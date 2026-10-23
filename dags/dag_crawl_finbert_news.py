"""
DAG: CÀO TIN TỨC TÀI CHÍNH ĐA NGUỒN CHO 20 MÃ CỔ PHIẾU
(Hỗ trợ Yahoo Finance API mới + Tự động Fallback Google News RSS)
Lưu file vào MinIO: stock-data/finbert/finbert_news_20tickers_YYYYMMDD.csv
"""
from datetime import datetime, timedelta, timezone
import io
import logging
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

MINIO_CONN_ID = 'minio_conn'
BUCKET_NAME = 'stock-data'

try:
    from config_shared import SECTOR_MAP
except ImportError:
    from dags.config_shared import SECTOR_MAP

try:
    from alert_utils import telegram_failure_callback
except ImportError:
    telegram_failure_callback = None

default_args = {
    'owner': 'quant_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': telegram_failure_callback,
    'execution_timeout': timedelta(minutes=45),
}

from news_sources import fetch_yahoo_news, fetch_rss_news, collect_ticker_news

def crawl_news_task(**context):
    from uuid import uuid4
    from news_data import NEWS_COLUMNS, NEWS_POLICY, select_eligible_news

    batches = []
    failures = {}
    logging.info("Collecting observed news with publication and collection timestamps.")
    for ticker, sector in SECTOR_MAP.items():
        batch, retrieval = collect_ticker_news(
            ticker, sector, yahoo_loader=fetch_yahoo_news, rss_loader=fetch_rss_news,
            supplement=True, use_cache=False)
        if retrieval['source_errors']:
            failures[ticker] = retrieval['source_errors']
        batches.append(batch)
        logging.info("%s: %s real article records; %s with verified timestamps.",
                     ticker, len(batch), int(batch.eligible_for_model.sum()))

    stamp = datetime.now(timezone.utc)
    date_nodash = stamp.strftime("%Y%m%d")
    df_news = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(columns=NEWS_COLUMNS)
    minio_key = f"finbert/finbert_news_20tickers_{date_nodash}.csv"
    csv_text = df_news.to_csv(index=False)
    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    archive_key = f"news_history/{stamp.strftime('%Y%m%dT%H%M%S%f')}_{uuid4().hex}.csv"
    # Immutable observations are the primary record. The daily file is a view.
    s3_hook.load_string(string_data=csv_text, key=archive_key,
                        bucket_name=BUCKET_NAME, replace=False)
    s3_hook.load_string(string_data=csv_text, key=minio_key,
                        bucket_name=BUCKET_NAME, replace=True)
    eligible = select_eligible_news(df_news, stamp)
    import json
    report = {'news_policy': NEWS_POLICY, 'collected_at': stamp.isoformat(),
              'archive_key': archive_key, 'rows': len(df_news), 'eligible_rows': len(eligible),
              'missing_tickers': sorted(set(SECTOR_MAP) - set(eligible.get('ma_co_phieu', []))),
              'status_counts': df_news.timestamp_status.value_counts().to_dict(),
              'source_failures': failures}
    s3_hook.load_string(string_data=json.dumps(report, ensure_ascii=False),
                        key=archive_key.replace('.csv', '.json'), bucket_name=BUCKET_NAME, replace=False)

    # Xuất định dạng Apache Parquet (nén Snappy)
    try:
        parquet_buf = io.BytesIO()
        df_news.to_parquet(parquet_buf, engine='pyarrow', compression='snappy', index=False)
        parquet_key = minio_key.replace('.csv', '.parquet')
        s3_hook.load_bytes(
            bytes_data=parquet_buf.getvalue(),
            key=parquet_key,
            bucket_name=BUCKET_NAME,
            replace=True
        )
        logging.info(f"📦 Đã xuất song song Parquet cho FinBERT News: {parquet_key}")
    except Exception as pq_err:
        logging.warning(f"⚠️ Lưu Parquet News không thành công: {pq_err}")

    logging.info(f"🎉 ĐÃ LƯU THÀNH CÔNG {len(df_news)} DÒNG TIN TỨC LÊN MINIO: {minio_key}")

    # === KIỂM ĐỊNH CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY MLOPS) ===
    try:
        from data_validator import validate_finbert_dataset, save_and_alert_quality_report
        qa_df = df_news.rename(columns={'ma_co_phieu': 'ticker', 'tieu_de': 'headline'})
        qa_result = validate_finbert_dataset(qa_df, expected_tickers=list(SECTOR_MAP.keys()))
        save_and_alert_quality_report(qa_result, s3_hook, BUCKET_NAME, date_nodash)
    except Exception as qa_err:
        logging.warning(f"⚠️ Không thể chạy QA validation FinBERT: {qa_err}")


with DAG(
    dag_id='dag_crawl_finbert_news',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['news', 'finbert', 'crawl', 'minio', 'nlp']
) as dag:

    dag.doc_md = """
# 📰 DAG: Cào Tin Tức Tài Chính Đa Nguồn (FinBERT NLP Pipeline)
---
### 1. Mục Đích & Cơ Chế Thu Thập
DAG đảm nhiệm tầng **News Ingestion** đa nguồn cho mô hình phân tích cảm xúc ngôn ngữ tự nhiên **FinBERT**:
- **Cơ chế tải kép (Dual-Source Fallback):**
  1. *Nguồn 1:* Yahoo Finance RSS (Trích xuất các tin tức tài chính chính thống mới nhất trong 72h).
  2. *Nguồn 2:* Tự động kích hoạt Google News RSS Fallback nếu Yahoo Finance bị chặn hoặc không trả về bài viết.
- Quét qua **20 mã cổ phiếu lớn**, phân bổ theo ngành và chuẩn hóa cấu trúc dữ liệu.

### 2. Định Dạng Dữ Liệu Thu Thập
Mỗi phiên bản bài báo lưu `published_at`, `first_seen_at`, `available_at`,
`story_id`, `article_version`, nguồn URL và trạng thái timestamp. Thiếu tin
được ghi nhận bằng số lượng 0; không tạo bài thay thế. Chỉ tin có nguồn và
thời gian hợp lệ mới được dùng cho mô hình. Mọi lần thu thập lưu bất biến
trong `news_history/`; file ngày là bản xem mới nhất.

### 3. Đầu Ra MinIO
- **Artifact:** `stock-data/finbert/finbert_news_20tickers_YYYYMMDD.csv`
"""

    crawl_news = PythonOperator(
        task_id='crawl_ticker_news',
        python_callable=crawl_news_task,
        provide_context=True,
    )

    crawl_news
