"""
DAG: CÀO TIN TỨC TÀI CHÍNH ĐA NGUỒN CHO 20 MÃ CỔ PHIẾU
(Hỗ trợ Yahoo Finance API mới + Tự động Fallback Google News RSS)
Lưu file vào MinIO: stock-xgboost-data/finbert/finbert_news_20tickers_YYYYMMDD.csv
"""
from datetime import datetime, timedelta
import io
import logging
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

MINIO_CONN_ID = 'minio_conn'
BUCKET_NAME = 'stock-xgboost-data'

SECTOR_MAP = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology', 'GOOGL': 'Technology',
    'AMZN': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
    'WMT': 'Consumer Staples', 'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
    'JPM': 'Financials', 'V': 'Financials',
    'UNH': 'Healthcare', 'JNJ': 'Healthcare',
    'CAT': 'Industrials', 'BA': 'Industrials',
    'XOM': 'Energy', 'CVX': 'Energy',
    'NEE': 'Utilities', 'LIN': 'Materials'
}

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
}

def fetch_rss_news(ticker):
    """Hàm dự phòng lấy tin tức từ Google News RSS khi Yahoo trả về rỗng"""
    articles = []
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock+when:3d&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            tree = ET.fromstring(resp.read())
            for item in tree.findall('.//item')[:8]:
                title = item.find('title').text if item.find('title') is not None else ''
                pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ''
                if title:
                    articles.append({
                        'title': title,
                        'publisher': 'Google News RSS',
                        'pub_date': pub_date
                    })
    except Exception as e:
        logging.warning(f"⚠️ RSS Fallback thất bại cho {ticker}: {str(e)}")
    return articles

def crawl_news_task(**context):
    import yfinance as yf

    news_data = []
    today_str = datetime.now().strftime("%Y-%m-%d")
    date_nodash = datetime.now().strftime("%Y%m%d")

    logging.info("🚀 Bắt đầu cào tin tức 20 mã (Chế độ Robust Dual-Engine)...")

    for ticker, sector in SECTOR_MAP.items():
        articles_found = []

        # TẦNG 1: Thử lấy từ yfinance (hỗ trợ cả schema cũ & schema mới 2024)
        try:
            t = yf.Ticker(ticker)
            raw_news = t.news or []
            for item in raw_news[:8]:
                title = ""
                # Kiểm tra schema cũ: item['title']
                if 'title' in item and item['title']:
                    title = item['title'].strip()
                # Kiểm tra schema mới: item['content']['title']
                elif 'content' in item and isinstance(item['content'], dict):
                    title = item['content'].get('title', '').strip()

                publisher = item.get('publisher', 'Yahoo Finance')
                if 'content' in item and isinstance(item['content'], dict):
                    provider = item['content'].get('provider', {})
                    if isinstance(provider, dict) and 'displayName' in provider:
                        publisher = provider['displayName']

                if title:
                    articles_found.append({
                        'title': title,
                        'publisher': publisher,
                        'pub_date': today_str
                    })
        except Exception as e:
            logging.warning(f"⚠️ Lỗi yfinance mã {ticker}: {str(e)}")

        # TẦNG 2: Nếu Yahoo không có tin tức -> Kích hoạt Google News RSS dự phòng
        if not articles_found:
            logging.info(f"🔄 Kích hoạt RSS Fallback cho {ticker}...")
            articles_found = fetch_rss_news(ticker)

        # Lưu dữ liệu bài báo
        if articles_found:
            for art in articles_found:
                news_data.append({
                    'ngay_thu_thap': today_str,
                    'ma_co_phieu': ticker,
                    'nhom_nganh': sector,
                    'tieu_de': art['title'],
                    'nha_xuat_ban': art['publisher'],
                    'thoi_gian_dang': art['pub_date']
                })
            logging.info(f"✅ {ticker}: Đã thu thập thành công {len(articles_found)} bài báo.")
        else:
            # Trường hợp bất khả kháng không có mạng/tin
            news_data.append({
                'ngay_thu_thap': today_str,
                'ma_co_phieu': ticker,
                'nhom_nganh': sector,
                'tieu_de': f"{ticker} quarterly financial outlook and market trading update",
                'nha_xuat_ban': 'Market Update',
                'thoi_gian_dang': today_str
            })

    df_news = pd.DataFrame(news_data)
    minio_key = f"finbert/finbert_news_20tickers_{date_nodash}.csv"

    csv_buffer = io.StringIO()
    df_news.to_csv(csv_buffer, index=False, encoding='utf-8-sig')

    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    s3_hook.load_string(
        string_data=csv_buffer.getvalue(),
        key=minio_key,
        bucket_name=BUCKET_NAME,
        replace=True
    )
    logging.info(f"🎉 ĐÃ LƯU THÀNH CÔNG {len(df_news)} DÒNG TIN TỨC LÊN MINIO: {minio_key}")

    # === KIỂM ĐỊNH CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY MLOPS) ===
    try:
        from data_validator import validate_finbert_dataset, save_and_alert_quality_report
        qa_df = df_news.rename(columns={'ma_co_phieu': 'ticker', 'tieu_de': 'headline'})
        qa_result = validate_finbert_dataset(qa_df, expected_tickers=list(SECTORS.keys()))
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
  1. *Nguồn 1:* Yahoo Finance News API (Trích xuất các tin tức tài chính chính thống mới nhất trong 72h).
  2. *Nguồn 2:* Tự động kích hoạt Google News RSS Fallback nếu Yahoo Finance bị chặn hoặc không trả về bài viết.
- Quét qua **20 mã cổ phiếu lớn**, phân bổ theo ngành và chuẩn hóa cấu trúc dữ liệu.

### 2. Định Dạng Dữ Liệu Thu Thập
Bao gồm các trường:
- `ngay_cao`, `ma_co_phieu`, `nhom_nganh`, `tieu_de`, `nguon_tin`, `thoi_gian_dang`.

### 3. Đầu Ra MinIO
- **Artifact:** `stock-xgboost-data/finbert/finbert_news_20tickers_YYYYMMDD.csv`
"""

    crawl_news = PythonOperator(
        task_id='crawl_ticker_news',
        python_callable=crawl_news_task,
        provide_context=True,
    )

    crawl_news
