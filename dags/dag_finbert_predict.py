"""
DAG: CHẠY MÔ HÌNH FINBERT & DỰ BÁO SENTIMENT (TÍCH HỢP RETRY TELEGRAM & TIMEOUT AN TOÀN)
Lưu file vào MinIO: stock-data/finbert/sentiment_finbert_YYYYMMDD.csv
"""
from datetime import datetime, timedelta
import io
import logging
import pandas as pd
import numpy as np

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

MINIO_CONN_ID = 'minio_conn'
BUCKET_NAME = 'stock-data'

try:
    from alert_utils import telegram_failure_callback, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, send_telegram_safe
except ImportError:
    telegram_failure_callback = None
    TELEGRAM_TOKEN = ""
    TELEGRAM_CHAT_ID = ""
    def send_telegram_safe(url, data=None, files=None, max_retries=3): return False

try:
    from config_shared import SECTOR_MAP
except ImportError:
    from dags.config_shared import SECTOR_MAP

default_args = {
    'owner': 'quant_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'on_failure_callback': telegram_failure_callback,
    'execution_timeout': timedelta(minutes=60),
}

def predict_finbert_task(**context):
    import torch
    from datetime import timezone
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sentiment_utils import SENTIMENT_PREFIX, sentiment_summary
    from signal_formulas import finbert_signal, signal_action, FINBERT_FORMULA_VERSION
    from pipeline_storage import new_model_version
    from chart_utils import load_price_data_from_minio

    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    stamp = datetime.now(timezone.utc)
    date_nodash = stamp.strftime('%Y%m%d')
    today_str = stamp.strftime('%Y-%m-%d')
    keys = (s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix='finbert/finbert_news_100tickers_')
            or s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix='finbert/finbert_news_20tickers_') or [])
    if not keys:
        raise ValueError('Không có dữ liệu tin tức để chạy FinBERT.')
    # CSV is the daily primary view; parquet can be stale after a failed upload.
    parquet_keys = sorted([k for k in keys if k.endswith('.parquet')])
    csv_keys = sorted([k for k in keys if k.endswith('.csv')])
    current_csv = [key for key in csv_keys if date_nodash in key]
    current_parquet = [key for key in parquet_keys if date_nodash in key]
    news_key = current_csv[-1] if current_csv else current_parquet[-1] if current_parquet else None
    if not news_key:
        raise ValueError('Không tìm thấy file tin tức (.parquet hoặc .csv) hợp lệ.')
    obj = s3_hook.get_key(news_key, bucket_name=BUCKET_NAME)
    raw_bytes = obj.get()["Body"].read()
    if news_key.endswith('.parquet'):
        df_news = pd.read_parquet(io.BytesIO(raw_bytes))
    else:
        df_news = pd.read_csv(io.BytesIO(raw_bytes), encoding='utf-8', encoding_errors='replace')
    # News from an earlier crawl cannot be issued as a fresh prediction.
    if date_nodash not in news_key:
        raise ValueError('Dữ liệu tin tức chưa cập nhật cho lần dự báo này.')
    prices = load_price_data_from_minio(s3_hook, BUCKET_NAME)
    references = (prices.sort_values('Date').groupby('Ticker')['Date'].last().to_dict()
                  if not prices.empty else {})
    from news_data import select_eligible_news, NEWS_POLICY
    import hashlib
    from uuid import uuid4
    df_news = select_eligible_news(df_news, stamp)
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert') if not df_news.empty else None
    model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert') if not df_news.empty else None
    if model is not None:
        model.eval()
    labels = {str(label).lower(): int(index) for index, label in model.config.id2label.items()} if model is not None else {}
    results, scored_articles = [], []
    for ticker, sector in SECTOR_MAP.items():
        group = df_news[df_news.ma_co_phieu.eq(ticker)]
        outputs = []
        for _, row in group.iterrows():
            title = str(row.get('tieu_de', '')).strip()
            if not title or title == 'nan':
                continue
            summary = str(row.get('tom_tat', '')).strip()
            text = title + ('. ' + summary if summary and summary != 'nan' else '')
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            with torch.no_grad():
                probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0].numpy()
                outputs.append(probabilities)
                scored_articles.append({**row.to_dict(),
                    **{f'sentiment_{label}': float(probabilities[labels[label]]) for label in ('positive', 'negative', 'neutral')},
                    'sentiment_scored_at': datetime.now(timezone.utc).isoformat(),
                    'sentiment_model_version': 'ProsusAI/finbert', 'news_source_key': news_key,
                    'news_source_sha256': hashlib.sha256(raw_bytes).hexdigest()})
        if not outputs:
            results.append({'ngay_du_bao': today_str, 'ma_co_phieu': ticker, 'nhom_nganh': sector,
                            'sentiment_label': 'Chưa có cảm xúc', 'issued_at': datetime.now(timezone.utc).isoformat(),
                            'reference_date': pd.Timestamp(references[ticker]).strftime('%Y-%m-%d') if ticker in references else None,
                            'sentiment_model_version': None,
                            'news_count': len(outputs), 'news_missing': not outputs, 'news_policy': NEWS_POLICY,
                            'inference_source': 'unavailable', 'sentiment_positive': None,
                            'sentiment_negative': None, 'sentiment_neutral': None})
            continue
        mean_probs = np.mean(outputs, axis=0)
        sentiment = {f'sentiment_{label}': float(mean_probs[labels[label]])
                     for label in ('positive', 'negative', 'neutral')}
        results.append({'ngay_du_bao': today_str, 'ma_co_phieu': ticker, 'nhom_nganh': sector,
                        'sentiment_label': sentiment_summary(sentiment),
                        'issued_at': datetime.now(timezone.utc).isoformat(),
                        'reference_date': pd.Timestamp(references[ticker]).strftime('%Y-%m-%d') if ticker in references else None,
                        'sentiment_model_version': 'ProsusAI/finbert', 'news_count': len(outputs),
                        'news_sources': ' | '.join(sorted(set(group.nha_xuat_ban.dropna().astype(str)))) if 'nha_xuat_ban' in group else '',
                        'news_missing': False, 'news_policy': NEWS_POLICY, 'news_source_key': news_key,
                        'news_source_sha256': hashlib.sha256(raw_bytes).hexdigest(),
                        'inference_source': 'sentiment_model_v2', **sentiment})
    if not results:
        raise ValueError('Chưa có tin tức và giá tham chiếu hợp lệ để dự báo.')
    frame = pd.DataFrame(results)
    frame['prob_num'] = [finbert_signal(row) for row in results]
    frame['xac_suat_tang_gia'] = frame['prob_num'].apply(lambda score: f'{score * 100:.2f}%' if pd.notna(score) else None)
    frame['khuyen_nghi'] = frame['prob_num'].apply(lambda score: signal_action(score) if pd.notna(score) else 'Không khả dụng')
    frame['score_policy'] = FINBERT_FORMULA_VERSION
    frame['score_type'] = 'sentiment_signal'

    conf = getattr(context.get('dag_run'), 'conf', None) or {}
    frame['pipeline_run_id'] = conf.get('pipeline_run_id')
    if scored_articles:
        scored_key = f"news_sentiment_history/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}_{uuid4().hex}.csv"
        s3_hook.load_string(string_data=pd.DataFrame(scored_articles).to_csv(index=False),
                            key=scored_key, bucket_name=BUCKET_NAME, replace=False)
    s3_hook.load_string(string_data=frame.to_csv(index=False),
                        key=f"sentiment_history/finbert/{stamp.strftime('%Y%m%dT%H%M%S%f')}_{uuid4().hex}.csv",
                        bucket_name=BUCKET_NAME, replace=False)
    s3_hook.load_string(string_data=frame.to_csv(index=False),
                        key=f'{SENTIMENT_PREFIX}{date_nodash}.csv',
                        bucket_name=BUCKET_NAME, replace=True)
    logging.info('FinBERT processed %s tickers and %s articles.', len(frame), frame['news_count'].sum())
    from model_registry import ModelCard, save_model_to_minio
    card = ModelCard(model_name='FinBERT news sentiment',
                     model_version=new_model_version(), trained_date=today_str, model_type='finbert',
                     dataset_size=int(frame['news_count'].sum()), num_tickers=len(frame),
                     hyperparameters={'model_name': 'ProsusAI/finbert', 'max_length': 128},
                     metrics={}, notes='News sentiment and legacy linear sentiment signal for 35/35/30 Ensemble.')
    save_model_to_minio(s3_hook, BUCKET_NAME, 'finbert', card,
                        extra_artifacts={'sentiment_outputs.json': frame.to_json(orient='records')})
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        text = (f'📰 FinBERT — {today_str}\nĐã phân tích {len(frame)} mã từ mô hình pretrained.\n'
                f"Số tin đã phân tích: {int(frame['news_count'].sum())}.\n"
                'Kết quả gồm cảm xúc tích cực, tiêu cực và trung tính.')
        send_telegram_safe(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',
                           data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})
        send_telegram_safe(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument',
                           data={'chat_id': TELEGRAM_CHAT_ID},
                           files={'document': (f'finbert_{date_nodash}.csv', frame.to_csv(index=False).encode(), 'text/csv')})



with DAG(
    dag_id='dag_finbert_predict',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['finbert', 'prediction', 'minio', 'telegram', 'nlp']
) as dag:

    dag.doc_md = """FinBERT phân loại cảm xúc tin tức: tích cực, tiêu cực, trung tính.
Điểm tín hiệu = clip((1 + positive - negative) / 2, 0.05, 0.95).
Master kết hợp 35% XGBoost, 35% LSTM, 30% điểm FinBERT.
Lưu thời điểm phân tích, nguồn tin và phiên bản mô hình cho mỗi quan sát."""

    predict_finbert = PythonOperator(
        task_id='predict_finbert_sentiment',
        python_callable=predict_finbert_task,
        provide_context=True,
    )
