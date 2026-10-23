"""
DAG: ĐỐI CHIẾU 3 PHƯƠNG PHÁP ĐỘC LẬP & TẠO PHƯƠNG PHÁP 4 TỔNG HỢP
- Lưu lịch sử dự báo bất biến: stock-data/forecast_history/ensemble/
- Lưu file So sánh: stock-data/comparison/bang_so_sanh_4_phuong_phap_YYYYMMDD.csv
- Gửi báo cáo so sánh 4 phương pháp về Telegram
"""
from datetime import datetime, timedelta
import io
import logging
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

MINIO_CONN_ID = 'minio_conn'
BUCKET_NAME = 'stock-data'

try:
    from alert_utils import telegram_failure_callback, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    telegram_failure_callback = None
    TELEGRAM_TOKEN = ""
    TELEGRAM_CHAT_ID = ""

default_args = {
    'owner': 'quant_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': telegram_failure_callback,
    'execution_timeout': timedelta(minutes=25),
}

try:
    from config_shared import SECTOR_MAP
except ImportError:
    from dags.config_shared import SECTOR_MAP

def compare_and_ensemble_task(**context):
    from datetime import timezone
    from signal_formulas import finbert_signal, weighted_ensemble, signal_action, finbert_signal_text, ENSEMBLE_FORMULA_VERSION, ENSEMBLE_SOURCE
    from sentiment_utils import SENTIMENT_PREFIX, LEGACY_SENTIMENT_PREFIX, stored_sentiment_details
    from forecast_history import save_forecast_history
    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    stamp = datetime.now(timezone.utc)
    date_nodash = stamp.strftime('%Y%m%d')
    frames = {}
    conf = getattr(context.get('dag_run'), 'conf', None) or {}
    pipeline_run_id = conf.get('pipeline_run_id')
    prefixes = {'xgb': 'xgboost/du_bao_tang_truong_',
                'lstm': 'lstm/du_bao_tang_truong_lstm_',
                'bert': SENTIMENT_PREFIX}
    for name, prefix in prefixes.items():
        key = prefix + date_nodash + '.csv'
        keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix=prefix) or []
        if name == 'bert' and key not in keys:
            key = LEGACY_SENTIMENT_PREFIX + date_nodash + '.csv'
            keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix=LEGACY_SENTIMENT_PREFIX) or []
        if key not in keys:
            continue
        frame = pd.read_csv(io.StringIO(s3_hook.read_key(key, bucket_name=BUCKET_NAME)))
        required = {'ma_co_phieu', 'issued_at', 'reference_date', 'inference_source'}
        if name != 'bert':
            required |= {'prob_num', 'model_version'}
        if not required.issubset(frame.columns):
            continue
        if name == 'bert':
            valid = frame.apply(lambda row: stored_sentiment_details(row)['finbert_status'] == 'available', axis=1)
        else:
            valid = frame['inference_source'].eq('trained_model_v2') & frame['model_version'].notna()
        if pipeline_run_id:
            valid &= frame.get('pipeline_run_id', pd.Series(index=frame.index, dtype=object)).eq(pipeline_run_id)
        frame = frame[valid & frame['reference_date'].notna()]
        frames[name] = frame.set_index('ma_co_phieu').to_dict('index')
    tickers = set().union(*(set(rows) for rows in frames.values())) if frames else set()
    if not tickers:
        logging.warning('⚠️ Chưa có dự báo hợp lệ từ mô hình đã huấn luyện. Xuất báo cáo trạng thái.')
        empty_report = pd.DataFrame([{'ngay_du_bao': stamp.strftime('%Y-%m-%d'),
                                      'trang_thai': 'Chưa có mô hình đạt chuẩn chất lượng'}])
        s3_hook.load_string(string_data=empty_report.to_csv(index=False),
                            key=f'comparison/bang_so_sanh_4_phuong_phap_{date_nodash}.csv',
                            bucket_name=BUCKET_NAME, replace=True)
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            from alert_utils import send_telegram_safe
            send_telegram_safe(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',
                               data={'chat_id': TELEGRAM_CHAT_ID,
                                     'text': f'⚠️ Ensemble — {stamp.strftime("%Y-%m-%d")}\n'
                                             'Tất cả mô hình đang ở trạng thái unavailable/SAFE MODE.\n'
                                             'Pipeline chạy thành công nhưng chưa có dự báo giá.'})
        return
    comparisons, forecasts = [], []
    for ticker in sorted(tickers):
        rows = {name: records[ticker] for name, records in frames.items() if ticker in records}
        reference = max(str(row['reference_date']) for row in rows.values())
        rows = {name: row for name, row in rows.items() if str(row['reference_date']) == reference}
        values = {f'prob_{name}': float(row['prob_num']) for name, row in rows.items()
                  if name in ('xgb', 'lstm') and pd.notna(row['prob_num']) and 0 <= float(row['prob_num']) <= 1}
        sentiment = {key: rows.get('bert', {}).get(key) for key in
                     ('sentiment_positive', 'sentiment_negative', 'sentiment_neutral')}
        bert_score = finbert_signal(sentiment)
        if bert_score is not None:
            values['prob_bert'] = bert_score
        probability, weights = weighted_ensemble(values)
        if probability is None:
            continue
        version = ENSEMBLE_FORMULA_VERSION
        action = signal_action(probability)
        def text(name):
            row = rows.get(name, {})
            value = values.get(f'prob_{name}')
            return f"{row.get('khuyen_nghi', '')} ({value*100:.2f}%)" if value is not None else 'Không khả dụng'
        issued_at = datetime.now(timezone.utc).isoformat()
        sector = next(iter(rows.values())).get('nhom_nganh', SECTOR_MAP.get(ticker, 'General'))
        forecasts.append({'ma_co_phieu': ticker, 'reference_date': reference, 'issued_at': issued_at,
                          'prediction_horizon': 1, 'pipeline_run_id': pipeline_run_id, 'prob_num': probability, 'model_version': version,
                          'news_policy': rows.get('bert', {}).get('news_policy'),
                          'news_count': rows.get('bert', {}).get('news_count', 0),
                          'sentiment_model_version': rows.get('bert', {}).get('sentiment_model_version'),
                          'inference_source': ENSEMBLE_SOURCE, 'ensemble_method': version,
                          'weight_xgb': weights.get('prob_xgb'), 'weight_lstm': weights.get('prob_lstm'),
                          'weight_bert': weights.get('prob_bert'), **values, **sentiment})
        xgb = rows.get('xgb', {})
        comparisons.append({'ngay_du_bao': stamp.strftime('%Y-%m-%d'), 'ma_co_phieu': ticker,
                            'nhom_nganh': sector, 'pp1_xgboost': text('xgb'), 'pp2_lstm': text('lstm'),
                            'pp3_finbert': finbert_signal_text(sentiment), **sentiment,
                            'news_policy': rows.get('bert', {}).get('news_policy'),
                            'news_count': rows.get('bert', {}).get('news_count', 0),
                            'news_sources': rows.get('bert', {}).get('news_sources', ''),
                            'sentiment_model_version': rows.get('bert', {}).get('sentiment_model_version'),
                            'pp4_tong_hop': f'{action} ({probability*100:.2f}%)' if probability is not None else action,
                            'trang_thai_doi_chieu': 'Tổng hợp theo trọng số 35/35/30',
                            'ensemble_model_version': version, 'ensemble_method': version,
                            'weight_xgb': weights.get('prob_xgb'), 'weight_lstm': weights.get('prob_lstm'),
                            'weight_bert': weights.get('prob_bert'),
                            'inference_source': ENSEMBLE_SOURCE,
                            'reference_date': reference, 'issued_at': issued_at,
                            'return_model_version': xgb.get('return_model_version'),
                            'gia_5d_du_kien': xgb.get('gia_5d_du_kien')})
    if not comparisons:
        logging.warning('⚠️ Có dữ liệu model nhưng không có xác suất hợp lệ nào.')
        return
    save_forecast_history(s3_hook, BUCKET_NAME, 'ensemble', pd.DataFrame(forecasts))
    s3_hook.load_string(string_data=pd.DataFrame(comparisons).to_csv(index=False),
                        key=f'comparison/bang_so_sanh_4_phuong_phap_{date_nodash}.csv',
                        bucket_name=BUCKET_NAME, replace=True)
    logging.info('Saved weighted signals; %s ensemble predictions.', sum(row['prob_num'] is not None for row in forecasts))
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        from alert_utils import send_telegram_safe
        valid_count = sum(row['prob_num'] is not None for row in forecasts)
        text = (f'Ensemble — {stamp.strftime("%Y-%m-%d")}\n'
                f'Kết quả từ các mô hình và tín hiệu tin tức: {len(comparisons)} mã.\n'
                f'Tín hiệu Ensemble theo trọng số 35/35/30: {valid_count} mã.\n'
                'Xác suất và mục tiêu chưa có mô hình tương ứng được để trống.')
        send_telegram_safe(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',
                           data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})
        send_telegram_safe(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument',
                           data={'chat_id': TELEGRAM_CHAT_ID},
                           files={'document': (f'comparison_{date_nodash}.csv', pd.DataFrame(comparisons).to_csv(index=False).encode(), 'text/csv')})



with DAG(
    dag_id='dag_ensemble_master',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['so_sanh', '4_phuong_phap', 'ensemble', 'xgboost', 'lstm', 'finbert', 'telegram']
) as dag:

    dag.doc_md = """Master Ensemble = 35% XGBoost + 35% LSTM + 30% điểm FinBERT.
FinBERT = clip((1 + positive - negative) / 2, 0.05, 0.95).
Khi thiếu đầu vào, chuẩn hóa trọng số của các đầu vào còn lại.
Đây là điểm tổng hợp theo công thức; lịch sử phát hành được kiểm chứng theo T+1."""

    run_compare_ensemble = PythonOperator(
        task_id='compare_and_ensemble_all_models',
        python_callable=compare_and_ensemble_task,
        provide_context=True,
    )
