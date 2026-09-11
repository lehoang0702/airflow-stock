"""
DAG: ĐÁNH GIÁ MÔ HÌNH DỰ BÁO CỔ PHIẾU (EVALUATION PIPELINE)
===============================================================
Chạy sau khi train xong, thực hiện đánh giá toàn diện 3 mô hình:
  Task 1: evaluate_xgboost
  Task 2: evaluate_lstm
  Task 3: evaluate_finbert
  Task 4: compare_all_models
  Task 5: send_evaluation_report
"""
from datetime import datetime, timedelta
import io
import time
import requests
import logging
import json
import pandas as pd
import numpy as np

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

MINIO_CONN_ID = 'minio_conn'
BUCKET_NAME = 'stock-xgboost-data'

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

default_args = {
    'owner': 'quant_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'on_failure_callback': telegram_failure_callback,
}


def _load_valid_dataset(s3_hook, prefix, min_rows=100):
    """Tải dataset hợp lệ mới nhất từ MinIO."""
    keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix=prefix)
    if not keys:
        return None

    for k in reversed(sorted(keys)):
        try:
            raw = s3_hook.read_key(k, bucket_name=BUCKET_NAME)
            df = pd.read_csv(io.StringIO(raw))
            if len(df) >= min_rows:
                logging.info(f"✅ Đã chọn dataset: {k} ({len(df):,} dòng)")
                return df
        except Exception:
            continue
    return None


# ==============================================================================
#  TASK 1: ĐÁNH GIÁ XGBOOST
# ==============================================================================

def evaluate_xgboost_task(**context):
    """Đánh giá XGBoost trên test set với đầy đủ metrics + Cross-Validation."""
    import xgboost as xgb
    from model_evaluator import (
        evaluate_binary_classifier,
        generate_classification_report_text,
        run_time_series_cv,
        save_evaluation_to_minio
    )
    from model_registry import XGBoostModelBuilder, save_model_to_minio

    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    date_nodash = datetime.now().strftime("%Y%m%d")

    # 1. Đọc dataset
    df = _load_valid_dataset(s3_hook, "xgboost/xgboost_stock_20tickers_10y_")
    if df is None:
        raise ValueError("❌ Không tìm thấy dataset XGBoost hợp lệ!")

    # 2. Train & Đánh giá (dùng ModelBuilder)
    builder = XGBoostModelBuilder()
    train_metrics = builder.train(df)

    # 3. Đánh giá chi tiết trên test set
    df_processed, feature_cols = builder.preprocess(df)
    train_mask = df_processed['Date_Std'] < pd.Timestamp('2024-01-01')
    if train_mask.sum() < 200:
        split_idx = int(len(df_processed) * 0.8)
        test_mask = df_processed.index >= split_idx
    else:
        test_mask = ~train_mask

    X_test = df_processed.loc[test_mask, feature_cols].values
    y_test = df_processed.loc[test_mask, 'Target_Std'].values
    y_prob = builder.model.predict_proba(X_test)[:, 1]

    eval_metrics = evaluate_binary_classifier(y_test, y_prob, "XGBoost")

    # 4. Classification Report
    report = generate_classification_report_text(y_test, y_prob, "XGBoost")
    logging.info(report)

    # 5. Cross-Validation (TimeSeriesSplit)
    X_all = df_processed[feature_cols].values
    y_all = df_processed['Target_Std'].values

    def xgb_builder_fn(X_tr, y_tr):
        m = xgb.XGBClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.03,
            subsample=0.75, colsample_bytree=0.75, min_child_weight=3,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )
        m.fit(X_tr, y_tr)
        return m

    cv_metrics = run_time_series_cv(X_all, y_all, xgb_builder_fn, n_splits=5, model_name="XGBoost")

    # 6. Cập nhật metrics vào model card
    full_metrics = {**eval_metrics, **cv_metrics}
    builder.model_card.metrics = full_metrics

    # 7. Lưu model + evaluation lên MinIO
    save_model_to_minio(
        s3_hook=s3_hook,
        bucket_name=BUCKET_NAME,
        model_type='xgboost',
        model_card=builder.model_card,
        model_bytes=builder.serialize()
    )

    save_evaluation_to_minio(s3_hook, BUCKET_NAME, 'xgboost', full_metrics, date_nodash)

    # Push metrics cho task compare
    context['ti'].xcom_push(key='xgboost_eval', value=json.dumps(full_metrics, default=str))
    context['ti'].xcom_push(key='xgboost_cv', value=json.dumps(cv_metrics, default=str))
    logging.info("✅ Đánh giá XGBoost hoàn tất!")


# ==============================================================================
#  TASK 2: ĐÁNH GIÁ LSTM
# ==============================================================================

def evaluate_lstm_task(**context):
    """Đánh giá LSTM trên test set với đầy đủ metrics."""
    import typing_extensions
    if not hasattr(typing_extensions, 'TypeIs'):
        class _TypeIsMeta(type):
            def __getitem__(self, item): return bool
        class _TypeIs(metaclass=_TypeIsMeta): pass
        typing_extensions.TypeIs = _TypeIs

    import torch
    from sklearn.metrics import accuracy_score, roc_auc_score
    from model_evaluator import (
        evaluate_binary_classifier,
        generate_classification_report_text,
        save_evaluation_to_minio
    )
    from model_registry import LSTMModelBuilder, save_model_to_minio

    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    date_nodash = datetime.now().strftime("%Y%m%d")

    # 1. Đọc dataset
    df = _load_valid_dataset(s3_hook, "lstm/lstm_stock_20tickers_10y_")
    if df is None:
        raise ValueError("❌ Không tìm thấy dataset LSTM hợp lệ!")

    # 2. Train & Đánh giá
    builder = LSTMModelBuilder()
    train_metrics = builder.train(df)

    # 3. Đánh giá chi tiết trên test set
    df_processed, feature_cols = builder.preprocess(df)
    lookback = builder.hyperparams['lookback']
    temp = builder.hyperparams['temperature']

    test_df = df_processed[df_processed['Date_Std'] >= '2024-01-01'].copy()
    test_df[feature_cols] = builder.scaler.transform(test_df[feature_cols].values)

    X_test, y_test = builder._create_sequences(test_df, feature_cols, lookback)

    if len(X_test) > 0:
        builder.model.eval()
        with torch.no_grad():
            test_logits = builder.model(torch.tensor(X_test)).numpy()
            y_prob = 1.0 / (1.0 + np.exp(-test_logits / temp))

        eval_metrics = evaluate_binary_classifier(y_test, y_prob, "LSTM")

        report = generate_classification_report_text(y_test, y_prob, "LSTM")
        logging.info(report)
    else:
        eval_metrics = {'accuracy': 0.5, 'auc_roc': 0.5, 'total_samples': 0}

    # 4. Cập nhật model card
    builder.model_card.metrics = eval_metrics

    # 5. Lưu model + scaler + evaluation lên MinIO
    save_model_to_minio(
        s3_hook=s3_hook,
        bucket_name=BUCKET_NAME,
        model_type='lstm',
        model_card=builder.model_card,
        model_bytes=builder.serialize_model(),
        scaler_bytes=builder.serialize_scaler()
    )

    save_evaluation_to_minio(s3_hook, BUCKET_NAME, 'lstm', eval_metrics, date_nodash)

    context['ti'].xcom_push(key='lstm_eval', value=json.dumps(eval_metrics, default=str))
    logging.info("✅ Đánh giá LSTM hoàn tất!")


# ==============================================================================
#  TASK 3: ĐÁNH GIÁ FINBERT
# ==============================================================================

def evaluate_finbert_task(**context):
    """Đánh giá FinBERT bằng cách đối chiếu sentiment vs giá thực tế."""
    from model_evaluator import (
        evaluate_finbert_sentiment,
        save_evaluation_to_minio
    )
    from model_registry import save_model_to_minio, ModelCard
    from chart_utils import load_price_data_from_minio

    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    date_nodash = datetime.now().strftime("%Y%m%d")
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 1. Đọc kết quả dự báo FinBERT mới nhất
    keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix="finbert/du_bao_tang_truong_finbert_")
    if not keys:
        logging.warning("⚠️ Không tìm thấy kết quả FinBERT để đánh giá")
        context['ti'].xcom_push(key='finbert_eval', value=json.dumps({}))
        return

    latest_key = sorted(keys)[-1]
    raw = s3_hook.read_key(latest_key, bucket_name=BUCKET_NAME)
    df_pred = pd.read_csv(io.StringIO(raw))

    # Thêm prob_num nếu chưa có
    if 'prob_num' not in df_pred.columns:
        df_pred['prob_num'] = df_pred['xac_suat_tang_gia'].str.replace('%', '').astype(float) / 100.0

    # 2. Đọc dữ liệu giá lịch sử
    df_prices = load_price_data_from_minio(s3_hook, BUCKET_NAME)

    # 3. Đánh giá sentiment vs giá thực tế
    eval_metrics = evaluate_finbert_sentiment(df_pred, df_prices, prediction_horizon=5)

    # 4. Tạo model card cho FinBERT
    model_card = ModelCard(
        model_name="FinBERT Sentiment Analysis (ProsusAI/finbert)",
        model_version=f"v{date_nodash}",
        trained_date=today_str,
        model_type="finbert",
        dataset_size=len(df_pred),
        num_tickers=df_pred['ma_co_phieu'].nunique(),
        hyperparameters={
            'model_name': 'ProsusAI/finbert',
            'max_length': 128,
            'buy_threshold': 0.55,
            'sell_threshold': 0.45
        },
        metrics=eval_metrics,
        notes=f"Evaluated on {eval_metrics.get('total_evaluated', 0)} tickers with T+5 price verification"
    )

    # 5. Lưu model card + evaluation
    save_model_to_minio(
        s3_hook=s3_hook,
        bucket_name=BUCKET_NAME,
        model_type='finbert',
        model_card=model_card,
        extra_artifacts={
            'pipeline_config.json': json.dumps(model_card.hyperparameters, indent=2),
        }
    )

    save_evaluation_to_minio(s3_hook, BUCKET_NAME, 'finbert', eval_metrics, date_nodash)

    context['ti'].xcom_push(key='finbert_eval', value=json.dumps(eval_metrics, default=str))
    logging.info("✅ Đánh giá FinBERT hoàn tất!")


# ==============================================================================
#  TASK 4: SO SÁNH TẤT CẢ MÔ HÌNH
# ==============================================================================

def compare_all_models_task(**context):
    """So sánh metrics giữa 3 mô hình và tạo bảng tổng hợp."""
    from model_evaluator import generate_evaluation_report, save_comparison_to_minio

    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    date_nodash = datetime.now().strftime("%Y%m%d")
    today_str = datetime.now().strftime("%Y-%m-%d")

    ti = context['ti']

    # Đọc metrics từ XCom
    eval_results = {}
    cv_results = {}

    xgb_eval = ti.xcom_pull(key='xgboost_eval', task_ids='evaluate_xgboost')
    if xgb_eval:
        eval_results['XGBoost'] = json.loads(xgb_eval)
        xgb_cv = ti.xcom_pull(key='xgboost_cv', task_ids='evaluate_xgboost')
        if xgb_cv:
            cv_results['XGBoost'] = json.loads(xgb_cv)

    lstm_eval = ti.xcom_pull(key='lstm_eval', task_ids='evaluate_lstm')
    if lstm_eval:
        eval_results['LSTM'] = json.loads(lstm_eval)

    finbert_eval = ti.xcom_pull(key='finbert_eval', task_ids='evaluate_finbert')
    if finbert_eval:
        eval_results['FinBERT'] = json.loads(finbert_eval)

    if not eval_results:
        logging.warning("⚠️ Không có kết quả đánh giá nào để so sánh")
        return

    # Tạo báo cáo
    telegram_msg, csv_string = generate_evaluation_report(eval_results, cv_results, today_str)

    # Lưu lên MinIO
    save_comparison_to_minio(s3_hook, BUCKET_NAME, csv_string, date_nodash)

    # Push cho task gửi Telegram
    ti.xcom_push(key='evaluation_telegram_msg', value=telegram_msg)
    ti.xcom_push(key='evaluation_csv', value=csv_string)

    logging.info("✅ So sánh 3 mô hình hoàn tất!")


# ==============================================================================
#  TASK 5: GỬI BÁO CÁO VỀ TELEGRAM
# ==============================================================================

def send_evaluation_report_task(**context):
    """Gửi báo cáo đánh giá về Telegram."""
    ti = context['ti']
    date_nodash = datetime.now().strftime("%Y%m%d")

    telegram_msg = ti.xcom_pull(key='evaluation_telegram_msg', task_ids='compare_all_models')
    csv_string = ti.xcom_pull(key='evaluation_csv', task_ids='compare_all_models')

    if not telegram_msg:
        logging.warning("⚠️ Không có nội dung báo cáo để gửi")
        return

    # 1. Gửi tin nhắn text
    url_msg = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    send_telegram_safe(url_msg, data={
        'chat_id': TELEGRAM_CHAT_ID,
        'text': telegram_msg,
        'parse_mode': 'HTML'
    })

    # 2. Gửi file CSV
    if csv_string:
        url_doc = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
        csv_bytes = csv_string.encode('utf-8-sig')
        files = {'document': (f"evaluation_report_{date_nodash}.csv", csv_bytes, 'text/csv')}
        send_telegram_safe(url_doc, data={
            'chat_id': TELEGRAM_CHAT_ID,
            'caption': f'📊 Báo cáo đánh giá chi tiết 3 mô hình ({datetime.now().strftime("%Y-%m-%d")})'
        }, files=files)

    logging.info("✅ Đã gửi báo cáo đánh giá về Telegram!")


# ==============================================================================
#  TỔNG HỢP: Chạy toàn bộ pipeline đánh giá (cho test manual)
# ==============================================================================

def evaluate_all_models_task(**context):
    """Hàm tổng hợp chạy toàn bộ pipeline đánh giá (dùng cho test thủ công)."""
    # Tạo mock context cho xcom
    class MockTI:
        def __init__(self):
            self._store = {}
        def xcom_push(self, key, value):
            self._store[key] = value
        def xcom_pull(self, key, task_ids=None):
            return self._store.get(key)

    mock_ctx = {'ti': MockTI()}

    logging.info("=" * 60)
    logging.info("🌲 ĐÁNH GIÁ XGBOOST")
    logging.info("=" * 60)
    evaluate_xgboost_task(**mock_ctx)

    logging.info("=" * 60)
    logging.info("🧠 ĐÁNH GIÁ LSTM")
    logging.info("=" * 60)
    evaluate_lstm_task(**mock_ctx)

    logging.info("=" * 60)
    logging.info("📰 ĐÁNH GIÁ FINBERT")
    logging.info("=" * 60)
    evaluate_finbert_task(**mock_ctx)

    logging.info("=" * 60)
    logging.info("📊 SO SÁNH 3 MÔ HÌNH")
    logging.info("=" * 60)
    compare_all_models_task(**mock_ctx)

    logging.info("=" * 60)
    logging.info("📤 GỬI BÁO CÁO")
    logging.info("=" * 60)
    send_evaluation_report_task(**mock_ctx)


# ==============================================================================
#  DAG DEFINITION
# ==============================================================================

with DAG(
    dag_id='dag_model_evaluation',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['evaluation', 'metrics', 'xgboost', 'lstm', 'finbert', 'cross_validation']
) as dag:

    dag.doc_md = """
# 📊 DAG: Đánh Giá Mô Hình Dự Báo Toàn Diện (Scientific Model Evaluation)
---
### 1. Mục Đích & Tiêu Chuẩn Học Thuật (MLOps Evaluation Pipeline)
DAG phụ trách đánh giá khoa học và khách quan hiệu năng của cả 3 mô hình học máy theo tiêu chuẩn công bố khoa học:
- **XGBoost (Tabular):** Đánh giá trên tập test độc lập + TimeSeriesSplit 5-Fold Cross-Validation.
- **LSTM (PyTorch):** Đánh giá chuỗi nến test set và ma trận nhầm lẫn (Confusion Matrix).
- **FinBERT (NLP):** Đối chiếu xác suất sentiment với biến động giá thực tế T+5 của thị trường.

### 2. Bộ Chỉ Số Đánh Giá Đầy Đủ
| Chỉ Số | Ý Nghĩa Thực Tiễn Trong Tài Chính |
|---|---|
| **Accuracy** | Tỷ lệ dự báo đúng chiều giá tăng/giảm trên toàn bộ mẫu |
| **AUC-ROC** | Khả năng phân tách giữa cổ phiếu tăng và cổ phiếu giảm |
| **Precision** | Xác suất cổ phiếu thực sự tăng khi mô hình hô MUA (Tránh bẫy Bull-trap) |
| **Recall** | Tỷ lệ bắt được các cơ hội tăng giá lớn của thị trường (Tránh bỏ lỡ sóng) |
| **F1-Score** | Trung bình điều hòa cân bằng giữa Precision và Recall |
| **Confusion Matrix** | Bảng chi tiết: TP (True Positive), FP, TN, FN |
| **TimeSeriesSplit CV** | Đo lường độ ổn định (Mean ± Std) qua 5 giai đoạn thị trường khác nhau |
| **Sentiment Correlation**| Hệ số tương quan Pearson giữa điểm cảm xúc tin tức và lợi suất giá thực tế |

### 3. Đầu Ra MinIO & Kênh Phân Phối
- **Báo cáo JSON từng model:** `evaluation/{model_type}/eval_YYYYMMDD.json`
- **Bảng so sánh tổng hợp:** `evaluation/comparison/eval_comparison_YYYYMMDD.csv`
- **Thông báo Telegram:** Bảng tổng hợp Markdown + tệp CSV đính kèm gửi trực tiếp về Telegram.
"""

    eval_xgb = PythonOperator(
        task_id='evaluate_xgboost',
        python_callable=evaluate_xgboost_task,
        provide_context=True,
    )

    eval_lstm = PythonOperator(
        task_id='evaluate_lstm',
        python_callable=evaluate_lstm_task,
        provide_context=True,
    )

    eval_finbert = PythonOperator(
        task_id='evaluate_finbert',
        python_callable=evaluate_finbert_task,
        provide_context=True,
    )

    compare_task = PythonOperator(
        task_id='compare_all_models',
        python_callable=compare_all_models_task,
        provide_context=True,
    )

    send_report = PythonOperator(
        task_id='send_evaluation_report',
        python_callable=send_evaluation_report_task,
        provide_context=True,
    )

    # Task flow: eval 3 models song song → compare → send report
    [eval_xgb, eval_lstm, eval_finbert] >> compare_task >> send_report
