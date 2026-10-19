"""
DAG: ĐÁNH GIÁ MÔ HÌNH DỰ BÁO CỔ PHIẾU (EVALUATION PIPELINE)
===============================================================
Chạy sau khi train xong, thực hiện đánh giá toàn diện 3 mô hình:
  Task 1: evaluate_xgboost
  Task 2: evaluate_lstm
  Task 3: evaluate_ensemble
  Task 4: compare_all_models
  Task 5: send_evaluation_report
"""
from datetime import datetime, timedelta, timezone
import io
import logging
import json
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

default_args = {
    'owner': 'quant_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'on_failure_callback': telegram_failure_callback,
    'execution_timeout': timedelta(minutes=20),
}


def _load_valid_dataset(s3_hook, prefix, min_rows=100, dataset_key=None, expected_sha256=None):
    """Tải dataset hợp lệ mới nhất từ MinIO."""
    if dataset_key is not None:
        import hashlib
        raw = s3_hook.read_key(dataset_key, bucket_name=BUCKET_NAME)
        if expected_sha256 and hashlib.sha256(raw.encode('utf-8')).hexdigest() != expected_sha256:
            raise ValueError('Training dataset was overwritten; holdout evaluation cannot be reproduced')
        df = pd.read_csv(io.StringIO(raw))
        if len(df) < min_rows:
            raise ValueError('Recorded training dataset is too small')
        return df
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

def evaluation_temporal_masks(df, hyperparameters):
    """Xác định training và test masks theo metadata đã ghi nhận."""
    from pipeline_data import temporal_masks, rolling_temporal_masks

    if not hyperparameters:
        return temporal_masks(df)

    keys = {'split_policy', 'train_start', 'validation_start', 'test_start', 'test_end'}
    if any(k in hyperparameters for k in keys):
        try:
            train_mask, val_mask, test_mask, _ = rolling_temporal_masks(
                df, split_metadata=hyperparameters
            )
            return train_mask, test_mask
        except Exception as e:
            raise ValueError(f"Incomplete/unknown evaluation metadata: {e}")
    else:
        return temporal_masks(df)


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
    from model_registry import XGBoostModelBuilder

    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    dag_run_conf = getattr(context.get('dag_run'), 'conf', None) or {}
    date_nodash = datetime.now(timezone.utc).strftime("%Y%m%d")

    # 2. Train & Đánh giá (dùng ModelBuilder)
    from model_registry import load_evaluation_model
    import xgboost as xgb
    builder = XGBoostModelBuilder()

    # Load model instead of retraining
    artifacts = load_evaluation_model(s3_hook, BUCKET_NAME, "xgboost", dag_run_conf.get('pipeline_run_id'))
    version = artifacts['model_card'].model_version if artifacts else None
    if not artifacts or not artifacts.get('model_bytes'):
        logging.warning(f"Không tìm thấy artifact XGBoost {version}, bỏ qua đánh giá nhánh này.")
        return

    builder.model = xgb.XGBClassifier()
    builder.model.load_model(bytearray(artifacts['model_bytes']))
    builder.model_card = artifacts['model_card']
    if builder.model_card.hyperparameters.get('dataset_key'):
        df = _load_valid_dataset(s3_hook, "xgboost/", dataset_key=builder.model_card.hyperparameters['dataset_key'],
                                 expected_sha256=builder.model_card.hyperparameters.get('dataset_sha256'))
    else:
        df = _load_valid_dataset(s3_hook, "xgboost/xgboost_stock_20tickers_10y_")
    if df is None:
        raise ValueError("❌ Không tìm thấy dataset XGBoost hợp lệ!")

    # 3. Đánh giá chi tiết trên test set
    df_processed, _ = builder.preprocess(
        df, feature_policy=builder.model_card.hyperparameters.get('feature_policy', 'legacy'))
    feature_cols = artifacts['feature_columns']
    if not feature_cols or set(feature_cols) - set(df_processed.columns):
        raise ValueError('Evaluation requires the exact trained feature schema')
    train_mask, test_mask = evaluation_temporal_masks(df_processed, builder.model_card.hyperparameters)

    X_test = df_processed.loc[test_mask, feature_cols].values
    y_test = df_processed.loc[test_mask, 'Target_Std'].values
    y_prob = builder.model.predict_proba(X_test)[:, 1]

    threshold = builder.model_card.hyperparameters.get('decision_threshold', 0.5)
    eval_metrics = evaluate_binary_classifier(y_test, y_prob, "XGBoost", decision_threshold=threshold)
    eval_metrics.update({'status': 'ok', 'evaluation_source': 'trained_model_holdout_v2',
                         'evaluation_date': datetime.now(timezone.utc).isoformat(),
                         'model_version': version, 'prediction_horizon': 1,
                         'test_start': str(df_processed.loc[test_mask, 'Date_Std'].min()),
                         'test_end': str(df_processed.loc[test_mask, 'Date_Std'].max())})

    # 4. Classification Report
    report = generate_classification_report_text(y_test, y_prob, "XGBoost", decision_threshold=threshold)
    logging.info(report)

    # 5. Cross-Validation (TimeSeriesSplit)
    df_train = df_processed[train_mask & df_processed['Target_Std'].notna()]
    X_all = df_train[feature_cols].values
    y_all = df_train['Target_Std'].values

    def xgb_builder_fn(X_tr, y_tr):
        m = xgb.XGBClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.03,
            subsample=0.75, colsample_bytree=0.75, min_child_weight=3,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )
        m.fit(X_tr, y_tr)
        return m

    cv_metrics = run_time_series_cv(X_all, y_all, xgb_builder_fn, n_splits=5, model_name="XGBoost", dates=df_train['Date_Std'])

    # 6. Cập nhật metrics vào model card
    full_metrics = {**builder.model_card.metrics, **eval_metrics, **cv_metrics}
    builder.model_card.metrics = full_metrics

    # 7. Cập nhật evaluation lên MinIO

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
    from model_evaluator import (
        evaluate_binary_classifier,
        generate_classification_report_text,
        save_evaluation_to_minio
    )
    from model_registry import LSTMModelBuilder

    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    dag_run_conf = getattr(context.get('dag_run'), 'conf', None) or {}
    date_nodash = datetime.now(timezone.utc).strftime("%Y%m%d")

    # 2. Train & Đánh giá
    from model_registry import load_evaluation_model
    import pickle
    builder = LSTMModelBuilder()

    # Load model instead of retraining
    artifacts = load_evaluation_model(s3_hook, BUCKET_NAME, "lstm", dag_run_conf.get('pipeline_run_id'))
    version = artifacts['model_card'].model_version if artifacts else None
    if not artifacts or not artifacts.get('model_bytes'):
        logging.warning(f"Không tìm thấy artifact LSTM {version}, bỏ qua đánh giá nhánh này.")
        return

    builder.hyperparams.update(artifacts['model_card'].hyperparameters)
    builder.model = builder._build_model_class()(len(artifacts['feature_columns']))
    builder.model.load_state_dict(torch.load(io.BytesIO(artifacts['model_bytes']), weights_only=True, map_location='cpu'))
    if artifacts.get('scaler_bytes'):
        builder.scaler = pickle.loads(artifacts['scaler_bytes'])
    builder.model_card = artifacts['model_card']
    if builder.model_card.hyperparameters.get('dataset_key'):
        df = _load_valid_dataset(s3_hook, "lstm/", dataset_key=builder.model_card.hyperparameters['dataset_key'],
                                 expected_sha256=builder.model_card.hyperparameters.get('dataset_sha256'))
    else:
        df = _load_valid_dataset(s3_hook, "lstm/lstm_stock_20tickers_10y_")
    if df is None:
        raise ValueError("❌ Không tìm thấy dataset LSTM hợp lệ!")

    # 3. Đánh giá chi tiết trên test set
    df_processed, _ = builder.preprocess(
        df, feature_policy=builder.model_card.hyperparameters.get('feature_policy', 'legacy'))
    feature_cols = artifacts['feature_columns']
    if not feature_cols or set(feature_cols) - set(df_processed.columns):
        raise ValueError('Evaluation requires the exact trained feature schema')
    lookback = builder.hyperparams['lookback']
    if artifacts.get('scaler_bytes') is None:
        raise ValueError('LSTM evaluation requires its trained scaler')

    train_mask, test_mask = evaluation_temporal_masks(df_processed, builder.model_card.hyperparameters)

    test_df = df_processed.copy()
    test_df[feature_cols] = builder.scaler.transform(test_df[feature_cols].values)

    X_test, y_test = builder._create_sequences(test_df, feature_cols, lookback, mask=test_mask)

    if len(X_test) > 0:
        builder.model.eval()
        from torch.utils.data import TensorDataset, DataLoader
        test_dataset = TensorDataset(torch.tensor(X_test))
        batch_size = builder.hyperparams.get('batch_size', 128)
        if batch_size not in [128, 256]:
            batch_size = 128
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        all_logits = []
        with torch.no_grad():
            for (bx,) in test_loader:
                all_logits.append(builder.model(bx).numpy())
        test_logits = np.concatenate(all_logits)
        y_prob = torch.sigmoid(torch.tensor(test_logits)).numpy()

        threshold = builder.model_card.hyperparameters.get('decision_threshold', 0.5)
        eval_metrics = {**evaluate_binary_classifier(y_test, y_prob, "LSTM", decision_threshold=threshold), 'status': 'ok'}

        report = generate_classification_report_text(y_test, y_prob, "LSTM", decision_threshold=threshold)
        logging.info(report)
    else:
        eval_metrics = {'status': 'unavailable', 'accuracy': None, 'auc_roc': None, 'total_samples': 0}

    test_dates = df_processed.loc[test_mask, 'Date_Std'] if test_mask.any() else test_df['Date_Std']
    eval_metrics.update({'evaluation_source': 'trained_model_holdout_v2',
                         'evaluation_date': datetime.now(timezone.utc).isoformat(),
                         'model_version': version, 'prediction_horizon': 1,
                         'test_start': str(test_dates.min()),
                         'test_end': str(test_dates.max())})

    # 4. Cập nhật model card
    builder.model_card.metrics = {**builder.model_card.metrics, **eval_metrics}

    # 5. Cập nhật evaluation lên MinIO

    save_evaluation_to_minio(s3_hook, BUCKET_NAME, 'lstm', eval_metrics, date_nodash)

    context['ti'].xcom_push(key='lstm_eval', value=json.dumps(eval_metrics, default=str))
    logging.info("✅ Đánh giá LSTM hoàn tất!")


# ==============================================================================
#  TASK 3: ĐÁNH GIÁ VÀ HUẤN LUYỆN ENSEMBLE
# ==============================================================================

def evaluate_ensemble_task(**context):
    """Evaluate issued weighted T+1 signals; fit a separate research Ensemble."""
    from model_evaluator import evaluate_binary_classifier, save_evaluation_to_minio
    from chart_utils import load_price_data_from_minio
    from forecast_history import load_forecast_history, observed_forecasts
    from learned_signals import train_signal_model
    from signal_formulas import ENSEMBLE_SOURCE, ENSEMBLE_FORMULA_VERSION
    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    prices = load_price_data_from_minio(s3_hook, BUCKET_NAME)
    ensemble_history = load_forecast_history(s3_hook, BUCKET_NAME, 'ensemble')
    observed, _, _ = observed_forecasts(ensemble_history, prices, horizon=1)
    current_history = ensemble_history
    if not ensemble_history.empty:
        source = ensemble_history.get('inference_source', pd.Series('', index=ensemble_history.index))
        version = ensemble_history.get('model_version', pd.Series('', index=ensemble_history.index))
        current_history = ensemble_history[source.eq(ENSEMBLE_SOURCE) & version.eq(ENSEMBLE_FORMULA_VERSION)]
    current, pending, invalid = observed_forecasts(current_history, prices, horizon=1)
    ensemble_metrics = {'status': 'unavailable', 'accuracy': None, 'auc_roc': None,
                        'total_samples': 0, 'pending_samples': pending, 'invalid_samples': invalid,
                        'evaluation_source': 'issued_weighted_forecasts_v1', 'ensemble_method': ENSEMBLE_FORMULA_VERSION,
                        'evaluation_date': datetime.now(timezone.utc).isoformat(), 'prediction_horizon': 1}
    if not current.empty and 'prob_num' in current:
        probabilities = pd.to_numeric(current['prob_num'], errors='coerce')
        valid = probabilities.between(0, 1)
        if valid.any():
            ensemble_metrics.update(evaluate_binary_classifier(current.loc[valid, 'actual_up'].to_numpy(),
                                                               probabilities.loc[valid].to_numpy(), 'Ensemble 35/35/30'))
            ensemble_metrics['status'] = 'ok'
    save_evaluation_to_minio(s3_hook, BUCKET_NAME, 'ensemble', ensemble_metrics)
    context['ti'].xcom_push(key='ensemble_eval', value=json.dumps(ensemble_metrics, default=str))
    training = train_signal_model(s3_hook.get_conn(), BUCKET_NAME, 'ensemble', observed)
    context['ti'].xcom_push(key='ensemble_training', value=json.dumps(training, default=str))


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

    ensemble_eval = ti.xcom_pull(key='ensemble_eval', task_ids='evaluate_ensemble')
    if ensemble_eval:
        eval_results['Ensemble Master'] = json.loads(ensemble_eval)

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
    evaluate_ensemble_task(**mock_ctx)

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
    tags=['evaluation', 'metrics', 'xgboost', 'lstm', 'ensemble', 'cross_validation']
) as dag:

    dag.doc_md = """
# 📊 DAG: Đánh Giá Mô Hình Dự Báo Toàn Diện (Scientific Model Evaluation)
---
### 1. Mục Đích & Tiêu Chuẩn Học Thuật (MLOps Evaluation Pipeline)
DAG phụ trách đánh giá khoa học và khách quan hiệu năng của cả 3 mô hình học máy theo tiêu chuẩn công bố khoa học:
- **XGBoost (Tabular):** Đánh giá trên tập test độc lập + TimeSeriesSplit 5-Fold Cross-Validation.
- **LSTM (PyTorch):** Đánh giá chuỗi nến test set và ma trận nhầm lẫn (Confusion Matrix).
- **Ensemble:** Kiểm chứng tín hiệu tổng hợp 35/35/30 theo T+1; huấn luyện thêm mô hình học trọng số để nghiên cứu.

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

    eval_ensemble = PythonOperator(
        task_id='evaluate_ensemble',
        python_callable=evaluate_ensemble_task,
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
    [eval_xgb, eval_lstm, eval_ensemble] >> compare_task >> send_report
