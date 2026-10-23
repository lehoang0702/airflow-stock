"""
DAG: TRAIN VÀ DỰ BÁO XGBOOST (BẢN TIN ĐẦY ĐỦ - LUÔN HIỂN THỊ TOP RANKING & THỊ TRƯỜNG)
Lưu file vào MinIO: stock-data/xgboost/du_bao_tang_truong_YYYYMMDD.csv
"""

from pipeline_data import prepare_target, clean_features, rolling_temporal_masks, normalize_price_history, add_cross_sectional_features
from datetime import datetime, timedelta, timezone
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

COLUMNS_9 = [
    'ngay_du_bao', 'ma_co_phieu', 'nhom_nganh', 'khuyen_nghi',
    'xac_suat_tang_gia', 'xu_huong_du_kien', 'do_tin_cay',
    'do_chinh_xac_mo_hinh', 'chi_so_auc', 'loi_nhuan_5d_du_kien', 'gia_5d_du_kien'
]

default_args = {
    'owner': 'quant_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'on_failure_callback': telegram_failure_callback,
    'execution_timeout': timedelta(minutes=45),
}

def train_and_predict_xgboost_task(**context):
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, roc_auc_score

    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    try:
        if not s3_hook.check_for_bucket(BUCKET_NAME):
            s3_hook.create_bucket(BUCKET_NAME)
    except Exception:
        pass

    from pipeline_storage import new_model_version, snapshot_training_dataset
    model_version = new_model_version()
    pipeline_run_id = (getattr(context.get('dag_run'), 'conf', None) or {}).get('pipeline_run_id')
    date_nodash = datetime.now(timezone.utc).strftime("%Y%m%d")
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Đọc dataset MinIO (Tự động lọc file hợp lệ > 100 dòng)
    keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix="xgboost/xgboost_stock_100tickers_5y_")
    if not keys:
        keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix="xgboost/xgboost_stock_20tickers_5y_")
    if not keys:
        keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix="xgboost/xgboost_stock_20tickers_10y_")
    if not keys:
        raise FileNotFoundError("❌ Không tìm thấy dataset trên MinIO xgboost/")

    data_key = None
    df = None
    for k in reversed(sorted([key for key in keys if key.endswith('.csv')])):
        try:
            raw_csv = s3_hook.read_key(k, bucket_name=BUCKET_NAME)
            temp_df = pd.read_csv(io.StringIO(raw_csv))
            if len(temp_df) > 100:
                data_key = k
                df = temp_df
                logging.info(f"✅ Đã chọn dataset hợp lệ: {k} ({len(df):,} dòng)")
                break
            else:
                logging.warning(f"⚠️ Bỏ qua file rỗng hoặc quá ít dòng: {k} ({len(temp_df)} dòng)")
        except Exception as e:
            logging.warning(f"⚠️ Lỗi đọc file {k}: {e}")

    if df is None or len(df) < 100:
        raise ValueError("❌ Không tìm thấy dataset hợp lệ (>100 dòng) trên MinIO xgboost/")

    cols_lower = {c.lower(): c for c in df.columns}
    date_col = cols_lower.get('date') or cols_lower.get('datetime') or df.columns[0]
    ticker_col = cols_lower.get('symbol') or cols_lower.get('ticker') or cols_lower.get('ma_co_phieu')
    sector_col = cols_lower.get('sector') or cols_lower.get('nhom_nganh')

    df['Date_Std'] = pd.to_datetime(df[date_col], errors='coerce').dt.tz_localize(None)
    df['Ticker_Std'] = df[ticker_col] if ticker_col else 'UNKNOWN'
    df['Sector_Std'] = df[sector_col] if sector_col else df['Ticker_Std'].map(SECTOR_MAP).fillna('General')

    # Target
    close_col = cols_lower.get('close') or cols_lower.get('adj close') or cols_lower.get('adj_close')
    target_col = cols_lower.get('future_direction_1d') or cols_lower.get('target')
    df = prepare_target(df, target_col, close_col)

    from pipeline_data import add_stationary_features, select_model_features
    df = add_stationary_features(df)
    df, _ = add_cross_sectional_features(df, stationary=True)
    feature_cols = select_model_features(df, stationary=True)
    df[feature_cols] = clean_features(df, feature_cols)

    # Train / Test
    train_mask, val_mask, test_mask, split_metadata = rolling_temporal_masks(df, horizon=1)

    X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, 'Target_Std']
    X_val, y_val = df.loc[val_mask, feature_cols], df.loc[val_mask, 'Target_Std']
    X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, 'Target_Std']

    if y_train.nunique() != 2:
        raise ValueError("y_train does not have exactly 2 classes before tuning.")

    # Tự động tối ưu hóa siêu tham số (Optuna Bayesian Optimization)
    try:
        from hyperparameter_tuner import tune_xgboost_hyperparameters, save_tuned_hyperparameters_to_minio, DEFAULT_XGB_PARAMS
        tuned_params = tune_xgboost_hyperparameters(X_train, y_train, n_trials=30, dates=df.loc[train_mask, 'Date_Std'])
        save_tuned_hyperparameters_to_minio(s3_hook, BUCKET_NAME, tuned_params)
    except Exception as tune_err:
        logging.warning(f"⚠️ Không thể chạy hyperparameter tuning ({tune_err}). Sử dụng bộ tham số mặc định.")
        from hyperparameter_tuner import DEFAULT_XGB_PARAMS
        tuned_params = DEFAULT_XGB_PARAMS.copy()
    from model_training import recency_weights, select_decision_threshold
    recency_half_life = tuned_params.pop('recency_half_life', 0)
    tuned_params['eval_metric'] = 'auc' if y_val.nunique() == 2 else 'logloss'
    tuned_params['early_stopping_rounds'] = 20

    import mlflow
    import mlflow.xgboost
    from mlflow_utils import log_xgboost_model
    import os

    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("MINIO_ROOT_USER", "minioadmin")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get("MINIO_ENDPOINT_URL", "http://minio:9000")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))
    mlflow.set_experiment("Quantum_XGBoost")

    with mlflow.start_run(run_name=f"XGBoost_{date_nodash}"):
        mlflow.log_params(tuned_params)

        model = xgb.XGBClassifier(**tuned_params)
        model.fit(X_train, y_train,
                  sample_weight=recency_weights(df.loc[train_mask, 'Date_Std'], recency_half_life),
                  eval_set=[(X_val, y_val)], verbose=False)

        # Calculate val_auc
        val_preds_prob = model.predict_proba(X_val)[:, 1]
        try:
            val_auc = float(roc_auc_score(y_val, val_preds_prob)) if len(np.unique(y_val)) == 2 else None
        except Exception:
            val_auc = None

        decision_threshold = select_decision_threshold(y_val, val_preds_prob, dates=df.loc[val_mask, 'Date_Std'])
        best_iteration = getattr(model, 'best_iteration', None)

        # Đánh giá Test set — METRICS THỰC SỰ (không clip)
        test_preds_prob = model.predict_proba(X_test)[:, 1]
        test_preds_bin = (test_preds_prob >= decision_threshold).astype(int)
        test_acc = float(accuracy_score(y_test, test_preds_bin))
        try:
            test_auc = float(roc_auc_score(y_test, test_preds_prob)) if len(np.unique(y_test)) == 2 else None
        except Exception:
            test_auc = None

        majority_class = int(y_train.mode()[0]) if len(y_train) > 0 else 0
        baseline_acc = float(accuracy_score(y_test, np.full_like(y_test, majority_class)))

        mlflow.log_metrics({k: v for k, v in {"accuracy": test_acc, "auc": test_auc, "val_auc": val_auc, "best_iteration": best_iteration, "baseline_acc": baseline_acc,
            "accuracy_at_0_5": float(accuracy_score(y_test, test_preds_prob >= 0.5)),
            "decision_threshold": decision_threshold}.items() if v is not None})
        log_xgboost_model(mlflow, model, X_train)

    # Lưu model artifact lên MinIO
    try:
        from model_registry import ModelCard, save_model_to_minio, select_champion_model
        snapshot_key, snapshot_hash = snapshot_training_dataset(s3_hook, BUCKET_NAME, 'xgboost', model_version, raw_csv)
        model_card = ModelCard(
            model_name='XGBoost Gradient Boosting Classifier (AutoML Tuned)',
            model_version=model_version,
            trained_date=today_str,
            model_type='xgboost',
            dataset_size=int(train_mask.sum()),
            num_features=len(feature_cols),
            num_tickers=int(df['Ticker_Std'].nunique()),
            hyperparameters={
                **{k: v for k, v in tuned_params.items() if isinstance(v, (int, float, str))},
                **{k: v for k, v in split_metadata.items() if isinstance(v, (int, float, str))},
                'best_iteration': best_iteration, 'feature_policy': 'stationary_v2',
                'decision_threshold': decision_threshold, 'recency_half_life': recency_half_life,
                'dataset_key': snapshot_key, 'pipeline_run_id': pipeline_run_id, 'dataset_sha256': snapshot_hash
            },
            metrics={'accuracy': test_acc, 'auc_roc': test_auc, 'val_auc': val_auc, 'baseline_acc': baseline_acc,
                     'accuracy_at_0_5': float(accuracy_score(y_test, test_preds_prob >= 0.5))},
            feature_columns=feature_cols
        )
        model_bytes = bytes(model.get_booster().save_raw(raw_format='json'))
        save_model_to_minio(s3_hook, BUCKET_NAME, 'xgboost', model_card, model_bytes)
    except Exception as e:
        raise RuntimeError('Cannot publish XGBoost artifacts; forecasts were not issued') from e

    # === KIỂM ĐỊNH CHẤT LƯỢNG MÔ HÌNH (CHAMPION - CHALLENGER GATE) ===
    champion_status = 'SAFE_MODE'
    champion_version = model_version
    champion_reason = "Đang kiểm định chất lượng mô hình..."
    try:
        decision = select_champion_model(
            s3_hook=s3_hook,
            bucket_name=BUCKET_NAME,
            model_type='xgboost',
            current_version=model_version,
            current_acc=test_acc,
            current_auc=test_auc,
            max_lookback_days=14,
            min_auc=0.45,
            min_acc=0.48
        )
        champion_status = decision['status']
        champion_reason = decision['reason']
        champion_version = decision['selected_version']
        logging.info(f"🏆 Champion Gate: status={champion_status}, version={champion_version}")

        if champion_status == 'FALLBACK':
            artifacts = decision.get('artifacts')
            if artifacts and artifacts.get('model_bytes'):
                import xgboost as xgb
                fallback_model = xgb.XGBClassifier()
                fallback_model.load_model(bytearray(artifacts['model_bytes']))
                model = fallback_model
                test_acc = decision['selected_metrics'].get('accuracy', test_acc)
                test_auc = decision['selected_metrics'].get('auc_roc', test_auc)
                if artifacts.get('feature_columns'):
                    feature_cols = artifacts['feature_columns']
                    logging.info(f"🔄 Đã đồng bộ feature_cols theo fallback model: {len(feature_cols)} features")
                logging.warning(f"🔄 ĐÃ LOAD FALLBACK MODEL ({champion_version}) ĐỂ DỰ BÁO!")
    except Exception as e:
        champion_status = 'SAFE_MODE'
        champion_reason = f"Lỗi trong bước chọn Champion Model: {e}. Kích hoạt SAFE_MODE để bảo vệ an toàn."
        logging.error(f"❌ {champion_reason}")

    # T+5 Return Regression Model
    logging.info("Tạo mô hình XGBoost Regression dự báo T+5...")
    df_reg = df.copy()
    df_reg = df_reg.sort_values(['Date_Std', 'Ticker_Std']).reset_index(drop=True)
    ret_5d_col = cols_lower.get('future_return_5d') or cols_lower.get('return_5d')
    if ret_5d_col and ret_5d_col in df_reg.columns:
        from pipeline_data import simple_return_target
        df_reg['return_5d'] = simple_return_target(df_reg, ret_5d_col)
    elif close_col and close_col in df_reg.columns:
        future_5d = df_reg.groupby('Ticker_Std')[close_col].shift(-5)
        df_reg['return_5d'] = (future_5d / df_reg[close_col]) - 1.0
    else:
        df_reg['return_5d'] = np.nan
        logging.warning('Không có giá hoặc nhãn lợi nhuận T+5 quan sát được; bỏ qua mô hình regression.')

    dates = df_reg['Date_Std']
    outcome_dates_5d = df_reg.groupby('Ticker_Std')['Date_Std'].shift(-5)
    observed_5d = df_reg['return_5d'].notna() & np.isfinite(df_reg['return_5d']) & outcome_dates_5d.notna()

    val_start = pd.Timestamp(split_metadata['validation_start'])
    test_start = pd.Timestamp(split_metadata['test_start'])
    test_end = pd.Timestamp(split_metadata['test_end'])

    train_mask_reg = observed_5d & (dates >= pd.Timestamp(split_metadata['train_start'])) & (dates < val_start) & (outcome_dates_5d < val_start)
    test_mask_reg = observed_5d & (dates >= test_start) & (dates <= test_end) & (outcome_dates_5d <= test_end)

    X_train_reg, y_train_reg = df_reg.loc[train_mask_reg, feature_cols], df_reg.loc[train_mask_reg, 'return_5d']
    X_test_reg, y_test_reg = df_reg.loc[test_mask_reg, feature_cols], df_reg.loc[test_mask_reg, 'return_5d']

    reg_model = xgb.XGBRegressor(
        n_estimators=120, max_depth=3, learning_rate=0.03,
        subsample=0.75, colsample_bytree=0.75, random_state=42, n_jobs=-1
    )
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    if train_mask_reg.sum() > 0:
        reg_model.fit(X_train_reg, y_train_reg)
        if len(X_test_reg):
            test_preds_reg = reg_model.predict(X_test_reg)
            mae = float(mean_absolute_error(y_test_reg, test_preds_reg))
            rmse = float(np.sqrt(mean_squared_error(y_test_reg, test_preds_reg)))
        else:
            mae = rmse = None
    else:
        mae = None
        rmse = None
        logging.warning("⚠️ Không có dữ liệu train cho regression T+5, bỏ qua fit.")
    logging.info(f"T+5 Regression - MAE: {mae}, RMSE: {rmse}")

    # ponytail: persists evaluated artifact, full-data refit needs separate serving artifact/prospective evaluation.
    return_model_version = None
    try:
        reg_model_card = ModelCard(
            model_name='XGBoost Regressor (T+5 Return)',
            model_version=model_version,
            trained_date=today_str,
            model_type='xgboost_return_5d',
            dataset_size=int(train_mask_reg.sum()),
            num_features=len(feature_cols),
            num_tickers=int(df_reg['Ticker_Std'].nunique()),
            hyperparameters={
                'n_estimators': 120, 'max_depth': 3, 'learning_rate': 0.03,
                **split_metadata,
                'prediction_horizon': 5, 'return_policy': 'simple_v2',
                'dataset_key': snapshot_key, 'dataset_sha256': snapshot_hash
            },
            metrics={'mae': mae, 'rmse': rmse},
            feature_columns=feature_cols
        )
        reg_model_bytes = bytes(reg_model.get_booster().save_raw(raw_format='json'))
        save_model_to_minio(s3_hook, BUCKET_NAME, 'xgboost_return_5d', reg_model_card, reg_model_bytes)
        return_model_version = model_version
    except Exception as e:
        logging.warning(f'⚠️ Không thể lưu model registry cho regression: {e}')

    # Dự báo phiên mới nhất
    latest_rows = df.sort_values(['Ticker_Std', 'Date_Std']).groupby('Ticker_Std').last().reset_index()
    X_latest = latest_rows.reindex(columns=feature_cols, fill_value=0.0).fillna(0.0)

    if champion_status == 'SAFE_MODE':
        probs = np.full(len(latest_rows), np.nan)
    else:
        probs = model.predict_proba(X_latest)[:, 1]
    try:
        returns_5d = reg_model.predict(X_latest) if return_model_version else np.full(len(latest_rows), np.nan)
    except Exception:
        returns_5d = np.full(len(latest_rows), np.nan)

    from chart_utils import (
        generate_individual_charts,
        save_charts_zip_minio_and_telegram,
        load_price_data_from_minio,
    )

    df_prices = normalize_price_history(load_price_data_from_minio(s3_hook, BUCKET_NAME))
    if df_prices.empty:
        df_prices = normalize_price_history(df)
        logging.warning('Không có lịch sử giá ngoài hợp lệ; dùng giá quan sát trong dataset XGBoost nếu có.')
    latest_prices = {}
    if not df_prices.empty:
        latest_price_df = df_prices.groupby('Ticker', sort=False).tail(1)
        latest_prices = dict(zip(latest_price_df['Ticker'], latest_price_df['Close']))

    results = []
    for idx, row in latest_rows.iterrows():
        ticker = row['Ticker_Std']
        sector = row['Sector_Std']
        prob_up = float(probs[idx])
        return_5d = float(returns_5d[idx])
        close_price = float(latest_prices.get(ticker, 0.0))
        pred_close_5d = close_price * (1.0 + return_5d) if close_price > 0 and np.isfinite(return_5d) else None

        if champion_status == 'SAFE_MODE':
            rec = "ĐỨNG NGOÀI"
            trend = "Chưa có dự báo đạt chuẩn (Safe Mode)"
            conf = "Không khả dụng (Safe Mode)"
        else:
            # Chuẩn hóa ngưỡng quyết định (Confidence Threshold Filtering)
            # Lọc nhiễu vùng phân vân 0.45 - 0.55 -> ĐỨNG NGOÀI, chỉ MUA khi >= 0.55, BÁN khi <= 0.45
            if prob_up >= 0.55:
                rec = "MUA"
                trend = "Tăng mạnh" if prob_up >= 0.60 else "Tăng tích lũy"
                conf = "Rất cao" if prob_up >= 0.60 else "Cao"
            elif prob_up <= 0.45:
                rec = "BÁN"
                trend = "Giảm mạnh" if prob_up <= 0.40 else "Giảm phân phối"
                conf = "Rất cao" if prob_up <= 0.40 else "Cao"
            else:
                rec = "ĐỨNG NGOÀI"
                trend = "Đi ngang / Lưỡng lự (Sideway)"
                conf = "Trung bình"

        if pred_close_5d is not None and (not np.isfinite(pred_close_5d) or pred_close_5d <= 0):
            pred_close_5d = None

        results.append({
            'ngay_du_bao': today_str,
            'ma_co_phieu': ticker,
            'nhom_nganh': sector,
            'khuyen_nghi': rec,
            'xac_suat_tang_gia': f"{prob_up*100:.2f}%" if np.isfinite(prob_up) else None,
            'prob_num': prob_up,
            'xu_huong_du_kien': trend,
            'do_tin_cay': conf,
            'do_chinh_xac_mo_hinh': f"{test_acc*100:.1f}%",
            'chi_so_auc': str(test_auc) if test_auc is not None else None,
            'loi_nhuan_5d_du_kien': f"{return_5d*100:.2f}%" if np.isfinite(return_5d) else None,
            'gia_5d_du_kien': round(pred_close_5d, 2) if pred_close_5d is not None else None,
            'return_model_version': return_model_version if pred_close_5d is not None else None
        })

    df_full = pd.DataFrame(results)
    df_out = df_full.copy()

    from forecast_history import save_forecast_history
    reference_dates = df.groupby('Ticker_Std')['Date_Std'].max()
    df_out['reference_date'] = df_out['ma_co_phieu'].map(reference_dates).apply(lambda d: pd.Timestamp(d).strftime('%Y-%m-%d'))
    df_out['issued_at'] = datetime.now(timezone.utc).isoformat()
    df_out['prediction_horizon'] = 1
    df_out['pipeline_run_id'] = pipeline_run_id
    df_out['model_version'] = champion_version
    df_out['inference_source'] = 'trained_model_v2'
    if champion_status == 'SAFE_MODE':
        df_out['model_version'] = None
        df_out['inference_source'] = 'unavailable'
    save_forecast_history(s3_hook, BUCKET_NAME, 'xgboost', df_out)

    # Lưu MinIO
    csv_buffer = io.StringIO()
    df_out.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    predict_key = f"xgboost/du_bao_tang_truong_{date_nodash}.csv"

    s3_hook.load_string(
        string_data=csv_buffer.getvalue(),
        key=predict_key,
        bucket_name=BUCKET_NAME,
        replace=True
    )
    logging.info(f"✅ Đã lưu XGBoost lên MinIO: {predict_key}")

    # === ĐỌC DỮ LIỆU GIÁ LỊCH SỬ & TÍNH KẾ HOẠCH GIAO DỊCH ===

    # Tạo nội dung báo cáo Telegram chuẩn chỉnh
    buy_list = df_full[df_full['khuyen_nghi'] == 'MUA'].sort_values('prob_num', ascending=False)
    sell_list = df_full[df_full['khuyen_nghi'] == 'BÁN'].sort_values('prob_num', ascending=True)
    top_5_potential = df_full[df_full['prob_num'].notna()].sort_values('prob_num', ascending=False).head(5)

    n_buy = len(buy_list)
    n_sell = len(sell_list)
    n_sideway = len(df_full) - n_buy - n_sell

    # Header Telegram linh hoạt theo trạng thái Champion Gate
    if champion_status == 'FALLBACK':
        gate_status_line = f"🔄 <b>CHAMPION GATE:</b> Fallback model <code>{champion_version}</code> (Model mới sụt giảm chất lượng)"
    elif champion_status == 'SAFE_MODE':
        gate_status_line = "🚨 <b>CHAMPION GATE:</b> KÍCH HOẠT SAFE MODE (AI tạm dừng khuyến nghị để bảo vệ vốn)"
    else:
        gate_status_line = f"🎯 <b>Hiệu năng Test (Holdout):</b> Acc {test_acc*100:.1f}% | AUC {test_auc} (Champion: {champion_version})"

    msg_lines = [
        "🌲 <b>BẢN TIN DỰ BÁO CỔ PHIẾU (XGBOOST TABULAR)</b>",
        f"📅 <b>Dữ liệu phiên:</b> {today_str}",
        gate_status_line,
        f"📊 <b>Thị trường {len(df_full)} mã:</b> 🟢 Mua: {n_buy} | 🟡 Đứng ngoài/chờ dự báo: {n_sideway} | 🔴 Bán: {n_sell}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ""
    ]

    if not buy_list.empty:
        msg_lines.append("🟢 <b>KHUYẾN NGHỊ MUA & KẾ HOẠCH T+5:</b>")
        for _, r in buy_list.iterrows():
            ticker = r['ma_co_phieu']
            prob_u = float(r['prob_num'])
            cur_p = latest_prices.get(ticker, 0.0)
            msg_lines.append(f"✅ <b>{ticker}</b> ({r['nhom_nganh']}): <b>{r['xac_suat_tang_gia']}</b>")
            if cur_p > 0:
                msg_lines.append('   Giá mục tiêu/cắt lỗ: chỉ hiển thị khi có mô hình đã huấn luyện tương ứng.')
        msg_lines.append("")

    elif not top_5_potential.empty:
        msg_lines.append("⭐ <b>TOP 5 CỔ PHIẾU ĐIỂM KỸ THUẬT CAO NHẤT:</b>")
        for _, r in top_5_potential.iterrows():
            ticker = r['ma_co_phieu']
            prob_u = float(r['prob_num'])
            cur_p = latest_prices.get(ticker, 0.0)
            p_text = f" | Giá: ${cur_p:.2f}" if cur_p > 0 else ""
            msg_lines.append(f"🔹 <b>{ticker}</b> ({r['nhom_nganh']}): <b>{r['xac_suat_tang_gia']}</b> ({r['khuyen_nghi']}){p_text}")
        msg_lines.append("")

    if champion_status == 'SAFE_MODE':
        msg_lines.append('Chưa có dự báo đạt chuẩn; xác suất và xếp hạng cổ phiếu được để trống.')

    if not sell_list.empty:
        msg_lines.append("🔴 <b>CẢNH BÁO BÁN / GIẢM TỶ TRỌNG:</b>")
        for _, r in sell_list.iterrows():
            ticker = r['ma_co_phieu']
            cur_p = latest_prices.get(ticker, 0.0)
            p_text = f" (${cur_p:.2f})" if cur_p > 0 else ""
            msg_lines.append(f"🔻 <b>{ticker}</b>{p_text}: {r['xu_huong_du_kien']} ({r['xac_suat_tang_gia']})")
    elif champion_status != 'SAFE_MODE':
        msg_lines.append("🔴 <b>CẢNH BÁO BÁN:</b> Không có tín hiệu bán tháo.")

    url_msg = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    send_telegram_safe(url_msg, data={'chat_id': TELEGRAM_CHAT_ID, 'text': "\n".join(msg_lines), 'parse_mode': 'HTML'})

    url_doc = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
    files = {'document': (f"du_bao_tang_truong_xgboost_{date_nodash}.csv", csv_bytes, 'text/csv')}
    send_telegram_safe(url_doc, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': '📊 Bảng phân tích chi tiết XGBoost 20 mã'}, files=files)

    # === CHART DỰ BÁO GIÁ CỔ PHIẾU (ZIP) ===
    try:
        if not df_prices.empty:
            chart_files = generate_individual_charts(
                df_price_history=df_prices,
                ticker_predictions=df_full.to_dict('records'),
                model_name='XGBoost',
                today_str=today_str
            )

            # Lưu MinIO + gửi ZIP qua Telegram
            save_charts_zip_minio_and_telegram(
                chart_files=chart_files,
                s3_hook=s3_hook,
                bucket_name=BUCKET_NAME,
                minio_folder=f"xgboost/charts_{date_nodash}/",
                zip_filename=f"charts_xgboost_{date_nodash}.zip",
                telegram_token=TELEGRAM_TOKEN,
                telegram_chat_id=TELEGRAM_CHAT_ID,
                caption=f'🌲 Biểu đồ dự báo giá 20 mã cổ phiếu - XGBoost ({today_str})\n📦 20 tệp PNG trong tệp ZIP',
                send_telegram_fn=send_telegram_safe
            )
        else:
            logging.warning("⚠️ Không có dữ liệu giá để vẽ chart XGBoost")
    except Exception as e:
        logging.warning(f"⚠️ Không thể tạo chart XGBoost: {e}")
        import traceback
        logging.warning(traceback.format_exc())


with DAG(
    dag_id='dag_xgboost_train_and_predict',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['xgboost', 'tabular', 'prediction', 'minio', 'telegram']
) as dag:

    dag.doc_md = r"""
# 🌲 DAG: Huấn Luyện & Dự Báo Mô Hình XGBoost (Tabular)
---
### 1. Tổng Quan & Kiến Trúc Mô Hình
DAG thực hiện quy trình huấn luyện mô hình **Gradient Boosted Decision Trees (GBDT)** trên tập dữ liệu bảng đa biến:
- **Thuật toán:** `XGBClassifier` với `n_estimators=120`, `max_depth=3`, `learning_rate=0.03`, `subsample=0.75`.
- **Target:** Biến nhị phân $y_{t} = 1$ nếu $Close_{t+1} > Close_{t}$, ngược lại $0$.
- **Data Split:** Time-based split (Tập train rolling 3 năm, validation 63 phiên, test 126 phiên) loại bỏ hoàn toàn Look-ahead bias.
- **Metrics thực tế:** Accuracy & AUC-ROC được tính trực tiếp từ tập test (không clip).

### 2. Quản Lý Phiên Bản (Model Registry MLOps)
- **Model Artifact:** `models/xgboost/vYYYYMMDD/model.json` (Serialized GBDT Booster).
- **Metadata Card:** `models/xgboost/vYYYYMMDD/model_card.json` (Dataset size, feature names, hyperparams).
- **Sổ cái phiên bản:** Tự động ghi nhận vào `models/registry/xgboost/`.

### 3. Đầu Ra & Trực Quan Hóa Đa Kênh
- **File dự báo:** `stock-data/xgboost/du_bao_tang_truong_YYYYMMDD.csv`
- **Biểu đồ nến & chỉ báo:** 20 biểu đồ Dual-Panel độc lập, nén `charts_xgboost_YYYYMMDD.zip` lên MinIO.
- **Thông báo:** Gửi bảng phân tích chỉ báo + tệp ZIP đầy đủ 20 biểu đồ qua Telegram Bot.
"""

    train_xgb = PythonOperator(
        task_id='train_and_export_single_forecast',
        python_callable=train_and_predict_xgboost_task,
        provide_context=True,
    )
