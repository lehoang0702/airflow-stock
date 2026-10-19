"""
DAG: TRAIN PYTORCH LSTM — VALIDATION-ONLY CONFIGURATION SEARCH
Lưu file vào MinIO: stock-data/lstm/du_bao_tang_truong_lstm_YYYYMMDD.csv
"""

import io
import os
import gc
import warnings
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from pipeline_data import prepare_target, clean_features, rolling_temporal_masks, create_sequences, add_cross_sectional_features
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

warnings.filterwarnings('ignore', category=UserWarning)

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
    'retries': 0,
    'execution_timeout': timedelta(minutes=35),
    'on_failure_callback': telegram_failure_callback,
}


def fit_lstm_with_validation(model, train_loader, val_loader, epochs=50, lr=0.001,
                             patience=5, max_grad_norm=1.0, weight_decay=1e-3):
    from model_training import fit_lstm_with_validation as fit
    return fit(model, train_loader, val_loader, epochs=epochs, lr=lr,
               patience=patience, max_grad_norm=max_grad_norm, weight_decay=weight_decay)


def train_and_predict_lstm_task(**context):
    import typing_extensions
    if not hasattr(typing_extensions, 'TypeIs'):
        class _TypeIsMeta(type):
            def __getitem__(self, item): return bool
        class _TypeIs(metaclass=_TypeIsMeta): pass
        typing_extensions.TypeIs = _TypeIs

    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    from torch.utils.data import TensorDataset, DataLoader

    class CalibratedStackedLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim=48, num_layers=2, dropout=0.2):
            super(CalibratedStackedLSTM, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            self.ln = nn.LayerNorm(hidden_dim)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 24),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(24, 1)
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.ln(out[:, -1, :])).squeeze(-1)

    torch.set_num_threads(2)
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

    keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix="lstm/lstm_stock_20tickers_10y_")
    if not keys:
        raise FileNotFoundError("❌ Không tìm thấy dataset trên MinIO lstm/")

    df = None
    for k in reversed(sorted([key for key in keys if key.endswith('.csv')])):
        try:
            raw_csv = s3_hook.read_key(k, bucket_name=BUCKET_NAME)
            temp_df = pd.read_csv(io.StringIO(raw_csv))
            if len(temp_df) > 100:
                df = temp_df
                data_key = k
                logging.info(f"✅ Đã chọn dataset hợp lệ: {k} ({len(df):,} dòng)")
                break
        except Exception as e:
            logging.warning(f"⚠️ Lỗi đọc file {k}: {e}")

    if df is None or len(df) < 100:
        raise ValueError("❌ Không tìm thấy dataset hợp lệ (>100 dòng) trên MinIO lstm/")

    cols_lower = {c.lower(): c for c in df.columns}
    date_col = cols_lower.get('date') or cols_lower.get('datetime') or df.columns[0]
    ticker_col = cols_lower.get('ticker') or cols_lower.get('symbol') or cols_lower.get('ma_co_phieu')
    sector_col = cols_lower.get('sector') or cols_lower.get('nhom_nganh')

    df['Date_Std'] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    df['Ticker_Std'] = df[ticker_col] if ticker_col else 'UNKNOWN'
    df['Sector_Std'] = df[sector_col] if sector_col else df['Ticker_Std'].map(SECTOR_MAP).fillna('General')

    close_col = cols_lower.get('close') or cols_lower.get('adj close') or cols_lower.get('adj_close')
    target_col = cols_lower.get('future_direction_1d') or cols_lower.get('target')
    df = prepare_target(df, target_col, close_col)
    from pipeline_data import add_stationary_features, select_model_features
    from model_training import select_lstm_configuration, select_decision_threshold
    df = add_stationary_features(df)
    df, _ = add_cross_sectional_features(df, stationary=True)
    feature_cols = select_model_features(df, stationary=True)
    df[feature_cols] = clean_features(df, feature_cols)
    df = df.sort_values(['Date_Std', 'Ticker_Std']).reset_index(drop=True)

    lookback = 30

    # --- MASKS WITH rolling_temporal_masks ---
    train_mask, val_mask, test_mask, split_metadata = rolling_temporal_masks(df, horizon=1)

    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, feature_cols].values)

    scaled_df = df.copy()
    scaled_df[feature_cols] = scaler.transform(df[feature_cols].values)

    X_train, y_train = create_sequences(scaled_df, feature_cols, lookback, mask=train_mask)
    X_val, y_val, val_indices = create_sequences(scaled_df, feature_cols, lookback, mask=val_mask, return_indices=True)
    X_test, y_test = create_sequences(scaled_df, feature_cols, lookback, mask=test_mask)

    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Train sequence creation failed, empty sequences.")
    if len(np.unique(y_train)) < 2:
        raise ValueError("Train sequence creation failed, target has less than 2 classes.")
    if not np.isfinite(X_train).all():
        raise ValueError("Non-finite values found in Train sequences.")
    if len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Val/Test sequence creation failed, empty sequences.")
    if not np.isfinite(X_val).all() or not np.isfinite(X_test).all():
        raise ValueError("Non-finite values found in Val/Test sequences.")

    # Seeds
    np.random.seed(42)
    torch.manual_seed(42)

    model, selected_config, validation_probabilities, search_history = select_lstm_configuration(
        CalibratedStackedLSTM, X_train, y_train, X_val, y_val,
        epochs=50, patience=5)
    decision_threshold = select_decision_threshold(
        y_val, validation_probabilities, dates=df.loc[val_indices, 'Date_Std'])
    best_epoch = selected_config['best_epoch']
    epochs_run = selected_config['epochs_run']
    best_val_metric = selected_config['val_metric']

    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=256, shuffle=False
    )

    import mlflow
    import mlflow.pytorch

    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("MINIO_ROOT_USER", "minioadmin")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get("MINIO_ENDPOINT_URL", "http://minio:9000")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))
    mlflow.set_experiment("Quantum_LSTM")

    with mlflow.start_run(run_name=f"LSTM_{date_nodash}"):
        mlflow.log_params({**selected_config, 'lookback': lookback, 'patience': 5,
                           'seed': 42, 'decision_threshold': decision_threshold})
        mlflow.log_dict(search_history, 'validation_search.json')

        model.eval()
        test_preds_prob = []
        test_y = []
        with torch.no_grad():
            for bx, by in test_loader:
                probs = torch.sigmoid(model(bx))
                test_preds_prob.extend(probs.cpu().numpy())
                test_y.extend(by.cpu().numpy())

        test_y = np.array(test_y)
        test_preds_prob = np.array(test_preds_prob)
        test_preds_bin = (test_preds_prob >= decision_threshold).astype(int)

        test_acc = float(accuracy_score(test_y, test_preds_bin))
        try:
            test_auc = float(roc_auc_score(test_y, test_preds_prob)) if len(np.unique(test_y)) == 2 else None
        except Exception:
            test_auc = None

        train_majority_class = float(pd.Series(y_train).mode()[0])
        baseline_acc = float(accuracy_score(test_y, np.full_like(test_y, train_majority_class)))

        mlflow.log_metrics({k: v for k, v in {
            "accuracy": test_acc,
            "auc": test_auc,
            "val_metric": best_val_metric,
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "baseline_acc": baseline_acc,
            "accuracy_at_0_5": float(accuracy_score(test_y, test_preds_prob >= 0.5))
        }.items() if v is not None})

        from mlflow_utils import log_model_compat
        log_model_compat(mlflow.pytorch.log_model, model, input_example=X_train[:1])

    # Lưu model artifact
    try:
        from model_registry import ModelCard, save_model_to_minio
        import pickle as _pkl

        snapshot_key, snapshot_hash = snapshot_training_dataset(s3_hook, BUCKET_NAME, 'lstm', model_version, raw_csv)
        hp = {**selected_config, 'epochs': 50, 'patience': 5, 'lookback': lookback,
              'seed': 42, 'decision_threshold': decision_threshold,
              'feature_policy': 'stationary_v2', 'dataset_key': snapshot_key, 'pipeline_run_id': pipeline_run_id,
              'dataset_sha256': snapshot_hash}
        hp.update(split_metadata)

        model_card = ModelCard(
            model_name='Calibrated Stacked LSTM (PyTorch)',
            model_version=model_version,
            trained_date=today_str,
            model_type='lstm',
            dataset_size=len(X_train),
            num_features=len(feature_cols),
            num_tickers=int(df['Ticker_Std'].nunique()),
            hyperparameters=hp,
            metrics={'accuracy': test_acc, 'auc_roc': test_auc, 'val_metric': best_val_metric, 'baseline_acc': baseline_acc,
                     'accuracy_at_0_5': float(accuracy_score(test_y, test_preds_prob >= 0.5))},
            feature_columns=feature_cols
        )
        model_buf = io.BytesIO()
        torch.save(model.state_dict(), model_buf)
        scaler_bytes = _pkl.dumps(scaler)
        save_model_to_minio(s3_hook, BUCKET_NAME, 'lstm', model_card,
                           model_buf.getvalue(), scaler_bytes)
    except Exception as e:
        raise RuntimeError('Cannot publish LSTM artifacts; forecasts were not issued') from e

    champion_status = 'SAFE_MODE'
    champion_version = model_version
    champion_reason = "Đang kiểm định chất lượng mô hình LSTM..."
    try:
        from model_registry import select_champion_model
        decision = select_champion_model(
            s3_hook=s3_hook,
            bucket_name=BUCKET_NAME,
            model_type='lstm',
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
        logging.info(f"🏆 Champion Gate LSTM: status={champion_status}, version={champion_version}")

        if champion_status == 'FALLBACK':
            artifacts = decision.get('artifacts')
            if artifacts and artifacts.get('model_bytes'):
                from model_registry import LSTMModelBuilder
                fallback_hp = {**LSTMModelBuilder.DEFAULT_HYPERPARAMS,
                               **artifacts['model_card'].hyperparameters}
                lookback = fallback_hp['lookback']
                fallback_builder = LSTMModelBuilder(fallback_hp)
                model = fallback_builder.deserialize_model(
                    artifacts['model_bytes'], len(artifacts['feature_columns']))
                from pipeline_data import add_cross_sectional_features as add_cs
                df, _ = add_cs(df, stationary=fallback_hp.get('feature_policy') == 'stationary_v2')
                model.eval()
                if not artifacts.get('scaler_bytes'):
                    raise ValueError('Fallback LSTM requires its matching trained scaler')
                scaler = _pkl.loads(artifacts['scaler_bytes'])
                if artifacts.get('feature_columns'):
                    feature_cols = artifacts['feature_columns']
                    df[feature_cols] = clean_features(df, feature_cols)
                    logging.info(f"🔄 Đã đồng bộ LSTM feature_cols theo fallback model: {len(feature_cols)} features")
                test_acc = decision['selected_metrics'].get('accuracy', test_acc)
                test_auc = decision['selected_metrics'].get('auc_roc', test_auc)
                logging.warning(f"🔄 ĐÃ LOAD FALLBACK MODEL LSTM ({champion_version}) ĐỂ DỰ BÁO!")
    except Exception as e:
        champion_status = 'SAFE_MODE'
        champion_reason = f"Lỗi trong bước chọn Champion Model LSTM: {e}. Kích hoạt SAFE_MODE để bảo vệ an toàn."
        logging.error(f"❌ {champion_reason}")

    # 5. Dự báo holdout
    latest_results = []
    for ticker, grp in df.groupby('Ticker_Std'):
        sector = grp['Sector_Std'].iloc[-1]
        feats = grp.reindex(columns=feature_cols, fill_value=0.0).fillna(0.0).values
        if len(feats) < lookback:
            continue

        last_seq = scaler.transform(feats[-lookback:])
        seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)

        if champion_status == 'SAFE_MODE':
            rec = "ĐỨNG NGOÀI"
            trend = "Thị trường biến động mạnh (Safe Mode)"
            conf = "Bảo vệ vốn (Safe Mode)"
            prob_up = np.nan
        else:
            with torch.no_grad():
                raw_logit = float(model(seq_tensor).item())
                prob_up = float(torch.sigmoid(torch.tensor(raw_logit)).item())

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

        latest_results.append({
            'ngay_du_bao': today_str,
            'ma_co_phieu': ticker,
            'nhom_nganh': sector,
            'khuyen_nghi': rec,
            'xac_suat_tang_gia': f"{prob_up*100:.2f}%" if np.isfinite(prob_up) else None,
            'prob_num': prob_up,
            'xu_huong_du_kien': trend,
            'do_tin_cay': conf,
            'do_chinh_xac_mo_hinh': f"{test_acc*100:.1f}%",
            'chi_so_auc': str(test_auc) if test_auc is not None else None
        })

    df_full = pd.DataFrame(latest_results)
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
    save_forecast_history(s3_hook, BUCKET_NAME, 'lstm', df_out)

    csv_buffer = io.StringIO()
    df_out.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    predict_key = f"lstm/du_bao_tang_truong_lstm_{date_nodash}.csv"

    s3_hook.load_string(
        string_data=csv_buffer.getvalue(),
        key=predict_key,
        bucket_name=BUCKET_NAME,
        replace=True
    )

    buy_list = df_full[df_full['khuyen_nghi'] == 'MUA'].sort_values('prob_num', ascending=False)
    sell_list = df_full[df_full['khuyen_nghi'] == 'BÁN'].sort_values('prob_num', ascending=True)
    top_5_potential = df_full.sort_values('prob_num', ascending=False).head(5)

    n_buy = len(buy_list)
    n_sell = len(sell_list)
    n_sideway = len(df_full) - n_buy - n_sell

    if champion_status == 'FALLBACK':
        gate_status_line = f"🔄 <b>CHAMPION GATE:</b> Fallback model <code>{champion_version}</code> (Model mới sụt giảm chất lượng)"
    elif champion_status == 'SAFE_MODE':
        gate_status_line = "🚨 <b>CHAMPION GATE:</b> KÍCH HOẠT SAFE MODE (AI tạm dừng khuyến nghị để bảo vệ vốn)"
    else:
        gate_status_line = f"🎯 <b>Hiệu năng Test (Holdout):</b> Acc {test_acc*100:.1f}% | AUC {test_auc} (Champion: {champion_version})"

    msg_lines = [
        "🧠 <b>BẢN TIN DỰ BÁO CỔ PHIẾU (LSTM DEEP LEARNING)</b>",
        f"📅 <b>Dữ liệu phiên:</b> {today_str}",
        gate_status_line,
        f"📊 <b>Thị trường 20 mã (Holdout validation):</b> 🟢 Mua: {n_buy} | 🟡 Đi ngang: {n_sideway} | 🔴 Bán: {n_sell}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ""
    ]

    df_prices = pd.DataFrame({
        'Date': df['Date_Std'],
        'Ticker': df['Ticker_Std'],
        'Close': pd.to_numeric(df[close_col], errors='coerce')
    }).dropna()

    latest_prices = {}
    if not df_prices.empty:
        latest_rows = df_prices.sort_values(['Ticker', 'Date']).groupby('Ticker').last().reset_index()
        latest_prices = dict(zip(latest_rows['Ticker'], latest_rows['Close']))

    if not buy_list.empty:
        msg_lines.append("🟢 <b>TOP KHUYẾN NGHỊ MUA & KẾ HOẠCH T+5:</b>")
        for _, r in buy_list.iterrows():
            ticker = r['ma_co_phieu']
            msg_lines.append(f"✅ <b>{ticker}</b> ({r['nhom_nganh']}): <b>{r['xac_suat_tang_gia']}</b>")
        msg_lines.append("")
    else:
        msg_lines.append("⭐ <b>TOP 5 CỔ PHIẾU ĐIỂM SÓNG NẾN CAO NHẤT:</b>")
        for _, r in top_5_potential.iterrows():
            ticker = r['ma_co_phieu']
            cur_p = latest_prices.get(ticker, 0.0)
            p_text = f" | Giá: ${cur_p:.2f}" if cur_p > 0 else ""
            msg_lines.append(f"🔹 <b>{ticker}</b> ({r['nhom_nganh']}): <b>{r['xac_suat_tang_gia']}</b> ({r['khuyen_nghi']}){p_text}")
        msg_lines.append("")

    if not sell_list.empty:
        msg_lines.append("🔴 <b>CẢNH BÁO BÁN / GIẢM TỶ TRỌNG:</b>")
        for _, r in sell_list.iterrows():
            ticker = r['ma_co_phieu']
            cur_p = latest_prices.get(ticker, 0.0)
            p_text = f" (${cur_p:.2f})" if cur_p > 0 else ""
            msg_lines.append(f"🔻 <b>{ticker}</b>{p_text}: {r['xu_huong_du_kien']} ({r['xac_suat_tang_gia']})")
    else:
        msg_lines.append("🔴 <b>CẢNH BÁO BÁN:</b> Không có cảnh báo bán tháo.")

    url_msg = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    send_telegram_safe(url_msg, data={'chat_id': TELEGRAM_CHAT_ID, 'text': "\n".join(msg_lines), 'parse_mode': 'HTML'})

    url_doc = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
    files = {'document': (f"du_bao_tang_truong_lstm_{date_nodash}.csv", csv_bytes, 'text/csv')}
    send_telegram_safe(url_doc, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': '📊 Bảng phân tích chi tiết LSTM Deep Learning 20 mã'}, files=files)

    try:
        from chart_utils import generate_individual_charts, save_charts_zip_minio_and_telegram
        if not df_prices.empty:
            chart_files = generate_individual_charts(
                df_price_history=df_prices,
                ticker_predictions=df_full.to_dict('records'),
                model_name='LSTM',
                today_str=today_str
            )
            save_charts_zip_minio_and_telegram(
                chart_files=chart_files,
                s3_hook=s3_hook,
                bucket_name=BUCKET_NAME,
                minio_folder=f"lstm/charts_{date_nodash}/",
                zip_filename=f"charts_lstm_{date_nodash}.zip",
                telegram_token=TELEGRAM_TOKEN,
                telegram_chat_id=TELEGRAM_CHAT_ID,
                caption=f'🧠 Biểu đồ dự báo giá 20 mã cổ phiếu - LSTM ({today_str})\n📦 20 tệp PNG trong tệp ZIP',
                send_telegram_fn=send_telegram_safe
            )
    except Exception as e:
        logging.warning(f"⚠️ Không thể tạo chart LSTM: {e}")

    gc.collect()


with DAG(
    dag_id='dag_lstm_train_and_predict',
    default_args=default_args,
    schedule_interval=None,
    max_active_runs=1,
    catchup=False,
    tags=['lstm', 'pytorch', 'prediction', 'minio', 'telegram']
) as dag:
    train_lstm = PythonOperator(
        task_id='train_and_predict_lstm_task',
        python_callable=train_and_predict_lstm_task,
        provide_context=True,
    )
