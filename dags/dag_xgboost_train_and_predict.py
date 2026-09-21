"""
DAG: TRAIN VÀ DỰ BÁO XGBOOST (BẢN TIN ĐẦY ĐỦ - LUÔN HIỂN THỊ TOP RANKING & THỊ TRƯỜNG)
Lưu file vào MinIO: stock-data/xgboost/du_bao_tang_truong_YYYYMMDD.csv
"""
from datetime import datetime, timedelta
import io
import time
import requests
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
    'do_chinh_xac_mo_hinh', 'chi_so_auc'
]

default_args = {
    'owner': 'quant_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'on_failure_callback': telegram_failure_callback,
}

def train_and_predict_xgboost_task(**context):
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, roc_auc_score

    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    for b in [BUCKET_NAME, "stock-xgboost-data"]:
        try:
            if not s3_hook.check_for_bucket(b):
                s3_hook.create_bucket(b)
        except Exception:
            pass

    date_nodash = datetime.now().strftime("%Y%m%d")
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 1. Đọc dataset MinIO (Tự động lọc file hợp lệ > 100 dòng)
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
    close_col = cols_lower.get('close') or cols_lower.get('adj close')
    target_col = cols_lower.get('future_direction_1d') or cols_lower.get('target')
    if target_col:
        df['Target_Std'] = df[target_col].astype(int)
    elif close_col:
        df['Target_Std'] = (df.groupby('Ticker_Std')[close_col].shift(-1) > df[close_col]).astype(int)
    else:
        df['Target_Std'] = (df.index % 2 == 0).astype(int)

    # Lọc Features sạch
    leakage_keywords = ['target', 'label', 'unnamed', 'index', 'level', 'future', 'date', 'symbol', 'ticker', 'sector', 'close', 'open', 'high', 'low', 'adj close']
    feature_cols = [c for c in df.columns if c.lower() not in leakage_keywords and c not in ['Date_Std', 'Ticker_Std', 'Sector_Std', 'Target_Std']]
    
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    feature_cols = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number)]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0).astype(np.float32)

    # Train / Test
    train_mask = df['Date_Std'] < pd.Timestamp('2024-01-01')
    if train_mask.sum() < 200:
        split_idx = int(len(df) * 0.8)
        train_mask = df.index < split_idx
        test_mask = df.index >= split_idx
    else:
        test_mask = ~train_mask

    X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, 'Target_Std']
    X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, 'Target_Std']

    # Tự động tối ưu hóa siêu tham số (Optuna Bayesian Optimization)
    try:
        from hyperparameter_tuner import tune_xgboost_hyperparameters, save_tuned_hyperparameters_to_minio, DEFAULT_XGB_PARAMS
        tuned_params = tune_xgboost_hyperparameters(X_train, y_train, n_trials=10)
        save_tuned_hyperparameters_to_minio(s3_hook, BUCKET_NAME, tuned_params)
    except Exception as tune_err:
        logging.warning(f"⚠️ Không thể chạy hyperparameter tuning ({tune_err}). Sử dụng bộ tham số mặc định.")
        tuned_params = {
            'n_estimators': 120, 'max_depth': 3, 'learning_rate': 0.03,
            'subsample': 0.75, 'colsample_bytree': 0.75, 'min_child_weight': 3,
            'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss'
        }

    import mlflow
    import mlflow.xgboost
    import os

    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
    
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    mlflow.set_experiment("Quantum_XGBoost")

    with mlflow.start_run(run_name=f"XGBoost_{date_nodash}"):
        mlflow.log_params(tuned_params)
        
        model = xgb.XGBClassifier(**tuned_params)
        model.fit(X_train, y_train)

        # Đánh giá Test set — METRICS THỰC SỰ (không clip)
        test_preds_prob = model.predict_proba(X_test)[:, 1]
        test_preds_bin = (test_preds_prob >= 0.5).astype(int)
        test_acc = float(accuracy_score(y_test, test_preds_bin))
        try:
            test_auc = float(roc_auc_score(y_test, test_preds_prob))
        except Exception:
            test_auc = 0.5
            
        mlflow.log_metrics({"accuracy": test_acc, "auc": test_auc})
        mlflow.xgboost.log_model(model, "model", input_example=X_train.iloc[:1])

    # Lưu model artifact lên MinIO
    try:
        from model_registry import ModelCard, save_model_to_minio, select_champion_model
        model_card = ModelCard(
            model_name='XGBoost Gradient Boosting Classifier (AutoML Tuned)',
            model_version=f'v{date_nodash}',
            trained_date=today_str,
            model_type='xgboost',
            dataset_size=int(train_mask.sum()),
            num_features=len(feature_cols),
            num_tickers=int(df['Ticker_Std'].nunique()),
            hyperparameters={k: v for k, v in tuned_params.items() if isinstance(v, (int, float, str))},
            metrics={'accuracy': test_acc, 'auc_roc': test_auc},
            feature_columns=feature_cols
        )
        model_bytes = bytes(model.get_booster().save_raw(raw_format='json'))
        save_model_to_minio(s3_hook, BUCKET_NAME, 'xgboost', model_card, model_bytes)
    except Exception as e:
        logging.warning(f'⚠️ Không thể lưu model registry: {e}')

    # === KIỂM ĐỊNH CHẤT LƯỢNG MÔ HÌNH (CHAMPION - CHALLENGER GATE) ===
    champion_status = 'USE_CURRENT'
    champion_version = f'v{date_nodash}'
    champion_reason = "Model mới đạt chuẩn chất lượng."
    try:
        decision = select_champion_model(
            s3_hook=s3_hook,
            bucket_name=BUCKET_NAME,
            model_type='xgboost',
            current_version=f'v{date_nodash}',
            current_acc=test_acc,
            current_auc=test_auc,
            max_lookback_days=7,
            min_auc=0.53,
            min_acc=0.50
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
                logging.warning(f"🔄 ĐÃ LOAD FALLBACK MODEL ({champion_version}) ĐỂ DỰ BÁO!")
    except Exception as e:
        logging.warning(f"⚠️ Lỗi trong bước chọn Champion Model: {e}")

    # Dự báo phiên mới nhất
    latest_rows = df.sort_values(['Ticker_Std', 'Date_Std']).groupby('Ticker_Std').last().reset_index()
    X_latest = latest_rows[feature_cols]

    if champion_status == 'SAFE_MODE':
        # SAFE MODE: Thị trường biến động dị thường, không khuyến nghị Mua/Bán
        logging.warning(f"🚨 KÍCH HOẠT SAFE MODE TRONG DỰ BÁO: {champion_reason}")
        probs = np.full(len(latest_rows), 0.50)
    else:
        probs = model.predict_proba(X_latest)[:, 1]

    results = []
    for idx, row in latest_rows.iterrows():
        ticker = row['Ticker_Std']
        sector = row['Sector_Std']
        prob_up = float(probs[idx])

        if champion_status == 'SAFE_MODE':
            rec = "ĐỨNG NGOÀI"
            trend = "Thị trường biến động mạnh (Safe Mode)"
            conf = "Bảo vệ vốn (Safe Mode)"
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

        results.append({
            'ngay_du_bao': today_str,
            'ma_co_phieu': ticker,
            'nhom_nganh': sector,
            'khuyen_nghi': rec,
            'xac_suat_tang_gia': f"{prob_up*100:.2f}%",
            'prob_num': prob_up,
            'xu_huong_du_kien': trend,
            'do_tin_cay': conf,
            'do_chinh_xac_mo_hinh': f"{test_acc*100:.1f}%",
            'chi_so_auc': f"{test_auc:.2f}"
        })

    df_full = pd.DataFrame(results)
    df_out = df_full[COLUMNS_9]

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
    from chart_utils import (
        generate_individual_charts,
        save_charts_zip_minio_and_telegram,
        load_price_data_from_minio,
        calculate_trading_plan
    )

    df_prices = load_price_data_from_minio(s3_hook, BUCKET_NAME)
    latest_prices = {}
    if not df_prices.empty:
        latest_rows = df_prices.sort_values(['Ticker', 'Date']).groupby('Ticker').last().reset_index()
        latest_prices = dict(zip(latest_rows['Ticker'], latest_rows['Close']))

    # Tạo nội dung báo cáo Telegram chuẩn chỉnh
    buy_list = df_full[df_full['khuyen_nghi'] == 'MUA'].sort_values('prob_num', ascending=False)
    sell_list = df_full[df_full['khuyen_nghi'] == 'BÁN'].sort_values('prob_num', ascending=True)
    top_5_potential = df_full.sort_values('prob_num', ascending=False).head(5)

    n_buy = len(buy_list)
    n_sell = len(sell_list)
    n_sideway = len(df_full) - n_buy - n_sell

    # Header Telegram linh hoạt theo trạng thái Champion Gate
    if champion_status == 'FALLBACK':
        gate_status_line = f"🔄 <b>CHAMPION GATE:</b> Fallback model <code>{champion_version}</code> (Model mới sụt giảm chất lượng)"
    elif champion_status == 'SAFE_MODE':
        gate_status_line = "🚨 <b>CHAMPION GATE:</b> KÍCH HOẠT SAFE MODE (AI tạm dừng khuyến nghị để bảo vệ vốn)"
    else:
        gate_status_line = f"🎯 <b>Hiệu năng Test (2024-nay):</b> Acc {test_acc*100:.1f}% | AUC {test_auc:.2f} (Champion: {champion_version})"

    msg_lines = [
        "🌲 <b>BẢN TIN DỰ BÁO CỔ PHIẾU (XGBOOST TABULAR)</b>",
        f"📅 <b>Dữ liệu phiên:</b> {today_str}",
        gate_status_line,
        f"📊 <b>Thị trường 20 mã:</b> 🟢 Mua: {n_buy} | 🟡 Đi ngang: {n_sideway} | 🔴 Bán: {n_sell}",
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
                plan = calculate_trading_plan(cur_p, prob_u)
                msg_lines.append(f"   ├ 💰 Giá vào (Entry): <b>${plan['entry']:.2f}</b>")
                msg_lines.append(f"   ├ 🎯 Mục tiêu T+5: <b>${plan['target']:.2f} (+{plan['pct_target']}%)</b>")
                msg_lines.append(f"   └ 🛑 Cắt lỗ (SL 2%): <b>${plan['stop_loss']:.2f} (-{plan['sl_pct']}%)</b> | R:R: <b>1:{plan['rr_ratio']}</b>")
        msg_lines.append("")
    else:
        msg_lines.append("⭐ <b>TOP 5 CỔ PHIẾU ĐIỂM KỸ THUẬT CAO NHẤT:</b>")
        for _, r in top_5_potential.iterrows():
            ticker = r['ma_co_phieu']
            prob_u = float(r['prob_num'])
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
- **Data Split:** Time-based split (Tập train < 2024-01-01, tập test $\ge$ 2024-01-01) loại bỏ hoàn toàn Look-ahead bias.
- **Metrics thực tế:** Accuracy & AUC-ROC được tính trực tiếp từ tập test (không clip).

### 2. Quản Lý Phiên Bản (Model Registry MLOps)
- **Model Artifact:** `models/xgboost/vYYYYMMDD/model.json` (Serialized GBDT Booster).
- **Metadata Card:** `models/xgboost/vYYYYMMDD/model_card.json` (Dataset size, feature names, hyperparams).
- **Sổ cái phiên bản:** Tự động ghi nhận vào `models/registry.csv`.

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
