"""
DAG: CÀO DỮ LIỆU & TÍNH FEATURES CHUỖI THỜI GIAN CHO LSTM
(Khắc phục lỗi lệch ngày 2026, tải an toàn ^VIX, ^TNX, SPY, QQQ)
Lưu file vào MinIO: stock-data/lstm/lstm_stock_100tickers_5y_YYYYMMDD.csv
"""
from datetime import datetime, timedelta
import io
import time
import logging
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

MINIO_CONN_ID = 'minio_conn'
BUCKET_NAME = 'stock-data'

try:
    from config_shared import TICKERS_LIST as TICKERS_100, MACRO_TICKERS as MACRO_DICT, SECTOR_MAP, DATASET_HISTORY_PERIOD
    TICKERS_20 = TICKERS_100  # Backward-compatible alias
    MACRO_TICKERS = list(MACRO_DICT.keys())
except ImportError:
    from dags.config_shared import TICKERS_LIST as TICKERS_100, MACRO_TICKERS as MACRO_DICT, SECTOR_MAP, DATASET_HISTORY_PERIOD
    TICKERS_20 = TICKERS_100
    MACRO_TICKERS = list(MACRO_DICT.keys())

try:
    from alert_utils import telegram_failure_callback
except ImportError:
    telegram_failure_callback = None

default_args = {
    'owner': 'quant_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=3),
    'on_failure_callback': telegram_failure_callback,
    'execution_timeout': timedelta(minutes=45),
}

from feature_engineering import calculate_technical_features

def crawl_and_extract_lstm_features(**context):
    import yfinance as yf

    date_nodash = datetime.now().strftime("%Y%m%d")
    logging.info("🚀 Bắt đầu cào dữ liệu vĩ mô và 20 mã cổ phiếu (Robust Dual-Engine)...")

    def fetch_stock_raw(ticker, period=DATASET_HISTORY_PERIOD, retries=3):
        for attempt in range(1, retries + 1):
            try:
                t_obj = yf.Ticker(ticker)
                hist = t_obj.history(period=period, auto_adjust=True)
                if not hist.empty and len(hist) >= 50:
                    return hist
            except Exception:
                pass
            try:
                raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
                if not raw.empty and len(raw) >= 50:
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    return raw
            except Exception:
                pass
            if attempt < retries:
                time.sleep(2 * attempt)
        return pd.DataFrame()

    # 1. CÀO DỮ LIỆU VĨ MÔ AN TOÀN (Dùng period=DATASET_HISTORY_PERIOD)
    macro_dfs = {}
    for m_ticker in MACRO_TICKERS:
        try:
            hist = fetch_stock_raw(m_ticker, period=DATASET_HISTORY_PERIOD)
            if not hist.empty:
                hist = hist.reset_index()
                hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None).dt.normalize()
                clean_name = m_ticker.replace('^', '').replace('=', '').replace('-', '').replace('.', '').lower()
                hist[f'{clean_name}_close'] = hist['Close']
                hist[f'{clean_name}_ret'] = hist['Close'].pct_change(fill_method=None)
                macro_dfs[m_ticker] = hist[['Date', f'{clean_name}_close', f'{clean_name}_ret']].dropna()
                logging.info(f"✅ Vĩ mô {m_ticker}: Đã lấy {len(hist)} phiên.")
            else:
                logging.warning(f"⚠️ Không tải được vĩ mô {m_ticker}")
        except Exception as e:
            logging.warning(f"⚠️ Lỗi vĩ mô {m_ticker}: {str(e)}")

    # Ghép dữ liệu vĩ mô
    if macro_dfs:
        base_macro = list(macro_dfs.values())[0]
        for other_m in list(macro_dfs.values())[1:]:
            base_macro = pd.merge(base_macro, other_m, on='Date', how='outer')
        df_macro = base_macro.sort_values('Date').ffill()
    else:
        df_macro = pd.DataFrame({'Date': []})

    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)

    # 2. CÀO 20 MÃ CỔ PHIẾU
    all_stock_rows = []
    failed_tickers = []
    for ticker in TICKERS_20:
        try:
            hist = fetch_stock_raw(ticker, period=DATASET_HISTORY_PERIOD)

            if hist.empty or len(hist) < 50:
                logging.warning(f"⚠️ Dữ liệu {ticker} quá ngắn hoặc rỗng từ Yahoo.")
                failed_tickers.append(ticker)
                continue

            hist = hist.reset_index()
            hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None).dt.normalize()
            hist['Ticker'] = ticker
            hist['Sector'] = SECTOR_MAP.get(ticker, 'General')

            if all(c in hist.columns for c in ['Open', 'High', 'Low', 'Close']):
                hist['High'] = hist[['High', 'Open', 'Close']].max(axis=1)
                hist['Low'] = hist[['Low', 'Open', 'Close']].min(axis=1)

            # Tính chỉ số kỹ thuật
            hist_feats = calculate_technical_features(hist)

            # Ghép với vĩ mô
            if not df_macro.empty:
                hist_merged = pd.merge(hist_feats, df_macro, on='Date', how='left')
            else:
                hist_merged = hist_feats

            feature_columns = hist_merged.columns.difference(['Target', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            hist_merged[feature_columns] = hist_merged[feature_columns].ffill().fillna(0)
            all_stock_rows.append(hist_merged)
            logging.info(f"✅ Cổ phiếu {ticker}: Đã tính toán xong {len(hist_merged)} dòng.")
        except Exception as e:
            logging.error(f"❌ Lỗi xử lý mã {ticker}: {str(e)}")
            failed_tickers.append(ticker)

    from pipeline_storage import load_backup_rows, require_complete_panel
    if failed_tickers:
        backups = load_backup_rows(
            s3_hook, BUCKET_NAME, 'lstm/lstm_stock_100tickers_5y_', 'Ticker',
            failed_tickers, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], date_nodash)
        if len(backups) < len(failed_tickers):
            still_failed = [t for t in failed_tickers if t not in backups]
            b20 = load_backup_rows(
                s3_hook, BUCKET_NAME, 'lstm/lstm_stock_20tickers_5y_', 'Ticker',
                still_failed, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], date_nodash)
            backups.update(b20)
        if len(backups) < len(failed_tickers):
            still_failed = [t for t in failed_tickers if t not in backups]
            b10 = load_backup_rows(
                s3_hook, BUCKET_NAME, 'lstm/lstm_stock_20tickers_10y_', 'Ticker',
                still_failed, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], date_nodash)
            backups.update(b10)
        for b_df in backups.values():
            if all(c in b_df.columns for c in ['Open', 'High', 'Low', 'Close']):
                b_df['High'] = b_df[['High', 'Open', 'Close']].max(axis=1)
                b_df['Low'] = b_df[['Low', 'Open', 'Close']].min(axis=1)
        all_stock_rows.extend(backups.values())

    if not all_stock_rows:
        raise ValueError("❌ Không thu thập được dữ liệu của bất kỳ mã cổ phiếu nào!")

    df_final = pd.concat(all_stock_rows, ignore_index=True)
    # CSV recovery supplies text dates while live observations use Timestamp.
    # Normalize the entire panel before sorting or creating horizon targets.
    df_final['Date'] = pd.to_datetime(df_final['Date'], errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()
    df_final = df_final.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    if all(c in df_final.columns for c in ['Open', 'High', 'Low', 'Close']):
        df_final['High'] = df_final[['High', 'Open', 'Close']].max(axis=1)
        df_final['Low'] = df_final[['Low', 'Open', 'Close']].min(axis=1)
    require_complete_panel(df_final, 'Ticker', 'Date', TICKERS_20)
    from data_validator import validate_lstm_dataset, save_and_alert_quality_report
    qa_result = validate_lstm_dataset(df_final, min_lookback=30, expected_tickers=TICKERS_20)
    save_and_alert_quality_report(qa_result, s3_hook, BUCKET_NAME, date_nodash)
    if not qa_result.is_valid:
        raise ValueError(f'LSTM data quality failed: {qa_result.errors}')
    from pipeline_data import direction_target
    future = df_final.groupby('Ticker')['Close'].shift(-1)
    df_final['Target'] = direction_target(df_final['Close'], future)

    # Chốt chặn an toàn (Quality Gate Guardrail): Bắt buộc đủ mã mới được ghi đè MinIO
    unique_tickers = df_final['Ticker'].nunique()
    if unique_tickers < len(TICKERS_20):
        missing = set(TICKERS_20) - set(df_final['Ticker'].unique())
        raise ValueError(f"❌ Chặn ghi file! Dataset chỉ có {unique_tickers}/{len(TICKERS_20)} mã. Còn thiếu: {missing}. Kích hoạt retry!")

    # 3. Ghi lên MinIO
    csv_buffer = io.StringIO()
    df_final.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    minio_key = f"lstm/lstm_stock_100tickers_5y_{date_nodash}.csv"

    s3_hook.load_string(
        string_data=csv_buffer.getvalue(),
        key=minio_key,
        bucket_name=BUCKET_NAME,
        replace=True
    )

    # Xuất định dạng Apache Parquet (nén Snappy) chuẩn Big Data
    try:
        parquet_buf = io.BytesIO()
        df_final.to_parquet(parquet_buf, engine='pyarrow', compression='snappy', index=False)
        parquet_key = minio_key.replace('.csv', '.parquet')
        s3_hook.load_bytes(
            bytes_data=parquet_buf.getvalue(),
            key=parquet_key,
            bucket_name=BUCKET_NAME,
            replace=True
        )
        logging.info(f"📦 Đã xuất song song định dạng Parquet cho LSTM: {parquet_key}")
    except Exception as pq_err:
        logging.warning(f"⚠️ Không thể xuất Parquet cho LSTM (vẫn giữ CSV): {pq_err}")

    logging.info(f"🎉 ĐÃ LƯU THÀNH CÔNG DATASET LSTM LÊN MINIO: {minio_key} ({len(df_final)} dòng, {unique_tickers} mã)")



with DAG(
    dag_id='dag_crawl_lstm_stock_features',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['lstm', 'crawl', 'features', 'minio', 'time_series']
) as dag:

    dag.doc_md = """
# 🧠 DAG: Cào Dữ Liệu & Trích Xuất Features Chuỗi Thời Gian (LSTM)
---
### 1. Mục Đích & Vai Trò
Phụ trách chuẩn bị tập dữ liệu chuỗi thời gian nến (OHLCV) và các đặc trưng động lượng (Momentum Sequences) cho mạng nơ-ron **LSTM (Long Short-Term Memory)**:
- Thu thập dữ liệu giá nến đầy đủ: `Open`, `High`, `Low`, `Close`, `Volume` cho **100 mã cổ phiếu**.
- Tích hợp 4 biến số vĩ mô: `SPY`, `QQQ`, `^VIX`, `^TNX`.
- Giữ nguyên cột giá gốc `Close` để phục vụ trực quan hóa biểu đồ và kiểm định thực tế sau này.

### 2. Các Đặc Trưng Chuỗi Tính Toán
- **Log Returns & Momentum:** `log_ret`, `ret_5`, `ret_10`, `ret_20`.
- **Độ Biến Động Chuỗi:** Realized Volatility 10 phiên & 20 phiên.
- **Biến Động Nến:** Tỷ lệ bóng nến `hl_spread`, tỷ lệ thân nến `co_spread`.
- **Khối Lượng & Chu kỳ:** Khối lượng chuẩn hóa log, chỉ số dao động RSI, vị thế dải Bollinger.

### 3. Đầu Ra MinIO
- **Artifact:** `stock-data/lstm/lstm_stock_100tickers_5y_YYYYMMDD.csv`
"""

    crawl_task = PythonOperator(
        task_id='crawl_and_extract_lstm_features',
        python_callable=crawl_and_extract_lstm_features,
        provide_context=True,
    )

    crawl_task
