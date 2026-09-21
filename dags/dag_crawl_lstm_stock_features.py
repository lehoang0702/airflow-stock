"""
DAG: CÀO DỮ LIỆU & TÍNH FEATURES CHUỖI THỜI GIAN CHO LSTM
(Khắc phục lỗi lệch ngày 2026, tải an toàn ^VIX, ^TNX, SPY, QQQ)
Lưu file vào MinIO: stock-data/lstm/lstm_stock_20tickers_10y_YYYYMMDD.csv
"""
from datetime import datetime, timedelta
import io
import time
import logging
import pandas as pd
import numpy as np

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

MINIO_CONN_ID = 'minio_conn'
BUCKET_NAME = 'stock-data'

try:
    from config_shared import TICKERS_LIST as TICKERS_20, MACRO_TICKERS as MACRO_DICT, SECTOR_MAP
    MACRO_TICKERS = list(MACRO_DICT.keys())
except ImportError:
    from dags.config_shared import TICKERS_LIST as TICKERS_20, MACRO_TICKERS as MACRO_DICT, SECTOR_MAP
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
}

def calculate_technical_features(df):
    """Tính toán 25+ chỉ số kỹ thuật động lượng cho LSTM"""
    df = df.copy()
    close = df['Close']
    open_p = df['Open']
    high = df['High']
    low = df['Low']
    vol = df['Volume']

    # Returns
    df['log_ret'] = np.log(close / close.shift(1))
    df['ret_5'] = close.pct_change(5)
    df['ret_10'] = close.pct_change(10)
    df['ret_20'] = close.pct_change(20)

    # MA ratios
    df['sma_10_ratio'] = close / close.rolling(10).mean()
    df['sma_20_ratio'] = close / close.rolling(20).mean()
    df['sma_50_ratio'] = close / close.rolling(50).mean()

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_pct_b'] = (close - (ma20 - 2 * std20)) / (4 * std20 + 1e-9)
    df['bb_width'] = (4 * std20) / (ma20 + 1e-9)

    # Volatility & Volume
    df['volatility_10'] = df['log_ret'].rolling(10).std()
    df['volatility_20'] = df['log_ret'].rolling(20).std()
    df['vol_sma_ratio'] = vol / (vol.rolling(20).mean() + 1e-9)
    
    # Biến động nến
    df['hl_spread'] = (high - low) / close
    df['co_spread'] = (close - open_p) / close

    # Target: 1 nếu ngày mai tăng, 0 nếu giảm
    df['Target'] = (close.shift(-1) > close).astype(float)
    return df

def crawl_and_extract_lstm_features(**context):
    import yfinance as yf

    date_nodash = datetime.now().strftime("%Y%m%d")
    logging.info("🚀 Bắt đầu cào dữ liệu vĩ mô và 20 mã cổ phiếu (Robust Dual-Engine)...")

    def fetch_stock_raw(ticker, period="15y", retries=3):
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
                    return raw
            except Exception:
                pass
            if attempt < retries:
                time.sleep(2 * attempt)
        return pd.DataFrame()

    # 1. CÀO DỮ LIỆU VĨ MÔ AN TOÀN (Dùng period="15y")
    macro_dfs = {}
    for m_ticker in MACRO_TICKERS:
        try:
            hist = fetch_stock_raw(m_ticker, period="15y")
            if not hist.empty:
                hist = hist.reset_index()
                hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None).dt.normalize()
                clean_name = m_ticker.replace('^', '').replace('=', '').replace('-', '').replace('.', '').lower()
                hist[f'{clean_name}_close'] = hist['Close']
                hist[f'{clean_name}_ret'] = hist['Close'].pct_change()
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
        df_macro = base_macro.sort_values('Date').ffill().bfill()
    else:
        df_macro = pd.DataFrame({'Date': []})

    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)

    # 2. CÀO 20 MÃ CỔ PHIẾU
    all_stock_rows = []
    failed_tickers = []
    for ticker in TICKERS_20:
        try:
            hist = fetch_stock_raw(ticker, period="15y")

            if hist.empty or len(hist) < 50:
                logging.warning(f"⚠️ Dữ liệu {ticker} quá ngắn hoặc rỗng từ Yahoo.")
                failed_tickers.append(ticker)
                continue

            hist = hist.reset_index()
            hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None).dt.normalize()
            hist['Ticker'] = ticker
            hist['Sector'] = SECTOR_MAP.get(ticker, 'General')

            # Tính chỉ số kỹ thuật
            hist_feats = calculate_technical_features(hist)

            # Ghép với vĩ mô
            if not df_macro.empty:
                hist_merged = pd.merge(hist_feats, df_macro, on='Date', how='left')
            else:
                hist_merged = hist_feats

            hist_merged = hist_merged.ffill().bfill().dropna(subset=['Target'])
            all_stock_rows.append(hist_merged)
            logging.info(f"✅ Cổ phiếu {ticker}: Đã tính toán xong {len(hist_merged)} dòng.")
        except Exception as e:
            logging.error(f"❌ Lỗi xử lý mã {ticker}: {str(e)}")
            failed_tickers.append(ticker)

    # NẾU CÓ MÃ BỊ LỖI MẠNG: Tự động Backfill từ dataset gần nhất trên MinIO để không bao giờ bị mất mã
    if failed_tickers:
        logging.warning(f"⚠️ Có {len(failed_tickers)} mã cào không thành công: {failed_tickers}. Đang kích hoạt cơ chế Backfill từ MinIO...")
        try:
            keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix="lstm/lstm_stock_20tickers_10y_")
            valid_prev_keys = [k for k in sorted(keys) if not k.endswith(f"{date_nodash}.csv")]
            if valid_prev_keys:
                latest_prev_key = valid_prev_keys[-1]
                logging.info(f"🔄 Đang đọc dữ liệu dự phòng từ: {latest_prev_key}")
                raw_prev_csv = s3_hook.read_key(latest_prev_key, bucket_name=BUCKET_NAME)
                df_prev = pd.read_csv(io.StringIO(raw_prev_csv))
                for ft in list(failed_tickers):
                    sub_prev = df_prev[df_prev['Ticker'] == ft].copy()
                    if not sub_prev.empty and len(sub_prev) >= 50:
                        all_stock_rows.append(sub_prev)
                        logging.info(f"🛡️ Đã bù đắp thành công mã {ft} từ {latest_prev_key} ({len(sub_prev)} dòng).")
                        failed_tickers.remove(ft)
        except Exception as bf_err:
            logging.warning(f"⚠️ Không thể backfill từ MinIO: {bf_err}")

    if not all_stock_rows:
        raise ValueError("❌ Không thu thập được dữ liệu của bất kỳ mã cổ phiếu nào!")

    df_final = pd.concat(all_stock_rows, ignore_index=True)
    df_final = df_final.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    df_final = df_final.ffill().bfill().fillna(0)

    # Chốt chặn an toàn (Quality Gate Guardrail): Bắt buộc đủ 20 mã mới được ghi đè MinIO
    unique_tickers = df_final['Ticker'].nunique()
    if unique_tickers < len(TICKERS_20):
        missing = set(TICKERS_20) - set(df_final['Ticker'].unique())
        raise ValueError(f"❌ Chặn ghi file! Dataset chỉ có {unique_tickers}/{len(TICKERS_20)} mã. Còn thiếu: {missing}. Kích hoạt retry!")

    # 3. Ghi lên MinIO
    csv_buffer = io.StringIO()
    df_final.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    minio_key = f"lstm/lstm_stock_20tickers_10y_{date_nodash}.csv"

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

    # === KIỂM ĐỊNH CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY MLOPS) ===
    try:
        from data_validator import validate_lstm_dataset, save_and_alert_quality_report
        qa_result = validate_lstm_dataset(df_final, min_lookback=30, expected_tickers=TICKERS_20)
        save_and_alert_quality_report(qa_result, s3_hook, BUCKET_NAME, date_nodash)
    except Exception as qa_err:
        logging.warning(f"⚠️ Không thể chạy QA validation LSTM: {qa_err}")


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
- Thu thập dữ liệu giá nến đầy đủ: `Open`, `High`, `Low`, `Close`, `Volume` cho **20 mã cổ phiếu**.
- Tích hợp 4 biến số vĩ mô: `SPY`, `QQQ`, `^VIX`, `^TNX`.
- Giữ nguyên cột giá gốc `Close` để phục vụ trực quan hóa biểu đồ và kiểm định thực tế sau này.

### 2. Các Đặc Trưng Chuỗi Tính Toán
- **Log Returns & Momentum:** `log_ret`, `ret_5`, `ret_10`, `ret_20`.
- **Độ Biến Động Chuỗi:** Realized Volatility 10 phiên & 20 phiên.
- **Biến Động Nến:** Tỷ lệ bóng nến `hl_spread`, tỷ lệ thân nến `co_spread`.
- **Khối Lượng & Chu kỳ:** Khối lượng chuẩn hóa log, chỉ số dao động RSI, vị thế dải Bollinger.

### 3. Đầu Ra MinIO
- **Artifact:** `stock-data/lstm/lstm_stock_20tickers_10y_YYYYMMDD.csv`
"""

    crawl_task = PythonOperator(
        task_id='crawl_and_extract_lstm_features',
        python_callable=crawl_and_extract_lstm_features,
        provide_context=True,
    )

    crawl_task
