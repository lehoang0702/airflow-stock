"""
DAG: CÀO VÀ XỬ LÝ DỮ LIỆU NẾN GIỜ INTRADAY (1H) CHO 20 MÃ CỔ PHIẾU
===================================================================
Khung thời gian siêu mịn: Nến 1 Giờ (interval='1h') trong 730 ngày gần nhất
Lưu trữ chuẩn Big Data trên MinIO:
  - CSV:     stock-xgboost-data/intraday/intraday_1h_20tickers_YYYYMMDD.csv
  - Parquet: stock-xgboost-data/intraday/intraday_1h_20tickers_YYYYMMDD.parquet
Sản sinh tập dữ liệu lớn (>50.000 dòng nến giờ) phục vụ phân tích vi cấu trúc thị trường.
"""

from datetime import datetime, timedelta
import io
import time
import logging
import numpy as np
import pandas as pd
import yfinance as yf

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

from config_shared import TICKERS_META, SECTOR_MAP

try:
    from alert_utils import telegram_failure_callback
except ImportError:
    telegram_failure_callback = None

MINIO_CONN_ID = 'minio_conn'
BUCKET_NAME = 'stock-xgboost-data'

default_args = {
    'owner': 'quant_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=3),
    'on_failure_callback': telegram_failure_callback,
}


def calculate_intraday_indicators(df_raw: pd.DataFrame, symbol: str, sector: str) -> pd.DataFrame:
    """Tính toán các chỉ báo kỹ thuật đặc trưng cho dữ liệu nến 1 giờ."""
    df = df_raw.copy()
    eps = 1e-9

    # Chuẩn hóa cột
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    for col in df.columns:
        if str(col).lower() in ['date', 'datetime', 'index']:
            df.rename(columns={col: 'datetime'}, inplace=True)
            break
    df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]

    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    df = df.sort_values('datetime').reset_index(drop=True)

    df['symbol'] = symbol
    df['sector'] = sector

    # 1. Intraday Log Returns
    df['ret_1h'] = np.log(df['close'] / (df['close'].shift(1) + eps))
    df['ret_4h'] = np.log(df['close'] / (df['close'].shift(4) + eps))
    df['ret_8h'] = np.log(df['close'] / (df['close'].shift(8) + eps))

    # 2. Intraday Trend (EMA 9, 21, 50)
    ema_9 = df['close'].ewm(span=9, adjust=False).mean()
    ema_21 = df['close'].ewm(span=21, adjust=False).mean()
    ema_50 = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_cross_9_21'] = (ema_9 - ema_21) / (ema_21 + eps)
    df['dist_ema_50'] = (df['close'] - ema_50) / (ema_50 + eps)

    # 3. Intraday Momentum (RSI 14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + eps)
    df['rsi_14h'] = 100.0 - (100.0 / (1.0 + rs))

    # 4. Intraday Volatility (Bollinger Bands & Parkinson Vol)
    ma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    df['bb_width_1h'] = (4.0 * std_20) / (ma_20 + eps)
    df['volatility_10h'] = df['ret_1h'].rolling(10).std()

    # 5. Intraday Volume & VWAP Proxy
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    cum_vol = df['volume'].cumsum()
    cum_vol_price = (typical_price * df['volume']).cumsum()
    df['vwap_proxy'] = cum_vol_price / (cum_vol + eps)
    df['dist_from_vwap'] = (df['close'] - df['vwap_proxy']) / (df['vwap_proxy'] + eps)

    # 6. Target nến giờ: Xu hướng 4 giờ tiếp theo (T+4h)
    df['future_ret_4h'] = df['ret_4h'].shift(-4)
    df['target_dir_4h'] = (df['future_ret_4h'] > 0).astype(int)

    return df.dropna().reset_index(drop=True)


def crawl_intraday_pipeline_task(**context):
    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    if not s3_hook.check_for_bucket(BUCKET_NAME):
        s3_hook.create_bucket(BUCKET_NAME)

    date_nodash = datetime.now().strftime("%Y%m%d")
    logging.info("🚀 Bắt đầu cào dữ liệu Intraday 1 Giờ (730 ngày) cho 20 mã cổ phiếu...")

    all_intraday_dfs = []
    failed = []

    for ticker, meta in TICKERS_META.items():
        sector = SECTOR_MAP.get(ticker, meta.get('sector', 'General'))
        logging.info(f"⏳ Cào nến 1h: {ticker} ({meta.get('name')})...")

        raw_df = pd.DataFrame()
        for attempt in range(1, 4):
            try:
                t = yf.Ticker(ticker)
                raw_df = t.history(period="730d", interval="1h", auto_adjust=False)
                if not raw_df.empty and len(raw_df) >= 100:
                    break
            except Exception:
                pass
            try:
                raw_df = yf.download(ticker, period="730d", interval="1h", auto_adjust=False, progress=False)
                if not raw_df.empty and len(raw_df) >= 100:
                    break
            except Exception:
                pass
            time.sleep(1.5 * attempt)

        if raw_df.empty or len(raw_df) < 50:
            logging.warning(f"⚠️ Không lấy được nến 1h cho {ticker}")
            failed.append(ticker)
            continue

        processed_df = calculate_intraday_indicators(raw_df, symbol=ticker, sector=sector)
        all_intraday_dfs.append(processed_df)
        logging.info(f"✅ {ticker}: Đã tạo xong {len(processed_df):,} dòng nến 1 giờ.")

    if not all_intraday_dfs:
        raise ValueError("❌ Không cào được dữ liệu nến 1h của bất kỳ mã nào!")

    combined_intraday = pd.concat(all_intraday_dfs, ignore_index=True)
    combined_intraday = combined_intraday.sort_values(['symbol', 'datetime']).reset_index(drop=True)

    csv_key = f"intraday/intraday_1h_20tickers_{date_nodash}.csv"
    parquet_key = f"intraday/intraday_1h_20tickers_{date_nodash}.parquet"

    logging.info(f"📊 TỔNG KẾT DATASET INTRADAY: {len(combined_intraday):,} dòng nến 1h | {combined_intraday.shape[1]} đặc trưng.")

    # 1. Ghi CSV
    csv_buf = io.StringIO()
    combined_intraday.to_csv(csv_buf, index=False)
    s3_hook.load_string(
        string_data=csv_buf.getvalue(),
        key=csv_key,
        bucket_name=BUCKET_NAME,
        replace=True
    )

    # 2. Ghi Apache Parquet (nén Snappy)
    try:
        pq_buf = io.BytesIO()
        combined_intraday.to_parquet(pq_buf, engine='pyarrow', compression='snappy', index=False)
        s3_hook.load_bytes(
            bytes_data=pq_buf.getvalue(),
            key=parquet_key,
            bucket_name=BUCKET_NAME,
            replace=True
        )
        logging.info(f"📦 Đã xuất thành công file Parquet Intraday: {parquet_key}")
    except Exception as pq_err:
        logging.warning(f"⚠️ Lỗi xuất Parquet Intraday: {pq_err}")

    logging.info(f"🎉 HOÀN TẤT DATASET INTRADAY 1H LÊN MINIO: {csv_key}")


with DAG(
    dag_id='dag_crawl_intraday_features',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['intraday', '1h', 'data_pipeline', 'minio', 'big_data', 'parquet'],
) as dag:

    dag.doc_md = """
# ⏱️ DAG: Cào Dữ Liệu Nến Giờ Intraday (1H) & Lưu Trữ Parquet
---
### 1. Mục Đích & Vai Trò
DAG này mở rộng hệ thống sang **Khung thời gian độ mịn cao (Intraday Microstructure)**:
- Thu thập dữ liệu nến **1 Giờ (`1h`)** cho 20 mã cổ phiếu Blue-chip trong **730 ngày gần nhất** (giới hạn tối đa của Yahoo Finance).
- Tính toán các chỉ báo kỹ thuật đặc thù cho giao dịch ngắn hạn:
  - Hourly Returns (`ret_1h`, `ret_4h`, `ret_8h`)
  - Hourly EMAs (`ema_cross_9_21`, `dist_ema_50`)
  - Hourly RSI 14 (`rsi_14h`)
  - Volume Weighted Average Price (`vwap_proxy`, `dist_from_vwap`)
  - Hourly Volatility (`bb_width_1h`, `volatility_10h`)
- Sản sinh tập dữ liệu lớn **hơn 50.000 – 80.000 dòng**.

### 2. Định Dạng Lưu Trữ Chuẩn Big Data
- **CSV:** `stock-xgboost-data/intraday/intraday_1h_20tickers_YYYYMMDD.csv`
- **Apache Parquet:** `stock-xgboost-data/intraday/intraday_1h_20tickers_YYYYMMDD.parquet` (nén Snappy, tối ưu I/O)
"""

    crawl_intraday = PythonOperator(
        task_id='crawl_intraday_1h_task',
        python_callable=crawl_intraday_pipeline_task,
        provide_context=True,
    )

    crawl_intraday
