import io
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

try:
    from config_shared import SECTOR_MAP, TICKERS_META, MACRO_TICKERS
except ImportError:
    from dags.config_shared import SECTOR_MAP, TICKERS_META, MACRO_TICKERS

TICKERS = {
    k: {'name': TICKERS_META.get(k, {}).get('name', k), 'sector': SECTOR_MAP.get(k, 'Unknown')}
    for k in SECTOR_MAP.keys()
}

MINIO_CONN_ID = 'minio_conn'
MINIO_BUCKET = 'stock-data'


def clean_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    
    if any(str(c).lower() in ['date', 'datetime'] for c in df_raw.columns):
        df = df_raw.copy()
    else:
        df = df_raw.reset_index().copy()

    for col in df.columns:
        if str(col).lower() in ['date', 'datetime', 'index']:
            df.rename(columns={col: 'date'}, inplace=True)
            break
            
    df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
    if 'adj_close' not in df.columns and 'close' in df.columns:
        df['adj_close'] = df['close']
        
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    if {'open', 'high', 'low', 'close'}.issubset(df.columns):
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
    return df.sort_values('date').reset_index(drop=True)


from feature_engineering import calculate_technical_indicators


def crawl_process_and_upload_to_minio(**kwargs):
    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    if not s3_hook.check_for_bucket(MINIO_BUCKET):
        s3_hook.create_bucket(MINIO_BUCKET)
    def fetch_stock_raw(ticker, period="15y", retries=3):
        for attempt in range(1, retries + 1):
            try:
                t = yf.Ticker(ticker)
                h = t.history(period=period, auto_adjust=False)
                if not h.empty and len(h) >= 50:
                    return h
            except Exception:
                pass
            try:
                raw = yf.download(ticker, period=period, auto_adjust=False, progress=False)
                if not raw.empty and len(raw) >= 50:
                    return raw
            except Exception:
                pass
            if attempt < retries:
                time.sleep(2 * attempt)
        return pd.DataFrame()

    def fetch_stock_fundamentals(ticker):
        """Trích xuất tự động các chỉ số tài chính cơ bản & định giá từ Yahoo Finance."""
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            return {
                'fund_pe_ratio': float(info.get('trailingPE') or info.get('forwardPE') or 0.0),
                'fund_pb_ratio': float(info.get('priceToBook') or 0.0),
                'fund_roe': float(info.get('returnOnEquity') or 0.0),
                'fund_profit_margin': float(info.get('profitMargins') or 0.0),
                'fund_debt_to_equity': float(info.get('debtToEquity') or 0.0),
                'fund_beta': float(info.get('beta') or 1.0),
                'fund_market_cap_log': float(np.log1p(info.get('marketCap') or 1e9)),
            }
        except Exception:
            return {
                'fund_pe_ratio': 0.0, 'fund_pb_ratio': 0.0, 'fund_roe': 0.0,
                'fund_profit_margin': 0.0, 'fund_debt_to_equity': 0.0,
                'fund_beta': 1.0, 'fund_market_cap_log': 0.0
            }

    print("=== 1. TẢI VÀ XỬ LÝ DỮ LIỆU MACRO & LIÊN THỊ TRƯỜNG (15Y) ===")
    macro_dfs = {}
    for ticker, prefix in MACRO_TICKERS.items():
        raw = fetch_stock_raw(ticker, period="15y")
        if raw.empty:
            continue
        cdf = clean_df(raw)
        cdf[f'{prefix}_log_ret'] = np.log(cdf['close'] / cdf['close'].shift(1))
        macro_dfs[prefix] = cdf[['date', f'{prefix}_log_ret']]

    if not {'spy', 'qqq'}.issubset(macro_dfs):
        raise ValueError('Không thể tải đủ dữ liệu vĩ mô bắt buộc SPY/QQQ')

    df_macro = macro_dfs['spy']
    for prefix in ['qqq', 'vix', 'tnx', 'gold', 'oil', 'dxy', 'hyg']:
        if prefix in macro_dfs:
            df_macro = pd.merge(df_macro, macro_dfs[prefix], on='date', how='left')

    all_dfs = []
    failed_tickers = []
    print("=== 2. TÍNH TOÁN ĐẶC TRƯNG CHO 20 MÃ (15Y + CHỈ SỐ CƠ BẢN) ===")
    for ticker, info in TICKERS.items():
        name = info['name']
        sector = info['sector']
        print(f"-> [{sector}] {ticker} ({name})...")
        df_raw = fetch_stock_raw(ticker, period="15y")
        if df_raw.empty:
            print(f"⚠️ Cảnh báo: {ticker} không có dữ liệu từ Yahoo.")
            failed_tickers.append(ticker)
            continue
        fund_dict = fetch_stock_fundamentals(ticker)
        df_stock = clean_df(df_raw)
        df_processed = calculate_technical_indicators(df_stock, df_macro, symbol=ticker, sector=sector, fundamentals=fund_dict)
        all_dfs.append(df_processed)

    # Recover raw OHLCV, then rebuild all features with the current macro schema.
    from pipeline_storage import load_backup_rows, require_complete_panel
    if failed_tickers:
        backups = load_backup_rows(
            s3_hook, MINIO_BUCKET, 'raw-data/xgboost/stock_prices_', 'symbol',
            failed_tickers, ['date', 'open', 'high', 'low', 'close', 'volume'],
            datetime.now().strftime('%Y%m%d'), exclude_today=False)
        for ticker, raw in backups.items():
            raw['date'] = pd.to_datetime(raw['date']).dt.tz_localize(None)
            if {'open', 'high', 'low', 'close'}.issubset(raw.columns):
                raw['high'] = raw[['high', 'open', 'close']].max(axis=1)
                raw['low'] = raw[['low', 'open', 'close']].min(axis=1)
            all_dfs.append(calculate_technical_indicators(
                raw.sort_values('date').reset_index(drop=True), df_macro, ticker,
                TICKERS[ticker]['sector'], fetch_stock_fundamentals(ticker)))

    if not all_dfs:
        raise ValueError("❌ Không tải được dữ liệu cho bất kỳ mã nào! Hủy quá trình để không ghi đè file rỗng lên MinIO.")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    if {'open', 'high', 'low', 'close'}.issubset(combined_df.columns):
        combined_df['high'] = combined_df[['high', 'open', 'close']].max(axis=1)
        combined_df['low'] = combined_df[['low', 'open', 'close']].min(axis=1)
    require_complete_panel(combined_df, 'symbol', 'date', list(TICKERS))
    from data_validator import validate_xgboost_dataset, save_and_alert_quality_report
    qa_raw = validate_xgboost_dataset(combined_df.rename(columns={
        'symbol': 'Ticker', 'date': 'Date', 'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close', 'volume': 'Volume'}), expected_tickers=list(TICKERS))
    save_and_alert_quality_report(qa_raw, s3_hook, MINIO_BUCKET, datetime.now().strftime('%Y%m%d'))
    if not qa_raw.is_valid:
        raise ValueError(f'XGBoost data quality failed: {qa_raw.errors}')

    # Chốt chặn an toàn: Bắt buộc đủ 20 mã
    unique_tickers = combined_df['symbol'].nunique()
    if unique_tickers < len(TICKERS):
        missing = set(TICKERS.keys()) - set(combined_df['symbol'].unique())
        raise ValueError(f"❌ Chặn xuất file XGBoost! Chỉ có {unique_tickers}/{len(TICKERS)} mã. Thiếu: {missing}. Kích hoạt retry!")


    print("=== 3. TẠO TARGET BẰNG GROUPBY VÀ DROPNAS DÒNG CUỐI ===")
    from pipeline_data import direction_target
    combined_df = combined_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    target_cols = []
    for horizon in (1, 5):
        future_price = combined_df.groupby('symbol')['close'].shift(-horizon)
        future_return = future_price / combined_df['close'] - 1.0
        ret_col = f'future_return_{horizon}d'
        label_col = f'future_direction_{horizon}d'
        combined_df[ret_col] = future_return
        combined_df[label_col] = direction_target(pd.Series(0., index=combined_df.index), future_return)
        target_cols.extend([ret_col, label_col])
    combined_df['return_policy'] = 'simple_v2'
    from pipeline_data import select_model_features
    # Only model inputs need a warm-up window. Optional provider metadata and
    # raw Adj Close/action columns may be absent in a recovered OHLCV archive.
    feat_cols = select_model_features(combined_df, stationary=True) + ['date', 'symbol']
    clean_df_final = combined_df.dropna(subset=feat_cols).reset_index(drop=True)
    require_complete_panel(clean_df_final, 'symbol', 'date', list(TICKERS))

    print("=== 4. TIME-BASED SPLIT VÀ BỎ GIÁ THÔ KHỎI X ===")
    # Phân chia tập dữ liệu
    conditions = [
        clean_df_final['date'] < '2023-01-01',
        (clean_df_final['date'] >= '2023-01-01') & (clean_df_final['date'] < '2024-01-01'),
        clean_df_final['date'] >= '2024-01-01'
    ]
    choices = ['train', 'val', 'test']
    clean_df_final['split_set'] = np.select(conditions, choices, default='train')

    # Bỏ các cột giá thô phi dừng (open, high, low, close, adj_close, volume)
    columns_to_drop = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    clean_df_final.drop(columns=[c for c in columns_to_drop if c in clean_df_final.columns], inplace=True)

    today_str = datetime.now().strftime('%Y%m%d')
    file_name = f"xgboost/xgboost_stock_20tickers_10y_{today_str}.csv"

    print(f"\n=== ĐANG XUẤT 1 FILE CSV DUY NHẤT LÊN MINIO ===")
    print(f"-> Tên file: {file_name}")
    print(f"-> Tổng số dòng: {len(clean_df_final):,} dòng | Tổng số cột: {clean_df_final.shape[1]} cột")

    # Publish a validated raw archive for subsequent recoveries.
    from pipeline_storage import new_model_version
    raw_columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    s3_hook.load_string(string_data=combined_df[raw_columns].to_csv(index=False),
                        key=f'raw-data/xgboost/stock_prices_{new_model_version()[1:]}.csv',
                        bucket_name=MINIO_BUCKET, replace=False)

    # Đẩy 1 file CSV duy nhất lên MinIO
    csv_buf = io.StringIO()
    clean_df_final.to_csv(csv_buf, index=False)
    s3_hook.load_string(
        string_data=csv_buf.getvalue(),
        key=file_name,
        bucket_name=MINIO_BUCKET,
        replace=True
    )

    # Đẩy thêm định dạng Apache Parquet (nén Snappy) chuẩn Big Data
    try:
        parquet_buf = io.BytesIO()
        clean_df_final.to_parquet(parquet_buf, engine='pyarrow', compression='snappy', index=False)
        parquet_key = file_name.replace('.csv', '.parquet')
        s3_hook.load_bytes(
            bytes_data=parquet_buf.getvalue(),
            key=parquet_key,
            bucket_name=MINIO_BUCKET,
            replace=True
        )
        print(f"📦 Đã xuất song song định dạng Parquet lên MinIO: {parquet_key}")
    except Exception as pq_err:
        print(f"⚠️ Lưu Parquet không thành công (vẫn duy trì CSV): {pq_err}")

    print(">>> ĐÃ GỬI DATASET XGBOOST LÊN MINIO THÀNH CÔNG! <<<")



try:
    from alert_utils import telegram_failure_callback
except ImportError:
    telegram_failure_callback = None

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'on_failure_callback': telegram_failure_callback,
    'execution_timeout': timedelta(minutes=25),
}

with DAG(
    dag_id='dag_crawl_xgboost_stock_features',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['stock', 'xgboost', 'single_csv', 'data_pipeline'],
) as dag:

    dag.doc_md = """
# 📊 DAG: Cào Dữ Liệu & Kỹ Thuật Đặc Trưng Cho XGBoost
---
### 1. Tổng Quan & Vai Trò Trong Hệ Thống
DAG này phụ trách tầng **Data Ingestion & Feature Engineering** cho nhánh mô hình Tabular **XGBoost**:
- Tự động kéo dữ liệu 10 năm của **20 mã cổ phiếu Blue-chips** thuộc 9 nhóm ngành S&P 500.
- Tính toán đồng thời các biến số kinh tế vĩ mô (Macro Benchmarks): `SPY`, `QQQ`, `^VIX`, `^TNX`.
- Sinh ra ma trận hơn **85 đặc trưng kỹ thuật (Technical Features)** có tính chất chuẩn hóa.

### 2. Các Nhóm Đặc Trưng Tính Toán
| Nhóm Chỉ Báo | Chỉ Báo Kỹ Thuật Chi Tiết |
|---|---|
| **Xu Hướng (Trend)** | SMA (5, 10, 20, 50, 200), EMA (9, 21), Golden/Death Cross, MACD & Signal |
| **Động Lượng (Momentum)** | RSI (14 ngày), Stochastic Oscillator (%K, %D), ROC (Rate of Change) |
| **Biến Động (Volatility)** | Bollinger Bands (Upper, Lower, Width, %B), ATR (Average True Range) |
| **Khối Lượng (Volume)** | Volume Moving Average 20 phiên, Volume Ratio, OBV (On-Balance Volume) |
| **Kinh Tế Vĩ Mô (Macro)** | Lợi suất TPCP Mỹ 10Y (^TNX), Chỉ số sợ hãi (^VIX), Tương quan SPY & QQQ |

### 3. Đầu Ra MinIO
- **Artifact:** `stock-data/xgboost/xgboost_stock_20tickers_10y_YYYYMMDD.csv`
"""

    crawl_task = PythonOperator(
        task_id='crawl_and_engineer_features',
        python_callable=crawl_process_and_upload_to_minio,
    )

    crawl_task
