import io
import time
from datetime import datetime
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
    
    df = df_raw.reset_index().copy()
    for col in df.columns:
        if str(col).lower() in ['date', 'datetime', 'index']:
            df.rename(columns={col: 'date'}, inplace=True)
            break
            
    df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
    if 'adj_close' not in df.columns and 'close' in df.columns:
        df['adj_close'] = df['close']
        
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    return df.sort_values('date').reset_index(drop=True)


def calculate_technical_indicators(df_stock: pd.DataFrame, df_macro: pd.DataFrame, symbol: str, sector: str, fundamentals: dict = None):
    df = df_stock.copy()
    eps = 1e-9

    df['symbol'] = symbol
    df['sector'] = sector
    df['volume_log'] = np.log1p(df['volume'])

    # 1. Multi-timeframe MA Returns
    for p in [5, 10, 20, 50, 100, 200]:
        sma = df['close'].rolling(p).mean()
        df[f'ret_close_sma{p}'] = (df['close'] - sma) / (sma + eps)
        ema = df['close'].ewm(span=p, adjust=False).mean()
        df[f'ret_close_ema{p}'] = (df['close'] - ema) / (ema + eps)

    df['sma_cross_5_20'] = (df['close'].rolling(5).mean() / (df['close'].rolling(20).mean() + eps)) - 1
    df['sma_cross_20_50'] = (df['close'].rolling(20).mean() / (df['close'].rolling(50).mean() + eps)) - 1
    df['sma_cross_50_200'] = (df['close'].rolling(50).mean() / (df['close'].rolling(200).mean() + eps)) - 1

    # 2. Momentum & Oscillators
    for rsi_p in [7, 14, 21]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_p).mean()
        rs = gain / (loss + eps)
        df[f'rsi_{rsi_p}'] = 100 - (100 / (1 + rs))

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = (ema_12 - ema_26) / (df['close'] + eps)
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14 + eps))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['williams_r_14'] = -100 * ((high_14 - df['close']) / (high_14 - low_14 + eps))

    tp = (df['high'] + df['low'] + df['close']) / 3
    df['cci_20'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean()) + eps)
    df['roc_10'] = (df['close'] - df['close'].shift(10)) / (df['close'].shift(10) + eps)

    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    up_m = df['high'] - df['high'].shift(1)
    dn_m = df['low'].shift(1) - df['low']
    plus_di = 100 * (pd.Series(np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)).rolling(14).mean() / (atr_14 + eps))
    minus_di = 100 * (pd.Series(np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)).rolling(14).mean() / (atr_14 + eps))
    df['adx_14'] = (100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + eps))).rolling(14).mean()

    # 3. Volatility Metrics
    bb_mean = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    df['bb_pct_b'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + eps)
    df['bb_width'] = (bb_upper - bb_lower) / (bb_mean + eps)
    df['atr_14_norm'] = atr_14 / (df['close'] + eps)

    kc_upper = bb_mean + (1.5 * atr_14)
    kc_lower = bb_mean - (1.5 * atr_14)
    df['squeeze_on'] = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)

    df['parkinson_vol_20'] = np.sqrt((1 / (4 * np.log(2))) * ((np.log(df['high'] / df['low'])) ** 2).rolling(20).mean())
    df['garman_klass_vol_20'] = np.sqrt((0.5 * (np.log(df['high'] / df['low']) ** 2) - (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2)).rolling(20).mean())

    # 4. Volume & Money Flow
    obv = (np.sign(df['close'].diff()).fillna(0) * df['volume']).cumsum()
    df['obv_diff'] = obv.diff()
    df['volume_sma_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + eps)
    df['volume_zscore_20'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).std() + eps)

    raw_money_flow = tp * df['volume']
    pos_flow = raw_money_flow.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_flow = raw_money_flow.where(tp < tp.shift(1), 0).rolling(14).sum()
    df['mfi_14'] = 100 - (100 / (1 + (pos_flow / (neg_flow + eps))))

    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + eps)
    df['cmf_20'] = (clv * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + eps)

    # 5. Returns, Lags & Stats
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    for lag in range(1, 6):
        df[f'log_return_lag{lag}'] = df['log_return'].shift(lag)
    df['rolling_mean_ret_20'] = df['log_return'].rolling(20).mean()
    df['rolling_std_ret_20'] = df['log_return'].rolling(20).std()
    df['rolling_skew_20'] = df['log_return'].rolling(20).skew()
    df['rolling_kurt_20'] = df['log_return'].rolling(20).kurt()

    high_52w = df['high'].rolling(252).max()
    low_52w = df['low'].rolling(252).min()
    df['dist_52w_high'] = (df['close'] - high_52w) / (high_52w + eps)
    df['dist_52w_low'] = (df['close'] - low_52w) / (low_52w + eps)

    # 6. Price Action & Calendar
    candle_range = df['high'] - df['low'] + eps
    df['high_low_range'] = candle_range / (df['close'] + eps)
    df['gap_open'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + eps)
    df['close_loc_in_range'] = (df['close'] - df['low']) / candle_range
    df['upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / candle_range
    df['lower_shadow_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / candle_range
    df['body_to_range_ratio'] = (df['close'] - df['open']).abs() / candle_range

    df['dow'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)

    # 7. Merge Macro & Cross-Asset
    df = pd.merge(df, df_macro, on='date', how='left')
    df['excess_ret_vs_spy'] = df['log_return'] - df['spy_log_ret']
    df['excess_ret_vs_qqq'] = df['log_return'] - df['qqq_log_ret']
    df['beta_60d_spy'] = df['log_return'].rolling(60).cov(df['spy_log_ret']) / (df['spy_log_ret'].rolling(60).var() + eps)
    df['beta_60d_qqq'] = df['log_return'].rolling(60).cov(df['qqq_log_ret']) / (df['qqq_log_ret'].rolling(60).var() + eps)

    # 8. Inter-Market Dynamics (Vàng, Dầu, Đô la Mỹ, Tín dụng Rủi ro cao)
    if 'gold_log_ret' in df.columns:
        df['excess_ret_vs_gold'] = df['log_return'] - df['gold_log_ret']
        df['corr_gold_30d'] = df['log_return'].rolling(30).corr(df['gold_log_ret']).fillna(0.0)
    if 'oil_log_ret' in df.columns:
        df['excess_ret_vs_oil'] = df['log_return'] - df['oil_log_ret']
    if 'dxy_log_ret' in df.columns:
        df['corr_dxy_30d'] = df['log_return'].rolling(30).corr(df['dxy_log_ret']).fillna(0.0)
    if 'hyg_log_ret' in df.columns and 'tnx_log_ret' in df.columns:
        df['credit_spread_proxy'] = df['hyg_log_ret'] - df['tnx_log_ret']

    # 9. Fundamental Metrics (Chỉ số tài chính cơ bản & định giá)
    if fundamentals:
        for k, v in fundamentals.items():
            df[k] = float(v)

    return df


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

    if 'spy' not in macro_dfs:
        raise ValueError("❌ Không thể tải dữ liệu vĩ mô SPY!")

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

    # Tự động Backfill mã lỗi mạng từ dataset MinIO gần nhất nếu có
    if failed_tickers:
        print(f"⚠️ Có {len(failed_tickers)} mã cào không thành công: {failed_tickers}. Kích hoạt Backfill...")
        try:
            today_str = datetime.now().strftime('%Y%m%d')
            keys = s3_hook.list_keys(bucket_name=MINIO_BUCKET, prefix="xgboost/xgboost_stock_20tickers_10y_")
            valid_prev = [k for k in sorted(keys) if not k.endswith(f"{today_str}.csv")]
            if valid_prev:
                latest_prev_key = valid_prev[-1]
                print(f"🔄 Đang đọc dự phòng từ: {latest_prev_key}")
                raw_prev = s3_hook.read_key(latest_prev_key, bucket_name=MINIO_BUCKET)
                df_prev = pd.read_csv(io.StringIO(raw_prev))
                for ft in list(failed_tickers):
                    sub_prev = df_prev[df_prev['symbol'] == ft].copy()
                    if not sub_prev.empty and len(sub_prev) >= 50:
                        all_dfs.append(sub_prev)
                        print(f"🛡️ Đã bù đắp thành công mã {ft} từ {latest_prev_key} ({len(sub_prev)} dòng).")
                        failed_tickers.remove(ft)
        except Exception as e:
            print(f"⚠️ Lỗi backfill XGBoost: {e}")

    if not all_dfs:
        raise ValueError("❌ Không tải được dữ liệu cho bất kỳ mã nào! Hủy quá trình để không ghi đè file rỗng lên MinIO.")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Chốt chặn an toàn: Bắt buộc đủ 20 mã
    unique_tickers = combined_df['symbol'].nunique()
    if unique_tickers < len(TICKERS):
        missing = set(TICKERS.keys()) - set(combined_df['symbol'].unique())
        raise ValueError(f"❌ Chặn xuất file XGBoost! Chỉ có {unique_tickers}/{len(TICKERS)} mã. Thiếu: {missing}. Kích hoạt retry!")


    print("=== 3. TẠO TARGET BẰNG GROUPBY VÀ DROPNAS DÒNG CUỐI ===")
    combined_df['future_return_1d'] = combined_df.groupby('symbol')['log_return'].shift(-1)
    combined_df['future_direction_1d'] = (combined_df['future_return_1d'] > 0).astype(int)
    combined_df['future_return_5d'] = combined_df.groupby('symbol')['log_return'].shift(-5)
    combined_df['future_direction_5d'] = (combined_df['future_return_5d'] > 0).astype(int)

    # Bỏ NA ở Features nhưng GIỮ LẠI dòng cuối để dự báo (fill target NaN)
    target_cols = ['future_return_1d', 'future_direction_1d', 'future_return_5d', 'future_direction_5d']
    feat_cols = [c for c in combined_df.columns if c not in target_cols]
    clean_df_final = combined_df.dropna(subset=feat_cols).reset_index(drop=True)
    clean_df_final[target_cols] = clean_df_final[target_cols].ffill().bfill().fillna(0)

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

    # === KIỂM ĐỊNH CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY MLOPS) ===
    try:
        from data_validator import validate_xgboost_dataset, save_and_alert_quality_report
        df_val = combined_df.rename(columns={
            'symbol': 'Ticker', 'date': 'Date', 'open': 'Open',
            'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })
        qa_result = validate_xgboost_dataset(df_val, expected_tickers=list(TICKERS.keys()))
        save_and_alert_quality_report(qa_result, s3_hook, MINIO_BUCKET, today_str)
    except Exception as qa_err:
        print(f"⚠️ Không thể chạy QA validation: {qa_err}")


try:
    from alert_utils import telegram_failure_callback
except ImportError:
    telegram_failure_callback = None

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'on_failure_callback': telegram_failure_callback,
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
