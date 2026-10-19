"""Identical feature definitions for training ingestion and live inference."""
import numpy as np
import pandas as pd

def calculate_technical_indicators(df_stock: pd.DataFrame, df_macro: pd.DataFrame, symbol: str, sector: str, fundamentals: dict = None):
    missing = {'date', 'spy_log_ret', 'qqq_log_ret'} - set(df_macro.columns)
    if missing:
        raise ValueError(f'Missing required macro features: {sorted(missing)}')
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
    df['ret_5'] = close.pct_change(5, fill_method=None)
    df['ret_10'] = close.pct_change(10, fill_method=None)
    df['ret_20'] = close.pct_change(20, fill_method=None)

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
    from pipeline_data import direction_target
    df['Target'] = direction_target(close, close.shift(-1))
    return df

