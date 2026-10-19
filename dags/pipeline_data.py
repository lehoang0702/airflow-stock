"""Time-aware data preparation shared by training and evaluation."""
import numpy as np
import pandas as pd


def normalize_price_history(df):
    """Canonical observed prices for lowercase XGBoost and uppercase LSTM CSVs."""
    columns = ['Date', 'Ticker', 'Close']
    empty = pd.DataFrame(columns=columns)
    if df is None or df.empty:
        return empty
    aliases = {str(column).strip().lower().replace(' ', '_'): column for column in df.columns}
    date_col = next((aliases[key] for key in ('date', 'datetime', 'date_std') if key in aliases), None)
    ticker_col = next((aliases[key] for key in ('ticker', 'symbol', 'ma_co_phieu', 'ticker_std') if key in aliases), None)
    close_col = next((aliases[key] for key in ('close', 'adj_close') if key in aliases), None)
    if date_col is None or ticker_col is None or close_col is None:
        return empty
    result = df[[date_col, ticker_col, close_col]].copy()
    result.columns = columns
    result['Date'] = pd.to_datetime(result['Date'], errors='coerce', utc=True).dt.tz_localize(None)
    result['Close'] = pd.to_numeric(result['Close'], errors='coerce')
    result = result.dropna(subset=columns)
    result = result[np.isfinite(result['Close']) & result['Close'].gt(0)]
    result['Ticker'] = result['Ticker'].astype(str).str.strip()
    result = result[result['Ticker'].ne('')]
    return result.sort_values(['Ticker', 'Date'], kind='stable').drop_duplicates(['Ticker', 'Date'], keep='last').reset_index(drop=True)


def direction_target(current, future):
    return (future > current).astype(float).where(future.notna() & current.notna())


def simple_return_target(df, column):
    """Feature-only legacy CSVs contain log returns; new CSVs tag simple returns."""
    stored = pd.to_numeric(df[column], errors='coerce')
    policy = df.get('return_policy', pd.Series('legacy_log_v1', index=df.index))
    if not policy.isin(['simple_v2', 'legacy_log_v1']).all():
        raise ValueError('Unknown return label policy')
    return stored.where(policy.eq('simple_v2'), np.expm1(stored))


def prepare_target(df, target_col=None, close_col=None, horizon=1, min_return_threshold=0.0):
    df = df.sort_values(['Date_Std', 'Ticker_Std']).reset_index(drop=True).copy()
    future_dates = df.groupby('Ticker_Std')['Date_Std'].shift(-horizon)
    if close_col:
        future = df.groupby('Ticker_Std')[close_col].shift(-horizon)
        target = direction_target(df[close_col], future)
        if min_return_threshold > 0:
            ret = (future - df[close_col]).abs() / (df[close_col] + 1e-9)
            target = target.where(ret >= min_return_threshold)
        df['Target_Std'] = target
    elif target_col:
        df['Target_Std'] = pd.to_numeric(df[target_col], errors='coerce').where(future_dates.notna())
    else:
        raise ValueError('Dataset requires prices or observed target labels')
    return df


def clean_features(df, columns):
    values = df[columns].replace([np.inf, -np.inf], np.nan)
    return values.groupby(df['Ticker_Std']).ffill().fillna(0).astype(np.float32)

def add_stationary_features(df):
    """Derive scale-free indicators using only the current observed price."""
    df = df.copy()
    aliases = {str(c).lower().replace(' ', '_'): c for c in df.columns}
    close_col = aliases.get('close') or aliases.get('adj_close')
    if close_col:
        close = pd.to_numeric(df[close_col], errors='coerce').where(lambda s: s > 0)
        for name in ('macd', 'macd_signal', 'macd_hist'):
            if name in aliases:
                df[f'{name}_pct'] = pd.to_numeric(df[aliases[name]], errors='coerce') / close
    return df


def select_model_features(df, stationary=False):
    """One causal feature policy shared by DAGs, builders and evaluation."""
    excluded = {'date_std', 'ticker_std', 'sector_std', 'target_std', 'split_set',
                'open', 'high', 'low', 'close', 'adj_close', 'volume', 'symbol',
                'ticker', 'ma_co_phieu', 'date', 'datetime', 'sector', 'nhom_nganh',
                'return_5d', 'return_policy'}
    if stationary:
        excluded.update({'dividends', 'stock_splits'})
        for name in ('macd', 'macd_signal', 'macd_hist'):
            if f'{name}_pct' in df.columns:
                excluded.add(name)
    features = []
    for col in df.columns:
        name = str(col).lower().replace(' ', '_')
        if (name in excluded or 'unnamed' in name
                or name.startswith(('future', 'target', 'label', 'fund_'))
                or (stationary and name.endswith('_close'))):
            continue
        values = pd.to_numeric(df[col], errors='coerce')
        if values.notna().any():
            df[col] = values
            features.append(col)
    if not features:
        raise ValueError('No observed numeric model features')
    return features


def add_cross_sectional_features(df, stationary=False):
    cs_features = []
    cols_lower = {str(c).lower(): c for c in df.columns}

    date_col = cols_lower.get('date_std') or cols_lower.get('date')
    if not date_col:
        return df, cs_features

    mappings = {
        'ret1': ['log_return', 'ret_1d', 'log_ret'] if stationary else ['log_return', 'ret_1d'],
        'rsi14': ['rsi_14'],
        'vol_ratio': ['volume_sma_ratio', 'vol_sma_ratio'] if stationary else ['volume_sma_ratio'],
        'ret5': ['ret_close_sma5', 'ret_5'],
        'macd_hist': ['macd_hist_pct', 'macd_hist'] if stationary else ['macd_hist']
    }

    for key, cands in mappings.items():
        col = next((cols_lower.get(c) for c in cands if cols_lower.get(c)), None)
        if col:
            rank_col = f'cs_rank_{key}'
            rel_col = f'cs_rel_{key}'
            df[rank_col] = df.groupby(date_col)[col].rank(pct=True)
            df[rel_col] = df[col] - df.groupby(date_col)[col].transform('mean')
            cs_features.extend([rank_col, rel_col])

    return df, cs_features


def inference_panel_features(target, peers, ticker, stationary=False):
    """Rebuild relative features from contemporaneous peers, never a one-stock rank."""
    def canonical(frame, symbol=None):
        frame = frame.copy()
        aliases = {str(c).lower(): c for c in frame.columns}
        date = aliases.get('date_std') or aliases.get('date') or aliases.get('datetime')
        symbol_col = aliases.get('ticker_std') or aliases.get('ticker') or aliases.get('symbol')
        frame['Date_Std'] = pd.to_datetime(frame[date], utc=True).dt.tz_localize(None).dt.normalize()
        frame['Ticker_Std'] = symbol if symbol is not None else frame[symbol_col]
        return add_stationary_features(frame) if stationary else frame
    target = canonical(target, ticker)
    peers = canonical(peers)
    universe = set(peers['Ticker_Std'].dropna())
    required_peers = len(universe - {ticker})
    peers = peers[peers['Date_Std'].isin(target['Date_Std']) & peers['Ticker_Std'].ne(ticker)]
    if peers.duplicated(['Date_Std', 'Ticker_Std']).any():
        raise ValueError('Duplicate observed peers')
    peer_counts = peers.groupby('Date_Std')['Ticker_Std'].nunique()
    target['_inference_row'] = np.arange(len(target))
    combined = pd.concat([peers, target], ignore_index=True)
    combined, columns = add_cross_sectional_features(combined, stationary=stationary)
    result = combined[combined['_inference_row'].notna()].sort_values('_inference_row')
    complete = target['Date_Std'].map(peer_counts).eq(required_peers) & (required_peers > 0)
    for col in columns:
        target[col] = result[col].to_numpy()
        target.loc[~complete, col] = np.nan
    return target.drop(columns='_inference_row')

def rolling_temporal_masks(df, train_years=3, validation_sessions=63, test_sessions=126, horizon=1, split_metadata=None):
    if df.index.has_duplicates:
        raise ValueError("Dataset has duplicate index which causes misalignment")
    if 'Date_Std' not in df.columns or 'Ticker_Std' not in df.columns or 'Target_Std' not in df.columns:
        raise ValueError("Dataset requires Date_Std, Ticker_Std, Target_Std")

    if type(horizon) is bool or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("Horizon must be positive integer")

    if split_metadata is None:
        if type(train_years) is bool or not isinstance(train_years, int) or train_years <= 0:
            raise ValueError("Counts must be positive integers")
        if type(validation_sessions) is bool or not isinstance(validation_sessions, int) or validation_sessions <= 0:
            raise ValueError("Counts must be positive integers")
        if type(test_sessions) is bool or not isinstance(test_sessions, int) or test_sessions <= 0:
            raise ValueError("Counts must be positive integers")

    dates = pd.to_datetime(df['Date_Std'], errors='coerce')
    if dates.isna().any():
        raise ValueError("Invalid dates found")

    if df['Ticker_Std'].isna().any() or (df['Ticker_Std'].astype(str).str.strip() == '').any():
        raise ValueError("Null or empty ticker found")

    if pd.DataFrame({'T': df['Ticker_Std'], 'D': dates}).duplicated().any():
        raise ValueError("Duplicate ticker-date pairs found")

    valid_dates = pd.Series(dates.unique()).sort_values().reset_index(drop=True)

    if split_metadata is not None:
        required = {'split_policy', 'train_start', 'validation_start', 'test_start', 'test_end', 'prediction_horizon'}
        missing = required - set(split_metadata.keys())
        if missing:
            raise ValueError(f"Incomplete split_metadata, missing: {missing}")
        if split_metadata['split_policy'] != 'rolling_v1':
            raise ValueError("Unsupported split_policy")

        train_start = pd.Timestamp(split_metadata['train_start'])
        val_start = pd.Timestamp(split_metadata['validation_start'])
        test_start = pd.Timestamp(split_metadata['test_start'])
        test_end = pd.Timestamp(split_metadata['test_end'])
        horizon = split_metadata['prediction_horizon']

        if pd.isna(train_start) or pd.isna(val_start) or pd.isna(test_start) or pd.isna(test_end):
            raise ValueError("NaT found in split_metadata")

        if type(horizon) is bool or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("Metadata horizon must be positive integer")

        if not (train_start < val_start < test_start <= test_end):
            raise ValueError("Invalid boundaries")

        meta = {k: split_metadata[k] for k in required}
    else:
        if len(valid_dates) < test_sessions + validation_sessions + 1:
            raise ValueError("Insufficient dates for rolling split")

        test_start = valid_dates.iloc[-test_sessions]
        val_start = valid_dates.iloc[-(test_sessions + validation_sessions)]
        train_start = val_start - pd.DateOffset(years=train_years)
        test_end = valid_dates.iloc[-1]

        meta = {
            'split_policy': 'rolling_v1',
            'train_start': train_start.isoformat(),
            'validation_start': val_start.isoformat(),
            'test_start': test_start.isoformat(),
            'test_end': test_end.isoformat(),
            'prediction_horizon': horizon
        }

    orig_idx = df.index
    df_temp = df[['Ticker_Std', 'Target_Std']].copy()
    df_temp['Date_Std'] = dates

    df_temp = df_temp.sort_values(['Ticker_Std', 'Date_Std'])
    df_temp['outcome_date'] = df_temp.groupby('Ticker_Std')['Date_Std'].shift(-horizon)
    df_temp = df_temp.loc[orig_idx]

    dt = df_temp['Date_Std']
    out_dt = df_temp['outcome_date']
    observed = df_temp['Target_Std'].notna()

    train_mask = observed & (dt >= train_start) & (dt < val_start) & (out_dt < val_start)
    val_mask = observed & (dt >= val_start) & (dt < test_start) & (out_dt < test_start)
    test_mask = observed & (dt >= test_start) & (dt <= test_end) & (out_dt <= test_end)

    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Empty labeled subsets")

    return train_mask, val_mask, test_mask, meta


def temporal_masks(df, cutoff='2024-01-01', horizon=1, max_train_years=None, min_return_threshold=0.0):
    dates = df['Date_Std']
    boundary = pd.Timestamp(cutoff)
    if (dates < boundary).sum() < 200:
        unique = dates.dropna().sort_values().unique()
        if len(unique) < 3:
            raise ValueError('Insufficient dates for chronological train/test split')
        boundary = pd.Timestamp(unique[min(int(len(unique) * .8), len(unique) - 1)])

    outcome_dates = df.groupby('Ticker_Std')['Date_Std'].shift(-horizon)
    observed = df['Target_Std'].notna()

    train_cond = (dates < boundary) & (outcome_dates < boundary)

    if max_train_years is not None:
        train_cond &= (dates >= boundary - pd.DateOffset(years=max_train_years))

    if min_return_threshold > 0.0:
        cols_lower = {str(c).lower(): c for c in df.columns}
        ret_col = (cols_lower.get('future_return_1d') or
                   cols_lower.get('future_ret') or
                   cols_lower.get('close_pct_change') or
                   cols_lower.get('ret_1d'))
        if not ret_col:
            close_col = cols_lower.get('close') or cols_lower.get('adj close')
            if close_col:
                future_ret = df.groupby('Ticker_Std')[close_col].shift(-1) / df[close_col] - 1.0
                train_cond &= (future_ret.abs() >= min_return_threshold)
        else:
            train_cond &= (df[ret_col].abs() >= min_return_threshold)

    return observed & train_cond, observed & (dates >= boundary)


def date_folds(dates, n_splits=3):
    from sklearn.model_selection import TimeSeriesSplit
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    if dates.isna().any():
        raise ValueError('Invalid dates in time series split')
    unique = np.sort(dates.unique())
    for train, valid in TimeSeriesSplit(n_splits=n_splits, gap=1).split(unique):
        yield np.flatnonzero(dates.isin(unique[train])), np.flatnonzero(dates.isin(unique[valid]))


def create_sequences(data, columns, lookback, mask=None, return_indices=False):
    X, y, end_indices = [], [], []
    for _, group in data.groupby('Ticker_Std'):
        group = group.sort_values('Date_Std')
        features = group[columns].to_numpy()
        targets = group['Target_Std'].to_numpy()
        indices = group.index.to_numpy()
        for end in range(lookback - 1, len(group)):
            if pd.isna(targets[end]):
                continue
            if mask is not None and not mask.loc[indices[end]]:
                continue
            X.append(features[end - lookback + 1:end + 1])
            y.append(targets[end])
            end_indices.append(indices[end])
    result = (np.asarray(X, dtype=np.float32).reshape(-1, lookback, len(columns)),
              np.asarray(y, dtype=np.float32))
    return (*result, np.asarray(end_indices)) if return_indices else result
