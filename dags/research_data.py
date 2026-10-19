"""Paired, adjusted-price research inputs. Never mutates serving datasets/artifacts."""
import numpy as np
import pandas as pd

from feature_engineering import calculate_technical_features, calculate_technical_indicators
from pipeline_data import (add_stationary_features, add_cross_sectional_features,
                           clean_features, prepare_target, select_model_features)


PRICE_POLICY = 'shared_lstm_adjusted_ohlcv_v1'
MACROS = {'spy': 'spy', 'qqq': 'qqq', 'vix': 'vix', 'tnx': 'tnx',
          'gcf': 'gold', 'clf': 'oil', 'dxynyb': 'dxy', 'hyg': 'hyg'}
CORE = ['log_ret', 'ret_5', 'ret_20', 'sma_20_ratio', 'rsi_14', 'macd_hist_pct',
        'bb_width', 'volatility_20', 'vol_sma_ratio', 'hl_spread', 'co_spread',
        'spy_ret', 'qqq_ret', 'vix_ret', 'cs_rank_ret1', 'cs_rank_vol_ratio']
REGIME = ['sector_return', 'sector_relative_return', 'market_breadth',
          'sector_breadth', 'volatility_ratio', 'relative_volume_change',
          'market_momentum_20', 'market_volatility_20']


def prepare_shared_panel(raw):
    """Recompute both feature families and labels from ONE adjusted OHLCV snapshot.

    Input is the existing LSTM crawler's auto_adjust=True CSV. No independently
    downloaded XGBoost prices, old features or labels are used for model fitting.
    Only observed same-date cross-sectional data and trailing windows are used.
    """
    required = ['Date', 'Ticker', 'Sector', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = set(required) - set(raw)
    if missing:
        raise ValueError(f'Missing shared OHLCV columns: {sorted(missing)}')
    macro_columns = [f'{name}_close' for name in MACROS]
    if set(macro_columns) - set(raw):
        raise ValueError('Shared snapshot requires all eight macro price series')
    data = raw[required + macro_columns].copy()
    data['Date'] = pd.to_datetime(data['Date'], errors='raise', utc=True).dt.tz_localize(None)
    if data[required].isna().any().any() or data.duplicated(['Date', 'Ticker']).any():
        raise ValueError('Null required values or duplicate ticker-date observations')
    sessions = pd.DatetimeIndex(sorted(data.Date.unique()))
    for ticker, group in data.groupby('Ticker'):
        expected = sessions[sessions >= group.Date.min()]
        if len(expected.difference(group.Date)):
            raise ValueError(f'Missing sessions or stale ticker in shared snapshot: {ticker}')
    numeric = ['Open', 'High', 'Low', 'Close', 'Volume'] + macro_columns
    data[numeric] = data[numeric].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(data[numeric].to_numpy()).all():
        raise ValueError('Nonfinite price observations')
    if (data[['Open', 'High', 'Low', 'Close']] <= 0).any().any() or (data.Volume < 0).any():
        raise ValueError('Invalid OHLCV observations')
    # Backfilled tickers must not silently supply a different macro snapshot.
    if data.groupby('Date')[macro_columns].nunique().gt(1).any().any():
        raise ValueError('Inconsistent same-date macro prices across tickers')
    macro = data[['Date'] + macro_columns].drop_duplicates('Date').sort_values('Date')
    macro_x = pd.DataFrame({'date': macro.Date})
    for source, name in MACROS.items():
        price = macro[f'{source}_close'].where(macro[f'{source}_close'] > 0)
        macro[f'{source}_ret'] = price.pct_change(fill_method=None)
        macro_x[f'{name}_log_ret'] = np.log(price / price.shift(1))
    pieces, xgb_columns = [], None
    for ticker, group in data.groupby('Ticker', sort=True):
        group = group.sort_values('Date').reset_index(drop=True)
        lstm = calculate_technical_features(group[required])
        lstm = lstm.merge(macro.drop(columns=macro_columns), on='Date', validate='many_to_one')
        lstm['Date_Std'], lstm['Ticker_Std'] = lstm.Date, lstm.Ticker
        lstm = add_stationary_features(lstm)
        # XGBoost's MACD is already normalized by price: do not divide it again.
        stock = group[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        stock.columns = [column.lower() for column in stock.columns]
        xgb = calculate_technical_indicators(stock, macro_x, ticker, str(group.Sector.iloc[0]))
        columns = select_model_features(xgb)
        xgb_columns = columns
        for column in columns:
            lstm[f'xgb_{column}'] = xgb[column].to_numpy()
        lstm['_history_count'] = np.arange(len(lstm)) + 1
        pieces.append(lstm)
    panel = pd.concat(pieces, ignore_index=True)
    panel = prepare_target(panel, close_col='Close')
    panel, _ = add_cross_sectional_features(panel, stationary=True)
    # Reproduce the XGBoost cross-section using its normalized feature definitions.
    xgb_panel = panel[['Date_Std'] + [f'xgb_{c}' for c in xgb_columns]].rename(
        columns={f'xgb_{c}': c for c in xgb_columns})
    xgb_panel, cross = add_cross_sectional_features(xgb_panel, stationary=True)
    for col in cross:
        panel[f'xgb_{col}'] = xgb_panel[col]
    xgb_columns = [f'xgb_{col}' for col in xgb_columns + cross]
    day_sector = panel.groupby(['Date_Std', 'Sector'])['log_ret']
    panel['sector_return'] = day_sector.transform('mean')
    panel['sector_relative_return'] = panel.log_ret - panel.sector_return
    panel['market_breadth'] = panel.log_ret.gt(0).groupby(panel.Date_Std).transform('mean')
    panel['sector_breadth'] = panel.log_ret.gt(0).groupby([panel.Date_Std, panel.Sector]).transform('mean')
    panel['volatility_ratio'] = panel.volatility_10 / panel.volatility_20.replace(0, np.nan)
    panel['relative_volume_change'] = panel.groupby('Ticker_Std').vol_sma_ratio.diff()
    market = macro.set_index('Date').spy_ret
    panel['market_momentum_20'] = panel.Date_Std.map((1 + market).rolling(20).apply(np.prod, raw=True) - 1)
    panel['market_volatility_20'] = panel.Date_Std.map(market.rolling(20).std())
    lstm_columns = select_model_features(panel.drop(columns=xgb_columns + REGIME + ['_history_count']).copy(), stationary=True)
    groups = {'xgboost': {'full': xgb_columns, 'core': CORE, 'enriched': CORE + REGIME},
              'lstm': {'full': lstm_columns, 'core': CORE, 'enriched': CORE + REGIME}}
    all_columns = sorted(set(xgb_columns + lstm_columns + CORE + REGIME))
    panel[all_columns] = clean_features(panel, all_columns)
    # Warm-up rows remain as LSTM context but cannot be scored or fitted.
    panel.loc[panel._history_count < 253, 'Target_Std'] = np.nan
    return panel, groups


def audit_original_labels(panel, xgboost_raw):
    """Quantify original label mismatches against the canonical adjusted target."""
    old = xgboost_raw.rename(columns={'date': 'Date_Std', 'symbol': 'Ticker_Std',
                                      'future_direction_1d': 'old_target'})
    required = ['Date_Std', 'Ticker_Std', 'old_target']
    if set(required) - set(old):
        raise ValueError('XGBoost snapshot has no auditable T+1 label')
    old = old[required].copy()
    old['Date_Std'] = pd.to_datetime(old.Date_Std, utc=True).dt.tz_localize(None)
    paired = panel[['Date_Std', 'Ticker_Std', 'Target_Std']].merge(
        old, on=['Date_Std', 'Ticker_Std'], how='inner', validate='one_to_one').dropna()
    mismatch = paired[paired.Target_Std.ne(paired.old_target)]
    return {'paired_labels': len(paired), 'mismatched_labels': len(mismatch),
            'mismatch_ratio': len(mismatch) / len(paired) if len(paired) else None,
            'examples': mismatch.head(20).astype({'Date_Std': str}).to_dict('records')}


def walk_forward_splits(panel, folds=3, test_sessions=63, validation_sessions=63,
                        end=None, train_years=3):
    """Nonoverlapping T+1 test windows, purged by actual per-ticker outcome dates."""
    from pipeline_data import rolling_temporal_masks
    for value in (folds, test_sessions, validation_sessions, train_years):
        if type(value) is not int or value < 1:
            raise ValueError('Window sizes must be positive integers')
    dates = np.sort(panel.Date_Std.unique())
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end).to_datetime64()]
    first = len(dates) - 1 - folds * test_sessions
    if first < validation_sessions + 1:
        raise ValueError('Insufficient dates for requested walk-forward windows')
    for fold in range(folds):
        start = first + fold * test_sessions
        val_start = pd.Timestamp(dates[start - validation_sessions])
        metadata = {'split_policy': 'rolling_v1', 'prediction_horizon': 1,
                    'train_start': (val_start - pd.DateOffset(years=train_years)).isoformat(),
                    'validation_start': val_start.isoformat(),
                    'test_start': pd.Timestamp(dates[start]).isoformat(),
                    'test_end': pd.Timestamp(dates[start + test_sessions]).isoformat()}
        if pd.Timestamp(metadata['train_start']) < pd.Timestamp(dates[0]):
            raise ValueError('Insufficient history for complete requested training window')
        yield rolling_temporal_masks(panel, split_metadata=metadata)
