"""Research-only context features and independent chronological model selection."""
import numpy as np
import pandas as pd

from config_shared import SECTOR_MAP
from pipeline_data import clean_features


def add_context_features(panel, universe=None):
    """Use observations available at the close; categories are fixed configuration.

    No target, future return, current company fundamental or fitted statistic is
    used. Unknown symbols receive zero identity indicators.
    """
    universe = SECTOR_MAP if universe is None else universe
    data = panel.sort_values(['Ticker_Std', 'Date_Std']).copy()
    groups = data.groupby('Ticker_Std', sort=False)
    previous = groups.Close.shift(1)
    data['ctx_overnight'] = data.Open / previous - 1
    data['ctx_intraday'] = data.Close / data.Open - 1
    data['ctx_close_position'] = (data.Close - data.Low) / (data.High - data.Low).replace(0, np.nan)
    data['ctx_overnight_scaled'] = data.ctx_overnight / data.volatility_20.replace(0, np.nan)
    data['ctx_market_residual'] = data.log_ret - np.log1p(data.spy_ret)
    columns = ['ctx_overnight', 'ctx_intraday', 'ctx_close_position',
               'ctx_overnight_scaled', 'ctx_market_residual']
    for source, windows in [('ctx_market_residual', (5, 20)),
                            ('ctx_overnight', (5,)), ('ctx_intraday', (5,))]:
        for window in windows:
            name = f'{source}_mean_{window}'
            data[name] = data.groupby('Ticker_Std')[source].transform(
                lambda values: values.rolling(window).mean())
            columns.append(name)
    for ticker in sorted(universe):
        name = f'ctx_ticker_{ticker}'
        data[name] = data.Ticker_Std.eq(ticker).astype(float)
        columns.append(name)
    for number, sector in enumerate(sorted(set(universe.values()))):
        name = f'ctx_sector_{number}'
        data[name] = data.Sector.eq(sector).astype(float)
        columns.append(name)
    data[columns] = clean_features(data, columns)
    # Preserve the exact labels, index and outer test cohort of the input.
    return data.loc[panel.index], columns


def independent_stopping_masks(panel, outer_train, stopping_sessions=63, horizon=1):
    """Reserve the train tail for stopping; outer validation stays untouched."""
    if type(stopping_sessions) is not int or stopping_sessions < 5:
        raise ValueError('At least five stopping sessions are required')
    if type(horizon) is not int or horizon < 1:
        raise ValueError('A positive prediction horizon is required')
    if not panel.index.equals(outer_train.index) or panel.index.has_duplicates:
        raise ValueError('Training masks and unique panel indices must align')
    dates = np.sort(panel.loc[outer_train, 'Date_Std'].unique())
    if len(dates) < stopping_sessions + horizon + 60:
        raise ValueError('Insufficient history for independent stopping')
    boundary = pd.Timestamp(dates[-stopping_sessions])
    ordered = panel.sort_values(['Ticker_Std', 'Date_Std'])
    outcomes = ordered.groupby('Ticker_Std').Date_Std.shift(-horizon).reindex(panel.index)
    fit = outer_train & panel.Date_Std.lt(boundary) & outcomes.lt(boundary)
    stop = outer_train & panel.Date_Std.ge(boundary)
    if not fit.any() or not stop.any() or (fit & stop).any():
        raise ValueError('Invalid independent stopping subsets')
    return fit, stop, {'stopping_start': boundary.isoformat(),
                       'fit_sessions': int(panel.loc[fit, 'Date_Std'].nunique()),
                       'stopping_sessions': int(panel.loc[stop, 'Date_Std'].nunique()),
                       'purged_rows': int((outer_train & ~(fit | stop)).sum())}


def select_generalizing_candidate(records, min_auc_gain=.005, max_logloss_increase=.005):
    """Require a validation improvement over control before changing recipe."""
    if not records:
        raise ValueError('A control candidate is required')
    baseline = records[0]['validation']
    if baseline['auc_roc'] is None:
        return 0
    eligible = [0]
    for index, record in enumerate(records[1:], 1):
        metric = record['validation']
        if (metric['auc_roc'] is not None
                and metric['auc_roc'] >= baseline['auc_roc'] + min_auc_gain
                and metric['logloss'] <= baseline['logloss'] + max_logloss_increase):
            eligible.append(index)
    return max(eligible, key=lambda index: (records[index]['validation']['auc_roc'],
                                           -records[index]['validation']['logloss'], -index))
