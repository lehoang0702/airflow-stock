"""Regression checks for chronological labels, inference windows and registry writes."""
import ast
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dags'))
from pipeline_data import prepare_target, clean_features, temporal_masks, date_folds, create_sequences
from model_registry import ModelCard, _update_registry, list_model_versions, XGBoostModelBuilder


def panel(n=20):
    return pd.DataFrame([
        {'Date_Std': date, 'Ticker_Std': ticker, 'Close': i + 10., 'feature': float(i)}
        for ticker in ['A', 'B']
        for i, date in enumerate(pd.date_range('2023-12-20', periods=n))
    ])


def test_latest_labels_remain_unknown_per_ticker():
    df = prepare_target(panel().sample(frac=1, random_state=1), close_col='Close')
    assert df.groupby('Ticker_Std').tail(1)['Target_Std'].isna().all()
    assert df['Target_Std'].dropna().eq(1).all()
    assert df['Date_Std'].is_monotonic_increasing


def test_legacy_filled_targets_are_not_scored_on_last_day():
    df = panel().drop(columns='Close').assign(label=0)
    result = prepare_target(df, target_col='label')
    assert result.groupby('Ticker_Std').tail(1)['Target_Std'].isna().all()


def test_lstm_window_ends_on_target_date_and_excludes_unknown():
    df = prepare_target(panel(5), close_col='Close')
    X, y = create_sequences(df, ['feature'], 3)
    assert X.shape == (4, 3, 1)
    np.testing.assert_array_equal(X[:, :, 0], [[0, 1, 2], [1, 2, 3]] * 2)
    np.testing.assert_array_equal(y, [1, 1, 1, 1])


def test_lstm_accepts_exactly_one_complete_window():
    df = panel(3).query("Ticker_Std == 'A'").assign(Target_Std=[0., 0., 1.])
    X, y = create_sequences(df, ['feature'], 3)
    assert X.shape == (1, 3, 1)
    assert y.tolist() == [1.]


def test_feature_fill_never_uses_other_ticker_or_future():
    df = pd.DataFrame({'Ticker_Std': ['A', 'B', 'A', 'B'], 'f': [np.nan, 9., 2., np.nan]})
    assert clean_features(df, ['f'])['f'].tolist() == [0., 9., 2., 9.]


def test_temporal_split_excludes_unknown_and_boundary_outcome():
    df = prepare_target(panel(300), close_col='Close')
    train, test = temporal_masks(df, '2024-06-01')
    assert df.loc[train, 'Date_Std'].max() < pd.Timestamp('2024-05-31')
    assert df.loc[test, 'Date_Std'].min() == pd.Timestamp('2024-06-01')
    assert df.loc[train | test, 'Target_Std'].notna().all()


def test_folds_keep_dates_together_even_with_ticker_ordered_input():
    dates = panel(40)['Date_Std']
    for train, valid in date_folds(dates):
        assert dates.iloc[train].max() < pd.Timestamp(dates.iloc[valid].min()) - pd.Timedelta(days=1)
        assert set(dates.iloc[train]).isdisjoint(dates.iloc[valid])
        for date in dates.iloc[valid].unique():
            assert set(np.flatnonzero(dates.eq(date))) <= set(valid)


class MemoryS3:
    def __init__(self):
        self.objects = {}
    def read_key(self, key, bucket_name):
        return self.objects[key]
    def list_keys(self, bucket_name, prefix):
        return [key for key in self.objects if key.startswith(prefix)]
    def load_string(self, string_data, key, bucket_name, replace):
        self.objects[key] = string_data


def test_parallel_registry_entries_preserve_all_families_and_legacy():
    s3 = MemoryS3()
    s3.objects['models/registry.csv'] = 'model_type,version,trained_date,accuracy\nxgboost,v20260920,2026-09-20,0.5\n'
    def write(kind):
        _update_registry(s3, 'test', ModelCard(kind, 'v20260922', '2026-09-22', kind, metrics={'accuracy': .6}))
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(write, ['xgboost', 'lstm', 'finbert']))
    assert len(list_model_versions(s3, 'test')) == 4
    write('xgboost')
    assert len(list_model_versions(s3, 'test', 'xgboost')) == 2


def test_evaluation_preprocess_handles_missing_latest_labels():
    df = panel().rename(columns={'Date_Std': 'Date', 'Ticker_Std': 'Ticker'})
    processed, columns = XGBoostModelBuilder().preprocess(df)
    assert processed['Target_Std'].isna().sum() == 2
    assert columns == ['feature']


def load_dashboard_functions(*names):
    path = Path(__file__).resolve().parents[1] / 'dashboard/app.py'
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_unavailable_dashboard_has_no_fabricated_prices_or_signals():
    fn = load_dashboard_functions('_unavailable_market_data')['_unavailable_market_data']
    result = fn('2026-09-22')
    assert result['status'] == 'unavailable'
    assert result['predictions'] == []
    assert result['date'] is None
    assert result['buy_count'] == result['sell_count'] == result['hold_count'] == 0
