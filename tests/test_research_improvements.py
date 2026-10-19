from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'dags'), str(ROOT)]
from research_improvements import (add_context_features, independent_stopping_masks,
                                   select_generalizing_candidate)
from scripts.improve_models import candidate_specs, fit_candidate, verify_reference
from scripts.collect_observed_news import collect_ticker


def panel_fixture(sessions=130):
    rng = np.random.default_rng(7)
    pieces = []
    for ticker in ('AAPL', 'MSFT'):
        close = 100 * np.exp(np.cumsum(rng.normal(0, .01, sessions)))
        signal = rng.normal(size=sessions)
        pieces.append(pd.DataFrame({'Ticker_Std': ticker, 'Sector': 'Technology',
            'Date_Std': pd.bdate_range('2024-01-01', periods=sessions),
            'Close': close, 'Open': close * .999, 'High': close * 1.01,
            'Low': close * .99, 'volatility_20': .01,
            'spy_ret': .001, 'log_ret': np.r_[0., np.diff(np.log(close))],
            'signal': signal, 'Target_Std': (signal > 0).astype(float)}))
    return pd.concat(pieces, ignore_index=True).sort_values(['Date_Std', 'Ticker_Std']).reset_index(drop=True)


def test_context_is_causal_per_symbol_and_preserves_labels_and_order():
    panel = panel_fixture()
    before, columns = add_context_features(panel)
    cutoff = panel.Date_Std.unique()[-10]
    changed = panel.copy()
    changed.loc[changed.Date_Std >= cutoff, ['Open', 'High', 'Low', 'Close']] *= 3
    changed['Target_Std'] = 1 - changed.Target_Std
    after, same_columns = add_context_features(changed)
    assert columns == same_columns
    pd.testing.assert_frame_equal(before.loc[before.Date_Std < cutoff, columns],
                                  after.loc[after.Date_Std < cutoff, columns])
    pd.testing.assert_series_equal(before.Target_Std, panel.Target_Std)
    assert before.index.equals(panel.index)
    assert np.isfinite(before[columns]).all().all()
    assert before.loc[before.Ticker_Std.eq('AAPL'), 'ctx_ticker_AAPL'].eq(1).all()
    # First observation of each ticker must not use the previous ticker's close.
    assert before.groupby('Ticker_Std').head(1).ctx_overnight.eq(0).all()


def test_independent_stopping_purges_outcomes_and_excludes_outer_validation():
    panel = panel_fixture()
    dates = sorted(panel.Date_Std.unique())
    train = panel.Date_Std.lt(dates[100])
    fit, stop, meta = independent_stopping_masks(panel, train, stopping_sessions=10)
    outcome = panel.groupby('Ticker_Std').Date_Std.shift(-1)
    assert outcome[fit].lt(pd.Timestamp(meta['stopping_start'])).all()
    assert not (fit & stop).any()
    assert not ((fit | stop) & ~train).any()
    assert meta['purged_rows'] == 2
    with pytest.raises(ValueError, match='Insufficient'):
        independent_stopping_masks(panel, train & panel.Date_Std.lt(dates[20]), 10)


def test_selection_requires_validation_gain_and_does_not_read_test():
    records = [{'validation': {'auc_roc': .52, 'logloss': .69}, 'test': .9},
               {'validation': {'auc_roc': .521, 'logloss': .689}, 'test': .99},
               {'validation': {'auc_roc': .54, 'logloss': .692}, 'test': .1},
               {'validation': {'auc_roc': .6, 'logloss': .71}, 'test': .99}]
    assert select_generalizing_candidate(records) == 2
    records[0]['test'], records[2]['test'] = .1, .99
    assert select_generalizing_candidate(records) == 2


@pytest.mark.parametrize('kind', ['xgboost', 'lstm'])
def test_real_training_is_independent_of_outer_validation_and_test_labels(kind):
    module = pytest.importorskip('xgboost' if kind == 'xgboost' else 'torch')
    if kind == 'lstm':
        module.set_num_threads(2)
    panel = panel_fixture()
    dates = sorted(panel.Date_Std.unique())
    train = panel.Date_Std.lt(dates[100])
    fit, stop, _ = independent_stopping_masks(panel, train, 10)
    validation = panel.Date_Std.ge(dates[100]) & panel.Date_Std.lt(dates[115])
    test = panel.Date_Std.ge(dates[115])
    spec = candidate_specs(kind)[0]
    first, predict = fit_candidate(kind, panel, ['signal'], fit, stop, validation, spec, [42], 2, 4)
    changed = panel.copy()
    changed.loc[validation | test, 'Target_Std'] = 1 - changed.loc[validation | test, 'Target_Std']
    second, other = fit_candidate(kind, changed, ['signal'], fit, stop, validation, spec, [42], 2, 4)
    assert first['checkpoints'] == second['checkpoints']
    np.testing.assert_allclose(predict(test)[1], other(test)[1])
    # Validation metrics/thresholds may change; checkpoints may not.


def test_reference_rejects_changed_source_or_cohort():
    frame = pd.DataFrame({'Date': pd.to_datetime(['2025-01-01']), 'Ticker': ['A'], 'y': [1.]})
    meta = {key: '2025-01-01' for key in ('train_start', 'validation_start', 'test_start', 'test_end')}
    meta['prediction_horizon'] = 1
    previous = {'source': {'sha256': 'same'}, 'split': meta}
    verify_reference(frame, frame, meta, previous['source'], previous)
    with pytest.raises(ValueError, match='source'):
        verify_reference(frame, frame, meta, {'sha256': 'changed'}, previous)
    changed = frame.copy()
    changed.y = 0.
    with pytest.raises(AssertionError):
        verify_reference(frame, changed, meta, previous['source'], previous)


def test_collection_records_real_observation_instead_of_publication():
    article = {'title': 'Apple earnings release', 'publisher': 'Reuters',
               'pub_date': '2024-01-01T10:00:00Z', 'link': 'https://example.org/news/apple-earnings'}
    frame, error = collect_ticker('AAPL', feed_reader=lambda url: [article])
    assert error is None and len(frame) == 1
    assert pd.Timestamp(frame.first_seen_at.iloc[0]) > pd.Timestamp(frame.published_at.iloc[0])
    assert 'sentiment_positive' not in frame
    failed, error = collect_ticker('AAPL', feed_reader=lambda url: (_ for _ in ()).throw(ValueError('no feed')))
    assert failed.empty and error == 'no feed'
