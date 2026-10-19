from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'dags'), str(ROOT)]
from research_data import prepare_shared_panel, audit_original_labels, walk_forward_splits, MACROS
from scripts.walk_forward_benchmark import (fit_candidate, select_candidate,
                                            paired_block_intervals, summarize, probability_diagnostics)


def prices(sessions=850):
    rng = np.random.default_rng(7)
    dates = pd.bdate_range('2020-01-02', periods=sessions)
    macro = {f'{name}_close': 100 * np.exp(np.cumsum(rng.normal(0, .01, sessions))) for name in MACROS}
    pieces = []
    for ticker in ('A', 'B'):
        close = 100 * np.exp(np.cumsum(rng.normal(.0001, .01, sessions)))
        pieces.append(pd.DataFrame({'Date': dates, 'Ticker': ticker, 'Sector': 'Tech',
                                   'Open': close * .999, 'High': close * 1.01, 'Low': close * .99,
                                   'Close': close, 'Volume': rng.integers(1000, 5000, sessions), **macro}))
    return pd.concat(pieces, ignore_index=True)


def test_both_feature_families_use_the_same_observed_target_and_no_future_columns():
    panel, groups = prepare_shared_panel(prices())
    expected = panel.groupby('Ticker_Std').Close.shift(-1).gt(panel.Close).astype(float)
    valid = panel.Target_Std.notna()
    np.testing.assert_array_equal(panel.loc[valid, 'Target_Std'], expected[valid])
    assert panel.groupby('Ticker_Std').tail(1).Target_Std.isna().all()
    for kind in groups.values():
        for columns in kind.values():
            assert not any('target' in c.lower() or 'future' in c.lower() for c in columns)
            assert np.isfinite(panel[columns].to_numpy()).all()
    # XGBoost already normalizes MACD; no accidental second division by price.
    assert 'xgb_macd' in groups['xgboost']['full']
    assert 'xgb_macd_pct' not in groups['xgboost']['full']


def test_future_prices_cannot_change_past_features():
    raw = prices(400)
    cutoff = raw.Date.sort_values().unique()[-30]
    original, groups = prepare_shared_panel(raw[raw.Date < cutoff])
    raw.loc[raw.Date >= cutoff, ['Open', 'High', 'Low', 'Close']] *= 10
    extended, _ = prepare_shared_panel(raw)
    columns = sorted(set(sum([c for model in groups.values() for c in model.values()], [])))
    pd.testing.assert_frame_equal(original[columns], extended[extended.Date < cutoff][columns].reset_index(drop=True))


def test_shared_data_rejects_duplicate_or_conflicting_snapshots():
    raw = prices(300)
    with pytest.raises(ValueError, match='duplicate'):
        prepare_shared_panel(pd.concat([raw, raw.iloc[[0]]]))
    raw.loc[300, 'spy_close'] += 1
    with pytest.raises(ValueError, match='Inconsistent'):
        prepare_shared_panel(raw)


def test_shared_data_rejects_missing_or_stale_sessions():
    raw = prices(300)
    for missing_index in (100, 299):
        with pytest.raises(ValueError, match='Missing sessions or stale ticker'):
            prepare_shared_panel(raw.drop(index=missing_index))


def test_label_audit_is_paired_by_ticker_date():
    panel, _ = prepare_shared_panel(prices(300))
    old = panel[['Date_Std', 'Ticker_Std', 'Target_Std']].rename(
        columns={'Date_Std': 'date', 'Ticker_Std': 'symbol', 'Target_Std': 'future_direction_1d'})
    index = old.future_direction_1d.first_valid_index()
    old.loc[index, 'future_direction_1d'] = 1 - old.loc[index, 'future_direction_1d']
    audit = audit_original_labels(panel, old.sample(frac=1, random_state=2))
    assert audit['mismatched_labels'] == 1
    assert audit['paired_labels'] == panel.Target_Std.notna().sum()


def test_walk_forward_nonoverlapping_tests_and_outcomes_do_not_cross_boundaries():
    panel, _ = prepare_shared_panel(prices())
    outcomes = panel.groupby('Ticker_Std').Date_Std.shift(-1)
    seen = set()
    splits = list(walk_forward_splits(panel, folds=3, test_sessions=30, validation_sessions=20, train_years=1))
    for train, val, test, meta in splits:
        assert (outcomes[train] < pd.Timestamp(meta['validation_start'])).all()
        assert (outcomes[val] < pd.Timestamp(meta['test_start'])).all()
        assert (outcomes[test] <= pd.Timestamp(meta['test_end'])).all()
        assert not seen.intersection(panel.index[test])
        seen.update(panel.index[test])
        assert test.sum() == 60
    with pytest.raises(ValueError, match='history'):
        list(walk_forward_splits(panel, train_years=5))


def test_selection_only_uses_validation_and_respects_probability_loss():
    records = [{'validation': {'auc_roc': .52, 'logloss': .69}, 'test_auc': .9},
               {'validation': {'auc_roc': .54, 'logloss': .692}, 'test_auc': .1},
               {'validation': {'auc_roc': .8, 'logloss': .9}, 'test_auc': .99}]
    assert select_candidate(records) == 1
    records[0]['test_auc'], records[1]['test_auc'] = .1, .99
    assert select_candidate(records) == 1


@pytest.mark.parametrize('kind', ['xgboost', 'lstm'])
def test_real_fit_is_independent_of_test_labels(kind):
    pytest.importorskip('torch' if kind == 'lstm' else 'xgboost')
    if kind == 'xgboost':
        pytest.importorskip('optuna')
    else:
        import torch
        torch.set_num_threads(2)
    panel, groups = prepare_shared_panel(prices())
    _, _, test, meta = next(walk_forward_splits(panel, folds=1, test_sessions=20,
                                               validation_sessions=20, train_years=1))
    spec = {'name': 'small', 'group': 'core', 'years': 1, 'lookback': 10}
    result, predict, _, val_p = fit_candidate(kind, panel, groups, meta, spec, 1, 2, [42])
    changed = panel.copy()
    changed.loc[test, 'Target_Std'] = 1 - changed.loc[test, 'Target_Std']
    other, predict_other, _, other_val = fit_candidate(kind, changed, groups, meta, spec, 1, 2, [42])
    assert result == other
    np.testing.assert_allclose(val_p, other_val)
    indices, p = predict(test)
    other_indices, other_p = predict_other(test)
    np.testing.assert_array_equal(indices, other_indices)
    np.testing.assert_allclose(p, other_p)


def test_identical_predictions_have_zero_paired_uncertainty():
    dates = np.repeat(pd.bdate_range('2025-01-01', periods=30), 2)
    y = np.tile([0, 1], 30)
    frame = pd.DataFrame({'Date': dates, 'Ticker': np.tile(['A', 'B'], 30), 'fold': 1, 'y': y,
                          'selected_probability': y * .8 + .1, 'reference_probability': y * .8 + .1,
                          'selected_label': y, 'reference_label': y})
    result = paired_block_intervals(frame, repeats=10)
    assert result['delta_95pct'] == {'accuracy': [0., 0.], 'auc_roc': [0., 0.]}


def test_probability_bins_preserve_boundaries_and_single_class_confusion_counts():
    p = np.r_[np.arange(11) / 10, .3]
    diagnostics = probability_diagnostics(np.ones(len(p)), p)
    assert sum(b['samples'] for b in diagnostics['calibration_bins']) == len(p)
    assert sum(sum(row) for row in diagnostics['confusion_matrix']) == len(p)
    assert diagnostics['confusion_matrix'][0] == [0, 0]


def test_learning_history_does_not_change_lstm_training_or_predictions():
    torch = pytest.importorskip('torch')
    from torch.utils.data import DataLoader, TensorDataset
    from model_registry import LSTMModelBuilder
    from model_training import fit_lstm_with_validation, predict_lstm_loader
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(32, 4, 3)).astype(np.float32)
    y = (X[:, -1, 0] > 0).astype(np.float32)
    data = TensorDataset(torch.as_tensor(X), torch.as_tensor(y))
    predictions = []
    for collect in (False, True):
        torch.manual_seed(42)
        model = LSTMModelBuilder()._build_model_class()(input_dim=3, hidden_dim=8, num_layers=1, dropout=0.)
        train = DataLoader(data, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(42))
        history = [] if collect else None
        _, _, _, model = fit_lstm_with_validation(model, train, DataLoader(data, batch_size=16), epochs=3, patience=3, history=history)
        predictions.append(predict_lstm_loader(model, DataLoader(data, batch_size=16)))
        if collect:
            assert len(history) == 3
            assert history[0]['train_auc'] is not None
    np.testing.assert_allclose(*predictions)
