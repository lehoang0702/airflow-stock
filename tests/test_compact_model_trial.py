"""Causality checks for the bounded simpler-model experiment."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'dags')]
from scripts.compact_model_trial import stable_features, fit_linear, average_predictors
from research_improvements import independent_stopping_masks


def fixture():
    rng = np.random.default_rng(19)
    dates = pd.bdate_range('2020-01-01', periods=330)
    data = pd.DataFrame({'Date_Std': np.repeat(dates, 4),
                         'Ticker_Std': ['A', 'B', 'C', 'D'] * len(dates)})
    data['signal'] = rng.normal(size=len(data))
    data['noise'] = rng.normal(size=len(data))
    data['Target_Std'] = (data.signal + rng.normal(0, .1, len(data)) > 0).astype(float)
    train = data.Date_Std.lt(dates[260])
    fit, stop, _ = independent_stopping_masks(data, train, 20)
    validation = data.Date_Std.ge(dates[260]) & data.Date_Std.lt(dates[300])
    test = data.Date_Std.ge(dates[300])
    return data, fit, stop, validation, test


def test_feature_selection_ignores_future_values_and_labels():
    data, fit, _, _, _ = fixture()
    selected, detail = stable_features(data, ['signal', 'noise'], fit, limit=1)
    assert selected == ['signal']
    changed = data.copy()
    changed.loc[~fit, ['signal', 'noise']] *= -100
    changed.loc[~fit, 'Target_Std'] = 1 - changed.loc[~fit, 'Target_Std']
    after, audit = stable_features(changed, ['signal', 'noise'], fit, limit=1)
    assert after == selected and audit == detail


def test_scaling_and_fitted_probabilities_ignore_validation_values_and_future_labels():
    data, fit, _, validation, test = fixture()
    first, predict = fit_linear(data, ['signal', 'noise'], fit, validation, 'linear')
    changed = data.copy()
    changed.loc[validation, ['signal', 'noise']] *= 10000
    changed.loc[~fit, 'Target_Std'] = 1 - changed.loc[~fit, 'Target_Std']
    _, other = fit_linear(changed, ['signal', 'noise'], fit, validation, 'linear')
    np.testing.assert_allclose(predict(test)[1], other(test)[1], atol=1e-12)
    assert first['validation']['auc_roc'] > .9


def test_empty_stable_features_use_fit_prior_without_dropping_observations():
    data, fit, _, validation, test = fixture()
    _, predict = fit_linear(data, [], fit, validation, 'empty')
    index, probability = predict(test)
    np.testing.assert_array_equal(index, data.index[test])
    np.testing.assert_allclose(probability, data.loc[fit, 'Target_Std'].mean())


def test_blend_rejects_misaligned_observations():
    first = lambda mask: (np.array([1, 2]), np.array([.2, .6]))
    second = lambda mask: (np.array([1, 2]), np.array([.4, .8]))
    np.testing.assert_allclose(average_predictors(first, second)(None)[1], [.3, .7])
    other = lambda mask: (np.array([2, 1]), np.array([.4, .8]))
    with pytest.raises(ValueError, match='align'):
        average_predictors(first, other)(None)
