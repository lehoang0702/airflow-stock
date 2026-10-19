"""Causal features, validation-only choices and saved evaluation semantics."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dags'))
from model_registry import LSTMModelBuilder, XGBoostModelBuilder
from model_training import recency_weights, select_decision_threshold
from model_evaluator import evaluate_binary_classifier
from pipeline_data import inference_panel_features, add_stationary_features, add_cross_sectional_features


def sample_panel():
    days = pd.bdate_range('2025-01-01', periods=6)
    return pd.DataFrame({'Date': np.tile(days, 2), 'Ticker': ['AAPL'] * 6 + ['MSFT'] * 6,
                         'Close': np.r_[np.arange(100., 106.), np.arange(200., 206.)],
                         'log_ret': np.r_[np.linspace(-.02, .02, 6), np.linspace(-.04, .04, 6)],
                         'vol_sma_ratio': np.tile(np.arange(1., 7.), 2),
                         'macd': np.arange(12.), 'macd_signal': np.arange(12.) / 2,
                         'macd_hist': np.arange(12.) / 2, 'spy_close': 500.,
                         'spy_ret': .01, 'label_1d': 1., 'future_return_5d': .1,
                         'return_5d': .1, 'fund_pe_ratio': 20., 'Adj Close': 100.,
                         'split_set': 'train', 'UnNaMeD: 0': np.arange(12.)})


@pytest.mark.parametrize('builder_class', [LSTMModelBuilder, XGBoostModelBuilder])
def test_stationary_features_are_price_scale_invariant_and_block_leakage(builder_class):
    panel = sample_panel()
    scaled = panel.copy()
    scaled[['Close', 'macd', 'macd_signal', 'macd_hist']] *= 10
    original, columns = builder_class().preprocess(panel, feature_policy='stationary_v2')
    other, other_columns = builder_class().preprocess(scaled, feature_policy='stationary_v2')
    assert columns == other_columns
    pd.testing.assert_frame_equal(original[columns], other[columns])
    assert not set(columns) & {'label_1d', 'future_return_5d', 'return_5d', 'fund_pe_ratio',
                               'Adj Close', 'spy_close', 'macd', 'split_set', 'UnNaMeD: 0'}
    assert {'macd_pct', 'macd_hist_pct', 'cs_rank_ret1', 'cs_rank_vol_ratio'} <= set(columns)


def test_feature_generation_cannot_read_appended_future_observations():
    panel = sample_panel()
    original, cols = LSTMModelBuilder().preprocess(panel.copy(), feature_policy='stationary_v2')
    future = panel.iloc[[-1]].copy()
    future['Date'] += pd.Timedelta(days=20)
    future['Close'], future['log_ret'], future['macd'] = 100000., 1000., 100000.
    extended, _ = LSTMModelBuilder().preprocess(pd.concat([panel, future]), feature_policy='stationary_v2')
    pd.testing.assert_frame_equal(original[cols], extended.iloc[:len(original)][cols])


def test_legacy_cross_sectional_definition_is_replayed_for_old_artifacts():
    old, _ = LSTMModelBuilder().preprocess(sample_panel(), feature_policy='legacy')
    new, _ = LSTMModelBuilder().preprocess(sample_panel(), feature_policy='stationary_v2')
    assert 'cs_rank_ret1' not in old
    np.testing.assert_allclose(old['cs_rel_macd_hist'], old['macd_hist'] - old.groupby('Date_Std')['macd_hist'].transform('mean'))
    np.testing.assert_allclose(new['cs_rel_macd_hist'], new['macd_hist_pct'] - new.groupby('Date_Std')['macd_hist_pct'].transform('mean'), atol=1e-8)


def test_recency_is_by_session_not_row_or_ticker():
    dates = pd.to_datetime(['2025-01-03', '2025-01-01', '2025-01-03', '2025-01-02'])
    weights = recency_weights(dates, 1)
    assert weights[0] == weights[2]
    assert weights[0] == pytest.approx(2 * weights[3])
    assert weights[3] == pytest.approx(2 * weights[1])
    assert weights.mean() == pytest.approx(1.)
    np.testing.assert_array_equal(recency_weights(dates, 0), np.ones(4))


def test_validation_threshold_changes_accuracy_and_preserves_auc():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([.41, .44, .46, .49])
    threshold = select_decision_threshold(y, probabilities)
    default = evaluate_binary_classifier(y, probabilities)
    tuned = evaluate_binary_classifier(y, probabilities, decision_threshold=threshold)
    assert tuned['accuracy'] == 1.
    assert default['accuracy'] == .5
    assert tuned['auc_roc'] == default['auc_roc']
    assert tuned['accuracy_at_0_5'] == .5
    assert tuned['decision_threshold'] == threshold


def test_threshold_does_not_claim_improvement_by_only_predicting_the_majority():
    assert select_decision_threshold([0, 1, 1, 1], [.6, .6, .6, .6]) == .5
    assert select_decision_threshold([1, 1], [.42, .49]) == .5
    with pytest.raises(ValueError):
        select_decision_threshold([0, 1], [np.nan, .5])


def test_temporal_threshold_rejects_gain_concentrated_in_one_regime():
    # A lowered threshold wins globally, while reversing correct negatives in the tail.
    labels = np.array([1, 1, 1, 0] * 2 + [0, 0, 0, 1])
    probabilities = np.array([.46, .46, .46, .2] * 2 + [.46, .46, .46, .7])
    dates = pd.bdate_range('2025-01-01', periods=12)
    assert select_decision_threshold(labels, probabilities) != .5
    assert select_decision_threshold(labels, probabilities, dates=dates) == .5


def test_inference_relative_features_match_training_and_require_same_date_peers():
    panel = sample_panel()
    observed = panel[panel['Ticker'].eq('AAPL')].copy()
    expected, cs = add_cross_sectional_features(add_stationary_features(panel), stationary=True)
    actual = inference_panel_features(observed, panel, 'AAPL', stationary=True)
    np.testing.assert_allclose(actual[cs], expected[expected['Ticker'].eq('AAPL')][cs], atol=1e-8)
    missing = panel.drop(panel[panel['Ticker'].eq('MSFT')].index[-1])
    unavailable = inference_panel_features(observed, missing, 'AAPL', stationary=True)
    assert unavailable[cs].iloc[-1].isna().all()


def test_recorded_dataset_is_verified_before_replaying_holdout():
    import ast
    import hashlib
    import io
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / 'dags/dag_model_evaluation.py'
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == '_load_valid_dataset')
    namespace = {'pd': pd, 'io': io, 'BUCKET_NAME': 'test'}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    raw = sample_panel().to_csv(index=False)
    store = SimpleNamespace(read_key=lambda *args, **kwargs: raw)
    function = namespace['_load_valid_dataset']
    assert len(function(store, 'lstm/', min_rows=1, dataset_key='chosen.csv',
                        expected_sha256=hashlib.sha256(raw.encode()).hexdigest())) == 12
    with pytest.raises(ValueError, match='overwritten'):
        function(store, 'lstm/', min_rows=1, dataset_key='chosen.csv', expected_sha256='incorrect')


def test_lstm_search_rejects_high_auc_with_deteriorating_probability_loss(monkeypatch):
    torch = pytest.importorskip('torch')
    import model_training
    predictions = {1: np.array([.51, .52, .53, .51]),
                   2: np.array([.95, .97, .96, .98])}
    y = np.array([0, 1, 0, 1], dtype=np.float32)
    class Model:
        def __init__(self, input_dim, hidden_dim, **kwargs):
            self.hidden_dim = hidden_dim
    def fit(model, *args, **kwargs):
        from sklearn.metrics import roc_auc_score
        return 0, 1, roc_auc_score(y, predictions[model.hidden_dim]), model
    monkeypatch.setattr(model_training, 'fit_lstm_with_validation', fit)
    monkeypatch.setattr(model_training, 'predict_lstm_loader', lambda model, loader: predictions[model.hidden_dim])
    candidates = [{'hidden_dim': dimension, 'num_layers': 1, 'dropout': 0., 'lr': .001,
                   'weight_decay': .001} for dimension in (1, 2)]
    X = np.zeros((4, 3, 2), dtype=np.float32)
    model, config, _, history = model_training.select_lstm_configuration(Model, X, y, X, y,
                                                                         candidates=candidates, epochs=1)
    assert history[1]['val_metric'] > history[0]['val_metric']
    assert not history[1]['eligible']
    assert model.hidden_dim == config['hidden_dim'] == 1


def test_xgboost_tuning_scores_only_observations_outside_fit_and_early_stopping(monkeypatch):
    pytest.importorskip('optuna')
    xgb = pytest.importorskip('xgboost')
    from hyperparameter_tuner import tune_xgboost_hyperparameters
    dates = pd.Series(np.repeat(pd.bdate_range('2024-01-01', periods=100), 3))
    features = pd.DataFrame({'day': np.repeat(np.arange(100), 3), 'signal': np.tile([0, 1, 0], 100)})
    target = features['signal']
    observations = []

    class TrackingClassifier:
        def __init__(self, **params):
            pass
        def fit(self, X, y, sample_weight, eval_set, verbose):
            self.fit_rows = set(X.index)
            self.stop_rows = set(eval_set[0][0].index)
            assert not self.fit_rows & self.stop_rows
            assert X['day'].max() < eval_set[0][0]['day'].min()
            return self
        def predict_proba(self, X):
            assert not set(X.index) & (self.fit_rows | self.stop_rows)
            observations.append(len(X))
            return np.column_stack([1 - X['signal'], X['signal']])

    monkeypatch.setattr(xgb, 'XGBClassifier', TrackingClassifier)
    params = tune_xgboost_hyperparameters(features, target, n_trials=1, dates=dates)
    assert len(observations) == 3
    assert params['tree_method'] == 'hist'
    assert 'recency_half_life' in params
