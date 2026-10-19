"""Check the legacy formulas, missing data, units and version provenance."""
import math

import pytest

from signal_formulas import finbert_signal, weighted_ensemble


def test_meta_formula_and_ensemble_use_all_three_models():
    bert = finbert_signal({'sentiment_positive': .255, 'sentiment_negative': .032, 'sentiment_neutral': .713})
    assert bert == pytest.approx(.6115)
    score, weights = weighted_ensemble({'prob_xgb': .529, 'prob_lstm': .501, 'prob_bert': bert})
    assert score == pytest.approx(.54395)
    assert weights == {'prob_xgb': .35, 'prob_lstm': .35, 'prob_bert': .3}


def test_neutral_articles_are_neutral_but_missing_articles_are_absent():
    assert finbert_signal({'sentiment_positive': 0, 'sentiment_negative': 0, 'sentiment_neutral': 1}) == .5
    assert finbert_signal({}) is None
    assert finbert_signal({'sentiment_positive': .8, 'sentiment_negative': .8, 'sentiment_neutral': .8}) is None
    assert finbert_signal({'sentiment_positive': 1, 'sentiment_negative': 0, 'sentiment_neutral': 0}) == .95
    assert finbert_signal({'sentiment_positive': 0, 'sentiment_negative': 1, 'sentiment_neutral': 0}) == .05


@pytest.mark.parametrize('missing', [None, math.nan, math.inf, -1, 60, '0.6', True])
def test_invalid_inputs_are_excluded_and_weights_are_normalized(missing):
    score, weights = weighted_ensemble({'prob_xgb': missing, 'prob_lstm': .6, 'prob_bert': .4})
    assert score == pytest.approx((.35 * .6 + .30 * .4) / .65)
    assert 'prob_xgb' not in weights and sum(weights.values()) == pytest.approx(1)


def test_all_missing_has_no_fabricated_fifty_percent_score():
    assert weighted_ensemble({}) == (None, {})
    assert weighted_ensemble({'prob_xgb': None, 'prob_lstm': math.nan, 'prob_bert': -1}) == (None, {})


def test_weighted_evaluation_excludes_older_learned_results(monkeypatch):
    pytest.importorskip('airflow')
    import pandas as pd
    from types import SimpleNamespace
    import chart_utils
    import forecast_history
    import learned_signals
    import model_evaluator
    import dag_model_evaluation as dag

    records = pd.DataFrame([
        {'ma_co_phieu': 'AAPL', 'reference_date': '2026-10-01', 'issued_at': '2026-10-01T22:00:00Z',
         'prediction_horizon': 1, 'inference_source': 'weighted_ensemble_v1',
         'model_version': 'weighted_35_35_30_v1', 'prob_num': .7},
        {'ma_co_phieu': 'MSFT', 'reference_date': '2026-10-01', 'issued_at': '2026-10-01T22:00:00Z',
         'prediction_horizon': 1, 'inference_source': 'trained_model_v2',
         'model_version': 'old-learned', 'prob_num': .1},
        {'ma_co_phieu': 'NVDA', 'reference_date': '2026-10-02', 'issued_at': '2026-10-02T22:00:00Z',
         'prediction_horizon': 1, 'inference_source': 'trained_model_v2',
         'model_version': 'old-learned', 'prob_num': .1}])
    prices = pd.DataFrame([{'Ticker': ticker, 'Date': date, 'Close': close}
                           for ticker in ['AAPL', 'MSFT', 'NVDA']
                           for date, close in [('2026-10-01', 100), ('2026-10-02', 110)]])
    saved = []
    monkeypatch.setattr(dag, 'S3Hook', lambda **kwargs: SimpleNamespace(get_conn=lambda: None))
    monkeypatch.setattr(chart_utils, 'load_price_data_from_minio', lambda *args: prices)
    monkeypatch.setattr(forecast_history, 'load_forecast_history', lambda *args: records)
    monkeypatch.setattr(model_evaluator, 'save_evaluation_to_minio', lambda hook, bucket, kind, metrics: saved.append(metrics))
    monkeypatch.setattr(learned_signals, 'train_signal_model', lambda *args: {'status': 'unavailable'})
    dag.evaluate_ensemble_task(ti=SimpleNamespace(xcom_push=lambda **kwargs: None))
    assert saved[0]['evaluation_source'] == 'issued_weighted_forecasts_v1'
    assert saved[0]['total_samples'] == 1 and saved[0]['accuracy'] == 1
    assert saved[0]['pending_samples'] == 0


def test_dashboard_only_reports_evaluation_of_the_active_formula(monkeypatch):
    import json
    from dashboard import services
    from test_forecast_provenance import ObjectStore
    store = ObjectStore()
    key = 'evaluation/ensemble/eval_20261007.json'
    report = {'accuracy': .99, 'total_samples': 100, 'evaluation_date': '2026-10-07',
              'evaluation_source': 'issued_trained_forecasts_v2'}
    store.put_object(Bucket='test', Key=key, Body=json.dumps(report).encode())
    monkeypatch.setattr(services, 'get_s3_client', lambda: store)
    assert services.get_evaluation_data()['models_comparison'] == []
    report.update(evaluation_source='issued_weighted_forecasts_v1', ensemble_method='weighted_35_35_30_v1')
    store.put_object(Bucket='test', Key=key, Body=json.dumps(report).encode())
    assert services.get_evaluation_data()['models_comparison'][0]['accuracy'] == .99
