"""Prevent fabricated results and retrospective scoring of live forecasts."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
from threading import Event
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dags'))
from forecast_history import observed_forecasts
from model_evaluator import evaluate_direction_forecasts, evaluate_binary_classifier, run_time_series_cv
from learned_signals import load_signal_model, predict_signal, train_signal_model
from dashboard import services, jobs


class ObjectStore:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        return {'Body': io.BytesIO(self.objects[Key])}

    def get_paginator(self, name):
        return self

    def paginate(self, Bucket, Prefix):
        yield {'Contents': [{'Key': key, 'LastModified': datetime.now(timezone.utc)}
                            for key in self.objects if key.startswith(Prefix)]}


def prices():
    # Actual sessions skip the weekend and an exchange holiday.
    dates = ['2026-09-04', '2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11', '2026-09-14', '2026-09-15']
    return pd.DataFrame({'Ticker': ['AAPL'] * 7, 'Date': dates, 'Close': [100, 101, 102, 103, 104, 90, 110]})


def forecast(**changes):
    return {'ma_co_phieu': 'AAPL', 'reference_date': '2026-09-04',
            'issued_at': '2026-09-04T22:00:00Z', 'prob_num': .8,
            'prediction_horizon': 5, 'model_version': 'v1',
            'inference_source': 'trained_model_v2', **changes}


def test_forecast_matches_exact_issued_horizon_instead_of_latest_price():
    metrics = evaluate_direction_forecasts(pd.DataFrame([forecast()]), prices())
    assert metrics['total_evaluated'] == 1
    assert metrics['sentiment_accuracy'] == 0  # T+5 is 90, latest session is 110.
    assert metrics['sentiment_auc_roc'] is None


def test_recent_forecasts_remain_pending_and_cannot_score_past_returns():
    recent = forecast(reference_date='2026-09-14', issued_at='2026-09-14T22:00:00Z')
    metrics = evaluate_direction_forecasts(pd.DataFrame([recent]), prices())
    assert metrics['pending_samples'] == 1
    assert metrics['total_samples'] == 0
    assert metrics['sentiment_accuracy'] is None


def test_old_or_late_issuance_is_not_scored():
    legacy = {'ma_co_phieu': 'AAPL', 'prob_num': .9, 'ngay_du_bao': '2026-09-04'}
    late = forecast(issued_at='2026-09-11T22:00:00Z')
    observed, pending, invalid = observed_forecasts(pd.DataFrame([legacy, late]), prices())
    assert observed.empty
    assert pending == 0
    assert invalid == 2


def test_prediction_on_a_weekend_uses_following_actual_session():
    issued = forecast(issued_at='2026-09-06T10:00:00Z')
    observed, _, _ = observed_forecasts(pd.DataFrame([issued]), prices())
    assert observed.iloc[0]['outcome_date'] == pd.Timestamp('2026-09-14')


def test_missing_probability_or_heuristic_source_is_not_evaluated():
    data = pd.DataFrame([forecast(prob_num=None), forecast(inference_source='heuristic')])
    metrics = evaluate_direction_forecasts(data, prices())
    assert metrics['total_samples'] == 0
    assert metrics['sentiment_accuracy'] is None


def test_confusion_matrix_preserves_single_class_counts_and_auc_is_absent():
    metrics = evaluate_binary_classifier(np.array([1, 1]), np.array([.8, .9]))
    assert metrics['true_positives'] == 2
    assert metrics['auc_roc'] is None
    json.dumps(metrics, allow_nan=False)


def test_failed_cv_has_no_fabricated_metrics():
    def fail(X, y):
        raise ValueError('training failed')
    metrics = run_time_series_cv(np.zeros((20, 1)), np.zeros(20), fail, n_splits=2)
    assert metrics['cv_accuracy_mean'] is None
    assert metrics['cv_auc_mean'] is None


def test_missing_reports_expose_no_quality_pass_or_evaluation_numbers():
    with patch.object(services, 'get_s3_client', return_value=None):
        quality = services.get_quality_data()
        evaluation = services.get_evaluation_data()
    assert quality['status'] == 'unavailable'
    assert quality['total_rows'] is None
    assert quality['integrity_checks'] == []
    assert evaluation['models_comparison'] == []
    assert evaluation['date'] is None


def test_legacy_evaluations_are_rejected_and_verified_metrics_kept_exactly():
    store = ObjectStore()
    legacy = {'accuracy': .65, 'auc_roc': .81}
    verified = {'accuracy': 0, 'auc_roc': None, 'total_samples': 20,
                'evaluation_source': 'trained_model_holdout_v2', 'evaluation_date': '2026-09-30T12:00:00Z'}
    store.put_object(Bucket='test', Key='evaluation/finbert/eval_20260930.json', Body=json.dumps(legacy).encode())
    store.put_object(Bucket='test', Key='evaluation/xgboost/eval_20260930.json', Body=json.dumps(verified).encode())
    with patch.object(services, 'get_s3_client', return_value=store):
        report = services.get_evaluation_data()
    assert len(report['models_comparison']) == 1
    assert report['models_comparison'][0]['accuracy'] == 0
    assert report['models_comparison'][0]['auc_roc'] is None
    assert report['models_comparison'][0]['total_samples'] == 20


def test_quality_coverage_and_counts_come_from_actual_reports():
    store = ObjectStore()
    report = {'dataset_name': 'XGBoost_Features', 'is_valid': False,
              'timestamp': '2026-09-29T22:00:00', 'total_rows': 100, 'total_columns': 10,
              'tickers_found': ['AAPL', 'MSFT'], 'errors': ['invalid price'], 'warnings': [],
              'metrics': {'avg_missing_pct': 0}}
    store.put_object(Bucket='test', Key='quality/quality_report_XGBoost_Features_20260929.json', Body=json.dumps(report).encode())
    with patch.object(services, 'get_s3_client', return_value=store):
        quality = services.get_quality_data()
    assert quality['status'] == 'FAILED'
    assert quality['total_rows'] == 100
    assert quality['tickers_count'] == 2
    assert quality['avg_missing_pct'] == 0
    assert quality['timestamp'] == report['timestamp']


def test_insufficient_history_cannot_publish_a_model_or_probability():
    store = ObjectStore()
    result = train_signal_model(store, 'test', 'ensemble', pd.DataFrame())
    assert result['status'] == 'unavailable'
    assert store.objects == {}
    assert predict_signal(store, 'test', 'ensemble', {'prob_xgb': .7, 'prob_lstm': .3}) == (None, None)


def test_supervised_ensemble_artifact_is_trained_and_temporally_purged():
    store = ObjectStore()
    observations = []
    for date in pd.bdate_range('2025-01-01', periods=50):
        for i in range(10):
            up = i % 2
            observations.append({'ma_co_phieu': f'S{i}', 'reference_date': date,
                                 'outcome_date': date + pd.offsets.BDay(1), 'actual_up': up,
                                 'prob_xgb': .85 if up else .15, 'prob_lstm': .4 if up else .6,
                                 'sentiment_positive': .5, 'sentiment_negative': .3, 'sentiment_neutral': .2,
                                 'news_policy': 'observed_news_v1', 'news_count': 3,
                                 'sentiment_model_version': 'ProsusAI/finbert',
                                 'inference_source': 'base_models_v2'})
    metrics = train_signal_model(store, 'test', 'ensemble', pd.DataFrame(observations))
    assert metrics['status'] == 'ok'
    assert metrics['total_samples'] == 100
    model, metadata = load_signal_model(store, 'test', 'ensemble')
    assert metadata['training_samples'] >= 200
    assert pd.Timestamp(metadata['train_end']) < pd.Timestamp(metadata['test_start'])
    probability, version = predict_signal(store, 'test', 'ensemble', {'prob_xgb': .85, 'prob_lstm': .4,
        'sentiment_positive': .5, 'sentiment_negative': .3, 'sentiment_neutral': .2})
    assert probability > .9  # Learned decision, not 35/35 fixed averaging.
    assert version == metrics['model_version']
    metadata_key = next(key for key in store.objects if key.endswith('metadata.json'))
    legacy = json.loads(store.objects[metadata_key])
    legacy.pop('news_policy')
    store.objects[metadata_key] = json.dumps(legacy).encode()
    assert load_signal_model(store, 'test', 'ensemble') == (None, None)


def test_legacy_sentiment_cannot_train_new_heads():
    store = ObjectStore()
    legacy = pd.DataFrame([{'reference_date': '2026-09-01', 'outcome_date': '2026-09-08',
                           'ma_co_phieu': 'AAPL', 'actual_up': 1,
                           'inference_source': 'base_models_v2', 'prob_xgb': .6, 'prob_lstm': .4,
                           'sentiment_positive': .7, 'sentiment_negative': .1, 'sentiment_neutral': .2}])
    result = train_signal_model(store, 'test', 'ensemble', legacy)
    assert result['status'] == 'unavailable'
    assert result['news_policy'] == 'observed_news_v1'
    assert store.objects == {}


def test_analysis_jobs_share_work_and_stay_pending_until_completion():
    from dashboard import inference
    started, release = Event(), Event()
    calls = []
    def analyze(ticker):
        calls.append(ticker)
        started.set()
        release.wait(5)
        return {'status': 'success', 'prediction': {'ticker': ticker}}
    with patch.object(inference, 'analyze_ticker_on_demand', analyze):
        first, code = jobs.analysis_job(ticker='TESTJOB')
        assert code == 202
        assert started.wait(3)
        second, code = jobs.analysis_job(ticker='TESTJOB')
        assert second['job_id'] == first['job_id']
        assert second['status'] == 'processing'
        release.set()
        # Completion callback runs under the same lock; wait on the future indirectly.
        import time
        for _ in range(100):
            result, code = jobs.analysis_job(job_id=first['job_id'])
            if code == 200:
                break
            time.sleep(.01)
    assert code == 200
    assert result['status'] == 'success'
    assert calls == ['TESTJOB']


def test_expired_job_returns_an_error_instead_of_a_placeholder_forecast():
    response, code = jobs.analysis_job(job_id='missing')
    assert code == 404
    assert response['status'] == 'error'
    assert 'prediction' not in response


def test_http_analysis_returns_pending_and_keeps_other_api_responsive():
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from urllib.request import urlopen
    from dashboard import handler, inference
    started, release = Event(), Event()
    def analyze(ticker):
        started.set()
        release.wait(5)
        return {'status': 'success', 'prediction': {'ticker': ticker}}
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler.DashboardHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f'http://127.0.0.1:{server.server_port}'
    try:
        with patch.object(inference, 'analyze_ticker_on_demand', analyze), patch.object(
                handler, 'get_evaluation_data', return_value={'status': 'unavailable', 'models_comparison': []}):
            with urlopen(root + '/api/analyze?symbol=APITEST', timeout=2) as response:
                assert response.status == 202
                job = json.load(response)
            assert started.wait(2)
            with urlopen(root + '/api/evaluation', timeout=2) as response:
                assert response.status == 200
                assert json.load(response)['models_comparison'] == []
            with urlopen(root + '/api/analyze?job_id=' + job['job_id'], timeout=2) as response:
                assert response.status == 202
                assert json.load(response)['status'] == 'processing'
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_partial_price_rows_and_missing_indicators_are_not_filled_with_numbers():
    assert services._observed_candle('2026-09-30', {'Close': 100}) is None
    candle = services._observed_candle('2026-09-30', {'Open': 100, 'High': 102,
                                                    'Low': 99, 'Close': 101, 'Volume': 1000})
    assert candle['rsi'] is None
    assert candle['close'] == 101


def test_live_feature_generation_uses_training_definitions():
    from feature_engineering import calculate_technical_features, calculate_technical_indicators
    from dashboard.inference import _calculate_lstm_features_ondemand, _calculate_xgboost_features_ondemand
    dates = pd.bdate_range('2025-01-01', periods=280)
    close = np.linspace(100, 150, len(dates)) + np.sin(np.arange(len(dates)))
    frame = pd.DataFrame({'Date': dates, 'Open': close - .3, 'High': close + 1,
                          'Low': close - 1, 'Close': close, 'Volume': 1000 + np.arange(len(dates))})
    pd.testing.assert_frame_equal(_calculate_lstm_features_ondemand(frame), calculate_technical_features(frame))
    xgb_frame = frame.rename(columns={column: column.lower() for column in frame})
    macro = pd.DataFrame({'date': dates, 'spy_log_ret': .01, 'qqq_log_ret': .005})
    actual = _calculate_xgboost_features_ondemand(xgb_frame, macro, 'AAPL', 'Technology')
    expected = calculate_technical_indicators(xgb_frame, macro, 'AAPL', 'Technology')
    pd.testing.assert_frame_equal(actual, expected)


def test_lstm_evaluation_runs_the_saved_trained_weights_and_scaler():
    import ast
    import logging
    import pickle
    import pytest
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    import model_registry
    import model_evaluator
    builder = model_registry.LSTMModelBuilder({**model_registry.LSTMModelBuilder.DEFAULT_HYPERPARAMS,
        'hidden_dim': 4, 'num_layers': 1, 'dropout': 0, 'epochs': 1, 'lookback': 3})
    dates = pd.bdate_range('2023-11-15', periods=65)
    frame = pd.DataFrame({'Date': dates, 'Ticker': 'AAPL', 'Close': 100 + np.sin(np.arange(65)),
                          'momentum': np.cos(np.arange(65))})
    builder.train(frame.copy())
    buffer = io.BytesIO()
    torch.save(builder.model.state_dict(), buffer)
    artifact = {'model_card': builder.model_card, 'model_bytes': buffer.getvalue(),
                'scaler_bytes': pickle.dumps(builder.scaler), 'feature_columns': builder.feature_cols}
    path = Path(__file__).resolve().parents[1] / 'dags/dag_model_evaluation.py'
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_lstm_task')
    captured = {}
    namespace = {'datetime': datetime, 'timezone': timezone, 'json': json, 'io': io,
                 'np': np, 'pd': pd, 'logging': logging, 'BUCKET_NAME': 'test', 'MINIO_CONN_ID': 'test',
                 'S3Hook': lambda **kwargs: ObjectStore(), '_load_valid_dataset': lambda *args: frame.copy()}
    mask_node = next(node for node in ast.parse(path.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name == 'evaluation_temporal_masks')
    exec(compile(ast.Module(body=[mask_node], type_ignores=[]), str(path), 'exec'), namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    def save(hook, bucket, kind, metrics, *args):
        captured.update(metrics)
    context = {'ti': SimpleNamespace(xcom_push=lambda **kwargs: None)}
    with patch.object(model_registry, 'load_model_from_minio', return_value=artifact), patch.object(
            model_registry, 'save_model_to_minio'), patch.object(model_evaluator, 'save_evaluation_to_minio', save):
        namespace['evaluate_lstm_task'](**context)
    assert captured['status'] == 'ok'
    assert captured['total_samples'] > 0
    assert captured['evaluation_source'] == 'trained_model_holdout_v2'
    assert captured['accuracy'] is not None
