"""Regressions for review findings, using real code and isolated storage/feeds."""
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dags'))
from pipeline_storage import (load_backup_rows, new_model_version,
                              require_complete_panel, snapshot_training_dataset)
from model_registry import (ModelCard, load_evaluation_model, load_model_from_minio,
                            save_model_to_minio, select_champion_model)


@pytest.mark.parametrize('kind', ['xgboost', 'lstm'])
@pytest.mark.parametrize('material_error', [False, True])
def test_candle_bounds_tolerate_roundoff_but_reject_wrong_prices(kind, material_error):
    from data_validator import validate_lstm_dataset, validate_xgboost_dataset
    frame = pd.DataFrame({
        'Date': pd.date_range('2026-01-01', periods=35), 'Ticker': 'AAPL',
        'Open': 100., 'Close': 110., 'Volume': 1000.,
        'High': 109.99 if material_error else np.nextafter(110., 0.),
        'Low': 100.01 if material_error else np.nextafter(100., np.inf),
    })
    # Exercise the same serialization used by live stored datasets.
    frame = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
    validator = validate_lstm_dataset if kind == 'lstm' else validate_xgboost_dataset
    result = validator(frame, expected_tickers=['AAPL'])
    assert result.is_valid is not material_error, result.errors


class MemoryStore:
    def __init__(self):
        self.objects = {}
        self.writes = []

    def check_for_bucket(self, bucket): return True
    def check_for_key(self, key, bucket_name): return key in self.objects
    def list_keys(self, bucket_name, prefix): return [k for k in self.objects if k.startswith(prefix)]
    def read_key(self, key, bucket_name): return self.objects[key].decode('utf-8')
    def get_key(self, key, bucket_name):
        return SimpleNamespace(get=lambda: {'Body': io.BytesIO(self.objects[key])})
    def load_bytes(self, bytes_data, key, bucket_name, replace):
        if key in self.objects and not replace:
            raise FileExistsError(key)
        self.objects[key] = bytes_data
        self.writes.append(key)
    def load_string(self, string_data, **kwargs):
        self.load_bytes(string_data.encode('utf-8'), **kwargs)


def card(version=None, run='run-one'):
    return ModelCard('test', version or new_model_version(), '2026-10-07', 'xgboost',
                     feature_columns=['momentum'], metrics={'accuracy': .6, 'auc_roc': .6},
                     hyperparameters={'pipeline_run_id': run})


def test_multiple_runs_keep_distinct_weights_and_training_inputs():
    store = MemoryStore()
    first, second = card(), card()
    assert first.model_version != second.model_version
    for c, raw, weights in ((first, 'date,close\n2026-10-01,100\n', b'one'),
                            (second, 'date,close\n2026-10-01,101\n', b'two')):
        key, digest = snapshot_training_dataset(store, 'test', 'xgboost', c.model_version, raw)
        c.hyperparameters.update(dataset_key=key, dataset_sha256=digest)
        save_model_to_minio(store, 'test', 'xgboost', c, weights)
    assert load_model_from_minio(store, 'test', 'xgboost', first.model_version)['model_bytes'] == b'one'
    assert load_model_from_minio(store, 'test', 'xgboost')['model_bytes'] == b'two'
    assert len([k for k in store.objects if k.startswith('dataset_history/')]) == 2
    with pytest.raises(FileExistsError):
        save_model_to_minio(store, 'test', 'xgboost', first, b'replacement')
    assert load_model_from_minio(store, 'test', 'xgboost', first.model_version)['model_bytes'] == b'one'


def test_failed_upload_does_not_publish_a_card_or_hide_previous_model(monkeypatch):
    store = MemoryStore()
    first, second = card(), card()
    save_model_to_minio(store, 'test', 'xgboost', first, b'one')
    original = store.load_string
    def fail_schema(string_data, **kwargs):
        if kwargs['key'].endswith('feature_columns.json'):
            raise OSError('interrupted upload')
        original(string_data, **kwargs)
    monkeypatch.setattr(store, 'load_string', fail_schema)
    with pytest.raises(OSError):
        save_model_to_minio(store, 'test', 'xgboost', second, b'two')
    assert f'models/xgboost/{second.model_version}/model_card.json' not in store.objects
    assert load_model_from_minio(store, 'test', 'xgboost')['version'] == first.model_version


def test_evaluation_resolves_the_master_run_without_rewriting_cards():
    store = MemoryStore()
    first, second = card(run='one'), card(run='two')
    save_model_to_minio(store, 'test', 'xgboost', first, b'one')
    save_model_to_minio(store, 'test', 'xgboost', second, b'two')
    assert load_evaluation_model(store, 'test', 'xgboost', 'one')['version'] == first.model_version
    assert load_evaluation_model(store, 'test', 'xgboost', 'missing') is None
    assert load_evaluation_model(store, 'test', 'xgboost')['version'] == second.model_version


@pytest.mark.parametrize('kind', ['xgboost', 'lstm'])
def test_evaluation_replays_snapshot_without_a_daily_dataset(kind, monkeypatch):
    pytest.importorskip('airflow')
    import dag_model_evaluation as evaluation
    import model_evaluator
    from model_registry import XGBoostModelBuilder, LSTMModelBuilder
    store = MemoryStore()
    dates = pd.bdate_range('2023-08-01', periods=160)
    frame = pd.DataFrame({'Date': dates, 'Ticker': 'AAPL',
                          'Close': 100 + np.sin(np.arange(160)),
                          'momentum': np.cos(np.arange(160))})
    if kind == 'lstm':
        builder = LSTMModelBuilder({**LSTMModelBuilder.DEFAULT_HYPERPARAMS,
            'hidden_dim': 4, 'num_layers': 1, 'dropout': 0,
            'epochs': 1, 'lookback': 3})
    else:
        builder = XGBoostModelBuilder({**XGBoostModelBuilder.DEFAULT_HYPERPARAMS,
                                      'n_estimators': 3, 'n_jobs': 1})
    builder.train(frame.copy())
    saved_card = builder.model_card
    key, digest = snapshot_training_dataset(store, 'test', kind,
                                            saved_card.model_version, frame.to_csv(index=False))
    saved_card.hyperparameters.update(dataset_key=key, dataset_sha256=digest,
                                     pipeline_run_id='snapshot-test')
    weights = builder.serialize_model() if kind == 'lstm' else builder.serialize()
    scaler = builder.serialize_scaler() if kind == 'lstm' else None
    save_model_to_minio(store, 'test', kind, saved_card, weights, scaler)
    before = store.objects[f'models/{kind}/{saved_card.model_version}/model_card.json']
    assert not any(k.startswith(kind + '/') for k in store.objects)
    monkeypatch.setattr(evaluation, 'S3Hook', lambda **kwargs: store)
    monkeypatch.setattr(evaluation, 'BUCKET_NAME', 'test')
    captured = {}
    monkeypatch.setattr(model_evaluator, 'save_evaluation_to_minio',
                        lambda hook, bucket, model_type, metrics, *args: captured.update(metrics))
    task = getattr(evaluation, f'evaluate_{kind}_task')
    task(dag_run=SimpleNamespace(conf={'pipeline_run_id': 'snapshot-test'}),
         ti=SimpleNamespace(xcom_push=lambda **kwargs: None))
    assert captured['status'] == 'ok'
    assert captured['model_version'] == saved_card.model_version
    assert captured['total_samples'] > 0
    assert store.objects[f'models/{kind}/{saved_card.model_version}/model_card.json'] == before


def test_same_day_fallback_accepts_timestamp_versions(monkeypatch):
    import model_registry
    store = MemoryStore()
    first = card('v20261007T000000000001_a')
    save_model_to_minio(store, 'test', 'xgboost', first, b'one')
    outcome = select_champion_model(store, 'test', 'xgboost',
                                    'v20261007T010000000001_b', .4, .4)
    assert outcome['status'] == 'FALLBACK'
    assert outcome['selected_version'] == first.model_version


@pytest.mark.parametrize('failure', ['incomplete', 'nonfinite'])
def test_fallback_skips_unusable_candidate_and_uses_next_model(failure):
    from model_registry import _update_registry
    store = MemoryStore()
    usable = card('v20261007T000000000001_a')
    save_model_to_minio(store, 'test', 'xgboost', usable, b'usable')
    broken = card('v20261007T010000000001_b')
    broken.metrics = {'accuracy': .9, 'auc_roc': np.inf if failure == 'nonfinite' else .9}
    if failure == 'incomplete':
        store.objects[f'models/xgboost/{broken.model_version}/model_card.json'] = broken.to_json().encode()
        _update_registry(store, 'test', broken)
    else:
        save_model_to_minio(store, 'test', 'xgboost', broken, b'nonfinite-score')
    result = select_champion_model(store, 'test', 'xgboost',
                                   'v20261007T020000000001_c', .4, .4)
    assert result['status'] == 'FALLBACK'
    assert result['selected_version'] == usable.model_version
    assert result['artifacts']['model_bytes'] == b'usable'


def price_history(periods=350):
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, periods)))
    return pd.DataFrame({'Open': close * .995, 'High': close * 1.02,
                         'Low': close * .98, 'Close': close,
                         'Volume': rng.integers(1000, 5000, periods)},
                        index=pd.bdate_range('2024-01-01', periods=periods, name='Date'))


@pytest.mark.parametrize('parquet_only', [False, True])
def test_backup_reads_formats_and_excludes_current_and_future_days(parquet_only):
    store = MemoryStore()
    frame = price_history(60).reset_index().assign(Ticker='AAPL')
    raw = io.BytesIO()
    frame.to_parquet(raw, index=False)
    store.objects['lstm/data_20261006.parquet'] = raw.getvalue()
    if not parquet_only:
        store.objects['lstm/data_20261006.csv'] = frame.to_csv(index=False).encode()
    # Corrupt or inappropriate objects must not hide a usable earlier version.
    store.objects['lstm/data_20261007.csv'] = b'bad'
    store.objects['lstm/data_20261008.csv'] = b'bad'
    recovered = load_backup_rows(store, 'test', 'lstm/data_', 'Ticker', ['AAPL'],
                                 ['Date', 'Close'], '20261007')
    assert len(recovered['AAPL']) == 60
    assert recovered['AAPL'].Close.iloc[-1] == pytest.approx(frame.Close.iloc[-1])


def test_coverage_and_stale_backfill_are_rejected_after_cleaning():
    panel = pd.DataFrame({'Ticker': ['AAPL', 'MSFT'], 'Date': ['2026-10-02', '2026-10-01']})
    with pytest.raises(ValueError, match='Stale backfill'):
        require_complete_panel(panel, 'Ticker', 'Date', ['AAPL', 'MSFT'])
    with pytest.raises(ValueError, match='Incomplete ticker coverage'):
        require_complete_panel(panel.iloc[:1], 'Ticker', 'Date', ['AAPL', 'MSFT'])


def crawler_module(kind, monkeypatch, store, history):
    pytest.importorskip('airflow')
    import importlib
    module = importlib.import_module('dag_crawl_' + kind + '_stock_features')
    import yfinance
    monkeypatch.setattr(module, 'S3Hook', lambda **kwargs: store)
    monkeypatch.setattr(yfinance, 'Ticker', lambda symbol: SimpleNamespace(
        history=lambda **kwargs: history.copy(), info={}))
    monkeypatch.setattr(yfinance, 'download', lambda *args, **kwargs: history.copy())
    if kind == 'lstm':
        monkeypatch.setattr(module, 'TICKERS_20', ['AAPL'])
        monkeypatch.setattr(module, 'MACRO_TICKERS', ['SPY'])
    else:
        monkeypatch.setattr(module, 'TICKERS', {'AAPL': {'name': 'Apple', 'sector': 'Technology'}})
        monkeypatch.setattr(module, 'MACRO_TICKERS', {'SPY': 'spy', 'QQQ': 'qqq'})
    return module


@pytest.mark.parametrize('kind', ['xgboost', 'lstm'])
@pytest.mark.parametrize('failure', ['invalid_high', 'nonfinite_close'])
def test_crawlers_log_quality_failure_without_publishing_bad_data(kind, failure, monkeypatch):
    store = MemoryStore()
    history = price_history()
    if failure == 'invalid_high': history['High'] = history.Close * .9
    else: history.iloc[-1, history.columns.get_loc('Close')] = np.nan
    module = crawler_module(kind, monkeypatch, store, history)
    task = module.crawl_process_and_upload_to_minio if kind == 'xgboost' else module.crawl_and_extract_lstm_features
    with pytest.raises(ValueError, match='quality failed'):
        task()
    assert any(k.startswith('quality/') for k in store.objects)
    assert not any(k.startswith(kind + '/') for k in store.objects)


def test_real_xgboost_crawler_keeps_simple_return_labels(monkeypatch):
    store = MemoryStore()
    history = price_history()
    module = crawler_module('xgboost', monkeypatch, store, history)
    module.crawl_process_and_upload_to_minio()
    key = next(k for k in store.objects if k.startswith('xgboost/') and k.endswith('.csv'))
    data = pd.read_csv(io.BytesIO(store.objects[key]))
    expected = history.Close.shift(-5) / history.Close - 1
    labels = dict(zip(expected.index.strftime('%Y-%m-%d'), expected))
    np.testing.assert_allclose(data.future_return_5d, data.date.map(labels), equal_nan=True)
    assert data.return_policy.eq('simple_v2').all()
    assert 'close' not in data  # Raw prices remain outside model input exports.


@pytest.mark.skip(reason="raw-data backup removed as requested")
def test_xgboost_raw_backfill_retains_failed_ticker_in_final_export(monkeypatch):
    store = MemoryStore()
    history = price_history()
    module = crawler_module('xgboost', monkeypatch, store, history)
    monkeypatch.setattr(module, 'TICKERS', {
        'AAPL': {'name': 'Apple', 'sector': 'Technology'},
        'MSFT': {'name': 'Microsoft', 'sector': 'Technology'}})
    monkeypatch.setattr(module.time, 'sleep', lambda *args: None)
    import yfinance
    monkeypatch.setattr(yfinance, 'Ticker', lambda symbol: SimpleNamespace(
        history=lambda **kwargs: pd.DataFrame() if symbol == 'MSFT' else history.copy(), info={}))
    monkeypatch.setattr(yfinance, 'download', lambda symbol, **kwargs:
                        pd.DataFrame() if symbol == 'MSFT' else history.copy())
    raw = history.reset_index()
    raw.columns = raw.columns.str.lower()
    raw['symbol'] = 'MSFT'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d')
    store.objects[f'raw-data/xgboost/stock_prices_{stamp}T000000000001.csv'] = raw.to_csv(index=False).encode()
    module.crawl_process_and_upload_to_minio()
    exported = pd.read_csv(io.BytesIO(next(value for key, value in store.objects.items()
                                          if key.startswith('xgboost/') and key.endswith('.csv'))))
    assert set(exported.symbol) == {'AAPL', 'MSFT'}
    assert exported.groupby('symbol').date.max().nunique() == 1


@pytest.mark.parametrize('backup_format', ['csv', 'parquet'])
def test_lstm_backfill_normalizes_dates_with_live_rows(backup_format, monkeypatch):
    from datetime import timedelta
    from feature_engineering import calculate_technical_features
    store = MemoryStore()
    history = price_history()
    module = crawler_module('lstm', monkeypatch, store, history)
    monkeypatch.setattr(module, 'TICKERS_20', ['AAPL', 'MSFT'])
    monkeypatch.setattr(module.time, 'sleep', lambda *args: None)
    import yfinance
    monkeypatch.setattr(yfinance, 'Ticker', lambda symbol: SimpleNamespace(
        history=lambda **kwargs: pd.DataFrame() if symbol == 'MSFT' else history.copy(), info={}))
    monkeypatch.setattr(yfinance, 'download', lambda symbol, **kwargs:
                        pd.DataFrame() if symbol == 'MSFT' else history.copy())
    raw = calculate_technical_features(history.reset_index()).assign(Ticker='MSFT', Sector='Technology')
    stamp = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y%m%d')
    key = f'lstm/lstm_stock_20tickers_10y_{stamp}.{backup_format}'
    if backup_format == 'csv':
        store.objects[key] = raw.to_csv(index=False).encode()
    else:
        buf = io.BytesIO()
        raw.to_parquet(buf, index=False)
        store.objects[key] = buf.getvalue()
    module.crawl_and_extract_lstm_features()
    latest = datetime.now(timezone.utc).strftime('%Y%m%d')
    exported = pd.read_csv(io.BytesIO(store.objects[f'lstm/lstm_stock_20tickers_10y_{latest}.csv']))
    assert set(exported.Ticker) == {'AAPL', 'MSFT'}
    assert exported.groupby('Ticker').Date.max().nunique() == 1
    assert exported.groupby('Ticker').tail(1).Target.isna().all()


def test_missing_champion_metrics_fail_closed_without_format_error():
    outcome = select_champion_model(MemoryStore(), 'test', 'xgboost', 'v20261007', None, None)
    assert outcome['status'] == 'SAFE_MODE'


@pytest.mark.parametrize('failed_kind', ['xgboost', 'lstm'])
def test_real_missing_model_keeps_surviving_model_and_unknown_ticker(failed_kind, monkeypatch):
    from dashboard import inference
    monkeypatch.setattr(inference, '_get_history', lambda *args: price_history(60))
    monkeypatch.setattr(inference, '_load_xgboost_from_minio', lambda: (None, None))
    monkeypatch.setattr(inference, '_load_lstm_from_minio', lambda: (None, None, None, 0))
    if failed_kind == 'xgboost':
        monkeypatch.setattr(inference, '_predict_lstm_real', lambda ticker: 60.)
    else:
        monkeypatch.setattr(inference, '_predict_xgboost_real', lambda *args: 60.)
    monkeypatch.setattr(inference, '_predict_finbert_real', lambda *args: (None, None))
    monkeypatch.setattr(inference, 'get_s3_client', lambda: None)
    result = inference.analyze_ticker_on_demand('XYZ')
    assert result['status'] == 'success'
    pred = result['prediction']
    assert pred['name'] == 'XYZ'
    assert pred['prob_xgb' if failed_kind == 'xgboost' else 'prob_lstm'] is None
    assert pred['prob_lstm' if failed_kind == 'xgboost' else 'prob_xgb'] == 60.
    inference._ON_DEMAND_PREDICTIONS.pop('XYZ', None)


@pytest.mark.parametrize('price', [np.nan, np.inf, 0, -1])
def test_on_demand_rejects_invalid_reference_price_before_running_models(price, monkeypatch):
    from dashboard import inference
    monkeypatch.setattr(inference, '_get_history', lambda *args: pd.DataFrame({'Close': [price]}))
    def unexpected_model(*args):
        pytest.fail('Models must not run against an invalid reference price')
    monkeypatch.setattr(inference, '_predict_xgboost_real', unexpected_model)
    monkeypatch.setattr(inference, '_predict_lstm_real', unexpected_model)
    monkeypatch.setattr(inference, '_predict_finbert_real', unexpected_model)
    result = inference.analyze_ticker_on_demand('IBM')
    assert result['status'] == 'error'
    assert 'Giá tham chiếu' in result['message']
    assert 'IBM' not in inference._ON_DEMAND_PREDICTIONS


def test_legacy_log_return_conversion_matches_new_return_labels():
    from pipeline_data import simple_return_target
    old = pd.DataFrame({'future_return_5d': [np.log(1.2), np.nan]})
    new = pd.DataFrame({'future_return_5d': [.2, np.nan], 'return_policy': 'simple_v2'})
    np.testing.assert_allclose(simple_return_target(old, 'future_return_5d'),
                               simple_return_target(new, 'future_return_5d'), equal_nan=True)


def test_quality_api_serves_failed_nonfinite_metrics_as_null(monkeypatch):
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import urlopen
    from dashboard import handler
    from data_validator import validate_xgboost_dataset
    raw = price_history(60).reset_index().assign(Ticker='AAPL')
    raw['Close'] = np.inf
    report = validate_xgboost_dataset(raw).to_dict()
    assert not report['is_valid']
    monkeypatch.setattr(handler, 'get_quality_data', lambda: {'status': 'FAILED', 'datasets': [report]})
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler.DashboardHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with urlopen(f'http://127.0.0.1:{server.server_port}/api/quality', timeout=5) as response:
            body = response.read().decode()
            assert response.status == 200
        data = json.loads(body, parse_constant=lambda value: pytest.fail(f'Invalid JSON number: {value}'))
        assert data['status'] == 'FAILED'
        assert data['datasets'][0]['metrics']['max_close_price'] is None
        assert data['datasets'][0]['errors']
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_ensemble_excludes_forecasts_from_another_run_on_the_same_day(monkeypatch):
    pytest.importorskip('airflow')
    import dag_ensemble_master as ensemble
    import learned_signals
    store = MemoryStore()
    monkeypatch.setattr(store, 'get_conn', lambda: store, raising=False)
    monkeypatch.setattr(ensemble, 'S3Hook', lambda **kwargs: store)
    monkeypatch.setattr(ensemble, 'TELEGRAM_TOKEN', '')
    monkeypatch.setattr(learned_signals, 'predict_signal', lambda *args: (None, None))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d')
    base = {'ma_co_phieu': 'AAPL', 'nhom_nganh': 'Technology', 'prob_num': .6,
            'issued_at': datetime.now(timezone.utc).isoformat(),
            'reference_date': '2026-10-06', 'inference_source': 'trained_model_v2',
            'model_version': 'v20261007', 'khuyen_nghi': 'MUA'}
    store.objects[f'xgboost/du_bao_tang_truong_{stamp}.csv'] = pd.DataFrame(
        [{**base, 'pipeline_run_id': 'old-run', 'prob_num': .9}]).to_csv(index=False).encode()
    store.objects[f'lstm/du_bao_tang_truong_lstm_{stamp}.csv'] = pd.DataFrame(
        [{**base, 'pipeline_run_id': 'current-run'}]).to_csv(index=False).encode()
    ensemble.compare_and_ensemble_task(dag_run=SimpleNamespace(conf={'pipeline_run_id': 'current-run'}))
    report = pd.read_csv(io.BytesIO(store.objects[f'comparison/bang_so_sanh_4_phuong_phap_{stamp}.csv']))
    assert report.iloc[0].pp1_xgboost == 'Không khả dụng'
    assert report.iloc[0].pp2_lstm == 'MUA (60.00%)'
    history_key = next(k for k in store.objects if k.startswith('forecast_history/ensemble/'))
    history = pd.read_csv(io.BytesIO(store.objects[history_key]))
    assert history.iloc[0].pipeline_run_id == 'current-run'
    assert 'prob_xgb' not in history.columns
