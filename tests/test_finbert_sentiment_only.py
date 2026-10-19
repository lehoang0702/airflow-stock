"""Sentiment stays available without a direction head and reaches Ensemble unchanged."""
import io
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from dashboard import services, charts
from sentiment_utils import SENTIMENT_PREFIX


def sentiment_row():
    return {'ma_co_phieu': 'AAPL', 'reference_date': '2026-10-06',
            'issued_at': '2026-10-07T01:00:00Z', 'news_policy': 'observed_news_v1',
            'news_count': 10, 'sentiment_model_version': 'ProsusAI/finbert',
            'inference_source': 'sentiment_model_v2',
            'sentiment_positive': .25, 'sentiment_negative': .05, 'sentiment_neutral': .7,
            'news_sources': 'Yahoo Finance RSS'}


def test_new_sentiment_export_reaches_ensemble_without_price_fields(monkeypatch):
    pytest.importorskip('airflow')
    import dag_ensemble_master as dag
    import signal_formulas
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d')
    price = {'ma_co_phieu': 'AAPL', 'reference_date': '2026-10-06',
             'issued_at': '2026-10-07T01:00:00Z', 'inference_source': 'trained_model_v2',
             'model_version': 'v1', 'prob_num': .6, 'khuyen_nghi': 'MUA'}
    objects = {f'{SENTIMENT_PREFIX}{stamp}.csv': pd.DataFrame([sentiment_row()]).to_csv(index=False),
               f'xgboost/du_bao_tang_truong_{stamp}.csv': pd.DataFrame([price]).to_csv(index=False),
               f'lstm/du_bao_tang_truong_lstm_{stamp}.csv': pd.DataFrame([price]).to_csv(index=False)}
    class Store:
        def get_conn(self): return self
        def list_keys(self, bucket_name, prefix): return [key for key in objects if key.startswith(prefix)]
        def read_key(self, key, bucket_name): return objects[key]
        def load_string(self, string_data, key, **kwargs): objects[key] = string_data
    calls = []
    real_weighted = signal_formulas.weighted_ensemble
    def predict(values):
        calls.append(values.copy())
        return real_weighted(values)
    monkeypatch.setattr(dag, 'S3Hook', lambda **kwargs: Store())
    monkeypatch.setattr(dag, 'TELEGRAM_TOKEN', '')
    monkeypatch.setattr(signal_formulas, 'weighted_ensemble', predict)
    dag.compare_and_ensemble_task()
    assert len(calls) == 1
    assert calls[0] == {'prob_xgb': .6, 'prob_lstm': .6, 'prob_bert': .6}
    history = pd.read_csv(io.StringIO(next(value for key, value in objects.items() if key.startswith('forecast_history/'))))
    assert history.iloc[0].prob_bert == .6
    assert history.iloc[0].prob_num == pytest.approx(.6)
    assert history.iloc[0].inference_source == 'weighted_ensemble_v1'
    assert history.iloc[0].prediction_horizon == 1
    comparison = pd.read_csv(io.StringIO(objects[f'comparison/bang_so_sanh_4_phuong_phap_{stamp}.csv']))
    assert comparison.iloc[0].pp3_finbert == 'TÍCH CỰC (60.00%)'


@pytest.mark.parametrize('legacy', [False, True])
def test_market_shows_sentiment_and_ignores_old_finbert_buy_signal(monkeypatch, legacy):
    comparison = {'ma_co_phieu': 'AAPL', 'reference_date': '2026-10-06',
                  'issued_at': '2026-10-07T01:00:00Z', 'inference_source': 'trained_model_v2',
                  'pp1_xgboost': 'MUA (60.0%)', 'pp2_lstm': 'MUA (60.0%)',
                  'pp3_finbert': 'MUA (99.0%)', 'pp4_tong_hop': 'CHƯA CÓ ENSEMBLE'}
    if not legacy:
        comparison.update({key: value for key, value in sentiment_row().items()
                           if key not in ('inference_source',)})
    def read(s3, key):
        if key.startswith('comparison/'):
            return pd.DataFrame([comparison])
        if key.startswith('lstm/'):
            return pd.DataFrame([{'Ticker': 'AAPL', 'Date': '2026-10-06', 'Close': 100}])
        return pd.DataFrame([sentiment_row()])
    monkeypatch.setattr(services, '_CACHE', {'market': {'data': None, 'time': 0}})
    monkeypatch.setattr(services, '_ON_DEMAND_PREDICTIONS', {})
    monkeypatch.setattr(services, 'get_s3_client', lambda: object())
    monkeypatch.setattr(services, 'get_latest_s3_key', lambda s3, prefix: prefix + '20261007.csv')
    monkeypatch.setattr(services, 'read_s3_dataframe', read)
    row = services.get_market_data()['predictions'][0]
    assert row['pp3_finbert'] == 'TÍCH CỰC (60.00%)'
    assert row['finbert_status'] == 'available' and row['finbert_news_count'] == 10
    assert row['finbert_sentiment']['sentiment_positive'] == .25
    assert row['prob_bert'] == 60.0 and row['prob_ensemble'] == 60.0
    assert row['ensemble_weights'] == {'prob_xgb': .35, 'prob_lstm': .35, 'prob_bert': .3}


def test_finbert_chart_needs_sentiment_only_and_does_not_fetch_prices(monkeypatch):
    import yfinance
    def unexpected(*args, **kwargs): raise AssertionError('Sentiment chart must not fetch price history')
    monkeypatch.setattr(yfinance, 'Ticker', unexpected)
    monkeypatch.setattr(charts, '_ON_DEMAND_PREDICTIONS', {'META': {
        'finbert_sentiment': {'sentiment_positive': .25, 'sentiment_negative': .05, 'sentiment_neutral': .7},
        'finbert_news_count': 10, 'finbert_sources': ['Yahoo Finance RSS']}})
    assert charts.generate_live_chart_image('META', 'finbert').startswith(b'\x89PNG')


def test_finbert_old_price_metrics_are_not_in_price_rankings(monkeypatch):
    from test_forecast_provenance import ObjectStore
    import json
    store = ObjectStore()
    store.put_object(Bucket='test', Key='evaluation/finbert/eval_20261007.json', Body=json.dumps({
        'evaluation_source': 'issued_trained_forecasts_v2', 'evaluation_date': '2026-10-07',
        'accuracy': .99, 'total_samples': 100}).encode())
    monkeypatch.setattr(services, 'get_s3_client', lambda: store)
    assert services.get_evaluation_data()['models_comparison'] == []
