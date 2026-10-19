from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dags'))
from news_data import (precise_utc, yahoo_article, normalize_batch, select_eligible_news,
                       point_in_time_news_features, SENTIMENT_COLUMNS, news_inventory)


def article(**changes):
    return {'title': 'Apple reports quarterly earnings', 'summary': 'Revenue increased.',
            'publisher': 'Reuters', 'pub_date': '2026-10-02T19:00:00Z',
            'link': 'https://example.com/business/apple?utm_source=feed', **changes}


def batch(items=None, seen='2026-10-02T20:00:00Z'):
    return normalize_batch([article()] if items is None else items, 'AAPL', 'Technology', seen)


def test_publication_time_is_not_replaced_by_collection_day():
    data = batch()
    assert data.published_at.iloc[0] == '2026-10-02T19:00:00+00:00'
    assert data.first_seen_at.iloc[0] == data.available_at.iloc[0] == '2026-10-02T20:00:00+00:00'
    assert data.duong_dan.iloc[0] == 'https://example.com/business/apple'
    assert len(select_eligible_news(data, '2026-10-02T20:00:00Z')) == 1
    assert select_eligible_news(data, '2026-10-02T19:30:00Z').empty


@pytest.mark.parametrize('date', [None, '', '2026-10-02', '2026-10-02 19:00:00', 'bad'])
def test_unknown_or_date_only_publication_is_never_guessed(date):
    data = batch([article(pub_date=date)])
    assert data.timestamp_status.iloc[0] == 'unknown_publication_time'
    assert pd.isna(data.published_at.iloc[0])
    assert select_eligible_news(data, '2026-10-02T22:00:00Z').empty


def test_yahoo_schemas_and_rss_timezones_have_precise_times():
    old = yahoo_article({'title': 'Old schema', 'providerPublishTime': 1790967600,
                         'link': 'https://example.com/old'})
    new = yahoo_article({'content': {'title': 'New schema', 'pubDate': '2026-10-02T19:00:00Z',
                                     'canonicalUrl': {'url': 'https://example.com/new'},
                                     'provider': {'displayName': 'Publisher'}}})
    assert precise_utc(old['pub_date']).tzinfo is not None
    assert new['publisher'] == 'Publisher'
    assert precise_utc('Fri, 02 Oct 2026 15:00:00 -0400') == precise_utc(new['pub_date'])


def test_missing_news_is_empty_and_duplicates_are_removed():
    assert batch([]).empty
    duplicate = article(link='https://example.com/business/apple?utm_campaign=other')
    data = batch([article(pub_date=None), article(), duplicate])
    assert len(data) == 1
    assert data.timestamp_status.iloc[0] == 'verified'


def test_future_or_stale_or_synthetic_news_cannot_enter_model():
    future = batch([article(pub_date='2026-10-03T19:00:00Z')])
    assert select_eligible_news(future, '2026-10-03T22:00:00Z').empty
    assert select_eligible_news(batch(), '2026-10-06T22:00:00Z').empty
    synthetic = batch([article(publisher='Market Update')])
    assert select_eligible_news(synthetic, '2026-10-02T22:00:00Z').empty
    assert select_eligible_news(pd.DataFrame({'ma_co_phieu': ['AAPL'], 'tieu_de': ['old headline']}),
                                '2026-10-02T22:00:00Z').empty


def test_tampered_availability_or_string_false_is_rejected():
    data = batch()
    data.available_at = '2026-10-02T18:00:00Z'
    assert select_eligible_news(data, '2026-10-02T22:00:00Z').empty
    data = batch()
    data.eligible_for_model = 'False'
    assert select_eligible_news(data, '2026-10-02T22:00:00Z').empty


def test_content_revision_cannot_rewrite_past_observed_features():
    original = batch()
    revised = batch([article(summary='Later corrected figures.')], seen='2026-10-02T23:00:00Z')
    all_versions = pd.concat([revised, original], ignore_index=True)
    assert select_eligible_news(all_versions, '2026-10-02T22:00:00Z').article_version.iloc[0] == original.article_version.iloc[0]
    assert select_eligible_news(all_versions, '2026-10-03T00:00:00Z').article_version.iloc[0] == revised.article_version.iloc[0]


def test_historical_news_features_require_observed_scored_inputs_and_keep_missing():
    data = batch()
    data[SENTIMENT_COLUMNS] = [.7, .1, .2]
    data['sentiment_scored_at'] = '2026-10-02T20:01:00Z'
    data['sentiment_model_version'] = 'test_model'
    decisions = pd.DataFrame({'Ticker_Std': ['AAPL', 'AAPL', 'MSFT'],
                               'prediction_at': ['2026-10-02T19:30:00Z', '2026-10-02T22:00:00Z', '2026-10-02T22:00:00Z']}, index=[4, 5, 6])
    features = point_in_time_news_features(data, decisions)
    assert features.news_count.tolist() == [0, 1, 0]
    assert features.news_missing.tolist() == [1, 0, 1]
    assert np.isnan(features.loc[4, 'sentiment_positive'])
    assert features.loc[5, 'sentiment_positive'] == .7
    data['sentiment_scored_at'] = '2026-10-03T00:00:00Z'
    assert point_in_time_news_features(data, decisions).news_count.eq(0).all()
    with pytest.raises(ValueError, match='timestamp'):
        point_in_time_news_features(data, decisions.drop(columns='prediction_at'))


def test_old_data_is_audited_without_backdating_observation():
    legacy = pd.DataFrame({'ma_co_phieu': ['AAPL', 'MSFT'], 'thoi_gian_dang': ['2026-10-02', 'Fri, 02 Oct 2026 15:00:00 -0400'],
                           'nha_xuat_ban': ['Market Update', 'Reuters']})
    report = news_inventory(legacy, '2026-10-02T22:00:00Z')
    assert report['synthetic_rows'] == 1
    assert report['precise_publication_rows'] == 1
    assert report['eligible_observed_rows'] == 0


def test_global_feed_relevance_does_not_match_short_ticker_substrings():
    from news_data import article_mentions_ticker
    assert not article_mentions_ticker('Revenue rose amid volatility', 'V')
    assert not article_mentions_ticker('A cat video became popular', 'CAT')
    assert not article_mentions_ticker('Bankers discuss interest rates', 'BA')
    assert article_mentions_ticker('Visa quarterly earnings improve', 'V')
    assert article_mentions_ticker('Caterpillar lowers forecast', 'CAT')
    assert article_mentions_ticker('NYSE: BA shares rebound', 'BA')
    assert article_mentions_ticker('Google revenue grows', 'GOOGL')


@pytest.mark.parametrize('verified', [False, True])
def test_finbert_only_scores_verified_news_and_archives_actual_outputs(monkeypatch, verified):
    pytest.importorskip('airflow')
    import dag_finbert_predict as predictor
    import chart_utils
    import model_registry
    import learned_signals
    import transformers
    import io
    import torch
    from datetime import datetime, timezone
    from types import SimpleNamespace

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 2, 22, tzinfo=timezone.utc)
    data = batch() if verified else pd.DataFrame({'ma_co_phieu': ['AAPL'], 'tieu_de': ['legacy headline']})
    key = 'finbert/finbert_news_20tickers_20261002.csv'
    objects = {key: data.to_csv(index=False)}
    calls = []
    class Store:
        def __init__(self, **kwargs): pass
        def get_conn(self): return self
        def list_keys(self, bucket_name, prefix): return [key]
        def get_key(self, *args, **kwargs):
            return SimpleNamespace(get=lambda: {'Body': io.BytesIO(objects[key].encode())})
        def load_string(self, string_data, key, bucket_name, replace):
            if not replace: assert key not in objects
            objects[key] = string_data
    class Model:
        config = SimpleNamespace(id2label={0: 'positive', 1: 'negative', 2: 'neutral'})
        def eval(self): pass
        def __call__(self, **kwargs): return SimpleNamespace(logits=torch.tensor([[2., 0., 1.]]))
    def tokenizer_loader(*args, **kwargs):
        assert verified
        calls.append('tokenizer')
        return lambda *args, **kwargs: {}
    def model_loader(*args, **kwargs):
        assert verified
        calls.append('model')
        return Model()
    monkeypatch.setattr(predictor, 'datetime', FixedDatetime)
    monkeypatch.setattr(predictor, 'S3Hook', Store)
    monkeypatch.setattr(predictor, 'SECTOR_MAP', {'AAPL': 'Technology', 'MSFT': 'Technology'})
    monkeypatch.setattr(predictor, 'TELEGRAM_TOKEN', '')
    monkeypatch.setattr(chart_utils, 'load_price_data_from_minio', lambda *args: pd.DataFrame())
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', tokenizer_loader)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, 'from_pretrained', model_loader)
    def unexpected(*args): raise AssertionError('Sentiment must not call a price head')
    monkeypatch.setattr(learned_signals, 'predict_signal', unexpected)
    monkeypatch.setattr(model_registry, 'save_model_to_minio', lambda *args, **kwargs: None)
    predictor.predict_finbert_task()
    forecast_key = next(k for k in objects if k.startswith('sentiment_history/finbert/'))
    forecasts = pd.read_csv(io.StringIO(objects[forecast_key])).set_index('ma_co_phieu')
    assert 'prediction_horizon' not in forecasts.columns
    assert forecasts.loc['MSFT', 'prob_num'] != forecasts.loc['MSFT', 'prob_num']
    if verified:
        from signal_formulas import finbert_signal
        assert forecasts.loc['AAPL', 'prob_num'] == pytest.approx(finbert_signal(forecasts.loc['AAPL']))
        assert forecasts.loc['AAPL', 'score_type'] == 'sentiment_signal'
    assert not any(key.startswith('forecast_history/finbert/') for key in objects)
    assert forecasts.loc['MSFT', 'news_count'] == 0
    archives = [k for k in objects if k.startswith('news_sentiment_history/')]
    if verified:
        assert calls == ['tokenizer', 'model']
        assert len(archives) == 1
        scored = pd.read_csv(io.StringIO(objects[archives[0]]))
        assert len(select_eligible_news(scored, '2026-10-02T22:01:00Z', require_sentiment=True)) == 1
        assert forecasts.loc['AAPL', 'news_count'] == 1
    else:
        assert calls == [] and archives == []
        assert forecasts.news_count.eq(0).all()


def test_crawler_preserves_immutable_runs_and_never_makes_missing_articles(monkeypatch):
    pytest.importorskip('airflow')
    import dag_crawl_finbert_news as crawler
    import data_validator
    import yfinance
    from types import SimpleNamespace
    objects, archive_keys = {}, []
    class Store:
        def __init__(self, **kwargs):
            pass
        def load_string(self, string_data, key, bucket_name, replace):
            if not replace:
                assert key not in objects
                archive_keys.append(key)
            objects[key] = string_data
        def load_bytes(self, bytes_data, key, bucket_name, replace):
            objects[key] = bytes_data
    monkeypatch.setattr(crawler, 'SECTOR_MAP', {'AAPL': 'Technology', 'MSFT': 'Technology'})
    monkeypatch.setattr(crawler, 'S3Hook', Store)
    monkeypatch.setattr(crawler, 'fetch_rss_news', lambda ticker: [])
    monkeypatch.setattr(crawler, 'fetch_yahoo_news', lambda ticker: [])
    monkeypatch.setattr(yfinance, 'Ticker', lambda ticker: SimpleNamespace(news=[]))
    monkeypatch.setattr(data_validator, 'save_and_alert_quality_report', lambda *args, **kwargs: None)
    crawler.crawl_news_task()
    crawler.crawl_news_task()
    assert len(archive_keys) == 4  # Two immutable CSVs and their audit reports.
    assert not any('quarterly financial outlook' in value for value in objects.values() if isinstance(value, str))
    import io
    current = next(key for key in objects if key.startswith('finbert/') and key.endswith('.csv'))
    assert pd.read_csv(io.StringIO(objects[current])).empty
