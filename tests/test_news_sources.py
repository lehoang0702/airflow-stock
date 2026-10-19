"""Provider failures must preserve usable news without inventing price scores."""
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dags'))
import news_sources as sources
from news_data import normalize_batch, select_eligible_news


@pytest.fixture(autouse=True)
def isolated_news(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 7, 6, tzinfo=timezone.utc)
    monkeypatch.setattr(sources, 'datetime', Clock)
    sources._CACHE.clear()


def article(**changes):
    return {'title': 'Meta earnings increase', 'summary': 'Observed report',
            'publisher': 'Yahoo Finance', 'pub_date': '2026-10-07T05:00:00Z',
            'link': 'https://finance.yahoo.com/news/meta-earnings-050000.html', **changes}


def test_yahoo_404_falls_back_to_real_timestamped_rss(monkeypatch):
    xml = b'''<rss><channel><item><title>Meta earnings increase</title>
    <link>https://example.com/meta</link><source>Reuters</source>
    <pubDate>Wed, 07 Oct 2026 05:00:00 GMT</pubDate></item></channel></rss>'''
    def fetch(request, timeout):
        if request.full_url.startswith('https://news.google.com/'):
            return io.BytesIO(xml)
        raise HTTPError(request.full_url, 404, 'Not Found', {}, None)
    monkeypatch.setattr(sources.urllib.request, 'urlopen', fetch)
    batch, report = sources.collect_ticker_news('META', use_cache=False)
    assert report['status'] == 'ok' and report['eligible_rows'] == 1
    assert '404' in report['source_errors']['Yahoo Finance RSS']
    assert report['sources'] == ['RSS fallback']
    assert batch.nha_xuat_ban.tolist() == ['Reuters']
    assert batch.published_at.tolist() == ['2026-10-07T05:00:00+00:00']


def test_valid_yahoo_rss_does_not_call_fallback():
    def unexpected(_): raise AssertionError('Valid primary should not need fallback')
    batch, report = sources.collect_ticker_news('META', yahoo_loader=lambda _: [article()],
                                               rss_loader=unexpected)
    assert report['status'] == 'ok' and report['sources'] == ['Yahoo Finance RSS']
    assert len(batch) == 1


@pytest.mark.parametrize('date', [None, '2026-09-30T05:00:00Z'])
def test_unusable_primary_news_uses_fallback(date):
    batch, report = sources.collect_ticker_news('META',
        yahoo_loader=lambda _: [article(pub_date=date)],
        rss_loader=lambda _: [article(title='Fresh report', link='https://example.com/fresh')])
    eligible = select_eligible_news(batch, sources.datetime.now(timezone.utc))
    assert eligible.tieu_de.tolist() == ['Fresh report']
    assert report['eligible_rows'] == 1


def test_provider_error_is_distinct_from_no_articles():
    def failed(_): raise HTTPError('https://example.com/rss', 404, 'Not Found', {}, None)
    batch, report = sources.collect_ticker_news('META', yahoo_loader=failed, rss_loader=failed)
    assert batch.empty and report['status'] == 'source_error'
    assert len(report['source_errors']) == 2
    _, empty = sources.collect_ticker_news('META', yahoo_loader=lambda _: [], rss_loader=lambda _: [])
    assert empty['status'] == 'no_eligible_news'


def test_cache_preserves_observation_time_and_isolates_returned_rows(monkeypatch):
    calls = []
    monkeypatch.setattr(sources, 'fetch_yahoo_news', lambda _: calls.append('fetch') or [article()])
    first, report = sources.collect_ticker_news('META')
    first.loc[0, 'tieu_de'] = 'mutated'
    second, second_report = sources.collect_ticker_news('META')
    assert calls == ['fetch'] and second.tieu_de.iloc[0] == article()['title']
    assert second.first_seen_at.iloc[0] == report['collected_at'] == second_report['collected_at']


def test_non_rss_response_activates_fallback(monkeypatch):
    monkeypatch.setattr(sources.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(b'<html>error</html>'))
    batch, report = sources.collect_ticker_news('META', rss_loader=lambda _: [article()])
    assert len(batch) == 1 and report['status'] == 'ok'
    assert 'RSS channel' in report['source_errors']['Yahoo Finance RSS']


def test_finbert_reports_sentiment_without_calling_price_head(monkeypatch):
    import torch
    from dashboard import inference
    batch = normalize_batch([article()], 'META', 'General', '2026-10-07T06:00:00Z')
    monkeypatch.setattr(sources, 'collect_ticker_news', lambda *a: (batch, {'status': 'ok', 'sources': ['Yahoo Finance RSS']}))
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None): return cls(2026, 10, 7, 6, tzinfo=timezone.utc)
    monkeypatch.setattr(inference, 'datetime', Clock)
    model = SimpleNamespace(config=SimpleNamespace(id2label={0:'positive',1:'negative',2:'neutral'}))
    class Scorer:
        config = model.config
        def __call__(self, **kw): return SimpleNamespace(logits=torch.tensor([[2., 0., 1.]]))
    monkeypatch.setattr(inference, '_load_finbert_model', lambda: (lambda *a, **kw: {}, Scorer()))
    def unexpected(*args): raise AssertionError('FinBERT must not call a price head')
    monkeypatch.setattr(inference, 'weighted_ensemble', unexpected)
    details = inference._predict_finbert_real('META')
    assert details['status'] == 'available'
    assert details['news_count'] == 1 and details['sources'] == ['Yahoo Finance RSS']
    assert sum(details[key] for key in ['sentiment_positive','sentiment_negative','sentiment_neutral']) == pytest.approx(1.)


def test_finbert_source_failure_does_not_load_model(monkeypatch):
    from dashboard import inference
    monkeypatch.setattr(sources, 'collect_ticker_news', lambda *a: (pd.DataFrame(), {'status':'source_error','sources':[]}))
    def unexpected(): raise AssertionError('No usable news')
    monkeypatch.setattr(inference, '_load_finbert_model', unexpected)
    details = inference._predict_finbert_real('META')
    assert details['status'] == 'source_error'
    assert details['news_count'] == 0


@pytest.mark.parametrize('news_available', [False, True])
def test_quote_prices_survive_news_failure_and_use_shared_feed(monkeypatch, news_available):
    import yfinance
    from dashboard import services
    now = pd.Timestamp.now(tz='UTC')
    batch = normalize_batch([article(pub_date=(now - pd.Timedelta(minutes=15)).isoformat())],
                            'META', 'General', now) if news_available else pd.DataFrame()
    report = {'status': 'ok' if news_available else 'source_error',
              'sources': ['Yahoo Finance RSS'] if news_available else [], 'eligible_rows': int(news_available)}
    monkeypatch.setattr(sources, 'collect_ticker_news', lambda *a: (batch, report))
    monkeypatch.setattr(services, 'get_s3_client', lambda: None)
    monkeypatch.setattr(services, 'get_quote_statistics', lambda _: {'status': 'unavailable'})
    history = pd.DataFrame({'Open':100., 'High':110., 'Low':90., 'Close':105., 'Volume':1000.},
                           index=pd.date_range('2026-01-01', periods=35))
    monkeypatch.setattr(yfinance, 'Ticker', lambda _: SimpleNamespace(history=lambda **kw: history))
    services._CACHE['tickers'].pop('META', None)
    result = services.get_ticker_chart('META')
    assert len(result['candles']) == 35 and result['candles'][-1]['close'] == 105.
    assert len(result['news']) == int(news_available)
    assert result['news_status']['status'] == report['status']
    assert result['statistics']['status'] == 'unavailable'
    json.dumps(result, allow_nan=False)  # Exercise the actual HTTP serialization contract.
