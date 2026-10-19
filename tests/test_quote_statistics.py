"""Quote metadata must survive source failures without inventing absent metrics."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from dashboard import quotes


@pytest.fixture(autouse=True)
def clear_quote_cache():
    quotes._CACHE.clear()
    yield
    quotes._CACHE.clear()


def provider(monkeypatch, info):
    import yfinance
    monkeypatch.setattr(yfinance, 'Ticker', lambda _: SimpleNamespace(get_info=lambda: info))


def test_real_fields_units_dates_and_copy_isolation(monkeypatch):
    date = (datetime.now(timezone.utc) + timedelta(days=20)).replace(hour=20, minute=0, second=0)
    provider(monkeypatch, {'marketCap': 1882301399040, 'beta': 1.185, 'trailingPE': 27.84024,
        'trailingEps': 26.54, 'bid': 718., 'ask': 742.89, 'fiftyTwoWeekLow': 520.26,
        'fiftyTwoWeekHigh': 779.82, 'averageVolume': 19329054, 'dividendRate': 2.1,
        'dividendYield': .28, 'earningsTimestampStart': date.timestamp(),
        'earningsTimestampEnd': date.timestamp(), 'currency': 'USD'})
    result = quotes.get_quote_statistics(' meta ')
    assert result['status'] == 'ok'
    assert result['market_cap'] == 1882301399040
    assert result['dividend_yield_pct'] == .28
    assert result['earnings_date'] == date.date().isoformat()
    assert result['earnings_is_estimate'] is True
    assert result['target_mean_price'] is None
    result['pe_ttm'] = -1
    assert quotes.get_quote_statistics('META')['pe_ttm'] == 27.84024
    json.dumps(result, allow_nan=False)


def test_invalid_values_not_serialized_but_zero_eps_valid(monkeypatch):
    provider(monkeypatch, {'marketCap': float('inf'), 'trailingPE': float('nan'),
        'trailingEps': 0, 'bid': 0, 'ask': -2, 'beta': False,
        'dividendYield': 0, 'exDividendDate': 1e100})
    data = quotes.get_quote_statistics('META')
    assert data['eps_ttm'] == data['dividend_yield_pct'] == 0
    assert all(data[key] is None for key in ('market_cap', 'pe_ttm', 'bid', 'ask', 'beta', 'ex_dividend_date'))
    json.dumps(data, allow_nan=False)


def test_source_error_retains_actual_cached_values_and_timestamp(monkeypatch):
    provider(monkeypatch, {'trailingEps': -2, 'marketCap': 100})
    first = quotes.get_quote_statistics('META')
    quotes._CACHE['META']['expires'] = 0
    provider(monkeypatch, {})
    failed = quotes.get_quote_statistics('META')
    assert failed['status'] == 'stale'
    assert failed['eps_ttm'] == -2
    assert failed['fetched_at'] == first['fetched_at']
    assert failed['message']


def test_no_cache_and_source_error_has_no_fake_numbers(monkeypatch):
    provider(monkeypatch, {})
    data = quotes.get_quote_statistics('META')
    assert data['status'] == 'unavailable'
    assert not any(key in data for key in quotes._FIELDS)


def test_cache_avoids_repeated_provider_requests(monkeypatch):
    import yfinance
    calls = []
    monkeypatch.setattr(yfinance, 'Ticker', lambda ticker: (
        calls.append(ticker) or SimpleNamespace(get_info=lambda: {'trailingEps': 10})))
    quotes.get_quote_statistics('META')
    quotes.get_quote_statistics('META')
    assert calls == ['META']


def test_chart_before_analysis_uses_existing_job_api(monkeypatch):
    from dashboard import charts, jobs
    from dashboard.config import _ON_DEMAND_PREDICTIONS
    monkeypatch.delitem(_ON_DEMAND_PREDICTIONS, 'TEST', raising=False)
    calls = []
    monkeypatch.setattr(jobs, 'analysis_job', lambda **kw: calls.append(kw))
    assert charts.generate_live_chart_image('test', 'xgboost') is None
    assert calls == [{'ticker': 'TEST'}]
