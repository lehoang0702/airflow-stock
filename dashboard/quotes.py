"""Yahoo quote statistics, cached independently of candles and news."""
import copy
import logging
import math
import time
from datetime import datetime, timezone
from threading import Lock

QUOTE_TTL = 900
RETRY_TTL = 30
_CACHE = {}
_CACHE_LOCK = Lock()
_FETCH_LOCKS = [Lock() for _ in range(16)]
_FIELDS = {
    'bid': 'bid', 'ask': 'ask', 'market_cap': 'marketCap', 'beta': 'beta',
    'pe_ttm': 'trailingPE', 'eps_ttm': 'trailingEps',
    'low_52w': 'fiftyTwoWeekLow', 'high_52w': 'fiftyTwoWeekHigh',
    'average_volume': 'averageVolume', 'dividend_rate': 'dividendRate',
    # Yahoo's dividendYield is already a percentage (e.g. 0.28 = 0.28%).
    'dividend_yield_pct': 'dividendYield', 'target_mean_price': 'targetMeanPrice',
}


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _date(value):
    number = _number(value)
    if number is None or number <= 0:
        return None
    try:
        return datetime.fromtimestamp(number, timezone.utc).date().isoformat()
    except (ValueError, OverflowError, OSError):
        return None


def get_quote_statistics(ticker):
    """Keep working prices/news when Yahoo fundamentals fail; retain real stale data."""
    ticker = ticker.strip().upper()
    with _FETCH_LOCKS[hash(ticker) % len(_FETCH_LOCKS)]:
        now = time.monotonic()
        with _CACHE_LOCK:
            cached = _CACHE.get(ticker)
        if cached and now < cached['expires']:
            return copy.deepcopy(cached['data'])
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).get_info()
            if not isinstance(info, dict) or not info:
                raise ValueError('Empty Yahoo quote')
            data = {key: _number(info.get(source)) for key, source in _FIELDS.items()}
            for key in ('bid', 'ask', 'market_cap', 'low_52w', 'high_52w', 'target_mean_price'):
                if data[key] is not None and data[key] <= 0:
                    data[key] = None
            if not any(value is not None for value in data.values()):
                raise ValueError('No quote statistics available')
            dates = [_date(info.get(key)) for key in ('earningsTimestampStart', 'earningsTimestampEnd')]
            dates = sorted(set(date for date in dates if date))
            today = datetime.now(timezone.utc).date().isoformat()
            upcoming = [date for date in dates if date >= today]
            data.update(
                status='ok', source='Yahoo Finance',
                currency=str(info.get('currency') or 'USD'),
                fetched_at=datetime.now(timezone.utc).isoformat(),
                market_date=_date(info.get('regularMarketTime')),
                earnings_date=(upcoming[0] if upcoming else _date(info.get('earningsTimestamp'))),
                earnings_date_end=(upcoming[-1] if upcoming else None),
                earnings_is_estimate=bool(upcoming),
                ex_dividend_date=_date(info.get('exDividendDate')),
                message=None,
            )
            ttl = QUOTE_TTL
        except Exception as exc:
            logging.warning('Yahoo statistics unavailable for %s (%s)', ticker, type(exc).__name__)
            data = copy.deepcopy(cached['data']) if cached else {}
            data.update(status='stale' if cached and cached['data'].get('fetched_at') else 'unavailable',
                        source='Yahoo Finance',
                        message='Chưa cập nhật được thống kê Yahoo; đang hiển thị dữ liệu đã lấy trước đó.'
                        if cached and cached['data'].get('fetched_at') else
                        'Chưa lấy được thống kê từ Yahoo Finance. Các trường thiếu được giữ trống.')
            ttl = RETRY_TTL
        with _CACHE_LOCK:
            if ticker not in _CACHE and len(_CACHE) >= 128:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[ticker] = {'expires': time.monotonic() + ttl, 'data': data}
        return copy.deepcopy(data)
