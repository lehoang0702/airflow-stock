"""Shared, timestamped news retrieval for ingestion and live inference."""
import logging
import re
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import urlencode

from news_data import normalize_batch, select_eligible_news, article_mentions_ticker

_CACHE = {}
_CACHE_LOCK = threading.Lock()
_FETCH_LOCKS = [threading.Lock() for _ in range(16)]
NEWS_CACHE_TTL = 300


def _rss_articles(url, publisher, limit=15, ticker=None):
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=10) as response:
        root = ET.fromstring(response.read())
    if root.find('channel') is None:
        raise ValueError('News source did not return an RSS channel')
    articles = []
    for item in root.findall('.//item'):
        title = (item.findtext('title') or '').strip()
        summary = re.sub(r'<[^>]+>', '', item.findtext('description') or '').strip()
        if not title or (ticker and not article_mentions_ticker(title + ' ' + summary, ticker)):
            continue
        articles.append({'title': title, 'summary': summary[:500],
                         'publisher': item.findtext('source') or publisher,
                         'pub_date': item.findtext('pubDate'), 'link': item.findtext('link')})
        if len(articles) >= limit:
            break
    return articles


def fetch_yahoo_news(ticker):
    # The yfinance /xhr/ncp news endpoint can return 404 even while price APIs work.
    query = urlencode({'s': ticker, 'region': 'US', 'lang': 'en-US'})
    return _rss_articles('https://feeds.finance.yahoo.com/rss/2.0/headline?' + query,
                         'Yahoo Finance')


def fetch_rss_news(ticker):
    """Scoped Google RSS and relevant CNBC items; preserve source failures."""
    articles, errors = [], []
    query = urlencode({'q': f'{ticker} stock OR {ticker} earnings', 'hl': 'en-US',
                       'gl': 'US', 'ceid': 'US:en'})
    for name, url, scoped in [
        ('Google News RSS', 'https://news.google.com/rss/search?' + query, None),
        ('CNBC RSS', 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664', ticker),
    ]:
        try:
            articles.extend(_rss_articles(url, name, limit=10, ticker=scoped))
        except Exception as exc:
            errors.append(f'{name}: {exc}')
            logging.warning('News source failed for %s: %s', ticker, errors[-1])
    if not articles and errors:
        raise RuntimeError('; '.join(errors))
    return articles


def collect_ticker_news(ticker, sector='General', *, yahoo_loader=None,
                        rss_loader=None, supplement=False, use_cache=True):
    """Return observed rows and diagnostics; fall back when no fresh usable news."""
    cacheable = use_cache and yahoo_loader is None and rss_loader is None and not supplement
    key = (ticker, sector)
    with _FETCH_LOCKS[hash(key) % len(_FETCH_LOCKS)]:
        if cacheable:
            with _CACHE_LOCK:
                cached = _CACHE.get(key)
                ttl = NEWS_CACHE_TTL if cached and cached[2]['status'] == 'ok' else 30
                if cached and time.monotonic() - cached[0] < ttl:
                    return cached[1].copy(deep=True), dict(cached[2])
        articles, errors, sources = [], {}, []
        yahoo_loader = yahoo_loader or fetch_yahoo_news
        rss_loader = rss_loader or fetch_rss_news
        try:
            primary = yahoo_loader(ticker) or []
            articles.extend(primary)
            if primary:
                sources.append('Yahoo Finance RSS')
        except Exception as exc:
            errors['Yahoo Finance RSS'] = str(exc)
            logging.warning('Yahoo news retrieval failed for %s: %s', ticker, exc)
        observed = datetime.now(timezone.utc)
        primary_batch = normalize_batch(articles, ticker, sector, observed)
        if supplement or select_eligible_news(primary_batch, observed).empty:
            try:
                fallback = rss_loader(ticker) or []
                articles.extend(fallback)
                if fallback:
                    sources.append('RSS fallback')
            except Exception as exc:
                errors['RSS fallback'] = str(exc)
                logging.warning('Fallback news retrieval failed for %s: %s', ticker, exc)
        observed = datetime.now(timezone.utc)
        batch = normalize_batch(articles, ticker, sector, observed)
        eligible = select_eligible_news(batch, observed)
        report = {'collected_at': observed.isoformat(), 'sources': sources,
                  'source_errors': errors, 'rows': len(batch), 'eligible_rows': len(eligible),
                  'status': 'ok' if not eligible.empty else 'source_error' if errors else 'no_eligible_news'}
        if cacheable:
            with _CACHE_LOCK:
                if len(_CACHE) >= 128:
                    _CACHE.pop(min(_CACHE, key=lambda entry: _CACHE[entry][0]))
                _CACHE[key] = (time.monotonic(), batch.copy(deep=True), dict(report))
        return batch, report
