"""Observed article versions and point-in-time news eligibility.

Unknown/date-only publication times are retained for audit, never guessed.
Legacy news without observation provenance cannot be used in historic features.
"""
from email.utils import parsedate_to_datetime
import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd

NEWS_POLICY = 'observed_news_v1'
NEWS_COLUMNS = ['ngay_thu_thap', 'ma_co_phieu', 'nhom_nganh', 'tieu_de', 'tom_tat',
                'nha_xuat_ban', 'thoi_gian_dang', 'duong_dan', 'published_at',
                'first_seen_at', 'available_at', 'story_id', 'article_version',
                'news_policy', 'timestamp_status', 'eligible_for_model']
SENTIMENT_COLUMNS = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']


def article_mentions_ticker(text, ticker):
    """Conservative relevance filter for an unscoped market news feed.

    Short tickers such as V/BA/CAT must not match letters or ordinary words.
    Yahoo's ticker feed and the scoped Google query are handled upstream.
    """
    from config_shared import TICKERS_META
    company = TICKERS_META.get(ticker, {}).get('name', '')
    company = re.sub(r'[\s,]+(Inc\.|Corp\.|Co\.|plc|Corporation|Company)$', '', company).strip()
    aliases = [company] if company else []
    aliases += {'GOOGL': ['Google'], 'AMZN': ['Amazon'], 'XOM': ['Exxon'],
                'JPM': ['JPMorgan'], 'UNH': ['UnitedHealth']}.get(ticker, [])
    if any(re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text, re.I) for name in aliases):
        return True
    tag = re.escape(ticker)
    if re.search(r'(?:\$|NASDAQ:\s*|NYSE:\s*)' + tag + r'\b|\(' + tag + r'\)', text, re.I):
        return True
    return len(ticker) > 3 and bool(re.search(r'\b' + tag + r'\b', text, re.I))


def precise_utc(value):
    """Accept timezone-aware instants and Unix seconds; reject calendar dates."""
    if value is None or value is pd.NaT:
        return pd.NaT
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            if not np.isfinite(value) or value <= 0:
                return pd.NaT
            return pd.to_datetime(value, unit='s', utc=True)
        text = str(value).strip()
        if not text or text.lower() in ('nan', 'nat', 'none') or re.fullmatch(r'\d{4}-\d{2}-\d{2}', text):
            return pd.NaT
        try:
            stamp = pd.Timestamp(value)
        except (ValueError, TypeError):
            stamp = pd.Timestamp(parsedate_to_datetime(text))
        if stamp.tzinfo is None:
            return pd.NaT
        return stamp.tz_convert('UTC')
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def article_url(value):
    try:
        parsed = urlsplit(str(value or '').strip())
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            return None
        path = parsed.path.rstrip('/')
        if not path or path in ('/finance', '/search') or re.fullmatch(r'/quote/[^/]+/news', path):
            return None
        query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                 if not k.lower().startswith('utm_') and k.lower() not in ('guccounter', 'guce_referrer', 'guce_referrer_sig')]
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, urlencode(sorted(query)), ''))
    except ValueError:
        return None


def normalize_article(article, ticker, sector, collected_at, source=None):
    seen = precise_utc(collected_at)
    if pd.isna(seen):
        raise ValueError('Collection requires a precise timezone-aware timestamp')
    title = str(article.get('title') or '').strip()
    if not title:
        return None
    summary = str(article.get('summary') or '').strip()
    publisher = str(article.get('publisher') or source or '').strip()
    url = article_url(article.get('link'))
    published = precise_utc(article.get('pub_date'))
    synthetic = publisher.casefold() == 'market update' or 'quarterly financial outlook and market trading update' in title.casefold()
    status = ('synthetic' if synthetic else 'missing_source_url' if url is None else
              'unknown_publication_time' if pd.isna(published) else
              'future_publication_time' if published > seen else 'verified')
    available = max(published, seen) if pd.notna(published) else pd.NaT
    story = url or ' '.join(title.casefold().split())
    version = '\n'.join((story, title, summary, str(published)))
    return {'ngay_thu_thap': seen.date().isoformat(), 'ma_co_phieu': ticker,
            'nhom_nganh': sector, 'tieu_de': title, 'tom_tat': summary,
            'nha_xuat_ban': publisher, 'duong_dan': url,
            'thoi_gian_dang': published.isoformat() if pd.notna(published) else None,
            'published_at': published.isoformat() if pd.notna(published) else None,
            'first_seen_at': seen.isoformat(),
            'available_at': available.isoformat() if pd.notna(available) else None,
            'story_id': hashlib.sha256(story.encode()).hexdigest(),
            'article_version': hashlib.sha256(version.encode()).hexdigest(),
            'news_policy': NEWS_POLICY, 'timestamp_status': status,
            'eligible_for_model': status == 'verified'}


def yahoo_article(item):
    content = item.get('content') if isinstance(item.get('content'), dict) else {}
    provider = content.get('provider') if isinstance(content.get('provider'), dict) else {}
    def link(name):
        value = content.get(name)
        return value.get('url') if isinstance(value, dict) else None
    return {'title': content.get('title') or item.get('title'),
            'summary': content.get('summary') or item.get('summary'),
            'publisher': provider.get('displayName') or item.get('publisher') or 'Yahoo Finance',
            'pub_date': content.get('pubDate') or item.get('providerPublishTime') or item.get('pubDate'),
            'link': link('canonicalUrl') or link('clickThroughUrl') or item.get('link')}


def normalize_batch(articles, ticker, sector, collected_at):
    rows = [normalize_article(article, ticker, sector, collected_at) for article in articles]
    frame = pd.DataFrame([row for row in rows if row], columns=NEWS_COLUMNS)
    if frame.empty:
        return frame
    # Prefer valid provenance when RSS/Yahoo return the same headline.
    frame['_title_key'] = frame.tieu_de.str.casefold().str.replace(r'\s+', ' ', regex=True).str.strip()
    return (frame.sort_values('eligible_for_model', ascending=False, kind='stable')
            .drop_duplicates(['ma_co_phieu', '_title_key'], keep='first')
            .drop(columns='_title_key').reset_index(drop=True))


def select_eligible_news(frame, cutoff, lookback_hours=72, require_sentiment=False):
    """Fail closed for legacy or manipulated availability metadata."""
    cutoff = precise_utc(cutoff)
    if pd.isna(cutoff) or lookback_hours <= 0:
        raise ValueError('News cutoff must be timezone-aware and lookback positive')
    required = set(NEWS_COLUMNS) - {'thoi_gian_dang', 'ngay_thu_thap', 'nhom_nganh', 'tom_tat'}
    if frame.empty or required - set(frame.columns):
        return frame.iloc[:0].copy()
    data = frame.copy()
    for name in ('published_at', 'first_seen_at', 'available_at'):
        data[name] = data[name].map(precise_utc)
    published = pd.to_datetime(data.published_at, utc=True)
    seen = pd.to_datetime(data.first_seen_at, utc=True)
    available = pd.to_datetime(data.available_at, utc=True)
    expected = pd.concat([published, seen], axis=1).max(axis=1)
    flag = data.eligible_for_model.astype(str).str.lower().isin(['true', '1'])
    valid = (data.news_policy.eq(NEWS_POLICY) & flag & data.timestamp_status.eq('verified')
             & published.notna() & seen.notna() & available.eq(expected)
             & published.le(seen) & seen.le(cutoff) & published.le(cutoff)
             & published.ge(cutoff - pd.Timedelta(hours=lookback_hours))
             & data.duong_dan.map(article_url).notna()
             & data.tieu_de.notna() & data.tieu_de.astype(str).str.strip().ne('')
             & data.nha_xuat_ban.astype(str).str.casefold().ne('market update')
             & ~data.tieu_de.astype(str).str.casefold().str.contains('quarterly financial outlook and market trading update', regex=False))
    if require_sentiment:
        if set(SENTIMENT_COLUMNS + ['sentiment_scored_at', 'sentiment_model_version']) - set(data):
            return data.iloc[:0]
        probabilities = data[SENTIMENT_COLUMNS].apply(pd.to_numeric, errors='coerce')
        scored = pd.to_datetime(data.sentiment_scored_at.map(precise_utc), utc=True)
        valid &= (probabilities.notna().all(axis=1) & probabilities.ge(0).all(axis=1)
                  & probabilities.le(1).all(axis=1) & probabilities.sum(axis=1).sub(1).abs().le(1e-4)
                  & scored.notna() & scored.ge(seen) & scored.le(cutoff)
                  & data.sentiment_model_version.notna())
        data[SENTIMENT_COLUMNS] = probabilities
    data = data[valid].sort_values('first_seen_at', kind='stable')
    # Select the latest observed content version of each story as of cutoff.
    data = data.drop_duplicates(['ma_co_phieu', 'article_version'], keep='first')
    return data.drop_duplicates(['ma_co_phieu', 'story_id'], keep='last').reset_index(drop=True)


def point_in_time_news_features(news, decisions, lookback_hours=72):
    """Explicit prediction timestamps are mandatory; absent sentiment stays NaN."""
    required = {'Ticker_Std', 'prediction_at'}
    if required - set(decisions):
        raise ValueError('News features require ticker and actual prediction timestamp')
    records = []
    for index, row in decisions.iterrows():
        eligible = select_eligible_news(news, row.prediction_at, lookback_hours, require_sentiment=True)
        eligible = eligible[eligible.ma_co_phieu.eq(row.Ticker_Std)] if 'ma_co_phieu' in eligible else eligible
        record = {'news_count': len(eligible), 'news_missing': int(eligible.empty)}
        record.update({name: float(eligible[name].mean()) if not eligible.empty else np.nan for name in SENTIMENT_COLUMNS})
        records.append(record)
    return pd.DataFrame(records, index=decisions.index)


def news_inventory(frame, cutoff):
    usable = select_eligible_news(frame, cutoff, lookback_hours=24 * 365 * 30)
    publication = frame.get('thoi_gian_dang', pd.Series(index=frame.index, dtype=object)).map(precise_utc)
    synthetic = frame.get('nha_xuat_ban', pd.Series('', index=frame.index)).astype(str).str.casefold().eq('market update')
    return {'rows': len(frame), 'precise_publication_rows': int(publication.notna().sum()),
            'synthetic_rows': int(synthetic.sum()), 'eligible_observed_rows': len(usable),
            'eligible_observation_days': int(pd.to_datetime(usable.first_seen_at, utc=True).dt.normalize().nunique()) if not usable.empty else 0,
            'legacy_or_unverified_rows': len(frame) - len(usable)}
