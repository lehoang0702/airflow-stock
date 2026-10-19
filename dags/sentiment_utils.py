"""Validated news sentiment shared by batch and live consumers."""
import math

from news_data import NEWS_POLICY

SENTIMENT_LABELS = {
    'sentiment_positive': 'Tích cực',
    'sentiment_negative': 'Tiêu cực',
    'sentiment_neutral': 'Trung tính',
}
SENTIMENT_PREFIX = 'finbert/sentiment_finbert_'
LEGACY_SENTIMENT_PREFIX = 'finbert/du_bao_tang_truong_finbert_'


def sentiment_scores(values):
    try:
        scores = {key: float(values[key]) for key in SENTIMENT_LABELS}
        if all(math.isfinite(value) and 0 <= value <= 1 for value in scores.values()) and math.isclose(sum(scores.values()), 1, abs_tol=.01):
            return scores
    except (KeyError, TypeError, ValueError):
        pass
    return {}


def sentiment_summary(values):
    scores = sentiment_scores(values)
    return SENTIMENT_LABELS[max(scores, key=scores.get)] if scores else 'Chưa có cảm xúc'


def stored_sentiment_details(row):
    """Read sentiment even from older exports, never their price probability."""
    scores = sentiment_scores(row)
    try:
        count = int(row.get('news_count', 0))
    except (ValueError, TypeError, OverflowError):
        count = 0
    version = row.get('sentiment_model_version')
    valid = (scores and count > 0 and row.get('news_policy') == NEWS_POLICY
             and isinstance(version, str) and bool(version.strip()))
    if not valid:
        return {'finbert_status': 'unavailable', 'finbert_sentiment': {},
                'finbert_news_count': 0, 'finbert_sources': [],
                'finbert_message': 'Chưa có kết quả phân tích cảm xúc hợp lệ.'}
    sources = row.get('news_sources')
    return {'finbert_status': 'available', 'finbert_sentiment': scores,
            'finbert_news_count': count,
            'finbert_sources': sources.split(' | ') if isinstance(sources, str) and sources else [],
            'finbert_message': f'Đã phân tích cảm xúc {count} tin.'}
