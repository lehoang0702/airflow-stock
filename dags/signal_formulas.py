"""Versioned legacy signal formulas, shared by batch and live analysis.

These are sentiment/composite scores, not calibrated price probabilities.
All internal values are fractions in [0, 1]; UI consumers convert to percent.
"""
import math
from numbers import Real

from sentiment_utils import sentiment_scores

FINBERT_FORMULA_VERSION = 'sentiment_linear_v1'
ENSEMBLE_FORMULA_VERSION = 'weighted_35_35_30_v1'
ENSEMBLE_SOURCE = 'weighted_ensemble_v1'
ENSEMBLE_WEIGHTS = {'prob_xgb': .35, 'prob_lstm': .35, 'prob_bert': .30}


def finbert_signal(values):
    scores = sentiment_scores(values)
    if not scores:
        return None
    score = (1 + scores['sentiment_positive'] - scores['sentiment_negative']) / 2
    return min(.95, max(.05, score))


def weighted_ensemble(values):
    """Normalize surviving weights; missing inputs never become neutral scores."""
    available = {key: float(values[key]) for key in ENSEMBLE_WEIGHTS
                 if isinstance(values.get(key), Real) and not isinstance(values[key], bool)
                 and math.isfinite(float(values[key])) and 0 <= values[key] <= 1}
    total = sum(ENSEMBLE_WEIGHTS[key] for key in available)
    if not total:
        return None, {}
    weights = {key: ENSEMBLE_WEIGHTS[key] / total for key in available}
    return sum(available[key] * weights[key] for key in available), weights


def signal_action(score):
    return ('Không khả dụng' if score is None else
            'MUA' if score >= .55 else 'BÁN' if score <= .45 else 'ĐỨNG NGOÀI')


def finbert_signal_text(values):
    score = finbert_signal(values)
    if score is None:
        return 'Không khả dụng'
    label = 'TÍCH CỰC' if score >= .55 else 'TIÊU CỰC' if score <= .45 else 'TRUNG LẬP'
    return f'{label} ({score * 100:.2f}%)'
