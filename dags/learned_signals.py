"""Supervised direction heads. No hand-written probability conversions."""
import io
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from news_data import NEWS_POLICY

FEATURES = {
    'ensemble': ['prob_xgb', 'prob_lstm', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral'],
}
MIN_TRAINING_SAMPLES = 100
MIN_TRAINING_SESSIONS = 10


def load_signal_model(s3, bucket, kind, before=None):
    """Read completed, versioned artifacts; ignore incomplete training uploads."""
    try:
        prefix = f'models/learned/{kind}/'
        keys = []
        for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=prefix):
            keys.extend(item['Key'] for item in page.get('Contents', []) if item['Key'].endswith('/metadata.json'))
        for key in sorted(keys, reverse=True):
            meta = json.loads(s3.get_object(Bucket=bucket, Key=key)['Body'].read())
            if meta.get('status') != 'trained' or meta.get('feature_columns') != FEATURES[kind]:
                continue
            # Heads fitted to legacy placeholder/undated news are incompatible.
            if meta.get('news_policy') != NEWS_POLICY:
                continue
            if meta.get('training_samples', 0) < MIN_TRAINING_SAMPLES:
                continue
            if before is not None and pd.Timestamp(meta['trained_at']) >= pd.Timestamp(before):
                continue
            model_key = key.replace('metadata.json', 'model.joblib')
            model = joblib.load(io.BytesIO(s3.get_object(Bucket=bucket, Key=model_key)['Body'].read()))
            return model, meta
    except Exception as exc:
        logging.warning('Cannot load trained %s: %s', kind, exc)
    return None, None


def predict_signal(s3, bucket, kind, values):
    """Probability and provenance come from a fitted artifact or remain absent."""
    if s3 is None:
        return None, None
    features = FEATURES[kind]
    try:
        X = pd.DataFrame([{column: values.get(column) for column in features}])
        if not np.isfinite(X.astype(float).to_numpy()).all():
            return None, None
        model, metadata = load_signal_model(s3, bucket, kind)
        if model is None:
            return None, None
        probability = float(model.predict_proba(X)[0, list(model.classes_).index(1)])
        return probability, metadata['version']
    except Exception as exc:
        logging.warning('Cannot infer %s: %s', kind, exc)
        return None, None


def train_signal_model(s3, bucket, kind, observations):
    """Train on matured live forecasts; validate on later, purged sessions."""
    from model_evaluator import evaluate_binary_classifier
    features = FEATURES[kind]
    missing = set(features + ['reference_date', 'outcome_date', 'actual_up']) - set(observations.columns)
    if missing:
        return {'status': 'unavailable', 'message': 'Chưa đủ lịch sử dự báo hợp lệ để huấn luyện.', 'total_samples': 0}
    data = observations.copy()
    if not {'news_policy', 'news_count', 'sentiment_model_version'}.issubset(data.columns):
        return {'status': 'unavailable', 'message': 'Lịch sử cảm xúc thiếu nguồn tin và thời điểm quan sát đáng tin cậy.',
                'total_samples': 0, 'news_policy': NEWS_POLICY}
    data = data[data.news_policy.eq(NEWS_POLICY)
                & pd.to_numeric(data.news_count, errors='coerce').gt(0)
                & data.sentiment_model_version.notna()]
    sources = ('base_models_v2', 'trained_model_v2', 'weighted_ensemble_v1')
    if 'inference_source' not in data:
        return {'status': 'unavailable', 'message': 'Lịch sử thiếu nguồn gốc mô hình.', 'total_samples': 0}
    data = data[data['inference_source'].isin(sources)]
    data[features] = data[features].apply(pd.to_numeric, errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['actual_up'])
    data['reference_date'] = pd.to_datetime(data['reference_date'], utc=True).dt.tz_localize(None).dt.normalize()
    data['outcome_date'] = pd.to_datetime(data['outcome_date'], utc=True).dt.tz_localize(None).dt.normalize()
    data = data.sort_values('reference_date').drop_duplicates(['reference_date', 'ma_co_phieu'])
    sessions = sorted(data['reference_date'].unique())
    if len(sessions) < MIN_TRAINING_SESSIONS:
        return {'status': 'unavailable', 'message': 'Đang tích lũy lịch sử dự báo để huấn luyện.', 'total_samples': 0,
                'observed_samples': len(data), 'observed_sessions': len(sessions)}
    boundary = pd.Timestamp(sessions[int(len(sessions) * .8)])
    train = data[data['outcome_date'].lt(boundary)]
    test = data[data['reference_date'].ge(boundary)]
    if len(train) < MIN_TRAINING_SAMPLES or train['actual_up'].nunique() != 2 or len(test) < 20:
        return {'status': 'unavailable', 'message': 'Chưa đủ mẫu huấn luyện và kiểm chứng độc lập.', 'total_samples': 0,
                'observed_samples': len(data)}
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(train[features], train['actual_up'])
    metrics = evaluate_binary_classifier(test['actual_up'].to_numpy(), model.predict_proba(test[features])[:, 1], kind)
    trained_at = datetime.now(timezone.utc).isoformat()
    version = datetime.now(timezone.utc).strftime('v%Y%m%dT%H%M%S%f')
    metadata = {'status': 'trained', 'kind': kind, 'version': version, 'trained_at': trained_at,
                'news_policy': NEWS_POLICY,
                'training_samples': len(train), 'feature_columns': features,
                'train_end': str(train['outcome_date'].max().date()),
                'test_start': str(boundary.date()), 'metrics': metrics}
    buf = io.BytesIO()
    joblib.dump(model, buf)
    prefix = f'models/learned/{kind}/{version}/'
    s3.put_object(Bucket=bucket, Key=prefix + 'model.joblib', Body=buf.getvalue())
    s3.put_object(Bucket=bucket, Key=prefix + 'metadata.json', Body=json.dumps(metadata).encode())
    return {**metrics, 'status': 'ok', 'model_version': version, 'evaluation_date': trained_at,
            'evaluation_source': 'trained_model_holdout_v2', 'prediction_horizon': 1,
            'test_start': str(boundary.date()), 'test_end': str(test['outcome_date'].max().date())}
