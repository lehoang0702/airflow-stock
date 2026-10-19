"""Immutable training inputs and format-aware ingestion recovery."""
import hashlib
import io
import re
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd


def new_model_version():
    return datetime.now(timezone.utc).strftime('v%Y%m%dT%H%M%S%f') + '_' + uuid4().hex[:8]


def version_date(version):
    return datetime.strptime(str(version)[1:9], '%Y%m%d')


def snapshot_training_dataset(s3_hook, bucket, kind, version, raw_csv):
    key = f'dataset_history/{kind}/{version}.csv'
    s3_hook.load_string(string_data=raw_csv, key=key, bucket_name=bucket, replace=False)
    return key, hashlib.sha256(raw_csv.encode('utf-8')).hexdigest()


def load_backup_rows(s3_hook, bucket, prefix, ticker_column, tickers,
                     required_columns, today, exclude_today=True):
    """Try every compatible backup; CSV takes precedence over its Parquet twin."""
    keys = s3_hook.list_keys(bucket_name=bucket, prefix=prefix) or []
    candidates = []
    for key in keys:
        match = re.search(r'(\d{8})', key.rsplit('/', 1)[-1])
        if not match or not key.endswith(('.csv', '.parquet')):
            continue
        day = match.group(1)
        if day > today or (exclude_today and day == today):
            continue
        candidates.append(key)
    candidates.sort(key=lambda key: (key.rsplit('.', 1)[0], key.endswith('.csv')), reverse=True)
    recovered = {}
    for key in candidates:
        if len(recovered) == len(tickers):
            break
        try:
            if key.endswith('.csv'):
                frame = pd.read_csv(io.StringIO(s3_hook.read_key(key, bucket_name=bucket)))
            else:
                raw = s3_hook.get_key(key, bucket_name=bucket).get()['Body'].read()
                frame = pd.read_parquet(io.BytesIO(raw))
            if set(required_columns + [ticker_column]) - set(frame.columns):
                continue
            for ticker in tickers:
                subset = frame[frame[ticker_column].eq(ticker)].copy()
                if ticker not in recovered and len(subset) >= 50:
                    recovered[ticker] = subset
        except Exception:
            # A corrupt newest backup must not hide older usable observations.
            continue
    return recovered


def require_complete_panel(frame, ticker_column, date_column, expected_tickers):
    missing = set(expected_tickers) - set(frame[ticker_column].dropna())
    if missing or frame.empty:
        raise ValueError(f'Incomplete ticker coverage after cleaning: {sorted(missing)}')
    dates = pd.to_datetime(frame[date_column], errors='coerce', utc=True)
    if dates.isna().any() or frame.duplicated([ticker_column, date_column]).any():
        raise ValueError('Invalid or duplicate ticker-date observations')
    latest = dates.groupby(frame[ticker_column]).max()
    stale = latest[latest.lt(latest.max())]
    if not stale.empty:
        raise ValueError(f'Stale backfill; latest observed sessions differ: {stale.index.tolist()}')
