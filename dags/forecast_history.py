"""Join issued forecasts to their subsequent, observed trading sessions."""
import pandas as pd
from datetime import datetime, timezone
from uuid import uuid4


def save_forecast_history(s3_hook, bucket, kind, frame):
    """Keep every issuance immutable, including repeated runs on the same day."""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    key = f'forecast_history/{kind}/{stamp}_{uuid4().hex}.csv'
    s3_hook.load_string(string_data=frame.to_csv(index=False), key=key,
                        bucket_name=bucket, replace=False)
    return key


def load_forecast_history(s3_hook, bucket, kind):
    frames = []
    for key in sorted(s3_hook.list_keys(bucket_name=bucket, prefix=f'forecast_history/{kind}/') or []):
        if key.endswith('.csv'):
            import io
            frames.append(pd.read_csv(io.StringIO(s3_hook.read_key(key, bucket_name=bucket))))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    ticker_col = 'ma_co_phieu' if 'ma_co_phieu' in combined.columns else ('ticker' if 'ticker' in combined.columns else None)
    if ticker_col and 'reference_date' in combined.columns:
        subset = [ticker_col, 'reference_date']
        if 'prediction_horizon' in combined.columns:
            subset.append('prediction_horizon')
        if 'issued_at' in combined.columns:
            combined = combined.sort_values('issued_at')
        combined = combined.drop_duplicates(subset=subset, keep='last').reset_index(drop=True)
    return combined


def observed_forecasts(predictions, prices, horizon=5):
    """Return mature rows only; never score today's forecast on past returns.

    reference_date is the last closed session available to inference. issued_at
    is the actual publication time, independent of Airflow's logical date.
    """
    matched = []
    pending = invalid = 0
    if predictions.empty or prices.empty:
        return pd.DataFrame(), len(predictions), 0
    predictions = predictions.copy()
    ticker_col = 'ma_co_phieu' if 'ma_co_phieu' in predictions.columns else ('ticker' if 'ticker' in predictions.columns else None)
    if ticker_col and 'reference_date' in predictions.columns:
        subset = [ticker_col, 'reference_date']
        if 'prediction_horizon' in predictions.columns:
            subset.append('prediction_horizon')
        if 'issued_at' in predictions.columns:
            predictions = predictions.sort_values('issued_at')
        predictions = predictions.drop_duplicates(subset=subset, keep='last').reset_index(drop=True)
    prices = prices.copy()
    prices['Date'] = pd.to_datetime(prices['Date'], errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()
    prices['Close'] = pd.to_numeric(prices['Close'], errors='coerce')
    prices = prices.dropna(subset=['Date', 'Ticker', 'Close'])
    prices = prices[prices['Close'].gt(0)].sort_values('Date').drop_duplicates(['Ticker', 'Date'])
    groups = {ticker: group.reset_index(drop=True) for ticker, group in prices.groupby('Ticker')}
    for record in predictions.to_dict('records'):
        ticker = record.get('ma_co_phieu', record.get('ticker'))
        reference = pd.to_datetime(record.get('reference_date'), errors='coerce', utc=True)
        issued = pd.to_datetime(record.get('issued_at'), errors='coerce', utc=True)
        row_horizon = record.get('prediction_horizon', horizon)
        if pd.isna(reference) or pd.isna(issued) or ticker not in groups or pd.isna(row_horizon):
            invalid += 1
            continue
        try:
            steps = int(row_horizon)
        except (TypeError, ValueError):
            invalid += 1
            continue
        if steps < 1:
            invalid += 1
            continue
        reference = reference.tz_localize(None).normalize()
        issued_day = issued.tz_convert('America/New_York').tz_localize(None).normalize()
        series = groups[ticker]
        positions = series.index[series['Date'].eq(reference)]
        if len(positions) != 1 or reference > issued_day:
            invalid += 1
            continue
        start = int(positions[0])
        if start + steps >= len(series):
            pending += 1
            continue
        end = series.iloc[start + steps]
        # Outcome cannot already have happened at publication time. Require the
        # FIRST outcome session to be later too, including for multi-day labels.
        if series.iloc[start + 1]['Date'] <= issued_day:
            invalid += 1
            continue
        baseline = float(series.iloc[start]['Close'])
        change = float(end['Close']) / baseline - 1
        matched.append({**record, 'outcome_date': end['Date'],
                        'actual_return': change, 'actual_up': int(change > 0)})
    return pd.DataFrame(matched), pending, invalid
