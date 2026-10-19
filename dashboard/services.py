"""
DATA & ANALYTICS SERVICES — QUANTUM DASHBOARD
=============================================
Cung cấp dữ liệu nghiệp vụ cho các REST endpoints:
  - get_market_data(): Tổng hợp dự báo 4 mô hình, tính trading plan, tâm lý thị trường
  - get_ticker_chart(): Trả về nến OHLCV 260 phiên và tin tức thời gian thực
  - get_quality_data(): Báo cáo kiểm định chất lượng dữ liệu (Data Quality Gate)
  - get_evaluation_data(): Chỉ số backtest, TimeSeriesSplit CV, ROC-AUC, ma trận nhầm lẫn
"""

import json
import time
import logging
import math
from datetime import datetime
import pandas as pd

from dashboard.config import (
    BUCKET_NAME,
    CACHE_TTL,
    _CACHE,
    _ON_DEMAND_PREDICTIONS,
    TICKERS_META,
    EXTENDED_TICKERS_META,
    SECTOR_TRANSLATION,
)
from dashboard.storage import get_s3_client, get_latest_s3_key, read_s3_dataframe, extract_percentage
from dashboard.quotes import get_quote_statistics
from sentiment_utils import SENTIMENT_PREFIX, LEGACY_SENTIMENT_PREFIX, stored_sentiment_details
from signal_formulas import finbert_signal, finbert_signal_text, weighted_ensemble, signal_action, ENSEMBLE_SOURCE, ENSEMBLE_FORMULA_VERSION


def _unavailable_market_data(today_str: str):
    return {
        "date": None,
        "status": "unavailable",
        "market_mood": "CHƯA CÓ DỮ LIỆU",
        "message": "Chưa đọc được dữ liệu dự báo. Vui lòng kiểm tra pipeline và kết nối dữ liệu.",
        "buy_count": 0, "sell_count": 0, "hold_count": 0,
        "predictions": [],
    }


def _market_result(today_str: str, predictions: list) -> dict:
    if not predictions:
        return _unavailable_market_data(today_str)
    n_buy = sum(1 for p in predictions if 'MUA' in p.get('pp4_tong_hop', ''))
    n_sell = sum(1 for p in predictions if 'BÁN' in p.get('pp4_tong_hop', ''))
    n_hold = len(predictions) - n_buy - n_sell
    mood = "BULLISH (TÍCH CỰC)" if n_buy > n_sell else ("BEARISH (TIÊU CỰC)" if n_sell > n_buy else "SIDEWAY (ĐI NGANG)")

    return {
        "date": max((p.get("issued_at") or "" for p in predictions), default="") or None,
        "market_mood": mood if any(p.get("prob_ensemble") is not None for p in predictions) else "CHƯA CÓ ENSEMBLE",
        "buy_count": n_buy,
        "sell_count": n_sell,
        "hold_count": n_hold,
        "predictions": predictions
    }


def get_market_data():
    """Đọc dữ liệu bảng đối chiếu 4 phương pháp từ MinIO (kèm cache)."""
    now_ts = time.time()
    if _CACHE['market']['data'] is not None and (now_ts - _CACHE['market']['time']) < CACHE_TTL:
        return _CACHE['market']['data']

    s3 = get_s3_client()
    today_str = datetime.now().strftime("%Y-%m-%d")

    if not s3:
        result = _market_result(today_str, list(_ON_DEMAND_PREDICTIONS.values()))
        _CACHE['market'] = {'time': now_ts, 'data': result}
        return result

    latest_prices = {}
    price_key = get_latest_s3_key(s3, "lstm/lstm_stock_20tickers_10y_")
    if price_key:
        try:
            df_p = read_s3_dataframe(s3, price_key)
            if 'Ticker' in df_p.columns and 'Close' in df_p.columns:
                last_rows = df_p.sort_values(['Ticker', 'Date']).groupby('Ticker').last().reset_index()
                latest_prices = dict(zip(last_rows['Ticker'], last_rows['Close']))
        except Exception as e:
            logging.warning(f"Error reading price from S3: {e}")

    cmp_key = get_latest_s3_key(s3, "comparison/bang_so_sanh_4_phuong_phap_")
    predictions = []

    if cmp_key:
        try:
            df_cmp = read_s3_dataframe(s3, cmp_key)
            stored_sentiments = {}
            if 'sentiment_positive' not in df_cmp.columns:
                # Older comparisons have only a retired price score; read actual sentiment instead.
                date_suffix = cmp_key.rsplit('_', 1)[-1]
                for prefix in (SENTIMENT_PREFIX, LEGACY_SENTIMENT_PREFIX):
                    try:
                        frame = read_s3_dataframe(s3, prefix + date_suffix)
                        stored_sentiments = frame.set_index('ma_co_phieu').to_dict('index')
                        break
                    except Exception:
                        continue

            for _, r in df_cmp.iterrows():
                if r.get('inference_source') not in ('trained_model_v2', ENSEMBLE_SOURCE):
                    continue
                ticker = str(r.get('ma_co_phieu', '')).strip()
                cur_p = float(latest_prices.get(ticker, float('nan')))

                if not math.isfinite(cur_p) or cur_p <= 0:
                    continue

                pp1 = str(r.get('pp1_xgboost', ''))
                pp2 = str(r.get('pp2_lstm', ''))
                sentiment_row = r if 'sentiment_positive' in df_cmp.columns else stored_sentiments.get(ticker, {})
                if str(sentiment_row.get('reference_date')) != str(r.get('reference_date')):
                    sentiment_row = {}
                sentiment_details = stored_sentiment_details(sentiment_row)
                pp3 = finbert_signal_text(sentiment_details['finbert_sentiment'])
                bert_score = finbert_signal(sentiment_details['finbert_sentiment'])

                available = {name: value for name, value in {
                    'xgboost': extract_percentage(pp1), 'lstm': extract_percentage(pp2)}.items() if value is not None and 0 <= value <= 100}
                if bert_score is not None:
                    available['finbert'] = bert_score * 100
                if not available:
                    continue
                values = {f'prob_{name}': available[key] / 100 for key, name in [('xgboost', 'xgb'), ('lstm', 'lstm'), ('finbert', 'bert')] if key in available}
                ensemble_score, weights = weighted_ensemble(values)
                prob_e = ensemble_score * 100 if ensemble_score is not None else None
                prob_xgb = available.get('xgboost')
                prob_lstm = available.get('lstm')
                action_clean = signal_action(ensemble_score)
                target = float(r['gia_5d_du_kien']) if pd.notna(r.get('gia_5d_du_kien')) and pd.notna(r.get('return_model_version')) else None

                raw_sector = str(r.get('nhom_nganh', ''))
                vn_sector = SECTOR_TRANSLATION.get(raw_sector, TICKERS_META.get(ticker, {}).get('sector', 'Công nghệ'))

                predictions.append({
                    "ticker": ticker,
                    "name": TICKERS_META.get(ticker, {}).get('name', ticker),
                    "sector": vn_sector,
                    "sector_en": raw_sector,
                    "current_price": round(cur_p, 2),
                    "prob_xgb": prob_xgb,
                    "prob_lstm": prob_lstm,
                    **sentiment_details,
                    "prob_bert": round(bert_score * 100, 1) if bert_score is not None else None,
                    "finbert_score_type": "sentiment_linear_v1",
                    "ensemble_method": ENSEMBLE_FORMULA_VERSION,
                    "ensemble_weights": weights,
                    "prob_ensemble": round(prob_e, 1) if prob_e is not None else None,
                    "pp1_xgboost": pp1,
                    "pp2_lstm": pp2,
                    "pp3_finbert": pp3,
                    "pp4_tong_hop": action_clean,
                    "trang_thai": str(r.get('trang_thai_doi_chieu', 'Tổng hợp theo trọng số')),
                    "real_models_used": [name for key, name in [('xgboost', 'XGBoost'), ('lstm', 'LSTM')] if key in available] + (['FinBERT'] if sentiment_details['finbert_status'] == 'available' else []),
                    "plan_entry": round(cur_p, 2),
                    "plan_target": target,
                    "plan_stop_loss": None,
                    "plan_tp_pct": (target / cur_p - 1) * 100 if target is not None else None,
                    "plan_sl_pct": None,
                    "plan_rr": None,
                    "inference_source": ENSEMBLE_SOURCE,
                    "issued_at": r.get("issued_at"),
                    "reference_date": r.get("reference_date")
                })
        except Exception as e:
            logging.error(f"Error reading comparison S3 key {cmp_key}: {e}")

    if not predictions and not _ON_DEMAND_PREDICTIONS:
        return _unavailable_market_data(today_str)

    for sym, p_obj in _ON_DEMAND_PREDICTIONS.items():
        found = False
        for i, p in enumerate(predictions):
            if p['ticker'] == sym:
                predictions[i] = p_obj
                found = True
                break
        if not found:
            predictions.append(p_obj)

    result = _market_result(today_str, predictions)
    _CACHE['market']['time'] = now_ts
    _CACHE['market']['data'] = result
    return result


def _observed_candle(date, row):
    import math
    required = ('Open', 'High', 'Low', 'Close', 'Volume')
    try:
        values = {name: float(row[name]) for name in required}
        if not all(math.isfinite(value) for value in values.values()):
            return None
        rsi = row.get('rsi_14')
        return {'date': str(date)[:10],
                **{name.lower(): round(value, 2) for name, value in values.items() if name != 'Volume'},
                'volume': int(values['Volume']),
                'rsi': round(float(rsi), 1) if rsi is not None and math.isfinite(float(rsi)) else None}
    except (KeyError, TypeError, ValueError):
        return None


def get_ticker_chart(ticker: str):
    """Combine stored/live candles, observed news and Yahoo quote statistics."""
    now_ts = time.time()
    cached = _CACHE['tickers'].get(ticker)
    if cached and (now_ts - cached['time']) < CACHE_TTL:
        return cached['data']

    s3 = get_s3_client()
    candles = []
    news = []

    if s3:
        price_key = get_latest_s3_key(s3, "lstm/lstm_stock_20tickers_10y_")
        if price_key:
            try:
                df = read_s3_dataframe(s3, price_key)
                if 'Ticker' in df.columns:
                    sub = df[df['Ticker'] == ticker].sort_values('Date').tail(260)
                    for _, r in sub.iterrows():
                        candle = _observed_candle(r.get('Date'), r)
                        if candle is not None:
                            candles.append(candle)
            except Exception as e:
                logging.warning(f"Error fetching candles for {ticker}: {e}")

    from news_sources import collect_ticker_news
    from news_data import select_eligible_news
    from config_shared import SECTOR_MAP
    batch, news_report = collect_ticker_news(ticker, SECTOR_MAP.get(ticker, 'General'))
    eligible = select_eligible_news(batch, pd.Timestamp.now(tz='UTC'))
    if eligible.empty and s3:
        news_key = get_latest_s3_key(s3, 'finbert/finbert_news_20tickers_')
        if news_key:
            try:
                stored = select_eligible_news(read_s3_dataframe(s3, news_key), pd.Timestamp.now(tz='UTC'))
                eligible = stored[stored.ma_co_phieu.eq(ticker)]
                if not eligible.empty:
                    news_report = {**news_report, 'status': 'ok', 'sources': ['MinIO observed news'],
                                   'eligible_rows': len(eligible)}
            except Exception as exc:
                logging.warning('Stored news unavailable for %s: %s', ticker, exc)
    news = [{'headline': row['tieu_de'], 'source': row['nha_xuat_ban'],
             'date': row['published_at'].isoformat(), 'url': row['duong_dan']}
            for row in eligible.head(8).to_dict('records')]

    if not candles:
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            if not hist.empty:
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_series = 100 - (100 / (1 + rs))
                hist['rsi_14'] = rsi_series

                for dt, r in hist.tail(260).iterrows():
                    candle = _observed_candle(dt.strftime('%Y-%m-%d'), r)
                    if candle is not None:
                        candles.append(candle)
        except Exception as e:
            logging.warning(f"Error fetching yfinance live data for {ticker}: {e}")

    meta = TICKERS_META.get(ticker) or EXTENDED_TICKERS_META.get(ticker) or {'name': ticker, 'sector': 'Thị trường Mỹ'}
    result = {
        "ticker": ticker,
        "name": meta.get('name', ticker),
        "sector": meta.get('sector', 'Công nghệ'),
        "candles": candles,
        "news": news,
        "news_status": news_report,
        "statistics": get_quote_statistics(ticker),
    }
    _CACHE['tickers'][ticker] = {'time': now_ts, 'data': result}
    return result


def _s3_objects(s3, prefix):
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        yield from page.get('Contents', [])


def get_quality_data():
    """Expose actual validation reports; absence is not a successful check."""
    empty = {'status': 'unavailable', 'timestamp': None, 'total_rows': None,
             'total_columns': None, 'tickers_count': None,
             'expected_tickers_count': len(TICKERS_META), 'avg_missing_pct': None,
             'error_count': None, 'warning_count': None, 'datasets': [],
             'integrity_checks': [], 'message': 'Chưa có kết quả kiểm định dữ liệu.'}
    s3 = get_s3_client()
    if not s3:
        return empty
    try:
        latest = {}
        for item in _s3_objects(s3, 'quality/quality_report_'):
            name = item['Key'].rsplit('_', 1)[0]
            if name not in latest or item['LastModified'] > latest[name]['LastModified']:
                latest[name] = item
        reports = [json.loads(s3.get_object(Bucket=BUCKET_NAME, Key=item['Key'])['Body'].read())
                   for item in latest.values()]
        reports = [r for r in reports if isinstance(r.get('is_valid'), bool) and r.get('timestamp')]
        if not reports:
            return empty
        missing = [(r['total_rows'] * r['total_columns'], r['metrics']['avg_missing_pct']) for r in reports
                   if r.get('metrics', {}).get('avg_missing_pct') is not None and r.get('total_rows', 0) > 0 and r.get('total_columns', 0) > 0]
        avg_missing = (sum(rows * pct for rows, pct in missing) / sum(rows for rows, _ in missing)) if missing else None
        datasets = [{'name': r['dataset_name'], 'rows': r.get('total_rows'),
                     'columns': r.get('total_columns'), 'tickers': len(r.get('tickers_found', [])),
                     'status': 'PASSED' if r['is_valid'] else 'FAILED', 'timestamp': r['timestamp'],
                     'warnings': r.get('warnings', []), 'errors': r.get('errors', []), 'metrics': r.get('metrics', {})}
                    for r in reports]
        return {**empty, 'status': 'PASSED' if all(r['is_valid'] for r in reports) else 'FAILED',
                'timestamp': max(r['timestamp'] for r in reports),
                'total_rows': sum(r.get('total_rows', 0) for r in reports),
                'total_columns': sum(r.get('total_columns', 0) for r in reports),
                'tickers_count': len(set().union(*(set(r.get('tickers_found', [])) for r in reports))),
                'avg_missing_pct': avg_missing, 'datasets': datasets,
                'error_count': sum(len(r.get('errors', [])) for r in reports),
                'warning_count': sum(len(r.get('warnings', [])) for r in reports), 'message': None}
    except Exception as exc:
        logging.warning('Cannot load quality reports: %s', exc)
        return empty


def get_evaluation_data():
    """Read verifiable, versioned evaluation JSON. Reject legacy placeholders."""
    empty = {'status': 'unavailable', 'date': None, 'models_comparison': [],
             'cross_validation_folds': [],
             'message': 'Chưa có kết quả đánh giá từ mô hình đã huấn luyện.'}
    s3 = get_s3_client()
    if not s3:
        return empty
    try:
        models = []
        names = {'xgboost': 'XGBoost', 'lstm': 'LSTM', 'ensemble': 'Ensemble Master'}
        for kind, name in names.items():
            items = [item for item in _s3_objects(s3, f'evaluation/{kind}/eval_') if item['Key'].endswith('.json')]
            if not items:
                continue
            item = max(items, key=lambda item: item['LastModified'])
            metrics = json.loads(s3.get_object(Bucket=BUCKET_NAME, Key=item['Key'])['Body'].read())
            if metrics.get('evaluation_source') not in ('trained_model_holdout_v2', 'issued_trained_forecasts_v2', 'issued_weighted_forecasts_v1'):
                continue
            if kind == 'ensemble' and (metrics.get('evaluation_source') != 'issued_weighted_forecasts_v1'
                                       or metrics.get('ensemble_method') != ENSEMBLE_FORMULA_VERSION):
                continue
            if not metrics.get('evaluation_date'):
                continue
            metrics['model'] = name
            metrics['source_key'] = item['Key']
            models.append(metrics)
        if not models:
            return empty
        xgb = next((m for m in models if m['model'] == 'XGBoost'), {})
        return {**empty, 'status': 'ok' if any(m.get('total_samples', 0) > 0 for m in models) else 'unavailable',
                'date': max(m['evaluation_date'] for m in models), 'models_comparison': models,
                'cross_validation_folds': xgb.get('fold_details') or []}
    except Exception as exc:
        logging.warning('Cannot load evaluation: %s', exc)
        return empty
