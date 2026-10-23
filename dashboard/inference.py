"""
INFERENCE & ON-DEMAND AI PIPELINE — QUANTUM DASHBOARD
======================================================
Load model weights từ MinIO (XGBoost, PyTorch LSTM, FinBERT Transformer),
tính toán đặc trưng kỹ thuật on-the-fly, và thực thi inference thời gian thực.
"""

import os
import io
import json
import time
import math
import logging
import tempfile
import threading
from datetime import datetime, timezone
import time
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np
import pandas as pd

from dashboard.config import (
    BUCKET_NAME,
    MODEL_CACHE_TTL,
    _MODEL_CACHE,
    _ON_DEMAND_PREDICTIONS,
    _CACHE,
    TICKERS_META,
    EXTENDED_TICKERS_META,
    SECTOR_TRANSLATION,
)
from sentiment_utils import sentiment_scores
from signal_formulas import finbert_signal, finbert_signal_text, weighted_ensemble, ENSEMBLE_SOURCE, ENSEMBLE_FORMULA_VERSION
from dashboard.storage import get_s3_client
from dashboard.charts import _CHART_CACHE

_FEATURE_PANEL_CACHE = {}

_FEATURE_PANEL_LOCK = threading.Lock()

_FETCH_CACHE = {}
_NUM_STRIPES = 16
_STRIPED_LOCKS = [threading.Lock() for _ in range(_NUM_STRIPES)]
_GLOBAL_FETCH_LOCK = threading.Lock()
FETCH_CACHE_TTL = 3600
MAX_FETCH_CACHE = 128

_MODEL_LOAD_LOCKS = {
    'xgboost': threading.Lock(),
    'lstm': threading.Lock(),
    'finbert': threading.Lock()
}

_MODEL_EXECUTOR = ThreadPoolExecutor(max_workers=6, thread_name_prefix='model_inference')

def _get_history(ticker, period, auto_adjust):
    key = (ticker, period, auto_adjust)
    now_mono = time.monotonic()

    with _GLOBAL_FETCH_LOCK:
        cached = _FETCH_CACHE.get(key)
        if cached is not None:
            if now_mono - cached['time'] < FETCH_CACHE_TTL:
                df = cached['data']
                return df.copy(deep=True) if hasattr(df, 'copy') else df
            else:
                del _FETCH_CACHE[key]
                
    stripe_lock = _STRIPED_LOCKS[hash(key) % _NUM_STRIPES]
    with stripe_lock:
        with _GLOBAL_FETCH_LOCK:
            cached = _FETCH_CACHE.get(key)
            if cached is not None and time.monotonic() - cached['time'] < FETCH_CACHE_TTL:
                df = cached['data']
                return df.copy(deep=True) if hasattr(df, 'copy') else df
                
        try:
            import yfinance as yf
            tk = yf.Ticker(ticker)
            df = tk.history(period=period, auto_adjust=auto_adjust, timeout=10.0)
            if df is not None and not df.empty:
                if df.index.name and 'date' in df.index.name.lower():
                    df.index.name = 'Date'
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                if all(c in df.columns for c in ['Open', 'High', 'Low', 'Close']):
                    df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
                    df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
        except Exception as e:
            logging.warning(f"⚠️ fetch history error {ticker}: {e}")
            df = pd.DataFrame()
            
        if df is not None and not df.empty:
            with _GLOBAL_FETCH_LOCK:
                if len(_FETCH_CACHE) >= MAX_FETCH_CACHE:
                    oldest = min(_FETCH_CACHE, key=lambda k: _FETCH_CACHE[k]['time'])
                    del _FETCH_CACHE[oldest]
                
                _FETCH_CACHE[key] = {
                    'time': time.monotonic(),
                    'data': df.copy(deep=True) if hasattr(df, 'copy') else df
                }
        
        return df.copy(deep=True) if df is not None and hasattr(df, 'copy') else df



def _complete_observed_model_features(frame, ticker, model_type, feature_cols):
    """Use the same scale-free indicators and observed peer panel as training."""
    from pipeline_data import add_stationary_features, inference_panel_features
    stationary = 'macd_hist_pct' in feature_cols
    if stationary:
        frame = add_stationary_features(frame)
    if not any(col.startswith('cs_') for col in feature_cols):
        return frame
    from dashboard.storage import read_s3_dataframe
    with _FEATURE_PANEL_LOCK:
        cached = _FEATURE_PANEL_CACHE.get(model_type)
        if cached is None or time.time() - cached['loaded_at'] >= MODEL_CACHE_TTL:
            s3 = get_s3_client()
            prefix = f'{model_type}/{model_type}_stock_100tickers_5y_'
            keys = [item['Key'] for page in s3.get_paginator('list_objects_v2').paginate(
                Bucket=BUCKET_NAME, Prefix=prefix) for item in page.get('Contents', [])
                if item['Key'].endswith('.csv')]
            if not keys:
                prefix = f'{model_type}/{model_type}_stock_20tickers_5y_'
                keys = [item['Key'] for page in s3.get_paginator('list_objects_v2').paginate(
                    Bucket=BUCKET_NAME, Prefix=prefix) for item in page.get('Contents', [])
                    if item['Key'].endswith('.csv')]
            if not keys:
                prefix = f'{model_type}/{model_type}_stock_20tickers_10y_'
                keys = [item['Key'] for page in s3.get_paginator('list_objects_v2').paginate(
                    Bucket=BUCKET_NAME, Prefix=prefix) for item in page.get('Contents', [])
                    if item['Key'].endswith('.csv')]
            if not keys:
                return frame
            panel = read_s3_dataframe(s3, sorted(keys)[-1])
            aliases = {str(col).lower(): col for col in panel.columns}
            ticker_col = aliases.get('ticker') or aliases.get('symbol')
            date_col = aliases.get('date') or aliases.get('datetime')
            needed = {'date', 'datetime', 'ticker', 'symbol', 'close', 'adj close', 'adj_close',
                      'log_return', 'ret_1d', 'log_ret', 'rsi_14', 'volume_sma_ratio',
                      'vol_sma_ratio', 'ret_close_sma5', 'ret_5', 'macd_hist'}
            panel = panel[[col for col in panel.columns if str(col).lower() in needed]]
            panel = panel.sort_values(date_col).groupby(ticker_col, sort=False).tail(300)
            cached = {'panel': panel, 'loaded_at': time.time()}
            _FEATURE_PANEL_CACHE[model_type] = cached
    result = inference_panel_features(frame, cached['panel'], ticker, stationary=stationary)
    required_cs = [col for col in feature_cols if col.startswith('cs_')]
    if any(col not in result for col in required_cs) or result[required_cs].iloc[-1].isna().any():
        raise ValueError('No complete contemporaneous peer panel for this forecast date')
    return result


def _load_xgboost_from_minio():
    """Load mô hình XGBoost mới nhất từ MinIO (boto3)."""
    with _MODEL_LOAD_LOCKS['xgboost']:
        now_ts = time.monotonic()
        cache = _MODEL_CACHE['xgboost']
        if cache['model'] is not None and (now_ts - cache['loaded_at']) < MODEL_CACHE_TTL:
            return cache['model'], cache['feature_cols']

        s3 = get_s3_client()
        if not s3:
            return None, None

        try:
            versions = set()
            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="models/xgboost/"):
                for obj in page.get('Contents', []):
                    parts = obj['Key'].replace("models/xgboost/", "").split("/")
                    if parts[0].startswith('v') and obj['Key'].endswith('model_card.json'):
                        versions.add(parts[0])

            if not versions:
                return None, None

            # Champion Gate: Lùi dần từ version mới nhất để tìm model đạt chuẩn (AUC >= 0.45, Acc >= 0.48, Lookback 14 ngày)
            selected_ver = None
            for ver in sorted(versions, reverse=True):
                try:
                    card_obj = s3.get_object(Bucket=BUCKET_NAME, Key=f"models/xgboost/{ver}/model_card.json")
                    card = json.loads(card_obj['Body'].read().decode('utf-8'))
                    from pipeline_storage import version_date
                    age = (datetime.now(timezone.utc).date() - version_date(ver).date()).days
                    if not 0 <= age <= 14:
                        continue
                    m = card.get('metrics', {})
                    auc = float(m.get('auc_roc', 0.0) or 0.0)
                    acc = float(m.get('accuracy', 0.0) or 0.0)
                    if math.isfinite(auc) and math.isfinite(acc) and 0.45 <= auc <= 1 and 0.48 <= acc <= 1:
                        selected_ver = ver
                        logging.info(f"🏆 Champion Gate Dashboard: Chọn XGBoost {ver} (AUC={auc:.3f}, Acc={acc:.1%})")
                        break
                except Exception as e:
                    logging.debug(f"Không thể kiểm tra model_card cho XGBoost {ver}: {e}")

            if not selected_ver:
                # Best-effort Fallback: Lấy version mới nhất có file model.json
                for ver in sorted(versions, reverse=True):
                    try:
                        s3.head_object(Bucket=BUCKET_NAME, Key=f"models/xgboost/{ver}/model.json")
                        selected_ver = ver
                        logging.warning(f"⚠️ Kích hoạt Best-Effort Fallback: Chọn XGBoost {ver} (mới nhất có sẵn)")
                        break
                    except Exception:
                        continue

            if not selected_ver:
                logging.warning("⚠️ Không tìm thấy phiên bản XGBoost khả dụng nào trên MinIO.")
                return None, None

            version = selected_ver
            prefix = f"models/xgboost/{version}/"
            logging.info(f"🧠 Đang load XGBoost model: {prefix}")

            import xgboost as xgb
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=f"{prefix}model.json")
            model_bytes = obj['Body'].read()

            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                f.write(model_bytes)
                tmp_path = f.name
            try:
                model = xgb.XGBClassifier()
                model.load_model(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            feature_cols = []
            try:
                obj_fc = s3.get_object(Bucket=BUCKET_NAME, Key=f"{prefix}feature_columns.json")
                feature_cols = json.loads(obj_fc['Body'].read().decode('utf-8'))
            except Exception:
                logging.warning("⚠️ Không tìm thấy feature_columns.json cho XGBoost")

            cache['version'] = version
            cache['model'] = model
            cache['feature_cols'] = feature_cols
            cache['loaded_at'] = time.monotonic()
            logging.info(f"✅ Đã load XGBoost model {version} ({len(feature_cols)} features)")
            return model, feature_cols

        except Exception as e:
            logging.warning(f"⚠️ Lỗi load XGBoost từ MinIO: {e}")
            return None, None


def _load_lstm_from_minio():
    """Load mô hình LSTM + scaler mới nhất từ MinIO (boto3)."""
    with _MODEL_LOAD_LOCKS['lstm']:
        now_ts = time.monotonic()
        cache = _MODEL_CACHE['lstm']
        if cache['model'] is not None and (now_ts - cache['loaded_at']) < MODEL_CACHE_TTL:
            return cache['model'], cache['scaler'], cache['feature_cols'], cache['input_dim']

        s3 = get_s3_client()
        if not s3:
            return None, None, None, 0

        try:
            import torch
            import torch.nn as nn
            import pickle

            versions = set()
            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="models/lstm/"):
                for obj in page.get('Contents', []):
                    parts = obj['Key'].replace("models/lstm/", "").split("/")
                    if parts[0].startswith('v') and obj['Key'].endswith('model_card.json'):
                        versions.add(parts[0])

            if not versions:
                return None, None, None, 0

            # Champion Gate: Lùi dần từ version mới nhất để tìm model đạt chuẩn (AUC >= 0.45, Acc >= 0.48, Lookback 14 ngày)
            selected_ver = None
            for ver in sorted(versions, reverse=True):
                try:
                    card_obj = s3.get_object(Bucket=BUCKET_NAME, Key=f"models/lstm/{ver}/model_card.json")
                    card = json.loads(card_obj['Body'].read().decode('utf-8'))
                    from pipeline_storage import version_date
                    age = (datetime.now(timezone.utc).date() - version_date(ver).date()).days
                    if not 0 <= age <= 14:
                        continue
                    m = card.get('metrics', {})
                    auc = float(m.get('auc_roc', 0.0) or 0.0)
                    acc = float(m.get('accuracy', 0.0) or 0.0)
                    if math.isfinite(auc) and math.isfinite(acc) and 0.45 <= auc <= 1 and 0.48 <= acc <= 1:
                        selected_ver = ver
                        logging.info(f"🏆 Champion Gate Dashboard: Chọn LSTM {ver} (AUC={auc:.3f}, Acc={acc:.1%})")
                        break
                except Exception as e:
                    logging.debug(f"Không thể kiểm tra model_card cho LSTM {ver}: {e}")

            if not selected_ver:
                # Best-effort Fallback: Lấy version mới nhất có file model.pt
                for ver in sorted(versions, reverse=True):
                    try:
                        s3.head_object(Bucket=BUCKET_NAME, Key=f"models/lstm/{ver}/model.pt")
                        selected_ver = ver
                        logging.warning(f"⚠️ Kích hoạt Best-Effort Fallback: Chọn LSTM {ver} (mới nhất có sẵn)")
                        break
                    except Exception:
                        continue

            if not selected_ver:
                logging.warning("⚠️ Không tìm thấy phiên bản LSTM khả dụng nào trên MinIO.")
                return None, None, None, 0

            version = selected_ver
            prefix = f"models/lstm/{version}/"
            logging.info(f"🧠 Đang load LSTM model: {prefix}")

            feature_cols = []
            try:
                obj_fc = s3.get_object(Bucket=BUCKET_NAME, Key=f"{prefix}feature_columns.json")
                feature_cols = json.loads(obj_fc['Body'].read().decode('utf-8'))
            except Exception:
                logging.warning("⚠️ Không tìm thấy feature_columns.json cho LSTM")
                return None, None, None, 0

            input_dim = len(feature_cols)

            hp = {'hidden_dim': 48, 'num_layers': 2, 'dropout': 0.2, 'temperature': 1.0}
            try:
                obj_card = s3.get_object(Bucket=BUCKET_NAME, Key=f"{prefix}model_card.json")
                card_data = json.loads(obj_card['Body'].read().decode('utf-8'))
                if 'hyperparameters' in card_data:
                    hp.update(card_data['hyperparameters'])
            except Exception:
                pass

            class CalibratedStackedLSTM(nn.Module):
                def __init__(self, in_dim, hidden_dim=48, num_layers=2, dropout=0.2):
                    super(CalibratedStackedLSTM, self).__init__()
                    self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers,
                                       batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
                    self.ln = nn.LayerNorm(hidden_dim)
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_dim, 24),
                        nn.ReLU(),
                        nn.Dropout(0.15),
                        nn.Linear(24, 1)
                    )

                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(self.ln(out[:, -1, :])).squeeze(-1)

            obj_model = s3.get_object(Bucket=BUCKET_NAME, Key=f"{prefix}model.pt")
            model_bytes = obj_model['Body'].read()

            model = CalibratedStackedLSTM(
                in_dim=input_dim,
                hidden_dim=int(hp.get('hidden_dim', 48)),
                num_layers=int(hp.get('num_layers', 2)),
                dropout=float(hp.get('dropout', 0.2))
            )
            buf = io.BytesIO(model_bytes)
            model.load_state_dict(torch.load(buf, weights_only=True, map_location='cpu'))
            model.eval()

            scaler = None
            try:
                obj_scaler = s3.get_object(Bucket=BUCKET_NAME, Key=f"{prefix}scaler.pkl")
                scaler_bytes = obj_scaler['Body'].read()
                scaler = pickle.loads(scaler_bytes)
            except Exception:
                logging.warning("⚠️ Không tìm thấy scaler.pkl cho LSTM")

            cache['version'] = version
            cache['model'] = model
            cache['scaler'] = scaler
            cache['feature_cols'] = feature_cols
            cache['input_dim'] = input_dim
            cache['loaded_at'] = time.monotonic()
            cache['lookback'] = int(hp.get('lookback', 30))
            logging.info(f"✅ Đã load LSTM model {version} (input_dim={input_dim})")
            return model, scaler, feature_cols, input_dim

        except Exception as e:
            logging.warning(f"⚠️ Lỗi load LSTM từ MinIO: {e}")
            return None, None, None, 0


def _load_finbert_model():
    """Load FinBERT tokenizer + model (cache trong RAM)."""
    with _MODEL_LOAD_LOCKS['finbert']:
        now_ts = time.monotonic()
        cache = _MODEL_CACHE['finbert']
        if cache['model'] is not None and (now_ts - cache['loaded_at']) < MODEL_CACHE_TTL:
            return cache['tokenizer'], cache['model']

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            logging.info("🧠 Đang load FinBERT model từ HuggingFace...")
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            model.eval()

            cache['tokenizer'] = tokenizer
            cache['model'] = model
            cache['loaded_at'] = time.monotonic()
            logging.info("✅ Đã load FinBERT model thành công")
            return tokenizer, model
        except Exception as e:
            logging.warning(f"⚠️ Lỗi load FinBERT: {e}")
            return None, None


def _calculate_xgboost_features_ondemand(df_stock, df_macro, symbol, sector):
    from feature_engineering import calculate_technical_indicators
    return calculate_technical_indicators(df_stock, df_macro, symbol, sector)


def _calculate_lstm_features_ondemand(df):
    from feature_engineering import calculate_technical_features
    return calculate_technical_features(df)


def _crawl_macro_data_ondemand():
    """Crawl dữ liệu vĩ mô (SPY, QQQ, VIX, TNX, Gold, Oil, DXY, HYG) cho XGBoost features."""
    import yfinance as yf

    try:
        from config_shared import MACRO_TICKERS as MACRO_DICT
    except ImportError:
        MACRO_DICT = {
            'SPY': 'spy', 'QQQ': 'qqq', '^VIX': 'vix', '^TNX': 'tnx',
            'GC=F': 'gold', 'CL=F': 'oil', 'DX-Y.NYB': 'dxy', 'HYG': 'hyg'
        }

    macro_dfs = {}
    for ticker, prefix in MACRO_DICT.items():
        retries = 3 if prefix in ('spy', 'qqq') else 2
        raw = None
        for attempt in range(retries):
            try:
                raw = _get_history(ticker, "2y", False)
                if raw is not None and not raw.empty:
                    break
            except Exception as e:
                pass
            if attempt < retries - 1:
                time.sleep(0.5)
        
        if raw is None or raw.empty:
            continue
            
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            cdf = raw.reset_index().copy()
            cdf.columns = [str(c).lower().replace(' ', '_') for c in cdf.columns]
            if 'date' not in cdf.columns:
                for c in cdf.columns:
                    if 'date' in c.lower():
                        cdf.rename(columns={c: 'date'}, inplace=True)
                        break
            cdf['date'] = pd.to_datetime(cdf['date']).dt.tz_localize(None)
            cdf[f'{prefix}_log_ret'] = np.log(cdf['close'] / cdf['close'].shift(1))
            macro_dfs[prefix] = cdf[['date', f'{prefix}_log_ret']].dropna()
        except Exception as e:
            logging.warning(f"⚠️ Không thể tải macro {ticker}: {e}")

    if 'spy' not in macro_dfs or 'qqq' not in macro_dfs:
        logging.warning("⚠️ Thiếu dữ liệu vĩ mô bắt buộc (SPY hoặc QQQ)")
        return pd.DataFrame()

    df_macro = pd.merge(macro_dfs['spy'], macro_dfs['qqq'], on='date', how='inner')
    for prefix in ['vix', 'tnx', 'gold', 'oil', 'dxy', 'hyg']:
        if prefix in macro_dfs:
            df_macro = pd.merge(df_macro, macro_dfs[prefix], on='date', how='left')

    return df_macro.ffill()


def _predict_xgboost_real(ticker, cur_price):
    """Chạy dự đoán XGBoost thật bằng model đã train."""
    model, feature_cols = _load_xgboost_from_minio()
    if model is None or not feature_cols:
        logging.info(f"XGBoost model không khả dụng cho {ticker}")
        return None

    try:
        import yfinance as yf
        df_raw = _get_history(ticker, "2y", False)
        if df_raw.empty or len(df_raw) < 60:
            return None

        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
        df = df_raw.reset_index().copy()
        df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]
        if 'adj_close' not in df.columns and 'close' in df.columns:
            df['adj_close'] = df['close']
        for c in df.columns:
            if 'date' in c.lower():
                df.rename(columns={c: 'date'}, inplace=True)
                break
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.sort_values('date').reset_index(drop=True)

        df_macro = _crawl_macro_data_ondemand()

        sector = 'General'
        meta_info = TICKERS_META.get(ticker) or EXTENDED_TICKERS_META.get(ticker)
        if meta_info:
            sector = SECTOR_TRANSLATION.get(meta_info.get('sector', ''), meta_info.get('sector', 'General'))

        df_feat = _calculate_xgboost_features_ondemand(df, df_macro, ticker, sector)
        df_feat = _complete_observed_model_features(df_feat, ticker, 'xgboost', feature_cols)

        available_feats = [c for c in feature_cols if c in df_feat.columns]
        missing_feats = [c for c in feature_cols if c not in df_feat.columns]

        if missing_feats:
            logging.warning(f"XGBoost không khả dụng cho {ticker}: thiếu {len(missing_feats)}/{len(feature_cols)} features")
            return None

        last_row = df_feat.iloc[[-1]].copy()
        for mf in missing_feats:
            last_row[mf] = 0.0

        X = df_feat[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().iloc[[-1]].astype(np.float32)
        if not np.isfinite(X.to_numpy()).all():
            return None
        prob = float(model.predict_proba(X)[:, 1][0])

        logging.info(f"✅ XGBoost real predict cho {ticker}: prob={prob:.4f}")
        return float(round(prob * 100.0, 1))

    except Exception as e:
        logging.warning(f"⚠️ XGBoost real predict lỗi cho {ticker}: {e}")
        return None


def _predict_lstm_real(ticker):
    """Chạy dự đoán LSTM thật bằng model đã train."""
    model, scaler, feature_cols, input_dim = _load_lstm_from_minio()
    if model is None or scaler is None or not feature_cols:
        logging.info(f"LSTM model không khả dụng cho {ticker}")
        return None

    try:
        import torch
        import yfinance as yf

        lookback = _MODEL_CACHE['lstm'].get('lookback', 30)
        df_raw = _get_history(ticker, "1y", True)
        if df_raw.empty or len(df_raw) < lookback + 20:
            return None

        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
        df = df_raw.reset_index().copy()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df['Ticker'] = ticker

        try:
            from config_shared import SECTOR_MAP
        except ImportError:
            SECTOR_MAP = {}
        df['Sector'] = SECTOR_MAP.get(ticker, 'General')

        df_feat = _calculate_lstm_features_ondemand(df)

        try:
            from config_shared import MACRO_TICKERS as MACRO_DICT
        except ImportError:
            MACRO_DICT = {'SPY': 'spy', 'QQQ': 'qqq', '^VIX': 'vix', '^TNX': 'tnx'}

        macro_dfs = {}
        for m_ticker_sym in MACRO_DICT:
            try:
                hist = _get_history(m_ticker_sym, "1y", True)
                if not hist.empty:
                    if isinstance(hist.columns, pd.MultiIndex):
                        hist.columns = hist.columns.get_level_values(0)
                    hist = hist.reset_index()
                    hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None).dt.normalize()
                    clean_name = m_ticker_sym.replace('^', '').replace('=', '').replace('-', '').replace('.', '').lower()
                    hist[f'{clean_name}_close'] = hist['Close']
                    hist[f'{clean_name}_ret'] = hist['Close'].pct_change(fill_method=None)
                    macro_dfs[m_ticker_sym] = hist[['Date', f'{clean_name}_close', f'{clean_name}_ret']].dropna()
            except Exception:
                pass

        if macro_dfs:
            base_macro = list(macro_dfs.values())[0]
            for other_m in list(macro_dfs.values())[1:]:
                base_macro = pd.merge(base_macro, other_m, on='Date', how='outer')
            df_macro_lstm = base_macro.sort_values('Date').ffill()
            df_feat = pd.merge(df_feat, df_macro_lstm, on='Date', how='left')

        df_feat = df_feat.ffill()
        df_feat = _complete_observed_model_features(df_feat, ticker, 'lstm', feature_cols)

        available_feats = [c for c in feature_cols if c in df_feat.columns]
        missing_feats = [c for c in feature_cols if c not in df_feat.columns]

        if missing_feats:
            logging.warning(f"⚠️ LSTM: quá nhiều features thiếu ({len(missing_feats)}/{len(feature_cols)})")
            return None

        for mf in missing_feats:
            df_feat[mf] = 0.0

        feat_data = df_feat[feature_cols].values.astype(np.float32)
        if not np.isfinite(feat_data[-lookback:]).all():
            return None

        if scaler is not None:
            feat_data = scaler.transform(feat_data)

        if len(feat_data) < lookback:
            return None

        seq = feat_data[-lookback:]
        X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            logit = model(X).item()
            prob = float(torch.sigmoid(torch.tensor(logit)).item())

        logging.info(f"✅ LSTM real predict cho {ticker}: prob={prob:.4f}")
        return float(round(prob * 100.0, 1))

    except Exception as e:
        logging.warning(f"⚠️ LSTM real predict lỗi cho {ticker}: {e}")
        return None


def _predict_finbert_real(ticker):
    """Score verified news and return sentiment without a price-direction forecast."""
    try:
        import torch
        from news_sources import collect_ticker_news
        from news_data import select_eligible_news
        from config_shared import SECTOR_MAP
        batch, report = collect_ticker_news(ticker, SECTOR_MAP.get(ticker, 'General'))
        batch = select_eligible_news(batch, datetime.now(timezone.utc)).head(10)
        details = {'news_count': len(batch), 'sources': report['sources'], 'news_report': report}
        if batch.empty:
            message = ('Không lấy được tin từ các nguồn hiện tại. Vui lòng thử lại.'
                       if report['status'] == 'source_error' else
                       'Chưa có tin hợp lệ trong 72 giờ qua để phân tích cảm xúc.')
            return {**details, 'status': report['status'], 'message': message}
        tokenizer, model = _load_finbert_model()
        if tokenizer is None or model is None:
            return {**details, 'status': 'model_error',
                          'message': 'Đã có tin tức nhưng chưa tải được mô hình phân tích cảm xúc FinBERT.'}
        scores = []
        for row in batch.to_dict('records'):
            inputs = tokenizer(row['tieu_de'], return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)[0].numpy()
            scores.append(probs)

        if scores:
            mean_probs = np.mean(scores, axis=0)
            labels = {str(label).lower(): int(index) for index, label in model.config.id2label.items()}
            values = {f'sentiment_{label}': float(mean_probs[labels[label]]) for label in ('positive', 'negative', 'neutral')}
            values.update(details)
            values.update(status='available', message=f'Đã phân tích cảm xúc {len(scores)} tin.')
            logging.info('FinBERT sentiment for %s: %s articles', ticker, len(scores))
            return values
        return {'status': 'unavailable', 'news_count': 0, 'message': 'Chưa có kết quả cảm xúc.'}

    except Exception as e:
        logging.warning(f"⚠️ FinBERT real predict lỗi cho {ticker}: {e}")
        return {'status': 'inference_error', 'news_count': 0,
                      'message': 'Không thể hoàn tất phân tích FinBERT. Vui lòng thử lại.'}


def _parse_scalar_prob(value):
    if value is None or isinstance(value, (bool, list, tuple, dict, set)):
        return None
    try:
        import numpy as np
        if not isinstance(value, (int, float, np.number)):
            return None
        val = float(value)
        if math.isfinite(val) and 0.0 <= val <= 100.0:
            return val
    except (TypeError, ValueError):
        pass
    return None

def _combine_available_model_predictions(predictions, sentiment=None):
    """Apply the legacy 35/35/30 formula to available, verified outputs."""
    available = {}
    for name, value in predictions.items():
        if name in ('xgboost', 'lstm', 'finbert'):
            val = _parse_scalar_prob(value)
            if val is not None:
                available[name] = val
    
    score = finbert_signal(sentiment or {})
    if score is not None:
        available['finbert'] = score * 100
    values = {f'prob_{name}': available.get(key) / 100 for key, name in
              [('xgboost', 'xgb'), ('lstm', 'lstm'), ('finbert', 'bert')] if key in available}
    probability, _ = weighted_ensemble(values)
    return (float(round(probability * 100, 1)) if probability is not None else None), available



def analyze_ticker_on_demand(ticker: str) -> dict:
    """
    Pipeline dự đoán thật sử dụng 4 mô hình AI:
    1. XGBoost — Load model từ MinIO, tính cùng bộ đặc trưng đã dùng khi huấn luyện, chạy predict_proba()
    2. LSTM — Load model PyTorch từ MinIO, chuẩn hóa features, chạy inference
    3. FinBERT — Load ProsusAI/finbert, phân tích sentiment tin tức thật
    4. Ensemble — tổng hợp 35% XGBoost + 35% LSTM + 30% điểm FinBERT.
    """
    ticker = ticker.strip().upper()
    _ON_DEMAND_PREDICTIONS.pop(ticker, None)
    _CACHE['market']['data'] = None
    for key in list(_CHART_CACHE):
        if key.startswith(f'{ticker}_'):
            del _CHART_CACHE[key]
    try:
        import yfinance as yf

        df = _get_history(ticker, "6mo", False)
        if df.empty:
            return {"status": "error", "message": f"Không tìm thấy dữ liệu giao dịch cho mã {ticker}"}

        cur_p = round(float(df['Close'].iloc[-1]), 2)
        if not math.isfinite(cur_p) or cur_p <= 0:
            return {'status': 'error', 'message': f'Giá tham chiếu của {ticker} không hợp lệ.'}
        logging.info(f"🚀 Bắt đầu phân tích on-demand THẬT cho {ticker} (3 models song song)...")

        prob_xgb_real = None
        prob_lstm_real = None
        bert_values = None

        futures = {
            _MODEL_EXECUTOR.submit(_predict_xgboost_real, ticker, cur_p): 'xgboost',
            _MODEL_EXECUTOR.submit(_predict_lstm_real, ticker): 'lstm',
            _MODEL_EXECUTOR.submit(_predict_finbert_real, ticker): 'finbert',
        }

        completed, _ = wait(futures)
        for future in completed:
            model_name = futures[future]
            try:
                result = future.result()
                if isinstance(result, (tuple, list)):
                    prob_val = result[0] if len(result) > 0 else None
                else:
                    prob_val = result
                    
                if model_name == 'xgboost':
                    prob_xgb_real = prob_val
                elif model_name == 'lstm':
                    prob_lstm_real = prob_val
                elif model_name == 'finbert':
                    bert_values = result if isinstance(result, dict) else None
            except Exception as e:
                logging.warning(f"⚠️ Model {model_name} lỗi cho {ticker}: {e}")

        prob_e, available = _combine_available_model_predictions({
            'xgboost': prob_xgb_real, 'lstm': prob_lstm_real
        }, bert_values)
        if not available and not sentiment_scores(bert_values or {}):
            return {'status': 'error', 'message': f'Không có mô hình AI khả dụng để dự đoán {ticker}.'}

        prob_xgb = available.get('xgboost')
        prob_lstm = available.get('lstm')
        prob_bert = round(available['finbert'], 1) if 'finbert' in available else None
        used_real_models = [name for key, name in [('xgboost', 'XGBoost'), ('lstm', 'LSTM')] if key in available]
        if sentiment_scores(bert_values or {}):
            used_real_models.append('FinBERT')
        logging.info('Models available for %s: %s', ticker, used_real_models)
        ensemble_values = {f'prob_{name}': available[key] / 100 for key, name in
                           [('xgboost', 'xgb'), ('lstm', 'lstm'), ('finbert', 'bert')] if key in available}
        _, ensemble_weights = weighted_ensemble(ensemble_values)

        def signal(prob, positive, negative, neutral):
            if prob is None:
                return 'Không khả dụng'
            return (f'{positive} ({prob}%)' if prob >= 55.0 else
                    f'{negative} ({prob}%)' if prob <= 45.0 else f'{neutral} ({prob}%)') + ' [AI]'

        pp1_str = signal(prob_xgb, 'MUA', 'BÁN', 'ĐỨNG NGOÀI')
        pp2_str = signal(prob_lstm, 'MUA', 'BÁN', 'ĐỨNG NGOÀI')
        pp3_str = finbert_signal_text(bert_values or {})

        if prob_e is None:
            action_clean = 'CHƯA CÓ ENSEMBLE'
        elif 48 <= prob_e <= 52:
            action_clean = 'Tín hiệu yếu / Trung tính (Neutral)'
        else:
            action_clean = 'MUA' if prob_e >= 55 else 'BÁN' if prob_e <= 45 else 'ĐỨNG NGOÀI'
        target = stop_loss = tp_pct = sl_pct = plan_rr = None

        recs = ['MUA' if p >= 55.0 else ('BÁN' if p <= 45.0 else 'HOLD') for p in available.values()]
        n_mua = recs.count('MUA')
        n_ban = recs.count('BÁN')
        if not recs:
            status = 'Chỉ có cảm xúc tin tức'
        elif len(recs) == 1:
            status = f'Chỉ có {used_real_models[0]} khả dụng'
        elif n_mua == len(recs):
            status = '🟢 MUA ĐỒNG THUẬN'
        elif n_ban == len(recs):
            status = '🔴 BÁN ĐỒNG THUẬN'
        elif n_mua > len(recs) / 2:
            status = '🟢 MUA (ĐA SỐ)'
        elif n_ban > len(recs) / 2:
            status = '🔴 BÁN (ĐA SỐ)'
        else:
            status = '🟡 PHÂN HÓA'

        meta_info = TICKERS_META.get(ticker) or EXTENDED_TICKERS_META.get(ticker) or {}
        name = meta_info.get('name') or ticker
        sector = meta_info.get('sector') or 'Chưa có thông tin'
        sector_vn = SECTOR_TRANSLATION.get(sector, sector)

        prediction_obj = {
            "ticker": ticker,
            "name": name,
            "sector": sector_vn,
            "current_price": cur_p,
            "prob_xgb": prob_xgb,
            "prob_lstm": prob_lstm,
            "prob_bert": prob_bert,
            "finbert_score_type": "sentiment_linear_v1",
            "ensemble_method": ENSEMBLE_FORMULA_VERSION,
            "ensemble_weights": ensemble_weights,
            "prob_ensemble": prob_e,
            "finbert_status": (bert_values or {}).get('status', 'unavailable'),
            "finbert_message": (bert_values or {}).get('message'),
            "finbert_news_count": (bert_values or {}).get('news_count', 0),
            "finbert_sources": (bert_values or {}).get('sources', []),
            "finbert_sentiment": {key: bert_values[key] for key in (
                'sentiment_positive', 'sentiment_negative', 'sentiment_neutral')
                if bert_values and key in bert_values},
            "pp1_xgboost": pp1_str,
            "pp2_lstm": pp2_str,
            "pp3_finbert": pp3_str,
            "pp4_tong_hop": action_clean,
            "trang_thai": status,
            "plan_entry": cur_p,
            "plan_target": target,
            "plan_stop_loss": stop_loss,
            "plan_tp_pct": tp_pct,
            "plan_sl_pct": sl_pct,
            "plan_rr": plan_rr,
            "on_demand": True,
            "inference_source": ENSEMBLE_SOURCE,
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "reference_date": str(df.index[-1]),
            "real_models_used": used_real_models,
            "analyzed_at": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
        }

        if len(_ON_DEMAND_PREDICTIONS) > 100:
            _ON_DEMAND_PREDICTIONS.pop(next(iter(_ON_DEMAND_PREDICTIONS)))
        _ON_DEMAND_PREDICTIONS[ticker] = prediction_obj
        _CACHE['market']['data'] = None

        return {"status": "success", "prediction": prediction_obj}
    except Exception as e:
        logging.exception(f"Lỗi phân tích on-demand cho mã {ticker}: {e}")
        return {"status": "error", "message": str(e)}
