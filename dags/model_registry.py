"""
MODULE: MODEL REGISTRY — XÂY DỰNG & QUẢN LÝ MÔ HÌNH
=====================================================
Đóng gói toàn bộ logic xây dựng, serialize, lưu trữ và quản lý phiên bản mô hình.

Schema MinIO (version = UTC timestamp + unique suffix):
    models/xgboost/<version>/model.json + model_card.json + feature_columns.json
    models/lstm/<version>/model.pt + scaler.pkl + model_card.json + feature_columns.json
    models/finbert/<version>/model_card.json + sentiment_outputs.json
    models/registry/<model_type>/<version>.json (legacy: models/registry.csv)
"""

from pipeline_data import (prepare_target, clean_features, temporal_masks, create_sequences,
                           add_stationary_features, select_model_features)

import io
import json
import logging
import pickle
from pipeline_storage import new_model_version, version_date
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ==============================================================================
#  MODEL CARD — Metadata cho mỗi phiên bản mô hình
# ==============================================================================

@dataclass
class ModelCard:
    """Thẻ mô hình chứa toàn bộ metadata của 1 phiên bản model."""
    model_name: str                        # VD: "XGBoost", "LSTM", "FinBERT"
    model_version: str                     # VD: "v20260904"
    trained_date: str                      # VD: "2026-09-04"
    model_type: str                        # "xgboost" | "lstm" | "finbert"

    # Dataset
    dataset_size: int = 0                  # Số dòng dữ liệu training
    num_features: int = 0                  # Số features đầu vào
    num_tickers: int = 0                   # Số mã cổ phiếu
    train_period: str = ""                 # VD: "2021-01-01 → 2023-12-31"
    test_period: str = ""                  # VD: "2024-01-01 → 2026-09-04"

    # Hyperparameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Metrics (được tính thực sự bởi model_evaluator.py)
    metrics: Dict[str, float] = field(default_factory=dict)

    # Feature list
    feature_columns: List[str] = field(default_factory=list)

    # Ghi chú bổ sung
    notes: str = ""

    def to_json(self) -> str:
        """Serialize ModelCard sang JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'ModelCard':
        """Deserialize ModelCard từ JSON string."""
        data = json.loads(json_str)
        return cls(**data)


# ==============================================================================
#  XGBOOST MODEL BUILDER
# ==============================================================================

class XGBoostModelBuilder:
    """
    Đóng gói toàn bộ pipeline xây dựng mô hình XGBoost:
    Tiền xử lý → Lọc features → Train XGBClassifier → Serialize
    """

    DEFAULT_HYPERPARAMS = {
        'n_estimators': 120,
        'max_depth': 3,
        'learning_rate': 0.03,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'min_child_weight': 3,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }

    def __init__(self, hyperparams: Optional[Dict] = None):
        self.hyperparams = hyperparams or self.DEFAULT_HYPERPARAMS.copy()
        self.model = None
        self.feature_cols = []
        self.model_card = None

    def preprocess(self, df: pd.DataFrame, feature_policy='legacy') -> Tuple[pd.DataFrame, List[str]]:
        """Tiền xử lý dữ liệu: chuẩn hóa cột, tạo target, lọc features."""
        cols_lower = {c.lower(): c for c in df.columns}
        date_col = cols_lower.get('date') or cols_lower.get('datetime') or df.columns[0]
        ticker_col = cols_lower.get('symbol') or cols_lower.get('ticker') or cols_lower.get('ma_co_phieu')
        sector_col = cols_lower.get('sector') or cols_lower.get('nhom_nganh')
        close_col = cols_lower.get('close') or cols_lower.get('adj close') or cols_lower.get('adj_close')

        df['Date_Std'] = pd.to_datetime(df[date_col], errors='coerce').dt.tz_localize(None)
        df['Ticker_Std'] = df[ticker_col] if ticker_col else 'UNKNOWN'
        df['Sector_Std'] = df[sector_col] if sector_col else 'General'

        # Target: giá phiên sau > giá phiên hiện tại
        target_col = cols_lower.get('future_direction_1d') or cols_lower.get('target')
        df = prepare_target(df, target_col, close_col)

        from pipeline_data import add_cross_sectional_features
        stationary = feature_policy == 'stationary_v2'
        if stationary:
            df = add_stationary_features(df)
        df, _ = add_cross_sectional_features(df, stationary=stationary)
        feature_cols = select_model_features(df, stationary=stationary)
        df[feature_cols] = clean_features(df, feature_cols)

        self.feature_cols = feature_cols
        return df, feature_cols

    def train(self, df: pd.DataFrame, train_cutoff: str = '2024-01-01') -> Dict[str, float]:
        """Train mô hình XGBoost và trả về metrics thực sự (KHÔNG clip)."""
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, roc_auc_score

        df, feature_cols = self.preprocess(df)

        # Split train/test theo thời gian
        train_mask, test_mask = temporal_masks(df, train_cutoff)

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, 'Target_Std']
        X_test = df.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, 'Target_Std']

        # Train
        self.model = xgb.XGBClassifier(**self.hyperparams)
        self.model.fit(X_train, y_train)

        # Đánh giá trên test set — metrics THỰC SỰ (không clip)
        test_probs = self.model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)
        test_acc = float(accuracy_score(y_test, test_preds))
        try:
            test_auc = float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) == 2 else None
        except Exception:
            test_auc = None

        # Tạo model card
        date_nodash = datetime.now().strftime("%Y%m%d")
        train_dates = df.loc[train_mask, 'Date_Std']
        test_dates = df.loc[test_mask, 'Date_Std']

        self.model_card = ModelCard(
            model_name="XGBoost Gradient Boosting Classifier",
            model_version=new_model_version(),
            trained_date=datetime.now().strftime("%Y-%m-%d"),
            model_type="xgboost",
            dataset_size=int(train_mask.sum()),
            num_features=len(feature_cols),
            num_tickers=int(df['Ticker_Std'].nunique()),
            train_period=f"{train_dates.min().strftime('%Y-%m-%d')} → {train_dates.max().strftime('%Y-%m-%d')}",
            test_period=f"{test_dates.min().strftime('%Y-%m-%d')} → {test_dates.max().strftime('%Y-%m-%d')}",
            hyperparameters=self.hyperparams,
            metrics={'accuracy': test_acc, 'auc_roc': test_auc},
            feature_columns=feature_cols,
            notes=f"Test set: {int(test_mask.sum())} samples"
        )

        metrics = {'accuracy': test_acc, 'auc_roc': test_auc}
        logging.info(f"✅ XGBoost đã train xong | Acc: {test_acc*100:.2f}% | AUC: {test_auc}")
        return metrics

    def serialize(self) -> bytes:
        """Serialize XGBoost model sang JSON bytes."""
        try:
            return bytes(self.model.get_booster().save_raw(raw_format='json'))
        except Exception:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                tmp_path = f.name
            try:
                self.model.save_model(tmp_path)
                with open(tmp_path, 'rb') as f:
                    return f.read()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    @staticmethod
    def deserialize(model_bytes: bytes):
        """Deserialize XGBoost model từ JSON bytes."""
        import xgboost as xgb
        model = xgb.XGBClassifier()
        try:
            model.load_model(bytearray(model_bytes))
            return model
        except Exception:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                f.write(model_bytes)
                tmp_path = f.name
            try:
                model.load_model(tmp_path)
                return model
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Dự báo xác suất tăng giá cho các hàng cuối cùng."""
        if self.model is None:
            raise ValueError("Model chưa được train hoặc load!")
        ticker_col = 'Ticker_Std' if 'Ticker_Std' in df.columns else ('Ticker' if 'Ticker' in df.columns else None)
        if ticker_col:
            X = df.groupby(ticker_col)[self.feature_cols].transform(
                lambda grp: grp.replace([np.inf, -np.inf], np.nan).ffill()
            ).fillna(0.0).astype(np.float32)
        else:
            X = df[self.feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).astype(np.float32)
        return self.model.predict_proba(X)[:, 1]


# ==============================================================================
#  LSTM MODEL BUILDER
# ==============================================================================

class LSTMModelBuilder:
    """
    Đóng gói toàn bộ pipeline xây dựng mô hình LSTM PyTorch:
    Chuẩn hóa → Tạo sliding windows → Train CalibratedStackedLSTM → Serialize
    """

    DEFAULT_HYPERPARAMS = {
        'hidden_dim': 48,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.0015,
        'weight_decay': 1e-3,
        'batch_size': 128,
        'epochs': 15,
        'lookback': 30,
        'temperature': 1.0
    }

    def __init__(self, hyperparams: Optional[Dict] = None):
        self.hyperparams = hyperparams or self.DEFAULT_HYPERPARAMS.copy()
        self.model = None
        self.scaler = None
        self.feature_cols = []
        self.model_card = None
        self._model_class = None  # Reference to CalibratedStackedLSTM class

    def _build_model_class(self):
        """Xây dựng class CalibratedStackedLSTM (lazy import torch)."""
        import torch.nn as nn

        hp = self.hyperparams

        class CalibratedStackedLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim=hp['hidden_dim'],
                         num_layers=hp['num_layers'], dropout=hp['dropout']):
                super(CalibratedStackedLSTM, self).__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
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

        self._model_class = CalibratedStackedLSTM
        return CalibratedStackedLSTM

    def preprocess(self, df: pd.DataFrame, feature_policy='legacy') -> Tuple[pd.DataFrame, List[str]]:
        """Tiền xử lý dữ liệu cho LSTM."""
        cols_lower = {c.lower(): c for c in df.columns}
        date_col = cols_lower.get('date') or cols_lower.get('datetime') or df.columns[0]
        ticker_col = cols_lower.get('ticker') or cols_lower.get('symbol') or cols_lower.get('ma_co_phieu')
        sector_col = cols_lower.get('sector') or cols_lower.get('nhom_nganh')
        close_col = cols_lower.get('close') or cols_lower.get('adj close') or cols_lower.get('adj_close')

        df['Date_Std'] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
        df['Ticker_Std'] = df[ticker_col] if ticker_col else 'UNKNOWN'
        df['Sector_Std'] = df[sector_col] if sector_col else 'General'

        target_col = cols_lower.get('future_direction_1d') or cols_lower.get('target')
        df = prepare_target(df, target_col, close_col)

        from pipeline_data import add_cross_sectional_features
        stationary = feature_policy == 'stationary_v2'
        if stationary:
            df = add_stationary_features(df)
        df, _ = add_cross_sectional_features(df, stationary=stationary)
        feature_cols = select_model_features(df, stationary=stationary)

        df[feature_cols] = clean_features(df, feature_cols)
        df = df.sort_values(['Date_Std', 'Ticker_Std']).reset_index(drop=True)

        self.feature_cols = feature_cols
        return df, feature_cols

    def _create_sequences(self, data_group, feature_cols, lookback, mask=None):
        return create_sequences(data_group, feature_cols, lookback, mask=mask)

    def train(self, df: pd.DataFrame, train_cutoff: str = '2024-01-01') -> Dict[str, float]:
        """Train mô hình LSTM và trả về metrics thực sự (KHÔNG clip)."""
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, roc_auc_score

        torch.set_num_threads(2)

        df, feature_cols = self.preprocess(df)
        hp = self.hyperparams
        lookback = hp['lookback']

        # Chuẩn hóa features
        self.scaler = StandardScaler()
        train_mask_idx = df['Date_Std'] < train_cutoff
        self.scaler.fit(df.loc[train_mask_idx, feature_cols].values)

        scaled_df = df.copy()
        scaled_df[feature_cols] = self.scaler.transform(df[feature_cols].values)
        # Bỏ Target ở phiên cuối của tập train để tránh leakage sang test
        last_train_rows = scaled_df[train_mask_idx].groupby('Ticker_Std').tail(1).index
        scaled_df.loc[last_train_rows, 'Target_Std'] = np.nan

        # Tạo sequences với history liền mạch
        test_mask_idx = scaled_df['Date_Std'] >= train_cutoff
        X_train, y_train = self._create_sequences(scaled_df, feature_cols, lookback, mask=train_mask_idx)
        X_test, y_test = self._create_sequences(scaled_df, feature_cols, lookback, mask=test_mask_idx)

        # Build và train model
        ModelClass = self._build_model_class()
        self.model = ModelClass(input_dim=len(feature_cols))
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=hp['learning_rate'],
                                      weight_decay=hp['weight_decay'])

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=hp['batch_size'], shuffle=True
        )

        self.model.train()
        for epoch in range(hp['epochs']):
            for bx, by in train_loader:
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        # Đánh giá test set — metrics THỰC SỰ (không clip)
        self.model.eval()
        with torch.no_grad():
            if len(X_test) > 0:
                test_logits = self.model(torch.tensor(X_test)).numpy()
                temp = 1.0
                test_probs = 1.0 / (1.0 + np.exp(-test_logits / temp))
                test_preds = (test_probs >= 0.5).astype(int)
                test_acc = float(accuracy_score(y_test, test_preds))
                try:
                    test_auc = float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) == 2 else None
                except Exception:
                    test_auc = None
            else:
                raise ValueError('No independent LSTM test observations')

        # Tạo model card
        date_nodash = datetime.now().strftime("%Y%m%d")
        train_dates = scaled_df.loc[train_mask_idx, 'Date_Std']
        test_dates = scaled_df.loc[test_mask_idx, 'Date_Std']

        self.model_card = ModelCard(
            model_name="Calibrated Stacked LSTM (PyTorch)",
            model_version=new_model_version(),
            trained_date=datetime.now().strftime("%Y-%m-%d"),
            model_type="lstm",
            dataset_size=len(X_train),
            num_features=len(feature_cols),
            num_tickers=int(df['Ticker_Std'].nunique()),
            train_period=f"{train_dates.min().strftime('%Y-%m-%d')} → {train_dates.max().strftime('%Y-%m-%d')}",
            test_period=f"{test_dates.min().strftime('%Y-%m-%d')} → {test_dates.max().strftime('%Y-%m-%d')}",
            hyperparameters=hp,
            metrics={'accuracy': test_acc, 'auc_roc': test_auc},
            feature_columns=feature_cols,
            notes=f"Test set: {len(X_test)} sequences, lookback={lookback}"
        )

        metrics = {'accuracy': test_acc, 'auc_roc': test_auc}
        logging.info(f"✅ LSTM đã train xong | Acc: {test_acc*100:.2f}% | AUC: {test_auc}")
        return metrics

    def serialize_model(self) -> bytes:
        """Serialize PyTorch model state_dict."""
        import torch
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        return buf.getvalue()

    def serialize_scaler(self) -> bytes:
        """Serialize StandardScaler."""
        return pickle.dumps(self.scaler)

    def deserialize_model(self, model_bytes: bytes, input_dim: int):
        """Deserialize PyTorch model từ bytes."""
        import torch
        ModelClass = self._build_model_class()
        model = ModelClass(input_dim=input_dim)
        buf = io.BytesIO(model_bytes)
        model.load_state_dict(torch.load(buf, weights_only=True))
        model.eval()
        self.model = model
        return model

    def deserialize_scaler(self, scaler_bytes: bytes):
        """Deserialize StandardScaler từ bytes."""
        self.scaler = pickle.loads(scaler_bytes)
        return self.scaler


# ==============================================================================
#  FINBERT MODEL BUILDER
# ==============================================================================

class FinBERTModelBuilder:
    """Pretrained sentiment inference; price forecasts require a fitted head."""
    DEFAULT_CONFIG = {'model_name': 'ProsusAI/finbert', 'max_length': 128}

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self.DEFAULT_CONFIG.copy()
        self.model_card = None
        self.sentiment_stats = {}

    def run_inference(self, df_news: pd.DataFrame, today_str: str) -> Tuple[List[Dict], Dict]:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        model = AutoModelForSequenceClassification.from_pretrained(self.config['model_name'])
        model.eval()
        labels = {str(label).lower(): int(index) for index, label in model.config.id2label.items()}
        results = []
        for (ticker, sector), group in df_news.groupby(['ma_co_phieu', 'nhom_nganh']):
            outputs = []
            for title in group['tieu_de'].dropna():
                if not str(title).strip():
                    continue
                inputs = tokenizer(str(title), return_tensors='pt', truncation=True, max_length=self.config['max_length'])
                with torch.no_grad():
                    outputs.append(torch.softmax(model(**inputs).logits, dim=-1)[0].numpy())
            if not outputs:
                continue
            average = np.mean(outputs, axis=0)
            sentiment = {f'sentiment_{label}': float(average[labels[label]]) for label in ('positive', 'negative', 'neutral')}
            results.append({'ngay_du_bao': today_str, 'ma_co_phieu': ticker, 'nhom_nganh': sector,
                            'prob_num': None, 'xac_suat_tang_gia': None,
                            'khuyen_nghi': 'CHƯA CÓ MÔ HÌNH DỰ BÁO GIÁ', **sentiment})
            self.sentiment_stats[ticker] = {'total_news': len(outputs), **sentiment}
        return results, self.sentiment_stats


# ==============================================================================
#  MINIO MODEL STORAGE — Lưu trữ & Quản lý phiên bản
# ==============================================================================

def save_model_to_minio(s3_hook, bucket_name: str, model_type: str,
                        model_card: ModelCard,
                        model_bytes: Optional[bytes] = None,
                        scaler_bytes: Optional[bytes] = None,
                        extra_artifacts: Optional[Dict[str, str]] = None):
    """
    Lưu model artifact + model_card lên MinIO.

    Args:
        s3_hook: Airflow S3Hook
        bucket_name: MinIO bucket
        model_type: "xgboost" | "lstm" | "finbert"
        model_card: ModelCard instance
        model_bytes: Serialized model (XGBoost JSON hoặc PyTorch state_dict)
        scaler_bytes: Serialized scaler (chỉ LSTM)
        extra_artifacts: Dict of {filename: json_string} để upload thêm
    """
    version = model_card.model_version
    prefix = f"models/{model_type}/{version}/"
    if s3_hook.check_for_key(key=prefix + 'model_card.json', bucket_name=bucket_name):
        raise FileExistsError(f'Model version already published: {model_type}/{version}')
    if model_type in ('xgboost', 'lstm', 'xgboost_return_5d') and model_bytes is None:
        raise ValueError('Publishing a model requires its trained weights')
    if model_type == 'lstm' and scaler_bytes is None:
        raise ValueError('Publishing LSTM requires its matching scaler')
    if model_bytes is not None:
        filename = {'xgboost': 'model.json', 'lstm': 'model.pt',
                    'xgboost_return_5d': 'model.json'}.get(model_type, 'model.bin')
        s3_hook.load_bytes(bytes_data=model_bytes, key=prefix + filename,
                           bucket_name=bucket_name, replace=False)
    if scaler_bytes is not None:
        s3_hook.load_bytes(bytes_data=scaler_bytes, key=prefix + 'scaler.pkl',
                           bucket_name=bucket_name, replace=False)
    if model_card.feature_columns:
        s3_hook.load_string(string_data=json.dumps(model_card.feature_columns),
                            key=prefix + 'feature_columns.json', bucket_name=bucket_name, replace=False)
    for filename, content in (extra_artifacts or {}).items():
        s3_hook.load_string(string_data=content, key=prefix + filename,
                            bucket_name=bucket_name, replace=False)
    # The card is the completion marker: readers never see a partly uploaded version.
    s3_hook.load_string(string_data=model_card.to_json(), key=prefix + 'model_card.json',
                        bucket_name=bucket_name, replace=False)
    _update_registry(s3_hook, bucket_name, model_card)
    logging.info('Published immutable model %s/%s', model_type, version)


def load_model_from_minio(s3_hook, bucket_name: str, model_type: str,
                          version: Optional[str] = None) -> Optional[Dict]:
    """
    Download model artifact từ MinIO.

    Args:
        s3_hook: Airflow S3Hook
        bucket_name: MinIO bucket
        model_type: "xgboost" | "lstm" | "finbert"
        version: VD "v20260904". None = lấy phiên bản mới nhất.

    Returns:
        Dict chứa: model_card, model_bytes, scaler_bytes (nếu có), feature_columns
    """
    prefix = f"models/{model_type}/"

    if version is None:
        keys = s3_hook.list_keys(bucket_name=bucket_name, prefix=prefix) or []
        versions = {key[len(prefix):].split('/')[0] for key in keys
                    if key.endswith('/model_card.json')}
        for candidate in sorted(versions, reverse=True):
            artifact = load_model_from_minio(s3_hook, bucket_name, model_type, candidate)
            if artifact:
                return artifact
        return None

    ver_prefix = f"{prefix}{version}/"

    result = {'version': version, 'model_type': model_type}

    # Load model_card
    card_key = f"{ver_prefix}model_card.json"
    try:
        card_json = s3_hook.read_key(card_key, bucket_name=bucket_name)
        result['model_card'] = ModelCard.from_json(card_json)
        if (result['model_card'].model_version != version or
                result['model_card'].model_type != model_type):
            return None
        logging.info(f"📋 Đã load model_card: {card_key}")
    except Exception:
        logging.warning(f"⚠️ Không tìm thấy model_card: {card_key}")
        return None

    # Load model artifact
    if model_type in ('xgboost', 'xgboost_return_5d'):
        model_key = f"{ver_prefix}model.json"
    elif model_type == 'lstm':
        model_key = f"{ver_prefix}model.pt"
    else:
        model_key = None

    if model_key:
        try:
            obj = s3_hook.get_key(model_key, bucket_name=bucket_name)
            result['model_bytes'] = obj.get()['Body'].read()
            logging.info(f"🧠 Đã load model: {model_key}")
        except Exception:
            logging.warning(f"⚠️ Không tìm thấy model artifact: {model_key}")

    # Load scaler (LSTM)
    if model_type == 'lstm':
        scaler_key = f"{ver_prefix}scaler.pkl"
        try:
            obj = s3_hook.get_key(scaler_key, bucket_name=bucket_name)
            result['scaler_bytes'] = obj.get()['Body'].read()
            logging.info(f"📐 Đã load scaler: {scaler_key}")
        except Exception:
            pass

    # Load feature columns
    fc_key = f"{ver_prefix}feature_columns.json"
    try:
        fc_json = s3_hook.read_key(fc_key, bucket_name=bucket_name)
        result['feature_columns'] = json.loads(fc_json)
    except Exception:
        result['feature_columns'] = []

    if model_key and (not result.get('model_bytes') or not result['feature_columns']):
        return None
    if model_type == 'lstm' and not result.get('scaler_bytes'):
        return None
    return result


def load_evaluation_model(s3_hook, bucket, kind, pipeline_run_id=None):
    if not pipeline_run_id:
        return load_model_from_minio(s3_hook, bucket, kind)
    prefix = f'models/{kind}/'
    keys = s3_hook.list_keys(bucket_name=bucket, prefix=prefix) or []
    versions = {key[len(prefix):].split('/')[0] for key in keys if key.endswith('/model_card.json')}
    for version in sorted(versions, reverse=True):
        artifacts = load_model_from_minio(s3_hook, bucket, kind, version)
        if artifacts and artifacts['model_card'].hyperparameters.get('pipeline_run_id') == pipeline_run_id:
            return artifacts
    return None


def list_model_versions(s3_hook, bucket_name: str, model_type: Optional[str] = None) -> pd.DataFrame:
    """
    Liệt kê tất cả phiên bản model trên MinIO.

    Returns:
        DataFrame với cột: model_type, version, trained_date, accuracy, auc_roc, dataset_size
    """
    frames = []
    try:
        frames.append(pd.read_csv(io.StringIO(s3_hook.read_key('models/registry.csv', bucket_name=bucket_name))))
    except Exception:
        pass  # Compatibility with registries created before per-version entries.
    keys = s3_hook.list_keys(bucket_name=bucket_name, prefix='models/registry/') or []
    for key in sorted(keys):
        if key.endswith('.json'):
            frames.append(pd.DataFrame([json.loads(s3_hook.read_key(key, bucket_name=bucket_name))]))
    if not frames:
        return pd.DataFrame(columns=['model_type', 'version', 'trained_date', 'accuracy', 'auc_roc', 'dataset_size'])
    df = pd.concat(frames, ignore_index=True).drop_duplicates(['model_type', 'version'], keep='last')
    if model_type:
        df = df[df['model_type'] == model_type]
    return df.sort_values(['trained_date', 'version']).reset_index(drop=True)


def _update_registry(s3_hook, bucket_name: str, model_card: ModelCard):
    """Write an independent entry so parallel model families cannot lose updates."""
    new_row = {
        'model_type': model_card.model_type,
        'version': model_card.model_version,
        'trained_date': model_card.trained_date,
        'accuracy': model_card.metrics.get('accuracy', model_card.metrics.get('sentiment_accuracy', None)),
        'auc_roc': model_card.metrics.get('auc_roc', model_card.metrics.get('sentiment_auc_roc', None)),
        'dataset_size': model_card.dataset_size,
        'num_features': model_card.num_features,
        'num_tickers': model_card.num_tickers,
        'notes': model_card.notes
    }

    s3_hook.load_string(
        string_data=json.dumps(new_row),
        key=f"models/registry/{model_card.model_type}/{model_card.model_version}.json",
        bucket_name=bucket_name,
        replace=False,
    )


# ==============================================================================
#  CHAMPION-CHALLENGER SELECTION & FALLBACK MECHANISM
# ==============================================================================

def select_champion_model(s3_hook,
                          bucket_name: str,
                          model_type: str,
                          current_version: str,
                          current_acc: float,
                          current_auc: float,
                          max_lookback_days: int = 7,
                          min_auc: float = 0.495,
                          min_acc: float = 0.50) -> Dict[str, Any]:
    """
    Cơ chế Model Validation Gate (Champion-Challenger):
    Kiểm định chất lượng model mới train hôm nay. Nếu không đạt chuẩn chất lượng
    (do thị trường biến động/nhiễu dữ liệu), tự động fallback về model tốt nhất
    đã được lưu trong vòng `max_lookback_days` (mặc định 7 ngày), hoặc kích hoạt Safe Mode.

    Args:
        s3_hook: Airflow S3Hook
        bucket_name: MinIO bucket
        model_type: "xgboost" | "lstm"
        current_version: Phiên bản vừa train hôm nay (VD: "v20260921")
        current_acc: Accuracy của model vừa train trên Test set
        current_auc: AUC-ROC của model vừa train trên Test set
        max_lookback_days: Giới hạn số ngày lùi về quá khứ (mặc định 7 ngày để tránh Concept Drift)
        min_auc: Ngưỡng AUC tối thiểu chấp nhận được (mặc định 0.495)
        min_acc: Ngưỡng Accuracy tối thiểu chấp nhận được (mặc định 0.50)

    Returns:
        Dict:
            'status': 'USE_CURRENT' | 'FALLBACK' | 'SAFE_MODE'
            'selected_version': str
            'selected_metrics': {'accuracy': float, 'auc_roc': float}
            'reason': str
            'artifacts': Optional[Dict]
    """
    # 1. Kiểm tra model mới hôm nay có đạt ngưỡng tối thiểu hay không (mặc định min_auc=0.495, min_acc=0.50)
    current_passed = (current_auc is not None and current_acc is not None and
                      np.isfinite(current_auc) and np.isfinite(current_acc) and
                      current_auc >= min_auc and current_acc >= min_acc)
    acc_text = f'{current_acc:.1%}' if current_acc is not None else 'unavailable'

    if current_passed:
        logging.info(f"✅ Model mới {current_version} ({model_type}) đạt chuẩn chất lượng "
                     f"(AUC={current_auc} >= {min_auc}, Acc={acc_text} >= {min_acc:.0%}). "
                     f"Chọn làm Champion!")
        return {
            'status': 'USE_CURRENT',
            'selected_version': current_version,
            'selected_metrics': {'accuracy': current_acc, 'auc_roc': current_auc},
            'reason': (f"Model mới ({current_version}: AUC={current_auc}, Acc={acc_text}) "
                       f"đạt chuẩn chất lượng (AUC>={min_auc}, Acc>={min_acc:.0%})."),
            'artifacts': None
        }

    # 2. Model hôm nay không đạt chuẩn -> Bắt đầu quét registry.csv tìm model cũ dự phòng
    logging.warning(f"⚠️ Model mới {current_version} ({model_type}) KHÔNG ĐẠT CHUẨN "
                    f"(AUC={current_auc} < {min_auc} hoặc Acc={acc_text} < {min_acc:.0%}). "
                    f"Bắt đầu quét registry.csv trong {max_lookback_days} ngày gần nhất...")

    try:
        current_dt = version_date(current_version)
    except Exception:
        current_dt = datetime.now()

    df_reg = list_model_versions(s3_hook, bucket_name, model_type)

    valid_candidates = []
    if not df_reg.empty:
        # Lọc đúng model_type và loại trừ version hiện tại
        df_type = df_reg[(df_reg['model_type'] == model_type) & (df_reg['version'] != current_version)].copy()

        for _, row in df_type.iterrows():
            ver = str(row.get('version', '')).strip()
            # Xác định ngày huấn luyện
            row_dt = None
            if pd.notna(row.get('trained_date')):
                try:
                    row_dt = datetime.strptime(str(row['trained_date']).strip(), "%Y-%m-%d")
                except Exception:
                    pass

            if row_dt is None and ver.startswith('v'):
                try:
                    row_dt = version_date(ver)
                except Exception:
                    pass

            if row_dt is None:
                continue

            # Kiểm tra khoảng thời gian (không quá max_lookback_days và không ở tương lai)
            days_diff = (current_dt - row_dt).days
            if 0 <= days_diff <= max_lookback_days:
                auc_val = pd.to_numeric(row.get('auc_roc'), errors='coerce')
                acc_val = pd.to_numeric(row.get('accuracy'), errors='coerce')

                # Kiểm tra ngưỡng chất lượng
                if pd.notna(auc_val) and pd.notna(acc_val) and np.isfinite(auc_val) and np.isfinite(acc_val):
                    if auc_val >= min_auc and acc_val >= min_acc:
                        valid_candidates.append({
                            'version': ver,
                            'trained_date': str(row.get('trained_date', '')),
                            'accuracy': float(acc_val),
                            'auc_roc': float(auc_val),
                            'days_ago': days_diff
                        })

    # 3. Đánh giá ứng viên Fallback
    if valid_candidates:
        # Sắp xếp ưu tiên AUC cao nhất, sau đó đến Accuracy
        valid_candidates.sort(key=lambda x: (x['auc_roc'], x['accuracy']), reverse=True)
        for best_candidate in valid_candidates:
            best_ver = best_candidate['version']
            artifacts = load_model_from_minio(s3_hook, bucket_name, model_type, version=best_ver)
            if not artifacts or not artifacts.get('model_bytes'):
                logging.warning('Incomplete fallback %s; checking the next candidate.', best_ver)
                continue
            return {
                'status': 'FALLBACK',
                'selected_version': best_ver,
                'selected_metrics': {
                    'accuracy': best_candidate['accuracy'],
                    'auc_roc': best_candidate['auc_roc']
                },
                'reason': (f"Model mới ({current_version}: AUC={current_auc}, Acc={acc_text}) không đạt chuẩn. "
                           f"Tự động fallback về phiên bản {best_ver} (AUC={best_candidate['auc_roc']:.3f}, "
                           f"Acc={best_candidate['accuracy']:.1%}, cách {best_candidate['days_ago']} ngày)."),
                'artifacts': artifacts
            }

    # 4. Kịch bản xấu nhất: Không có model nào trong max_lookback_days đạt chuẩn -> Safe Mode
    logging.warning(f"🚨 KÍCH HOẠT SAFE MODE: Model mới {current_version} và tất cả model trong {max_lookback_days} ngày qua "
                    f"đều không đạt chuẩn chất lượng (AUC < {min_auc} hoặc Acc < {min_acc:.0%}).")
    return {
        'status': 'SAFE_MODE',
        'selected_version': current_version,
        'selected_metrics': {'accuracy': current_acc, 'auc_roc': current_auc},
        'reason': (f"CẢNH BÁO THỊ TRƯỜNG BIẾN ĐỘNG MẠNH: Model mới ({current_version}: AUC={current_auc}, Acc={acc_text}) "
                   f"và toàn bộ model trong {max_lookback_days} ngày gần nhất đều không đạt ngưỡng tin cậy. "
                   f"Kích hoạt SAFE MODE (Đứng ngoài quan sát, không giao dịch để bảo vệ vốn)."),
        'artifacts': None
    }
