"""
MODULE: MODEL REGISTRY — XÂY DỰNG & QUẢN LÝ MÔ HÌNH
=====================================================
Đóng gói toàn bộ logic xây dựng, serialize, lưu trữ và quản lý phiên bản mô hình.

Schema MinIO:
    models/xgboost/v{YYYYMMDD}/model.json + model_card.json + feature_columns.json
    models/lstm/v{YYYYMMDD}/model.pt + scaler.pkl + model_card.json + feature_columns.json
    models/finbert/v{YYYYMMDD}/pipeline_config.json + model_card.json + sentiment_stats.json
    models/registry.csv
"""

import io
import json
import logging
import pickle
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

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Tiền xử lý dữ liệu: chuẩn hóa cột, tạo target, lọc features."""
        cols_lower = {c.lower(): c for c in df.columns}
        date_col = cols_lower.get('date') or cols_lower.get('datetime') or df.columns[0]
        ticker_col = cols_lower.get('symbol') or cols_lower.get('ticker') or cols_lower.get('ma_co_phieu')
        sector_col = cols_lower.get('sector') or cols_lower.get('nhom_nganh')
        close_col = cols_lower.get('close') or cols_lower.get('adj close')

        df['Date_Std'] = pd.to_datetime(df[date_col], errors='coerce').dt.tz_localize(None)
        df['Ticker_Std'] = df[ticker_col] if ticker_col else 'UNKNOWN'
        df['Sector_Std'] = df[sector_col] if sector_col else 'General'

        # Target: giá phiên sau > giá phiên hiện tại
        if close_col:
            df['Target_Std'] = (df.groupby('Ticker_Std')[close_col].shift(-1) > df[close_col]).astype(int)
        else:
            df['Target_Std'] = (df.index % 2 == 0).astype(int)

        # Lọc features (loại bỏ data leakage)
        leakage_keywords = [
            'target', 'label', 'unnamed', 'index', 'level', 'future',
            'date', 'symbol', 'ticker', 'sector', 'close', 'open',
            'high', 'low', 'adj close'
        ]
        feature_cols = [
            c for c in df.columns
            if not any(k in c.lower() for k in leakage_keywords)
            and c not in ['Date_Std', 'Ticker_Std', 'Sector_Std', 'Target_Std']
        ]

        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        feature_cols = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number)]
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0).astype(np.float32)

        self.feature_cols = feature_cols
        return df, feature_cols

    def train(self, df: pd.DataFrame, train_cutoff: str = '2024-01-01') -> Dict[str, float]:
        """Train mô hình XGBoost và trả về metrics thực sự (KHÔNG clip)."""
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, roc_auc_score

        df, feature_cols = self.preprocess(df)

        # Split train/test theo thời gian
        train_mask = df['Date_Std'] < pd.Timestamp(train_cutoff)
        if train_mask.sum() < 200:
            split_idx = int(len(df) * 0.8)
            train_mask = df.index < split_idx
            test_mask = df.index >= split_idx
        else:
            test_mask = ~train_mask

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
            test_auc = float(roc_auc_score(y_test, test_probs))
        except Exception:
            test_auc = 0.5

        # Tạo model card
        date_nodash = datetime.now().strftime("%Y%m%d")
        train_dates = df.loc[train_mask, 'Date_Std']
        test_dates = df.loc[test_mask, 'Date_Std']

        self.model_card = ModelCard(
            model_name="XGBoost Gradient Boosting Classifier",
            model_version=f"v{date_nodash}",
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
        logging.info(f"✅ XGBoost đã train xong | Acc: {test_acc*100:.2f}% | AUC: {test_auc:.4f}")
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
        X = df[self.feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0).astype(np.float32)
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
        'temperature': 2.5
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
                                   batch_first=True, dropout=dropout)
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

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Tiền xử lý dữ liệu cho LSTM."""
        cols_lower = {c.lower(): c for c in df.columns}
        date_col = cols_lower.get('date') or cols_lower.get('datetime') or df.columns[0]
        ticker_col = cols_lower.get('ticker') or cols_lower.get('symbol') or cols_lower.get('ma_co_phieu')
        sector_col = cols_lower.get('sector') or cols_lower.get('nhom_nganh')
        close_col = cols_lower.get('close') or cols_lower.get('adj close')

        df['Date_Std'] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
        df['Ticker_Std'] = df[ticker_col] if ticker_col else 'UNKNOWN'
        df['Sector_Std'] = df[sector_col] if sector_col else 'General'

        if close_col:
            df['Target_Std'] = (df.groupby('Ticker_Std')[close_col].shift(-1) > df[close_col]).astype(float)
        else:
            df['Target_Std'] = (df.index % 2 == 0).astype(float)

        # Lọc features — loại bỏ giá thô để tránh bullish bias
        raw_price_keywords = ['close', 'open', 'high', 'low', 'adj close', 'volume',
                              'date', 'ticker', 'symbol', 'sector', 'target']
        feature_cols = []
        for c in df.columns:
            c_low = c.lower()
            if (not any(k == c_low for k in raw_price_keywords)
                    and 'target' not in c_low
                    and c not in ['Date_Std', 'Ticker_Std', 'Sector_Std', 'Target_Std']):
                df[c] = pd.to_numeric(df[c], errors='coerce')
                if np.issubdtype(df[c].dtype, np.number):
                    feature_cols.append(c)

        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0).astype(np.float32)
        df = df.sort_values(['Date_Std', 'Ticker_Std']).reset_index(drop=True)

        self.feature_cols = feature_cols
        return df, feature_cols

    def _create_sequences(self, data_group, feature_cols, lookback):
        """Tạo sliding windows cho LSTM."""
        X, y = [], []
        for _, grp in data_group.groupby('Ticker_Std'):
            feats = grp[feature_cols].values
            targets = grp['Target_Std'].values
            n = len(feats)
            if n <= lookback:
                continue
            for i in range(lookback, n):
                X.append(feats[i - lookback:i])
                y.append(targets[i])
        if not X:
            return np.empty((0, lookback, len(feature_cols))), np.empty((0,))
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

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

        train_df = df[df['Date_Std'] < train_cutoff].copy()
        test_df = df[df['Date_Std'] >= train_cutoff].copy()

        # Chuẩn hóa features
        self.scaler = StandardScaler()
        train_df[feature_cols] = self.scaler.fit_transform(train_df[feature_cols].values)
        test_df[feature_cols] = self.scaler.transform(test_df[feature_cols].values)

        # Tạo sequences
        X_train, y_train = self._create_sequences(train_df, feature_cols, lookback)
        X_test, y_test = self._create_sequences(test_df, feature_cols, lookback)

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
                temp = hp['temperature']
                test_probs = 1.0 / (1.0 + np.exp(-test_logits / temp))
                test_preds = (test_probs >= 0.5).astype(int)
                test_acc = float(accuracy_score(y_test, test_preds))
                try:
                    test_auc = float(roc_auc_score(y_test, test_probs))
                except Exception:
                    test_auc = 0.5
            else:
                test_acc, test_auc = 0.5, 0.5

        # Tạo model card
        date_nodash = datetime.now().strftime("%Y%m%d")
        train_dates = train_df['Date_Std']
        test_dates = test_df['Date_Std']

        self.model_card = ModelCard(
            model_name="Calibrated Stacked LSTM (PyTorch)",
            model_version=f"v{date_nodash}",
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
        logging.info(f"✅ LSTM đã train xong | Acc: {test_acc*100:.2f}% | AUC: {test_auc:.4f}")
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
    """
    Đóng gói pipeline FinBERT Sentiment Analysis:
    Load pre-trained FinBERT → Chạy sentiment inference → Lưu pipeline config + thống kê
    """

    DEFAULT_CONFIG = {
        'model_name': 'ProsusAI/finbert',
        'max_length': 128,
        'prob_formula': '(avg_score + 1.0) / 2.0',
        'prob_clip_min': 0.05,
        'prob_clip_max': 0.95,
        'buy_threshold': 0.55,
        'sell_threshold': 0.45
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self.DEFAULT_CONFIG.copy()
        self.model_card = None
        self.sentiment_stats = {}

    def run_inference(self, df_news: pd.DataFrame, today_str: str) -> Tuple[List[Dict], Dict]:
        """
        Chạy FinBERT inference trên tập tin tức.
        Trả về: (predictions_list, sentiment_stats_dict)
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        model = AutoModelForSequenceClassification.from_pretrained(self.config['model_name'])
        model.eval()

        results = []
        sentiment_stats = {}

        for (ticker, sector), group in df_news.groupby(['ma_co_phieu', 'nhom_nganh']):
            scores = []
            pos_count, neg_count, neu_count = 0, 0, 0

            for title in group['tieu_de'].tolist():
                inputs = tokenizer(str(title), return_tensors="pt",
                                   truncation=True, max_length=self.config['max_length'])
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)[0].numpy()

                score = float(probs[0] - probs[1])
                scores.append(score)

                # Đếm sentiment phân bố
                if probs[0] > probs[1] and probs[0] > probs[2]:
                    pos_count += 1
                elif probs[1] > probs[0] and probs[1] > probs[2]:
                    neg_count += 1
                else:
                    neu_count += 1

            if scores:
                avg_score = float(np.mean(scores))
                prob_up = float(np.clip(
                    (avg_score + 1.0) / 2.0,
                    self.config['prob_clip_min'],
                    self.config['prob_clip_max']
                ))
            else:
                avg_score = 0.0
                prob_up = 0.50

            # Phân loại khuyến nghị (Confidence Threshold Filtering)
            if prob_up >= self.config['buy_threshold']:
                rec = "MUA"
                trend = "Tăng mạnh" if prob_up >= 0.60 else "Tăng tích lũy"
                conf = "Rất cao" if prob_up >= 0.60 else "Cao"
            elif prob_up <= self.config['sell_threshold']:
                rec = "BÁN"
                trend = "Giảm mạnh" if prob_up <= 0.40 else "Giảm phân phối"
                conf = "Rất cao" if prob_up <= 0.40 else "Cao"
            else:
                rec = "ĐỨNG NGOÀI"
                trend = "Đi ngang / Lưỡng lự (Sideway)"
                conf = "Trung bình"

            results.append({
                'ngay_du_bao': today_str,
                'ma_co_phieu': ticker,
                'nhom_nganh': sector,
                'khuyen_nghi': rec,
                'xac_suat_tang_gia': f"{prob_up * 100:.2f}%",
                'xu_huong_du_kien': trend,
                'do_tin_cay': conf,
                'prob_num': prob_up,
                'avg_sentiment_score': avg_score
            })

            sentiment_stats[ticker] = {
                'positive': pos_count,
                'negative': neg_count,
                'neutral': neu_count,
                'total_news': len(scores),
                'avg_score': round(avg_score, 4),
                'prob_up': round(prob_up, 4)
            }

        self.sentiment_stats = sentiment_stats

        # Tạo model card
        date_nodash = datetime.now().strftime("%Y%m%d")
        total_news = sum(s['total_news'] for s in sentiment_stats.values())
        total_pos = sum(s['positive'] for s in sentiment_stats.values())
        total_neg = sum(s['negative'] for s in sentiment_stats.values())

        self.model_card = ModelCard(
            model_name="FinBERT Sentiment Analysis (ProsusAI/finbert)",
            model_version=f"v{date_nodash}",
            trained_date=datetime.now().strftime("%Y-%m-%d"),
            model_type="finbert",
            dataset_size=total_news,
            num_features=0,
            num_tickers=len(sentiment_stats),
            hyperparameters=self.config,
            metrics={},  # Sẽ được cập nhật bởi model_evaluator.py
            notes=f"Pre-trained model. {total_news} tin tức phân tích: {total_pos} positive, {total_neg} negative"
        )

        logging.info(f"✅ FinBERT đã phân tích {total_news} tin tức cho {len(sentiment_stats)} mã")
        return results, sentiment_stats


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

    # 1. Lưu model_card.json
    s3_hook.load_string(
        string_data=model_card.to_json(),
        key=f"{prefix}model_card.json",
        bucket_name=bucket_name,
        replace=True
    )
    logging.info(f"📋 Đã lưu model_card.json → {prefix}model_card.json")

    # 2. Lưu model artifact
    if model_bytes is not None:
        if model_type == 'xgboost':
            model_key = f"{prefix}model.json"
        elif model_type == 'lstm':
            model_key = f"{prefix}model.pt"
        else:
            model_key = f"{prefix}model.bin"

        s3_hook.load_bytes(
            bytes_data=model_bytes,
            key=model_key,
            bucket_name=bucket_name,
            replace=True
        )
        logging.info(f"🧠 Đã lưu model artifact → {model_key} ({len(model_bytes):,} bytes)")

    # 3. Lưu scaler (LSTM)
    if scaler_bytes is not None:
        scaler_key = f"{prefix}scaler.pkl"
        s3_hook.load_bytes(
            bytes_data=scaler_bytes,
            key=scaler_key,
            bucket_name=bucket_name,
            replace=True
        )
        logging.info(f"📐 Đã lưu scaler → {scaler_key}")

    # 4. Lưu feature columns
    if model_card.feature_columns:
        s3_hook.load_string(
            string_data=json.dumps(model_card.feature_columns, ensure_ascii=False),
            key=f"{prefix}feature_columns.json",
            bucket_name=bucket_name,
            replace=True
        )

    # 5. Lưu extra artifacts (VD: pipeline_config, sentiment_stats cho FinBERT)
    if extra_artifacts:
        for filename, content in extra_artifacts.items():
            s3_hook.load_string(
                string_data=content,
                key=f"{prefix}{filename}",
                bucket_name=bucket_name,
                replace=True
            )
            logging.info(f"📎 Đã lưu {filename} → {prefix}{filename}")

    # 6. Cập nhật registry.csv
    _update_registry(s3_hook, bucket_name, model_card)

    logging.info(f"✅ Đã lưu phiên bản {model_type}/{version} lên MinIO thành công!")


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
        # Tìm phiên bản mới nhất
        keys = s3_hook.list_keys(bucket_name=bucket_name, prefix=prefix)
        if not keys:
            logging.warning(f"⚠️ Không tìm thấy model {model_type} nào trên MinIO")
            return None

        versions = set()
        for k in keys:
            parts = k.replace(prefix, '').split('/')
            if parts[0].startswith('v'):
                versions.add(parts[0])

        if not versions:
            return None
        version = sorted(versions)[-1]

    ver_prefix = f"{prefix}{version}/"

    result = {'version': version, 'model_type': model_type}

    # Load model_card
    card_key = f"{ver_prefix}model_card.json"
    try:
        card_json = s3_hook.read_key(card_key, bucket_name=bucket_name)
        result['model_card'] = ModelCard.from_json(card_json)
        logging.info(f"📋 Đã load model_card: {card_key}")
    except Exception:
        logging.warning(f"⚠️ Không tìm thấy model_card: {card_key}")
        return None

    # Load model artifact
    if model_type == 'xgboost':
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

    return result


def list_model_versions(s3_hook, bucket_name: str, model_type: Optional[str] = None) -> pd.DataFrame:
    """
    Liệt kê tất cả phiên bản model trên MinIO.

    Returns:
        DataFrame với cột: model_type, version, trained_date, accuracy, auc_roc, dataset_size
    """
    registry_key = "models/registry.csv"
    try:
        raw = s3_hook.read_key(registry_key, bucket_name=bucket_name)
        df = pd.read_csv(io.StringIO(raw))
        if model_type:
            df = df[df['model_type'] == model_type]
        return df
    except Exception:
        return pd.DataFrame(columns=['model_type', 'version', 'trained_date',
                                     'accuracy', 'auc_roc', 'dataset_size'])


def _update_registry(s3_hook, bucket_name: str, model_card: ModelCard):
    """Cập nhật file registry.csv tổng hợp trên MinIO."""
    registry_key = "models/registry.csv"

    # Đọc registry hiện tại (nếu có)
    try:
        raw = s3_hook.read_key(registry_key, bucket_name=bucket_name)
        df = pd.read_csv(io.StringIO(raw))
    except Exception:
        df = pd.DataFrame(columns=['model_type', 'version', 'trained_date',
                                   'accuracy', 'auc_roc', 'dataset_size',
                                   'num_features', 'num_tickers', 'notes'])

    # Thêm hoặc cập nhật dòng mới
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

    # Xóa dòng cũ nếu cùng model_type + version
    if not df.empty:
        mask = ~((df['model_type'] == model_card.model_type) & (df['version'] == model_card.model_version))
        df = df[mask]
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Ghi lại
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding='utf-8-sig')
    s3_hook.load_string(
        string_data=buf.getvalue(),
        key=registry_key,
        bucket_name=bucket_name,
        replace=True
    )
    logging.info(f"📝 Đã cập nhật registry.csv ({len(df)} phiên bản)")
