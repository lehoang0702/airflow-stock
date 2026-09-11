"""
MODULE: HYPERPARAMETER TUNER — TỐI ƯU HÓA SIÊU THAM SỐ TỰ ĐỘNG (AutoML)
========================================================================
Sử dụng thuật toán tối ưu hóa Bayes (Bayesian Optimization via Optuna)
kết hợp kiểm định chuỗi thời gian TimeSeriesSplit để tự động dò tìm
bộ tham số tối ưu cho mô hình GBDT (XGBoost) mà không gây rò rỉ dữ liệu tương lai.
"""

import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd

# Mặc định bộ tham số chuẩn chất lượng cao khi không chạy tuning
DEFAULT_XGB_PARAMS = {
    'n_estimators': 120,
    'max_depth': 3,
    'learning_rate': 0.03,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss'
}


def tune_xgboost_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 12) -> dict:
    """
    Dò tìm siêu tham số tối ưu bằng Optuna Bayesian Optimization.
    Sử dụng TimeSeriesSplit(n_splits=3) để tối đa hóa ROC-AUC trung bình.
    """
    try:
        import optuna
        import xgboost as xgb
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        logging.warning(f"⚠️ Optuna hoặc thư viện chưa sẵn sàng ({e}). Sử dụng bộ siêu tham số chuẩn định sẵn.")
        return DEFAULT_XGB_PARAMS.copy()

    # Tắt log rác của Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Đảm bảo đủ dữ liệu để split
    if len(X_train) < 300:
        logging.info("ℹ️ Kích thước tập huấn luyện nhỏ (<300), sử dụng cấu hình mặc định an toàn.")
        return DEFAULT_XGB_PARAMS.copy()

    logging.info(f"🧠 Bắt đầu Bayesian Optimization ({n_trials} trials) qua TimeSeriesSplit...")

    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 80, 160, step=20),
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.08, log=True),
            'subsample': trial.suggest_float('subsample', 0.65, 0.85),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.85),
            'min_child_weight': trial.suggest_int('min_child_weight', 2, 6),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }

        auc_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Kiểm tra phân phối nhãn
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                continue

            clf = xgb.XGBClassifier(**params)
            clf.fit(X_tr, y_tr)
            preds = clf.predict_proba(X_val)[:, 1]

            try:
                score = roc_auc_score(y_val, preds)
                auc_scores.append(score)
            except Exception:
                pass

        return float(np.mean(auc_scores)) if auc_scores else 0.5

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=60)

    best_params = DEFAULT_XGB_PARAMS.copy()
    best_params.update(study.best_params)

    logging.info(f"🎯 OPTUNA HOÀN TẤT: Best Trial ROC-AUC = {study.best_value:.4f}")
    logging.info(f"🔧 Siêu tham số tối ưu: max_depth={best_params['max_depth']}, lr={best_params['learning_rate']:.4f}, n_est={best_params['n_estimators']}")

    return best_params


def save_tuned_hyperparameters_to_minio(s3_hook, bucket_name: str, best_params: dict):
    """Lưu siêu tham số tối ưu vào MinIO để phục vụ Versioning & Audit MLOps."""
    try:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "algorithm": "XGBoost",
            "tuning_engine": "Optuna Bayesian Optimization (TPE Sampler)",
            "cross_validation": "TimeSeriesSplit (n_splits=3)",
            "best_parameters": best_params
        }
        json_str = json.dumps(payload, indent=2)
        s3_hook.load_string(
            string_data=json_str,
            key="models/xgboost/best_hyperparams.json",
            bucket_name=bucket_name,
            replace=True
        )
        logging.info("💾 Đã lưu cấu hình siêu tham số tối ưu lên MinIO: models/xgboost/best_hyperparams.json")
    except Exception as e:
        logging.warning(f"⚠️ Không thể lưu best_hyperparams lên MinIO: {e}")
