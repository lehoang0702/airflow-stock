"""
MODULE: MODEL EVALUATOR — ĐÁNH GIÁ MÔ HÌNH HỆ THỐNG
=====================================================
Đánh giá mô hình toàn diện với đầy đủ metrics khoa học:
- Accuracy, AUC-ROC, Precision, Recall, F1-Score, Confusion Matrix
- TimeSeriesSplit Cross-Validation (5-fold)
- Historical price-direction forecast evaluation
- So sánh phiên bản & xuất báo cáo

Schema MinIO:
    evaluation/{model_type}/eval_{YYYYMMDD}.csv
    evaluation/comparison/eval_comparison_{YYYYMMDD}.csv
"""

import io
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==============================================================================
#  ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI NHỊ PHÂN (XGBoost, LSTM)
# ==============================================================================

def evaluate_binary_classifier(y_true: np.ndarray, y_prob: np.ndarray,
                               model_name: str = "Model", dates=None,
                               decision_threshold: float = 0.5,
                               confidence_margin: float = 0.05) -> Dict[str, Any]:
    """
    Đánh giá đầy đủ mô hình phân loại nhị phân.

    Args:
        y_true: Nhãn thực tế (0/1)
        y_prob: Xác suất dự báo (0.0 → 1.0)
        model_name: Tên mô hình (để ghi log)

    Returns:
        Dict chứa toàn bộ metrics
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if len(y_true) == 0 or len(y_true) != len(y_prob):
        raise ValueError('Evaluation requires matched, nonempty observations')
    if not np.isfinite(y_prob).all() or not ((y_prob >= 0) & (y_prob <= 1)).all():
        raise ValueError('Invalid model probabilities')
    if not np.isfinite(decision_threshold) or not 0 < decision_threshold < 1:
        raise ValueError('Invalid decision threshold')
    y_pred = (y_prob >= decision_threshold).astype(int)

    # Metrics cơ bản
    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None

    # Per-class metrics
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Precision / Recall / F1 cho class 0 (Giảm)
    prec_0 = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_0 = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_0 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    high_conv_mask = (y_prob >= 0.5 + confidence_margin) | (y_prob <= 0.5 - confidence_margin)
    if np.any(high_conv_mask):
        high_conv_samples = int(np.sum(high_conv_mask))
        high_conv_acc = round(float(accuracy_score(y_true[high_conv_mask], (y_prob[high_conv_mask] >= decision_threshold).astype(int))), 4)
        high_conv_coverage = round(float(high_conv_samples / len(y_true)), 4)
    else:
        high_conv_samples = 0
        high_conv_acc = None
        high_conv_coverage = 0.0

    metrics = {
        'accuracy': round(acc, 4),
        'accuracy_at_0_5': round(float(accuracy_score(y_true, y_prob >= 0.5)), 4),
        'decision_threshold': float(decision_threshold),
        'confidence_margin': float(confidence_margin),
        'high_conviction_samples': high_conv_samples,
        'high_conviction_accuracy': high_conv_acc,
        'high_conviction_coverage': high_conv_coverage,
        'auc_roc': round(auc, 4) if auc is not None else None,
        'precision_class1': round(prec, 4),
        'recall_class1': round(rec, 4),
        'f1_score_class1': round(f1, 4),
        'precision_class0': round(prec_0, 4),
        'recall_class0': round(rec_0, 4),
        'f1_score_class0': round(f1_0, 4),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'total_samples': int(len(y_true)),
        'positive_ratio': round(float(y_true.mean()), 4),
        'predicted_positive_ratio': round(float(y_pred.mean()), 4)
    }

    logging.info(
        f"📊 {model_name} | Acc: {acc*100:.2f}% | AUC: {auc} | "
        f"Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}%"
    )

    return metrics


def generate_classification_report_text(y_true: np.ndarray, y_prob: np.ndarray,
                                         model_name: str = "Model", decision_threshold: float = 0.5) -> str:
    """
    Xuất Classification Report dạng text table (per-class metrics).

    Returns:
        String chứa bảng Classification Report chuẩn sklearn
    """
    y_pred = (y_prob >= decision_threshold).astype(int)

    report = classification_report(
        y_true, y_pred,
        labels=[0, 1], target_names=["Giam (0)", "Tang (1)"],
        digits=4,
        zero_division=0
    )

    header = f"\n{'='*55}\n📊 CLASSIFICATION REPORT: {model_name}\n{'='*55}\n"
    return header + report


# ==============================================================================
#  ĐÁNH GIÁ LỊCH SỬ DỰ BÁO HƯỚNG GIÁ
# ==============================================================================

def evaluate_direction_forecasts(predictions: pd.DataFrame,
                               price_data: pd.DataFrame,
                               prediction_horizon: int = 5) -> Dict[str, Any]:
    """Score issued, trained direction forecasts after their exact horizon."""
    from forecast_history import observed_forecasts
    observed, pending, invalid = observed_forecasts(predictions, price_data, prediction_horizon)
    result = {
        'status': 'unavailable', 'sentiment_accuracy': None,
        'sentiment_auc_roc': None, 'sentiment_correlation': None,
        'total_evaluated': 0, 'total_samples': 0,
        'pending_samples': pending, 'invalid_samples': invalid,
        'prediction_horizon': prediction_horizon,
        'evaluation_source': 'issued_trained_forecasts_v2',
        'evaluation_date': datetime.now().isoformat(),
        'message': 'Chưa có dự báo từ mô hình đã huấn luyện đủ phiên để kiểm chứng.'
    }
    if observed.empty or 'prob_num' not in observed:
        return result
    probabilities = pd.to_numeric(observed['prob_num'], errors='coerce')
    valid = probabilities.between(0, 1) & observed.get(
        'inference_source', pd.Series('', index=observed.index)).eq('trained_model_v2')
    observed = observed.loc[valid].copy()
    if observed.empty:
        return result
    probabilities = probabilities.loc[valid].to_numpy()
    metrics = evaluate_binary_classifier(observed['actual_up'].to_numpy(), probabilities, 'Price direction')
    changes = observed['actual_return'].to_numpy()
    correlation = (float(np.corrcoef(probabilities, changes)[0, 1])
                   if len(changes) > 1 and np.std(changes) > 0 and np.std(probabilities) > 0 else None)
    return {**result, **metrics, 'status': 'ok', 'message': None,
            'sentiment_accuracy': metrics['accuracy'],
            'sentiment_auc_roc': metrics['auc_roc'],
            'sentiment_correlation': correlation,
            'total_evaluated': len(observed),
            'test_start': str(observed['reference_date'].min()),
            'test_end': str(observed['outcome_date'].max().date())}


# ==============================================================================
#  CROSS-VALIDATION (TimeSeriesSplit)
# ==============================================================================

def run_time_series_cv(X: np.ndarray, y: np.ndarray,
                       model_builder_fn, n_splits: int = 5,
                       model_name: str = "Model", dates=None) -> Dict[str, Any]:
    """
    Chạy TimeSeriesSplit cross-validation để đánh giá tính ổn định của mô hình.

    Args:
        X: Feature matrix (numpy array)
        y: Target vector (numpy array)
        model_builder_fn: Callable trả về fitted model có .predict_proba()
        n_splits: Số fold (mặc định 5)
        model_name: Tên mô hình (để ghi log)

    Returns:
        Dict chứa: cv_accuracy_mean, cv_accuracy_std, cv_auc_mean, cv_auc_std, fold_details
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_accs = []
    fold_aucs = []
    fold_details = []

    from pipeline_data import date_folds
    folds = date_folds(dates, n_splits) if dates is not None else tscv.split(X)
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            model = model_builder_fn(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            fold_acc = float(accuracy_score(y_test, y_pred))
            fold_auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) == 2 else None

            fold_accs.append(fold_acc)
            if fold_auc is not None:
                fold_aucs.append(fold_auc)
            fold_details.append({
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'accuracy': round(fold_acc, 4),
                'auc_roc': round(fold_auc, 4) if fold_auc is not None else None
            })
        except Exception as e:
            logging.warning(f"⚠️ CV Fold {fold_idx+1} thất bại: {e}")
            fold_details.append({
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'accuracy': None,
                'auc_roc': None,
                'error': str(e)
            })

    if fold_accs:
        cv_result = {
            'cv_accuracy_mean': round(float(np.mean(fold_accs)), 4),
            'cv_accuracy_std': round(float(np.std(fold_accs)), 4),
            'cv_auc_mean': round(float(np.mean(fold_aucs)), 4) if fold_aucs else None,
            'cv_auc_std': round(float(np.std(fold_aucs)), 4) if fold_aucs else None,
            'n_splits': n_splits,
            'fold_details': fold_details
        }
        logging.info(
            f"🔄 {model_name} CV ({n_splits}-fold) | "
            f"Acc: {cv_result['cv_accuracy_mean']*100:.2f}% ± {cv_result['cv_accuracy_std']*100:.2f}% | "
            f"AUC: {cv_result['cv_auc_mean']} ± {cv_result['cv_auc_std']}"
        )
    else:
        cv_result = {
            'cv_accuracy_mean': None, 'cv_accuracy_std': None,
            'cv_auc_mean': None, 'cv_auc_std': None,
            'n_splits': n_splits, 'fold_details': fold_details
        }

    return cv_result


# ==============================================================================
#  SO SÁNH PHIÊN BẢN & TẠO BÁO CÁO
# ==============================================================================

def compare_model_versions(eval_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    So sánh metrics giữa nhiều mô hình / phiên bản.

    Args:
        eval_results: Dict {model_name: metrics_dict}
            VD: {"XGBoost v20260904": {...}, "LSTM v20260904": {...}, "FinBERT v20260904": {...}}

    Returns:
        DataFrame chứa bảng so sánh
    """
    rows = []
    for model_name, metrics in eval_results.items():
        row = {'model': model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sắp xếp theo AUC hoặc Accuracy giảm dần
    sort_col = 'auc_roc' if 'auc_roc' in df.columns else 'accuracy'
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)

    return df


def generate_evaluation_report(eval_results: Dict[str, Dict],
                                cv_results: Optional[Dict[str, Dict]] = None,
                                today_str: Optional[str] = None) -> Tuple[str, str]:
    """
    Tạo báo cáo đánh giá dạng text (cho Telegram) và CSV (cho lưu trữ).

    Args:
        eval_results: Dict {model_name: metrics_dict}
        cv_results: Dict {model_name: cv_metrics_dict} (optional)
        today_str: Ngày đánh giá

    Returns:
        Tuple (telegram_message, csv_string)
    """
    if today_str is None:
        today_str = datetime.now().strftime("%Y-%m-%d")

    msg_lines = [
        "📊 <b>BÁO CÁO ĐÁNH GIÁ MÔ HÌNH DỰ BÁO CỔ PHIẾU</b>",
        f"📅 <b>Ngày đánh giá:</b> {today_str}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ""
    ]

    def number(value, percent=False):
        if value is None or not np.isfinite(value):
            return 'Chưa có kết quả'
        return f'{value * 100:.2f}%' if percent else f'{value:.4f}'

    for model_name, metrics in eval_results.items():
        msg_lines.append(f'📊 <b>{model_name}</b>')
        for title, key, percent in [('Accuracy', 'accuracy', True), ('AUC-ROC', 'auc_roc', False),
                                    ('Precision', 'precision_class1', True), ('Recall', 'recall_class1', True),
                                    ('F1', 'f1_score_class1', False)]:
            value = metrics.get(key)
            msg_lines.append(f'   {title}: <b>{number(value, percent)}</b>')
        msg_lines.append(f"   Tổng mẫu đánh giá: {metrics.get('total_samples', metrics.get('total_evaluated', 0))}")
        msg_lines.append(f"   Ngày đánh giá: {metrics.get('evaluation_date', today_str)}")
        if metrics.get('message'):
            msg_lines.append(metrics['message'])
        if cv_results and model_name in cv_results:
            cv = cv_results[model_name]
            msg_lines.append(f"   CV Accuracy: {number(cv.get('cv_accuracy_mean'), True)}")
            msg_lines.append(f"   CV AUC: {number(cv.get('cv_auc_mean'))}")
        msg_lines.append('')

    # Tạo CSV
    df_compare = compare_model_versions(eval_results)
    csv_buf = io.StringIO()
    df_compare.to_csv(csv_buf, index=False, encoding='utf-8-sig')

    return "\n".join(msg_lines), csv_buf.getvalue()


# ==============================================================================
#  LƯU BÁO CÁO LÊN MINIO
# ==============================================================================

def save_evaluation_to_minio(s3_hook, bucket_name: str, model_type: str,
                              metrics: Dict, date_nodash: Optional[str] = None):
    """
    Lưu báo cáo đánh giá lên MinIO.

    Schema: evaluation/{model_type}/eval_{YYYYMMDD}.json
    """
    if date_nodash is None:
        date_nodash = datetime.now().strftime("%Y%m%d")

    eval_key = f"evaluation/{model_type}/eval_{date_nodash}.json"
    eval_json = json.dumps(metrics, ensure_ascii=False, indent=2, default=str)

    s3_hook.load_string(
        string_data=eval_json,
        key=eval_key,
        bucket_name=bucket_name,
        replace=True
    )
    logging.info(f"📊 Đã lưu báo cáo đánh giá lên MinIO: {eval_key}")


def save_comparison_to_minio(s3_hook, bucket_name: str,
                              csv_string: str, date_nodash: Optional[str] = None):
    """Lưu bảng so sánh tổng hợp lên MinIO."""
    if date_nodash is None:
        date_nodash = datetime.now().strftime("%Y%m%d")

    cmp_key = f"evaluation/comparison/eval_comparison_{date_nodash}.csv"
    s3_hook.load_string(
        string_data=csv_string,
        key=cmp_key,
        bucket_name=bucket_name,
        replace=True
    )
    logging.info(f"📊 Đã lưu bảng so sánh đánh giá lên MinIO: {cmp_key}")
