"""
MODULE: MODEL EVALUATOR — ĐÁNH GIÁ MÔ HÌNH HỆ THỐNG
=====================================================
Đánh giá mô hình toàn diện với đầy đủ metrics khoa học:
- Accuracy, AUC-ROC, Precision, Recall, F1-Score, Confusion Matrix
- TimeSeriesSplit Cross-Validation (5-fold)
- FinBERT Sentiment Accuracy (đối chiếu vs giá thực tế)
- So sánh phiên bản & xuất báo cáo

Schema MinIO:
    evaluation/{model_type}/eval_{YYYYMMDD}.csv
    evaluation/comparison/eval_comparison_{YYYYMMDD}.csv
"""

import io
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
                               model_name: str = "Model") -> Dict[str, Any]:
    """
    Đánh giá đầy đủ mô hình phân loại nhị phân.

    Args:
        y_true: Nhãn thực tế (0/1)
        y_prob: Xác suất dự báo (0.0 → 1.0)
        model_name: Tên mô hình (để ghi log)

    Returns:
        Dict chứa toàn bộ metrics
    """
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics cơ bản
    acc = float(accuracy_score(y_true, y_pred))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    # Per-class metrics
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Precision / Recall / F1 cho class 0 (Giảm)
    prec_0 = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_0 = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_0 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        'accuracy': round(acc, 4),
        'auc_roc': round(auc, 4),
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
        f"📊 {model_name} | Acc: {acc*100:.2f}% | AUC: {auc:.4f} | "
        f"Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}%"
    )

    return metrics


def generate_classification_report_text(y_true: np.ndarray, y_prob: np.ndarray,
                                        model_name: str = "Model") -> str:
    """
    Xuất Classification Report dạng text table (per-class metrics).

    Returns:
        String chứa bảng Classification Report chuẩn sklearn
    """
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(
        y_true, y_pred,
        target_names=["Giam (0)", "Tang (1)"],
        digits=4,
        zero_division=0
    )

    header = f"\n{'='*55}\n📊 CLASSIFICATION REPORT: {model_name}\n{'='*55}\n"
    return header + report


# ==============================================================================
#  ĐÁNH GIÁ FINBERT SENTIMENT (ĐỐI CHIẾU VS GIÁ THỰC TẾ)
# ==============================================================================

def evaluate_finbert_sentiment(predictions: pd.DataFrame,
                               price_data: pd.DataFrame,
                               prediction_horizon: int = 5) -> Dict[str, Any]:
    """
    Đánh giá FinBERT bằng cách đối chiếu sentiment dự báo vs biến động giá thực tế.

    Logic:
    - Lấy kết quả FinBERT dự báo tuần trước (hoặc phiên gần nhất có đủ dữ liệu T+5)
    - So sánh: nếu FinBERT dự báo MUA → giá có thực sự tăng sau 5 phiên không?
    - Tính: Sentiment Accuracy, Correlation, Confusion Matrix

    Args:
        predictions: DataFrame chứa cột ['ma_co_phieu', 'khuyen_nghi', 'prob_num']
        price_data: DataFrame chứa cột ['Date', 'Ticker', 'Close']
        prediction_horizon: Số phiên để kiểm chứng (mặc định 5)

    Returns:
        Dict chứa metrics đánh giá FinBERT
    """
    if predictions.empty or price_data.empty:
        logging.warning("⚠️ Không đủ dữ liệu để đánh giá FinBERT sentiment")
        return {'sentiment_accuracy': 0.0, 'sentiment_correlation': 0.0, 'total_evaluated': 0}

    price_data = price_data.copy()
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    price_data = price_data.sort_values(['Ticker', 'Date'])

    y_true_list = []
    y_pred_list = []
    prob_list = []

    for _, row in predictions.iterrows():
        ticker = row['ma_co_phieu']
        prob = float(row.get('prob_num', 0.5))

        ticker_prices = price_data[price_data['Ticker'] == ticker].copy()
        if len(ticker_prices) < prediction_horizon + 1:
            continue

        # Giá phiên cuối cùng tính từ cuối dataset
        last_idx = len(ticker_prices) - prediction_horizon - 1
        if last_idx < 0:
            continue

        price_at_prediction = float(ticker_prices.iloc[last_idx]['Close'])
        price_after_horizon = float(ticker_prices.iloc[-1]['Close'])

        # Ground truth: giá có tăng sau T+5 không?
        actual_up = 1 if price_after_horizon > price_at_prediction else 0

        # Prediction: FinBERT dự báo tăng (prob >= 0.5) hay không
        pred_up = 1 if prob >= 0.5 else 0

        y_true_list.append(actual_up)
        y_pred_list.append(pred_up)
        prob_list.append(prob)

    if not y_true_list:
        logging.warning("⚠️ Không tìm thấy đủ dữ liệu giá để kiểm chứng FinBERT")
        return {'sentiment_accuracy': 0.0, 'sentiment_correlation': 0.0, 'total_evaluated': 0}

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_prob = np.array(prob_list)

    # Tính metrics
    sent_acc = float(accuracy_score(y_true, y_pred))

    # Tương quan giữa sentiment score và biến động giá
    try:
        price_changes = []
        for _, row in predictions.iterrows():
            ticker = row['ma_co_phieu']
            ticker_prices = price_data[price_data['Ticker'] == ticker]
            if len(ticker_prices) >= prediction_horizon + 1:
                last_idx = len(ticker_prices) - prediction_horizon - 1
                p0 = float(ticker_prices.iloc[last_idx]['Close'])
                p1 = float(ticker_prices.iloc[-1]['Close'])
                price_changes.append((p1 - p0) / p0)
            else:
                price_changes.append(0.0)
        correlation = float(np.corrcoef(prob_list[:len(price_changes)], price_changes[:len(prob_list)])[0, 1])
        if np.isnan(correlation):
            correlation = 0.0
    except Exception:
        correlation = 0.0

    try:
        sent_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        sent_auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    metrics = {
        'sentiment_accuracy': round(sent_acc, 4),
        'sentiment_auc_roc': round(sent_auc, 4),
        'sentiment_correlation': round(correlation, 4),
        'precision_class1': round(prec, 4),
        'recall_class1': round(rec, 4),
        'f1_score_class1': round(f1, 4),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'total_evaluated': int(len(y_true))
    }

    logging.info(
        f"📰 FinBERT Sentiment | Acc: {sent_acc*100:.2f}% | AUC: {sent_auc:.4f} | "
        f"Correlation: {correlation:.4f} | Evaluated: {len(y_true)} mã"
    )

    return metrics


# ==============================================================================
#  CROSS-VALIDATION (TimeSeriesSplit)
# ==============================================================================

def run_time_series_cv(X: np.ndarray, y: np.ndarray,
                       model_builder_fn, n_splits: int = 5,
                       model_name: str = "Model") -> Dict[str, Any]:
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

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            model = model_builder_fn(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            fold_acc = float(accuracy_score(y_test, y_pred))
            try:
                fold_auc = float(roc_auc_score(y_test, y_prob))
            except Exception:
                fold_auc = 0.5

            fold_accs.append(fold_acc)
            fold_aucs.append(fold_auc)
            fold_details.append({
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'accuracy': round(fold_acc, 4),
                'auc_roc': round(fold_auc, 4)
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
            'cv_auc_mean': round(float(np.mean(fold_aucs)), 4),
            'cv_auc_std': round(float(np.std(fold_aucs)), 4),
            'n_splits': n_splits,
            'fold_details': fold_details
        }
        logging.info(
            f"🔄 {model_name} CV ({n_splits}-fold) | "
            f"Acc: {cv_result['cv_accuracy_mean']*100:.2f}% ± {cv_result['cv_accuracy_std']*100:.2f}% | "
            f"AUC: {cv_result['cv_auc_mean']:.4f} ± {cv_result['cv_auc_std']:.4f}"
        )
    else:
        cv_result = {
            'cv_accuracy_mean': 0.0, 'cv_accuracy_std': 0.0,
            'cv_auc_mean': 0.0, 'cv_auc_std': 0.0,
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

    for model_name, metrics in eval_results.items():
        # Xác định icon theo loại model
        if 'xgboost' in model_name.lower():
            icon = "🌲"
        elif 'lstm' in model_name.lower():
            icon = "🧠"
        elif 'finbert' in model_name.lower():
            icon = "📰"
        else:
            icon = "📊"

        msg_lines.append(f"{icon} <b>{model_name}</b>")

        acc = metrics.get('accuracy', metrics.get('sentiment_accuracy', 0))
        auc = metrics.get('auc_roc', metrics.get('sentiment_auc_roc', 0))
        prec = metrics.get('precision_class1', 0)
        rec = metrics.get('recall_class1', 0)
        f1 = metrics.get('f1_score_class1', 0)
        total = metrics.get('total_samples', metrics.get('total_evaluated', 0))

        msg_lines.append(f"   ├ Accuracy:  <b>{acc*100:.2f}%</b>")
        msg_lines.append(f"   ├ AUC-ROC:   <b>{auc:.4f}</b>")
        msg_lines.append(f"   ├ Precision:  <b>{prec*100:.2f}%</b>")
        msg_lines.append(f"   ├ Recall:     <b>{rec*100:.2f}%</b>")
        msg_lines.append(f"   ├ F1-Score:   <b>{f1*100:.2f}%</b>")

        # Confusion Matrix
        tp = metrics.get('true_positives', 0)
        fp = metrics.get('false_positives', 0)
        tn = metrics.get('true_negatives', 0)
        fn = metrics.get('false_negatives', 0)
        msg_lines.append(f"   ├ CM: TP={tp} FP={fp} TN={tn} FN={fn}")

        # Correlation (FinBERT)
        corr = metrics.get('sentiment_correlation')
        if corr is not None:
            msg_lines.append(f"   ├ Sentiment-Price Corr: <b>{corr:.4f}</b>")

        msg_lines.append(f"   └ Tổng mẫu đánh giá: <b>{total}</b>")

        # Cross-Validation
        if cv_results and model_name in cv_results:
            cv = cv_results[model_name]
            cv_acc = cv.get('cv_accuracy_mean', 0)
            cv_acc_std = cv.get('cv_accuracy_std', 0)
            cv_auc = cv.get('cv_auc_mean', 0)
            cv_auc_std = cv.get('cv_auc_std', 0)
            msg_lines.append(f"   🔄 <b>Cross-Validation ({cv.get('n_splits', 5)}-fold):</b>")
            msg_lines.append(f"      ├ CV Acc:  {cv_acc*100:.2f}% ± {cv_acc_std*100:.2f}%")
            msg_lines.append(f"      └ CV AUC:  {cv_auc:.4f} ± {cv_auc_std:.4f}")

        msg_lines.append("")

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
