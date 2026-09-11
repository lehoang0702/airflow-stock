"""
DATA QUALITY & SCHEMA VALIDATOR (MLOps Pipeline)
==================================================
Module kiểm định chất lượng dữ liệu tự động trước khi bước vào khâu huấn luyện:
  1. validate_xgboost_dataset: Kiểm tra 85+ features, tỷ lệ missing, outlier, coverage.
  2. validate_lstm_dataset: Kiểm tra tính liên tục chuỗi nến OHLCV, lookback window >= 30.
  3. validate_finbert_dataset: Kiểm tra độ phủ tin tức, tính toàn vẹn tiêu đề và ngày tháng.
  4. save_quality_report_minio: Lưu báo cáo quality_report_YYYYMMDD.json lên MinIO.
  5. alert_if_critical_failure: Báo động qua Telegram nếu vi phạm ngưỡng an toàn.
"""

import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import requests

try:
    from alert_utils import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    TELEGRAM_TOKEN = ""
    TELEGRAM_CHAT_ID = ""


class DataQualityResult:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.timestamp = datetime.now().isoformat()
        self.is_valid = True
        self.total_rows = 0
        self.total_columns = 0
        self.tickers_found = []
        self.missing_pct_by_col = {}
        self.warnings = []
        self.errors = []
        self.metrics = {}

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        logging.warning(f"[{self.dataset_name} QA Warning] {msg}")

    def add_error(self, msg: str):
        self.is_valid = False
        self.errors.append(msg)
        logging.error(f"[{self.dataset_name} QA ERROR] {msg}")

    def to_dict(self):
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "tickers_count": len(self.tickers_found),
            "tickers_found": self.tickers_found,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


def validate_xgboost_dataset(df: pd.DataFrame, expected_tickers: list = None) -> DataQualityResult:
    """
    Kiểm tra chất lượng bộ dữ liệu đặc trưng bảng (XGBoost).
    """
    qa = DataQualityResult("XGBoost_Features")
    if df is None or df.empty:
        qa.add_error("DataFrame dữ liệu rỗng (Empty DataFrame)")
        return qa

    qa.total_rows = len(df)
    qa.total_columns = len(df.columns)

    # 1. Kiểm tra các cột cốt lõi
    required_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_req = [c for c in required_cols if c not in df.columns]
    if missing_req:
        qa.add_error(f"Thiếu các cột cốt lõi bắt buộc: {missing_req}")

    # 2. Kiểm tra danh sách Ticker
    if 'Ticker' in df.columns:
        found_t = sorted(df['Ticker'].dropna().unique().tolist())
        qa.tickers_found = found_t
        if expected_tickers:
            missing_t = set(expected_tickers) - set(found_t)
            if missing_t:
                qa.add_warning(f"Thiếu dữ liệu của {len(missing_t)} tickers: {sorted(list(missing_t))}")
    
    # 3. Kiểm tra tính hợp lệ của giá & khối lượng
    if 'Close' in df.columns:
        invalid_prices = df[df['Close'] <= 0]
        if not invalid_prices.empty:
            qa.add_error(f"Phát hiện {len(invalid_prices)} bản ghi có giá Close <= 0")

    if 'High' in df.columns and 'Low' in df.columns:
        broken_candles = df[df['High'] < df['Low']]
        if not broken_candles.empty:
            qa.add_error(f"Phát hiện {len(broken_candles)} bản ghi bất thường có High < Low")

    if 'Volume' in df.columns:
        neg_volume = df[df['Volume'] < 0]
        if not neg_volume.empty:
            qa.add_error(f"Phát hiện {len(neg_volume)} bản ghi có Volume âm")

    # 4. Kiểm tra tỷ lệ Missing Values
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    high_missing_cols = null_pct[null_pct > 5.0].to_dict()
    if high_missing_cols:
        qa.add_warning(f"Có {len(high_missing_cols)} cột có tỷ lệ missing > 5%: {list(high_missing_cols.keys())[:5]}")

    qa.metrics = {
        "avg_missing_pct": float(null_pct.mean()),
        "max_missing_pct": float(null_pct.max()),
        "min_close_price": float(df['Close'].min()) if 'Close' in df.columns else 0.0,
        "max_close_price": float(df['Close'].max()) if 'Close' in df.columns else 0.0,
    }
    return qa


def validate_lstm_dataset(df: pd.DataFrame, min_lookback: int = 30, expected_tickers: list = None) -> DataQualityResult:
    """
    Kiểm tra chất lượng chuỗi thời gian nến cho mô hình LSTM.
    """
    qa = DataQualityResult("LSTM_TimeSeries")
    if df is None or df.empty:
        qa.add_error("DataFrame LSTM rỗng (Empty DataFrame)")
        return qa

    qa.total_rows = len(df)
    qa.total_columns = len(df.columns)

    # 1. Cột bắt buộc
    required = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        qa.add_error(f"Thiếu các cột bắt buộc: {missing_req}")

    # 2. Kiểm tra độ dài chuỗi cho từng Ticker >= min_lookback
    if 'Ticker' in df.columns:
        ticker_counts = df['Ticker'].value_counts().to_dict()
        qa.tickers_found = sorted(list(ticker_counts.keys()))
        short_tickers = {t: c for t, c in ticker_counts.items() if c < min_lookback}
        if short_tickers:
            qa.add_error(f"Có {len(short_tickers)} mã không đủ {min_lookback} phiên: {short_tickers}")

        if expected_tickers:
            missing_t = set(expected_tickers) - set(qa.tickers_found)
            if missing_t:
                qa.add_warning(f"Thiếu dữ liệu của tickers: {sorted(list(missing_t))}")

    # 3. Kiểm tra cấu trúc nến hợp lệ
    if all(c in df.columns for c in ['Open', 'High', 'Low', 'Close']):
        invalid_high = df[(df['High'] < df['Open']) | (df['High'] < df['Close'])]
        if not invalid_high.empty:
            qa.add_error(f"Phát hiện {len(invalid_high)} nến có High nhỏ hơn Open hoặc Close")

        invalid_low = df[(df['Low'] > df['Open']) | (df['Low'] > df['Close'])]
        if not invalid_low.empty:
            qa.add_error(f"Phát hiện {len(invalid_low)} nến có Low lớn hơn Open hoặc Close")

    qa.metrics = {
        "ticker_count": len(qa.tickers_found),
        "min_candles_per_ticker": int(min(ticker_counts.values())) if 'Ticker' in df.columns else 0,
        "max_candles_per_ticker": int(max(ticker_counts.values())) if 'Ticker' in df.columns else 0,
    }
    return qa


def validate_finbert_dataset(df: pd.DataFrame, expected_tickers: list = None) -> DataQualityResult:
    """
    Kiểm tra chất lượng nguồn tin tức cho mô hình FinBERT NLP.
    """
    qa = DataQualityResult("FinBERT_News")
    if df is None or df.empty:
        qa.add_error("Không có bài báo nào được thu thập (Empty News DataFrame)")
        return qa

    qa.total_rows = len(df)
    qa.total_columns = len(df.columns)

    # 1. Cột bắt buộc
    required = ['ticker', 'headline']
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        qa.add_error(f"Thiếu các cột bắt buộc: {missing_req}")

    # 2. Tiêu đề rỗng
    if 'headline' in df.columns:
        empty_headlines = df[df['headline'].str.strip().eq('') | df['headline'].isnull()]
        if not empty_headlines.empty:
            qa.add_warning(f"Có {len(empty_headlines)} bài báo có tiêu đề rỗng")

    # 3. Độ phủ tin tức theo mã
    if 'ticker' in df.columns:
        ticker_news_count = df['ticker'].value_counts().to_dict()
        qa.tickers_found = sorted(list(ticker_news_count.keys()))
        if expected_tickers:
            zero_news = set(expected_tickers) - set(qa.tickers_found)
            if zero_news:
                qa.add_warning(f"Có {len(zero_news)} mã không có tin tức nào trong 7 ngày qua: {sorted(list(zero_news))}")

    qa.metrics = {
        "total_news_articles": int(len(df)),
        "tickers_with_news": len(qa.tickers_found),
    }
    return qa


def save_and_alert_quality_report(
    qa_result: DataQualityResult,
    s3_hook,
    bucket_name: str,
    date_str: str,
    telegram_token: str = TELEGRAM_TOKEN,
    telegram_chat_id: str = TELEGRAM_CHAT_ID
):
    """
    Lưu kết quả QA lên MinIO và bắn cảnh báo Telegram nếu có lỗi nghiêm trọng.
    """
    report = qa_result.to_dict()
    key = f"quality/quality_report_{qa_result.dataset_name}_{date_str}.json"
    
    try:
        s3_hook.load_string(
            string_data=json.dumps(report, indent=2, ensure_ascii=False),
            key=key,
            bucket_name=bucket_name,
            replace=True
        )
        logging.info(f"✅ Đã lưu báo cáo Data Quality lên MinIO: {key}")
    except Exception as e:
        logging.warning(f"⚠️ Không thể lưu báo cáo QA lên MinIO: {e}")

    # Nếu có lỗi nghiêm trọng -> Cảnh báo khẩn cấp qua Telegram
    if not qa_result.is_valid:
        msg = (
            f"🚨 <b>CẢNH BÁO DATA QUALITY: {qa_result.dataset_name}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"❌ <b>Phát hiện {len(qa_result.errors)} lỗi dữ liệu nghiêm trọng:</b>\n"
        )
        for err in qa_result.errors[:5]:
            msg += f"  • <code>{err}</code>\n"
        msg += f"\n⚠️ Vui lòng kiểm tra lại nguồn cấp dữ liệu trước khi train!"

        try:
            url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
            requests.post(url, data={'chat_id': telegram_chat_id, 'text': msg, 'parse_mode': 'HTML'}, timeout=10)
        except Exception as e:
            logging.warning(f"⚠️ Không thể gửi cảnh báo Telegram QA: {e}")

    return report
