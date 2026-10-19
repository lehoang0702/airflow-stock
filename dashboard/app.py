#!/usr/bin/env python3
"""
QUANTUM MULTI-MODAL DASHBOARD — ENTRYPOINT
==========================================
Web Server tích hợp MinIO S3 & Real-time AI Inference phục vụ Quantum Web Dashboard.
Port: 8050 | Static: dashboard/static/ (HTML, CSS, JS)

Kiến trúc module hóa:
  - dashboard.config: Cấu hình hệ thống, constants, cache & logging
  - dashboard.storage: Kết nối MinIO S3, tìm latest keys, đọc Parquet/CSV
  - dashboard.charts: Render biểu đồ nến kỹ thuật Dual-Panel, cache ảnh biểu đồ
  - dashboard.services: Business logic cho market, ticker, quality, evaluation
  - dashboard.inference: Model loading (XGBoost, LSTM, FinBERT) & on-demand AI pipeline
  - dashboard.handler: HTTP Request Router & Static File Server
"""

import os
import sys
import logging
from http.server import ThreadingHTTPServer

# Cấu hình đường dẫn module
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DAGS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "dags"))
for p in [BASE_DIR, DAGS_DIR, "/opt/airflow/dags", "/opt/airflow"]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import từ các module thành phần
from dashboard.config import (
    PORT,
    STATIC_DIR,
    BUCKET_NAME,
    CACHE_TTL,
    MODEL_CACHE_TTL,
    TICKERS_META,
    EXTENDED_TICKERS_META,
    SECTOR_TRANSLATION,
    _CACHE,
    _MODEL_CACHE,
    _ON_DEMAND_PREDICTIONS,
    setup_dashboard_logging,
)

from dashboard.storage import (
    get_s3_client,
    get_latest_s3_key,
    read_s3_dataframe,
    extract_percentage,
)

from dashboard.charts import (
    _CHART_CACHE,
    generate_live_chart_image,
    get_chart_image,
)

from dashboard.services import (
    get_market_data,
    get_ticker_chart,
    get_quality_data,
    get_evaluation_data,
    _market_result,
)

from dashboard.inference import (
    _load_xgboost_from_minio,
    _load_lstm_from_minio,
    _load_finbert_model,
    _calculate_xgboost_features_ondemand,
    _calculate_lstm_features_ondemand,
    _crawl_macro_data_ondemand,
    _predict_xgboost_real,
    _predict_lstm_real,
    _predict_finbert_real,
    analyze_ticker_on_demand,
    _combine_available_model_predictions,
)

from dashboard.handler import DashboardHandler


def _unavailable_market_data(today_str: str):
    """Trả về cấu trúc dữ liệu mặc định khi pipeline chưa có dự báo."""
    return {
        "date": None,
        "status": "unavailable",
        "market_mood": "CHƯA CÓ DỮ LIỆU",
        "message": "Chưa đọc được dữ liệu dự báo. Vui lòng kiểm tra pipeline và kết nối dữ liệu.",
        "buy_count": 0, "sell_count": 0, "hold_count": 0,
        "predictions": [],
    }


def run_server():
    """Khởi động Dashboard HTTP Server."""
    setup_dashboard_logging()
    server_address = ('0.0.0.0', PORT)
    httpd = ThreadingHTTPServer(server_address, DashboardHandler)
    logging.info(f"🚀 Quantum Dashboard Server đang chạy tại: http://0.0.0.0:{PORT}")
    logging.info(f"📂 Thư mục static: {STATIC_DIR}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        logging.info("🛑 Đã dừng Dashboard Server.")


if __name__ == '__main__':
    run_server()
