"""
CONFIG & CONSTANTS — QUANTUM DASHBOARD
======================================
Cấu hình đường dẫn, metadata mã cổ phiếu, cache và tham số vận hành của Dashboard.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DAGS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "dags"))
for p in [DAGS_DIR, "/opt/airflow/dags", "/opt/airflow"]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from config_shared import TICKERS_META as _TICKERS_META_SHARED
except ImportError:
    from dags.config_shared import TICKERS_META as _TICKERS_META_SHARED

# Paths
LOGS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "logs"))
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR, exist_ok=True)

STATIC_DIR = os.path.join(BASE_DIR, "static")
PORT = 8050
BUCKET_NAME = "stock-data"

# Tickers & Sectors
TICKERS_META = _TICKERS_META_SHARED

EXTENDED_TICKERS_META = {
    'PYPL': {'name': 'PayPal Holdings Inc.', 'sector': 'Tài chính & Fintech'},
    'COIN': {'name': 'Coinbase Global Inc.', 'sector': 'Tài chính & Crypto'},
    'BABA': {'name': 'Alibaba Group Holding', 'sector': 'Thương mại điện tử'},
    'TGT': {'name': 'Target Corporation', 'sector': 'Hàng tiêu dùng'},
    'F': {'name': 'Ford Motor Co.', 'sector': 'Hàng tiêu dùng & Ô tô'},
    'GM': {'name': 'General Motors Co.', 'sector': 'Hàng tiêu dùng & Ô tô'},
    'NKE': {'name': 'Nike Inc.', 'sector': 'Hàng tiêu dùng & Thời trang'},
    'INTU': {'name': 'Intuit Inc.', 'sector': 'Công nghệ'},
}

SECTOR_TRANSLATION = {
    'Technology': 'Công nghệ',
    'Financials': 'Tài chính - Ngân hàng',
    'Financial Services': 'Tài chính - Ngân hàng',
    'Healthcare': 'Y tế & Chăm sóc sức khỏe',
    'Consumer Discretionary': 'Hàng tiêu dùng không thiết yếu',
    'Consumer Staples': 'Hàng tiêu dùng thiết yếu',
    'Energy': 'Năng lượng & Dầu khí',
    'Utilities': 'Năng lượng & Tiện ích',
    'Industrials': 'Công nghiệp Chế tạo',
    'Materials': 'Vật liệu Công nghiệp',
    'Communication Services': 'Truyền thông & Viễn thông',
    'Real Estate': 'Bất động sản',
}

# Cache for heavy datasets
CACHE_TTL = 5
_CACHE = {
    'market': {'time': 0, 'data': None},
    'tickers': {}  # ticker -> {'time': 0, 'data': None}
}

_ON_DEMAND_PREDICTIONS = {}

# Model Cache (1 hour TTL)
MODEL_CACHE_TTL = 3600
_MODEL_CACHE = {
    'xgboost': {'model': None, 'feature_cols': None, 'loaded_at': 0},
    'lstm': {'model': None, 'scaler': None, 'feature_cols': None, 'input_dim': 0, 'loaded_at': 0},
    'finbert': {'tokenizer': None, 'model': None, 'loaded_at': 0},
}


def setup_dashboard_logging():
    """Thiết lập logging luân chuyển cho Dashboard server."""
    log_file = os.path.join(LOGS_DIR, "dashboard.log")
    handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        logger.addHandler(handler)
    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger
