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
    'TSLA': {'name': 'Tesla Inc.', 'sector': 'Hàng tiêu dùng / Ô tô điện'},
    'T': {'name': 'AT&T Inc.', 'sector': 'Viễn thông & Công nghệ'},
    'TXN': {'name': 'Texas Instruments Inc.', 'sector': 'Bán dẫn & Công nghệ'},
    'TMO': {'name': 'Thermo Fisher Scientific', 'sector': 'Y tế & Thiết bị'},
    'TMUS': {'name': 'T-Mobile US Inc.', 'sector': 'Viễn thông'},
    'TGT': {'name': 'Target Corporation', 'sector': 'Hàng tiêu dùng'},
    'META': {'name': 'Meta Platforms Inc.', 'sector': 'Công nghệ'},
    'AMD': {'name': 'Advanced Micro Devices', 'sector': 'Bán dẫn & Công nghệ'},
    'NFLX': {'name': 'Netflix Inc.', 'sector': 'Truyền thông & Giải trí'},
    'INTC': {'name': 'Intel Corporation', 'sector': 'Bán dẫn & Công nghệ'},
    'CRM': {'name': 'Salesforce Inc.', 'sector': 'Công nghệ'},
    'ADBE': {'name': 'Adobe Inc.', 'sector': 'Công nghệ'},
    'ORCL': {'name': 'Oracle Corporation', 'sector': 'Công nghệ'},
    'QCOM': {'name': 'Qualcomm Inc.', 'sector': 'Bán dẫn & Công nghệ'},
    'AVGO': {'name': 'Broadcom Inc.', 'sector': 'Bán dẫn & Công nghệ'},
    'CSCO': {'name': 'Cisco Systems Inc.', 'sector': 'Công nghệ'},
    'BAC': {'name': 'Bank of America Corp.', 'sector': 'Tài chính'},
    'MA': {'name': 'Mastercard Inc.', 'sector': 'Tài chính'},
    'DIS': {'name': 'Walt Disney Co.', 'sector': 'Truyền thông & Giải trí'},
    'PYPL': {'name': 'PayPal Holdings Inc.', 'sector': 'Tài chính & Fintech'},
    'UBER': {'name': 'Uber Technologies Inc.', 'sector': 'Công nghệ & Vận tải'},
    'PLTR': {'name': 'Palantir Technologies', 'sector': 'Trí tuệ nhân tạo (AI)'},
    'COIN': {'name': 'Coinbase Global Inc.', 'sector': 'Tài chính & Crypto'},
    'BABA': {'name': 'Alibaba Group Holding', 'sector': 'Thương mại điện tử'},
    'COST': {'name': 'Costco Wholesale Corp.', 'sector': 'Hàng tiêu dùng'},
    'F': {'name': 'Ford Motor Co.', 'sector': 'Hàng tiêu dùng & Ô tô'},
    'GM': {'name': 'General Motors Co.', 'sector': 'Hàng tiêu dùng & Ô tô'},
    'PFE': {'name': 'Pfizer Inc.', 'sector': 'Y tế & Dược phẩm'},
    'LLY': {'name': 'Eli Lilly and Company', 'sector': 'Y tế & Dược phẩm'},
    'ABBV': {'name': 'AbbVie Inc.', 'sector': 'Y tế & Dược phẩm'},
    'COP': {'name': 'ConocoPhillips', 'sector': 'Năng lượng'},
}

SECTOR_TRANSLATION = {
    'Technology': 'Công nghệ',
    'Financials': 'Tài chính',
    'Financial Services': 'Tài chính',
    'Healthcare': 'Y tế',
    'Consumer Discretionary': 'Hàng tiêu dùng',
    'Consumer Staples': 'Hàng tiêu dùng',
    'Energy': 'Năng lượng',
    'Utilities': 'Năng lượng & Tiện ích',
    'Industrials': 'Công nghiệp',
    'Materials': 'Công nghiệp',
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
