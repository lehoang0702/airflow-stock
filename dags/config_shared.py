"""
MODULE: CONFIG SHARED — CẤU HÌNH DÙNG CHUNG TOÀN HỆ THỐNG
=========================================================
Tập trung hóa các định nghĩa chung về danh mục cổ phiếu, phân loại ngành (GICS),
siêu dữ liệu và lịch vận hành tự động cho toàn bộ các DAGs và Dashboard.
"""

# 1. Ánh xạ 20 mã cổ phiếu với nhóm ngành chuẩn quốc tế GICS (tiếng Anh)
SECTOR_MAP = {
    # Technology (Công nghệ)
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'NVDA': 'Technology',
    'GOOGL': 'Technology',

    # Consumer Discretionary (Hàng tiêu dùng không thiết yếu - Bao gồm Amazon)
    'AMZN': 'Consumer Discretionary',
    'MCD': 'Consumer Discretionary',
    'NKE': 'Consumer Discretionary',

    # Consumer Staples (Hàng tiêu dùng thiết yếu)
    'WMT': 'Consumer Staples',
    'PG': 'Consumer Staples',
    'KO': 'Consumer Staples',

    # Financials (Tài chính - Ngân hàng)
    'JPM': 'Financials',
    'V': 'Financials',

    # Healthcare (Y tế & Dược phẩm)
    'UNH': 'Healthcare',
    'JNJ': 'Healthcare',

    # Industrials (Công nghiệp Chế tạo)
    'CAT': 'Industrials',
    'BA': 'Industrials',

    # Energy (Năng lượng & Dầu khí)
    'XOM': 'Energy',
    'CVX': 'Energy',

    # Utilities (Tiện ích & Năng lượng tái tạo)
    'NEE': 'Utilities',

    # Materials (Vật liệu Công nghiệp)
    'LIN': 'Materials'
}

# 2. Danh sách 20 mã cổ phiếu S&P 500 theo thứ tự chuẩn
TICKERS_LIST = list(SECTOR_MAP.keys())

# 3. Siêu dữ liệu chi tiết hiển thị cho Dashboard & Báo cáo (tiếng Việt)
TICKERS_META = {
    'AAPL': {'name': 'Apple Inc.', 'sector': 'Công nghệ'},
    'MSFT': {'name': 'Microsoft Corp.', 'sector': 'Công nghệ'},
    'NVDA': {'name': 'NVIDIA Corp.', 'sector': 'Công nghệ'},
    'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Công nghệ'},
    'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'Hàng tiêu dùng không thiết yếu'},
    'JPM': {'name': 'JPMorgan Chase & Co.', 'sector': 'Tài chính - Ngân hàng'},
    'V': {'name': 'Visa Inc.', 'sector': 'Dịch vụ Tài chính'},
    'JNJ': {'name': 'Johnson & Johnson', 'sector': 'Y tế & Chăm sóc sức khỏe'},
    'UNH': {'name': 'UnitedHealth Group', 'sector': 'Bảo hiểm Y tế'},
    'XOM': {'name': 'Exxon Mobil Corp.', 'sector': 'Năng lượng & Dầu khí'},
    'CVX': {'name': 'Chevron Corp.', 'sector': 'Năng lượng & Dầu khí'},
    'PG': {'name': 'Procter & Gamble Co.', 'sector': 'Hàng tiêu dùng thiết yếu'},
    'KO': {'name': 'Coca-Cola Co.', 'sector': 'Đồ uống & Hàng tiêu dùng'},
    'WMT': {'name': 'Walmart Inc.', 'sector': 'Bán lẻ & Tiêu dùng'},
    'MCD': {'name': "McDonald's Corp.", 'sector': 'Dịch vụ Ăn uống'},
    'NKE': {'name': 'Nike Inc.', 'sector': 'Thời trang & Thể thao'},
    'CAT': {'name': 'Caterpillar Inc.', 'sector': 'Công nghiệp Chế tạo'},
    'BA': {'name': 'Boeing Co.', 'sector': 'Hàng không & Quốc phòng'},
    'NEE': {'name': 'NextEra Energy Inc.', 'sector': 'Năng lượng & Tiện ích'},
    'LIN': {'name': 'Linde plc', 'sector': 'Vật liệu Công nghiệp'},
}

# 4. Chỉ số thị trường vĩ mô tham chiếu (Macro Benchmarks)
MACRO_TICKERS = {
    'SPY': 'spy',      # S&P 500 ETF
    'QQQ': 'qqq',      # Nasdaq 100 ETF
    '^VIX': 'vix',     # CBOE Volatility Index
    '^TNX': 'tnx'      # 10Y US Treasury Yield
}

# 5. Lịch chạy tự động tối ưu cho thị trường chứng khoán Mỹ
# 22:00 UTC (05:00 sáng VN) từ Thứ Hai đến Thứ Sáu (sau khi phiên NYSE/NASDAQ đóng cửa)
DEFAULT_CRON_SCHEDULE = '0 22 * * 1-5'
