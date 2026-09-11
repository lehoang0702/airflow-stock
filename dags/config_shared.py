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

# 4. Chỉ số thị trường vĩ mô & Liên thị trường mở rộng (Inter-Market Benchmarks)
MACRO_TICKERS = {
    'SPY': 'spy',           # S&P 500 ETF (Chỉ số thị trường chung)
    'QQQ': 'qqq',           # Nasdaq 100 ETF (Công nghệ & Đổi mới)
    '^VIX': 'vix',          # CBOE Volatility Index (Thước đo nỗi sợ thị trường)
    '^TNX': 'tnx',          # 10Y US Treasury Yield (Lợi suất trái phiếu chính phủ Mỹ 10 năm)
    'GC=F': 'gold',         # Vàng tương lai (Gold Futures - Tài sản trú ẩn an toàn)
    'CL=F': 'oil',          # Dầu thô WTI (Crude Oil - Năng lượng & Lạm phát)
    'DX-Y.NYB': 'dxy',      # US Dollar Index (Chỉ số sức mạnh đồng Đô la Mỹ)
    'HYG': 'hyg',           # iShares High Yield Corporate Bond (Khẩu vị rủi ro tín dụng)
}

# 5. Danh mục các trường dữ liệu Báo cáo Tài chính & Phân tích Cơ bản (Fundamental Metrics)
FUNDAMENTAL_FIELDS = [
    'trailingPE',           # P/E 12 tháng qua
    'forwardPE',            # P/E dự phóng
    'priceToBook',          # Chỉ số P/B
    'returnOnEquity',       # Tỷ suất sinh lời trên vốn chủ sở hữu (ROE)
    'profitMargins',        # Biên lợi nhuận ròng (Profit Margin)
    'debtToEquity',         # Tỷ lệ Nợ / Vốn chủ sở hữu
    'beta',                 # Hệ số rủi ro hệ thống Beta
    'marketCap',            # Vốn hóa thị trường
]

# 6. Cấu hình khung thời gian nến Intraday (Độ mịn cao)
INTRADAY_INTERVAL = '1h'    # Nến 1 giờ
INTRADAY_PERIOD = '730d'    # 730 ngày gần nhất (tối đa Yahoo Finance hỗ trợ cho nến 1h)

# 7. Cấu hình lịch sử dữ liệu chuẩn cho Machine Learning
DATASET_HISTORY_PERIOD = '15y'  # Lịch sử 15 năm (2011 đến nay)

# 8. Lịch chạy tự động tối ưu cho thị trường chứng khoán Mỹ
# 22:00 UTC (05:00 sáng VN) từ Thứ Hai đến Thứ Sáu (sau khi phiên NYSE/NASDAQ đóng cửa)
DEFAULT_CRON_SCHEDULE = '0 22 * * 1-5'
