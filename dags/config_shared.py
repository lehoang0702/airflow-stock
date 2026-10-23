"""
MODULE: CONFIG SHARED — CẤU HÌNH DÙNG CHUNG TOÀN HỆ THỐNG
=========================================================
Tập trung hóa các định nghĩa chung về danh mục 100 cổ phiếu vốn hóa lớn nhất (Top 100 S&P),
phân loại ngành (GICS), siêu dữ liệu và lịch vận hành tự động cho toàn bộ các DAGs và Dashboard.
"""

# 1. Ánh xạ 100 mã cổ phiếu lớn nhất theo vốn hóa thị trường với nhóm ngành GICS
SECTOR_MAP = {
    'NVDA': 'Technology',  # Top Market Cap: ~5113.1B USD
    'AAPL': 'Technology',  # Top Market Cap: ~5058.9B USD
    'GOOGL': 'Communication Services',  # Top Market Cap: ~3945.9B USD
    'GOOG': 'Communication Services',  # Top Market Cap: ~3908.0B USD
    'MSFT': 'Technology',  # Top Market Cap: ~3901.5B USD
    'AMZN': 'Consumer Discretionary',  # Top Market Cap: ~2690.6B USD
    'META': 'Communication Services',  # Top Market Cap: ~1641.2B USD
    'AVGO': 'Technology',  # Top Market Cap: ~1550.7B USD
    'TSLA': 'Consumer Discretionary',  # Top Market Cap: ~1472.8B USD
    'BRK-B': 'Financials',  # Top Market Cap: ~1161.5B USD
    'LLY': 'Healthcare',  # Top Market Cap: ~1138.2B USD
    'MU': 'Technology',  # Top Market Cap: ~1001.1B USD
    'WMT': 'Consumer Staples',  # Top Market Cap: ~949.7B USD
    'JPM': 'Financials',  # Top Market Cap: ~880.3B USD
    'AMD': 'Technology',  # Top Market Cap: ~842.3B USD
    'V': 'Financials',  # Top Market Cap: ~767.0B USD
    'XOM': 'Energy',  # Top Market Cap: ~751.4B USD
    'JNJ': 'Healthcare',  # Top Market Cap: ~633.0B USD
    'MA': 'Financials',  # Top Market Cap: ~544.0B USD
    'ABBV': 'Healthcare',  # Top Market Cap: ~504.9B USD
    'PLTR': 'Technology',  # Top Market Cap: ~470.8B USD
    'CVX': 'Energy',  # Top Market Cap: ~456.5B USD
    'INTC': 'Technology',  # Top Market Cap: ~455.9B USD
    'COST': 'Consumer Staples',  # Top Market Cap: ~445.5B USD
    'CSCO': 'Technology',  # Top Market Cap: ~419.3B USD
    'KO': 'Consumer Staples',  # Top Market Cap: ~400.2B USD
    'ORCL': 'Technology',  # Top Market Cap: ~386.9B USD
    'UNH': 'Healthcare',  # Top Market Cap: ~379.5B USD
    'PG': 'Consumer Staples',  # Top Market Cap: ~373.2B USD
    'MRK': 'Healthcare',  # Top Market Cap: ~356.9B USD
    'BAC': 'Financials',  # Top Market Cap: ~355.2B USD
    'HD': 'Consumer Discretionary',  # Top Market Cap: ~317.6B USD
    'CAT': 'Industrials',  # Top Market Cap: ~316.0B USD
    'PM': 'Consumer Staples',  # Top Market Cap: ~312.6B USD
    'PANW': 'Technology',  # Top Market Cap: ~310.1B USD
    'AMAT': 'Technology',  # Top Market Cap: ~301.9B USD
    'GE': 'Industrials',  # Top Market Cap: ~297.6B USD
    'DELL': 'Technology',  # Top Market Cap: ~294.2B USD
    'LRCX': 'Technology',  # Top Market Cap: ~289.8B USD
    'MS': 'Financials',  # Top Market Cap: ~284.7B USD
    'NFLX': 'Communication Services',  # Top Market Cap: ~277.1B USD
    'TXN': 'Technology',  # Top Market Cap: ~259.4B USD
    'RTX': 'Industrials',  # Top Market Cap: ~259.1B USD
    'GS': 'Financials',  # Top Market Cap: ~257.3B USD
    'TMO': 'Healthcare',  # Top Market Cap: ~247.0B USD
    'WFC': 'Financials',  # Top Market Cap: ~247.0B USD
    'IBM': 'Technology',  # Top Market Cap: ~239.8B USD
    'AMGN': 'Healthcare',  # Top Market Cap: ~238.8B USD
    'LIN': 'Materials',  # Top Market Cap: ~235.0B USD
    'GEV': 'Industrials',  # Top Market Cap: ~220.4B USD
    'ANET': 'Technology',  # Top Market Cap: ~218.8B USD
    'AXP': 'Financials',  # Top Market Cap: ~210.8B USD
    'CRM': 'Technology',  # Top Market Cap: ~203.2B USD
    'C': 'Financials',  # Top Market Cap: ~202.9B USD
    'GILD': 'Healthcare',  # Top Market Cap: ~195.0B USD
    'DIS': 'Communication Services',  # Top Market Cap: ~194.5B USD
    'QCOM': 'Technology',  # Top Market Cap: ~186.5B USD
    'VZ': 'Communication Services',  # Top Market Cap: ~185.6B USD
    'COP': 'Energy',  # Top Market Cap: ~184.9B USD
    'PEP': 'Consumer Staples',  # Top Market Cap: ~183.6B USD
    'SNDK': 'Technology',  # Top Market Cap: ~183.2B USD
    'ABT': 'Healthcare',  # Top Market Cap: ~178.9B USD
    'MCD': 'Consumer Discretionary',  # Top Market Cap: ~175.1B USD
    'UNP': 'Industrials',  # Top Market Cap: ~174.2B USD
    'DE': 'Industrials',  # Top Market Cap: ~173.6B USD
    'BLK': 'Financials',  # Top Market Cap: ~172.8B USD
    'PFE': 'Healthcare',  # Top Market Cap: ~171.3B USD
    'NEE': 'Utilities',  # Top Market Cap: ~171.1B USD
    'NOW': 'Technology',  # Top Market Cap: ~162.9B USD
    'ISRG': 'Healthcare',  # Top Market Cap: ~161.8B USD
    'TMUS': 'Communication Services',  # Top Market Cap: ~161.0B USD
    'DHR': 'Healthcare',  # Top Market Cap: ~159.4B USD
    'SCHW': 'Financials',  # Top Market Cap: ~156.7B USD
    'ACN': 'Technology',  # Top Market Cap: ~151.4B USD
    'BA': 'Industrials',  # Top Market Cap: ~148.9B USD
    'T': 'Communication Services',  # Top Market Cap: ~144.1B USD
    'LMT': 'Industrials',  # Top Market Cap: ~139.1B USD
    'UBER': 'Technology',  # Top Market Cap: ~138.3B USD
    'BMY': 'Healthcare',  # Top Market Cap: ~126.0B USD
    'MO': 'Consumer Staples',  # Top Market Cap: ~124.9B USD
    'COF': 'Financials',  # Top Market Cap: ~123.2B USD
    'MDT': 'Healthcare',  # Top Market Cap: ~122.8B USD
    'CVS': 'Healthcare',  # Top Market Cap: ~119.5B USD
    'BKNG': 'Consumer Discretionary',  # Top Market Cap: ~117.5B USD
    'SBUX': 'Consumer Discretionary',  # Top Market Cap: ~114.7B USD
    'LOW': 'Consumer Discretionary',  # Top Market Cap: ~111.3B USD
    'SO': 'Utilities',  # Top Market Cap: ~106.8B USD
    'ADBE': 'Technology',  # Top Market Cap: ~103.2B USD
    'BNY': 'Financials',  # Top Market Cap: ~98.3B USD
    'DUK': 'Utilities',  # Top Market Cap: ~97.0B USD
    'GD': 'Industrials',  # Top Market Cap: ~94.6B USD
    'USB': 'Financials',  # Top Market Cap: ~92.8B USD
    'INTU': 'Technology',  # Top Market Cap: ~92.6B USD
    'EMR': 'Industrials',  # Top Market Cap: ~90.1B USD
    'MMM': 'Industrials',  # Top Market Cap: ~89.7B USD
    'AMT': 'Real Estate',  # Top Market Cap: ~87.7B USD
    'UPS': 'Industrials',  # Top Market Cap: ~85.5B USD
    'MDLZ': 'Consumer Staples',  # Top Market Cap: ~81.1B USD
    'CMCSA': 'Communication Services',  # Top Market Cap: ~80.5B USD
    'FDX': 'Industrials',  # Top Market Cap: ~70.7B USD
}


# 2. Danh sách 100 mã cổ phiếu theo thứ tự vốn hóa từ lớn nhất đến nhỏ nhất
TICKERS_LIST = list(SECTOR_MAP.keys())

# 3. Siêu dữ liệu chi tiết hiển thị cho Dashboard & Báo cáo
TICKERS_META = {
'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Công nghệ', 'market_cap_b': 5113.13},
    'AAPL': {'name': 'Apple Inc.', 'sector': 'Công nghệ', 'market_cap_b': 5058.93},
    'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Truyền thông & Viễn thông', 'market_cap_b': 3945.87},
    'GOOG': {'name': 'Alphabet Inc.', 'sector': 'Truyền thông & Viễn thông', 'market_cap_b': 3908.01},
    'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Công nghệ', 'market_cap_b': 3901.53},
    'AMZN': {'name': 'Amazon.com, Inc.', 'sector': 'Hàng tiêu dùng không thiết yếu', 'market_cap_b': 2690.65},
    'META': {'name': 'Meta Platforms, Inc.', 'sector': 'Truyền thông & Viễn thông', 'market_cap_b': 1641.17},
    'AVGO': {'name': 'Broadcom Inc.', 'sector': 'Công nghệ', 'market_cap_b': 1550.74},
    'TSLA': {'name': 'Tesla, Inc.', 'sector': 'Hàng tiêu dùng không thiết yếu', 'market_cap_b': 1472.83},
    'BRK-B': {'name': 'Berkshire Hathaway Inc. New', 'sector': 'Tài chính - Ngân hàng', 'market_cap_b': 1161.50},
    'LLY': {'name': 'Eli Lilly and Company', 'sector': 'Y tế & Chăm sóc sức khỏe', 'market_cap_b': 1138.25},
    'MU': {'name': 'Micron Technology, Inc.', 'sector': 'Công nghệ', 'market_cap_b': 1001.09},
    'WMT': {'name': 'Walmart Inc.', 'sector': 'Hàng tiêu dùng thiết yếu', 'market_cap_b': 949.75},
    'JPM': {'name': 'JP Morgan Chase & Co.', 'sector': 'Tài chính - Ngân hàng', 'market_cap_b': 880.33},
    'AMD': {'name': 'Advanced Micro Devices, Inc.', 'sector': 'Công nghệ', 'market_cap_b': 842.29},
    'V': {'name': 'Visa Inc.', 'sector': 'Tài chính - Ngân hàng', 'market_cap_b': 766.96},
    'XOM': {'name': 'ExxonMobil Holdings Corporation', 'sector': 'Năng lượng & Dầu khí', 'market_cap_b': 751.39},
    'JNJ': {'name': 'Johnson & Johnson', 'sector': 'Y tế & Chăm sóc sức khỏe', 'market_cap_b': 633.01},
    'MA': {'name': 'Mastercard Incorporated', 'sector': 'Tài chính - Ngân hàng', 'market_cap_b': 544.01},
    'ABBV': {'name': 'AbbVie Inc.', 'sector': 'Y tế & Chăm sóc sức khỏe', 'market_cap_b': 504.90},
    'PLTR': {'name': 'Palantir Technologies Inc.', 'sector': 'Công nghệ', 'market_cap_b': 470.82},
    'CVX': {'name': 'Chevron Corporation', 'sector': 'Năng lượng & Dầu khí', 'market_cap_b': 456.50},
    'INTC': {'name': 'Intel Corporation', 'sector': 'Công nghệ', 'market_cap_b': 455.86},
    'COST': {'name': 'Costco Wholesale Corporation', 'sector': 'Hàng tiêu dùng thiết yếu', 'market_cap_b': 445.49},
    'CSCO': {'name': 'Cisco Systems, Inc.', 'sector': 'Công nghệ', 'market_cap_b': 419.33},
    'KO': {'name': 'Coca-Cola Company (The)', 'sector': 'Hàng tiêu dùng thiết yếu', 'market_cap_b': 400.20},
    'ORCL': {'name': 'Oracle Corporation', 'sector': 'Công nghệ', 'market_cap_b': 386.94},
    'UNH': {'name': 'UnitedHealth Group Incorporated', 'sector': 'Y tế & Chăm sóc sức khỏe', 'market_cap_b': 379.47},
    'PG': {'name': 'Procter & Gamble Company (The)', 'sector': 'Hàng tiêu dùng thiết yếu', 'market_cap_b': 373.16},
    'MRK': {'name': 'Merck & Company, Inc.', 'sector': 'Y tế & Chăm sóc sức khỏe', 'market_cap_b': 356.90},
    'BAC': {'name': 'Bank of America Corporation', 'sector': 'Tài chính - Ngân hàng', 'market_cap_b': 355.16},
    'HD': {'name': 'Home Depot, Inc. (The)', 'sector': 'Hàng tiêu dùng không thiết yếu', 'market_cap_b': 317.61},
    'CAT': {'name': 'Caterpillar, Inc.', 'sector': 'Công nghiệp Chế tạo', 'market_cap_b': 315.96},
    'PM': {'name': 'Philip Morris International Inc', 'sector': 'Hàng tiêu dùng thiết yếu', 'market_cap_b': 312.64},
    'PANW': {'name': 'Palo Alto Networks, Inc.', 'sector': 'Công nghệ', 'market_cap_b': 310.13},
    'AMAT': {'name': 'Applied Materials, Inc.', 'sector': 'Công nghệ', 'market_cap_b': 301.89},
    'GE': {'name': 'GE Aerospace', 'sector': 'Công nghiệp Chế tạo', 'market_cap_b': 297.63},
    'DELL': {'name': 'Dell Technologies Inc.', 'sector': 'Công nghệ', 'market_cap_b': 294.19},
    'LRCX': {'name': 'Lam Research Corporation', 'sector': 'Công nghệ', 'market_cap_b': 289.77},
    'MS': {'name': 'Morgan Stanley', 'sector': 'Tài chính - Ngân hàng', 'market_cap_b': 284.70},
    'NFLX': {'name': 'Netflix, Inc.', 'sector': 'Truyền thông & Viễn thông', 'market_cap_b': 277.11},
    'TXN': {'name': 'Texas Instruments Incorporated', 'sector': 'Công nghệ', 'market_cap_b': 259.35},
    'RTX': {'name': 'RTX Corporation', 'sector': 'Công nghiệp Chế tạo', 'market_cap_b': 259.14},
    'GS': {'name': 'Goldman Sachs Group, Inc. (The)', 'sector': 'Tài chính - Ngân hàng', 'market_cap_b': 257.29},
    'TMO': {'name': 'Thermo Fisher Scientific Inc', 'sector': 'Y tế & Chăm sóc sức khỏe', 'market_cap_b': 247.04},
    'WFC': {'name': 'Wells Fargo & Company', 'sector': 'Tài chính - Ngân hàng', 'market_cap_b': 246.96},
    'IBM': {'name': 'International Business Machines', 'sector': 'Công nghệ', 'market_cap_b': 239.77},
    'AMGN': {'name': 'Amgen Inc.', 'sector': 'Y tế & Chăm sóc sức khỏe', 'market_cap_b': 238.75},
    'LIN': {'name': 'Linde plc', 'sector': 'Vật liệu Công nghiệp', 'market_cap_b': 234.95},
    'GEV': {'name': 'GE Vernova Inc.', 'sector': 'Công nghiệp Chế tạo', 'market_cap_b': 220.38},
    'ANET': {'name': 'Arista Networks, Inc.', 'sector': 'Công nghệ', 'market_cap_b': 218.81},
    'AXP': {'name': 'American Express Company', 'sector': 'Tài chính - Ngân hàng', 'market_cap_b': 210.82},
    'CRM': {'name': 'Salesforce, Inc.', 'sector': 'Công nghệ', 'market_cap_b': 203.16},
    'C': {'name': 'Citigroup, Inc.', 'sector': 'Tài chính - Ngân hàng', 'market_cap_b': 202.92},
    'GILD': {'name': 'Gilead Sciences, Inc.', 'sector': 'Y tế & Chăm sóc sức khỏe', 'market_cap_b': 195.00},
    'DIS': {'name': 'Walt Disney Company (The)', 'sector': 'Truyền thông & Viễn thông', 'market_cap_b': 194.55},
    'QCOM': {'name': 'QUALCOMM Incorporated', 'sector': 'Công nghệ', 'market_cap_b': 186.47},
    'VZ': {'name': 'Verizon Communications Inc.', 'sector': 'Truyền thông & Viễn thông', 'market_cap_b': 185.57},
    'COP': {'name': 'ConocoPhillips', 'sector': 'Năng lượng & Dầu khí', 'market_cap_b': 184.91},
    'PEP': {'name': 'Pepsico, Inc.', 'sector': 'Hàng tiêu dùng thiết yếu', 'market_cap_b': 183.63},
    'SNDK': {'name': 'Sandisk Corporation', 'sector': 'Công nghệ', 'market_cap_b': 183.21},
    'ABT': {'name': 'Abbott Laboratories', 'sector': 'Y tế & Chăm sóc sức khỏe', 'market_cap_b': 178.90},
    'MCD': {'name': 'McDonald\'s Corporation', 'sector': 'Hàng tiêu dùng không thiết yếu', 'market_cap_b': 175.14},
    'UNP': {'name': 'Union Pacific Corporation', 'sector': 'Công nghiệp Chế tạo', 'market_cap_b': 174.21},
    'DE': {'name': 'Deere & Company', 'sector': 'Công nghiệp Chế tạo', 'market_cap_b': 173.59},
    'BLK': {'name': 'BlackRock, Inc.', 'sector': 'Tài chính - Ngân hàng', 'market_cap_b': 172.81},
    'PFE': {'name': 'Pfizer, Inc.', 'sector': 'Y tế & Chăm sóc sức khỏe', 'market_cap_b': 171.28},
    'NEE': {'name': 'NextEra Energy, Inc.', 'sector': 'Năng lượng & Tiện ích', 'market_cap_b': 171.08},
    'NOW': {'name': 'ServiceNow, Inc.', 'sector': 'Công nghệ', 'market_cap_b': 162.85},
    'ISRG': {'name': 'Intuitive Surgical, Inc.', 'sector': 'Y tế & Chăm sóc sức khỏe', 'market_cap_b': 161.80},
    'TMUS': {'name': 'T-Mobile US, Inc.', 'sector': 'Truyền thông & Viễn thông', 'market_cap_b': 160.95},
    'DHR': {'name': 'Danaher Corporation', 'sector': 'Y tế & Chăm sóc sức khỏe', 'market_cap_b': 159.38},
    'SCHW': {'name': 'Charles Schwab Corporation (The', 'sector': 'Tài chính - Ngân hàng', 'market_cap_b': 156.65},
    'ACN': {'name': 'Accenture plc', 'sector': 'Công nghệ', 'market_cap_b': 151.35},
    'BA': {'name': 'Boeing Company (The)', 'sector': 'Công nghiệp Chế tạo', 'market_cap_b': 148.86},
    'T': {'name': 'AT&T Inc.', 'sector': 'Truyền thông & Viễn thông', 'market_cap_b': 144.09},
    'LMT': {'name': 'Lockheed Martin Corporation', 'sector': 'Công nghiệp Chế tạo', 'market_cap_b': 139.05},
    'UBER': {'name': 'Uber Technologies, Inc.', 'sector': 'Công nghệ', 'market_cap_b': 138.29},
    'BMY': {'name': 'Bristol-Myers Squibb Company', 'sector': 'Y tế & Chăm sóc sức khỏe', 'market_cap_b': 126.04},
    'MO': {'name': 'Altria Group, Inc.', 'sector': 'Hàng tiêu dùng thiết yếu', 'market_cap_b': 124.95},
    'COF': {'name': 'Capital One Financial Corporati', 'sector': 'Tài chính - Ngân hàng', 'market_cap_b': 123.24},
    'MDT': {'name': 'Medtronic plc.', 'sector': 'Y tế & Chăm sóc sức khỏe', 'market_cap_b': 122.77},
    'CVS': {'name': 'CVS Health Corporation', 'sector': 'Y tế & Chăm sóc sức khỏe', 'market_cap_b': 119.51},
    'BKNG': {'name': 'Booking Holdings Inc. Common St', 'sector': 'Hàng tiêu dùng không thiết yếu', 'market_cap_b': 117.48},
    'SBUX': {'name': 'Starbucks Corporation', 'sector': 'Hàng tiêu dùng không thiết yếu', 'market_cap_b': 114.67},
    'LOW': {'name': 'Lowe\'s Companies, Inc.', 'sector': 'Hàng tiêu dùng không thiết yếu', 'market_cap_b': 111.27},
    'SO': {'name': 'Southern Company (The)', 'sector': 'Năng lượng & Tiện ích', 'market_cap_b': 106.83},
    'ADBE': {'name': 'Adobe Inc.', 'sector': 'Công nghệ', 'market_cap_b': 103.20},
    'BNY': {'name': 'The Bank of New York Mellon Cor', 'sector': 'Tài chính - Ngân hàng', 'market_cap_b': 98.31},
    'DUK': {'name': 'Duke Energy Corporation (Holdin', 'sector': 'Năng lượng & Tiện ích', 'market_cap_b': 97.05},
    'GD': {'name': 'General Dynamics Corporation', 'sector': 'Công nghiệp Chế tạo', 'market_cap_b': 94.58},
    'USB': {'name': 'U.S. Bancorp', 'sector': 'Tài chính - Ngân hàng', 'market_cap_b': 92.80},
    'INTU': {'name': 'Intuit Inc.', 'sector': 'Công nghệ', 'market_cap_b': 92.58},
    'EMR': {'name': 'Emerson Electric Company', 'sector': 'Công nghiệp Chế tạo', 'market_cap_b': 90.11},
    'MMM': {'name': '3M Company', 'sector': 'Công nghiệp Chế tạo', 'market_cap_b': 89.67},
    'AMT': {'name': 'American Tower Corporation (REI', 'sector': 'Bất động sản', 'market_cap_b': 87.70},
    'UPS': {'name': 'United Parcel Service, Inc.', 'sector': 'Công nghiệp Chế tạo', 'market_cap_b': 85.48},
    'MDLZ': {'name': 'Mondelez International, Inc.', 'sector': 'Hàng tiêu dùng thiết yếu', 'market_cap_b': 81.05},
    'CMCSA': {'name': 'Comcast Corporation', 'sector': 'Truyền thông & Viễn thông', 'market_cap_b': 80.54},
    'FDX': {'name': 'FedEx Corporation', 'sector': 'Công nghiệp Chế tạo', 'market_cap_b': 70.73},
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

# 6. Cấu hình lịch sử dữ liệu chuẩn cho Machine Learning
DATASET_HISTORY_PERIOD = '5y'  # Lịch sử 5 năm (tối ưu: 1y warm-up + 3y train + 3m val + 6m test)

# 7. Lịch chạy tự động tối ưu cho thị trường chứng khoán Mỹ
# 22:00 UTC (05:00 sáng VN) từ Thứ Hai đến Thứ Sáu (sau khi phiên NYSE/NASDAQ đóng cửa)
DEFAULT_CRON_SCHEDULE = '0 22 * * 1-5'

# 8. Ngưỡng tín hiệu giao dịch thống nhất toàn hệ thống (Trading Decision Thresholds)
# Xác suất tăng giá >= BUY_THRESHOLD -> MUA
# Xác suất tăng giá <= SELL_THRESHOLD -> BÁN
# Vùng 45% - 55% -> ĐỨNG NGOÀI / THEO DÕI (Lọc nhiễu thị trường đi ngang)
BUY_THRESHOLD = 0.55        # 55.0%
SELL_THRESHOLD = 0.45       # 45.0%
BUY_THRESHOLD_PCT = 55.0    # Tỷ lệ phần trăm cho Dashboard (55.0%)
SELL_THRESHOLD_PCT = 45.0   # Tỷ lệ phần trăm cho Dashboard (45.0%)

