/**
 * QUANTUM MULTI-MODAL DASHBOARD - CLIENT APP LOGIC (v6.0)
 * =========================================================
 * Smart Market Screener (100 Stocks Table), Interactive Candlestick with TP/SL Targets,
 * 4-Model Probability Gauges, FinBERT News Feed, and AI Auditing.
 */

const TICKERS = [
  'NVDA', 'AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN', 'META', 'AVGO', 'TSLA', 'BRK-B',
  'LLY', 'MU', 'WMT', 'JPM', 'AMD', 'V', 'XOM', 'JNJ', 'MA', 'ABBV',
  'PLTR', 'CVX', 'INTC', 'COST', 'CSCO', 'KO', 'ORCL', 'UNH', 'PG', 'MRK',
  'BAC', 'HD', 'CAT', 'PM', 'PANW', 'AMAT', 'GE', 'DELL', 'LRCX', 'MS',
  'NFLX', 'TXN', 'RTX', 'GS', 'TMO', 'WFC', 'IBM', 'AMGN', 'LIN', 'GEV',
  'ANET', 'AXP', 'CRM', 'C', 'GILD', 'DIS', 'QCOM', 'VZ', 'COP', 'PEP',
  'SNDK', 'ABT', 'MCD', 'UNP', 'DE', 'BLK', 'PFE', 'NEE', 'NOW', 'ISRG',
  'TMUS', 'DHR', 'SCHW', 'ACN', 'BA', 'T', 'LMT', 'UBER', 'BMY', 'MO',
  'COF', 'MDT', 'CVS', 'BKNG', 'SBUX', 'LOW', 'SO', 'ADBE', 'BNY', 'DUK',
  'GD', 'USB', 'INTU', 'EMR', 'MMM', 'AMT', 'UPS', 'MDLZ', 'CMCSA', 'FDX'
];

const SECTOR_MAP_VN = {
  'Technology': 'Công Nghệ',
  'Communication Services': 'Truyền Thông',
  'Consumer Discretionary': 'Tiêu Dùng',
  'Consumer Staples': 'Hàng Thiết Yếu',
  'Financials': 'Tài Chính',
  'Healthcare': 'Y Tế',
  'Industrials': 'Công Nghiệp',
  'Energy': 'Năng Lượng',
  'Utilities': 'Tiện Ích',
  'Materials': 'Vật Liệu Cơ Bản',
  'Real Estate': 'Bất Động Sản'
};

let state = {
  activeTicker: 'META',
  activeView: 'candlestick', // 'candlestick' (default) | 'ensemble' | 'xgboost' | 'lstm' | 'finbert'
  activeTab: 'tab-charts',    // 'tab-charts' (Yahoo Finance Quote View) | 'tab-screener' | 'tab-audit'
  marketData: null,
  chartData: {},
  filterSignal: 'all',        // 'all' | 'buy' | 'hold' | 'sell'
  filterSector: 'all',        // 'all' | sector string
  searchKeyword: ''
};

// ==========================================================================
// INITIALIZATION
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
  const searchInput = document.getElementById('screener-search-input');
  if (searchInput) searchInput.value = '';
  state.searchKeyword = '';
  state.filterSignal = 'all';
  state.filterSector = 'all';

  initMainTabs();
  initTickerPills();
  initViewButtons();
  initLightbox();
  initScreenerFilters();
  initEventListeners();
  initAutoRefresh();
  loadAllData();
});

// 1. Quản lý Tab chính (Screener vs Charts vs Audit)
function initMainTabs() {
  const buttons = document.querySelectorAll('#main-nav-pills .nav-pill-btn');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.dataset.tab;
      switchTab(targetTab);
    });
  });
}

function switchTab(tabId) {
  state.activeTab = tabId;

  // Cập nhật trạng thái nút
  document.querySelectorAll('#main-nav-pills .nav-pill-btn').forEach(btn => {
    if (btn.dataset.tab === tabId) btn.classList.add('active');
    else btn.classList.remove('active');
  });

  // Hiển thị panel tương ứng
  document.querySelectorAll('.tab-panel').forEach(panel => {
    if (panel.id === tabId) {
      panel.classList.add('active');
    } else {
      panel.classList.remove('active');
    }
  });

  // Nếu chuyển sang tab chart thì vẽ lại biểu đồ để fit kích thước
  if (tabId === 'tab-charts' && state.chartData[state.activeTicker]) {
    const candles = state.chartData[state.activeTicker].candles || [];
    setTimeout(() => renderCandlestickSVG(candles), 50);
  }
}

// 2. Quản lý thanh chọn nhanh Ticker Pills
function initTickerPills() {
  const container = document.getElementById('ticker-pills');
  if (!container) return;

  container.innerHTML = '';
  TICKERS.forEach(ticker => {
    const pill = document.createElement('button');
    pill.className = `ticker-pill ${ticker === state.activeTicker ? 'active' : ''}`;
    pill.textContent = ticker;
    pill.addEventListener('click', () => {
      selectTicker(ticker);
    });
    container.appendChild(pill);
  });
}

function selectTicker(ticker, shouldSwitchTab = false) {
  state.activeTicker = ticker;

  // Cập nhật pills
  document.querySelectorAll('.ticker-pill').forEach(pill => {
    if (pill.textContent === ticker) pill.classList.add('active');
    else pill.classList.remove('active');
  });

  // Cập nhật header
  updateHeaderActiveTicker();

  // Tải dữ liệu biểu đồ cho mã mới
  loadTickerChart(ticker);

  const isOutside = SEARCH_DIRECTORY.some(i => i.ticker === ticker && !i.isCore);
  if (isOutside) {
    // Với mã mở rộng, ưu tiên chế độ Nến Nhật Tương Tác & MA20
    switchViewMode('candlestick');
  } else if (state.activeView !== 'candlestick') {
    updateReportImage();
  }

  // Highlight hàng tương ứng trong bảng Screener nếu đang ở bảng
  highlightScreenerRow(ticker);

  if (shouldSwitchTab) {
    switchTab('tab-charts');
    const chartCard = document.querySelector('.chart-main-card');
    if (chartCard) {
      chartCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }
}

// 3. Toolbar chuyển chế độ xem biểu đồ (Nến tương tác vs Ảnh báo cáo)
function switchViewMode(view) {
  state.activeView = view;

  const buttons = document.querySelectorAll('#view-buttons .view-btn');
  buttons.forEach(b => {
    if (b.dataset.view === view) b.classList.add('active');
    else b.classList.remove('active');
  });

  const imgContainer = document.getElementById('report-img-container');
  const candleSection = document.getElementById('candlestick-section');
  const actionsBlock = document.getElementById('view-actions-block');

  if (view === 'candlestick') {
    if (imgContainer) imgContainer.classList.add('hidden');
    if (candleSection) candleSection.classList.remove('hidden');
    if (actionsBlock) actionsBlock.classList.add('hidden');
    if (state.chartData[state.activeTicker]) {
      renderCandlestickSVG(state.chartData[state.activeTicker].candles || []);
    }
  } else {
    if (imgContainer) imgContainer.classList.remove('hidden');
    if (candleSection) candleSection.classList.add('hidden');
    if (actionsBlock) actionsBlock.classList.remove('hidden');
    updateReportImage();
  }
}
window.switchToCandlestickView = () => switchViewMode('candlestick');
window.switchTab = switchTab;
window.selectTicker = selectTicker;


function initViewButtons() {
  const buttons = document.querySelectorAll('#view-buttons .view-btn');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const view = btn.dataset.view;
      switchViewMode(view);
    });
  });
}

// Timeframe buttons removed - all candles shown by default

function updateReportImage() {
  const img = document.getElementById('report-chart-img');
  const emptyBox = document.getElementById('report-img-empty');
  const downloadBtn = document.getElementById('btn-download-img');
  if (!img) return;

  const model = state.activeView === 'candlestick' ? 'ensemble' : state.activeView;
  const imgUrl = `/api/chart-image?symbol=${state.activeTicker}&model=${model}&t=${Date.now()}`;

  img.style.display = 'block';
  img.style.opacity = '0.5';
  if (emptyBox) emptyBox.classList.add('hidden');

  img.src = imgUrl;
  img.onload = () => {
    img.style.opacity = '1';
    img.style.display = 'block';
    if (emptyBox) emptyBox.classList.add('hidden');
  };
  img.onerror = () => {
    img.style.display = 'none';
    if (emptyBox) {
      emptyBox.classList.remove('hidden');
      const titleEl = document.getElementById('empty-img-title');
      const descEl = document.getElementById('empty-img-desc');
      if (titleEl) titleEl.textContent = `Chưa có ảnh báo cáo AI cho ${state.activeTicker}`;
      if (descEl) descEl.innerHTML = `Hệ thống chưa tạo ảnh báo cáo cho <strong>${state.activeTicker}</strong>. Vui lòng bấm nút bên dưới để chuyển sang <strong>Biểu Đồ Nến Nhật & RSI Tương Tác</strong>.`;
    }
  };

  if (downloadBtn) {
    downloadBtn.href = `/api/chart-image?symbol=${state.activeTicker}&model=${model}`;
    downloadBtn.download = `${state.activeTicker}_${model}_report.png`;
  }
}

// 4. Lightbox xem ảnh HD
function initLightbox() {
  const modal = document.getElementById('lightbox-modal');
  const modalImg = document.getElementById('lightbox-img');
  const closeBtn = document.getElementById('lightbox-close');
  const zoomBtn = document.getElementById('btn-zoom-img');
  const reportImg = document.getElementById('report-chart-img');

  const openLightbox = () => {
    if (!modal || !modalImg || !reportImg) return;
    modalImg.src = reportImg.src;
    modal.classList.add('active');
  };

  const closeLightbox = () => {
    if (modal) modal.classList.remove('active');
  };

  if (reportImg) reportImg.addEventListener('click', openLightbox);
  if (zoomBtn) zoomBtn.addEventListener('click', openLightbox);
  if (closeBtn) closeBtn.addEventListener('click', closeLightbox);
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) closeLightbox();
    });
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal && modal.classList.contains('active')) {
      closeLightbox();
    }
  });
}

// 5. Bộ lọc và tìm kiếm trên Bảng 100 Cổ Phiếu (Smart Screener)
function initScreenerFilters() {
  // Lọc theo Tín hiệu (MUA / BÁN / CHỜ)
  const signalButtons = document.querySelectorAll('#signal-filter-group .filter-chip');
  signalButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      signalButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.filterSignal = btn.dataset.filter;
      applyScreenerFilters();
    });
  });

  // Lọc theo Ngành
  const sectorButtons = document.querySelectorAll('#sector-filter-group .sector-chip');
  sectorButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      sectorButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.filterSector = btn.dataset.sector;
      applyScreenerFilters();
    });
  });

  initSearchAutocomplete();
}

// ==========================================================================
// AUTOCOMPLETE SEARCH DIRECTORY & ON-DEMAND EXTENSION
// ==========================================================================
const SEARCH_DIRECTORY = [
  // 100 Cổ phiếu cốt lõi S&P 500 (Có đầy đủ mô hình AI XGBoost, LSTM, FinBERT, Ensemble)
  { ticker: 'NVDA', name: 'NVIDIA Corporation', sector: 'Công nghệ', isCore: true },
  { ticker: 'AAPL', name: 'Apple Inc.', sector: 'Công nghệ', isCore: true },
  { ticker: 'GOOGL', name: 'Alphabet Inc.', sector: 'Truyền thông & Viễn thông', isCore: true },
  { ticker: 'GOOG', name: 'Alphabet Inc.', sector: 'Truyền thông & Viễn thông', isCore: true },
  { ticker: 'MSFT', name: 'Microsoft Corporation', sector: 'Công nghệ', isCore: true },
  { ticker: 'AMZN', name: 'Amazon.com, Inc.', sector: 'Hàng tiêu dùng không thiết yếu', isCore: true },
  { ticker: 'META', name: 'Meta Platforms, Inc.', sector: 'Truyền thông & Viễn thông', isCore: true },
  { ticker: 'AVGO', name: 'Broadcom Inc.', sector: 'Công nghệ', isCore: true },
  { ticker: 'TSLA', name: 'Tesla, Inc.', sector: 'Hàng tiêu dùng không thiết yếu', isCore: true },
  { ticker: 'BRK-B', name: 'Berkshire Hathaway Inc. New', sector: 'Tài chính - Ngân hàng', isCore: true },
  { ticker: 'LLY', name: 'Eli Lilly and Company', sector: 'Y tế & Chăm sóc sức khỏe', isCore: true },
  { ticker: 'MU', name: 'Micron Technology, Inc.', sector: 'Công nghệ', isCore: true },
  { ticker: 'WMT', name: 'Walmart Inc.', sector: 'Hàng tiêu dùng thiết yếu', isCore: true },
  { ticker: 'JPM', name: 'JP Morgan Chase & Co.', sector: 'Tài chính - Ngân hàng', isCore: true },
  { ticker: 'AMD', name: 'Advanced Micro Devices, Inc.', sector: 'Công nghệ', isCore: true },
  { ticker: 'V', name: 'Visa Inc.', sector: 'Tài chính - Ngân hàng', isCore: true },
  { ticker: 'XOM', name: 'ExxonMobil Holdings Corporation', sector: 'Năng lượng & Dầu khí', isCore: true },
  { ticker: 'JNJ', name: 'Johnson & Johnson', sector: 'Y tế & Chăm sóc sức khỏe', isCore: true },
  { ticker: 'MA', name: 'Mastercard Incorporated', sector: 'Tài chính - Ngân hàng', isCore: true },
  { ticker: 'ABBV', name: 'AbbVie Inc.', sector: 'Y tế & Chăm sóc sức khỏe', isCore: true },
  { ticker: 'PLTR', name: 'Palantir Technologies Inc.', sector: 'Công nghệ', isCore: true },
  { ticker: 'CVX', name: 'Chevron Corporation', sector: 'Năng lượng & Dầu khí', isCore: true },
  { ticker: 'INTC', name: 'Intel Corporation', sector: 'Công nghệ', isCore: true },
  { ticker: 'COST', name: 'Costco Wholesale Corporation', sector: 'Hàng tiêu dùng thiết yếu', isCore: true },
  { ticker: 'CSCO', name: 'Cisco Systems, Inc.', sector: 'Công nghệ', isCore: true },
  { ticker: 'KO', name: 'Coca-Cola Company (The)', sector: 'Hàng tiêu dùng thiết yếu', isCore: true },
  { ticker: 'ORCL', name: 'Oracle Corporation', sector: 'Công nghệ', isCore: true },
  { ticker: 'UNH', name: 'UnitedHealth Group Incorporated', sector: 'Y tế & Chăm sóc sức khỏe', isCore: true },
  { ticker: 'PG', name: 'Procter & Gamble Company (The)', sector: 'Hàng tiêu dùng thiết yếu', isCore: true },
  { ticker: 'MRK', name: 'Merck & Company, Inc.', sector: 'Y tế & Chăm sóc sức khỏe', isCore: true },
  { ticker: 'BAC', name: 'Bank of America Corporation', sector: 'Tài chính - Ngân hàng', isCore: true },
  { ticker: 'HD', name: 'Home Depot, Inc. (The)', sector: 'Hàng tiêu dùng không thiết yếu', isCore: true },
  { ticker: 'CAT', name: 'Caterpillar, Inc.', sector: 'Công nghiệp Chế tạo', isCore: true },
  { ticker: 'PM', name: 'Philip Morris International Inc', sector: 'Hàng tiêu dùng thiết yếu', isCore: true },
  { ticker: 'PANW', name: 'Palo Alto Networks, Inc.', sector: 'Công nghệ', isCore: true },
  { ticker: 'AMAT', name: 'Applied Materials, Inc.', sector: 'Công nghệ', isCore: true },
  { ticker: 'GE', name: 'GE Aerospace', sector: 'Công nghiệp Chế tạo', isCore: true },
  { ticker: 'DELL', name: 'Dell Technologies Inc.', sector: 'Công nghệ', isCore: true },
  { ticker: 'LRCX', name: 'Lam Research Corporation', sector: 'Công nghệ', isCore: true },
  { ticker: 'MS', name: 'Morgan Stanley', sector: 'Tài chính - Ngân hàng', isCore: true },
  { ticker: 'NFLX', name: 'Netflix, Inc.', sector: 'Truyền thông & Viễn thông', isCore: true },
  { ticker: 'TXN', name: 'Texas Instruments Incorporated', sector: 'Công nghệ', isCore: true },
  { ticker: 'RTX', name: 'RTX Corporation', sector: 'Công nghiệp Chế tạo', isCore: true },
  { ticker: 'GS', name: 'Goldman Sachs Group, Inc. (The)', sector: 'Tài chính - Ngân hàng', isCore: true },
  { ticker: 'TMO', name: 'Thermo Fisher Scientific Inc', sector: 'Y tế & Chăm sóc sức khỏe', isCore: true },
  { ticker: 'WFC', name: 'Wells Fargo & Company', sector: 'Tài chính - Ngân hàng', isCore: true },
  { ticker: 'IBM', name: 'International Business Machines', sector: 'Công nghệ', isCore: true },
  { ticker: 'AMGN', name: 'Amgen Inc.', sector: 'Y tế & Chăm sóc sức khỏe', isCore: true },
  { ticker: 'LIN', name: 'Linde plc', sector: 'Vật liệu Công nghiệp', isCore: true },
  { ticker: 'GEV', name: 'GE Vernova Inc.', sector: 'Công nghiệp Chế tạo', isCore: true },
  { ticker: 'ANET', name: 'Arista Networks, Inc.', sector: 'Công nghệ', isCore: true },
  { ticker: 'AXP', name: 'American Express Company', sector: 'Tài chính - Ngân hàng', isCore: true },
  { ticker: 'CRM', name: 'Salesforce, Inc.', sector: 'Công nghệ', isCore: true },
  { ticker: 'C', name: 'Citigroup, Inc.', sector: 'Tài chính - Ngân hàng', isCore: true },
  { ticker: 'GILD', name: 'Gilead Sciences, Inc.', sector: 'Y tế & Chăm sóc sức khỏe', isCore: true },
  { ticker: 'DIS', name: 'Walt Disney Company (The)', sector: 'Truyền thông & Viễn thông', isCore: true },
  { ticker: 'QCOM', name: 'QUALCOMM Incorporated', sector: 'Công nghệ', isCore: true },
  { ticker: 'VZ', name: 'Verizon Communications Inc.', sector: 'Truyền thông & Viễn thông', isCore: true },
  { ticker: 'COP', name: 'ConocoPhillips', sector: 'Năng lượng & Dầu khí', isCore: true },
  { ticker: 'PEP', name: 'Pepsico, Inc.', sector: 'Hàng tiêu dùng thiết yếu', isCore: true },
  { ticker: 'SNDK', name: 'Sandisk Corporation', sector: 'Công nghệ', isCore: true },
  { ticker: 'ABT', name: 'Abbott Laboratories', sector: 'Y tế & Chăm sóc sức khỏe', isCore: true },
  { ticker: 'MCD', name: 'McDonald\'s Corporation', sector: 'Hàng tiêu dùng không thiết yếu', isCore: true },
  { ticker: 'UNP', name: 'Union Pacific Corporation', sector: 'Công nghiệp Chế tạo', isCore: true },
  { ticker: 'DE', name: 'Deere & Company', sector: 'Công nghiệp Chế tạo', isCore: true },
  { ticker: 'BLK', name: 'BlackRock, Inc.', sector: 'Tài chính - Ngân hàng', isCore: true },
  { ticker: 'PFE', name: 'Pfizer, Inc.', sector: 'Y tế & Chăm sóc sức khỏe', isCore: true },
  { ticker: 'NEE', name: 'NextEra Energy, Inc.', sector: 'Năng lượng & Tiện ích', isCore: true },
  { ticker: 'NOW', name: 'ServiceNow, Inc.', sector: 'Công nghệ', isCore: true },
  { ticker: 'ISRG', name: 'Intuitive Surgical, Inc.', sector: 'Y tế & Chăm sóc sức khỏe', isCore: true },
  { ticker: 'TMUS', name: 'T-Mobile US, Inc.', sector: 'Truyền thông & Viễn thông', isCore: true },
  { ticker: 'DHR', name: 'Danaher Corporation', sector: 'Y tế & Chăm sóc sức khỏe', isCore: true },
  { ticker: 'SCHW', name: 'Charles Schwab Corporation (The', sector: 'Tài chính - Ngân hàng', isCore: true },
  { ticker: 'ACN', name: 'Accenture plc', sector: 'Công nghệ', isCore: true },
  { ticker: 'BA', name: 'Boeing Company (The)', sector: 'Công nghiệp Chế tạo', isCore: true },
  { ticker: 'T', name: 'AT&T Inc.', sector: 'Truyền thông & Viễn thông', isCore: true },
  { ticker: 'LMT', name: 'Lockheed Martin Corporation', sector: 'Công nghiệp Chế tạo', isCore: true },
  { ticker: 'UBER', name: 'Uber Technologies, Inc.', sector: 'Công nghệ', isCore: true },
  { ticker: 'BMY', name: 'Bristol-Myers Squibb Company', sector: 'Y tế & Chăm sóc sức khỏe', isCore: true },
  { ticker: 'MO', name: 'Altria Group, Inc.', sector: 'Hàng tiêu dùng thiết yếu', isCore: true },
  { ticker: 'COF', name: 'Capital One Financial Corporati', sector: 'Tài chính - Ngân hàng', isCore: true },
  { ticker: 'MDT', name: 'Medtronic plc.', sector: 'Y tế & Chăm sóc sức khỏe', isCore: true },
  { ticker: 'CVS', name: 'CVS Health Corporation', sector: 'Y tế & Chăm sóc sức khỏe', isCore: true },
  { ticker: 'BKNG', name: 'Booking Holdings Inc. Common St', sector: 'Hàng tiêu dùng không thiết yếu', isCore: true },
  { ticker: 'SBUX', name: 'Starbucks Corporation', sector: 'Hàng tiêu dùng không thiết yếu', isCore: true },
  { ticker: 'LOW', name: 'Lowe\'s Companies, Inc.', sector: 'Hàng tiêu dùng không thiết yếu', isCore: true },
  { ticker: 'SO', name: 'Southern Company (The)', sector: 'Năng lượng & Tiện ích', isCore: true },
  { ticker: 'ADBE', name: 'Adobe Inc.', sector: 'Công nghệ', isCore: true },
  { ticker: 'BNY', name: 'The Bank of New York Mellon Cor', sector: 'Tài chính - Ngân hàng', isCore: true },
  { ticker: 'DUK', name: 'Duke Energy Corporation (Holdin', sector: 'Năng lượng & Tiện ích', isCore: true },
  { ticker: 'GD', name: 'General Dynamics Corporation', sector: 'Công nghiệp Chế tạo', isCore: true },
  { ticker: 'USB', name: 'U.S. Bancorp', sector: 'Tài chính - Ngân hàng', isCore: true },
  { ticker: 'INTU', name: 'Intuit Inc.', sector: 'Công nghệ', isCore: true },
  { ticker: 'EMR', name: 'Emerson Electric Company', sector: 'Công nghiệp Chế tạo', isCore: true },
  { ticker: 'MMM', name: '3M Company', sector: 'Công nghiệp Chế tạo', isCore: true },
  { ticker: 'AMT', name: 'American Tower Corporation (REI', sector: 'Bất động sản', isCore: true },
  { ticker: 'UPS', name: 'United Parcel Service, Inc.', sector: 'Công nghiệp Chế tạo', isCore: true },
  { ticker: 'MDLZ', name: 'Mondelez International, Inc.', sector: 'Hàng tiêu dùng thiết yếu', isCore: true },
  { ticker: 'CMCSA', name: 'Comcast Corporation', sector: 'Truyền thông & Viễn thông', isCore: true },
  { ticker: 'FDX', name: 'FedEx Corporation', sector: 'Công nghiệp Chế tạo', isCore: true },

  // Cổ phiếu mở rộng ngoài Top 100 (Hỗ trợ On-Demand AI & Biểu đồ Nến Live)
  { ticker: 'PYPL', name: 'PayPal Holdings Inc.', sector: 'Tài chính & Fintech', isCore: false },
  { ticker: 'COIN', name: 'Coinbase Global Inc.', sector: 'Tài chính & Crypto', isCore: false },
  { ticker: 'BABA', name: 'Alibaba Group Holding', sector: 'Thương mại điện tử', isCore: false },
  { ticker: 'TGT', name: 'Target Corporation', sector: 'Hàng tiêu dùng', isCore: false },
  { ticker: 'F', name: 'Ford Motor Co.', sector: 'Hàng tiêu dùng & Ô tô', isCore: false },
  { ticker: 'GM', name: 'General Motors Co.', sector: 'Hàng tiêu dùng & Ô tô', isCore: false },
  { ticker: 'NKE', name: 'Nike Inc.', sector: 'Hàng tiêu dùng & Thời trang', isCore: false },
];

function highlightMatch(text, query) {
  if (!query) return text;
  const idx = text.toLowerCase().indexOf(query.toLowerCase());
  if (idx === -1) return text;
  const before = text.substring(0, idx);
  const match = text.substring(idx, idx + query.length);
  const after = text.substring(idx + query.length);
  return `${before}<mark class="search-match">${match}</mark>${after}`;
}

let activeSuggestionIndex = -1;

function setupAutocompleteFlyout(config) {
  const {
    inputId,
    dropdownId,
    clearBtnId,
    submitBtnId,
    wrapperId,
    onSelect,
    onInput
  } = config;

  const searchInput = document.getElementById(inputId);
  const dropdown = document.getElementById(dropdownId);
  const clearBtn = clearBtnId ? document.getElementById(clearBtnId) : null;
  const submitBtn = submitBtnId ? document.getElementById(submitBtnId) : null;
  if (!searchInput || !dropdown) return;

  let activeIndex = -1;

  function renderSuggestions(query) {
    activeIndex = -1;
    const q = query.trim().toLowerCase();
    if (!q) {
      dropdown.classList.add('hidden');
      dropdown.innerHTML = '';
      return;
    }

    const prefixTicker = [];
    const prefixName = [];
    const substringMatch = [];

    SEARCH_DIRECTORY.forEach(item => {
      const t = item.ticker.toLowerCase();
      const n = item.name.toLowerCase();
      const s = (item.sector || '').toLowerCase();
      if (t.startsWith(q)) {
        prefixTicker.push(item);
      } else if (n.startsWith(q)) {
        prefixName.push(item);
      } else if (t.includes(q) || n.includes(q) || s.includes(q)) {
        substringMatch.push(item);
      }
    });

    const matches = [...prefixTicker, ...prefixName, ...substringMatch].slice(0, 8);

    if (matches.length === 0) {
      dropdown.innerHTML = `
        <div class="suggestion-header">
          <span class="sugg-header-title">GỢI Ý TÌM KIẾM</span>
        </div>
        <div class="sugg-empty-state">
          <span class="sugg-empty-icon">🔍</span>
          <span>Không tìm thấy mã hoặc công ty nào khớp với "<strong>${query}</strong>"</span>
        </div>
      `;
      dropdown.classList.remove('hidden');
      return;
    }

    dropdown.innerHTML = `
      <div class="suggestion-header">
        <span class="sugg-header-title">GỢI Ý CỔ PHIẾU (${matches.length})</span>
        <span class="sugg-header-hint"><kbd>↵</kbd> hoặc Click để chọn</span>
      </div>
      <div class="suggestion-list">
        ${matches.map((s, idx) => `
          <div class="suggestion-item" data-index="${idx}" data-ticker="${s.ticker}">
            <div class="sugg-left">
              <span class="sugg-ticker">${highlightMatch(s.ticker, q)}</span>
              <div class="sugg-details">
                <span class="sugg-name">${highlightMatch(s.name, q)}</span>
                <span class="sugg-sector-tag">${s.sector}</span>
              </div>
            </div>
            <div class="sugg-right">
              <span class="sugg-badge ${s.isCore ? 'core' : 'ondemand'}">
                ${s.isCore ? '⚡ Top 100 S&P (AI)' : '📈 Biểu đồ Live'}
              </span>
              <span class="sugg-arrow">→</span>
            </div>
          </div>
        `).join('')}
      </div>
    `;

    dropdown.classList.remove('hidden');

    dropdown.querySelectorAll('.suggestion-item').forEach(el => {
      el.addEventListener('click', (e) => {
        e.stopPropagation();
        const ticker = el.dataset.ticker;
        dropdown.classList.add('hidden');
        searchInput.value = ticker;
        if (clearBtn) clearBtn.style.display = 'block';
        onSelect(ticker);
      });
    });
  }

  function updateActiveSuggestion(items) {
    items.forEach((item, idx) => {
      if (idx === activeIndex) {
        item.classList.add('active');
        item.scrollIntoView({ block: 'nearest' });
      } else {
        item.classList.remove('active');
      }
    });
  }

  searchInput.addEventListener('input', (e) => {
    const val = e.target.value;
    if (clearBtn) clearBtn.style.display = val ? 'block' : 'none';
    renderSuggestions(val);
    if (onInput) onInput(val);
  });

  searchInput.addEventListener('focus', () => {
    if (searchInput.value.trim()) {
      renderSuggestions(searchInput.value);
    }
  });

  searchInput.addEventListener('keydown', (e) => {
    const items = dropdown.querySelectorAll('.suggestion-item');
    if (dropdown.classList.contains('hidden') || items.length === 0) {
      if (e.key === 'Enter') {
        e.preventDefault();
        const val = searchInput.value.trim().toUpperCase();
        if (val) {
          dropdown.classList.add('hidden');
          onSelect(val);
        }
      }
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      activeIndex = (activeIndex + 1) % items.length;
      updateActiveSuggestion(items);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      activeIndex = (activeIndex - 1 + items.length) % items.length;
      updateActiveSuggestion(items);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      let selectedTicker = '';
      if (activeIndex >= 0 && items[activeIndex]) {
        selectedTicker = items[activeIndex].dataset.ticker;
      } else if (items.length > 0) {
        selectedTicker = items[0].dataset.ticker;
      }
      if (selectedTicker) {
        dropdown.classList.add('hidden');
        searchInput.value = selectedTicker;
        if (clearBtn) clearBtn.style.display = 'block';
        onSelect(selectedTicker);
      }
    } else if (e.key === 'Escape') {
      dropdown.classList.add('hidden');
    }
  });

  if (clearBtn) {
    clearBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      searchInput.value = '';
      clearBtn.style.display = 'none';
      dropdown.classList.add('hidden');
      dropdown.innerHTML = '';
      searchInput.focus();
      if (onInput) onInput('');
    });
  }

  if (submitBtn) {
    submitBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const val = searchInput.value.trim().toUpperCase();
      if (val) {
        dropdown.classList.add('hidden');
        onSelect(val);
      }
    });
  }

  document.addEventListener('click', (e) => {
    const wrap = wrapperId ? document.getElementById(wrapperId) : null;
    if (wrap) {
      if (!wrap.contains(e.target)) dropdown.classList.add('hidden');
    } else {
      if (!searchInput.contains(e.target) && !dropdown.contains(e.target)) {
        dropdown.classList.add('hidden');
      }
    }
  });
}

function initSearchAutocomplete() {
  // 1. Header Search Bar (Yahoo Finance Pill Search)
  setupAutocompleteFlyout({
    inputId: 'header-search-input',
    dropdownId: 'header-search-dropdown',
    clearBtnId: 'header-search-clear',
    submitBtnId: 'header-search-submit',
    wrapperId: 'header-search-wrap',
    onSelect: (ticker) => {
      selectTicker(ticker, true);
    }
  });

  // 2. Screener Tab Search Bar (Table Filter + Quick Pick)
  setupAutocompleteFlyout({
    inputId: 'screener-search-input',
    dropdownId: 'search-suggestions-dropdown',
    clearBtnId: 'search-clear-btn',
    submitBtnId: null,
    wrapperId: 'search-box-wrap',
    onSelect: (ticker) => {
      selectSearchSuggestion(ticker);
    },
    onInput: (val) => {
      state.searchKeyword = val.trim().toLowerCase();
      applyScreenerFilters();
    }
  });

  // 3. Sidebar Quote Lookup
  const sidebarInput = document.getElementById('sidebar-quote-lookup');
  if (sidebarInput) {
    sidebarInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        const sym = sidebarInput.value.trim().toUpperCase();
        if (sym) selectTicker(sym, true);
      }
    });
  }
}

function selectSearchSuggestion(ticker) {
  const searchInput = document.getElementById('screener-search-input');
  const dropdown = document.getElementById('search-suggestions-dropdown');
  const clearBtn = document.getElementById('search-clear-btn');

  if (searchInput) searchInput.value = ticker;
  state.searchKeyword = ticker.toLowerCase();
  if (clearBtn) clearBtn.style.display = 'block';
  if (dropdown) {
    dropdown.classList.add('hidden');
    dropdown.innerHTML = '';
  }

  const hasPred = (state.marketData && state.marketData.predictions || []).some(p => p.ticker.toUpperCase() === ticker.toUpperCase());
  if (hasPred) {
    selectTicker(ticker, false);
    applyScreenerFilters();
  } else {
    runOnDemandAnalysis(ticker);
  }
}
window.selectSearchSuggestion = selectSearchSuggestion;


// 6. Nút làm mới dữ liệu & Sự kiện Modal
function initEventListeners() {
  const btnRefresh = document.getElementById('btn-refresh');
  if (btnRefresh) {
    btnRefresh.addEventListener('click', () => {
      btnRefresh.classList.add('rotating');
      loadAllData().finally(() => {
        setTimeout(() => btnRefresh.classList.remove('rotating'), 500);
      });
    });
  }
  initAIModalListeners();
}

// 7. Tự động làm mới chu kỳ 5 giây
function initAutoRefresh() {
  setInterval(() => {
    loadAllData();
  }, 5000); // 5 giây
}

// ==========================================================================
// MINIMALIST ON-DEMAND AI PIPELINE (THANH TIẾN TRÌNH TỐI GIẢN)
// ==========================================================================
let isAnalyzingOnDemand = false;

function showToast(message) {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = 'toast-msg';
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.2s ease';
    setTimeout(() => toast.remove(), 250);
  }, 2500);
}

function initAIModalListeners() {
  const modal = document.getElementById('ai-modal-overlay') || document.getElementById('ai-loading-overlay');
  const closeBtn = document.getElementById('ai-modal-close-btn');

  if (closeBtn) {
    closeBtn.addEventListener('click', () => {
      if (modal) modal.classList.remove('active');
    });
  }

  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.classList.remove('active');
      }
    });
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal && modal.classList.contains('active')) {
      modal.classList.remove('active');
    }
  });
}

function setModalProgress(pct, statusText) {
  const bar = document.getElementById('ai-progress-bar-fill');
  const label = document.getElementById('ai-progress-pct-val');
  const status = document.getElementById('ai-progress-status-text');
  if (bar) { bar.classList.toggle('is-pending', pct !== 100); bar.style.width = pct === 100 ? '100%' : ''; }
  if (label) label.textContent = pct === 100 ? 'Hoàn tất' : 'Đang xử lý';
  if (status) status.textContent = statusText || 'Đang xử lý dữ liệu…';
}

async function runOnDemandAnalysis(ticker) {
  ticker = (ticker || state.activeTicker || 'META').toUpperCase().trim();
  if (isAnalyzingOnDemand) return;
  isAnalyzingOnDemand = true;

  const modal = document.getElementById('ai-modal-overlay') || document.getElementById('ai-loading-overlay');
  const symEl = document.getElementById('ai-modal-ticker-symbol');

  if (symEl) symEl.textContent = ticker;
  setModalProgress(null, 'Yêu cầu đang chờ hệ thống xử lý…');

  if (modal) modal.classList.add('active');

  let data;
  try {
    data = await fetch(`/api/analyze?symbol=${encodeURIComponent(ticker)}`).then(r => r.json());
    while (data && ['queued', 'processing'].includes(data.status)) {
      setModalProgress(null, data.message);
      await new Promise(resolve => setTimeout(resolve, 1500));
      data = await fetch(`/api/analyze?job_id=${encodeURIComponent(data.job_id)}`).then(r => r.json());
    }
  } catch (error) {
    data = {status: 'error', message: 'Không thể lấy trạng thái xử lý. Vui lòng thử lại.'};
  }

  if (data && data.status === 'success' && data.prediction) {
    const pred = data.prediction;
    setModalProgress(100, 'Hoàn tất phân tích!');

    // Hợp nhất vào state.marketData.predictions
    if (!state.marketData) state.marketData = { predictions: [] };
    if (!state.marketData.predictions) state.marketData.predictions = [];
    delete state.marketData.status;
    delete state.marketData.message;
    const idx = state.marketData.predictions.findIndex(p => p.ticker === pred.ticker);
    if (idx >= 0) {
      state.marketData.predictions[idx] = pred;
    } else {
      state.marketData.predictions.push(pred);
    }

    // Cập nhật danh bạ tìm kiếm
    let dirItem = SEARCH_DIRECTORY.find(i => i.ticker.toUpperCase() === pred.ticker.toUpperCase());
    if (dirItem) {
      dirItem.isAnalyzed = true;
    } else {
      SEARCH_DIRECTORY.push({
        ticker: pred.ticker,
        name: pred.name,
        sector: pred.sector,
        isCore: false,
        isAnalyzed: true
      });
    }

    applyScreenerFilters();
    selectTicker(pred.ticker, false);

    showToast(`Đã phân tích xong ${pred.ticker} (${pred.pp4_tong_hop})`);

    setTimeout(() => {
      if (modal) modal.classList.remove('active');
      isAnalyzingOnDemand = false;
    }, 450);
  } else {
    setModalProgress(null, 'Chưa có kết quả phân tích');
    showToast(data?.message || `Không thể phân tích mã ${ticker}`);
    setTimeout(() => {
      if (modal) modal.classList.remove('active');
      isAnalyzingOnDemand = false;
    }, 1000);
  }
}
window.runOnDemandAnalysis = runOnDemandAnalysis;

function syncSearchDirectoryWithPredictions(predictions) {
  if (!predictions || !Array.isArray(predictions)) return;
  predictions.forEach(p => {
    let item = SEARCH_DIRECTORY.find(i => i.ticker.toUpperCase() === p.ticker.toUpperCase());
    if (item) {
      item.isCore = true;
      if (p.name && !item.name) item.name = p.name;
      if (p.sector) item.sector = p.sector;
    } else {
      SEARCH_DIRECTORY.push({
        ticker: p.ticker,
        name: p.name || p.ticker,
        sector: p.sector || 'Công nghệ',
        isCore: true
      });
    }
  });
}

// ==========================================================================
// DATA FETCHING & SYNCHRONIZATION
// ==========================================================================
let activeDataLoads = 0;
let loadingAllData = false;
async function loadAllData() {
  if (loadingAllData) return;
  loadingAllData = true;
  activeDataLoads++;
  document.getElementById('data-loading')?.classList.remove('hidden');
  try {
  const syncEl = document.getElementById('sync-timestamp');
  if (syncEl) {
    const now = new Date();
    syncEl.textContent = now.toLocaleTimeString('vi-VN') + ' ' + now.toLocaleDateString('vi-VN');
  }

  const chartPromise = loadTickerChart(state.activeTicker);
  const qualityPromise = loadQualityData();
  const evalPromise = loadEvaluationData();

  try {
    const marketRes = await fetch('/api/market').then(r => r.json());
    state.marketData = marketRes;
    syncSearchDirectoryWithPredictions(marketRes?.predictions);
    renderHeaderMarketOverview(marketRes);
    renderTopPicksBanner(marketRes);
    renderScreenerTable(marketRes);
    updateTickerPredictions(state.activeTicker);
  } catch (err) {
    console.error('Lỗi nạp dữ liệu Market:', err);
  }

  await Promise.allSettled([chartPromise, qualityPromise, evalPromise]);
  } finally {
    loadingAllData = false;
    activeDataLoads--;
    if (activeDataLoads === 0) document.getElementById('data-loading')?.classList.add('hidden');
  }
}

// Cập nhật Header Market Overview
function renderHeaderMarketOverview(market) {
  if (!market) return;

  const moodEl = document.getElementById('header-market-mood');
  if (moodEl) {
    moodEl.textContent = market.market_mood || 'SIDEWAY (ĐI NGANG)';
    const isBull = (market.buy_count || 0) > (market.sell_count || 0);
    const isBear = (market.sell_count || 0) > (market.buy_count || 0);
    moodEl.style.color = isBull ? 'var(--accent-green)' : (isBear ? 'var(--accent-red)' : 'var(--accent-yellow)');
  }

  const buyCountEl = document.getElementById('summary-buy-count');
  const holdCountEl = document.getElementById('summary-hold-count');
  const sellCountEl = document.getElementById('summary-sell-count');

  if (buyCountEl) buyCountEl.textContent = `${market.buy_count || 0} MUA`;
  if (holdCountEl) holdCountEl.textContent = `${market.hold_count || 0} CHỜ`;
  if (sellCountEl) sellCountEl.textContent = `${market.sell_count || 0} BÁN`;

  updateHeaderActiveTicker();
}

function updateHeaderActiveTicker() {
  const activeEl = document.getElementById('header-active-ticker');
  if (activeEl) activeEl.textContent = state.activeTicker;

  const signalEl = document.getElementById('header-active-signal');
  if (signalEl) {
    if (state.marketData && state.marketData.predictions) {
      const pred = state.marketData.predictions.find(p => p.ticker === state.activeTicker);
      if (pred) {
        const action = pred.pp4_tong_hop || 'THEO DÕI';
        signalEl.textContent = `${action} (${pred.prob_ensemble == null ? 'Chưa có kết quả' : `${pred.prob_ensemble}%`})`;
        signalEl.className = `ticker-signal-chip ${action.includes('MUA') ? 'chip-green' : (action.includes('BÁN') ? 'chip-red' : 'chip-yellow')}`;
        return;
      }
    }
    signalEl.textContent = 'THEO DÕI (LIVE DATA)';
    signalEl.className = 'ticker-signal-chip chip-yellow';
  }
}

// ==========================================================================
// TAB 1: BẢNG 100 CỔ PHIẾU THÔNG MINH (SMART MARKET SCREENER)
// ==========================================================================

// Top Opportunity Spotlight Banner (Top 3 mã MUA)
function renderTopPicksBanner(market) {
  const container = document.getElementById('top-picks-container');
  if (!container || !market || !market.predictions) return;

  if (market.status === 'unavailable') {
    container.textContent = market.message || 'Chưa có dữ liệu dự báo.';
    return;
  }

  const buyPicks = market.predictions
    .filter(p => (p.pp4_tong_hop || '').includes('MUA') || p.prob_ensemble >= 55)
    .sort((a, b) => b.prob_ensemble - a.prob_ensemble)
    .slice(0, 3);

  if (buyPicks.length === 0) {
    container.innerHTML = `
      <div class="empty-top-picks">
        <span>Chưa có tín hiệu mua từ mô hình Ensemble đã huấn luyện.</span>
      </div>
    `;
    return;
  }

  container.innerHTML = buyPicks.map((pick, idx) => {
    const isStrong = pick.pp4_tong_hop.includes('MẠNH') || pick.prob_ensemble >= 58;
    return `
      <div class="top-pick-card" onclick="selectTicker('${pick.ticker}', true)">
        <div class="pick-badge-rank">#${idx + 1} TIỀM NĂNG NHẤT</div>
        <div class="pick-top-row">
          <span class="pick-ticker">${pick.ticker}</span>
          <span class="pick-action ${isStrong ? 'strong' : ''}">${pick.pp4_tong_hop}</span>
        </div>
        <div class="pick-name">${pick.name}</div>
        <div class="pick-metrics-row">
          <div>
            <span class="m-lbl">Giá Vào (Entry):</span>
            <strong class="m-val">$${pick.current_price.toFixed(2)}</strong>
          </div>
          <div>
            <span class="m-lbl">Mục Tiêu (TP):</span>
            <strong class="m-val text-success">${pick.plan_target == null ? 'Chưa có kết quả' : `$${pick.plan_target.toFixed(2)} (${pick.plan_tp_pct.toFixed(2)}%)`}</strong>
          </div>
          <div>
            <span class="m-lbl">Xác Suất Tăng T+1:</span>
            <strong class="m-val text-purple">${pick.prob_ensemble}%</strong>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

const SECTOR_ALIASES = {
  'tech': ['công nghệ', 'technology', 'tech'],
  'comm': ['truyền thông', 'viễn thông', 'communication', 'media'],
  'finance': ['tài chính', 'financials', 'financial services', 'finance', 'ngân hàng', 'bank'],
  'health': ['y tế', 'healthcare', 'health', 'chăm sóc sức khỏe', 'dược phẩm'],
  'consumer': ['tiêu dùng', 'consumer', 'hàng tiêu dùng', 'staples', 'discretionary', 'đồ uống', 'bán lẻ', 'thời trang', 'thiết yếu'],
  'energy': ['năng lượng', 'energy', 'tiện ích', 'utilities', 'dầu khí', 'điện'],
  'industry': ['công nghiệp', 'industrials', 'industry', 'vật liệu', 'materials', 'chế tạo', 'hàng không'],
  'realestate': ['bất động sản', 'real estate', 'reit']
};

function matchStockSector(p, filterKey) {
  if (!filterKey || filterKey === 'all') return true;
  const targetAliases = SECTOR_ALIASES[filterKey] || [filterKey.toLowerCase()];
  const stockSec = `${p.sector || ''} ${p.sector_en || ''}`.toLowerCase();
  return targetAliases.some(alias => stockSec.includes(alias));
}

function updateFilterChipCounts(preds) {
  if (!preds || !preds.length) return;

  const buyCount = preds.filter(p => (p.pp4_tong_hop || '').includes('MUA') || p.prob_ensemble >= 55).length;
  const sellCount = preds.filter(p => (p.pp4_tong_hop || '').includes('BÁN') || (p.prob_ensemble != null && p.prob_ensemble <= 45)).length;
  const holdCount = preds.length - buyCount - sellCount;

  const chipAll = document.getElementById('filter-chip-all');
  const chipBuy = document.getElementById('filter-chip-buy');
  const chipHold = document.getElementById('filter-chip-hold');
  const chipSell = document.getElementById('filter-chip-sell');

  if (chipAll) chipAll.textContent = `Tất Cả (${preds.length})`;
  if (chipBuy) chipBuy.textContent = `🟢 Khuyến Nghị MUA (${buyCount})`;
  if (chipHold) chipHold.textContent = `🟡 Theo Dõi (${holdCount})`;
  if (chipSell) chipSell.textContent = `🔴 Cảnh Báo BÁN (${sellCount})`;

  const sectorMeta = {
    'tech': { icon: '💻', name: 'Công Nghệ' },
    'comm': { icon: '📡', name: 'Truyền Thông' },
    'finance': { icon: '🏦', name: 'Tài Chính' },
    'health': { icon: '🏥', name: 'Y Tế' },
    'consumer': { icon: '🛒', name: 'Tiêu Dùng' },
    'energy': { icon: '⚡', name: 'Năng Lượng & Tiện Ích' },
    'industry': { icon: '🏭', name: 'Công Nghiệp' },
    'realestate': { icon: '🏢', name: 'Bất Động Sản' }
  };

  document.querySelectorAll('#sector-filter-group .sector-chip').forEach(btn => {
    const sKey = btn.dataset.sector;
    if (sKey === 'all') {
      btn.textContent = `Tất Cả Ngành (${preds.length})`;
    } else {
      const count = preds.filter(p => matchStockSector(p, sKey)).length;
      const meta = sectorMeta[sKey] || { icon: '', name: sKey };
      btn.textContent = `${meta.icon} ${meta.name} (${count})`;
    }
  });
}

function resetScreenerFilters() {
  state.filterSignal = 'all';
  state.filterSector = 'all';
  state.searchKeyword = '';

  const searchInput = document.getElementById('screener-search-input');
  if (searchInput) searchInput.value = '';
  const clearBtn = document.getElementById('search-clear-btn');
  if (clearBtn) clearBtn.style.display = 'none';

  document.querySelectorAll('#signal-filter-group .filter-chip').forEach(b => {
    if (b.dataset.filter === 'all') b.classList.add('active');
    else b.classList.remove('active');
  });
  document.querySelectorAll('#sector-filter-group .sector-chip').forEach(b => {
    if (b.dataset.sector === 'all') b.classList.add('active');
    else b.classList.remove('active');
  });

  applyScreenerFilters();
}
window.resetScreenerFilters = resetScreenerFilters;

// Render toàn bộ dữ liệu bảng Screener
function renderScreenerTable(market) {
  if (market && market.predictions) {
    updateFilterChipCounts(market.predictions);
  }
  applyScreenerFilters();
}

function applyScreenerFilters() {
  const tbody = document.getElementById('screener-table-body');
  const countEl = document.getElementById('screener-records-count');
  if (!tbody || !state.marketData || !state.marketData.predictions) return;

  const preds = state.marketData.predictions;

  const filtered = preds.filter(p => {
    // 1. Lọc theo Signal
    if (state.filterSignal === 'buy') {
      const isBuy = (p.pp4_tong_hop || '').includes('MUA') || p.prob_ensemble >= 55;
      if (!isBuy) return false;
    } else if (state.filterSignal === 'hold') {
      const isBuy = (p.pp4_tong_hop || '').includes('MUA') || p.prob_ensemble >= 55;
      const isSell = (p.pp4_tong_hop || '').includes('BÁN') || (p.prob_ensemble != null && p.prob_ensemble <= 45);
      if (isBuy || isSell) return false;
    } else if (state.filterSignal === 'sell') {
      const isSell = (p.pp4_tong_hop || '').includes('BÁN') || (p.prob_ensemble != null && p.prob_ensemble <= 45);
      if (!isSell) return false;
    }

    // 2. Lọc theo Sector
    if (!matchStockSector(p, state.filterSector)) {
      return false;
    }

    // 3. Tìm kiếm theo keyword
    if (state.searchKeyword) {
      const kw = state.searchKeyword;
      const matchTicker = p.ticker.toLowerCase().includes(kw);
      const matchName = (p.name || '').toLowerCase().includes(kw);
      const matchSector = `${p.sector || ''} ${p.sector_en || ''}`.toLowerCase().includes(kw);
      if (!matchTicker && !matchName && !matchSector) return false;
    }

    return true;
  });

  if (countEl) {
    countEl.textContent = `Hiển thị ${filtered.length}/${preds.length} cổ phiếu`;
  }

  if (filtered.length === 0) {
    if (state.searchKeyword) {
      const kw = state.searchKeyword.toLowerCase().trim();
      const rawKw = state.searchKeyword.toUpperCase().trim();
      const outsideMatch = SEARCH_DIRECTORY.find(item =>
        item.ticker.toLowerCase() === kw ||
        item.ticker.toLowerCase().startsWith(kw) ||
        item.name.toLowerCase().includes(kw)
      );

      if (outsideMatch && !outsideMatch.isCore) {
        tbody.innerHTML = `
          <tr>
            <td colspan="9" class="table-empty">
              <div class="outside-stock-simple">
                <span>Mã <strong>${outsideMatch.ticker}</strong> (${outsideMatch.name}) chưa phân tích AI.</span>
                <button class="btn-analyze-now" onclick="runOnDemandAnalysis('${outsideMatch.ticker}')">
                  ⚡ Phân Tích ${outsideMatch.ticker}
                </button>
                <button class="btn-analyze-now" style="background:#334155; border-color:#475569;" onclick="selectTicker('${outsideMatch.ticker}', true)">
                  Biểu Đồ Nến
                </button>
                <button class="btn-analyze-now" style="background:transparent; border-color:#475569; color:#94a3b8;" onclick="resetScreenerFilters()">
                  ✕ Xóa
                </button>
              </div>
            </td>
          </tr>
        `;
        return;
      } else if (/^[A-Z]{1,6}$/.test(rawKw)) {
        tbody.innerHTML = `
          <tr>
            <td colspan="9" class="table-empty">
              <div class="outside-stock-simple">
                <span>Mã <strong>${rawKw}</strong> chưa có dữ liệu AI.</span>
                <button class="btn-analyze-now" onclick="runOnDemandAnalysis('${rawKw}')">
                  ⚡ Phân Tích ${rawKw}
                </button>
                <button class="btn-analyze-now" style="background:transparent; border-color:#475569; color:#94a3b8;" onclick="resetScreenerFilters()">
                  ✕ Xóa
                </button>
              </div>
            </td>
          </tr>
        `;
        return;
      }
    }

    let helpMsg = 'Không tìm thấy mã cổ phiếu nào phù hợp với bộ lọc hiện tại.';
    if (state.searchKeyword) {
      helpMsg = `Không tìm thấy mã hoặc doanh nghiệp nào khớp với từ khóa "<strong>${state.searchKeyword}</strong>".`;
    } else if (state.filterSignal === 'buy') {
      helpMsg = 'Hiện tại hệ thống không có khuyến nghị MUA cho phiên này do thị trường đang trong pha điều chỉnh / đi ngang. Bạn có thể xem các mã ở nhóm "Theo Dõi" hoặc bấm "Tất Cả".';
    } else if (state.filterSignal === 'sell') {
      helpMsg = 'Không có mã cổ phiếu nào có tín hiệu BÁN trong danh mục đã chọn.';
    }

    tbody.innerHTML = `
      <tr>
        <td colspan="9" class="table-empty">
          <div class="empty-state-box">
            <span class="empty-state-icon">🔍</span>
            <p>${helpMsg}</p>
            <button class="btn-reset-filters" onclick="resetScreenerFilters()">
              ✕ Xóa Tìm Kiếm & Hiển Thị Toàn Bộ Cổ Phiếu
            </button>
          </div>
        </td>
      </tr>
    `;
    return;
  }

  function getModelPill(prob, type) {
    if (prob === null || prob === undefined || !Number.isFinite(Number(prob))) {
      return '<span class="model-pill neutral">Không khả dụng</span>';
    }
    const pVal = Number(prob);
    let label = 'Trung tính';
    let cls = 'neutral';
    if (type === 'sentiment') {
      if (pVal >= 55) { label = 'Tích cực'; cls = 'bullish'; }
      else if (pVal <= 45) { label = 'Tiêu cực'; cls = 'bearish'; }
      else { label = 'Trung tính'; cls = 'neutral'; }
    } else {
      if (pVal >= 52) { label = 'Tăng'; cls = 'bullish'; }
      else if (pVal <= 48) { label = 'Giảm'; cls = 'bearish'; }
      else { label = 'Đi ngang'; cls = 'neutral'; }
    }
    return `<span class="model-pill ${cls}">${label} <span class="model-pct">${pVal.toFixed(1)}%</span></span>`;
  }

  tbody.innerHTML = filtered.map(p => {
    const isSelected = (p.ticker === state.activeTicker);
    const action = p.pp4_tong_hop || 'ĐỨNG NGOÀI';
    const isBuy = action.includes('MUA');
    const isSell = action.includes('BÁN');

    let badgeClass = 'badge-hold';
    if (action.includes('MUA MẠNH')) badgeClass = 'badge-strong-buy';
    else if (isBuy) badgeClass = 'badge-buy';
    else if (isSell) badgeClass = 'badge-sell';

    const onDemandTag = p.on_demand ? `<span class="sugg-badge ondemand" style="font-size: 9px; padding: 2px 6px; margin-left: 6px;">⚡ On-Demand</span>` : '';

    return `
      <tr class="screener-row ${isSelected ? 'selected' : ''}" data-ticker="${p.ticker}" onclick="selectTicker('${p.ticker}', false)">
        <td class="td-ticker">
          <div class="ticker-identity">
            <div style="display: flex; align-items: center;">
              <span class="ticker-sym">${p.ticker}</span>
              ${onDemandTag}
            </div>
            <span class="ticker-corp">${p.name}</span>
          </div>
        </td>
        <td class="td-sector">
          <span class="sector-pill">${p.sector}</span>
        </td>
        <td class="td-price">
          <strong>$${p.current_price.toFixed(2)}</strong>
        </td>
        <td class="td-xgb">
          ${getModelPill(p.prob_xgb, 'signal')}
        </td>
        <td class="td-lstm">
          ${getModelPill(p.prob_lstm, 'signal')}
        </td>
        <td class="td-bert">
          ${getModelPill(p.prob_bert, 'sentiment')}
        </td>
        <td class="td-signal">
          <span class="ai-signal-badge ${badgeClass}">${action}</span>
        </td>
        <td class="td-confidence">
          <div class="confidence-bar-wrap">
            <div class="confidence-bar-track">
              <div class="confidence-bar-fill ${isBuy ? 'green' : (isSell ? 'red' : 'yellow')}" style="width: ${p.prob_ensemble ?? 0}%;"></div>
            </div>
            <span class="confidence-num">${p.prob_ensemble == null ? 'Chưa có kết quả' : `${p.prob_ensemble}%`}</span>
          </div>
        </td>
        <td class="td-action">
          <button class="btn-table-action" onclick="event.stopPropagation(); selectTicker('${p.ticker}', true)">
            Xem Biểu Đồ ➔
          </button>
        </td>
      </tr>
    `;
  }).join('');
}

function highlightScreenerRow(ticker) {
  document.querySelectorAll('.screener-row').forEach(row => {
    if (row.dataset.ticker === ticker) row.classList.add('selected');
    else row.classList.remove('selected');
  });
}

// ==========================================================================
// TAB 1: YAHOO FINANCE KEY STATISTICS & INTERACTIVE CANDLESTICK CHART
// ==========================================================================

function updateKeyStatistics(ticker, res) {
  const candles = (res && res.candles) || [];
  const stats = (res && res.statistics) || {};
  const hasNumber = value => typeof value === 'number' && Number.isFinite(value);
  const number = value => hasNumber(value) ? value.toFixed(2) : '--';
  const calendarDate = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value)
    ? value.split('-').reverse().join('/') : '--';
  const fields = {
    'stat-prev-close': '--', 'stat-open': '--', 'stat-bid': '--',
    'stat-ask': '--', 'stat-day-range': '--', 'stat-52w-range': '--',
    'stat-volume': '--', 'stat-avg-volume': '--', 'stat-market-cap': '--',
    'stat-beta': '--', 'stat-pe': '--', 'stat-eps': '--',
    'stat-earnings': '--', 'stat-dividend': '--',
    'stat-ex-dividend': '--', 'stat-target-est': '--'
  };
  const put = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  };

  const overnightBlock = document.getElementById('chart-overnight-block');
  const dividendBanner = document.getElementById('dividend-alert-banner');
  if (overnightBlock) overnightBlock.style.display = 'none';
  if (dividendBanner) dividendBanner.style.display = 'none';
  put('chart-current-price', '--');
  put('chart-change-tag', '--');
  put('chart-time-tag', 'Chưa có dữ liệu giá');
  put('chart-exchange-info', 'Dữ liệu nến ngày • USD');
  put('chart-buy-time-tag', `Thông tin ${ticker}`);
  put('stat-avg-volume-label', 'KLGD TB (tối đa 60 phiên)');
  put('stat-earnings-label', stats.earnings_is_estimate === false
    ? 'Ngày Báo Cáo KQKD Gần Nhất' : 'Ngày Báo Cáo KQKD (Dự Kiến)');

  if (candles.length > 0) {
    const last = candles[candles.length - 1];
    const prev = candles.length > 1 ? candles[candles.length - 2] : null;
    fields['stat-open'] = last.open.toFixed(2);
    fields['stat-day-range'] = `${last.low.toFixed(2)} - ${last.high.toFixed(2)}`;
    fields['stat-volume'] = Number(last.volume).toLocaleString();
    const recent = candles.slice(-60);
    fields['stat-avg-volume'] = Math.round(recent.reduce((sum, c) => sum + Number(c.volume || 0), 0) / recent.length).toLocaleString();
    if (candles.length >= 252) {
      const year = candles.slice(-252);
      fields['stat-52w-range'] = `${Math.min(...year.map(c => c.low)).toFixed(2)} - ${Math.max(...year.map(c => c.high)).toFixed(2)}`;
    }
    put('chart-current-price', `$${last.close.toFixed(2)}`);
    put('chart-time-tag', `Phiên gần nhất: ${last.date}`);
    if (prev) {
      fields['stat-prev-close'] = prev.close.toFixed(2);
      const diff = last.close - prev.close;
      const pct = prev.close > 0 ? diff / prev.close * 100 : 0;
      const positive = diff >= 0;
      const change = document.getElementById('chart-change-tag');
      if (change) {
        change.textContent = `${positive ? '+' : ''}${diff.toFixed(2)} (${positive ? '+' : ''}${pct.toFixed(2)}%)`;
        change.className = `quote-price-diff ${positive ? 'text-success' : 'text-danger'}`;
      }
    }
  }
  const bindings = {'stat-bid': 'bid', 'stat-ask': 'ask', 'stat-beta': 'beta',
    'stat-pe': 'pe_ttm', 'stat-eps': 'eps_ttm', 'stat-target-est': 'target_mean_price'};
  Object.entries(bindings).forEach(([id, key]) => { fields[id] = number(stats[key]); });
  if (hasNumber(stats.market_cap)) {
    const units = [[1e12, 'nghìn tỷ'], [1e9, 'tỷ'], [1e6, 'triệu']];
    const unit = units.find(([threshold]) => stats.market_cap >= threshold);
    fields['stat-market-cap'] = `${unit ? (stats.market_cap / unit[0]).toFixed(2) + ' ' + unit[1]
      : stats.market_cap.toLocaleString()} ${stats.currency || 'USD'}`;
  }
  if (hasNumber(stats.low_52w) && hasNumber(stats.high_52w)) {
    fields['stat-52w-range'] = `${number(stats.low_52w)} - ${number(stats.high_52w)}`;
  }
  if (hasNumber(stats.average_volume)) {
    fields['stat-avg-volume'] = Math.round(stats.average_volume).toLocaleString();
    put('stat-avg-volume-label', 'KLGD TB (3 tháng • Yahoo)');
  }
  fields['stat-earnings'] = calendarDate(stats.earnings_date);
  if (stats.earnings_date_end && stats.earnings_date_end !== stats.earnings_date) {
    fields['stat-earnings'] += ` - ${calendarDate(stats.earnings_date_end)}`;
  }
  fields['stat-ex-dividend'] = calendarDate(stats.ex_dividend_date);
  if (hasNumber(stats.dividend_rate) || hasNumber(stats.dividend_yield_pct)) {
    const rate = hasNumber(stats.dividend_rate) ? number(stats.dividend_rate) : '--';
    const yieldPct = hasNumber(stats.dividend_yield_pct) ? `${number(stats.dividend_yield_pct)}%` : '--';
    fields['stat-dividend'] = `${rate} (${yieldPct})`;
  }
  const retrieved = stats.fetched_at ? new Date(stats.fetched_at) : null;
  const updated = retrieved && Number.isFinite(retrieved.getTime())
    ? ` • Lấy lúc ${retrieved.toLocaleString('vi-VN')}` : '';
  const sourceNote = stats.status === 'ok'
    ? `Thống kê: Yahoo Finance${updated}. “--”: nguồn chưa cung cấp chỉ số.`
    : (stats.message || 'Chưa lấy được thống kê Yahoo Finance.');
  put('stat-source-note', sourceNote);
  Object.entries(fields).forEach(([id, value]) => put(id, value));
}

async function loadTickerChart(ticker) {
  try {
    const res = await fetch(`/api/ticker?symbol=${ticker}`).then(r => r.json());
    state.chartData[ticker] = res;

    // Cập nhật Tên & Thông tin mã
    const tickerNameEl = document.getElementById('chart-ticker-name');
    if (tickerNameEl) tickerNameEl.textContent = res.ticker || ticker;

    const companyNameEl = document.getElementById('chart-company-name');
    if (companyNameEl) companyNameEl.textContent = res.name || ticker;

    const sectorEl = document.getElementById('chart-sector');
    if (sectorEl) {
      const rawSector = res.sector || 'Chưa có thông tin';
      sectorEl.textContent = SECTOR_MAP_VN[rawSector] || rawSector;
    }

    const sideSymEl = document.getElementById('side-ticker-symbol');
    if (sideSymEl) sideSymEl.textContent = ticker;

    // Cập nhật toàn bộ Key Statistics & Quote Hero Banner
    updateKeyStatistics(ticker, res);

    // Cập nhật giá nến gần nhất
    const candles = res.candles || [];
    if (candles.length > 0) {
      const lastCandle = candles[candles.length - 1];
      const curPriceEl = document.getElementById('chart-current-price');
      if (curPriceEl) {
        curPriceEl.textContent = `$${lastCandle.close.toFixed(2)}`;
      }
    }

    // Cập nhật Trading Plan summary
    updateTickerPredictions(ticker);

    // Cập nhật tin tức FinBERT
    renderNews(res.news || [], res.news_status);

    // Vẽ biểu đồ nến SVG tương tác
    renderCandlestickSVG(candles);
  } catch (e) {
    console.error(`Lỗi tải biểu đồ ${ticker}:`, e);
  }
}

function updateTickerPredictions(ticker) {
  const pred = (state.marketData && state.marketData.predictions)
    ? state.marketData.predictions.find(p => p.ticker === ticker)
    : null;

  if (pred) {
    const curPriceEl = document.getElementById('chart-current-price');
    if (curPriceEl && pred.current_price) {
      curPriceEl.textContent = `$${pred.current_price.toFixed(2)}`;
    }

    const badge = document.getElementById('chart-signal-badge');
    if (badge) {
      const action = pred.pp4_tong_hop || 'THEO DÕI';
      badge.textContent = action;
      badge.className = `price-badge ${action.includes('MUA') ? 'stat-badge green' : (action.includes('BÁN') ? 'stat-badge red' : 'stat-badge yellow')}`;
    }

    // Cập nhật Trading Plan Header Mini Card
    const pEntry = document.getElementById('chart-plan-entry');
    const pTp = document.getElementById('chart-plan-tp');
    const pSl = document.getElementById('chart-plan-sl');
    const pRr = document.getElementById('chart-plan-rr');

    if (pEntry) pEntry.textContent = `$${pred.plan_entry.toFixed(2)}`;
    if (pTp) {
      pTp.textContent = pred.plan_target > 0 ? `$${pred.plan_target.toFixed(2)} (+${pred.plan_tp_pct}%)` : 'Chưa có kết quả';
    }
    if (pSl) {
      pSl.textContent = pred.plan_stop_loss > 0 ? `$${pred.plan_stop_loss.toFixed(2)} (-${pred.plan_sl_pct}%)` : 'Chưa có kết quả';
    }
    if (pRr) pRr.textContent = pred.plan_rr || 'Chưa có kết quả';

    // 4 Model Gauges
    const setGauge = (idVal, idBar, val) => {
      const vEl = document.getElementById(idVal);
      const bEl = document.getElementById(idBar);
      const available = val !== null && val !== undefined && Number.isFinite(Number(val));
      if (vEl) vEl.textContent = available ? `${val}%` : 'Không khả dụng';
      if (bEl) bEl.style.width = available ? `${Math.min(100, Math.max(0, val))}%` : '0%';
    };

    setGauge('side-xgb-val', 'side-xgb-bar', pred.prob_xgb);
    setGauge('side-lstm-val', 'side-lstm-bar', pred.prob_lstm);
    setGauge('side-ensemble-val', 'side-ensemble-bar', pred.prob_ensemble);
  } else {
    // Trường hợp mã mở rộng (như TSLA, META, AMD, T...)
    const cData = state.chartData[ticker];
    const candles = cData ? (cData.candles || []) : [];
    const lastPrice = candles.length > 0 ? candles[candles.length - 1].close : null;

    const curPriceEl = document.getElementById('chart-current-price');
    if (curPriceEl && lastPrice) {
      curPriceEl.textContent = `$${lastPrice.toFixed(2)}`;
    }

    const badge = document.getElementById('chart-signal-badge');
    if (badge) {
      badge.textContent = state.marketData?.status === 'unavailable' ? 'CHƯA CÓ DỮ LIỆU DỰ BÁO' : 'THEO DÕI (LIVE DATA)';
      badge.className = 'price-badge stat-badge yellow';
    }

    const pEntry = document.getElementById('chart-plan-entry');
    const pTp = document.getElementById('chart-plan-tp');
    const pSl = document.getElementById('chart-plan-sl');
    const pRr = document.getElementById('chart-plan-rr');

    if (pEntry) pEntry.textContent = lastPrice ? `$${lastPrice.toFixed(2)}` : '--';
    if (pTp) pTp.textContent = 'Chưa có dự báo';
    if (pSl) pSl.textContent = 'Chưa có dự báo';
    if (pRr) pRr.textContent = '--';

    const setGauge = (idVal, idBar, val, text) => {
      const vEl = document.getElementById(idVal);
      const bEl = document.getElementById(idBar);
      if (vEl) vEl.textContent = text || `${val}%`;
      if (bEl) bEl.style.width = '0%';
    };

    setGauge('side-xgb-val', 'side-xgb-bar', null, 'Chưa có kết quả');
    setGauge('side-lstm-val', 'side-lstm-bar', null, 'Chưa có kết quả');
    setGauge('side-ensemble-val', 'side-ensemble-bar', null, 'Chưa có kết quả');
  }

  const statusNoteEl = document.getElementById('side-model-status-note');
  if (statusNoteEl) {
    if (pred) {
      const isOnDemand = !!pred.on_demand;
      const consensusText = pred.pp4_tong_hop || pred.trang_thai || 'THEO DÕI';
      let consensusClass = 'neutral';
      let consensusIcon = '🟡';
      if (consensusText.includes('MUA')) {
        consensusClass = 'buy';
        consensusIcon = '🟢';
      } else if (consensusText.includes('BÁN')) {
        consensusClass = 'sell';
        consensusIcon = '🔴';
      }

      const timeText = pred.analyzed_at 
        ? `${pred.analyzed_at}`
        : (isOnDemand ? 'Vừa phân tích' : 'Airflow tự động');

      statusNoteEl.innerHTML = `
        <div class="model-status-card ${isOnDemand ? 'is-ondemand' : 'is-core'}">
          <div class="model-status-header">
            <div class="status-type-badge">
              <span class="status-pulse-dot ${isOnDemand ? 'pulse-amber' : 'pulse-green'}"></span>
              <span class="status-type-text">${isOnDemand ? '⚡ On-Demand Live' : '🛡️ Top 100 S&P'}</span>
              <span class="status-ticker-tag">${ticker}</span>
            </div>
            <div class="status-consensus-badge ${consensusClass}">
              <span>${consensusIcon}</span>
              <span>${consensusText}</span>
            </div>
          </div>

          <div class="model-mini-stats">
            <span class="stat-cell" title="🌲 XGBoost (Kỹ thuật & Vĩ mô)"><span class="stat-name">XGB</span> <strong class="stat-num">${pred.prob_xgb == null ? 'Không khả dụng' : `${pred.prob_xgb}%`}</strong></span>
            <span class="stat-dot">•</span>
            <span class="stat-cell" title="🧠 LSTM (Chuỗi Nến 5 Năm)"><span class="stat-name">LSTM</span> <strong class="stat-num">${pred.prob_lstm == null ? 'Không khả dụng' : `${pred.prob_lstm}%`}</strong></span>
            <span class="stat-dot">•</span>
            <span class="stat-cell" title="📰 FinBERT — Điểm tín hiệu tin tức"><span class="stat-name">BERT</span> <strong class="stat-num">${pred.prob_bert == null ? 'Không khả dụng' : `${pred.prob_bert}%`}</strong></span>
            <span class="stat-dot">•</span>
            <span class="stat-cell highlight" title="👑 Master Ensemble"><span class="stat-name">Ens</span> <strong class="stat-num">${pred.prob_ensemble == null ? 'Chưa có kết quả' : `${pred.prob_ensemble}%`}</strong></span>
          </div>

          <div class="model-status-meta">
            <span class="model-meta-time">
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
              </svg>
              <span>${timeText}</span>
            </span>
            <span class="model-meta-tag">${isOnDemand ? (pred.real_models_used || []).join(' + ') : 'Daily Pipeline'}</span>
          </div>

          <button class="btn-reanalyze" onclick="runOnDemandAnalysis('${ticker}')" title="Chạy lại phân tích nến và tin tức mới nhất cho ${ticker}">
            <svg class="reanalyze-svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="23 4 23 10 17 10"></polyline>
              <polyline points="1 20 1 14 7 14"></polyline>
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
            </svg>
            <span>Phân tích lại <strong>${ticker}</strong></span>
          </button>
        </div>
      `;
    } else {
      statusNoteEl.innerHTML = `
        <div class="model-status-card is-empty">
          <div class="model-status-header">
            <div class="status-type-badge">
              <span class="status-pulse-dot pulse-amber"></span>
              <span class="status-type-text">Mã Mở Rộng</span>
              <span class="status-ticker-tag">${ticker}</span>
            </div>
            <div class="status-consensus-badge neutral">
              <span>⚠️</span>
              <span>Chưa chạy AI</span>
            </div>
          </div>

          <p class="model-empty-desc">
            Mã <strong>${ticker}</strong> chưa có kết quả từ 4 mô hình AI (XGBoost, LSTM, FinBERT, Ensemble).
          </p>

          <button class="btn-analyze-now-primary" onclick="runOnDemandAnalysis('${ticker}')" title="Kích hoạt phân tích AI tức thời cho ${ticker}">
            <svg class="analyze-svg" width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
            </svg>
            <span>⚡ Phân Tích AI Ngay</span>
          </button>
        </div>
      `;
    }
  }

  renderFinbertStatus(pred);
  updateHeaderActiveTicker();
}

function finbertSentimentScores(pred) {
  const values = pred?.finbert_sentiment || {};
  const scores = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral'].map(key => values[key]);
  return scores.every(value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1)
    && Math.abs(scores.reduce((sum, value) => sum + value, 0) - 1) <= .01 ? scores : null;
}

function finbertSignalPercent(pred) {
  const scores = finbertSentimentScores(pred);
  if (!scores) return null;
  return Math.min(95, Math.max(5, (1 + scores[0] - scores[1]) * 50));
}

function renderFinbertStatus(pred) {
  const scores = finbertSentimentScores(pred);
  const label = document.getElementById('side-bert-val');
  const score = finbertSignalPercent(pred);
  if (label) label.textContent = score == null ? 'Không khả dụng' : `${score.toFixed(1)}%`;
  const bar = document.getElementById('side-bert-bar');
  if (bar) bar.style.width = score == null ? '0%' : `${score.toFixed(1)}%`;
  const container = document.getElementById('side-finbert-status');
  if (!container) return;
  container.replaceChildren();
  container.hidden = !pred?.finbert_message && !scores;
  if (container.hidden) return;
  const message = document.createElement('div');
  message.textContent = `FinBERT: ${pred.finbert_message || "Cảm xúc tin tức"}`;
  container.append(message);
  const sentiment = pred.finbert_sentiment || {};
  const labels = [['sentiment_positive', 'Tích cực'], ['sentiment_negative', 'Tiêu cực'], ['sentiment_neutral', 'Trung tính']];
  if (scores) {
    const scores = document.createElement('div');
    scores.textContent = 'Cảm xúc tin tức: ' + labels.map(([key, label]) => `${label} ${(sentiment[key] * 100).toFixed(1)}%`).join(' · ');
    container.append(scores);
  }
  if (score != null) {
    const signal = document.createElement('div');
    signal.textContent = `Điểm tín hiệu tin tức: Tăng ${score.toFixed(1)}% · Giảm ${(100 - score).toFixed(1)}%`;
    container.append(signal);
  }
  if (pred.ensemble_weights) {
    const weights = document.createElement('div');
    const labels = {prob_xgb: 'XGBoost', prob_lstm: 'LSTM', prob_bert: 'FinBERT'};
    weights.textContent = 'Master: ' + Object.entries(pred.ensemble_weights).filter(([key, value]) => labels[key] && typeof value === 'number' && Number.isFinite(value)).map(([key, value]) => `${labels[key]} ${(value * 100).toFixed(0)}%`).join(' + ');
    container.append(weights);
  }
  if (Array.isArray(pred.finbert_sources) && pred.finbert_sources.length) {
    const sources = document.createElement('div');
    sources.textContent = `Nguồn tin: ${pred.finbert_sources.join(', ')}`;
    container.append(sources);
  }

}

function renderNews(newsList, newsStatus) {
  const container = document.getElementById('side-news-list');
  if (!container) return;
  container.replaceChildren();

  if (!Array.isArray(newsList) || newsList.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'empty-news';
    empty.style.cssText = 'padding: 16px; color: var(--text-muted); font-size: 12px; text-align: center;';
    empty.textContent = newsStatus?.status === 'source_error'
      ? 'Nguồn tin đang lỗi kết nối. Vui lòng thử lại sau.'
      : 'Chưa có tin hợp lệ trong 72 giờ qua.';
    container.append(empty);
    return;
  }

  const fallback = `https://finance.yahoo.com/quote/${encodeURIComponent(state.activeTicker)}/news/`;
  newsList.forEach(n => {
    if (!n || typeof n !== 'object') return;
    const link = document.createElement('a');
    let url = fallback;
    try {
      const parsed = new URL(String(n.url || fallback));
      if (['http:', 'https:'].includes(parsed.protocol)) url = parsed.href;
    } catch (_) { /* Invalid article links use the ticker's news page. */ }
    link.href = url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.className = 'news-item';
    link.title = `Đọc bài báo gốc: ${String(n.headline || '')}`;

    const headline = document.createElement('div');
    headline.className = 'news-headline';
    headline.textContent = String(n.headline || '');
    const meta = document.createElement('div');
    meta.className = 'news-meta';
    const source = document.createElement('span');
    source.textContent = `📰 ${String(n.source || '')}`;
    const date = document.createElement('span');
    date.textContent = String(n.date || '').substring(0, 10);
    const read = document.createElement('span');
    read.className = 'news-read-link';
    read.textContent = 'Đọc bài gốc ↗';
    meta.append(source, date, read);
    link.append(headline, meta);
    container.append(link);
  });
}

// Vẽ biểu đồ nến SVG kèm đường Take Profit & Stop Loss trực quan
function renderCandlestickSVG(candles) {
  const svg = document.getElementById('candlestick-svg');
  const rsiSvg = document.getElementById('rsi-svg');
  const tooltip = document.getElementById('chart-tooltip');
  if (!svg || candles.length === 0) {
    if (svg) svg.innerHTML = `<text x="50%" y="50%" text-anchor="middle" fill="#64748B" font-size="14">Đang chuẩn bị nến cho mã này...</text>`;
    return;
  }

  const width = svg.clientWidth || 800;
  const height = 340;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

  // Lấy thông tin plan để vẽ TP/SL
  const pred = (state.marketData && state.marketData.predictions)
    ? state.marketData.predictions.find(p => p.ticker === state.activeTicker)
    : null;

  const hasPlan = pred && pred.plan_target > 0 && pred.plan_stop_loss > 0;

  // Tính min / max giá
  let minP = Math.min(...candles.map(c => c.low));
  let maxP = Math.max(...candles.map(c => c.high));

  if (hasPlan) {
    minP = Math.min(minP, pred.plan_stop_loss * 0.99);
    maxP = Math.max(maxP, pred.plan_target * 1.01);
  }

  const padP = (maxP - minP) * 0.05 || 1.0;
  minP -= padP;
  maxP += padP;

  const paddingLeft = 15;
  const paddingRight = 75;
  const paddingTop = 25;
  const paddingBottom = 30;
  const chartW = width - paddingLeft - paddingRight;
  const chartH = height - paddingTop - paddingBottom;

  const candleW = Math.max(3, Math.min(14, (chartW / candles.length) * 0.7));
  const stepX = chartW / candles.length;

  function getY(price) {
    return paddingTop + chartH - ((price - minP) / (maxP - minP)) * chartH;
  }

  let html = '';

  // 1. Gridlines ngang & Price Labels
  const gridCount = 5;
  for (let i = 0; i <= gridCount; i++) {
    const p = minP + (maxP - minP) * (i / gridCount);
    const y = getY(p);
    html += `<line x1="${paddingLeft}" y1="${y}" x2="${width - paddingRight}" y2="${y}" stroke="rgba(255,255,255,0.06)" stroke-dasharray="3,3" />`;
    html += `<text x="${width - paddingRight + 8}" y="${y + 4}" fill="#64748B" font-size="10" font-family="monospace">$${p.toFixed(2)}</text>`;
  }

  // 2. Vẽ đường Take Profit & Stop Loss nếu có
  if (hasPlan) {
    const yTp = getY(pred.plan_target);
    const ySl = getY(pred.plan_stop_loss);

    // Take Profit Line
    html += `<line x1="${paddingLeft}" y1="${yTp}" x2="${width - paddingRight}" y2="${yTp}" stroke="#10B981" stroke-width="1.5" stroke-dasharray="4,4" />`;
    html += `<rect x="${width - paddingRight + 4}" y="${yTp - 9}" width="68" height="18" fill="#10B981" rx="3" opacity="0.9" />`;
    html += `<text x="${width - paddingRight + 8}" y="${yTp + 3}" fill="#FFFFFF" font-size="9" font-weight="700">TP: $${pred.plan_target.toFixed(1)}</text>`;

    // Stop Loss Line
    html += `<line x1="${paddingLeft}" y1="${ySl}" x2="${width - paddingRight}" y2="${ySl}" stroke="#F43F5E" stroke-width="1.5" stroke-dasharray="4,4" />`;
    html += `<rect x="${width - paddingRight + 4}" y="${ySl - 9}" width="68" height="18" fill="#F43F5E" rx="3" opacity="0.9" />`;
    html += `<text x="${width - paddingRight + 8}" y="${ySl + 3}" fill="#FFFFFF" font-size="9" font-weight="700">SL: $${pred.plan_stop_loss.toFixed(1)}</text>`;
  }

  // 3. Đường MA20
  let maPoints = [];
  for (let i = 0; i < candles.length; i++) {
    if (i >= 19) {
      const slice = candles.slice(i - 19, i + 1);
      const avg = slice.reduce((sum, c) => sum + c.close, 0) / 20;
      const x = paddingLeft + i * stepX + candleW / 2;
      maPoints.push(`${x},${getY(avg)}`);
    }
  }
  if (maPoints.length > 1) {
    html += `<polyline fill="none" stroke="#F59E0B" stroke-width="2" points="${maPoints.join(' ')}" opacity="0.85" />`;
  }

  // 4. Vẽ nến OHLCV
  candles.forEach((c, i) => {
    const x = paddingLeft + i * stepX;
    const isGreen = c.close >= c.open;
    const color = isGreen ? '#10B981' : '#F43F5E';

    const yHigh = getY(c.high);
    const yLow = getY(c.low);
    const yOpen = getY(c.open);
    const yClose = getY(c.close);

    const bodyY = Math.min(yOpen, yClose);
    const bodyH = Math.max(2, Math.abs(yOpen - yClose));

    // Wick
    html += `<line x1="${x + candleW / 2}" y1="${yHigh}" x2="${x + candleW / 2}" y2="${yLow}" stroke="${color}" stroke-width="1.5" />`;
    // Body
    html += `<rect class="candle-bar" data-idx="${i}" x="${x}" y="${bodyY}" width="${candleW}" height="${bodyH}" fill="${color}" rx="1" style="cursor: pointer;" />`;
  });

  svg.innerHTML = html;

  // Tooltip interaction
  if (tooltip) {
    const bars = svg.querySelectorAll('.candle-bar');
    bars.forEach(bar => {
      bar.addEventListener('mouseenter', (e) => {
        const idx = parseInt(e.target.getAttribute('data-idx'));
        const c = candles[idx];
        if (!c) return;
        const isUp = c.close >= c.open;
        const diff = c.close - c.open;
        const diffPct = (diff / c.open) * 100;

        tooltip.innerHTML = `
          <div style="font-weight: 700; margin-bottom: 4px; color: #F8FAFC;">${c.date || 'Phiên'}</div>
          <div>Mở: <b>$${c.open.toFixed(2)}</b> | Đóng: <b style="color: ${isUp ? '#10B981' : '#F43F5E'}">$${c.close.toFixed(2)}</b></div>
          <div>Cao: $${c.high.toFixed(2)} | Thấp: $${c.low.toFixed(2)}</div>
          <div>Biến động: <b style="color: ${isUp ? '#10B981' : '#F43F5E'}">${diff >= 0 ? '+' : ''}${diff.toFixed(2)} (${diffPct.toFixed(2)}%)</b></div>
          ${c.volume ? `<div>Khối lượng: ${c.volume.toLocaleString()}</div>` : ''}
          ${c.rsi ? `<div>RSI: <b>${c.rsi.toFixed(1)}</b></div>` : ''}
        `;
        tooltip.classList.remove('hidden');

        const rect = e.target.getBoundingClientRect();
        const parentRect = svg.parentElement.getBoundingClientRect();
        tooltip.style.left = `${Math.min(parentRect.width - 170, Math.max(10, rect.left - parentRect.left - 50))}px`;
        tooltip.style.top = `${Math.max(10, rect.top - parentRect.top - 90)}px`;
      });

      bar.addEventListener('mouseleave', () => {
        tooltip.classList.add('hidden');
      });
    });
  }

  // 5. RSI Subchart
  if (rsiSvg && candles.some(c => c.rsi)) {
    const rsiW = rsiSvg.clientWidth || width;
    const rsiH = 75;
    rsiSvg.setAttribute('viewBox', `0 0 ${rsiW} ${rsiH}`);

    let rsiHtml = `
      <line x1="${paddingLeft}" y1="22" x2="${rsiW - paddingRight}" y2="22" stroke="rgba(244,63,94,0.35)" stroke-dasharray="2,2" />
      <text x="${rsiW - paddingRight + 6}" y="25" fill="#F43F5E" font-size="9">70 (Quá mua)</text>
      <line x1="${paddingLeft}" y1="52" x2="${rsiW - paddingRight}" y2="52" stroke="rgba(16,185,129,0.35)" stroke-dasharray="2,2" />
      <text x="${rsiW - paddingRight + 6}" y="55" fill="#10B981" font-size="9">30 (Quá bán)</text>
    `;

    const rsiPoints = candles.map((c, i) => {
      if (c.rsi == null) return null;
      const x = paddingLeft + i * stepX + candleW / 2;
      const y = 6 + (65 - 6) - ((c.rsi - 20) / (80 - 20)) * (65 - 6);
      return `${x},${Math.max(4, Math.min(70, y))}`;
    }).filter(Boolean).join(' ');

    rsiHtml += `<polyline fill="none" stroke="#A855F7" stroke-width="1.5" points="${rsiPoints}" />`;
    rsiSvg.innerHTML = rsiHtml;

    const lastRsi = candles[candles.length - 1].rsi;
    const rsiValEl = document.getElementById('rsi-latest-val');
    if (rsiValEl) rsiValEl.textContent = lastRsi == null ? 'RSI: Chưa có dữ liệu' : `RSI: ${lastRsi.toFixed(1)}`;
  }
}

// ==========================================================================
// TAB 3: DATA QUALITY GATE & MODEL EVALUATION AUDITING
// ==========================================================================

function measured(value, digits = 2, percent = false) {
  if (value == null || !Number.isFinite(Number(value))) return '—';
  return (Number(value) * (percent ? 100 : 1)).toFixed(digits) + (percent ? '%' : '');
}
function displayText(id, text) { const el = document.getElementById(id); if (el) el.textContent = text; }

async function loadQualityData() {
  let report;
  try { report = await fetch('/api/quality').then(r => r.json()); }
  catch { report = {status: 'unavailable', message: 'Không thể tải báo cáo kiểm định.'}; }
  displayText('quality-status-text', report.status === 'PASSED' ? 'ĐÃ KIỂM ĐỊNH' : report.status === 'FAILED' ? 'CÓ LỖI DỮ LIỆU' : 'Chưa có kết quả');
  displayText('quality-report-message', report.timestamp ? `Ngày kiểm định: ${report.timestamp} · ${report.error_count} lỗi · ${report.warning_count} cảnh báo` : report.message || 'Chưa có kết quả.');
  displayText('qa-ticker-coverage', report.tickers_count == null ? '—' : `${report.tickers_count} / ${report.expected_tickers_count} mã`);
  displayText('qa-total-rows', report.total_rows == null ? '—' : `${report.total_rows.toLocaleString('vi-VN')} bản ghi`);
  displayText('qa-total-cols', report.total_columns == null ? '—' : `${report.total_columns} cột`);
  displayText('qa-clean-pct', report.avg_missing_pct == null ? '—' : `${(100 - report.avg_missing_pct).toFixed(2)}%`);
}

async function loadEvaluationData() {
  let report;
  try { report = await fetch('/api/evaluation').then(r => r.json()); }
  catch { report = {status: 'unavailable', message: 'Không thể tải kết quả đánh giá.', models_comparison: []}; }
  displayText('eval-date-str', report.date || '—');
  displayText('evaluation-message', report.status === 'ok' ? 'Đã kiểm chứng bằng dữ liệu thực tế. Xem ngày đánh giá, kỳ hạn và số mẫu bên dưới.' : report.message || 'Chưa có kết quả đánh giá.');
  const mapping = {'Ensemble Master': 'ensemble', 'XGBoost': 'xgboost', 'LSTM': 'lstm'};
  for (const key of Object.values(mapping)) {
    displayText(`leaderboard-${key}-winrate`, 'Chưa có kết quả');
    displayText(`leaderboard-${key}-details`, 'Ngày đánh giá: — · Số mẫu: —');
    const bar = document.getElementById(`leaderboard-${key}-bar`); if (bar) bar.style.width = '0%';
  }
  const models = (report.models_comparison || []).filter(model => model.model !== 'FinBERT');
  for (const model of models) {
    const key = mapping[model.model]; if (!key) continue;
    const valid = model.accuracy != null && model.total_samples > 0;
    displayText(`leaderboard-${key}-winrate`, valid ? measured(model.accuracy, 1, true) : 'Chưa có kết quả');
    displayText(`leaderboard-${key}-details`, `ROC-AUC: ${measured(model.auc_roc, 4)} · Ngày: ${model.evaluation_date || '—'} · Số mẫu: ${model.total_samples ?? 0} · T+${model.prediction_horizon ?? '—'}`);
    const bar = document.getElementById(`leaderboard-${key}-bar`); if (bar) bar.style.width = valid ? `${model.accuracy * 100}%` : '0%';
  }
  const tbody = document.getElementById('eval-benchmark-body');
  if (tbody) {
    tbody.replaceChildren();
    if (!models.length) {
      const row = tbody.insertRow(); const cell = row.insertCell(); cell.colSpan = 9; cell.textContent = report.message || 'Chưa có kết quả đánh giá.';
    }
    for (const model of models) {
      const row = tbody.insertRow();
      const fields = [model.model, measured(model.accuracy, 2, true), measured(model.auc_roc, 4),
                      `${measured(model.precision_class1, 1, true)} / ${measured(model.precision_class0, 1, true)}`,
                      `${measured(model.recall_class1, 1, true)} / ${measured(model.recall_class0, 1, true)}`,
                      `${measured(model.f1_score_class1, 3)} / ${measured(model.f1_score_class0, 3)}`,
                      String(model.total_samples ?? 0), model.evaluation_date || '—', `T+${model.prediction_horizon ?? '—'}`];
      fields.forEach(value => { row.insertCell().textContent = value; });
    }
  }
  const folds = document.getElementById('cv-folds-list');
  if (folds) {
    folds.replaceChildren();
    for (const fold of report.cross_validation_folds || []) {
      const entry = document.createElement('p'); entry.textContent = `Fold ${fold.fold}: Accuracy ${measured(fold.accuracy, 2, true)} · ROC-AUC ${measured(fold.auc_roc, 4)} · ${fold.test_size} mẫu`; folds.appendChild(entry);
    }
  }
}
