"""
UNIT TESTS SUITE — PIPELINE KIỂM THỬ TỰ ĐỘNG (CI/CD)
=====================================================
Bộ kiểm thử đơn vị độc lập (chạy được cả trên môi trường Host tối giản
và môi trường CI/Docker đầy đủ thư viện numpy/pandas).
"""

import sys
import os
import unittest

# Thêm đường dẫn dags/ vào sys.path để import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dags')))

from config_shared import (
    SECTOR_MAP,
    TICKERS_LIST,
    TICKERS_META,
    MACRO_TICKERS,
    FUNDAMENTAL_FIELDS,
    INTRADAY_INTERVAL
)

try:
    import numpy as np
    import pandas as pd
    from dag_crawl_intraday_features import calculate_intraday_indicators
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class TestConfigIntegrity(unittest.TestCase):
    """Kiểm thử tính toàn vẹn và nhất quán của toàn bộ cấu hình hệ thống (Pure Python)."""

    def test_tickers_count(self):
        """Xác nhận danh mục phải có đúng 20 mã cổ phiếu Blue-chip S&P 500."""
        self.assertEqual(len(TICKERS_LIST), 20)
        self.assertEqual(len(SECTOR_MAP), 20)
        self.assertEqual(len(TICKERS_META), 20)

    def test_sectors_mapping(self):
        """Xác nhận các nhóm ngành GICS chuẩn quốc tế được gán đầy đủ."""
        valid_sectors = {
            'Technology', 'Consumer Discretionary', 'Consumer Staples',
            'Financials', 'Healthcare', 'Industrials', 'Energy',
            'Utilities', 'Materials'
        }
        assigned_sectors = set(SECTOR_MAP.values())
        self.assertTrue(assigned_sectors.issubset(valid_sectors))

    def test_macro_intermarket_benchmarks(self):
        """Xác nhận các chỉ số vĩ mô và liên thị trường mở rộng được khai báo chuẩn."""
        expected_benchmarks = ['SPY', 'QQQ', '^VIX', '^TNX', 'GC=F', 'CL=F', 'DX-Y.NYB', 'HYG']
        for bm in expected_benchmarks:
            self.assertIn(bm, MACRO_TICKERS)

    def test_fundamental_fields(self):
        """Xác nhận danh mục chỉ số phân tích cơ bản cần thiết."""
        expected_fields = ['trailingPE', 'priceToBook', 'returnOnEquity', 'profitMargins', 'debtToEquity', 'beta']
        for f in expected_fields:
            self.assertIn(f, FUNDAMENTAL_FIELDS)

    def test_intraday_config(self):
        """Xác nhận cấu hình nến Intraday là 1h."""
        self.assertEqual(INTRADAY_INTERVAL, '1h')


class TestIntradayFeatureEngineering(unittest.TestCase):
    """Kiểm thử hàm tính toán đặc trưng nến 1 giờ khi môi trường có pandas/numpy."""

    def setUp(self):
        if not HAS_PANDAS:
            self.skipTest("Bỏ qua test này do môi trường host chưa cài numpy/pandas (chạy đầy đủ trên CI Docker).")
        dates = pd.date_range(start='2024-01-01 09:30', periods=100, freq='h')
        np.random.seed(42)
        close_prices = 150.0 + np.cumsum(np.random.randn(100) * 0.5)
        high_prices = close_prices + np.random.uniform(0.1, 1.0, 100)
        low_prices = close_prices - np.random.uniform(0.1, 1.0, 100)
        open_prices = close_prices + np.random.uniform(-0.3, 0.3, 100)
        volumes = np.random.randint(100000, 500000, 100)

        self.sample_df = pd.DataFrame({
            'datetime': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })

    def test_intraday_indicators_calculation(self):
        """Xác nhận hàm tính đúng và đủ các chỉ báo nến giờ (RSI, EMA, VWAP, Volatility)."""
        res_df = calculate_intraday_indicators(self.sample_df, symbol='AAPL', sector='Technology')
        self.assertFalse(res_df.empty)
        required_cols = [
            'ret_1h', 'ret_4h', 'ema_cross_9_21', 'dist_ema_50',
            'rsi_14h', 'bb_width_1h', 'volatility_10h',
            'vwap_proxy', 'dist_from_vwap', 'target_dir_4h'
        ]
        for col in required_cols:
            self.assertIn(col, res_df.columns)


if __name__ == '__main__':
    unittest.main()
