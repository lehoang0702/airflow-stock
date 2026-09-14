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
)


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


if __name__ == '__main__':
    unittest.main()
