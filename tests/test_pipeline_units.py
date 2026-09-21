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


    def test_tickers_list_matches_sector_map(self):
        """TICKERS_LIST phải khớp chính xác với keys của SECTOR_MAP."""
        self.assertEqual(set(TICKERS_LIST), set(SECTOR_MAP.keys()))

    def test_tickers_meta_matches_sector_map(self):
        """TICKERS_META phải có đúng cùng tập mã với SECTOR_MAP."""
        self.assertEqual(set(TICKERS_META.keys()), set(SECTOR_MAP.keys()))

    def test_tickers_meta_has_required_fields(self):
        """Mỗi ticker trong TICKERS_META phải có key 'name' và 'sector'."""
        for ticker, meta in TICKERS_META.items():
            self.assertIn('name', meta, f"{ticker} thiếu 'name'")
            self.assertIn('sector', meta, f"{ticker} thiếu 'sector'")
            self.assertTrue(len(meta['name']) > 0, f"{ticker} có name rỗng")


class TestDataValidator(unittest.TestCase):
    """Kiểm thử module data_validator (logic thuần, không cần S3)."""

    def test_validate_xgboost_empty_df(self):
        import pandas as pd
        from data_validator import validate_xgboost_dataset
        qa = validate_xgboost_dataset(pd.DataFrame())
        self.assertFalse(qa.is_valid)
        self.assertGreater(len(qa.errors), 0)

    def test_validate_xgboost_valid_df(self):
        import pandas as pd
        import numpy as np
        from data_validator import validate_xgboost_dataset
        base_price = np.random.uniform(150, 200, 100)
        df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100),
            'Ticker': ['AAPL'] * 100,
            'Open': base_price,
            'High': base_price + 5.0,
            'Low': base_price - 5.0,
            'Close': base_price,
            'Volume': np.random.randint(1_000_000, 10_000_000, 100),
        })
        qa = validate_xgboost_dataset(df)
        self.assertTrue(qa.is_valid)

    def test_validate_finbert_empty(self):
        import pandas as pd
        from data_validator import validate_finbert_dataset
        qa = validate_finbert_dataset(pd.DataFrame())
        self.assertFalse(qa.is_valid)

    def test_validate_lstm_empty(self):
        import pandas as pd
        from data_validator import validate_lstm_dataset
        qa = validate_lstm_dataset(pd.DataFrame())
        self.assertFalse(qa.is_valid)


class TestAlertUtils(unittest.TestCase):
    """Kiểm thử module alert_utils (logic, không gửi thật)."""

    def test_send_telegram_alert_returns_bool(self):
        from alert_utils import send_telegram_alert
        result = send_telegram_alert("test message")
        self.assertIsInstance(result, bool)


class TestHyperparameterTuner(unittest.TestCase):
    """Kiểm thử module hyperparameter_tuner."""

    def test_default_params_structure(self):
        from hyperparameter_tuner import DEFAULT_XGB_PARAMS
        required_keys = ['n_estimators', 'max_depth', 'learning_rate', 'random_state']
        for k in required_keys:
            self.assertIn(k, DEFAULT_XGB_PARAMS)

    def test_tune_small_dataset_returns_defaults(self):
        import pandas as pd
        import numpy as np
        from hyperparameter_tuner import tune_xgboost_hyperparameters, DEFAULT_XGB_PARAMS
        X = pd.DataFrame(np.random.randn(50, 5), columns=[f'f{i}' for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 50))
        result = tune_xgboost_hyperparameters(X, y)
        self.assertEqual(result, DEFAULT_XGB_PARAMS)


if __name__ == '__main__':
    unittest.main()
