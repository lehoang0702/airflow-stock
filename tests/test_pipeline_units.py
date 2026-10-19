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


class TestMinioComposeConnection(unittest.TestCase):
    def test_airflow_services_share_minio_connection(self):
        import json
        from pathlib import Path
        import yaml

        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / 'docker-compose.yaml').read_text())
        for service in ('airflow-init', 'airflow-webserver', 'airflow-scheduler'):
            environment = dict(item.split('=', 1) for item in config['services'][service]['environment'])
            connection = json.loads(environment['AIRFLOW_CONN_MINIO_CONN'])
            self.assertEqual(connection['conn_type'], 'aws')
            self.assertEqual(connection['extra']['endpoint_url'], '${MINIO_ENDPOINT_URL:-http://minio:9000}')
            self.assertEqual(environment['AWS_ACCESS_KEY_ID'], '${MINIO_ROOT_USER:-minioadmin}')
            self.assertEqual(environment['AWS_SECRET_ACCESS_KEY'], '${MINIO_ROOT_PASSWORD:-minioadmin}')


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

class TestCrossSectionalAndTemporalMasks(unittest.TestCase):
    def setUp(self):
        import pandas as pd
        import numpy as np
        dates = pd.date_range('2020-01-01', periods=10).tolist() * 3
        tickers = ['AAPL']*10 + ['MSFT']*10 + ['GOOGL']*10
        ret_1d = [0.01, 0.02, -0.01, 0.03, 0.00]*6
        future_ret = [0.03, -0.01, 0.00, 0.05, 0.02]*6
        
        self.df = pd.DataFrame({
            'Date_Std': dates,
            'Ticker_Std': tickers,
            'ret_1d': ret_1d,
            'rsi_14': np.random.uniform(30, 70, 30),
            'future_return_1d': future_ret,
            'Target_Std': [1, 0, 0, 1, 1]*6
        })
        self.df['Date_Std'] = pd.to_datetime(self.df['Date_Std'])

    def test_add_cross_sectional_features(self):
        from pipeline_data import add_cross_sectional_features
        df_out, cs_features = add_cross_sectional_features(self.df.copy())
        
        self.assertIn('cs_rank_ret1', cs_features)
        self.assertIn('cs_rel_ret1', cs_features)
        self.assertIn('cs_rank_rsi14', cs_features)
        
        # Verify correctness for the first date
        first_date = df_out['Date_Std'].iloc[0]
        slice_first_date = df_out[df_out['Date_Std'] == first_date]
        mean_ret1 = slice_first_date['ret_1d'].mean()
        
        rel_diff = slice_first_date['ret_1d'].iloc[0] - mean_ret1
        self.assertAlmostEqual(slice_first_date['cs_rel_ret1'].iloc[0], rel_diff)

    def test_temporal_masks_max_train_years(self):
        from pipeline_data import temporal_masks
        import pandas as pd
        
        train_mask, test_mask = temporal_masks(
            self.df.copy(), 
            cutoff='2020-01-08', 
            max_train_years=3, 
            min_return_threshold=0.0
        )
        
        # Dates must be < boundary (2020-01-08) and >= 2017-01-08
        self.assertTrue(train_mask.any())
        self.assertTrue(test_mask.any())

    def test_temporal_masks_min_return_threshold(self):
        from pipeline_data import temporal_masks
        
        train_mask, test_mask = temporal_masks(
            self.df.copy(), 
            cutoff='2020-01-08', 
            max_train_years=3, 
            min_return_threshold=0.015
        )
        
        df_train = self.df[train_mask]
        # Check if min return threshold is respected
        self.assertTrue((df_train['future_return_1d'].abs() >= 0.015).all())

