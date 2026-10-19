"""
UNIT TESTS — MODEL REGISTRY (MLOps LifeCycle & Versioning)
===========================================================
Kiểm thử toàn diện:
  1. ModelCard serialization & deserialization (JSON metadata)
  2. XGBoostModelBuilder & LSTMModelBuilder configuration & leakage prevention
  3. Champion-Challenger validation gate & fallback mechanism
  4. Registry indexing and model versioning logic
"""

import sys
import os
import unittest
import json
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# Thêm đường dẫn dags/ vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dags')))

from model_registry import (
    ModelCard,
    XGBoostModelBuilder,
    select_champion_model,
)


class TestModelCard(unittest.TestCase):
    """Kiểm thử tính toàn vẹn của ModelCard metadata."""

    def test_model_card_serialization_roundtrip(self):
        """ModelCard phải serialize ra JSON và deserialize ngược lại nguyên vẹn."""
        card = ModelCard(
            model_name="XGBoost Classifier",
            model_version="v20260923",
            trained_date="2026-09-23",
            model_type="xgboost",
            dataset_size=5000,
            num_features=115,
            num_tickers=20,
            train_period="2020-01-01 -> 2024-12-31",
            test_period="2025-01-01 -> 2026-09-23",
            hyperparameters={"max_depth": 3, "n_estimators": 120, "learning_rate": 0.03},
            metrics={"accuracy": 0.585, "auc_roc": 0.612, "brier_score": 0.231},
            feature_columns=["RSI_14", "MACD_12_26", "SMA_50_Ratio"],
            notes="Trained with purged time-series split."
        )

        json_str = card.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn("v20260923", json_str)
        self.assertIn("RSI_14", json_str)

        restored_card = ModelCard.from_json(json_str)
        self.assertEqual(restored_card.model_name, card.model_name)
        self.assertEqual(restored_card.model_version, card.model_version)
        self.assertEqual(restored_card.dataset_size, 5000)
        self.assertEqual(restored_card.hyperparameters["max_depth"], 3)
        self.assertAlmostEqual(restored_card.metrics["auc_roc"], 0.612)
        self.assertEqual(len(restored_card.feature_columns), 3)


class TestXGBoostModelBuilder(unittest.TestCase):
    """Kiểm thử XGBoostModelBuilder và khả năng lọc bỏ data leakage."""

    def test_preprocess_filters_leakage_columns(self):
        """Preprocess phải loại bỏ triệt để các cột target, future, và raw price."""
        builder = XGBoostModelBuilder()

        dates = pd.date_range("2024-01-01", periods=10)
        df = pd.DataFrame({
            'Date': dates,
            'Ticker': ['AAPL'] * 10,
            'Close': np.linspace(150, 160, 10),
            'Volume': [1000000] * 10,
            'RSI_14': np.random.uniform(30, 70, 10),
            'future_direction_1d': [1, 0, 1, 0, 1, 0, 1, 0, 1, np.nan],
            'leak_target_col': np.random.randint(0, 2, 10),
        })

        processed_df, feature_cols = builder.preprocess(df)
        self.assertIn('Date_Std', processed_df.columns)
        self.assertIn('Target_Std', processed_df.columns)

        # Cột features đầu vào để train không được chứa target hoặc future
        for col in feature_cols:
            c = col.lower()
            self.assertFalse(c.startswith('future'))
            self.assertFalse(c.startswith('target'))
            self.assertNotIn(c, ['close', 'date', 'ticker', 'open', 'high', 'low', 'volume'])


class TestChampionChallengerSelection(unittest.TestCase):
    """Kiểm thử Model Validation Gate và cơ chế Fallback / Safe Mode."""

    def test_champion_selection_passes_with_sufficient_metrics(self):
        """Model mới vượt ngưỡng chất lượng (AUC >= 0.495, Acc >= 0.50) được chọn làm Champion."""
        mock_s3 = MagicMock()
        result = select_champion_model(
            s3_hook=mock_s3,
            bucket_name="stock-data",
            model_type="xgboost",
            current_version="v20260923",
            current_acc=0.58,
            current_auc=0.62,
            min_auc=0.495,
            min_acc=0.50
        )
        self.assertEqual(result['status'], 'USE_CURRENT')
        self.assertEqual(result['selected_version'], 'v20260923')
        self.assertAlmostEqual(result['selected_metrics']['auc_roc'], 0.62)

    def test_fallback_to_best_candidate_when_current_fails(self):
        """Khi model mới không đạt chuẩn, tìm model tốt nhất trong 7 ngày để fallback."""
        mock_s3 = MagicMock()

        # Giả lập lịch sử model trong registry
        df_registry = pd.DataFrame([
            {
                'model_type': 'xgboost',
                'version': 'v20260920',
                'trained_date': '2026-09-20',
                'accuracy': 0.57,
                'auc_roc': 0.60
            },
            {
                'model_type': 'xgboost',
                'version': 'v20260918',
                'trained_date': '2026-09-18',
                'accuracy': 0.55,
                'auc_roc': 0.58
            }
        ])

        with patch('model_registry.list_model_versions', return_value=df_registry):
            with patch('model_registry.load_model_from_minio', return_value={'model_bytes': b'mock_weights'}):
                result = select_champion_model(
                    s3_hook=mock_s3,
                    bucket_name="stock-data",
                    model_type="xgboost",
                    current_version="v20260923",
                    current_acc=0.45,
                    current_auc=0.45,
                    max_lookback_days=7,
                    min_auc=0.495,
                    min_acc=0.50
                )
                self.assertEqual(result['status'], 'FALLBACK')
                self.assertEqual(result['selected_version'], 'v20260920')
                self.assertIsNotNone(result['artifacts'])

    def test_default_threshold_boundary(self):
        """Kiểm tra ngưỡng mặc định chính xác (AUC=0.495, Acc=0.50)."""
        mock_s3 = MagicMock()
        
        # Test boundary pass
        result = select_champion_model(
            s3_hook=mock_s3,
            bucket_name="stock-data",
            model_type="xgboost",
            current_version="v20260923",
            current_acc=0.50,
            current_auc=0.495
        )
        self.assertEqual(result['status'], 'USE_CURRENT')

        # Test boundary fail (either below fails with empty mocked registry)
        df_empty_registry = pd.DataFrame()
        with patch('model_registry.list_model_versions', return_value=df_empty_registry):
            result_fail = select_champion_model(
                s3_hook=mock_s3,
                bucket_name="stock-data",
                model_type="xgboost",
                current_version="v20260923",
                current_acc=0.494,  
                current_auc=0.495
            )
            self.assertEqual(result_fail['status'], 'SAFE_MODE')

            result_fail_2 = select_champion_model(
                s3_hook=mock_s3,
                bucket_name="stock-data",
                model_type="xgboost",
                current_version="v20260923",
                current_acc=0.50,
                current_auc=0.494  
            )
            self.assertEqual(result_fail_2['status'], 'SAFE_MODE')

    def test_safe_mode_when_no_valid_candidates(self):
        """Khi model mới trượt và không có model cũ nào đạt chuẩn trong lookback -> Kích hoạt SAFE_MODE."""
        mock_s3 = MagicMock()
        df_empty_registry = pd.DataFrame()

        with patch('model_registry.list_model_versions', return_value=df_empty_registry):
            result = select_champion_model(
                s3_hook=mock_s3,
                bucket_name="stock-data",
                model_type="xgboost",
                current_version="v20260923",
                current_acc=0.40,
                current_auc=0.42,
                max_lookback_days=7
            )
            self.assertEqual(result['status'], 'SAFE_MODE')
            self.assertIn("SAFE MODE", result['reason'])


if __name__ == '__main__':
    unittest.main()
