import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../dags'))

from data_validator import (
    DataQualityResult,
    validate_xgboost_dataset,
    validate_lstm_dataset,
    validate_finbert_dataset
)

class TestDataValidator(unittest.TestCase):
    def test_data_quality_result(self):
        qa = DataQualityResult("test_dataset")
        self.assertTrue(qa.is_valid)
        self.assertEqual(qa.errors, [])
        
        qa.add_warning("warn1")
        self.assertEqual(len(qa.warnings), 1)
        self.assertTrue(qa.is_valid)
        
        qa.add_error("err1")
        self.assertEqual(len(qa.errors), 1)
        self.assertFalse(qa.is_valid)
        
        d = qa.to_dict()
        self.assertEqual(d["dataset_name"], "test_dataset")
        self.assertFalse(d["is_valid"])
        self.assertIn("total_rows", d)
        self.assertIn("error_count", d)
        self.assertIn("warning_count", d)
        self.assertIn("errors", d)
        self.assertIn("warnings", d)
        
    def test_validate_xgboost_dataset(self):
        qa = validate_xgboost_dataset(pd.DataFrame())
        self.assertFalse(qa.is_valid)
        self.assertGreater(len(qa.errors), 0)
        
        df_missing = pd.DataFrame({'Date': ['2023-01-01'], 'Ticker': ['AAPL']})
        qa_missing = validate_xgboost_dataset(df_missing)
        self.assertFalse(qa_missing.is_valid)
        
        df_invalid = pd.DataFrame({
            'Date': ['2023-01-01'], 'Ticker': ['AAPL'], 
            'Open': [100], 'High': [90], 'Low': [110], 'Close': [-1], 'Volume': [1000]
        })
        qa_invalid = validate_xgboost_dataset(df_invalid)
        self.assertFalse(qa_invalid.is_valid)
        
        df_valid = pd.DataFrame({
            'Date': ['2023-01-01'], 'Ticker': ['AAPL'], 
            'Open': [100], 'High': [110], 'Low': [90], 'Close': [105], 'Volume': [1000]
        })
        qa_valid = validate_xgboost_dataset(df_valid)
        self.assertTrue(qa_valid.is_valid)

    def test_validate_lstm_dataset(self):
        qa = validate_lstm_dataset(pd.DataFrame())
        self.assertFalse(qa.is_valid)
        
        df_missing = pd.DataFrame({'Date': ['2023-01-01']})
        qa_missing = validate_lstm_dataset(df_missing)
        self.assertFalse(qa_missing.is_valid)
        
        df_short = pd.DataFrame({
            'Date': pd.date_range(start='1/1/2023', periods=10),
            'Ticker': ['AAPL']*10,
            'Open': [100]*10, 'High': [110]*10, 'Low': [90]*10, 'Close': [105]*10, 'Volume': [1000]*10
        })
        qa_short = validate_lstm_dataset(df_short, min_lookback=30)
        self.assertFalse(qa_short.is_valid)
        
        df_valid = pd.DataFrame({
            'Date': pd.date_range(start='1/1/2023', periods=35),
            'Ticker': ['AAPL']*35,
            'Open': [100]*35, 'High': [110]*35, 'Low': [90]*35, 'Close': [105]*35, 'Volume': [1000]*35
        })
        qa_valid = validate_lstm_dataset(df_valid, min_lookback=30)
        self.assertTrue(qa_valid.is_valid)

    def test_validate_finbert_dataset(self):
        qa = validate_finbert_dataset(pd.DataFrame())
        self.assertFalse(qa.is_valid)
        
        df_valid = pd.DataFrame({'ticker': ['AAPL'], 'headline': ['Good news']})
        qa_valid = validate_finbert_dataset(df_valid)
        self.assertTrue(qa_valid.is_valid)

if __name__ == '__main__':
    unittest.main()
