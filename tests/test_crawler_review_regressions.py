import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dags'))

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Test feature_engineering missing macro
from feature_engineering import calculate_technical_indicators

def test_feature_engineering_missing_macro():
    df_stock = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'open': np.random.rand(10),
        'high': np.random.rand(10),
        'low': np.random.rand(10),
        'close': np.random.rand(10),
        'volume': np.random.rand(10) * 1000
    })
    # missing QQQ
    df_macro = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'spy_log_ret': np.random.rand(10)
    })
    
    with pytest.raises(ValueError, match="Missing required macro features"):
        calculate_technical_indicators(df_stock, df_macro, 'AAPL', 'Tech')


def test_clean_df_normalizes_candle_bounds():
    with patch.dict('sys.modules', {
        'airflow': MagicMock(),
        'airflow.operators': MagicMock(),
        'airflow.operators.python': MagicMock(),
        'airflow.providers': MagicMock(),
        'airflow.providers.amazon': MagicMock(),
        'airflow.providers.amazon.aws': MagicMock(),
        'airflow.providers.amazon.aws.hooks': MagicMock(),
        'airflow.providers.amazon.aws.hooks.s3': MagicMock(),
    }):
        from dag_crawl_xgboost_stock_features import clean_df
    from data_validator import validate_xgboost_dataset

    # Simulate KO candle anomaly on 2026-10-19: Open: 92.56, High: 93.99, Low: 92.6222, Close: 93.435
    df_anomaly = pd.DataFrame({
        'Date': pd.date_range('2026-01-01', periods=35),
        'Open': [92.56] * 35,
        'High': [93.99] * 35,
        'Low': [92.6222] * 35,  # Low > Open!
        'Close': [93.435] * 35,
        'Volume': [10000.0] * 35
    })

    df_cleaned = clean_df(df_anomaly)
    # Check that Low was clamped to min(Low, Open, Close)
    assert (df_cleaned['low'] <= df_cleaned['open']).all()
    assert (df_cleaned['high'] >= df_cleaned['close']).all()
    assert df_cleaned['low'].iloc[0] == pytest.approx(92.56)

    # Check validator passes with cleaned dataframe
    qa = validate_xgboost_dataset(df_cleaned.rename(columns={
        'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    }).assign(Ticker='KO'))
    assert qa.is_valid, f"Validation failed: {qa.errors}"
