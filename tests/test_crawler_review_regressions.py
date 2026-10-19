import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Test feature_engineering missing macro
from dags.feature_engineering import calculate_technical_indicators

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

# We can also mock airflow and test the crawler DAG functions directly
# But they require context and hooks.
