import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dags')))

from pipeline_data import rolling_temporal_masks

class TestTemporalTraining(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2020-01-01', periods=1000, freq='B')
        tickers = ['AAPL', 'MSFT']
        dfs = []
        for t in tickers:
            df_t = pd.DataFrame({
                'Date_Std': dates,
                'Ticker_Std': t,
                'Target_Std': (np.arange(len(dates)) % 2).astype(float)
            })
            dfs.append(df_t)
        self.df = pd.concat(dfs, ignore_index=True)

    def test_disjointness(self):
        train_mask, val_mask, test_mask, meta = rolling_temporal_masks(
            self.df, train_years=1, validation_sessions=20, test_sessions=40, horizon=1
        )
        self.assertFalse((train_mask & val_mask).any())
        self.assertFalse((train_mask & test_mask).any())
        self.assertFalse((val_mask & test_mask).any())
        
    def test_purge_horizon(self):
        train_m1, val_m1, test_m1, meta1 = rolling_temporal_masks(
            self.df, train_years=1, validation_sessions=20, test_sessions=40, horizon=1
        )
        train_m5, val_m5, test_m5, meta5 = rolling_temporal_masks(
            self.df, train_years=1, validation_sessions=20, test_sessions=40, horizon=5
        )
        self.assertLess(train_m5.sum(), train_m1.sum())
        
        val_start = pd.Timestamp(meta5['validation_start'])
        df_sorted = self.df.sort_values(['Ticker_Std', 'Date_Std'])
        df_sorted['outcome_date'] = df_sorted.groupby('Ticker_Std')['Date_Std'].shift(-5)
        train_rows = df_sorted.loc[train_m5]
        self.assertTrue((train_rows['outcome_date'] < val_start).all())

    def test_purge_irregular_missing_dates(self):
        df_drop = self.df.drop(self.df.index[500:510]).reset_index(drop=True)
        train_m, val_m, test_m, meta = rolling_temporal_masks(
            df_drop, train_years=1, validation_sessions=20, test_sessions=40, horizon=5
        )
        val_start = pd.Timestamp(meta['validation_start'])
        test_start = pd.Timestamp(meta['test_start'])
        df_sorted = df_drop.sort_values(['Ticker_Std', 'Date_Std'])
        df_sorted['outcome_date'] = df_sorted.groupby('Ticker_Std')['Date_Std'].shift(-5)
        self.assertTrue((df_sorted.loc[train_m, 'outcome_date'] < val_start).all())
        self.assertTrue((df_sorted.loc[val_m, 'outcome_date'] < test_start).all())

    def test_multi_ticker_shuffled(self):
        shuffled = self.df.sample(frac=1.0, random_state=42)
        train_mask, val_mask, test_mask, meta = rolling_temporal_masks(
            shuffled, train_years=1, validation_sessions=20, test_sessions=40, horizon=1
        )
        self.assertTrue(train_mask.any())
        self.assertTrue(val_mask.any())
        self.assertTrue(test_mask.any())
        self.assertFalse((train_mask & val_mask).any())

    def test_insufficient_history(self):
        short_df = self.df[self.df['Date_Std'] < '2020-01-15']
        with self.assertRaises(ValueError):
            rolling_temporal_masks(short_df, train_years=1, validation_sessions=20, test_sessions=40)

    def test_replay_with_extra_recent_rows_excludes_outside_test_end(self):
        train_mask, val_mask, test_mask, meta = rolling_temporal_masks(
            self.df, train_years=1, validation_sessions=20, test_sessions=40, horizon=1
        )
        
        latest_date = self.df['Date_Std'].max()
        future_dates = pd.date_range(latest_date + pd.Timedelta(days=1), periods=10, freq='B')
        extra_df = pd.DataFrame({
            'Date_Std': list(future_dates) * 2,
            'Ticker_Std': ['AAPL'] * 10 + ['MSFT'] * 10,
            'Target_Std': [1.0] * 20
        })
        extended_df = pd.concat([self.df, extra_df], ignore_index=True)
        
        re_train, re_val, re_test, re_meta = rolling_temporal_masks(extended_df, split_metadata=meta)
        
        test_end = pd.Timestamp(meta['test_end'])
        re_test_dates = extended_df.loc[re_test, 'Date_Std']
        self.assertTrue((re_test_dates <= test_end).all())
        
        df_sorted = extended_df.sort_values(['Ticker_Std', 'Date_Std'])
        df_sorted['out_dt'] = df_sorted.groupby('Ticker_Std')['Date_Std'].shift(-1)
        re_test_outcomes = df_sorted.loc[re_test, 'out_dt']
        self.assertTrue((re_test_outcomes <= test_end).all())
        
        extra_indices = extended_df.index[-20:]
        self.assertFalse(re_train.loc[extra_indices].any())
        self.assertFalse(re_val.loc[extra_indices].any())
        self.assertFalse(re_test.loc[extra_indices].any())

    def test_invalid_replay(self):
        meta = {
            'split_policy': 'rolling_v1',
            'train_start': '2020-01-01',
            'validation_start': '2019-01-01',
            'test_start': '2020-06-01',
            'test_end': '2020-12-31',
            'prediction_horizon': 1
        }
        with self.assertRaises(ValueError):
            rolling_temporal_masks(self.df, split_metadata=meta)

    def test_unlabeled_tails(self):
        df_nan = self.df.copy()
        df_nan.loc[df_nan.index % 5 == 0, 'Target_Std'] = np.nan
        train_mask, val_mask, test_mask, meta = rolling_temporal_masks(
            df_nan, train_years=1, validation_sessions=20, test_sessions=40, horizon=1
        )
        self.assertFalse(df_nan.loc[train_mask, 'Target_Std'].isna().any())
        self.assertFalse(df_nan.loc[val_mask, 'Target_Std'].isna().any())
        self.assertFalse(df_nan.loc[test_mask, 'Target_Std'].isna().any())

    def test_invalid_counts_horizons(self):
        with self.assertRaises(ValueError):
            rolling_temporal_masks(self.df, horizon=0)
        with self.assertRaises(ValueError):
            rolling_temporal_masks(self.df, horizon=False)
        with self.assertRaises(ValueError):
            rolling_temporal_masks(self.df, train_years=-1)
        with self.assertRaises(ValueError):
            rolling_temporal_masks(self.df, validation_sessions=0)
        with self.assertRaises(ValueError):
            rolling_temporal_masks(self.df, test_sessions=-5)

    def test_invalid_dates(self):
        df_invalid = self.df.copy()
        df_invalid.loc[0, 'Date_Std'] = pd.NaT
        with self.assertRaises(ValueError):
            rolling_temporal_masks(df_invalid)
            
    def test_null_ticker(self):
        df_invalid = self.df.copy()
        df_invalid.loc[0, 'Ticker_Std'] = ''
        with self.assertRaises(ValueError):
            rolling_temporal_masks(df_invalid)

    def test_duplicate_ticker_date(self):
        df_dup = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            rolling_temporal_masks(df_dup)

if __name__ == '__main__':
    unittest.main()
