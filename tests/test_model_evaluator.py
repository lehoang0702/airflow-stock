import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../dags'))

from model_evaluator import (
    evaluate_binary_classifier,
    generate_classification_report_text,
    evaluate_direction_forecasts
)

class TestModelEvaluator(unittest.TestCase):
    def test_evaluate_binary_classifier(self):
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.85, 0.15])
        
        res = evaluate_binary_classifier(y_true, y_prob)
        self.assertEqual(res['accuracy'], 1.0)
        self.assertEqual(res['auc_roc'], 1.0)
        
        keys = ['accuracy', 'auc_roc', 'precision_class1', 'recall_class1', 'f1_score_class1', 
                'true_positives', 'false_positives', 'true_negatives', 'false_negatives']
        for k in keys:
            self.assertIn(k, res)
            
        y_true_edge = np.array([1, 1, 1])
        y_prob_edge = np.array([0.9, 0.8, 0.7])
        res_edge = evaluate_binary_classifier(y_true_edge, y_prob_edge)
        self.assertIsNone(res_edge['auc_roc'])

    def test_selective_confidence_evaluation(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.49, 0.51, 0.1])
        res = evaluate_binary_classifier(y_true, y_prob, confidence_margin=0.05)
        self.assertEqual(res['high_conviction_samples'], 2)
        self.assertEqual(res['high_conviction_accuracy'], 1.0)
        self.assertEqual(res['high_conviction_coverage'], 0.5)

    def test_prepare_target_with_horizon_and_threshold(self):
        from pipeline_data import prepare_target
        df = pd.DataFrame({
            'Ticker_Std': ['AAPL'] * 5,
            'Date_Std': pd.date_range('2023-01-01', periods=5),
            'Close': [100, 101, 102, 103, 104]
        })
        res_h2 = prepare_target(df, close_col='Close', horizon=2)
        self.assertEqual(res_h2['Target_Std'].iloc[0], 1.0)
        self.assertTrue(pd.isna(res_h2['Target_Std'].iloc[3]))
        
        res_thresh = prepare_target(df, close_col='Close', horizon=1, min_return_threshold=0.015)
        self.assertTrue(res_thresh['Target_Std'].isna().all())

    def test_generate_classification_report_text(self):
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.85, 0.15])
        report = generate_classification_report_text(y_true, y_prob)
        self.assertIn("CLASSIFICATION REPORT", report)
        self.assertIn("Giam (0)", report)
        self.assertIn("Tang (1)", report)

    def test_evaluate_direction_forecasts(self):
        res_empty = evaluate_direction_forecasts(pd.DataFrame(), pd.DataFrame())
        self.assertIsNone(res_empty['sentiment_accuracy'])
        
        predictions = pd.DataFrame([{'ma_co_phieu': 'AAPL', 'prob_num': 0.8,
            'reference_date': '2023-01-01', 'issued_at': '2023-01-02T01:00:00Z',
            'prediction_horizon': 5, 'inference_source': 'trained_model_v2'}])
        price_data = pd.DataFrame({
            'Date': pd.date_range(start='1/1/2023', periods=10),
            'Ticker': ['AAPL']*10,
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })
        res = evaluate_direction_forecasts(predictions, price_data, prediction_horizon=5)
        self.assertGreater(res['sentiment_accuracy'], 0.0)

if __name__ == '__main__':
    unittest.main()
