"""Independently replay saved model-improvement metrics and cohort constraints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dags'))
from research_improvements import select_generalizing_candidate


def verify(output, reference):
    report = json.loads((output / 'results.json').read_text())
    previous = json.loads((reference / 'results.json').read_text())
    if report['status'] != 'complete' or report['source'] != previous['source']:
        raise ValueError('Complete report and identical frozen source required')
    result, common = {'status': 'verified', 'models': {}, 'source': report['source']}, None
    for kind, model in report['models'].items():
        path = output / f'{kind}_predictions.csv'
        frame = pd.read_csv(path, parse_dates=['Date']).sort_values(['fold', 'Date', 'Ticker']).reset_index(drop=True)
        old = pd.read_csv(reference / f'{kind}_predictions.csv', parse_dates=['Date']).sort_values(['fold', 'Date', 'Ticker']).reset_index(drop=True)
        assert len(model['folds']) == report['protocol']['folds']
        if frame.duplicated(['Date', 'Ticker']).any():
            raise ValueError('Duplicate test observation')
        pd.testing.assert_frame_equal(frame[['Date', 'Ticker', 'y', 'fold']], old[['Date', 'Ticker', 'y', 'fold']])
        if common is None:
            common = frame[['Date', 'Ticker', 'y', 'fold']]
        else:
            pd.testing.assert_frame_equal(common, frame[['Date', 'Ticker', 'y', 'fold']])
        np.testing.assert_array_equal(frame.reference_probability, old.reference_probability)
        np.testing.assert_array_equal(frame.reference_label, old.reference_label)
        for fold in model['folds']:
            selected = select_generalizing_candidate(fold['candidates'],
                report['selection_policy']['min_validation_auc_gain'],
                report['selection_policy']['max_validation_logloss_increase'])
            assert fold['selected_index'] == selected
            assert fold['selection'] == fold['candidates'][selected]['spec']['name']
            assert pd.Timestamp(fold['stopping_split']['stopping_start']) < pd.Timestamp(fold['split']['validation_start'])
            part = frame[frame.fold.eq(fold['fold'])]
            for name, index in [('control', 0), ('selected', selected)]:
                threshold = fold['candidates'][index]['validation']['decision_threshold']
                np.testing.assert_array_equal(part[f'{name}_label'], (part[f'{name}_probability'] >= threshold).astype(int))
                np.testing.assert_allclose(accuracy_score(part.y, part[f'{name}_label']), fold[name]['accuracy'], atol=1e-10, rtol=0)
                np.testing.assert_allclose(roc_auc_score(part.y, part[f'{name}_probability']), fold[name]['auc_roc'], atol=1e-10, rtol=0)
        recomputed = {}
        for name in ('reference', 'control', 'selected', 'majority'):
            p, label = frame[f'{name}_probability'], frame[f'{name}_label']
            actual = {'accuracy': float(accuracy_score(frame.y, label)),
                      'accuracy_at_0_5': float(accuracy_score(frame.y, p >= .5)),
                      'balanced_accuracy': float(balanced_accuracy_score(frame.y, label)),
                      'auc_roc': float(roc_auc_score(frame.y, p)),
                      'logloss': float(log_loss(frame.y, p, labels=[0, 1])),
                      'predicted_positive_ratio': float(label.mean())}
            for metric, value in actual.items():
                np.testing.assert_allclose(value, model['summary'][name][metric], rtol=0,
                                           atol=1e-7 if metric == 'logloss' else 1e-10)
            recomputed[name] = actual
        result['models'][kind] = {'samples': len(frame), 'days': frame.Date.nunique(),
            'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'metrics': recomputed}
    result['checks'] = ['Identical source and reference probabilities', 'Identical paired labels and dates',
                        'No duplicate test observations', 'Selected recipe matches validation-only policy',
                        'Labels match frozen per-fold thresholds', 'CSV metrics match saved report']
    (output / 'verification.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=ROOT / 'output/walk_forward')
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.reference), indent=2, ensure_ascii=False))
