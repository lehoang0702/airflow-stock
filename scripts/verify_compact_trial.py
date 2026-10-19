"""Independently verify trial labels, frozen decisions and reported metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dags'))
from research_improvements import select_generalizing_candidate


def verify(output, snapshot):
    report = json.loads((output / 'results.json').read_text())
    manifest = json.loads((output / 'selection_manifest.json').read_text())
    assert report['status'] == 'complete' and manifest['status'] == 'selections_frozen'
    assert report['selection_manifest_sha256'] == hashlib.sha256((output / 'selection_manifest.json').read_bytes()).hexdigest()
    assert report['source']['sha256'] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    for path, digest in report['runtime']['source_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    frame = pd.read_csv(output / 'predictions.csv', parse_dates=['Date'])
    assert not frame.duplicated(['Date', 'Ticker']).any()
    assert frame.fold.nunique() == report['protocol']['folds']
    raw = pd.read_csv(snapshot, parse_dates=['Date'])
    raw = raw[raw.Date.le(report['source']['last_allowed_outcome'])].sort_values(['Ticker', 'Date'])
    raw['outcome_date'] = raw.groupby('Ticker').Date.shift(-1)
    raw['future_close'] = raw.groupby('Ticker').Close.shift(-1)
    raw['y'] = (raw.future_close > raw.Close).astype(float).where(raw.future_close.notna())
    paired = frame.merge(raw[['Date', 'Ticker', 'y', 'outcome_date']], on=['Date', 'Ticker'],
                         how='left', validate='one_to_one', suffixes=('', '_raw'))
    np.testing.assert_array_equal(paired.y, paired.y_raw)
    names = report['protocol']['candidate_names']
    for fold, frozen in zip(report['folds'], manifest['folds']):
        assert 'test' not in frozen
        for key in ('selection', 'selected_index', 'candidates', 'reference', 'split', 'train_prior'):
            assert fold[key] == frozen[key], key
        expected = select_generalizing_candidate(fold['candidates'],
            report['protocol']['min_validation_auc_gain'], report['protocol']['max_validation_logloss_increase'])
        assert fold['selected_index'] == expected
        assert fold['selection'] == fold['candidates'][expected]['spec']['name']
        group = paired[paired.fold.eq(fold['fold'])].copy()
        meta = fold['split']
        cohort = raw[raw.Date.ge(meta['test_start']) & raw.Date.le(meta['test_end'])
                     & raw.outcome_date.le(meta['test_end']) & raw.y.notna()]
        pd.testing.assert_frame_equal(group[['Date', 'Ticker']].sort_values(['Date', 'Ticker']).reset_index(drop=True),
            cohort[['Date', 'Ticker']].sort_values(['Date', 'Ticker']).reset_index(drop=True))
        assert group.Date.nunique() == report['protocol']['test_sessions']
        np.testing.assert_array_equal(group.selected_probability, group[f"{fold['selection']}_probability"])
        np.testing.assert_array_equal(group.selected_label, group[f"{fold['selection']}_label"])
        for name in names:
            p = group[f'{name}_probability'].to_numpy()
            threshold = fold['test'][name]['decision_threshold']
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            np.testing.assert_array_equal(group[f'{name}_label'], p >= threshold)
            if name != 'majority':
                assert confusion_matrix(group.y, p >= threshold, labels=[0, 1]).tolist() == fold['test'][name]['diagnostics']['confusion_matrix']
    recomputed = {}
    for section, group in [('summary', frame), ('recent_two_folds', frame[frame.fold.gt(report['protocol']['folds'] - 2)])]:
        recomputed[section] = {}
        for name in names:
            p, labels = group[f'{name}_probability'], group[f'{name}_label']
            actual = {'accuracy': float(accuracy_score(group.y, labels)),
                'accuracy_at_0_5': float(accuracy_score(group.y, p >= .5)),
                'balanced_accuracy': float(balanced_accuracy_score(group.y, labels)),
                'auc_roc': float(roc_auc_score(group.y, p)),
                'logloss': float(log_loss(group.y, p, labels=[0, 1])),
                'predicted_positive_ratio': float(labels.mean()),
                'fold_auc_mean': float(np.mean([roc_auc_score(f.y, f[f'{name}_probability']) for _, f in group.groupby('fold')]))}
            for key, value in actual.items():
                np.testing.assert_allclose(value, report[section][name][key], atol=1e-7 if key == 'logloss' else 1e-10, rtol=0)
            recomputed[section][name] = actual
    result = {'status': 'verified', 'samples': len(frame), 'sessions': frame.Date.nunique(),
        'tickers': frame.Ticker.nunique(), 'first_feature_date': str(frame.Date.min().date()),
        'last_feature_date': str(frame.Date.max().date()), 'last_outcome_date': str(paired.outcome_date.max().date()),
        'prediction_sha256': hashlib.sha256((output / 'predictions.csv').read_bytes()).hexdigest(),
        'checks': ['Frozen source and source code hashes', 'Raw-price T+1 labels independently reconstructed',
            'Every test row retained and no overlapping test windows', 'Choices frozen before test scoring',
            'Validation-only selection replayed', 'Threshold labels and confusion matrices match',
            'All aggregate and recent-window metrics independently recomputed'], 'metrics': recomputed}
    (output / 'verification.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.output, args.snapshot)
    print(json.dumps({key: value for key, value in result.items() if key != 'metrics'}, indent=2))
