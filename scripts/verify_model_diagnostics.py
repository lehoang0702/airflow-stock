"""Independently replay saved diagnostic metrics and check paired cohorts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def verify(output, reference):
    report = json.loads((output / 'diagnostics.json').read_text())
    previous = json.loads((reference / 'results.json').read_text())
    if report['status'] != 'complete' or report['source'] != previous['source']:
        raise ValueError('Complete diagnostic report and identical frozen price source required')
    results, cohorts = {}, {}
    protocol = report['protocol']
    for kind, model in report['models'].items():
        if len(model['records']) != protocol['folds'] * len(protocol['fractions']):
            raise ValueError('Missing diagnostic records')
        frames = []
        for fold in range(1, protocol['folds'] + 1):
            frame = pd.read_csv(output / f'{kind}_fold_{fold}_predictions.csv')
            for fraction in protocol['fractions']:
                part = frame[frame.train_fraction.eq(fraction)].sort_values(['Date_Std', 'Ticker_Std'])
                record = next(r for r in model['records'] if r['fold'] == fold and r['spec']['train_fraction'] == fraction)
                cohort = part[['Date_Std', 'Ticker_Std', 'Target_Std']].reset_index(drop=True)
                if fold not in cohorts:
                    cohorts[fold] = cohort
                pd.testing.assert_frame_equal(cohort, cohorts[fold])
                if part.duplicated(['Date_Std', 'Ticker_Std']).any():
                    raise ValueError('Duplicate held-out observation')
                threshold = record['validation']['decision_threshold']
                if not np.array_equal(part.prediction, (part.probability >= threshold).astype(int)):
                    raise ValueError('Prediction labels do not match frozen validation threshold')
                computed = {'accuracy': accuracy_score(part.Target_Std, part.prediction),
                            'auc_roc': roc_auc_score(part.Target_Std, part.probability),
                            'logloss': log_loss(part.Target_Std, part.probability, labels=[0, 1])}
                for metric, value in computed.items():
                    # Training computes log loss with float32 probabilities;
                    # CSV replay parses float64 and may differ in accumulation.
                    tolerance = 1e-7 if metric == 'logloss' else 1e-10
                    np.testing.assert_allclose(value, record['test'][metric], rtol=0, atol=tolerance)
                assert len(part) == record['test']['samples']
                assert pd.Timestamp(record['training_period']['end']) < pd.Timestamp(record['split']['validation_start'])
                for split in ('training', 'validation', 'test'):
                    bins = record['probability_diagnostics'][split]['calibration_bins']
                    assert sum(b['samples'] for b in bins) == record[split]['samples']
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        by_fraction = {}
        for fraction in protocol['fractions']:
            part = combined[combined.train_fraction.eq(fraction)]
            records = [r for r in model['records'] if r['spec']['train_fraction'] == fraction]
            by_fraction[str(fraction)] = {
                'samples': len(part), 'pooled_accuracy': float(accuracy_score(part.Target_Std, part.prediction)),
                'pooled_auc': float(roc_auc_score(part.Target_Std, part.probability)),
                'mean_fold_auc': {s: float(np.mean([r[s]['auc_roc'] for r in records]))
                                  for s in ('training', 'validation', 'test')}}
        old = pd.read_csv(reference / f'{kind}_predictions.csv').sort_values(['fold', 'Date', 'Ticker'])
        full = combined[combined.train_fraction.eq(1)].sort_values(['fold', 'Date_Std', 'Ticker_Std'])
        pd.testing.assert_frame_equal(full[['fold', 'Date_Std', 'Ticker_Std', 'Target_Std']].reset_index(drop=True),
            old[['fold', 'Date', 'Ticker', 'y']].rename(columns={'Date': 'Date_Std', 'Ticker': 'Ticker_Std', 'y': 'Target_Std'}).reset_index(drop=True), check_dtype=False)
        difference = np.abs(full.probability.to_numpy() - old.reference_probability.to_numpy())
        np.testing.assert_allclose(full.probability, old.reference_probability, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(full.prediction, old.reference_label)
        results[kind] = {'fractions': by_fraction,
                         'reference_max_probability_difference': float(difference.max()),
                         'reference_labels_match': True}
    result = {'status': 'verified', 'source': report['source'], 'models': results,
              'metric_tolerances': {'accuracy': 1e-10, 'auc_roc': 1e-10, 'logloss': 1e-7},
              'checks': ['Same held-out cohorts across fractions and models', 'Saved test metrics independently replayed',
                         'Frozen validation thresholds match labels', 'Calibration bins preserve every observation',
                         'Full-data diagnostic predictions reproduce the prior reference']}
    (output / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/model_diagnostics'))
    parser.add_argument('--reference', type=Path, default=Path('output/walk_forward'))
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.reference), indent=2))
