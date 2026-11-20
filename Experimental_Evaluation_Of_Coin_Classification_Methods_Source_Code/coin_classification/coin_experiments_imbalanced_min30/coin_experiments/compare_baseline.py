"""Compare expanded and original predictions on the identical 840 anchor coins."""
import argparse, collections, itertools
from pathlib import Path
import numpy as np
from common import read_csv, read_json, write_csv, write_json, digest, metric_values, METRICS
from config import METHODS, EXPERIMENTS, PROTOCOL
from analyze_results import corrected, holm


def compare(base, out):
    base, out = Path(base), Path(out)
    q = read_json(out / 'protocol.json')
    mapping = read_json(out / 'label_mapping.json')
    ids = {r['coin_id'] for r in read_csv(base / 'manifest.csv')}
    if digest(base / 'manifest.csv') != q['baseline_manifest_sha256'] or digest(base / 'splits.csv') != q['baseline_splits_sha256'] or len(ids) != 840:
        raise ValueError('Wrong baseline')
    def read(folder):
        groups = collections.defaultdict(dict)
        for r in read_csv(folder / 'analysis' / 'all_test_predictions.csv'):
            if r['coin_id'] not in ids:
                continue
            key = (r['experiment'], r['method'], int(r['seed']), int(r['fold']))
            if r['coin_id'] in groups[key]:
                raise ValueError('Duplicated anchor prediction')
            groups[key][r['coin_id']] = r
        return groups
    a, b = read(base), read(out)
    foldrows = []
    summary = []
    tests = []
    keys = list(itertools.product(PROTOCOL['seeds'], range(1, 6)))

    for exp in EXPERIMENTS:
        ix = {c: i for i, c in enumerate(mapping[exp])}
        for method in METHODS:
            pairs = []
            for seed, fold in keys:
                k = (exp, method, seed, fold)
                aa, bb = a[k], b[k]
                coins = sorted(aa)
                if set(aa) != set(bb) or len(aa) != 168:
                    raise ValueError('Different anchor test members: ' + str(k))
                if any((aa[c]['true_label'], aa[c]['path']) != (bb[c]['true_label'], bb[c]['path']) for c in coins):
                    raise ValueError('Different labels or images')
                y = [ix[aa[c]['true_label']] for c in coins]
                x = metric_values(y, [ix[aa[c]['predicted_label']] for c in coins], len(ix))
                z = metric_values(y, [ix[bb[c]['predicted_label']] for c in coins], len(ix))
                pairs.append((x, z))
                foldrows.extend(dict(
                    experiment=exp,
                    method=method,
                    seed=seed,
                    fold=fold,
                    metric=m,
                    balanced=x[m],
                    expanded=z[m],
                    difference=z[m] - x[m]
                ) for m in METRICS)
            for m in METRICS:
                x = np.array([a[m] for a, b in pairs])
                z = np.array([b[m] for a, b in pairs])
                d = z - x
                summary.append(dict(
                    experiment=exp,
                    method=method,
                    metric=m,
                    balanced_mean=x.mean(),
                    balanced_sd=x.std(ddof=1),
                    expanded_mean=z.mean(),
                    expanded_sd=z.std(ddof=1),
                    difference_mean=d.mean(),
                    difference_sd=d.std(ddof=1),
                    n=len(d)
                ))
                if m in ('accuracy', 'macro_f1'):
                    test = corrected(d, 168 / 588, PROTOCOL['alpha'])
                    tests.append(dict(experiment=exp, method=method, metric=m, **test))
        for m in ('accuracy', 'macro_f1'):
            holm([r for r in tests if r['experiment'] == exp and r['metric'] == m], PROTOCOL['alpha'])

    dest = out / 'analysis'
    write_csv(dest / 'anchor_fold_comparison.csv', foldrows)
    write_csv(dest / 'anchor_summary_comparison.csv', summary)
    write_csv(dest / 'anchor_paired_tests_holm.csv', tests)
    write_json(
        dest / 'anchor_comparison_notes.json',
        {
            'scope': 'Same 840 Coin IDs, same seed/fold test assignments; expanded training/validation has additional naturally distributed coins. Difference combines training size and class distribution, not isolated imbalance. Full expanded OOF is a separate evaluation.',
            'tests': 'Exploratory corrected paired t tests using baseline test/train ratio 168/588; Holm over four methods per metric and label space. No independent external test.',
            'units': 'Fractions 0-1; difference expanded minus balanced.'
        }
    )
    print('PASS: same 840 anchor coins, same folds; comparison files saved.')
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    compare(a.baseline, a.output)
