"""Recompute all paper metrics from stored test predictions; no retraining."""
import argparse, collections, itertools, math
from pathlib import Path
import numpy as np
from scipy.stats import t
from config import METHODS, EXPERIMENTS
from common import METRICS, read_json, read_csv, write_csv, write_json, metric_values, predictions_and_metrics
from run_experiments import validate_prepared

REPORT_METRICS = METRICS + (
    'training_seconds',
    'total_search_seconds',
    'inference_ms_per_image',
    'total_parameters',
    'trainable_parameters',
    'best_epoch',
    'epochs_run'
)


def corrected(values, ratio, alpha):
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    se = math.sqrt((1 / n + ratio) * sd * sd)
    crit = float(t.ppf(1 - alpha / 2, n - 1))
    # Zero observed variance does not establish certainty; inference undefined in this case.
    if se == 0:
        return dict(
            mean=mean,
            sd=sd,
            corrected_se=se,
            ci_low=None,
            ci_high=None,
            t_stat=None,
            p_value=None,
            n=n
        )

    stat = mean / se
    pv = float(2 * t.sf(abs(stat), n - 1))

    return dict(
        mean=mean,
        sd=sd,
        corrected_se=se,
        ci_low=mean - crit * se,
        ci_high=mean + crit * se,
        t_stat=stat,
        p_value=pv,
        n=n
    )


def holm(group, alpha):
    # Treat any undefined tests as non-rejections while counting all predeclared comparisons.
    ranked = sorted(group, key=lambda r: r['p_value'] if r['p_value'] is not None else 1.)
    running = 0.
    m = len(ranked)

    for i, r in enumerate(ranked):
        if r['p_value'] is None:
            r['holm_p'] = None
            r['reject'] = False
            continue
        running = max(running, min(1., (m - i) * r['p_value']))
        r['holm_p'] = running
        r['reject'] = running < alpha


def analyze(out, check_code=True):
    q, root, mapping, folds = validate_prepared(out, verify_files=False, check_code=check_code)
    p = q['settings']
    seeds = p['seeds']
    nf = p['folds']
    alpha = p['alpha']
    dest = out / 'analysis'
    dest.mkdir(exist_ok=True)
    missing = []
    results = {}
    predictions = {}
    candidate_rows = []

    for exp, method, seed, fold in itertools.product(EXPERIMENTS, METHODS, seeds, range(1, nf + 1)):
        folder = out / 'runs' / exp / f'seed_{seed}' / f'fold_{fold:02d}' / method
        if not (folder / 'result.json').is_file():
            missing.append(str(folder.relative_to(out)))
            continue
        r = read_json(folder / 'result.json')
        pred = read_csv(folder / 'test' / 'predictions.csv')
        expected = folds[seed, fold]['test']
        key = 'original_type' if exp == 'original' else 'merged_type'
        classes = mapping[exp]
        ix = {c: i for i, c in enumerate(classes)}
        if (r['experiment'], r['method'], r['seed'], r['fold']) != (exp, method, seed, fold):
            raise ValueError('Incorrect result identity')
        if [(
            x['coin_id'],
            x['path'],
            x['original_type'],
            x['true_label']
        ) for x in pred] != [(x['coin_id'], x['path'], x['original_type'], x[key]) for x in expected]:
            raise ValueError(f'Wrong test rows: {folder}')

        metrics = metric_values(
            [ix[x['true_label']] for x in pred],
            [ix[x['predicted_label']] for x in pred],
            len(classes)
        )
        for m, v in metrics.items():
            if abs(v - r[m]) > 1e-12:
                raise ValueError(f'Metric/prediction mismatch: {folder} {m}')
        sel = read_json(folder / 'selection.json')
        chosen = max(sel['candidates'], key=lambda x: x['validation']['macro_f1'])['candidate']
        if chosen != r['selected_candidate'] or chosen != sel['selected']:
            raise ValueError(f'Invalid validation selection: {folder}')
        results[exp, method, seed, fold] = r
        predictions[exp, method, seed, fold] = pred

        for c in sel['candidates']:
            candidate_rows.append(dict(
                experiment=exp,
                method=method,
                seed=seed,
                fold=fold,
                candidate=c['candidate'],
                selected=c['candidate'] == chosen,
                validation_accuracy=c['validation']['accuracy'],
                validation_macro_f1=c['validation']['macro_f1'],
                training_seconds=c['training_seconds'],
                epochs_run=c['epochs_run'],
                best_epoch=c['best_epoch'],
                total_parameters=c['total_parameters'],
                trainable_parameters=c['trainable_parameters']
            ))
    if missing:
        write_json(dest / 'missing_runs.json', missing)
        raise RuntimeError(f'{len(missing)} of {len(EXPERIMENTS)*len(METHODS)*len(seeds)*nf} runs missing. See analysis/missing_runs.json; resume training before final statistics.')
    (dest / 'missing_runs.json').unlink(missing_ok=True)
    write_csv(dest / 'all_fold_results.csv', results.values())
    write_csv(dest / 'all_candidate_validation_results.csv', candidate_rows)
    summaries = []
    per_seed = []
    tests = []
    selected_counts = []
    all_pred = []
    per_class_seed = []

    for exp in EXPERIMENTS:
        classes = mapping[exp]
        ix = {c: i for i, c in enumerate(classes)}
        ratio = float(np.mean([results[exp, METHODS[0], s, f]['n_test'] / results[exp, METHODS[0], s, f]['n_train'] for s in seeds for f in range(1, nf + 1)]))
        keys = [(s, f) for s in seeds for f in range(1, nf + 1)]

        for method in METHODS:
            rs = [results[exp, method, s, f] for s, f in keys]
            for m in REPORT_METRICS:
                vals = [r[m] for r in rs if r.get(m) is not None]
                if not vals:
                    continue
                if m in METRICS:
                    z = corrected(vals, ratio, alpha)
                    low = None if z['ci_low'] is None else max(0., z['ci_low'])
                    high = None if z['ci_high'] is None else min(1., z['ci_high'])
                else:
                    z = {
                        'mean': float(np.mean(vals)),
                        'sd': float(np.std(vals, ddof=1)),
                        'n': len(vals)
                    }
                    low = high = None

                summaries.append(dict(
                    experiment=exp,
                    method=method,
                    metric=m,
                    mean=z['mean'],
                    sd=z['sd'],
                    ci95_low=low,
                    ci95_high=high,
                    n=z['n']
                ))

            counts = collections.Counter(r['selected_candidate'] for r in rs)
            selected_counts.extend(
                dict(experiment=exp, method=method, candidate=c, count=n) for c, n in sorted(counts.items())
            )
            pooled = []

            for s in seeds:
                rows = [row for f in range(1, nf + 1) for row in predictions[exp, method, s, f]]
                if len({r['coin_id'] for r in rows}) != p['n_classes'] * p['samples_per_class'] or len(rows) != p['n_classes'] * p['samples_per_class']:
                    raise ValueError('OOF coin count mismatch')
                y = [ix[r['true_label']] for r in rows]
                pred = [ix[r['predicted_label']] for r in rows]
                rr = [dict(r, label=r['true_label']) for r in rows]
                oof_dir = dest / exp / method / f'seed_{s}_oof'
                z = predictions_and_metrics(oof_dir, rr, classes, y, pred)
                per_seed.append(dict(experiment=exp, method=method, seed=s, **z))
                for cr in read_csv(oof_dir / 'per_class.csv'):
                    per_class_seed.append(dict(experiment=exp, method=method, seed=s, **cr))
                pooled.extend(rr)

                for f in range(1, nf + 1):
                    all_pred.extend(dict(
                        experiment=exp,
                        method=method,
                        seed=s,
                        fold=f,
                        **row
                    ) for row in predictions[exp, method, s, f])

            # Counts pooled over three repeats; NOT independent coins; only descriptive.
            predictions_and_metrics(
                dest / exp / method / 'pooled_descriptive',
                pooled,
                classes,
                [ix[r['true_label']] for r in pooled],
                [ix[r['predicted_label']] for r in pooled]
            )

        for m in ('accuracy', 'macro_f1'):
            family = []
            for a, b in itertools.combinations(METHODS, 2):
                diff = [results[exp, a, s, f][m] - results[exp, b, s, f][m] for s, f in keys]
                z = corrected(diff, ratio, alpha)
                family.append(dict(
                    experiment=exp,
                    metric=m,
                    method_a=a,
                    method_b=b,
                    n=z['n'],
                    mean_difference=z['mean'],
                    sd_difference=z['sd'],
                    corrected_se=z['corrected_se'],
                    ci95_low=z['ci_low'],
                    ci95_high=z['ci_high'],
                    t_stat=z['t_stat'],
                    p_value=z['p_value'],
                    test_train_ratio=ratio
                ))

            holm(family, alpha)
            tests.extend(family)

    write_csv(dest / 'summary_mean_sd_ci.csv', summaries)
    write_csv(dest / 'paired_corrected_tests_holm.csv', tests)
    write_csv(dest / 'selection_counts.csv', selected_counts)
    write_csv(dest / 'oof_metrics_by_seed.csv', per_seed)
    write_csv(dest / 'all_test_predictions.csv', all_pred)
    write_csv(dest / 'per_class_by_seed.csv', per_class_seed)
    groups = collections.defaultdict(list)
    for r in per_class_seed:
        groups[r['experiment'], r['method'], r['label']].append(r)
    pc = []

    for (exp, method, label), rs in groups.items():
        row = dict(experiment=exp, method=method, label=label, support_per_seed=int(rs[0]['support']))
        for m in ('precision', 'recall', 'f1'):
            vals = [float(r[m]) for r in rs]
            row[m + '_mean'] = float(np.mean(vals))
            row[m + '_sd'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        pc.append(row)

    write_csv(dest / 'per_class_mean_sd.csv', pc)
    write_json(
        dest / 'statistics_notes.json',
        {
            'performance_units': 'Fractions 0-1, not percent. Differences in percentage points = fraction * 100.',
            'descriptive_mean_sd': 'Mean and sample SD over 15 fold scores, not independent replicates.',
            'ci': 'Approximate corrected resampled t interval using SE=sqrt((1/n + n_test/n_train)*s^2), df=n-1. n_train excludes validation (conservative approximation). Marginal metric intervals clipped to [0,1]. Zero variance: interval/test not reported.',
            'paired_tests': 'Nadeau-Bengio-style corrected paired t tests using identical fold/seed comparisons, six pairs per experiment per metric; Holm adjustment separately for accuracy (primary) and macro F1 (secondary) within each experiment. Unadjusted confidence intervals for differences; exploratory/approximate, not proof.',
            'pooled': 'Pooled confusion counts use each coin once per repeat, three times overall; descriptive only. OOF results for each seed also saved.',
            'timing': 'Wall-clock includes data loading, training, validation and best-weight revalidation, excludes model construction/download/checkpoint writes (history writes included). SIFT selected time includes the shared feature/vocabulary stage and selected fit, total search counts shared work once. Neural test inference includes loading, preprocessing, transfer, forward, predicted labels; validation batch warmup excluded. Siamese uses saved training-only centroids. SIFT runs on CPU; neural timings depend on selected device.',
            'comparison_between_experiments': 'Original and merged label spaces have different difficulty/priors. Do not interpret improved accuracy as same-task model improvement.',
            'scope': 'Conditional on the selected dataset; uncertainty not independent external validation.'
        }
    )
    print(f'Analysis complete: {len(results)} selected test runs; {len(candidate_rows)} candidates. Results: {dest}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    analyze(Path(a.output).resolve())
if __name__ == '__main__':
    main()
