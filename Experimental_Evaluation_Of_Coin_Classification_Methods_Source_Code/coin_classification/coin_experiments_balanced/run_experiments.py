"""Run fixed protocol; resume completed candidates/folds; one process per output folder."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, collections, importlib, time, sys, socket, contextlib
from pathlib import Path
import numpy as np
from config import PROTOCOL, METHODS, EXPERIMENTS
from common import read_json, write_json, read_csv, write_csv, digest, code_hashes, environment, predictions_and_metrics

MODULES = dict(zip(
    METHODS,
    [
        'algorithm_1_sift_svm',
        'algorithm_2_resnet50',
        'algorithm_3_siamese_resnet18',
        'algorithm_4_efficientnet_b0'
    ]
))


def validate_prepared(out, verify_files=True, check_code=True):
    q = read_json(out / 'protocol.json')
    p = q['settings']
    root = Path(q['dataset_root'])

    for file, key in [
        ('manifest.csv', 'manifest_sha256'),
        ('splits.csv', 'splits_sha256'),
        ('label_mapping.json', 'mapping_sha256')
    ]:
        if digest(out / file) != q[key]:
            raise ValueError(f'Prepared {file} was modified. Refusing to run.')
    if check_code and code_hashes() != q['code_sha256']:
        raise ValueError('Source code changed after preparation; refusing to mix incompatible runs.')
    manifest = read_csv(out / 'manifest.csv')
    mapping = read_json(out / 'label_mapping.json')
    all_rows = read_csv(out / 'splits.csv')
    ids = {r['coin_id']: r for r in manifest}
    expected = p['n_classes'] * p['samples_per_class']
    k = p['samples_per_class']
    n = p['n_classes']
    if len(ids) != expected or len(manifest) != expected or len({r['sha256'] for r in manifest}) != expected or len({r['pixel_sha256'] for r in manifest}) != expected:
        raise ValueError('Coin or image duplicate, or incorrect manifest size')
    if collections.Counter(r['original_type'] for r in manifest) != dict.fromkeys(mapping['original'], k):
        raise ValueError('Incorrect class count')
    if verify_files:
        for r in manifest:
            path = (root / r['path']).resolve()
            if not path.is_relative_to(root.resolve()) or digest(path) != r['sha256']:
                raise ValueError(f'Image changed: {path}')
    folds = collections.defaultdict(lambda: collections.defaultdict(list))

    for r in all_rows:
        m = ids[r['coin_id']]
        if any(r[key] != m[key] for key in [
            'path',
            'original_type',
            'merged_type'
        ]) or r['merged_type'] != mapping['original_to_merged'][r['original_type']]:
            raise ValueError('Manifest/split label mismatch')
        folds[(int(r['seed']), int(r['fold']))][r['split']].append(r)

    expected_keys = {(s, f) for s in p['seeds'] for f in range(1, p['folds'] + 1)}
    if set(folds) != expected_keys:
        raise ValueError('Fold grid incomplete')
    for s in p['seeds']:
        tested = []
        for f in range(1, p['folds'] + 1):
            d = folds[s, f]
            combined = [r['coin_id'] for rs in d.values() for r in rs]
            if set(d) != {'train', 'val', 'test'} or len(combined) != len(ids) or set(combined) != set(ids):
                raise ValueError('Overlap/missing samples in fold')
            for split, rs in d.items():
                frac = {
                    'test': 1 / p['folds'],
                    'val': p['validation_fraction_of_all'],
                    'train': 1 - 1 / p['folds'] - p['validation_fraction_of_all']
                }[split]
                counts = collections.Counter(r['original_type'] for r in rs)
                if set(counts) != set(mapping['original']) or any(abs(c - k * frac) > 1e-8 for c in counts.values()):
                    raise ValueError(f'Incorrect stratification for {s}/{f}/{split}')

            tested.extend(r['coin_id'] for r in d['test'])
        if collections.Counter(tested) != dict.fromkeys(ids, 1):
            raise ValueError('One test prediction per coin/seed violated')

    return q, root, mapping, folds


def run_one(out, exp, seed, fold, method, ctx):
    module = importlib.import_module(MODULES[method])
    dest = out / 'runs' / exp / f'seed_{seed}' / f'fold_{fold:02d}' / method
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'result.json').exists():
        done = read_json(dest / 'result.json')
        if not (dest / 'test' / 'predictions.csv').is_file() or not (dest / 'test' / 'metrics.json').is_file():
            raise ValueError(f'Incomplete completed run: {dest}')
        print(f'SKIP completed {exp}/{seed}/{fold}/{method}', flush=True)
        return done

    candidates = []
    # Training functions receive no test rows. A candidate cannot use test examples for selection.
    train_ctx = {k: v for k, v in ctx.items() if k != 'test'}

    for candidate in module.CANDIDATES:
        path = dest / candidate
        path.mkdir(exist_ok=True)
        if (path / 'done.json').exists():
            result = read_json(path / 'done.json')
            model_file = path / ('model.joblib' if method == 'sift_svm' else 'model.pth')
            if not model_file.is_file() or result['candidate'] != candidate:
                raise ValueError(f'Broken completed candidate: {path}')
            print(f'SKIP completed candidate {path}', flush=True)
        else:
            result = module.train(candidate, train_ctx, path)

        candidates.append(result)

    # Deterministic ties: prefer frozen or C=1 (first candidate). No test predictions exist yet.
    selected = max(candidates, key=lambda r: r['validation']['macro_f1'])
    chosen = selected['candidate']
    if method == 'sift_svm':
        search_seconds = sum(c['candidate_fit_seconds'] for c in candidates) + selected['shared_feature_seconds']
    else:
        search_seconds = sum(c['training_seconds'] for c in candidates)
    write_json(
        dest / 'selection.json',
        {
            'selection_metric': 'validation_macro_f1',
            'tie_break': 'first candidate (frozen or C1)',
            'candidates': candidates,
            'selected': chosen
        }
    )
    y, pred, elapsed = module.evaluate(ctx, chosen, dest / chosen)
    test = predictions_and_metrics(dest / 'test', ctx['test'], ctx['classes'], y, pred)
    result = {
        'experiment': exp,
        'seed': seed,
        'fold': fold,
        'method': method,
        'selected_candidate': chosen,
        'run_seed': ctx['run_seed'],
        'n_train': len(ctx['train']),
        'n_validation': len(ctx['val']),
        'n_test': len(ctx['test']),
        'n_classes': len(ctx['classes']),
        **test,
        'validation_macro_f1': selected['validation']['macro_f1'],
        'validation_accuracy': selected['validation']['accuracy'],
        'training_seconds': selected['training_seconds'],
        'total_search_seconds': search_seconds,
        'inference_seconds': elapsed,
        'inference_ms_per_image': 1000 * elapsed / len(y),
        'total_parameters': selected['total_parameters'],
        'trainable_parameters': selected['trainable_parameters'],
        'best_epoch': selected['best_epoch'],
        'epochs_run': selected['epochs_run'],
        'stop_reason': selected['stop_reason']
    }
    write_json(dest / 'result.json', result)
    print(
        f'COMPLETE {exp}/{seed}/{fold}/{method} {chosen} | test accuracy={test["accuracy"]:.4f}, macro F1={test["macro_f1"]:.4f}',
        flush=True
    )

    return result


def run(out, experiments, methods, device, pretrained=True, check_code=True):
    q, root, mapping, folds = validate_prepared(out, check_code=check_code)
    p = q['settings']
    import torch
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available. Install matching CUDA PyTorch or explicitly use --device cpu.')
    if device.type == 'cpu':
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    info = environment()
    info.update({
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'torch_cpu_threads': torch.get_num_threads(),
        'pretrained': pretrained,
        'hostname': socket.gethostname()
    })
    env_path = out / 'execution_environment.json'
    if env_path.exists() and read_json(env_path) != info:
        raise ValueError('Execution environment changed (including hardware, package versions or device). Use same environment to keep timing comparable.')
    write_json(env_path, info)

    for exp in experiments:
        classes = mapping[exp]
        local = dict(p, num_classes=len(classes))
        key = 'original_type' if exp == 'original' else 'merged_type'

        for seed in p['seeds']:
            for fold in range(1, p['folds'] + 1):
                ctx = {
                    'root': root,
                    'classes': classes,
                    'p': local,
                    'device': device,
                    'pretrained': pretrained,
                    'run_seed': seed + 1009 * fold
                }
                ctx.update({
                    sp: [dict(r, label=r[key]) for r in rs] for sp, rs in folds[seed, fold].items()
                })
                for method in methods:
                    run_one(out, exp, seed, fold, method, ctx)

    print('Requested runs completed.', flush=True)


class Tee:
    def __init__(self, original, file):
        self.original = original
        self.file = file

    def write(self, msg):
        self.original.write(msg)
        self.file.write(msg)
        return len(msg)

    def flush(self):
        self.original.flush()
        self.file.flush()

    @ property
    def encoding(self):
        return self.original.encoding


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    ap.add_argument('--experiment', choices=['all', *EXPERIMENTS], default='all')
    ap.add_argument('--method', choices=['all', *METHODS], default='all')
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument(
        '--check-device',
        action='store_true',
        help='With --preflight-only: test both CNN configurations using real ImageNet weights and synthetic batch'
    )
    ap.add_argument(
        '--confirm-audit',
        action='store_true',
        help='Confirm inspected class selection, coin IDs and combined image audit before launch'
    )
    a = ap.parse_args()
    out = Path(a.output).resolve()
    q, _, mapping, _ = validate_prepared(out)
    if q['settings'] != PROTOCOL:
        raise ValueError('Production protocol differs from config.py')
    if a.preflight_only:
        print(f'PASS: {len(mapping["original"])} original / {len(mapping["merged"])} merged classes. Shared splits and all selected image SHA-256 hashes verified.')
        if a.check_device:
            from preflight import device_check
            device_check(out, q['settings'], len(mapping['original']), a.device)
            print('PASS: all six neural-model configurations on the requested device.')
        return
    if a.check_device:
        ap.error('--check-device requires --preflight-only')
    if not a.confirm_audit:
        ap.error('Review manifest/audits, then pass --confirm-audit')
    # Exclusive lock: no parallel processes writing into same output folder.
    lock = out / 'RUNNING.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'{lock} exists. If previous process stopped, verify it is not running, then remove this lock to resume.')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(f'{socket.gethostname()} PID {os.getpid()}\n')
        with (out / 'console.log').open('a', encoding='utf-8', buffering=1) as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                print(
                    f'\n--- Launch {time.strftime("%Y-%m-%d %H:%M:%S")} experiment={a.experiment} method={a.method} ---',
                    flush=True
                )
                run(
                    out,
                    EXPERIMENTS if a.experiment == 'all' else [a.experiment],
                    METHODS if a.method == 'all' else [a.method],
                    a.device
                )
    finally:
        lock.unlink(missing_ok=True)
if __name__ == '__main__':
    main()
