import csv, json, os, hashlib, platform, sys, random
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

METRICS = ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1')


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def read_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_joblib(path, obj):
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def atomic_torch(path, obj):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def metric_values(y, pred, n):
    labs = list(range(n))
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_precision': float(precision_score(
                y,
                pred,
                labels=labs,
                average='macro',
                zero_division=0
            )),
            'macro_recall': float(recall_score(y, pred, labels=labs, average='macro', zero_division=0)),
            'macro_f1': float(f1_score(y, pred, labels=labs, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(y, pred, labels=labs, average='weighted', zero_division=0))}


def predictions_and_metrics(out, rows, classes, y, pred):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    assert len(rows) == len(y) == len(pred) and len(y) > 0
    assert np.all((y >= 0) & (y < len(classes))) and np.all((pred >= 0) & (pred < len(classes)))
    assert [classes[i] for i in y] == [r['label'] for r in rows], 'Prediction order mismatch'
    write_csv(
        out / 'predictions.csv',
        [
            dict(
                coin_id=r['coin_id'],
                path=r['path'],
                original_type=r['original_type'],
                true_label=classes[i],
                predicted_label=classes[j],
                correct=int(i == j)
            ) for r, i, j in zip(rows, y, pred)
        ]
    )
    m = metric_values(y, pred, len(classes))
    write_json(out / 'metrics.json', m)
    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    write_csv(
        out / 'confusion_matrix.csv',
        [dict(true_label=c, **{k: int(v) for k, v in zip(classes, row)}) for c, row in zip(classes, cm)]
    )
    report = classification_report(
        y,
        pred,
        labels=range(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0
    )
    write_csv(
        out / 'per_class.csv',
        [dict(
            label=c,
            precision=report[c]['precision'],
            recall=report[c]['recall'],
            f1=report[c]['f1-score'],
            support=int(report[c]['support'])
        ) for c in classes]
    )

    return m


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def environment():
    from importlib.metadata import version, PackageNotFoundError
    versions = {}

    for pkg in [
        'numpy',
        'scipy',
        'scikit-learn',
        'torch',
        'torchvision',
        'opencv-python',
        'opencv-python-headless',
        'Pillow',
        'joblib'
    ]:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            pass

    return {
        'python': sys.version,
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'packages': versions
    }


def code_hashes():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py')) if p.name != 'smoke_test.py'}
