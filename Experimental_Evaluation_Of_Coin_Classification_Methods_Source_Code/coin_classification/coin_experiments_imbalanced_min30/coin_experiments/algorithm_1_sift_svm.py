"""Algorithm 1: detected SIFT keypoints, L1 BoVW, training-only vocabulary/scaler, RBF SVM."""
import time
from pathlib import Path
import cv2, joblib
from PIL import Image, ImageOps
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from common import atomic_joblib, write_json, read_json, predictions_and_metrics, set_seed

METHOD = 'sift_svm'
CANDIDATES = ('C1', 'C10')


def descriptors(root, rows, p):
    sift = cv2.SIFT_create()
    result = []

    for r in rows:
        with Image.open(root / r['path']) as im:
            image = np.array(ImageOps.exif_transpose(im).convert('L'))
        image = cv2.resize(image, (p['sift_size'], p['sift_size']))
        _, d = sift.detectAndCompute(image, None)
        result.append(d if d is not None else np.empty((0, 128), dtype=np.float32))

    return result


def features(ds, kmeans, p):
    out = []
    for d in ds:
        h = np.zeros(p['visual_words'], np.float32)
        if len(d):
            h = np.bincount(kmeans.predict(d), minlength=p['visual_words']).astype(np.float32)
            h /= h.sum()
        out.append(h)
    return np.asarray(out)


def labels(rows, classes):
    m = {c: i for i, c in enumerate(classes)}
    return np.array([m[r['label']] for r in rows])


def shared(ctx, out):
    path = out / 'shared_bovw.joblib'
    p = ctx['p']
    root = ctx['root']
    classes = ctx['classes']
    if path.exists():
        return joblib.load(path)
    set_seed(ctx['run_seed'])
    cv2.setRNGSeed(ctx['run_seed'])
    cv2.setNumThreads(1)
    start = time.perf_counter()
    tr = descriptors(root, ctx['train'], p)
    va = descriptors(root, ctx['val'], p)
    valid = [d for d in tr if len(d)]
    if not valid or sum(map(len, valid)) < p['visual_words']:
        raise ValueError('Insufficient TRAIN SIFT descriptors for vocabulary')
    # Only training descriptors are fitted. Validation descriptors are transformed, never fitted.
    kmeans = MiniBatchKMeans(
        n_clusters=p['visual_words'],
        random_state=ctx['run_seed'],
        batch_size=p['sift_kmeans_batch_size'],
        n_init=p['sift_kmeans_n_init'],
        max_iter=100
    )
    kmeans.fit(np.vstack(valid))
    X = features(tr, kmeans, p)
    V = features(va, kmeans, p)
    obj = {
        'kmeans': kmeans,
        'X_train': X,
        'X_val': V,
        'y_train': labels(ctx['train'], classes),
        'y_val': labels(ctx['val'], classes),
        'shared_seconds': time.perf_counter() - start,
        'train_images_without_keypoints': sum(not len(d) for d in tr),
        'val_images_without_keypoints': sum(not len(d) for d in va)
    }
    atomic_joblib(path, obj)

    return obj


def train(candidate, ctx, out):
    p = ctx['p']
    C = int(candidate[1:])
    assert C in p['sift_C']
    obj = shared(ctx, out.parent)
    start = time.perf_counter()
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=p['sift_kernel'], C=C, gamma=p['sift_gamma']))
    ])
    pipe.fit(obj['X_train'], obj['y_train'])
    pred = pipe.predict(obj['X_val'])
    elapsed = time.perf_counter() - start
    vm = predictions_and_metrics(out / 'validation', ctx['val'], ctx['classes'], obj['y_val'], pred)
    atomic_joblib(
        out / 'model.joblib',
        {
            'classes': ctx['classes'],
            'pipeline': pipe,
            'kmeans': obj['kmeans'],
            'protocol': p,
            'candidate': candidate
        }
    )
    summary = {
        'candidate': candidate,
        'validation': vm,
        'training_seconds': elapsed + obj['shared_seconds'],
        'candidate_fit_seconds': elapsed,
        'shared_feature_seconds': obj['shared_seconds'],
        'epochs_run': None,
        'best_epoch': None,
        'stop_reason': 'fitted',
        'total_parameters': None,
        'trainable_parameters': None,
        'support_vectors': int(pipe['svm'].n_support_.sum()),
        'train_images_without_keypoints': obj['train_images_without_keypoints'],
        'val_images_without_keypoints': obj['val_images_without_keypoints']
    }
    write_json(out / 'done.json', summary)
    print(f'{METHOD} {candidate}: val F1 {vm["macro_f1"]:.4f}', flush=True)

    return summary


def evaluate(ctx, candidate, out):
    p = ctx['p']
    root = ctx['root']
    obj = joblib.load(out / 'model.joblib')
    assert obj['candidate'] == candidate and obj['classes'] == ctx['classes'] and obj['protocol'] == p
    cv2.setNumThreads(1)
    warm = ctx['val'][:p['batch_size'] * p['inference_warmup_batches']]
    if warm:
        obj['pipeline'].predict(features(descriptors(root, warm, p), obj['kmeans'], p))
    start = time.perf_counter()
    d = descriptors(root, ctx['test'], p)
    X = features(d, obj['kmeans'], p)
    pred = obj['pipeline'].predict(X)
    elapsed = time.perf_counter() - start

    return labels(ctx['test'], ctx['classes']), pred, elapsed
