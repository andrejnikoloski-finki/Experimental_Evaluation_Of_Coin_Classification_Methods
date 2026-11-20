"""Algorithm 3: shared ResNet18, 128-D normalized embeddings, contrastive loss, nearest centroid."""
import time, random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models
from neural_common import (
    CoinDataset,
    get_transforms,
    loader,
    seed_torch,
    sync,
    set_grad,
    training_mode,
    optimizer_for,
    EarlyStopping
)
from common import metric_values, predictions_and_metrics, write_csv, write_json, atomic_torch

METHOD = 'siamese_resnet18'
CANDIDATES = ('frozen', 'partial')


class SiameseNetwork(nn.Module):
    def __init__(self, p, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(weights=getattr(
            models.ResNet18_Weights,
            p['weights'][METHOD]
        ) if pretrained else None)
        d = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        e = p['embedding_dim']
        self.embedding = nn.Sequential(nn.Linear(d, e), nn.ReLU(), nn.Linear(e, e))

    def forward_once(self, x):
        return F.normalize(self.embedding(self.backbone(x)), p=2, dim=1)

    def forward(self, a, b):
        return self.forward_once(a), self.forward_once(b)


def build(candidate, p, pretrained=True):
    if candidate not in CANDIDATES:
        raise ValueError(candidate)
    model = SiameseNetwork(p, pretrained)
    set_grad(model.backbone, False)
    unfrozen = model.backbone.layer4 if candidate == 'partial' else None
    if unfrozen is not None:
        set_grad(unfrozen, True)
    return model, unfrozen


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, a, b, y):
        dist = F.pairwise_distance(a, b)
        return (y * dist.square() + (1 - y) * torch.clamp(self.margin - dist, min=0).square()).mean()


def pair_indices(labels, count, seed):
    # Uniform class choice; positive and negative pairs in equal proportions.
    rng = random.Random(seed)
    by = {int(c): np.flatnonzero(labels == c).tolist() for c in np.unique(labels)}
    classes = sorted(by)
    if len(classes) < 2 or min(map(len, by.values())) < 2:
        raise ValueError('Siamese sampling requires >=2 classes and >=2 coins per class')
    choices = [1] * (count // 2) + [0] * (count - count // 2)
    rng.shuffle(choices)
    pairs = []

    for same in choices:
        if same:
            c = rng.choice(classes)
            a, b = rng.sample(by[c], 2)
        else:
            c, d = rng.sample(classes, 2)
            a, b = rng.choice(by[c]), rng.choice(by[d])
        pairs.append((a, b, same))

    return pairs


class PairDataset(Dataset):
    def __init__(self, base, pairs):
        self.base = base
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b, same = self.pairs[i]
        return self.base[a][0], self.base[b][0], torch.tensor(same, dtype=torch.float32)


def embeddings(model, dl, device):
    model.eval()
    em = []
    labels = []
    with torch.inference_mode():
        for x, y in dl:
            em.append(model.forward_once(x.to(device)).cpu())
            labels.extend(y.tolist())
    return torch.cat(em), np.array(labels)


def centroids_from_training(emb, y, n):
    return torch.stack([F.normalize(emb[y == i].mean(dim=0), p=2, dim=0) for i in range(n)])


def classify(emb, centroids):
    return torch.cdist(emb, centroids).argmin(1).numpy()


def validate(model, train_dl, val_dl, device, n, pairs, criterion):
    te, ty = embeddings(model, train_dl, device)
    centers = centroids_from_training(te, ty, n)
    ve, vy = embeddings(model, val_dl, device)
    vp = classify(ve, centers)
    i, j, s = zip(*pairs)
    loss = criterion(ve[list(i)], ve[list(j)], torch.tensor(s, dtype=torch.float32)).item()
    return vy, vp, loss, centers


def train(candidate, ctx, out):
    p = ctx['p']
    device = ctx['device']
    seed_torch(ctx['run_seed'])
    model, unfrozen = build(candidate, p, ctx['pretrained'])
    model.to(device)
    opt = optimizer_for(model.embedding, unfrozen, p)
    criterion = ContrastiveLoss(p['margin'])
    tr, ev = get_transforms(p)
    classes = ctx['classes']
    root = ctx['root']
    train = CoinDataset(ctx['train'], root, classes, tr)
    train_eval = loader(CoinDataset(ctx['train'], root, classes, ev), p)
    val = CoinDataset(ctx['val'], root, classes, ev)
    val_dl = loader(val, p)
    val_pairs = pair_indices(val.y, p['siamese_pairs_per_epoch'], ctx['run_seed'] + 8917)
    stopper = EarlyStopping(p)
    history = []
    sync(device)
    start = time.perf_counter()

    for epoch in range(1, p['max_epochs'] + 1):
        training_mode(model, model.backbone, unfrozen, model.embedding)
        # Fresh deterministic pairs every epoch; same pair schedule for both candidates.
        pairs = pair_indices(train.y, p['siamese_pairs_per_epoch'], ctx['run_seed'] + epoch * 1009)
        dl = loader(PairDataset(train, pairs), p)
        running = 0.
        total = 0

        for a, b, y in dl:
            a, b, y = a.to(device), b.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            e1, e2 = model(a, b)
            loss = criterion(e1, e2, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite Siamese loss')
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            total += len(y)

        vy, vp, vl, _ = validate(model, train_eval, val_dl, device, len(classes), val_pairs, criterion)
        vm = metric_values(vy, vp, len(classes))
        stop = stopper.update(vm['macro_f1'], model, epoch)
        history.append({
            'epoch': epoch,
            'training_loss': running / total,
            'validation_loss': vl,
            'validation_accuracy': vm['accuracy'],
            'validation_macro_f1': vm['macro_f1']
        })
        print(
            f'{METHOD} {candidate} epoch {epoch}/{p["max_epochs"]}: loss {running/total:.4f} val accuracy {vm["accuracy"]:.4f} val F1 {vm["macro_f1"]:.4f}',
            flush=True
        )
        write_csv(out / 'history.csv', history)
        if stop:
            break

    model.load_state_dict(stopper.state)
    vy, vp, _, centers = validate(model, train_eval, val_dl, device, len(classes), val_pairs, criterion)
    sync(device)
    elapsed = time.perf_counter() - start
    vm = predictions_and_metrics(out / 'validation', ctx['val'], classes, vy, vp)
    assert abs(vm['macro_f1'] - stopper.best) < 1e-10
    atomic_torch(
        out / 'model.pth',
        {
            'model_state_dict': stopper.state,
            'centroids': centers,
            'classes': classes,
            'method': METHOD,
            'candidate': candidate,
            'protocol': p,
            'run_seed': ctx['run_seed']
        }
    )
    summary = {
        'candidate': candidate,
        'validation': vm,
        'training_seconds': elapsed,
        'epochs_run': len(history),
        'best_epoch': stopper.best_epoch,
        'stop_reason': 'patience' if stop else 'max_epochs',
        'total_parameters': sum(q.numel() for q in model.parameters()),
        'trainable_parameters': sum(q.numel() for q in model.parameters() if q.requires_grad)
    }
    write_json(out / 'done.json', summary)

    return summary


def evaluate(ctx, candidate, out):
    p = ctx['p']
    device = ctx['device']
    model, _ = build(candidate, p, False)
    model.to(device)
    ck = torch.load(out / 'model.pth', map_location='cpu', weights_only=True)
    assert ck['classes'] == ctx['classes'] and ck['candidate'] == candidate and ck['protocol'] == p
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    centers = ck['centroids']
    _, ev = get_transforms(p)
    val = loader(CoinDataset(ctx['val'], ctx['root'], ctx['classes'], ev), p)
    test = loader(CoinDataset(ctx['test'], ctx['root'], ctx['classes'], ev), p)

    with torch.inference_mode():
        for j, (x, _) in enumerate(val):
            if j >= p['inference_warmup_batches']:
                break
            classify(model.forward_once(x.to(device)).cpu(), centers)
    sync(device)
    start = time.perf_counter()
    te, y = embeddings(model, test, device)
    pred = classify(te, centers)
    sync(device)

    return y, pred, time.perf_counter() - start
