"""Shared neural training, checkpoints, evaluation and timing."""
import time, copy, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from common import set_seed, metric_values, write_csv, write_json, read_json, atomic_torch, predictions_and_metrics


class CoinDataset(Dataset):
    def __init__(self, rows, root, classes, transform):
        self.rows = rows
        self.root = Path(root)
        self.index = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.y = np.array([self.index[r['label']] for r in rows], dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.root / self.rows[i]['path']) as im:
            x = self.transform(ImageOps.exif_transpose(im).convert('RGB'))
        return x, int(self.y[i])


def get_transforms(p):
    # No horizontal flips: they mirror inscriptions and exchange obverse/reverse positions.
    base = [transforms.Resize((p['image_size'], p['image_size']))]
    norm = [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])]
    train = transforms.Compose(base + [
        transforms.RandomRotation(p['rotation_degrees']),
        transforms.ColorJitter(brightness=p['brightness'], contrast=p['contrast'])
    ] + norm)
    return train, transforms.Compose(base + norm)


def loader(dataset, p, shuffle=False, seed=42):
    return DataLoader(
        dataset,
        batch_size=p['batch_size'],
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed)
    )


def seed_torch(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Some CUDA kernels have no deterministic implementation; emit a warning rather than crash long runs.
    torch.use_deterministic_algorithms(True, warn_only=True)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_grad(module, value):
    for q in module.parameters():
        q.requires_grad = value


def training_mode(model, backbone, unfrozen, head):
    model.train()
    backbone.eval()
    # Frozen BatchNorm running statistics must remain fixed, not just gradients.
    if unfrozen is not None:
        unfrozen.train()
    head.train()


def optimizer_for(head, unfrozen, p):
    groups = [{'params': list(head.parameters()), 'lr': p['head_lr']}]
    if unfrozen is not None:
        groups.append({'params': list(unfrozen.parameters()), 'lr': p['backbone_lr']})
    return torch.optim.Adam(groups)


class EarlyStopping:
    def __init__(self, p):
        self.p = p
        self.best = -float('inf')
        self.reference = -float('inf')
        self.best_epoch = 0
        self.stale = 0
        self.state = None

    def update(self, score, model, epoch):
        # Save exact best F1; min_delta only controls patience. Earliest true tie wins.
        if score > self.best:
            self.best = score
            self.best_epoch = epoch
            self.state = cpu_state(model)
        if score > self.reference + self.p['min_delta']:
            self.reference = score
            self.stale = 0
        else:
            self.stale += 1
        return self.stale >= self.p['patience']


def classifier_eval(model, dl, device):
    model.eval()
    truth = []
    pred = []
    loss = 0.
    total = 0

    with torch.inference_mode():
        for x, y in dl:
            x = x.to(device)
            ydev = y.to(device)
            logits = model(x)
            loss += nn.functional.cross_entropy(logits, ydev, reduction='sum').item()
            total += len(y)
            pred.extend(logits.argmax(1).cpu().tolist())
            truth.extend(y.tolist())

    return np.array(truth), np.array(pred), loss / total


def classifier_predict(model, dl, device):
    model.eval()
    truth = []
    pred = []
    with torch.inference_mode():
        for x, y in dl:
            pred.extend(model(x.to(device)).argmax(1).cpu().tolist())
            truth.extend(y.tolist())
    return np.array(truth), np.array(pred)


def train_classifier(builder, method, candidate, ctx, out):
    p = ctx['p']
    device = ctx['device']
    seed_torch(ctx['run_seed'])
    model, backbone, unfrozen, head = builder(candidate, p, ctx['pretrained'])
    model.to(device)
    opt = optimizer_for(head, unfrozen, p)
    criterion = nn.CrossEntropyLoss()
    tr, ev = get_transforms(p)
    root = ctx['root']
    classes = ctx['classes']
    train_dl = loader(CoinDataset(ctx['train'], root, classes, tr), p, True, ctx['run_seed'])
    val_dl = loader(CoinDataset(ctx['val'], root, classes, ev), p)
    stopper = EarlyStopping(p)
    history = []
    sync(device)
    start = time.perf_counter()

    for epoch in range(1, p['max_epochs'] + 1):
        training_mode(model, backbone, unfrozen, head)
        total = 0
        running = 0.
        correct = 0

        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        vy, vp, vl = classifier_eval(model, val_dl, device)
        vm = metric_values(vy, vp, len(classes))
        stop = stopper.update(vm['macro_f1'], model, epoch)
        history.append({
            'epoch': epoch,
            'training_loss': running / total,
            'training_accuracy': correct / total,
            'validation_loss': vl,
            'validation_accuracy': vm['accuracy'],
            'validation_macro_f1': vm['macro_f1']
        })
        print(
            f'{method} {candidate} epoch {epoch}/{p["max_epochs"]}: loss {running/total:.4f} val accuracy {vm["accuracy"]:.4f} val F1 {vm["macro_f1"]:.4f}',
            flush=True
        )
        write_csv(out / 'history.csv', history)
        if stop:
            break

    model.load_state_dict(stopper.state)
    vy, vp, _ = classifier_eval(model, val_dl, device)
    sync(device)
    elapsed = time.perf_counter() - start
    vm = predictions_and_metrics(out / 'validation', ctx['val'], classes, vy, vp)
    assert abs(vm['macro_f1'] - stopper.best) < 1e-10
    atomic_torch(
        out / 'model.pth',
        {
            'model_state_dict': stopper.state,
            'classes': classes,
            'candidate': candidate,
            'method': method,
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


def evaluate_classifier(builder, ctx, chosen, out):
    p = ctx['p']
    device = ctx['device']
    model, _, _, _ = builder(chosen, p, False)
    model.to(device)
    ck = torch.load(out / 'model.pth', map_location='cpu', weights_only=True)
    assert ck['classes'] == ctx['classes'] and ck['candidate'] == chosen and ck['protocol'] == p
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    _, ev = get_transforms(p)
    val = loader(CoinDataset(ctx['val'], ctx['root'], ctx['classes'], ev), p)
    test = loader(CoinDataset(ctx['test'], ctx['root'], ctx['classes'], ev), p)

    with torch.inference_mode():
        for j, (x, _) in enumerate(val):
            if j >= p['inference_warmup_batches']:
                break
            model(x.to(device))
    sync(device)
    start = time.perf_counter()
    y, pred = classifier_predict(model, test, device)
    sync(device)

    return y, pred, time.perf_counter() - start
