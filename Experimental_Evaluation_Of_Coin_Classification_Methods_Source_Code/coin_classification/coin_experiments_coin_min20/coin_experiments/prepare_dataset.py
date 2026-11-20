"""Create a single immutable image list and shared folds; no image copies or training."""
import argparse, collections, re, hashlib, random
from pathlib import Path
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import numpy as np
from config import PROTOCOL
from common import write_csv, write_json, digest, read_csv, environment, code_hashes

EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
CN_ID = re.compile(r'(?:^|_)cn_coin_(\d+)(?=_|$)', re.I)
CN_TYPE = re.compile(r'(?:^|_)cn_type_(\d+)(?=_|$)', re.I)
SINGLE = re.compile(r'(?:^|_)(?:obv|rev|obverse|reverse)(?=_|$)', re.I)
ORIGINAL = re.compile(r'(?:^|_)o(?:_merged)?$', re.I)


def discover(root, metadata):
    records = []
    rejected = []
    if metadata:
        source = read_csv(metadata)
        for r in source:
            p = (root / r['path']).resolve()
            if not p.is_relative_to(root):
                raise ValueError(f'Path outside dataset: {p}')
            if not p.is_file() or p.suffix.lower() not in EXT:
                raise ValueError(f'Missing image: {p}')
            if not re.fullmatch(r'\d+', r['coin_id']) or not re.fullmatch(r'\d+', r['original_type']):
                raise ValueError(f'Expected numeric Coin ID and type: {r}')
            if r.get('both_sides', '').lower() not in ('1', 'yes', 'true'):
                raise ValueError(f'Cannot confirm both sides: {p}')
            if SINGLE.search(p.stem):
                raise ValueError(f'Single-side filename: {p}')
            if CN_ID.search(p.stem) and int(CN_ID.search(p.stem).group(1)) != int(r['coin_id']):
                raise ValueError(f'Coin ID mismatch in metadata: {p}')
            if CN_TYPE.search(p.stem) and int(CN_TYPE.search(p.stem).group(1)) != int(r['original_type']):
                raise ValueError(f'Type mismatch in metadata: {p}')
            records.append((p, str(int(r['original_type'])), str(int(r['coin_id']))))
    else:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.suffix.lower() not in EXT:
                continue
            # Supports raw dataset_types/<type>/... and old train/val/test/<type>/...
            folders = [s for s in p.relative_to(root).parts[:-1] if s.isdigit()]
            if not folders:
                continue
            t = str(int(folders[-1]))
            m = CN_ID.search(p.stem)
            mt = CN_TYPE.search(p.stem)
            if SINGLE.search(p.stem):
                rejected.append((str(p), 'single_side'))
                continue
            if not m:
                rejected.append((str(p), 'unrecognized_coin_id'))
                continue
            if mt and str(int(mt.group(1))) != t:
                raise ValueError(f'Filename type differs from class folder: {p}')
            records.append((p, t, str(int(m.group(1)))))
    if not records:
        raise ValueError('No usable images. Use dataset_types root or supply verified metadata CSV.')

    return records, rejected


def prepare(root, out, protocol, metadata=None, classes_file=None, confirmed=False, exclude_coins=None):
    root = Path(root).resolve()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError(f'Dataset not found: {root}')
    if (out / 'protocol.json').exists() or (out / 'runs').exists():
        raise ValueError('Output already contains a prepared protocol. Do not overwrite; use an empty output directory.')
    if not confirmed and not metadata:
        raise ValueError('Confirm the folder contains combined obverse+reverse images using --confirm-combined. Do not use dataset_coins (single sides).')
    records, rejected = discover(root, metadata)
    write_csv(
        out / 'rejected_files.csv',
        [{'path': p, 'reason': r} for p, r in rejected],
        ['path', 'reason']
    )
    unknown = [r for r in rejected if r[1] == 'unrecognized_coin_id']
    if unknown:
        raise ValueError(f'{len(unknown)} image filenames have no CN Coin ID; see rejected_files.csv. Supply verified --metadata CSV rather than guessing IDs.')
    excluded = set()
    if exclude_coins:
        excluded = {x.strip() for x in Path(exclude_coins).read_text(encoding='utf-8-sig').splitlines() if x.strip()}
        if not all(x.isdigit() for x in excluded):
            raise ValueError('Excluded IDs must be numeric')
        excluded = {str(int(x)) for x in excluded}
    write_csv(out / 'excluded_coin_ids.csv', [{'coin_id': c} for c in sorted(excluded)], ['coin_id'])
    by_coin = collections.defaultdict(list)
    for p, t, c in records:
        if c not in excluded:
            by_coin[c].append((p, t))
    conflicting = {c for c, v in by_coin.items() if len({t for p, t in v}) > 1}
    write_csv(
        out / 'conflicting_coin_ids.csv',
        [{
            'coin_id': c,
            'types': ';'.join(sorted({t for p, t in by_coin[c]}))
        } for c in sorted(conflicting)],
        ['coin_id', 'types']
    )
    pool = collections.defaultdict(list)
    for c, items in by_coin.items():
        if c not in conflicting:
            pool[items[0][1]].append((c, items))
    write_csv(
        out / 'class_inventory.csv',
        [
            {
                'original_type': t,
                'unique_coin_ids': len(v),
                'eligible': len(v) >= protocol['samples_per_class']
            } for t, v in sorted(pool.items(), key=lambda kv: (-len(kv[1]), int(kv[0])))
        ]
    )
    n = protocol['n_classes']
    k = protocol['samples_per_class']
    required = protocol['required_classes']
    eligible = {t for t, v in pool.items() if len(v) >= k}
    if not set(required) <= eligible:
        raise ValueError(f'Required types have fewer than {k} eligible distinct Coin IDs: {sorted(set(required)-eligible)}. See class_inventory.csv; do not change the protocol after training.')
    if classes_file:
        candidates = [x.strip() for x in Path(classes_file).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
        if len(candidates) != n or len(set(candidates)) != n or not set(required) <= set(candidates) or not set(candidates) <= eligible:
            raise ValueError('Explicit classes list must have exactly the required number of eligible unique types, including reviewer pairs.')
        candidates = list(dict.fromkeys(required + candidates))
    else:
        preferred = list(dict.fromkeys(required + protocol['preferred_classes']))
        candidates = [t for t in preferred if t in eligible]
        candidates += sorted(eligible - set(candidates), key=lambda t: (-len(pool[t]), int(t)))
    if len(candidates) < n:
        raise ValueError(f'Only {len(candidates)} eligible types; need {n}. See class_inventory.csv.')
    selected = []
    audits = []
    seen_pixels = {}
    seen_bytes = {}
    perceptual = []
    classes = []
    attempts = []
    def save_selection_audit():
        write_csv(out / 'selection_audit.csv', audits, ['coin_id', 'path', 'reason'])
        write_csv(
            out / 'class_selection.csv',
            attempts,
            ['original_type', 'unique_coin_ids', 'accepted_coins', 'status']
        )
    for t in candidates:
        # Each class has its own deterministic sampling stream. Rejected classes
        # do not consume global sampling state or reserve image hashes.
        rng = random.Random(protocol['selection_seed'] + int(t))
        coins = sorted(pool[t], key=lambda x: int(x[0]))
        rng.shuffle(coins)
        local = []
        local_pixels = {}
        local_bytes = {}
        local_perceptual = []

        for c, items in coins:
            # Prefer original photo to plaster cast; deterministic tie-break.
            for p, _ in sorted(
                items,
                key=lambda item: (not bool(ORIGINAL.search(item[0].stem)), str(item[0]))
            ):
                try:
                    with Image.open(p) as im:
                        im = ImageOps.exif_transpose(im).convert('RGB')
                        im.load()
                        h = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                        gray = np.array(im.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
                        dhash = int.from_bytes(np.packbits(gray[:, 1:] > gray[:,:-1]).tobytes(), 'big')
                    b = digest(p)
                except Exception as e:
                    audits.append({'coin_id': c, 'path': str(p), 'reason': 'unreadable: ' + str(e)})
                    continue
                if h in seen_pixels or b in seen_bytes or h in local_pixels or b in local_bytes:
                    duplicate = seen_pixels.get(
                        h,
                        seen_bytes.get(b, local_pixels.get(h, local_bytes.get(b)))
                    )
                    audits.append({
                        'coin_id': c,
                        'path': str(p),
                        'reason': 'exact duplicate of Coin ID ' + str(duplicate)
                    })
                    continue

                local_pixels[h] = c
                local_bytes[b] = c
                local.append({
                    'coin_id': c,
                    'original_type': t,
                    'path': p.relative_to(root).as_posix(),
                    'sha256': b,
                    'pixel_sha256': h,
                    'available_views': len(items)
                })
                local_perceptual.append((c, t, p.relative_to(root).as_posix(), dhash))
                break
            if len(local) == k:
                break

        accepted = len(local) == k
        attempts.append({
            'original_type': t,
            'unique_coin_ids': len(pool[t]),
            'accepted_coins': len(local),
            'status': 'selected' if accepted else 'insufficient_valid_coins'
        })
        if not accepted:
            save_selection_audit()
            if t in required or classes_file:
                raise ValueError(f'Class {t}: only {len(local)} readable distinct nonduplicate coins. Required or explicitly selected type cannot be replaced. See selection_audit.csv and class_selection.csv. No training allowed.')
            print(
                f'Skipping optional class {t}: only {len(local)}/{k} valid coins. Trying next eligible type.',
                flush=True
            )
            continue

        classes.append(t)
        selected.extend(local)
        seen_pixels.update(local_pixels)
        seen_bytes.update(local_bytes)
        perceptual.extend(local_perceptual)
        print(f'Selected class {t}: {k} coins ({len(classes)}/{n} classes).', flush=True)
        if len(classes) == n:
            break

    save_selection_audit()
    if len(classes) != n:
        raise ValueError(f'Only {len(classes)} classes passed image verification; need {n} with {k} coins each. See class_selection.csv and selection_audit.csv. No training allowed.')
    classes = sorted(classes, key=int)
    near = []

    for i, (ca, ta, pa, ha) in enumerate(perceptual):
        for cb, tb, pb, hb in perceptual[i + 1:]:
            distance = (ha ^ hb).bit_count()
            if distance <= 3:
                near.append(dict(
                    coin_id_a=ca,
                    type_a=ta,
                    path_a=pa,
                    coin_id_b=cb,
                    type_b=tb,
                    path_b=pb,
                    dhash_distance=distance
                ))

    write_csv(
        out / 'near_duplicate_review.csv',
        near,
        ['coin_id_a', 'type_a', 'path_a', 'coin_id_b', 'type_b', 'path_b', 'dhash_distance']
    )
    print(f'Near-duplicate review: {len(near)} flagged pairs (heuristic only; review before launch).')
    selected.sort(key=lambda r: (int(r['original_type']), int(r['coin_id'])))
    assert len(selected) == n * k == len({r['coin_id'] for r in selected})
    mapping = {t: t for t in classes}

    for group in protocol['merge_groups']:
        if not set(group) <= set(classes) or any(mapping[t] != t for t in group):
            raise ValueError('Invalid or overlapping merge groups')
        for t in group:
            mapping[t] = '+'.join(group)
    for r in selected:
        r['merged_type'] = mapping[r['original_type']]
    write_csv(out / 'manifest.csv', selected)
    y = np.array([r['original_type'] for r in selected])
    indices = np.arange(len(y))
    split_rows = []

    for seed in protocol['seeds']:
        outer = StratifiedKFold(n_splits=protocol['folds'], shuffle=True, random_state=seed)
        seen = []

        for fold, (dev, test) in enumerate(outer.split(indices, y), 1):
            vsize = int(round(len(y) * protocol['validation_fraction_of_all']))
            inner = StratifiedShuffleSplit(n_splits=1, test_size=vsize, random_state=seed + fold * 1009)
            tr, va = next(inner.split(dev, y[dev]))
            train = dev[tr]
            val = dev[va]
            seen.extend(test.tolist())

            for split, arr in [('train', train), ('val', val), ('test', test)]:
                assert len(set(y[arr])) == n and len(arr) == len(set(arr))
                for i in sorted(arr):
                    split_rows.append({
                        'seed': seed,
                        'fold': fold,
                        'split': split,
                        **{q: selected[i][q] for q in [
                            'coin_id',
                            'original_type',
                            'merged_type',
                            'path'
                        ]}
                    })
            assert len(set(train) & set(val)) == len(set(train) & set(test)) == len(set(val) & set(test)) == 0
        assert sorted(seen) == indices.tolist()

    write_csv(out / 'splits.csv', split_rows)
    merged_classes = sorted(set(mapping.values()), key=lambda c: tuple(map(int, c.split('+'))))
    write_json(
        out / 'label_mapping.json',
        {'original': classes, 'merged': merged_classes, 'original_to_merged': mapping}
    )
    write_json(out / 'preparation_environment.json', environment())
    write_json(
        out / 'protocol.json',
        {
            'settings': protocol,
            'dataset_root': str(root),
            'manifest_sha256': digest(out / 'manifest.csv'),
            'splits_sha256': digest(out / 'splits.csv'),
            'mapping_sha256': digest(out / 'label_mapping.json'),
            'code_sha256': code_hashes(),
            'combined_images_confirmed': True,
            'selection_rule': 'Required reviewer types, eligible original types, then highest unique ID counts (numeric type tie-break); skip optional types failing image verification; per-type seed = selection_seed + numeric type ID; original image preferred; exclude conflicting IDs and exact duplicates.' if not classes_file else 'Explicit class list, required types first; per-type seed = selection_seed + numeric type ID; fail if any type fails image verification; original image preferred; exclude conflicting IDs and exact duplicates.'
        }
    )
    print(f'Prepared {len(selected)} unique coins: {len(classes)} original / {len(merged_classes)} merged classes. {len(protocol["seeds"])*protocol["folds"]} shared folds. Review manifest and audit before training.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--metadata', help='Verified CSV: path,original_type,coin_id,both_sides')
    p.add_argument('--classes-file', help=f'Optional {PROTOCOL["n_classes"]} type IDs, one per line')
    p.add_argument('--confirm-combined', action='store_true')
    p.add_argument(
        '--exclude-coins',
        help='Optional verified duplicate Coin IDs, one per line, excluded before selection'
    )
    a = p.parse_args()
    prepare(
        a.dataset,
        a.output,
        PROTOCOL,
        a.metadata,
        a.classes_file,
        a.confirm_combined,
        a.exclude_coins
    )
if __name__ == '__main__':
    main()
