"""Expand the exact balanced cohort; anchor all existing fold memberships."""
import argparse, collections, hashlib, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from common import read_csv, read_json, write_csv, write_json, digest, code_hashes, environment
from config import PROTOCOL
from prepare_dataset import discover, ORIGINAL
from run_experiments import validate_prepared


def image_info(p):
    with Image.open(p) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        im.load()
        pix = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        gray = np.array(im.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
        dh = int.from_bytes(np.packbits(gray[:, 1:] > gray[:,:-1]).tobytes(), 'big')
    return digest(p), pix, dh


def prepare(root, base, out):
    root, base, out = map(lambda x: Path(x).resolve(), (root, base, out))
    if (out / 'protocol.json').exists() or (out / 'runs').exists():
        raise ValueError('Output already prepared; do not overwrite')
    q = read_json(base / 'protocol.json')
    mapping = read_json(base / 'label_mapping.json')

    for f, k in [
        ('manifest.csv', 'manifest_sha256'),
        ('splits.csv', 'splits_sha256'),
        ('label_mapping.json', 'mapping_sha256')
    ]:
        if digest(base / f) != q[k]:
            raise ValueError('Baseline changed: ' + f)

    expected = dict(PROTOCOL)
    expected.pop('dataset_mode')
    expected['samples_per_class'] = 30
    if q['settings'] != expected:
        raise ValueError('Baseline protocol differs from expected original settings')
    old = read_csv(base / 'manifest.csv')
    old_splits = read_csv(base / 'splits.csv')
    classes = mapping['original']
    counts = collections.Counter(r['original_type'] for r in old)
    if len(old) != 840 or len(classes) != 28 or counts != dict.fromkeys(classes, 30) or len({r['coin_id'] for r in old}) != 840:
        raise ValueError('Invalid balanced anchor')
    rows = list(old)
    anchor = {r['coin_id']: r for r in old}
    seen_bytes = {}
    seen_pixels = {}
    perceptual = []
    audit = []

    for r in old:
        p = (root / r['path']).resolve()
        if not p.is_relative_to(root):
            raise ValueError('Unsafe image path')
        b, h, d = image_info(p)
        if (b, h) != (r['sha256'], r['pixel_sha256']) or b in seen_bytes or h in seen_pixels:
            raise ValueError('Anchor file changed or duplicated: ' + str(p))
        seen_bytes[b] = r['coin_id']
        seen_pixels[h] = r['coin_id']
        perceptual.append((r, d))

    rec, rejected = discover(root, None)
    grouped = collections.defaultdict(list)
    for path, t, c in rec:
        grouped[c].append((path, t))
    for c, items in grouped.items():
        if c in anchor and {t for _, t in items} != {anchor[c]['original_type']}:
            raise ValueError('Anchor Coin ID has conflicting type: ' + c)
    for c, items in sorted(grouped.items(), key=lambda item: int(item[0])):
        if c in anchor:
            continue
        types = {t for _, t in items}
        if len(types) != 1:
            audit.append(dict(coin_id=c, path='', reason='conflicting types'))
            continue
        t = next(iter(types))
        if t not in classes:
            continue
        for path, _ in sorted(
            items,
            key=lambda item: (not bool(ORIGINAL.search(item[0].stem)), str(item[0]))
        ):
            try:
                b, h, d = image_info(path)
            except Exception as e:
                audit.append(dict(coin_id=c, path=str(path), reason='unreadable: ' + str(e)))
                continue
            if b in seen_bytes or h in seen_pixels:
                audit.append(dict(
                    coin_id=c,
                    path=str(path),
                    reason='exact duplicate of ' + str(seen_bytes.get(b, seen_pixels.get(h)))
                ))
                continue

            r = dict(
                coin_id=c,
                original_type=t,
                path=path.relative_to(root).as_posix(),
                sha256=b,
                pixel_sha256=h,
                available_views=len(items),
                merged_type=mapping['original_to_merged'][t]
            )
            rows.append(r)
            perceptual.append((r, d))
            seen_bytes[b] = c
            seen_pixels[h] = c
            break

    rows.sort(key=lambda r: (int(r['original_type']), int(r['coin_id'])))
    extras = collections.defaultdict(list)
    for r in rows:
        if r['coin_id'] not in anchor:
            extras[r['original_type']].append(r)
    if len(rows) <= 840:
        raise ValueError('No additional valid coins: upload the full source dataset, not the selected 840')
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'manifest.csv', rows)
    write_json(out / 'label_mapping.json', mapping)
    write_csv(out / 'selection_audit.csv', audit, ['coin_id', 'path', 'reason'])
    write_csv(
        out / 'rejected_files.csv',
        [dict(path=p, reason=r) for p, r in rejected],
        ['path', 'reason']
    )
    if any(r == 'unrecognized_coin_id' for p, r in rejected):
        raise ValueError('Unrecognized Coin IDs: inspect rejected_files.csv')
    near = []

    for i, (a, ha) in enumerate(perceptual):
        for b, hb in perceptual[i + 1:]:
            d = (ha ^ hb).bit_count()
            if d <= 3:
                near.append(dict(
                    coin_id_a=a['coin_id'],
                    type_a=a['original_type'],
                    path_a=a['path'],
                    coin_id_b=b['coin_id'],
                    type_b=b['original_type'],
                    path_b=b['path'],
                    dhash_distance=d
                ))

    write_csv(
        out / 'near_duplicate_review.csv',
        near,
        ['coin_id_a', 'type_a', 'path_a', 'coin_id_b', 'type_b', 'path_b', 'dhash_distance']
    )
    split_rows = []

    for seed in PROTOCOL['seeds']:
        # Cycle extra coins across test folds within each class; leave old memberships untouched.
        assignment = {}
        for t in classes:
            a = list(extras[t])
            rng = random.Random(seed + int(t) * 1009)
            rng.shuffle(a)
            offset = rng.randrange(5)
            for j, r in enumerate(a):
                assignment[r['coin_id']] = (j + offset) % 5 + 1
        for fold in range(1, 6):
            group = [r for r in old_splits if int(r['seed']) == seed and int(r['fold']) == fold]
            for t in classes:
                a = extras[t]
                dev = [r for r in a if assignment[r['coin_id']] != fold]
                random.Random(seed + fold * 1009 + int(t) * 100003).shuffle(dev)
                v = {r['coin_id'] for r in dev[:min(len(dev), int(len(a) * .1 + .5))]}

                for r in a:
                    sp = 'test' if assignment[r['coin_id']] == fold else ('val' if r['coin_id'] in v else 'train')
                    group.append(dict(
                        seed=seed,
                        fold=fold,
                        split=sp,
                        **{k: r[k] for k in ('coin_id', 'original_type', 'merged_type', 'path')}
                    ))

            group.sort(key=lambda r: (
                ('train', 'val', 'test').index(r['split']),
                int(r['original_type']),
                int(r['coin_id'])
            ))
            split_rows.extend(group)

    write_csv(out / 'splits.csv', split_rows)
    actual = collections.Counter(r['original_type'] for r in rows)
    write_csv(
        out / 'class_counts.csv',
        [dict(original_type=t, balanced=30, expanded=actual[t], extra=actual[t] - 30) for t in classes]
    )
    write_json(out / 'preparation_environment.json', environment())
    write_json(
        out / 'protocol.json',
        dict(
            settings=PROTOCOL,
            dataset_root=str(root),
            manifest_sha256=digest(out / 'manifest.csv'),
            splits_sha256=digest(out / 'splits.csv'),
            mapping_sha256=digest(out / 'label_mapping.json'),
            code_sha256=code_hashes(),
            combined_images_confirmed=True,
            baseline_manifest_sha256=digest(base / 'manifest.csv'),
            baseline_splits_sha256=digest(base / 'splits.csv'),
            selection_rule='All valid additional Coin IDs in the same 28 types; exact duplicates excluded with baseline priority; original baseline fold assignments preserved; new IDs stratified by original type; no balancing or tuning changes.'
        )
    )
    validate_prepared(out)
    print(
        f'PASS: {len(rows)} coins, {len(rows)-840} additional; class sizes {min(actual.values())}–{max(actual.values())}; near duplicate flags {len(near)}. Review class_counts.csv and audit before launch.',
        flush=True
    )

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--confirm-combined', action='store_true')
    a = p.parse_args()
    if not a.confirm_combined:
        p.error('Confirm same combined image source with --confirm-combined')
    prepare(a.dataset, a.baseline, a.output)
