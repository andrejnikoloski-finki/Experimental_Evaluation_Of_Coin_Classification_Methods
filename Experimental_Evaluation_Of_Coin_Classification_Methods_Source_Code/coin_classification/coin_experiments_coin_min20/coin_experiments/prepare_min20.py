"""Use all distinct valid IDs in all classes with >=20 accepted coins; preserve expanded cohort folds for original and merged labels."""
import argparse, collections, random
from pathlib import Path
from common import read_csv, read_json, write_csv, write_json, digest, code_hashes, environment
from config import PROTOCOL
from prepare_dataset import discover, ORIGINAL
from prepare_imbalanced import image_info
from run_experiments import validate_prepared


def prepare(root, base, out, conflicts):
    root, base, out = map(lambda x: Path(x).resolve(), (root, base, out))
    if (out / 'protocol.json').exists() or (out / 'runs').exists():
        raise ValueError('Output already prepared; do not overwrite')
    q = read_json(base / 'protocol.json')
    oldmap = read_json(base / 'label_mapping.json')

    for f, k in [
        ('manifest.csv', 'manifest_sha256'),
        ('splits.csv', 'splits_sha256'),
        ('label_mapping.json', 'mapping_sha256')
    ]:
        if digest(base / f) != q[k]:
            raise ValueError('Baseline changed: ' + f)

    expected = dict(PROTOCOL, n_classes=28, dataset_mode='expanded_anchored')
    expected.pop('minimum_per_class')
    if q['settings'] != expected:
        raise ValueError('Unexpected expanded baseline settings')
    old = read_csv(base / 'manifest.csv')
    old_splits = read_csv(base / 'splits.csv')
    anchor = {r['coin_id']: r for r in old}
    oldclasses = set(oldmap['original'])
    if len(old) != 1602 or len(anchor) != 1602 or len(oldclasses) != 28:
        raise ValueError('Expected completed 1602-coin baseline')
    rec, rejected = discover(root, None)
    grouped = collections.defaultdict(list)
    for path, t, c in rec:
        grouped[c].append((path, t))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(
        out / 'rejected_files.csv',
        [dict(path=p, reason=r) for p, r in rejected],
        ['path', 'reason']
    )
    if any(r == 'unrecognized_coin_id' for p, r in rejected):
        raise ValueError('Unrecognized Coin IDs: inspect rejected_files.csv')
    conflicts = set(Path(conflicts).read_text(encoding='utf-8-sig').split())
    conflicts.update(c for c, items in grouped.items() if len({t for _, t in items}) > 1)
    excluded = conflicts & set(anchor)
    write_csv(
        out / 'excluded_baseline_conflicts.csv',
        [dict(
            coin_id=c,
            baseline_type=anchor[c]['original_type'],
            baseline_path=anchor[c]['path'],
            types_in_uploaded_data=';'.join(sorted({t for _, t in grouped[c]}, key=int)),
            reason='cross-type Coin ID in full source'
        ) for c in sorted(excluded, key=int)],
        ['coin_id', 'baseline_type', 'baseline_path', 'types_in_uploaded_data', 'reason']
    )
    old = [r for r in old if r['coin_id'] not in excluded]
    old_splits = [r for r in old_splits if r['coin_id'] not in excluded]
    anchor = {r['coin_id']: r for r in old}
    print(
        f'Excluded {len(excluded)} conflicting baseline Coin IDs: {sorted(excluded,key=int)}; retained {len(anchor)} baseline coins.',
        flush=True
    )
    rows = []
    seen_bytes = {}
    seen_pixels = {}
    perceptual = []
    audit = []

    for r in old:
        r = dict(r, merged_type=r['original_type'])
        p = (root / r['path']).resolve()
        c = r['coin_id']
        if not p.is_relative_to(root) or {t for _, t in grouped[c]} != {r['original_type']}:
            raise ValueError('Anchor path/type mismatch: ' + c)
        b, h, d = image_info(p)
        if (b, h) != (r['sha256'], r['pixel_sha256']) or b in seen_bytes or h in seen_pixels:
            raise ValueError('Anchor file changed: ' + str(p))
        seen_bytes[b] = c
        seen_pixels[h] = c
        rows.append(r)
        perceptual.append((r, d))

    pool = collections.defaultdict(list)

    for c, items in grouped.items():
        if c in conflicts:
            audit.append(dict(coin_id=c, path='', reason='conflicting types'))
            continue
        if c not in anchor:
            pool[items[0][1]].append((c, items))

    # Existing classes first; then numeric order. Rejected classes do not reserve hashes.
    counts = []
    oldcounts = collections.Counter(r['original_type'] for r in old)

    for t in sorted(set(pool) | oldclasses, key=lambda t: (t not in oldclasses, int(t))):
        local = []
        lb = {}
        lh = {}
        lp = []

        for c, items in sorted(pool[t], key=lambda x: int(x[0])):
            for path, _ in sorted(
                items,
                key=lambda item: (not bool(ORIGINAL.search(item[0].stem)), str(item[0]))
            ):
                try:
                    b, h, d = image_info(path)
                except Exception as e:
                    audit.append(dict(coin_id=c, path=str(path), reason='unreadable: ' + str(e)))
                    continue
                duplicate = seen_bytes.get(b) or seen_pixels.get(h) or lb.get(b) or lh.get(h)
                if duplicate:
                    audit.append(dict(
                        coin_id=c,
                        path=str(path),
                        reason='exact duplicate of ' + duplicate
                    ))
                    continue

                r = dict(
                    coin_id=c,
                    original_type=t,
                    path=path.relative_to(root).as_posix(),
                    sha256=b,
                    pixel_sha256=h,
                    available_views=len(items),
                    merged_type=t
                )
                local.append(r)
                lp.append((r, d))
                lb[b] = c
                lh[h] = c
                break

        n = oldcounts[t] + len(local)
        keep = n >= PROTOCOL['minimum_per_class']
        counts.append(dict(
            original_type=t,
            baseline=oldcounts[t],
            valid_coins=n,
            selected=keep,
            extra=len(local) if keep else 0
        ))
        if keep:
            rows.extend(local)
            perceptual.extend(lp)
            seen_bytes.update(lb)
            seen_pixels.update(lh)

    write_csv(out / 'class_selection.csv', counts)
    write_csv(out / 'selection_audit.csv', audit, ['coin_id', 'path', 'reason'])
    rows.sort(key=lambda r: (int(r['original_type']), int(r['coin_id'])))
    actual = collections.Counter(r['original_type'] for r in rows)
    classes = sorted(actual, key=int)
    if not oldclasses <= set(classes) or len(classes) <= 28:
        raise ValueError('No new valid classes')
    label_map = {t: t for t in classes}

    for g in PROTOCOL['merge_groups']:
        if not set(g) <= set(classes) or any(label_map[t] != t for t in g):
            raise ValueError('Invalid or overlapping merge groups')
        for t in g:
            label_map[t] = '+'.join(g)
    if any(label_map[t] != oldmap['original_to_merged'][t] for t in oldclasses):
        raise ValueError('Baseline merge mapping differs')
    mapping = dict(
        original=classes,
        merged=sorted(set(label_map.values()), key=lambda c: tuple(map(int, c.split('+')))),
        original_to_merged=label_map
    )
    for r in rows:
        r['merged_type'] = label_map[r['original_type']]
    merged_counts = collections.Counter(r['merged_type'] for r in rows)
    write_csv(
        out / 'merged_class_counts.csv',
        [dict(merged_type=t, valid_coins=merged_counts[t]) for t in mapping['merged']]
    )
    write_csv(out / 'manifest.csv', rows)
    write_json(out / 'label_mapping.json', mapping)
    write_csv(
        out / 'class_counts.csv',
        [dict(
            original_type=t,
            baseline=oldcounts[t],
            expanded=actual[t],
            extra=actual[t] - oldcounts[t]
        ) for t in classes]
    )
    # Stream near-duplicate pairs instead of holding them all in RAM.
    import csv
    fields = ['coin_id_a', 'type_a', 'path_a', 'coin_id_b', 'type_b', 'path_b', 'dhash_distance']
    near = 0

    with (out / 'near_duplicate_review.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i, (a, ha) in enumerate(perceptual):
            for b, hb in perceptual[i + 1:]:
                d = (ha ^ hb).bit_count()
                if d <= 3:
                    w.writerow([
                        a['coin_id'],
                        a['original_type'],
                        a['path'],
                        b['coin_id'],
                        b['original_type'],
                        b['path'],
                        d
                    ])
                    near += 1

    extras = collections.defaultdict(list)
    for r in rows:
        if r['coin_id'] not in anchor:
            extras[r['original_type']].append(r)
    split_rows = []

    for seed in PROTOCOL['seeds']:
        assignment = {}
        for t in classes:
            a = list(extras[t])
            rng = random.Random(seed + int(t) * 1009)
            rng.shuffle(a)
            offset = rng.randrange(5)
            for j, r in enumerate(a):
                assignment[r['coin_id']] = (j + offset) % 5 + 1
        for fold in range(1, 6):
            group = [dict(
                r,
                merged_type=label_map[r['original_type']]
            ) for r in old_splits if int(r['seed']) == seed and int(r['fold']) == fold]
            for t in classes:
                a = extras[t]
                dev = [r for r in a if assignment[r['coin_id']] != fold]
                random.Random(seed + fold * 1009 + int(t) * 100003).shuffle(dev)
                # Target total per-class validation count, accounting for preserved old members.
                oldval = sum(r['original_type'] == t and r['split'] == 'val' for r in group)
                nv = max(0, min(len(dev), int(actual[t] * .1 + .5) - oldval))
                v = {r['coin_id'] for r in dev[:nv]}

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
    # Verify preserved assignments exactly before training.
    def keys(rs):
        return {(int(r['seed']), int(r['fold']), r['coin_id'], r['split']) for r in rs}
    if keys(r for r in split_rows if r['coin_id'] in anchor) != keys(old_splits):
        raise ValueError('Anchor split mismatch')
    write_csv(out / 'splits.csv', split_rows)
    write_json(out / 'preparation_environment.json', environment())
    p = dict(PROTOCOL, n_classes=len(classes))
    write_json(
        out / 'protocol.json',
        dict(
            settings=p,
            dataset_root=str(root),
            manifest_sha256=digest(out / 'manifest.csv'),
            splits_sha256=digest(out / 'splits.csv'),
            mapping_sha256=digest(out / 'label_mapping.json'),
            code_sha256=code_hashes(),
            combined_images_confirmed=True,
            baseline_manifest_sha256=digest(base / 'manifest.csv'),
            baseline_splits_sha256=digest(base / 'splits.csv'),
            excluded_baseline_coin_ids=sorted(excluded, key=int),
            retained_baseline_coins=len(anchor),
            selection_rule='All valid IDs in classes with at least 20 accepted coins; baseline cohort prioritized except globally conflicting IDs, excluded from every split. Remaining baseline folds preserved. New types ordered numerically, exact duplicates and cross-type IDs excluded; original labels and exactly the two predefined merged pairs use identical folds.'
        )
    )
    validate_prepared(out)
    print(
        f'PASS: {len(rows)} coins, {len(classes)} original / {len(mapping["merged"])} merged classes, class sizes {min(actual.values())}–{max(actual.values())}, {len(rows)-len(anchor)} additional coins, {near} near-duplicate flags. Preserved all {len(anchor)} retained anchor fold assignments.',
        flush=True
    )

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--conflicts', required=True)
    p.add_argument('--confirm-combined', action='store_true')
    a = p.parse_args()
    if not a.confirm_combined:
        p.error('Confirm same combined image source with --confirm-combined')
    prepare(a.dataset, a.baseline, a.output, a.conflicts)
