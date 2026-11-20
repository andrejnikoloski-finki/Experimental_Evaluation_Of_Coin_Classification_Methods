"""Package all potentially eligible raw class folders using distinct CN Coin IDs."""
import argparse, collections, csv, io, re
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
CN_ID = re.compile(r'(?:^|_)cn_coin_(\d+)(?=_|$)', re.I)
SINGLE = re.compile(r'(?:^|_)(?:obv|rev|obverse|reverse)(?=_|$)', re.I)


def pack(root, dest):
    root, dest = Path(root).resolve(), Path(dest).resolve()
    here = Path(__file__).resolve().parent
    if not root.is_dir() or dest.exists():
        raise ValueError('Invalid source or ZIP already exists')
    files = collections.defaultdict(list)
    ids = collections.defaultdict(set)
    types = collections.defaultdict(set)

    for f in root.rglob('*'):
        if not f.is_file() or f.suffix.lower() not in EXT:
            continue
        folders = [x for x in f.relative_to(root).parts[:-1] if x.isdigit()]
        if not folders:
            continue
        t = str(int(folders[-1]))
        files[t].append(f)
        m = CN_ID.search(f.stem)
        if m and not SINGLE.search(f.stem):
            c = str(int(m.group(1)))
            ids[t].add(c)
            types[c].add(t)

    candidates = sorted((t for t, v in ids.items() if len(v) >= 20), key=int)
    if len(candidates) <= 28:
        raise ValueError('Expected more than 28 potential classes; use full dataset_types')
    conflicts = sorted((c for c, v in types.items() if len(v) > 1), key=int)
    inv = io.StringIO()
    w = csv.writer(inv)
    w.writerow(['original_type', 'raw_distinct_ids'])
    w.writerows((t, len(ids[t])) for t in candidates)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp')

    try:
        with ZipFile(tmp, 'w', compression=ZIP_DEFLATED, compresslevel=1) as z:
            for t in candidates:
                for f in sorted(files[t]):
                    z.write(
                        f,
                        'coin_experiments_coin_min20/dataset_types/' + f.relative_to(root).as_posix()
                    )
            for f in sorted((here / 'coin_experiments').iterdir()):
                if f.is_file() and f.suffix in ('.py', '.txt'):
                    z.write(f, 'coin_experiments_coin_min20/coin_experiments/' + f.name)
            z.write(here / 'train_min20.slurm', 'coin_experiments_coin_min20/train_min20.slurm')
            z.writestr('coin_experiments_coin_min20/raw_inventory.csv', inv.getvalue())
            z.writestr('coin_experiments_coin_min20/conflicting_ids.txt', '\n'.join(conflicts))
        with ZipFile(tmp) as z:
            if z.testzip():
                raise ValueError('ZIP integrity failure')
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)

    print(f'Packaged {len(candidates)} candidate classes, {sum(len(ids[t]) for t in candidates)} raw distinct IDs per class, {sum(len(files[t]) for t in candidates)} image files; {len(conflicts)} cross-type IDs in full source. {dest}')
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--zip', required=True)
    a = p.parse_args()
    pack(a.dataset, a.zip)
