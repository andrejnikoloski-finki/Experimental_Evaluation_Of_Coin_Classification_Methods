import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZIP_DEFLATED


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def package(project, prepared, destination):
    project, prepared, destination = (Path(p).resolve() for p in (project, prepared, destination))
    q = json.loads((prepared / 'protocol.json').read_text(encoding='utf-8'))
    root = Path(q['dataset_root']).resolve()
    settings = q['settings']
    if (
        settings['n_classes'],
        settings['samples_per_class'],
        settings['folds'],
        settings['seeds']
    ) != (28, 30, 5, [42, 123, 2026]):
        raise ValueError('Unexpected protocol, expected 28 x 30 and 3 x 5 folds')
    if (prepared / 'runs').exists() or (prepared / 'RUNNING.lock').exists() or (prepared / 'execution_environment.json').exists():
        raise ValueError('Training has already started or an execution environment exists. This tool is for pre-training transfer only.')
    for filename, key in [
        ('manifest.csv', 'manifest_sha256'),
        ('splits.csv', 'splits_sha256'),
        ('label_mapping.json', 'mapping_sha256')
    ]:
        if sha(prepared / filename) != q[key]:
            raise ValueError(f'Changed prepared file: {filename}')

    actual_code = {p.name: sha(p) for p in project.glob('*.py') if p.name != 'smoke_test.py'}
    if actual_code != q['code_sha256']:
        raise ValueError('Project code differs from the code used for dataset preparation. Use the exact local project that generated the protocol.')
    with (prepared / 'manifest.csv').open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 840 or len({r['coin_id'] for r in rows}) != 840 or len({r['path'] for r in rows}) != 840:
        raise ValueError('Unexpected manifest size or duplicates')
    images = []

    for r in rows:
        rel = PurePosixPath(r['path'].replace('\\', '/'))
        if rel.is_absolute() or '..' in rel.parts or ':' in str(rel):
            raise ValueError(f'Invalid relative path: {rel}')
        p = (root / Path(*rel.parts)).resolve()
        if not p.is_relative_to(root) or sha(p) != r['sha256']:
            raise ValueError(f'Image missing or changed: {p}')
        images.append((p, 'dataset_types/' + rel.as_posix()))

    files = [(
        p,
        'coin_experiments_balanced/' + p.name
    ) for p in sorted(project.iterdir()) if p.is_file() and p.suffix in ('.py', '.txt', '.md')]
    files += [(p, 'experiments_28/' + p.name) for p in sorted(prepared.iterdir()) if p.is_file() and p.suffix in ('.csv', '.json')]
    files += images
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    tmp = destination.with_suffix('.zip.tmp')

    try:
        with ZipFile(tmp, 'w', compression=ZIP_DEFLATED, compresslevel=1) as z:
            for p, rel in files:
                z.write(p, 'coin_classification/' + rel)
            z.writestr('coin_classification/relocate.py', RELOCATE)
        with ZipFile(tmp) as z:
            if z.testzip() is not None:
                raise ValueError('ZIP integrity check failed')
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)

    print(f'PASS: 840 image hashes verified; protocol, splits and project code unchanged.\nZIP: {destination}\nSize: {destination.stat().st_size / 1024**2:.1f} MiB')


RELOCATE = """import json
from pathlib import Path


root = Path(__file__).resolve().parent
p = root / "experiments_28" / "protocol.json"
q = json.loads(p.read_text(encoding="utf-8"))

if (p.parent / "runs").exists():
    raise RuntimeError("Refusing to relocate after training has started")

q["dataset_root"] = str((root / "dataset_types").resolve())
tmp = p.with_suffix(".json.tmp")
tmp.write_text(json.dumps(q, indent=2, ensure_ascii=False), encoding="utf-8")
tmp.replace(p)
print("Dataset root:", q["dataset_root"])
"""

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', required=True)
    p.add_argument('--prepared', required=True)
    p.add_argument('--zip', required=True)
    a = p.parse_args()
    package(a.project, a.prepared, a.zip)
