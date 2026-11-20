"""Windows: package this code + all raw images from the 28 original class folders."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

CLASSES = '3314 3987 20178 20184 7686 7697 12884 6099 8455 7702 11243 5859 6007 20234 11758 12606 20231 864 285 2360 8778 11128 12997 4911 5224 7910 11761 20745'.split()
EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--zip', required=True)
    a = p.parse_args()
    root = Path(a.dataset).resolve()
    dest = Path(a.zip).resolve()
    here = Path(__file__).resolve().parent
    if dest.exists():
        raise FileExistsError(dest)
    files = []

    for t in CLASSES:
        folder = root / t
        if not folder.is_dir():
            raise ValueError(f'Missing source class folder: {folder}')
        images = [f for f in folder.rglob('*') if f.is_file() and f.suffix.lower() in EXT]
        if len(images) < 30:
            raise ValueError(f'Incomplete class folder: {folder}')
        files.extend(images)
    if len(files) <= 840:
        raise ValueError('Use full raw dataset, not the 840 selected images')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix('.tmp')

    try:
        with ZipFile(temp, 'w', compression=ZIP_DEFLATED, compresslevel=1) as z:
            for f in sorted(files):
                z.write(
                    f,
                    'coin_experiments_imbalanced_min30/dataset_types/' + f.relative_to(root).as_posix()
                )
            for f in sorted((here / 'coin_experiments').iterdir()):
                if f.is_file() and f.suffix in ('.py', '.txt'):
                    z.write(f, 'coin_experiments_imbalanced_min30/coin_experiments/' + f.name)
            z.write(
                here / 'train_imbalanced.slurm',
                'coin_experiments_imbalanced_min30/train_imbalanced.slurm'
            )
        with ZipFile(temp) as z:
            if z.testzip():
                raise ValueError('ZIP integrity failure')
        temp.replace(dest)
    finally:
        temp.unlink(missing_ok=True)

    print(f'Packaged {len(files)} raw images from 28 classes: {dest} ({dest.stat().st_size/1024**2:.1f} MiB)')
if __name__ == '__main__':
    main()
