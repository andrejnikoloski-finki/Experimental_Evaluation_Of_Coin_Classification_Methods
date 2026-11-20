"""Functional integration test using artificial images, random weights, and one epoch. NOT research results."""
import tempfile, copy
from pathlib import Path
import numpy as np
from PIL import Image
from config import PROTOCOL, METHODS, EXPERIMENTS
from prepare_dataset import prepare
from run_experiments import validate_prepared, run
from analyze_results import analyze
from common import read_csv


def main():
    p = copy.deepcopy(PROTOCOL)
    p.update(
        n_classes=3,
        samples_per_class=8,
        preferred_classes=['1', '2', '3'],
        required_classes=['1', '2', '3'],
        merge_groups=[['1', '2']],
        seeds=[42],
        folds=2,
        validation_fraction_of_all=.25,
        max_epochs=1,
        patience=1,
        image_size=64,
        sift_size=100,
        visual_words=6,
        batch_size=4,
        siamese_pairs_per_epoch=8
    )
    with tempfile.TemporaryDirectory(prefix='coin_smoke_') as d:
        root = Path(d) / 'dataset_types'
        out = Path(d) / 'output'
        rng = np.random.default_rng(99)

        for c in range(1, 4):
            (root / str(c)).mkdir(parents=True)
            for j in range(8):
                im = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
                path = root / str(c) / f'CN_type_{c}_cn_coin_{c*100+j}_o_merged.png'
                Image.fromarray(im).save(path)

        Image.fromarray(rng.integers(
            0,
            256,
            size=(96, 96, 3),
            dtype=np.uint8
        )).save(root / '1' / 'CN_type_1_cn_coin_100_p_merged.png')
        prepare(root, out, p, confirmed=True)
        validate_prepared(out)
        run(out, EXPERIMENTS, METHODS, 'cpu', pretrained=False)
        analyze(out)
        assert len(read_csv(out / 'analysis' / 'all_fold_results.csv')) == 16
        assert len(read_csv(out / 'analysis' / 'all_candidate_validation_results.csv')) == 32
        assert len(read_csv(out / 'analysis' / 'paired_corrected_tests_holm.csv')) == 24
        run(out, ['original'], ['sift_svm'], 'cpu', pretrained=False)
        print('SMOKE TEST PASSED: all methods, both variants and label setups, checkpoint reload, selection, saved predictions, analysis, resume.')
if __name__ == '__main__':
    main()
