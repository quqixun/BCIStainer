import os
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt

from tqdm import tqdm
from os.path import join as opj


if __name__ == '__main__':

    he_dir = './data/test/HE'
    ihc_dir = './data/test/IHC'
    exps = [
        'stainer_basic/exp1',
        'stainer_basic/exp2',
        'stainer_basic/exp3',
        'stainer_basic/exp4',
        'stainer_basic_cmp/exp1',
        'stainer_basic_cmp/exp2',
        'stainer_basic_cmp/exp3',
        'stainer_basic_cmp/exp4'
    ]
    ihc_pred_dir_base = './evaluations/{}/model_best_psnr_tta/IHC_pred'

    files = os.listdir(he_dir)
    files.sort()

    num_cols = 4
    num_rows = int(np.ceil(len(exps) / num_cols) + 1)

    for file in tqdm(files, ncols=66):
        he = iio.imread(opj(he_dir, file))
        ihc = iio.imread(opj(ihc_dir, file))

        plt.figure(figsize=(num_cols * 5, num_rows * 5))
        plt.subplot(num_rows, num_cols, 1)
        plt.title('HE')
        plt.imshow(he)
        plt.axis('off')
        plt.subplot(num_rows, num_cols, 2)
        plt.title('IHC')
        plt.imshow(ihc)
        plt.axis('off')

        for i, exp in enumerate(exps):
            ihc_pred = iio.imread(opj(ihc_pred_dir_base.format(exp), file))
            plt.subplot(num_rows, num_cols, num_cols + i + 1)
            plt.title(exp)
            plt.imshow(ihc_pred)
            plt.axis('off')

        plt.tight_layout()
        plt.show()
