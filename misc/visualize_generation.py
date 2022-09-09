import os
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt

from tqdm import tqdm
from os.path import join as opj


if __name__ == '__main__':

    he_dir = './data/test/HE'
    ihc_dir = './data/test/IHC'

    # exp = 
    exps = [
        'style_translator_cahr/exp13',
        'style_translator_cmp/exp2',
        'style_translator_cmp/exp4',
        'style_translator/exp6',
        'style_translator/exp6_1',
        'style_translator/exp4',
        'style_translator/exp4_1',
        'style_translator_cahr/exp9',
        'style_translator_cahr/exp9_1',
        'style_translator_cahr/exp14',
        'style_translator_cahr/exp6',
        'style_translator_cahr_cmp/exp4'
    ]
    ihc_pred_dir_base = './evaluations/{}/model_best_psnr_tta/IHC_pred'

    files = os.listdir(he_dir)
    files.sort()

    num_cols = 4
    num_rows = int(np.ceil(len(exps) / num_cols) + 1)

    for file in tqdm(files, ncols=66):
        he = iio.imread(opj(he_dir, file))
        ihc = iio.imread(opj(ihc_dir, file))

        plt.figure(figsize=(num_cols * 4, num_rows * 4))
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
