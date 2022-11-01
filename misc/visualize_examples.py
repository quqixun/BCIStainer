import os
import random
import imageio.v2 as iio
import matplotlib.pyplot as plt

from os.path import join as opj


if __name__ == '__main__':

    he_dir = './data/test/HE'
    ihc_dir = './data/test/IHC'
    ihc_pred_dir = './evaluations/stainer_basic_cmp/exp3/model_best_psnr_tta/IHC_pred'

    files = os.listdir(he_dir)
    files.sort()

    num_cols = 8
    num_rows = 3

    while True:
        rnd_files = random.sample(files, num_cols)

        plt.figure(figsize=(16, 6))
        for i, file in enumerate(rnd_files):
            he = iio.imread(opj(he_dir, file))
            ihc = iio.imread(opj(ihc_dir, file))
            ihc_pred = iio.imread(opj(ihc_pred_dir, file))

            plt.subplot(num_rows, num_cols, i + 1)
            plt.title(file.split('.')[0])
            plt.imshow(he)
            plt.xticks([])
            plt.yticks([])
            if i == 0:
                plt.ylabel('HE', fontsize=16)
            plt.subplot(num_rows, num_cols, i + 1 + num_cols)
            plt.imshow(ihc)
            plt.xticks([])
            plt.yticks([])
            if i == 0:
                plt.ylabel('IHC', fontsize=16)
            plt.subplot(num_rows, num_cols, i + 1 + num_cols * 2)
            plt.imshow(ihc_pred)
            plt.xticks([])
            plt.yticks([])
            if i == 0:
                plt.ylabel('IHC Pred', fontsize=16)
        
        plt.tight_layout()
        plt.show()
