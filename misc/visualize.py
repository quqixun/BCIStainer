import os
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt

from os.path import join as opj


if __name__ == '__main__':

    # he_dir = './data/test/HE'
    # ihc_dir = './data/test/IHC'

    # exp = 'style_translator/exp1'
    # ihc_pred_dir = f'./evaluations/{exp}/model_best_psnr_tta/IHC_pred'

    # files = os.listdir(he_dir)
    # files.sort()

    # for file in files:
    #     he = iio.imread(opj(he_dir, file))
    #     ihc = iio.imread(opj(ihc_dir, file))
    #     ihc_pred = iio.imread(opj(ihc_pred_dir, file))

    #     plt.figure(figsize=(18, 6))
    #     plt.subplot(131)
    #     plt.imshow(he)
    #     plt.axis('off')
    #     plt.subplot(132)
    #     plt.imshow(ihc)
    #     plt.axis('off')
    #     plt.subplot(133)
    #     plt.imshow(ihc_pred)
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.show()

    he_dir = './data/test/HE'
    ihc_dir = './data/test/IHC'
    files = os.listdir(he_dir)
    files.sort()

    for file in files:
        he = iio.imread(opj(he_dir, file))
        ihc = iio.imread(opj(ihc_dir, file))

        plt.figure(figsize=(13, 7))
        plt.subplot(121)
        plt.title(f'{file} - HE')
        plt.imshow(he)
        plt.axis('off')
        plt.subplot(122)
        plt.title(f'{file} - IHC')
        plt.imshow(ihc)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
