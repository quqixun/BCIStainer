import os
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt

from os.path import join as opj


if __name__ == '__main__':

    he_dir = './data/test/HE'
    ihc_dir = './data/test/IHC'

    # exp = 'resnet_ada_l_v2_exp5'
    # exp = 'resnet_mod_v2_exp3'
    exp = 'resnet_ada_v2_exp18'
    ihc_pred_dir = f'./outputs_test/{exp}/model_best_psnr_tta/IHC_pred'

    files = os.listdir(he_dir)
    files.sort()

    for file in files:
        he = iio.imread(opj(he_dir, file))
        ihc = iio.imread(opj(ihc_dir, file))
        ihc_pred = iio.imread(opj(ihc_pred_dir, file))

        plt.figure(figsize=(18, 6))
        plt.subplot(131)
        plt.imshow(he)
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(ihc)
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(ihc_pred)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
