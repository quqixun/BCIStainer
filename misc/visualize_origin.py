import os
import imageio.v2 as iio
import matplotlib.pyplot as plt

from tqdm import tqdm
from os.path import join as opj


if __name__ == '__main__':

    he_dir = './data/test/HE'
    ihc_dir = './data/test/IHC'
    files = os.listdir(he_dir)
    files.sort()

    for file in tqdm(files, ncols=66):
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
