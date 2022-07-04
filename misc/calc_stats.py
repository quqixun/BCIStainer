import os
import gc
import imageio
import numpy as np

from tqdm import tqdm
from os.path import join as opj


def load_image(image_path):
    image = imageio.v2.imread(image_path)
    return np.array(image)


def get_mean_std(data_dir):

    files = os.listdir(data_dir)
    data_list = [load_image(opj(data_dir, f))
                 for f in tqdm(files, ncols=66)]
    data_array = np.array(data_list)
    del data_list
    gc.collect()

    print(data_dir)
    print('mean:', np.mean(data_array, axis=(0, 1, 2)))
    print('std: ', np.std(data_array, axis=(0, 1, 2)))
    print('min: ', np.min(data_array, axis=(0, 1, 2)))
    print('max: ', np.max(data_array, axis=(0, 1, 2)))
    print('global_mean:', np.mean(data_array))
    print('global_std: ', np.std(data_array))
    print('global_min: ', np.min(data_array))
    print('global_max: ', np.max(data_array))
    print()


if __name__ == '__main__':

    he_dir  = '../data/train/HE'
    ihc_dir = '../data/train/IHC'

    get_mean_std(he_dir)
    get_mean_std(ihc_dir)
