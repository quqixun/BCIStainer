import os
import numpy as np
import imageio.v2 as iio
import albumentations as A
# import matplotlib.pyplot as plt

from itertools import product
from os.path import join as opj
from ..utils import normalize_image
from torch.utils.data import Dataset, DataLoader


class BCICAHRDataset(Dataset):

    def __init__(self, data_dir, mode, crop_size=512, random_crop=False,
                 augment=False, norm_method='global_minmax'):
        super(BCICAHRDataset, self).__init__()

        he_dir  = opj(data_dir, 'HE')
        ihc_dir = opj(data_dir, 'IHC')
        files   = os.listdir(he_dir)

        self.he_list = []
        self.ihc_list = []
        self.level_list = []
        for f in files:
            self.he_list.append(opj(he_dir, f))
            self.ihc_list.append(opj(ihc_dir, f))
            self.level_list.append(int(f.split('_')[2][0]))

        self.augment = augment
        if self.augment:
            self.transform = A.Compose(
                [
                    A.Flip(p=0.5),
                    A.Transpose(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomResizedCrop(
                        p=0.2, height=1024, width=1024,
                        scale=(0.8, 1.0)
                    ),
                    A.OneOf([
                        A.GridDistortion(p=0.3),
                        A.ElasticTransform(p=0.3),
                        A.OpticalDistortion(p=0.3)
                    ], p=0.3)
                ],
                additional_targets={'image0': 'image'}
            )

        self.mode = mode
        self.full_size = 1024
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.norm_method = norm_method
        self.crop_range  = self.full_size - self.crop_size

        if not self.random_crop:
            upper_idx = self.full_size - self.crop_size + 1
            self.crop_row_idxs  = list(range(0, upper_idx, crop_size // 2))
            self.crop_col_idxs  = list(range(0, upper_idx, crop_size // 2))
            self.crop_rowx_cols = list(product(
                self.crop_row_idxs, self.crop_col_idxs
            ))

    def __len__(self):
        return len(self.he_list)

    def _getitem_train(self, he, ihc):

        if self.augment:
            transformed = self.transform(image=he, image0=ihc)
            he  = transformed['image']
            ihc = transformed['image0']

        # crop image
        if self.random_crop:
            row_idx = np.random.randint(self.crop_range)
            col_idx = np.random.randint(self.crop_range)
        else:
            row_idx = np.random.choice(self.crop_row_idxs)
            col_idx = np.random.choice(self.crop_col_idxs)

        crop_idx = np.array([row_idx, col_idx])
        he_crop = he[
            row_idx:row_idx + self.crop_size,
            col_idx:col_idx + self.crop_size
        ].copy()
        ihc_crop = ihc[
            row_idx:row_idx + self.crop_size,
            col_idx:col_idx + self.crop_size
        ].copy()

        # plt.figure(figsize=(15, 6))
        # plt.subplot(131)
        # plt.title(f'HE  - {he.shape}')
        # plt.imshow(he)
        # plt.axis('off')
        # plt.subplot(132)
        # plt.title(f'IHC - {ihc.shape}')
        # plt.imshow(ihc)
        # plt.axis('off')
        # plt.subplot(133)
        # plt.title(f'HE Crop  - {he_crop.shape} - ({row_idx}, {col_idx})')
        # plt.imshow(he_crop)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

        he  = normalize_image(he, 'he', self.norm_method)
        he  = he.transpose(2, 0, 1).astype(np.float32)
        ihc = normalize_image(ihc, 'ihc', self.norm_method)
        ihc = ihc.transpose(2, 0, 1).astype(np.float32)
        he_crop  = normalize_image(he_crop, 'he', self.norm_method)
        he_crop  = he_crop.transpose(2, 0, 1).astype(np.float32)
        ihc_crop = normalize_image(ihc_crop, 'ihc', self.norm_method)
        ihc_crop = ihc_crop.transpose(2, 0, 1).astype(np.float32)

        return he, ihc, he_crop, ihc_crop, crop_idx

    def _getitem_val(self, he, ihc):

        he_crop_list  = []
        for row_idx, col_idx in self.crop_rowx_cols:
            he_crop = he[
                row_idx:row_idx + self.crop_size,
                col_idx:col_idx + self.crop_size
            ].copy()
            he_crop_list.append(he_crop)
        he_crop  = np.array(he_crop_list)

        # print(he.shape, he_crop.shape)

        # plt.figure(figsize=(8, 8))
        # plt.imshow(he[0])
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

        # plt.figure(figsize=(10, 10))
        # for i in range(len(he_crop)):
        #     image = he_crop[i]
        #     row_idx, col_idx = self.crop_rowx_cols[i]
        #     plt.subplot(3, 3, i + 1)
        #     plt.title(f'IHC Crop {i + 1}- {image.shape} - ({row_idx}, {col_idx})')
        #     plt.imshow(image)
        #     plt.axis('off')
        # plt.tight_layout()
        # plt.show()

        he       = normalize_image(he, 'he', self.norm_method)
        he       = he.transpose(2, 0, 1).astype(np.float32)
        ihc      = normalize_image(ihc, 'ihc', self.norm_method)
        ihc      = ihc.transpose(2, 0, 1).astype(np.float32)
        he_crop  = normalize_image(he_crop, 'he', self.norm_method)
        he_crop  = he_crop.transpose(0, 3, 1, 2).astype(np.float32)
        crop_idx = np.array(self.crop_rowx_cols)

        return he, ihc, he_crop, crop_idx

    def __getitem__(self, index):

        he    = np.array(iio.imread(self.he_list[index]))
        ihc   = np.array(iio.imread(self.ihc_list[index]))
        level = self.level_list[index]

        if self.mode == 'train':
            he, ihc, he_crop, ihc_crop, crop_idx = self._getitem_train(he, ihc)
            return he, ihc, level, he_crop, ihc_crop, crop_idx
        else:  # self.mode == 'val'
            he, ihc, he_crop, crop_idx = self._getitem_val(he, ihc)
            return he, ihc, he_crop, crop_idx


def get_cahr_dataloader(mode, data_dir, configs):
    assert mode in ['train', 'val']

    if mode == 'train':
        batch_size  = configs.train_batch
        drop_last   = True
        shuffle     = True
        augment     = True
        random_crop = configs.random_crop
    else:  # mode == 'val'
        assert configs.val_batch == 1
        batch_size  = configs.val_batch
        drop_last   = False
        shuffle     = False
        augment     = False
        random_crop = False

    dataset = BCICAHRDataset(
        data_dir=data_dir,
        mode=mode,
        crop_size=configs.crop_size,
        random_crop=random_crop,
        augment=augment,
        norm_method=configs.norm_method
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=configs.num_workers,
        pin_memory=configs.pin_memory,
        drop_last=drop_last,
        shuffle=shuffle
    )

    return dataloader
