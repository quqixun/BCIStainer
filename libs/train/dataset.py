import os
import numpy as np
import imageio.v2 as iio
import albumentations as A
import matplotlib.pyplot as plt

from os.path import join as opj
from ..utils import normalize_image
from torch.utils.data import Dataset, DataLoader


class BCIDataset(Dataset):

    def __init__(self, data_dir, augment=False, norm_method='global_minmax'):
        super(BCIDataset, self).__init__()

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

        self.norm_method = norm_method

        return

    def __len__(self):
        return len(self.he_list)

    def __getitem__(self, index):

        he    = np.array(iio.imread(self.he_list[index]))
        ihc   = np.array(iio.imread(self.ihc_list[index]))
        level = self.level_list[index]

        if self.augment:
            transformed = self.transform(image=he, image0=ihc)
            he  = transformed['image']
            ihc = transformed['image0']

            # plt.figure(figsize=(18, 10))
            # plt.subplot(221)
            # plt.title(f'HE  - {he.shape}')
            # plt.imshow(he)
            # plt.axis('off')
            # plt.subplot(222)
            # plt.title(f'IHC - {ihc.shape}')
            # plt.imshow(ihc)
            # plt.axis('off')
            # plt.subplot(223)
            # plt.title(f'HE A  - {hea.shape}')
            # plt.imshow(hea)
            # plt.axis('off')
            # plt.subplot(224)
            # plt.title(f'IHC A - {ihca.shape}')
            # plt.imshow(ihca)
            # plt.axis('off')
            # plt.tight_layout()
            # plt.show()

        he  = normalize_image(he, 'he', self.norm_method)
        he  = he.transpose(2, 0, 1).astype(np.float32)
        ihc = normalize_image(ihc, 'ihc', self.norm_method)
        ihc = ihc.transpose(2, 0, 1).astype(np.float32)

        return he, ihc, level


def get_dataloader(mode, data_dir, configs):
    assert mode in ['train', 'val', 'test']

    if mode == 'train':
        batch_size = configs.train_batch
        drop_last  = True
        shuffle    = True
        augment    = True
    else:  # mode in ['val', 'test']
        batch_size = configs.val_batch
        drop_last  = False
        shuffle    = False
        augment    = False

    dataset = BCIDataset(
        data_dir, augment, configs.norm_method
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
