import os
import imageio
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from tqdm import tqdm
from os.path import join as opj
from ..utils import HE_STAT, IHC_STAT
from torch.utils.data import Dataset, DataLoader


class BCIDataset(Dataset):

    def __init__(self, data_dir, augment=False):
        super(BCIDataset, self).__init__()

        he_dir  = opj(data_dir, 'HE')
        ihc_dir = opj(data_dir, 'IHC')
        files   = os.listdir(he_dir)

        self.he_list = []
        self.ihc_list = []
        self.level_list = []

        print(f'Loading data from {data_dir} ...')
        for f in tqdm(files, ncols=66):
            self.he_list.append(self.load_image(opj(he_dir, f)))
            self.ihc_list.append(self.load_image(opj(ihc_dir, f)))
            self.level_list.append(int(f.split('_')[2][0]))

        self.augment = augment
        if self.augment:
            self.transform = A.Compose(
                [
                    A.Flip(p=0.5),
                    A.Transpose(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomResizedCrop(
                        p=0.3, height=1024, width=1024,
                        scale=(0.5, 1.0), interpolation=4
                    )
                ],
                additional_targets={'image0': 'image'}
            )

        return

    @staticmethod
    def load_image(image_path):
        image = imageio.v2.imread(image_path)
        return np.array(image)

    def __len__(self):
        return len(self.he_list)

    def __getitem__(self, index):

        he    = self.he_list[index]
        ihc   = self.ihc_list[index]
        level = self.level_list[index]

        if self.augment:
            transformed = self.transform(image=he, image0=ihc)
            he  = transformed['image']
            ihc = transformed['image0']

        # plt.figure(figsize=(18, 10))
        # plt.subplot(121)
        # plt.title(f'HE  - {he.shape}')
        # plt.imshow(he)
        # plt.axis('off')
        # plt.subplot(122)
        # plt.title(f'IHC - {ihc.shape}')
        # plt.imshow(ihc)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

        he  = (he - HE_STAT['mean']) / HE_STAT['std']
        he  = he.transpose(2, 0, 1).astype(np.float32)
        ihc = (ihc - IHC_STAT['mean']) / IHC_STAT['std']
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

    dataset = BCIDataset(data_dir, augment)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=configs.num_workers,
        pin_memory=configs.pin_memory,
        drop_last=drop_last,
        shuffle=shuffle
    )

    return dataloader
