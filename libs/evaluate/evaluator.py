import os
import cv2
import torch
import numpy as np
import pandas as pd
import imageio.v2 as iio
import matplotlib.pyplot as plt

from tqdm import tqdm
from ..models import define_G
from os.path import join as opj
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from ..utils import normalize_image, unnormalize_image


class BCIEvaluator(object):

    def __init__(self, configs, model_path, apply_tta=False):

        self.apply_tta = apply_tta

        # model
        self.G_params = configs.G
        self.device   = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.norm_method = configs.loader.norm_method

        self._load_model(model_path)

    def _load_model(self, model_path):

        self.G = define_G(self.G_params)
        G_dict = torch.load(model_path, map_location='cpu')
        self.G.load_state_dict(G_dict)
        self.G = self.G.to(self.device)
        self.G.eval()

        return
    
    def forward(self, data_dir, output_dir):

        output_dir += '_tta' if self.apply_tta else ''
        pred_dir = opj(output_dir, 'IHC_pred')
        os.makedirs(pred_dir, exist_ok=True)
        metrics_path = opj(output_dir, 'metrics.csv')

        he_dir  = opj(data_dir, 'HE')
        ihc_dir = opj(data_dir, 'IHC')
        files   = os.listdir(he_dir)
        files.sort()

        metrics_list = []
        for file in tqdm(files, ncols=88):
            he_path = opj(he_dir, file)
            ihc_path = opj(ihc_dir, file)
            ihc_pred_path = opj(pred_dir, file)

            if self.apply_tta:
                self.predict_tta(he_path, ihc_pred_path)
            else:
                self.predict(he_path, ihc_pred_path)

            psnr, ssim = self.evaluate(ihc_path, ihc_pred_path)

            metrics_list.append([he_path, ihc_path, ihc_pred_path, psnr, ssim])

        columns = ['he', 'ihc', 'ihc_pred', 'psnr', 'ssim']
        metrics = pd.DataFrame(metrics_list, columns=columns)
        metrics.to_csv(metrics_path, index=False)

        psnr_avg = np.mean(metrics['psnr'])
        psnr_std = np.std(metrics['psnr'])
        ssim_avg = np.mean(metrics['ssim'])
        ssim_std = np.std(metrics['ssim'])

        print(f'- Output: {output_dir}')
        print(f'- PSNR:   {psnr_avg:.3f} ± {psnr_std:.3f}')
        print(f'- SSIM:   {ssim_avg:.3f} ± {ssim_std:.3f}')

        return

    @torch.no_grad()
    def predict(self, he_path, ihc_pred_path):

        he_ori = iio.imread(he_path)
        he = normalize_image(he_ori, 'he', self.norm_method)
        he = he.transpose(2, 0, 1).astype(np.float32)[None, ...]
        he = torch.Tensor(he).to(self.device)

        try:
            ihc_pred, cls_pred = self.G(he)
        except:
            ihc_pred = self.G(he)
            cls_pred = None

        ihc_pred = ihc_pred[0].cpu().numpy()
        ihc_pred = ihc_pred.transpose(1, 2, 0)
        ihc_pred = unnormalize_image(ihc_pred, 'ihc', self.norm_method)
        ihc_pred = ihc_pred.astype(np.uint8)

        iio.imwrite(ihc_pred_path, ihc_pred)

        # plt.figure(figsize=(18, 9))
        # plt.subplot(121)
        # plt.imshow(he_ori)
        # plt.subplot(122)
        # plt.imshow(ihc_pred)
        # plt.tight_layout()
        # plt.show()

        return
    
    @torch.no_grad()
    def predict_tta(self, he_path, ihc_pred_path):

        he_ori = iio.imread(he_path)

        ihc_pred_tta = np.zeros_like(he_ori).astype(np.float32)
        for i in range(7):
            he_tta = self.tta(he_ori, i)
            he = normalize_image(he_tta, 'he', self.norm_method)
            he = he.transpose(2, 0, 1).astype(np.float32)[None, ...]
            he = torch.Tensor(he).to(self.device)

            try:
                ihc_pred, cls_pred = self.G(he)
            except:
                ihc_pred = self.G(he)
                cls_pred = None

            ihc_pred = ihc_pred[0].cpu().numpy()
            ihc_pred = ihc_pred.transpose(1, 2, 0)
            ihc_pred = unnormalize_image(ihc_pred, 'ihc', self.norm_method)
            ihe_pred_untta = self.untta(ihc_pred, i)
            ihc_pred_tta += ihe_pred_untta
        
        ihc_pred_tta /= 7
        ihc_pred_tta = ihc_pred_tta.astype(np.uint8)
        iio.imwrite(ihc_pred_path, ihc_pred_tta)

        # plt.figure(figsize=(18, 9))
        # plt.subplot(121)
        # plt.imshow(he_ori)
        # plt.subplot(122)
        # plt.imshow(ihc_pred_tta)
        # plt.tight_layout()
        # plt.show()

        return

    def evaluate(self, ihc_path, ihc_pred_path):

        real = cv2.imread(ihc_path)
        fake = cv2.imread(ihc_pred_path)
        psnr = peak_signal_noise_ratio(fake, real)
        ssim = structural_similarity(fake, real, multichannel=True)

        return psnr, ssim

    @staticmethod
    def tta(image, no):

        if no == 0:
            return image
        elif no == 1:
            return np.rot90(image, 1)
        elif no == 2:
            return np.rot90(image, 2)
        elif no == 3:
            return np.rot90(image, 3)
        elif no == 4:
            return np.fliplr(image)
        elif no == 5:
            return np.flipud(image)
        elif no == 6:
            return image.transpose(1, 0, 2)
        else:
            raise NotImplemented('unknown no fo tta')

    @staticmethod
    def untta(image, no):

        if no == 0:
            return image
        elif no == 1:
            return np.rot90(image, 3)
        elif no == 2:
            return np.rot90(image, 2)
        elif no == 3:
            return np.rot90(image, 1)
        elif no == 4:
            return np.fliplr(image)
        elif no == 5:
            return np.flipud(image)
        elif no == 6:
            return image.transpose(1, 0, 2)
        else:
            raise NotImplemented('unknown no fo tta')
