import os
import cv2
import torch
import numpy as np
import pandas as pd
import imageio.v2 as iio
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from tqdm import tqdm
from os.path import join as opj
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from ..utils import normalize_image, unnormalize_image


class BCIEvaluator(object):

    def __init__(self, configs, model_path):


        self.model_name  = configs.model.name
        self.model_prms  = configs.model.params
        self.device      = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.norm_method = configs.loader.norm_method

        self._load_model(model_path)

    def _load_model(self, model_path):

        if self.model_name == 'UNet':
            model_func = smp.Unet
        else:
            raise ValueError('Unknown model')

        self.model = model_func(**self.model_prms)
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)

        return
    
    def forward(self, data_dir, output_dir):

        pred_dir = opj(output_dir, 'IHC_pred')
        os.makedirs(pred_dir, exist_ok=True)

        he_dir  = opj(data_dir, 'HE')
        ihc_dir = opj(data_dir, 'IHC')
        files   = os.listdir(he_dir)
        files.sort()

        metrics_list = []
        for file in tqdm(files, ncols=66):
            he_path = opj(he_dir, file)
            ihc_path = opj(ihc_dir, file)
            ihc_pred_path = opj(pred_dir, file)

            self.predict(he_path, ihc_pred_path)
            psnr, ssim = self.evaluate(ihc_path, ihc_pred_path)

            metrics_list.append([he_path, ihc_path, ihc_pred_path, psnr, ssim])

        metrics = pd.DataFrame(metrics_list, columns=['he', 'ihc', 'ihc_pred', 'psnr', 'ssim'])
        metrics.to_csv(opj(output_dir, 'metrics.csv'), index=False)

        psnr_avg = np.mean(metrics['psnr'])
        psnr_std = np.std(metrics['psnr'])
        ssim_avg = np.mean(metrics['ssim'])
        ssim_std = np.std(metrics['ssim'])

        print(output_dir)
        print(f'PSNE: {psnr_avg:.3f} ± {psnr_std:.3f}')
        print(f'SSIM:  {ssim_avg:.3f} ± {ssim_std:.3f}')

        return

    @torch.no_grad()
    def predict(self, he_path, ihc_pred_path):

        he_ori = iio.imread(he_path)
        he = normalize_image(he_ori, 'he', self.norm_method)
        he = he.transpose(2, 0, 1).astype(np.float32)[None, ...]
        he = torch.Tensor(he).to(self.device)

        ihc_pred = self.model(he)
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

    def evaluate(self, ihc_path, ihc_pred_path):

        real = cv2.imread(ihc_path)
        fake = cv2.imread(ihc_pred_path)
        psnr = peak_signal_noise_ratio(fake, real)
        ssim = structural_similarity(fake, real, multichannel=True)

        return psnr, ssim
