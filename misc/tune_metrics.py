import os
import torch
import numpy as np
import imageio.v2 as iio

from tqdm import tqdm
from piqa import SSIM, PSNR
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


if __name__ == '__main__':

    real_dir = './data/test/IHC'
    fake_dir = './outputs_test/resnet_mod_v2_exp3/model_best_psnr_tta/IHC_pred'

    files = os.listdir(real_dir)
    files.sort()

    sk_psnr_list, sk_ssim_list = [], []
    piqa_psnr_list, piqa_ssim_list = [], []

    for file in tqdm(files, ncols=66):
        # images in same shape
        real_image = iio.imread(os.path.join(real_dir, file))
        fake_image = iio.imread(os.path.join(fake_dir, file))

        real_tensor = torch.from_numpy(real_image.transpose(2, 0, 1)[None]).float().contiguous() / 255.0
        fake_tensor = torch.from_numpy(fake_image.transpose(2, 0, 1)[None]).float().contiguous() / 255.0

        sk_psnr = peak_signal_noise_ratio(real_image, fake_image)
        sk_ssim = structural_similarity(
            real_image, fake_image, multichannel=True
        )
        sk_psnr_list.append(sk_psnr)
        sk_ssim_list.append(sk_ssim)

        piqa_psnr = PSNR()(real_tensor, fake_tensor)
        piqa_ssim = SSIM(
            window_size=9, sigma=2.375, n_channels=3
        )(real_tensor, fake_tensor)
        piqa_psnr_list.append(piqa_psnr.item())
        piqa_ssim_list.append(piqa_ssim.item())

    print(f'SKImage PSNR: {np.mean(sk_psnr_list)}')
    print(f'SKImage SSIM: {np.mean(sk_ssim_list)}')
    print(f'PIQA PSNR:    {np.mean(piqa_psnr_list)}')
    print(f'PIQA SSIM:    {np.mean(piqa_ssim_list)}')
