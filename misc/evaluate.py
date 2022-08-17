import os
import cv2

from tqdm import tqdm
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def psnr_and_ssim():

    he_dir = './data/test/HE'
    ihc_dir = './data/test/IHC'

    exp = 'resnet_ada_l_v2_exp5'
    # exp = 'resnet_mod_v2_exp3'
    # exp = 'resnet_ada_v2_exp18'
    ihc_pred_dir = f'./outputs_test/{exp}/model_best_psnr_tta/IHC_pred'

    psnr = []
    ssim = []
    for i in tqdm(os.listdir(he_dir)):
        fake = cv2.imread(os.path.join(ihc_dir, i))
        real = cv2.imread(os.path.join(ihc_pred_dir, i))
        PSNR = peak_signal_noise_ratio(fake, real)
        psnr.append(PSNR)
        SSIM = structural_similarity(fake, real, multichannel=True)
        ssim.append(SSIM)

    average_psnr = sum(psnr) / len(psnr)
    average_ssim = sum(ssim) / len(ssim)
    print("The average psnr is " + str(average_psnr))
    print("The average ssim is " + str(average_ssim))

psnr_and_ssim()