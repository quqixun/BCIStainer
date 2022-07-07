import torch
import torch.nn as nn

from piqa import SSIM, MS_SSIM, HaarPSI, PSNR


class RecLoss(nn.Module):

    def __init__(self, mode, weight=1.0):
        super(RecLoss, self).__init__()

        self.weight = weight

        if mode == 'mse':
            self.loss = nn.MSELoss()
        elif mode == 'mae':
            self.loss = nn.L1Loss()
        elif mode == 'smae':
            self.loss = nn.SmoothL1Loss()
        else:
            raise NotImplementedError(f'rec mode {mode} not implemented')

    def forward(self, target, prediction):
        return self.loss(target, prediction) * self.weight


class SimLoss(nn.Module):

    def __init__(self, mode, weight=1.0):
        super(SimLoss, self).__init__()

        self.weight = weight

        if mode == 'ssim':
            self.sim = SSIM(window_size=7, sigma=1.5, n_channels=3)
        elif mode == 'ms_ssim':
            self.sim = MS_SSIM(window_size=7, sigma=1.5, n_channels=3)
        elif mode == 'haarpsi':
            self.sim = HaarPSI(chromatic=True, downsample=True)
        else:
            raise NotImplementedError(f'sim mode {mode} not implemented')

    def forward(self, target, prediction):
        # range in [0, 1]
        target_ = (target + 1.0) / 2.0
        prediction_ = (prediction + 1.0) / 2.0
        sim_loss = 1 - self.sim(prediction_, target_)
        return sim_loss * self.weight


class GANLoss(nn.Module):

    def __init__(self, mode, weight=1.0, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.mode = mode
        self.weight = weight

        if mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError(f'gan mode {mode} not implemented')

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):

        if self.mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss * self.weight


class EvalMetrics(nn.Module):

    def __init__(self):
        super(EvalMetrics, self).__init__()

        self.psnr = PSNR(value_range=255)
        self.ssim = SSIM(window_size=7, sigma=1.5, value_range=255)

    def forward(self, target, prediction):
        # range in [0, 255]
        target_ = (target + 1.0) / 2.0 * 255.0
        prediction_ = (prediction + 1.0) / 2.0 * 255.0

        psnr = self.psnr(prediction_, target_)
        ssim = self.ssim(prediction_, target_)

        return psnr, ssim
