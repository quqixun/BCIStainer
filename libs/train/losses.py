import torch
import torch.nn as nn


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


class GANLoss(nn.Module):

    def __init__(self, mode, weight=1.0, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = mode
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

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss * self.weight
