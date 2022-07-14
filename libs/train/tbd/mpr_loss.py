import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class EdgeLoss(nn.Module):

    def __init__(self):
        super(EdgeLoss, self).__init__()

        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)      # filter
        down = filtered[:,:, ::2, ::2]           # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4    # upsample
        filtered = self.conv_gauss(new_filter)   # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class MPRLoss1(nn.Module):

    def __init__(self, eps=1e-3):
        super(MPRLoss1, self).__init__()
        self.charb_loss = CharbonnierLoss(eps=eps)
        self.edge_loss = EdgeLoss()

    def forward(self, x, y, face, mouth):
        charb_loss = self.charb_loss(x, y)
        edge_loss = self.edge_loss(x, y)
        return charb_loss + edge_loss


class MPRLoss2(nn.Module):

    def __init__(self, eps=1e-3):
        super(MPRLoss2, self).__init__()
        self.charb_loss = CharbonnierLoss(eps=eps)
        self.edge_loss = EdgeLoss()

    def forward(self, x, y, face, mouth):

        charb_loss = self.charb_loss(x, y)
        charb_loss += self.charb_loss(x * face, y * face)
        charb_loss += self.charb_loss(x * mouth, y * mouth)

        edge_loss = self.edge_loss(x, y)
        edge_loss += self.edge_loss(x * face, y * face)
        edge_loss += self.edge_loss(x * mouth, y * mouth)

        return charb_loss + edge_loss


def load_mprloss(mode):

    assert mode in ['mpr1', 'mpr2']

    if mode == 'mpr1':
        return MPRLoss1()
    elif mode == 'mpr2':
        return MPRLoss2()
