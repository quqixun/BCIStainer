'''
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
'''


import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ResBlk(nn.Module):

    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2, True),
                 normalize=False, downsample=False):
        super(ResBlk, self).__init__()

        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class EqualizedWeight(nn.Module):

    def __init__(self, shape):
        super(EqualizedWeight, self).__init__()

        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.data = nn.Parameter(torch.randn(shape))

    def forward(self):
         return self.data * self.c


class ModConv2D(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size, demodulate=True, use_bias=True, eps=1e-8):
        super(ModConv2D, self).__init__()

        self.out_dim = out_dim
        self.use_bias = use_bias
        self.demodulate = demodulate
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.weight = EqualizedWeight([out_dim, in_dim, kernel_size, kernel_size])
        self.eps = eps

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, style):
        b, _, h, w = x.shape

        style_ = style[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        # print(weights.size(), style_.size())
        weights = weights * (style_ + 1)

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_dim, *ws)

        x = x.reshape(1, -1, h, w)
        x = F.conv2d(x, weights, padding=self.padding, groups=b)
        x = x.reshape(-1, self.out_dim, h, w)

        if self.use_bias:
            x += self.bias[None, :, None, None]

        return x


class ModResBlk(nn.Module):

    def __init__(self, dim_in, dim_out, style_dim=64,
                 actv=nn.LeakyReLU(0.2, True), upsample=False):
        super(ModResBlk, self).__init__()

        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = ModConv2D(dim_in, dim_out, 3)
        self.conv2 = ModConv2D(dim_out, dim_out, 3)
        self.style1 = nn.Linear(style_dim, dim_in)
        self.style2 = nn.Linear(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.actv(x)
        s1 = self.style1(s)
        x = self.conv1(x, s1)
        
        x = self.actv(x)
        s2 = self.style2(s)
        x = self.conv2(x, s2)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = out + self._shortcut(x)
        return out / math.sqrt(2)


class Classifier(nn.Module):

    def __init__(self, input_nc=3, n_classes=4, ngf=32, cls_size=256, dropout=0.0, max_conv_dim=256):
        super(Classifier, self).__init__()

        self.cls_size = cls_size

        self.inconv = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, True)
        )

        num_layers = int(np.log2(cls_size)) - 3
        layers = []
        in_dim = ngf
        for i in range(num_layers):
            out_dim = min(max_conv_dim, in_dim * 2)
            layers += [
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2, True)
            ]
            in_dim = out_dim
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(1)]
        self.layers = nn.Sequential(*layers)
        self.style_dim = out_dim

        cls_head = []
        if dropout > 0:
            cls_head += [nn.Dropout(dropout)]
        cls_head += [nn.Linear(out_dim, n_classes)]
        self.cls_head = nn.Sequential(*cls_head)

    def forward(self, x):

        out = F.interpolate(x, size=self.cls_size, mode='bilinear', align_corners=True)
        out = self.inconv(out)
        style = self.layers(out)
        level = self.cls_head(style)

        return level, style


class StarModGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, n_classes=4, n_enc1=3,
                 n_blocks=2, ngf=32, dropout=0.0, cls_size=256,
                 max_conv_dim=256, lowres=False):
        super(StarModGenerator, self).__init__()

        self.classifier = Classifier(input_nc, n_classes, ngf, cls_size, dropout, max_conv_dim)
        style_dim = self.classifier.style_dim

        self.from_rgb = nn.Conv2d(input_nc, ngf, 3, 1, 1)
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(ngf, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf, output_nc, 1, 1, 0),
            nn.Tanh()
        )

        # down/up-sampling blocks
        repeat_num = n_enc1
        dim_in = ngf
        self.encode1 = nn.ModuleList()
        self.decode2 = nn.ModuleList()
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode1.append(ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode2.insert(0, ModResBlk(dim_out, dim_in, style_dim, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        self.encode2 = nn.ModuleList()
        self.decode1 = nn.ModuleList()
        for _ in range(n_blocks):
            self.encode2.append(ResBlk(dim_out, dim_out, normalize=True))
            self.decode1.insert(0, ModResBlk(dim_out, dim_out, style_dim))

        self.lowres = lowres
        if self.lowres:
            self.lowdec = nn.Sequential(
                nn.InstanceNorm2d(dim_out, affine=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(dim_out, output_nc, 1, 1, 0),
                nn.Tanh()
            )

    def forward(self, x):

        level, style = self.classifier(x)
        out = self.from_rgb(x)

        for block in self.encode1:
            out = block(out)

        for block in self.encode2:
            out = block(out)

        for block in self.decode1:
            out = block(out, style)

        if self.lowres:
            out_low = self.lowdec(out)

        for block in self.decode2:
            out = block(out, style)

        out = self.to_rgb(out)

        if self.lowres:
            return out, out_low, level
        else:
            return out, level


if __name__ == '__main__':

    x = torch.rand(1, 3, 1024, 1024).cuda()
    # style = torch.rand(1, 512).cuda()

    model = StarModGenerator(
        input_nc=3, output_nc=3, n_classes=4, n_enc1=3,
        n_blocks=2, ngf=32, dropout=0.2, cls_size=256,
        max_conv_dim=256, lowres=True
    )
    model.cuda()
    # print(model)

    out, out_low, level = model(x)
    print(out.size(), out_low.size(), level.size())
