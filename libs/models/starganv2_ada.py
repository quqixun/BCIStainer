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
        super().__init__()
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
        if self.downsample:
            self.avgpool = nn.AvgPool2d(2)

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
            x = self.avgpool(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64,
                 actv=nn.LeakyReLU(0.2, True), upsample=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
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
        print(num_layers)
        layers = []
        in_dim = ngf
        for i in range(num_layers):
            out_dim = min(max_conv_dim, in_dim * 2)
            print(in_dim, out_dim)
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
        print(out.size())
        out = self.inconv(out)
        style = self.layers(out)
        level = self.cls_head(style)

        return level, style


class StarAdaGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, n_classes=4, n_enc1=3,
                 n_blocks=2, ngf=32, dropout=0.0, cls_size=256, max_conv_dim=256):
        super(StarAdaGenerator, self).__init__()

        self.classifier = Classifier(input_nc, n_classes, ngf, cls_size, dropout, max_conv_dim)
        style_dim = self.classifier.style_dim

        self.from_rgb = nn.Conv2d(input_nc, ngf, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(ngf, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf, output_nc, 1, 1, 0)
        )

        # down/up-sampling blocks
        repeat_num = n_enc1
        dim_in = ngf
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(n_blocks):
            self.encode.append(ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(0, AdainResBlk(dim_out, dim_out, style_dim))

    def forward(self, x):

        level, style = self.classifier(x)
        print(level.size(), style.size())
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        for block in self.decode:
            x = block(x, style)
        return self.to_rgb(x)


if __name__ == '__main__':

    x = torch.rand(1, 3, 1024, 1024).cuda()
    # style = torch.rand(1, 512).cuda()

    model = StarAdaGenerator()
    model.cuda()
    # print(model)

    output = model(x)
    print(output.size())

    # classifier = Classifier(input_nc=3, n_classes=4, ngf=32, cls_size=256, dropout=0.0)
    # classifier.cuda()
    # level, style = classifier(x)
    # print(level.size(), style.size())
