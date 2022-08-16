import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class ResnetAdaBlock(nn.Module):

    def __init__(self, style_dim, conv_dim, norm_layer, dropout, use_bias):
        super(ResnetAdaBlock, self).__init__()

        self.style1 = AdaIN(style_dim, conv_dim)
        self.style2 = AdaIN(style_dim, conv_dim)

        conv1 = [
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(conv_dim),
            nn.LeakyReLU(0.2, True)
        ]
        if dropout > 0:
            conv1 += [nn.Dropout(dropout)]
        self.conv1 = nn.Sequential(*conv1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(conv_dim),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        x_in, style = x
        out = self.style1(x_in, style)
        out = self.conv1(out)
        out = self.style2(out, style)
        out = self.conv2(out)
        return (x_in + out) / math.sqrt(2), style


class ResnetAdaBlock2(nn.Module):

    def __init__(self, style_dim, conv_dim, dropout, use_bias):
        super(ResnetAdaBlock2, self).__init__()

        self.style1 = AdaIN(style_dim, conv_dim)
        self.style2 = AdaIN(style_dim, conv_dim)

        self.act = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(dropout)if dropout > 0 else None

        self.conv1 = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1, bias=use_bias)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1, bias=use_bias)

    def forward(self, x):
        x_in, style = x
        out = self.conv1(x_in)
        out = self.style1(out, style)
        out = self.act(out)
        out = self.conv2(x_in)
        out = self.style2(out, style)
        out = self.act(out)
        return (x_in + out) / math.sqrt(2), style


class DownBlock(nn.Module):

    def __init__(self, in_dim, out_dim, norm_layer, dropout, use_bias):
        super(DownBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(out_dim),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_dim),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.conv(x)


class DownResBlock(nn.Module):

    def __init__(self, in_dim, out_dim, norm_layer, dropout, use_bias):
        super(DownResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_dim),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_dim),
            nn.LeakyReLU(0.2, True)
        )

        self.conv2 = None
        if in_dim != out_dim:
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=use_bias),
                norm_layer(out_dim),
                nn.LeakyReLU(0.2, True)
            )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x1 = self.conv1(x)
        x2 = self.conv2(x) if self.conv2 is not None else x
        return (x1 + x2) / math.sqrt(2)


class UpBlock(nn.Module):

    def __init__(self, in_dim, out_dim, norm_layer, dropout, use_bias):
        super(UpBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=use_bias),
            norm_layer(out_dim),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_dim),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.conv(x)


class UpResBlock(nn.Module):

    def __init__(self, in_dim, out_dim, norm_layer, dropout, use_bias):
        super(UpResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_dim),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_dim),
            nn.LeakyReLU(0.2, True)
        )

        self.conv2 = None
        if in_dim != out_dim:
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=use_bias),
                norm_layer(out_dim),
                nn.LeakyReLU(0.2, True)
            )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.conv1(x)
        x2 = self.conv2(x) if self.conv2 is not None else x
        return (x1 + x2) / math.sqrt(2)


class UpSkipBlock(nn.Module):

    def __init__(self, in_dim, out_dim, norm_layer, dropout, use_bias):
        super(UpSkipBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_dim),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_dim),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class UpAdaBlock(nn.Module):

    def __init__(self, style_dim, in_dim, out_dim, norm_layer, dropout, use_bias):
        super(UpAdaBlock, self).__init__()

        self.style1 = AdaIN(style_dim, in_dim)
        self.style2 = AdaIN(style_dim, out_dim)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=use_bias),
            norm_layer(out_dim),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_dim),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x, style):
        out = self.style1(x, style)
        out = self.conv1(out)
        out = self.style2(out, style)
        out = self.conv2(out)
        return out


class UpSkipAdaBlock(nn.Module):

    def __init__(self, style_dim, in_dim, out_dim, norm_layer, dropout, use_bias):
        super(UpSkipAdaBlock, self).__init__()

        self.style1 = AdaIN(style_dim, in_dim)
        self.style2 = AdaIN(style_dim, out_dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_dim),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_dim),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x1, x2, style):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2], dim=1)

        out = self.style1(x, style)
        out = self.conv1(out)
        out = self.style2(out, style)
        out = self.conv2(out)
        return out


class EqualizedWeight(nn.Module):

    def __init__(self, shape):
        super(EqualizedWeight, self).__init__()

        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.data = nn.Parameter(torch.randn(shape))

    def forward(self):
         return self.data * self.c


class EqualizedLinear(nn.Module):

    def __init__(self, in_dim, out_dim, bias=0.0):
        super(EqualizedLinear, self).__init__()

        self.weight = EqualizedWeight([out_dim, in_dim])
        self.bias = nn.Parameter(torch.ones(out_dim) * bias)

    def forward(self, x):
        return F.linear(x, self.weight(), bias=self.bias)


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

    def forward(self, x_in):
        x, style = x_in
        b, _, h, w = x.shape

        style_ = style[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
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


class ResnetModBlock(nn.Module):

    def __init__(self, style_dim, conv_dim, norm_layer, dropout, use_bias, style_layer=False):
        super(ResnetModBlock, self).__init__()
        assert style_dim == conv_dim

        self.style_layer = style_layer
        if self.style_layer:
            self.style1 = nn.Linear(style_dim, conv_dim, bias=True)
            self.style2 = nn.Linear(style_dim, conv_dim, bias=True)

        conv1 = [
            ModConv2D(conv_dim, conv_dim, kernel_size=3, demodulate=True, use_bias=use_bias),
            norm_layer(conv_dim),
            nn.LeakyReLU(0.2, True)
        ]
        if dropout > 0:
            conv1 += [nn.Dropout(dropout)]
        self.conv1 = nn.Sequential(*conv1)

        self.conv2 = nn.Sequential(
            ModConv2D(conv_dim, conv_dim, kernel_size=3, demodulate=True, use_bias=use_bias),
            norm_layer(conv_dim),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        x_in, style = x
        if self.style_layer:
            out = self.conv1((x_in, self.style1(style)))
            out = self.conv2((out, self.style2(style)))
        else:
            out = self.conv1((x_in, style))
            out = self.conv2((out, style))
        return (x_in + out) / math.sqrt(2), style


class ResnetModBlock2(nn.Module):

    def __init__(self, style_dim, conv_dim, dropout, use_bias, style_layer=False):
        super(ResnetModBlock2, self).__init__()

        self.style_layer = style_layer
        if self.style_layer:
            self.style1 = nn.Linear(style_dim, conv_dim, bias=True)
            self.style2 = nn.Linear(style_dim, conv_dim, bias=True)
        else:
            assert style_dim == conv_dim

        conv1 = [
            ModConv2D(conv_dim, conv_dim, kernel_size=3, demodulate=True, use_bias=use_bias),
            nn.LeakyReLU(0.2, True)
        ]
        if dropout > 0:
            conv1 += [nn.Dropout(dropout)]
        self.conv1 = nn.Sequential(*conv1)

        self.conv2 = nn.Sequential(
            ModConv2D(conv_dim, conv_dim, kernel_size=3, demodulate=True, use_bias=use_bias),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        x_in, style = x
        if self.style_layer:
            out = self.conv1((x_in, self.style1(style)))
            out = self.conv2((out, self.style2(style)))
        else:
            out = self.conv1((x_in, style))
            out = self.conv2((out, style))
        return (x_in + out) / math.sqrt(2), style


if __name__ == '__main__':

    x = torch.rand(2, 32, 128, 128)
    style = torch.rand(2, 32)

    m = ResnetModBlock(32, 32, nn.BatchNorm2d, 0.0, False)
    out, _ = m((x, style))
    print(out.size())
