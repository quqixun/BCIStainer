import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SimAM(nn.Module):

    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.e_lambda = e_lambda

    def forward(self, x):

        _, _, h, w = x.size()
        n = w * h - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5

        return x * torch.sigmoid(e_inv)


class ConvNormAct(nn.Module):

    def __init__(self, in_dims, out_dims, conv_type='conv2d',
                 kernel_size=3, stride=1, padding=1, bias=False,
                 norm_layer=nn.BatchNorm2d, sampling='none',
                 attention=False):
        super(ConvNormAct, self).__init__()

        assert sampling in ['down', 'up', 'none']
        self.sampling = sampling

        assert conv_type in ['conv2d', 'convTranspose2d']
        self.conv_type = conv_type

        layers = []
        if self.conv_type == 'conv2d':
            layers += [
                nn.ReflectionPad2d(padding),
                nn.Conv2d(
                    in_dims, out_dims, kernel_size=kernel_size,
                    stride=stride, padding=0, bias=bias
                )
            ]
        else:  # self.conv_type == 'convTranspose2d'
            layers += [
                nn.ConvTranspose2d(
                    in_dims, out_dims, kernel_size=kernel_size,
                    stride=stride, padding=padding,
                    output_padding=padding, bias=bias
                )
            ]

        if attention:
            layers += [SimAM()]
        
        layers += [
            norm_layer(out_dims),
            nn.LeakyReLU(0.2, True)
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        if self.conv_type == 'conv2d':
            if self.sampling == 'down':
                out = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
                out = self.conv(out)
            elif self.sampling == 'up':
                out = self.conv(x)
                out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
            else:
                out = self.conv(x)
        else:  # self.conv_type == 'convTranspose2d'
            out = self.conv(x)

        return out


class ResnetBlock(nn.Module):

    def __init__(self, conv_dims, norm_layer, use_bias, attention=False):
        super(ResnetBlock, self).__init__()

        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(conv_dims, conv_dims, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(conv_dims),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(conv_dims, conv_dims, kernel_size=3, padding=0, bias=use_bias),
        ]

        if attention:
            layers += [SimAM()]
        
        layers += [
            norm_layer(conv_dims),
            nn.LeakyReLU(0.2, True)
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return (x + self.conv(x)) / math.sqrt(2)


class AdaIN(nn.Module):

    def __init__(self, style_dims, num_features):
        super(AdaIN, self).__init__()

        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.linear = nn.Linear(style_dims, num_features * 2)

    def forward(self, x, style):
        h = self.linear(style)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class ResnetAdaBlock(nn.Module):

    def __init__(self, style_dims, conv_dims, use_bias, attention=False):
        super(ResnetAdaBlock, self).__init__()

        self.act    = nn.LeakyReLU(0.2, True)
        self.style1 = AdaIN(style_dims, conv_dims)
        self.style2 = AdaIN(style_dims, conv_dims)
        conv_params = dict(kernel_size=3, padding=1, bias=use_bias)
        self.conv1  = nn.Conv2d(conv_dims, conv_dims, **conv_params)
        self.conv2  = nn.Conv2d(conv_dims, conv_dims, **conv_params)
        self.atten  = SimAM() if attention else None

    def forward(self, x):
        x_in, style = x
        out = self.conv1(x_in)
        out = self.style1(out, style)
        out = self.act(out)
        out = self.conv2(out)
        if self.atten:
            out = self.atten(out)
        out = self.style2(out, style)
        out = self.act(out)
        return (x_in + out) / math.sqrt(2), style


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


class ModConv2d(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size,
                 demodulate=True, use_bias=True, eps=1e-8):
        super(ModConv2d, self).__init__()

        self.out_dim = out_dim
        self.use_bias = use_bias
        self.demodulate = demodulate
        self.kernel_size = kernel_size
        self.padding = (kernel_size // 2,) * 4
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
        x = F.pad(x, self.padding, mode='reflect')
        x = F.conv2d(x, weights, padding=0, groups=b)
        x = x.reshape(-1, self.out_dim, h, w)

        if self.use_bias:
            x += self.bias[None, :, None, None]

        return x


class ResnetModBlock(nn.Module):

    def __init__(self, style_dims, conv_dims, use_bias, style_linear=True, attention=False):
        super(ResnetModBlock, self).__init__()

        self.style_linear = style_linear
        if self.style_linear:
            self.style1 = nn.Linear(style_dims, conv_dims, bias=True)
            self.style2 = nn.Linear(style_dims, conv_dims, bias=True)
        else:
            assert style_dims == conv_dims

        conv1 = []
        conv1.append(
            ModConv2d(
                conv_dims, conv_dims, kernel_size=3,
                demodulate=True, use_bias=use_bias
            )
        )
        conv1.append(nn.LeakyReLU(0.2, True))
        self.conv1 = nn.Sequential(*conv1)

        self.conv2 = nn.Sequential(
            ModConv2d(
                conv_dims, conv_dims, kernel_size=3,
                demodulate=True, use_bias=use_bias
            ),
            nn.LeakyReLU(0.2, True)
        )

        self.atten = SimAM() if attention else None

    def forward(self, x):
        x_in, style = x
        if self.style_linear:
            out = self.conv1((x_in, self.style1(style)))
            out = self.conv2((out, self.style2(style)))
        else:
            out = self.conv1((x_in, style))
            out = self.conv2((out, style))
        
        if self.atten:
            out = self.atten(out)

        return (x_in + out) / math.sqrt(2), style
