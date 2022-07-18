import functools
import torch.nn as nn
import segmentation_models_pytorch as smp

from .utils import *
from .layers import *


def define_G(configs):

    if configs.name == 'resnet_nblocks':
        net = ResnetGenerator(**configs.params)
    elif configs.name == 'resnet_ada_nblocks':
        net = ResnetAdaGenerator(**configs.params)
    elif configs.name == 'unet':
        net = smp.Unet(**configs.params)
    elif configs.name == 'unet++':
        net = smp.UnetPlusPlus(**configs.params)
    else:
        raise NotImplementedError(f'unknown G model name {configs.name}')

    init_weights(net, **configs.init)
    return net


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, n_blocks=6, ngf=64,
                 norm_type='batch', use_dropout=False, padding_type='reflect'):
        super(ResnetGenerator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1,
                                   bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetAdaGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, n_classes=4, n_enc1=3,
                 n_blocks=9, ngf=32, norm_type='none', dropout=0.0):
        super(ResnetAdaGenerator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.inconv = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3, bias=use_bias),
            norm_layer(ngf),
            nn.LeakyReLU(0.2, True)
        )

        encoder1 = []
        enc1_downsampling = n_enc1
        for i in range(enc1_downsampling):
            mult     = 2 ** i
            in_dims  = ngf * mult
            out_dims = ngf * mult * 2
            encoder1 += [
                nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(out_dims),
                nn.LeakyReLU(0.2, True)
            ]
        self.encoder1 = nn.Sequential(*encoder1)

        encoder2 = []
        style_dims = out_dims
        enc2_downsampling = 7 - n_enc1
        for i in range(enc2_downsampling):
            encoder2 += [
                nn.Conv2d(style_dims, style_dims, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(style_dims),
                nn.LeakyReLU(0.2, True)
            ]
        encoder2 += [nn.AdaptiveAvgPool2d(1), nn.Flatten(1)]
        self.encoder2 = nn.Sequential(*encoder2)

        cls_head = []
        if dropout > 0:
            cls_head += [nn.Dropout(dropout)]
        cls_head += [nn.Linear(style_dims, n_classes)]
        self.cls_head = nn.Sequential(*cls_head)

        decoder1 = []
        conv_dims = out_dims
        for i in range(n_blocks):
            decoder1 += [
                ResnetAdaBlock(
                    style_dims, conv_dims, norm_layer=norm_layer,
                    dropout=dropout, use_bias=use_bias
                )
            ]
        self.decoder1 = nn.Sequential(*decoder1)

        decoder2 = []
        for i in range(enc1_downsampling):
            mult = 2 ** (enc1_downsampling - i)
            decoder2 += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1,
                                   bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.LeakyReLU(0.2, True),
            ]
        decoder2 += [
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            nn.Tanh()
        ]
        self.decoder2 = nn.Sequential(*decoder2)

    def forward(self, x):

        x_in = self.inconv(x)
        enc1 = self.encoder1(x_in)
        style = self.encoder2(enc1)
        level = self.cls_head(style)

        dec1, _ = self.decoder1([enc1, style])
        out = self.decoder2(dec1)

        return out, level
