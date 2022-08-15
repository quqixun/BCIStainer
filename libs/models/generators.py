import functools
import torch.nn as nn
import segmentation_models_pytorch as smp

from .utils import *
from .layers import *
from .starganv2_ada import StarAdaGenerator
from .starganv2_mod import StarModGenerator


def define_G(configs):

    if configs.name == 'resnet_nblocks':
        net = ResnetGenerator(**configs.params)
    elif configs.name == 'resnet_ada_nblocks':
        net = ResnetAdaGenerator(**configs.params)
    elif configs.name == 'resnet_mod_nblocks':
        net = ResnetModGenerator(**configs.params)
    elif configs.name == 'resnet_ada_l_nblocks':
        net = ResnetAdaLGenerator(**configs.params)
    elif configs.name == 'resnet_ada_h_nblocks':
        net = ResnetAdaHGenerator(**configs.params)
    elif configs.name == 'star_ada':
        net = StarAdaGenerator(**configs.params)
    elif configs.name == 'star_mod':
        net = StarModGenerator(**configs.params)
    elif configs.name == 'unet':
        net = smp.Unet(**configs.params)
    elif configs.name == 'unet++':
        net = smp.UnetPlusPlus(**configs.params)
    elif configs.name == 'unet_ada':
        net = UnetAdaGenerator(**configs.params)
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
                 n_blocks=9, ngf=32, norm_type='none', dropout=0.0,
                 last_ks=3, lowres=False, ada_block='ada_block'):
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
            if ada_block == 'ada_block':
                layer = ResnetAdaBlock(
                    style_dims, conv_dims, norm_layer=norm_layer,
                    dropout=dropout, use_bias=use_bias
                )
            elif ada_block == 'ada_block2':
                layer = ResnetAdaBlock2(
                    style_dims, conv_dims, dropout=dropout, use_bias=use_bias
                )
            decoder1 += [layer]
        self.decoder1 = nn.Sequential(*decoder1)

        self.lowres = lowres
        if self.lowres:
            self.lowdec = nn.Sequential(
                nn.Conv2d(conv_dims, 3, kernel_size=3, padding=1),
                nn.Tanh()
            )

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
            nn.Conv2d(ngf, output_nc, kernel_size=last_ks, padding=last_ks // 2),
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

        if self.lowres:
            out_low = self.lowdec(dec1)
            return out, out_low, level
        else:
            return out, level


class ResnetAdaLGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, n_classes=4, n_enc1=3,
                 n_blocks=9, ngf=32, norm_type='none', dropout=0.0,
                 last_ks=3, lowres=False):
        super(ResnetAdaLGenerator, self).__init__()

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

        enc1_dims = [ngf]
        self.n_enc1 = n_enc1
        for i in range(self.n_enc1):
            mult     = 2 ** i
            in_dims  = ngf * mult
            out_dims = ngf * mult * 2
            enc1_dims = [out_dims] + enc1_dims
            setattr(
                self, f'encoder1_{i + 1}',
                DownBlock(in_dims, out_dims, norm_layer, dropout, use_bias)
            )

        encoder2 = []
        style_dims = out_dims
        n_enc2 = 7 - n_enc1
        for i in range(n_enc2):
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

        self.lowres = lowres
        if self.lowres:
            self.lowdec = nn.Sequential(
                nn.Conv2d(conv_dims, 3, kernel_size=3, padding=1),
                nn.Tanh()
            )

        for i in range(self.n_enc1):
            in_dims  = enc1_dims[i]
            out_dims = enc1_dims[i + 1]
            setattr(
                self, f'decoder2_{i + 1}',
                UpBlock(in_dims, out_dims, norm_layer, dropout, use_bias)
            )

        self.outconv = nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=last_ks, padding=last_ks // 2),
            nn.Tanh()
        )

    def forward(self, x):

        out = self.inconv(x)

        for i in range(self.n_enc1):
            layer = getattr(self, f'encoder1_{i + 1}')
            out = layer(out)

        style = self.encoder2(out)
        level = self.cls_head(style)

        out, _ = self.decoder1([out, style])
        if self.lowres:
            out_low = self.lowdec(out)

        for i in range(self.n_enc1):
            layer = getattr(self, f'decoder2_{i + 1}')
            out = layer(out)

        out = self.outconv(out)

        if self.lowres:
            return out, out_low, level
        else:
            return out, level


class ResnetAdaHGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, n_classes=4, n_enc1=3,
                 n_blocks=9, ngf=32, norm_type='none', dropout=0.0,
                 last_ks=3, lowres=False):
        super(ResnetAdaHGenerator, self).__init__()

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

        enc1_dims = [ngf]
        self.n_enc1 = n_enc1
        for i in range(self.n_enc1):
            mult     = 2 ** i
            in_dims  = ngf * mult
            out_dims = ngf * mult * 2
            enc1_dims = [out_dims] + enc1_dims
            setattr(
                self, f'encoder1_{i + 1}',
                DownBlock(in_dims, out_dims, norm_layer, dropout, use_bias)
            )

        encoder2 = []
        style_dims = out_dims
        n_enc2 = 7 - n_enc1
        for i in range(n_enc2):
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

        self.lowres = lowres
        if self.lowres:
            self.lowdec = nn.Sequential(
                nn.Conv2d(conv_dims, 3, kernel_size=3, padding=1),
                nn.Tanh()
            )

        for i in range(self.n_enc1):
            in_dims  = enc1_dims[i]
            out_dims = enc1_dims[i + 1]
            setattr(
                self, f'decoder2_{i + 1}',
                UpAdaBlock(style_dims, in_dims, out_dims, norm_layer, dropout, use_bias)
            )

        self.outconv = nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=last_ks, padding=last_ks // 2),
            nn.Tanh()
        )

    def forward(self, x):

        out = self.inconv(x)

        for i in range(self.n_enc1):
            layer = getattr(self, f'encoder1_{i + 1}')
            out = layer(out)

        style = self.encoder2(out)
        level = self.cls_head(style)

        out, _ = self.decoder1([out, style])
        if self.lowres:
            out_low = self.lowdec(out)

        for i in range(self.n_enc1):
            layer = getattr(self, f'decoder2_{i + 1}')
            out = layer(out, style)

        out = self.outconv(out)

        if self.lowres:
            return out, out_low, level
        else:
            return out, level


class UnetAdaGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, n_classes=4, n_enc1=5,
                 ngf=32, norm_type='none', dropout=0.0, last_ks=3):
        super(UnetAdaGenerator, self).__init__()
        n_enc1 = max(min(n_enc1, 5), 3)  # n_enc1: [3, 4, 5]

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

        enc1_dims = [ngf]
        self.n_enc1 = n_enc1
        for i in range(self.n_enc1):
            mult     = 2 ** i
            in_dims  = min(ngf * mult, 512)
            out_dims = min(ngf * mult * 2, 512)
            enc1_dims = [out_dims] + enc1_dims
            setattr(
                self, f'encoder1_{i + 1}',
                DownBlock(in_dims, out_dims, norm_layer, dropout, use_bias)
            )

        encoder2 = []
        style_dims = out_dims
        n_enc2 = 7 - n_enc1
        for i in range(n_enc2):
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

        for i in range(self.n_enc1):
            in_dims  = enc1_dims[i] + enc1_dims[i + 1]
            out_dims = enc1_dims[i + 1]
            setattr(
                self, f'decoder2_{i + 1}',
                UpSkipAdaBlock(style_dims, in_dims, out_dims, norm_layer, dropout, use_bias)
            )

        self.outconv = nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=last_ks, padding=last_ks // 2),
            nn.Tanh()
        )

    def forward(self, x):

        out = self.inconv(x)
        enc_list = [out]

        for i in range(self.n_enc1):
            layer = getattr(self, f'encoder1_{i + 1}')
            out = layer(out)
            enc_list = [out] + enc_list

        style = self.encoder2(out)
        level = self.cls_head(style)

        for i in range(self.n_enc1):
            x1 = out
            x2 = enc_list[i + 1]
            layer = getattr(self, f'decoder2_{i + 1}')
            out = layer(x1, x2, style)

        out = self.outconv(out)
        return out, level


class ResnetModGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, n_classes=4, n_enc1=3,
                 n_blocks=9, ngf=32, norm_type='none', dropout=0.0,
                 last_ks=3, lowres=False, mod_block='mod_block',
                 style_layer=False):
        super(ResnetModGenerator, self).__init__()

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
            if mod_block == 'mod_block':
                layer = ResnetModBlock(
                    style_dims, conv_dims, norm_layer=norm_layer,
                    dropout=dropout, use_bias=use_bias,
                    style_layer=style_layer
                )
            elif mod_block == 'mod_block2':
                layer = ResnetModBlock2(
                    style_dims, conv_dims,
                    dropout=dropout, use_bias=use_bias,
                    style_layer=style_layer
                )

            decoder1 += [layer]
        self.decoder1 = nn.Sequential(*decoder1)

        self.lowres = lowres
        if self.lowres:
            self.lowdec = nn.Sequential(
                nn.Conv2d(conv_dims, output_nc, kernel_size=3, padding=1),
                nn.Tanh()
            )

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
            nn.Conv2d(ngf, output_nc, kernel_size=last_ks, padding=last_ks // 2),
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

        if self.lowres:
            out_low = self.lowdec(dec1)
            return out, out_low, level
        else:
            return out, level
