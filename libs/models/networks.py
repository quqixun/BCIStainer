import math
import torch
import functools
import torch.nn as nn
import segmentation_models_pytorch as smp

from torch.nn import init


# ------------------------------------------------------------------------------
# Helper Functions


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):

    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


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


def define_D(configs):

    if configs.name == 'patch_gan':
        net = NLayerDiscriminator(**configs.params)
    elif configs.name == 'ms_gan':
        net = MultiscaleDiscriminator(**configs.params)
    else:
        raise NotImplementedError(f'unknown D model name {configs.name}')

    init_weights(net, **configs.init)
    return net


# ------------------------------------------------------------------------------
# Generators


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
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class ResnetAdaGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, n_classes=4, n_blocks=6, ngf=32,
                 norm_type='none', dropout=0.0):
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
        enc1_downsampling = 3
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
        enc2_downsampling = 4
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


# ------------------------------------------------------------------------------
# Discriminators


class MultiscaleDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='batch', num_D=3):
        super(MultiscaleDiscriminator, self).__init__()

        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_type)
            setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def singleD_forward(self, model, input):
        return [model(input)]

    def forward(self, input):
        result = []
        input_downsampled = input
        for i in range(self.num_D):
            model = getattr(self, 'layer' + str(self.num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (self.num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='batch'):
        super(NLayerDiscriminator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
