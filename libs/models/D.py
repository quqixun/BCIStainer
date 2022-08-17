import functools
import torch.nn as nn
import torch.nn.functional as F

from .utils import *


def define_D(configs):

    if configs.name == 'patch_gan':
        net = NLayerDiscriminator(**configs.params)
    elif configs.name == 'ms_gan':
        net = MultiscaleDiscriminator(**configs.params)
    else:
        raise NotImplementedError(f'unknown D model name {configs.name}')

    init_weights(net, **configs.init)
    return net


class MultiscaleDiscriminator(nn.Module):

    def __init__(self, input_channels, init_channels=64,
                 num_layers=3, norm_type='batch', num_depths=3):
        super(MultiscaleDiscriminator, self).__init__()

        self.num_depths = num_depths
        self.num_layers = num_layers

        for i in range(num_depths):
            netD = NLayerDiscriminator(input_channels, init_channels, num_layers, norm_type)
            setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def singleD_forward(self, model, input):
        return [model(input)]

    def forward(self, input):
        result = []
        input_downsampled = input
        for i in range(self.num_depths):
            model = getattr(self, 'layer' + str(self.num_depths - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (self.num_depths - 1):
                # input_downsampled = self.downsample(input_downsampled)
                input_downsampled = F.interpolate(
                    input_downsampled, scale_factor=0.5,
                    mode='bilinear', align_corners=True
                )
        return result


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_channels, init_channels=64, num_layers=3, norm_type='batch'):
        super(NLayerDiscriminator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_channels, init_channels, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(init_channels * nf_mult_prev, init_channels * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(init_channels * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        sequence += [
            nn.Conv2d(init_channels * nf_mult_prev, init_channels * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(init_channels * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(init_channels * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
