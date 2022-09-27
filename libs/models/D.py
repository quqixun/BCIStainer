import functools
import torch.nn as nn
import torch.nn.functional as F

from .utils import *
from .layers import *


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
                 num_layers=3, norm_type='batch',
                 num_depths=3, attention=False):
        super(MultiscaleDiscriminator, self).__init__()

        self.num_depths = num_depths
        self.num_layers = num_layers

        for i in range(num_depths):
            netD = NLayerDiscriminator(
                input_channels, init_channels,
                num_layers, norm_type, attention
            )
            setattr(self, 'layer' + str(i), netD.model)

    def singleD_forward(self, model, input):
        return [model(input)]

    def forward(self, input):
        result = []
        input_downsampled = input
        for i in range(self.num_depths):
            model = getattr(self, 'layer' + str(self.num_depths - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (self.num_depths - 1):
                input_downsampled = F.interpolate(
                    input_downsampled, scale_factor=0.5,
                    mode='bilinear', align_corners=True
                )
        return result


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_channels, init_channels=64,
                 num_layers=3, norm_type='batch', attention=False):
        super(NLayerDiscriminator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm_type)
        use_bias = False if norm_type == 'batch' else True
        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            # nn.Conv2d(
            #     input_channels, init_channels,
            #     kernel_size=kw, stride=2, padding=padw
            # ),
            # norm_layer(init_channels),
            # nn.LeakyReLU(0.2, True)
            ConvNormAct(
                in_dims=input_channels, out_dims=init_channels,
                conv_type='conv2d', kernel_size=3, stride=2, padding=1,
                bias=use_bias, norm_layer=norm_layer, sampling='none',
                attention=attention
            )
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            in_dims = init_channels * nf_mult_prev
            out_dims = init_channels * nf_mult
            sequence += [
                # nn.Conv2d(
                #     init_channels * nf_mult_prev, init_channels * nf_mult,
                #     kernel_size=kw, stride=2, padding=padw, bias=use_bias
                # ),
                # norm_layer(init_channels * nf_mult),
                # nn.LeakyReLU(0.2, True)
                ConvNormAct(
                    in_dims=in_dims, out_dims=out_dims,
                    conv_type='conv2d', kernel_size=3, stride=2, padding=1,
                    bias=use_bias, norm_layer=norm_layer, sampling='none',
                    attention=attention
                )
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        in_dims = init_channels * nf_mult_prev
        out_dims = init_channels * nf_mult
        sequence += [
            # nn.Conv2d(
            #     init_channels * nf_mult_prev, init_channels * nf_mult,
            #     kernel_size=kw, stride=1, padding=padw, bias=use_bias
            # ),
            # norm_layer(init_channels * nf_mult),
            # nn.LeakyReLU(0.2, True)
            ConvNormAct(
                in_dims=in_dims, out_dims=out_dims,
                conv_type='conv2d', kernel_size=3, stride=1, padding=1,
                bias=use_bias, norm_layer=norm_layer, sampling='none',
                attention=attention
            )
        ]

        sequence += [
            nn.Conv2d(
                init_channels * nf_mult, 1,
                kernel_size=kw, stride=1, padding=padw
            )
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
