import numpy as np
import torch.nn as nn

from .utils import *
from .layers import *


def define_C(configs):
    net = Comparator(**configs.params)
    init_weights(net, **configs.init)
    return net


class Comparator(nn.Module):

    def __init__(self,
        full_size=1024,
        input_channels=3,
        init_channels=32,
        max_channels=256,
        levels=4,
        norm_type='batch',
        dropout=0.2
    ):
        super(Comparator, self).__init__()

        assert norm_type in ['batch', 'instance', 'none']
        norm_layer = get_norm_layer(norm_type=norm_type)
        use_bias = False if norm_type == 'batch' else True

        encoder = [
            ConvNormAct(
                in_dims=input_channels, out_dims=init_channels,
                conv_type='conv2d', kernel_size=7, stride=1,
                padding=3, bias=use_bias, norm_layer=norm_layer,
                sampling='none'
            )
        ]

        num_blocks = int(np.log2(full_size / 8))
        for i in range(num_blocks):
            mult     = 2 ** i
            in_dims  = min(init_channels * mult, max_channels)
            out_dims = min(init_channels * mult * 2, max_channels)
            encoder.append(
                ConvNormAct(
                    in_dims=in_dims, out_dims=out_dims,
                    conv_type='conv2d', kernel_size=3, stride=2,
                    padding=1, bias=use_bias, norm_layer=norm_layer,
                    sampling='none'
                )
            )
        encoder.append(nn.AdaptiveAvgPool2d(1))
        encoder.append(nn.Flatten(1))
        self.encoder = nn.Sequential(*encoder)

        classify_head = []
        if dropout > 0:
            classify_head.append(nn.Dropout(dropout))
        classify_head.append(nn.Linear(out_dims, levels))
        self.classify_head = nn.Sequential(*classify_head)
    
    def forward(self, x):
        latent = self.encoder(x)
        levels = self.classify_head(latent)
        return levels, latent
