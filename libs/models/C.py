import numpy as np
import torch.nn as nn

from .utils import *
from .layers import *
from torchvision.models.mobilenetv2 import MobileNetV2
from torchvision.models.shufflenetv2 import ShuffleNetV2


def define_C(configs):

    if configs.name == 'basic':
        net = ComparatorBasic(**configs.params)
    elif configs.name == 'mobilenetv2':
        net = ComparatorMobileNetV2(**configs.params)
    elif configs.name == 'shufflenetv2':
        net = ComparatorShuffleNetV2(**configs.params)
    else:
        raise NotImplementedError(f'unknown C model name {configs.name}')

    init_weights(net, **configs.init)
    return net


class ComparatorBasic(nn.Module):

    def __init__(self,
        full_size=1024,
        input_channels=3,
        init_channels=32,
        max_channels=256,
        levels=4,
        norm_type='batch',
        dropout=0.2,
        attention=False
    ):
        super(ComparatorBasic, self).__init__()

        assert norm_type in ['batch', 'instance', 'none']
        norm_layer = get_norm_layer(norm_type=norm_type)
        use_bias = False if norm_type == 'batch' else True

        encoder = [
            ConvNormAct(
                in_dims=input_channels, out_dims=init_channels,
                conv_type='conv2d', kernel_size=7, stride=2,
                padding=3, bias=use_bias, norm_layer=norm_layer,
                sampling='none', attention=False
            )
        ]

        num_blocks = int(np.log2(full_size / 2 / 8))
        for i in range(num_blocks):
            mult     = 2 ** i
            in_dims  = min(init_channels * mult, max_channels)
            out_dims = min(init_channels * mult * 2, max_channels)
            encoder.append(
                ConvNormAct(
                    in_dims=in_dims, out_dims=out_dims,
                    conv_type='conv2d', kernel_size=3, stride=2,
                    padding=1, bias=use_bias, norm_layer=norm_layer,
                    sampling='none', attention=attention
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


class ComparatorMobileNetV2(nn.Module):

    def __init__(self,
        levels=4,
        width_mult=1.0,
        norm_type='batch',
        dropout=0.2
    ):
        super(ComparatorMobileNetV2, self).__init__()

        assert norm_type in ['batch', 'instance', 'none']
        norm_layer = get_norm_layer(norm_type=norm_type)

        model = MobileNetV2(
            num_classes=levels,
            width_mult=width_mult,
            norm_layer=norm_layer,
            dropout=dropout
        )

        self.encoder = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )

        self.classify_head = model.classifier

    def forward(self, x):
        latent = self.encoder(x)
        levels = self.classify_head(latent)
        return levels, latent


class ComparatorShuffleNetV2(nn.Module):

    def __init__(self, levels=4, width_mult=1.0):
        super(ComparatorShuffleNetV2, self).__init__()
        assert width_mult in [0.5, 1.0, 1.5, 2.0]

        if width_mult == 0.5:
            stages_out_channels = [24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            stages_out_channels = [24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            stages_out_channels = [24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            stages_out_channels = [24, 244, 488, 976, 2048]

        model = ShuffleNetV2(
            stages_repeats=[4, 8, 4],
            stages_out_channels=stages_out_channels,
            num_classes=levels
        )

        self.encoder = nn.Sequential(
            model.conv1,
            model.maxpool,
            model.stage2,
            model.stage3,
            model.stage4,
            model.conv5
        )

        self.classify_head = model.fc

    def forward(self, x):
        latent = self.encoder(x)
        latent = latent.mean([2, 3])  # globalpool
        levels = self.classify_head(latent)
        return levels, latent
