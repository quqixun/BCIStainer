import torch.nn as nn

from .utils import *
from .layers import *


def define_G(configs):

    if configs.name == 'style_translator':
        net = StyleTranslator(**configs.params)
    else:
        raise NotImplementedError(f'unknown G model name {configs.name}')

    init_weights(net, **configs.init)
    return net


class StyleTranslator(nn.Module):

    def __init__(self,
        image_size=1024,
        input_channels=3,
        output_channels=3,
        init_channels=32,
        levels=4,
        encoder1_blocks=3,
        style_type='ada',
        style_linear=True,
        style_blocks=9,
        norm_type='batch',
        dropout=0.0,
        output_lowres=False
    ):
        super(StyleTranslator, self).__init__()

        assert norm_type in ['batch', 'instance', 'none']
        norm_layer = get_norm_layer(norm_type=norm_type)
        use_bias = False if norm_type == 'batch' else True

        self.inconv = ConvNormAct(
            in_dims=input_channels, out_dims=init_channels,
            conv_type='conv2d', kernel_size=7, stride=1,
            padding=3, bias=use_bias, norm_layer=norm_layer,
            sampling='none'
        )

        encoder1 = []
        for i in range(encoder1_blocks):
            mult     = 2 ** i
            in_dims  = init_channels * mult
            out_dims = init_channels * mult * 2
            encoder1.append(
                ConvNormAct(
                    in_dims=in_dims, out_dims=out_dims,
                    conv_type='conv2d', kernel_size=3, stride=2,
                    padding=1, bias=use_bias, norm_layer=norm_layer,
                    sampling='none'
                )
            )
        self.encoder1 = nn.Sequential(*encoder1)

        style_dims = out_dims
        total_encoder_blocks = int(np.log2(image_size / 8))
        num_encoder2_blocks = total_encoder_blocks - encoder1_blocks

        encoder2 = []
        for i in range(num_encoder2_blocks):
            encoder2.append(
                ConvNormAct(
                    in_dims=style_dims, out_dims=style_dims,
                    conv_type='conv2d', kernel_size=3, stride=2,
                    padding=1, bias=use_bias, norm_layer=norm_layer,
                    sampling='none'
                )
            )
        encoder2.append(nn.AdaptiveAvgPool2d(1))
        encoder2.append(nn.Flatten(1))
        self.encoder2 = nn.Sequential(*encoder2)

        classify_head = []
        if dropout > 0:
            classify_head.append(nn.Dropout(dropout))
        classify_head.append(nn.Linear(style_dims, levels))
        self.classify_head = nn.Sequential(*classify_head)

        assert style_type in ['ada', 'mod']
        conv_dims = out_dims
        decoder1 = []
        for i in range(style_blocks):
            if style_type == 'ada':
                layer = ResnetAdaBlock(
                    style_dims, conv_dims,
                    dropout=dropout,use_bias=use_bias
                )
            elif style_type == 'mod':
                layer = ResnetModBlock(
                    style_dims, conv_dims, dropout=dropout,
                    use_bias=use_bias, style_linear=style_linear
                )
            decoder1.append(layer)
        self.decoder1 = nn.Sequential(*decoder1)

        self.output_lowres = output_lowres
        if self.output_lowres:
            self.lowres_outconv = nn.Sequential(
                nn.Conv2d(
                    conv_dims, output_channels,
                    kernel_size=3, padding=1
                ),
                nn.Tanh()
            )

        decoder2 = []
        for i in range(encoder1_blocks):
            mult     = 2 ** (encoder1_blocks - i)
            in_dims  = init_channels * mult
            out_dims = int(init_channels * mult / 2)
            decoder2.append(
                ConvNormAct(
                    in_dims=in_dims, out_dims=out_dims,
                    conv_type='convTranspose2d',
                    kernel_size=3, stride=2, padding=1,
                    bias=use_bias, norm_layer=norm_layer,
                    sampling='none'
                )
            )
        self.decoder2 = nn.Sequential(*decoder2)

        self.highres_outconv = nn.Sequential(
            nn.Conv2d(
                init_channels, output_channels,
                kernel_size=7, padding=3
            ),
            nn.Tanh()
        )

    def forward(self, x):

        x_in = self.inconv(x)
        enc1 = self.encoder1(x_in)
        style = self.encoder2(enc1)
        level = self.classify_head(style)

        dec1, _ = self.decoder1([enc1, style])
        dec2 = self.decoder2(dec1)
        hr_out = self.highres_outconv(dec2)

        if self.output_lowres:
            lr_out = self.lowres_outconv(dec1)
            return hr_out, lr_out, level
        else:
            return hr_out, level
