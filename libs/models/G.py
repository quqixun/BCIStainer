import torch.nn as nn

from .utils import *
from .layers import *
from copy import deepcopy


def define_G(configs):

    if configs.name == 'style_translator':
        net = StyleTranslator(**configs.params)
    elif configs.name == 'style_translator_cahr':
        net = StyleTranslatorCAHR(**configs.params)
    else:
        raise NotImplementedError(f'unknown G model name {configs.name}')

    init_weights(net, **configs.init)
    return net


class StyleTranslator(nn.Module):

    def __init__(self,
        full_size=1024,
        input_channels=3,
        output_channels=3,
        init_channels=32,
        levels=4,
        encoder1_blocks=3,
        style_type='mod',
        style_linear=True,
        style_blocks=9,
        norm_type='batch',
        dropout=0.2,
        output_lowres=True
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

        style_dims = conv_dims = out_dims
        total_encoder_blocks = int(np.log2(full_size / 8))
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

        assert style_type in ['ada', 'mod', 'none']
        self.style_type = style_type

        decoder1 = []
        for i in range(style_blocks):
            if self.style_type == 'ada':
                layer = ResnetAdaBlock(
                    style_dims, conv_dims, use_bias=use_bias
                )
            elif self.style_type == 'mod':
                layer = ResnetModBlock(
                    style_dims, conv_dims, use_bias=use_bias,
                    style_linear=style_linear
                )
            else:  # self.style_type == 'none'
                layer = ResnetBlock(
                    conv_dims, norm_layer=norm_layer, use_bias=use_bias
                )
            decoder1.append(layer)
        self.decoder1 = nn.Sequential(*decoder1)

        self.output_lowres = output_lowres
        if self.output_lowres:
            self.lowres_outconv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(
                    conv_dims, output_channels,
                    kernel_size=3, padding=0
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
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                init_channels, output_channels,
                kernel_size=7, padding=0
            ),
            nn.Tanh()
        )

    def forward(self, he):

        he_in = self.inconv(he)
        enc1 = self.encoder1(he_in)
        style = self.encoder2(enc1)
        level = self.classify_head(style)

        if self.style_type == 'none':
            dec1 = self.decoder1(enc1)
        else:
            dec1, _ = self.decoder1([enc1, style])

        dec2 = self.decoder2(dec1)
        ihc_hr = self.highres_outconv(dec2)

        if self.output_lowres:
            ihc_lr = self.lowres_outconv(dec1)
            return ihc_hr, ihc_lr, level
        else:
            return ihc_hr, level


class StyleTranslatorCAHR(nn.Module):

    def __init__(self,
        full_size=1024,
        crop_size=512,
        input_channels=3,
        output_channels=3,
        init_channels=64,
        levels=4,
        encoder1_blocks=2,
        style_type='mod',
        style_linear=True,
        style_blocks=9,
        norm_type='batch',
        dropout=0.2,
        output_lowres=True,
        mask_dec_input='dec1'
    ):
        super(StyleTranslatorCAHR, self).__init__()

        self.full_size = full_size
        self.crop_size = crop_size

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

        style_dims = conv_dims = out_dims
        total_encoder_blocks = int(np.log2(full_size / 8))
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

        assert style_type in ['ada', 'mod', 'none']
        self.style_type = style_type

        decoder1 = []
        for i in range(style_blocks):
            if self.style_type == 'ada':
                layer = ResnetAdaBlock(
                    style_dims, conv_dims, use_bias=use_bias
                )
            elif self.style_type == 'mod':
                layer = ResnetModBlock(
                    style_dims, conv_dims,  use_bias=use_bias,
                    style_linear=style_linear
                )
            else:  # self.style_type == 'none'
                layer = ResnetBlock(
                    conv_dims, norm_layer=norm_layer, use_bias=use_bias
                )
            decoder1.append(layer)
        self.decoder1 = nn.Sequential(*decoder1)

        self.output_lowres = output_lowres
        if self.output_lowres:
            self.lowres_outconv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(
                    conv_dims, output_channels,
                    kernel_size=3, padding=0
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
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                init_channels, output_channels,
                kernel_size=7, padding=0
            ),
            nn.Tanh()
        )

        assert mask_dec_input in ['dec1', 'enc1'], \
            f'mask_dec_input {mask_dec_input} is invalid'
        self.mask_dec_input = mask_dec_input
        self.mask_decoder = deepcopy(self.decoder2)
        self.mask_outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                init_channels, 1,
                kernel_size=7, padding=0
            ),
            nn.Sigmoid()
        )

    def _forward_full(self, he):

        outputs_dict = {}

        he_in = self.inconv(he)
        enc1 = self.encoder1(he_in)
        style = self.encoder2(enc1)
        level = self.classify_head(style)

        if self.style_type == 'none':
            dec1 = self.decoder1(enc1)
        else:
            outputs_dict['style'] = style
            dec1, _ = self.decoder1([enc1, style])

        dec2 = self.decoder2(dec1)
        ihc_full = self.highres_outconv(dec2)

        if self.output_lowres:
            ihc_lr = self.lowres_outconv(dec1)
            outputs_dict['ihc_lr'] = ihc_lr
        
        outputs_dict['level']    = level
        outputs_dict['ihc_full'] = ihc_full
        return outputs_dict

    def _forward_crop(self, he_crop, style):

        he_in = self.inconv(he_crop)
        enc1 = self.encoder1(he_in)

        if self.style_type == 'none':
            dec1 = self.decoder1(enc1)
        else:
            if style.size(0) != he_crop.size(0):
                style = style.repeat(he_crop.size(0), 1)
            dec1, _ = self.decoder1([enc1, style])
        
        dec2 = self.decoder2(dec1)
        ihc_crop = self.highres_outconv(dec2)

        if self.mask_dec_input == 'dec1':
            mask_dec = self.mask_decoder(dec1)
        elif self.mask_dec_input == 'enc1':
            mask_dec = self.mask_decoder(enc1)
        mask_crop = self.mask_outconv(mask_dec)

        outputs_dict = {
            'ihc_crop': ihc_crop,
            'mask_crop': mask_crop
        }
        return outputs_dict

    def _train_merge(self, ihc_full, ihc_crop, mask_crop, crop_idxs):

        ihc_hr_list = []
        for i in range(ihc_full.size(0)):
            row1, col1 = crop_idxs[i]
            row2, col2 = crop_idxs[i] + self.crop_size
            row_pad = [row1, self.full_size - row2]
            col_pad = [col1, self.full_size - col2]

            ihc_crop_pad  = F.pad(ihc_crop[i], col_pad + row_pad)
            mask_crop_pad = F.pad(mask_crop[i], col_pad + row_pad)

            ihc_full_ = ihc_full[i] * (1 - mask_crop_pad)
            ihc_crop_ = ihc_crop_pad * mask_crop_pad
            ihc_hr_   = ihc_full_ + ihc_crop_
            ihc_hr_list.append(ihc_hr_)

        ihc_hr = torch.stack(ihc_hr_list)
        return ihc_hr

    def _infer_full_merge(self, ihc_full, ihc_crop, mask_crop, crop_idxs):
        assert ihc_full.size(0) == 1

        ihc_full = ihc_full.squeeze(0)
        ihc_hr   = torch.zeros_like(ihc_full)

        for i in range(ihc_crop.size(0)):
            row1, col1 = crop_idxs[i]
            row2, col2 = crop_idxs[i] + self.crop_size
            row_pad = [row1, self.full_size - row2]
            col_pad = [col1, self.full_size - col2]

            ihc_crop_pad  = F.pad(ihc_crop[i], col_pad + row_pad)
            mask_crop_pad = F.pad(mask_crop[i], col_pad + row_pad)

            ihc_full_ = ihc_full * (1 - mask_crop_pad)
            ihc_crop_ = ihc_crop_pad * mask_crop_pad
            ihc_hr_   = ihc_full_ + ihc_crop_
            ihc_hr   += ihc_hr_

        ihc_hr /= ihc_crop.size(0)
        return ihc_hr.unsqueeze(0)

    def _infer_crop_merge(self, ihc_full, ihc_crop, mask_crop, crop_idxs):
        assert ihc_full.size(0) == 1

        ihc_full  = ihc_full.squeeze(0)
        ihc_hr    = torch.zeros_like(ihc_full)
        ihc_count = torch.zeros_like(ihc_full)

        for i in range(ihc_crop.size(0)):
            row1, col1 = crop_idxs[i]
            row2, col2 = crop_idxs[i] + self.crop_size

            ihc_full_crop_ = ihc_full[:, row1:row2, col1:col2]
            ihc_full_crop_ = ihc_full_crop_ * (1 - mask_crop[i])
            ihc_crop_ = ihc_crop[i] * mask_crop[i]
            ihc_hr_   = ihc_full_crop_ + ihc_crop_

            ihc_hr[:, row1:row2, col1:col2]    += ihc_hr_
            ihc_count[:, row1:row2, col1:col2] += 1.0

        ihc_hr /= ihc_count
        return ihc_hr.unsqueeze(0)

    def forward(self, he, he_crop, crop_idxs, mode):
        assert mode in ['train', 'infer_full', 'infer_crop'], \
            f'mode {mode} is invalid'

        full_outputs = self._forward_full(he)
        level    = full_outputs['level']
        ihc_full = full_outputs['ihc_full']
        style    = full_outputs.get('style', None)
        ihc_lr   = full_outputs.get('ihc_lr', None)

        crop_outputs = self._forward_crop(he_crop, style)
        ihc_crop  = crop_outputs['ihc_crop']
        mask_crop = crop_outputs['mask_crop']

        if mode == 'train':
            merge_func = self._train_merge
        elif mode == 'infer_full':
            merge_func = self._infer_full_merge
        elif mode == 'infer_crop':
            merge_func = self._infer_crop_merge

        ihc_hr = merge_func(ihc_full, ihc_crop, mask_crop, crop_idxs)

        if self.output_lowres:
            return ihc_hr, ihc_lr, ihc_crop, level
        else:
            return ihc_hr, ihc_crop, level
