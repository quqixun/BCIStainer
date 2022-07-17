import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):

    def __init__(self, input_size, output_size, input_nc, output_nc, norm=True):
        super(ResnetBlock, self).__init__()

        if input_size == output_size * 2:
            resample = nn.AvgPool2d(2)
        elif input_size * 2 == output_size:
            resample = nn.Upsample(None, 2, 'bilinear')
        elif input_size == output_size:
            resample = nn.Identity()
        else:
            raise ValueError('not implemented')

        if norm:
            self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(input_nc, affine=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1),
                resample,
                nn.InstanceNorm2d(output_nc, affine=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1),
                resample,
                nn.LeakyReLU(0.2),
                nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1)
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=1, padding=0),
            resample
        )

    def forward(self, x):
        return (self.conv1(x) + self.conv2(x)) / math.sqrt(2)


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


class ResnetBlock_AdaIN(nn.Module):
    def __init__(self, input_size, output_size, input_nc, output_nc, latent_size):
        super(ResnetBlock_AdaIN, self).__init__()

        if input_size == output_size * 2:
            resample = nn.AvgPool2d(2)
        elif input_size * 2 == output_size:
            resample = nn.Upsample(None, 2, 'bilinear')
        elif input_size == output_size:
            resample = nn.Identity()
        else:
            raise ValueError('not implemented')

        self.style1 = AdaIN(latent_size, input_nc)
        self.style2 = AdaIN(latent_size, output_nc)

        self.conv1_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1),
            resample
        )

        self.conv1_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1),
        )

        self.conv2 = nn.Sequential(
            resample,
            nn.Conv2d(input_nc, output_nc, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x_in, style = x
        x1 = self.style1(x_in, style)
        x1 = self.conv1_1(x1)
        x1 = self.style2(x1, style)
        x1 = self.conv1_2(x1)
        x2 = self.conv2(x_in)
        return (x1 + x2) / math.sqrt(2), style


class SelfAttention(nn.Module):

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        
        self.qconv = nn.Conv2d(in_dim, in_dim // 8, 1, 1, 0)
        self.kconv = nn.Conv2d(in_dim, in_dim // 8, 1, 1, 0)
        self.vconv = nn.Conv2d(in_dim, in_dim, 1, 1, 0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

        return

    def forward(self, x):

        b, c, h, w = x.size()
        q = self.qconv(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.kconv(x).view(b, -1, h * w)
        v = self.vconv(x).view(b, -1, h * w)

        attention = self.softmax(torch.bmm(q, k)).permute(0, 2, 1)
        out = torch.bmm(v, attention).view(b, c, h, w)
        out = self.gamma * out + x

        return out


class ModulatedConv(nn.Module):

    def __init__(self, in_chan, out_chan, kernel, demod=True,
                 stride=1, dilation=1, eps = 1e-8, **kwargs):
        super(ModulatedConv, self).__init__()

        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps

        return

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, _, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x


class ModulatedBlock(nn.Module):

    def __init__(self, latent_dim, input_channels, filters, upsample=True):
        super(ModulatedBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) \
                        if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.conv1 = ModulatedConv(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.conv2 = ModulatedConv(filters, filters, 3)

        self.activation = nn.LeakyReLU(0.2)
        return

    def forward(self, x):

        x_in, style = x

        if self.upsample is not None:
            x_in = self.upsample(x_in)

        style1 = self.to_style1(style)
        out = self.conv1(x_in, style1)
        out = self.activation(out)

        style2 = self.to_style2(style)
        out = self.conv2(out, style2)
        out = self.activation(out)

        return out, style


class ResnetBlock_AdaIN2(nn.Module):
    def __init__(self, input_size, output_size, input_nc, output_nc, audio_dim, pose_dim):
        super(ResnetBlock_AdaIN2, self).__init__()

        if input_size == output_size * 2:
            resample = nn.AvgPool2d(2)
        elif input_size * 2 == output_size:
            resample = nn.Upsample(None, 2, 'bilinear')
        elif input_size == output_size:
            resample = nn.Identity()
        else:
            raise ValueError('not implemented')

        self.style1 = AdaIN(pose_dim, input_nc)
        self.style2 = AdaIN(audio_dim, output_nc)

        self.conv1_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1),
            resample
        )

        self.conv1_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1),
        )

        self.conv2 = nn.Sequential(
            resample,
            nn.Conv2d(input_nc, output_nc, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x_in, pose, audio = x
        x1 = self.style1(x_in, pose)
        x1 = self.conv1_1(x1)
        x1 = self.style2(x1, audio)
        x1 = self.conv1_2(x1)
        x2 = self.conv2(x_in)
        return (x1 + x2) / math.sqrt(2), pose, audio


class ModulatedBlock2(nn.Module):

    def __init__(self, audio_dim, pose_dim, input_channels, filters, upsample=True):
        super(ModulatedBlock2, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) \
                        if upsample else None

        self.to_style1 = nn.Linear(pose_dim, input_channels)
        self.conv1 = ModulatedConv(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(audio_dim, filters)
        self.conv2 = ModulatedConv(filters, filters, 3)

        self.activation = nn.LeakyReLU(0.2)
        return

    def forward(self, x):

        x_in, pose, audio = x

        if self.upsample is not None:
            x_in = self.upsample(x_in)

        style1 = self.to_style1(pose)
        out = self.conv1(x_in, style1)
        out = self.activation(out)

        style2 = self.to_style2(audio)
        out = self.conv2(out, style2)

        return out, pose, audio


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResnetBlock_AdaIN_CBAM(nn.Module):

    def __init__(self, input_size, output_size, input_nc, output_nc, latent_size):
        super(ResnetBlock_AdaIN_CBAM, self).__init__()

        if input_size == output_size * 2:
            resample = nn.AvgPool2d(2)
        elif input_size * 2 == output_size:
            resample = nn.Upsample(None, 2, 'bilinear')
        elif input_size == output_size:
            resample = nn.Identity()
        else:
            raise ValueError('not implemented')

        self.style1 = AdaIN(latent_size, input_nc)
        self.style2 = AdaIN(latent_size, output_nc)

        self.conv1_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1),
            resample
        )

        self.conv1_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1),
        )

        self.conv2 = nn.Sequential(
            resample,
            nn.Conv2d(input_nc, output_nc, kernel_size=1, padding=0)
        )

        self.ca = ChannelAttention(output_nc)
        self.sa = SpatialAttention()

    def forward(self, x):
        x_in, style = x
        x1 = self.style1(x_in, style)
        x1 = self.conv1_1(x1)
        x1 = self.style2(x1, style)
        x1 = self.conv1_2(x1)
        x1 = self.ca(x1) * x1
        x1 = self.sa(x1) * x1
        x2 = self.conv2(x_in)
        return (x1 + x2) / math.sqrt(2), style


class ResnetBlock_CBAM(nn.Module):

    def __init__(self, input_size, output_size, input_nc, output_nc, norm=True):
        super(ResnetBlock_CBAM, self).__init__()

        if input_size == output_size * 2:
            resample = nn.AvgPool2d(2)
        elif input_size * 2 == output_size:
            resample = nn.Upsample(None, 2, 'bilinear')
        elif input_size == output_size:
            resample = nn.Identity()
        else:
            raise ValueError('not implemented')

        if norm:
            self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(input_nc, affine=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1),
                resample,
                nn.InstanceNorm2d(output_nc, affine=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1),
                resample,
                nn.LeakyReLU(0.2),
                nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1)
            )

        self.ca = ChannelAttention(output_nc)
        self.sa = SpatialAttention()

        self.conv2 = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=1, padding=0),
            resample
        )

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.ca(x1) * x1
        x1 = self.sa(x1) * x1
        x2 = self.conv2(x)

        return (x1 + x2) / math.sqrt(2)
