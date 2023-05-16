# ------------------------------------------------------------------------
# PowerBEV
# Copyright (c) 2023 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from FIERY (https://github.com/wayveai/fiery)
# Copyright (c) 2021 Wayve Technologies Limited. All Rights Reserved.
# ------------------------------------------------------------------------

from collections import OrderedDict

from torch import nn


class Residual(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        dilation=1,
        upsample=False,
        downsample=False,
    ):
        super().__init__()
        self._downsample = downsample
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        if upsample:
            assert not downsample, 'downsample and upsample not possible simultaneously.'
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, dilation=1, stride=2, output_padding=padding_size, padding=padding_size)
        elif downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, dilation=dilation, stride=2, padding=padding_size)
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, dilation=dilation, padding=padding_size)
        
        self.layers = nn.Sequential(conv, nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=True))

        if out_channels == in_channels and not downsample and not upsample:
            self.projection = None
        else:
            projection = OrderedDict()
            if upsample:
                projection.update({'upsample_skip_proj': nn.Upsample(scale_factor=2, mode='bilinear')})
            elif downsample:
                projection.update({'upsample_skip_proj': nn.MaxPool2d(kernel_size=2, stride=2)})
            projection.update(
                {
                    'conv_skip_proj': nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    'bn_skip_proj': nn.BatchNorm2d(out_channels),
                }
            )
            self.projection = nn.Sequential(projection)
    
    def forward(self, *args):
        (x,) = args
        x_residual = self.layers(x)
        if self.projection is not None:
            if self._downsample:
                # pad h/w dimensions if they are odd to prevent shape mismatch with residual layer
                x = nn.functional.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0)
            return x_residual + self.projection(x)
        return x_residual + x


class Bottleneck(nn.Module):
    """expand + depthwise + pointwise"""
    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        dilation=1,
        upsample=False,
        downsample=False,
        expand_ratio=2,
        dropout=0.0,
    ):
        super().__init__()
        self._downsample = downsample
        expand_channels = round(in_channels * expand_ratio)
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        if upsample:
            assert not downsample, 'downsample and upsample not possible simultaneously.'
            conv = nn.ConvTranspose2d(expand_channels, expand_channels, kernel_size=kernel_size, bias=False, dilation=1, stride=2, output_padding=padding_size, padding=padding_size, groups=expand_channels)
        elif downsample:
            conv = nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size, bias=False, dilation=dilation, stride=2, padding=padding_size, groups=expand_channels)
        else:
            conv = nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size, bias=False, dilation=dilation, padding=padding_size, groups=expand_channels)
        
        self.layers = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, expand_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.Hardswish(inplace=True),
            # dw
            conv,
            nn.BatchNorm2d(expand_channels),
            nn.Hardswish(inplace=True),
            # pw-linear
            nn.Conv2d(expand_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=dropout),
            # SeModule(out_channels),
        )

        if out_channels == in_channels and not downsample and not upsample:
            self.projection = None
        else:
            projection = OrderedDict()
            if upsample:
                projection.update({'upsample_skip_proj': nn.Upsample(scale_factor=2, mode='bilinear')})
            elif downsample:
                projection.update({'upsample_skip_proj': nn.MaxPool2d(kernel_size=2, stride=2)})
            projection.update(
                {
                    'conv_skip_proj': nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    'bn_skip_proj': nn.BatchNorm2d(out_channels),
                }
            )
            self.projection = nn.Sequential(projection)

    def forward(self, *args):
        (x,) = args
        x_residual = self.layers(x)
        if self.projection is not None:
            if self._downsample:
                # pad h/w dimensions if they are odd to prevent shape mismatch with residual layer
                x = nn.functional.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0)
            return x_residual + self.projection(x)
        return x_residual + x