import numpy as np
import torch
import torch.nn as nn


def int2tensor(input):
    return torch.from_numpy(np.array(input))


class MLP(nn.Module):
    def __init__(self, input_ch, planes, n):
        super(MLP, self).__init__()
        self.planes = planes
        self.num_FiLM = n
        self.model = nn.Sequential(
            nn.Linear(input_ch, int(0.5 * n * self.planes), bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(int(0.5 * n * self.planes), n * self.planes, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(n * self.planes, 2 * n * self.planes, bias=True),
        )

    def forward(self, x):
        out = self.model(x)
        out1 = out[:, :self.num_FiLM * self.planes]
        out2 = out[:, self.num_FiLM * self.planes:]
        return out1, out2


class ResidualBlock(nn.Module):
    def __init__(self, planes):
        super(ResidualBlock, self).__init__()
        self.filter1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.ln1 = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu1 = nn.LeakyReLU(0.2, True)
        self.filter2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.ln2 = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu2 = nn.LeakyReLU(0.2, True)
        self.SA = SA_layer(7)

    def forward(self, input):
        output = self.leakyrelu1(self.ln1(self.filter1(input)))
        output = self.ln2(self.filter2(output))
        output = self.leakyrelu2(output)
        output = self.SA(output) * output
        return output + input


class ResidualBlock_1d(nn.Module):
    def __init__(self, planes, reduction):
        super(ResidualBlock_1d, self).__init__()
        self.filter1 = nn.Conv2d(planes, planes, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.ln1 = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu1 = nn.LeakyReLU(0.2, True)
        self.filter2 = nn.Conv2d(planes, planes, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.ln2 = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu2 = nn.LeakyReLU(0.2, True)
        self.CA = CALayer(planes, reduction)

    def forward(self, input):
        output = self.leakyrelu1(self.ln1(self.filter1(input)))
        output = self.ln2(self.filter2(output))
        output = self.leakyrelu2(output)
        output = self.CA(output)
        return output + input


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SA_layer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA_layer, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(relu_slope, inplace=True),
        )

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=False)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        return self.conv_block(out)


def conv3x3(in_chn, out_chn, bias=True):
    return nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)


def conv_down(in_chn, out_chn, bias=False):
    return nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
