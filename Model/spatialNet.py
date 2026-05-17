import torch.nn as nn

from Model.utils import ResidualBlock, UNetConvBlock, UNetUpBlock, conv3x3


class SpatialNet(nn.Module):
    def __init__(self, opt):
        super(SpatialNet, self).__init__()
        self.planes = opt.num_filters * 4
        self.model_start = nn.Sequential(
            nn.Conv2d(1, self.planes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=self.planes, num_groups=1, affine=False),
            nn.LeakyReLU(0.2, True),
        )
        self.layer0 = nn.Sequential(ResidualBlock(self.planes), ResidualBlock(self.planes))
        self.layer1 = nn.Sequential(ResidualBlock(self.planes), ResidualBlock(self.planes))
        self.layer2 = nn.Sequential(ResidualBlock(self.planes), ResidualBlock(self.planes))
        self.layer3 = nn.Sequential(ResidualBlock(self.planes), ResidualBlock(self.planes))
        self.model_end = nn.Sequential(
            nn.Conv2d(self.planes, opt.num_filters, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(opt.num_filters, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, input):
        output = self.model_start(input)
        output = self.layer0(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        return self.model_end(output)


class SpatialNet_1(nn.Module):
    def __init__(self, opt):
        super(SpatialNet_1, self).__init__()
        self.planes = opt.num_filters * 4
        self.model_start = nn.Sequential(
            nn.Conv2d(1, self.planes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=self.planes, num_groups=1, affine=False),
            nn.LeakyReLU(0.2, True),
        )
        self.layer0 = nn.Sequential(ResidualBlock(self.planes), ResidualBlock(self.planes), ResidualBlock(self.planes))
        self.layer1 = nn.Sequential(ResidualBlock(self.planes), ResidualBlock(self.planes), ResidualBlock(self.planes))
        self.layer2 = nn.Sequential(ResidualBlock(self.planes), ResidualBlock(self.planes), ResidualBlock(self.planes))
        self.layer3 = nn.Sequential(ResidualBlock(self.planes), ResidualBlock(self.planes), ResidualBlock(self.planes))
        self.model_end = nn.Sequential(
            nn.Conv2d(self.planes, opt.num_filters, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(opt.num_filters, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, input):
        output = self.model_start(input)
        output = self.layer0(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        return self.model_end(output)


class SpatialNet_2(nn.Module):
    def __init__(self, out_ch=64):
        super(SpatialNet_2, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False)
        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        out = self.tconv1(out)
        out = out + residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out = out + residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        return out + residual_1


class SpatialNet_3(nn.Module):
    def __init__(self, in_chn, wf=32, depth=5, relu_slope=0.2):
        super(SpatialNet_3, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        prev_channels = in_chn
        for i in range(depth):
            downsample = i + 1 < depth
            self.down_path.append(UNetConvBlock(prev_channels, (2 ** i) * wf, downsample, relu_slope))
            prev_channels = (2 ** i) * wf
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2 ** i) * wf, relu_slope))
            prev_channels = (2 ** i) * wf
        self.last = conv3x3(prev_channels, in_chn, bias=False)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            if i + 1 < self.depth:
                x, x_up = down(x)
                blocks.append(x_up)
            else:
                x = down(x)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        return self.last(x)


class SpatialNet_4(SpatialNet_3):
    def __init__(self, in_chn, wf=32, depth=4, relu_slope=0.2):
        super(SpatialNet_4, self).__init__(in_chn, wf, depth, relu_slope)


SpatialNet1 = SpatialNet
