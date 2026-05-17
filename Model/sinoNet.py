import torch.nn as nn
import torch.nn.functional as F
from Model.utils import ResidualBlock_1d


class SinoNet(nn.Module):
    def __init__(self, opt):
        super(SinoNet, self).__init__()
        self.planes = 4 * opt.num_filters
        self.model_start = nn.Sequential(
            nn.Conv2d(1, opt.num_filters, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.GroupNorm(num_channels=opt.num_filters, num_groups=1, affine=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(opt.num_filters, self.planes, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.GroupNorm(num_channels=self.planes, num_groups=1, affine=False),
            nn.LeakyReLU(0.2, True),
        )
        self.filter1 = self._make_filter(opt)
        self.filter2 = self._make_filter(opt)
        self.filter3 = self._make_filter(opt)
        self.model_final = nn.Sequential(
            nn.Conv2d(self.planes, opt.num_filters, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.Conv2d(opt.num_filters, 1, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
        )

    def _make_filter(self, opt):
        return nn.Sequential(
            ResidualBlock_1d(self.planes, opt.reduction),
            ResidualBlock_1d(self.planes, opt.reduction),
            ResidualBlock_1d(self.planes, opt.reduction),
        )

    def forward(self, input):
        output = self.model_start(input)
        output = self.filter1(output)
        output = self.filter2(output)
        output = self.filter3(output)
        return self.model_final(output)


class SinoNet_1(SinoNet):
    def __init__(self, opt):
        super(SinoNet_1, self).__init__(opt)
        self.filter4 = self._make_filter(opt)

    def forward(self, input):
        output = self.model_start(input)
        output = self.filter1(output)
        output = self.filter2(output)
        output = self.filter3(output)
        output = self.filter4(output)
        return self.model_final(output)


class SinoNet_2(SinoNet):
    def forward(self, input):
        output = self.model_start(input)
        output = self.filter1(output)
        output = self.filter2(output)
        return self.model_final(output)


class iRadonMap_subNet1(nn.Module):
    def __init__(self, geo, opt):
        super(iRadonMap_subNet1, self).__init__()
        self.geo = geo

        self.RampFilter1 = nn.Linear(geo['nDetecU'], geo['nDetecU'])
        self.RampFilter2 = nn.Linear(geo['nDetecU'], geo['nDetecU'])
        self.RampFilter3 = nn.Linear(geo['nDetecU'], geo['nDetecU'])

        nn.init.xavier_uniform_(self.RampFilter1.weight)
        nn.init.xavier_uniform_(self.RampFilter2.weight)
        nn.init.xavier_uniform_(self.RampFilter3.weight)

    def forward(self, input):
        input = input.view(-1, self.geo['nDetecU'])
        output = F.tanh(self.RampFilter1(input))
        output = F.tanh(self.RampFilter2(input))
        output = F.tanh(self.RampFilter3(input))
        return output


class FCN(nn.Module):
    def __init__(self, opt):
        super(FCN, self).__init__()
        conv_start  = [nn.Conv2d(1, opt.num_filters, kernel_size=(1,3), stride=1, padding=(0,1), bias=True),nn.GroupNorm(num_channels=opt.num_filters, num_groups=1, affine=False),nn.LeakyReLU(0.2, True)]
        conv_start += [nn.Conv2d(opt.num_filters, 4*opt.num_filters, kernel_size=(1,3), stride=1, padding=(0,1), bias=True),nn.GroupNorm(num_channels=4*opt.num_filters, num_groups=1, affine=False),nn.LeakyReLU(0.2, True)]
        self.model_start = nn.Sequential(*conv_start)
        
        self.filter1 = nn.Sequential(*[nn.Conv2d(4*opt.num_filters, 4*opt.num_filters, kernel_size=(1,3), stride=1, padding=(0,1), bias=True),nn.GroupNorm(num_channels=opt.num_filters, num_groups=1, affine=False),nn.LeakyReLU(0.2, True)])
        self.filter2 = nn.Sequential(*[nn.Conv2d(4*opt.num_filters, 4*opt.num_filters, kernel_size=(1,3), stride=1, padding=(0,1), bias=True),nn.GroupNorm(num_channels=opt.num_filters, num_groups=1, affine=False),nn.LeakyReLU(0.2, True)])
        self.filter3 = nn.Sequential(*[nn.Conv2d(4*opt.num_filters, 4*opt.num_filters, kernel_size=(1,3), stride=1, padding=(0,1), bias=True),nn.GroupNorm(num_channels=opt.num_filters, num_groups=1, affine=False),nn.LeakyReLU(0.2, True)])
        self.filter4 = nn.Sequential(*[nn.Conv2d(4*opt.num_filters, 4*opt.num_filters, kernel_size=(1,3), stride=1, padding=(0,1), bias=True),nn.GroupNorm(num_channels=opt.num_filters, num_groups=1, affine=False),nn.LeakyReLU(0.2, True)])

        model_list_final = [nn.Conv2d(4*opt.num_filters, opt.num_filters, kernel_size=(1,3), stride=1, padding=(0,1), bias=True)]
        model_list_final += [nn.Conv2d(opt.num_filters, 1, kernel_size=(1,3), stride=1, padding=(0,1), bias=True)]
        self.model_final =  nn.Sequential(*model_list_final)

    def forward(self, input):
        output = self.model_start(input)
        output = self.filter1(output)
        output = self.filter2(output)
        output = self.filter3(output)
        output = self.filter4(output)
        output = self.model_final(output)

        return output

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)