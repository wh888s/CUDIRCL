import torch.nn as nn

from Model.spatialNet import SpatialNet, SpatialNet_1, SpatialNet_2, SpatialNet_3


def _to_device(module, opt, gpu_id):
    if opt.use_cuda:
        return module.cuda(gpu_id)
    return module


class imgMAP(nn.Module):
    def __init__(self, opt, gpu_id):
        super(imgMAP, self).__init__()
        self.SpatialNet = _to_device(SpatialNet(opt), opt, gpu_id)

    def forward(self, input):
        return self.SpatialNet(input)


class imgMAP_1(nn.Module):
    def __init__(self, opt, gpu_id):
        super(imgMAP_1, self).__init__()
        self.SpatialNet = _to_device(SpatialNet_1(opt), opt, gpu_id)

    def forward(self, input):
        return self.SpatialNet(input)


class imgMAP_2(nn.Module):
    def __init__(self, opt, gpu_id):
        super(imgMAP_2, self).__init__()
        self.SpatialNet = _to_device(SpatialNet_2(opt.out_ch), opt, gpu_id)

    def forward(self, input):
        return self.SpatialNet(input)


class imgMAP_3(nn.Module):
    def __init__(self, opt, gpu_id):
        super(imgMAP_3, self).__init__()
        self.SpatialNet = _to_device(SpatialNet_3(in_chn=1), opt, gpu_id)

    def forward(self, input):
        return self.SpatialNet(input)
