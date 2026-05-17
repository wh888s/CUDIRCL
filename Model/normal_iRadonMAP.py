import torch.nn as nn

from Model.backProj import BackProj
from Model.sinoNet import SinoNet, SinoNet_1
from Model.spatialNet import SpatialNet, SpatialNet_1, SpatialNet_4


def _to_device(module, opt, gpu_id):
    if opt.use_cuda:
        return module.cuda(gpu_id)
    return module


class normal_iRadonMAP_1(nn.Module):
    def __init__(self, geo, opt, gpu_id):
        super(normal_iRadonMAP_1, self).__init__()
        self.SinoNet = _to_device(SinoNet(opt), opt, gpu_id)
        self.BackProjNet = _to_device(BackProj(geo), opt, gpu_id)
        self.SpatialNet = _to_device(SpatialNet(opt), opt, gpu_id)
        self.gpu_id = gpu_id

    def forward(self, input):
        sino = self.SinoNet(input)
        bp = self.BackProjNet(sino, self.gpu_id)
        output = self.SpatialNet(bp)
        return bp, output


class normal_iRadonMAP_7(nn.Module):
    def __init__(self, geo, opt, gpu_id):
        super(normal_iRadonMAP_7, self).__init__()
        self.SinoNet = _to_device(SinoNet(opt), opt, gpu_id)
        self.BackProjNet = _to_device(BackProj(geo), opt, gpu_id)
        self.SpatialNet = _to_device(SpatialNet_1(opt), opt, gpu_id)
        self.gpu_id = gpu_id

    def forward(self, input):
        sino = self.SinoNet(input)
        bp = self.BackProjNet(sino, self.gpu_id)
        output = self.SpatialNet(bp)
        return bp, output


class normal_iRadonMAP_8(nn.Module):
    def __init__(self, geo, opt, gpu_id):
        super(normal_iRadonMAP_8, self).__init__()
        self.SinoNet = _to_device(SinoNet_1(opt), opt, gpu_id)
        self.BackProjNet = _to_device(BackProj(geo), opt, gpu_id)
        self.SpatialNet = _to_device(SpatialNet_4(in_chn=1), opt, gpu_id)
        self.gpu_id = gpu_id

    def forward(self, input):
        sino = self.SinoNet(input)
        bp = self.BackProjNet(sino, self.gpu_id)
        output = self.SpatialNet(bp)
        return bp, output


normal_iRadonMAP = normal_iRadonMAP_1
