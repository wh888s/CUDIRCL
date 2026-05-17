import copy
import warnings

import astra
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from Model.utils import int2tensor

warnings.filterwarnings('ignore')


class backPrj_fun(Function):
    @staticmethod
    def forward(ctx, input, options, cfg, proj_id, geo, gpu_id):
        sinogram = input.detach().cpu().numpy().reshape(geo['views'], geo['nDetecU'])
        sinogram_id = astra.data2d.create('-sino', options['proj_geom'])
        astra.data2d.store(sinogram_id, sinogram)
        rec_id = astra.data2d.create('-vol', options['vol_geom'])

        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        recon = np.expand_dims(astra.data2d.get(rec_id), axis=0)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sinogram_id)
        astra.algorithm.delete(alg_id)

        output = torch.from_numpy(recon).cuda(gpu_id)
        ctx.save_for_backward(
            int2tensor(proj_id),
            int2tensor(geo['views']),
            int2tensor(geo['nDetecU']),
            int2tensor(gpu_id),
            int2tensor(geo['nVoxelX']),
            int2tensor(geo['nVoxelY']),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        proj_id, views, n_detec_u, gpu_id, nx, ny = ctx.saved_tensors
        proj_id = int(proj_id)
        views = int(views)
        n_detec_u = int(n_detec_u)
        gpu_id = int(gpu_id)
        nx = int(nx)
        ny = int(ny)

        img = grad_output.detach().cpu().numpy().reshape(nx, ny)
        sinogram_id, sino = astra.create_sino(img, proj_id)
        astra.data2d.delete(sinogram_id)
        proj = copy.deepcopy(sino)
        grad_input = torch.from_numpy(proj)
        grad_input = grad_input.view(-1, 1, views, n_detec_u).cuda(gpu_id)
        return grad_input, None, None, None, None, None


class BackProj(nn.Module):
    def __init__(self, geo):
        super(BackProj, self).__init__()
        self.channel = 1
        self.geo = geo
        self.options = {
            'proj_geom': astra.create_proj_geom(
                geo['mode'],
                geo['dDetecU'],
                geo['nDetecU'],
                np.linspace(geo['start_angle'], geo['end_angle'], geo['views'], False),
                geo['DSO'],
                geo['DOD'],
            ),
            'vol_geom': astra.create_vol_geom(
                geo['nVoxelY'],
                geo['nVoxelX'],
                -geo['sVoxelY'] / 2,
                geo['sVoxelY'] / 2,
                -geo['sVoxelX'] / 2,
                geo['sVoxelX'] / 2,
            ),
        }

        if geo['mode'] == 'parallel':
            self.proj_id = astra.create_projector('cuda', self.options['proj_geom'], self.options['vol_geom'])
        elif geo['mode'] == 'fanflat':
            self.proj_id = astra.create_projector('line_fanflat', self.options['proj_geom'], self.options['vol_geom'])
        else:
            raise ValueError('Unsupported projection mode: {}'.format(geo['mode']))

        self.cfg = astra.astra_dict('FBP_CUDA')
        self.cfg['ProjectorId'] = self.proj_id

    def forward(self, input, gpu_id_conv):
        output = backPrj_fun.apply(input, self.options, self.cfg, self.proj_id, self.geo, gpu_id_conv)
        return output.view(-1, self.channel, self.geo['nVoxelX'], self.geo['nVoxelY'])
