import warnings

from Fed_model.mutual_init import mutual_init
from Solver.test import test_model_mutual
from Solver.train_mutual import train_model_mutual
from Utils.initParameter import InitPara

warnings.filterwarnings('ignore')


def build_geo(opt):
    geo_0 = {
        'nVoxelX': int(512), 'sVoxelX': 384.0, 'dVoxelX': 0.75,
        'nVoxelY': int(512), 'sVoxelY': 384.0, 'dVoxelY': 0.75,
        'nDetecU': int(1024), 'sDetecU': 665.6, 'dDetecU': 0.65,
        'offOriginX': 0.0, 'offOriginY': 0.0,
        'views': int(512 / opt.ds_scale), 'slices': 1, 'Dose': 2e5, 'sigma': 3.0, 'scale': 2,
        'DSD': 750.1740, 'DSO': 476.8300, 'DOD': 273.344,
        'start_angle': 0.0, 'end_angle': opt.angle_range[opt.geo_mode],
        'mode': opt.geo_mode, 'extent': 1, 'name': 'client0',
    }
    geo_1 = {
        'nVoxelX': int(512), 'sVoxelX': 384.0, 'dVoxelX': 0.75,
        'nVoxelY': int(512), 'sVoxelY': 384.0, 'dVoxelY': 0.75,
        'nDetecU': int(1024), 'sDetecU': 665.6, 'dDetecU': 0.65,
        'offOriginX': 0.0, 'offOriginY': 0.0,
        'views': int(512 / opt.ds_scale), 'slices': 1, 'Dose': 1e5, 'sigma': 3.0, 'scale': 2,
        'DSD': 750.1740, 'DSO': 476.8300, 'DOD': 273.344,
        'start_angle': 0.0, 'end_angle': opt.angle_range[opt.geo_mode],
        'mode': opt.geo_mode, 'extent': 1, 'name': 'client3',
    }
    geo_2 = {
        'nVoxelX': int(512), 'sVoxelX': 340.0192, 'dVoxelX': 0.6641,
        'nVoxelY': int(512), 'sVoxelY': 340.0192, 'dVoxelY': 0.6641,
        'nDetecU': int(904), 'sDetecU': 552, 'dDetecU': 0.5,
        'offOriginX': 0.0, 'offOriginY': 0.0,
        'views': int(768 / opt.ds_scale), 'slices': 1, 'Dose': 3e5, 'sigma': 1.0, 'scale': 2,
        'DSD': 1085.6, 'DSO': 595.0, 'DOD': 490.6, 'filter': 'cosine',
        'start_angle': 0.0, 'end_angle': opt.angle_range[opt.geo_mode],
        'mode': opt.geo_mode, 'extent': 1, 'name': 'client2',
    }
    geo_3 = {
        'nVoxelX': 512, 'sVoxelX': 384.0, 'dVoxelX': 0.75,
        'nVoxelY': 512, 'sVoxelY': 384.0, 'dVoxelY': 0.75,
        'nDetecU': 1008, 'sDetecU': 665.6, 'dDetecU': 0.65,
        'offOriginX': 0.0, 'offOriginY': 0.0,
        'views': int(896 / opt.ds_scale), 'slices': 1, 'Dose': 1e5, 'sigma': 3.0, 'scale': 1,
        'DSD': 750.1740, 'DSO': 476.8300, 'DOD': 273.344,
        'start_angle': 0.0, 'end_angle': opt.angle_range[opt.geo_mode],
        'mode': opt.geo_mode, 'extent': 1, 'name': 'client1',
    }
    geo_4 = {
        'nVoxelX': 512, 'sVoxelX': 340.0192, 'dVoxelX': 0.6641,
        'nVoxelY': 512, 'sVoxelY': 340.0192, 'dVoxelY': 0.6641,
        'nDetecU': 1024, 'sDetecU': 552, 'dDetecU': 0.5,
        'offOriginX': 0.0, 'offOriginY': 0.0,
        'views': int(896 / opt.ds_scale), 'slices': 1, 'Dose': 3e5, 'sigma': 1.0, 'scale': 1,
        'DSD': 946.7460, 'DSO': 538.5200, 'DOD': 408.2260,
        'start_angle': 0.0, 'end_angle': opt.angle_range[opt.geo_mode],
        'mode': opt.geo_mode, 'extent': 1, 'name': 'client4',
    }
    geo_5 = {
        'nVoxelX': 512, 'sVoxelX': 380.0, 'dVoxelX': 0.7421875,
        'nVoxelY': 512, 'sVoxelY': 380.0, 'dVoxelY': 0.7421875,
        'nDetecU': 904, 'sDetecU': 554.4, 'dDetecU': 0.548,
        'offOriginX': 0.0, 'offOriginY': 0.0,
        'views': int(512 / opt.ds_scale), 'slices': 1, 'Dose': 1e5, 'sigma': 5.0, 'scale': 1,
        'DSD': 865.20, 'DSO': 613.0, 'DOD': 252.20,
        'start_angle': 0.0, 'end_angle': opt.angle_range[opt.geo_mode],
        'mode': opt.geo_mode, 'extent': 1, 'name': 'client5',
    }
    return [geo_0, geo_1, geo_2, geo_3, geo_4, geo_5]


def main(opt):
    geo = build_geo(opt)
    clients = getattr(opt, 'all_clients', opt.ac_list)
    opt.geo_id = {client: idx for idx, client in enumerate(clients)}

    print('Mutual initing begins!')
    result = mutual_init(opt, geo)
    local_model_list, dataloaders, val_dataloaders, dataset_sizes = result[:4]
    local_criterion_list, local_optimizer_list, local_lr_scheduler_list = result[4:7]
    mutual_model, mutual_optimizer = result[7:]

    if opt.is_train:
        print('Mutual training begins!')
        train_model_mutual(
            dataloaders,
            val_dataloaders,
            dataset_sizes,
            local_model_list,
            local_criterion_list,
            local_optimizer_list,
            local_lr_scheduler_list,
            mutual_model,
            mutual_optimizer,
            opt=opt,
        )
    else:
        print('Testing begins!')
        test_model_mutual(dataloaders, local_model_list, local_criterion_list, opt)


if __name__ == '__main__':
    main(InitPara())
