import os

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from Datasets.datasets_pre_load import TrainDataSet
from Model.imgMAP import imgMAP_2, imgMAP_3
from Model.mutualNet import MutualNet
from Model.normal_iRadonMAP import normal_iRadonMAP_1, normal_iRadonMAP_7, normal_iRadonMAP_8


def _clients(opt):
    return list(getattr(opt, 'all_clients', opt.ac_list))


def _client_task(opt, client):
    if getattr(opt, 'task_mode', 'reconstruction') in ['reconstruction', 'denoising']:
        return opt.task_mode
    return opt.client_task[client]


def _to_device(module, opt, client):
    if opt.use_cuda:
        return module.cuda(opt.gpu_id[client])
    return module


def _load_state(model, path):
    if os.path.isfile(path):
        print('Loading previously trained network: {}...'.format(os.path.basename(path)))
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('Done!')


def _load_optimizer(optimizer, path):
    if os.path.isfile(path):
        print('Loading previous optimizer: {}...'.format(os.path.basename(path)))
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        try:
            optimizer.load_state_dict(checkpoint)
        except ValueError:
            print('Skip incompatible optimizer: {}'.format(os.path.basename(path)))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('Done!')


def _build_local_model(client, opt, geo):
    task = _client_task(opt, client)
    if task == 'denoising':
        if client == 'client1':
            return imgMAP_2(opt, opt.gpu_id[client])
        return imgMAP_3(opt, opt.gpu_id[client])

    if client == 'client2':
        return normal_iRadonMAP_7(geo[opt.geo_id[client]], opt, opt.gpu_id[client])
    if client == 'client4':
        return normal_iRadonMAP_8(geo[opt.geo_id[client]], opt, opt.gpu_id[client])
    return normal_iRadonMAP_1(geo[opt.geo_id[client]], opt, opt.gpu_id[client])


def mutual_init(opt, geo):
    clients = _clients(opt)
    mutual_model = _to_device(MutualNet(opt), opt, clients[0])
    mutual_model_path = os.path.join(opt.target_path, 'Model_save', 'mutual_model.pkl')
    _load_state(mutual_model, mutual_model_path)

    mutual_optimizer = optim.RMSprop(
        mutual_model.parameters(),
        lr=opt.lr[clients[0]],
        momentum=0.9,
        weight_decay=0.0,
    )

    if opt.is_train:
        mutual_optimizer_path = os.path.join(opt.target_path, 'Optimizer_save', 'mutual_optimizer.pkl')
        _load_optimizer(mutual_optimizer, mutual_optimizer_path)

    local_model_list = [_build_local_model(client, opt, geo) for client in clients]

    for idx, client in enumerate(clients):
        model_name = 'trans_{}_model'.format(client) if opt.is_train else 'best_{}_model'.format(client)
        model_path = os.path.join(opt.target_path, 'Model_save', '{}.pkl'.format(model_name))
        _load_state(local_model_list[idx], model_path)

    local_criterion_list = [
        MutualLoss(opt, type_loss=opt.type_loss[client]).cuda(opt.gpu_id[client])
        if opt.use_cuda else MutualLoss(opt, type_loss=opt.type_loss[client])
        for client in clients
    ]

    local_optimizer_list = [
        optim.RMSprop(
            local_model_list[idx].parameters(),
            lr=opt.lr[client],
            momentum=0.9,
            weight_decay=0.0,
        )
        for idx, client in enumerate(clients)
    ]

    if opt.is_train:
        for idx, client in enumerate(clients):
            optimizer_path = os.path.join(opt.target_path, 'Optimizer_save', 'trans_{}_optimizer.pkl'.format(client))
            _load_optimizer(local_optimizer_list[idx], optimizer_path)

    local_lr_scheduler_list = [
        lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) if opt.is_lr_scheduler else None
        for optimizer in local_optimizer_list
    ]

    if opt.is_train:
        opt.client_record_name = os.path.join(opt.target_path, 'client_record.txt')
        if not os.path.isfile(opt.client_record_name):
            open(opt.client_record_name, mode='a', encoding='utf-8').close()

        opt.record_valid_path = os.path.join(opt.target_path, 'site_valid_record')
        if not os.path.isdir(opt.record_valid_path):
            os.makedirs(opt.record_valid_path)

        for client in clients:
            record_path = os.path.join(opt.record_valid_path, '{}_record.txt'.format(client))
            if not os.path.isfile(record_path):
                open(record_path, mode='a', encoding='utf-8').close()

    print('Constructing Datasets...')
    datasets = {
        client: TrainDataSet(
            opt.root_path,
            opt.folder[client],
            is_train=opt.is_train,
            task_mode=_client_task(opt, client),
            loss_type=opt.type_loss[client],
        )
        for client in clients
    }
    dataloaders = {
        client: DataLoader(
            datasets[client],
            opt.batch_size[client],
            shuffle=opt.is_shuffle,
            pin_memory=opt.use_cuda,
            num_workers=opt.num_workers[client],
        )
        for client in clients
    }
    dataset_sizes = {
        client: opt.batch_num[client] * opt.batch_size[client]
        for client in clients
    }
    val_datasets = {
        client: TrainDataSet(
            opt.root_path,
            opt.folder[client],
            is_train=False,
            task_mode=_client_task(opt, client),
            loss_type=opt.type_loss[client],
        )
        for client in clients
    }
    val_dataloaders = {
        client: DataLoader(
            val_datasets[client],
            opt.batch_size[client],
            shuffle=False,
            pin_memory=opt.use_cuda,
            num_workers=opt.num_workers[client],
        )
        for client in clients
    }
    return (
        local_model_list,
        dataloaders,
        val_dataloaders,
        dataset_sizes,
        local_criterion_list,
        local_optimizer_list,
        local_lr_scheduler_list,
        mutual_model,
        mutual_optimizer,
    )


class MutualLoss(torch.nn.Module):
    def __init__(self, opt, type_loss='L1Loss', size_average=True):
        super(MutualLoss, self).__init__()
        self.type_loss = type_loss
        self.opt = opt
        reduction = 'mean' if size_average else 'sum'

        if self.type_loss == 'L1Loss':
            self.img_loss = torch.nn.L1Loss(reduction=reduction)
        elif self.type_loss == 'MSELoss':
            self.img_loss = torch.nn.MSELoss(reduction=reduction)
        elif self.type_loss == 'unsuper_loss':
            self.img_loss = self.unsupervised_loss
        else:
            raise NameError('There is no supported loss: %s.' % self.type_loss)

    def forward(self, output, noisy_img, label_img):
        if self.type_loss == 'unsuper_loss':
            return self.img_loss(output, noisy_img.detach())
        return self.img_loss(output, label_img.detach()), 0, 0

    def unsupervised_loss(self, denoised_img, noisy_img):
        loss_self = torch.mean(torch.abs(denoised_img - noisy_img))
        loss_tv = torch_TV(denoised_img)
        return loss_self + self.opt.lambda_TV * loss_tv, loss_self.item(), loss_tv.item()


def torch_TV(out_fbp):
    out_fbp = out_fbp.squeeze()
    pixel_dif1 = out_fbp[1:, :] - out_fbp[:-1, :]
    pixel_dif2 = out_fbp[:, 1:] - out_fbp[:, :-1]
    return torch.mean(torch.abs(pixel_dif1)) + torch.mean(torch.abs(pixel_dif2))
