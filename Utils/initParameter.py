import os
import numpy as np
import torch


class InitPara(object):
    def __init__(self):
        self.net_name = 'CUDIRCL-CT-mixtask-res'
        self.use_cuda = torch.cuda.is_available()
        self.root_path = '/mnt/kunlun/users/wanghao/data/across/'
        self.target_path = '/mnt/kunlun/users/wanghao/CUDIRCL/code/CT-res/all-train/save-p0-c5/'

        self.all_clients = ['client0', 'client1', 'client2', 'client3', 'client4', 'client5']
        self.initial_clients = ['client0', 'client1', 'client2', 'client3']
        self.join_clients = ['client4', 'client5']
        self.ac_list = list(self.initial_clients)
        self.task_mode = 'mixed'
        self.client_task = {
            'client0': 'reconstruction',
            'client1': 'denoising',
            'client2': 'reconstruction',
            'client3': 'denoising',
            'client4': 'reconstruction',
            'client5': 'denoising',
        }
        self.folder = {
            'client0': 'chest_2e5',
            'client1': 'siemens_1e5',
            'client2': 'ge_3e5',
            'client3': 'kidney',
            'client4': 'c1_3e5',
            'client5': 'c2_1e5',
        }

        self.batch_num = {client: 1000 for client in self.all_clients}
        self.batch_size = {client: 1 for client in self.all_clients}
        self.num_workers = {client: 10 for client in self.all_clients}
        self.test_batch_num = {client: 100 for client in self.all_clients}
        self.v_batch_num = {client: 100 for client in self.all_clients}
        self.lr = {client: 2e-5 for client in self.all_clients}

        self.num_filters = 16
        self.out_ch = 64
        self.reduction = 16
        self.type_loss = {
            'client0': 'MSELoss',
            'client1': 'MSELoss',
            'client2': 'MSELoss',
            'client3': 'MSELoss',
            'client4': 'MSELoss',
            'client5': 'unsuper_loss',
        }
        self.lambda_TV = 0.3
        self.mutual_weight_1 = {client: 1 for client in self.all_clients}
        self.mutual_weight_2 = {client: 1 for client in self.all_clients}
        self.huber_delta = 0.1

        self.is_train = True
        self.ds_scale = 1
        self.is_lr_scheduler = False
        self.geo_mode = 'parallel'
        self.angle_range = {'fanflat': 2 * np.pi, 'parallel': np.pi}

        gpu_id = 0
        self.gpu_id = {client: gpu_id for client in self.all_clients}

        self.save_as_mat = True
        self.is_shuffle = self.is_train

        self.client_epochs = 10
        self.pre_epoch = 0
        self.c_epochs = 10
        self.join_epoch = 1
        self.add_epoch = 5
        self.start_epoch = 0
        self.score_target = 98.5
        self.switch = True
        self.trans_method = 1
        self.save_method = 0

        self.result_folder = self.net_name + '_result'
        self.model_path = os.path.join(self.target_path, 'Model')

        for folder in [self.result_folder, 'Model_save', 'Loss_save', 'Optimizer_save', 'Model']:
            path = os.path.join(self.target_path, folder)
            if not os.path.isdir(path):
                os.makedirs(path)
