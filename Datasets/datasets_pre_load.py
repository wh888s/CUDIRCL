import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from Datasets.imageProcess import Normalize
from Datasets.utils import findFiles


class TrainData(Dataset):
    def __init__(self, root_dir, folder, is_train=False):
        phase = 'train' if is_train else 'test'
        self.image_paths = findFiles(root_dir + 'local/{}/{}/*.mat'.format(folder, phase))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return loadmat(self.image_paths[idx])


class TrainDataSet(Dataset):
    def __init__(self, root_dir, folder, is_train=False, task_mode='reconstruction', loss_type='MSELoss'):
        self.task_mode = task_mode
        self.loss_type = loss_type
        self.Normalize_d = Normalize('dicom')
        self.Normalize_s = Normalize('sino')
        self.imgset = TrainData(root_dir, folder, is_train=is_train)

    def __len__(self):
        return len(self.imgset)

    def __getitem__(self, idx):
        mat_data = self.imgset[idx]
        recon = self.Normalize_d(np.array([mat_data['fbp']]))
        if self.loss_type == 'unsuper_loss' or 'ndct' not in mat_data:
            img = recon.copy()
        else:
            img = self.Normalize_d(np.array([mat_data['ndct']]))

        if self.task_mode == 'reconstruction':
            sinogram = self.Normalize_s(np.array([mat_data['sinogram']]))
        else:
            sinogram = np.zeros_like(recon)

        if 'feature_vec' in mat_data:
            feature_vec = torch.FloatTensor(mat_data['feature_vec']).view(-1)
        else:
            feature_vec = torch.zeros(8).type(torch.FloatTensor)

        if feature_vec.numel() < 8:
            padding = torch.zeros(8 - feature_vec.numel()).type(torch.FloatTensor)
            feature_vec = torch.cat([feature_vec, padding], dim=0)
        else:
            feature_vec = feature_vec[:8]

        if feature_vec[6].item() > 0:
            feature_vec[6] = torch.log10(feature_vec[6])

        return {
            'ndct': torch.from_numpy(img).type(torch.FloatTensor),
            'sinogram': torch.from_numpy(sinogram).type(torch.FloatTensor),
            'fbp': torch.from_numpy(recon).type(torch.FloatTensor),
            'feature_vec': feature_vec,
        }
