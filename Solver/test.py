import os

import numpy as np
import scipy.io
import torch

from Datasets.imageProcess import DeNormalize
from Solver.measure import Normarlize2_0_255, compute_measure


def _clients(opt):
    return list(getattr(opt, 'all_clients', opt.ac_list))


def _client_task(opt, client):
    if getattr(opt, 'task_mode', 'reconstruction') in ['reconstruction', 'denoising']:
        return opt.task_mode
    return opt.client_task[client]


def _target_for_loss(opt, phase, low_labels, labels):
    if opt.type_loss[phase] == 'unsuper_loss':
        return low_labels
    return labels


def _forward_local(model, task, sinogram, low_labels):
    if task == 'reconstruction':
        _, outputs = model(sinogram)
        return outputs
    return model(low_labels)


def test_model_mutual(dataloaders, local_model_list, local_criterion_list, opt=None):
    denormalize = DeNormalize('dicom')
    clients = _clients(opt)
    measure_fbp = {client: [] for client in clients}
    measure_output = {client: [] for client in clients}
    local_model_id = {client: idx for idx, client in enumerate(clients)}

    for phase in clients:
        task = _client_task(opt, phase)
        model_idx = local_model_id[phase]
        local_model_list[model_idx].eval()

        for i_batch, data in enumerate(dataloaders[phase]):
            if i_batch == opt.test_batch_num[phase]:
                break

            print('Processing {} batch_{}...'.format(phase, i_batch + 1))

            low_labels = data['fbp']
            labels = data['ndct']
            sinogram = data['sinogram']

            if opt.use_cuda:
                low_labels = low_labels.cuda(opt.gpu_id[phase])
                labels = labels.cuda(opt.gpu_id[phase])
                sinogram = sinogram.cuda(opt.gpu_id[phase])

            target = _target_for_loss(opt, phase, low_labels, labels)

            with torch.no_grad():
                outputs = _forward_local(local_model_list[model_idx], task, sinogram, low_labels)
                loss, _, _ = local_criterion_list[model_idx](outputs, low_labels, target)

            data['output'] = outputs.cpu()
            data['loss'] = loss.cpu()
            data['output'] = denormalize(data['output'])
            data['ndct'] = denormalize(data['ndct'])
            data['fbp'] = denormalize(data['fbp'])

            fbp_measure, output_measure = compute_measure(
                Normarlize2_0_255(data['fbp'][0, 0, 128:384, 128:384]),
                Normarlize2_0_255(data['ndct'][0, 0, 128:384, 128:384]),
                Normarlize2_0_255(data['output'][0, 0, 128:384, 128:384]),
                255,
            )

            data['fbp_measure'] = fbp_measure
            data['output_measure'] = output_measure
            measure_fbp[phase].append(np.array(fbp_measure))
            measure_output[phase].append(np.array(output_measure))

            if opt.save_as_mat:
                data_save = {}
                for key, value in data.items():
                    if torch.is_tensor(value):
                        data_save[key] = value.cpu().squeeze_().data.numpy()
                    else:
                        data_save[key] = value
                scipy.io.savemat(
                    os.path.join(opt.target_path, opt.result_folder, '{}_{}.mat'.format(phase, i_batch)),
                    mdict=data_save,
                )

    scipy.io.savemat(
        os.path.join(opt.target_path, opt.result_folder, 'measure_fbp.mat'),
        mdict={key: value for key, value in measure_fbp.items()},
    )
    scipy.io.savemat(
        os.path.join(opt.target_path, opt.result_folder, 'measure_output.mat'),
        mdict={key: value for key, value in measure_output.items()},
    )
