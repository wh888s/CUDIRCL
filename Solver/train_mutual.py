import os
import pickle

import scipy.io
import torch

from Datasets.imageProcess import DeNormalize
from Fed_model.utils import (
    batch_MSE,
    batch_PSNR,
    batch_SSIM,
    calculate_score,
    check_client,
    draw_data,
    elimate,
    ex_seq,
    get_score,
    rank,
    tag_save,
    write_data,
)
from Model.Loss import HuberLoss


def _all_clients(opt):
    return list(getattr(opt, 'all_clients', opt.ac_list))


def _epoch_clients(opt, c_epoch):
    if hasattr(opt, 'initial_clients') and hasattr(opt, 'join_clients'):
        if c_epoch < opt.join_epoch:
            return list(opt.initial_clients)
        clients = list(opt.initial_clients)
        clients.extend([client for client in opt.join_clients if client not in clients])
        return clients
    return list(opt.ac_list)


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
        bp, outputs = model(sinogram)
        return bp, outputs, sinogram
    outputs = model(low_labels)
    return low_labels, outputs, low_labels


def _move_batch(data, phase, opt):
    low_labels = data['fbp']
    labels = data['ndct']
    sinogram = data['sinogram']
    feature_vec = data['feature_vec']
    if opt.use_cuda:
        low_labels = low_labels.cuda(opt.gpu_id[phase])
        labels = labels.cuda(opt.gpu_id[phase])
        sinogram = sinogram.cuda(opt.gpu_id[phase])
        feature_vec = feature_vec.cuda(opt.gpu_id[phase])
    return low_labels, labels, sinogram, feature_vec


def train_model_mutual(
    dataloaders,
    val_dataloaders,
    dataset_sizes,
    local_model_list,
    local_criterion_list,
    local_optimizer_list,
    local_lr_scheduler_list,
    mutual_model,
    mutual_optimizer,
    opt=None,
):
    denormalize = DeNormalize('dicom')
    clients_all = _all_clients(opt)
    clientlist_org = list(clients_all)
    local_model_id = {client: idx for idx, client in enumerate(clients_all)}
    mutual_criterion = HuberLoss(delta=opt.huber_delta)
    if opt.use_cuda:
        mutual_criterion = mutual_criterion.cuda(opt.gpu_id[clients_all[0]])
    min_loss = {client: 1.0 for client in clients_all}
    tag_list = []
    master_controller_flag = False

    for c_epoch in range(opt.pre_epoch, opt.c_epochs):
        opt.ac_list = _epoch_clients(opt, c_epoch)

        with open(opt.client_record_name, 'a') as f:
            f.write(' '.join(opt.ac_list) + '\n')

        if opt.is_lr_scheduler:
            for phase in opt.ac_list:
                scheduler = local_lr_scheduler_list[local_model_id[phase]]
                if scheduler is not None:
                    scheduler.step()

        for phase in opt.ac_list:
            print('********** Mutual-Learning train mode begin **********')
            model_idx = local_model_id[phase]
            task = _client_task(opt, phase)
            local_model_list[model_idx].train()
            mutual_model.train()

            if master_controller_flag and phase == opt.ac_list[-1]:
                now_client_train_epoch = opt.client_epochs + opt.add_epoch
            else:
                now_client_train_epoch = opt.client_epochs

            losses = torch.zeros(now_client_train_epoch, opt.batch_num[phase])

            for client_train_epoch in range(now_client_train_epoch):
                running_loss = 0.0

                for i_batch, data in enumerate(dataloaders[phase]):
                    if i_batch == opt.batch_num[phase]:
                        break

                    low_labels, labels, sinogram, feature_vec = _move_batch(data, phase, opt)
                    target = _target_for_loss(opt, phase, low_labels, labels)

                    local_optimizer_list[model_idx].zero_grad()
                    local_model_list[model_idx].zero_grad()
                    mutual_optimizer.zero_grad()
                    mutual_model.zero_grad()

                    proxy_input, local_outputs, batch_input = _forward_local(
                        local_model_list[model_idx],
                        task,
                        sinogram,
                        low_labels,
                    )
                    local_loss, loss_self, loss_tv = local_criterion_list[model_idx](
                        local_outputs,
                        low_labels,
                        target,
                    )

                    pred_mutual = mutual_model(proxy_input.detach(), feature_vec)
                    mutual_local_loss, mutual_loss_self, mutual_loss_tv = local_criterion_list[model_idx](
                        pred_mutual,
                        low_labels,
                        target,
                    )
                    local_proxy_loss = mutual_criterion(local_outputs, pred_mutual.detach())
                    mutual_proxy_loss = mutual_criterion(pred_mutual, local_outputs.detach())
                    loss = local_loss + opt.mutual_weight_1[phase] * local_proxy_loss
                    mutual_loss = mutual_local_loss + opt.mutual_weight_2[phase] * mutual_proxy_loss

                    loss.backward()
                    mutual_loss.backward()

                    psnr_train = batch_PSNR(local_outputs, target, 1.0)
                    ssim_train = batch_SSIM(local_outputs, target)
                    mse_train = batch_MSE(local_outputs, target)
                    psnr_mutual = batch_PSNR(pred_mutual, target, 1.0)
                    ssim_mutual = batch_SSIM(pred_mutual, target)
                    mse_mutual = batch_MSE(pred_mutual, target)

                    local_optimizer_list[model_idx].step()
                    mutual_optimizer.step()

                    if opt.type_loss[phase] == 'unsuper_loss':
                        print(
                            '{} Training || {}, c_epoch: {}, client_epoch: {} || {}/{}, local_loss: {:.8f}, self_loss: {:.8f}, tv_loss: {:.8f}, mutual_loss: {:.8f}, proxy_loss: {:.8f}, PSNR: {:.6f}, SSIM: {:.6f}, MSE: {:.6f}'.format(
                                opt.net_name,
                                phase,
                                c_epoch,
                                client_train_epoch,
                                i_batch,
                                opt.batch_num[phase],
                                loss.item(),
                                loss_self,
                                loss_tv,
                                mutual_loss.item(),
                                local_proxy_loss.item(),
                                psnr_train,
                                ssim_train,
                                mse_train,
                            )
                        )
                    else:
                        print(
                            '{} Training || {}, c_epoch: {}, client_epoch: {} || {}/{}, local_loss: {:.8f}, mutual_loss: {:.8f}, proxy_loss: {:.8f}, PSNR: {:.6f}, SSIM: {:.6f}, MSE: {:.6f}'.format(
                                opt.net_name,
                                phase,
                                c_epoch,
                                client_train_epoch,
                                i_batch,
                                opt.batch_num[phase],
                                loss.item(),
                                mutual_loss.item(),
                                local_proxy_loss.item(),
                                psnr_train,
                                ssim_train,
                                mse_train,
                            )
                        )
                    print(
                        'Mutual Training || {}, c_epoch: {}, client_epoch: {} || {}/{}, mutual_task_loss: {:.8f}, mutual_proxy_loss: {:.8f}, PSNR: {:.6f}, SSIM: {:.6f}, MSE: {:.6f}'.format(
                            phase,
                            c_epoch,
                            client_train_epoch,
                            i_batch,
                            opt.batch_num[phase],
                            mutual_local_loss.item(),
                            mutual_proxy_loss.item(),
                            psnr_mutual,
                            ssim_mutual,
                            mse_mutual,
                        )
                    )

                    losses[client_train_epoch, i_batch] = loss.item()
                    running_loss += loss.item() * batch_input.size(0)

                epoch_loss = running_loss / dataset_sizes[phase]
                print('{} Loss: {:.8f}'.format(phase, epoch_loss))

                if epoch_loss < min_loss[phase]:
                    min_loss[phase] = epoch_loss
                    scipy.io.savemat(
                        os.path.join(opt.target_path, 'Loss_save', 'min_loss_{}.mat'.format(c_epoch)),
                        mdict={key: value for key, value in min_loss.items()},
                    )
                    with open(os.path.join(opt.target_path, 'Loss_save', 'min_loss_{}.dat'.format(c_epoch)), 'wb') as f:
                        pickle.dump(min_loss, f, True)

                    torch.save(
                        local_model_list[model_idx].state_dict(),
                        os.path.join(opt.target_path, 'Model_save', 'best_{}_model.pkl'.format(phase)),
                    )
                    torch.save(
                        local_optimizer_list[model_idx].state_dict(),
                        os.path.join(opt.target_path, 'Optimizer_save', 'best_{}_optimizer.pkl'.format(phase)),
                    )

                tmp_losses = {phase: losses[:client_train_epoch + 1]}
                scipy.io.savemat(
                    os.path.join(opt.target_path, 'Loss_save', 'losses_{}.mat'.format(c_epoch)),
                    mdict={key: value.numpy() for key, value in tmp_losses.items()},
                )
                with open(os.path.join(opt.target_path, 'Loss_save', 'losses_{}.dat'.format(c_epoch)), 'wb') as f:
                    pickle.dump(tmp_losses, f, True)

            torch.save(
                local_model_list[model_idx].state_dict(),
                os.path.join(opt.target_path, 'Model_save', 'trans_{}_model.pkl'.format(phase)),
            )
            torch.save(
                local_optimizer_list[model_idx].state_dict(),
                os.path.join(opt.target_path, 'Optimizer_save', 'trans_{}_optimizer.pkl'.format(phase)),
            )
            torch.save(mutual_model.state_dict(), os.path.join(opt.target_path, 'Model_save', 'mutual_model.pkl'))
            torch.save(
                mutual_optimizer.state_dict(),
                os.path.join(opt.target_path, 'Optimizer_save', 'mutual_optimizer.pkl'),
            )
            print('***************** model saved!!! *****************')

        print('***************** model validing!!! *****************')
        for phase in opt.ac_list:
            content_valid = ''
            add_psnr_valid = 0
            add_ssim_valid = 0
            add_mse_valid = 0
            add_score_valid = 0
            model_idx = local_model_id[phase]
            task = _client_task(opt, phase)
            local_model_list[model_idx].eval()

            for i_batch, data in enumerate(val_dataloaders[phase]):
                if i_batch == opt.v_batch_num[phase]:
                    break

                low_labels, labels, sinogram, feature_vec = _move_batch(data, phase, opt)
                target = _target_for_loss(opt, phase, low_labels, labels)

                with torch.no_grad():
                    _, local_outputs, _ = _forward_local(
                        local_model_list[model_idx],
                        task,
                        sinogram,
                        low_labels,
                    )
                    local_loss, _, _ = local_criterion_list[model_idx](local_outputs, low_labels, target)
                    psnr_valid = batch_PSNR(local_outputs, target, 1.0)
                    ssim_valid = batch_SSIM(local_outputs, target)
                    mse_valid = batch_MSE(local_outputs, target)
                    score_valid = calculate_score(psnr_valid, ssim_valid, mse_valid)

                data['output'] = local_outputs.cpu()
                data['loss'] = local_loss.cpu()
                data['output'] = denormalize(data['output'])
                data['ndct'] = denormalize(data['ndct'])
                data['fbp'] = denormalize(data['fbp'])
                data_save = {key: value.cpu().squeeze_().data.numpy() for key, value in data.items()}
                scipy.io.savemat(
                    os.path.join(opt.target_path, opt.result_folder, 'valid_{}_{}.mat'.format(phase, i_batch)),
                    mdict=data_save,
                )

                add_psnr_valid += psnr_valid
                add_ssim_valid += ssim_valid
                add_mse_valid += mse_valid
                add_score_valid += score_valid
                print(
                    '{} Validing || {}, c_epoch: {} || {}/{}, subLoss: {:.8f}, PSNR: {:.6f}, SSIM: {:.6f}, MSE: {:.6f}'.format(
                        opt.net_name,
                        phase,
                        c_epoch,
                        i_batch,
                        opt.v_batch_num[phase],
                        local_loss.item(),
                        psnr_valid,
                        ssim_valid,
                        mse_valid,
                    )
                )

            mean_psnr_valid = add_psnr_valid / opt.v_batch_num[phase]
            mean_ssim_valid = add_ssim_valid / opt.v_batch_num[phase]
            mean_mse_valid = add_mse_valid / opt.v_batch_num[phase]
            mean_score_valid = add_score_valid / opt.v_batch_num[phase]
            content_valid += '{}  {}  {}  {}  {}  \n'.format(
                c_epoch,
                mean_psnr_valid,
                mean_ssim_valid,
                mean_mse_valid,
                mean_score_valid,
            )
            write_data(opt.record_valid_path, phase, content_valid)
            print('client: {}, score: {:.8f}'.format(phase, mean_score_valid))

        if opt.switch and c_epoch >= opt.start_epoch:
            master_controller_flag = True
            score_list = get_score(opt.record_valid_path, opt.ac_list)
            train_new_clientindex, eva_score_list = rank(score_list, opt.ac_list)
            train_new_clientlist = ex_seq(opt.ac_list, train_new_clientindex)
            save_list, tag_list = tag_save(
                opt.score_target,
                train_new_clientlist,
                eva_score_list,
                tag_list,
                opt.save_method,
            )

            for client in save_list:
                model_save_name = 'trans_model_{}_{}.pkl'.format(c_epoch, client)
                torch.save(
                    local_model_list[local_model_id[client]].state_dict(),
                    os.path.join(opt.model_path, model_save_name),
                )
                print('trans_epoch:{} {} meets training requirements'.format(c_epoch, client))

            if len(tag_list) < len(clientlist_org):
                if opt.trans_method == 0:
                    opt.ac_list = train_new_clientlist
                elif opt.trans_method == 1:
                    elimate_num = check_client(train_new_clientlist, tag_list)
                    opt.ac_list = elimate(train_new_clientlist, elimate_num)
                print('The new client training sequence is {}'.format(opt.ac_list))
            else:
                print('All clients have completed training')

    draw_data(opt.record_valid_path, clientlist_org, opt.score_target)
