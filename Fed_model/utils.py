import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def write_data(path, site_name, data):
    record_path = os.path.join(path, site_name + '_record.txt')
    with open(record_path, 'a') as f:
        f.write(data)


def get_score(record_path, site_name):
    score_list = []
    for site in site_name:
        txt_path = os.path.join(record_path, site + '_record.txt')
        site_score_record = np.loadtxt(txt_path)
        score = site_score_record[-1, -1] if site_score_record.ndim == 2 else site_score_record[-1]
        score_list.append(score)
    return score_list


def rank(score_list, site_name):
    sorted_items = sorted(zip(score_list, range(len(site_name))), reverse=True)
    eva_score_list = [score for score, _ in sorted_items]
    site_new_index = [idx for _, idx in sorted_items]
    return site_new_index, eva_score_list


def ex_seq(org_list, train_new_siteindex):
    return [org_list[idx] for idx in train_new_siteindex]


def tag_save(threshold_value, train_new_sitelist, eva_score_list, tag_list, save_method):
    save_list = []
    for site, score in zip(train_new_sitelist, eva_score_list):
        if score >= threshold_value and site not in tag_list:
            save_list.append(site)
            tag_list.append(site)
            if save_method == 0:
                break
    return save_list, tag_list


def check_client(org_list, tag_list):
    eliminate_num = 0
    for idx, client in enumerate(org_list):
        if client in tag_list:
            eliminate_num = idx + 1
    return eliminate_num


def elimate(org_list, elimate_num):
    return list(org_list[elimate_num:])


def draw_data(site_valid_record_path, siteslist, threshold):
    records = {}
    for site in siteslist:
        site_record_path = os.path.join(site_valid_record_path, site + '_record.txt')
        if os.path.isfile(site_record_path):
            data = np.loadtxt(site_record_path)
            if data.size:
                records[site] = np.atleast_2d(data)

    if not records:
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=600)
    metric_map = [
        ('PSNR', 1, axes[0][0]),
        ('SSIM', 2, axes[0][1]),
        ('MSE', 3, axes[1][0]),
        ('SCORE', 4, axes[1][1]),
    ]

    for title, column, axis in metric_map:
        for site, data in records.items():
            total_epoch = range(len(data))
            axis.plot(total_epoch, data[:, column], linewidth=1, label=site)
        if title == 'SCORE':
            axis.axhline(y=threshold, alpha=0.5, linewidth=1, label='threshold')
        axis.set_xlabel('epoch')
        axis.set_ylabel(title)

    axes[1][1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(site_valid_record_path, 'fig.png'), format='png')
    plt.close(fig)
    print('The picture is finished')


def batch_PSNR(img, imclean, data_range):
    img_np = img.data.cpu().numpy().astype(np.float32)
    clean_np = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for idx in range(img_np.shape[0]):
        psnr += compare_psnr(clean_np[idx, :, :, :], img_np[idx, :, :, :], data_range=data_range)
    return psnr / img_np.shape[0]


def batch_SSIM(img, imclean):
    img_np = img.data.cpu().numpy().astype(np.float32)
    clean_np = imclean.data.cpu().numpy().astype(np.float32)
    ssim = 0
    for idx in range(img_np.shape[0]):
        clean_img = clean_np[idx, 0, :, :]
        pred_img = img_np[idx, 0, :, :]
        ssim += compare_ssim(clean_img, pred_img, gaussian_weights=True, data_range=1.0)
    return ssim / img_np.shape[0]


def batch_MSE(img, imclean):
    img_np = img.data.cpu().numpy().astype(np.float32)
    clean_np = imclean.data.cpu().numpy().astype(np.float32)
    mse = 0
    for idx in range(img_np.shape[0]):
        mse += compare_mse(clean_np[idx, :, :, :], img_np[idx, :, :, :])
    return mse / img_np.shape[0]


def calculate_score(psnr, ssim, mse):
    if psnr <= 0 or ssim <= 0 or mse >= 1:
        return 0

    target_psnr = 45
    target_ssim = 1
    target_mse = 0

    psnr_error = max(0, target_psnr - psnr)
    ssim_error = np.abs(ssim - target_ssim)
    mse_error = np.abs(mse - target_mse)

    psnr_error_mapped = (psnr_error / target_psnr) * 100
    ssim_error_mapped = (ssim_error / target_ssim) * 100
    mse_error_mapped = mse_error * 100
    average_error = (psnr_error_mapped + ssim_error_mapped + mse_error_mapped) / 3
    return max(0, 100 - average_error)
