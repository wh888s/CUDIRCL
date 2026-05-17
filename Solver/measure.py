from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def Normarlize2_0_255(input):
    original_min = torch.min(input)
    original_max = torch.max(input)
    original_range = original_max - original_min
    desired_min = 0
    desired_max = 255
    desired_range = desired_max - desired_min
    return desired_range * (input - original_min) / original_range + desired_min


def compute_measure(x, y, pred, data_range):
    original_psnr = compute_PSNR(x, y, data_range)
    original_ssim = compute_SSIM(x, y, data_range)
    original_rmse = compute_RMSE(x, y)
    pred_psnr = compute_PSNR(pred, y, data_range)
    pred_ssim = compute_SSIM(pred, y, data_range)
    pred_rmse = compute_RMSE(pred, y)
    return torch.FloatTensor([original_psnr, original_ssim, original_rmse]), torch.FloatTensor(
        [pred_psnr, pred_ssim, pred_rmse]
    )


def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()


def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2, data_range):
    mse = compute_MSE(img1, img2)
    if type(img1) == torch.Tensor:
        return 10 * torch.log10((data_range ** 2) / mse).item()
    return 10 * np.log10((data_range ** 2) / mse)


def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
    if len(img1.size()) == 2:
        shape = img1.shape[-1]
        img1 = img1.view(1, 1, shape, shape)
        img2 = img2.view(1, 1, shape, shape)

    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2)
    mu2 = F.conv2d(img2, window, padding=window_size // 2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if size_average:
        return ssim_map.mean().item()
    return ssim_map.mean(1).mean(1).mean(1).item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return Variable(window_2d.expand(channel, 1, window_size, window_size).contiguous())
