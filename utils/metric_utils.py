import torch
from functools import partial

import torch

from pytorch_msssim import ms_ssim, ssim
# from lpipsPyTorch import lpips
import lpips


def psnr_func(img1, img2, data_range=1):
    err = torch.mean((img1 - img2) ** 2)
    # err = mean_squared_error(img1, img2)
    return 10 * torch.log10((data_range ** 2) / err)




# def msssim_fn(output_list, target_list):
#     msssim_list = []
#     for output, target in zip(output_list, target_list):
#         if output.size(-2) >= 160:
#             msssim = ms_ssim(output.float().detach(), target.detach(), data_range=1, size_average=True)
#         else:
#             msssim = torch.tensor(0).to(output.device)
#         msssim_list.append(msssim.view(1))
#     msssim = torch.cat(msssim_list, dim=0) #(num_stage)
#     msssim = msssim.view(1, -1).expand(output_list[-1].size(0), -1) #(batchsize, num_stage)
#     return msssim


def msssim_fn(output, target):
    assert output.size(-2) >= 160
    msssim_val = ms_ssim(output.float().detach(), target.detach(), data_range=1, size_average=True)
    return msssim_val




# lpips_fn = partial(lpips, net_type='vgg')
lpips_fn = lpips.LPIPS().cuda()



