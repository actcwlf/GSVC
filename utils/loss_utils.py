#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from functools import lru_cache

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable
from math import exp


def l1_loss_func(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss_func(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


@lru_cache(maxsize=1)
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim_func(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def calc_optical_loss_one_frame(
        render_results1,
        render_results2,
        optical_flow,
        x_min, y_min, scale,
        x_pix_max: int, y_pix_max: int,
        n_offsets=10
):
    indices = torch.arange(0, render_results1.visible_mask.shape[0] * n_offsets, device=render_results1.visible_mask.device)

    part1 = rearrange(indices, '(n k) -> n k', n=render_results1.visible_mask.shape[0], k=n_offsets)
    part1 = part1[render_results1.visible_mask].reshape(-1)
    part1 = part1[render_results1.generated_gaussians.mask]
    mask1 = torch.zeros_like(indices).bool()
    mask1[part1] = True

    #
    part2 = rearrange(indices, '(n k) -> n k', n=render_results1.visible_mask.shape[0], k=n_offsets)
    part2 = part2[render_results2.visible_mask].reshape(-1)
    part2 = part2[render_results2.generated_gaussians.mask]
    mask2 = torch.zeros_like(indices).bool()
    mask2[part2] = True

    common_mask = torch.logical_and(mask1, mask2) # 公共anchor点

    reshaped_common_mask = common_mask.reshape(-1, n_offsets)

    #
    mask1 = reshaped_common_mask[render_results1.visible_mask].reshape(-1)
    mask1 = mask1 * render_results1.generated_gaussians.mask

    concat_part1 = render_results1.generated_gaussians.concatenated_all[mask1]
    scaling_repeat, repeat_anchor, _, _, offsets = concat_part1.split([6, 3, 3, 7, 3], dim=-1)
    offsets = offsets * scaling_repeat[:, :3]  # [N_opacity_pos_gaussian, 3]
    xyz1 = repeat_anchor + offsets  # [N_opacity_pos_gaussian, 3]
    xy1 = xyz1[:, :2]

    mask2 = reshaped_common_mask[render_results2.visible_mask].reshape(-1)
    mask2 = mask2 * render_results2.generated_gaussians.mask
    concat_part2 = render_results2.generated_gaussians.concatenated_all[mask2]
    scaling_repeat, repeat_anchor, _, _, offsets = concat_part2.split([6, 3, 3, 7, 3], dim=-1)
    offsets = offsets * scaling_repeat[:, :3]  # [N_opacity_pos_gaussian, 3]
    xyz2 = repeat_anchor + offsets  # [N_opacity_pos_gaussian, 3]
    xy2 = xyz2[:, :2]

    #
    pix_xy = xy1 - torch.Tensor([[x_min, y_min]]).to(xy1.device)
    pix_xy = (pix_xy * scale).round().long()

    mask1 = torch.logical_and(pix_xy[:, 0] >= 0, pix_xy[:, 1] >= 0)
    mask2 = torch.logical_and(pix_xy[:, 0] < x_pix_max, pix_xy[:, 1] < y_pix_max)
    mask = torch.logical_and(mask1, mask2)

    pix_xy = pix_xy[mask]

    #
    optical_flow = optical_flow.permute(2, 1, 0).to(xy1.device)
    uv = optical_flow[pix_xy[:, 0], pix_xy[:, 1], ...] / scale

    loss = ((xy2[mask] - xy1[mask]) - uv).abs().mean()

    pix_gaussian_uv = (xy2[mask] - xy1[mask]) * scale
    return loss, pix_xy, pix_gaussian_uv


def calc_optical_loss(
        render_results1_f, render_results1_b,
        render_results2_f, render_results2_b,
        optical_flow,
        x_min, y_min, scale,
        x_pix_max: int, y_pix_max: int,
        n_offsets=10
):
    loss_f, _, _ = calc_optical_loss_one_frame(render_results1_f, render_results2_f, optical_flow,
        x_min, y_min, scale, x_pix_max, y_pix_max, n_offsets)

    loss_b, _, _ = calc_optical_loss_one_frame(render_results1_b, render_results2_b, optical_flow,
        x_min, y_min, scale, x_pix_max, y_pix_max, n_offsets)

    return loss_f + loss_b