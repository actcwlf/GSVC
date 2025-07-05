import os.path
import pickle
import time
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from einops import repeat

# import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from frame_cube.frame import Frame
from scene.gaussian_model import GaussianModel
from utils.encodings import STE_binary, STE_multistep
# from py_module.gaussian_rasterizer import GaussianRasterizer as PyGaussianRasterizer


class GenerateMode(Enum):
    TRAINING_FULL_PRECISION = 0
    TRAINING_QUANTIZED = 1
    TRAINING_ENTROPY = 2
    TRAININ_STE_ENTROPY = 3
    DECODING_AS_IS = 4






@dataclass
class RatePack:
    bit_per_param: torch.Tensor = None
    bit_per_feat_param: torch.Tensor = None
    bit_per_scaling_param: torch.Tensor = None
    bit_per_offsets_param: torch.Tensor = None


@dataclass
class GeneratedGaussians:
    xyz: torch.Tensor
    color: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rot: torch.Tensor
    neural_opacity: torch.Tensor = None
    visable_mask: torch.Tensor = None
    mask: torch.Tensor = None
    bit_per_param: torch.Tensor = None
    bit_per_feat_param: torch.Tensor = None
    bit_per_scaling_param: torch.Tensor = None
    bit_per_offsets_param: torch.Tensor = None
    concatenated_all: torch.Tensor = None
    time_sub: float = None


# def calc_entropy_context(pc: GaussianModel, anchor):
#
#
#     feat_context = pc.calc_interp_feat(anchor)
#     feat_context = pc.get_grid_mlp(feat_context)
#     mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
#         torch.split(feat_context,
#                     split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3 * pc.n_offsets, 3 * pc.n_offsets, 1,
#                                             1, 1], dim=-1)
#
#     return EntropyContext(
#         mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj
#     )

def calc_sampled_rate(
        pc: GaussianModel,
        visible_mask,
        feat, grid_scaling, grid_offsets,
        Q_feat, Q_scaling, Q_offsets,
        entropy_context
):
    # l, ref_bit = pc.estimate_final_bits()
    #
    # mask_anchor = pc.get_mask_anchor.detach()
    #
    # _anchor = pc.get_anchor[mask_anchor].detach()  # [:100]
    # _feat = pc._anchor_feat[mask_anchor].detach()  # [:100]
    # t_est_bit_per_feat = ref_bit.bit_feat / _feat.numel()



    anchor = pc.get_anchor[visible_mask]

    mask_anchor = pc.get_mask_anchor[visible_mask]
    mask_anchor_bool = mask_anchor.to(torch.bool)
    mask_anchor_rate = (mask_anchor.sum() / mask_anchor.numel()).detach()

    binary_grid_masks = pc.get_mask[visible_mask]  # differentiable mask


    choose_idx = torch.rand_like(anchor[:, 0]) <= 0.05
    choose_idx = choose_idx & mask_anchor_bool
    feat_chosen = feat[choose_idx]
    grid_scaling_chosen = grid_scaling[choose_idx]
    grid_offsets_chosen = grid_offsets[choose_idx].view(-1, 3 * pc.n_offsets)
    mean = entropy_context.mean_feat[choose_idx]
    scale = entropy_context.scale_feat[choose_idx]
    mean_scaling = entropy_context.mean_scaling[choose_idx]
    scale_scaling = entropy_context.scale_scaling[choose_idx]
    mean_offsets = entropy_context.mean_offsets[choose_idx]
    scale_offsets = entropy_context.scale_offsets[choose_idx]
    Q_feat = Q_feat[choose_idx]
    Q_scaling = Q_scaling[choose_idx]
    Q_offsets = Q_offsets[choose_idx]
    binary_grid_masks_chosen = binary_grid_masks[choose_idx].repeat(1, 1, 3).view(-1, 3 * pc.n_offsets)
    bit_feat = pc.entropy_gaussian(feat_chosen, mean, scale, Q_feat, pc._anchor_feat.mean())
    bit_scaling = pc.entropy_gaussian(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling, pc.get_scaling.mean())
    bit_offsets = pc.entropy_gaussian(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets, pc._offset.mean())
    bit_offsets = bit_offsets * binary_grid_masks_chosen

    # t_bit_feat = bit_feat.sum()
    bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel() * mask_anchor_rate
    bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel() * mask_anchor_rate
    bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel() * mask_anchor_rate
    bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                    (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel()) * mask_anchor_rate

    return RatePack(
        bit_per_param=bit_per_param,
        bit_per_feat_param=bit_per_feat_param,
        bit_per_scaling_param=bit_per_scaling_param,
        bit_per_offsets_param=bit_per_offsets_param,

    )

def generate_neural_gaussians(
        frame: Frame,
        pc : GaussianModel,
        visible_mask=None,
        mode=GenerateMode.TRAINING_FULL_PRECISION
):
    ## view frustum filtering for acceleration

    time_sub = 0

    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    #
    feat = pc._anchor_feat[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    binary_grid_masks = pc.get_mask[visible_mask]  # differentiable mask
    # mask_anchor = pc.get_mask_anchor[visible_mask]
    # mask_anchor_bool = mask_anchor.to(torch.bool)
    # mask_anchor_rate = (mask_anchor.sum() / mask_anchor.numel()).detach()

    # bit_per_param = None
    # bit_per_feat_param = None
    # bit_per_scaling_param = None
    # bit_per_offsets_param = None

    rate_pack = RatePack()

    Q_feat = 1
    Q_scaling = 0.001
    Q_offsets = 0.2
    if mode == GenerateMode.TRAINING_FULL_PRECISION or mode == GenerateMode.DECODING_AS_IS:
        # 全精度训练和解码后推断都不对feature进行处理
        pass

    elif mode == GenerateMode.TRAINING_QUANTIZED:
        # 一阶段量化，采用均匀分布模拟量化
        feat = pc.noise_quantizer(feat, Q_feat)
        grid_scaling = pc.noise_quantizer(grid_scaling, Q_scaling)
        grid_offsets = pc.noise_quantizer(grid_offsets, Q_offsets)

    elif mode == GenerateMode.TRAINING_ENTROPY:
        # 二阶段量化，量化的同时计算码率

        entropy_context = pc.calc_entropy_context(anchor)

        Q_feat = Q_feat * entropy_context.Q_feat_adj
        Q_scaling = Q_scaling * entropy_context.Q_scaling_adj
        Q_offsets = Q_offsets * entropy_context.Q_offsets_adj

        feat = pc.noise_quantizer(feat, Q_feat)
        grid_scaling = pc.noise_quantizer(grid_scaling, Q_scaling)
        grid_offsets = pc.noise_quantizer(grid_offsets, Q_offsets.unsqueeze(1))

        rate_pack = calc_sampled_rate(
            pc, visible_mask,
            feat, grid_scaling, grid_offsets,
            Q_feat, Q_scaling, Q_offsets,
            entropy_context
        )

    elif mode == GenerateMode.TRAININ_STE_ENTROPY:
        # 编码，对参数执行确定性量化
        torch.cuda.synchronize(); t1 = time.time()

        entropy_context = pc.calc_entropy_context(anchor)

        Q_feat = Q_feat * entropy_context.Q_feat_adj.detach()
        Q_scaling = Q_scaling * entropy_context.Q_scaling_adj.detach()
        Q_offsets = Q_offsets * entropy_context.Q_offsets_adj.detach()  # [N_visible_anchor, 1]

        feat = (STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean())).detach()
        grid_scaling = (STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean())).detach()
        grid_offsets = (STE_multistep.apply(grid_offsets, Q_offsets.unsqueeze(1), pc._offset.mean())).detach()


        torch.cuda.synchronize(); time_sub = time.time() - t1

        rate_pack = calc_sampled_rate(
            pc, visible_mask,
            feat, grid_scaling, grid_offsets,
            Q_feat, Q_scaling, Q_offsets,
            entropy_context
        )

    else:
        raise ValueError(f'Unknown mode {mode}')

    ob_view = anchor - frame.cam_pos.to(anchor.device)
    ob_view = ob_view[:, 2:]
    abs_z = torch.zeros_like(ob_view) + frame.cam_pos[-1].to(anchor.device)

    time_emb = pc.embed_time_fn(abs_z)
    z_emb = pc.embed_fn(ob_view)


    ## view-adaptive feature
    # if False: # pc.use_feat_bank: # disable feature bank completely for now
    #     cat_view = torch.cat([ob_view, ob_dist], dim=1)  # [3+1]
    #
    #     bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)  # [N_visible_anchor, 1, 3]
    #
    #     feat = feat.unsqueeze(dim=-1)  # feat: [N_visible_anchor, 32]
    #     feat = \
    #         feat[:, ::4, :1].repeat([1, 4, 1])*bank_weight[:, :, :1] + \
    #         feat[:, ::2, :1].repeat([1, 2, 1])*bank_weight[:, :, 1:2] + \
    #         feat[:, ::1, :1]*bank_weight[:, :, 2:]
    #     feat = feat.squeeze(dim=-1)  # [N_visible_anchor, 32]

    # cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N_visible_anchor, 32+3+1]
    cat_local_view = torch.cat([feat, time_emb, z_emb], dim=1)  # [N_visible_anchor, 32+3+1]

    pe = torch.cat([time_emb, z_emb], dim=1)  # [N_visible_anchor, 32+3+1]

    neural_opacity = pc.get_opacity_mlp(feat, pe)  # [N_visible_anchor, K]
    neural_opacity = neural_opacity.reshape([-1, 1])  # [N_visible_anchor*K, 1]
    neural_opacity = neural_opacity * binary_grid_masks.view(-1, 1)
    mask = (neural_opacity > 0.0)
    mask = mask.view(-1)  # [N_visible_anchor*K]

    # select opacity
    opacity = neural_opacity[mask]  # [N_opacity_pos_gaussian, 1]

    # get offset's color
    color = pc.get_color_mlp(feat, pe)  # [N_visible_anchor, K*3]
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [N_visible_anchor*K, 3]

    # get offset's cov
    scale_rot = pc.get_cov_mlp(feat, pe)  # [N_visible_anchor, K*7]
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [N_visible_anchor*K, 7]

    neural_offset = pc.get_deform_mlp(cat_local_view)
    neural_offset = neural_offset.reshape([anchor.shape[0] * pc.n_offsets, 3])

    offsets = grid_offsets.view([-1, 3])  # [N_visible_anchor*K, 3]

    offsets = offsets + neural_offset

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # [N_visible_anchor, 6+3]
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  # [N_visible_anchor*K, 6+3]
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets],
                                 dim=-1)  # [N_visible_anchor*K, (6+3)+3+7+3]
    masked = concatenated_all[mask]  # [N_opacity_pos_gaussian, (6+3)+3+7+3]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)


    # post-process cov
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(
        scale_rot[:, :3])
    rot = pc.rotation_activation(scale_rot[:, 3:7])  # [N_opacity_pos_gaussian, 4]

    offsets = offsets * scaling_repeat[:, :3]# [N_opacity_pos_gaussian, 3]
    xyz = repeat_anchor + offsets  # [N_opacity_pos_gaussian, 3]
    scaling = scaling

    xyz = torch.clamp(xyz, pc.x_bound_min, pc.x_bound_max)
    gss = GeneratedGaussians(
        xyz=xyz,
        color=color,
        opacity=opacity,
        scaling=scaling,
        rot=rot,
        neural_opacity=neural_opacity,
        visable_mask=visible_mask,
        mask=mask,
        bit_per_param=rate_pack.bit_per_param,
        bit_per_feat_param=rate_pack.bit_per_feat_param,
        bit_per_scaling_param=rate_pack.bit_per_scaling_param,
        bit_per_offsets_param=rate_pack.bit_per_offsets_param,
        concatenated_all=concatenated_all,
        time_sub=time_sub
    )
    return gss

