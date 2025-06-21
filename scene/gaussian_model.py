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

import os
import pathlib
import pickle
# import subprocess
# import pickle
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass, asdict
from functools import reduce
from dahuffman import HuffmanCodec
import numpy as np
import torch
from loguru import logger
# from matplotlib import pyplot as plt
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2  # noqa
from torch import nn, Tensor
from torch_scatter import scatter_max

from arguments import ModelParams, PipelineParams
from common.base import RenderResults
from utils.codec_utils import EncodeMeta
from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric)
from utils.graphics_utils import BasicPointCloud
from utils.mask import encode_mask
from utils.param_utils import quantize_tensor
# from utils.inspector import check_tensor
from utils.system_utils import mkdir_p
from utils.entropy_models import EntropyGaussian

from utils.encodings import \
    STE_binary, STE_multistep, Quantize_anchor, \
    GridEncoder, \
    anchor_round_digits, \
    encoder_gaussian, decoder_gaussian, \
    get_binary_vxl_size, UniformQuantizer, encode_anchor, decode_anchor, encode_binary, decode_binary, reorder_and_split
from utils.time_util import get_embedder

bit2MB_scale = 8 * 1024 * 1024


@dataclass
class BitInfo:
    bit_anchor: int
    bit_anchor_gpcc: int
    bit_feat: int
    bit_scaling: int
    bit_offsets: int
    bit_hash: int
    bit_masks: int
    bit_mlp: int
    bit_mlp_encoded: int


@dataclass
class EntropyContext:
    mean_feat: torch.Tensor
    scale_feat: torch.Tensor
    mean_scaling: torch.Tensor
    scale_scaling: torch.Tensor
    mean_offsets: torch.Tensor
    scale_offsets: torch.Tensor
    Q_feat_adj: torch.Tensor
    Q_scaling_adj: torch.Tensor
    Q_offsets_adj: torch.Tensor


class Mix3d2dEncoding(nn.Module):
    def __init__(
            self,
            n_features,
            resolutions_list,
            log2_hashmap_size,
            resolutions_list_2D,
            log2_hashmap_size_2D,
            ste_binary,
            ste_multistep,
            add_noise,
            Q,
    ):
        super().__init__()
        self.encoding_xyz = GridEncoder(
            num_dim=3,
            n_features=n_features,
            resolutions_list=resolutions_list,
            log2_hashmap_size=log2_hashmap_size,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xy = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_yz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.output_dim = self.encoding_xyz.output_dim + \
                          self.encoding_xy.output_dim + \
                          self.encoding_xz.output_dim + \
                          self.encoding_yz.output_dim

    def forward(self, x):
        x_x, y_y, z_z = torch.chunk(x, 3, dim=-1)
        out_xyz = self.encoding_xyz(x)  # [..., 2*16]
        out_xy = self.encoding_xy(torch.cat([x_x, y_y], dim=-1))  # [..., 2*4]
        out_xz = self.encoding_xz(torch.cat([x_x, z_z], dim=-1))  # [..., 2*4]
        out_yz = self.encoding_yz(torch.cat([y_y, z_z], dim=-1))  # [..., 2*4]
        out_i = torch.cat([out_xyz, out_xy, out_xz, out_yz], dim=-1)  # [..., 56]
        return out_i


class FiLM(nn.Module):
    def __init__(self, condition_dim, input_dim):
        super(FiLM, self).__init__()

        # 全连接层，用于生成γ和β参数
        self.fc_gamma0 = nn.Linear(condition_dim, condition_dim)
        self.fc_beta0 = nn.Linear(condition_dim, condition_dim)

        self.fc_gamma1 = nn.Linear(condition_dim, input_dim)
        self.fc_beta1 = nn.Linear(condition_dim, input_dim)
        self.act = nn.ReLU()


    def forward(self, x, condition):
        # 根据条件特征获取缩放scale参数和移位参数shift，即计算γ和β参数
        gamma = self.fc_gamma1(self.act(self.fc_gamma0(condition)))
        beta = self.fc_beta1(self.act(self.fc_beta0(condition)))

        # 对输入特征x进行缩放和偏移，实现条件特征调整输入特征
        y = gamma * x + beta
        return y


class GeneratorNet(nn.Module):
    def __init__(self, input_dim, output_dim, inner_dim, condition_dim, out_act=None):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, inner_dim)
        self.act = nn.GELU()
        self.out_linear = nn.Linear(inner_dim, output_dim)
        self.film = FiLM(condition_dim, inner_dim)
        self.out_act = nn.Identity() if out_act is None else out_act

    def forward(self, feature, condition):

        feature = self.linear1(feature)
        feature = self.act(feature)
        feature = self.linear2(feature)

        affine_feat = self.film(feature, condition)





        out = self.out_linear(affine_feat)
        return self.out_act(out)

class EntropyParamsNet(nn.Module):
    def __init__(self, input_dim, inner_dim, inner_dim2, output_dim, layer=2):
        super().__init__()
        if layer == 2:
            self.dist_net = nn.Sequential(
                nn.Linear(input_dim, inner_dim),
                nn.GELU(),
                # nn.Linear(inner_dim, inner_dim),
                # nn.ReLU(True),
                nn.Linear(inner_dim, output_dim * 2),
            )
        else:
            assert layer == 3
            self.dist_net = nn.Sequential(
                nn.Linear(input_dim, inner_dim),
                nn.GELU(),
                nn.Linear(inner_dim, inner_dim),
                nn.GELU(),
                nn.Linear(inner_dim, output_dim * 2),
            )


        self.quant_step_net = nn.Sequential(
            nn.Linear(input_dim, inner_dim2),
            nn.GELU(),
            nn.Linear(inner_dim2, 1),
        )

    def forward(self, x):
        params = self.dist_net(x)
        feature_dim = params.shape[1] // 2

        mean, scale = torch.split(params, split_size_or_sections=[feature_dim, feature_dim], dim=-1)
        quant_step = self.quant_step_net(x)
        return mean, scale, quant_step



def calc_symbol_min_max(x_mean, Q, bound=15000):
    x_min = x_mean.mean() / Q.mean() - bound
    x_max = x_mean.mean() / Q.mean() + bound
    return int(x_min), int(x_max)


class ResidualLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        assert in_features == out_features
        super().__init__(in_features, out_features, bias,
                 device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        y = super().forward(input)
        return y + input
class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self,
                 model_config: ModelParams,
                 feat_dim: int = 32,
                 n_offsets: int = 5,
                 voxel_size: float = 0.01,
                 update_depth: int = 3,
                 update_init_factor: int = 100,
                 update_hierachy_factor: int = 4,
                 use_feat_bank = False,
                 n_features_per_level: int = 2,
                 log2_hashmap_size: int = 19,
                 log2_hashmap_size_2D: int = 17,
                 resolutions_list = (18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
                 resolutions_list_2D = (130, 258, 514, 1026),
                 ste_binary: bool = True,
                 ste_multistep: bool = False,
                 add_noise: bool = False,
                 Q=1,
                 use_2D: bool = True,
                 decoded_version: bool = False,
                 ):
        super().__init__()
        logger.info(
            'hash_params: ' +
            str((
                use_2D,
                n_features_per_level,
                log2_hashmap_size,
                resolutions_list,
                log2_hashmap_size_2D,
                resolutions_list_2D,
                ste_binary,
                ste_multistep,
                add_noise
            ))
        )

        self.model_config = model_config

        # feat_dim = 50
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.log2_hashmap_size_2D = log2_hashmap_size_2D
        self.resolutions_list = resolutions_list
        self.resolutions_list_2D = resolutions_list_2D
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q
        self.use_2D = use_2D
        self.decoded_version = decoded_version

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._mask = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if use_2D:
            self.encoding_xyz = Mix3d2dEncoding(
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                resolutions_list_2D=resolutions_list_2D,
                log2_hashmap_size_2D=log2_hashmap_size_2D,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()
        else:
            self.encoding_xyz = GridEncoder(
                num_dim=3,
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()

        encoding_params_num = 0
        for n, p in self.encoding_xyz.named_parameters():
            encoding_params_num += p.numel()
        encoding_MB = encoding_params_num / 8 / 1024 / 1024
        if not ste_binary: encoding_MB *= 32
        logger.info(f'encoding_param_num={encoding_params_num}, size={encoding_MB}MB.')

        self.embed_time_fn, time_input_ch = get_embedder(self.model_config.time_multi_res, 1)
        self.embed_fn, z_input_ch = get_embedder(self.model_config.offset_multi_res, 1)

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.GELU(),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        # mlp_input_feat_dim = feat_dim + 3 + 1
        mlp_input_feat_dim = feat_dim  + 2

        mlp_input_feat_dim = feat_dim + time_input_ch + z_input_ch

        input_dim = feat_dim
        inner_dim = feat_dim * 2
        condition_dim = time_input_ch + z_input_ch

        # self.mlp_opacity = nn.Sequential(
        #     nn.Linear(mlp_input_feat_dim, feat_dim),
        #     nn.ReLU(True),
        #     nn.Linear(feat_dim, feat_dim),
        #     nn.ReLU(True),
        #     nn.Linear(feat_dim, n_offsets),
        #     nn.Tanh()
        # ).cuda()

        self.mlp_opacity = GeneratorNet(
            input_dim=input_dim, output_dim=n_offsets, inner_dim=inner_dim, condition_dim=condition_dim,
            out_act=nn.Tanh()
        ).cuda()

        # self.mlp_cov = nn.Sequential(
        #     nn.Linear(mlp_input_feat_dim, feat_dim),
        #     nn.ReLU(True),
        #     nn.Linear(feat_dim, feat_dim),
        #     nn.ReLU(True),
        #     nn.Linear(feat_dim, 7*self.n_offsets),
        #     # nn.Linear(feat_dim, 7),
        # ).cuda()

        self.mlp_cov = GeneratorNet(
            input_dim=input_dim, output_dim=7 * self.n_offsets, inner_dim=inner_dim, condition_dim=condition_dim
        ).cuda()

        # self.mlp_color = nn.Sequential(
        #     nn.Linear(mlp_input_feat_dim, feat_dim),
        #     nn.ReLU(True),
        #     nn.Linear(feat_dim, feat_dim),
        #     nn.ReLU(True),
        #     nn.Linear(feat_dim, 3*self.n_offsets),
        #     nn.Sigmoid()
        # ).cuda()

        self.mlp_color = GeneratorNet(
            input_dim=input_dim, output_dim=3 * self.n_offsets, inner_dim=inner_dim, condition_dim=condition_dim,
            out_act=nn.Sigmoid()
        ).cuda()

        # self.mlp_deform = nn.Sequential(
        #     nn.Linear(mlp_input_feat_dim, feat_dim * 2),
        #     nn.GELU(),
        #     nn.Linear(feat_dim * 2, feat_dim * 2),
        #     nn.GELU(),
        #     nn.Linear(feat_dim * 2, feat_dim * 2),
        #     nn.GELU(),
        #     # nn.Linear(feat_dim, feat_dim),
        #     # nn.ReLU(True),
        #     # nn.Linear(feat_dim, feat_dim),
        #     # nn.ReLU(True),
        #
        #     # ResidualLinear(feat_dim * 2, feat_dim* 2),
        #     # nn.ReLU(True),
        #     # ResidualLinear(feat_dim* 2, feat_dim* 2),
        #     # nn.ReLU(True),
        #     # ResidualLinear(feat_dim* 2, feat_dim* 2),
        #     # nn.ReLU(True),
        #     nn.Linear(feat_dim * 2, 3 * self.n_offsets),
        # ).cuda()

        # self.mlp_deform = GeneratorNet(
        #     input_dim=input_dim, output_dim=3 * self.n_offsets, inner_dim=feat_dim * 2, condition_dim=condition_dim,
        # ).cuda()

        self.mlp_deform = nn.Sequential(
            nn.Linear(mlp_input_feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim * 2),
            nn.GELU(),
            # nn.Linear(feat_dim, feat_dim),
            # nn.ReLU(True),
            # nn.Linear(feat_dim, feat_dim),
            # nn.ReLU(True),

            # ResidualLinear(feat_dim * 2, feat_dim* 2),
            # nn.ReLU(True),
            # ResidualLinear(feat_dim* 2, feat_dim* 2),
            # nn.ReLU(True),
            # ResidualLinear(feat_dim* 2, feat_dim* 2),
            # nn.ReLU(True),
            nn.Linear(feat_dim * 2, 3 * self.n_offsets),
        ).cuda()

        self.mlp_feature_enet = EntropyParamsNet(
            input_dim=self.encoding_xyz.output_dim, inner_dim=feat_dim * 3, inner_dim2=feat_dim, output_dim=feat_dim
        ).cuda()

        self.mlp_scaling_enet = EntropyParamsNet(
            input_dim=self.encoding_xyz.output_dim, inner_dim=feat_dim * 2, inner_dim2=feat_dim, output_dim=6, layer=3
        ).cuda()

        self.mlp_offset_enet = EntropyParamsNet(
            input_dim=self.encoding_xyz.output_dim, inner_dim=feat_dim * 3, inner_dim2=feat_dim, output_dim=3 * self.n_offsets
        ).cuda()

        self.noise_quantizer = UniformQuantizer()

        self.entropy_gaussian = EntropyGaussian(Q=1).cuda()

    def get_encoding_params(self):
        params = []
        if self.use_2D:
            params.append(self.encoding_xyz.encoding_xyz.params)
            params.append(self.encoding_xyz.encoding_xy.params)
            params.append(self.encoding_xyz.encoding_xz.params)
            params.append(self.encoding_xyz.encoding_yz.params)
        else:
            params.append(self.encoding_xyz.params)
        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params

    def get_mlp_size(self, digit=32):
        mlp_size = 0
        for n, p in self.named_parameters():
            if 'mlp' in n:
                mlp_size += p.numel() * digit
        return mlp_size, mlp_size / 8 / 1024 / 1024

    def eval(self):
        # self.mlp_opacity.eval()
        # self.mlp_cov.eval()
        # self.mlp_color.eval() # 影响eval和train
        # self.encoding_xyz.eval()
        # self.mlp_grid.eval()
        # self.mlp_deform.eval()
        super().eval()

        assert self.use_feat_bank is False
        # if self.use_feat_bank:
        #     self.mlp_feature_bank.eval()

    def train(self, mode: bool = True):
        # self.mlp_opacity.train()
        # self.mlp_cov.train()
        # self.mlp_color.train()
        # self.encoding_xyz.train()
        # self.mlp_grid.train()
        # self.mlp_deform.train()

        super().train(mode)

        assert self.use_feat_bank is False

        # if self.use_feat_bank:
        #     self.mlp_feature_bank.train()

    def capture(self):
        state_dict = self.state_dict()

        return (
            state_dict,
            self.x_bound_min,
            self.x_bound_max,
            # self._anchor,
            # self._anchor_feat,
            # self._offset,
            # self._mask,
            # self._scaling,
            # self._rotation,
            # self._opacity,
            self.max_radii2D,
            self.offset_denom,
            self.anchor_demon,
            # self.mlp_opacity.state_dict(),
            # self.mlp_cov.state_dict(),
            # self.mlp_color.state_dict(),
            # self.encoding_xyz.state_dict(),
            # self.mlp_grid.state_dict(),
            # self.mlp_deform.state_dict(),

            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            state_dict,
            self.x_bound_min,
            self.x_bound_max,
            # self._anchor,
            # self._anchor_feat,
            # self._offset,
            # self._mask,
            # self._scaling,
            # self._rotation,
            # self._opacity,
            self.max_radii2D,
            offset_denom,
            anchor_demon,
            # mlp_opacity_dict,
            # mlp_cov_dict,
            # mlp_color_dict,
            # encoding_xyz_dict,
            # mlp_grid_dict,
            # mlp_deform_dict,
            opt_dict,
            self.spatial_lr_scale
        ) = model_args
        anchor_num = state_dict['_anchor'].shape[0]
        self.init_anchor_params(anchor_num)

        self.training_setup(training_args)
        self.offset_denom = offset_denom
        self.anchor_demon = anchor_demon
        self.load_state_dict(state_dict)
        # self.mlp_opacity.load_state_dict(mlp_opacity_dict)
        # self.mlp_cov.load_state_dict(mlp_cov_dict)
        # self.mlp_color.load_state_dict(mlp_color_dict)
        # self.encoding_xyz.load_state_dict(encoding_xyz_dict)
        # # self.mlp_grid.load_state_dict(mlp_grid_dict)
        # self.mlp_deform.load_state_dict(mlp_deform_dict)

        # current_state = self.optimizer.state_dict()
        # current_state['state'] = opt_dict['state']
        # old_params_group = opt_dict['param_groups']
        # flatten = {}
        # for entry in old_params_group:
        #     flatten[entry['name']] = entry
        #
        # for i in range(len(current_state['param_groups'])):
        #     entry = current_state['param_groups'][i]
        #     name = entry['name']
        #     try:
        #         old_entry = flatten[name]
        #         current_state['param_groups'][i] = old_entry
        #     except KeyError:
        #         pass
        #
        #
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.decoded_version:
            return self._scaling
        return 1.0 * self.scaling_activation(self._scaling)

    @property
    def get_mask(self):
        if self.decoded_version:
            # print('call decoded ver')
            return self._mask
        # print('call normal ver')
        mask_sig = torch.sigmoid(self._mask)
        return ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig

    @property
    def get_mask_anchor(self):
        with torch.no_grad():
            if self.decoded_version:
                mask_anchor = (torch.sum(self._mask, dim=1)[:, 0]) > 0
                return mask_anchor
            mask_sig = torch.sigmoid(self._mask)
            mask = ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig
            mask_anchor = (torch.sum(mask, dim=1)[:, 0]) > 0
            return mask_anchor

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_grid_mlp(self):
        return self.mlp_grid

    @property
    def get_deform_mlp(self):
        return self.mlp_deform

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        if self.decoded_version:
            return self._anchor
        anchor, quantized_v = Quantize_anchor.apply(self._anchor, self.x_bound_min, self.x_bound_max)
        return anchor

    @property
    def quantized_anchor(self):
        return Quantize_anchor.quantized(self._anchor, self.x_bound_min, self.x_bound_max)

    @torch.no_grad()
    def update_anchor_bound(self, x_lim, y_lim, z_lim, bleed=0.1):
        # x_bound_min = (torch.min(self._anchor, dim=0, keepdim=True)[0]).detach()
        # x_bound_max = (torch.max(self._anchor, dim=0, keepdim=True)[0]).detach()
        # for c in range(x_bound_min.shape[-1]):
        #     x_bound_min[0, c] = x_bound_min[0, c] * 1.2 if x_bound_min[0, c] < 0 else x_bound_min[0, c] * 1.2
        # for c in range(x_bound_max.shape[-1]):
        #     x_bound_max[0, c] = x_bound_max[0, c] * 1.2 if x_bound_max[0, c] > 0 else x_bound_max[0, c] * 1.2
        x_lim = x_lim * (1 + bleed)
        y_lim = y_lim * (1 + bleed)
        z_lim = z_lim * (1 + bleed)

        x_bound_min = torch.Tensor([[x_lim, y_lim, z_lim]]).to(self._anchor.device)
        x_bound_max = torch.Tensor([[-x_lim, -y_lim, -z_lim]]).to(self._anchor.device)
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        logger.info('anchor_bound_updated')
        logger.info(f'x_bound_min {x_bound_min}')
        logger.info(f'x_bound_max {x_bound_max}')

    def calc_interp_feat(self, x):
        # x: [N, 3]
        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]
        features = self.encoding_xyz(x)  # [N, 4*12]
        return features

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        # 一个voxel里只保留一个点，且根据voxel尺寸进行量化
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        return data

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        ratio = 1
        points = pcd.points[::ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            # 调用simple_knn的distCUDA2函数，计算点云中的每个点到与其最近的K个点的平均距离的平方，K如何指定待检查
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        logger.info(f'Initial voxel_size: {self.voxel_size}')

        # 根据voxel size对初始点云进行量化下采样
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()

        # 每个voxel生成n_offset个gaussians
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        # 生成gaussians的mask
        masks = torch.ones((fused_point_cloud.shape[0], self.n_offsets, 1)).float().cuda()
        # voxel 的特征
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        logger.info("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 初始化的voxelized的点云作为锚点
        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._mask = nn.Parameter(masks.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def init_anchor_params(self, anchor_num):

        fused_point_cloud = torch.zeros((anchor_num, 3)).cuda()

        # 每个voxel生成n_offset个gaussians
        offsets = torch.zeros((anchor_num, self.n_offsets, 3)).float().cuda()
        # 生成gaussians的mask
        masks = torch.ones((anchor_num, self.n_offsets, 1)).float().cuda()
        # voxel 的特征
        anchors_feat = torch.zeros((anchor_num, self.feat_dim)).float().cuda()

        logger.info("Number of points at initialisation : ", anchor_num)

        # dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.zeros((anchor_num, 6), device="cuda")

        rots = torch.zeros((anchor_num, 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((anchor_num, 1), dtype=torch.float, device="cuda"))


        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._mask = nn.Parameter(masks.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def register_training_params(self, name, module, lr, scheduler_func=None):
        entry = {
            'params': module if isinstance(module, list) else module.parameters(),
            'lr': lr,
            'name': name,
            # 'scheduler': scheduler_func if scheduler_func is not None else lambda x: lr
        }
        assert name not in self.net_params_registry.keys()
        self.net_params_registry[name] = entry
        self.scheduler_registry[name] = scheduler_func if scheduler_func is not None else lambda x: lr

    def training_setup(self, training_args):
        self.net_params_registry = OrderedDict()
        self.scheduler_registry = OrderedDict()
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        # if self.use_feat_bank:
        #     l = [
        #         {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
        #         {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
        #         {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
        #         {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
        #         {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        #         {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
        #         {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        #
        #         {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
        #         {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
        #         {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
        #         {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
        #
        #         {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
        #         {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},
        #
        #         {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
        #
        #         # {'params': self.conv_super_res.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "super_res"},
        #     ]
        # else:
        # l = [
        #     {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
        #     {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
        #     {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
        #     {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
        #     {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        #     {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
        #     {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        #
        #     # 注意mlp前缀要用于区分是否为高斯点
        #
        #     # {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
        #     # {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
        #     # {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
        #     #
        #     # {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
        #     # {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},
        #     #
        #     # {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
        #
        #     # {'params': self.conv_super_res.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "conv_super_res"},
        # ]



        self.register_training_params(
            name='anchor', module=[self._anchor],
            lr=training_args.position_lr_init * self.spatial_lr_scale,
            scheduler_func=get_expon_lr_func(
                lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                lr_delay_mult=training_args.position_lr_delay_mult,
                max_steps=training_args.position_lr_max_steps
            )
        )

        self.register_training_params(
            name='offset', module=[self._offset],
            lr=training_args.offset_lr_init * self.spatial_lr_scale,
            scheduler_func=get_expon_lr_func(
                lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                lr_delay_mult=training_args.offset_lr_delay_mult,
                max_steps=training_args.offset_lr_max_steps)
        )

        self.register_training_params(
            name='mask', module=[self._mask],
            lr=training_args.mask_lr_init * self.spatial_lr_scale,
            scheduler_func= get_expon_lr_func(
                lr_init=training_args.mask_lr_init*self.spatial_lr_scale,
                lr_final=training_args.mask_lr_final*self.spatial_lr_scale,
                lr_delay_mult=training_args.mask_lr_delay_mult,
                max_steps=training_args.mask_lr_max_steps
            )
        )

        self.register_training_params(
            name='anchor_feat', module=[self._anchor_feat],
            lr= training_args.feature_lr
        )

        self.register_training_params(
            name='opacity', module=[self._opacity],
            lr= training_args.opacity_lr
        )
        self.register_training_params(
            name='scaling', module=[self._scaling],
            lr= training_args.scaling_lr
        )
        self.register_training_params(
            name='rotation', module=[self._rotation],
            lr= training_args.rotation_lr
        )


        self.register_training_params(
            'mlp_opacity', self.mlp_opacity,
            lr=training_args.mlp_opacity_lr_init,
            scheduler_func=get_expon_lr_func(
                lr_init=training_args.mlp_opacity_lr_init,
                lr_final=training_args.mlp_opacity_lr_final,
                lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                max_steps=training_args.mlp_opacity_lr_max_steps
            )
        )

        self.register_training_params(
            'mlp_cov', self.mlp_cov,
            lr=training_args.mlp_cov_lr_init,
            scheduler_func=get_expon_lr_func(
                lr_init=training_args.mlp_cov_lr_init,
                lr_final=training_args.mlp_cov_lr_final,
                lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                max_steps=training_args.mlp_cov_lr_max_steps
            )
        )

        self.register_training_params(
            'mlp_color', self.mlp_color,
            lr=training_args.mlp_color_lr_init,
            scheduler_func=get_expon_lr_func(
                lr_init=training_args.mlp_color_lr_init,
                lr_final=training_args.mlp_color_lr_final,
                lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                max_steps=training_args.mlp_color_lr_max_steps
            )
        )

        self.register_training_params(
            'encoding_xyz', self.encoding_xyz,
            lr=training_args.encoding_xyz_lr_init,
            scheduler_func=get_expon_lr_func(
                lr_init=training_args.encoding_xyz_lr_init,
                lr_final=training_args.encoding_xyz_lr_final,
                lr_delay_mult=training_args.encoding_xyz_lr_delay_mult,
                max_steps=training_args.encoding_xyz_lr_max_steps,
                         step_sub=0 if self.ste_binary else 10000,
            )
        )

        self.register_training_params(
            'mlp_deform', self.mlp_deform,
            lr=training_args.mlp_deform_lr_init,
            scheduler_func=get_expon_lr_func(
                lr_init=training_args.mlp_deform_lr_init,
                lr_final=training_args.mlp_deform_lr_final,
                lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                max_steps=training_args.mlp_deform_lr_max_steps
            )
        )



        # self.register_training_params(
        #     'mlp_grid', self.mlp_grid,
        #     lr=training_args.mlp_grid_lr_init,
        #     scheduler_func=get_expon_lr_func(
        #         lr_init=training_args.encoding_xyz_lr_init,
        #         lr_final=training_args.encoding_xyz_lr_final,
        #         lr_delay_mult=training_args.encoding_xyz_lr_delay_mult,
        #         max_steps=training_args.encoding_xyz_lr_max_steps,
        #                  step_sub=0 if self.ste_binary else 10000,
        #     )
        # )

        self.register_training_params(
            'mlp_feature_enet', self.mlp_feature_enet,
            lr=training_args.mlp_entropy_net_lr_init,
            scheduler_func=get_expon_lr_func(
                lr_init=training_args.mlp_entropy_net_lr_init,
                lr_final=training_args.mlp_entropy_net_lr_final,
                lr_delay_mult=training_args.mlp_entropy_net_lr_delay_mult,
                max_steps=training_args.mlp_entropy_net_lr_max_steps
            )
        )

        self.register_training_params(
            'mlp_scaling_enet', self.mlp_scaling_enet,
            lr=training_args.mlp_entropy_net_lr_init,
            scheduler_func=get_expon_lr_func(
                lr_init=training_args.mlp_entropy_net_lr_init,
                lr_final=training_args.mlp_entropy_net_lr_final,
                lr_delay_mult=training_args.mlp_entropy_net_lr_delay_mult,
                max_steps=training_args.mlp_entropy_net_lr_max_steps
            )
        )

        self.register_training_params(
            'mlp_offset_enet', self.mlp_offset_enet,
            lr=training_args.mlp_entropy_net_lr_init,
            scheduler_func=get_expon_lr_func(
                lr_init=training_args.mlp_entropy_net_lr_init,
                lr_final=training_args.mlp_entropy_net_lr_final,
                lr_delay_mult=training_args.mlp_entropy_net_lr_delay_mult,
                max_steps=training_args.mlp_entropy_net_lr_max_steps
            )
        )

        l = self.net_params_registry.values()
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
        #                                             lr_final=training_args.position_lr_final*self.spatial_lr_scale,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)
        # self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
        #                                             lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
        #                                             lr_delay_mult=training_args.offset_lr_delay_mult,
        #                                             max_steps=training_args.offset_lr_max_steps)
        # self.mask_scheduler_args = get_expon_lr_func(lr_init=training_args.mask_lr_init*self.spatial_lr_scale,
        #                                             lr_final=training_args.mask_lr_final*self.spatial_lr_scale,
        #                                             lr_delay_mult=training_args.mask_lr_delay_mult,
        #                                             max_steps=training_args.mask_lr_max_steps)
        #
        #
        #
        #
        # self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
        #                                             lr_final=training_args.mlp_opacity_lr_final,
        #                                             lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
        #                                             max_steps=training_args.mlp_opacity_lr_max_steps)
        #
        # self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
        #                                             lr_final=training_args.mlp_cov_lr_final,
        #                                             lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
        #                                             max_steps=training_args.mlp_cov_lr_max_steps)
        #
        # self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
        #                                             lr_final=training_args.mlp_color_lr_final,
        #                                             lr_delay_mult=training_args.mlp_color_lr_delay_mult,
        #                                             max_steps=training_args.mlp_color_lr_max_steps)
        # if self.use_feat_bank:
        #     self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
        #                                                 lr_final=training_args.mlp_featurebank_lr_final,
        #                                                 lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
        #                                                 max_steps=training_args.mlp_featurebank_lr_max_steps)
        #
        # self.encoding_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.encoding_xyz_lr_init,
        #                                             lr_final=training_args.encoding_xyz_lr_final,
        #                                             lr_delay_mult=training_args.encoding_xyz_lr_delay_mult,
        #                                             max_steps=training_args.encoding_xyz_lr_max_steps,
        #                                                      step_sub=0 if self.ste_binary else 10000,
        #                                                      )
        # self.mlp_grid_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_grid_lr_init,
        #                                             lr_final=training_args.mlp_grid_lr_final,
        #                                             lr_delay_mult=training_args.mlp_grid_lr_delay_mult,
        #                                             max_steps=training_args.mlp_grid_lr_max_steps,
        #                                                  step_sub=0 if self.ste_binary else 10000,
        #                                                  )
        #
        # self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init,
        #                                             lr_final=training_args.mlp_deform_lr_final,
        #                                             lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
        #                                             max_steps=training_args.mlp_deform_lr_max_steps)

    # def update_learning_rate(self, iteration):
    #     ''' Learning rate scheduling per step '''
    #     for param_group in self.optimizer.param_groups:
    #         if param_group["name"] == "offset":
    #             lr = self.offset_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #         if param_group["name"] == "mask":
    #             lr = self.mask_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #         if param_group["name"] == "anchor":
    #             lr = self.anchor_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #         if param_group["name"] == "mlp_opacity":
    #             lr = self.mlp_opacity_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #         if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
    #             lr = self.mlp_featurebank_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #         if param_group["name"] == "mlp_cov":
    #             lr = self.mlp_cov_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #         if param_group["name"] == "mlp_color":
    #             lr = self.mlp_color_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #         if param_group["name"] == "encoding_xyz":
    #             lr = self.encoding_xyz_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #         if param_group["name"] == "mlp_grid":
    #             lr = self.mlp_grid_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #         if param_group["name"] == "mlp_deform":
    #             lr = self.mlp_deform_scheduler_args(iteration)
    #             param_group['lr'] = lr

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            name = param_group['name']
            lr_scheduler = self.scheduler_registry[name]
            lr = lr_scheduler(iteration)
            param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._mask.shape[1]*self._mask.shape[2]):
            l.append('f_mask_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        mask = self._mask.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, mask, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        mask_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_mask")]
        mask_names = sorted(mask_names, key = lambda x: int(x.split('_')[-1]))
        masks = np.zeros((anchor.shape[0], len(mask_names)))
        for idx, attr_name in enumerate(mask_names):
            masks[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        masks = masks.reshape((masks.shape[0], 1, -1))

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._mask = nn.Parameter(torch.tensor(masks, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:  # Only for opacity, rotation. But seems they two are useless?
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def training_statis(self, render_results: RenderResults): # viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        """
        统计高斯点的梯度等信息
        :param viewspace_point_tensor:
        :param opacity:
        :param update_filter:
        :param offset_selection_mask:
        :param anchor_visible_mask:
        :return:
        """
        viewspace_point_tensor = render_results.viewspace_points
        opacity = render_results.neural_opacity
        update_filter = render_results.visibility_filter
        offset_selection_mask = render_results.selection_mask
        anchor_visible_mask = render_results.visible_mask


        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])

        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]


        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._mask = optimizable_tensors["mask"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):  # 3
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()
                new_masks = torch.ones_like(candidate_anchor[:, 0:1]).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "mask": new_masks,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._mask = optimizable_tensors["mask"]
                self._opacity = optimizable_tensors["opacity"]

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path):
        mkdir_p(os.path.dirname(path))

        if self.use_feat_bank:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'mlp_feature_bank': self.mlp_feature_bank.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'encoding_xyz': self.encoding_xyz.state_dict(),
                # 'grid_mlp': self.mlp_grid.state_dict(),
                'deform_mlp': self.mlp_deform.state_dict(),
            }, path)
        else:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'encoding_xyz': self.encoding_xyz.state_dict(),
                # 'grid_mlp': self.mlp_grid.state_dict(),
                'deform_mlp': self.mlp_deform.state_dict(),
            }, path)


    def load_mlp_checkpoints(self, path):
        checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(checkpoint['mlp_feature_bank'])
        self.encoding_xyz.load_state_dict(checkpoint['encoding_xyz'])
        # self.mlp_grid.load_state_dict(checkpoint['grid_mlp'])
        self.mlp_deform.load_state_dict(checkpoint['deform_mlp'])

    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            mask = mask.unsqueeze(-1) + 0.0
            x_c = (2 - 1 / mag) * (x / mag)
            x = x_c * mask + x * (1 - mask)
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x

    def calc_entropy_context(self, anchor):

        # check_tensor(anchor)

        feat_context = self.calc_interp_feat(anchor)
        # anchor_pe_emb = self.anchor_embed_fn(anchor)

        # feat_context = torch.cat([feat_context, anchor_pe_emb], dim=1)

        mean_feat, scale_feat, Q_feat_adj = self.mlp_feature_enet(feat_context)
        mean_scaling, scale_scaling, Q_scaling_adj = self.mlp_scaling_enet(feat_context)
        mean_offsets, scale_offsets, Q_offsets_adj = self.mlp_offset_enet(feat_context)

        # feat_context = self.get_grid_mlp(feat_context)
        # mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
        #     torch.split(feat_context,
        #                 split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1,
        #                                         1, 1], dim=-1)

        Q_feat_adj = torch.exp(torch.clamp(Q_feat_adj, min=-10, max=10))
        Q_scaling_adj = torch.exp(torch.clamp(Q_scaling_adj, min=-10, max=10))
        Q_offsets_adj = torch.exp(torch.clamp(Q_offsets_adj, min=-10, max=10))

        return EntropyContext(
            mean_feat, torch.clamp(scale_feat, 1e-9),
            mean_scaling, torch.clamp(scale_scaling, 1e-9),
            mean_offsets, torch.clamp(scale_offsets,1e-9),
            Q_feat_adj, Q_scaling_adj, Q_offsets_adj
        )

    @torch.no_grad()
    def estimate_final_bits(self):

        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        mask_anchor = self.get_mask_anchor

        _anchor = self.get_anchor[mask_anchor] #[:100]
        _feat = self._anchor_feat[mask_anchor] #[:100]
        _grid_offsets = self._offset[mask_anchor] #[:100]
        _scaling = self.get_scaling[mask_anchor] #[:100]
        _mask = self.get_mask[mask_anchor] #[:100]
        hash_embeddings = self.get_encoding_params()

        # feat_context = self.calc_interp_feat(_anchor)  # [N_visible_anchor*0.2, 32]
        # mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
        #     torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]
        entropy_context = self.calc_entropy_context(_anchor)

        Q_feat = Q_feat * entropy_context.Q_feat_adj
        Q_scaling = Q_scaling * entropy_context.Q_scaling_adj
        Q_offsets = Q_offsets * entropy_context.Q_offsets_adj

        # print('estimate', _feat[0])

        feat_min, feat_max = calc_symbol_min_max(entropy_context.mean_feat, Q_feat)
        scaling_min, scaling_max = calc_symbol_min_max(entropy_context.mean_scaling, Q_scaling)
        offsets_min, offsets_max = calc_symbol_min_max(entropy_context.mean_offsets, Q_offsets)


        # _feat = (STE_multistep.apply(_feat, Q_feat)).detach()
        # grid_scaling = (STE_multistep.apply(_scaling, Q_scaling)).detach()
        # offsets = (STE_multistep.apply(_grid_offsets, Q_offsets.unsqueeze(1))).detach()

        quantized_feat = STE_multistep.quantize(_feat, Q_feat, feat_min, feat_max).detach()
        quantized_grid_scaling = STE_multistep.quantize(_scaling, Q_scaling, scaling_min, scaling_max).detach()
        quantized_offsets = STE_multistep.quantize(_grid_offsets, Q_offsets.unsqueeze(1), offsets_min, offsets_max).detach()


        quantized_offsets = quantized_offsets.view(-1, 3 * self.n_offsets)

        mask_ratio = _mask.sum() / _mask.shape[0]
        mask_tmp = _mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets)

        bit_feat = self.entropy_gaussian(
            quantized_feat,
            entropy_context.mean_feat,
            entropy_context.scale_feat,
            Q_feat,
            quantized=True
        )

        # check_tensor(_feat[:100])
        # check_tensor(entropy_context.mean_feat[:100])
        # check_tensor(entropy_context.scale_feat[:100])
        # check_tensor(Q_feat[:100])

        bit_scaling = self.entropy_gaussian(
            quantized_grid_scaling,
            entropy_context.mean_scaling,
            entropy_context.scale_scaling,
            Q_scaling,
            quantized=True
        )

        # check_tensor(grid_scaling[:100])
        # check_tensor(entropy_context.mean_scaling[:100])
        # check_tensor(entropy_context.scale_scaling[:100])
        # check_tensor(Q_scaling[:100])


        bit_offsets = self.entropy_gaussian(
            quantized_offsets,
            entropy_context.mean_offsets,
            entropy_context.scale_offsets,
            Q_offsets,
            quantized=True
        )
        bit_offsets = bit_offsets * mask_tmp


        # print('bit_feat', bit_feat.view(-1)[:5000].sum())
        # print('bit_scaling', bit_scaling.view(-1)[:600].sum())
        # print('bit_offsets', bit_offsets.view(-1)[:10].sum())

        # print('bit_feat', bit_feat.view(-1).sum())
        # print('bit_scaling', bit_scaling.view(-1).sum())
        # print('bit_offsets', bit_offsets.view(-1).sum())

        bit_anchor = _anchor.shape[0] * 3 * anchor_round_digits
        bit_feat = torch.sum(bit_feat).item()
        bit_scaling = torch.sum(bit_scaling).item()
        bit_offsets = torch.sum(bit_offsets).item()
        if self.ste_binary:
            bit_hash = get_binary_vxl_size((hash_embeddings+1)/2)[1].item()
        else:
            bit_hash = hash_embeddings.numel()*32
        bit_masks = get_binary_vxl_size(_mask)[1].item()

        # logger.info(str((bit_anchor, bit_feat, bit_scaling, bit_offsets, bit_hash, bit_masks)))

        estimated_mlp_bit = self.get_mlp_size()[0]
        bit_info = BitInfo(
            bit_anchor,
            bit_anchor / 2,
            bit_feat,
            bit_scaling,
            bit_offsets,
            bit_hash,
            bit_masks,
            estimated_mlp_bit,
            int(estimated_mlp_bit * 0.3),
        )

        log_info = f"Estimated sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
                   f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + self.get_mlp_size()[0])/bit2MB_scale, 4)}"

        return log_info, bit_info

    def quantize_model(self, replace=True):

        cur_ckt = self.state_dict()

        valid_mask_list = []
        quant_weight_list = []  #
        meta_info_list = []
        for k, v in cur_ckt.items():
            if not k.startswith('mlp'):
                continue
            logger.info(f'{k}, {v.shape}')

            # if k.endswith('_mask'):  # ignore prune mask
            #     continue
            # elif k.endswith('_orig'):
            #     valid_mask_key = k.replace('_orig', '_mask')
            #     valid_mask = cur_ckt[valid_mask_key]
            #     v = v * valid_mask
            large_tf = (v.dim() in {2, 4} and 'bias' not in k)
            quant_t, valid_mask, new_t, meta_info = quantize_tensor(v, 8,
                                                                    0 if large_tf else -1)
            logger.info(f'mask {valid_mask.sum()}')
            valid_quant_v = quant_t[valid_mask]  # only include non-zero weights
            cur_ckt[k] = new_t

            quant_weight_list.append(valid_quant_v.flatten())
            valid_mask_list.append(valid_mask.flatten())
            meta_info['key'] = k
            meta_info['shape'] = [int(i) for i in v.shape]
            meta_info_list.append(meta_info)

        if replace:
            logger.info('replacing mlp params...')
            self.load_state_dict(cur_ckt)

        self.quantized_model_cache = (valid_mask_list, quant_weight_list, meta_info_list)
        return self.quantized_model_cache


    def encode_mlp(self, file_path):
        valid_mask_list, quant_weight_list, meta_info_list = self.quantized_model_cache
        cat_mask = torch.cat(valid_mask_list)
        # mask_bytes = mask_to_bytes([bool(i) for i in cat_mask])

        compressed_mask = encode_mask(cat_mask)
        logger.info(f'mask length: {len(compressed_mask)}B')

        cat_param = torch.cat(quant_weight_list)
        input_code_list = cat_param.tolist()

        # plt.hist(cat_param.detach().flatten().cpu().numpy(), bins=256)
        # plt.show()
        #
        # data = cat_param.detach().flatten().cpu().numpy()
        # from collections import Counter
        # c = Counter(data)
        #
        # total = 0
        # for k, v in c.items():
        #     total += v
        #
        # mapping = {**c}
        # total_entropy = 0
        # min_bit_len = 10000000
        # max_bit_len = 0
        # for k, v in c.items():
        #     p = v / total
        #     bit_len = - np.log2(p)
        #     min_bit_len = min(bit_len, min_bit_len)
        #     max_bit_len = max(bit_len, max_bit_len)
        #     total_entropy += v * bit_len
        #
        # print('best entropy', total_entropy, 'total params', total, min_bit_len, max_bit_len)

        # generating HuffmanCoding table
        codec = HuffmanCodec.from_data(input_code_list)
        encoded_repr = codec.encode([int(i) for i in cat_param])
        logger.info(f'param length: {len(encoded_repr)}')

        code_table = codec.get_code_table()

        meta_info = {
            'code_table': code_table,
            'meta_list': meta_info_list
        }

        encoded_meta = pickle.dumps(meta_info)

        compressed_meta = zlib.compress(encoded_meta, level=9)

        logger.info(f'meta length: {len(compressed_meta)}')



        compressed = {
            # 'model_meta': self.model.config.to_bytes(),
            'meta': compressed_meta,
            'mask': compressed_mask,
            'params': encoded_repr
        }

        with open(file_path, 'wb') as f:
            pickle.dump(compressed, f)

        real_rate_byte = os.path.getsize(file_path)

        logger.info(f'mlp total size: {real_rate_byte * 8 / bit2MB_scale} MB')

        return real_rate_byte * 8

    @torch.no_grad()
    def conduct_encoding(self, pre_path_name, pipe: PipelineParams, replace=True):

        valid_mask_list, quant_weight_list, meta_info_list = self.quantize_model(replace)
        mlp_real_rate_bit = self.encode_mlp(pathlib.Path(pre_path_name) / 'mlp.pkl')

        t_codec = 0

        torch.cuda.synchronize(); t1 = time.time()
        logger.info('Start encoding ...')

        mask_anchor = self.get_mask_anchor

        quantized_anchor, anchor_interval, anchor_min = self.quantized_anchor
        quantized_anchor = quantized_anchor[mask_anchor]

        selection, anchor_gpcc_bit = encode_anchor(quantized_anchor.long().detach().cpu().numpy(), pathlib.Path(pre_path_name), pathlib.Path(pipe.tmc3_executable))
        selection = torch.Tensor(selection).to(quantized_anchor.device).long()
        anchor_infos_list = [anchor_interval, anchor_min]



        _anchor = self.get_anchor[mask_anchor][selection]
        _feat = self._anchor_feat[mask_anchor][selection]
        _grid_offsets = self._offset[mask_anchor][selection]
        _scaling = self.get_scaling[mask_anchor][selection]
        _mask = self.get_mask[mask_anchor][selection]

        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        # feat_context = self.calc_interp_feat(_anchor)  # [N_visible_anchor*0.2, 32]
        # mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
        #     torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]
        ec = self.calc_entropy_context(_anchor)
        Q_feat = Q_feat * ec.Q_feat_adj
        Q_scaling = Q_scaling * ec.Q_scaling_adj
        Q_offsets = Q_offsets * ec.Q_offsets_adj

        feat_min, feat_max = calc_symbol_min_max(ec.mean_feat, Q_feat)
        scaling_min, scaling_max = calc_symbol_min_max(ec.mean_scaling, Q_scaling)
        offsets_min, offsets_max = calc_symbol_min_max(ec.mean_offsets, Q_offsets)
        # Q_feat_mean = Q_feat.mean()
        # Q_scaling_mean = Q_scaling.mean()
        # Q_offsets_mean = Q_offsets.mean()



        N = _anchor.shape[0]
        MAX_batch_size = 1000
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        bit_feat_list = []
        bit_scaling_list = []
        bit_offsets_list = []
        # anchor_infos_list = []
        indices_list = []
        min_feat_list = []
        max_feat_list = []
        min_scaling_list = []
        max_scaling_list = []
        min_offsets_list = []
        max_offsets_list = []

        feat_list = []
        scaling_list = []
        offsets_list = []


        file_list = []



        # torch.save(_anchor, os.path.join(pre_path_name, 'anchor.pkl'))

        for s in range(steps):
            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)

            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            indices = torch.tensor(data=range(N_num), device='cuda', dtype=torch.long)  # [N_num]
            # anchor_infos = None
            # anchor_infos_list.append(anchor_infos)
            indices_list.append(indices+N_start)

            anchor_sort = _anchor[N_start:N_end][indices]  # [N_num, 3]

            # encode feat
            # feat_context = self.calc_interp_feat(anchor_sort)  # [N_num, ?]
            # # many [N_num, ?]
            # mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            #     torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)
            ec = self.calc_entropy_context(anchor_sort)

            # Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1]).view(-1)
            # Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            # Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)



            Q_feat = Q_feat * ec.Q_feat_adj
            Q_scaling = Q_scaling * ec.Q_scaling_adj
            Q_offsets = Q_offsets * ec.Q_offsets_adj

            Q_feat = Q_feat.repeat(1, ec.mean_feat.shape[-1])
            Q_scaling = Q_scaling.repeat(1, ec.mean_scaling.shape[-1])
            Q_offsets = Q_offsets.repeat(1, ec.mean_offsets.shape[-1])


            mean_feat = ec.mean_feat.contiguous()
            mean_scaling = ec.mean_scaling.contiguous()
            mean_offsets = ec.mean_offsets.contiguous()
            scale_feat = ec.scale_feat.contiguous()
            scale_scaling = ec.scale_scaling.contiguous()
            scale_offsets = ec.scale_offsets.contiguous()
            # Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            # Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            # Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            feat = _feat[N_start:N_end][indices]

            # print('encode', feat[0])

            # print('feat shape', feat.shape)

            # feat = feat.view(-1)# [N_num*32]


            # with open('feat.pkl', 'wb') as f:
            #     pack = (
            #         feat, Q_feat, feat_min, feat_max,
            #          mean_feat, scale_feat, Q_feat, feat_min, feat_max
            #     )
            #
            #     pickle.dump(pack, f)

            # feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat

            # feat = STE_multistep.apply(feat, Q_feat, _feat.mean())

            quantized_feat = STE_multistep.quantize(feat, Q_feat, feat_min, feat_max)
            torch.cuda.synchronize(); t0 = time.time()
            # bit_feat, min_feat, max_feat = encoder_gaussian(feat, mean, scale, Q_feat, _feat.mean(), file_name=feat_b_name)

            bit_feat, min_feat, max_feat = encoder_gaussian(quantized_feat, mean_feat, scale_feat, Q_feat, feat_min, feat_max, file_name=feat_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            file_list.append(feat_b_name)
            bit_feat_list.append(bit_feat)
            min_feat_list.append(min_feat)
            max_feat_list.append(max_feat)
            feat_list.append(feat)

            # print('bit feat', bit_feat)

            scaling = _scaling[N_start:N_end][indices] # .view(-1)  # [N_num*6]

            # print('scaling shape', scaling.shape)

            # scaling = scaling + torch.empty_like(scaling).uniform_(-0.5, 0.5) * Q_scaling

            # with open('scaling.pkl', 'wb') as f:
            #     pack = (
            #         scaling, Q_scaling, scaling_min, scaling_max,
            #         mean_scaling, scale_scaling, Q_scaling, scaling_min, scaling_max
            #     )
            #
            #     pickle.dump(pack, f)

            # scaling = STE_multistep.apply(scaling, Q_scaling, _scaling.mean())
            quantized_scaling = STE_multistep.quantize(scaling, Q_scaling, scaling_min, scaling_max)
            torch.cuda.synchronize(); t0 = time.time()
            # bit_scaling, min_scaling, max_scaling = encoder_gaussian(scaling, mean_scaling, scale_scaling, Q_scaling, _scaling.mean(), file_name=scaling_b_name)
            bit_scaling, min_scaling, max_scaling = encoder_gaussian(
                quantized_scaling, mean_scaling, scale_scaling, Q_scaling, scaling_min, scaling_max, file_name=scaling_b_name
            )
            torch.cuda.synchronize(); t_codec += time.time() - t0
            file_list.append(scaling_b_name)
            bit_scaling_list.append(bit_scaling)
            min_scaling_list.append(min_scaling)
            max_scaling_list.append(max_scaling)
            scaling_list.append(scaling)

            # print('bit scaling', bit_scaling)

            mask = _mask[N_start:N_end][indices]  # {0, 1}  # [N_num, K, 1]
            mask = mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets).to(torch.bool)  # [N_num*K*3]
            offsets = _grid_offsets[N_start:N_end][indices].view(-1, 3*self.n_offsets)  # [N_num*K*3]


            # offsets = offsets + torch.empty_like(offsets).uniform_(-0.5, 0.5) * Q_offsets

            # offsets = STE_multistep.apply(offsets, Q_offsets, _grid_offsets.mean())
            quantized_offsets = STE_multistep.quantize(offsets, Q_offsets, offsets_min, offsets_max)
            offsets[~mask] = 0.0
            torch.cuda.synchronize(); t0 = time.time()
            # bit_offsets, min_offsets, max_offsets = encoder_gaussian(offsets[mask], mean_offsets[mask], scale_offsets[mask], Q_offsets[mask], _grid_offsets.mean(),  file_name=offsets_b_name)
            bit_offsets, min_offsets, max_offsets = encoder_gaussian(quantized_offsets[mask], mean_offsets[mask], scale_offsets[mask], Q_offsets[mask], offsets_min, offsets_max,  file_name=offsets_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            file_list.append(offsets_b_name)
            bit_offsets_list.append(bit_offsets)
            min_offsets_list.append(min_offsets)
            max_offsets_list.append(max_offsets)
            offsets_list.append(offsets)

            # print('bit offsets', bit_offsets)
            #
            # torch.cuda.empty_cache()
            #
            # exit()

        bit_anchor = N * 3 * anchor_round_digits
        bit_feat = sum(bit_feat_list)
        bit_scaling = sum(bit_scaling_list)
        bit_offsets = sum(bit_offsets_list)

        hash_b_name = os.path.join(pre_path_name, 'hash.b')
        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        hash_embeddings = self.get_encoding_params()  # {-1, 1}
        if self.ste_binary:
            p = torch.zeros_like(hash_embeddings).to(torch.float32)
            prob_hash = (((hash_embeddings + 1) / 2).sum() / hash_embeddings.numel()).item()
            p[...] = prob_hash
            bit_hash = encode_binary(hash_embeddings.view(-1), p.view(-1), file_name=hash_b_name)
        else:
            prob_hash = 0
            bit_hash = hash_embeddings.numel()*32


        indices = torch.cat(indices_list, dim=0)
        assert indices.shape[0] == _mask.shape[0]
        mask = _mask[indices]  # {0, 1}
        p = torch.zeros_like(mask).to(torch.float32)
        prob_masks = (mask.sum() / mask.numel()).item()
        p[...] = prob_masks
        bit_masks = encode_binary((mask * 2 - 1).view(-1), p.view(-1), file_name=masks_b_name)

        meta = EncodeMeta(
            model_config=self.model_config,
            total_anchor_num=self._anchor.shape[0],
            anchor_num=N,
            batch_size=MAX_batch_size,
            anchor_infos_list=anchor_infos_list,
            min_feat_list=min_feat_list,
            max_feat_list=max_feat_list,
            min_scaling_list=min_scaling_list,
            max_scaling_list=max_scaling_list,
            min_offsets_list=min_offsets_list,
            max_offsets_list=max_offsets_list
        )

        # combined_bitstream = str(pathlib.Path(pre_path_name, 'combined.b').absolute())
        # result = subprocess.run(['touch', combined_bitstream])
        # for file in file_list:
        #     result = subprocess.run(['cat', file, '>>', combined_bitstream])
        #     assert result.returncode == 0

        encoded_meta = pickle.dumps(asdict(meta))

        compressed_meta = zlib.compress(encoded_meta, level=9)

        meta_bit = len(compressed_meta) * 8

        torch.cuda.synchronize(); t2 = time.time()
        logger.info(f'encoding time: {t2 - t1}')
        logger.info(f'codec time: {t_codec}')

        bit_info = BitInfo(
            bit_anchor,
            anchor_gpcc_bit,
            bit_feat,
            bit_scaling,
            bit_offsets,
            bit_hash,
            bit_masks,
            self.get_mlp_size()[0],
            mlp_real_rate_bit
        )

        logger.info(f"Encoded sizes in MB: ")
        logger.info(f"  meta:    {round(meta_bit / bit2MB_scale, 4)}")
        logger.info(f"  anchor:  {round(bit_anchor/bit2MB_scale, 4)} -> G-PCC {round(anchor_gpcc_bit/bit2MB_scale, 4)}, ")
        logger.info(f"  feat:    {round(bit_feat/bit2MB_scale, 4)}, ")
        logger.info(f"  scaling: {round(bit_scaling/bit2MB_scale, 4)}, ")
        logger.info(f"  offsets: {round(bit_offsets/bit2MB_scale, 4)}, ")
        logger.info(f"  hash:    {round(bit_hash/bit2MB_scale, 4)}, ")
        logger.info(f"  masks:   {round(bit_masks/bit2MB_scale, 4)}, ")
        logger.info(f"  MLPs:    {round(self.get_mlp_size()[0]/bit2MB_scale, 4)} -> encode {round(mlp_real_rate_bit / bit2MB_scale, 4)}")
        logger.info(f"Total {round((meta_bit + anchor_gpcc_bit + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + mlp_real_rate_bit)/bit2MB_scale, 4)}, ")
        logger.info(f"EncTime {round(t2 - t1, 4)}")


        return meta, prob_hash, prob_masks, bit_info


    @torch.no_grad()
    def conduct_decoding(self, pre_path_name, meta: EncodeMeta, prob_hash, prob_masks, tmc3_path):
        torch.cuda.synchronize(); t1 = time.time()
        logger.info('Start decoding ...')
        # [N_full, N, MAX_batch_size, anchor_infos_list, min_feat_list, max_feat_list, min_scaling_list, max_scaling_list, min_offsets_list, max_offsets_list, prob_hash, prob_masks] = patched_infos




        N_full = meta.total_anchor_num
        N = meta.anchor_num
        MAX_batch_size = meta.batch_size
        anchor_infos_list = meta.anchor_infos_list
        min_feat_list = meta.min_feat_list
        max_feat_list = meta.max_feat_list
        min_scaling_list = meta.min_scaling_list
        max_scaling_list = meta.max_scaling_list
        min_offsets_list = meta.min_offsets_list
        max_offsets_list = meta.max_offsets_list

        q_anchor = decode_anchor(pre_path_name, tmc3_path)
        q_anchor = torch.Tensor(q_anchor).cuda()
        anchor_decoded = Quantize_anchor.dequantized(q_anchor, anchor_infos_list[0], anchor_infos_list[1])
        # anchor_decoded = torch.Tensor(anchor_decoded).cuda()


        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        feat_decoded_list = []
        scaling_decoded_list = []
        offsets_decoded_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')
        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        p = torch.zeros(size=[N, self.n_offsets, 1], device='cuda').to(torch.float32)
        p[...] = prob_masks
        masks_decoded = decode_binary(p.view(-1), masks_b_name)  # {-1, 1}
        masks_decoded = (masks_decoded + 1) / 2  # {0, 1}
        masks_decoded = masks_decoded.view(-1, self.n_offsets, 1)

        assert self.ste_binary
        p = torch.zeros_like(self.get_encoding_params()).to(torch.float32)
        p[...] = prob_hash
        hash_embeddings = decode_binary(p.view(-1), hash_b_name)  # {-1, 1}
        hash_embeddings = hash_embeddings.view(-1, self.n_features_per_level)

        Q_feat_list = []
        Q_scaling_list = []
        Q_offsets_list = []

        # anchor_decoded = torch.load(os.path.join(pre_path_name, 'anchor.pkl')).cuda()

        for s in range(steps):
            min_feat = min_feat_list[s]
            max_feat = max_feat_list[s]
            min_scaling = min_scaling_list[s]
            max_scaling = max_scaling_list[s]
            min_offsets = min_offsets_list[s]
            max_offsets = max_offsets_list[s]

            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)
            # sizes of MLPs is not included here
            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            # # encode feat
            # feat_context = self.calc_interp_feat(anchor_decoded[N_start:N_end])  # [N_num, ?]
            # # many [N_num, ?]
            # mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            #     torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)
            #
            ec = self.calc_entropy_context(anchor_decoded[N_start:N_end])
            Q_feat_list.append(Q_feat * ec.Q_feat_adj.contiguous())
            Q_scaling_list.append(Q_scaling * ec.Q_scaling_adj.contiguous())
            Q_offsets_list.append(Q_offsets * ec.Q_offsets_adj.contiguous())

            Q_feat_adj = ec.Q_feat_adj.contiguous().repeat(1, ec.mean_feat.shape[-1]).view(-1)
            Q_scaling_adj = ec.Q_scaling_adj.contiguous().repeat(1, ec.mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = ec.Q_offsets_adj.contiguous().repeat(1, ec.mean_offsets.shape[-1]).view(-1)
            mean_feat = ec.mean_feat.contiguous().view(-1)
            mean_scaling = ec.mean_scaling.contiguous().view(-1)
            mean_offsets = ec.mean_offsets.contiguous().view(-1)
            # scale_feat = torch.clamp(ec.scale_feat.contiguous().view(-1), min=1e-9)
            # scale_scaling = torch.clamp(ec.scale_scaling.contiguous().view(-1), min=1e-9)
            # scale_offsets = torch.clamp(ec.scale_offsets.contiguous().view(-1), min=1e-9)

            scale_feat = ec.scale_feat.contiguous().view(-1)
            scale_scaling = ec.scale_scaling.contiguous().view(-1)
            scale_offsets = ec.scale_offsets.contiguous().view(-1)

            Q_feat = Q_feat * Q_feat_adj
            Q_scaling = Q_scaling * Q_scaling_adj
            Q_offsets = Q_offsets * Q_offsets_adj

            feat_decoded = decoder_gaussian(mean_feat, scale_feat, Q_feat, file_name=feat_b_name, min_value=min_feat, max_value=max_feat)
            feat_decoded = feat_decoded.view(N_num, self.feat_dim)  # [N_num, 32]
            feat_decoded_list.append(feat_decoded)

            scaling_decoded = decoder_gaussian(mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, min_value=min_scaling, max_value=max_scaling)
            scaling_decoded = scaling_decoded.view(N_num, 6)  # [N_num, 6]
            scaling_decoded_list.append(scaling_decoded)

            masks_tmp = masks_decoded[N_start:N_end].repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)
            offsets_decoded_tmp = decoder_gaussian(mean_offsets[masks_tmp], scale_offsets[masks_tmp], Q_offsets[masks_tmp], file_name=offsets_b_name, min_value=min_offsets, max_value=max_offsets)
            offsets_decoded = torch.zeros_like(mean_offsets)
            offsets_decoded[masks_tmp] = offsets_decoded_tmp
            offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]
            offsets_decoded_list.append(offsets_decoded)

            torch.cuda.empty_cache()

        feat_decoded = torch.cat(feat_decoded_list, dim=0)
        scaling_decoded = torch.cat(scaling_decoded_list, dim=0)
        offsets_decoded = torch.cat(offsets_decoded_list, dim=0)

        torch.cuda.synchronize(); t2 = time.time()
        logger.info('decoding time:', t2 - t1)

        # fill back N_full
        _anchor = torch.zeros(size=[N_full, 3], device='cuda')
        _anchor_feat = torch.zeros(size=[N_full, self.feat_dim], device='cuda')
        _offset = torch.zeros(size=[N_full, self.n_offsets, 3], device='cuda')
        _scaling = torch.zeros(size=[N_full, 6], device='cuda')
        _mask = torch.zeros(size=[N_full, self.n_offsets, 1], device='cuda')

        _anchor[:N] = anchor_decoded
        _anchor_feat[:N] = feat_decoded
        _offset[:N] = offsets_decoded
        _scaling[:N] = scaling_decoded
        _mask[:N] = masks_decoded

        logger.info('Start replacing parameters with decoded ones...')
        # replace attributes by decoded ones
        assert self._anchor_feat.shape == _anchor_feat.shape
        self._anchor_feat = nn.Parameter(_anchor_feat)
        assert self._offset.shape == _offset.shape
        self._offset = nn.Parameter(_offset)
        # If change the following attributes, decoded_version must be set True
        self.decoded_version = True
        assert self.get_anchor.shape == _anchor.shape
        self._anchor = nn.Parameter(_anchor)
        assert self._scaling.shape == _scaling.shape
        self._scaling = nn.Parameter(_scaling)
        assert self._mask.shape == _mask.shape
        self._mask = nn.Parameter(_mask)

        if self.ste_binary:
            if self.use_2D:
                len_3D = self.encoding_xyz.encoding_xyz.params.shape[0]
                len_2D = self.encoding_xyz.encoding_xy.params.shape[0]
                # print(len_3D, len_2D, hash_embeddings.shape)
                self.encoding_xyz.encoding_xyz.params = nn.Parameter(hash_embeddings[0:len_3D])
                self.encoding_xyz.encoding_xy.params = nn.Parameter(hash_embeddings[len_3D:len_3D+len_2D])
                self.encoding_xyz.encoding_xz.params = nn.Parameter(hash_embeddings[len_3D+len_2D:len_3D+len_2D*2])
                self.encoding_xyz.encoding_yz.params = nn.Parameter(hash_embeddings[len_3D+len_2D*2:len_3D+len_2D*3])
            else:
                self.encoding_xyz.params = nn.Parameter(hash_embeddings)

        logger.info('Parameters are successfully replaced by decoded ones!')

        log_info = f"DecTime {round(t2 - t1, 4)}"

        return log_info

    @torch.no_grad()
    def conduct_stream_encoding(self, pre_path_name, pipe: PipelineParams, replace=True):

        valid_mask_list, quant_weight_list, meta_info_list = self.quantize_model(replace)
        mlp_real_rate_bit = self.encode_mlp(pathlib.Path(pre_path_name) / 'mlp.pkl')

        t_codec = 0

        torch.cuda.synchronize();
        t1 = time.time()
        logger.info('Start encoding ...')

        mask_anchor = self.get_mask_anchor

        quantized_anchor, anchor_interval, anchor_min = self.quantized_anchor
        quantized_anchor = quantized_anchor[mask_anchor]

        selection, anchor_gpcc_bit = encode_anchor(quantized_anchor.long().detach().cpu().numpy(),
                                                   pathlib.Path(pre_path_name), pathlib.Path(pipe.tmc3_executable))
        selection = torch.Tensor(selection).to(quantized_anchor.device).long()
        anchor_infos_list = [anchor_interval, anchor_min]

        _anchor = self.get_anchor[mask_anchor][selection]
        _feat = self._anchor_feat[mask_anchor][selection]
        _grid_offsets = self._offset[mask_anchor][selection]
        _scaling = self.get_scaling[mask_anchor][selection]
        _mask = self.get_mask[mask_anchor][selection]

        # 根据z顺序重新排列所有的属性
        z_order, index_splits = reorder_and_split(_anchor)

        _anchor = _anchor[z_order]
        _feat = _feat[z_order]
        _grid_offsets = _grid_offsets[z_order]
        _scaling = _scaling[z_order]
        _mask = _mask[z_order]


        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        ec = self.calc_entropy_context(_anchor)
        Q_feat = Q_feat * ec.Q_feat_adj
        Q_scaling = Q_scaling * ec.Q_scaling_adj
        Q_offsets = Q_offsets * ec.Q_offsets_adj

        feat_min, feat_max = calc_symbol_min_max(ec.mean_feat, Q_feat)
        scaling_min, scaling_max = calc_symbol_min_max(ec.mean_scaling, Q_scaling)
        offsets_min, offsets_max = calc_symbol_min_max(ec.mean_offsets, Q_offsets)
        # Q_feat_mean = Q_feat.mean()
        # Q_scaling_mean = Q_scaling.mean()
        # Q_offsets_mean = Q_offsets.mean()

        N = _anchor.shape[0]
        MAX_batch_size = 1000
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        bit_feat_list = []
        bit_scaling_list = []
        bit_offsets_list = []
        # anchor_infos_list = []
        indices_list = []
        min_feat_list = []
        max_feat_list = []
        min_scaling_list = []
        max_scaling_list = []
        min_offsets_list = []
        max_offsets_list = []

        feat_list = []
        scaling_list = []
        offsets_list = []

        file_list = []

        # torch.save(_anchor, os.path.join(pre_path_name, 'anchor.pkl'))

        for s, (N_start, N_end) in enumerate(index_splits):
            N_num = N_end - N_start


            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            indices = torch.tensor(data=range(N_num), device='cuda', dtype=torch.long)  # [N_num]
            # anchor_infos = None
            # anchor_infos_list.append(anchor_infos)
            indices_list.append(indices + N_start)

            anchor_sort = _anchor[N_start:N_end][indices]  # [N_num, 3]

            # encode feat
            # feat_context = self.calc_interp_feat(anchor_sort)  # [N_num, ?]
            # # many [N_num, ?]
            # mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            #     torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)
            ec = self.calc_entropy_context(anchor_sort)

            # Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1]).view(-1)
            # Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            # Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)

            Q_feat = Q_feat * ec.Q_feat_adj
            Q_scaling = Q_scaling * ec.Q_scaling_adj
            Q_offsets = Q_offsets * ec.Q_offsets_adj

            Q_feat = Q_feat.repeat(1, ec.mean_feat.shape[-1])
            Q_scaling = Q_scaling.repeat(1, ec.mean_scaling.shape[-1])
            Q_offsets = Q_offsets.repeat(1, ec.mean_offsets.shape[-1])

            mean_feat = ec.mean_feat.contiguous()
            mean_scaling = ec.mean_scaling.contiguous()
            mean_offsets = ec.mean_offsets.contiguous()
            scale_feat = ec.scale_feat.contiguous()
            scale_scaling = ec.scale_scaling.contiguous()
            scale_offsets = ec.scale_offsets.contiguous()
            # Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            # Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            # Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            feat = _feat[N_start:N_end][indices]

            # print('encode', feat[0])

            # print('feat shape', feat.shape)

            # feat = feat.view(-1)# [N_num*32]

            # with open('feat.pkl', 'wb') as f:
            #     pack = (
            #         feat, Q_feat, feat_min, feat_max,
            #          mean_feat, scale_feat, Q_feat, feat_min, feat_max
            #     )
            #
            #     pickle.dump(pack, f)

            # feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat

            # feat = STE_multistep.apply(feat, Q_feat, _feat.mean())

            quantized_feat = STE_multistep.quantize(feat, Q_feat, feat_min, feat_max)
            torch.cuda.synchronize();
            t0 = time.time()
            # bit_feat, min_feat, max_feat = encoder_gaussian(feat, mean, scale, Q_feat, _feat.mean(), file_name=feat_b_name)

            bit_feat, min_feat, max_feat = encoder_gaussian(quantized_feat, mean_feat, scale_feat, Q_feat, feat_min,
                                                            feat_max, file_name=feat_b_name)
            torch.cuda.synchronize();
            t_codec += time.time() - t0
            file_list.append(feat_b_name)
            bit_feat_list.append(bit_feat)
            min_feat_list.append(min_feat)
            max_feat_list.append(max_feat)
            feat_list.append(feat)

            # print('bit feat', bit_feat)

            scaling = _scaling[N_start:N_end][indices]  # .view(-1)  # [N_num*6]

            # print('scaling shape', scaling.shape)

            # scaling = scaling + torch.empty_like(scaling).uniform_(-0.5, 0.5) * Q_scaling

            # with open('scaling.pkl', 'wb') as f:
            #     pack = (
            #         scaling, Q_scaling, scaling_min, scaling_max,
            #         mean_scaling, scale_scaling, Q_scaling, scaling_min, scaling_max
            #     )
            #
            #     pickle.dump(pack, f)

            # scaling = STE_multistep.apply(scaling, Q_scaling, _scaling.mean())
            quantized_scaling = STE_multistep.quantize(scaling, Q_scaling, scaling_min, scaling_max)
            torch.cuda.synchronize();
            t0 = time.time()
            # bit_scaling, min_scaling, max_scaling = encoder_gaussian(scaling, mean_scaling, scale_scaling, Q_scaling, _scaling.mean(), file_name=scaling_b_name)
            bit_scaling, min_scaling, max_scaling = encoder_gaussian(
                quantized_scaling, mean_scaling, scale_scaling, Q_scaling, scaling_min, scaling_max,
                file_name=scaling_b_name
            )
            torch.cuda.synchronize();
            t_codec += time.time() - t0
            file_list.append(scaling_b_name)
            bit_scaling_list.append(bit_scaling)
            min_scaling_list.append(min_scaling)
            max_scaling_list.append(max_scaling)
            scaling_list.append(scaling)

            # print('bit scaling', bit_scaling)

            mask = _mask[N_start:N_end][indices]  # {0, 1}  # [N_num, K, 1]
            mask = mask.repeat(1, 1, 3).view(-1, 3 * self.n_offsets).to(torch.bool)  # [N_num*K*3]
            offsets = _grid_offsets[N_start:N_end][indices].view(-1, 3 * self.n_offsets)  # [N_num*K*3]

            # offsets = offsets + torch.empty_like(offsets).uniform_(-0.5, 0.5) * Q_offsets

            # offsets = STE_multistep.apply(offsets, Q_offsets, _grid_offsets.mean())
            quantized_offsets = STE_multistep.quantize(offsets, Q_offsets, offsets_min, offsets_max)
            offsets[~mask] = 0.0
            torch.cuda.synchronize();
            t0 = time.time()
            # bit_offsets, min_offsets, max_offsets = encoder_gaussian(offsets[mask], mean_offsets[mask], scale_offsets[mask], Q_offsets[mask], _grid_offsets.mean(),  file_name=offsets_b_name)
            bit_offsets, min_offsets, max_offsets = encoder_gaussian(quantized_offsets[mask], mean_offsets[mask],
                                                                     scale_offsets[mask], Q_offsets[mask], offsets_min,
                                                                     offsets_max, file_name=offsets_b_name)
            torch.cuda.synchronize();
            t_codec += time.time() - t0
            file_list.append(offsets_b_name)
            bit_offsets_list.append(bit_offsets)
            min_offsets_list.append(min_offsets)
            max_offsets_list.append(max_offsets)
            offsets_list.append(offsets)

            # print('bit offsets', bit_offsets)
            #
            # torch.cuda.empty_cache()
            #
            # exit()

        bit_anchor = N * 3 * anchor_round_digits
        bit_feat = sum(bit_feat_list)
        bit_scaling = sum(bit_scaling_list)
        bit_offsets = sum(bit_offsets_list)

        hash_b_name = os.path.join(pre_path_name, 'hash.b')
        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        hash_embeddings = self.get_encoding_params()  # {-1, 1}
        if self.ste_binary:
            p = torch.zeros_like(hash_embeddings).to(torch.float32)
            prob_hash = (((hash_embeddings + 1) / 2).sum() / hash_embeddings.numel()).item()
            p[...] = prob_hash
            bit_hash = encode_binary(hash_embeddings.view(-1), p.view(-1), file_name=hash_b_name)
        else:
            prob_hash = 0
            bit_hash = hash_embeddings.numel() * 32

        indices = torch.cat(indices_list, dim=0)
        assert indices.shape[0] == _mask.shape[0]
        mask = _mask[indices]  # {0, 1}
        p = torch.zeros_like(mask).to(torch.float32)
        prob_masks = (mask.sum() / mask.numel()).item()
        p[...] = prob_masks
        bit_masks = encode_binary((mask * 2 - 1).view(-1), p.view(-1), file_name=masks_b_name)

        meta = EncodeMeta(
            model_config=self.model_config,
            total_anchor_num=self._anchor.shape[0],
            anchor_num=N,
            batch_size=MAX_batch_size,
            anchor_infos_list=anchor_infos_list,
            min_feat_list=min_feat_list,
            max_feat_list=max_feat_list,
            min_scaling_list=min_scaling_list,
            max_scaling_list=max_scaling_list,
            min_offsets_list=min_offsets_list,
            max_offsets_list=max_offsets_list
        )

        # combined_bitstream = str(pathlib.Path(pre_path_name, 'combined.b').absolute())
        # result = subprocess.run(['touch', combined_bitstream])
        # for file in file_list:
        #     result = subprocess.run(['cat', file, '>>', combined_bitstream])
        #     assert result.returncode == 0

        encoded_meta = pickle.dumps(asdict(meta))

        compressed_meta = zlib.compress(encoded_meta, level=9)

        meta_bit = len(compressed_meta) * 8

        torch.cuda.synchronize();
        t2 = time.time()
        logger.info(f'encoding time: {t2 - t1}')
        logger.info(f'codec time: {t_codec}')

        bit_info = BitInfo(
            bit_anchor,
            anchor_gpcc_bit,
            bit_feat,
            bit_scaling,
            bit_offsets,
            bit_hash,
            bit_masks,
            self.get_mlp_size()[0],
            mlp_real_rate_bit
        )

        logger.info(f"Encoded sizes in MB: ")
        logger.info(f"  meta:    {round(meta_bit / bit2MB_scale, 4)}")
        logger.info(
            f"  anchor:  {round(bit_anchor / bit2MB_scale, 4)} -> G-PCC {round(anchor_gpcc_bit / bit2MB_scale, 4)}, ")
        logger.info(f"  feat:    {round(bit_feat / bit2MB_scale, 4)}, ")
        logger.info(f"  scaling: {round(bit_scaling / bit2MB_scale, 4)}, ")
        logger.info(f"  offsets: {round(bit_offsets / bit2MB_scale, 4)}, ")
        logger.info(f"  hash:    {round(bit_hash / bit2MB_scale, 4)}, ")
        logger.info(f"  masks:   {round(bit_masks / bit2MB_scale, 4)}, ")
        logger.info(
            f"  MLPs:    {round(self.get_mlp_size()[0] / bit2MB_scale, 4)} -> encode {round(mlp_real_rate_bit / bit2MB_scale, 4)}")
        logger.info(
            f"Total {round((meta_bit + anchor_gpcc_bit + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + mlp_real_rate_bit) / bit2MB_scale, 4)}, ")
        logger.info(f"EncTime {round(t2 - t1, 4)}")

        return meta, prob_hash, prob_masks, bit_info

    @torch.no_grad()
    def conduct_stream_decoding(self, pre_path_name, meta: EncodeMeta, prob_hash, prob_masks, tmc3_path):
        torch.cuda.synchronize()
        t1 = time.time()
        logger.info('Start stream decoding ...')
        # [N_full, N, MAX_batch_size, anchor_infos_list, min_feat_list, max_feat_list, min_scaling_list, max_scaling_list, min_offsets_list, max_offsets_list, prob_hash, prob_masks] = patched_infos

        N_full = meta.total_anchor_num
        N = meta.anchor_num
        MAX_batch_size = meta.batch_size
        anchor_infos_list = meta.anchor_infos_list
        min_feat_list = meta.min_feat_list
        max_feat_list = meta.max_feat_list
        min_scaling_list = meta.min_scaling_list
        max_scaling_list = meta.max_scaling_list
        min_offsets_list = meta.min_offsets_list
        max_offsets_list = meta.max_offsets_list

        q_anchor = decode_anchor(pre_path_name, tmc3_path)
        q_anchor = torch.Tensor(q_anchor).cuda()
        anchor_decoded = Quantize_anchor.dequantized(q_anchor, anchor_infos_list[0], anchor_infos_list[1])
        # anchor_decoded = torch.Tensor(anchor_decoded).cuda()

        # steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        # 根据z顺序重新排列所有的属性
        z_order, index_splits = reorder_and_split(anchor_decoded)
        anchor_decoded = anchor_decoded[z_order]

        feat_decoded_list = []
        scaling_decoded_list = []
        offsets_decoded_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')
        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        p = torch.zeros(size=[N, self.n_offsets, 1], device='cuda').to(torch.float32)
        p[...] = prob_masks
        masks_decoded = decode_binary(p.view(-1), masks_b_name)  # {-1, 1}
        masks_decoded = (masks_decoded + 1) / 2  # {0, 1}
        masks_decoded = masks_decoded.view(-1, self.n_offsets, 1)

        assert self.ste_binary
        p = torch.zeros_like(self.get_encoding_params()).to(torch.float32)
        p[...] = prob_hash
        hash_embeddings = decode_binary(p.view(-1), hash_b_name)  # {-1, 1}
        hash_embeddings = hash_embeddings.view(-1, self.n_features_per_level)

        Q_feat_list = []
        Q_scaling_list = []
        Q_offsets_list = []

        # anchor_decoded = torch.load(os.path.join(pre_path_name, 'anchor.pkl')).cuda()

        for s, (N_start, N_end) in enumerate(index_splits):
            min_feat = min_feat_list[s]
            max_feat = max_feat_list[s]
            min_scaling = min_scaling_list[s]
            max_scaling = max_scaling_list[s]
            min_offsets = min_offsets_list[s]
            max_offsets = max_offsets_list[s]

            N_num = N_end - N_start
            # N_start = s * MAX_batch_size
            # N_end = min((s + 1) * MAX_batch_size, N)
            # sizes of MLPs is not included here
            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            # # encode feat
            # feat_context = self.calc_interp_feat(anchor_decoded[N_start:N_end])  # [N_num, ?]
            # # many [N_num, ?]
            # mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            #     torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)
            #
            ec = self.calc_entropy_context(anchor_decoded[N_start:N_end])
            Q_feat_list.append(Q_feat * ec.Q_feat_adj.contiguous())
            Q_scaling_list.append(Q_scaling * ec.Q_scaling_adj.contiguous())
            Q_offsets_list.append(Q_offsets * ec.Q_offsets_adj.contiguous())

            Q_feat_adj = ec.Q_feat_adj.contiguous().repeat(1, ec.mean_feat.shape[-1]).view(-1)
            Q_scaling_adj = ec.Q_scaling_adj.contiguous().repeat(1, ec.mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = ec.Q_offsets_adj.contiguous().repeat(1, ec.mean_offsets.shape[-1]).view(-1)
            mean_feat = ec.mean_feat.contiguous().view(-1)
            mean_scaling = ec.mean_scaling.contiguous().view(-1)
            mean_offsets = ec.mean_offsets.contiguous().view(-1)
            # scale_feat = torch.clamp(ec.scale_feat.contiguous().view(-1), min=1e-9)
            # scale_scaling = torch.clamp(ec.scale_scaling.contiguous().view(-1), min=1e-9)
            # scale_offsets = torch.clamp(ec.scale_offsets.contiguous().view(-1), min=1e-9)

            scale_feat = ec.scale_feat.contiguous().view(-1)
            scale_scaling = ec.scale_scaling.contiguous().view(-1)
            scale_offsets = ec.scale_offsets.contiguous().view(-1)

            Q_feat = Q_feat * Q_feat_adj
            Q_scaling = Q_scaling * Q_scaling_adj
            Q_offsets = Q_offsets * Q_offsets_adj

            feat_decoded = decoder_gaussian(mean_feat, scale_feat, Q_feat, file_name=feat_b_name, min_value=min_feat,
                                            max_value=max_feat)
            feat_decoded = feat_decoded.view(N_num, self.feat_dim)  # [N_num, 32]
            feat_decoded_list.append(feat_decoded)

            scaling_decoded = decoder_gaussian(mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name,
                                               min_value=min_scaling, max_value=max_scaling)
            scaling_decoded = scaling_decoded.view(N_num, 6)  # [N_num, 6]
            scaling_decoded_list.append(scaling_decoded)

            masks_tmp = masks_decoded[N_start:N_end].repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(
                torch.bool)
            offsets_decoded_tmp = decoder_gaussian(mean_offsets[masks_tmp], scale_offsets[masks_tmp],
                                                   Q_offsets[masks_tmp], file_name=offsets_b_name,
                                                   min_value=min_offsets, max_value=max_offsets)
            offsets_decoded = torch.zeros_like(mean_offsets)
            offsets_decoded[masks_tmp] = offsets_decoded_tmp
            offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]
            offsets_decoded_list.append(offsets_decoded)

            torch.cuda.empty_cache()

        feat_decoded = torch.cat(feat_decoded_list, dim=0)
        scaling_decoded = torch.cat(scaling_decoded_list, dim=0)
        offsets_decoded = torch.cat(offsets_decoded_list, dim=0)
        # N = feat_decoded.shape[0]
        torch.cuda.synchronize();
        t2 = time.time()
        logger.info('decoding time:', t2 - t1)
        logger.info(f'N_full {N_full}, N {N}')

        # fill back N_full
        _anchor = torch.zeros(size=[N_full, 3], device='cuda')
        _anchor_feat = torch.zeros(size=[N_full, self.feat_dim], device='cuda')
        _offset = torch.zeros(size=[N_full, self.n_offsets, 3], device='cuda')
        _scaling = torch.zeros(size=[N_full, 6], device='cuda')
        _mask = torch.zeros(size=[N_full, self.n_offsets, 1], device='cuda')

        _anchor[:N] = anchor_decoded
        _anchor_feat[:N] = feat_decoded
        _offset[:N] = offsets_decoded
        _scaling[:N] = scaling_decoded
        _mask[:N] = masks_decoded

        logger.info('Start replacing parameters with decoded ones...')
        # replace attributes by decoded ones
        assert self._anchor_feat.shape == _anchor_feat.shape
        self._anchor_feat = nn.Parameter(_anchor_feat)
        assert self._offset.shape == _offset.shape
        self._offset = nn.Parameter(_offset)
        # If change the following attributes, decoded_version must be set True
        self.decoded_version = True
        assert self.get_anchor.shape == _anchor.shape
        self._anchor = nn.Parameter(_anchor)
        assert self._scaling.shape == _scaling.shape
        self._scaling = nn.Parameter(_scaling)
        assert self._mask.shape == _mask.shape
        self._mask = nn.Parameter(_mask)

        if self.ste_binary:
            if self.use_2D:
                len_3D = self.encoding_xyz.encoding_xyz.params.shape[0]
                len_2D = self.encoding_xyz.encoding_xy.params.shape[0]
                # print(len_3D, len_2D, hash_embeddings.shape)
                self.encoding_xyz.encoding_xyz.params = nn.Parameter(hash_embeddings[0:len_3D])
                self.encoding_xyz.encoding_xy.params = nn.Parameter(hash_embeddings[len_3D:len_3D + len_2D])
                self.encoding_xyz.encoding_xz.params = nn.Parameter(
                    hash_embeddings[len_3D + len_2D:len_3D + len_2D * 2])
                self.encoding_xyz.encoding_yz.params = nn.Parameter(
                    hash_embeddings[len_3D + len_2D * 2:len_3D + len_2D * 3])
            else:
                self.encoding_xyz.params = nn.Parameter(hash_embeddings)

        logger.info('Parameters are successfully replaced by decoded ones!')

        log_info = f"DecTime {round(t2 - t1, 4)}"

        return log_info