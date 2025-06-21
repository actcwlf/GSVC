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
import os.path
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from einops import repeat

import math
from py_module.cuda_ortho_gaussian_rasterizer import GaussianRasterizationSettings, GaussianRasterizer

from frame_cube.frame import Frame
from scene.gaussian_model import GaussianModel
from utils.encodings import STE_binary, STE_multistep
from py_module.gaussian_rasterizer import GaussianRasterizer as PyGaussianRasterizer
from utils.inspector import check_tensor


def prefilter_voxel(
        frame: Frame,
        pc: GaussianModel,
        pipe,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True,
                                          device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # visible_mask = (pc.get_anchor[:, 2] - frame.z).abs() < pc.model_config.threshold
    #
    # return visible_mask.detach()

    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(frame.image_height),
        image_width=int(frame.image_width),
        # tanfovx=tanfovx,
        # tanfovy=tanfovy,
        x_min=frame.x_min,
        y_min=frame.y_min,
        scale=frame.scale,
        threshold=pc.model_config.threshold,
        kernel_size=pc.model_config.kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # viewmatrix=viewpoint_camera.world_view_transform,
        # projmatrix=viewpoint_camera.full_proj_transform,
        viewmatrix=frame.view_matrix.permute(1, 0).cuda(),
        # viewmatrix=frame.view_matrix.cuda(),
        # sh_degree=pipe.dr,
        sh_degree=pc.model_config.sh_degree,
        campos=frame.cam_pos,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    py_rasterizer = PyGaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor

    # print('mean3D max', means3D.max(), 'min', means3D.min())

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:  # False
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:  # into here
        scales = pc.get_scaling  # requires_grad = True
        rotations = pc.get_rotation  # requires_grad = True

    radii_pure = rasterizer.visible_filter(
        means3D=means3D,
        scales=scales[:, :3],
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,  # None
    )



    raster_mask = radii_pure > 0
    # anchor = pc.get_anchor
    # delta_z = anchor - frame.cam_pos.to(anchor.device)
    # delta_z = delta_z[:, 2]
    # delta_z_mask = delta_z.abs() <= pc.model_config.threshold
    #
    # flag = delta_z_mask[raster_mask]
    #
    # check_tensor(flag)

    return raster_mask
