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
# from os import makedirs
import pathlib
import shutil
# import torchvision
# import json
# import wandb
import time
from functools import partial
# from pathlib import Path
# from PIL import Image
# import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
from random import randint

# torch.set_num_threads(32)
# lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
import matplotlib
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from loguru import logger
from skimage.metrics import peak_signal_noise_ratio
from torchvision.utils import flow_to_image
# from utils.general_utils import safe_state
# import uuid
from tqdm import tqdm

# from utils.image_utils import psnr
# from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
# from frame_cube.frame import Frame
from frame_cube.frame_cube import FrameCube
from ortho_gaussian_renderer import render, network_gui, GenerateMode
# import sys
from scene import GaussianModel
# from scene.gaussian_model import BitInfo
from utils.codec_utils import encode_gaussian
from utils.encodings import get_binary_vxl_size
# from utils.log_utils import prepare_output_and_logger
from utils.loss_utils import calc_optical_loss_one_frame
# from lpipsPyTorch import lpips
from utils.report_utils import evaluate, log_training_results
from utils.train_util import TrainingController

BIT2MB_SCALE = 8 * 1024 * 1024



def psnr_func2(img1, img2):
    mse = torch.mean((img1.detach() - img2.detach()) ** 2)  # [B, T]
    psnr = -10 * torch.log10(mse)
    psnr = torch.clamp(psnr, min=0, max=50)
    # psnr = psnr[frame_mask]
    # psnr = torch.mean(psnr)
    return psnr


def calc_optical_loss_eval(
        render_results1_f, render_results1_b,
        render_results2_f, render_results2_b,
        optical_flow,
        x_min, y_min, scale,
        x_pix_max: int, y_pix_max: int,
        n_offsets=10
):
    loss_f, pix_xy, pix_gaussian_uv = calc_optical_loss_one_frame(render_results1_f, render_results2_f, optical_flow,
        x_min, y_min, scale, x_pix_max, y_pix_max, n_offsets)

    t = pix_gaussian_uv.unsqueeze(-1).unsqueeze(-1)
    uv_rgb = flow_to_image(t).squeeze(-1).squeeze(-1).detach().cpu().numpy()

    # white_board = torch.ones((optical_flow.shape[-2], optical_flow.shape[-1], 3)).long().to(pix_xy.device) * 255
    #
    # white_board[pix_xy[:, 1], pix_xy[:, 0], :] = uv_rgb
    #
    # plt.imshow(white_board.detach().cpu().numpy())
    # plt.show()
    # for xy, rgb in zip(pix_xy, uv_rgb):
    #     plt.scatter([xy[0].item()], [xy[1].item()], c=[rgb / 255])
    # plt.show()

    plt.scatter(
        pix_xy[:, 0].cpu().numpy(),
        pix_xy[:, 1].cpu().numpy(),
        c=uv_rgb / 255,
        s=0.1
    )
    plt.show()






    loss_b, _, _ = calc_optical_loss_one_frame(render_results1_b, render_results2_b, optical_flow,
        x_min, y_min, scale, x_pix_max, y_pix_max, n_offsets)

    # return loss_f + loss_b



def eval_model(
        model_params: ModelParams,
        opt: OptimizationParams,
        pipe: PipelineParams,
        checkpoint
):

    gaussians = GaussianModel(
        model_params,
        model_params.anchor_feature_dim,
        model_params.n_offsets,
        model_params.voxel_size,
        model_params.update_depth,
        model_params.update_init_factor,
        model_params.update_hierarchy_factor,
        model_params.use_feat_bank,
        # n_features_per_level=args_param.n_features,
        # log2_hashmap_size=args_param.log2,
        # log2_hashmap_size_2D=args_param.log2_2D,
        n_features_per_level=model_params.grid_feature_dim,
        log2_hashmap_size=model_params.log2,
        log2_hashmap_size_2D=model_params.log2_2D,
    )

    frame_cube = FrameCube(opt, pipe, gaussians)

    gaussians.update_anchor_bound(frame_cube.dataset.x_min, frame_cube.dataset.y_min, frame_cube.dataset.z_min)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_ckpt, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_ckpt, opt)

    # iter_start = torch.cuda.Event(enable_timing=True)
    # iter_end = torch.cuda.Event(enable_timing=True)
    #
    # ema_loss_for_log = 0.0
    # progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    # first_iter += 1
    # torch.cuda.synchronize();
    # t_start = time.time()
    # log_time_sub = 0

    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    retain_grad = False
    render_mode = GenerateMode.TRAINING_FULL_PRECISION

    for frame_idx in tqdm(range(frame_cube.dataset.len_z_frames - 1), disable=True):
        frame1 = frame_cube.dataset[frame_idx]
        frame2 = frame_cube.dataset[frame_idx + 1]
        optical_flow = frame_cube.dataset.get_optical_flow(frame_idx)

        render_results1_f = render(
            frame1, gaussians, pipe, background,
            retain_grad=retain_grad,
            mode=render_mode
        )

        frame1.view_matrix, frame1.view_matrix_s = frame1.view_matrix_s.cuda(), frame1.view_matrix.cuda()
        render_results1_b = render(
            frame1, gaussians, pipe, background,
            retain_grad=retain_grad,
            mode=render_mode
        )

        xyz = render_results1_f.generated_gaussians.xyz
        xyz = xyz[render_results1_f.radii > 0]
        xy = xyz[:, :2].detach().cpu().numpy()

        # xy = xy[xy[:, 0] > -1.1]
        # xy = xy[xy[:, 1] > -0.6]
        plt.scatter(xy[:, 0], xy[:, 1], s=0.1)
        plt.show()



        image1_f = render_results1_f.rendered_image
        image1_b = render_results1_b.rendered_image
        image1_bf = torch.flip(image1_b, dims=(-1,))
        image1 = (image1_f + image1_bf) / 2


        render_results2_f = render(
            frame2, gaussians, pipe, background,
            retain_grad=retain_grad,
            mode=render_mode
        )
        frame2.view_matrix, frame2.view_matrix_s = frame2.view_matrix_s.cuda(), frame2.view_matrix.cuda()
        render_results2_b = render(
            frame2, gaussians, pipe, background,
            retain_grad=retain_grad,
            mode=render_mode
        )

        image2_f = render_results2_f.rendered_image
        image2_b = render_results2_b.rendered_image
        image2_bf = torch.flip(image2_b, dims=(-1,))

        image2 = (image2_f + image2_bf) / 2

        gt_image1 = frame1.image.cuda().permute(0, 2, 1)
        gt_image2 = frame2.image.cuda().permute(0, 2, 1)


        optical_loss = calc_optical_loss_eval(
            render_results1_f, render_results1_b,
            render_results2_f, render_results2_b,
            optical_flow,
            frame_cube.dataset.x_min, frame_cube.dataset.y_min, frame_cube.dataset.scale,
            frame_cube.dataset.width, frame_cube.dataset.height, gaussians.n_offsets
        )
        break

