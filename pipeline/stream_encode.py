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
import pathlib
import time
from functools import partial

import torch
from loguru import logger



from arguments import ModelParams, PipelineParams, OptimizationParams

from frame_cube.frame_cube import FrameCube
from ortho_gaussian_renderer import render, network_gui

from scene import GaussianModel
# from scene.gaussian_model import BitInfo
from utils.codec_utils import encode_gaussian, stream_encode_gaussian
from utils.encodings import get_binary_vxl_size
# from utils.log_utils import prepare_output_and_logger
from utils.loss_utils import l1_loss_func, ssim_func, calc_optical_loss
from utils.metric_utils import psnr_func
# from lpipsPyTorch import lpips
from utils.report_utils import evaluate, log_training_results, render_frames
from utils.train_util import TrainingController







def training(
        # args_param,
        model_params: ModelParams,
        opt: OptimizationParams,
        pipe: PipelineParams,
        dataset_name,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        debug_from,
        tb_writer,
        wandb=None,
        # logger=None,
        ply_path=None):
    first_iter = 0

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
    # scene = Scene(dataset, gaussians, ply_path=ply_path)
    # gaussians.update_anchor_bound()
    gaussians.update_anchor_bound(frame_cube.dataset.x_min, frame_cube.dataset.y_min, frame_cube.dataset.z_min)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_ckpt, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_ckpt, opt)


    stream_encode_gaussian(frame_cube.gaussians, pipe)

    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    curried_render = partial(
        render,
        pipe=pipe,
        bg_color=background,
    )
    output_frame_dir = pathlib.Path(pipe.model_path) / 'decoded_frames'
    output_frame_dir.mkdir(exist_ok=True)
    render_frames(
        frame_cube,
        curried_render,
        output_frame_dir

    )

    torch.cuda.synchronize()
    t_end = time.time()
    # logger.info("Total Training time: {}".format(t_end-t_start-log_time_sub))

    return gaussians.x_bound_min, gaussians.x_bound_max
