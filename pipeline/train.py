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
# from utils.general_utils import safe_state
# import uuid
from tqdm import tqdm

# from utils.image_utils import psnr
# from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
# from frame_cube.frame import Frame
from frame_cube.frame_cube import FrameCube
from ortho_gaussian_renderer import render, network_gui
# import sys
from scene import GaussianModel
# from scene.gaussian_model import BitInfo
from utils.codec_utils import encode_gaussian
from utils.encodings import get_binary_vxl_size
# from utils.log_utils import prepare_output_and_logger
from utils.loss_utils import l1_loss_func, ssim_func, calc_optical_loss
from utils.metric_utils import psnr_func
# from lpipsPyTorch import lpips
from utils.report_utils import evaluate, log_training_results
from utils.train_util import TrainingController

# import numpy as np
#
# import subprocess
# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
# os.system('echo $CUDA_VISIBLE_DEVICES')

BIT2MB_SCALE = 8 * 1024 * 1024
# run_codec = True

# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
#     print("found tf board")
# except ImportError:
#     TENSORBOARD_FOUND = False
#     print("not found tf board")


# def save_runtime_code(dst: str) -> None:
#     additional_ignore_patterns = ['.git', '.gitignore']
#     ignore_patterns = set()
#     ROOT = '.'
#     with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
#         for line in gitIgnoreFile:
#             if not line.startswith('#'):
#                 if line.endswith('\n'):
#                     line = line[:-1]
#                 if line.endswith('/'):
#                     line = line[:-1]
#                 ignore_patterns.add(line)
#     ignore_patterns = list(ignore_patterns)
#     for additionalPattern in additional_ignore_patterns:
#         ignore_patterns.append(additionalPattern)
#
#     log_dir = pathlib.Path(__file__).parent.resolve()
#
#     shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignore_patterns))
#
#     print('Backup Finished!')


def show_image(img: torch.Tensor, iteration):
    # plt.figure(dpi=300)
    # plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
    # plt.title(f'iter {iter}')
    # plt.show()

    dpi = matplotlib.rcParams['figure.dpi']

    # Determine the figures size in inches to fit your image
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    height, width, depth = img.shape
    figsize = width / float(dpi), height / float(dpi)

    # plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(f'iter {iteration}')
    plt.show()


def psnr_func2(img1, img2):
    mse = torch.mean((img1.detach() - img2.detach()) ** 2) #[B, T]
    psnr = -10 * torch.log10(mse)
    psnr = torch.clamp(psnr, min=0, max=50)
    # psnr = psnr[frame_mask]
    # psnr = torch.mean(psnr)
    return psnr


def network_gui_render(
    model_params: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    gaussians: GaussianModel,
    background,
    iteration
):
    if network_gui.conn == None:
        network_gui.try_connect()
    while network_gui.conn != None:
        try:
            net_image_bytes = None
            custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
            if custom_cam != None:
                net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                net_image_bytes = memoryview(
                    (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            network_gui.send(net_image_bytes, pipe.source_path)
            if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                break
        except Exception as e:
            network_gui.conn = None


def log_bit_info(tb_writer, gaussians, dataset_name,
                 render_results1_f, render_results1_b,
                 render_results2_f, render_results2_b,
                 iteration):

    assert render_results1_f.entropy_constrained
    assert render_results1_b.entropy_constrained
    assert render_results2_f.entropy_constrained
    assert render_results2_b.entropy_constrained
    bit_per_param = (render_results1_f.bit_per_param + render_results1_b.bit_per_param
                     + render_results2_f.bit_per_param + render_results2_b.bit_per_param)
    bit_per_feat_param = (render_results1_f.bit_per_feat_param + render_results1_b.bit_per_feat_param
                          + render_results2_f.bit_per_feat_param + render_results2_b.bit_per_feat_param)
    bit_per_scaling_param = (render_results1_f.bit_per_scaling_param + render_results1_b.bit_per_scaling_param
                             + render_results2_f.bit_per_scaling_param + render_results2_b.bit_per_scaling_param)
    bit_per_offsets_param = (render_results1_f.bit_per_offsets_param + render_results1_b.bit_per_offsets_param
                             + render_results2_f.bit_per_offsets_param + render_results2_b.bit_per_offsets_param)

    bit_per_param = bit_per_param / 4
    bit_per_feat_param = bit_per_feat_param / 4
    bit_per_scaling_param = bit_per_scaling_param / 4
    bit_per_offsets_param = bit_per_offsets_param / 4

    ttl_size_feat_MB = bit_per_feat_param.item() * gaussians.get_anchor.shape[0] * gaussians.feat_dim / BIT2MB_SCALE
    ttl_size_scaling_MB = bit_per_scaling_param.item() * gaussians.get_anchor.shape[0] * 6 / BIT2MB_SCALE
    ttl_size_offsets_MB = bit_per_offsets_param.item() * gaussians.get_anchor.shape[
        0] * 3 * gaussians.n_offsets / BIT2MB_SCALE
    ttl_size_MB = ttl_size_feat_MB + ttl_size_scaling_MB + ttl_size_offsets_MB

    with torch.no_grad():
        grid_masks = gaussians._mask.data
        binary_grid_masks = (torch.sigmoid(grid_masks) > 0.01).float()
        mask_1_rate, mask_size_bit, mask_size_MB, mask_numel = get_binary_vxl_size(
            binary_grid_masks + 0.0)  # [0, 1] -> [-1, 1]

    order = 1
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_sampled_rate/bit_per_feat_param', bit_per_feat_param.item(), iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_sampled_rate/ttl_size_feat_MB',  ttl_size_feat_MB, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_sampled_rate/bit_per_scaling_param', bit_per_scaling_param.item(), iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_sampled_rate/ttl_size_scaling_MB',  ttl_size_scaling_MB, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_sampled_rate/bit_per_offsets_param', bit_per_offsets_param.item(), iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_sampled_rate/ttl_size_offsets_MB',  ttl_size_offsets_MB, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_sampled_rate/1_rate_mask', mask_1_rate, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_sampled_rate/mask_numel', mask_numel, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_sampled_rate/mask_size_MB', mask_size_MB, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_sampled_rate/bit_per_param', bit_per_param.item(), iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_sampled_rate/ttl_size_MB', ttl_size_MB, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/train_sampled_rate/ttl_size_MB_with_mask', ttl_size_MB + mask_size_MB, iteration)


def log_estimated_bit_info(tb_writer, gaussians, dataset_name, iteration):
    order = 2
    _, bit_info = gaussians.estimate_final_bits()

    tb_writer.add_scalar(f'{dataset_name}/{order}/estimated_rate/feat(MB)', bit_info.bit_feat / BIT2MB_SCALE, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/estimated_rate/scaling(MB)', bit_info.bit_scaling / BIT2MB_SCALE, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/estimated_rate/offsets(MB)', bit_info.bit_offsets / BIT2MB_SCALE, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/estimated_rate/anchor(MB)', bit_info.bit_anchor / BIT2MB_SCALE, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/estimated_rate/masks(MB)', bit_info.bit_masks / BIT2MB_SCALE, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/estimated_rate/hashs(MB)', bit_info.bit_hash / BIT2MB_SCALE, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/estimated_rate/MLP(MB)', bit_info.bit_mlp / BIT2MB_SCALE, iteration)

    total_bit = bit_info.bit_anchor \
                + bit_info.bit_feat \
                + bit_info.bit_scaling \
                + bit_info.bit_offsets \
                + bit_info.bit_hash \
                + bit_info.bit_masks \
                + bit_info.bit_mlp

    tb_writer.add_scalar(f'{dataset_name}/{order}/estimated_rate/total(MB)', total_bit / BIT2MB_SCALE, iteration)


def log_real_bit_info(tb_writer, gaussians, dataset_name, iteration, pipe):
    tmp_path = pipe.model_path
    order = 2
    bit_stream_path = os.path.join(tmp_path, 'tmp_bitstreams')
    os.makedirs(bit_stream_path, exist_ok=True)
    meta, prob_hash, prob_masks, bit_info = gaussians.conduct_encoding(pre_path_name=bit_stream_path, pipe=pipe, replace=False)

    tb_writer.add_scalar(f'{dataset_name}/{order}/real_rate/feat(MB)',    bit_info.bit_feat / BIT2MB_SCALE, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/real_rate/scaling(MB)', bit_info.bit_scaling / BIT2MB_SCALE, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/real_rate/offsets(MB)', bit_info.bit_offsets / BIT2MB_SCALE, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/real_rate/anchor(MB)',  bit_info.bit_anchor / BIT2MB_SCALE, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/real_rate/anchor-gpcc(MB)', bit_info.bit_anchor_gpcc / BIT2MB_SCALE, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/real_rate/masks(MB)',   bit_info.bit_masks / BIT2MB_SCALE, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/real_rate/hashs(MB)',   bit_info.bit_hash / BIT2MB_SCALE , iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/real_rate/MLP(MB)',     bit_info.bit_mlp / BIT2MB_SCALE, iteration)
    tb_writer.add_scalar(f'{dataset_name}/{order}/real_rate/MLP-encoded(MB)', bit_info.bit_mlp_encoded / BIT2MB_SCALE, iteration)

    total_bit = bit_info.bit_anchor \
            + bit_info.bit_feat \
            + bit_info.bit_scaling \
            + bit_info.bit_offsets \
            + bit_info.bit_hash \
            + bit_info.bit_masks \
            + bit_info.bit_mlp

    tb_writer.add_scalar(f'{dataset_name}/{order}/real_rate/total(MB)', total_bit / BIT2MB_SCALE, iteration)

    total_bit2 = bit_info.bit_anchor_gpcc \
                + bit_info.bit_feat \
                + bit_info.bit_scaling \
                + bit_info.bit_offsets \
                + bit_info.bit_hash \
                + bit_info.bit_masks \
                + bit_info.bit_mlp_encoded

    tb_writer.add_scalar(f'{dataset_name}/{order}/real_rate/total-2stage(MB)', total_bit2 / BIT2MB_SCALE, iteration)





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

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    torch.cuda.synchronize(); t_start = time.time()
    log_time_sub = 0

    controller = TrainingController(opt)
    controller.step()
    if first_iter != 1:
        controller.current_iteration = first_iter

    for iteration in range(first_iter, opt.iterations + 1):

        bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # network_gui_render(model_params, opt, pipe, gaussians, background, iteration)

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        frame_idx = randint(0, frame_cube.dataset.len_z_frames - 2)

        # frame_idx = iteration % frame_cube.dataset.len_z_frames
        #
        # frame_idx = randint(0, len(frame_cube.dataset) - 1)
        frame1 = frame_cube.dataset[frame_idx]
        frame2 = frame_cube.dataset[frame_idx+1]
        optical_flow = frame_cube.dataset.get_optical_flow(frame_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_mode = controller.render_mode

        # voxel_visible_mask:bool = radii_pure > 0: 应该是[N_anchor]?
        retain_grad = (opt.update_until > iteration >= 0)

        render_results1_f = render(
            frame1, gaussians, pipe, background,
            retain_grad=retain_grad,
            mode=render_mode
        )
        frame1.view_matrix, frame1.view_matrix_s = frame1.view_matrix_s.cuda(), frame1.view_matrix.cuda()
        # voxel_visible_mask2 = prefilter_voxel(frame, gaussians, pipe, background)
        # # voxel_visible_mask:bool = radii_pure > 0: 应该是[N_anchor]?
        # retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_results1_b = render(
            frame1, gaussians, pipe, background,
            retain_grad=retain_grad,
            mode=render_mode
        )

        image1_f = render_results1_f.rendered_image
        image1_b = render_results1_b.rendered_image
        image1_bf = torch.flip(image1_b, dims=(-1,))

        # img_input = torch.cat([image1, image2_f], dim=0).unsqueeze(0)
        # image = gaussians.conv_super_res(img_input).squeeze(0)
        #
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

        # image = image + image2 / 2
        # if iteration % 100 == 1:
        #     # show_image(image1, iteration)
        #     # show_image(image2, iteration)
        #     show_image(image1, f'idx {frame_idx} iter {iteration}')
        #     show_image(image2, f'idx {frame_idx+1} iter {iteration}')
        #     # exit()

        # if iteration % 2000 == 0 and bit_per_param is not None:
        # if iteration % 2000 == 0 and controller.entropy_constrained:
        #     log_bit_info(gaussians, render_results1, render_results2, iteration, logger)

        gt_image1 = frame1.image.cuda().permute(0, 2, 1)
        gt_image2 = frame2.image.cuda().permute(0, 2, 1)
        # gt_image = torch.flip(gt_image, dims=(1,))
        Ll1 = l1_loss_func(image1, gt_image1) + l1_loss_func(image2, gt_image2)  # + 0.01 * l1_loss_func(image1, image2_f)

        ssim_loss = (1.0 - ssim_func(image1, gt_image1)) + (1.0 - ssim_func(image2, gt_image2))

        # scaling_reg = scaling.prod(dim=1).mean()
        scaling_reg1_f = render_results1_f.scaling.prod(dim=1).mean()
        scaling_reg1_b = render_results1_b.scaling.prod(dim=1).mean()
        scaling_reg2_f = render_results2_f.scaling.prod(dim=1).mean()
        scaling_reg2_b = render_results2_b.scaling.prod(dim=1).mean()
        scaling_reg = scaling_reg1_f + scaling_reg1_b + scaling_reg2_f + scaling_reg2_b

        if opt.optical_lambda == 0:
            optical_loss = 0
        else:
            optical_loss = calc_optical_loss(
                render_results1_f, render_results1_b,
                render_results2_f, render_results2_b,
                optical_flow,
                frame_cube.dataset.x_min, frame_cube.dataset.y_min, frame_cube.dataset.scale,
                frame_cube.dataset.width, frame_cube.dataset.height, gaussians.n_offsets
            )

        opacity_reg1_f = (1 - render_results1_f.neural_opacity).mean()
        opacity_reg1_b = (1 - render_results1_b.neural_opacity).mean()
        opacity_reg2_f = (1 - render_results2_f.neural_opacity).mean()
        opacity_reg2_b = (1 - render_results2_b.neural_opacity).mean()
        opacity_reg = opacity_reg1_f + opacity_reg1_b + opacity_reg2_f + opacity_reg2_b

        loss = (
                (1.0 - opt.lambda_dssim) * Ll1
                + opt.lambda_dssim * ssim_loss
                + opt.scaling_reg * scaling_reg
                + opt.opacity_reg * opacity_reg
                + opt.optical_lambda * optical_loss
                )

        # if bit_per_param is not None:
        if controller.entropy_constrained:
            assert render_results1_f.entropy_constrained
            assert render_results1_b.entropy_constrained
            assert render_results2_f.entropy_constrained
            assert render_results2_b.entropy_constrained
            bit_per_param = render_results1_f.bit_per_param + render_results1_b.bit_per_param \
                        + render_results2_f.bit_per_param + render_results2_b.bit_per_param

            bit_per_param = bit_per_param
            _, bit_hash_grid, MB_hash_grid, _ = get_binary_vxl_size((gaussians.get_encoding_params()+1)/2)
            denom = gaussians._anchor.shape[0]*(gaussians.feat_dim+6+3*gaussians.n_offsets)
            loss = loss + opt.lmbda * (bit_per_param + bit_hash_grid / denom)

            loss = loss + 5e-4 * torch.mean(torch.sigmoid(gaussians._mask))

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            # psnr_val1 = peak_signal_noise_ratio(image1.cpu().numpy(), gt_image1.cpu().numpy(), data_range=1)
            # psnr_val2 = peak_signal_noise_ratio(image2.cpu().numpy(), gt_image2.cpu().numpy(), data_range=1)

            psnr_val1 = psnr_func(image1, gt_image1, data_range=1)
            psnr_val2 = psnr_func(image2, gt_image2, data_range=1)

            psnr_val = (psnr_val1 + psnr_val2) / 2
            # psnr_val2 = psnr_func2(image, gt_image).item()

            if iteration in checkpoint_iterations:

                logger.info(f"[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), pipe.model_path + "/chkpnt" + str(iteration) + ".pth")

            # if iteration % 100 == 1:
            #     evaluate_one_frame(
            #         tb_writer,
            #         dataset_name,
            #         frame_cube,
            #         render, (pipe, background), iteration=iteration
            #     )
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration % 1000 == 1 and iteration != 1:

                curried_render = partial(
                    render,
                    pipe=pipe,
                    bg_color=background,
                    mode=render_mode
                )
                evaluate(
                    tb_writer,
                    dataset_name,
                    frame_cube,
                    curried_render, iteration=iteration, order=0
                )

            # Log and save
            torch.cuda.synchronize()
            t_start_log = time.time()

            num_rendered = render_results1_f.num_rendered + render_results1_b.num_rendered \
                           + render_results2_f.num_rendered + render_results2_b.num_rendered
            active_gaussian = render_results1_f.active_gaussains + render_results1_b.active_gaussains \
                              + render_results2_f.active_gaussains + render_results2_b.active_gaussains
            mask_ratio = (
                    (render_results1_f.selection_mask.sum() + render_results1_b.selection_mask.sum()
                     + render_results2_f.selection_mask.sum() + render_results2_b.selection_mask.sum())
                    / (render_results1_f.selection_mask.shape[0] + render_results1_b.selection_mask.shape[0]
                       + render_results2_f.selection_mask.shape[0] + render_results2_b.selection_mask.shape[0]
                       )
            )
            log_training_results(
                tb_writer,
                dataset_name,
                psnr_val,
                iteration,
                Ll1,
                loss,
                gaussians.get_anchor.shape[0],
                active_gaussians=active_gaussian,
                num_rendered=num_rendered,
                mask_ratio=mask_ratio,
                elapsed=iter_start.elapsed_time(iter_end)
            )

            if controller.entropy_constrained:
                log_bit_info(tb_writer, gaussians, dataset_name,
                             render_results1_f, render_results1_b,
                             render_results2_f, render_results2_b,
                             iteration)
                if iteration % 100 == 0:
                    log_estimated_bit_info(tb_writer, gaussians, dataset_name, iteration)
                if iteration % 1000 == 0:
                    log_real_bit_info(tb_writer, gaussians, dataset_name, iteration, pipe)

            if iteration in saving_iterations:
                logger.info("[ITER {}] Saving Gaussians".format(iteration))
                frame_cube.save(pipe.model_path, iteration)
            torch.cuda.synchronize()
            t_end_log = time.time()
            t_log = t_end_log - t_start_log
            log_time_sub += t_log

            # densification
            if controller.gaussian_statis:
                # 修改为双向
                gaussians.training_statis(render_results1_f)
                gaussians.training_statis(render_results1_b)
                gaussians.training_statis(render_results2_f)
                gaussians.training_statis(render_results2_b)

            if controller.gaussian_adjust_anchor:
                gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold,
                                        grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)

            if controller.clean_denorm:
                gaussians.opacity_accum = None
                gaussians.offset_gradient_accum = None
                gaussians.offset_denom = None
                torch.cuda.empty_cache()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

        controller.step()

    encode_gaussian(frame_cube.gaussians, pipe)

    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    curried_render = partial(
        render,
        pipe=pipe,
        bg_color=background,
    )

    evaluate(
        tb_writer,
        dataset_name,
        frame_cube,
        curried_render,
    )

    torch.cuda.synchronize()
    t_end = time.time()
    logger.info("Total Training time: {}".format(t_end-t_start-log_time_sub))

    return gaussians.x_bound_min, gaussians.x_bound_max
