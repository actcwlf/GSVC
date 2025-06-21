import os
import pathlib
import typing
from dataclasses import dataclass

from loguru import logger
# import torch
# from frame_cube.frame_cube import FrameCube
# from utils.log_utils import prepare_output_and_logger
# from scene import GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
# from utils.report_utils import evaluate
# from ortho_gaussian_renderer import render, GenerateMode
import torch

# from typing import Optional, Tuple
# from torch import Tensor



@dataclass
class EncodeMeta:
    model_config: ModelParams
    total_anchor_num: int
    anchor_num: int
    batch_size: int
    anchor_infos_list: typing.List[typing.Any]
    min_feat_list: typing.List[typing.Any]
    max_feat_list: typing.List[typing.Any]
    min_scaling_list: typing.List[typing.Any]
    max_scaling_list: typing.List[typing.Any]
    min_offsets_list: typing.List[typing.Any]
    max_offsets_list: typing.List[typing.Any]


    # prob_hash,
    # prob_masks], log_info, bit_info)

# def test_model(
#         args_param,
#         model_params: ModelParams,
#         opt: OptimizationParams,
#         pipe: PipelineParams,
#         dataset_name,
#         checkpoint,
#         # logger=None
# ):
#
#     tb_writer = prepare_output_and_logger(model_params)
#
#     gaussians = GaussianModel(
#         model_params,
#         model_params.feat_dim,
#         model_params.n_offsets,
#         model_params.voxel_size,
#         model_params.update_depth,
#         model_params.update_init_factor,
#         model_params.update_hierarchy_factor,
#         model_params.use_feat_bank,
#         n_features_per_level=args_param.n_features,
#         log2_hashmap_size=args_param.log2,
#         log2_hashmap_size_2D=args_param.log2_2D,
#     )
#
#     frame_cube = FrameCube(model_params, gaussians)
#     # scene = Scene(dataset, gaussians, ply_path=ply_path)
#     gaussians.update_anchor_bound()
#
#     gaussians.training_setup(opt)
#     if checkpoint:
#         (model_checkpoint, first_iter) = torch.load(checkpoint)
#         gaussians.restore(model_checkpoint, opt)
#
#     bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
#     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
#
#     encode_gaussian(frame_cube.gaussians, args_param.model_path)
#
#     evaluate(
#         tb_writer,
#         dataset_name,
#         frame_cube,
#         render, (pipe, background, 1.0, False, GenerateMode.DECODING_AS_IS),
#         # logger
#     )


@torch.no_grad()
def encode_gaussian(gaussians, pipe: PipelineParams):
    log_info, _ = gaussians.estimate_final_bits()
    logger.info(log_info)

    bit_stream_path = pathlib.Path(pipe.model_path) / 'bitstreams'


    os.makedirs(bit_stream_path, exist_ok=True)
    # conduct encoding
    meta, prob_hash, prob_masks, bit_info = gaussians.conduct_encoding(pre_path_name=bit_stream_path,pipe=pipe)




    logger.info(log_info)
    # conduct decoding
    log_info = gaussians.conduct_decoding(
        pre_path_name=bit_stream_path, meta=meta, prob_hash=prob_hash, prob_masks=prob_masks, tmc3_path=pathlib.Path(pipe.tmc3_executable))
    logger.info(log_info)



@torch.no_grad()
def stream_encode_gaussian(gaussians, pipe: PipelineParams):
    log_info, _ = gaussians.estimate_final_bits()
    logger.info(log_info)

    bit_stream_path = pathlib.Path(pipe.model_path) / 'stream_bitstreams'


    os.makedirs(bit_stream_path, exist_ok=True)
    # conduct encoding
    meta, prob_hash, prob_masks, bit_info = gaussians.conduct_stream_encoding(pre_path_name=bit_stream_path,pipe=pipe)




    logger.info(log_info)
    # conduct decoding
    log_info = gaussians.conduct_stream_decoding(
        pre_path_name=bit_stream_path, meta=meta, prob_hash=prob_hash, prob_masks=prob_masks, tmc3_path=pathlib.Path(pipe.tmc3_executable))
    logger.info(log_info)