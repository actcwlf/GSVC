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

# from argparse import ArgumentParser, Namespace
# import sys
# import os
from dataclasses import dataclass


# class GroupParams:
#     pass
# 
# class ParamGroup:
#     def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
#         group = parser.add_argument_group(name)
#         for key, value in vars(self).items():
#             shorthand = False
#             if key.startswith("_"):
#                 shorthand = True
#                 key = key[1:]
#             t = type(value)
#             value = value if not fill_none else None 
#             if shorthand:
#                 if t == bool:
#                     group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
#                 else:
#                     group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
#             else:
#                 if t == bool:
#                     group.add_argument("--" + key, default=value, action="store_true")
#                 else:
#                     group.add_argument("--" + key, default=value, type=t)
# 
#     def extract(self, args):
#         group = GroupParams()
#         for arg in vars(args).items():
#             if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
#                 setattr(group, arg[0], arg[1])
#         return group


@dataclass
class ModelParams:
    """ Model configuration """
    sh_degree: int = 0                   # dimension of sh 
    threshold: float = 0.1               # render horizon
    kernel_size: float = 0.3             # kernel size of low pass filter
    anchor_feature_dim: int = 50                   # Scaffold-GS feature dimension
    n_offsets: int = 10                  # Scaffold-GS number of generated gaussians per anchor
    voxel_size: float = 0.001            # if voxel_size<=0, using 1nn dist
    update_depth: int = 3                # 
    update_init_factor: int = 16         # 
    update_hierarchy_factor: int = 4     # 



    time_multi_res: int = 16             #
    offset_multi_res: int = 16           #

    log2: int = 13
    log2_2D: int = 15
    grid_feature_dim: int = 4
    # parser.add_argument("--lmbda", type=float, default = 0.001)

    use_feat_bank: bool = False          #

    resolution: int = -1                 #
    white_background: bool = False       #

    
    
# class ModelParams(ParamGroup): 
#     def __init__(self, parser, sentinel=False):
#         self.sh_degree = 0
#         self.threshold = 0.1
#         self.kernel_size = 0.3
#         self.feat_dim = 50
#         self.n_offsets = 10
#         self.voxel_size = 0.001  # if voxel_size<=0, using 1nn dist
#         self.update_depth = 3
#         self.update_init_factor = 16
#         self.update_hierarchy_factor = 4
#         self.init_anchor_num = 10_000
#         self.init_point_cloud = ''
# 
#         self.time_multi_res = 4
#         self.offset_multi_res = 4
# 
#         self.use_feat_bank = False
#         self.source_path = ""
#         self.optical_path = ''
#         self.model_path = ""
#         self.images = "images"
#         self.resolution = -1
#         self.white_background = False
#         self.data_device = "cuda"
#         self.eval = True
#         self.lod = 0
#         super().__init__(parser, "Loading Parameters", sentinel)
# 
#     def extract(self, args):
#         g = super().extract(args)
#         g.source_path = os.path.abspath(g.source_path)
#         return g


@dataclass
class PipelineParams:
    """ pipeline config """
    source_path: str = ""  #
    optical_path: str = ''  #
    model_path: str = ""  #
    tmc3_executable: str = None
    # images: str = "images"  #

    init_point_cloud: str = ''           #

    # data_device: str = "cuda"            #
    # eval: bool = True                    #
    # lod: int = 0                         #

    convert_SHs_python: bool = False     #
    compute_cov3D_python: bool = False   #
    debug: bool = False                  #
    skip_prefetch: bool = False
    

# class PipelineParams(ParamGroup):
#     def __init__(self, parser):
#         self.convert_SHs_python = False
#         self.compute_cov3D_python = False
#         self.debug = False
#         super().__init__(parser, "Pipeline Parameters")


@dataclass
class OptimizationParams:
    """ Optimization config """
    iterations: int = 40_000                #
    position_lr_init: float = 0.0   #
    position_lr_final: float = 0.0   #
    position_lr_delay_mult: float = 0.01   #
    position_lr_max_steps: int = 40_000   #

    offset_lr_init: float = 0.01   #
    offset_lr_final: float = 0.0001   #
    offset_lr_delay_mult: float = 0.01   #
    offset_lr_max_steps: int = 40_000   #

    mask_lr_init: float = 0.01   #
    mask_lr_final: float = 0.0001   # 
    mask_lr_delay_mult: float = 0.01   #
    mask_lr_max_steps: int = 40_000   #

    feature_lr: float = 0.0075   #
    opacity_lr: float = 0.02   #
    scaling_lr: float = 0.007   #
    rotation_lr: float = 0.002   #

    mlp_opacity_lr_init: float = 0.002   #
    mlp_opacity_lr_final: float = 0.00002   #
    mlp_opacity_lr_delay_mult: float = 0.01   #
    mlp_opacity_lr_max_steps: int = 40_000   #

    mlp_cov_lr_init: float = 0.004   #
    mlp_cov_lr_final: float = 0.004   #
    mlp_cov_lr_delay_mult: float = 0.01   #
    mlp_cov_lr_max_steps: int = 40_000   #

    mlp_color_lr_init: float = 0.008  #
    mlp_color_lr_final: float = 0.00005  #
    mlp_color_lr_delay_mult: float = 0.01  #
    mlp_color_lr_max_steps: int = 40_000  #

    mlp_featurebank_lr_init: float = 0.01  #
    mlp_featurebank_lr_final: float = 0.00001  #
    mlp_featurebank_lr_delay_mult: float = 0.01  #
    mlp_featurebank_lr_max_steps: int = 40_000  #

    encoding_xyz_lr_init: float = 0.005  #
    encoding_xyz_lr_final: float = 0.00001  #
    encoding_xyz_lr_delay_mult: float = 0.33  #
    encoding_xyz_lr_max_steps: int = 40_000  #

    mlp_grid_lr_init: float = 0.005  #
    mlp_grid_lr_final: float = 0.00001  #
    mlp_grid_lr_delay_mult: float = 0.01  #
    mlp_grid_lr_max_steps: int = 40_000  #

    mlp_deform_lr_init: float = 0.005  #
    mlp_deform_lr_final: float = 0.0005  #
    mlp_deform_lr_delay_mult: float = 0.01  #
    mlp_deform_lr_max_steps: int = 40_000  #

    mlp_entropy_net_lr_init: float = 0.005
    mlp_entropy_net_lr_final: float = 0.0005
    mlp_entropy_net_lr_delay_mult: float = 0.01
    mlp_entropy_net_lr_max_steps: int = 40_000

    init_anchor_num: int = 10_000        #

    lmbda: float = 0.001             # control RD trade-off

    percent_dense: float = 0.01  #
    lambda_dssim: float = 0.2  #

    # for anchor densification
    start_stat: int = 500  #
    update_from: int = 1500  #
    update_interval: int = 100  #
    update_until: int = 25_000  #
    pause_densification: int = 1_000   # 在量化开始后一段时间不进行desification，同时也不统计相关梯度信息
    # self.update_until = 30_000  # for dev only

    # for dev only
    # self.start_stat = 200
    # self.update_from = 500
    # self.update_interval = 100
    # self.update_until = 2_500
    scaling_reg: float = 0.01  #
    opacity_reg: float = 0  #
    optical_lambda: float = 5

    full_precision_training_total: int = 10_000   # 3_000  # 全精度训练10_000
    quantized_training_total: int = 5_000   # 量化训练10_000
    entropy_constrained_train_total: int = 20_000   # 带有约束训练10_000
    ste_entropy_constrained_train_total: int = 5_000   # 带有约束训练10_000

    # self.full_precision_training_total = 1_000   # 3_000  # 全精度训练10_000
    # self.quantized_training_total = 1_000   # 量化训练10_000
    # self.entropy_constrained_train_total = 1_000   # 带有约束训练10_000
    # self.ste_entropy_constrained_train_total = 1_000   # 带有约束训练10_000

    min_opacity: float = 0.005   # 0.2
    success_threshold: float = 0.8  #
    densify_grad_threshold: float = 0.0005    # 0.0002 -> 0.0004 -> 0.0005


# class OptimizationParams(ParamGroup):
#     def __init__(self, parser):
#         self.iterations = 40_000
#         self.position_lr_init = 0.0
#         self.position_lr_final = 0.0
#         self.position_lr_delay_mult = 0.01
#         self.position_lr_max_steps = 40_000
#         
#         self.offset_lr_init = 0.01
#         self.offset_lr_final = 0.0001
#         self.offset_lr_delay_mult = 0.01
#         self.offset_lr_max_steps = 40_000
#         
#         self.mask_lr_init = 0.01
#         self.mask_lr_final = 0.0001
#         self.mask_lr_delay_mult = 0.01
#         self.mask_lr_max_steps = 40_000
# 
#         self.feature_lr = 0.0075
#         self.opacity_lr = 0.02
#         self.scaling_lr = 0.007
#         self.rotation_lr = 0.002
#         
#         self.mlp_opacity_lr_init = 0.002
#         self.mlp_opacity_lr_final = 0.00002  
#         self.mlp_opacity_lr_delay_mult = 0.01
#         self.mlp_opacity_lr_max_steps = 40_000
# 
#         self.mlp_cov_lr_init = 0.004
#         self.mlp_cov_lr_final = 0.004
#         self.mlp_cov_lr_delay_mult = 0.01
#         self.mlp_cov_lr_max_steps = 40_000
#         
#         self.mlp_color_lr_init = 0.008
#         self.mlp_color_lr_final = 0.00005
#         self.mlp_color_lr_delay_mult = 0.01
#         self.mlp_color_lr_max_steps = 40_000
#         
#         self.mlp_featurebank_lr_init = 0.01
#         self.mlp_featurebank_lr_final = 0.00001
#         self.mlp_featurebank_lr_delay_mult = 0.01
#         self.mlp_featurebank_lr_max_steps = 40_000
# 
#         self.encoding_xyz_lr_init = 0.005
#         self.encoding_xyz_lr_final = 0.00001
#         self.encoding_xyz_lr_delay_mult = 0.33
#         self.encoding_xyz_lr_max_steps = 40_000
# 
#         self.mlp_grid_lr_init = 0.005
#         self.mlp_grid_lr_final = 0.00001
#         self.mlp_grid_lr_delay_mult = 0.01
#         self.mlp_grid_lr_max_steps = 40_000
# 
#         self.mlp_deform_lr_init = 0.005
#         self.mlp_deform_lr_final = 0.0005
#         self.mlp_deform_lr_delay_mult = 0.01
#         self.mlp_deform_lr_max_steps = 40_000
# 
#         self.percent_dense = 0.01
#         self.lambda_dssim = 0.2
#         
#          # for anchor densification
#         self.start_stat = 500
#         self.update_from = 1500
#         self.update_interval = 100
#         self.update_until = 25_000
#         self.pause_densification = 1_000  # 在量化开始后一段时间不进行desification，同时也不统计相关梯度信息
#          # self.update_until = 30_000  # for dev only
# 
#          # for dev only
#          # self.start_stat = 200
#          # self.update_from = 500
#          # self.update_interval = 100
#          # self.update_until = 2_500
#         self.scaling_reg = 0.01
#         self.opacity_reg = 0.01
# 
# 
#         self.full_precision_training_total = 10_000  # 3_000  # 全精度训练10_000
#         self.quantized_training_total = 5_000       # 量化训练10_000
#         self.entropy_constrained_train_total = 20_000  # 带有约束训练10_000
#         self.ste_entropy_constrained_train_total = 5_000   # 带有约束训练10_000
# 
#          # self.full_precision_training_total = 1_000   # 3_000  # 全精度训练10_000
#          # self.quantized_training_total = 1_000   # 量化训练10_000
#          # self.entropy_constrained_train_total = 1_000   # 带有约束训练10_000
#          # self.ste_entropy_constrained_train_total = 1_000   # 带有约束训练10_000
#         
#         self.min_opacity = 0.005   # 0.2
#         self.success_threshold = 0.8
#         self.densify_grad_threshold = 0.0004
# 
#         super().__init__(parser, "Optimization Parameters")
# 
# def get_combined_args(parser : ArgumentParser):
#     cmdlne_string = sys.argv[1:]
#     cfgfile_string = "Namespace()"
#     args_cmdline = parser.parse_args(cmdlne_string)
# 
#     try:
#         cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
#         print("Looking for config file in", cfgfilepath)
#         with open(cfgfilepath) as cfg_file:
#             print("Config file found: {}".format(cfgfilepath))
#             cfgfile_string = cfg_file.read()
#     except TypeError:
#         print("Config file not found at")
#         pass
#     args_cfgfile = eval(cfgfile_string)
# 
#     merged_dict = vars(args_cfgfile).copy()
#     for k,v in vars(args_cmdline).items():
#         if v != None:
#             merged_dict[k] = v
#     return Namespace(**merged_dict)
