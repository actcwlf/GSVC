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
import random
import json

import numpy as np
from loguru import logger
from frame_cube.frame import FrameCubeDataset
from frame_cube.utils import init_point_cloud, load_point_cloud
from utils.graphics_utils import BasicPointCloud
# from utils.system_utils import searchForMaxIteration
# from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams


# from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON




class FrameCube:
    gaussians: GaussianModel

    def __init__(self,
                 opt: OptimizationParams,
                 pipe: PipelineParams,
                 gaussians: GaussianModel,
                 load_iteration=None,
                 shuffle=True,
                 resolution_scales=[1.0],
                 ply_path=None):
        """b
        :param path: Path to colmap scene main folder.
        """

        self.dataset = FrameCubeDataset(pipe.source_path, pipe.optical_path, prefetch=not pipe.skip_prefetch)

        if pipe.init_point_cloud != '':
            logger.info(f'using init point cloud {pipe.init_point_cloud}')
            pcd = BasicPointCloud(
                points=load_point_cloud(
                    pipe.init_point_cloud
                ),
                colors=None,
                normals=None
            )
        else:
            logger.info(f'using random point cloud {opt.init_anchor_num}')
            pcd = BasicPointCloud(
                points=init_point_cloud(
                    self.dataset.x_min,
                    self.dataset.y_min,
                    self.dataset.z_min,
                    n=opt.init_anchor_num
                ),
                colors=None,
                normals=None
            )

        self.gaussians = gaussians

        self.gaussians.create_from_pcd(pcd, 1) # TODO: 确定spatial_lr_scale的取值

        self.x_bound = 1.1

        # self.model_path = args.model_path
        # self.loaded_iter = None
        # self.gaussians = gaussians
        #
        # if load_iteration:
        #     if load_iteration == -1:
        #         self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
        #     else:
        #         self.loaded_iter = load_iteration
        #
        #     print("Loading trained model at iteration {}".format(self.loaded_iter))
        #
        # self.train_cameras = {}
        # self.test_cameras = {}
        #
        # self.x_bound = None
        # if os.path.exists(os.path.join(args.source_path, "sparse")):
        #     scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod)
        # elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        #     print("Found transforms_train.json file, assuming Blender data set!")
        #     scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval,
        #                                                    ply_path=ply_path)
        #     self.x_bound = 1.3
        # else:
        #     assert False, "Could not recognize scene type!"
        #
        # if not self.loaded_iter:
        #     if ply_path is not None:
        #         with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
        #                                                     'wb') as dest_file:
        #             dest_file.write(src_file.read())
        #     else:
        #         with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
        #                                                                'wb') as dest_file:
        #             dest_file.write(src_file.read())
        #     json_cams = []
        #     camlist = []  # 相机数量就是数据集中图像的数量
        #     if scene_info.test_cameras:
        #         camlist.extend(scene_info.test_cameras)
        #     if scene_info.train_cameras:
        #         camlist.extend(scene_info.train_cameras)
        #     for id, cam in enumerate(camlist):
        #         json_cams.append(camera_to_JSON(id, cam))
        #     with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
        #         json.dump(json_cams, file)
        #
        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        #     random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        #
        # self.cameras_extent = scene_info.nerf_normalization["radius"]
        #
        # # print(f'self.cameras_extent: {self.cameras_extent}')
        #
        # for resolution_scale in resolution_scales:
        #     print("Loading Training Cameras")
        #     self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
        #                                                                     args)
        #     print("Loading Test Cameras")
        #     self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
        #                                                                    args)
        #
        # if self.loaded_iter:
        #     self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
        #                                                          "point_cloud",
        #                                                          "iteration_" + str(self.loaded_iter),
        #                                                          "point_cloud.ply"))
        #     self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
        #                                                      "point_cloud",
        #                                                      "iteration_" + str(self.loaded_iter),
        #                                                      "checkpoint.pth"))
        # else:
        #     self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, model_path, iteration):
        point_cloud_path = os.path.join(model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(os.path.join(point_cloud_path, "checkpoint.pth"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]