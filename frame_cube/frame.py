import pathlib
import os
import pickle
from dataclasses import dataclass

import numpy as np
from typing import Union
from PIL import Image
from loguru import logger
from torch.utils.data import Dataset
import torch
from torchvision import transforms

import glm
from tqdm import tqdm


def make_view_matrix(x=0, y=0, z=0, plane='xy'):
    eye = glm.vec3(x, y, z)
    if plane == 'xy':
        center = glm.vec3(x, y, z - 0.1) # 由于view空间取正向轴，因此这里看向负向
        center_s = glm.vec3(x, y, z + 0.1)  # 第二视角方向相反
        up = glm.vec3(0, 1, 0)
    elif plane == 'yz':
        center = glm.vec3(x - 0.1, y, z)
        center_s = glm.vec3(x + 0.1, y, z)
        up = glm.vec3(0, 0, 1)
    elif plane == 'zx':
        center = glm.vec3(x, y - 0.1, z)
        center_s = glm.vec3(x, y + 0.1, z)
        up = glm.vec3(1, 0, 0)
    else:
        raise ValueError()

    view_mat = glm.lookAt(eye, center, up)
    view_matrix = torch.Tensor(np.array(view_mat))

    view_mat_s = glm.lookAt(eye, center_s, up)
    view_matrix_s = torch.Tensor(np.array(view_mat_s))

    cam_pos = torch.Tensor(np.array(eye))

    return view_matrix, view_matrix_s, cam_pos


@dataclass
class Frame:
    image_id: int
    plane: str
    image: torch.Tensor
    x_min: float
    y_min: float
    z: float
    image_width: int
    image_height: int
    view_matrix: torch.Tensor
    view_matrix_s: torch.Tensor
    scale: float
    cam_pos: torch.Tensor





class FrameCubeDataset(Dataset):
    def __init__(self, main_dir: Union[pathlib.Path, str], optical_flow_dir, transform=None, prefetch=True):

        if isinstance(main_dir, str):
            main_dir = pathlib.Path(main_dir)

        if isinstance(optical_flow_dir, str):
            optical_flow_dir = pathlib.Path(optical_flow_dir)

        # self.optical_flow_dir = optical_flow_dir
        self.optical_flow_paths = sorted(optical_flow_dir.iterdir())
        self.main_dir = main_dir
        self.z_frame_dir = main_dir #/ 'z_frame'
        # self.x_frame_dir = main_dir / 'x_frame'
        # self.y_frame_dir = main_dir / 'y_frame'

        self.z_frame_paths = sorted(self.z_frame_dir.iterdir())
        # self.x_frame_paths = sorted(self.x_frame_dir.iterdir())
        # self.y_frame_paths = sorted(self.y_frame_dir.iterdir())

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform



        z_image = Image.open(self.z_frame_paths[0]).convert("RGB")


        tensor_image = self.transform(z_image)
        self.height =  tensor_image.shape[-2]
        self.width =   tensor_image.shape[-1]
        self.scale = max(self.height, self.width, self.frame_num) / 2 # ndc_scale = 2 * scale
        self.x_min = -self.width / 2 / self.scale
        self.y_min = -self.height / 2 / self.scale
        self.z_min = -len(self.z_frame_paths) / 2 / self.scale


        self.prefetched_images = []
        self.prefetched_of = []

        if prefetch:
            self.prefetch()



    def __len__(self):
        # return self.len_z_frames + self.len_x_frames + self.len_y_frames

        return self.len_z_frames

    @property
    def frame_num(self):
        return self.len_z_frames

    @property
    def frame_height(self):
        return self.height

    @property
    def frame_width(self):
        return self.width

    @property
    def len_z_frames(self):
        return len(self.z_frame_paths)

    # @property
    # def len_x_frames(self):
    #     return len(self.x_frame_paths)
    #
    # @property
    # def len_y_frames(self):
    #     return len(self.y_frame_paths)

    def prefetch(self):
        logger.info('prefetching data...')
        for img_name in tqdm(self.z_frame_paths):
            image = Image.open(img_name).convert("RGB")
            tensor_image = self.transform(image).permute(0, 2, 1)
            self.prefetched_images.append(tensor_image)

        for of_pkl_name in tqdm(self.optical_flow_paths):
            with open(of_pkl_name, 'rb') as f:
                uv_tensor = torch.Tensor(pickle.load(f))
            uv_tensor = torch.Tensor(uv_tensor)
            self.prefetched_of.append(uv_tensor)



    def get_z_frame(self, image_id, load_image=True):
        max_id = self.len_z_frames
        z = (image_id - max_id / 2) / self.scale
        image_width = self.width
        image_height = self.height
        x_min = - image_width / 2 / self.scale
        y_min = - image_height / 2 / self.scale

        view_matrix, view_matrix_s, cam_pos = make_view_matrix(z=z, plane='xy')

        if load_image:
            if self.prefetched_images:
                tensor_image = self.prefetched_images[image_id]
            else:
                img_name = self.z_frame_paths[image_id]
                image = Image.open(img_name).convert("RGB")
                tensor_image = self.transform(image).permute(0, 2, 1)
        else:
            tensor_image = None

        frame = Frame(
            image_id=image_id,
            plane='xy',
            image=tensor_image,
            x_min=x_min,
            y_min=y_min,
            z=z,
            image_width=image_width,
            image_height=image_height,
            view_matrix=view_matrix,
            view_matrix_s=view_matrix_s,
            scale=self.scale,
            cam_pos=cam_pos
        )
        return frame


    def get_dummy_frame(self, image_id):
        return self.get_z_frame(image_id, load_image=False)

    # def get_x_frame(self, image_id):
    #     img_name = self.x_frame_paths[image_id]
    #     max_id = self.len_x_frames
    #
    #     z = (image_id - max_id / 2) / self.scale
    #     image_width = self.len_y_frames
    #     image_height = self.len_z_frames
    #     x_min = - image_width / 2 / self.scale
    #     y_min = - image_height / 2 / self.scale
    #
    #     view_matrix, view_matrix_s, cam_pos = make_view_matrix(x=z, plane='yz')
    #
    #
    #     image = Image.open(img_name).convert("RGB")
    #     tensor_image = self.transform(image).permute(0, 2, 1)
    #
    #     frame = Frame(
    #         image_id=image_id,
    #         plane='yz',
    #         image=tensor_image,
    #         x_min=x_min,
    #         y_min=y_min,
    #         z=z,
    #         image_width=image_width,
    #         image_height=image_height,
    #         view_matrix=view_matrix,
    #         view_matrix_s=view_matrix_s,
    #         scale=self.scale,
    #         cam_pos=cam_pos
    #     )
    #     return frame
    #
    # def get_y_frame(self, image_id):
    #     img_name = self.y_frame_paths[image_id]
    #     max_id = self.len_y_frames
    #
    #     z = (image_id - max_id / 2) / self.scale
    #     image_width = self.len_z_frames
    #     image_height = self.len_x_frames
    #     x_min = - image_width / 2 / self.scale
    #     y_min = - image_height / 2 / self.scale
    #
    #     view_matrix, view_matrix_s, cam_pos = make_view_matrix(y=z, plane='zx')
    #
    #     image = Image.open(img_name).convert("RGB")
    #     tensor_image = self.transform(image)
    #     frame = Frame(
    #         image_id=image_id,
    #         plane='zx',
    #         image=tensor_image,
    #         x_min=x_min,
    #         y_min=y_min,
    #         z=z,
    #         image_width=image_width,
    #         image_height=image_height,
    #         view_matrix=view_matrix,
    #         view_matrix_s=view_matrix_s,
    #         scale=self.scale,
    #         cam_pos=cam_pos
    #     )
    #     return frame

    def __getitem__(self, idx):
        return self.get_z_frame(idx)
        # idx = idx + self.len_z_frames # for debug only
        # idx = idx + self.len_x_frames # for debug only
        # if idx < self.len_z_frames:
        #     return self.get_z_frame(idx)
        # elif self.len_z_frames <= idx < self.len_z_frames + self.len_x_frames:
        #     return self.get_x_frame(idx - self.len_z_frames)
        # else:
        #     assert idx >= self.len_z_frames + self.len_x_frames
        #     return self.get_y_frame(idx - self.len_z_frames - self.len_x_frames)

    def get_optical_flow(self, idx):
        if self.prefetched_of:
            return self.prefetched_of[idx]

        with open(self.optical_flow_paths[idx], 'rb') as f:
            uv_tensor = torch.Tensor(pickle.load(f))
        return  torch.Tensor(uv_tensor)


