import pathlib
import os
import numpy as np
from typing import Union
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms


class NeRVDataset(Dataset):
    def __init__(self, main_dir: Union[pathlib.Path,str], transform=None, vid_list=None, frame_gap=1, visualize=False):

        if isinstance(main_dir, str):
            main_dir = pathlib.Path(main_dir)

        self.main_dir = main_dir

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        frame_idx, self.frame_path = [], []
        accum_img_num = []
        all_imgs = list(main_dir.iterdir())
        all_imgs.sort()

        num_frame = 0
        for img_id in all_imgs:
            self.frame_path.append(img_id)
            frame_idx.append(num_frame)
            num_frame += 1

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        accum_img_num.append(num_frame)
        self.frame_idx = [float(x) / len(frame_idx) for x in frame_idx]
        self.accum_img_num = np.asfarray(accum_img_num)
        if vid_list is not None:
            self.frame_idx = [self.frame_idx[i] for i in vid_list]
        self.frame_gap = frame_gap

        image = Image.open(all_imgs[0]).convert("RGB")


        tensor_image = self.transform(image)
        self.height = tensor_image.shape[-2]
        self.width = tensor_image.shape[-1]

    def __len__(self):
        return len(self.frame_idx) // self.frame_gap

    @property
    def frame_height(self):
        return self.height

    @property
    def frame_width(self):
        return self.width

    def __getitem__(self, idx):
        valid_idx = idx * self.frame_gap
        img_id = self.frame_path[valid_idx]
        img_name = os.path.join(self.main_dir, img_id)
        image = Image.open(img_name).convert("RGB")
        tensor_image = self.transform(image)
        if tensor_image.size(1) > tensor_image.size(2):
            tensor_image = tensor_image.permute(0, 2, 1)
        frame_idx = torch.tensor(self.frame_idx[valid_idx])

        return tensor_image, frame_idx


