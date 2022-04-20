from cmath import inf
from torch.utils.data import Dataset
import cv2 as cv2
import PIL.Image as Image
import numpy as np
import pandas as pd
import torchvision
import os
import albumentations as A
import torch
import os.path as osp

from tqdm import trange

"""
    传入全局cfg文件
"""


def isfilp(x):
    return not x.endswith("flip")


class MSDataSet(Dataset):
    """
        cfg： 全局模型参数配置文件
        mode: 控制训练验证和测试的部分,str
    """

    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg = cfg  # 传入全局cfg文件
        self.mode = mode
        self.image_path = osp.join(cfg.dataset, "images")
        self.mask_path = osp.join(cfg.dataset, "labels")
        cur_file = open(osp.join(cfg.dataset, mode + ".txt"))
        # 根据Mode 选择对应的方式
        self.cur_list = list(map(lambda x: x.rstrip("\n"), cur_file.readlines()))
        self.cur_list = list(filter(isfilp, self.cur_list))
        print(mode, len(self.cur_list))
        self.augtransform_train = A.Compose(
            [
                A.Resize(height=cfg.height, width=cfg.width, interpolation=cv2.INTER_CUBIC, p=1),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5),
                # A.Transpose(p=0.5),
                # Todo RandomCrop
                # Todo CutMix @圈圈师姐
            ]
        )
        self.augtransform_test = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                # Todo RandomCrop
                # Todo CutMix @圈圈师姐
            ]
        )
        # self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        # 最最最简单的数据增强的实现

    def __getitem__(self, index):
        img = np.array(Image.open(osp.join(self.image_path, self.cur_list[index] + ".png")))
        mask = np.array(Image.open(osp.join(self.mask_path, self.cur_list[index] + ".png")))
        augumentation = A.Compose([])
        # print(self.cur_list[index])

        if self.mode == "train":
            augumentation = self.augtransform_train(image=img, mask=mask)
        img = augumentation["image"]
        mask = augumentation["mask"]

        # elif self.mode == "test":
        #     augumentation = self.augtransform_test(image=img, mask=mask)
        rgb_img = img[:, :, :3].astype(np.float32)
        infrared = np.expand_dims(img[:, :, 3], axis=-1)
        infrared = np.concatenate([infrared, infrared, infrared], axis=-1).astype(np.float32)
        # 转换成
       
        # 在这个尺度上进行对应的concate操作
        rgb_img = torch.from_numpy(rgb_img).permute(-1, 0, 1)
        infrared = torch.from_numpy(infrared).permute(-1, 0, 1)
        mask = torch.from_numpy(mask).long()

        return rgb_img, infrared, mask
        # Todo 随机剪裁和多尺度测试还没有做
        # 前三个波段为对应的rgb 波段 后面为对应的可见光波段,Multi-sca

    def __len__(self):
        return len(self.cur_list)
