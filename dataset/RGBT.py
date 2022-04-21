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
import random 
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
        # 根据Mode 选择对应的方式,随机裁剪是默认开始的，这边不开MixUP了
        self.cur_list = list(map(lambda x: x.rstrip("\n"), cur_file.readlines()))
        self.cur_list = list(filter(isfilp, self.cur_list))
        self.multi_scale = cfg.multi_scale
        self.augtransform_train = A.Compose(
            [
                A.RandomCrop(height = cfg.train_sr.height, width = cfg.test_sr.width ,p = 0.8),
                A.Resize(height=cfg.train_sr.height, width=cfg.sr.width, p=1),
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
            ]
        )
        self.augtransform_test = A.Compose(
            [
                A.Resize(height=cfg.test_sr.height, width=cfg.test_sr.width)
            ]
        )
        # self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        # 最最最简单的数据增强的实现

    def __getitem__(self, index):
        img = np.array(Image.open(osp.join(self.image_path, self.cur_list[index] + ".png")))
        mask = np.array(Image.open(osp.join(self.mask_path, self.cur_list[index] + ".png")))
        # print(self.cur_list[index])

        if self.mode == "train":
            augumentation = self.augtransform_train(image=img, mask=mask)
        else:
            augumentation = self.augtransform_test(image=img, mask=mask)
        
        img = augumentation["image"]
        mask = augumentation["mask"]

        # elif self.mode == "test":
        #     augumentation = self.augtransform_test(image=img, mask=mask)
        rgb_img = img[:, :, :3].astype(np.float32)
        infrared = np.expand_dims(img[:, :, 3], axis=-1)
        infrared = np.concatenate([infrared, infrared, infrared], axis=-1).astype(np.float32)
        # 转换成
        """
            需要注意以下几个点：
                PIL读取图像的顺序为(H,W,C)
                opencv读取的(C,H,W),
                Test 部分不做多尺度标签分配
        """
        # 在这个尺度上进行对应的concate操作
        rgb_img = torch.from_numpy(rgb_img).permute(-1, 0, 1)
        infrared = torch.from_numpy(infrared).permute(-1, 0, 1)
        mask = torch.from_numpy(mask).long()
        if (self.multi_scale and self.mode =='Train'):
            rgb_img, infrared, mask = MSDataSet.multi_scale_label(rgb_img, infrared, mask)
            
        return rgb_img, infrared, mask
        # Todo 随机剪裁和多尺度测试还没有做，多尺度分布
        # 前三个波段为对应的rgb 波段 后面为对应的可见光波段,Multi-sca
    @staticmethod
    def multi_scale_label(rgb_img,infrared_img,mask):
        # 多尺度测试的方式
        f_scale = 0.5 + random.randint(0, 15) / 10.0
        rgb_img = cv2.resize(rgb_img,None,fx = f_scale,fy = f_scale, interpolation=cv2.INTER_LINEAR)
        infrared_img = cv2.resize(infrared_img,None,fx = f_scale,fy = f_scale, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, None, fx=f_scale,fy=f_scale,interpolation = cv2.INTER_NEAREST)
        return (rgb_img, infrared_img, mask)
    def __len__(self):
        return len(self.cur_list)
