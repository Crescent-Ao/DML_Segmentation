from torch.utils.data import Dataset
import cv2 as cv2
import PIL.Image as Image
import numpy as np
import pandas as pd
import os
import albumentations as A
import os.path as osp
"""
    传入全局cfg文件
"""
class MSDataSet(Dataset):
    """
        cfg： 全局模型参数配置文件
        mode: 控制训练验证和测试的部分,str
    """
    def __init__(self,cfg,mode):
        super().__init__()
        self.cfg = cfg # 传入全局cfg文件
        self.mode = mode
        self.image_path = osp.join(cfg.dataset,'images')
        self.mask_path = osp.join(cfg.dataset,'masks')
        cur_file = open(osp.join(cfg.dataset,mode+'.txt'))
        # 根据Mode 选择对应的方式
        self.cur_list = list(map(lambda x:x.rstrip('\n'),cur_file.readlines()))
    def __getitem__(self, index):
        
    def __len__(self):
        return len(self.cur_list)