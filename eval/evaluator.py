from itertools import Predicate
from cv2 import initUndistortRectifyMap
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2 as cv2
from .evaluation import *
## 将所有的测试
class Evalutator:
    def __init__(self, cfg, trainer):
        self.cfg = cfg
    @staticmethod
    def predict_whole(model,image,img_size):
        inter = nn.Upsample(size = img_size,mod = 'bilinear',align_corners=True)
        # 支持(N,C,H,W)的上采样的操作, 支持下面的额cazuo
        prediction = model(image)
        if (len(prediction)!=0):
            prediction = prediction[0] # 获取第0个特征的操作
        prediction = inter(prediction).detach().cpu().numpy()
        return prediction
    @staticmethod
    def predict_multiscale(model,cfg):
        



    