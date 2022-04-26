from itertools import Predicate
from cv2 import initUndistortRectifyMap
import torch
from scipy import ndimage
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2 as cv2
from .evaluation import *

# 将所有的测试


class Evalutator:
    def __init__(self, cfg, trainer):
        self.trainer = trainer
        self.cfg = cfg

    def eval_main(self, ):


    @staticmethod
    def predict_whole(model, image, img_size):
        inter = nn.Upsample(size=img_size, mod="bilinear", align_corners=True)
        # 支持(N,C,H,W)的上采样的操作, 支持下面的额cazuo
        prediction = model(image)
        if len(prediction) != 0:
            prediction = prediction[0]  # 获取第0个特征的操作
        prediction = inter(prediction).detach().cpu().numpy()[0].transpose(1, 2, 0)
        return prediction

    @staticmethod
    def predict_multiscale(model, image, classes, scales, cfg):
        # 这边放到cpu去跑
        N, C, H, W = image.shape
        full_Probs = np.zeros((H, W, classes))
        for scale in scales:
            scale_image = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
            scaled_probs = Evalutator.predict_whole(model, scale_image,)
