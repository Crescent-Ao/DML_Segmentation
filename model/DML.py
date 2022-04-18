# Demo测试，为了简化代码这个目前仅仅支持单卡的版本
from cProfile import label
from cmath import cos, inf
from loss import CriterionDSN, CriterionPixelWise, CriterionPairWiseforWholeFeatAfterPool
from utils.config import ConfigDict
from utils.utils import AverageMeter
from utils.config import Config
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pspnet import Res_pspnet
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils.logging import get_logger
from dataset.RGBT import MSDataSet

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class Trainer:
    def __init__(self, cfg, log_path):
        # Todo：目前仅仅是单卡的版本，多卡版本应该是可见光一个模型，红外一个模型
        self.cfg = cfg
        self.visible = Res_pspnet(cfg.visible.Block, cfg.visible.Block_num, num_classes=cfg.classes)
        self.thermal = Res_pspnet(cfg.thermal.Block, cfg.thermal.Block_num, num_classes=cfg.classes)
        # Todo: 对抗学习策略目前还没有采用，只初始化了两个迭代器
        self.v_solver = optim.SGD(
            [{"params": filter(lambda p: p.requires_grad, self.visible.parameters()), "initial_lr": cfg.lr_v}],
            cfg.lr_v,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        self.v_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.v_solver, T_0=cfg.visbile.T_0, T_mult=cfg.visbile.T_mult, verbose=False
        )
        self.t_solver = optim.SGD(
            [{"params": filter(lambda p: p.requires_grad, self.thermal.parameters()), "initial_lr": cfg.lr_t}],
            cfg.lr_t,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        self.t_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.t_solver, T_0=cfg.thermal.T_0, T_mult=cfg.thermal.T_mult, verbose=False
        )
        # Todo: 同上对抗学习的技巧还没用上，并且相互编码和解码的奇淫技巧还没用上
        self.criterion = CriterionDSN()  # BCE
        self.criterion_pixel = CriterionPixelWise()
        self.criterion_pair_wise = CriterionPairWiseforWholeFeatAfterPool(scale=cfg.pool_scale, feat_ind=-5)
        # 引入CPS loss
        self.criterion_cps = nn.CrossEntropyLoss(reduction="mean")
        self.thermal.cuda()
        self.visible.cuda()
        # Todo: DataLoader部分还没写
        self.train_loader = DataLoader(
            MSDataSet(cfg=self.cfg, mode="train"), batch_size=cfg.train_batch, num_worker=4, pin_memory=True
        )
        self.test_loader = DataLoader(
            MSDataSet(cfg=self.cfg, mode="test"), batch_size=cfg.test_batch, num_worker=4, pin_memory=True
        )
        self.logger = get_logger("DML", log_file=log_path)
        self.tensor_writer = SummaryWriter(log_dir=log_path, comment="QAQ")
        self.v_loss = 0.0
        self.t_loss = 0.0
        self.pi_v_t = 0.0
        self.pi_t_v = 0.0
        self.pa_v_t = 0.0
        self.pa_t_v = 0.0
        self.pi_G_loss = 0.0
        self.pa_G_loss = 0.0
        cudnn.benchmark = True

    # 训练之前两个分支独自学习
    def train_self_branch(self, epoch):
        self.visible.train()
        self.thermal.train()
        # 切换到训练模式

        losses = [AverageMeter() for i in range(2)]
        tbar = tqdm(enumerate(self.train_loader))
        for batch_index, (rgb_img, infrared_img, mask) in tbar:
            rgb_img = rgb_img.cuda()
            infrared_img = infrared_img.cuda()
            # 全部给上张量，这个时候开始算loss
            predict_rgb = self.visible(rgb_img)
            predict_thermal = self.thermal(infrared_img)
            # Todo: Holistic Loss,下面是红外网络的Demo
            # 红外loss
            BCE_thermal = self.criterion(predict_thermal, mask, is_target_scattered=False)
            losses[0].update(BCE_thermal.item(), self.cfg.train_batch)

            # 可见光网络的Loss
            BCE_visible = self.criterion(predict_rgb, mask, is_target_scattered=False)
            losses[1].update(BCE_visible.item(), self.cfg.train_batch)

            # 红外反向传播
            self.t_solver.zero_grad()
            BCE_thermal.backward()
            self.t_solver.step()
            self.t_scheduler.step(len(self.train_loader) * epoch + batch_index)
            # 可见光反向传播
            self.v_solver.zero_grad()
            BCE_visible.backward()
            self.v_solver.step()
            self.v_scheduler.step(len(self.train_loader) * epoch + batch_index)
        self.logger.info("train_self_branch:Epoch {}: Thermal  BCE:{:.10f}:".format(epoch, losses[0].avg,))
        self.tensor_writer.add_scalar("train_self_branch:Thermal_loss/BCE_Thermal", losses[0].avg, epoch)
        self.logger.info("train_self_branch:Epoch {}: Visible BCE:{:.10f}:".format(epoch, losses[1].avg))
        self.tensor_writer.add_scalar("train_self_branch:Visible_loss/BCE_Visible", losses[1].avg, epoch)

    def DML_training(self, epoch):

        self.visible.train()
        self.thermal.train()
        # 切换到训练模式
        # Todo 创建一个列表生成器记录单个Epoch 所有的Loss
        if self.cfg.cps_flag:
            cps_avg = AverageMeter()
        losses = [AverageMeter() for i in range(8)]
        tbar = tqdm(enumerate(self.train_loader))
        for batch_index, (rgb_img, infrared_img, mask) in tbar:
            rgb_img = rgb_img.cuda()
            infrared_img = infrared_img.cuda()
            # 全部给上张量，这个时候开始算loss
            predict_rgb = self.visible(rgb_img)
            predict_thermal = self.thermal(infrared_img)
            thermal_loss = 0.0
            # Todo: Holistic Loss,下面是红外网络的Demo
            BCE_thermal = self.criterion(predict_thermal, mask, is_target_scattered=False)
            losses[0].update(BCE_thermal.item(), self.cfg.train_batch)
            KL_loss = self.criterion_pixel(predict_thermal, predict_rgb, is_target_scattered=False)
            losses[1].update(KL_loss.item(), self.cfg.train_batch)
            Pa_loss = self.criterion_pair_wise(predict_thermal, predict_rgb, is_target_scattered=True)
            losses[2].update(Pa_loss.item(), self.cfg.train_batch)
            thermal_loss = (
                BCE_thermal * self.cfg.thermal.lambda_1
                + KL_loss * self.cfg.thermal.lambda_2
                + Pa_loss * self.cfg.thermal.lambda_3
            )
            losses[3].update(thermal_loss.item(), self.cfg.train_batch)
            # Todo: 可见光网络的Loss
            BCE_visible = self.criterion(predict_rgb, mask, is_target_scattered=False)
            losses[4].update(BCE_visible.item(), self.cfg.train_batch)
            KL_loss_2 = self.criterion_pixel(predict_thermal, predict_rgb, is_target_scattered=False)
            losses[5].update(KL_loss_2.item(), self.cfg.train_batch)
            Pa_loss_2 = self.criterion_pair_wise(predict_thermal, predict_rgb, is_target_scattered=True)
            losses[6].update(Pa_loss_2.item(), self.cfg.train_batch)
            visible_loss = (
                BCE_visible * self.cfg.visible.lambda_1
                + KL_loss_2 * self.cfg.visible.lambda_2
                + Pa_loss_2 * self.cfg.visible.lambda_3
            )
            losses[7].update(visible_loss.item(), self.cfg.train_batch)
            if self.cfg.cps_flag:
                cps_loss = self.cps_loss(predict_rgb, predict_thermal)
                thermal_loss = thermal_loss + cps_loss
                visible_loss = visible_loss + cps_loss
                cps_avg.update(cps_loss.item(), self.cfg.train_batch)
                self.tensor_writer.add_scalar("cps loss", cps_avg)
            # 红外反向传播
            self.t_solver.zero_grad()

            thermal_loss.backward()
            self.t_solver.step()
            self.t_scheduler.step(len(self.train_loader) * epoch + batch_index)
            # 可见光反向传播
            self.v_solver.zero_grad()
            visible_loss.backward()
            self.v_solver.step()
            self.v_scheduler.step(len(self.train_loader) * epoch + batch_index)
        self.logger.info(
            "Epoch {}: Thermal  BCE:{:.10f}, Pi loss:{:.10f}, Pa loss:{:.10f},Loss:{:.10f}:".format(
                epoch, losses[0].avg, losses[1].avg, losses[2].avg, losses[3].avg
            )
        )
        self.tensor_writer.add_scalar("Thermal_loss/BCE_Thermal", losses[0].avg, epoch)
        self.tensor_writer.add_scalar("Thermal_loss/KL_loss", losses[1].avg, epoch)
        self.tensor_writer.add_scalar("Thermal_loss/pa_loss", losses[2].avg, epoch)
        self.tensor_writer.add_scalar("Thermal_loss/Concat_loss", losses[3].avg, epoch)
        self.logger.info(
            "Epoch {}: Visible BCE:{:.10f}, Pi loss:{:.10f}, Pa loss:{:.10f},Loss:{:.10f}:".format(
                epoch, losses[4].avg, losses[5].avg, losses[6].avg, losses[7].avg
            )
        )
        self.tensor_writer.add_scalar("Visible_loss/BCE_Visible", losses[4].avg, epoch)
        self.tensor_writer.add_scalar("Visible_loss/KL_loss", losses[5].avg, epoch)
        self.tensor_writer.add_scalar("Visible_loss/pa_loss", losses[6].avg, epoch)
        self.tensor_writer.add_scalar("Visible_loss/Concat_loss", losses[7].avg, epoch)
        if self.cfg.cps_flag:
            self.tensor_writer.add_scalar("cps loss", cps_avg.avg, epoch)
            self.logger.info("Epoch{}: Cps loss{.10f}".format(epoch, cps_avg.avg))

    def cps_loss(self, predict_rgb, predict_thermal):
        # CVPR 2021 半监督伪标签代码的实现
        pre_rgb = predict_rgb[0]
        pre_thermal = predict_thermal[0]
        _, maxr = torch.max(predict_rgb, dim=1)
        _, maxt = torch.max(predict_thermal, 1)
        cps_loss = self.criterion_cps(pre_rgb, maxt) + self.criterion_cps(pre_thermal, maxr)
        return cps_loss

    def testing(self, epoch):
        self.visible.eval()
        self.thermal.eval()
        # 切换到训练模式

        losses = [AverageMeter() for i in range(2)]
        tbar = tqdm(enumerate(self.train_loader))

        for batch_index, (rgb_img, infrared_img, mask) in tbar:
            with torch.no_grad():
                rgb_img = rgb_img.cuda()
                infrared_img = infrared_img.cuda()
                # 全部给上张量，这个时候开始算loss
                predict_rgb = self.visible(rgb_img)
                predict_thermal = self.thermal(infrared_img)
                # Todo: Holistic Loss,下面是红外网络的Demo
                # 红外loss
                BCE_thermal = self.criterion(predict_thermal, mask, is_target_scattered=False)
                losses[0].update(BCE_thermal.item(), self.cfg.train_batch)

                # 可见光网络的Loss
                BCE_visible = self.criterion(predict_rgb, mask, is_target_scattered=False)
                losses[1].update(BCE_visible.item(), self.cfg.train_batch)

        self.logger.info("train_self_branch:Epoch {}: Thermal  BCE:{:.10f}:".format(epoch, losses[0].avg,))
        self.tensor_writer.add_scalar("train_self_branch:Thermal_loss/BCE_Thermal", losses[0].avg, epoch)
        self.logger.info("train_self_branch:Epoch {}: Visible BCE:{:.10f}:".format(epoch, losses[1].avg))
        self.tensor_writer.add_scalar("train_self_branch:Visible_loss/BCE_Visible", losses[1].avg, epoch)

    def validation(self, epoch):
        pass


def main():
    cfg = Config.fromfile(r"/home/guoshibo/DML_Segmentation/Config/dml_esp.py")
    print(cfg)
