# Demo测试，为了简化代码这个目前仅仅支持单卡的版本
from cgitb import enable
import os
from cmath import cos, inf
from cProfile import label
import random
import numpy as np
import torch
from torch.cuda import amp
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import utils.gpu as gpu
from dataset.RGBT import MSDataSet
from model.loss import *
from model.pspnet import Res_pspnet
from model.Unet import UNet
from utils.config import Config, ConfigDict
from utils.logging import get_logger
from utils.utils import *
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from prefetch_generator import BackgroundGenerator
from eval.evaluation import total_intersect_and_union
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
print(LOCAL_RANK, RANK, WORLD_SIZE, "QAQ")

# torch.cuda.set_device(1)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class InfiniteDataLoader(DataLoaderX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class Trainer:
    def __init__(self, cfg, log_path, gpu_id=1):
        self.best_thermal_miou = 0.0
        self.best_visible_miou = 0.0
        self.cfg = cfg
        self.logger = get_logger("DML", log_file=log_path)

        gpu.init_seeds(1 + RANK)
        # device = gpu.select_device_v5(gpu_id, batch_size=cfg.train_batch)
        if LOCAL_RANK != -1:
            torch.cuda.set_device(LOCAL_RANK)
            device = torch.device("cuda", LOCAL_RANK)
            dist.init_process_group(backend="nccl")
            self.logger.info(f"[init] == local rank: {LOCAL_RANK}, global rank: {RANK} ==")
        # 多卡的版本的实现
        self.device = device
        self.cuda = self.device.type != "cpu"
        self.test_loader = DataLoader(
            MSDataSet(cfg=self.cfg, mode="test"),
            batch_size=cfg.test_batch,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        with gpu.torch_distributed_zero_first(LOCAL_RANK):
            self.train_dataset = MSDataSet(cfg=self.cfg, mode="train")
        sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset) if LOCAL_RANK != -1 else None
        self.train_loader = InfiniteDataLoader(
            self.train_dataset,
            batch_size=self.cfg.train_batch // WORLD_SIZE,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        self.visible = Res_pspnet(layers=cfg.visible.Block_num, num_classes=cfg.classes)
        self.visible = nn.SyncBatchNorm.convert_sync_batchnorm(self.visible).to(device)
        self.thermal = Res_pspnet(layers=cfg.thermal.Block_num, num_classes=cfg.classes)
        self.thermal = nn.SyncBatchNorm.convert_sync_batchnorm(self.thermal).to(device)
        self.scaler = amp.GradScaler(enabled=self.cuda)
        if RANK != -1:
            self.visible = DDP(
                self.visible, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True
            )
            self.thermal = DDP(
                self.thermal, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True
            )

        # Todo: 对抗学习策略目前还没有采用，只初始化了两个迭代器
        self.v_solver = optim.SGD(
            [{"params": filter(lambda p: p.requires_grad, self.visible.parameters()), "initial_lr": cfg.lr_v}],
            cfg.lr_v,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        self.v_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.v_solver, T_0=cfg.visible.T_0, T_mult=cfg.visible.T_mult,
        )
        self.t_solver = optim.SGD(
            [{"params": filter(lambda p: p.requires_grad, self.thermal.parameters()), "initial_lr": cfg.lr_t}],
            cfg.lr_t,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        self.t_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.t_solver, T_0=cfg.thermal.T_0, T_mult=cfg.thermal.T_mult,
        )
        # Todo: 同上对抗学习的技巧还没用上，并且相互编码和解码的奇淫技巧还没用上
        self.criterion = CriterionDSN()  # BCE
        self.criterion_pixel = CriterionKD(temperature=cfg.KD_temperature)
        self.criterion_pair_wise = CriterionPairWiseforWholeFeatAfterPool(scale=cfg.pool_scale, feat_ind=-5)
        self.criterion_cwd = CriterionCWD(cfg.CWD.norm_type, cfg.CWD.divergence, cfg.CWD.temperature)
        self.criterion_test =  torch.nn.CrossEntropyLoss(ignore_index=0, reduce=True)
        # 引入CPS loss
        self.criterion_cps = nn.CrossEntropyLoss(reduction="mean")

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
        cudnn.deterministic = False

    def synchronize(self):
        """
        Helper function to synchronize (barrier) among all processes when
        using distributed training
        """
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        if world_size == 1:
            return
        dist.barrier()

    def init_seeds(seed=0, cuda_deterministic=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda_deterministic:  # slower, more reproducible
            cudnn.deterministic = True
            cudnn.benchmark = False
        else:  # faster, less reproducible
            cudnn.deterministic = False
            cudnn.benchmark = True

    # 训练之前两个分支独自学习
    def train_self_branch(self, epoch):
        self.visible.train()
        self.thermal.train()
        # 切换到训练模式
        if RANK != -1:
            self.train_loader.sampler.set_epoch(epoch)
        losses = [AverageMeter() for i in range(2)]
        tbar = tqdm(enumerate(self.train_loader))
        for batch_index, (rgb_img, infrared_img, mask) in tbar:
            # print(rgb_img.shape, "rgb")
            # print(infrared_img.shape, "infrared")
            # print(mask.shape, "mask")
            rgb_img = rgb_img.to(self.device, non_blocking=True)
            infrared_img = infrared_img.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            # 全部给上张量，这个时候开始算loss
            with amp.autocast():
                predict_rgb = self.visible(rgb_img)
                predict_thermal = self.thermal(infrared_img)
                # Todo: Holistic Loss,下面是红外网络的Demo

                # 红外loss

                BCE_thermal = self.criterion(predict_thermal, mask)
                losses[0].update(BCE_thermal.item(), self.cfg.train_batch)

                # 可见光网络的Loss
                BCE_visible = self.criterion(predict_rgb, mask)
                losses[1].update(BCE_visible.item(), self.cfg.train_batch)
                if RANK != -1:
                    BCE_thermal *= WORLD_SIZE
                    BCE_visible *= WORLD_SIZE

            # 红外反向传播
            self.scaler.scale(BCE_thermal).backward()
            self.scaler.step(self.t_solver)
            self.scaler.update()
            self.t_solver.zero_grad()
            self.t_scheduler.step(len(self.train_loader) * epoch + batch_index)
            # 可见光反向传播
            self.scaler.scale(BCE_visible).backward()
            self.scaler.step(self.v_solver)
            self.scaler.update()
            self.v_solver.zero_grad()
            self.v_scheduler.step(len(self.train_loader) * epoch + batch_index)
        if RANK in [-1, 0]:
            self.logger.info("train_self_branch:Epoch {}: Thermal  BCE:{:.10f}:".format(epoch, losses[0].avg,))
            self.tensor_writer.add_scalar("train_self_branch:Thermal_loss/BCE_Thermal", losses[0].avg, epoch)
            self.logger.info("train_self_branch:Epoch {}: Visible BCE:{:.10f}:".format(epoch, losses[1].avg))
            self.tensor_writer.add_scalar("train_self_branch:Visible_loss/BCE_Visible", losses[1].avg, epoch)

    def DML_training_visible(self, epoch):
        self.visible.train()
        self.thermal.eval()
        # 切换到训练模式
        # Todo 创建一个列表生成器记录单个Epoch 所有的Loss
        if self.cfg.cps_flag:
            cps_avg = AverageMeter()
        losses = [AverageMeter() for i in range(4)]
        tbar = tqdm(enumerate(self.train_loader))
        if RANK != -1:
            self.train_loader.sampler.set_epoch(epoch)
        for batch_index, (rgb_img, infrared_img, mask) in tbar:
            rgb_img = rgb_img.to(self.device, non_blocking=True)
            infrared_img = infrared_img.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            # 全部给上张量，这个时候开始算loss
            with amp.autocast():
                predict_rgb = self.visible(rgb_img)
                predict_thermal = self.thermal(infrared_img)
                BCE_visible = self.criterion(predict_rgb, mask)
                losses[0].update(BCE_visible.item(), self.cfg.train_batch)
                KL_loss_2 = self.criterion_pixel(predict_thermal, predict_rgb)
                losses[1].update(KL_loss_2.item(), self.cfg.train_batch)
                Pa_loss_2 = self.criterion_pair_wise(predict_thermal, predict_rgb)
                losses[2].update(Pa_loss_2.item(), self.cfg.train_batch)
                visible_loss = (
                    BCE_visible * self.cfg.visible.lambda_1
                    + KL_loss_2 * self.cfg.visible.lambda_2
                    + Pa_loss_2 * self.cfg.visible.lambda_3
                )
                losses[3].update(visible_loss.item(), self.cfg.train_batch)
                if RANK != -1:
                    visible_loss *= WORLD_SIZE

            self.scaler.scale(visible_loss).backward()
            self.scaler.step(self.v_solver)
            self.scaler.update()
            self.v_solver.zero_grad()
            self.v_scheduler.step(len(self.train_loader) * epoch + batch_index)
            # Todo: 可见光网络的Loss

            if self.cfg.cps_flag:
                cps_loss = self.cps_loss(predict_rgb, predict_thermal)
                thermal_loss = thermal_loss + cps_loss
                visible_loss = visible_loss + cps_loss
                cps_avg.update(cps_loss.item(), self.cfg.train_batch)
                self.tensor_writer.add_scalar("cps loss", cps_avg)
            """
            # 红外反向传播
            self.t_solver.zero_grad()
            thermal_loss.backward(retain_graph=True)
            self.t_solver.step()
            self.t_scheduler.step(len(self.train_loader) * epoch + batch_index)
           """
            # 可见光反向传播
        if RANK in [-1, 0]:

            self.logger.info(
                "Epoch {}: Visible BCE:{:.10f}, Pi loss:{:.10f}, Pa loss:{:.10f},Loss:{:.10f}:".format(
                    epoch, losses[0].avg, losses[1].avg, losses[2].avg, losses[3].avg
                )
            )
            self.tensor_writer.add_scalar("Visible_loss/BCE_Visible", losses[0].avg, epoch)
            self.tensor_writer.add_scalar("Visible_loss/KL_loss", losses[1].avg, epoch)
            self.tensor_writer.add_scalar("Visible_loss/pa_loss", losses[2].avg, epoch)
            self.tensor_writer.add_scalar("Visible_loss/Concat_loss", losses[3].avg, epoch)
            if self.cfg.cps_flag:
                self.tensor_writer.add_scalar("cps loss", cps_avg.avg, epoch)
                self.logger.info("Epoch{}: Cps loss{.10f}".format(epoch, cps_avg.avg))

    def DML_training_thermal(self, epoch):

        self.visible.eval()
        self.thermal.train()
        # 切换到训练模式
        # Todo 创建一个列表生成器记录单个Epoch 所有的Loss
        if self.cfg.cps_flag:
            cps_avg = AverageMeter()
        losses = [AverageMeter() for i in range(4)]
        tbar = tqdm(enumerate(self.train_loader))
        if RANK != -1:
            self.train_loader.sampler.set_epoch(epoch)
        for batch_index, (rgb_img, infrared_img, mask) in tbar:
            rgb_img = rgb_img.to(self.device, non_blocking=True)
            infrared_img = infrared_img.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            # 全部给上张量，这个时候开始算loss
            with amp.autocast():
                predict_rgb = self.visible(rgb_img)
                predict_thermal = self.thermal(infrared_img)
                thermal_loss = 0.0
                # Todo: Holistic Loss,下面是红外网络的Demo
                BCE_thermal = self.criterion(predict_thermal, mask)
                losses[0].update(BCE_thermal.item(), self.cfg.train_batch)
                KL_loss = self.criterion_pixel(predict_thermal, predict_rgb)
                losses[1].update(KL_loss.item(), self.cfg.train_batch)
                Pa_loss = self.criterion_pair_wise(predict_thermal, predict_rgb)
                losses[2].update(Pa_loss.item(), self.cfg.train_batch)
                thermal_loss = (
                    BCE_thermal * self.cfg.thermal.lambda_1
                    + KL_loss * self.cfg.thermal.lambda_2
                    + Pa_loss * self.cfg.thermal.lambda_3
                )
                losses[3].update(thermal_loss.item(), self.cfg.train_batch)
                if RANK != -1:
                    thermal_loss *= WORLD_SIZE

            self.scaler.scale(thermal_loss).backward()
            self.scaler.step(self.t_solver)
            self.scaler.update()
            self.t_solver.zero_grad()
            self.t_scheduler.step(len(self.train_loader) * epoch + batch_index)

            # Todo: 可见光网络的Loss

            if self.cfg.cps_flag:
                cps_loss = self.cps_loss(predict_rgb, predict_thermal)
                thermal_loss = thermal_loss + cps_loss
                visible_loss = visible_loss + cps_loss
                cps_avg.update(cps_loss.item(), self.cfg.train_batch)
                self.tensor_writer.add_scalar("cps loss", cps_avg)
            """
            # 红外反向传播
            self.t_solver.zero_grad()
            thermal_loss.backward(retain_graph=True)
            self.t_solver.step()
            self.t_scheduler.step(len(self.train_loader) * epoch + batch_index)
           """
            # 可见光反向传播
        if RANK in [-1, 0]:
            self.logger.info(
                "Epoch {}: Thermal  BCE:{:.10f}, Pi loss:{:.10f}, Pa loss:{:.10f},Loss:{:.10f}:".format(
                    epoch, losses[0].avg, losses[1].avg, losses[2].avg, losses[3].avg
                )
            )
            self.tensor_writer.add_scalar("Thermal_loss/BCE_Thermal", losses[0].avg, epoch)
            self.tensor_writer.add_scalar("Thermal_loss/KL_loss", losses[1].avg, epoch)
            self.tensor_writer.add_scalar("Thermal_loss/pa_loss", losses[2].avg, epoch)
            self.tensor_writer.add_scalar("Thermal_loss/Concat_loss", losses[3].avg, epoch)
            if self.cfg.cps_flag:
                self.tensor_writer.add_scalar("cps loss", cps_avg.avg, epoch)
                self.logger.info("Epoch{}: Cps loss{.10f}".format(epoch, cps_avg.avg))

    def cps_loss(self, predict_rgb, predict_thermal):
        # CVPR 2021 半监督伪标签代码的实现
        pre_rgb = predict_rgb[0]
        pre_thermal = predict_thermal[0]
        _, maxr = torch.max(predict_rgb, dim=1)
        _, maxt = torch.max(predict_thermal, dim=1)
        cps_loss = self.criterion_cps(pre_rgb, maxt) + self.criterion_cps(pre_thermal, maxr)
        return cps_loss

    def testing(self, epoch, weight_path):
        self.visible.eval()
        self.thermal.eval()
        # 切换到训练模式

        losses = [AverageMeter() for i in range(2)]
        tbar = tqdm(enumerate(self.test_loader))

        rgb_result = None
        thermal_result = None
        mask_result = None
        for batch_index, (rgb_img, infrared_img, mask) in tbar:
            with torch.no_grad():
                interp = nn.Upsample(
                    size=(self.cfg.test_sr.height, self.cfg.test_sr.width), mode="bilinear", align_corners=True
                )

                rgb_img = rgb_img.cuda()
                infrared_img = infrared_img.cuda()
                mask = mask.cuda()
                with amp.autocast():
                    predict_rgb = self.visible(rgb_img)
                    predict_thermal = self.thermal(infrared_img)

                # Todo: Holistic Loss,下面是红外网络的Demo
                # 红外loss
                BCE_thermal = self.criterion_test(predict_thermal, mask)
                losses[0].update(BCE_thermal.item(), self.cfg.test_batch)

                # 可见光网络的Loss
                BCE_visible = self.criterion_test(predict_rgb, mask)
                losses[1].update(BCE_visible.item(), self.cfg.test_batch)

                # Todo 这边计算多尺度的训练模式，DataSet也要集成多尺度的训练模式,完成
                # Todo 编程的主体思路如下，实现一个评估类，类中主要的实现方式为@staicMethod的方式
                # Todo log目前先不同实现

                rgb_heatmap = interp(predict_rgb[0])  # .detach().cpu()
                thermal_heatmap = interp(predict_thermal[0])  # .detach().cpu()
                if batch_index == 0:
                    rgb_result = rgb_heatmap
                    thermal_result = thermal_heatmap
                    mask_result = mask
                else:
                    rgb_result = torch.cat((rgb_result, rgb_heatmap), 0)
                    thermal_result = torch.cat((thermal_result, thermal_heatmap), 0)
                    mask_result = torch.cat((mask_result, mask), 0)

        rgb_all_acc, rgb_iou, rgb_miou, rgb_acc, rgb_precision, rgb_recall = total_intersect_and_union(
            self.cfg.classes, rgb_result, mask_result
        )

        (
            thermal_all_acc,
            thermal_iou,
            thermal_miou,
            thermal_acc,
            thermal_precision,
            thermal_recall,
        ) = total_intersect_and_union(self.cfg.classes, thermal_result, mask_result)

        print(rgb_miou.item(), "rgbmiou")
        print(self.best_visible_miou, "best miou")

        if rgb_miou.item() > self.best_visible_miou:
            self.best_visible_miou = rgb_miou.item()
            save_path = os.path.join(
                os.path.join(weight_path, "rgb_weight_%s_miou%s.pth" % (str(epoch), str(rgb_miou.item())))
            )
            torch.save(self.visible.state_dict(), save_path)

        if thermal_miou.item() > self.best_thermal_miou:
            self.best_thermal_miou = thermal_miou.item()
            save_path = os.path.join(
                os.path.join(weight_path, "thermal_weight_%s_miou%s.pth" % (str(epoch), str(thermal_miou.item())))
            )
            torch.save(self.thermal.state_dict(), save_path)

        self.logger.info(
            "test:Epoch {}: Thermal  BCE:{:.10f} all_all:{} acc:{} iou:{} miou:{} precision:{} recall:{}".format(
                epoch,
                losses[0].avg,
                thermal_all_acc.detach().cpu().numpy(),
                thermal_acc.detach().cpu().numpy(),
                thermal_iou.detach().cpu().numpy(),
                thermal_miou.detach().cpu().numpy(),
                thermal_precision.detach().cpu().numpy(),
                thermal_recall.detach().cpu().numpy(),
            )
        )
        self.tensor_writer.add_scalar("test:Thermal_loss/BCE_Thermal", losses[0].avg, epoch)
        self.logger.info(
            "test:Epoch {}: Visible BCE:{:.10f} all_all:{} acc:{} iou:{} miou:{} precision:{} recall:{}".format(
                epoch,
                losses[1].avg,
                rgb_all_acc.detach().cpu().numpy(),
                rgb_acc.detach().cpu().numpy(),
                rgb_iou.detach().cpu().numpy(),
                rgb_miou.detach().cpu().numpy(),
                rgb_precision.detach().cpu().numpy(),
                rgb_recall.detach().cpu().numpy(),
            )
        )
        self.tensor_writer.add_scalar("test:Visible_loss/BCE_Visible", losses[1].avg, epoch)


def main():
    global create_file
    cfg = Config.fromfile(os.path.join(os.getcwd(), "Config/dml_esp.py"))
    print(cfg)
    start_epoch = 0
    ckpt_path =  get_exp("ckpt")
    weight_path = get_exp("weights")
    log_path = get_exp("log")

    print(ckpt_path,"ckpt")


    trainer = Trainer(cfg, log_path=log_path)

    for epoch in range(start_epoch, cfg.self_branch_epochs):
        trainer.train_self_branch(epoch)
        if RANK in [-1, 0] and epoch % trainer.cfg.eval_freq == 0:
            print("testing!!!")
            trainer.testing(epoch, weight_path)
        if RANK in [-1, 0] and epoch % trainer.cfg.ckpt_freq == 0:
            print("save_ckpt!!!")
            save_ckpt(epoch=epoch, ckpt_path=ckpt_path, trainer=trainer)

    for epoch in range(cfg.self_branch_epochs, cfg.DML_epochs):
        trainer.DML_training_thermal(epoch)
        trainer.DML_training_visible(epoch)
        if RANK in [-1, 0] and epoch % trainer.cfg.eval_freq == 0:
            print("testing!!!")
            trainer.testing(epoch, weight_path)
        if RANK in [-1, 0] and epoch % trainer.cfg.ckpt_freq == 0:
            print("save_ckpt!!!")
            save_ckpt(epoch=epoch, ckpt_path=ckpt_path, trainer=trainer)

    # trainer.testing(epoch)


if __name__ == "__main__":
    main()
    if WORLD_SIZE > 1 and RANK == 0:
        dist.destroy_process_group()
