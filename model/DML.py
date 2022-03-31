# Demo测试，为了简化代码这个目前仅仅支持单卡的版本
from cProfile import label
from cmath import inf
from .loss import *
from utils.config import *
from utils.utils import *
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from .pspnet import Res_pspnet, BasicBlock, Bottleneck
from torch.utils.tensorboard import SummaryWriter
import tqdm
from utils.logging import get_logger
from dataset.RGBT import *
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
class Trainer:
    def __init__(self,cfg,log_path):
        # Todo：目前仅仅是单卡的版本，多卡版本应该是可见光一个模型，红外一个模型
        self.cfg = cfg
        self.visible = Res_pspnet(cfg.visible.Block,cfg.visible.Block_num,num_classes= cfg.classes)
        self.thermal = Res_pspnet(cfg.thermal.Block,cfg.thermal.Block_num,num_classes=cfg.classes)
        # Todo: 对抗学习策略目前还没有采用，只初始化了两个迭代器
        self.v_solver = optim.SGD([{'params': filter(lambda p: p.requires_grad, self.visible.parameters()), 'initial_lr': cfg.lr_v}], cfg.lr_v, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        self.v_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.v_solver,T_0 = cfg.visbile.T_0,T_mult = cfg.visbile.T_mult,verbose=False)
        self.t_solver = optim.SGD([{'params': filter(lambda p: p.requires_grad, self.thermal.parameters()), 'initial_lr': cfg.lr_t}], cfg.lr_t, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        self.t_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.t_solver, T_0 = cfg.thermal.T_0,T_mult=cfg.thermal.T_mult,verbose = False)
        # Todo: 同上对抗学习的技巧还没用上，并且相互编码和解码的奇淫技巧还没用上
        self.criterion = CriterionDSN() # BCE
        self.criterion_pixel = CriterionPixelWise()
        self.criterion_pair_wise = CriterionPairWiseforWholeFeatAfterPool(scale = cfg.pool_scale, feat_ind=-5)
        self.thermal.cuda()
        self.visible.cuda()
        # Todo: DataLoader部分还没写
        self.train_loader = DataLoader(MSDataSet(cfg=self.cfg,mode='train'),batch_size = cfg.train_batch,num_worker = 4,\
            pin_memory=True)
        self.train_loader = DataLoader(MSDataSet(cfg=self.cfg,mode='train'),batch_size = cfg.test_batch,num_worker = 4,\
            pin_memory=True)
        self.logger = get_logger(log_file=log_path)
        self.tensor_writer = SummaryWriter(log_dir = log_path,comment = 'QAQ')
        self.v_loss = 0.0
        self.t_loss = 0.0
        self.pi_v_t = 0.0
        self.pi_t_v = 0.0
        self.pa_v_t = 0.0
        self.pa_t_v = 0.0
        self.pi_G_loss = 0.0
        self.pa_G_loss = 0.0    
        cudnn.benchmark = True
    def training(self,epoch):
        
        self.visible.train()
        self.thermal.train()
        # 切换到训练模式 
        ## Todo 创建一个列表生成器记录单个Epoch 所有的Loss
        losses =[ AverageMeter() for i in range(8)]
        tbar = tqdm(enumerate(self.dataloader))
        for batch_index,(rgb_img,infrared_img,mask) in tbar:
            rgb_img = rgb_img.cuda()
            infrared_img = infrared_img.cuda()
            ### 全部给上张量，这个时候开始算loss 
            predict_rgb = self.visible(rgb_img)
            predict_thermal = self.thermal(infrared_img)
            thermal_loss = 0.0
            # Todo: Holistic Loss,下面是红外网络的Demo
            BCE_thermal = self.criterion(predict_thermal,mask,is_target_scattered = False)
            self.t_loss += BCE_thermal.item()
            KL_loss = self.criterion_pixel(predict_thermal,predict_rgb,is_target_scattered = False)
            self.pi_t_v += KL_loss.item()
            Pa_loss = self.criterion_pair_wise(predict_thermal,predict_rgb, is_target_scattered = True)
            self.pa_t_v = Pa_loss.item()
            thermal_loss = BCE_thermal*self.cfg.thermal.lambda_1+\
            KL_loss*self.cfg.thermal.lambda_2+Pa_loss*self.cfg.thermal.lambda_3
            self.tensor_writer.add_scalar('Thermal_loss/BCE_Thermal',self.t_loss,epoch)
            self.tensor_writer.add_scalar('Thermal_loss/KL_loss',self.pi_t_v,epoch)
            self.tensor_writer.add_scalar('Thermal_loss/pa_loss',self.pa_t_v,epoch)
            self.tensor_writer.add_scalar('Thermal_loss/Concat_loss',thermal_loss)
            # Todo: 可见光网络的Loss
            BCE_visble = self.criterion(predict_thermal,mask,is_target_scattered = False)
            self.v_loss = BCE_visble.item()
            KL_loss_2 = self.criterion_pixel(predict_thermal,predict_rgb,is_target_scattered = False)
            self.pi_v_t = KL_loss_2.item()
            Pa_loss_2 = self.criterion_pair_wise(predict_thermal,predict_rgb, is_target_scattered = True)
            self.pa_v_t = Pa_loss_2.item()
            visible_loss = BCE_visble*self.cfg.visible.lambda_1+\
            KL_loss_2*self.cfg.visible.lambda_2+Pa_loss_2*self.cfg.visible.lambda_3
            self.tensor_writer.add_scalar('Visible_loss/BCE_Thermal',self.v_loss,epoch)
            self.tensor_writer.add_scalar('Visible_loss/KL_loss',self.pi_v_t,epoch)
            self.tensor_writer.add_scalar('Visible_loss/pa_loss',self.pa_v_t,epoch)
            self.tensor_writer.add_scalar('Visible_loss/Concat_loss',visible_loss)
            ## 红外反向传播
            self.t_solver.zero_grad()
            thermal_loss.backward()
            self.t_solver.step()
            self.t_scheduler.step(len(self.dataloader)*epoch + batch_index)
            ## 可见光反向传播
            self.v_solver.zero_grad()
            visible_loss.backward()
            self.v_solver.step()
            self.v_scheduler.step(len(self.dataloader)*epoch + batch_index)
            ## 

    
