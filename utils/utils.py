import torch
import os
import json


def get_exp(*property):

    if len(property) == 1:
        property_path = os.path.join(os.getcwd(), property[0])
    else:
        property_path = os.path.join(os.getcwd(), property[0], property[1])
    cur_sequence = sorted(list(map(lambda x: int(x[3:]), os.listdir(property_path))))
    cur_path = os.path.join(property_path, "exp" + str(cur_sequence[-1]))
    return cur_path


def mkdir_exp(*property):
    """
    :param property: 输入的是当前的文件名
    :return:文件的返回地址，最多支持二级地址的索引,然后就可以了
    """
    if len(property) == 1:
        property_path = os.path.join(os.getcwd(), property[0])
    else:
        property_path = os.path.join(os.getcwd(), property[0], property[1])
    if not os.listdir(property_path):
        cur_path = os.path.join(property_path, "exp1")
        os.mkdir(cur_path)
    else:
        cur_sequence = sorted(list(map(lambda x: int(x[3:]), os.listdir(property_path))))
        cur_path = os.path.join(property_path, "exp" + str(cur_sequence[-1] + 1))
        os.mkdir(cur_path)
    return cur_path


def save_ckpt(epoch, ckpt_path, trainer, filename=None):
    if filename:
        save_path_rgb = os.path.join(os.path.join(ckpt_path, "%s_ckpt_epoch%s_rgb.pth" % (filename, str(epoch))))
        save_path_thermal = os.path.join(
            os.path.join(ckpt_path, "%s_ckpt_epoch%s_thermal.pth" % (filename, str(epoch)))
        )
    else:
        save_path_rgb = os.path.join(os.path.join(ckpt_path, "ckpt_epoch%s_rgb.pth" % (str(epoch))))
        save_path_thermal = os.path.join(os.path.join(ckpt_path, "ckpt_epoch%s_thermal.pth" % (str(epoch))))
    checkpoint_rgb = {
        "net": trainer.visible.state_dict(),
        "optimizer": trainer.v_solver.state_dict(),
        "epoch": epoch,
        "lr_schedule": trainer.v_scheduler.state_dict(),
    }
    checkpoint_infrared = {
        "net": trainer.thermal.state_dict(),
        "optimizer": trainer.t_solver.state_dict(),
        "epoch": epoch,
        "lr_schedule": trainer.t_scheduler.state_dict(),
    }
    torch.save(checkpoint_rgb, save_path_rgb)
    torch.save(checkpoint_infrared, save_path_thermal)


class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
