import os

classes = 9
lr_v = 1e-4
lr_t = 1e-4
dataset = os.path.join(os.path.join(os.getcwd(), ".."), "ir_seg_dataset")
momentum = 0.9
weight_decay = 1e-4
cps_flag = False
ckpt_freq = 5
self_branch_epochs = 50
DML_epochs = 100
multi_scale = True
KD_temperature = 3
# 新增支持CWD
CWD = {
    "norm_type": "channel",
    "divergence": "kl",
    "temperature": "3",
}
visible = {
    "Block": "Bottleneck",
    "Block_num": [3, 4, 23, 3],
    "T_0": 20,
    "T_mult": 2,
    "lambda_1": 1,
    "lambda_2": 1,
    "lambda_3": 1,
}
thermal = {
    "Block": "Bottleneck",
    "Block_num": [3, 4, 23, 3],
    "T_0": 20,
    "T_mult": 2,
    "lambda_1": 1,
    "lambda_2": 1,
    "lambda_3": 1,
}
pool_scale = 5
train_batch = 4
test_batch = 1
# 这个是随机裁剪之后的尺度的高宽
train_sr = {"height": 320, "width": 320}
test_sr = {"height": 480, "width": 640}
