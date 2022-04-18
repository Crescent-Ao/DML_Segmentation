classes = 9
lr_v = 1e-4
lr_t = 1e-4
dataset = '/home/guoshibo/ir_seg_dataset/'
momentum = 0.9
weight_decay = 1e-4
cps_flag = True
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
train_batch = 8
test_batch = 1
