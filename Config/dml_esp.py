### 最简单的控制文件
from model.pspnet import Bottleneck,BasicBlock
classes = 9
visible = {
    'Block':Bottleneck,
    'Block_num':[3, 4, 23, 3],
    'T_0':20,
    'T_mult':2,
    'lambda_1':1,
    'lambda_2':1,
    'lambda_3':1,
}
thermal = {
    'Block':Bottleneck,
    'Block_num':[3, 4, 23, 3],
    'T_0':20,
    'T_mult':2,
    'lambda_1':1,
    'lambda_2':1,
    'lambda_3':1,
}
pool_scale = 5
train_batch = 8
test_batch = 1
