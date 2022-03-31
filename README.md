# Deep Mutual Learning for Segmentation
这个是互学习机制在跨域数据集上的探索，常见的跨模态融合的方式，无非以下三种方式
- 第一种 RGB+Thermal 直接在输入维度上进行concat的操作的方式
- 第二种 红外端 可见光端放上两个骨架网络作为特征提取的手段，然后再利用中间某个模块相融合
- 第三种 RGB Thermal 两个网络相互解耦合，然后对两个网络中的输出进行决策，属于结果融合一种
目前，跨模态融合的算法大致分为以上三种方式，能不能提出另外一种新的范式，针对上述跨模态数据,我们采用互学习机制，
加上对抗学习策略进行交叉重建的方式，通过设计不同的网络，将可见光蒸馏的知识，通过logits方式传递给红外的手段，通过这种方式，我们期望得到的结果如下
- 可见光和红外光在新的范式下，各自的指标都有所提升
- 可见光通过红外机制的引导，在夜间的情况下也能获取较好的分割效果
- 红外光通过可见光较好地的纹理细节信息，也能够在部分指标中得到相应的提升
## Cfg
cfg 最简单的配置文件，目前已支持.py文件的解析功能，全局解析 3.28完成
## logger
log 也是采用mmcv中方式，
## Dataset
