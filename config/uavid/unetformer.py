"""
UnetFormer for uavid datasets with supervision training
Libo Wang, 2022.02.22
"""
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.uavid_dataset import *
from geoseg.models.UNetFormer import UNetFormer
from tools.utils import Lookahead
from tools.utils import process_model_params

# training hparam
max_epoch = 40  #max_epoch: 设置训练的最大轮数
ignore_index = 255  #用于指定在计算损失时忽略的标签（通常用于背景类或无效像素）。
train_batch_size = 8
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01     # 权重衰减
backbone_lr = 6e-5  #backbone_lr 和 backbone_weight_decay: 用于预训练网络的学习率和权重衰减。
backbone_weight_decay = 0.01    #backbone_lr 和 backbone_weight_decay: 用于预训练网络的学习率和权重衰减。
num_classes = len(CLASSES)
classes = CLASSES   #目标类别数目

weights_name = "unetformer-r18-1024-768crop-e40"    #weights_name: 设置模型权重的文件名，包含模型结构和其他训练细节。
weights_path = "model_weights/uavid/{}".format(weights_name)    #weights_path: 权重文件保存的路径，包含 weights_name。
test_weights_name = "last"  #test_weights_name: 测试时加载的权重名称，默认为 "last"。
log_name = 'uavid/{}'.format(weights_name)  #log_name: 日志文件保存的路径，包含 weights_name。
monitor = 'val_mIoU'    #用于监控的指标，这里监控 val_mIoU（验证集的平均交并比）。
monitor_mode = 'max'    #monitor_mode: 设置为 'max'，表示最大化 val_mIoU，即模型将根据验证集的 mIoU 最优值保存。
save_top_k = 1  #save_top_k: 设置为 1，表示只保存最好的模型。
save_last = True    #save_last: 设置为 True，表示保存训练过程中的最后一个模型。
check_val_every_n_epoch = 10  #check_val_every_n_epoch: 每训练一个 epoch，就进行一次验证。
pretrained_ckpt_path = None # pretrained_ckpt_path: 如果有预训练的模型权重，设置这个路径来加载预训练的权重。
gpus = 'auto'  # default or gpu ids:[0] or gpu nums:  2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # resume_ckpt_path: 如果训练过程中需要继续训练，设置该路径。

#  define the network
net = UNetFormer(num_classes=num_classes)
# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)    #ignore_index 参数用于指示应该忽略的标签，在计算损失时不会对该标签的像素进行梯度更新。

use_aux_loss = True

# define the dataloader

train_dataset = UAVIDDataset(data_root='data/uavid/train_val', img_dir='images', mask_dir='masks',
                             mode='train', mosaic_ratio=0.25, transform=train_aug, img_size=(1024, 1024))
#mosaic_ratio：训练时的数据增强策略，mosaic_ratio=0.25 表示使用 25% 的概率进行 mosaic 增强。

val_dataset = UAVIDDataset(data_root='data/uavid/val', img_dir='images', mask_dir='masks', mode='val',
                           mosaic_ratio=0.0, transform=val_aug, img_size=(1024, 1024))
#val_dataset：验证集的数据集对象，用于验证模型性能。


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,      #提高数据加载效率，特别是在使用 GPU 时。
                          shuffle=True,     #是否打乱
                          drop_last=False)   #drop_last=True 和 drop_last=False：控制是否丢弃最后一批数据（如果该批数据不足 batch_size）。

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
#layerwise_params：定义了不同层次的学习率和权重衰减。backbone.* 指示对骨干网络（backbone）使用不同的学习率和权重衰减。
net_params = process_model_params(net, layerwise_params=layerwise_params)
#process_model_params()：该函数用于处理模型参数，将其分为不同的组，以便为不同的层分配不同的学习率和权重衰减。
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
#base_optimizer：使用 AdamW 作为基础优化器
optimizer = Lookahead(base_optimizer)
#optimizer：使用 Lookahead 优化器，它是在 base_optimizer 基础上增加了 Lookahead 技术来加速收敛。
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
#使用余弦退火学习率调度器 CosineAnnealingLR，可以在训练过程中动态调整学习率。

