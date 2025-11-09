import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import sys
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

'''
固定大部分随机性，尽量复现实验。
deterministic=True 让 cuDNN 选择确定性算法；benchmark=True 会根据输入大小选择最快算法，这两者在某些场景矛盾，一般选择其一；你这里测试阶段影响不大。
'''
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



'''
输入：二维标签图 H×W（每个像素是类 ID：0,1,2,3,4,5）。
输出：H×W×3 的 RGB 可视化。
注意：cv2.imwrite 期望 BGR。你这里写的是 RGB 值，如果直接 imwrite，颜色会偏。
解决：保存前 mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)，或者直接把调色板按 BGR 写。
'''
def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


"""
单图保存任务（供多进程调用）
inp 是一个三元组：预测的 mask、输出基名 mask_id（不含扩展名）、以及是否着色 rgb。
rgb=True：先 label2rgb()，再写 PNG。
rgb=False：直接写单通道标签 PNG。

"""

#这个函数是多进程调用，将预测结果保存为RGB图像还是保存为单通道标签图。
def img_writer(inp):
    (mask,  mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label2rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)    # 保存 RGB 图像
    else:
        mask_png = mask.astype(np.uint8)    ## 转换为单通道图像
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)    # # 保存为单通道标签图



#命令行参数
'''
不着色：python infer.py -c ./config/potsdam/AAAANet.py -o ./runs/pred/
开 TTA、且保存彩色图：python infer.py -c ... -o ... -t d4 --rgb
'''
# def get_args():
#     parser = argparse.ArgumentParser()
#     arg = parser.add_argument
#     arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
#     arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
#     arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
#     arg("--rgb", help="whether output rgb images", action='store_true')
#     return parser.parse_args()
def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # 定义参数
    arg("-c", "--config_path", type=Path, required=False, help="Path to config")
    arg("-o", "--output_path", type=Path, required=False, help="Path where to save resulting masks.")
    arg("-t", "--tta", default=None, choices=[None, "d4", "lr"], help="Test time augmentation.")
    arg("--rgb", help="Whether output RGB images", action='store_true')

    # 如果没有传参，则使用默认值
    if len(sys.argv) == 1:  # 检测是否有传入参数，如果没有则使用默认值
        return parser.parse_args([
            "-c", "./config/potsdam/AAAANet.py",  # 默认配置路径
            "-o", "./results/potsdam/AAAA_11/", # 默认输出路径
            "--rgb"  #如果需要输出彩色图像，则添加 --rgb 参数。
        ])
    else:
        return parser.parse_args()


"""
main：核心流程
py2cfg：把 Python 配置文件加载成对象；里面应提供模型权重、测试数据集等字段。

"""
def main():
    args = get_args()
    seed_everything(42)

    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True) #确保输出目录存在。

    #模型加载 & TTA
    model = Supervision_Train.load_from_checkpoint(     #用 Lightning 的 load_from_checkpoint 恢复 LightningModule（含模型结构等）。
        os.path.join(config.weights_path, config.test_weights_name + '.ckpt'), config=config)
    model.cuda()
    model.eval()
    evaluator = Evaluator(num_class=config.num_classes) #初始化评估器（计算 IoU/F1/OA）。
    evaluator.reset()
    if args.tta == "lr":        #左右（以及上下）翻转集成。
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":      #上下/左右翻转 + 多尺度（0.75, 1.0, 1.25, 1.5）。
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                # tta.Rotate90(angles=[90]),
                tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
        #SegmentationTTAWrapper
        """
        对输入做一组增强，分别前向，反变换回原尺度，聚合（默认平均）。
        """

    #数据加载
    test_dataset = config.test_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,       #测试数据集
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions = model(input['img'].cuda())

            image_ids = input["img_id"]
            masks_true = input['gt_semantic_seg']

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)    #每像素类别概率（按通道 C）；Softmax → argmax 得到离散标签图。
            predictions = raw_predictions.argmax(dim=1) #predictions 是 LongTensor；写图前转 numpy/uint8。

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())   #把预测与 GT 喂进去，内部会累计混淆矩阵或 TP/FP/FN 之类的计数。
                mask_name = image_ids[i]
                results.append((mask, str(args.output_path / mask_name), args.rgb)) #把要写的内容攒起来，最后统一多进程写磁盘。


    #指标汇总
    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()
    for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
        print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
    print('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), OA))    #通常忽略最后一个类（很多数据集把最后一类作为“背景/忽略”，看你的配置而定）。
    #多进程写图
    # t0 = time.time()
    # mpp.Pool(processes=mp.cpu_count()).map(img_writer, results) #用 CPU 核心数开的进程池对 results 逐项调用 img_writer。
    # t1 = time.time()
    # img_write_time = t1 - t0
    # print('images writing spends: {} s'.format(img_write_time))
    #改成了单进程写图,在Windows中，笔记本内存不足。
    t0 = time.time()
    for r in results:
        img_writer(r)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends (serial): {:.2f} s'.format(img_write_time))


if __name__ == "__main__":
    main()
