from tkinter import Label
import utils
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNN
from dataset import SingleShapeDataset
from utils import plot_save_output
import torch
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt


dataset_test = SingleShapeDataset(10)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

num_classes = 4

# get the model using the helper function
model = utils.get_instance_segmentation_model(num_classes).double()

device = torch.device('cpu')

# replace the 'cpu' to 'cuda' if you have a gpu
model.load_state_dict(torch.load("./results/maskrcnn_2.pth", map_location='cpu'))

model.eval()
path = "results/"

# save visual results
for i in range(10):
    imgs, labels = dataset_test[i]
    output = model([imgs])
    plot_save_output(path + str(i) + "_result.png", imgs, output[0])


#================================================================================#
# compute AP
# Reference: https://zhuanlan.zhihu.com/p/399837729

def calc_IoU_for_boxes(box1, box2) -> float:
    # box: [y1, x1, y2, x2], 其中 y2 > y1 且 x2 > x1
    y1 = max(box1[0], box2[0])
    y2 = min(box1[2], box2[2])
    x1 = max(box1[1], box2[1])
    x2 = min(box1[3], box2[3])
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    union = area1 + area2 - intersection
    return float(intersection / union)

def calc_IoU_for_masks(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    # mask: uint8, only 0u and 1u
    return float((mask1 * mask2).sum() / torch.count_nonzero(mask1 + mask2))


# the outputs includes: 'boxes', 'labels', 'masks', 'scores'
def compute_detection_ap(output_list, gt_list) -> float:
    # for detection: 衡量 box 的 IoU
    APs = []
    for cls in [1, 2, 3]:
        """ 对于固定的类别 cls:
            1. 统计这一组图中, 包含的类别为 cls 的全部物体 (的物体编号).
               其中, 假设第 i 张图的第 j 个物品是 cls 类的, 记 f"{i}-{j}" 为这个物体的 ｢物体编号｣.
            2. 统计每个 ｢预测类别为 cls｣ 的预测的 ｢物体编号｣、｢confidence｣、｢IoU｣.
        """
        # 1.
        objects = []
        for im_idx, gt in enumerate(gt_list):
            for obj_idx, label in enumerate(gt["labels"]):
                if label == cls:
                    objects.append(f"{im_idx}-{obj_idx}")
        if len(objects) == 0:   # gt 中没有这一 cls 的物体...
            continue

        # 2.
        # [注: 以下针对题意进行了限定: 每个图片的 ground_truth 中最多包含一个类别为 cls 的物体! 因为用的是 Single 数据集!]
        stat = []
        for im_idx, pred in enumerate(output_list):
            for obj_idx, label in enumerate(pred["labels"]):
                if label == cls:
                    # 计算这个预测的 IoU 和相应的 ｢物体编号｣
                    GT_ID = "这张图上不存在的物体!"
                    IoU = 0
                    IoU_ge_0_5 = False

                    gt = gt_list[im_idx]
                    if gt["labels"][0] == cls:
                        GT_ID = f"{im_idx}-{0}"
                        IoU = calc_IoU_for_boxes(gt["boxes"][0], pred["boxes"][obj_idx])
                        IoU_ge_0_5 = (IoU >= 0.5)

                    stat.append({
                        "GT_ID": GT_ID,
                        "Confidence": float(pred["scores"][obj_idx]),
                        "IoU_for_debug": IoU,
                        "IoU >= 0.5": IoU_ge_0_5,
                    })

        stat.sort(key=lambda d: d["Confidence"], reverse=True)      # 按 Confidence 排序
        print("当前物体类别为", cls)
        for _ in stat: print(_)

        # 计算 Precision-Recall
        P_on_R = dict()
        for length in range(1, len(stat) + 1):
            predictions = list(filter(lambda d: d["IoU >= 0.5"], stat[:length]))    # confidence 前 length 高中, IoU >= 0.5 的那些预测
            precision = len(predictions) / length
            recall = len(set(map(lambda d: d["GT_ID"], predictions)) - {"这张图上不存在的物体!"}) / len(objects)
            P_on_R[recall] = max(P_on_R.get(recall, 0), precision)

        if len(P_on_R) == 0:
            APs.append(0)
            continue

        RPs = sorted(list(P_on_R.items()), key=lambda pair: pair[0])
        Rs = list(zip(*RPs))[0]
        Ps = list(zip(*RPs))[1]

        print("RPs =", RPs)
        samples = np.linspace(0.0, 1.0, 11)
        PR_curve = np.interp(samples, Rs, Ps)
        print("PR曲线为:", PR_curve)
        plt.plot(samples, PR_curve)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.show()

        APs.append(np.mean(PR_curve))

    mAP_det = np.mean(APs)
    return float(mAP_det)


def compute_segmentation_ap(output_list, gt_list) -> float:
    # for segmentation: 衡量 mask 的 IoU
    APs = []
    for cls in [1, 2, 3]:
        """ 对于固定的类别 cls:
            1. 统计这一组图中, 包含的类别为 cls 的全部物体 (的物体编号).
               其中, 假设第 i 张图的第 j 个物品是 cls 类的, 记 f"{i}-{j}" 为这个物体的 ｢物体编号｣.
            2. 统计每个 ｢预测类别为 cls｣ 的预测的 ｢物体编号｣、｢confidence｣、｢IoU｣.
        """
        # 1.
        objects = []
        for im_idx, gt in enumerate(gt_list):
            for obj_idx, label in enumerate(gt["labels"]):
                if label == cls:
                    objects.append(f"{im_idx}-{obj_idx}")
        if len(objects) == 0:   # gt 中没有这一 cls 的物体...
            continue

        # 2.
        # [注: 以下针对题意进行了限定: 每个图片的 ground_truth 中最多包含一个类别为 cls 的物体! 因为用的是 Single 数据集!]
        stat = []
        for im_idx, pred in enumerate(output_list):
            for obj_idx, label in enumerate(pred["labels"]):
                if label == cls:
                    # 计算这个预测的 IoU 和相应的 ｢物体编号｣
                    GT_ID = "这张图上不存在的物体!"
                    IoU = 0
                    IoU_ge_0_5 = False

                    gt = gt_list[im_idx]
                    if gt["labels"][0] == cls:
                        GT_ID = f"{im_idx}-{0}"
                        IoU = calc_IoU_for_masks(gt["masks"][0], pred["masks"][obj_idx])
                        IoU_ge_0_5 = (IoU >= 0.5)

                    stat.append({
                        "GT_ID": GT_ID,
                        "Confidence": float(pred["scores"][obj_idx]),
                        "IoU_for_debug": IoU,
                        "IoU >= 0.5": IoU_ge_0_5,
                    })

        stat.sort(key=lambda d: d["Confidence"], reverse=True)      # 按 Confidence 排序
        print("当前物体类别为", cls)
        for _ in stat: print(_)

        # 计算 Precision-Recall
        P_on_R = dict()
        for length in range(1, len(stat) + 1):
            predictions = list(filter(lambda d: d["IoU >= 0.5"], stat[:length]))    # confidence 前 length 高中, IoU >= 0.5 的那些预测
            precision = len(predictions) / length
            recall = len(set(map(lambda d: d["GT_ID"], predictions)) - {"这张图上不存在的物体!"}) / len(objects)
            P_on_R[recall] = max(P_on_R.get(recall, 0), precision)

        if len(P_on_R) == 0:
            APs.append(0)
            continue

        RPs = sorted(list(P_on_R.items()), key=lambda pair: pair[0])
        Rs = list(zip(*RPs))[0]
        Ps = list(zip(*RPs))[1]

        print("RPs =", RPs)

        samples = np.linspace(0.0, 1.0, 11)
        PR_curve = np.interp(samples, Rs, Ps)
        print("PR曲线为:", PR_curve)
        plt.plot(samples, PR_curve)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.show()

        APs.append((np.mean(PR_curve), ))

    mAP_seg = np.mean(APs)

    return float(mAP_seg)




gt_labels_list = []
output_label_list = []
with torch.no_grad():
    for i in range(100):
        print(i)
        imgs, labels = dataset_test[i]
        gt_labels_list.append(labels)
        output = model([imgs])
        output_label_list.append(output[0])
mAP_detection = compute_detection_ap(output_label_list, gt_labels_list)
mAP_segmentation = compute_segmentation_ap(output_label_list, gt_labels_list)

np.savetxt(path + "mAP.txt", np.asarray([mAP_detection, mAP_segmentation]))
print([mAP_detection, mAP_segmentation])
