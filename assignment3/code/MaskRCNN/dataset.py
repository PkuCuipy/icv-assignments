import os
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import random
import cv2
import math
from utils import plot_save_dataset


class SingleShapeDataset(torch.utils.data.Dataset):
    def __init__(self, size):

        self.w = 128
        self.h = 128
        self.size = size
        print("size", self.size)

    def _draw_shape(self, img, mask, shape_id):
        buffer = 20
        y = random.randint(buffer, self.h - buffer - 1)
        x = random.randint(buffer, self.w - buffer - 1)
        s = random.randint(buffer, self.h // 4)
        color = tuple([random.randint(0, 255) for _ in range(3)])

        if shape_id == 1:
            cv2.rectangle(mask, (x - s, y - s), (x + s, y + s), 1, -1)
            cv2.rectangle(img, (x - s, y - s), (x + s, y + s), color, -1)

        elif shape_id == 2:
            cv2.circle(mask, (x, y), s, 1, -1)
            cv2.circle(img, (x, y), s, color, -1)

        elif shape_id == 3:
            points = np.array([[(x, y - s),
                                (x - s / math.sin(math.radians(60)), y + s),
                                (x + s / math.sin(math.radians(60)), y + s),
                                ]], dtype=np.int32)
            cv2.fillPoly(mask, points, 1)
            cv2.fillPoly(img, points, color)

    def __getitem__(self, idx):
        np.random.seed(idx)

        n_class = 1
        masks = np.zeros((n_class, self.h, self.w))
        img = np.zeros((self.h, self.w, 3))
        img[..., :] = np.asarray([random.randint(0, 255) for _ in range(3)])[None, None, :]

        obj_ids = np.zeros((n_class))

        shape_code = random.randint(1, 3)
        self._draw_shape(img, masks[0, :], shape_code)
        obj_ids[0] = shape_code

        boxes = np.zeros((n_class, 4))
        pos = np.where(masks[0])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes[0, :] = np.asarray([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        img = torch.tensor(img)
        img = img.permute(2, 0, 1)

        return img, target

    def __len__(self):
        return self.size


# ----------TODO------------
# Implement ShapeDataset.
# ----------TODO------------

class ShapeDataset(torch.utils.data.Dataset):

    def __init__(self, size):
        self.w = 128
        self.h = 128
        self.size = size
        print("size", self.size)

    def __getitem__(self, idx):
        np.random.seed(idx)

        # 随机背景色的图片
        img = np.zeros((self.h, self.w, 3))
        bg_color = np.random.randint(0, 255, size=[1, 1, 3])
        img[:] = bg_color

        # 画几个形状?
        n_shapes = np.random.randint(1, 3)

        # 开始生成每个形状的参数 (但先不绘制, 因为某些形状也许会被 NMS 机制剔除)
        shapes = []
        for i in range(n_shapes):
            type_, color, (y, x, s) = self.random_shape()
            shapes.append((
                type_,                          # [0]: 什么形状? 三角形? 圆? 正方形?
                color,                          # [1]: 什么颜色? RGB
                (y, x, s),                      # [2]: 中心点坐标, 图形尺寸
                (y - s, x - s, y + s, x + s),   # [3]: 包围盒的左上和右下坐标
                (2 * s) ** 2                    # [4]: 包围盒面积
            ))

        # 执行 NMS, 去除被严重重叠遮挡 (从而 IoU 会比较高) 的图形
        IoU_threashold = 0.3
        shapes_after_NMS = []
        while shapes:
            # 取出一个形状添加到 saN[] 中
            shape = shapes.pop()
            shapes_after_NMS.append(shape)
            # 计算剩余形状和这个形状的 IoU, 如果小于阈值, 就留下, 否则不留下
            left = []
            for s in shapes:
                if self.calc_IoU(shape, s) <= IoU_threashold:
                    left.append(s)
            shapes = left

        # 绘制 shapes_after_NMS[]
        n_shapes = len(shapes_after_NMS)
        masks = np.zeros((n_shapes, self.h, self.w))    # 该图形对应的 mask
        boxes = np.zeros((n_shapes, 4))                 # 该图形的包围盒
        area = np.zeros((n_shapes, ))                   # 该图形的包围盒面积
        labels = np.zeros((n_shapes, ))                 # 该图形的形状类型
        for i, shape in enumerate(reversed(shapes_after_NMS)):  # 靠后的是最下层的, 因此要先画! 所以要先翻转!
            self.draw_shape(shape, img, masks, i)
            boxes[i] = np.array([shape[3][1], shape[3][0], shape[3][3], shape[3][2]])
            area[i] = shape[4]
            labels[i] = shape[0]

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(masks, dtype=torch.uint8),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(area, dtype=torch.int32),
        }
        img = torch.tensor(img)
        img = img.permute(2, 0, 1)

        return img, target

    def __len__(self):
        return self.size

    def random_shape(self):
        shape = np.random.randint(1, 3)                     # 随机指定形状类型
        color = np.random.randint(0, 255, (3, ))            # 随机指定颜色
        buffer = 20
        y = np.random.randint(buffer, self.h - buffer - 1)  # 随机指定中心点位置 (x, y)
        x = np.random.randint(buffer, self.w - buffer - 1)
        s = np.random.randint(buffer, self.h // 4)          # 随机指定大小 s
        return shape, color, (y, x, s)

    def calc_IoU(self, shape1, shape2):
        box1, area1 = shape1[3:]
        box2, area2 = shape2[3:]
        # box: [y1, x1, y2, x2], 其中 y2 > y1 且 x2 > x1
        y1 = max(box1[0], box2[0])
        y2 = min(box1[2], box2[2])
        x1 = max(box1[1], box2[1])
        x2 = min(box1[3], box2[3])
        intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
        union = area1 + area2 - intersection
        return intersection / union

    def draw_shape(self, shape, img, masks, which):
        type_, color, (y, x, s) = shape[:3]
        # 在 img[][] 上绘制图形, 在 mask[][] 上标记这些像素为 True
        if type_ == 1:       # 1. 正方形
            # 绘制自己的 mask
            cv2.rectangle(masks[which, :], (x - s, y - s), (x + s, y + s), 1, -1)
            for i in range(which):  # 吃掉自己底下的形状的 mask
                cv2.rectangle(masks[i, :], (x - s, y - s), (x + s, y + s), 0, -1)
            # 绘制自己的图形
            color = tuple(color.tolist())
            cv2.rectangle(img, (x - s, y - s), (x + s, y + s), color, -1)

        elif type_ == 2:     # 2. 圆形
            # 绘制自己的 mask
            cv2.circle(masks[which, :], (x, y), s, 1, -1)
            for i in range(which):      # 吃掉自己底下的形状的 mask
                cv2.circle(masks[i, :], (x, y), s, 0, -1)
            # 绘制自己的图形
            color = tuple(color.tolist())
            cv2.circle(img, (x, y), s, color, -1)

        elif type_ == 3:     # 3. 三角形
            points = np.array([[
                (x, y - s),
                (x - s / math.sin(math.radians(60)), y + s),
                (x + s / math.sin(math.radians(60)), y + s)]], dtype=np.int32)
            # 绘制自己的 mask
            cv2.fillPoly(masks[which, :], points, 1)
            for i in range(which):      # 吃掉自己底下的形状的 mask
                cv2.fillPoly(masks[i, :], points, 0)
            # 绘制自己的图形
            color = tuple(color.tolist())
            cv2.fillPoly(img, points, color)


if __name__ == '__main__':
    dataset = ShapeDataset(size=10)
    path = "./results/"
    for i in range(10):
        imgs, labels = dataset[i]
        print(labels)
        plot_save_dataset(path + str(i) + "_data.png", imgs, labels)

