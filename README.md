## 计算机视觉导论(PKU 2022 春)课程作业
- 代码和作业报告会在作业截止后上传. ~~(如果我没有中期退课的话)~~
- PDF 里的链接在 Github 上是点不了的, 下载到本地就能点了.

### Course projects of *Introduction to Computer Vision*, Peking University, 2022 Spring.
- Codes and report will be updated once the assignment is due. ~~(if I do not quit in the midterm, I mean)~~
- Hyperlinks in PDF file could not be clicked in Github preview page, so you may want to download it before happily accessing them.

---

### 作业 1
1. 滑动窗口卷积 & Toeplitz 矩阵 (禁止使用 for 循环的 Numpy 练习);
2. Canny 边缘检测;
3. Harris 角点检测;
4. RANSAC 算法在平面拟合的应用;
5. 多层感知机 (MLP) 的反向传播算法.

### Assignment 1
1. Convolution by sliding window & by Toeplitz matrix (for-loops are banned for Numpy practice);
2. Canny Edge Detector;
3. Harris Corner Detector;
4. RANSAC algorithm for plane-fitting;
5. Implement backpropagation for multi-layer perceptron (MLP).

---

### 作业 2
1. 实现 MLP 的 BatchNorm 层, 包括训练前向、测试前向和反向传播;
2. 在 CIFAR-10 数据集上搭建一个卷积神经网络, 并通过调参、数据增加等技巧提高网络性能;
3. 透视投影相机的矫正.

### Assignment 2
1. Implement BatchNorm layer for MLP, including train-time forward, test-time forward and backpropagation;
2. Build and train a CNN on CIFAR-10, and further improve it by tricks like tunning and data augmentation;
3. Camera calibration, solving the intrinsic matrix K and extrinsic matrix \[R, T\] of a perspective camera;


---

### 作业 3
1. 将一个透视投影的深度图转换为点云表示;
2. 使用两种方法 (均匀采样 & 远距离点采样) 在三角网面上进行采样 (构成点云); 使用 ｢推土机距离｣ 和 ｢Chamfer 距离｣ 衡量点云之间的距离;
3. 实现一种将隐式模型表示转换为三角网面表示的算法——Marching Cube 算法;
4. 实现一个经典的、可处理 3D 点云的深度学习模型——PointNet;
5. 应用 Mask RCNN 算法在自己定义的一个数据集上, 并使用 mAP 对模型效果进行评价;

### Assignment 3
1. Transform a depth image to a point cloud;
2. Implement Uniform-sampling & Furthest-point-sampling Algorithm for sampling on mesh; Use EMD & CD to measure distance between point clouds;
3. Implement marching-cube algorithm, which can transform the implicit representation of a 3D-model to mesh representation;
4. Implement the classic deeplearning model for 3D point-cloud learning: PointNet;
5. Finetune a Mask RCNN model to fit a dataset created by yourself and then use mAP to evaluate the performance of that model on your dataset;
