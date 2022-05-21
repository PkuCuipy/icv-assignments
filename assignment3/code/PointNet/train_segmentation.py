from __future__ import print_function
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetSegmentationDataset
from model import PointNetDenseCls
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils import log_writter, setting

if __name__ == '__main__':

    opt = setting()
    writter = log_writter(opt.out_folder, "seg_1024D")

    random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    dataset = ShapeNetSegmentationDataset(
        root=opt.dataset_folder,
        npoints=opt.n_points,
        class_choice=["Airplane"])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.n_workers))

    test_dataset = ShapeNetSegmentationDataset(
        root=opt.dataset_folder,
        npoints=opt.n_points,
        class_choice=["Airplane"],
        split='test',
        with_data_augmentation=False)
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.n_workers))

    print(len(dataset), len(test_dataset))
    num_classes = dataset.num_seg_classes
    print('classes', num_classes)

    blue = lambda x: '\033[94m' + x + '\033[0m'

    classifier = PointNetDenseCls(k=num_classes)

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    num_batch = len(dataset) / opt.batch_size
    count = 0
    for epoch in range(opt.n_epoch):

        for i, data in enumerate(dataloader, 0):
            points, target = data
            points = points.transpose(2, 1)
            optimizer.zero_grad()
            classifier = classifier.train()
            pred = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0] - 1
            # print(pred.size(), target.size())
            loss = F.nll_loss(pred, target)
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (
            epoch, i, num_batch, loss.item(), correct.item() / float(opt.batch_size * 1024)))
            writter.add_train_scalar("Loss", loss.item(), count)
            writter.add_train_scalar("Acc", correct.item() / float(opt.batch_size * 1024), count)

            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points = points.transpose(2, 1)
                classifier = classifier.eval()
                pred = classifier(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0] - 1
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
                epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(opt.batch_size * 1024)))
                writter.add_test_scalar("Loss", loss.item(), count)
                writter.add_test_scalar("Acc", correct.item() / float(opt.batch_size * 1024), count)

            count += 1
        # torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))
        scheduler.step()

    ## benchmark mIOU
    shape_ious = []
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        points = points.transpose(2, 1)
        classifier = classifier.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(2)[1]

        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy() - 1

        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)  # np.unique(target_np[shape_idx])
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))

    print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))
