from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

# ----------TODO------------
# Implement the PointNet 
# ----------TODO------------

class PointNetFeat(nn.Module):
    """ classification 和 segmentation 的共同部分 """

    def __init__(self, *, for_dense_cls, global_feat_dim, need_visualize=False):
        """
        for_dense_cls: 是否是为了 dense 的 (即逐点的) classification?
            - 如果是, 那么需要把 local_feature 和 global_feature 拼在一起
            - 否过不是, 那么只需要单纯返回 global_feature 即可

        global_feat_dim: global_feature 的维度.
            - dense_classification 时仅支持 1024;
            - classification 时支持 256 或 1024.
        """

        if for_dense_cls: assert global_feat_dim in {1024}
        else: assert global_feat_dim in {256, 1024}

        super(PointNetFeat, self).__init__()
        self.global_feat_dim = global_feat_dim
        self.for_dense_cls = for_dense_cls
        self.need_vis = need_visualize

        # MLP (3 -> 64 -> 128 -> global_feat_dim)
        self.l0 = nn.Linear(in_features=3, out_features=64)
        self.l1 = nn.Linear(in_features=64, out_features=128)
        self.l2 = nn.Linear(in_features=128, out_features=self.global_feat_dim)


    def forward(self, x: torch.Tensor):
        batch_size, _, n_points = x.size()
        x = x                       # -> (BatchSize, 3, nPoints)
        x = x.transpose(1, 2)       # -> (BatchSize, nPoints, 3)
        x = torch.relu(self.l0(x))
        local_feat = x              # -> (BatchSize, nPoints, 64)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))  # -> (BatchSize, nPoints, self.global_feat_dim)
        vis_feat = torch.max(x, 2)[0]                                   # -> (BatchSize, nPoints)
        global_feat = torch.max(x, 1, keepdim=True)[0]                  # -> (BatchSize, 1, self.global_feat_dim)
        if self.for_dense_cls:
            # 把 local_feature 和 global_feature 拼在一起再 return
            local_feat = local_feat                                     # -> (BatchSize, nPoints, 64)
            repeat_global_feat = global_feat.repeat(1, n_points, 1)     # -> (BatchSize, nPoints, self.global_feat_dim)
            return torch.cat([local_feat, repeat_global_feat], dim=2)   # -> (BatchSize, nPoints, self.global_feat_dim + 64)
        else:
            # 直接 return global_feature 即可.
            return global_feat if not self.need_vis else (global_feat, vis_feat)



class PointNetCls256D(nn.Module):
    """ classification (global feature dimension = 256) """

    def __init__(self, k=2):
        super(PointNetCls256D, self).__init__()
        self.feat = PointNetFeat(for_dense_cls=False, global_feat_dim=256)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),   nn.ReLU(),
            nn.Linear(in_features=128, out_features=k),
        )

    def forward(self, x):                           # x: (BatchSize, 3, nPoints)
        global_feat = self.feat(x)                  # -> (BatchSize, 1, 256)
        global_feat = torch.squeeze(global_feat, 1) # -> (BatchSize, 256)
        output_scores = self.mlp(global_feat)       # -> (BatchSize, k)
        return F.log_softmax(output_scores, dim=1)  # -> (BatchSize, k)



class PointNetCls1024D(nn.Module):
    """ classification (global feature dimension = 1024) """

    def __init__(self, k=2, need_visualize=False):
        super(PointNetCls1024D, self).__init__()
        self.need_visualize = need_visualize
        self.feat = PointNetFeat(for_dense_cls=False, global_feat_dim=1024, need_visualize=True)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),  nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),   nn.ReLU(),
            nn.Linear(in_features=256, out_features=k),
        )

    def forward(self, x):                           # x: (BatchSize, 3, nPoints)
        global_feat, vis = self.feat(x)             # -> (BatchSize, 1, 1024)
        global_feat = torch.squeeze(global_feat, 1) # -> (BatchSize, 1024)
        output_scores = self.mlp(global_feat)       # -> (BatchSize, k)
        if not self.need_visualize:
            return F.log_softmax(output_scores, dim=1), None  # -> (BatchSize, k)
        else:
            return F.log_softmax(output_scores, dim=1), vis




class PointNetDenseCls(nn.Module):
    """ segmentation (dense classification) (global feature dimension = 1024) """

    def __init__(self, k=2):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feat = PointNetFeat(for_dense_cls=True, global_feat_dim=1024)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=1088, out_features=512), nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),  nn.ReLU(),
            nn.Linear(in_features=256, out_features=128), nn.ReLU(),
            nn.Linear(in_features=128, out_features=k),
        )

    def forward(self, x):       # x: (BatchSize, 3, nPoints)
        feat = self.feat(x)     # -> (BatchSize, nPoints, 1024 + 64)
        output_scores = self.mlp(feat)   # -> (BatchSize, nPoints, k)
        return F.log_softmax(output_scores, dim=-1)  # -> (BatchSize, nPoints, k)
