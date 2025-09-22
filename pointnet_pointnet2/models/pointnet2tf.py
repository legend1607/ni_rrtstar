import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_pointnet2.models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation


class PositionalEncodingLearned(nn.Module):
    """位置编码 (Positional Encoding) 模块，用于把点的坐标 (x,y,z) 转成 Transformer 可以利用的 embedding"""
    def __init__(self, in_dim=3, out_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, coords):
        # coords: (B, 3, N) -> (B, N, 3)
        coords = coords.permute(0, 2, 1)
        return self.fc(coords)  # (B, N, out_dim)


class PointTransformerModule(nn.Module):
    """对输入点的 全局特征 做 Transformer 注意力建模"""
    def __init__(self, feature_dim=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=nhead,
            batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x):
        # x: (B, N, C)
        x_out = self.transformer(x)
        return self.mlp(x_out) + x


class get_model(nn.Module):
    def __init__(self, num_classes, coord_dim=3, in_channels=3, use_dir_head=False):
        super(get_model, self).__init__()
        self.coord_dim = coord_dim
        self.use_dir_head = use_dir_head

        # === PointNet++ backbone ===
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32],
            in_channel=in_channels, mlp_list=[[16, 16, 32], [32, 32, 64]],
            coord_dim=coord_dim
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=256, radius_list=[0.1, 0.2], nsample_list=[16, 32],
            in_channel=32+64, mlp_list=[[64, 64, 128], [64, 96, 128]],
            coord_dim=coord_dim
        )
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=64, radius_list=[0.2, 0.4], nsample_list=[16, 32],
            in_channel=128+128, mlp_list=[[128, 196, 256], [128, 196, 256]],
            coord_dim=coord_dim
        )
        self.sa4 = PointNetSetAbstractionMsg(
            npoint=16, radius_list=[0.4, 0.8], nsample_list=[16, 32],
            in_channel=256+256, mlp_list=[[256, 256, 512], [256, 384, 512]],
            coord_dim=coord_dim
        )

        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        # ===新增 Transformer on global features ===
        self.feature_dim = 512  # match sa4 output
        self.pos_enc = PositionalEncodingLearned(in_dim=coord_dim, out_dim=self.feature_dim)
        self.transformer = PointTransformerModule(feature_dim=self.feature_dim, nhead=4, num_layers=2)

        # === Heads ===
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

        if self.use_dir_head:
            self.dir_head = nn.Sequential(
                nn.Conv1d(128, 64, 1),
                nn.ReLU(),
                nn.Conv1d(64, coord_dim, 1),
            )

    def forward(self, xyz):
        # xyz: (B, n_features, N)
        l0_points = xyz
        l0_xyz = xyz[:, :self.coord_dim, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B, C, 16)

        # === 新增Transformer at global tokens ===
        pos = self.pos_enc(l4_xyz)         # (B, 16, C)
        feat = l4_points.permute(0, 2, 1)  # (B, 16, C)
        feat = feat + pos
        feat = self.transformer(feat)      # (B, 16, C)
        l4_points = feat.permute(0, 2, 1)  # (B, C, 16)

        # === Feature propagation ===
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        # === Classification ===
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1).permute(0, 2, 1)  # (B, N, num_classes)

        if self.use_dir_head:
            d = self.dir_head(l0_points).permute(0, 2, 1)  # (B, N, 3)
            d = F.normalize(d, p=2, dim=-1)
            return x, l4_points, d
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self, use_dir_head=False, lambda_dir=0.1):
        super(get_loss, self).__init__()
        self.use_dir_head = use_dir_head
        self.lambda_dir = lambda_dir

    def forward(self, pred, target, trans_feat, weight, dir_pred=None, gt_dir=None):
        # 分类 loss
        total_loss = F.nll_loss(pred, target, weight=weight)

        # 方向 loss
        if self.use_dir_head and dir_pred is not None and gt_dir is not None:
            dir_loss = F.cosine_embedding_loss(
                dir_pred.view(-1, 3),
                gt_dir.view(-1, 3),
                torch.ones(dir_pred.shape[0]*dir_pred.shape[1], device=dir_pred.device),
                reduction="mean"
            )
            total_loss = total_loss + self.lambda_dir * dir_loss

        return total_loss
