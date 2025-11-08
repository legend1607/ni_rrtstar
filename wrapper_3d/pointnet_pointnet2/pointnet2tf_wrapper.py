from os.path import join

import torch
import numpy as np

from pointnet_pointnet2tf.models.pointnet2tf import get_model
from pointnet_pointnet2tf.models.pointnet2_utils import pc_normalize

class PNGWrapper:
    def __init__(
        self,
        num_classes=1,
        root_dir='.',
        coord_dim=3,
        device='cuda',
        use_dir_head=True,  # 新增参数
    ):
        """
        - inputs:
            - num_classes: default 2, for path and not path.
            - use_dir_head: whether to predict path direction vectors
        """
        self.use_dir_head = use_dir_head
        self.model = get_model(num_classes, coord_dim=coord_dim, use_dir_head=use_dir_head).to(device)
        model_filepath = join(
            root_dir,
            'results/model_training/random_pointnet2tf_3d/checkpoints/best_random_pointnet2tf_3d.pth'
        )
        checkpoint = torch.load(model_filepath, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.device = device
        print(f"PointNet++ wrapper 3d initialized. use_dir_head={use_dir_head}")

    
    def classify_path_points(
        self,
        pc,
        start_mask,
        goal_mask,
    ):
        """
        - outputs:
            - path_pred: (n_points,), 1-0 mask
            - path_score: (n_points,), probability
            - path_dir: (n_points,3) if use_dir_head else None
        """
        with torch.no_grad():
            n_points = pc.shape[0]
            pc_xyz = torch.from_numpy(pc_normalize(pc)).to(self.device)

            free_mask = 1-(start_mask+goal_mask).astype(bool)
            pc_features = torch.from_numpy(np.stack(
                (start_mask, goal_mask, free_mask.astype(np.float32)),
                axis=-1
            )).to(self.device)

            model_inputs = torch.cat([pc_xyz, pc_features], dim=1).permute(1, 0).unsqueeze(0)

            if self.use_dir_head:
                seg_pred, _, dir_pred = self.model(model_inputs)
                path_dir = torch.nn.functional.normalize(dir_pred, p=2, dim=-1)[0].detach().cpu().numpy()
            else:
                seg_pred, _ , dir_pred = self.model(model_inputs)  # 返回 None 或忽略
                path_dir = None

            path_pred = np.argmax(seg_pred.detach().cpu().numpy(), 2)[0]
            path_score = torch.softmax(seg_pred, dim=-1)[0, :, 1].detach().cpu().numpy()

            return path_pred, path_score, path_dir
