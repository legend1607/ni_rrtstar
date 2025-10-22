# wrapper/pointnet_pointnet2/pointnet2tf_wrapper.py
from os.path import join
import torch
import numpy as np

from pointnet_pointnet2tf.models.pointnet2tf import get_model
from pointnet_pointnet2.models.pointnet2_utils import pc_normalize


class PNGWrapper:
    def __init__(
        self,
        num_classes=1,
        num_keypoints=1,
        root_dir='.',
        coord_dim=2,
        device='cuda',
        use_direction=True,
    ):
        """
        Wrapper for PointNet++ + Transformer model
        -----------------------------------------
        Args:
            num_classes: number of path prediction channels (default 1)
            num_keypoints: number of keypoint prediction channels (default 1)
            coord_dim: spatial dimension (2 or 3)
            use_direction: whether model predicts direction vectors
        """
        self.device = device
        self.coord_dim = coord_dim
        self.use_direction = use_direction

        # Initialize model
        self.model = get_model(
            num_classes=num_classes,
            coord_dim=coord_dim,
            feature_dim=3,  # start, goal, free
            use_direction=use_direction
        ).to(device)

        # Load checkpoint
        model_filepath = join(
            root_dir,
            'results/model_training/random_pointnet2tf_2d_bitstar/checkpoints/best_random_pointnet2tf_2d_bitstar.pth'
        )
        checkpoint = torch.load(model_filepath, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"[INFO] PointNet++ + Transformer model loaded from:\n{model_filepath}")

    @torch.no_grad()
    def classify_path_points(self, pc, start_mask=None, goal_mask=None):
        """
        Predict path probability, keypoint probability, and direction field
        -------------------------------------------------------------------
        Args:
            pc: np.ndarray (N, 2 or 3), input point cloud
            start_mask: (N,), 1 for start area, 0 otherwise
            goal_mask:  (N,), 1 for goal area, 0 otherwise
        Returns:
            path_score: (N,) float32, probability [0,1]
            keypoint_score: (N,) float32, probability [0,1]
            direction: (N, coord_dim) float32, normalized direction vectors
        """
        n_points = pc.shape[0]

        # Normalize coordinates
        pc_xyz = torch.from_numpy(pc_normalize(pc)).float().to(self.device)  # (N, coord_dim)

        # Prepare masks
        if start_mask is None:
            start_mask = np.zeros(n_points, dtype=np.float32)
        if goal_mask is None:
            goal_mask = np.zeros(n_points, dtype=np.float32)
        free_mask = 1 - np.clip(start_mask + goal_mask, 0, 1).astype(np.float32)

        # Stack as features: (N, 3)
        pc_features = np.stack([start_mask, goal_mask, free_mask], axis=-1)
        pc_features = torch.from_numpy(pc_features).float().to(self.device)

        # Build input tensor: (B, C, N)
        model_input = torch.cat([pc_xyz, pc_features], dim=1).unsqueeze(0).permute(0, 2, 1)

        # Forward pass
        path_logits, keypoint_logits, direction = self.model(model_input)

        # Convert to numpy
        path_score = torch.sigmoid(path_logits[0, 0]).cpu().numpy()
        keypoint_score = torch.sigmoid(keypoint_logits[0, 0]).cpu().numpy()

        if self.use_direction and direction is not None:
            direction = direction[0].permute(1, 0).cpu().numpy()  # (N, coord_dim)
        else:
            direction = np.zeros((n_points, self.coord_dim), dtype=np.float32)

        return path_score, keypoint_score, direction
