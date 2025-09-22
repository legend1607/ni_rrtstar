import numpy as np
from torch.utils.data import Dataset

from pointnet_pointnet2.models.pointnet2_utils import pc_normalize


class PathPlanDataset(Dataset):
    def __init__(
        self,
        env_type,
        dataset_filepath,
    ):
        """
        dataset_filepath: 'data/random_2d/'+mode+'.npz', where mode is 'train', 'val', or 'test'.
        """
        data = np.load(dataset_filepath)
        self.pc = data['pc'].astype(np.float32)
        self.start_mask = data['start'].astype(np.float32)
        self.goal_mask = data['goal'].astype(np.float32) 
        self.free_mask = data['free'].astype(np.float32) 
        if env_type.startswith('random'):
            self.path_mask = data['astar'].astype(np.float32) 
        elif env_type.startswith('kuka'):
            self.path_mask = data['bitstar'].astype(np.float32)
        self.token = data['token']
        # print(f"self.token.dtype: {self.token.dtype}")
        if self.pc.shape[2]==2:
            self.pc = np.concatenate(
                (self.pc, np.zeros((self.pc.shape[0], self.pc.shape[1], 1)).astype(np.float32)),
                axis=2,
            )
        self.d = self.pc.shape[2]
        self.n_points = self.pc.shape[1]  
        print(f"[PathPlanDataset] Loaded point cloud with dimension = {self.d}")
        labelweights, _ = np.histogram(self.path_mask, range(3))
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)

    def __len__(self):
        return len(self.pc)
    
    def __getitem__(self, index):
        pc_pos_raw = self.pc[index] # (2048, 3)
        pc_pos = pc_normalize(pc_pos_raw)
        pc_features = np.stack(
            (self.start_mask[index], self.goal_mask[index], self.free_mask[index]),
            axis=-1,
        ) # (2048, 3)
        pc_labels = self.path_mask[index] # (2048,)
        return pc_pos_raw, pc_pos, pc_features, pc_labels, self.token[index]