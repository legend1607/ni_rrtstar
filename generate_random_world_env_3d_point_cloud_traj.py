# generate_random_world_env_3d_point_cloud_traj.py
import json
import time
from os.path import join

import yaml
import numpy as np

from path_planning_utils_3d.env_3d import Env
from datasets.point_cloud_mask_utils import get_point_cloud_mask_around_points
from datasets_3d.point_cloud_mask_utils_3d import generate_rectangle_point_cloud_3d_v1


def save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=False):
    """保存 npz 数据集"""
    raw_dataset_saved = {}
    for k in raw_dataset.keys():
        if k == 'token':
            raw_dataset_saved[k] = np.array(raw_dataset[k])
        else:
            raw_dataset_saved[k] = np.stack(raw_dataset[k], axis=0)
    filename = f"{mode}_tmp.npz" if tmp else f"{mode}.npz"
    np.savez(join(dataset_dir, filename), **raw_dataset_saved)


def interpolate_path(path, n_samples=10):
    """对稀疏路径做等距插值"""
    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum_len = np.concatenate(([0], np.cumsum(seg_len)))
    total_len = cum_len[-1]
    sample_pos = np.linspace(0, total_len, n_samples)
    interp_path = np.zeros((n_samples, path.shape[1]))
    for i in range(path.shape[1]):
        interp_path[:, i] = np.interp(sample_pos, cum_len, path[:, i])
    return interp_path.astype(np.float32)


def compute_direction_and_step(pc, path_points):
    """
    为每个点计算最近路径点的方向向量和步长 (3D)
    pc: (N,3)
    path_points: (M,3)
    """
    dist_matrix = np.linalg.norm(pc[:, None, :] - path_points[None, :, :], axis=-1)
    nearest_idx = np.argmin(dist_matrix, axis=1)
    next_idx = np.clip(nearest_idx + 1, 0, path_points.shape[0] - 1)
    dir_vectors = path_points[next_idx] - pc
    step_lengths = np.linalg.norm(dir_vectors, axis=-1, keepdims=True) + 1e-6
    dir_vectors = dir_vectors / step_lengths  # 单位向量
    return dir_vectors.astype(np.float32), step_lengths.astype(np.float32)


config_name = "random_3d"
with open(join("env_configs", config_name + ".yml"), 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

dataset_dir = join("data", config_name)

for mode in ['train', 'val', 'test']:
    with open(join(dataset_dir, mode, "envs.json"), 'r') as f:
        env_list = json.load(f)
    env_list = sorted(env_list, key=lambda x: x['env_id'])

    raw_dataset = {
        'token': [],
        'pc': [],
        'start': [],
        'goal': [],
        'free': [],
        'astar': [],
        'dir': [],
        'step': [],
        'traj': [],
    }

    start_time = time.time()
    for env_idx, env_dict in enumerate(env_list):
        env = Env(
            env_dict['env_dims'],
            env_dict['box_obstacles'],
            env_dict['ball_obstacles'],
            clearance=config['path_clearance'],
            resolution=config['astar_resolution'],
        )

        env_id = env_dict['env_id']
        for sample_idx, (x_start, x_goal) in enumerate(zip(env_dict['start'], env_dict['goal'])):
            env.set_start_goal(x_start, x_goal)
            sample_title = f"{env_id}_{sample_idx}"
            path = np.loadtxt(
                join(dataset_dir, mode, "astar_paths", sample_title + ".txt"),
                delimiter=',',
            )
            token = f"{mode}-{sample_title}"

            # 生成 3D 点云
            pc = generate_rectangle_point_cloud_3d_v1(
                env,
                config['n_points'],
                over_sample_scale=config['over_sample_scale'],
            )

            # 分类 mask
            around_start_mask = get_point_cloud_mask_around_points(pc, np.array(x_start)[np.newaxis, :],
                                                                  neighbor_radius=config['start_radius'])
            around_goal_mask = get_point_cloud_mask_around_points(pc, np.array(x_goal)[np.newaxis, :],
                                                                 neighbor_radius=config['goal_radius'])
            around_path_mask = get_point_cloud_mask_around_points(pc, path, neighbor_radius=config['path_radius'])
            freespace_mask = (1 - around_start_mask) * (1 - around_goal_mask)

            # 插值轨迹
            traj = interpolate_path(path, n_samples=5)

            # 方向和步长
            dir_vectors, step_lengths = compute_direction_and_step(pc, path)

            # 存入数据集
            raw_dataset['token'].append(token)
            raw_dataset['pc'].append(pc.astype(np.float32))
            raw_dataset['start'].append(around_start_mask.astype(np.float32))
            raw_dataset['goal'].append(around_goal_mask.astype(np.float32))
            raw_dataset['free'].append(freespace_mask.astype(np.float32))
            raw_dataset['astar'].append(around_path_mask.astype(np.float32))
            raw_dataset['dir'].append(dir_vectors)
            raw_dataset['step'].append(step_lengths)
            raw_dataset['traj'].append(traj)

        if (env_idx + 1) % 25 == 0:
            time_left = (time.time() - start_time) * (len(env_list) / (env_idx + 1) - 1) / 60
            print(f"{mode} {env_idx + 1}/{len(env_list)}, remaining time: {int(time_left)} min")

    # 保存最终 npz
    save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=False)
    print(f"[{mode}] saved {len(raw_dataset['token'])} samples to {mode}.npz")
