import json
import time
from matplotlib import pyplot as plt
import yaml
from os.path import join

import cv2
import numpy as np

from datasets.point_cloud_mask_utils import (
    get_binary_mask,
    get_point_cloud_mask_around_points,
    generate_rectangle_point_cloud,
)


def save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=False):
    """保存 npz 数据集"""
    raw_dataset_saved = {}
    for k in raw_dataset.keys():
        if k == "token":
            raw_dataset_saved[k] = np.array(raw_dataset[k])
        else:
            raw_dataset_saved[k] = np.stack(raw_dataset[k], axis=0)  # (b, n_points, ...)
    filename = mode + "_tmp.npz" if tmp else mode + ".npz"
    np.savez(join(dataset_dir, filename), **raw_dataset_saved)

def path_and_direction_label(
    pc, path, s_goal=None, sigma=2.0, path_influence_radius=10.0,
    goal_mix_scale=50.0
):
    """
    生成平滑的 soft path label + 混合方向标签（路径切线方向 + 指向目标方向）
    ------------------------------------------------
    pc: [N,2] 点云
    paths: list of np.array (M_i,2)
    s_goal: 终点坐标
    sigma: 路径label的高斯半径
    path_influence_radius: 超出此距离的点不生成方向
    goal_mix_scale: 控制切线方向与目标方向的融合范围（越大越强调路径切线）
    """
    N = pc.shape[0]
    path_label = np.zeros(N, dtype=np.float32)
    direction_label = np.zeros((N, 2), dtype=np.float32)

    path = np.array(path)
    if s_goal is None:
        s_goal = path[-1]

    # --- 1️⃣ soft path label: 高斯衰减 ---
    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i + 1]
        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-6:
            continue
        vec = pc - p0
        t = np.clip(np.sum(vec * seg_vec, axis=1) / (seg_len**2), 0, 1)
        proj = p0 + t[:, None] * seg_vec
        dist = np.linalg.norm(pc - proj, axis=1)
        label = np.exp(-(dist**2) / (2 * sigma**2))
        path_label = np.maximum(path_label, label)

    # --- 2️⃣ direction label: 切线方向 + 目标方向 融合 ---
    for i, p in enumerate(pc):
        # 找最近路径点
        dists_to_path = np.linalg.norm(path - p, axis=1)
        nearest_idx = np.argmin(dists_to_path)
        min_dist = dists_to_path[nearest_idx]
        if min_dist > path_influence_radius:
            continue

        # ① 路径切线方向
        if nearest_idx < len(path) - 1:
            next_point = path[nearest_idx + 1]
        else:
            next_point = s_goal
        tangent_vec = next_point - path[nearest_idx]
        tangent_norm = np.linalg.norm(tangent_vec)
        if tangent_norm < 1e-6:
            continue
        tangent_vec /= tangent_norm

        # ② 指向终点方向
        goal_vec = s_goal - p
        goal_norm = np.linalg.norm(goal_vec)
        if goal_norm < 1e-6:
            continue
        goal_vec /= goal_norm

        # ③ 混合权重: 越靠近终点 → 越看重 goal_vec
        dist_to_goal = np.linalg.norm(s_goal - p)
        alpha = np.exp(-dist_to_goal / goal_mix_scale)  # [0~1]

        mix_vec = (1 - alpha) * tangent_vec + alpha * goal_vec
        direction_label[i] = mix_vec / (np.linalg.norm(mix_vec) + 1e-6)

    return path_label, direction_label

# def path_and_direction_label(pc, path, s_goal=None, sigma=2.0, path_influence_radius=10):
#     """
#     生成 soft path label + 方向向量
#     ---------------------------
#     pc: [N,2] 点云
#     paths: list of np.array (M_i,2)
#     s_goal: 默认目标点 (终点)
#     sigma: 高斯衰减半径，用于 path label
#     path_influence_radius: 点到路径超过该距离就不生成方向
#     """
#     N = pc.shape[0]
#     path_label = np.zeros(N, dtype=np.float32)
#     direction_label = np.zeros((N, 2), dtype=np.float32)

#     path = np.array(path)
#     if s_goal is None:
#         s_goal = path[-1]

#     # ----------------------------
#     # 1️⃣ path label: 高斯衰减
#     # ----------------------------
#     for i in range(len(path) - 1):
#         p0, p1 = path[i], path[i + 1]
#         seg_vec = p1 - p0
#         seg_len = np.linalg.norm(seg_vec)
#         if seg_len < 1e-6:
#             continue
#         vec = pc - p0
#         t = np.clip(np.sum(vec * seg_vec, axis=1) / (seg_len**2), 0, 1)
#         proj = p0 + t[:, None] * seg_vec
#         dist = np.linalg.norm(pc - proj, axis=1)
#         label = np.exp(-(dist**2) / (2 * sigma**2))
#         path_label = np.maximum(path_label, label)

#     # ----------------------------
#     # 2️⃣ direction label: 基于最近路径点 + 目标关键点
#     # ----------------------------
#     for i, p in enumerate(pc):
#         # 计算 pc 点到路径的所有点距离
#         dists_to_path = np.linalg.norm(path - p, axis=1)
#         nearest_idx = np.argmin(dists_to_path)
#         min_dist = dists_to_path[nearest_idx]

#         # 距离过远就跳过
#         if min_dist > path_influence_radius:
#             continue

#         # 默认目标点为终点
#         target_point = s_goal

#         # 可以用路径关键点做更细的分段目标（这里直接用 path 本身）
#         key_seq = path[::max(len(path)//10, 1)]  # 取10个关键点均匀采样
#         for kpt in key_seq[:-1]:
#             kpt_idx = np.argmin(np.linalg.norm(path - kpt, axis=1))
#             if nearest_idx < kpt_idx:
#                 target_point = kpt
#                 break

#         # 方向向量
#         vec = target_point - p
#         norm = np.linalg.norm(vec) + 1e-6
#         direction_label[i] = vec / norm

#     return path_label, direction_label

# def path_and_direction_label(pc, paths, sigma=2.0, path_influence_radius=10):
#     """
#     生成 soft path label + 沿路径切线方向
#     ---------------------------
#     pc: [N,2] 点云
#     paths: list of np.array (M_i,2)
#     sigma: 高斯衰减半径，用于 path label
#     path_influence_radius: 点到路径超过该距离就不生成方向
#     """
#     N = pc.shape[0]
#     path_label = np.zeros(N, dtype=np.float32)
#     direction_label = np.zeros((N, 2), dtype=np.float32)

#     for path in paths:
#         path = np.array(path)

#         # ----------------------------
#         # 1️⃣ soft path label: 高斯衰减
#         # ----------------------------
#         for i in range(len(path) - 1):
#             p0, p1 = path[i], path[i + 1]
#             seg_vec = p1 - p0
#             seg_len = np.linalg.norm(seg_vec)
#             if seg_len < 1e-6:
#                 continue
#             vec = pc - p0
#             t = np.clip(np.sum(vec * seg_vec, axis=1) / (seg_len**2), 0, 1)
#             proj = p0 + t[:, None] * seg_vec
#             dist = np.linalg.norm(pc - proj, axis=1)
#             label = np.exp(-(dist**2) / (2 * sigma**2))
#             path_label = np.maximum(path_label, label)

#         # ----------------------------
#         # 2️⃣ direction label: 沿路径切线
#         # ----------------------------
#         for i, p in enumerate(pc):
#             # 找最近路径点索引
#             dists_to_path = np.linalg.norm(path - p, axis=1)
#             nearest_idx = np.argmin(dists_to_path)
#             min_dist = dists_to_path[nearest_idx]

#             # 距离过远就跳过
#             if min_dist > path_influence_radius:
#                 continue

#             # 沿路径方向：取当前点到下一路径点的向量
#             if nearest_idx < len(path) - 1:
#                 next_point = path[nearest_idx + 1]
#                 vec = next_point - path[nearest_idx]
#                 norm = np.linalg.norm(vec) + 1e-6
#                 direction_label[i] = vec / norm
#             else:
#                 # 如果是终点附近，方向置零
#                 direction_label[i] = np.zeros(2)

#     return path_label, direction_label

def generate_npz_dataset(config_name="random_2d", planner_type="bitstar"):
    """把 generate_dataset 生成的环境转换成 .npz 训练集"""

    # 读取配置文件
    with open(join("env_configs", config_name + ".yml"), "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    dataset_dir = join("data", f"{config_name}_{planner_type}")
    n_points = config["n_points"]
    over_sample_scale = config["over_sample_scale"]
    start_radius = config["start_radius"]
    goal_radius = config["goal_radius"]
    path_radius = config["path_radius"]

    for mode in ["test"]:
        env_json_path = join(dataset_dir, mode, "envs.json")
        with open(env_json_path, "r") as f:
            env_list = json.load(f)

        raw_dataset = {"token": [], "pc": [], "start": [], "goal": [], "free": [], "path": [], "direction": []}

        start_time = time.time()
        for env_dict in env_list:
            env_idx=env_dict["env_idx"]
            env_img_path = join(dataset_dir, mode, "env_imgs", f"{env_idx}.png")
            env_img = cv2.imread(env_img_path)
            binary_mask = get_binary_mask(env_img)

            skip_env = False
            samples_data = []  # 暂存该环境的有效样本

            for sample_idx, (s_start, s_goal) in enumerate(zip(env_dict["start"], env_dict["goal"])):
                sample_title = f"{env_idx}_{sample_idx}"
                path_file = join(dataset_dir, mode, f"{planner_type}_paths", sample_title + ".txt")

                # 检查路径文件是否存在且非空
                try:
                    path = np.loadtxt(path_file, delimiter=",")
                    if path.ndim == 1:
                        path = path[np.newaxis, :]
                    if path.shape[0] == 0:
                        skip_env = True
                        print(f"[{mode}] Skipping env {env_idx} because path {sample_title} is empty.")
                        break
                except Exception:
                    skip_env = True
                    print(f"[{mode}] Skipping env {env_idx} because path file {sample_title} is missing or invalid.")
                    break
                s_start, s_goal = np.array(s_start), np.array(s_goal)
                start_point = s_start[np.newaxis, :]
                goal_point = s_goal[np.newaxis, :]

                sample_title = f"{env_idx}_{sample_idx}"
                path_file = join(dataset_dir, mode, f"{planner_type}_paths", sample_title + ".txt")
                path = np.loadtxt(path_file, delimiter=",")

                token = mode + "-" + sample_title

                # 生成点云
                pc = generate_rectangle_point_cloud(
                    binary_mask,
                    n_points,
                    over_sample_scale=over_sample_scale,
                )

                # soft path label + direction
                path_label, direction_label = path_and_direction_label(pc, path, sigma=path_radius)

                # start/goal/freespace mask
                around_start_mask = get_point_cloud_mask_around_points(pc, start_point, neighbor_radius=start_radius)
                around_goal_mask = get_point_cloud_mask_around_points(pc, goal_point, neighbor_radius=goal_radius)
                freespace_mask = (1 - around_start_mask) * (1 - around_goal_mask)

                # 绘制环境 + 点云 + start/goal +路径 +方向场
                plt.figure(figsize=(6,6))
                plt.imshow(binary_mask, cmap='gray', origin='lower')
                
                # 点云
                plt.scatter(pc[:,0], pc[:,1], s=2, c='lightgray', label='Point cloud')
                
                # 起点/终点
                plt.scatter(s_start[0], s_start[1], c='green', s=100, marker='*', edgecolors='k', label='Start')
                plt.scatter(s_goal[0], s_goal[1], c='magenta', s=100, marker='*', edgecolors='k', label='Goal')
                
                # 路径
                plt.plot(path[:,0], path[:,1], 'r-o', lw=2, label='Path')
                
                # soft label 热力可选（灰色点+红色渐变）
                plt.scatter(pc[:,0], pc[:,1], c=path_label, cmap='Reds', s=15, alpha=0.6)
                
                # 方向箭头
                nonzero_mask = np.linalg.norm(direction_label, axis=1) > 1e-2
                plt.quiver(
                    pc[nonzero_mask][::10, 0],
                    pc[nonzero_mask][::10, 1],
                    direction_label[nonzero_mask][::10, 0],
                    direction_label[nonzero_mask][::10, 1],
                    color='blue',
                    scale=10,
                    width=0.002,
                    alpha=0.8,
                    label='Direction field'
                )
                
                plt.legend()
                plt.axis('equal')
                plt.title("Path + Soft Label + Direction Field")
                plt.show()


                # 保存样本
                raw_dataset["token"].append(token)
                raw_dataset["pc"].append(pc.astype(np.float32))
                raw_dataset["start"].append(around_start_mask.astype(np.float32))
                raw_dataset["goal"].append(around_goal_mask.astype(np.float32))
                raw_dataset["free"].append(freespace_mask.astype(np.float32))
                raw_dataset["path"].append(path_label.astype(np.float32))
                raw_dataset["direction"].append(direction_label.astype(np.float32))

            if (env_idx + 1) % 25 == 0:
                save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=True)
                time_left = (time.time() - start_time) * (len(env_list) / (env_idx + 1) - 1) / 60
                print(f"{mode} {env_idx + 1}/{len(env_list)}, remaining time: {int(time_left)} min")

        save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=False)
        print(f"[{mode}] saved {len(raw_dataset['token'])} samples to {mode}.npz")


if __name__ == "__main__":
    # 先运行 generate_dataset("random_2d", planner_type="bitstar") 生成 envs.json + PNG + BIT* 路径
    # 然后运行本脚本，生成 train/val/test 的 .npz 文件
    generate_npz_dataset("random_2d", planner_type="bitstar")
