#generate_random_world_env_2d_point_cloud_tf.py
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

def get_path_label(pc, path, s_goal=None, sigma=None, sigma_ratio=0.03):
    pc = np.asarray(pc)
    path = np.asarray(path)
    if s_goal is None:
        s_goal = path[-1]

    # 自动计算 sigma
    if sigma is None:
        range_vec = pc.max(axis=0) - pc.min(axis=0)
        sigma = sigma_ratio * range_vec
    sigma = np.asarray(sigma)
    
    path_label = np.zeros(len(pc), dtype=np.float32)
    sigma_eps = np.maximum(sigma, 1e-8)

    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i + 1]
        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-6:
            continue
        vec = pc - p0
        t = np.clip(np.sum(vec * seg_vec, axis=1) / (seg_len**2), 0, 1)
        proj = p0 + t[:, None] * seg_vec

        # 各维度加权距离
        diff = (pc - proj) / sigma_eps
        dist2 = np.sum(diff**2, axis=1)

        label = np.exp(-0.5 * dist2)
        path_label = np.maximum(path_label, label)

    return path_label


def get_path_and_direction_label(
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
        min_proj_dist = np.inf
        nearest_idx = None

        # 找最近路径线段（基于点到线段投影距离）
        for j in range(len(path) - 1):
            p0, p1 = path[j], path[j + 1]
            seg_vec = p1 - p0
            seg_len2 = np.dot(seg_vec, seg_vec)
            if seg_len2 < 1e-8:
                continue
            # 点到线段的投影参数 t∈[0,1]
            t = np.clip(np.dot(p - p0, seg_vec) / seg_len2, 0.0, 1.0)
            proj = p0 + t * seg_vec
            dist = np.linalg.norm(p - proj)
            if dist < min_proj_dist:
                min_proj_dist = dist
                nearest_idx = j

        # 如果点太远，不生成方向
        if min_proj_dist > path_influence_radius or nearest_idx is None:
            continue

        # ① 路径切线方向（使用最近线段）
        next_point = path[nearest_idx + 1] if nearest_idx < len(path) - 1 else s_goal
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
        direction_label[i] = tangent_vec#mix_vec / (np.linalg.norm(mix_vec) + 1e-6)

    return path_label, direction_label
def get_keypoint_label(pc, keypoints, sigma=5.0):
    """
    根据关键点生成soft标签（高斯衰减）
    pc: (N,2)
    keypoints: (K,2)
    sigma: 控制影响范围
    """
    if len(keypoints) == 0:
        return np.zeros(len(pc), dtype=np.float32)
    
    dist = np.linalg.norm(pc[:, None, :] - keypoints[None, :, :], axis=2)
    label = np.exp(-(np.min(dist, axis=1)**2) / (2 * sigma**2))
    return label.astype(np.float32)

def save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=False):
    """保存 npz 数据集"""
    raw_dataset_saved = {}
    for k in raw_dataset.keys():
        raw_dataset_saved[k] = np.stack(raw_dataset[k], axis=0) if k != "token" else np.array(raw_dataset[k])
    filename = f"{mode}_tmp.npz" if tmp else f"{mode}.npz"
    np.savez(join(dataset_dir, filename), **raw_dataset_saved)
def extract_keypoints_from_path(path, threshold=0.05):
    """
    从路径 polyline 提取关键点（高曲率拐点）
    path: (M, 2) np.array
    """
    keypoints = []
    for i in range(1, len(path) - 1):
        v1 = path[i] - path[i-1]
        v2 = path[i+1] - path[i]
        v1 = v1 / (np.linalg.norm(v1) + 1e-6)
        v2 = v2 / (np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
        if angle > threshold:  # 弯曲角度大于阈值
            keypoints.append(path[i])
    return np.array(keypoints)

def generate_npz_dataset(config_name="random_2d", planner_type="astar"):
    """生成符合 pointnet2tf 网络的数据集"""

    with open(join("env_configs", config_name + ".yml"), "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    data_subdir = config_name + "_" + planner_type
    dataset_dir = join("data", data_subdir)
    img_height, img_width = config["env_height"], config["env_width"]
    n_points = config["n_points"]
    over_sample_scale = config["over_sample_scale"]
    start_radius = config["start_radius"]
    goal_radius = config["goal_radius"]
    path_radius = config["path_radius"]

    for mode in ["test"]:
        with open(join(dataset_dir, mode, "envs.json"), "r") as f:
            env_list = json.load(f)

        raw_dataset = {
            "token": [],
            "pc": [],
            "start": [],
            "goal": [],
            "free": [],
            "path": [],
            "keypoint": [],
            "direction": [], 
        }

        start_time = time.time()
        for env_dict in env_list:
            env_idx=env_dict["env_idx"]
            env_img = cv2.imread(join(dataset_dir, mode, "env_imgs", f"{env_idx}.png"))
            binary_mask = get_binary_mask(env_img)

            for sample_idx, (s_start, s_goal) in enumerate(zip(env_dict["start"], env_dict["goal"])):
                s_start, s_goal = np.array(s_start), np.array(s_goal)
                start_point = s_start[np.newaxis, :]
                goal_point = s_goal[np.newaxis, :]

                sample_title = f"{env_idx}_{sample_idx}"
                # path=env_dict["paths"][sample_idx]
                # path = np.array(path)
                path = np.loadtxt(join(dataset_dir, mode, f"{planner_type}_paths", sample_title + ".txt"), delimiter=",")

                token = mode + "-" + sample_title

                # 生成无碰撞点云
                pc = generate_rectangle_point_cloud(binary_mask, n_points, over_sample_scale=over_sample_scale)  # (N, 2)

                # 分类 mask
                around_start_mask = get_point_cloud_mask_around_points(pc, start_point, neighbor_radius=start_radius)
                around_goal_mask = get_point_cloud_mask_around_points(pc, goal_point, neighbor_radius=goal_radius)
                
                path_label = get_path_label(pc, path)
                freespace_mask = (1 - around_start_mask) * (1 - around_goal_mask)

                # 提取关键点
                keypoints = path[1:]
                if len(keypoints) > 0:
                    keypoint_label = get_keypoint_label(pc, keypoints, sigma=10)
                else:
                    keypoint_label = np.zeros(pc.shape[0])
                plt.figure(figsize=(6,6))
                plt.imshow(binary_mask, cmap='gray')
                # 绘制点云
                plt.scatter(pc[:,0], pc[:,1], s=2, c='lightgray', label='Point cloud')

                # 绘制起点和终点
                plt.scatter(s_start[0], s_start[1], c='green', s=100, marker='*', edgecolors='k', label='Start')
                plt.scatter(s_goal[0], s_goal[1], c='magenta', s=100, marker='*', edgecolors='k', label='Goal')
                # 绘制路径
                # plt.plot(path[:,0], path[:,1], 'r-', lw=2, label='path')
                plt.scatter(pc[:,0], pc[:,1], c=path_label, cmap='Reds', s=15, alpha=0.6)

                # 绘制关键点
                if len(keypoints) > 0:
                    plt.scatter(pc[:,0], pc[:,1], c=keypoint_label, cmap='Oranges', s=15, alpha=0.6)


                plt.title(f"Env {env_idx} Sample {sample_idx}\nDirection field around path & keypoints")
                plt.legend(loc='upper right', fontsize=8)
                plt.axis('equal')
                plt.gca().invert_yaxis()  # 与图像坐标方向一致
                plt.tight_layout()
                plt.show()

                # 存入数据集
                raw_dataset["token"].append(token)
                raw_dataset["pc"].append(pc.astype(np.float32))
                raw_dataset["start"].append(around_start_mask.astype(np.float32))
                raw_dataset["goal"].append(around_goal_mask.astype(np.float32))
                raw_dataset["free"].append(freespace_mask.astype(np.float32))
                raw_dataset["path"].append(path_label.astype(np.float32))
                raw_dataset["keypoint"].append(keypoint_label.astype(np.float32))

            # 临时保存
            if (env_idx + 1) % 25 == 0:
                save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=True)
                time_left = (time.time() - start_time) * (len(env_list) / (env_idx + 1) - 1) / 60
                print(f"{mode} {env_idx+1}/{len(env_list)}, remaining time: {int(time_left)} min")

        # 保存最终 npz
        save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=False)
        print(f"[{mode}] saved {len(raw_dataset['token'])} samples to {mode}.npz")


if __name__ == "__main__":
    generate_npz_dataset("random_2d", "bitstar")
