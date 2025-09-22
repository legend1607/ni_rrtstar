import json
import time
import yaml
from os.path import join

import cv2
import numpy as np

from datasets.point_cloud_mask_utils import (
    get_binary_mask,
    get_point_cloud_mask_around_points,
    generate_rectangle_point_cloud,
)


def save_raw_dataset(
    raw_dataset,
    dataset_dir,
    mode,
    tmp=False,
):
    """保存 npz 数据集"""
    raw_dataset_saved = {}
    for k in raw_dataset.keys():
        if k == "token":
            raw_dataset_saved[k] = np.array(raw_dataset[k])
        else:
            raw_dataset_saved[k] = np.stack(raw_dataset[k], axis=0)  # (b, n_points, ...)
    if tmp:
        filename = mode + "_tmp.npz"
    else:
        filename = mode + ".npz"
    np.savez(join(dataset_dir, filename), **raw_dataset_saved)


def generate_npz_dataset(config_name="random_2d"):
    """把 generate_dataset 生成的环境转换成 .npz 训练集"""

    # 读取配置文件
    with open(join("env_configs", config_name + ".yml"), "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    dataset_dir = join("data", config_name)
    img_height, img_width = config["env_height"], config["env_width"]
    n_points = config["n_points"]
    over_sample_scale = config["over_sample_scale"]
    start_radius = config["start_radius"]
    goal_radius = config["goal_radius"]
    path_radius = config["path_radius"]

    # 三个模式：train/val/test
    for mode in ["train", "val", "test"]:
        with open(join(dataset_dir, mode, "envs.json"), "r") as f:
            env_list = json.load(f)

        raw_dataset = {"token": [], "pc": [], "start": [], "goal": [], "free": [], "astar": []}

        start_time = time.time()
        for env_idx, env_dict in enumerate(env_list):
            env_img = cv2.imread(join(dataset_dir, mode, "env_imgs", f"{env_idx}.png"))
            binary_mask = get_binary_mask(env_img)

            # 遍历这个环境里的多个起点-终点对
            for sample_idx, (s_start, s_goal) in enumerate(zip(env_dict["start"], env_dict["goal"])):
                s_start, s_goal = np.array(s_start), np.array(s_goal)
                start_point = s_start[np.newaxis, :]
                goal_point = s_goal[np.newaxis, :]

                sample_title = f"{env_idx}_{sample_idx}"
                path = np.loadtxt(
                    join(dataset_dir, mode, "astar_paths", sample_title + ".txt"),
                    delimiter=",",
                )

                # 样本 token：方便索引
                token = mode + "-" + sample_title

                # 生成点云
                pc = generate_rectangle_point_cloud(
                    binary_mask,
                    n_points,
                    over_sample_scale=over_sample_scale,
                )  # (n_points, 2)

                # 构造 mask
                around_start_mask = get_point_cloud_mask_around_points(
                    pc, start_point, neighbor_radius=start_radius
                )
                around_goal_mask = get_point_cloud_mask_around_points(
                    pc, goal_point, neighbor_radius=goal_radius
                )
                around_path_mask = get_point_cloud_mask_around_points(
                    pc, path, neighbor_radius=path_radius
                )
                freespace_mask = (1 - around_start_mask) * (1 - around_goal_mask)

                # 存入 raw_dataset
                raw_dataset["token"].append(token)
                raw_dataset["pc"].append(pc.astype(np.float32))
                raw_dataset["start"].append(around_start_mask.astype(np.float32))
                raw_dataset["goal"].append(around_goal_mask.astype(np.float32))
                raw_dataset["free"].append(freespace_mask.astype(np.float32))
                raw_dataset["astar"].append(around_path_mask.astype(np.float32))

            # 每处理 25 个环境，先存一次临时数据
            if (env_idx + 1) % 25 == 0:
                save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=True)
                time_left = (time.time() - start_time) * (len(env_list) / (env_idx + 1) - 1) / 60
                print(f"{mode} {env_idx+1}/{len(env_list)}, remaining time: {int(time_left)} min")

        # 最终保存 npz
        save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=False)
        print(f"[{mode}] saved {len(raw_dataset['token'])} samples to {mode}.npz")


if __name__ == "__main__":
    # 先运行 generate_dataset("random_2d") 生成 envs.json + PNG + A* 路径
    # 然后运行本脚本，生成 train/val/test 的 .npz 文件
    generate_npz_dataset("random_2d")
