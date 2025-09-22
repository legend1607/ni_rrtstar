import json
import time
import yaml
import random
from os import makedirs
from os.path import join, exists

import cv2
import numpy as np

from path_planning_utils.Astar_with_clearance import generate_start_goal_points, AStar


def generate_env(
    img_height,
    img_width,
    rectangle_width_range,
    circle_radius_range,
    num_rectangles_range,
    num_circles_range,
):
    """生成一个环境，包括图像、二值地图、障碍物参数"""
    env_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    env_dims = (img_height, img_width)
    num_rectangles = random.randint(num_rectangles_range[0], num_rectangles_range[1])
    num_circles = random.randint(num_circles_range[0], num_circles_range[1])
    rectangle_obstacles = []
    circle_obstacles = []

    # 随机矩形障碍
    for i in range(num_rectangles):
        x = random.randint(0, img_width)
        y = random.randint(0, img_height)
        w = random.randint(rectangle_width_range[0], rectangle_width_range[1])
        h = random.randint(rectangle_width_range[0], rectangle_width_range[1])
        cv2.rectangle(env_img, (x, y), (x + w, y + h), (0, 0, 0), -1)
        rectangle_obstacles.append([x, y, w, h])

    # 随机圆形障碍
    for i in range(num_circles):
        x = random.randint(0, img_width)
        y = random.randint(0, img_height)
        r = random.randint(circle_radius_range[0], circle_radius_range[1])
        cv2.circle(env_img, (x, y), r, (0, 0, 0), -1)
        circle_obstacles.append([x, y, r])

    # 二值地图（1=可行走，0=障碍）
    binary_env = np.zeros(env_dims).astype(int)
    binary_env[env_img[:, :, 0] != 0] = 1

    return env_img, binary_env, env_dims, rectangle_obstacles, circle_obstacles


def generate_astar_path(
    binary_env,
    s_start,
    s_goal,
    clearance=3,
):
    """调用A*搜索，返回路径和耗时"""
    start_time = time.time()
    astar = AStar(s_start, s_goal, binary_env, clearance, "euclidean")
    path, visited = astar.searching()
    path = astar.get_path_from_start_to_goal(path)  # list of tuples
    path_success = astar.check_success(path)
    exec_time = time.time() - start_time
    if path_success:
        return path, exec_time
    else:
        return None, exec_time


def generate_dataset(config_name="random_2d"):
    """主函数：根据配置文件生成 train/val/test 数据集"""

    # 读取配置
    with open(join("env_configs", config_name + ".yml"), "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    img_height, img_width = config["env_height"], config["env_width"]
    rectangle_width_range = config["rectangle_width_range"]
    circle_radius_range = config["circle_radius_range"]
    num_rectangles_range = config["num_rectangles_range"]
    num_circles_range = config["num_circles_range"]
    path_clearance = config["path_clearance"]
    start_goal_dim_distance_limit = config["start_goal_dim_distance_limit"]
    start_goal_sampling_attempt_count = config["start_goal_sampling_attempt_count"]

    env_size = {
        "train": config["train_env_size"],
        "val": config["val_env_size"],
        "test": config["test_env_size"],
    }

    num_samples_per_env = config["num_samples_per_env"]

    total_env_count = 0
    invalid_env_count = 0

    for mode in ["train", "val", "test"]:
        mode_dir = join("data", config_name, mode)
        mode_img_dir = join(mode_dir, "env_imgs")
        mode_path_dir = join(mode_dir, "astar_paths")

        # 确保目录存在
        makedirs(mode_img_dir, exist_ok=True)
        makedirs(mode_path_dir, exist_ok=True)

        mode_env_list = []

        while len(mode_env_list) < env_size[mode]:
            total_env_count += 1
            (
                env_img,
                binary_env,
                env_dims,
                rectangle_obstacles,
                circle_obstacles,
            ) = generate_env(
                img_height,
                img_width,
                rectangle_width_range,
                circle_radius_range,
                num_rectangles_range,
                num_circles_range,
            )

            valid_env = True
            s_start_list, s_goal_list, path_list, exec_time_list = [], [], [], []

            for _ in range(num_samples_per_env):
                s_start, s_goal = generate_start_goal_points(
                    binary_env,
                    clearance=path_clearance,
                    distance_lower_limit=start_goal_dim_distance_limit,
                    max_attempt_count=start_goal_sampling_attempt_count,
                )
                if s_start is None:
                    valid_env = False
                    break

                path, exec_time = generate_astar_path(
                    binary_env,
                    s_start,
                    s_goal,
                    clearance=path_clearance,
                )
                if path is None:
                    valid_env = False
                    break

                s_start_list.append(s_start)
                s_goal_list.append(s_goal)
                path_list.append(path)
                exec_time_list.append(exec_time)

            if not valid_env:
                invalid_env_count += 1
                print(
                    f"Invalid env: {invalid_env_count}/{total_env_count} (discarded)"
                )
                continue

            # 保存环境信息
            env_dict = {
                "env_dims": env_dims,
                "rectangle_obstacles": rectangle_obstacles,
                "circle_obstacles": circle_obstacles,
                "start": s_start_list,
                "goal": s_goal_list,
                "astar_time": exec_time_list,
            }

            mode_env_list.append(env_dict)
            env_idx = len(mode_env_list) - 1

            # 保存图片
            cv2.imwrite(join(mode_img_dir, f"{env_idx}.png"), env_img)

            # 保存路径
            for path_idx, path in enumerate(path_list):
                path_np = np.array(path)
                np.savetxt(
                    join(mode_path_dir, f"{env_idx}_{path_idx}.txt"),
                    path_np,
                    fmt="%d",
                    delimiter=",",
                )

            # 更新 envs.json
            with open(join(mode_dir, "envs.json"), "w") as f:
                json.dump(mode_env_list, f, indent=2)

            print(
                f"[{mode}] {len(mode_env_list)} envs / "
                f"{num_samples_per_env * len(mode_env_list)} samples saved."
            )


if __name__ == "__main__":
    generate_dataset("random_2d")
