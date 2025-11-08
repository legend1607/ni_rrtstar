import os
import json
import cv2
import time
from matplotlib import pyplot as plt
import yaml
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from path_planning_utils.Astar_with_clearance import generate_start_goal_points, AStar
from os.path import join, exists

from path_planning_classes.bit_star import BITStar  # BIT* 实现


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

def generate_path_astar(binary_env, s_start, s_goal, clearance=3):
    astar = AStar(s_start, s_goal, binary_env, clearance, "euclidean")
    path, visited = astar.searching()
    path = astar.get_path_from_start_to_goal(path)
    if astar.check_success(path):
        return path
    return None


def generate_path_bitstar(binary_env, s_start, s_goal, batch_size=100):
    """生成 BIT* 路径（单个点对）并可视化"""
    class SimpleEnv:
        def __init__(self, binary_map):
            self.map = binary_map
            self.init_state = s_start
            self.goal_state = s_goal
            self.bound = [(0, 0), binary_map.shape[::-1]]
            self.config_dim = 2
            self.collision_check_count = 0

        def _state_fp(self, state):
            x, y = int(state[0]), int(state[1])
            if 0 <= x < self.map.shape[1] and 0 <= y < self.map.shape[0]:
                self.collision_check_count += 1
                return self.map[y, x] == 1
            return False

        def _edge_fp(self, p1, p2):
            self.collision_check_count += 1
            num = max(int(np.linalg.norm(np.array(p2) - np.array(p1)) * 2), 2)
            line = list(zip(
                np.linspace(p1[0], p2[0], num=num).astype(int),
                np.linspace(p1[1], p2[1], num=num).astype(int)
            ))
            return all(0 <= x < self.map.shape[1] and 0 <= y < self.map.shape[0] and self.map[y, x] == 1 for x, y in line)

    env = SimpleEnv(binary_env)
    planner = BITStar(env, plot_flag=False, batch_size=batch_size)
    planner.plan(refine_time_budget=5)
    path = planner.get_best_path()

    # plt.figure(figsize=(6,6))
    # plt.imshow(binary_env, cmap='gray', origin='lower')
    # plt.scatter(s_start[0], s_start[1], c='green', s=100, marker='*', edgecolors='k', label='Start')
    # plt.scatter(s_goal[0], s_goal[1], c='magenta', s=100, marker='*', edgecolors='k', label='Goal')
    # if path:
    #     path = np.array(path)
    #     plt.plot(path[:,0], path[:,1], 'r-', lw=2, label='Path')
    # plt.axis('equal')
    # plt.legend()
    # plt.show()

    return path if len(path) > 0 else None


def generate_single_env_bitstar(env_idx, config, mode):
    """生成单个 BIT* 环境及其多点对路径"""
    env_img, binary_env, env_dims, rectangle_obstacles, circle_obstacles = generate_env(

        config["env_height"], 
        config["env_width"],
        config["rectangle_width_range"],
        config["circle_radius_range"],
        config["num_rectangles_range"],
        config["num_circles_range"],
    )

    s_start_list, s_goal_list, path_list = [], [], []

    # 内层并行生成每个 start-goal 点对路径
    futures = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for _ in range(config["num_samples_per_env"]):
            s_start, s_goal = generate_start_goal_points(
                binary_env,
                clearance=config["path_clearance"],
                distance_lower_limit=config["start_goal_dim_distance_limit"],
                max_attempt_count=config["start_goal_sampling_attempt_count"],
            )
            if s_start is None:
                continue
            futures.append(executor.submit(generate_path_bitstar, binary_env, s_start, s_goal, config.get("batch_size", 100)))

        for fut in as_completed(futures):
            path = fut.result()
            if path is None or len(path) == 0:
                # 如果有任何路径为空 → 放弃整个环境
                return None, None, None
            if path:
                path_list.append(path)
                s_start_list.append(path[0])
                s_goal_list.append(path[-1])

    env_dict = {
        "env_dims": env_dims,
        "rectangle_obstacles": rectangle_obstacles,
        "circle_obstacles": circle_obstacles,
        "start": s_start_list,
        "goal": s_goal_list,
        "paths": path_list
    }

    return env_idx, env_img, env_dict


def generate_dataset(config_name="random_2d", planner_type="astar"):
    """生成数据集，支持 A* 串行和 BIT* 多进程并行"""
    # 读取配置
    with open(join("env_configs", config_name + ".yml"), "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    img_height, img_width = config["env_height"], config["env_width"]
    num_samples_per_env = config["num_samples_per_env"]

    env_size = {
        "train": config["train_env_size"],
        "val": config["val_env_size"],
        "test": config["test_env_size"],
    }

    data_subdir = config_name + "_" + planner_type

    for mode in ["train", "val", "test"]:
        mode_dir = join("data", data_subdir, mode)
        img_dir = join(mode_dir, "env_imgs")
        path_dir = join(mode_dir, f"{planner_type}_paths")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(path_dir, exist_ok=True)

        env_list = []

        if planner_type.lower() == "astar":
            # 串行逻辑
            for env_idx in range(env_size[mode]):
                valid_env = False
                while not valid_env:
                    env_img, binary_env, env_dims, rectangle_obstacles, circle_obstacles = generate_env(
                        img_height,
                        img_width,
                        config["rectangle_width_range"],
                        config["circle_radius_range"],
                        config["num_rectangles_range"],
                        config["num_circles_range"],
                    )
                    s_start_list, s_goal_list, path_list = [], [], []
                    valid_env = True
                    for _ in range(num_samples_per_env):
                        s_start, s_goal = generate_start_goal_points(
                            binary_env,
                            clearance=config["path_clearance"],
                            distance_lower_limit=config["start_goal_dim_distance_limit"],
                            max_attempt_count=config["start_goal_sampling_attempt_count"],
                        )
                        if s_start is None:
                            valid_env = False
                            break
                        path = generate_path_astar(binary_env, s_start, s_goal, config["path_clearance"])
                        if path is None:
                            valid_env = False
                            break
                        s_start_list.append(s_start)
                        s_goal_list.append(s_goal)
                        path_list.append(path)

                # 保存数据
                env_dict = {
                    "env_dims": env_dims,
                    "rectangle_obstacles": rectangle_obstacles,
                    "circle_obstacles": circle_obstacles,
                    "start": s_start_list,
                    "goal": s_goal_list,
                    "paths": path_list
                }
                env_list.append(env_dict)
                cv2.imwrite(join(img_dir, f"{env_idx}.png"), env_img)
                for i, path in enumerate(path_list):
                    np.savetxt(join(path_dir, f"{env_idx}_{i}.txt"), np.array(path), fmt="%.2f", delimiter=",")
                with open(join(mode_dir, "envs.json"), "w") as f:
                    json.dump(env_list, f, indent=2)
                print(
                    f"[{mode}] {len(env_list)} envs / "
                    f"{num_samples_per_env * len(env_list)} samples saved."
                )

        elif planner_type.lower() == "bitstar":
            # 多进程并行生成环境
            with ProcessPoolExecutor(max_workers=config.get("max_workers_env", 4)) as executor:
                futures = [executor.submit(generate_single_env_bitstar, env_idx, config, mode)
                           for env_idx in range(env_size[mode])]
                for fut in as_completed(futures):
                    env_idx, env_img, env_dict = fut.result()
                    if env_dict is None:
                        print(f"[{mode}] Skipping environment {env_idx} due to empty paths.")
                        continue
                    env_list.append(env_dict)
                    cv2.imwrite(join(img_dir, f"{env_idx}.png"), env_img)
                    for i, path in enumerate(env_dict["paths"]):
                        if path is not None and len(path) > 0:
                            np.savetxt(join(path_dir, f"{env_idx}_{i}.txt"), np.array(path), fmt="%.2f", delimiter=",")
                    with open(join(mode_dir, "envs.json"), "w") as f:
                        json.dump(env_list, f, indent=2)
                    print(
                        f"[{mode}] {len(env_list)} envs / "
                        f"{num_samples_per_env * len(env_list)} samples saved."
                    )

        else:
            raise ValueError(f"Unknown planner_type: {planner_type}")

        print(f"[{mode}] {len(env_list)} environments generated and saved.")


if __name__ == "__main__":
    generate_dataset(config_name="random_2d", planner_type="bitstar")
