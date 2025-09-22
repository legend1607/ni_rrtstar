import json
import numpy as np
from os import makedirs
from os.path import join
from time import sleep
from environment.kuka_env import KukaEnv
import pybullet as p

# ---------------- Configuration ----------------
config_name = "kuka_random_3d"
dataset_dir = join("data", config_name)

config = {
    'random_seed': 42,
    'num_obstacles_range': [5, 15],       
    'box_size_range': [0.05, 0.2],   
    'space_range_xy': [-1, 1],             # x/y 坐标范围
    'space_range_z': [0, 1],               # z 坐标范围
    'num_samples_per_env': 5,             
    'redundant_env_size_scale': 1.5,        
    'train_env_size': 4000,
    'val_env_size': 500,
    'test_env_size': 500,
    'GUI': False                          
}

np.random.seed(config['random_seed'])
env_size = {
    'train': config['train_env_size'],
    'val': config['val_env_size'],
    'test': config['test_env_size'],
}

# ---------------- Helper Functions ----------------
def generate_random_obstacles(env):
    num_obstacles = np.random.randint(*config['num_obstacles_range'])
    for _ in range(num_obstacles):
        half_extents = np.random.uniform(config['box_size_range'][0],
                                         config['box_size_range'][1], 3)
        base_position = np.array([
            np.random.uniform(*config['space_range_xy']),  # x
            np.random.uniform(*config['space_range_xy']),  # y
            np.random.uniform(*config['space_range_z'])    # z
        ])
        env.add_box_obstacle(half_extents, base_position)

def visualize_start_goal(env, start, goal):
    """在 GUI 中显示障碍物和一组 start-goal"""

    # 设置起点机械臂
    env.set_config(start)

    # 设置目标机械臂
    goal_kuka = p.loadURDF(env.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
    env.set_config(goal, goal_kuka)

    print("Visualizing first start-goal pair. Close GUI window to continue...")
    try:
        input()  # 等待用户按回车
    finally:
        p.removeBody(goal_kuka)

# ---------------- Dataset Generation ----------------
total_env_count = 0
invalid_env_count = 0

for mode in ['train', 'val', 'test']:
    mode_dir = join(dataset_dir, mode)
    makedirs(mode_dir, exist_ok=True)
    mode_env_list = []

    env = KukaEnv(GUI=config['GUI'])

    while len(mode_env_list) < env_size[mode] * config['redundant_env_size_scale']:
        total_env_count += 1
        env.reset_obstacles()
        env.reset_arm()
        generate_random_obstacles(env)

        valid_env = True
        x_start_list, x_goal_list = [], []

        for sample_i in range(config['num_samples_per_env']):
            x_start, x_goal = env.sample_start_goal()
            if x_start is None:
                valid_env = False
                break
            x_start_list.append(x_start.tolist())
            x_goal_list.append(x_goal.tolist())

            if config['GUI']:
                visualize_start_goal(env, x_start, x_goal)

        if not valid_env:
            invalid_env_count += 1
            print(f"Invalid env: {invalid_env_count}/{total_env_count}")
            continue

        box_obstacles = []
        for obs in env.obstacles:
            half_extents, position = obs
            x, y, z = position
            w, h, d = half_extents * 2
            box_obstacles.append([x, y, z, w, h, d])

        env_dict = {
            'env_dims': [
                config['space_range_xy'][0],  # xmin
                config['space_range_xy'][0],  # ymin
                config['space_range_z'][0],   # zmin
                config['space_range_xy'][1],  # xmax
                config['space_range_xy'][1],  # ymax
                config['space_range_z'][1]    # zmax
            ],
            'joint_bounds': env.bound.tolist(),
            'box_obstacles': box_obstacles,
            'ball_obstacles': [],
            'start': x_start_list,
            'goal': x_goal_list
        }
        mode_env_list.append(env_dict)
        print(f"Valid env: {len(mode_env_list)}/{total_env_count}")
        if len(mode_env_list) % 100 == 0:
            with open(join(mode_dir, "raw_envs.json"), "w") as f:
                json.dump(mode_env_list, f)

        if len(mode_env_list) % 100 == 0:
            print(f"{len(mode_env_list)} {mode} envs and {len(mode_env_list)*config['num_samples_per_env']} samples saved.")

print(f"Finished generating dataset, saved in {dataset_dir}")
print(f"Total environments: {total_env_count}, Invalid: {invalid_env_count}")
