import json
import traceback
import os
from os import makedirs
from os.path import join
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from path_planning_utils_3d.env_3d import Env
from path_planning_utils_3d.Astar_3d import Weighted_A_star
from path_planning_utils_3d.collision_check_utils import points_in_AABB_3d, points_in_ball_3d

def json_converter(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)   # 兜底

def generate_env_3d(
    xyz_max,
    box_size_range,
    ball_radius_range,
    num_boxes_range,
    num_balls_range,
):
    """
    - inputs
        - xyz_max: tuple, (xmax, ymax, zmax)
        - box_size_range: list, (min, max)
        - ball_radius_range: list, (min, max)
        - num_boxes_range: list, (min, max)
        - num_balls_range: list, (min, max)
    - outputs
        - env_dims: tuple (xmax, ymax, zmax)
        - box_obstacles: list of [x,y,z,w,h,d]
        - ball_obstacles: list of [x,y,z,r]
    """
    xmax, ymax, zmax = xyz_max
    env_dims = xyz_max
    num_boxes = np.random.randint(num_boxes_range[0], num_boxes_range[1])
    num_balls = np.random.randint(num_balls_range[0], num_balls_range[1])
    box_obstacles = []
    ball_obstacles = []

    for i in range(num_boxes):
        not_in_env_3d = True
        while not_in_env_3d:
            x = np.random.randint(0, xmax)
            y = np.random.randint(0, ymax)
            z = np.random.randint(0, zmax)
            w = np.random.randint(box_size_range[0], box_size_range[1])
            h = np.random.randint(box_size_range[0], box_size_range[1])
            d = np.random.randint(box_size_range[0], box_size_range[1])
            if 0 <= x < xmax-w and 0 <= y < ymax-h and 0 <= z < zmax-d:
                not_in_env_3d = False
        box_obstacles.append([x,y,z,w,h,d])

    for i in range(num_balls):
        not_in_env_3d = True
        while not_in_env_3d:
            x = np.random.randint(0, xmax)
            y = np.random.randint(0, ymax)
            z = np.random.randint(0, zmax)
            r = np.random.randint(ball_radius_range[0], ball_radius_range[1])
            if r < x < xmax-r and r < y < ymax-r and r < z < zmax-r:
                not_in_env_3d = False
        ball_obstacles.append([x,y,z,r])
    
    return env_dims, box_obstacles, ball_obstacles


def generate_start_goal_points_3d(env, distance_lower_limit=50, max_attempt_count=100):
    attempt_count = 0
    while True:
        start_goal = np.random.randint(
            low=env.boundary[:3],
            high=env.boundary[3:],
            size=(2, 3),
        )
        start, goal = start_goal[0], start_goal[1]
        if ((start-goal)**2).sum() > distance_lower_limit**2:
            start_goal_in_aabb = points_in_AABB_3d(start_goal, env.box_obstacles, clearance=env.clearance)
            start_goal_in_ball = points_in_ball_3d(start_goal, env.balls_no_clearance, clearance=env.clearance)
            # 1 in obstacle, 0 not in obstacle # (2,) (2,)
            # 0,0
            # 0,0
            if (start_goal_in_aabb+start_goal_in_ball).sum()==0:
                return tuple(start.tolist()), tuple(goal.tolist())
        attempt_count += 1
        if attempt_count>max_attempt_count:
            return None, None

# ==================== ASTAR Helper ====================
def generate_astar_path(env):
    import time
    start_time = time.time()
    astar = Weighted_A_star(env)
    success = astar.run()
    if success:
        path = astar.get_path_solution()
        if astar.check_success(path):
            return path, time.time() - start_time
    return None, time.time() - start_time


# ==================== 单个环境处理 ====================
def process_single_env_astar(args):
    env_dict, num_samples, config = args
    pid = os.getpid()

    try:
        env = Env(
            env_dict['env_dims'],
            env_dict['box_obstacles'],
            env_dict['ball_obstacles'],
            clearance=config['path_clearance'],
            resolution=config['astar_resolution']
        )
        x_start_list, x_goal_list = env_dict['start'], env_dict['goal']
        path_list, exec_time_list = [], []

        for i in range(num_samples):
            env.set_start_goal(tuple(x_start_list[i]), tuple(x_goal_list[i]))
            path, exec_time = generate_astar_path(env)
            if path is None:
                print(f"[Sub {pid}] env FAILED sample {i}")
                return None   # 失败 → 整个环境丢弃
            path_list.append(path)
            exec_time_list.append(exec_time)

        env_dict['astar_time'] = exec_time_list
        return env_dict, path_list

    except Exception:
        tb = traceback.format_exc()
        print(f"[Sub {pid}] Exception:\n{tb}", flush=True)
        return None

# ==================== 保存函数 ====================
def save_env_paths(mode_path_dir, env_item):
    """
    保存单个环境的 A* 路径到 txt 文件
    - mode_path_dir: 保存目录
    - env_item: 带有 env_id 和 paths 的字典
    """
    env_id = env_item['env_id']
    path_list = env_item.get('paths', [])

    for path_idx, path in enumerate(path_list):
        filename = join(mode_path_dir, f"{env_id}_{path_idx}.txt")
        np.savetxt(filename, np.array(path), fmt='%d', delimiter=',')

# ==================== 并行生成 ASTAR 数据集 ====================
def generate_astar_dataset_parallel(mode, env_size, redundant_mode_env_list,
                                    config, dataset_dir,
                                    save_interval=10, n_process=4):
    print(f"Generating A* paths for {mode} set (parallel, {n_process} processes)...")
    mode_dir = join(dataset_dir, mode)
    mode_path_dir = join(mode_dir, "astar_paths")
    makedirs(mode_path_dir, exist_ok=True)
    final_file = join(mode_dir, "envs.json")

    mode_env_list = []
    valid_env_count, invalid_env_count = 0, 0

    # 如果有旧数据，先加载（断点续跑）
    if os.path.exists(final_file):
        with open(final_file, "r", encoding="utf-8") as f_json:
            try:
                old_envs = json.load(f_json)

                recovered_envs = []
                for e in old_envs:
                    env_id = e.get("env_id", None)
                    paths = e.get("paths", [])
                    # 检查每条路径文件是否存在
                    all_exist = True
                    for path_idx in range(len(paths)):
                        filename = join(mode_path_dir, f"{env_id}_{path_idx}.txt")
                        if not os.path.exists(filename):
                            print(f"[Warning] 缺少 {filename}，丢弃 env_id={env_id}")
                            all_exist = False
                            break
                    if all_exist:
                        recovered_envs.append(e)

                mode_env_list.extend(recovered_envs)
                valid_env_count = max([e["env_id"] for e in recovered_envs]) + 1 if recovered_envs else 0

                if valid_env_count >= env_size:
                    print(f"{mode} set already completed with {valid_env_count} envs.")
                    return
                print(f"已有 {valid_env_count} 个环境完成，继续剩余部分... "
                      f"(丢弃 {len(old_envs) - len(recovered_envs)} 个不完整环境)")
            except Exception as e:
                print(f"[Warning] Failed to load {final_file}: {e}, start from scratch.")


    env_iter = iter(redundant_mode_env_list)

    with ProcessPoolExecutor(max_workers=n_process) as executor:
        futures = {}
        # 初始任务池
        for _ in range(n_process):
            try:
                env_dict = next(env_iter)
                args = (env_dict, config['num_samples_per_env'], config)
                futures[executor.submit(process_single_env_astar, args)] = env_dict
            except StopIteration:
                break

        with tqdm(total=env_size, initial=valid_env_count, desc=f"[{mode}] 有效环境进度") as pbar:
            while futures and valid_env_count < env_size:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)

                for f in done:
                    futures.pop(f)
                    try:
                        result = f.result()
                    except Exception as e:
                        print(f"[Main] task raised exception: {e}")
                        result = None

                    if result is not None:
                        res_env_dict, path_list = result
                        res_env_dict['env_id'] = valid_env_count  # 连续编号
                        res_env_dict['paths'] = path_list
                        valid_env_count += 1

                        mode_env_list.append(res_env_dict)

                        # 保存路径文件
                        save_env_paths(mode_path_dir, res_env_dict)

                        # 定期保存 JSON
                        if len(mode_env_list) % save_interval == 0:
                            with open(final_file, "w", encoding="utf-8") as f_json:
                                json.dump(mode_env_list, f_json, ensure_ascii=False, default=json_converter)

                        pbar.update(1)
                    else:
                        invalid_env_count += 1  # 统计失败环境

                    # 补任务
                    if valid_env_count < env_size:
                        try:
                            env_dict = next(env_iter)
                            args = (env_dict, config['num_samples_per_env'], config)
                            futures[executor.submit(process_single_env_astar, args)] = env_dict
                        except StopIteration:
                            pass

            for f in futures:
                f.cancel()
            executor.shutdown(wait=False)

    # 最终保存
    print(f"所有进程完成，保存最终结果...")
    with open(final_file, "w", encoding="utf-8") as f_json:
        json.dump(mode_env_list, f_json, ensure_ascii=False, default=json_converter)

    print(f"Finished {mode} set. Total valid envs: {valid_env_count}, invalid envs: {invalid_env_count}")


# ==================== Main ====================
if __name__ == "__main__":
    import yaml
    config_name = "random_3d"
    with open(join("env_configs", config_name + ".yml"), "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    dataset_dir = join("data", config_name)

    env_size_dict = {
        'train': config['train_env_size'],
        'val': config['val_env_size'],
        'test': config['test_env_size'],
    }

    for mode in ['train', 'val', 'test']:
        mode_dir = join(dataset_dir, mode)
        with open(join(mode_dir, "raw_envs.json"), "r") as f:
            redundant_mode_env_list = json.load(f)
            print(f"Loaded {len(redundant_mode_env_list)} redundant envs for {mode} set")
        generate_astar_dataset_parallel(mode, env_size_dict[mode],
                                        redundant_mode_env_list, config, dataset_dir,
                                        n_process=10, save_interval=10)
