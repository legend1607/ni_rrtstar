import json
import traceback, os, sys
from os import makedirs
from os.path import join
from matplotlib import pyplot as plt
import numpy as np
import pybullet as p
import time
from concurrent.futures import wait
from tqdm import tqdm
from multiprocessing import Pool
from environment.kuka_env import KukaEnv
from path_planning_classes_3d.bit_star import BITStar
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------- Configuration ----------------
config_name = "kuka_random_3d"
dataset_dir = join("data", config_name)

config = {
    'random_seed': 42,
    'num_obstacles_range': [5, 15],
    'box_size_range': [0.05, 0.2],
    'space_range': [-1, 1],
    'num_samples_per_env': 1,
    'redundant_env_size_scale': 2,
    'train_env_size': 4000,
    'val_env_size': 500,
    'test_env_size': 500,
    'GUI': False,
    'refine_time_budget': 10,  
    'time_budget': 30, 
}

np.random.seed(config['random_seed'])

# ---------------- Helper Functions ----------------
def generate_bitstar_path(env, x_start, x_goal, max_attempts=3, gui=False, refine_time_budget=np.inf,time_budget=np.inf):
    env.init_state = np.array(x_start)
    env.goal_state = np.array(x_goal)
    for attempt in range(max_attempts):
        bitstar = BITStar(env, maxIter=5, batch_size=200,T=1000000)
        samples, edges, collision_count, goal_score, total_samples, exec_time = bitstar.plan(pathLengthLimit=np.inf, refine_time_budget=refine_time_budget,time_budget=time_budget)
        path = bitstar.get_best_path()
        if path and len(path) > 0:
            if gui:
                visualize_path(env, path)
            return path, exec_time
    return None, None

def visualize_path(env, path, step_ratio=0.1, snapshot_interval=50, ee_marker_scale=0.05):
    if not env.GUI:
        raise RuntimeError("GUI must be enabled to visualize path")
    path = np.array(path)
    interp_counter = 0
    red_spheres, kuka_snaps = [], []

    try:
        env.set_config(path[0], env.kukaId)
        goal_kuka = p.loadURDF(env.kuka_file, [0, 0, 0], [0, 0, 0, 1],
                                useFixedBase=True, flags=p.URDF_IGNORE_COLLISION_SHAPES)
        env.set_config(path[-1], goal_kuka)
        for i in range(len(path) - 1):
            q_start, q_end = path[i], path[i + 1]
            disp = q_end - q_start
            n_steps = max(int(np.linalg.norm(disp) / step_ratio), 1)

            for k in range(n_steps + 1):
                q_interp = q_start + (k / n_steps) * disp
                # 克隆透明机械臂
                if interp_counter % snapshot_interval == 0:
                    kuka_snap = p.loadURDF(env.kuka_file, [0, 0, 0], [0, 0, 0, 1],
                                           useFixedBase=True, flags=p.URDF_IGNORE_COLLISION_SHAPES)
                    for data in p.getVisualShapeData(kuka_snap):
                        color = list(data[-1])
                        color[-1] = 0.3
                        p.changeVisualShape(kuka_snap, data[1], rgbaColor=color)
                    kuka_snaps.append(kuka_snap)
                env.set_config(q_interp, kuka_snaps[-1] if kuka_snaps else env.kukaId)
                ee_pos = p.getLinkState(kuka_snaps[-1] if kuka_snaps else env.kukaId, env.kukaEndEffectorIndex)[0]
                red_ball = p.loadURDF("sphere2red.urdf", ee_pos, globalScaling=ee_marker_scale,
                                      flags=p.URDF_IGNORE_COLLISION_SHAPES)
                red_spheres.append(red_ball)
                interp_counter += 1
                p.stepSimulation()
        # print("路径可视化完成，按回车关闭并清空快照与红球...")
        # input()
    finally:
        for ball in red_spheres: p.removeBody(ball)
        for snap in kuka_snaps: p.removeBody(snap)
        p.removeBody(goal_kuka)
        p.removeAllUserDebugItems()


def visualize_env_sample(mode, env_id, sample_idx, dataset_dir="data/kuka_random_3d"):
    """
    可视化指定环境和 start-goal 样本的运动路径，并绘制关节角度曲线
    :param mode: 'train' / 'val' / 'test'
    :param env_id: 环境编号
    :param sample_idx: 样本编号 (即 start-goal 对序号)
    :param dataset_dir: 数据集目录
    """
    mode_dir = join(dataset_dir, mode)

    # 1. 读取环境配置
    with open(join(mode_dir, "envs.json"), "r") as f:
        envs = json.load(f)
    env_dict = next(e for e in envs if e["env_id"] == env_id)
    if env_dict is None:
        raise ValueError(f"Env id {env_id} not found in {mode} set")
    # 2. 读取路径文件
    path_file = join(mode_dir, "bitstar_paths", f"{env_id}_{sample_idx}.txt")
    if not os.path.exists(path_file):
        raise ValueError(f"Path file {path_file} not found")
    path = np.loadtxt(path_file, delimiter=',')

    # 3. 创建环境并添加障碍物
    env = KukaEnv(GUI=True)
    env.reset_obstacles()
    env.reset_arm()
    for obs in env_dict["box_obstacles"]:
        half_extents = np.array(obs[3:]) / 2.0
        base_position = np.array(obs[:3])
        env.add_box_obstacle(half_extents, base_position)

    # 4. 可视化路径
    visualize_path(env, path)

    # 5. 绘制关节角度变化曲线
    path = np.array(path)  # (N, D)
    num_points, num_joints = path.shape
    plt.figure(figsize=(10, 6))
    for j in range(num_joints):
        plt.plot(range(num_points), path[:, j], label=f"Joint {j}", marker='o', markersize=5)
    plt.xlabel("Path step")
    plt.ylabel("Joint angle (rad)")
    plt.title(f"Env {env_id}, Sample {sample_idx} Joint Trajectories")
    plt.legend()
    plt.grid(True)
    plt.show()

    env.close()


def save_path_and_env(mode_path_dir, env_idx, path_list, env_dict, mode_dir, mode_env_list):
    for path_idx, path in enumerate(path_list):
        path_np = np.array(path)
        np.savetxt(join(mode_path_dir, f"{env_idx}_{path_idx}.txt"), path_np, fmt='%.6f', delimiter=',')
    with open(join(mode_dir, "envs.json"), "w") as f:
        json.dump(mode_env_list, f)

def generate_bitstar_dataset(mode, env_size, redundant_mode_env_list, config, dataset_dir):
    print(f"Generating BIT* paths for {mode} set (single process)...")
    mode_dir = join(dataset_dir, mode)
    mode_path_dir = join(mode_dir, "bitstar_paths")
    makedirs(mode_path_dir, exist_ok=True)

    mode_env_list = []
    total_env_count = 0
    invalid_env_count = 0

    env = KukaEnv(GUI=config['GUI'])
    while len(mode_env_list) < env_size:
        if total_env_count >= len(redundant_mode_env_list):
            print("redundant_mode_env not enough")
            break
        total_env_count += 1
        env.reset_obstacles()
        env.reset_arm()

        env_dict = redundant_mode_env_list[total_env_count - 1]
        for obs in env_dict['box_obstacles']:
            half_extents = np.array(obs[3:]) / 2.0
            base_position = np.array(obs[:3])
            env.add_box_obstacle(half_extents, base_position)

        x_start_list, x_goal_list = env_dict['start'], env_dict['goal']
        valid_env = True
        path_list, exec_time_list = [], []

        for idx in range(config['num_samples_per_env']):
            x_start, x_goal = x_start_list[idx], x_goal_list[idx]
            path, exec_time = generate_bitstar_path(
                env, x_start, x_goal,
                gui=config['GUI'],
                refine_time_budget=config['refine_time_budget'],
                time_budget=config['time_budget']
            )
            if path is None:
                valid_env = False
                break
            path_list.append(path)
            exec_time_list.append(exec_time)

        if not valid_env:
            invalid_env_count += 1
            print(f"Invalid env: {invalid_env_count}/{total_env_count}")
            continue

        # 用有效编号
        env_idx = len(mode_env_list)
        env_dict['env_id'] = env_idx
        env_dict['bitstar_time'] = exec_time_list
        mode_env_list.append(env_dict)

        # 保存路径
        for path_idx, path in enumerate(path_list):
            path_np = np.array(path)
            np.savetxt(join(mode_path_dir, f"{env_idx}_{path_idx}.txt"), path_np, fmt='%.6f', delimiter=',')

        # 保存 json
        with open(join(mode_dir, "envs.json"), "w", encoding="utf-8") as f:
            json.dump(mode_env_list, f, ensure_ascii=False)

        print(f"{len(mode_env_list)} {mode} envs and {config['num_samples_per_env']*len(mode_env_list)} samples saved.")

    print(f"Finished {mode} set. Total valid envs: {len(mode_env_list)}, Invalid: {invalid_env_count}")


# ==================== 单个环境处理 ====================
def process_single_env(args):
    raw_idx, env_dict, num_samples, gui, refine_time_budget, time_budget = args
    pid = os.getpid()
    try:
        print(f"[Subprocess {pid}] start raw env {raw_idx}", flush=True)
        env = KukaEnv(GUI=gui)

        # 添加障碍物
        for obs in env_dict['box_obstacles']:
            half_extents = np.array(obs[3:]) / 2.0
            base_position = np.array(obs[:3])
            env.add_box_obstacle(half_extents, base_position)

        x_start_list, x_goal_list = env_dict['start'], env_dict['goal']
        path_list, exec_time_list = [], []

        for i in range(num_samples):
            x_start, x_goal = x_start_list[i], x_goal_list[i]
            path, exec_time = generate_bitstar_path(env, x_start, x_goal,
                                                   gui=gui,
                                                   refine_time_budget=refine_time_budget,
                                                   time_budget=time_budget)
            if path is None:
                print(f"[Subprocess {pid}] raw env {raw_idx} failed at sample {i}", flush=True)
                return None
            path_list.append(path)
            exec_time_list.append(exec_time)

        env_dict['bitstar_time'] = exec_time_list
        print(f"[Subprocess {pid}] finished raw env {raw_idx}", flush=True)
        return env_dict, path_list

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Subprocess {pid}] Exception processing raw env {raw_idx}:\n{tb}", flush=True)
        return None

    finally:
        try:
            if 'env' in locals():
                if hasattr(env, 'close'):
                    env.close()
                else:
                    try: p.disconnect()
                    except Exception: pass
        except Exception as e:
            print(f"[Subprocess {pid}] cleanup exception: {e}", flush=True)
        finally:
            print(f"[Subprocess {pid}] exiting raw env {raw_idx}...", flush=True) 

# ==================== 保存函数 ====================
def save_env_paths(mode_path_dir, mode_env_list, env_size):
    for env_item in mode_env_list:
        env_id = env_item['env_id']
        path_list = env_item.get('paths', [])

        # 单独保存到 .txt
        for path_idx, path in enumerate(path_list):
            filename = join(mode_path_dir, f"{env_id}_{path_idx}.txt")
            np.savetxt(filename, np.array(path), fmt='%.6f', delimiter=',')

    return mode_env_list

# ==================== 并行生成 BIT* 数据集 ====================
def generate_bitstar_dataset_parallel(mode, env_size, redundant_mode_env_list,
                                      config, dataset_dir, save_interval=10, n_process=4):
    print(f"Generating BIT* paths for {mode} set (parallel, {n_process} processes)...")
    mode_dir = join(dataset_dir, mode)
    mode_path_dir = join(mode_dir, "bitstar_paths")
    makedirs(mode_path_dir, exist_ok=True)
    final_file = join(mode_dir, "envs.json")

    mode_env_list = []
    invalid_env_count = 0

    # 读取已有 JSON（只保留完整环境）
    if os.path.exists(final_file):
        with open(final_file, "r", encoding="utf-8") as f_json:
            try:
                old_envs = json.load(f_json)

                # === 新增完整性检查 ===
                complete_envs = []
                for env_item in old_envs:
                    env_id = env_item['env_id']
                    ok = True
                    for path_idx in range(config['num_samples_per_env']):
                        path_file = join(mode_path_dir, f"{env_id}_{path_idx}.txt")
                        if not os.path.exists(path_file):
                            ok = False
                            break
                    if ok:
                        complete_envs.append(env_item)

                mode_env_list = complete_envs
                if len(mode_env_list) >= env_size:
                    print(f"{mode} set already completed with {len(mode_env_list)} envs.")
                    return
                print(f"已有 {len(mode_env_list)} 个完整环境，继续剩余部分...")

            except Exception:
                pass

    env_iter = ((i, env_dict) for i, env_dict in enumerate(redundant_mode_env_list))
    from concurrent.futures import wait, FIRST_COMPLETED

    with ProcessPoolExecutor(max_workers=n_process) as executor:
        futures = {}
        # 初始填满任务池
        for _ in range(n_process):
            try:
                i, env_dict = next(env_iter)
                args = (i, env_dict, config['num_samples_per_env'],
                        False, config['refine_time_budget'], config['time_budget'])
                futures[executor.submit(process_single_env, args)] = i
            except StopIteration:
                break

        with tqdm(total=env_size, initial=len(mode_env_list), desc=f"[{mode}] 有效环境进度") as pbar:
            while futures and len(mode_env_list) < env_size:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)

                for f in done:
                    raw_idx = futures.pop(f)
                    result = f.result()
                    if result is None:
                        invalid_env_count += 1
                        print(f"[Main] raw env {raw_idx} invalid -> skipped ({invalid_env_count} so far)")
                    else:
                        res_env_dict, path_list = result
                        # 重新分配有效编号
                        env_id = len(mode_env_list)
                        res_env_dict['env_id'] = env_id
                        res_env_dict['paths'] = path_list

                        mode_env_list.append(res_env_dict)
                        save_env_paths(mode_path_dir, [res_env_dict], env_size=env_size)

                        if len(mode_env_list) % save_interval == 0:
                            with open(final_file, "w", encoding="utf-8") as f_json:
                                json.dump(mode_env_list, f_json, ensure_ascii=False)

                        pbar.update(1)

                    # 提交新任务
                    if len(mode_env_list) < env_size:
                        try:
                            i, env_dict = next(env_iter)
                            args = (i, env_dict, config['num_samples_per_env'],
                                    False, config['refine_time_budget'], config['time_budget'])
                            futures[executor.submit(process_single_env, args)] = i
                        except StopIteration:
                            pass
            for f in futures:
                f.cancel()
            executor.shutdown(wait=False)

    # 最终保存
    print(f"所有进程完成，保存最终结果...")
    save_env_paths(mode_path_dir, mode_env_list, env_size=env_size)
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(mode_env_list, f, ensure_ascii=False)
    print(f"Finished {mode} set. Total valid envs: {len(mode_env_list)}, Invalid: {invalid_env_count}")

# ---------------- Main Program ----------------
if __name__ == "__main__":
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
        # generate_bitstar_dataset(mode, env_size_dict[mode], redundant_mode_env_list, config, dataset_dir)
        generate_bitstar_dataset_parallel(mode, env_size_dict[mode], redundant_mode_env_list, config, dataset_dir, n_process=10, save_interval=10)
    # visualize_env_sample('train', env_id=300, sample_idx=2, dataset_dir=dataset_dir)
