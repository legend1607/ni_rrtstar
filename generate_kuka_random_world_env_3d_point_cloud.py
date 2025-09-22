import glob
import json
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import os
import time
import traceback
import numpy as np
import pybullet as p
from os.path import join, exists
from os import makedirs
from tqdm import tqdm

from environment.kuka_env import KukaEnv

# ---------------- 配置参数 ----------------
config = {
    'dataset_dir': "data/kuka_random_3d",
    'modes': ['train','val','test'],
    'n_points': 4096,          # 点云采样数量
    'over_sample_scale': 5,
    'point_cloud_clearance': 0.0,
    'start_radius': 0.5,
    'goal_radius': 0.5,
    'path_radius': 0.5,
}

# ---------------- 工具函数 ----------------
def get_point_cloud_mask_around_points(point_cloud, points, neighbor_radius=3):
    point_cloud = np.asarray(point_cloud)
    points = np.asarray(points)
    assert point_cloud.shape[1] == points.shape[1], "point_cloud 和 points 维度不一致"
    diff = point_cloud[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    neighbor_mask = dist < neighbor_radius
    return np.any(neighbor_mask, axis=1)

def generate_joint_space_point_cloud(env, n_points, path=None, path_radius=0.1, path_sample_ratio=0.3):
    """
    生成关节空间点云，并保证 path_sample_ratio 的点来自路径附近。
    
    参数:
        env: KukaEnv 实例
        n_points: 点云总数
        path: 路径点数组, shape=(num_path_points, dof)
        path_radius: 路径附近采样半径
        path_sample_ratio: 点云中路径附近点比例
    """
    point_cloud = []

    # 计算路径附近点数量
    n_path_points = int(n_points * path_sample_ratio) if path is not None else 0

    # ---------------- 采样路径附近点 ----------------
    if path is not None and n_path_points > 0:
        path_points_collected = 0
        while path_points_collected < n_path_points:
            idx = np.random.randint(len(path))
            q_center = path[idx]
            q = q_center + np.random.uniform(-path_radius, path_radius, size=q_center.shape)
            if env.is_state_free(q):
                point_cloud.append(q)
                path_points_collected += 1

    # ---------------- 随机采样剩余自由点 ----------------
    while len(point_cloud) < n_points:
        q = env.uniform_sample()
        if env.is_state_free(q):
            point_cloud.append(q)

    return np.array(point_cloud, dtype=np.float32)

# ---------------- 保存最终 npz ----------------
def save_dataset_full(pc_dataset, dataset_dir, mode):
    makedirs(dataset_dir, exist_ok=True)
    filename = join(dataset_dir, f"{mode}.npz")
    pc_dataset_saved = {
        k: np.array(v) if k == 'token' else np.stack(v, axis=0)
        for k, v in pc_dataset.items() if len(v) > 0
    }
    np.savez(filename, **pc_dataset_saved)
    print(f"{mode} dataset saved: {len(pc_dataset['token'])} samples to {filename}")
    # 打印最终 npz 结构
    print("[最终NPZ结构信息]")
    with np.load(filename, allow_pickle=True) as data:
        for k in data.files:
            print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")
    return filename

# ---------------- 子进程函数 ----------------
def process_env(args):
    mode, env_dict, config = args
    env_idx = env_dict['env_id']
    env = None
    try:
        env = KukaEnv(GUI=False)
        env.reset_obstacles()
        env.reset_arm()
        for obs in env_dict['box_obstacles']:
            half_extents = np.array(obs[3:]) / 2.0
            base_position = np.array(obs[:3])
            env.add_box_obstacle(half_extents, base_position)

        tokens, pcs, starts, goals, frees, paths = [], [], [], [], [], []

        for sample_idx, (x_start, x_goal) in enumerate(zip(env_dict['start'], env_dict['goal'])):
            if(sample_idx>1):
                break
            env.init_state = np.array(x_start)
            env.goal_state = np.array(x_goal)
            token = f"{mode}-{env_idx}_{sample_idx}"

            # BIT* 路径
            if "paths" in env_dict:
                path = np.array(env_dict['paths'][sample_idx])
            else:
                path_file = join(config['dataset_dir'], mode, "bitstar_paths", f"{env_idx}_{sample_idx}.txt")
                path = np.loadtxt(path_file, delimiter=',')

            pc = generate_joint_space_point_cloud(
                env,
                n_points=config['n_points'],
                path=path,
                path_radius=0.4,
                path_sample_ratio=0.3
            )

            around_start_mask = get_point_cloud_mask_around_points(pc, np.array(x_start)[np.newaxis, :], config['start_radius'])
            around_goal_mask = get_point_cloud_mask_around_points(pc, np.array(x_goal)[np.newaxis, :], config['goal_radius'])
            around_path_mask = get_point_cloud_mask_around_points(pc, path, config['path_radius'])
            freespace_mask = (1 - around_start_mask) * (1 - around_goal_mask)

            tokens.append(token)
            pcs.append(pc.astype(np.float32))
            starts.append(around_start_mask.astype(np.float32))
            goals.append(around_goal_mask.astype(np.float32))
            frees.append(freespace_mask.astype(np.float32))
            paths.append(around_path_mask.astype(np.float32))


        return {'token': tokens, 'pc': pcs, 'start': starts, 'goal': goals, 'free': frees, 'path': paths}

    except Exception:
        tb = traceback.format_exc()
        print(f"[Subprocess] Exception processing env {env_idx}:\n{tb}", flush=True)
        return None

    finally:
        if env is not None:
            try:
                env.close()
            except:
                try:
                    p.disconnect()
                except:
                    pass

# ---------------- 并行生成最终点云 ----------------
def generate_point_cloud_parallel(mode, env_list, config, num_workers=None):
    if num_workers is None:
        from multiprocessing import cpu_count
        num_workers = max(1, cpu_count() - 1)

    dataset_dir = join(config['dataset_dir'], mode)
    makedirs(dataset_dir, exist_ok=True)

    pc_dataset = {k: [] for k in ['token','pc','start','goal','free','path']}

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = {}
    env_iter = iter(env_list)

    # 初始提交任务
    for _ in range(num_workers):
        try:
            e = next(env_iter)
            futures[executor.submit(process_env, (mode,e,config))] = e['env_id']
        except StopIteration:
            break

    with tqdm(total=len(env_list), desc=f"[{mode}] env progress") as pbar:
        while futures:
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for f in done:
                env_id = futures.pop(f)
                try:
                    result = f.result()
                except Exception as e:
                    print(f"[Main] env {env_id} raised exception: {e}")
                    result = None

                if result is not None:
                    for k in pc_dataset.keys():
                        pc_dataset[k].extend(result[k])
                pbar.update(1)

                # 提交新任务
                try:
                    e = next(env_iter)
                    futures[executor.submit(process_env, (mode,e,config))] = e['env_id']
                except StopIteration:
                    pass

    # 关闭 Executor
    executor.shutdown(wait=True, cancel_futures=True)

    # 保存最终 npz
    save_dataset_full(pc_dataset, dataset_dir, mode)

# ---------------- 主程序 ----------------
if __name__ == "__main__":
    for mode in config['modes']:
        envs_file = join(config['dataset_dir'], mode, "envs.json")
        if not exists(envs_file):
            print(f"环境文件不存在: {envs_file}")
            continue

        with open(envs_file, 'r') as f:
            env_list = json.load(f)
        env_list = sorted(env_list, key=lambda e: e["env_id"])

        # 并行生成并保存最终 npz
        generate_point_cloud_parallel(mode, env_list, config, num_workers=8)
