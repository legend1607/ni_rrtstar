import time
import pickle
from copy import copy
from os import makedirs
from os.path import join, exists
from importlib import import_module
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm

# =====================
# 参数配置
# =====================
class Args:
    # 算法选择
    path_planner = 'niarrt_star'   # 'rrt_star', 'irrt_star', 'nrrt_star', 'nirrt_star'
    neural_net = 'pointnet2tf'   # 'none', 'pointnet2', 'unet', 'pointnet'
    connect = 'none'              # 'none', 'bfs'
    device = 'cpu'                # 多进程下 GPU 不建议直接共享，默认 CPU

    # 规划参数
    step_len = 10
    iter_max = 5000
    clearance = 0
    pc_n_points = 2048
    pc_over_sample_scale = 5
    pc_sample_rate = 0.5
    pc_update_cost_ratio = 0.9
    connect_max_trial_attempts = 5

    # 任务相关
    problem = 'random_2d_simple'         # 'block', 'gap', 'random_2d'
    path_len_threshold_percentage = 0.02
    iter_after_initial = 3000
    num_problems = 200         # None 表示全部

args = Args()

# =====================
# 参数合法性检查
# =====================
if args.path_planner in ['rrt_star', 'irrt_star']:
    assert args.neural_net == 'none'
else:
    assert args.neural_net != 'none'

# =====================
# 选择路径规划器
# =====================
if args.neural_net == 'none':
    path_planner_name = args.path_planner
elif args.neural_net in ['pointnet2', 'pointnet','pointnet2tf']:
    path_planner_name = args.path_planner + '_png'
elif args.neural_net == 'unet':
    path_planner_name = args.path_planner + '_gng'
else:
    raise NotImplementedError

if args.connect != 'none':
    path_planner_name += '_c'
path_planner_name += '_2d'

get_path_planner = getattr(
    import_module('path_planning_classes.' + path_planner_name),
    'get_path_planner'
)

# =====================
# 选择神经网络包装器
# =====================
if args.neural_net == 'none':
    NeuralWrapper = None
elif args.neural_net in ['pointnet2', 'pointnet','pointnet2tf']:
    neural_wrapper_name = args.neural_net + '_wrapper'
    if args.connect != 'none':
        neural_wrapper_name += '_connect_' + args.connect
    NeuralWrapper = getattr(
        import_module('wrapper.pointnet_pointnet2.' + neural_wrapper_name),
        'PNGWrapper'
    )
elif args.neural_net == 'unet':
    neural_wrapper_name = args.neural_net + '_wrapper'
    if args.connect != 'none':
        raise NotImplementedError
    NeuralWrapper = getattr(
        import_module('wrapper.unet.' + neural_wrapper_name),
        'GNGWrapper'
    )
else:
    raise NotImplementedError

# =====================
# 选择环境生成器
# =====================
get_env_configs = getattr(
    import_module('datasets.planning_problem_utils_2d'),
    'get_' + args.problem + '_env_configs'
)
get_problem_input = getattr(
    import_module('datasets.planning_problem_utils_2d'),
    'get_' + args.problem + '_problem_input'
)

# =====================
# 初始化神经网络包装器
# =====================
if NeuralWrapper is None:
    neural_wrapper = None
else:
    neural_wrapper = NeuralWrapper(device=args.device)

# =====================
# 获取环境配置列表
# =====================
if args.problem == 'random_2d' or args.problem == 'random_2d_simple':
    args.clearance = 3

env_config_list = get_env_configs()
if args.num_problems is None:
    num_problems = len(env_config_list)
else:
    assert args.num_problems <= len(env_config_list)
    num_problems = args.num_problems

result_folderpath = 'results/evaluation/2d'
makedirs(result_folderpath, exist_ok=True)

connect_str = f"-c-{args.connect}" if args.connect != 'none' else ''
eval_setting = f"{args.problem}-{args.path_planner}{connect_str}-{args.neural_net}-{num_problems}"
result_filepath = join(result_folderpath, eval_setting + '.pickle')

# =====================
# 加载已存在结果
# =====================
if exists(result_filepath):
    with open(result_filepath, 'rb') as f:
        env_result_config_list = pickle.load(f)
else:
    env_result_config_list = []

# =====================
# 单环境评估函数
# =====================
def evaluate_env(env_idx_config):
    env_idx, env_config = env_idx_config
    # 跳过已计算过的环境
    if env_idx < len(env_result_config_list):
        print(f"Env {env_idx} already evaluated, skipping...")
        return None

    problem = get_problem_input(env_config)
    path_planner = get_path_planner(args, problem, neural_wrapper)

    if args.problem == 'block':
        path_len_threshold = problem['best_path_len'] * (1 + args.path_len_threshold_percentage)
        path_len_list = path_planner.planning_block_gap(path_len_threshold)
    elif args.problem == 'gap':
        path_len_list = path_planner.planning_block_gap(problem['flank_path_len'])
    elif args.problem == 'random_2d' or args.problem == 'random_2d_simple':
        path_len_list, time_list=path_planner.planning_random(args.iter_after_initial)
    else:
        raise NotImplementedError

    env_result_config = copy(env_config)
    env_result_config['result'] = path_len_list
    env_result_config['time'] = time_list
    return (env_idx, env_result_config)

# =====================
# 多进程评估
# =====================
if __name__ == '__main__':
    start_time = time.time()
    num_workers = min(cpu_count(), num_problems)
    print(f"Using {num_workers} parallel workers for evaluation...")

    # 初始化结果字典并恢复旧结果
    results_dict = {i: None for i in range(num_problems)}
    if len(env_result_config_list) > 0:
        print(f"[Info] Loaded {len(env_result_config_list)} previous results.")
        for i, env_res in enumerate(env_result_config_list):
            results_dict[i] = env_res

    # 并行执行
    with Pool(num_workers) as pool:
        for res in tqdm(pool.imap_unordered(evaluate_env, enumerate(env_config_list[:num_problems])),
                        total=num_problems,
                        desc="Evaluating Envs"):
            if res is None:
                continue

            env_idx, env_result_config = res
            results_dict[env_idx] = env_result_config

            # 每次更新保存所有非空结果
            sorted_results = [results_dict[i] for i in sorted(results_dict.keys()) if results_dict[i] is not None]

            with open(result_filepath, 'wb') as f:
                pickle.dump(sorted_results, f)

            # 估算剩余时间
            elapsed = time.time() - start_time
            remaining = elapsed * (num_problems / (env_idx + 1) - 1) / 60
            print(f"Evaluated {env_idx+1}/{num_problems}, remaining ~{int(remaining)} min")
