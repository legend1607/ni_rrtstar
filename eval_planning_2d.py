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
    path_planner = 'nirrt_star'   # 'rrt_star', 'irrt_star', 'nrrt_star', 'nirrt_star'
    neural_net = 'pointnet2'      # 'none', 'pointnet2', 'unet', 'pointnet'
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
    problem = 'random_2d'         # 'block', 'gap', 'random_2d'
    path_len_threshold_percentage = 0.02
    iter_after_initial = 3000
    num_problems = None           # None 表示全部

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
elif args.neural_net in ['pointnet2', 'pointnet']:
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
elif args.neural_net in ['pointnet2', 'pointnet']:
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
if args.problem == 'random_2d':
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
        return None

    problem = get_problem_input(env_config)
    path_planner = get_path_planner(args, problem, neural_wrapper)

    if args.problem == 'block':
        path_len_threshold = problem['best_path_len'] * (1 + args.path_len_threshold_percentage)
        path_len_list = path_planner.planning_block_gap(path_len_threshold)
    elif args.problem == 'gap':
        path_len_list = path_planner.planning_block_gap(problem['flank_path_len'])
    elif args.problem == 'random_2d':
        path_len_list = path_planner.planning_random(args.iter_after_initial)
    else:
        raise NotImplementedError

    env_result_config = copy(env_config)
    env_result_config['result'] = path_len_list
    return (env_idx, env_result_config)

# =====================
# 多进程评估
# =====================
if __name__ == '__main__':
    start_time = time.time()
    num_workers = min(cpu_count(), num_problems)
    print(f"Using {num_workers} parallel workers for evaluation...")

    # 使用 Pool.map 并行
    with Pool(num_workers) as pool:
        results_dict = {}
        for res in tqdm(pool.imap_unordered(evaluate_env, enumerate(env_config_list[:num_problems])),
                        total=num_problems,
                        desc="Evaluating Envs"):
            if res is None:
                continue
            env_idx, env_result_config = res
            results_dict[env_idx] = env_result_config
            # 每完成一个环境就保存，防止意外中断丢数据
            with open(result_filepath, 'wb') as f:
                pickle.dump(env_result_config_list, f)
            # 估算剩余时间
            elapsed = time.time() - start_time
            remaining = elapsed * (num_problems / (env_idx + 1) - 1) / 60
            print(f"Evaluated {env_idx+1}/{num_problems}, remaining ~{int(remaining)} min")
