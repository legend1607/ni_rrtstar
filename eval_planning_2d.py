import time
import pickle
from copy import copy
from os import makedirs
from os.path import join, exists
from importlib import import_module


# =====================
# 参数在这里直接写死
# =====================
class Args:
    # 算法选择
    path_planner = 'nirrt_star'   # 'rrt_star', 'irrt_star', 'nrrt_star', 'nirrt_star'
    neural_net = 'pointnet2'      # 'none', 'pointnet2', 'unet', 'pointnet'
    connect = 'none'              # 'none', 'bfs'
    device = 'cuda'               # 'cuda', 'cpu'

    # 规划参数
    step_len = 10
    iter_max = 50000
    clearance = 0
    pc_n_points = 2048
    pc_over_sample_scale = 5
    pc_sample_rate = 0.5
    pc_update_cost_ratio = 0.9
    connect_max_trial_attempts = 5

    # 任务相关
    problem = 'random_2d'         # 'block', 'gap', 'random_2d'
    path_len_threshold_percentage = 0.02
    iter_after_initial = 5000
    num_problems = None           # None 表示全部


args = Args()

# * sanity check
if args.path_planner in ['rrt_star', 'irrt_star']:
    assert args.neural_net == 'none'
else:
    assert args.neural_net != 'none'

# * set get_path_planner
if args.neural_net == 'none':
    path_planner_name = args.path_planner
elif args.neural_net in ['pointnet2', 'pointnet']:
    path_planner_name = args.path_planner + '_png'
elif args.neural_net == 'unet':
    path_planner_name = args.path_planner + '_gng'
else:
    raise NotImplementedError
if args.connect != 'none':
    path_planner_name = path_planner_name + '_c'
path_planner_name = path_planner_name + '_2d'
get_path_planner = getattr(import_module('path_planning_classes.' + path_planner_name), 'get_path_planner')

# * set NeuralWrapper
if args.neural_net == 'none':
    NeuralWrapper = None
elif args.neural_net in ['pointnet2', 'pointnet']:
    neural_wrapper_name = args.neural_net + '_wrapper'
    if args.connect != 'none':
        neural_wrapper_name = neural_wrapper_name + '_connect_' + args.connect
    NeuralWrapper = getattr(import_module('wrapper.pointnet_pointnet2.' + neural_wrapper_name), 'PNGWrapper')
elif args.neural_net == 'unet':
    neural_wrapper_name = args.neural_net + '_wrapper'
    if args.connect != 'none':
        raise NotImplementedError
    NeuralWrapper = getattr(import_module('wrapper.unet.' + neural_wrapper_name), 'GNGWrapper')
else:
    raise NotImplementedError

# * set planning problem
get_env_configs = getattr(import_module('datasets.planning_problem_utils_2d'), 'get_' + args.problem + '_env_configs')
get_problem_input = getattr(import_module('datasets.planning_problem_utils_2d'), 'get_' + args.problem + '_problem_input')

# * main
if NeuralWrapper is None:
    neural_wrapper = None
else:
    neural_wrapper = NeuralWrapper(device=args.device)

if args.problem == 'random_2d':
    args.clearance = 3
print(vars(args))  # 打印所有参数，方便检查

env_config_list = get_env_configs()
if args.num_problems is None:
    num_problems = len(env_config_list)
else:
    assert args.num_problems <= len(env_config_list)
    num_problems = args.num_problems

result_folderpath = 'results/evaluation/2d'
makedirs(result_folderpath, exist_ok=True)

if args.connect != 'none':
    connect_str = '-c-' + args.connect
else:
    connect_str = ''
eval_setting = args.problem + '-' + args.path_planner + connect_str + '-' + args.neural_net + '-' + str(num_problems)
result_filepath = join(result_folderpath, eval_setting + '.pickle')

if not exists(result_filepath):
    env_result_config_list = []
else:
    with open(result_filepath, 'rb') as f:
        env_result_config_list = pickle.load(f)

eval_start_time = time.time()
for env_idx, env_config in enumerate(env_config_list[:num_problems]):
    if env_idx < len(env_result_config_list):
        time_left = (time.time() - eval_start_time) * (num_problems / (env_idx + 1) - 1) / 60
        print("Evaluated {0}/{1} in the loaded file, remaining time: {2} min for {3}".format(
            env_idx + 1, num_problems, int(time_left), eval_setting))
        continue

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
    env_result_config_list.append(env_result_config)

    with open(result_filepath, 'wb') as f:
        pickle.dump(env_result_config_list, f)

    time_left = (time.time() - eval_start_time) * (num_problems / (env_idx + 1) - 1) / 60
    print("Evaluated {0}/{1}, remaining time: {2} min for {3}".format(
        env_idx + 1, num_problems, int(time_left), eval_setting))
