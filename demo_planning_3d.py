from matplotlib import pyplot as plt
import numpy as np
from importlib import import_module

# ================================
# 直接写参数
# ================================
class Args:
    # Path planner options: rrt_star, irrt_star, nrrt_star, nirrt_star
    path_planner = 'nirrt_star'
    # Neural net: none, pointnet2, unet, pointnet
    neural_net = 'pointnet2'
    # Connection strategy: none, bfs, astar
    connect = 'none'
    device = 'cuda'

    # Planner hyperparameters
    step_len = 10
    iter_max = 500
    clearance = 2
    pc_n_points = 2048
    pc_over_sample_scale = 5
    pc_sample_rate = 0.5
    pc_update_cost_ratio = 1.0
    connect_max_trial_attempts = 5

    # Problem and dataset
    problem = 'random_3d'
    result_folderpath = 'results'
    path_len_threshold_percentage = 0.02
    iter_after_initial = 1000

args = Args()

# ================================
# sanity check
# ================================
if args.path_planner in ['rrt_star', 'irrt_star']:
    assert args.neural_net == 'none'
else:
    assert args.neural_net != 'none'

# ================================
# set get_path_planner
# ================================
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
path_planner_name += '_3d'

get_path_planner = getattr(
    import_module('path_planning_classes_3d.' + path_planner_name),
    'get_path_planner'
)

# ================================
# set NeuralWrapper
# ================================
if args.neural_net in ['none']:
    NeuralWrapper = None
elif args.neural_net in ['pointnet2', 'pointnet']:
    neural_wrapper_name = args.neural_net + '_wrapper'
    if args.connect != 'none':
        neural_wrapper_name += '_connect_' + args.connect
    NeuralWrapper = getattr(
        import_module('wrapper_3d.pointnet_pointnet2.' + neural_wrapper_name),
        'PNGWrapper'
    )
else:
    raise NotImplementedError

# ================================
# set planning problem
# ================================
if args.problem != 'random_3d':
    raise NotImplementedError

get_env_configs = getattr(
    import_module('datasets_3d.planning_problem_utils_3d'),
    'get_' + args.problem + '_env_configs'
)
get_problem_input = getattr(
    import_module('datasets_3d.planning_problem_utils_3d'),
    'get_' + args.problem + '_problem_input'
)

# ================================
# main
# ================================
def main():
    if NeuralWrapper is None:
        neural_wrapper = None
    else:
        if args.problem.startswith('random'):
            neural_wrapper = NeuralWrapper(device=args.device)
        elif args.problem.startswith('kuka'):
            neural_wrapper = NeuralWrapper(coord_dim=7, device=args.device)

    # For random_3d, default clearance is 2
    if args.problem == 'random_3d':
        args.clearance = 2

    print("Planner and environment parameters:")
    print(args.__dict__)

    # Load environment configurations
    env_config_list = get_env_configs()
    env_config_index = np.random.randint(len(env_config_list))
    print("Selected env_config_index:", env_config_index)
    env_config = env_config_list[env_config_index]

    # Get problem input
    problem = get_problem_input(env_config)

    # Create path planner
    path_planner = get_path_planner(args, problem, neural_wrapper)

    # Run planning with visualization
    path_planner.planning(visualize=True)


if __name__ == "__main__":
    main()
