import numpy as np
from importlib import import_module
from copy import copy
import cv2

# -------------------------------
# 参数设置
# -------------------------------
args = {
    # 基础设置
    "path_planner": "nirrt_star",    # rrt_star, irrt_star, nrrt_star, nirrt_star
    "neural_net": "pointnet2",       # none, pointnet2, pointnet, unet
    "connect": "none",               # none, bfs, astar
    "device": "cuda",                # cuda 或 cpu

    # 规划参数
    "step_len": 10,
    "iter_max": 1000,
    "clearance": 0,                  # block/gap=0, random_2d=3
    "pc_n_points": 2048,
    "pc_over_sample_scale": 5,
    "pc_sample_rate": 0.5,
    "pc_update_cost_ratio": 0.9,
    "connect_max_trial_attempts": 5,

    # 任务设置
    "problem": "random_2d",          # block, gap, random_2d
    "result_folderpath": "results",
    "path_len_threshold_percentage": 0.02,  # block 用
    "iter_after_initial": 1000,      # random_2d 用
}

# -------------------------------
# 参数字典 -> 点访问
# -------------------------------
class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)
    def __setattr__(self, key, value):
        self[key] = value

args = AttrDict(args)

# -------------------------------
# 初始化 NeuralWrapper
# -------------------------------
if args.neural_net == "none":
    NeuralWrapper = None
elif args.neural_net in ["pointnet2", "pointnet"]:
    neural_wrapper_name = args.neural_net + "_wrapper"
    if args.connect != "none":
        neural_wrapper_name += "_connect_" + args.connect
    NeuralWrapper = getattr(
        import_module("wrapper.pointnet_pointnet2." + neural_wrapper_name),
        "PNGWrapper"
    )
elif args.neural_net == "unet":
    neural_wrapper_name = args.neural_net + "_wrapper"
    if args.connect != "none":
        raise NotImplementedError("Unet 不支持 connect 选项")
    NeuralWrapper = getattr(
        import_module("wrapper.unet." + neural_wrapper_name),
        "GNGWrapper"
    )
else:
    raise NotImplementedError(f"不支持的神经网络类型: {args.neural_net}")

# 初始化神经网络包装器
if NeuralWrapper is None:
    neural_wrapper = None
else:
    neural_wrapper = NeuralWrapper(device=args.device)
    print("PointNet++ wrapper is initialized.")

# -------------------------------
# 获取环境配置函数
# -------------------------------
get_env_configs = getattr(
    import_module("datasets.planning_problem_utils_2d"),
    f"get_{args.problem}_env_configs"
)
get_problem_input = getattr(
    import_module("datasets.planning_problem_utils_2d"),
    f"get_{args.problem}_problem_input"
)

# -------------------------------
# 设置 clearance
# -------------------------------
if args.problem == "random_2d":
    args.clearance = 3

# -------------------------------
# 选择一个环境并规划
# -------------------------------
env_config_list = get_env_configs()
env_config_index = np.random.randint(len(env_config_list))
print("选中的环境编号: ", env_config_index)

problem = get_problem_input(env_config_list[env_config_index])

# 获取路径规划器
path_planner_name = args.path_planner
if args.neural_net != "none":
    path_planner_name += "_png" if args.neural_net in ["pointnet2", "pointnet"] else "_gng"
if args.connect != "none":
    path_planner_name += "_c"
path_planner_name += "_2d"

get_path_planner = getattr(
    import_module("path_planning_classes." + path_planner_name),
    "get_path_planner"
)
print("使用的路径规划器: ", path_planner_name)
path_planner = get_path_planner(args, problem, neural_wrapper)

# 执行规划
path_planner.planning(visualize=True)
