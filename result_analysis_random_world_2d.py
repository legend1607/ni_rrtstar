import os
import pickle
import argparse
from os.path import join, exists
import csv

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 参数解析
# ------------------------------
argparser = argparse.ArgumentParser()
argparser.add_argument('--random_dataset_len', type=int, default=200)
args = argparser.parse_args()
random_dataset_len = args.random_dataset_len

# ------------------------------
# 方法与文件名
# ------------------------------
methods = ['rrt', 'irrt', 'nrrt_png', 'nrrt_gng', 'nrrt_png_c', 'nirrt_png', 'nirrt_png_c','niarrt_png']
result_filenames = [
    'random_2d_simple-rrt_star-none',
    'random_2d_simple-irrt_star-none',
    'random_2d_simple-nrrt_star-pointnet2',
    'random_2d_simple-nrrt_star-unet',
    'random_2d_simple-nrrt_star-c-bfs-pointnet2',
    'random_2d_simple-nirrt_star-pointnet2',
    'random_2d_simple-nirrt_star-c-bfs-pointnet2',
    'random_2d_simple-niarrt_star-pointnet2tf'
]
for i in range(len(result_filenames)):
    result_filenames[i] += '-' + str(random_dataset_len)

# ------------------------------
# 文件路径设置
# ------------------------------
visualization_folderpath = join('visualization', 'evaluation')
os.makedirs(visualization_folderpath, exist_ok=True)
results_folderpath = 'results/evaluation/2d'
os.makedirs(visualization_folderpath, exist_ok=True)

# ------------------------------
# 加载结果文件（跳过不存在的）
# ------------------------------
random_results = {}
available_methods = []
for method, result_filename in zip(methods, result_filenames):
    filepath = join(results_folderpath, result_filename + '.pickle')
    if not exists(filepath):
        print(f"⚠️  跳过 {filepath} （文件不存在）")
        continue
    with open(filepath, 'rb') as f:
        random_results[method] = pickle.load(f)
        available_methods.append(method)

if not available_methods:
    raise FileNotFoundError("❌ 没有找到任何有效的 .pickle 文件，请检查路径和文件名。")

print(f"✅ 已加载结果：{available_methods}")

# ------------------------------
# 初始化指标
# ------------------------------
success_rate_dict = {}
IFS_dict = {}
planning_time_dict = {}
path_len_var_dict = {}
max_iter = max([len(res['result']) for method in available_methods for res in random_results[method]])
success_per_iter_dict = {}
avg_path_cost_ratio_dict = {}

# ------------------------------
# Success Rate、IFS、Planning Time、路径方差、Success per Iteration
# ------------------------------
for method in available_methods:
    total_env = len(random_results[method])
    success_count = 0
    IFS_list = []
    planning_time_list = []
    final_path_len_list = []
    ratio_per_env = []

    for i, res in enumerate(random_results[method]):
        feasible_indices = np.where(np.array(res['result']) < np.inf)[0]
        if len(feasible_indices) > 0:
            success_count += 1
            first_success_idx = feasible_indices[0]
            IFS_list.append(first_success_idx)
            planning_time_list.append(np.sum(res.get('time', [0.0])))
            final_path_len_list.append(res['result'][-1])

        # 路径成本比曲线（相对RRT*初始可行路径）
        if 'rrt' in available_methods:
            rrt_res = random_results['rrt'][i]
            feasible_idx_rrt = np.where(np.array(rrt_res['result']) < np.inf)[0]
            if len(feasible_idx_rrt) > 0:
                rrt_initial = rrt_res['result'][feasible_idx_rrt[0]]
                ratio = np.array(res['result']) / rrt_initial
                ratio_per_env.append(ratio)
    # 平均路径成本比
    if ratio_per_env:
        avg_ratio = np.mean(np.vstack([r[:max_iter] if len(r) >= max_iter else np.pad(r,(0,max_iter-len(r)),'edge') 
                                       for r in ratio_per_env]), axis=0)
        avg_path_cost_ratio_dict[method] = avg_ratio

    success_rate_dict[method] = success_count / total_env
    IFS_dict[method] = IFS_list
    planning_time_dict[method] = planning_time_list
    path_len_var_dict[method] = np.var(final_path_len_list) if final_path_len_list else np.nan

    # Success per Iteration
    success_per_iter = np.zeros(max_iter)
    for res in random_results[method]:
        feasible_indices = np.where(np.array(res['result']) < np.inf)[0]
        if len(feasible_indices) > 0:
            success_per_iter[feasible_indices[0]:] += 1
    success_per_iter_dict[method] = success_per_iter / total_env

# ------------------------------
# 绘制 Success Rate vs Iterations
# ------------------------------
fig, ax = plt.subplots(figsize=(8,5))
colors = ['k','gray','C0','C1','C2','C4','C5']
for method, color in zip(available_methods, colors[:len(available_methods)]):
    ax.plot(range(max_iter), success_per_iter_dict[method], label=method, color=color)
ax.set_xlabel("Iteration")
ax.set_ylabel("Success Rate")
ax.set_title("Planning Success Rate vs Iterations")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(join(visualization_folderpath, 'success_rate_vs_iterations.png'))
plt.close(fig)

# ------------------------------
# IFS 分布柱状图
# ------------------------------
fig, ax = plt.subplots(figsize=(8,5))
bin_edges = np.arange(0, max_iter+100, 100)
for method, color in zip(available_methods, colors[:len(available_methods)]):
    ax.hist(IFS_dict[method], bins=bin_edges, alpha=0.5, label=method, color=color)
ax.set_xlabel("Iteration to First Success (IFS)")
ax.set_ylabel("Number of Environments")
ax.set_title("IFS Distribution")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(join(visualization_folderpath, 'IFS_distribution.png'))
plt.close(fig)

# ------------------------------
# 规划时间箱线图
# ------------------------------
fig, ax = plt.subplots(figsize=(8,5))
data = [planning_time_dict[method] for method in available_methods]
ax.boxplot(data, labels=available_methods)
ax.set_ylabel("Total Planning Time (s)")
ax.set_title("Planning Time Distribution")
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
fig.savefig(join(visualization_folderpath, 'planning_time_boxplot.png'))
plt.close(fig)

# ------------------------------
# 平滑连续路径成本比曲线
# ------------------------------
avg_path_cost_ratio_dict = {}
for method in available_methods:
    ratio_per_env = []
    for i, res in enumerate(random_results[method]):
        if 'rrt' not in available_methods:
            continue
        rrt_res = random_results['rrt'][i]
        feasible_idx_rrt = np.where(np.array(rrt_res['result']) < np.inf)[0]
        if len(feasible_idx_rrt) == 0:
            continue
        rrt_initial = rrt_res['result'][feasible_idx_rrt[0]]
        ratio = np.array(res['result']) / rrt_initial
        ratio_per_env.append(ratio)
    if ratio_per_env:
        max_iter = max([len(r) for r in ratio_per_env])
        avg_ratio = np.mean(np.vstack([
            r[:max_iter] if len(r) >= max_iter else np.pad(r, (0, max_iter-len(r)), 'edge')
            for r in ratio_per_env]), axis=0)
        avg_path_cost_ratio_dict[method] = avg_ratio

fig, ax = plt.subplots(figsize=(8,5))
for method, color in zip(available_methods, ['k','gray','C0','C1','C2','C4','C5']):
    if method in avg_path_cost_ratio_dict:
        ax.plot(range(len(avg_path_cost_ratio_dict[method])), avg_path_cost_ratio_dict[method], label=method)
ax.set_xlabel("Iteration")
ax.set_ylabel("Average Path Cost Ratio (vs RRT*)")
ax.set_title("Average Path Cost Ratio Over Iterations (Smooth)")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(join(visualization_folderpath,'avg_path_cost_ratio_smooth.png'))
plt.close(fig)

# ------------------------------
# 稀疏点路径成本比曲线（iter_after_initial_list）
# ------------------------------
iter_after_initial_list = range(0, 3000+250, 250)
random_analysis = {}
path_cost_mean = {}

for method in available_methods:
    random_analysis[method] = {}
    for iter_after_initial in iter_after_initial_list:
        random_analysis[method][iter_after_initial] = []
    for i in range(len(random_results[method])):
        feasible_indices = np.where(np.array(random_results[method][i]['result']) < np.inf)[0]
        if len(feasible_indices) == 0 or 'rrt' not in available_methods:
            continue
        initial_idx = feasible_indices[0]
        rrt_res = random_results['rrt'][i]
        feasible_idx_rrt = np.where(np.array(rrt_res['result']) < np.inf)[0]
        if len(feasible_idx_rrt) == 0:
            continue
        rrt_initial = rrt_res['result'][feasible_idx_rrt[0]]
        for iter_after_initial in iter_after_initial_list:
            idx = initial_idx + iter_after_initial
            if idx < len(random_results[method][i]['result']):
                ratio = random_results[method][i]['result'][idx] / rrt_initial
            else:
                ratio = random_results[method][i]['result'][-1] / rrt_initial
            random_analysis[method][iter_after_initial].append(ratio)
    # 计算均值
    path_cost_mean[method] = [np.mean(random_analysis[method][iter_key]) for iter_key in iter_after_initial_list]

fig, ax = plt.subplots(figsize=(8,5))
for method, color in zip(available_methods, ['k','gray','C0','C1','C2','C4','C5']):
    ax.plot(list(iter_after_initial_list), path_cost_mean[method], marker='.', linestyle='-', label=method)
ax.set_xlabel("Iterations after first feasible path")
ax.set_ylabel("Average Path Cost Ratio (vs RRT*)")
ax.set_title("Average Path Cost Ratio Over Iterations (Sparse)")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(join(visualization_folderpath,'avg_path_cost_ratio_sparse.png'))
plt.close(fig)

print("✅ 平滑与稀疏路径成本比曲线已保存")

# ------------------------------
# 保存其他指标到 CSV
# ------------------------------
csv_path = join(visualization_folderpath, 'planning_performance_summary.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Method','SuccessRate','IFS_Mean','IFS_Var','PlanningTime_Mean','PathLen_Var'])
    for method in available_methods:
        IFS_array = np.array(IFS_dict[method])
        time_array = np.array(planning_time_dict[method])
        writer.writerow([
            method,
            success_rate_dict[method],
            np.mean(IFS_array) if len(IFS_array)>0 else np.nan,
            np.var(IFS_array) if len(IFS_array)>0 else np.nan,
            np.mean(time_array) if len(time_array)>0 else np.nan,
            path_len_var_dict[method]
        ])
print(f"✅ 指标 CSV 已保存: {csv_path}")
print(f"✅ 图像已保存到 {visualization_folderpath}")
