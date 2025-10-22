import os
import pickle
import argparse
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# 参数
# -------------------------------
argparser = argparse.ArgumentParser()
argparser.add_argument('--random_dataset_len', type=int, default=500)
args = argparser.parse_args()
random_dataset_len = args.random_dataset_len

methods = ['rrt', 'irrt', 'nrrt_png', 'nrrt_gng', 'nrrt_png_c', 'nirrt_png', 'nirrt_png_c']
result_filenames = [
    'random_2d-rrt_star-none',
    'random_2d-irrt_star-none',
    'random_2d-nrrt_star-pointnet2',
    'random_2d-nrrt_star-unet',
    'random_2d-nrrt_star-c-bfs-pointnet2',
    'random_2d-nirrt_star-pointnet2',
    'random_2d-nirrt_star-c-bfs-pointnet2',
]
for i in range(len(result_filenames)):
    result_filenames[i] += '-'+str(random_dataset_len)

visualization_folderpath = join('visualization', 'evaluation')
os.makedirs(visualization_folderpath, exist_ok=True)
results_folderpath = 'results/evaluation/2d'

# -------------------------------
# 加载结果
# -------------------------------
random_results = {}
for method, filename in zip(methods, result_filenames):
    with open(join(results_folderpath, filename + '.pickle'), 'rb') as f:
        random_results[method] = pickle.load(f)
print(f"✅ 已加载 {len(random_results['rrt'])} 个环境结果。")

# -------------------------------
# 计算指标
# -------------------------------
metrics = []

for method in methods:
    results = random_results[method]
    total_envs = len(results)
    success_envs = []
    first_success_iters = []
    final_path_lengths = []
    convergence_speeds = []

    for env in results:
        arr = np.array(env['result'])
        finite_idx = np.where(np.isfinite(arr))[0]
        if len(finite_idx) > 0:
            first_success = finite_idx[0]
            first_success_iters.append(first_success)
            final_path = arr[-1]
            final_path_lengths.append(final_path)

            if len(arr) > 1000:
                start_val = np.mean(arr[-1000:])
                min_val = np.min(arr)
                conv_speed = (start_val - min_val) / start_val if start_val > 0 else 0
            else:
                conv_speed = 0
            convergence_speeds.append(conv_speed)

            success_envs.append(True)
        else:
            success_envs.append(False)

    metrics.append({
        'Method': method,
        'Success Rate (%)': round(np.mean(success_envs) * 100, 2),
        'Average Path Length': round(np.mean(final_path_lengths), 2),
        'Iteration to First Success': round(np.mean(first_success_iters), 2),
        'Convergence Speed': round(np.mean(convergence_speeds), 4)
    })

metrics_df = pd.DataFrame(metrics)
csv_path = join(visualization_folderpath, 'random_2d_summary_metrics.csv')
metrics_df.to_csv(csv_path, index=False)
print(f"📄 指标保存至: {csv_path}")
print(metrics_df)

# -------------------------------
# 1️⃣ 路径收敛曲线
# -------------------------------
iter_after_initial_list = range(0, 3000 + 250, 250)
random_analysis = {}

for method in methods:
    random_analysis[method] = {}
    for iter_after_initial in iter_after_initial_list:
        random_analysis[method][iter_after_initial] = []
    for i in range(random_dataset_len):
        r = np.array(random_results[method][i]['result'])
        finite_idx = np.where(r < np.inf)[0]
        if len(finite_idx) == 0:
            continue
        initial_idx = finite_idx[0]
        initial_path_cost_rrt = np.array(random_results['rrt'][i]['result'])[finite_idx[0]]
        for iter_after_initial in iter_after_initial_list:
            idx = min(initial_idx + iter_after_initial, len(r) - 1)
            random_analysis[method][iter_after_initial].append(r[idx] / initial_path_cost_rrt)

path_cost_mean = {m: [np.mean(random_analysis[m][k]) for k in iter_after_initial_list] for m in methods}

fig, ax = plt.subplots(figsize=(7,5))
for method, color, label in zip(
    ['rrt', 'irrt', 'nrrt_png', 'nirrt_png', 'nirrt_png_c', 'nrrt_png_c', 'nrrt_gng'],
    ['k', 'gray', 'C0', 'C1', 'C2', 'C4', 'C5'],
    ['RRT*', 'IRRT*', 'NRRT*-PNG',  'NIRRT*-PNG (F)', 'NIRRT*-PNG (FC)', 'NRRT*-PNG (C)', 'NRRT*-GNG']):
    plt.plot(iter_after_initial_list, path_cost_mean[method], c=color, marker='.', linestyle='-', label=label)

plt.legend()
plt.xlabel("Iteration after first success")
plt.ylabel("Normalized Path Length")
plt.title("Path Cost Ratio vs Iteration")
plt.grid(True, linestyle='--', alpha=0.4)
fig.savefig(join(visualization_folderpath,'random_2d_path_cost_ratio_results.png'))
plt.close(fig)
print("📈 路径收敛曲线已保存。")

# -------------------------------
# 2️⃣ 成功率曲线
# -------------------------------
max_iter = max(len(env['result']) for env in random_results['rrt'])
iter_points = np.arange(0, max_iter, 250)
success_rate_curve = {m: [] for m in methods}

for method in methods:
    results = random_results[method]
    total_envs = len(results)
    for it in iter_points:
        success = sum([np.any(np.isfinite(np.array(env['result'][:min(it+1, len(env['result']))]))) for env in results])
        success_rate_curve[method].append(success / total_envs * 100)

fig, ax = plt.subplots(figsize=(7,5))
for method, color, label in zip(
    ['rrt', 'irrt', 'nrrt_png', 'nirrt_png', 'nirrt_png_c', 'nrrt_png_c', 'nrrt_gng'],
    ['k', 'gray', 'C0', 'C1', 'C2', 'C4', 'C5'],
    ['RRT*', 'IRRT*', 'NRRT*-PNG', 'NIRRT*-PNG (F)', 'NIRRT*-PNG (FC)', 'NRRT*-PNG (C)', 'NRRT*-GNG']
):
    plt.plot(iter_points, success_rate_curve[method], color=color, label=label)

plt.xlabel("Iteration")
plt.ylabel("Success Rate (%)")
plt.title("Success Rate vs Iteration (2D Random Worlds)")
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
fig.savefig(join(visualization_folderpath, "random_2d_success_rate_curve.png"))
plt.close(fig)
print("📈 成功率曲线已保存。")

# -------------------------------
# 3️⃣ 首次成功迭代散点图
# -------------------------------
random_analysis = {}
for method in methods:
    random_analysis[method] = []
    for env in random_results[method]:
        arr = np.array(env['result'])
        finite_idx = np.where(np.isfinite(arr))[0]
        if len(finite_idx) > 0:
            random_analysis[method].append(finite_idx[0])
        else:
            random_analysis[method].append(np.inf)

fig, ax = plt.subplots(figsize=(6,6))
range_limit = 2000
ax.scatter(random_analysis['nirrt_png_c'], random_analysis['irrt'], s=8, c='C1', alpha=0.7, label='Environments')
plt.plot([0, range_limit], [0, range_limit], 'k--', lw=1)
plt.xlabel('NIRRT*-PNG(FC): Iteration to First Success')
plt.ylabel('IRRT*: Iteration to First Success')
plt.xlim(0, range_limit)
plt.ylim(0, range_limit)
plt.xticks([0, 500, 1000, 1500, 2000])
plt.yticks([0, 500, 1000, 1500, 2000])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
fig.savefig(join(visualization_folderpath, 'random_2d_iter_scatter_nirrt_irrt.png'))
plt.close(fig)
print("📊 首次成功迭代散点图已保存。")

# -------------------------------
# 输出完成
# -------------------------------
print("\n✅ 全部分析完成！结果文件：")
print(f" - {csv_path}")
print(f" - random_2d_path_cost_ratio_results.png")
print(f" - random_2d_success_rate_curve.png")
print(f" - random_2d_iter_scatter_nirrt_irrt.png")
