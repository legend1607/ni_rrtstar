# path_planning_classes/niarrt_star_png_2d.py
import math
import os
from matplotlib import pyplot as plt
import numpy as np
import time
from path_planning_utils.rrt_env import Env
from path_planning_classes.irrt_star_2d import IRRTStar2D
from path_planning_classes.rrt_base_2d import RRTBase2D
from path_planning_classes.rrt_visualizer_2d import NIARRTStarVisualizer
from datasets.point_cloud_mask_utils import get_point_cloud_mask_around_points, \
    generate_rectangle_point_cloud, ellipsoid_point_cloud_sampling

class NIARRTStarPNG2D(IRRTStar2D):
    def __init__(
        self,
        x_start,
        x_goal,
        step_len,
        search_radius,
        iter_max,
        env_dict,
        png_wrapper,
        binary_mask,
        clearance,
        pc_n_points,
        pc_over_sample_scale,
        pc_sample_rate,
        pc_update_cost_ratio,
    ):
        RRTBase2D.__init__(
            self,
            x_start,
            x_goal,
            step_len,
            search_radius,
            iter_max,
            Env(env_dict),
            clearance,
            "NIARRT*-PNG 2D",
        )
        self.png_wrapper = png_wrapper
        self.binary_mask = binary_mask
        self.pc_n_points = pc_n_points # * number of points in pc
        self.pc_over_sample_scale = pc_over_sample_scale
        self.pc_sample_rate = pc_sample_rate
        self.pc_neighbor_radius = self.step_len
        self.pc_update_cost_ratio = pc_update_cost_ratio
        self.path_solutions = [] # * a list of valid goal parent vertex indices
        self.visualizer = NIARRTStarVisualizer(self.x_start, self.x_goal, self.env)
        # ---------- for adaptive pred-direction trust ----------
        # 记录预测方向最近碰撞的“分数”，越高表示预测常导致碰撞，alpha 会减小
        self.pred_collision_score = 0.0
        self.pred_collision_decay = 0.995  # 每次迭代衰减（接近 1.0 更慢）
        self.pred_base_alpha = 0.7         # 初始信任度（可调）
        self.pred_min_alpha = 0.1          # 最小信任度下限
        self.pred_collision_sensitivity = 10.0  # 敏感度（越小对碰撞更敏感）

    def init(self):
        # 起点与终点应为 np.ndarray (n,)
        c_min, self.theta = self.get_distance_and_angle(self.x_start, self.x_goal)
        C = self.RotationToWorldFrame(self.x_start, self.x_goal, c_min)
        dim = self.x_start.shape[0]

        x_center = np.zeros((dim, 1))
        x_center[:, 0] = (self.x_start + self.x_goal) / 2.0
        return c_min, x_center, C
    
    def SampleInformedSubset(self, c_best, c_min, x_center, C):
        """
        在任意维空间中进行 Informed RRT* 椭球采样。
        
        参数：
            c_best: 当前最优路径长度
            c_min:  起点到终点的最短距离（直线距离）
            x_center: 椭球中心 (n,)
            C: 从世界坐标到椭球坐标的旋转矩阵 (n, n)
        返回：
            node_rand: np.ndarray (n,)
        """
        n = len(x_center)

        # --- 当还没有找到可行解时，直接全局采样 ---
        if np.isinf(c_best):
            return self.SampleFree()

        # --- 数值稳定处理 ---
        diff_sq = c_best**2 - c_min**2
        if diff_sq < 0:
            diff_sq = 1e-9  # 防止 sqrt 负数

        # --- 椭球主轴长度 ---
        r = np.zeros(n)
        r[0] = c_best / 2.0
        for i in range(1, n):
            r[i] = math.sqrt(diff_sq) / 2.0
        L = np.diag(r)  # 对角伸缩矩阵 (n, n)

        # --- 从单位超球内采样 ---
        while True:
            x_ball = self.SampleUnitBall(n)  # (n, 1)
            # 世界坐标变换
            node_rand = C @ L @ x_ball + x_center.reshape(-1, 1)
            node_rand = node_rand.flatten()
            # 碰撞检测（或边界有效性检测）
            if self.utils.is_valid(tuple(node_rand[:2])):  # 默认2D环境判定
                return node_rand
            
    @staticmethod
    def SampleUnitBall(dim=2):
        while True:
            x = np.random.uniform(-1, 1, (dim, 1))
            if np.linalg.norm(x) <= 1:
                return x

    @staticmethod
    def RotationToWorldFrame(x_start, x_goal, L):
        """
        生成局部椭球坐标到全局坐标的旋转矩阵 (任意维)
        - inputs:
            - x_start, x_goal: np.ndarray (n,)
            - L: 起点到终点的距离
        - output:
            - C: np.ndarray (n, n)
        """
        dim = x_start.shape[0]

        # 主轴方向（局部 x 轴）
        a1 = (x_goal - x_start).reshape(dim, 1) / L

        # 局部坐标系中的基向量 e1 = [1, 0, ..., 0]^T
        e1 = np.zeros((dim, 1))
        e1[0, 0] = 1.0

        # 使用 SVD 求解旋转矩阵，使 a1 对齐到 e1
        M = a1 @ e1.T
        U, _, V_T = np.linalg.svd(M, full_matrices=True)

        # 确保右手系旋转矩阵
        det_correction = np.eye(dim)
        det_correction[-1, -1] = np.linalg.det(U @ V_T)
        C = U @ det_correction @ V_T
        return C
    
    def init_pc(self):
        self.update_point_cloud(
            cmax=np.inf,
            cmin=None,
        )

    def planning(self, visualize=False):
        self.start_goal_straightline_dist, x_center, C = self.init()
        self.init_pc()  # 初始化点云
        c_best = np.inf
        c_update = c_best
        cost_curve = []
        start_time = time.time()  # 记录迭代开始时间

        for k in range(self.iter_max):

            if len(self.path_solutions) > 0:
                c_best, x_best = self.find_best_path_solution()
                if k%10==0:
                    self.update_point_cloud(c_best, self.start_goal_straightline_dist)

            node_rand, c_update = self.generate_random_node(c_best, self.start_goal_straightline_dist, x_center, C, c_update)
            node_nearest, node_nearest_index = self.nearest_neighbor(self.vertices[:self.num_vertices], node_rand)
            self.visualizer.set_current_node_nearest(node_nearest)

            pred_dir = None
            used_pred = False
            if self.png_wrapper.use_direction and getattr(self, "path_point_cloud_pred", None) is not None and len(self.path_point_cloud_pred) > 0 and len(self.path_solutions) <= 0:
                # 找与 node_nearest 最近的预测点（这里使用 pred-only 集合）
                nearest_idx = np.argmin(np.linalg.norm(self.path_point_cloud_pred - node_nearest, axis=1))
                pred_dir = None
                if getattr(self, "path_point_cloud_direction", None) is not None and nearest_idx < len(self.path_point_cloud_direction):
                    pred_dir = self.path_point_cloud_direction[nearest_idx]
                    self.visualizer.set_pred_direction(pred_dir)
                if pred_dir is not None:
                    pred_dir = pred_dir / (np.linalg.norm(pred_dir) + 1e-8)

                    # RRT* 基础方向（指向 node_rand）
                    dir_to_rand = node_rand - node_nearest
                    dir_to_rand = dir_to_rand / (np.linalg.norm(dir_to_rand) + 1e-8)

                    # 扇形扰动（使得扩展不完全僵化于 pred_dir）
                    max_angle = np.deg2rad(10.0)  # 扩展时扰动幅度通常比采样时小
                    delta = np.random.uniform(-max_angle, max_angle)
                    cosd, sind = np.cos(delta), np.sin(delta)
                    R = np.array([[cosd, -sind], [sind, cosd]])
                    pred_dir_perturbed = R @ pred_dir
                    pred_dir_perturbed /= (np.linalg.norm(pred_dir_perturbed) + 1e-8)

                    # 动态 alpha：根据最近预测碰撞情况自适应
                    # alpha = base * exp(-score / sensitivity), 且不低于 pred_min_alpha
                    alpha = self.pred_base_alpha * np.exp(- self.pred_collision_score / self.pred_collision_sensitivity)
                    alpha = float(np.clip(alpha, self.pred_min_alpha, 1.0))

                    # 融合方向
                    blended_dir = alpha * pred_dir_perturbed + (1.0 - alpha) * dir_to_rand
                    blended_dir /= (np.linalg.norm(blended_dir) + 1e-8)
                    self.visualizer.set_pred_direction(blended_dir)

                    # 先尝试沿 blended_dir 扩展；若碰撞则 fallback 回 dir_to_rand；若仍然碰撞退化为 new_state
                    candidate_new = node_nearest + self.step_len * blended_dir
                    self.visualizer.set_pred_direction(pred_dir)
                    if not self.utils.is_collision(node_nearest, candidate_new):
                        node_new = candidate_new
                        used_pred = True
                    else:
                        candidate_fallback = node_nearest + self.step_len * dir_to_rand
                        if not self.utils.is_collision(node_nearest, candidate_fallback):
                            node_new = candidate_fallback
                            self.visualizer.set_pred_direction(dir_to_rand)
                        else:
                            node_new = self.new_state(node_nearest, node_rand)
                            self.visualizer.set_pred_direction((node_rand-node_nearest)/ (np.linalg.norm(node_rand-node_nearest) + 1e-8))

                    # 更新 pred 碰撞统计：如果尝试过 blended_dir 且其碰撞，则提高 score
                    if used_pred is False:
                        # blended_dir 未成功（发生碰撞），视作一次预测碰撞事件
                        self.pred_collision_score += 1.0
                else:
                    # 没有可对齐的 pred direction，退化为常规扩展
                    node_new = self.new_state(node_nearest, node_rand)
                    self.visualizer.set_pred_direction((node_rand-node_nearest)/ (np.linalg.norm(node_rand-node_nearest) + 1e-8))
            else:
                node_new = self.new_state(node_nearest, node_rand)
                self.visualizer.set_pred_direction((node_rand-node_nearest)/ (np.linalg.norm(node_rand-node_nearest) + 1e-8))
            self.visualizer.set_current_expansion_new(node_new)
            self.pred_collision_score *= self.pred_collision_decay

            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new - node_nearest) < 1e-8:
                    node_new = node_nearest
                    self.visualizer.set_current_expansion_new(node_new)
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index) + self.Line(node_nearest, node_new)

                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices) > 0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)

                if self.InGoalRegion(node_new):
                    self.path_solutions.append(node_new_index)

            if len(self.path_solutions) > 0:
                c_best, x_best = self.find_best_path_solution()
                self.path = self.extract_path(x_best)
            else:
                self.path = []

            cost_curve.append(c_best)
            end_time = time.time()
            planning_time = end_time - start_time
            if k % 10 == 0:
                # print(f"Iteration {k} finished in {planning_time:.4f} seconds, current best path length: {c_best}")
                if visualize:
                    self.visualize(x_center, c_best, self.start_goal_straightline_dist, self.theta, cost_curve, iter_suffix=k)
                if c_best != np.inf:
                    print(f"Iteration {k} finished in {planning_time:.4f} seconds, current best path length: {c_best}, self.path length: {len(self.path)}")
                # 可视化
                # if visualize:
                #     self.visualize(x_center, c_best, self.start_goal_straightline_dist, self.theta, cost_curve, iter_suffix=k)
        plt.figure()
        plt.plot(range(len(cost_curve)), cost_curve)
        plt.xlabel("Iteration")
        plt.ylabel("Path Cost (c_best)")
        plt.title("Path Cost vs Iterations")
        plt.grid(True)
        planner_name = self.__class__.__name__
        img_dir = os.path.join("visualization", "planning_demo", planner_name)
        plt.savefig(os.path.join(img_dir,"path_cost_curve.png"), dpi=300)
        plt.close()

    def is_in_informed_ellipse(self, point, x_center, C, c_best, c_min):
        """
        判断任意维空间中某点是否在 Informed RRT* 的椭球采样域内。
        """
        if c_best == np.inf:
            return True  # 没有可行解时，视作不限制

        n = len(point)
        if c_best ** 2 - c_min ** 2 < 0:
            eps = 1e-9
        else:
            eps = 0.0

        # 构造 L（半轴长度）
        r = np.zeros(n)
        r[0] = c_best / 2.0
        r[1:] = np.sqrt(c_best ** 2 - c_min ** 2 + eps) / 2.0
        L_inv = np.diag(1.0 / r)

        # 转换到椭圆坐标系
        diff = point - x_center[:n, 0]
        z = L_inv @ C.T @ diff

        # 是否在单位球内
        return np.dot(z, z) <= 1.0

    def generate_random_node(
        self,
        c_curr,
        c_min,
        x_center,
        C,
        c_update,
    ):
        """
        自适应采样策略：
        - 动态调整 path/key 权重；
        - 在已有路径周围与瓶颈区域集中采样；
        - 同时保留 Informed 椭圆采样。
        """
        # --- 动态点云更新 ---
        if c_curr < self.pc_update_cost_ratio * c_update:
            self.update_point_cloud(c_curr, c_min)
            c_update = c_curr

        # --- 动态采样权重调节 ---
        if len(self.path_solutions) == 0:
            p_path, p_key = 0.3, 0.7
        else:
            c_best, _ = self.find_best_path_solution()
            ratio = np.clip(self.start_goal_straightline_dist / (c_best + 1e-8), 0, 1)
            p_pred = 0.4 * (1 - ratio) + 0.2
            p_key = 0.3 * (1 - ratio) + 0.2
            p_path = 1.0 - (p_pred + p_key)

        def sample_around_path(sigma_scale=0.3):
            """在当前最优路径附近采样"""
            if not hasattr(self, "path") or len(self.path) == 0:
                return None
            idx = int(len(self.path) * np.clip(np.random.normal(0.5, 0.15), 0, 1))
            p = self.path[idx]
            sigma_scale = 0.5 * (1 - ratio) + 0.05
            p = self.path[np.random.randint(0, len(self.path))]
            noise = np.random.randn(2) * self.step_len * sigma_scale
            return p + noise

        # --- 阶段 1：未找到可行解 ---
        if c_curr == np.inf:
            r = np.random.rand()
            if (
                r < p_key
                and self.path_point_cloud_pred is not None
                and len(self.path_point_cloud_pred) > 0
                and self.png_wrapper.use_direction
                and getattr(self, "path_point_cloud_direction", None) is not None
            ):
                # 随机取当前树的一个活跃节点（模拟当前扩展方向）
                node_nearest = self.vertices[np.random.randint(self.num_vertices)]
                # 1️⃣ 找到 node_nearest 在预测点云中最近的预测点
                nearest_idx = np.argmin(np.linalg.norm(self.path_point_cloud_dir - node_nearest, axis=1))

                # 2️⃣ 获取该点的预测方向（局部方向）
                dir_local = self.path_point_cloud_direction[nearest_idx]
                dir_local = dir_local / (np.linalg.norm(dir_local) + 1e-8)

                # 从 node_nearest 指向这些点的向量
                vecs = self.path_point_cloud_dir - node_nearest
                # 过滤掉距离为 0 的点
                norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
                dirs_to_pts = vecs / norms
                cos_sim = np.dot(dirs_to_pts, dir_local)
                # 方向一致性阈值：5°
                valid_mask = cos_sim > np.cos(np.deg2rad(5.0))
                valid_pts = self.path_point_cloud_dir[valid_mask]
                valid_dirs = self.path_point_cloud_direction[valid_mask]

                self.visualizer.set_path_point_cloud_pred(valid_pts)
                if len(valid_pts) > 0:
                    idx = np.random.randint(len(valid_pts))
                    base_point = valid_pts[idx]
                    base_dir = valid_dirs[idx]

                    # 扇形扰动：在 pred_dir 附近随机旋转 ±10°
                    max_angle = np.deg2rad(10.0)
                    delta = np.random.uniform(-max_angle, max_angle)
                    cosd, sind = np.cos(delta), np.sin(delta)
                    R = np.array([[cosd, -sind], [sind, cosd]])
                    perturbed_dir = R @ base_dir
                    perturbed_dir /= (np.linalg.norm(perturbed_dir) + 1e-8)

                    # 沿扰动方向做小幅采样（保留探索）
                    offset_scale = np.random.uniform(0.5, 1.5) * self.step_len
                    node_rand_candidate = base_point + offset_scale * perturbed_dir

                    # 如果 candidate 与 node_nearest 直接连线碰撞，则尝试回退策略
                    if not self.utils.is_collision(node_nearest, node_rand_candidate):
                        node_rand = node_rand_candidate
                    else:
                        # 若碰撞，50% 概率退到 keypoint（保探索），否则全局采样
                        if np.random.rand() < 0.5 and self.path_point_cloud_keypoints is not None and len(self.path_point_cloud_keypoints) > 0:
                            node_rand = self.path_point_cloud_keypoints[np.random.randint(len(self.path_point_cloud_keypoints))]
                        else:
                            node_rand = self.SampleFree()
                else:
                    # 无满足方向一致性的预测点 -> 退化为全局采样
                    node_rand = self.SampleFree()

            else:
                # 保留部分随机探索能力
                if np.random.rand() < 0.3:
                    node_rand = self.SampleFree()
                else:
                    if (
                        self.path_point_cloud_keypoints is not None
                        and len(self.path_point_cloud_keypoints) > 0
                    ):
                        node_rand = self.path_point_cloud_keypoints[
                            np.random.randint(len(self.path_point_cloud_keypoints))
                        ]
                    else:
                        node_rand = self.SampleFree()

        # --- 阶段 2：已有可行解 ---
        else:
            r = np.random.rand()
            node_rand = None

            # --- (1) 椭圆域内预测点采样 ---
            if r < p_pred and self.path_point_cloud_pred is not None and len(self.path_point_cloud_pred) > 0:
                candidates = self.path_point_cloud_pred
                valid = [p for p in candidates if self.is_in_informed_ellipse(p, x_center, C, c_curr, c_min)]
                if len(valid) > 0:
                    node_rand = valid[np.random.randint(len(valid))]

            # --- (2) 椭圆域内关键点采样 ---
            if node_rand is None and r < p_pred + p_key and self.path_point_cloud_keypoints is not None and len(self.path_point_cloud_keypoints) > 0:
                candidates = self.path_point_cloud_keypoints
                valid = [p for p in candidates if self.is_in_informed_ellipse(p, x_center, C, c_curr, c_min)]
                if len(valid) > 0:
                    node_rand = valid[np.random.randint(len(valid))]

            # --- (3) 椭圆域内路径附近采样 ---
            if node_rand is None and np.random.rand() < p_path:
                p = sample_around_path()
                if p is not None and self.is_in_informed_ellipse(p, x_center, C, c_curr, c_min):
                    node_rand = p

            # --- (4) 椭圆域内均匀采样（回退策略） ---
            if node_rand is None:
                node_rand = self.SampleInformedSubset(c_curr, c_min, x_center, C)

        self.visualizer.set_current_expansion(node_rand)
        return node_rand, c_update

    def update_point_cloud(
        self,
        cmax,
        cmin,
    ):
        """
        自适应更新点云采样分布：
        - 无解前扩大关键点范围；
        - 有解后逐步缩小路径范围，重点在已有路径与瓶颈区域优化。
        """
        if self.pc_sample_rate == 0:
            self.path_point_cloud_pred = None
            self.visualizer.set_path_point_cloud_pred(self.path_point_cloud_pred)
            return

        # --- 1. 生成基础点云 ---
        pc = generate_rectangle_point_cloud(
            self.binary_mask,
            self.pc_n_points,
            self.pc_over_sample_scale,
        )

        start_mask = get_point_cloud_mask_around_points(pc, self.x_start[np.newaxis, :], self.pc_neighbor_radius)
        goal_mask = get_point_cloud_mask_around_points(pc, self.x_goal[np.newaxis, :], self.pc_neighbor_radius)

        # --- 2. 调用模型预测 ---
        path_score, keypoint_score, direction = self.png_wrapper.classify_path_points(
            pc.astype(np.float32),
            start_mask.astype(np.float32),
            goal_mask.astype(np.float32),
        )

        # --- 3. 动态阈值调节 ---
        if len(self.path_solutions) == 0:
            # 尚无可行解：扩大关键点范围，路径范围较松
            path_thr, key_thr = 0.5, 0.7
        else:
            c_best, _ = self.find_best_path_solution()
            ratio = np.clip(self.start_goal_straightline_dist / (c_best + 1e-8), 0, 1)
            # ratio 越大表示路径越接近最优 → 收缩路径范围，强化关键点
            path_thr = 0.5 + 0.1 * ratio  # 路径筛选变严格
            key_thr  = 0.5 - 0.1 * ratio  # 关键点筛选变宽松

        # --- 4. 阈值筛选 ---
        path_mask = path_score > path_thr
        keypoint_mask = keypoint_score > key_thr
        combined_mask = np.logical_or(path_mask, keypoint_mask)

        # --- 5. 分类 ---
        self.path_point_cloud_pred = pc[path_mask]
        self.path_point_cloud_keypoints = pc[keypoint_mask]
        self.path_point_cloud_dir = pc[combined_mask]
        self.path_point_cloud_other = pc[~combined_mask]

        # --- 6. 保存方向预测（可选） ---
        if self.png_wrapper.use_direction:
            self.path_point_cloud_direction = direction[combined_mask]
            self.visualizer.set_pred_direction(self.path_point_cloud_direction)
        else:
            self.path_point_cloud_direction = None

        # --- 7. 更新可视化 ---
        self.visualizer.set_path_point_cloud_pred(self.path_point_cloud_pred)
        self.visualizer.set_keypoint_cloud_pred(self.path_point_cloud_keypoints)
        self.visualizer.set_path_point_cloud_other(self.path_point_cloud_other)

        # 可选可视化
        fig, ax = plt.subplots()
        ax.scatter(self.x_start[0], self.x_start[1], c='green', s=100, label='Start')
        ax.scatter(self.x_goal[0], self.x_goal[1], c='purple', s=100, label='Goal')

        if self.path_point_cloud_pred is not None:
            ax.scatter(self.path_point_cloud_pred[:, 0], self.path_point_cloud_pred[:, 1], c='r', label='Pred Path')
        if self.path_point_cloud_keypoints is not None:
            ax.scatter(self.path_point_cloud_keypoints[:, 0], self.path_point_cloud_keypoints[:, 1], c='b', label='Keypoints')
        if self.path_point_cloud_other is not None:
            ax.scatter(self.path_point_cloud_other[:, 0], self.path_point_cloud_other[:, 1], c='gray', alpha=0.3, label='Other')
        if self.png_wrapper.use_direction and self.path_point_cloud_direction is not None:
            ax.quiver(
                self.path_point_cloud_dir[::2, 0],
                self.path_point_cloud_dir[::2, 1],
                self.path_point_cloud_direction[::2, 0],
                self.path_point_cloud_direction[::2, 1],
                angles='xy', scale_units='xy', scale=0.1, color='orange', width=0.005, label='Direction'
            )

        ax.set_title(f"Adaptive Point Cloud (path_thr={path_thr:.2f}, key_thr={key_thr:.2f})")
        ax.legend()
        ax.set_aspect('equal')
        plt.close()

    def visualize(self, x_center, c_best, start_goal_straightline_dist, theta, cost_curve, figure_title=None, img_filename=None, iter_suffix=None):
        if figure_title is None:
            figure_title = "niarrt* 2D"
            if iter_suffix is not None:
                figure_title += f", iteration {iter_suffix}"
        if img_filename is None:
            img_filename = f"niarrt_2d_example_{iter_suffix}.png" if iter_suffix is not None else "niarrt_2d_example.png"
        planner_name = self.__class__.__name__
        img_dir = os.path.join("visualization", "planning_demo", planner_name)
        os.makedirs(img_dir, exist_ok=True)
        img_filename = os.path.join(img_dir,img_filename)
        self.visualizer.animation(
            self.vertices[:self.num_vertices],
            self.vertex_parents[:self.num_vertices],
            self.path,
            figure_title,
            x_center,
            c_best,
            start_goal_straightline_dist,
            theta,
            img_filename=img_filename,
        )

    def planning_block_gap(
        self,
        path_len_threshold,
    ):
        path_len_list = []
        theta, start_goal_straightline_dist, x_center, C = self.init()
        self.init_pc() # * niarrt*
        c_best = np.inf
        c_update = c_best # * niarrt*
        better_than_path_len_threshold = False
        for k in range(self.iter_max):
            if len(self.path_solutions)>0:
                c_best, x_best = self.find_best_path_solution()
            path_len_list.append(c_best)
            if k % 1000 == 0:
                print("{0}/{1} - current: {2:.2f}, threshold: {3:.2f}".format(\
                    k, self.iter_max, c_best, path_len_threshold)) #* not k+1, because we are not getting c_best after iteration is done
            if c_best < path_len_threshold:
                better_than_path_len_threshold = True
                break
            node_rand, c_update = self.generate_random_node(c_best, start_goal_straightline_dist, x_center, C, c_update) # * niarrt*
            node_nearest, node_nearest_index = self.nearest_neighbor(self.vertices[:self.num_vertices], node_rand)
            node_new = self.new_state(node_nearest, node_rand)
            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new-node_nearest)<1e-8:
                    # * do not create a new node if it is actually the same point
                    node_new = node_nearest
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index)+self.Line(node_nearest, node_new)
                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices)>0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)
                if self.InGoalRegion(node_new):
                    self.path_solutions.append(node_new_index)
        path_len_list = path_len_list[1:] # * the first one is the initialized c_best before iteration
        if better_than_path_len_threshold:
            return path_len_list
        # * path cost for the last iteration
        if len(self.path_solutions)>0:
            c_best, x_best = self.find_best_path_solution()
        path_len_list.append(c_best)
        print("{0}/{1} - current: {2:.2f}, threshold: {3:.2f}".format(\
            len(path_len_list), self.iter_max, c_best, path_len_threshold)) #* not k+1, because we are not getting c_best after iteration is done
        return path_len_list

    def planning_random(
        self,
        iter_after_initial,
    ):
        path_len_list = []
        self.start_goal_straightline_dist, x_center, C = self.init()
        self.init_pc() # * niarrt*
        c_best = np.inf
        c_update = c_best # * niarrt*
        better_than_inf = False

        for k in range(self.iter_max):
            if len(self.path_solutions)>0:
                c_best, x_best = self.find_best_path_solution()
                if k%10==0:
                    self.update_point_cloud(c_best, self.start_goal_straightline_dist)

            path_len_list.append(c_best)
            if k % 500 == 0:
                if c_best == np.inf:
                    print("{0}/{1} - current: inf".format(k, self.iter_max)) #* not k+1, because we are not getting c_best after iteration is done
            if c_best < np.inf:
                better_than_inf = True
                print("{0}/{1} - current: {2:.2f}".format(k, self.iter_max, c_best))
                break

            node_rand, c_update = self.generate_random_node(c_best, self.start_goal_straightline_dist, x_center, C, c_update)
            node_nearest, node_nearest_index = self.nearest_neighbor(self.vertices[:self.num_vertices], node_rand)
            
            pred_dir = None
            used_pred = False
            if self.png_wrapper.use_direction and getattr(self, "path_point_cloud_pred", None) is not None and len(self.path_point_cloud_pred) > 0 and len(self.path_solutions) <= 0:
                # 找与 node_nearest 最近的预测点（这里使用 pred-only 集合）
                nearest_idx = np.argmin(np.linalg.norm(self.path_point_cloud_pred - node_nearest, axis=1))
                pred_dir = None
                if getattr(self, "path_point_cloud_direction", None) is not None and nearest_idx < len(self.path_point_cloud_direction):
                    pred_dir = self.path_point_cloud_direction[nearest_idx]
                if pred_dir is not None:
                    pred_dir = pred_dir / (np.linalg.norm(pred_dir) + 1e-8)

                    # RRT* 基础方向（指向 node_rand）
                    dir_to_rand = node_rand - node_nearest
                    dir_to_rand = dir_to_rand / (np.linalg.norm(dir_to_rand) + 1e-8)

                    # 扇形扰动（使得扩展不完全僵化于 pred_dir）
                    max_angle = np.deg2rad(10.0)  # 扩展时扰动幅度通常比采样时小
                    delta = np.random.uniform(-max_angle, max_angle)
                    cosd, sind = np.cos(delta), np.sin(delta)
                    R = np.array([[cosd, -sind], [sind, cosd]])
                    pred_dir_perturbed = R @ pred_dir
                    pred_dir_perturbed /= (np.linalg.norm(pred_dir_perturbed) + 1e-8)

                    # 动态 alpha：根据最近预测碰撞情况自适应
                    # alpha = base * exp(-score / sensitivity), 且不低于 pred_min_alpha
                    alpha = self.pred_base_alpha * np.exp(- self.pred_collision_score / self.pred_collision_sensitivity)
                    alpha = float(np.clip(alpha, self.pred_min_alpha, 1.0))

                    # 融合方向
                    blended_dir = alpha * pred_dir_perturbed + (1.0 - alpha) * dir_to_rand
                    blended_dir /= (np.linalg.norm(blended_dir) + 1e-8)

                    # 先尝试沿 blended_dir 扩展；若碰撞则 fallback 回 dir_to_rand；若仍然碰撞退化为 new_state
                    candidate_new = node_nearest + self.step_len * blended_dir
                    if not self.utils.is_collision(node_nearest, candidate_new):
                        node_new = candidate_new
                        used_pred = True
                    else:
                        candidate_fallback = node_nearest + self.step_len * dir_to_rand
                        if not self.utils.is_collision(node_nearest, candidate_fallback):
                            node_new = candidate_fallback
                        else:
                            node_new = self.new_state(node_nearest, node_rand)

                    # 更新 pred 碰撞统计：如果尝试过 blended_dir 且其碰撞，则提高 score
                    if used_pred is False:
                        # blended_dir 未成功（发生碰撞），视作一次预测碰撞事件
                        self.pred_collision_score += 1.0
                else:
                    # 没有可对齐的 pred direction，退化为常规扩展
                    node_new = self.new_state(node_nearest, node_rand)
            else:
                node_new = self.new_state(node_nearest, node_rand)
            self.pred_collision_score *= self.pred_collision_decay
            
            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new-node_nearest)<1e-8:
                    # * do not create a new node if it is actually the same point
                    node_new = node_nearest
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index)+self.Line(node_nearest, node_new)

                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices)>0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)
                if self.InGoalRegion(node_new):
                    self.path_solutions.append(node_new_index)
        path_len_list = path_len_list[1:] # * the first one is the initialized c_best before iteration
        if better_than_inf:
            initial_path_len = path_len_list[-1]
        else:
            # * path cost for the last iteration
            if len(self.path_solutions)>0:
                c_best, x_best = self.find_best_path_solution()
            path_len_list.append(c_best)
            initial_path_len = path_len_list[-1]
            if initial_path_len == np.inf:
                # * fail to find initial path solution
                return path_len_list
        path_len_list = path_len_list[:-1] # * for loop below will add initial_path_len to path_len_list
        # * iteration after finding initial solution
        for k in range(iter_after_initial):
            c_best, x_best = self.find_best_path_solution() # * there must be path solutions
            path_len_list.append(c_best)
            if k % 1000 == 0:
                print("{0}/{1} - current: {2:.2f}, initial: {3:.2f}, cmin: {4:.2f}".format(\
                    k, iter_after_initial, c_best, initial_path_len, self.start_goal_straightline_dist))
            node_rand, c_update = self.generate_random_node(c_best, self.start_goal_straightline_dist, x_center, C, c_update) # * niarrt*
            node_nearest, node_nearest_index = self.nearest_neighbor(self.vertices[:self.num_vertices], node_rand)

            pred_dir = None
            used_pred = False
            if self.png_wrapper.use_direction and getattr(self, "path_point_cloud_pred", None) is not None and len(self.path_point_cloud_pred) > 0 and len(self.path_solutions) <= 0:
                # 找与 node_nearest 最近的预测点（这里使用 pred-only 集合）
                nearest_idx = np.argmin(np.linalg.norm(self.path_point_cloud_pred - node_nearest, axis=1))
                pred_dir = None
                if getattr(self, "path_point_cloud_direction", None) is not None and nearest_idx < len(self.path_point_cloud_direction):
                    pred_dir = self.path_point_cloud_direction[nearest_idx]
                if pred_dir is not None:
                    pred_dir = pred_dir / (np.linalg.norm(pred_dir) + 1e-8)

                    # RRT* 基础方向（指向 node_rand）
                    dir_to_rand = node_rand - node_nearest
                    dir_to_rand = dir_to_rand / (np.linalg.norm(dir_to_rand) + 1e-8)

                    # 扇形扰动（使得扩展不完全僵化于 pred_dir）
                    max_angle = np.deg2rad(10.0)  # 扩展时扰动幅度通常比采样时小
                    delta = np.random.uniform(-max_angle, max_angle)
                    cosd, sind = np.cos(delta), np.sin(delta)
                    R = np.array([[cosd, -sind], [sind, cosd]])
                    pred_dir_perturbed = R @ pred_dir
                    pred_dir_perturbed /= (np.linalg.norm(pred_dir_perturbed) + 1e-8)

                    # 动态 alpha：根据最近预测碰撞情况自适应
                    # alpha = base * exp(-score / sensitivity), 且不低于 pred_min_alpha
                    alpha = self.pred_base_alpha * np.exp(- self.pred_collision_score / self.pred_collision_sensitivity)
                    alpha = float(np.clip(alpha, self.pred_min_alpha, 1.0))

                    # 融合方向
                    blended_dir = alpha * pred_dir_perturbed + (1.0 - alpha) * dir_to_rand
                    blended_dir /= (np.linalg.norm(blended_dir) + 1e-8)

                    # 先尝试沿 blended_dir 扩展；若碰撞则 fallback 回 dir_to_rand；若仍然碰撞退化为 new_state
                    candidate_new = node_nearest + self.step_len * blended_dir
                    if not self.utils.is_collision(node_nearest, candidate_new):
                        node_new = candidate_new
                        used_pred = True
                    else:
                        candidate_fallback = node_nearest + self.step_len * dir_to_rand
                        if not self.utils.is_collision(node_nearest, candidate_fallback):
                            node_new = candidate_fallback
                        else:
                            node_new = self.new_state(node_nearest, node_rand)

                    # 更新 pred 碰撞统计：如果尝试过 blended_dir 且其碰撞，则提高 score
                    if used_pred is False:
                        # blended_dir 未成功（发生碰撞），视作一次预测碰撞事件
                        self.pred_collision_score += 1.0
                else:
                    # 没有可对齐的 pred direction，退化为常规扩展
                    node_new = self.new_state(node_nearest, node_rand)
            else:
                node_new = self.new_state(node_nearest, node_rand)
            self.pred_collision_score *= self.pred_collision_decay
            
            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new-node_nearest)<1e-8:
                    # * do not create a new node if it is actually the same point
                    node_new = node_nearest
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index)+self.Line(node_nearest, node_new)
                    
                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices)>0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)
                if self.InGoalRegion(node_new):
                    self.path_solutions.append(node_new_index)
        # * path cost for the last iteration
        c_best, x_best = self.find_best_path_solution() # * there must be path solutions
        path_len_list.append(c_best)
        print("{0}/{1} - current: {2:.2f}, initial: {3:.2f}".format(\
            iter_after_initial, iter_after_initial, c_best, initial_path_len))
        return path_len_list

def get_path_planner(
    args,
    problem,
    neural_wrapper,
):
    return NIARRTStarPNG2D(
        problem['x_start'],
        problem['x_goal'],
        args.step_len,
        problem['search_radius'],
        args.iter_max,
        problem['env_dict'],
        neural_wrapper,
        problem['binary_mask'],
        args.clearance,
        args.pc_n_points,
        args.pc_over_sample_scale,
        args.pc_sample_rate,
        args.pc_update_cost_ratio,
    )


    