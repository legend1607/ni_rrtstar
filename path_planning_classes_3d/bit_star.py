import numpy as np
import os
import sys
# 将项目根目录加入 Python 搜索路径
# __file__ 是当前脚本路径，os.path.dirname(__file__) 是 algorithm/ 目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import math
import yaml
import heapq
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from shapely.geometry import Point, LineString, Polygon
from descartes import PolygonPatch
from shapely import affinity
import itertools
from time import time
from environment.timer import Timer
import torch
import random
INF = float("inf")


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

class BITStar:
    def __init__(self, environment, maxIter=5, plot_flag=False, batch_size=200, T=10000, sampling=None, timer=None):
        if timer is None:
            self.timer = Timer()
        else:
            self.timer = timer

        self.env = environment

        start, goal, bounds = tuple(environment.init_state), tuple(environment.goal_state), environment.bound

        self.start = start
        self.goal = goal

        self.bounds = bounds
        self.bounds = np.array(self.bounds).reshape((2, -1)).T
        self.ranges = self.bounds[:, 1] - self.bounds[:, 0]
        self.dimension = environment.config_dim

        # This is the tree
        self.vertices = []
        self.edges = dict()  # key = point，value = parent
        self.g_scores = dict()

        self.samples = []
        self.vertex_queue = []
        self.edge_queue = []
        self.old_vertices = set()

        self.maxIter = maxIter
        self.r = INF
        self.batch_size = batch_size
        self.T, self.T_max = 0, T
        self.eta = 1.1  # tunable parameter
        self.obj_radius = 1
        self.resolution = 3

        # the parameters for informed sampling
        self.c_min = self.distance(self.start, self.goal)
        self.center_point = None
        self.C = None

        # whether plot the middle planning process
        self.plot_planning_process = plot_flag

        if sampling is None:
            self.sampling = self.informed_sample
        else:
            self.sampling = sampling

        self.n_collision_points = 0
        self.n_free_points = 2

    def setup_planning(self):
        # add goal to the samples
        self.samples.append(self.goal)
        self.g_scores[self.goal] = INF

        # add start to the tree
        self.vertices.append(self.start)
        self.g_scores[self.start] = 0

        # Computing the sampling space
        self.informed_sample_init()
        radius_constant = self.radius_init()

        return radius_constant

    def radius_init(self):
        from scipy import special
        # Hypersphere radius calculation
        n = self.dimension
        unit_ball_volume = np.pi ** (n / 2.0) / special.gamma(n / 2.0 + 1)
        volume = np.abs(np.prod(self.ranges)) * self.n_free_points / (self.n_collision_points + self.n_free_points)
        gamma = (1.0 + 1.0 / n) * volume / unit_ball_volume
        radius_constant = 2 * self.eta * (gamma ** (1.0 / n))
        return radius_constant

    def informed_sample_init(self):
        """
        初始化椭球采样所需的旋转矩阵 C 和中心点。
        加入鲁棒性检查：如果 c_min 非法（0 或 nan），禁用椭球采样。
        """
        # 防护：如果 start 与 goal 相同或 c_min 非正，标记为不可用
        try:
            if not np.isfinite(self.c_min) or self.c_min <= 1e-12:
                # disable informed sampling (will fallback to uniform)
                self.center_point = None
                self.C = None
                return
        except Exception:
            self.center_point = None
            self.C = None
            return

        self.center_point = np.array([(self.start[i] + self.goal[i]) / 2.0 for i in range(self.dimension)])
        # unit vector along the major axis
        a_1 = (np.array(self.goal) - np.array(self.start)) / self.c_min
        # build matrix for SVD
        id1_t = np.zeros(self.dimension)
        id1_t[-1] = 1.0
        # Construct M as outer product (a_1) * (e1^T) but ensuring shapes are correct
        M = np.zeros((self.dimension, self.dimension))
        M[:, -1] = a_1  # put a_1 as last column
        try:
            U, S, Vh = np.linalg.svd(M)
            # construct rotation matrix C
            # ensure determinant sign parity as in original implementation
            detU = np.linalg.det(U)
            detV = np.linalg.det(Vh.T)
            S_diag = np.ones(self.dimension)
            S_diag[-1] = detU * detV
            self.C = U @ np.diag(S_diag) @ Vh
        except Exception:
            # SVD failed for degenerate cases; disable informed sampling
            self.center_point = None
            self.C = None
            return

    def sample_unit_ball(self):
        u = np.random.normal(0, 1, self.dimension)  # an array of d normally distributed random variables
        norm = np.sum(u ** 2) ** (0.5)
        r = np.random.random() ** (1.0 / self.dimension)
        x = r * u / norm
        return x

    def informed_sample(self, c_best, sample_num, vertices):
        """
        Informed sampling with safety:
        - if c_best is infinite / not found, fallback to uniform sampling
        - if c_best <= c_min (or numeric noise), fallback to uniform
        - ensure sqrt argument non-negative using max(0.0, ...)
        """
        sample_array = []
        cur_num = 0

        # safety epsilon
        eps = 1e-12

        use_informed = (np.isfinite(c_best) and self.C is not None and c_best > (self.c_min + eps))

        # Precompute L if informed
        if use_informed:
            val = c_best ** 2 - self.c_min ** 2
            # numeric protection
            val = max(0.0, val)
            c_b = math.sqrt(val) / 2.0
            r = [c_best / 2.0] + [c_b] * (self.dimension - 1)
            L = np.diag(r)
        else:
            # if informed sampling not usable, do uniform sampling across bounds
            pass

        # Sample loop
        while cur_num < sample_num:
            if use_informed:
                # sample inside unit ball then transform
                x_ball = self.sample_unit_ball()
                random_point = tuple(np.dot(np.dot(self.C, L), x_ball) + self.center_point)
            else:
                # uniform random in bounds
                random_point = self.get_random_point()

            # accept only free points
            if self.is_point_free(random_point):
                sample_array.append(random_point)
                cur_num += 1

        return sample_array

    def get_random_point(self):
        point = self.bounds[:, 0] + np.random.random(self.dimension) * self.ranges
        return tuple(point)

    def is_point_free(self, point):
        result = self.env._state_fp(np.array(point))
        if result:
            self.n_free_points += 1
        else:
            self.n_collision_points += 1
        return result

    def is_edge_free(self, edge):
        result = self.env._edge_fp(np.array(edge[0]), np.array(edge[1]))
        # self.T += self.env.k
        return result

    def get_g_score(self, point):
        # gT(x)
        if point == self.start:
            return 0
        if point not in self.edges:
            return INF
        else:
            return self.g_scores.get(point)

    def get_f_score(self, point):
        # f^(x)
        return self.heuristic_cost(self.start, point) + self.heuristic_cost(point, self.goal)

    def actual_edge_cost(self, point1, point2):
        # c(x1,x2)
        if not self.is_edge_free([point1, point2]):
            return INF
        return self.distance(point1, point2)

    def heuristic_cost(self, point1, point2):
        # Euler distance as the heuristic distance
        return self.distance(point1, point2)

    def distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def get_edge_value(self, edge):
        # sort value for edge
        return self.get_g_score(edge[0]) + self.heuristic_cost(edge[0], edge[1]) + self.heuristic_cost(edge[1],
                                                                                                       self.goal)

    def get_point_value(self, point):
        # sort value for point
        return self.get_g_score(point) + self.heuristic_cost(point, self.goal)

    def bestVertexQueueValue(self):
        if not self.vertex_queue:
            return INF
        else:
            return self.vertex_queue[0][0]

    def bestEdgeQueueValue(self):
        if not self.edge_queue:
            return INF
        else:
            return self.edge_queue[0][0]

    def prune_edge(self, c_best):
        edge_array = list(self.edges.items())
        for point, parent in edge_array:
            if self.get_f_score(point) > c_best or self.get_f_score(parent) > c_best:
                self.edges.pop(point)

    def prune(self, c_best):
        self.samples = [point for point in self.samples if self.get_f_score(point) < c_best]
        self.prune_edge(c_best)
        vertices_temp = []
        for point in self.vertices:
            if self.get_f_score(point) <= c_best:
                if self.get_g_score(point) == INF:
                    self.samples.append(point)
                else:
                    vertices_temp.append(point)
        self.vertices = vertices_temp

    def expand_vertex(self, point):
        self.timer.start()

        # get the nearest value in vertex for every one in samples where difference is less than the radius
        neigbors_sample = []
        for sample in self.samples:
            if self.distance(point, sample) <= self.r:
                neigbors_sample.append(sample)

        self.timer.finish(Timer.NN)

        self.timer.start()

        # add an edge to the edge queue is the path might improve the solution
        for neighbor in neigbors_sample:
            estimated_f_score = self.heuristic_cost(self.start, point) + \
                                self.heuristic_cost(point, neighbor) + self.heuristic_cost(neighbor, self.goal)
            if estimated_f_score < self.g_scores[self.goal]:
                heapq.heappush(self.edge_queue, (self.get_edge_value((point, neighbor)), (point, neighbor)))

        # add the vertex to the edge queue
        if point not in self.old_vertices:
            neigbors_vertex = []
            for ver in self.vertices:
                if self.distance(point, ver) <= self.r:
                    neigbors_vertex.append(ver)
            for neighbor in neigbors_vertex:
                if neighbor not in self.edges or point != self.edges.get(neighbor):
                    estimated_f_score = self.heuristic_cost(self.start, point) + \
                                        self.heuristic_cost(point, neighbor) + self.heuristic_cost(neighbor, self.goal)
                    if estimated_f_score < self.g_scores[self.goal]:
                        estimated_g_score = self.get_g_score(point) + self.heuristic_cost(point, neighbor)
                        if estimated_g_score < self.get_g_score(neighbor):
                            heapq.heappush(self.edge_queue, (self.get_edge_value((point, neighbor)), (point, neighbor)))

        self.timer.finish(Timer.EXPAND)

    def get_best_path(self):
        path = []
        if self.g_scores[self.goal] != INF:
            path.append(self.goal)
            point = self.goal
            while point != self.start:
                point = self.edges[point]
                path.append(point)
            path.reverse()
        return path

    def path_length_calculate(self, path):
        path_length = 0
        for i in range(len(path) - 1):
            path_length += self.distance(path[i], path[i + 1])
        return path_length

    def plan(self, pathLengthLimit=np.inf, refine_time_budget=20, time_budget=INF):
        collision_checks = self.env.collision_check_count

        self.setup_planning()
        init_time = time()

        while self.T < self.T_max and (time() - init_time < time_budget):
            if not self.vertex_queue and not self.edge_queue:
                c_best = self.g_scores[self.goal]
                self.prune(c_best)
                if math.isinf(c_best):
                    # 没有找到路径 → uniform 采样
                    new_samples = [self.get_random_point() for _ in range(self.batch_size)]
                else:
                    # 已有路径 → 尝试 informed 采样（内部有保护，必要时回退 uniform）
                    new_samples = self.sampling(c_best, self.batch_size, self.vertices)
                self.samples.extend(self.sampling(c_best, self.batch_size, self.vertices))
                self.T += self.batch_size

                self.timer.start()
                self.old_vertices = set(self.vertices)
                self.vertex_queue = [(self.get_point_value(point), point) for point in self.vertices]
                heapq.heapify(self.vertex_queue)  # change to op priority queue
                q = len(self.vertices) + len(self.samples)
                self.r = self.radius_init() * ((math.log(q) / q) ** (1.0 / self.dimension))
                self.timer.finish(Timer.HEAP)

            try:
                while self.bestVertexQueueValue() <= self.bestEdgeQueueValue():
                    self.timer.start()
                    _, point = heapq.heappop(self.vertex_queue)
                    self.timer.finish(Timer.HEAP)
                    self.expand_vertex(point)
            except Exception as e:
                if (not self.edge_queue) and (not self.vertex_queue):
                    continue
                else:
                    raise e

            best_edge_value, bestEdge = heapq.heappop(self.edge_queue)

            # Check if this can improve the current solution
            if best_edge_value < self.g_scores[self.goal]:
                actual_cost_of_edge = self.actual_edge_cost(bestEdge[0], bestEdge[1])
                self.timer.start()
                actual_f_edge = self.heuristic_cost(self.start, bestEdge[0]) + actual_cost_of_edge + self.heuristic_cost(bestEdge[1], self.goal)
                if actual_f_edge < self.g_scores[self.goal]:
                    actual_g_score_of_point = self.get_g_score(bestEdge[0]) + actual_cost_of_edge
                    if actual_g_score_of_point < self.get_g_score(bestEdge[1]):
                        self.g_scores[bestEdge[1]] = actual_g_score_of_point
                        self.edges[bestEdge[1]] = bestEdge[0]
                        if bestEdge[1] not in self.vertices:
                            self.samples.remove(bestEdge[1])
                            self.vertices.append(bestEdge[1])
                            heapq.heappush(self.vertex_queue, (self.get_point_value(bestEdge[1]), bestEdge[1]))

                        self.edge_queue = [item for item in self.edge_queue if item[1][1] != bestEdge[1] or \
                                           self.get_g_score(item[1][0]) + self.heuristic_cost(item[1][0], item[1][
                            1]) < self.get_g_score(item[1][0])]
                        heapq.heapify(
                            self.edge_queue)  # Rebuild the priority queue because it will be destroyed after the element is removed

                self.timer.finish(Timer.HEAP)

            else:
                self.vertex_queue = []
                self.edge_queue = []
            if self.g_scores[self.goal] < pathLengthLimit and (time() - init_time > refine_time_budget):
                # print(f"Refining path: {self.g_scores[self.goal]} < {pathLengthLimit} and {time() - init_time} > {refine_time_budget}")
                break
        # print(self.T, self.g_scores[self.goal], time() - init_time)
        return self.samples, self.edges, self.env.collision_check_count - collision_checks, \
               self.g_scores[self.goal], self.T, time() - init_time

