import os
import json
import numpy as np
import cv2
from .env_config import RRT_EPS, LIMITS


class Random2DEnv:
    """
    随机2D环境类
    用于加载 generate_dataset() 生成的数据集。
    """

    def __init__(self, data_root, mode="train"):
        """
        Args:
            data_root (str): 数据集根目录，如 'data/random_2d_bitstar'
            mode (str): 'train' / 'val' / 'test'
        """
        self.mode = mode
        self.data_dir = os.path.join(data_root, mode)
        self.img_dir = os.path.join(self.data_dir, "env_imgs")

        env_json_path = os.path.join(self.data_dir, "envs.json")
        if not os.path.exists(env_json_path):
            raise FileNotFoundError(f"Env JSON not found: {env_json_path}")

        with open(env_json_path, "r") as f:
            self.envs = json.load(f)

        self.size = len(self.envs)
        self.episode_i = 0
        self.collision_check_count = 0
        self.bound = (-1, -1, 1, 1)
        self.map = None
        self.width = None

        print(f"Loaded {self.size} environments from {self.data_dir}")

    def __str__(self):
        return f"Random2DEnv({self.mode})"

    # ======================================================
    # 核心接口
    # ======================================================

    def init_new_problem(self, index=None):
        """加载一个新的地图及 start/goal"""
        if index is None:
            index = self.episode_i % self.size
        self.episode_i += 1

        env_info = self.envs[index]
        img_path = os.path.join(self.img_dir, f"{index}.png")
        env_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if env_img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # 二值化地图（1=可行走，0=障碍）
        self.map = (env_img > 128).astype(int)
        self.width = self.map.shape[0]
        self.env_info = env_info
        self.bound = [(0, 0), (self.map.shape[1], self.map.shape[0])]

        # 选择随机一个 start-goal 对
        pair_id = np.random.randint(len(env_info["start"]))
        self.init_state = np.array(env_info["start"][pair_id])
        self.goal_state = np.array(env_info["goal"][pair_id])

        self.collision_check_count = 0
        return self.get_problem()

    def get_problem(self):
        return {
            "map": self.map,
            "init_state": self.init_state,
            "goal_state": self.goal_state
        }

    def uniform_sample(self):
        """随机采样一个点"""
        h, w = self.map.shape
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        return np.array([x, y], dtype=float)

    def distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def interpolate(self, a, b, ratio):
        a, b = np.array(a), np.array(b)
        return a + ratio * (b - a)

    def in_goal_region(self, state):
        return self.distance(state, self.goal_state) < RRT_EPS and self._point_in_free_space(state)

    def step(self, state, action=None, new_state=None, check_collision=True):
        if action is not None:
            new_state = state + action

        new_state = np.clip(new_state, 0, self.map.shape[0]-1)
        if not check_collision:
            return new_state, action, True, self.in_goal_region(new_state)

        no_collision = self._edge_fp(state, new_state)
        done = no_collision and self.in_goal_region(new_state)
        return new_state, action, no_collision, done

    # ======================================================
    # 内部函数（碰撞检测）
    # ======================================================

    def _point_in_free_space(self, state):
        x, y = int(state[0]), int(state[1])
        if x < 0 or y < 0 or x >= self.map.shape[1] or y >= self.map.shape[0]:
            return False
        self.collision_check_count += 1
        return self.map[y, x] == 1

    def _edge_fp(self, a, b):
        """检查线段 (a,b) 是否碰撞"""
        num = int(self.distance(a, b)) * 2 + 1
        xs = np.linspace(a[0], b[0], num).astype(int)
        ys = np.linspace(a[1], b[1], num).astype(int)
        for x, y in zip(xs, ys):
            if not self._point_in_free_space((x, y)):
                return False
        return True

    def sample_empty_points(self):
        """随机取一个可行点"""
        while True:
            p = self.uniform_sample()
            if self._point_in_free_space(p):
                return p

    def set_random_init_goal(self):
        self.init_state = self.sample_empty_points()
        self.goal_state = self.sample_empty_points()
        while np.linalg.norm(self.init_state - self.goal_state) < 10:
            self.goal_state = self.sample_empty_points()
        return self.get_problem()
