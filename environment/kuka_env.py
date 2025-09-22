import os
import numpy as np
import pybullet as p
import pybullet_data
from time import sleep, time
from environment.timer import Timer

class KukaEnv:
    """
    KUKA 环境类，支持：
    - 随机障碍物生成
    - start-goal 状态采样
    - 机械臂路径可视化
    """

    EPS = 0.05  # 插值步长
    RRT_EPS = 0.5

    def __init__(self, GUI=False, kuka_file="kuka_iiwa/model.urdf"):
        self.GUI = GUI
        self.kuka_file = kuka_file
        self.dim = 3
        self.obstacles = []
        self.init_state = None
        self.goal_state = None
        self.collision_check_count = 0
        self.timer = Timer()
        self.collision_time = 0

        # 连接 PyBullet
        if GUI:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            target = p.getDebugVisualizerCamera()[11]
            p.resetDebugVisualizerCamera(cameraDistance=1.2,
                                         cameraYaw=90,
                                         cameraPitch=-25,
                                         cameraTargetPosition=[target[0], target[1], 0.7])
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # 加载平面和机械臂
        self.planeId = p.loadURDF("plane.urdf")
        self.kukaId = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)

        # 配置空间
        self.config_dim = p.getNumJoints(self.kukaId)
        self.pose_range = [(p.getJointInfo(self.kukaId, j)[8], p.getJointInfo(self.kukaId, j)[9])
                           for j in range(self.config_dim)]
        self.bound = np.array(self.pose_range).T.reshape(-1)
        self.kukaEndEffectorIndex = self.config_dim - 1
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,     # 摄像机到目标点的距离
            cameraYaw=30,           # 绕Z轴旋转角度
            cameraPitch=-40,        # 上下旋转角度
            cameraTargetPosition=[0, 0, 0.5]  # 目标点位置
        )
        # p.stepSimulation()

    # ----------------- 障碍物 -----------------
    def add_box_obstacle(self, half_extents, base_position):
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                     rgbaColor=np.random.uniform(0, 1, 3).tolist() + [0.8])
        body_id = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=col_id,
                                    baseVisualShapeIndex=vis_id,
                                    basePosition=base_position)
        self.obstacles.append((half_extents, base_position))
        return body_id

    def reset_obstacles(self):
        p.resetSimulation()
        self.obstacles = []
        self.planeId = p.loadURDF("plane.urdf")
    def reset_arm(self):
        self.kukaId = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)

    # ----------------- 状态与采样 -----------------
    def set_config(self, state, kukaId=None):
        if kukaId is None:
            kukaId = self.kukaId
        for j in range(self.config_dim):
            p.resetJointState(kukaId, j, state[j])
        p.performCollisionDetection()

    def valid_state(self, state):
        return (state >= np.array(self.pose_range)[:, 0]).all() and (state <= np.array(self.pose_range)[:, 1]).all()

    def is_state_free(self, state):
        if not self.valid_state(state):
            return False
        self.set_config(state)
        self.collision_check_count += 1  # 每次调用计一次碰撞检测
        return len(p.getContactPoints(self.kukaId)) == 0

    def _point_in_free_space(self, state):
        t0 = time()
        if not self.valid_state(state):
            return False

        for i in range(p.getNumJoints(self.kukaId)):
            p.resetJointState(self.kukaId, i, state[i])
        p.performCollisionDetection()
        if len(p.getContactPoints(self.kukaId)) == 0:
            self.collision_check_count += 1
            self.collision_time += time() - t0
            return True
        else:
            self.collision_point = state
            self.collision_check_count += 1
            self.collision_time += time() - t0
            return False

    def is_edge_free(self, state1, state2):
        assert state1.size == state2.size
        if not self.valid_state(state1) or not self.valid_state(state2):
            return False
        if not self.is_state_free(state1) or not self.is_state_free(state2):
            return False
        disp = state2 - state1
        d = self.distance(state1, state2)
        K = max(int(d / self.EPS), 1)
        for k in range(K + 1):
            interp_state = state1 + k / K * disp
            if not self.is_state_free(interp_state):
                return False
        return True

    def _state_fp(self, state):
        self.timer.start()
        free = self._point_in_free_space(state)
        self.timer.finish(Timer.VERTEX_CHECK)
        return free
    
    def _edge_fp(self, state, new_state):
        self.timer.start()
        self.k = 0
        assert state.size == new_state.size
        # 检查起点和终点有效性
        if not self.valid_state(state) or not self.valid_state(new_state):
            self.timer.finish(Timer.EDGE_CHECK)
            return False
        if not self._point_in_free_space(state) or not self._point_in_free_space(new_state):
            self.timer.finish(Timer.EDGE_CHECK)
            return False
        # 直线插值
        disp = new_state - state

        d = self.distance(state, new_state)
        K = int(d / self.RRT_EPS)
        # 遍历中间点
        for k in range(0, K):
            c = state + k * 1. / K * disp
            if not self._point_in_free_space(c):
                self.timer.finish(Timer.EDGE_CHECK)
                return False
        self.timer.finish(Timer.EDGE_CHECK)
        return True
    
    def uniform_sample(self, n=1):
        sample = np.random.uniform(np.array(self.pose_range)[:, 0],
                                   np.array(self.pose_range)[:, 1],
                                   size=(n, self.config_dim))
        if n == 1:
            return sample.reshape(-1)
        return sample

    def sample_start_goal(self, max_attempts=100):
        for _ in range(max_attempts):
            start = self.uniform_sample()
            goal = self.uniform_sample()
            if self.is_state_free(start) and self.is_state_free(goal) and np.linalg.norm(start - goal) > 0.1:
                self.init_state = start
                self.goal_state = goal
                return start, goal
        return None, None

    # ----------------- 工具函数 -----------------
    def distance(self, from_state, to_state):
        diff = to_state - from_state
        return np.sqrt(np.sum(diff ** 2))

    def interpolate(self, from_state, to_state, ratio):
        interp_state = from_state + ratio * (to_state - from_state)
        interp_state = np.clip(interp_state, np.array(self.pose_range)[:, 0], np.array(self.pose_range)[:, 1])
        return interp_state

    def in_goal_region(self, state, eps=None):
        if eps is None:
            eps = self.EPS
        return self.distance(state, self.goal_state) < eps and self.is_state_free(state)

    # ----------------- 可视化 -----------------
    def get_robot_points(self, config, end_point=True):
        points = []
        for i in range(self.config_dim):
            p.resetJointState(self.kukaId, i, config[i])
        if end_point:
            point = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)[0]
            return point
        for effector in range(self.kukaEndEffectorIndex + 1):
            point = p.getLinkState(self.kukaId, effector)[0]
            points.append(point)
        return points

    def plot(self, path, make_gif=False):
        path = np.array(path)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        for halfExtents, basePosition in self.obstacles:
            self.add_box_obstacle(halfExtents, basePosition)
        self.kukaId = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
        self.set_config(path[0])
        target_kukaId = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
        self.set_config(path[-1], target_kukaId)
        prev_pos = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)[0]
        final_pos = p.getLinkState(target_kukaId, self.kukaEndEffectorIndex)[0]
        gifs = []
        for idx in range(len(path) - 1):
            disp = path[idx + 1] - path[idx]
            d = self.distance(path[idx], path[idx + 1])
            K = max(int(d / self.EPS), 1)
            new_kuka = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
            for data in p.getVisualShapeData(new_kuka):
                color = list(data[-1])
                color[-1] = 0.5
                p.changeVisualShape(new_kuka, data[1], rgbaColor=color)
            for k in range(K + 1):
                c = path[idx] + k / K * disp
                self.set_config(c, new_kuka)
                new_pos = p.getLinkState(new_kuka, self.kukaEndEffectorIndex)[0]
                p.addUserDebugLine(prev_pos, new_pos, [1, 0, 0], 5, 0)
                prev_pos = new_pos
                if make_gif:
                    gifs.append(p.getCameraImage(width=1080, height=900)[2])
        return gifs

    def close(self):
        try:
            # remove added bodies if you store their ids
            for bid in getattr(self, 'created_body_ids', []):
                try: p.removeBody(bid)
                except Exception: pass
            try:
                p.removeAllUserDebugItems()
            except Exception:
                pass
            # finally disconnect if connected
            try:
                p.disconnect()
            except Exception:
                pass
        except Exception as e:
            print("[KukaEnv.close] exception:", e)