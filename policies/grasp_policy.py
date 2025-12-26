"""
自动抓取策略

功能：实现完整的抓取控制流程
"""
import numpy as np
import sys
import os
import logging
from scipy.spatial.transform import Rotation, Slerp

# 导入 Policy 基类
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from policies.policies import Policy

from robot_controller.ik_solver import IKSolver
from grasp_client import AnyGraspClient
from anygrasp.grasp_converter import GraspConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraspPolicy(Policy):
    def __init__(self, 
                 camera_to_base_transform,
                 ik_solver=None,
                 anygrasp_host='localhost',
                 anygrasp_port=50000,
                 anygrasp_authkey=b'anygrasp'):
        """
        初始化自动抓取策略

        Args:
            camera_to_base_transform (np.ndarray): 相机到机器人基坐标系的 4x4 变换矩阵
            ik_solver (IKSolver, optional): 逆运动学求解器实例
            anygrasp_host (str): AnyGrasp RPC 服务器地址
            anygrasp_port (int): AnyGrasp RPC 服务器端口
            anygrasp_authkey (bytes): RPC 认证密钥
        """

        # 连接 AnyGrasp RPC 服务器
        logger.info(f"[GraspPolicy] 正在连接到 AnyGrasp 服务器: {anygrasp_host}:{anygrasp_port}")
        self.anygrasp = AnyGraspClient(
            host=anygrasp_host,
            port=anygrasp_port,
            authkey=anygrasp_authkey
        )
        
        # 初始化坐标系转换器
        self.converter = GraspConverter(
            camera_intrinsics=None,
            camera_to_base_transform=camera_to_base_transform
        )
        
        # 初始化IK求解器对象
        self.ik_solver = ik_solver
        
        logger.info("[GraspPolicy] 初始化完成")
        
        # 状态变量
        self.state = 'WAITING'  # 当前状态
        self.action_sequence = []  # 动作序列
        self.action_index = 0  # 当前执行到的动作索引
        self.selected_grasp = None  # 选中的抓取
            
    def reset(self):
        """重置策略状态"""
        self.state = 'WAITING'
        self.action_sequence = []
        self.action_index = 0
        self.selected_grasp = None
        
        logger.info("GraspPolicy 已重置")
    
    def step(self, obs):
        """
        执行一步自动抓取

        Args:
            obs (dict): 环境观测字典

        Returns:
            dict or str: 动作字典或控制命令
        """

        # 状态机逻辑
        if self.state == 'WAITING':
            return self._state_waiting(obs)
        elif self.state == 'DETECTING':
            return self._state_detecting(obs)
        elif self.state == 'EXECUTING':
            return self._state_executing(obs)
        elif self.state == 'COMPLETED':
            return 'end_episode' # 返回结束信号
    
    def _state_waiting(self, obs):
        """等待状态 - 确认开始"""

        logger.info("开始抓取检测")
        self.state = 'DETECTING'
        return None
    
    def _state_detecting(self, obs):
        """检测状态 - 执行抓取检测"""
        
        # 调用 AnyGrasp RPC 服务器进行检测
        logger.info("[GraspPolicy] 请求 AnyGrasp 服务器进行抓取检测...")
        grasp_list = self.anygrasp.detect_grasps()
        
        if len(grasp_list) == 0:
            logger.warning("[GraspPolicy] AnyGrasp 未检测到任何抓取")
            self.state = 'COMPLETED'
            return 'end_episode'
        
        logger.info(f"[GraspPolicy] 收到 {len(grasp_list)} 个抓取候选")
        
        # 遍历候选抓取，寻找可达的抓取
        selected_grasp = self._select_reachable_grasp(grasp_list)
        
        if selected_grasp is None:
            logger.warning("没有找到可达的抓取点")
            self.state = 'COMPLETED'
            return 'end_episode'
        
        # 生成动作序列
        self.selected_grasp = selected_grasp
        self.action_sequence = self._generate_action_sequence(selected_grasp, obs)
        self.action_index = 0
        self.state = 'EXECUTING'
        
        logger.info(f"已选择抓取，分数: {selected_grasp['score']:.3f}")
        
        # 返回第一个动作
        return self._state_executing(obs)
    
    def _select_reachable_grasp(self, grasp_list):
        """
        遍历抓取候选并筛选

        Args:
            grasp_list (list): 抓取候选列表，每个元素为抓取字典

        Returns:
            dict or None: 选中的抓取字典（包含可达末端位姿），若无可达抓取则返回 None
        """

        # 遍历抓取候选
        for i, grasp in enumerate(grasp_list):
            logger.debug(f"尝试第 {i+1} 个抓取，分数: {grasp['score']:.3f}")
            
            # 转换为末端位姿
            position, quaternion = self.converter.grasp_to_ee_pose(grasp)
            
            # 验证可达性
            reachable = self.converter.verify_reachability(position, quaternion, self.ik_solver)
            
            if reachable:
                logger.info(f"找到可达抓取: 第 {i+1} 个")
                # 将转换后的位姿保存到抓取字典中
                grasp['ee_position'] = position
                grasp['ee_quaternion'] = quaternion
                return grasp
        
        return None
    
    def _interpolate_trajectory(self, start_pos, start_quat, end_pos, end_quat, num_steps=10):
        """
        在起点和终点之间进行轨迹插值
        
        Args:
            start_pos (np.ndarray): 起始位置 (3,)
            start_quat (np.ndarray): 起始四元数 (4,) [x, y, z, w]
            end_pos (np.ndarray): 目标位置 (3,)
            end_quat (np.ndarray): 目标四元数 (4,) [x, y, z, w]
            num_steps (int): 插值步数，默认 10
            
        Returns:
            list: 插值后的位姿列表 [(pos, quat), ...]
        """
        trajectory = []
        
        # 位置线性插值
        for i in range(num_steps):
            alpha = (i + 1) / num_steps  # 0 到 1 的插值系数
            interp_pos = (1 - alpha) * start_pos + alpha * end_pos
            
            # 四元数球面线性插值 (SLERP)
            # 创建旋转对象
            key_rots = Rotation.from_quat([start_quat, end_quat])
            slerp = Slerp([0, 1], key_rots)
            interp_rot = slerp(alpha)
            interp_quat = interp_rot.as_quat()
            
            trajectory.append((interp_pos, interp_quat))
        
        logger.debug(f"[GraspPolicy] 生成了 {len(trajectory)} 个插值路径点")
        return trajectory
    
    def _generate_action_sequence(self, grasp, obs):
        """
        生成完整的抓取动作序列

        Args:
            grasp (dict): 选中的抓取字典
            obs (dict): 当前观测信息

        Returns:
            list: 动作序列列表，每个元素为动作字典
        """

        actions = []
        
        # 动作1 - 打开夹爪
        actions.append({
            'arm_pos': obs['arm_pos'].copy(),
            'arm_quat': obs['arm_quat'].copy(),
            'gripper_pos': np.array(0.0)
        })
        
        # 动作2  - 移动到距离抓取接近位置 10cm 的地方 (预接近点)
        r = Rotation.from_quat(grasp['ee_quaternion'])
        approach_vector = r.apply([0, 0, 1]) # 获取 Z 轴向量
        pre_approach_pos = grasp['ee_position'] - 0.10 * approach_vector
        
        actions.append({
            'arm_pos': pre_approach_pos,
            'arm_quat': grasp['ee_quaternion'].copy(),
            'gripper_pos': np.array([0.0])
        })

        # 动作3 - 移动到目标抓取的接近位置
        # # 计算当前位置到目标位置的距离
        # distance = np.linalg.norm(grasp['ee_position'] - obs['arm_pos'])

        # # 根据距离自动调整插值步数
        # num_steps = max(10, int(distance / 0.05))
        # logger.info(f"[GraspPolicy] 距离: {distance:.3f}m, 插值步数: {num_steps}")
        
        # trajectory = self._interpolate_trajectory(
        #     start_pos=obs['arm_pos'],
        #     start_quat=obs['arm_quat'],
        #     end_pos=grasp['ee_position'],
        #     end_quat=grasp['ee_quaternion'],
        #     num_steps=num_steps
        # )
        
        # # 将插值路径点添加到动作序列
        # for pos, quat in trajectory:
        #     actions.append({
        #         'arm_pos': pos,
        #         'arm_quat': quat,
        #         'gripper_pos': np.array([0.0])
        #     })

        actions.append({
            'arm_pos': grasp['ee_position'].copy(),
            'arm_quat': grasp['ee_quaternion'].copy(),
            'gripper_pos': np.array([0.0])
        })
        
        # 动作3 - 闭合夹爪
        actions.append({
            'arm_pos': grasp['ee_position'].copy(),
            'arm_quat': grasp['ee_quaternion'].copy(),
            'gripper_pos': np.array([1.0])
        })
        
        # 动作4 - 提升物体
        lift_pos = grasp['ee_position'].copy()
        lift_pos[2] += 0.2
        
        # lift_trajectory = self._interpolate_trajectory(
        #     start_pos=grasp['ee_position'],
        #     start_quat=grasp['ee_quaternion'],
        #     end_pos=lift_pos,
        #     end_quat=grasp['ee_quaternion'],
        #     num_steps=5
        # )
        
        # for pos, quat in lift_trajectory:
        #     actions.append({
        #         'arm_pos': pos,
        #         'arm_quat': quat,
        #         'gripper_pos': np.array([1.0])
        #     })

        actions.append({
            'arm_pos': lift_pos.copy(),
            'arm_quat': grasp['ee_quaternion'].copy(),
            'gripper_pos': np.array([1.0])
        })
        
        return actions
    
    def _state_executing(self, obs):
        """执行状态 - 执行动作序列"""

        if self.action_index >= len(self.action_sequence):
            # 所有动作执行完毕
            logger.info("抓取动作序列执行完成")
            self.state = 'COMPLETED'
            return 'end_episode'
        
        # 获取当前动作
        action = self.action_sequence[self.action_index]
        logger.debug(f"执行动作 {self.action_index + 1}/{len(self.action_sequence)}")
        
        # 更新索引
        self.action_index += 1
        
        return action

    def get_state(self):
        """
        获取当前策略状态

        Returns:
            str: 当前状态字符串
        """

        return self.state

# 测试代码
if __name__ == '__main__':
    # TODO: 创建测试用的策略
    # 需要准备模型路径、相机参数等
    pass