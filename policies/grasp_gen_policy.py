import numpy as np
from rich.logging import RichHandler
import logging
import xmlrpc.client
from scipy.spatial.transform import Rotation as R

# 日志输出配置
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger("rich")

class GraspGenPolicy:
    def __init__(self, camera_to_base_transform, grasp_gen_host='localhost', grasp_gen_port=60000):
        """
        初始化 GraspGenPolicy 策略。

        Args:
            camera_to_base_transform (np.ndarray): 4x4 变换矩阵，将相机坐标系下的点变换到基座坐标系。
            grasp_gen_host (str, optional): GraspGen RPC 服务器的主机名或 IP 地址。默认 'localhost'。
            grasp_gen_port (int, optional): GraspGen RPC 服务器端口号。默认 60000。
        """

        self.camera_to_base = camera_to_base_transform
        
        # 连接 RPC
        url = f"http://{grasp_gen_host}:{grasp_gen_port}"
        logger.info(f"正在连接 GraspGen 服务器: {url}")
        self.rpc_client = xmlrpc.client.ServerProxy(url)
        
        # 状态机
        self.state = 'DETECTING'
        self.action_sequence = []
        self.action_index = 0
        self.latest_grasps = []

    def reset(self):
        """
        重置策略状态机和动作序列。

        Returns:
            None
        """
        self.state = 'DETECTING'
        self.action_sequence = []
        self.action_index = 0
        self.latest_grasps = []
        logger.info("策略已重置")

    def step(self, obs, point_cloud=None):
        """
        执行一步策略，根据当前状态机决定动作。

        Args:
            obs (dict): 当前机械臂状态信息，包含位置、姿态等。
            point_cloud (np.ndarray, optional): (N, 3) 物体点云数据。

        Returns:
            dict or str or None: 返回动作字典、'end_episode' 或 None。
        """

        if self.state == 'DETECTING':
            return self._state_detecting(obs, point_cloud)
        elif self.state == 'EXECUTING':
            return self._state_executing(obs)
        elif self.state == 'COMPLETED':
            return 'end_episode'
        return None

    def _state_detecting(self, obs, point_cloud):
        """
        检测抓取点并生成动作序列。

        Args:
            obs (dict): 当前机械臂状态信息。
            point_cloud (np.ndarray): (N, 3) 物体点云数据。

        Returns:
            dict or None: 第一个动作字典，或 None（无可用抓取）。
        """
        if point_cloud is None:
            logger.warning("未提供点云数据，无法进行抓取检测")
            return None

        logger.info(f"发送点云数据 ({len(point_cloud)} points) 到 GraspGen...")
        
        # 调用 RPC 获取抓取
        try:
            grasps = self.rpc_client.get_grasps(point_cloud.tolist(), 20)
        except Exception as e:
            logger.error(f"\nRPC 调用失败: {e}\n")
            return None

        if not grasps:
            logger.warning("\nGraspGen 未返回任何抓取\n")
            return None

        logger.info(f"收到 {len(grasps)} 个抓取候选")

        # 筛选最佳抓取
        selected_grasp = self._select_best_grasp(grasps)

        if selected_grasp is None:
            logger.warning("\n没有找到合适的可达抓取\n")
            return None

        # 生成动作序列
        self.action_sequence = self._generate_action_sequence(selected_grasp, obs)
        self.state = 'EXECUTING'
        
        # 立即执行第一个动作
        return self._state_executing(obs)

    def _select_best_grasp(self, grasps):
        """
        从抓取候选中筛选最佳抓取，并进行坐标系转换。

        Args:
            grasps (list): 抓取候选列表，每项为 {'score': float, 'matrix': list}。

        Returns:
            dict or None: 最佳抓取的末端执行器位姿字典（包含 arm_pos, arm_quat, score），或 None。
        """
        processed_grasps = []
        for grasp in grasps:
            # 获取相机坐标系下的位姿矩阵
            T_camera_grasp = np.array(grasp['matrix'])
            
            # 转换到基座坐标系
            T_base_grasp = self.camera_to_base @ T_camera_grasp
            
            # 提取位置和姿态
            position = T_base_grasp[:3, 3]
            rotation_matrix = T_base_grasp[:3, :3]
            
            # 过滤掉从下方接近的抓取
            approach_vector = rotation_matrix[:3, 2]
            if approach_vector[2] > -0.1:
                continue

            # 构造一个包含必要信息的字典返回
            r = R.from_matrix(rotation_matrix)
            quat = r.as_quat() # [x, y, z, w]
            
            processed_grasps.append({
                'arm_pos': position,
                'arm_quat': quat,
                'score': grasp['score']
            })
            
        self.latest_grasps = processed_grasps
        
        if not processed_grasps:
            return None
            
        # 按分数降序排序
        processed_grasps.sort(key=lambda x: x['score'], reverse=True)
        
        return processed_grasps[0]

    def _generate_action_sequence(self, grasp_pose, obs):
        """
        根据抓取位姿和当前机械臂状态生成动作序列。

        动作序列包括：打开夹爪、预接近、接近、闭合夹爪、提升。

        Args:
            grasp_pose (dict): 目标抓取的末端执行器位姿（arm_pos, arm_quat, score）。
            obs (dict): 当前机械臂状态。

        Returns:
            list: 动作字典列表，每项包含 arm_pos, arm_quat, gripper_pos。
        """
        actions = []
        
        target_pos = grasp_pose['arm_pos']
        target_quat = grasp_pose['arm_quat']
        
        # 打开夹爪
        actions.append({
            'arm_pos': obs['arm_pos'],
            'arm_quat': obs['arm_quat'],
            'gripper_pos': np.array([0.0])
        })
        
        # 预接近 (沿 Z 轴后退 10cm)
        r = R.from_quat(target_quat)
        approach_vector = r.apply([0, 0, 1])
        pre_pos = target_pos - 0.10 * approach_vector
        
        actions.append({
            'arm_pos': pre_pos,
            'arm_quat': target_quat,
            'gripper_pos': np.array([0.0])
        })
        
        # 接近抓取点
        actions.append({
            'arm_pos': target_pos,
            'arm_quat': target_quat,
            'gripper_pos': np.array([0.0])
        })
        
        # 闭合夹爪
        actions.append({
            'arm_pos': target_pos,
            'arm_quat': target_quat,
            'gripper_pos': np.array([1.0])
        })
        
        # 提升
        lift_pos = target_pos.copy()
        lift_pos[2] += 0.2
        
        actions.append({
            'arm_pos': lift_pos,
            'arm_quat': target_quat,
            'gripper_pos': np.array([1.0])
        })
        
        return actions

    def _state_executing(self, obs):
        """
        执行动作序列中的下一个动作。

        Args:
            obs (dict): 当前机械臂状态。

        Returns:
            dict or str: 当前动作字典，或 'end_episode' 表示动作序列结束。
        """
        if self.action_index >= len(self.action_sequence):
            self.state = 'COMPLETED'
            return 'end_episode'
        
        action = self.action_sequence[self.action_index]
        self.action_index += 1
        return action