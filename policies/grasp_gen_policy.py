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

        # # === 调试：打印点云统计信息 ===
        # pc_min = point_cloud.min(axis=0)
        # pc_max = point_cloud.max(axis=0)
        # pc_mean = point_cloud.mean(axis=0)
        # logger.info(f"[调试] 点云统计信息:")
        # logger.info(f"  数量: {len(point_cloud)} points")
        # logger.info(f"  范围: x=[{pc_min[0]:.3f}, {pc_max[0]:.3f}], y=[{pc_min[1]:.3f}, {pc_max[1]:.3f}], z=[{pc_min[2]:.3f}, {pc_max[2]:.3f}]")
        # logger.info(f"  中心: [{pc_mean[0]:.3f}, {pc_mean[1]:.3f}, {pc_mean[2]:.3f}]")
        
        logger.info(f"发送点云数据到 GraspGen...")
        
        # 调用 RPC 获取抓取
        try:
            grasps = self.rpc_client.get_grasps(point_cloud.tolist(), 20)
        except Exception as e:
            logger.error(f"RPC 调用失败: {e}")
            return None

        if not grasps:
            logger.warning("GraspGen 未返回任何抓取")
            return None

        logger.info(f"收到 {len(grasps)} 个抓取候选")

        # 筛选最佳抓取
        selected_grasp = self._select_best_grasp(grasps)

        if selected_grasp is None:
            logger.warning("没有找到合适的可达抓取")
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
        
        # === Robotiq 2F-85 夹爪TCP偏移 ===
        # GraspGen返回的是夹爪base_frame的位置，但实际接触点在前方
        # 根据 robotiq_2f_85.yaml 的 contact_points，z坐标约为 0.13m
        # 我们需要沿接近方向（Z轴）前移，使夹爪指尖到达抓取点
        TCP_OFFSET = 0.13 + 0.06  # 米，夹爪base到接触点的距离,此处加上 0.05 是为了适配高精度的齐次变换矩阵，将抓取往接近方向前移若干厘米
        
        for i, grasp in enumerate(grasps):
            # 获取相机坐标系下的位姿矩阵
            T_camera_grasp = np.array(grasp['matrix'])
            
            # # === 调试：打印第一个抓取的原始信息 ===
            # if i == 0:
            #     camera_grasp_pos = T_camera_grasp[:3, 3]
            #     logger.info(f"[调试] GraspGen返回的抓取位置（相机坐标系）: [{camera_grasp_pos[0]:.3f}, {camera_grasp_pos[1]:.3f}, {camera_grasp_pos[2]:.3f}]")
                # logger.info(f"[调试] 抓取评分: {grasp['score']:.3f}")
            
            # === 应用TCP偏移：沿Z轴（接近方向）前移 ===
            # 提取旋转矩阵和位置
            rotation_matrix = T_camera_grasp[:3, :3]
            position = T_camera_grasp[:3, 3]
            
            # Z轴方向（接近方向）是旋转矩阵的第三列
            approach_direction = rotation_matrix[:, 2]
            
            # 沿接近方向前移TCP_OFFSET
            position_corrected = position + TCP_OFFSET * approach_direction
            
            # 更新变换矩阵
            T_camera_grasp_corrected = T_camera_grasp.copy()
            T_camera_grasp_corrected[:3, 3] = position_corrected
            
            # if i == 0:
                # logger.info(f"[调试] TCP偏移补偿: 沿Z轴前移 {TCP_OFFSET}m")
                # logger.info(f"[调试] 补偿后抓取位置（相机坐标系）: [{position_corrected[0]:.3f}, {position_corrected[1]:.3f}, {position_corrected[2]:.3f}]")
            
            # 转换到基座坐标系
            T_base_grasp = self.camera_to_base @ T_camera_grasp_corrected
            
            # 提取位置和姿态
            position = T_base_grasp[:3, 3]
            rotation_matrix = T_base_grasp[:3, :3]
            
            # # 记录第一个抓取的变换结果
            # if i == 0:
            #     logger.info(f"[调试] 变换后抓取位置（基座坐标系）: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
                # logger.info(f"[调试] camera_to_base平移部分: {self.camera_to_base[:3, 3]}")

            # 剔除过低的抓取
            if position[2] < 0.03:
                continue

            # 将旋转矩阵转换为欧拉角（xyz 顺序，单位度）
            r = R.from_matrix(rotation_matrix)

            # 构造一个包含必要信息的字典返回
            quat = r.as_quat()  # [x, y, z, w]

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

        logger.info(f"选择评分最高的抓取，位置（基座坐标系）: [{processed_grasps[0]['arm_pos'][0]:.3f}, {processed_grasps[0]['arm_pos'][1]:.3f}, {processed_grasps[0]['arm_pos'][2]:.3f}], 评分: {processed_grasps[0]['score']:.3f}")

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