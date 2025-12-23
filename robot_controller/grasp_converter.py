"""
坐标系转换器

功能：将相机坐标系的抓取点转换为机器人末端位姿
"""

import numpy as np
import logging
from scipy.spatial.transform import Rotation as R

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraspConverter:
    def __init__(self, camera_intrinsics=None, camera_to_base_transform=None):
        """
        初始化坐标转换器

        Args:
            camera_intrinsics (dict): 相机内参字典
            camera_to_base_transform (np.ndarray): 相机到机器人基坐标系的 4x4 变换矩阵
        """

        # 保存变换矩阵
        if camera_to_base_transform is not None:
            # 检验变换矩阵形状是否正确
            if camera_to_base_transform.shape != (4, 4):
                raise ValueError(f"变换矩阵形状错误: {camera_to_base_transform.shape}")
            
            try:
                self.camera_to_base = camera_to_base_transform
            except Exception as e:
                logger.error(f'未正确传入变换矩阵: {e}')
                raise
        else:
            logger.error('camera_to_base_transform 不能为空')
            raise ValueError('camera_to_base_transform 不能为空')
        
        # 获取旋转矩阵
        self.rotation_c2b = self.camera_to_base[:3, :3]

        # 获取平移部分
        self.translation_c2b = self.camera_to_base[:3, 3]
    
    def grasp_to_ee_pose(self, grasp, approach_distance=0.05):
        """
        将抓取候选转换为末端执行器目标位姿

        Args:
            grasp (dict or GraspGroup): 抓取字典
            approach_distance (float): 接近距离（米）。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - position: np.array([x, y, z])，末端执行器目标位置。
                - quaternion: np.array([qx, qy, qz, qw])，末端执行器目标姿态（四元数）。
        """

        # 获取抓取位置和接近方向
        grasp_pos = grasp['position']
        approach_dir = grasp['approach_direction']
        
        # 转换到机器人基坐标系
        grasp_pos_homo = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2], 1])
        base_pos_homo = self.camera_to_base @ grasp_pos_homo
        base_pos = base_pos_homo[:3]
        logger.info(f"position: ({base_pos[0]}, {base_pos[1]}, {base_pos[2]})")
        
        # 计算接近位置
        approach_dir_base = self.rotation_c2b @ approach_dir
        approach_dir_base = approach_dir_base / np.linalg.norm(approach_dir_base)
        approach_pos_base = base_pos - approach_dir_base * approach_distance
        
        # 将旋转矩阵转换为四元数
        rotation_matrix_camera = grasp['rotation_matrix']    
        rotation_matrix_base = self.rotation_c2b @ rotation_matrix_camera
        rotation = R.from_matrix(rotation_matrix_base)
        quaternion = rotation.as_quat()
        euler = rotation.as_euler('xyz', degrees=True)

        logger.info(f"approach_position_base: ({approach_pos_base[0]}, {approach_pos_base[1]}, {approach_pos_base[2]})")
        logger.info(f"approach_euler_base (xyz, deg): ({euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f})")

        return approach_pos_base, quaternion

    def verify_reachability(self, position, quaternion, ik_solver):
        """
        验证目标位姿是否可达
        
        Args:
            position (np.ndarray): 末端执行器目标位置，形状为 [3]
            quaternion (np.ndarray): 末端执行器目标姿态（四元数），形状为 [4]
            ik_solver (IKSolver): 逆解求解器实例
            
        Returns:
            bool: 目标位姿是否可达
        """

        # 逆运动学求解末端执行器目标姿态
        try:
            # 获取初始关节角
            curr_qpos = ik_solver.qpos0

            # 求解
            qpos = ik_solver.solve(position, quaternion, curr_qpos)
            return True
        except Exception as e:
            logger.debug(f"位姿不可达：{e}")
            return False