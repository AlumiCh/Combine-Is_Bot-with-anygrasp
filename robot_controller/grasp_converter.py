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
        
        # AnyGrasp坐标系到机械臂末端执行器坐标系的变换
        # AnyGrasp坐标系：X轴=接近方向向外，Y轴=夹爪之间，Z轴=正交
        # 机械臂末端执行器坐标系：Z轴=接近方向，X轴=夹爪之间，Y轴=正交
        self.anygrasp_to_ee = np.array([
            [ 0,  1,  0],  # EE的X轴 = AnyGrasp的Y轴
            [ 0,  0,  1],  # EE的Y轴 = AnyGrasp的Z轴
            [ 1,  0,  0]   # EE的Z轴 = -AnyGrasp的X轴
        ])
    
    def grasp_to_ee_pose(self, grasp, approach_distance=0.03):
        """
        将抓取候选转换为末端执行器目标位姿
        
        坐标变换流程：
        1. AnyGrasp坐标系 → 末端执行器坐标系（相机参考系下）
        2. 相机坐标系 → 机器人基座坐标系
        3. 计算接近位置（沿接近方向后退一定距离）

        Args:
            grasp (dict): 抓取字典，包含：
                - position: [x,y,z] 抓取位置（相机坐标系）
                - rotation_matrix: 3x3旋转矩阵（AnyGrasp坐标系）
            approach_distance (float): 接近距离（米），默认0.05m

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - position: 末端执行器目标位置，形状(3,)
                - quaternion: 末端执行器目标姿态，形状(4,) [x,y,z,w]
        """
        
        # 提取 AnyGrasp 传入的数据
        grasp_pos_camera = grasp['position']  # 相机坐标系下的抓取位置
        rotation_anygrasp = grasp['rotation_matrix']  # AnyGrasp坐标系的旋转矩阵
        
        # AnyGrasp坐标系 → 末端执行器坐标系
        # 在相机坐标系下，重组旋转矩阵的列向量，得到EE的坐标轴表示
        rotation_ee_in_camera = rotation_anygrasp @ self.anygrasp_to_ee.T
        
        # 相机坐标系 → 基座坐标系
        grasp_pos_homo = np.append(grasp_pos_camera, 1) # 转为齐次坐标
        grasp_pos_base = (self.camera_to_base @ grasp_pos_homo)[:3]
        
        logger.info(f"\n基座坐标系抓取位置: [{grasp_pos_base[0]:.3f}, {grasp_pos_base[1]:.3f}, {grasp_pos_base[2]:.3f}]\n")
        
        rotation_ee_in_base = self.rotation_c2b @ rotation_ee_in_camera
             
        # 计算接近位置
        ee_z_axis_base = rotation_ee_in_base[:, 2]
        ee_z_axis_base_normalized = ee_z_axis_base / np.linalg.norm(ee_z_axis_base)
        
        # 沿Z轴负方向后退，得到接近位置
        approach_pos_base = grasp_pos_base - ee_z_axis_base_normalized * approach_distance
        
        logger.info(f"\n接近位置(基座): [{approach_pos_base[0]:.3f}, {approach_pos_base[1]:.3f}, {approach_pos_base[2]:.3f}]\n")
        
        # 转换为四元数和欧拉角
        rotation_obj = R.from_matrix(rotation_ee_in_base)
        quaternion = rotation_obj.as_quat()  # [x, y, z, w]
        euler_xyz_deg = rotation_obj.as_euler('xyz', degrees=True)
        
        logger.info(f"\n欧拉角XYZ(度):   [{euler_xyz_deg[0]:7.2f}, {euler_xyz_deg[1]:7.2f}, {euler_xyz_deg[2]:7.2f}]\n")

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