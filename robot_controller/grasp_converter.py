"""
坐标系转换器

功能：将相机坐标系的抓取点转换为机器人末端位姿
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

class GraspConverter:
    def __init__(self, camera_intrinsics=None, camera_to_base_transform=None):
        """
        初始化坐标转换器
        
        Args:
            camera_intrinsics: 相机内参字典（暂时可能用不到）
            camera_to_base_transform: 4x4 变换矩阵（相机→机器人基坐标系）
                                     如果为 None，需要提供默认值或报错
        """
        # TODO: 保存变换矩阵
        if camera_to_base_transform is None:
            # 选项1: 使用单位矩阵（假设相机和机器人对齐）
            # 选项2: 报错要求用户提供
            pass
        
        self.camera_to_base = camera_to_base_transform
    
    def grasp_to_ee_pose(self, grasp, approach_distance=0.05):
        """
        将抓取候选转换为末端执行器目标位姿
        
        Args:
            grasp: 抓取字典，包含 'position', 'approach_direction' 等
                  或者 GraspGroup 对象（如果暂时没实现解析）
            approach_distance: 接近距离（米）
            
        Returns:
            (position, quaternion): 
            - position: np.array([x, y, z])
            - quaternion: np.array([qx, qy, qz, qw])
        """
        # TODO: 
        # 1. 提取抓取位置和方向
        #    如果 grasp 是字典：
        #    grasp_pos = grasp['position']
        #    approach_dir = grasp['approach_direction']
        #    
        #    如果 grasp 是 GraspGroup[i]：
        #    需要你探索如何提取
        
        # 2. 转换到机器人基坐标系
        #    grasp_pos_homo = np.array([x, y, z, 1])
        #    base_pos_homo = self.camera_to_base @ grasp_pos_homo
        #    base_pos = base_pos_homo[:3]
        
        # 3. 计算接近位置（沿接近方向后退）
        #    approach_pos = base_pos - approach_dir * approach_distance
        
        # 4. 构造旋转矩阵并转换为四元数
        #    这是最复杂的部分，需要从接近方向构造完整的旋转
        #    提示：接近方向通常对应 z 轴
        #    需要构造 x, y, z 三个正交轴
        
        # 5. 返回位置和四元数
        pass
    
    def verify_reachability(self, position, quaternion, ik_solver):
        """
        验证目标位姿是否可达
        
        Args:
            position: np.array([x, y, z])
            quaternion: np.array([qx, qy, qz, qw])
            ik_solver: IKSolver 实例
            
        Returns:
            bool: 是否可达
        """
        # TODO:
        # 1. 尝试调用 IK 求解
        #    try:
        #        qpos = ik_solver.solve(position, quaternion, curr_qpos)
        #        return True
        #    except:
        #        return False
        
        # 注意：需要提供 curr_qpos（当前关节角度）
        # 可以使用默认的初始配置
        pass