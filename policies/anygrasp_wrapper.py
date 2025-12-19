"""
AnyGrasp 推理接口封装

功能：加载模型，执行抓取检测推理
"""

import sys
import os
import numpy as np
import argparse
import logging

sys.path.append(os.path.expanduser('~/documents/anygrasp_sdk'))

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnyGraspWrapper:
    def __init__(self, 
                 checkpoint_path,
                 camera_intrinsics=None,
                 workspace_limits=None,
                 max_gripper_width=0.1,
                 gripper_height=0.03,
                 top_down_grasp=False):
        """
        初始化 AnyGrasp 模型
        
        Args:
            checkpoint_path: 模型权重路径
            camera_intrinsics: 相机内参字典 {'fx', 'fy', 'cx', 'cy', 'scale'}
            workspace_limits: 工作空间限制[xmin, xmax, ymin, ymax, zmin, zmax]
            max_gripper_width: 最大夹爪宽度 (米)
            gripper_height: 夹爪高度 (米)
            top_down_grasp: 是否只输出从上往下的抓取
        """

        # 保存相机内参
        if camera_intrinsics is not None:
            try:
                self.fx = camera_intrinsics['fx']
                self.fy = camera_intrinsics['fy']
                self.cx = camera_intrinsics['cx']
                self.cy = camera_intrinsics['cy']
            except Exception as e:
                logger.error(f'未正确传入相机内参: {e}')
                raise
        else:
            logger.error('camera_intrinsics 不能为空')
            raise ValueError('camera_intrinsics 不能为空')
        
        # 保存工作空间限制
        if workspace_limits is not None:
            try:
                self.workspace_limits = workspace_limits
            except Exception as e:
                logger.error(f'未正确传入工作空间限制: {e}')
                raise
        else:
            logger.error('workspace_limits 不能为空')
            raise ValueError('workspace_limits 不能为空')
        
        # 创建配置对象
        cfgs = argparse.Namespace(
            checkpoint_path = checkpoint_path,
            max_gripper_width = max(0, min(0.1, max_gripper_width)),
            gripper_height = gripper_height,
            top_down_grasp = top_down_grasp,
            debug = False
        )
        
        # 初始化并加载模型
        self.anygrasp = AnyGrasp(cfgs)
        self.anygrasp.load_net()
        
        logger.info(f'[AnyGraspWrapper] 模型加载成功: {checkpoint_path}')
    
    def _rgbd_to_pointcloud(self, rgb, depth):
        """
        将 RGB-D 图像转换为点云

        Args:
            rgb (np.ndarray): 颜色信息，形状为 [H, W, 3]，类型为 uint8
            depth (np.ndarray): 深度信息，形状为 [H, W]，类型为 float32

        Returns:
            points (np.ndarray): 点云三维坐标，形状为 [N, 3]
            colors (np.ndarray): 点云颜色，形状为 [N, 3]
        """

        # 输入验证
        assert rgb.ndim == 3 and rgb.shape[2] == 3
        assert rgb.dtype == np.uint8
        assert depth.ndim == 2
        
        # 生成点云
        xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth
        points_x = (xmap - self.cx) / self.fx * points_z
        points_y = (ymap - self.cy) / self.fy * points_z
        
        # 打印深度统计信息
        valid_depth_mask = depth > 0
        if valid_depth_mask.any():
            logger.info(f"[_rgbd_to_pointcloud] 深度范围: {depth[valid_depth_mask].min():.3f}m ~ {depth[valid_depth_mask].max():.3f}m")
        else:
            logger.warning("[_rgbd_to_pointcloud] 深度图全为0！")
        
        # 创建有效点mask
        # 假设工作距离在 0.1m 到 5.0m 之间
        mask = (points_z > 0.1) & (points_z < 5.0)
        
        # 提取有效点和颜色
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = rgb.astype(np.float32) / 255.0 # 归一化
        colors = colors[mask].astype(np.float32)

        # 检查点云是否为空
        if len(points) == 0:
            logger.warning("[_rgbd_to_pointcloud] 没有有效点云！深度值可能不在有效范围内")
        else:
            logger.info(f"[_rgbd_to_pointcloud] 点云坐标范围: "
                       f"x=[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], "
                       f"y=[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], "
                       f"z=[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        return points, colors
    
    def _parse_grasp_group(self, gg, max_grasps=50):
        """
        将 GraspGroup 转换为标准字典列表
        
        Args:
            gg (GraspGroup): AnyGrasp 返回的抓取组
            max_grasps (int): 最大返回的抓取数量
            
        Returns:
            List[dict]: 抓取候选列表，每个元素包含：
                - position: [x, y, z] 抓取位置
                - approach_direction: [dx, dy, dz] 接近方向
                - angle: 旋转角度
                - width: 夹爪宽度
                - score: 抓取评分
        """

        if len(gg) == 0:
            logger.warning("GraspGroup 对象为空")
            return []
        
        # 限制抓取数量
        # num_grasps = min(len(gg), max_grasps)
        num_grasps = len(gg) # 未知抓取数量对最终性能的影响，此处暂时保留全部抓取
        logger.info(f"解析 {num_grasps} 个抓取候选")
        
        # 获取三维坐标数据
        positions = gg.translations[:num_grasps]
        
        # 获取旋转矩阵
        rotation_matrices = gg.rotation_matrices[:num_grasps]
        
        # 获取夹爪宽度
        widths = gg.widths[:num_grasps]
        
        # 获取深度
        depths = gg.depths[:num_grasps]
        
        # 获取抓取分数
        scores = gg.scores[:num_grasps]
        
        # 逐个处理每个抓取
        grasp_list = []
        
        for i in range(num_grasps):
            # 获取接近方向
            approach_direction = rotation_matrices[i][:, 2] # 此处未知旋转矩阵的构建规则，假设是旋转变换矩阵
            
            # 归一化接近方向为单位向量
            approach_direction = approach_direction / np.linalg.norm(approach_direction)
            
            # 构造抓取字典
            grasp_dict = {
                'position': positions[i].copy(),                # 三维坐标
                'approach_direction': approach_direction,       # 接近方向
                'rotation_matrix': rotation_matrices[i].copy(), # 旋转矩阵
                'angle': 0.0,                                   # 角度，后续明确旋转矩阵构造后可从中计算得欧拉角
                'width': float(widths[i]),                      # 宽度
                'score': float(scores[i])                       # 抓取分数
            }
            
            # 添加到列表
            grasp_list.append(grasp_dict)
        
        logger.info(f"成功将 {len(grasp_list)} 个抓取处理为字典")
        return grasp_list
    
    def predict(self, rgb, depth):
        """
        执行抓取检测推理

        Args:
            rgb (np.ndarray): 颜色信息，形状为 [H, W, 3]，类型为 uint8
            depth (np.ndarray): 深度信息，形状为 [H, W]，类型为 float32

        Returns:
            list[dict]: 抓取字典列表
        """
        
        # 生成点云
        points, colors = self._rgbd_to_pointcloud(rgb, depth)
        
        # 检查点云是否为空
        if len(points) == 0:
            logger.warning("[AnyGraspWrapper] 点云为空，无法进行抓取检测")
            return []
        
        # 调用 AnyGrasp 推理
        logger.info("[AnyGraspWrapper] 调用 AnyGrasp 推理...")
        gg, cloud = self.anygrasp.get_grasp(
            points, colors,
            lims=self.workspace_limits,
            apply_object_mask=True,
            dense_grasp=False,
            collision_detection=True
        )
        
        # 检查是否检测到抓取
        if len(gg) == 0:
            logger.warning("[AnyGraspWrapper] 未检测到任何抓取")
            return []
        
        # 后处理
        gg = gg.nms().sort_by_score()  # nms()为非极大值抑制
        gg_pick = gg[0:20]  # 选取分数靠前的20个点

        # 打印抓取分数
        logger.info(f"[AnyGraspWrapper] 筛选后抓取数量: {len(gg_pick)}")
        if len(gg_pick) > 0:
            logger.info(f"[AnyGraspWrapper] 最佳抓取分数: {gg_pick.scores[0]:.4f}")
        
        # 转换为字典列表
        grasp_list = self._parse_grasp_group(gg_pick)
        
        return grasp_list


# 测试代码
if __name__ == '__main__':
    # 模型路径
    checkpoint_path = 'path/to/checkpoint'
    
    # 创建实例
    wrapper = AnyGraspWrapper(checkpoint_path)
    
    # 创建虚拟输入或加载真实数据
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.zeros((480, 640), dtype=np.float32)
    
    # 推理
    grasps = wrapper.predict(rgb, depth)
    
    print(f"最佳抓取: {grasps[0]}")