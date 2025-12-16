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
                self.scale = camera_intrinsics['scale']
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
        points_z = depth / self.scale
        points_x = (xmap - self.cx) / self.fx * points_z
        points_y = (ymap - self.cy) / self.fy * points_z
        
        # 创建有效点mask
        mask = (points_z > 0) & (points_z < 1) # 此处深度依据相机与物体间距离做出更改
        
        # 提取有效点和颜色
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = rgb.astype(np.float32) / 255.0 # 归一化
        colors = colors[mask].astype(np.float32)

        # 打印出来以便检查点云坐标是否合理
        logger.debug(f"点云坐标范围: min={points.min(axis=0)}, max={points.max(axis=0)}")
        
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
        # TODO: 后续实现
        pass
    
    def predict(self, rgb, depth):
        """
        执行抓取检测推理

        Args:
            rgb (np.ndarray): 颜色信息，形状为 [H, W, 3]，类型为 uint8
            depth (np.ndarray): 深度信息，形状为 [H, W]，类型为 float32

        Returns:
            GraspGroup: AnyGrasp 返回的抓取组对象（已排序和NMS处理）
                       TODO: 后续改为返回 List[dict] 格式
        """
        
        # 生成点云
        points, colors = self._rgbd_to_pointcloud(rgb, depth)
        
        # 调用 AnyGrasp 推理
        gg, cloud = self.anygrasp.get_grasp(
            points, colors,
            lims=self.workspace_limits,
            apply_object_mask=True,
            dense_grasp=False,
            collision_detection=True
        )
        
        # 检查是否检测到抓取
        if len(gg) == 0:
            logger.warning("未检测到抓取")
            return gg
        
        # 后处理
        gg = gg.nms().sort_by_score()  # nms()为非极大值抑制
        gg_pick = gg[0:20]  # 选取分数靠前的20个点

        # 打印抓取分数
        logger.info(f"检测到 {len(gg_pick)} 个抓取候选")
        logger.info(f"最佳抓取分数: {gg_pick[0].score:.4f}")
        
        return gg_pick


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