"""
AnyGrasp RPC 服务器

功能：
    - 运行在 AnyGrasp 环境中，提供抓取检测服务
"""
import argparse
import logging
import os
import sys
import numpy as np
from multiprocessing.managers import BaseManager as MPBaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from policies.anygrasp_wrapper import AnyGraspWrapper


class AnyGraspService:
    """
    封装 AnyGraspWrapper，提供 RPC 接口
    """
    
    def __init__(self, checkpoint_path, cfgs):
        """
        初始化 AnyGrasp 模型

        Args:
            checkpoint_path (str): 模型权重路径
            cfgs (dict): AnyGrasp 配置字典
        """

        self.anygrasp = AnyGraspWrapper(checkpoint_path, cfgs)
        logger.info("[AnyGraspService] 模型初始化完成")
    
    def predict(self, rgb, depth):
        """
        执行抓取检测

        Args:
            rgb (np.ndarray): [H, W, 3] uint8 RGB 图像。
            depth (np.ndarray): [H, W] float32 深度图（米）。

        Returns:
            list[dict]: 抓取列表，每个抓取包含：
                - position (list): [3] 抓取位置 (x, y, z)
                - rotation_matrix (np.ndarray): [3, 3] 旋转矩阵
                - approach_direction (list): [3] 接近方向
                - width (float): 抓取宽度
                - score (float): 抓取分数
        """

        logger.info(f"[AnyGraspService] 收到图像: rgb={rgb.shape}, depth={depth.shape}")

        # 调用 AnyGraspWrapper
        grasp_list = self.anygrasp.predict(rgb, depth)

        logger.info(f"[AnyGraspService] 推理完成，返回 {len(grasp_list)} 个抓取")
        return grasp_list


class AnyGraspManager(MPBaseManager):
    """
    RPC Manager
    """
    pass


# 注册服务
AnyGraspManager.register('AnyGraspService', AnyGraspService)


if __name__ == '__main__':
    # 配置参数
    ANYGRASP_RPC_HOST = 'localhost'
    ANYGRASP_RPC_PORT = 50000
    RPC_AUTHKEY = b'anygrasp'

    # AnyGrasp 模型配置
    checkpoint_path = 'todo'
    max_gripper_width = 0.1  # 最大夹爪宽度（米）
    gripper_height = 0.03    # 夹爪高度（米）
    top_down_grasp = False   # 是否只检测俯视抓取
    
    cfgs = argparse.Namespace(
        checkpoint_path=checkpoint_path,
        max_gripper_width=max(0, min(0.1, max_gripper_width)),
        gripper_height=gripper_height,
        top_down_grasp=top_down_grasp,
        debug=False
    )

    # 创建服务实例
    logger.info(f"[AnyGraspServer] 正在加载模型: {checkpoint_path}")
    service = AnyGraspService(checkpoint_path, cfgs)
    
    # 创建 Manager
    manager = AnyGraspManager(
        address=(ANYGRASP_RPC_HOST, ANYGRASP_RPC_PORT),
        authkey=RPC_AUTHKEY
    )

    # 启动服务器
    server = manager.get_server()
    logger.info(f'AnyGrasp RPC 服务器已启动: {ANYGRASP_RPC_HOST}:{ANYGRASP_RPC_PORT}')
    logger.info('等待客户端连接...')
    server.serve_forever()