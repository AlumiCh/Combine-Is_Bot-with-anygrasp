"""
AnyGrasp RPC 客户端

功能：
    - 在 Is_Bot 环境中使用，连接到 anygrasp_server
"""
import logging
import numpy as np
import time
from multiprocessing.managers import BaseManager
from cameras import RealSenseCamera

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnyGraspClient:
    """
    AnyGrasp RPC 客户端
    提供与 AnyGraspWrapper 相同的接口
    """
    
    def __init__(self, host='localhost', port=50000, authkey=b'anygrasp'):
        """
        连接到 AnyGrasp RPC 服务器

        Args:
            host (str): 服务器地址
            port (int): 服务器端口
            authkey (bytes): 认证密钥
        """

        self.host = host
        self.port = port
        
        # 创建 Manager 并连接
        class ClientManager(BaseManager):
            pass
        
        ClientManager.register('AnyGraspService')
        
        self.manager = ClientManager(address=(host, port), authkey=authkey)
        
        logger.info(f"[AnyGraspClient] 正在连接到 {host}:{port}...")
        try:
            self.manager.connect()
            self.service = self.manager.AnyGraspService()
            logger.info("[AnyGraspClient] 连接成功")
        except ConnectionRefusedError as e:
            raise Exception(
                f'无法连接到 AnyGrasp RPC 服务器 ({host}:{port})，'
                '请确保 anygrasp_server.py 正在 AnyGrasp 环境中运行。'
            ) from e
    
    def predict(self, rgb, depth):
        """
        执行抓取检测

        Args:
            rgb (np.ndarray): [H, W, 3] uint8 RGB 图像
            depth (np.ndarray): [H, W] float32 深度图（米）

        Returns:
            list[dict]: 抓取列表
        """
        return self.service.predict(rgb, depth)
    
    def close(self):
        """
        关闭连接
        """
        logger.info("[AnyGraspClient] 关闭连接")


# 测试代码
if __name__ == '__main__':
    # 创建客户端
    client = AnyGraspClient(
        host='localhost',
        port=50000,
        authkey=b'anygrasp'
    )

    logger.info("初始化相机")
    camera = RealSenseCamera(resolution=(640, 480), fps=30)

    # 获取相机内参
    intrinsics = camera.get_intrinsics()
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    logger.info(f"相机内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # 工作区域设置
    xmin, xmax = -0.3, 0.3
    ymin, ymax = -0.3, 0.3
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    try:
        while True:
            # 获取 RGB 和深度图
            rgb, depth = camera.get_rgb_depth()
            
            if rgb is None or depth is None:
                logger.info("等待相机数据")
                time.sleep(0.1)
                continue
            
            break
    finally:
        camera.close()
        logger.info("相机已关闭")

    # 调用检测
    logger.info("正在发送数据到 AnyGrasp 服务器...")
    grasp_list = client.predict(rgb, depth)
    logger.info(f"收到 {len(grasp_list)} 个抓取")
    
    # 打印前几个抓取的信息
    for i, grasp in enumerate(grasp_list[:3]):
        logger.info(f"抓取 {i}: position={grasp['position']}, score={grasp['score']:.3f}")

    client.close()