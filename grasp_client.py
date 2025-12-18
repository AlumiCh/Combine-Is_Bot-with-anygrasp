"""
AnyGrasp RPC 客户端

功能：
    - 在 Is_Bot 环境中使用，连接到 anygrasp_server
"""
import logging
import numpy as np
from multiprocessing.managers import BaseManager

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

    # 模拟数据
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.random.rand(480, 640).astype(np.float32)

    # 调用检测
    grasp_list = client.predict(rgb, depth)
    logger.info(f"收到 {len(grasp_list)} 个抓取")

    client.close()