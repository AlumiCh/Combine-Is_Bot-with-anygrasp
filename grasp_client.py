"""
AnyGrasp RPC 客户端

功能：
    - 在 Is_Bot 环境中使用，连接到 anygrasp_server
    - 服务器端负责采集图像和推理，客户端只接收抓取结果
"""
import logging
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
    
    def detect_grasps(self):
        """
        请求抓取检测

        Returns:
            list[dict]: 抓取列表，每个抓取包含：
                - position: [3] 抓取位置 (x, y, z)
                - rotation_matrix: [3, 3] 旋转矩阵
                - approach_direction: [3] 接近方向
                - width: float 抓取宽度
                - score: float 抓取分数
        """
        logger.info("[AnyGraspClient] 请求抓取检测...")
        return self.service.detect_grasps()
    
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

    try:
        # 调用检测
        logger.info("正在请求 AnyGrasp 服务器进行抓取检测...")
        grasp_list = client.detect_grasps()
        
        logger.info(f"收到 {len(grasp_list)} 个抓取")
        
        # 打印前几个抓取的详细信息
        for i, grasp in enumerate(grasp_list[:5]):
            logger.info(f"\n抓取 {i+1}:")
            logger.info(f"  位置: {grasp['position']}")
            logger.info(f"  分数: {grasp['score']:.3f}")
            logger.info(f"  宽度: {grasp['width']:.3f}m")
            logger.info(f"  接近方向: {grasp['approach_direction']}")
    
    except Exception as e:
        logger.error(f"测试失败: {e}")
    
    finally:
        client.close()
        logger.info("测试完成") 