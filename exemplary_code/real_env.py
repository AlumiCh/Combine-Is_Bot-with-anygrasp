# Update: 2025-03-19

import time
import logging

from cameras import KinovaCamera, LogitechCamera
from configs.constants import BASE_RPC_HOST, BASE_RPC_PORT, ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from configs.constants import BASE_CAMERA_SERIAL
from arm_server import ArmManager 
# from base_server import BaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealEnv:
    """
    实现真实环境的类，负责与机器人底座和手臂的RPC连接。
    
    属性:
        base: 底座的RPC代理对象。
        arm: 手臂的RPC代理对象。
        base_camera: 底座相机的实例。
        wrist_camera: 手腕相机的实例。

    异常:
        ConnectionRefusedError: 如果无法连接到RPC服务器，将引发此异常。
    """
    def __init__(self):
        # only gen3 -> base,camera 都先关闭 
        
        # RPC服务器连接底座
        # base_manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
        # try:
        #     base_manager.connect()
        # except ConnectionRefusedError as e:
        #     raise Exception('无法连接到底座RPC服务器，请确保base_server.py正在运行。') from e

        # RPC服务器连接手臂
        arm_manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            arm_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception('无法连接到手臂RPC服务器，请确保arm_server.py正在运行。') from e

        # RPC代理对象
        # self.base = base_manager.Base(max_vel=(0.5, 0.5, 1.57), max_accel=(0.5, 0.5, 1.57))
        self.arm = arm_manager.Arm()

        # 相机
        # self.base_camera = LogitechCamera(BASE_CAMERA_SERIAL)
        # self.wrist_camera = KinovaCamera()

    def get_obs(self):
        """
        获取当前环境的观测数据。

        返回:
            dict: 包含底座状态、手臂状态和相机图像的字典。
        """
        obs = {}
        # obs.update(self.base.get_state())
        obs.update(self.arm.get_state())
        # obs['base_image'] = self.base_camera.get_image()
        # obs['wrist_image'] = self.wrist_camera.get_image()
        return obs

    def reset(self):
        """
        重置底座和手臂的状态。
        """
        # print('正在重置底座...')
        # self.base.reset()

        print('[real_env-reset] 正在重置手臂...')
        self.arm.reset()

        print('[real_env-reset] 机器人已重置')

    def step(self, action, wait_for_arrival=True, position_threshold=0.022, timeout=10.0):
        """
        执行给定的动作。

        参数:
            action (dict): 包含底座和手臂动作的字典。
            wait_for_arrival (bool): 是否等待机械臂到达目标位置，默认 True
            position_threshold (float): 位置收敛阈值（米），默认 0.01m = 1cm
            timeout (float): 超时时间（秒），默认 10.0s

        注意: 
            我们故意不在此处返回观测数据，以防止策略使用过时的数据。
        """
        # self.base.execute_action(action)  # 非阻塞
        # 执行动作
        print('[real_env-step] 执行动作...')

        # target position
        target_pos = action['arm_pos']
        logger.info(f"target position: {target_pos}")

        #
        target_quat = action['arm_quat']
        logger.info(f"target quaternion: {target_quat}")

        # current position
        curr_pos = self.get_obs()['arm_pos']
        logger.info(f"current position: {curr_pos}")

        #
        curr_quat = self.get_obs()['arm_quat']
        logger.info(f"current quaternion: {curr_quat}")

        # 如果需要等待到达
        if wait_for_arrival:
            import numpy as np
            start_time = time.time()
            
            while True:
                # move
                self.arm.execute_action(action)   # 非阻塞

                time.sleep(0.05)
                
                # 获取当前位置
                curr_pos = self.get_obs()['arm_pos']

                #
                curr_quat = self.get_obs()['arm_quat']
                
                # 计算位置误差
                position_error = np.linalg.norm(curr_pos - target_pos)

                #
                quaternion_error = np.linalg.norm(curr_quat - target_quat)
                
                # 检查是否到达
                if position_error < position_threshold:
                    logger.info(f"[real_env-step] 已到达目标位置，position误差: {position_error:.4f}m")
                    logger.info(f"[real_env-step] 已到达目标位置，quaternion误差: {quaternion_error:.4f}")
                    break
                
                # 检查超时
                if time.time() - start_time > timeout:
                    logger.warning(f"[real_env-step] 等待超时 ({timeout}s)，当前position误差: {position_error:.4f}m")
                    logger.warning(f"[real_env-step] 等待超时 ({timeout}s)，当前quaternion误差: {quaternion_error:.4f}")
                    break
        else:
            self.arm.execute_action(action)   # 非阻塞
            time.sleep(0.05)

    def close(self):
        """
        关闭所有连接的相机和RPC代理。
        """
        # self.base.close()
        self.arm.close()
        # self.base_camera.close()
        # self.wrist_camera.close()

if __name__ == '__main__':
    import time
    import numpy as np
    from configs.constants import POLICY_CONTROL_PERIOD
    env = RealEnv()
    try:
        while True:
            env.reset()
            for _ in range(100):
                action = {
                    # 'base_pose': 0.1 * np.random.rand(3) - 0.05,
                    'arm_pos': 0.1 * np.random.rand(3) + np.array([0.55, 0.0, 0.4]),
                    'arm_quat': np.random.rand(4),
                    'gripper_pos': np.random.rand(1),
                }
                env.step(action)
                obs = env.get_obs()
                print([(k, v.shape) if v.ndim == 3 else (k, v) for (k, v) in obs.items()])
                time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    finally:
        env.close()
