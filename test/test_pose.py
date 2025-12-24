"""
test whether the robot can reach a specific pose
"""


import numpy as np
import time
import logging
import os
import sys
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from real_env import RealEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_specific_pose():
    try:
        logger.info("\n初始化 RealEnv...\n")
        env = RealEnv()
        
        logger.info("\n重置环境...\n")
        env.reset()
        
        # 获取并打印当前位姿
        obs = env.get_obs()
        curr_pos = obs['arm_pos']
        curr_quat = obs['arm_quat']
        curr_euler = R.from_quat(curr_quat).as_euler('xyz', degrees=True)
        
        logger.info(f"当前位置: {np.round(curr_pos, 3)}")
        logger.info(f"当前姿态 (quat [x, y, z, w]): {np.round(curr_quat, 3)}")
        logger.info(f"当前姿态 (euler [x, y, z] degrees): {np.round(curr_euler, 3)}\n")
        
        target_pos = np.array([0.25, 0.10, 0.1])
        
        target_quat = np.array([0.687, 0.726, 0.021, 0.039]) 
        
        action = {
            'arm_pos': target_pos,
            'arm_quat': target_quat,
            'gripper_pos': np.array([0.0])
        }
        
        target_euler = R.from_quat(target_quat).as_euler('xyz', degrees=True)
        logger.info(f"目标位置: {np.round(target_pos, 3)}")
        logger.info(f"目标姿态 (quat [x, y, z, w]): {np.round(target_quat, 3)}")
        logger.info(f"目标姿态 (euler [x, y, z] degrees): {np.round(target_euler, 3)}\n")
        
        env.step(action, wait_for_arrival=True, timeout=15)
        
        time.sleep(10)
        
        logger.info("\n测试完成，正在重置并退出...\n")
        env.reset()
        
    except Exception as e:
        logger.error(f"\n测试过程中发生错误: {e}\n")
    finally:
        pass

if __name__ == "__main__":
    test_specific_pose()
