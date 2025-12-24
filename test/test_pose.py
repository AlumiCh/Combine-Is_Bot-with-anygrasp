"""
test whether the robot can reach a specific pose
"""


import numpy as np
import time
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from real_env import RealEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_specific_pose():
    try:
        logger.info("初始化 RealEnv...")
        env = RealEnv()
        
        logger.info("重置环境...")
        env.reset()
        
        target_pos = np.array([0.25, 0.10, 0.1])
        
        target_quat = np.array([0.687, 0.726, 0.021, 0.039]) 
        
        action = {
            'arm_pos': target_pos,
            'arm_quat': target_quat,
            'gripper_pos': np.array([0.0])
        }
        
        logger.info(f"正在移动到目标位姿: pos={target_pos}, quat={target_quat}")
        
        env.step(action, wait_for_arrival=True, timeout=30)
        
        logger.info("到达目标位姿。")
        
        time.sleep(30)
        
        logger.info("测试完成，正在重置并退出...")
        env.reset()
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
    finally:
        pass

if __name__ == "__main__":
    test_specific_pose()
