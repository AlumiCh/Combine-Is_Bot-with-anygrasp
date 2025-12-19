import numpy as np
import time
import logging
from policies.grasp_policy import GraspPolicy
from robot_controller.ik_solver import IKSolver
from real_env import RealEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 相机到基座的齐次变换矩阵
transform_c2b = np.array([ # 暂未测量，先用单位矩阵代替
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# 创建 IK 求解器实例
ik_solver = IKSolver(ee_offset=0.12) # 该值来源于arm_server.py

# 创建 GraspPolicy 实例
logger.info("初始化 GraspPolicy...")
policy = GraspPolicy(
    camera_to_base_transform=transform_c2b,
    ik_solver=ik_solver,
    anygrasp_host='localhost',
    anygrasp_port=50000,
    anygrasp_authkey=b'anygrasp'
)

# 创建 RealEnv 实例
logger.info("初始化 RealEnv...")
env = RealEnv()
env.reset()

# 重置策略
policy.reset()

# 创建观测字典
obs = env.get_obs()

logger.info("\n开始测试抓取检测...")

# 执行策略步骤
num_steps = 10 # 执行的步数
for i in range(num_steps):
    logger.info(f"步骤 {i + 1}:")
    
    # 执行一次抓取
    action = policy.step(obs)
    
    # 根据返回值类型进行不同处理
    if action is None:
        state = policy.get_state()
        logger.info(f"当前状态为 {state} ,继续")
        continue
    elif action == 'end_episode':
        state = policy.get_state()
        logger.info(f"当前状态为 {state} ,任务结束")
        break
    elif action == 'reset_env':
        state = policy.get_state()
        logger.info(f"当前状态为 {state} ,需要重置环境")
        break
    else:
        state = policy.get_state()
        logger.info(f"当前状态为 {state},将执行动作：\narm_pos={action['arm_pos']}\narm_quat={action['arm_quat']}\ngripper_pos={action['gripper_pos']}")
    
    time.sleep(0.1)

logger.info("测试完成！")

env.close()