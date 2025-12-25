"""
直接驱动机械臂进行 AnyGrasp 抓取任务

功能：
1. 直接控制 Kinova Gen3 机械臂
2. 通过 RPC 连接 AnyGrasp 服务器进行抓取检测
3. 执行完整的 抓取 -> 提升 -> 放置 流程
"""

import time
import numpy as np
import logging

# 导入必要的模块
from exemplary_code.arm_policy import Arm
from exemplary_code.ik_solver import IKSolver
from policies.grasp_policy import GraspPolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraspController:
    def __init__(self, anygrasp_host='localhost', anygrasp_port=50000):
        logger.info("正在初始化直接抓取控制器...")
        
        # 初始化机械臂
        logger.info("\n连接机械臂...\n")
        self.arm = Arm()
        self.arm.reset()
        
        # 初始化 IK 求解器
        self.ik_solver = IKSolver(ee_offset=0.12)
        
        # 初始化抓取策略
        camera_to_base = np.array([ 
                [0, 1, 0, 0.57],
                [1, 0, 0, 0.09],
                [0, 0, -1, 1.01],
                [0, 0, 0, 1]
            ])
        
        logger.info(f"\n连接 AnyGrasp 服务器 ({anygrasp_host}:{anygrasp_port})...\n")
        self.policy = GraspPolicy(
            camera_to_base_transform=camera_to_base,
            ik_solver=self.ik_solver,
        )
        
    def get_obs(self):
        """
        获取当前观测。

        Returns:
            dict: 当前机械臂的观测字典，包含末端位置、姿态、关节角度和夹爪状态。
        """

        state = self.arm.get_state()
        
        # 构造观测字典，与 GraspPolicy 兼容
        obs = {
            'arm_pos': state['arm_gpos'],  # 末端位置
            'arm_quat': state['arm_quat'], # 末端姿态 [x, y, z, w]
            'arm_qpos': np.deg2rad(state['arm_qpos']), # 关节角度
            'gripper_pos': state['gripper_pos'] # 夹爪状态
        }
        return obs

    def run_episode(self):
        """
        执行一次完整的抓取流程。

        包括机械臂复位、策略重置、循环决策与动作执行，直到任务完成或超时。
        """

        logger.info("\n开始新的抓取任务...\n")
        
        # 机械臂复位
        logger.info("\n机械臂复位到初始位置...\n")
        self.arm.reset()
        time.sleep(1.0)
        
        # 重置策略
        self.policy.reset()
        
        # 获取初始观测
        obs = self.get_obs()
        
        step_count = 0
        max_steps = 100
        
        while step_count < max_steps:
            # 策略决策
            action = self.policy.step(obs)
            
            # 处理结束信号
            if action == 'end_episode':
                logger.info("\n抓取任务完成\n")
                break
            
            if action is None:
                # 策略可能在等待或处理中
                time.sleep(0.1)
                continue
            
            # 执行动作
            step_count += 1
            logger.info(f"执行步骤 {step_count}")
            
            # 记录目标位置用于验证
            target_pos = action['arm_pos']

            # 求解出目标关节角度(此处这样处理而不是使用 exucute_action_g 方法是为了避免多次解逆)
            qpos = self.ik_solver.solve(action["arm_pos"], action["arm_quat"], np.deg2rad(self.arm.get_state()['arm_qpos']))
            action_q = {
                'arm_pos': np.rad2deg(qpos),
                'gripper_pos': action['gripper_pos']
            }
            
            # 执行运动
            start_time = time.time()
            while True:
                self.arm.execute_action_q(action_q)
                time.sleep(0.05) # 控制机械臂运动速度
                
                # 计算位置误差
                curr_pos = self.arm.get_state()['arm_gpos']
                error = np.linalg.norm(curr_pos - target_pos)
                
                pos_error_threshold = 0.02 # 位置误差阈值 (m)
                if error < pos_error_threshold:
                    logger.info(f"\n已到达目标，位置误差: {error*1000:.2f}mm\n")
                    break

                time_threshold = 10 # 超时阈值 (s)
                if time.time() - start_time > time_threshold:
                    logger.warning(f"\n移动超时，最终位置误差：{error*1000:.2f}mm\n")
                    break
            
            # 更新观测
            obs = self.get_obs()

    def close(self):
        """
        清理资源。

        关闭机械臂连接，释放相关资源。
        """

        logger.info("\n关闭连接...\n")
        self.arm.close()

if __name__ == "__main__":
    controller = None
    try:
        controller = GraspController()
    except KeyboardInterrupt:
        logger.info("\n用户中断任务\n")
    except Exception as e:
        logger.error(f"\n发生错误: {e}\n", exc_info=True)
    finally:
        if controller:
            controller.close()
