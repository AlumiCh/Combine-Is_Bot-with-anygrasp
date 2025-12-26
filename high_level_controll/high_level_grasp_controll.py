"""
基于 Kortex API (High-Level) 的 AnyGrasp 抓取控制器
使用 utilities.py 进行连接管理

功能：
1. 使用 BaseClient 发送 High-Level Action (ReachPose)
2. 利用机械臂内部控制器保证毫米级定位精度
3. 结合 AnyGrasp 进行视觉抓取
"""

import sys
import os
import time
import threading
import numpy as np
import logging
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Kortex API
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2

# 本地工具类
import utilities

# 策略与数学工具
from policies.grasp_policy import GraspPolicy
from scipy.spatial.transform import Rotation as R

from robot_controller.ik_solver import IKSolver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TIMEOUT_DURATION = 60

class HighLevelGraspController:
    def __init__(self, router):
        """
        初始化高层抓取控制器。

        Args:
            router: Kortex API 路由器对象，用于建立 RPC 通信。
        """

        self.router = router

        # 初始化 BaseClient 用于发送高层控制指令（如 ReachPose, ExecuteAction）
        self.base = BaseClient(self.router)

        # 初始化 BaseCyclicClient 用于获取实时的机械臂状态反馈
        self.base_cyclic = BaseCyclicClient(self.router)
        
        # 初始化抓取策略
        camera_to_base = np.array([ 
                [0, 1, 0, 0.57],
                [1, 0, 0, 0.09],
                [0, 0, -1, 1.01],
                [0, 0, 0, 1]
            ])
        
        # 创建抓取策略实例
        # 传入变换矩阵和 IK 求解器（IK 仅用于策略内部的可达性预判）
        self.policy = GraspPolicy(
            camera_to_base_transform=camera_to_base,
            ik_solver=IKSolver(ee_offset=0.12) 
        )

    def check_for_end_or_abort(self, e):
        """
        生成通知回调函数，用于检测异步动作是否结束或被中止。

        Args:
            e (threading.Event): 线程事件对象，当动作结束时会被 set()。

        Returns:
            function: 符合 Kortex API 要求的通知回调函数。
        """

        def check(notification, e=e):
            # 检查通知事件是否为“动作结束”或“动作中止”
            if notification.action_event == Base_pb2.ACTION_END or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
        return check

    def move_to_retract(self):
        """
        控制机械臂移动到预设的 Retract 位置。

        Returns:
            bool: 如果在超时时间内成功到达则返回 True，否则返回 False。
        """

        logger.info("\n移动至 Retract 预设位置...\n")
        # 设置伺服模式为单级伺服（High-Level 控制必需）
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        # 从机械臂中读取所有预定义的动作列表
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        
        # 遍历列表寻找名为 "Retract" 的动作
        for action in action_list.action_list:
            if action.name == "Retract":
                action_handle = action.handle

        if action_handle is None:
            logger.warning("\n找不到 'Retract' 动作")
            return False

        # 使用事件机制等待异步动作完成
        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        # 执行引用的动作
        self.base.ExecuteActionFromReference(action_handle)
        # 等待动作完成或超时
        finished = e.wait(TIMEOUT_DURATION)
        # 取消订阅通知
        self.base.Unsubscribe(notification_handle)
        return finished

    def send_gripper_command(self, value):
        """
        发送夹爪控制指令。

        Args:
            value (float): 夹爪目标开合度，0.0 表示完全打开，1.0 表示完全闭合。
        """
        
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = value
        self.base.SendGripperCommand(gripper_command)
        # 给予夹爪一定的动作时间
        time.sleep(1.0) 

    def reach_cartesian_pose(self, target_pos, target_quat):
        """
        使用 ReachPose 动作将机械臂末端移动到指定的笛卡尔位姿。
        该方法利用机械臂内部的轨迹规划器，保证运动平滑。

        Args:
            target_pos (array-like): 目标位置 [x, y, z]，单位为米。
            target_quat (array-like): 目标姿态四元数 [x, y, z, w]。

        Returns:
            bool: 动作是否成功完成。
        """
        action = Base_pb2.Action()
        action.name = "Grasp Pose"
        action.application_data = ""

        # Kortex High-Level API 使用欧拉角 (ThetaX, ThetaY, ThetaZ) 描述姿态
        # 因此需要将四元数转换为欧拉角 (使用 xyz 顺序，单位为度)
        r = R.from_quat(target_quat) 
        euler_angles = r.as_euler('xyz', degrees=True)

        # 填充目标位姿数据
        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = target_pos[0]
        cartesian_pose.y = target_pos[1]
        cartesian_pose.z = target_pos[2]
        cartesian_pose.theta_x = euler_angles[0]
        cartesian_pose.theta_y = euler_angles[1]
        cartesian_pose.theta_z = euler_angles[2]

        # 订阅动作通知以监控执行状态
        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        logger.info(f"\n执行笛卡尔移动: 位置={target_pos}, 欧拉角={euler_angles}\n")
        # 发送动作指令
        self.base.ExecuteAction(action)

        # 等待完成
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)
        
        if finished:
            logger.info("移动完成。\n")            
            # 打印详细的运动分析信息
            try:
                feedback = self.base_cyclic.RefreshFeedback()
                current_pos = np.array([
                    feedback.base.tool_pose_x,
                    feedback.base.tool_pose_y,
                    feedback.base.tool_pose_z
                ])
                current_euler = np.array([
                    feedback.base.tool_pose_theta_x,
                    feedback.base.tool_pose_theta_y,
                    feedback.base.tool_pose_theta_z
                ])
                
                # 计算偏差
                pos_diff = np.linalg.norm(current_pos - target_pos)
                euler_diff = np.linalg.norm(current_euler - euler_angles)
                
                logger.info(f"目标位置(cm)  : {target_pos * 100}")
                logger.info(f"当前位置(cm)  : {current_pos * 100}")
                logger.info(f"位置偏差(cm)  : {pos_diff * 100:.4f}")
                logger.info(f"目标欧拉角(deg): {euler_angles}")
                logger.info(f"当前欧拉角(deg): {current_euler}")
                logger.info(f"欧拉角偏差(deg): {euler_diff:.4f}\n")
            except Exception as err:
                logger.warning(f"\n无法获取反馈信息以打印分析: {err}\n")
        else:
            logger.warning("移动超时。")
        return finished

    def get_state(self):
        """
        获取机械臂当前状态，并将其格式化为 GraspPolicy 所需的观测字典。

        Returns:
            dict: 包含末端位置、姿态、关节角和夹爪状态的字典。
        """

        # 获取最新的循环反馈数据
        feedback = self.base_cyclic.RefreshFeedback()
        
        # 提取末端位置 (x, y, z)
        arm_pos = np.array([
            feedback.base.tool_pose_x,
            feedback.base.tool_pose_y,
            feedback.base.tool_pose_z
        ])

        # 提取末端姿态并从欧拉角转换为四元数 [x, y, z, w]
        r = R.from_euler('xyz', [
            feedback.base.tool_pose_theta_x,
            feedback.base.tool_pose_theta_y,
            feedback.base.tool_pose_theta_z
        ], degrees=True)
        arm_quat = r.as_quat() 

        # 提取各关节角度并从度转换为弧度
        arm_qpos = np.array([a.position for a in feedback.actuators]) 
        arm_qpos = np.deg2rad(arm_qpos) 

        # 提取夹爪位置 (归一化到 0.0-1.0)
        gripper_pos = np.array([feedback.interconnect.gripper_feedback.motor[0].position / 100.0]) 

        return {
            'arm_pos': arm_pos,
            'arm_quat': arm_quat,
            'arm_qpos': arm_qpos,
            'gripper_pos': gripper_pos
        }

    def run_episode(self):
        """
        执行一个完整的抓取任务周期。
        流程：复位 -> 开启夹爪 -> 循环检测与移动 -> 抓取 -> 结束。
        """

        logger.info("\n正在重置机器人...\n")
        # 移动到初始位置并打开夹爪
        self.move_to_retract()
        self.send_gripper_command(0.0) 
        # 重置策略内部状态
        self.policy.reset()
        
        step_count = 0
        max_steps = 20
        
        while step_count < max_steps:
            # 获取当前环境观测
            obs = self.get_state()
            
            # 获取下一步动作
            action = self.policy.step(obs)
            
            # 检查是否收到结束信号
            if action == 'end_episode':
                logger.info("\n任务执行完毕\n")
                break
            
            if action is None:
                time.sleep(0.1)
                continue
            
            step_count += 1
            logger.info(f"执行第 {step_count} 步")
            
            # 执行移动动作
            target_pos = action['arm_pos']
            target_quat = action['arm_quat']
            self.reach_cartesian_pose(target_pos, target_quat)
            
            # 执行夹爪动作
            target_gripper = action['gripper_pos']
            # 将策略输出的连续值映射为二值控制指令
            cmd_val = 1.0 if target_gripper > 0.5 else 0.0
            self.send_gripper_command(cmd_val)
            
            # 动作间的短暂停顿
            time.sleep(0.5)

def main():
    """
    主程序入口，解析参数并运行高层抓取控制器。
    """

    args = utilities.parseConnectionArguments()
    
    # 创建连接并运行
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        controller = HighLevelGraspController(router)
        try:
            controller.run_episode()
        except KeyboardInterrupt:
            logger.info("User interrupted.")
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
