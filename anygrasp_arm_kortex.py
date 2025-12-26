"""
基于 Kortex API (High-Level) 的 AnyGrasp 抓取控制器
解决定位精度和关节角无法到达的问题。

功能：
1. 使用 BaseClient 发送 High-Level Action (ReachPose/ReachJointAngles)
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

# Kortex API 导入
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2

# 导入工具类 (假设 utilities 在上级或同级目录，如果找不到请调整路径)
# 这里我们直接实现连接逻辑，避免依赖 utilities
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager

# 导入策略
from policies.grasp_policy import GraspPolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 连接参数 (请根据实际情况修改)
ROBOT_IP = "192.168.1.10" # 默认 IP，请修改
USERNAME = "admin"
PASSWORD = "admin"

TIMEOUT_DURATION = 20

class KortexArmClient:
    def __init__(self, ip, username, password):
        self.transport = TCPTransport()
        self.router = RouterClient(self.transport, self.router.create_error_callback())
        self.transport.connect(ip, 10000)

        self.session_manager = SessionManager(self.router)
        self.session_info = Base_pb2.SessionCreateInfo()
        self.session_info.username = username
        self.session_info.password = password
        self.session_info.session_inactivity_timeout = 60000   # (milliseconds)
        self.session_info.connection_type = Base_pb2.TCP      # (0=TCP)

        self.session_handle = self.session_manager.CreateSession(self.session_info)

        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)

    def clean_up(self):
        if self.session_manager is not None:
            self.session_manager.CloseSession()
        if self.transport is not None:
            self.transport.disconnect()

    def check_for_end_or_abort(self, e):
        def check(notification, e=e):
            # print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
        return check

    def move_to_home(self):
        logger.info("Moving to Home position...")
        # 确保是单层伺服模式
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle

        if action_handle is None:
            logger.warning("Can't find 'Home' action. Using default angles.")
            # 如果没有 Home 动作，可以定义一个默认角度
            return False

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        self.base.ExecuteActionFromReference(action_handle)
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)
        return finished

    def send_gripper_command(self, value):
        """
        控制夹爪
        value: 0.0 (Open) to 1.0 (Close)
        """
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = value
        self.base.SendGripperCommand(gripper_command)
        time.sleep(1.0) # 等待夹爪动作

    def reach_cartesian_pose(self, target_pos, target_quat):
        """
        使用 ReachPose Action 移动到笛卡尔坐标
        target_pos: [x, y, z] (meters)
        target_quat: [x, y, z, w] (scipy/numpy format) -> 需要转换为 Euler Angles 或直接使用 Pose
        注意：Kinova Action 的 ReachPose 通常使用 Euler Angles (Theta X, Y, Z) 或者 Quaternion
        但在提供的示例 01 中使用的是 Theta X,Y,Z。
        为了精确控制姿态，我们最好将四元数转换为欧拉角，或者查看是否支持四元数。
        Base_pb2.CartesianPose 定义包含 theta_x, theta_y, theta_z (degrees)。
        """
        
        action = Base_pb2.Action()
        action.name = "Grasp Pose"
        action.application_data = ""

        # 转换四元数到欧拉角 (XYZ 顺序, degrees)
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(target_quat) # input [x, y, z, w]
        euler_angles = r.as_euler('xyz', degrees=True)

        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = target_pos[0]
        cartesian_pose.y = target_pos[1]
        cartesian_pose.z = target_pos[2]
        cartesian_pose.theta_x = euler_angles[0]
        cartesian_pose.theta_y = euler_angles[1]
        cartesian_pose.theta_z = euler_angles[2]

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        logger.info(f"Executing Cartesian Move: Pos={target_pos}, Euler={euler_angles}")
        self.base.ExecuteAction(action)

        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)
        
        if finished:
            logger.info("Movement completed.")
        else:
            logger.warning("Movement timed out.")
        return finished

    def get_state(self):
        """获取当前状态，适配 GraspPolicy 的接口"""
        feedback = self.base_cyclic.RefreshFeedback()
        
        # 位置
        arm_pos = np.array([
            feedback.base.tool_pose_x,
            feedback.base.tool_pose_y,
            feedback.base.tool_pose_z
        ])

        # 姿态 (Euler -> Quat)
        # Kinova 反馈的是 Theta X, Y, Z (degrees)
        r = R.from_euler('xyz', [
            feedback.base.tool_pose_theta_x,
            feedback.base.tool_pose_theta_y,
            feedback.base.tool_pose_theta_z
        ], degrees=True)
        arm_quat = r.as_quat() # [x, y, z, w]

        # 关节角
        arm_qpos = np.array([a.position for a in feedback.actuators]) # degrees
        arm_qpos = np.deg2rad(arm_qpos) # convert to rad for policy

        # 夹爪
        gripper_pos = np.array([feedback.interconnect.gripper_feedback.motor[0].position / 100.0]) # 0-100 -> 0-1

        return {
            'arm_pos': arm_pos,
            'arm_quat': arm_quat,
            'arm_qpos': arm_qpos,
            'gripper_pos': gripper_pos
        }

# 引入 scipy 用于旋转转换
from scipy.spatial.transform import Rotation as R

class GraspControllerKortex:
    def __init__(self, robot_ip='192.168.1.10'):
        self.robot = KortexArmClient(robot_ip, "admin", "admin")
        
        # 初始化抓取策略
        # 注意：这里不需要 IKSolver 了，因为我们直接发笛卡尔指令给机械臂
        camera_to_base = np.array([ 
                [0, 1, 0, 0.57],
                [1, 0, 0, 0.09],
                [0, 0, -1, 1.01],
                [0, 0, 0, 1]
            ])
        
        self.policy = GraspPolicy(
            camera_to_base_transform=camera_to_base,
            ik_solver=None # 传入 None，因为我们将在 execute 中处理运动
        )

    def run_episode(self):
        logger.info("Resetting robot...")
        self.robot.move_to_home()
        self.robot.send_gripper_command(0.0) # Open
        self.policy.reset()
        
        step_count = 0
        max_steps = 20 # 减少步数，因为 High-Level 动作更准确，不需要微调循环
        
        while step_count < max_steps:
            obs = self.robot.get_state()
            
            # 策略决策
            # 注意：GraspPolicy 内部可能依赖 ik_solver 来计算某些中间状态
            # 如果 GraspPolicy 强依赖 ik_solver，我们可能需要保留它仅用于计算，
            # 但实际执行使用 robot.reach_cartesian_pose
            
            # 这里假设 GraspPolicy.step 返回的是目标笛卡尔位姿
            action = self.policy.step(obs)
            
            if action == 'end_episode':
                logger.info("Episode finished.")
                break
            
            if action is None:
                time.sleep(0.1)
                continue
            
            step_count += 1
            logger.info(f"Step {step_count}")
            
            # 执行动作
            # 1. 移动手臂
            target_pos = action['arm_pos']
            target_quat = action['arm_quat']
            self.robot.reach_cartesian_pose(target_pos, target_quat)
            
            # 2. 控制夹爪
            target_gripper = action['gripper_pos'][0]
            # 简单的阈值判断：>0.5 关，<0.5 开
            # 或者根据策略的具体输出调整
            # 假设策略输出 1.0 为关，0.0 为开
            # Kinova GripperCommand: 0.0 (Open) -> 1.0 (Close) ? 
            # 实际上 Kinova API 中 GripperCommand value 通常是 0.0 - 1.0
            # 需要确认 GraspPolicy 的输出定义。假设 1=Close, 0=Open
            self.robot.send_gripper_command(target_gripper)
            
            time.sleep(0.5)

    def close(self):
        self.robot.clean_up()

if __name__ == "__main__":
    # 解析命令行参数获取 IP
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.1.10", help="Robot IP address")
    args = parser.parse_args()

    controller = GraspControllerKortex(args.ip)
    try:
        controller.run_episode()
    except KeyboardInterrupt:
        pass
    finally:
        controller.close()
