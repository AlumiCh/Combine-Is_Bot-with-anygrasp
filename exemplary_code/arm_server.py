# Author: ZMAI
# Date: 2025-05-04
# Version: 1.1
#
# This RPC server allows other processes to communicate with the Kinova arm
# low-level controller, which runs in its own, dedicated real-time process.
#
# Note: Operations that are not time-sensitive should be run in a separate,
# non-real-time process to avoid interfering with the low-level control and
# causing latency spikes.

import queue
import time
import logging
from multiprocessing.managers import BaseManager as MPBaseManager
import numpy as np
from robot_controller.gen3.arm_controller import JointCompliantController
from robot_controller.gen3.kinova import TorqueControlledArm
import os, sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(parent_dir)
sys.path.append(parent_dir)
from configs.constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from robot_controller.ik_solver import IKSolver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Arm:
    def __init__(self):
        self.arm = TorqueControlledArm()
        self.arm.set_joint_limits(
            speed_limits=(7 * (30,)), acceleration_limits=(7 * (80,))
        )
        self.command_queue = queue.Queue(1)
        self.controller = None
        self.ik_solver = IKSolver(ee_offset=0.12)
        self.target_qpos = None # 目标关节角

    def reset(self):
        # Stop low-level control
        if self.arm.cyclic_running:
            time.sleep(0.75)  # Wait for arm to stop moving
            self.arm.stop_cyclic()

        # Clear faults
        self.arm.clear_faults()

        # Reset arm configuration
        self.arm.open_gripper()
        self.arm.retract()

        # Create new instance of controller
        self.controller = JointCompliantController(self.command_queue)

        # Start low-level control
        self.arm.init_cyclic(self.controller.control_callback)
        while not self.arm.cyclic_running:
            time.sleep(0.01)

    def execute_action(self, action):
        qpos = self.ik_solver.solve(action["arm_pos"], action["arm_quat"], self.arm.q)

        # 获取逆运动学求解器得到的关节角
        self.target_qpos = qpos

        self.command_queue.put((qpos, action["gripper_pos"].item()))

    def get_target_qpos(self):
        """
        获取逆运动学求解器得到的关节角

        Returns:
            target_qpos: 目标关节角
        """

        try:
            if self.target_qpos is None:
                logger.error("\n目标关节角为空\n")
            else:
                return self.target_qpos
        except Exception as e:
                logger.error(f'未正确获取目标关节角: {e}')
                raise

    def get_state(self):
        arm_pos, arm_quat = self.arm.get_tool_pose()
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)
        state = {
            "arm_pos": arm_pos,
            "arm_quat": arm_quat,
            "gripper_pos": np.array([self.arm.gripper_pos])
        }
        return state

    def close(self):
        if self.arm.cyclic_running:
            time.sleep(0.75)  # Wait for arm to stop moving
            self.arm.stop_cyclic()
        self.arm.disconnect()


class ArmManager(MPBaseManager):
    pass


ArmManager.register("Arm", Arm)

if __name__ == "__main__":
    # # manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    # # server = manager.get_server()
    # # print(f'Arm manager server started at {ARM_RPC_HOST}:{ARM_RPC_PORT}')
    # # server.serve_forever()

    # import numpy as np
    # from configs.constants import POLICY_CONTROL_PERIOD

    # # manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    # # manager.connect()
    # # arm = manager.Arm()
    # # try:
    # #     arm.reset()
    # #     for i in range(50):
    # #         arm.execute_action({
    # #             'arm_pos': np.array([0.135, 0.002, 0.211]),
    # #             'arm_quat': np.array([0.706, 0.707, 0.029, 0.029]),
    # #             'gripper_pos': np.zeros(1),
    # #         })
    # #         print(arm.get_state())
    # #         time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    # # finally:
    # #     arm.close()

    # arm = Arm()
    # arm.reset()

    # input("Enter")

    # for i in range(50):
    #     arm.execute_action(
    #         {
    #             "arm_pos": np.array([0.135, 0.002, 0.211]),
    #             "arm_quat": np.array([0.706, 0.707, 0.029, 0.029]),
    #             "gripper_pos": np.zeros(1),
    #         }
    #     )
    #     print(arm.get_state())
    #     time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise

    manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(f'Arm manager server started at {ARM_RPC_HOST}:{ARM_RPC_PORT}')
    server.serve_forever()