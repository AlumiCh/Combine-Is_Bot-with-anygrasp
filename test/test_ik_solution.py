import numpy as np
import mujoco
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from robot_controller.ik_solver import IKSolver
from scipy.spatial.transform import Rotation as R


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def forward_kinematics(ik_solver, qpos):
    """
    使用 IKSolver 的 MuJoCo 模型计算正运动学
    
    Args:
        ik_solver: IKSolver 实例
        qpos: 关节角度数组 (7,)
    
    Returns:
        pos: 末端位置 (3,)
        quat: 末端四元数 (4,) [x, y, z, w]
    """
    # 设置关节角度
    ik_solver.data.qpos[:] = qpos
    
    # 更新运动学
    mujoco.mj_kinematics(ik_solver.model, ik_solver.data)
    mujoco.mj_comPos(ik_solver.model, ik_solver.data)
    
    # 读取末端位置
    pos = ik_solver.site_pos.copy()
    
    # 读取末端姿态（旋转矩阵转四元数）
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, ik_solver.site_mat)
    
    # 转换四元数格式：MuJoCo (w,x,y,z) -> scipy (x,y,z,w)
    quat = quat[[1, 2, 3, 0]]
    
    return pos, quat


# ===== 验证代码 =====
def verify_ik_solution(ik_solver, target_pos, target_qpos, actual_qpos):
    """
    对比 IK 求解的准确性
    
    Args:
        ik_solver: IKSolver 实例
        target_pos: 目标末端位置
        target_qpos: IK 求解出的目标关节角
        actual_qpos: 机械臂实际关节角
    """

    target_pos = np.array(target_pos)
    target_qpos = np.array(target_qpos)
    actual_qpos = np.array(actual_qpos)

    logger.info("正运动学验证")
    
    # 计算目标关节角的正运动学
    fk_pos_target, fk_quat_target = forward_kinematics(ik_solver, target_qpos)
    fk_euler_target = R.from_quat(fk_quat_target).as_euler('xyz', degrees=True)
    
    logger.info("\n[目标关节角 -> 正运动学]")
    logger.info(f"  关节角度(度): {np.degrees(target_qpos).round(2)}")
    logger.info(f"  FK 位置: {fk_pos_target}")
    logger.info(f"  FK 欧拉角(度): {fk_euler_target.round(2)}")
    
    # 计算实际关节角的正运动学
    fk_pos_actual, fk_quat_actual = forward_kinematics(ik_solver, actual_qpos)
    fk_euler_actual = R.from_quat(fk_quat_actual).as_euler('xyz', degrees=True)
    
    logger.info("\n[实际关节角 -> 正运动学]")
    logger.info(f"  关节角度(度): {np.degrees(actual_qpos).round(2)}")
    logger.info(f"  FK 位置: {fk_pos_actual}")
    logger.info(f"  FK 欧拉角(度): {fk_euler_actual.round(2)}")
    
    # 对比分析
    logger.info("\n[误差分析]")
    
    # IK 求解精度
    ik_pos_error = np.linalg.norm(target_pos - fk_pos_target)
    
    logger.info("\nIK 求解精度:")
    logger.info(f"  期望位置: {target_pos}")
    logger.info(f"  FK 位置:   {fk_pos_target}")
    logger.info(f"  位置误差: {ik_pos_error*1000:.2f} mm")
    logger.info(f"  FK 欧拉角(度):   {fk_euler_target.round(2)}")
    
    # 关节跟踪误差（目标关节角 vs 实际关节角）
    joint_error = target_qpos - actual_qpos
    
    logger.info("\n关节跟踪误差:")
    logger.info(f"  各关节偏差(度): {np.degrees(joint_error).round(2)}")
    logger.info(f"  最大偏差: {np.degrees(np.abs(joint_error).max()):.2f}° (关节{np.abs(joint_error).argmax()})")
    
    # 最终末端位置误差
    final_pos_error = np.linalg.norm(fk_pos_target - fk_pos_actual)
    
    logger.info(f"\n末端位置误差:")
    logger.info(f"  位置误差: {final_pos_error*100:.2f} cm")
    
if __name__ == '__main__':
    ik_solver = IKSolver(ee_offset=0.12)
    
    # 目标位置
    target_pos = [0.346, 0.206, 0.076]
    
    # 关节角度（输入为角度，转换为弧度进行计算）
    target_qpos_deg = [347.771,  44.72,  177.699, 258.689, 332.576, 303.922,  55.96]
    actual_qpos_deg = [350.175,  45.221, 179.356, 256.785, 335.564, 307.631,  59.674]
    
    verify_ik_solution(ik_solver, target_pos, 
                       np.radians(target_qpos_deg), 
                       np.radians(actual_qpos_deg))