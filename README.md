    def solve_basic(self, pos, quat, curr_qpos, max_iters=50, err_thresh=1e-6):
        """
        基础版 IK 求解器：
        - 仅使用雅可比伪逆
        - 不包含阻尼（不规避奇异点）
        - 不包含零空间任务（不考虑参考姿态）
        - 追求高精度迭代
        """
        # 1. 初始化关节位置
        self.data.qpos = curr_qpos

        for _ in range(max_iters):
            # 2. 正向运动学更新：计算当前末端位姿
            mujoco.mj_kinematics(self.model, self.data)

            # 3. 计算 6 维误差 (位置 + 旋转)
            # 平移误差
            self.err_pos[:] = pos - self.site_pos
            # 旋转误差 (四元数差转角速度向量)
            mujoco.mju_mat2Quat(self.site_quat, self.site_mat)
            mujoco.mju_negQuat(self.site_quat_inv, self.site_quat)
            mujoco.mju_mulQuat(self.err_quat, quat, self.site_quat_inv)
            mujoco.mju_quat2Vel(self.err_rot, self.err_quat, 1.0)

            # 4. 精度检查：如果误差足够小，直接返回
            if np.linalg.norm(self.err) < err_thresh:
                break

            # 5. 获取当前雅可比矩阵 (6 x 关节数)
            mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, self.site_id)

            # 6. 核心计算：使用 Moore-Penrose 伪逆直接求解关节增量
            # update = J⁺ * err
            update = np.linalg.pinv(self.jac) @ self.err

            # 7. 更新关节位置
            mujoco.mj_integratePos(self.model, self.data.qpos, update, 1.0)

        return self.data.qpos.copy()