import numpy as np

def solve_transformation(P_camera, P_base):
    """
    使用 Kabsch 算法求解 P_base = R @ P_camera + t
    P_camera, P_base: (N, 3) 的 numpy 数组
    """
    assert P_camera.shape == P_base.shape
    
    # 1. 计算质心
    centroid_camera = np.mean(P_camera, axis=0)
    centroid_base = np.mean(P_base, axis=0)
    
    # 2. 去质心
    q_camera = P_camera - centroid_camera
    q_base = P_base - centroid_base
    
    # 3. 计算协方差矩阵 H
    H = q_camera.T @ q_base
    
    # 4. SVD 分解
    U, S, Vt = np.linalg.svd(H)
    
    # 5. 计算旋转矩阵 R
    R = Vt.T @ U.T
    
    # 处理反射情况（确保是旋转矩阵而非反射矩阵）
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    # 6. 计算平移向量 t
    t = centroid_base - R @ centroid_camera
    
    # 7. 构建 4x4 齐次变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T

def flange_to_tool(positions, rotations_deg, offset=0.13):
    """
    根据法兰坐标和姿态计算工具末端(接触点)坐标
    positions: (N, 3) 法兰位置 [x, y, z]
    rotations_deg: (N, 3) 法兰姿态 [ThetaX, ThetaY, ThetaZ] (单位: 度)
    offset: 偏移长度，默认 0.13m
    """
    from scipy.spatial.transform import Rotation as R
    tool_points = []
    for pos, orient in zip(positions, rotations_deg):
        # Kinova Gen3 Web UI 显示的是 Euler XYZ 角度 (Intrinsic)
        rot_matrix = R.from_euler('xyz', orient, degrees=True).as_matrix()
        # 在 Kinova/GraspGen 坐标系中，Z 轴 (第2列) 是接近方向
        approach_direction = rot_matrix[:, 2]
        # 计算接触点坐标
        tool_pos = pos + offset * approach_direction
        tool_points.append(tool_pos)
    return np.array(tool_points)

if __name__ == "__main__":
    # --- 1. 相机数据 (从 get_click_points_3d.py 中获取) ---
    points_camera = np.array([
        [0.106, -0.279, 0.808], 
        [0.306, -0.279, 0.815],
        [0.106,  0.102, 0.810],
        [0.256,  0.052, 0.812]
    ])
    
    # --- 2. 机械臂数据 (从 Kinova Web UI 或 API 读取) ---
    # 法兰中心位置 [x, y, z] (米)
    flange_pos = np.array([
        [0.450, 0.120, 0.150],
        [0.460, -0.080, 0.155],
        [0.320, 0.110, 0.148],
        [0.380, -0.020, 0.152]
    ])
    # 法兰姿态 [ThetaX, ThetaY, ThetaZ] (度)
    flange_ori = np.array([
        [180.0, 0.0, 90.0],
        [180.0, 0.0, 90.0],
        [180.0, 0.0, 90.0],
        [180.0, 0.0, 90.0]
    ])

    # 自动计算工具末端坐标 (P_base)
    # 逻辑：法兰位置 + 0.13m * 接近方向向量
    points_base = flange_to_tool(flange_pos, flange_ori, offset=0.13)
    
    print("\n[计算] 得到的工具末端 (接触点) 坐标:")
    for i, p in enumerate(points_base):
        print(f"  点 {i+1}: {p}")

    # 执行 Kabsch 算法求解变换矩阵
    T_cam_to_base = solve_transformation(points_camera, points_base)
    
    print("\n=== 计算得到的相机到底座变换矩阵 (camera_to_base) ===")
    print("np.array(")
    print(np.array2string(T_cam_to_base, separator=', ', formatter={'float_kind':lambda x: "%.6f" % x}))
    print(")")
    
    # 验证误差
    errors = []
    for i in range(len(points_camera)):
        p_transformed = (T_cam_to_base[:3, :3] @ points_camera[i]) + T_cam_to_base[:3, 3]
        error = np.linalg.norm(p_transformed - points_base[i])
        errors.append(error)
    print(f"\n平均对齐误差: {np.mean(errors)*100:.2f} 厘米")
