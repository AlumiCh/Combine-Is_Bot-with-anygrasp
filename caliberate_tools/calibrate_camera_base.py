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

if __name__ == "__main__":
    # --- 请在此处输入你记录的数据 ---
    # 建议至少取 4 个点，点分布越广、不共面，效果越好
    
    # P_camera: 在相机点云中看到的点的坐标 (x, y, z)
    points_camera = np.array([
        [0.10, -0.20, 0.95], 
        [0.30, -0.20, 0.96],
        [0.10,  0.10, 0.94],
        [0.25,  0.05, 0.85]
    ])
    
    # P_base: 机器人移动到对应位置处，指尖接触点的底座坐标 (x, y, z)
    # 注意：如果你之前算出夹爪长度偏移是 0.13m，那么 P_base 应该是 机器人法兰坐标 + 0.13m*接近方向
    points_base = np.array([
        [0.55, 0.12, 0.02],
        [0.56, -0.08, 0.01],
        [0.42, 0.11, 0.03],
        [0.48, -0.02, 0.12]
    ])
    
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
