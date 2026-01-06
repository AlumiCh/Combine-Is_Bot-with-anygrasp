import cv2
import numpy as np
import open3d as o3d
import time
import sys
import os

# 将根目录添加到路径以便导入 cameras
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cameras import RealSenseCamera

def get_clicked_points_3d():
    # 1. 初始化 RealSense 相机
    print("正在初始化 RealSense D435i...")
    camera = RealSenseCamera()

    # 先获取一帧图像
    rgb_image, depth_image = camera.get_rgb_depth()
    
    # 等待相机稳定
    time.sleep(1)

    # 再获取一帧，保证图像对齐无误
    rgb_image, depth_image = camera.get_rgb_depth()
    
    if rgb_image is None or depth_image is None:
        print("错误：无法获取相机数据。")
        camera.close()
        return []

    # BGR 用于 OpenCV 显示
    display_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    # 获取内参
    intr = camera.get_intrinsics()
    fx, fy = intr['fx'], intr['fy']
    cx, cy = intr['cx'], intr['cy']
    
    clicked_pixels = []

    # 鼠标回调函数
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_pixels.append((x, y))
            # 在图上画个圈标记一下
            cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display_image, str(len(clicked_pixels)), (x+10, y+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Select Calibration Points (RealSense)", display_image)
            print(f"已选择第 {len(clicked_pixels)} 个点: 像素坐标 ({x}, {y})")

    # 显示图片并等待点击
    print("\n操作说明:")
    print("1. 在弹出窗口中，用鼠标左键顺次点击标定点。")
    print("2. 点击完成后，在图片窗口按下 'q' 键退出并计算坐标。")
    
    cv2.namedWindow("Select Calibration Points (RealSense)")
    cv2.setMouseCallback("Select Calibration Points (RealSense)", on_mouse)
    cv2.imshow("Select Calibration Points (RealSense)", display_image)
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

    points_3d = []

    print("\n=== 选定点的空间 3D 坐标 (相机坐标系) ===")
    print("请按照此顺序在标定脚本中使用 (单位: 米)：")
    
    for i, (px, py) in enumerate(clicked_pixels):
        # depth_image 已经是单位为米的对齐深度图
        z = depth_image[py, px] 
        
        if z <= 0:
            print(f"警告：点 {i+1} ({px}, {py}) 深度值为 {z} (无效)，请尝试点击点云更完整的区域。")
            continue
            
        # 使用内参将像素坐标转换为 3D 坐标
        # 公式: x = (u - cx) * z / fx, y = (v - cy) * z / fy
        x = (px - cx) * z / fx
        y = (py - cy) * z / fy
        
        pos_3d_m = [x, y, z]
        points_3d.append(pos_3d_m)
        
        print(f"点 {i+1}: [{pos_3d_m[0]:.6f}, {pos_3d_m[1]:.6f}, {pos_3d_m[2]:.6f}]")

    camera.close()
    return points_3d

if __name__ == "__main__":
    get_clicked_points_3d()
