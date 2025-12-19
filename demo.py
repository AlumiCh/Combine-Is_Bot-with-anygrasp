import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import cv2
import time
import sys

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

def project_point_to_image(point_3d, fx, fy, cx, cy):
    """将 3D 点投影到 2D 图像坐标"""
    if point_3d[2] <= 0:
        return None
    pixel_x = int(fx * point_3d[0] / point_3d[2] + cx)
    pixel_y = int(fy * point_3d[1] / point_3d[2] + cy)
    return np.array([pixel_x, pixel_y])

def draw_o3d_gripper_on_image(image, vertices, edges, color, fx, fy, cx, cy):
    """
    将 O3D 夹爪的顶点和边投影到图像上并绘制
    
    Args:
        image: 输入图像
        vertices: 夹爪顶点 (N, 3) 的 numpy 数组
        edges: 边的连接关系 (M, 2) 的列表，每行是 [v1_idx, v2_idx]
        color: BGR 颜色元组
        fx, fy, cx, cy: 相机内参
    """
    image_copy = image.copy()
    
    # 投影所有顶点
    projected_vertices = []
    for vertex in vertices:
        proj = project_point_to_image(vertex, fx, fy, cx, cy)
        if proj is not None and 0 <= proj[0] < image.shape[1] and 0 <= proj[1] < image.shape[0]:
            projected_vertices.append(proj)
        else:
            projected_vertices.append(None)
    
    # 绘制边
    if edges is not None:
        for edge in edges:
            v1_idx, v2_idx = edge
            if v1_idx < len(projected_vertices) and v2_idx < len(projected_vertices):
                proj1 = projected_vertices[v1_idx]
                proj2 = projected_vertices[v2_idx]
                if proj1 is not None and proj2 is not None:
                    cv2.line(image_copy, tuple(proj1), tuple(proj2), color, 2)
    
    # 绘制顶点
    for proj in projected_vertices:
        if proj is not None:
            cv2.circle(image_copy, tuple(proj), 3, color, -1)
    
    return image_copy

def draw_grasps_on_image(rgb_image, grasps, camera_intrinsics, max_grasps=5):
    """
    在 RGB 图像上绘制抓取
    
    Args:
        rgb_image: [H, W, 3] RGB 图像 (uint8)
        grasps: GraspGroup 对象，包含检测到的抓取
        camera_intrinsics: 相机内参字典 {fx, fy, cx, cy}
        max_grasps: 最多绘制多少个抓取
    
    Returns:
        绘制了抓取的 RGB 图像
    """
    image_with_grasps = rgb_image.copy()
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']
    
    # 只绘制前 max_grasps 个最优抓取
    for i, grasp in enumerate(grasps[:max_grasps]):
        # 获取抓取的位置（3D 点）
        grasp_pos = grasp.translation  # [x, y, z]
        
        # 投影到图像平面
        if grasp_pos[2] > 0:  # 检查是否在相机前方
            # 颜色：根据抓取得分（得分越高越绿）
            score_normalized = min(grasp.score, 1.0)
            color_b = int(255 * (1 - score_normalized))
            color_g = int(255 * score_normalized)
            color_r = 100
            color = (color_b, color_g, color_r)
            
            # 使用 O3D 夹爪几何体投影
            try:
                # 获取 O3D 夹爪几何体
                gripper_geometry = grasp.to_open3d_geometry()
                
                # 获取顶点
                vertices = np.asarray(gripper_geometry.vertices)
                
                # 获取三角形网格或线
                if hasattr(gripper_geometry, 'triangles'):
                    triangles = np.asarray(gripper_geometry.triangles)
                    # 将三角形转换为边
                    edges = []
                    for tri in triangles:
                        edges.append([tri[0], tri[1]])
                        edges.append([tri[1], tri[2]])
                        edges.append([tri[2], tri[0]])
                else:
                    edges = None
                
                # 投影夹爪到图像上
                image_with_grasps = draw_o3d_gripper_on_image(
                    image_with_grasps, vertices, edges, color, fx, fy, cx, cy
                )
            except Exception as e:
                pass
                        
            # 绘制排名和得分文字
            pixel_x = int(fx * grasp_pos[0] / grasp_pos[2] + cx)
            pixel_y = int(fy * grasp_pos[1] / grasp_pos[2] + cy)
            if 0 <= pixel_x < image_with_grasps.shape[1] and 0 <= pixel_y < image_with_grasps.shape[0]:
                score_text = f"#{i+1} {grasp.score:.2f}"
                cv2.putText(image_with_grasps, score_text, 
                           (pixel_x + 12, pixel_y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
    
    return image_with_grasps

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--use_camera', action='store_true', help='Use RealSense camera instead of example data')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

if cfgs.use_camera:
    try:
        from cameras import RealSenseCamera
    except ImportError as e:
        print(f"无法导入相机模块。请确保已安装必要的依赖:")
        print("  pip install pyrealsense2")
        print(f"具体错误: {e}")
        sys.exit(1)

def demo_from_file(data_dir):
    """演示：从文件加载数据"""
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # get data
    colors = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depths = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    # get camera intrinsics
    fx, fy = 927.17, 927.37
    cx, cy = 651.32, 349.62
    scale = 1000.0
    # set workspace to filter output grasps
    xmin, xmax = -0.19, 0.12
    ymin, ymax = 0.02, 0.15
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    print(points.min(axis=0), points.max(axis=0))

    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:20]
    print(gg_pick.scores)
    print('grasp score:', gg_pick[0].score)

    # visualization
    if cfgs.debug:
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)
        o3d.visualization.draw_geometries([*grippers, cloud])
        o3d.visualization.draw_geometries([grippers[0], cloud])

def demo_from_camera():
    """演示：从 RealSense 相机实时抓取检测"""
    
    print("初始化相机")
    camera = RealSenseCamera(resolution=(640, 480), fps=30)

    print("加载抓取检测网络")
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # 获取相机内参
    intrinsics = camera.get_intrinsics()
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    print(f"相机内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # 工作区域设置
    xmin, xmax = -0.3, 0.3
    ymin, ymax = -0.3, 0.3
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    try:
        frame_count = 0
        fps_timer = time.time()
        
        while True:
            # 获取 RGB 和深度图
            rgb, depth = camera.get_rgb_depth()
            
            if rgb is None or depth is None:
                print("等待相机数据")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # 计算 FPS
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_timer)
                print(f"FPS: {fps:.1f}")
                fps_timer = time.time()
            
            # 从深度图生成点云
            height, width = depth.shape
            xmap, ymap = np.arange(width), np.arange(height)
            xmap, ymap = np.meshgrid(xmap, ymap)
            
            points_z = depth  # 深度值已经是米
            points_x = (xmap - cx) / fx * points_z
            points_y = (ymap - cy) / fy * points_z
            
            # 掩码：过滤有效深度范围
            mask = (points_z > 0.1) & (points_z < 2.0)
            points = np.stack([points_x, points_y, points_z], axis=-1)
            points = points[mask].astype(np.float32)
            
            # 转换颜色格式 (RGB float32)
            colors = rgb.astype(np.float32) / 255.0
            colors = colors[mask].astype(np.float32)
            
            if len(points) == 0:
                print("点云为空，跳过本帧")
                continue
            
            print(f"点云范围: [{points.min(axis=0)} to {points.max(axis=0)}]")
            
            # 抓取检测
            start_time = time.time()
            gg, cloud = anygrasp.get_grasp(
                points, colors, 
                lims=lims, 
                apply_object_mask=True, 
                dense_grasp=False, 
                collision_detection=True
            )
            inference_time = time.time() - start_time
            
            # 在 RGB 图像上绘制抓取
            rgb_with_grasps = rgb.copy()
            
            if len(gg) == 0:
                print(f"未检测到抓取 (推理耗时: {inference_time:.2f}s)")
            else:
                gg = gg.nms().sort_by_score()
                gg_pick = gg[0:20]
                
                print(f"\n检测到 {len(gg_pick)} 个抓取 (推理耗时: {inference_time:.2f}s)")
                print(f"抓取得分: {gg_pick.scores[:5]}")
                print(f"最佳抓取得分: {gg_pick[0].score:.4f}\n")
                
                # 在图像上绘制最优抓取
                rgb_with_grasps = draw_grasps_on_image(rgb, gg_pick, intrinsics, max_grasps=5)
            
            # 显示图像
            cv2.imshow('RGB Image with Grasps', cv2.cvtColor(rgb_with_grasps, cv2.COLOR_RGB2BGR))
            
            # 深度图可视化
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            cv2.imshow('Depth Image', depth_colormap)
            
            # 按 'q' 退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("相机已关闭")


if __name__ == '__main__':
    if cfgs.use_camera:
        demo_from_camera()
    else:
        demo_from_file('./example_data/')
