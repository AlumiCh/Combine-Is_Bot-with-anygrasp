import sys
import os
import time
import numpy as np
from rich.logging import RichHandler
import logging
import threading
import argparse
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from segmentation import SAM2Wrapper

# 导入 Kortex API 相关
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
import utilities

# 导入策略和相机
from policies.grasp_gen_policy import GraspGenPolicy
from cameras import RealSenseCamera

# 日志输出配置
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger("rich")

TIMEOUT_DURATION = 20

# 全局变量用于鼠标回调
selected_points = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])
        logger.info(f"Selected point: {x}, {y}")

class GraspSystem:
    def __init__(self, router):
        """
        初始化 GraspSystem，包括机械臂客户端、相机、策略等。

        Args:
            router: Kortex API 的 TCP 连接路由对象。
        """
        self.router = router
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)
        
        # 初始化相机
        logger.info("正在初始化相机...")
        self.camera = RealSenseCamera()

        # 获取相机内参
        self.camera_intrinsics = self.camera.get_intrinsics()
        if self.camera_intrinsics is not None:
            try:
                self.fx = self.camera_intrinsics['fx']
                self.fy = self.camera_intrinsics['fy']
                self.cx = self.camera_intrinsics['cx']
                self.cy = self.camera_intrinsics['cy']
            except Exception as e:
                logger.error(f'未正确获取相机内参: {e}')
                raise
        else:
            logger.error('camera_intrinsics 为空')
            raise ValueError('camera_intrinsics 为空')
        
        # 初始化策略
        camera_to_base = np.array([ 
            [0, 1, 0, 0.57],
            [1, 0, 0, 0.09],
            [0, 0, -1, 1.01],
            [0, 0, 0, 1]
        ])
        
        self.policy = GraspGenPolicy(
            camera_to_base_transform=camera_to_base,
            grasp_gen_host='localhost',
            grasp_gen_port=60000
        )

        # 初始化 SAM2
        self.seg_model = SAM2Wrapper(
            checkpoint_path="sam2/checkpoints/sam2.1_hiera_large.pt",
            model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"
        )



    def get_user_prompt(self, rgb):
        """
        显示图像并允许用户点击选择物体。
        Args:
            rgb: RGB 图像
        Returns:
            points: (N, 2) 用户点击的点
            labels: (N,) 对应的标签 (全为1)
        """
        global selected_points
        selected_points = []
        
        # OpenCV 需要 BGR 格式显示
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        window_name = "Select Object (Click points, then press 'q')"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        logger.info("请在图像中点击要抓取的物体，按 'q' 确认...")
        
        while True:
            img_disp = bgr.copy()
            # 绘制已选点
            for pt in selected_points:
                cv2.circle(img_disp, tuple(pt), 5, (0, 255, 0), -1)
            
            cv2.imshow(window_name, img_disp)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyWindow(window_name)
        
        if not selected_points:
            logger.warning("未选择任何点，将使用默认中心点策略")
            return None, None
            
        return np.array(selected_points), np.ones(len(selected_points))

    def get_robot_state(self):
        """
        获取机械臂当前状态，并将其格式化为字典。

        Returns:
            dict: 包含末端执行器位置 (arm_pos)、姿态四元数 (arm_quat)、关节角 (arm_qpos)、夹爪状态 (gripper_pos) 的字典。
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
    
    def check_for_end_or_abort(self, e):
        """
        生成通知回调函数，用于检测异步动作是否结束或被中止。

        Args:
            e (threading.Event): 线程事件对象，当动作结束或中止时会被 set()。

        Returns:
            function: 符合 Kortex API 要求的通知回调函数。
        """

        def check(notification, e=e):
            # 检查通知事件是否为“动作结束”或“动作中止”
            if notification.action_event == Base_pb2.ACTION_END or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
        return check
    
    def move_cartesian(self, target_pos, target_quat):
        """
        使用 ReachPose 动作将机械臂末端移动到指定的笛卡尔位姿。
        该方法利用机械臂内部的轨迹规划器，保证运动平滑。

        Args:
            target_pos (array-like): 目标位置 [x, y, z]，单位为米。
            target_quat (array-like): 目标姿态四元数 [x, y, z, w]。

        Returns:
            bool: 动作是否成功完成（True 表示到达目标，False 表示超时或失败）。
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

    def control_gripper(self, value):
        """
        发送夹爪控制指令。

        Args:
            value (float): 夹爪目标开合度，0.0 表示完全打开，1.0 表示完全闭合。

        Returns:
            None
        """
        
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = value
        self.base.SendGripperCommand(gripper_command)
        # 给予夹爪一定的动作时间
        time.sleep(1.0) 

    def move_retract(self):
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
    
    def _rgbd_to_pointcloud(self, rgb, depth, prompt_points=None, prompt_labels=None):
        """
        将 RGB-D 图像转换为分割的点云。

        Args:
            rgb (np.ndarray): 颜色信息，形状为 [H, W, 3]，类型为 uint8 (RGB格式)。
            depth (np.ndarray): 深度信息，形状为 [H, W]，类型为 float32。
            prompt_points: 提示点
            prompt_labels: 提示标签

        Returns:
            tuple:
                points (np.ndarray): 点云三维坐标，形状为 [N, 3]。
                colors (np.ndarray): 点云颜色，形状为 [N, 3]。
        """

        # 输入验证
        assert rgb.ndim == 3 and rgb.shape[2] == 3
        assert rgb.dtype == np.uint8
        assert depth.ndim == 2

        # 生成分割点云的掩码
        mask_seg = self.seg_model.segment(rgb, point_coords=prompt_points, point_labels=prompt_labels)

        # 显示分割结果
        cv2.imshow("Mask", (mask_seg * 255).astype(np.uint8)); cv2.waitKey(1)
        
        # 生成点云
        xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth
        points_x = (xmap - self.cx) / self.fx * points_z
        points_y = (ymap - self.cy) / self.fy * points_z
        
        # 打印深度统计信息
        valid_depth_mask = depth > 0
        if valid_depth_mask.any():
            logger.info(f"[_rgbd_to_pointcloud] 深度范围: {depth[valid_depth_mask].min():.3f}m ~ {depth[valid_depth_mask].max():.3f}m")
        else:
            logger.warning("\n[_rgbd_to_pointcloud] 深度图全为0！\n")
        
        # 创建有效点mask
        mask = (points_z > 0.9) & (points_z < 1.1) & mask_seg
        
        # 提取有效点和颜色
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = rgb.astype(np.float32) / 255.0 # 归一化
        colors = colors[mask].astype(np.float32)

        # 检查点云是否为空
        if len(points) == 0:
            logger.warning("\n[_rgbd_to_pointcloud] 没有有效点云！深度值可能不在有效范围内\n")
        else:
            logger.info(f"[_rgbd_to_pointcloud] 点云坐标范围: "
                       f"x=[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], "
                       f"y=[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], "
                       f"z=[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        return points, colors
    
    def create_gripper_geometry(self, transform_matrix, gripper_width=0.08, gripper_height=0.02):
        """
        创建夹爪几何体模型
        Args:
            transform_matrix: 4x4 变换矩阵
            gripper_width: 夹爪宽度
            gripper_height: 夹爪高度
        Returns:
            list of o3d.geometry: 夹爪几何体列表
        """
        # 创建夹爪基座（手掌部分）
        palm_width = 0.02
        palm_length = 0.06
        palm = o3d.geometry.TriangleMesh.create_box(width=palm_width, height=palm_length, depth=gripper_height)
        palm.translate([-palm_width/2, -palm_length/2, 0])
        palm.paint_uniform_color([0.1, 0.1, 0.1])  # 深灰色
        
        # 创建左手指
        finger_thickness = 0.005
        finger_length = 0.04
        left_finger = o3d.geometry.TriangleMesh.create_box(
            width=finger_thickness, height=finger_length, depth=gripper_height)
        left_finger.translate([-gripper_width/2, palm_length/2, 0])
        left_finger.paint_uniform_color([0.0, 0.5, 1.0])  # 蓝色
        
        # 创建右手指
        right_finger = o3d.geometry.TriangleMesh.create_box(
            width=finger_thickness, height=finger_length, depth=gripper_height)
        right_finger.translate([gripper_width/2 - finger_thickness, palm_length/2, 0])
        right_finger.paint_uniform_color([0.0, 0.5, 1.0])  # 蓝色
        
        # 应用变换
        palm.transform(transform_matrix)
        left_finger.transform(transform_matrix)
        right_finger.transform(transform_matrix)
        
        return [palm, left_finger, right_finger]
    
    def visualize_grasps(self, points, colors, grasps):
        """
        可视化点云和抓取（仅显示最高分抓取）
        Args:
            points: (N, 3) numpy array - 相机坐标系下的点云
            colors: (N, 3) numpy array
            grasps: list of dicts (from policy) - 基座坐标系下的抓取姿态
        """
        if len(grasps) == 0:
            logger.warning("[visualize_grasps] 未检测到任何抓取，跳过可视化")
            return
        
        # 构建点云（相机坐标系）
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        if colors is not None and len(colors) > 0:
            cloud.colors = o3d.utility.Vector3dVector(colors)
        
        # 应用变换矩阵（与 anygrasp 一致：翻转Z轴用于可视化）
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        
        # 只显示最高分的抓取（第一个）
        best_grasp = grasps[0]
        logger.info(f"[visualize_grasps] 显示最佳抓取，基座坐标系位置: {best_grasp['arm_pos']}")
        
        # 抓取姿态从基座坐标系转回相机坐标系用于可视化
        T_base_grasp = np.eye(4)
        T_base_grasp[:3, 3] = best_grasp['arm_pos']
        r = R.from_quat(best_grasp['arm_quat'])
        T_base_grasp[:3, :3] = r.as_matrix()
        
        # 逆变换：基座 -> 相机
        camera_to_base_inv = np.linalg.inv(self.policy.camera_to_base)
        T_camera_grasp = camera_to_base_inv @ T_base_grasp
        
        logger.info(f"[visualize_grasps] 相机坐标系位置: {T_camera_grasp[:3, 3]}")
        
        # 应用可视化变换矩阵
        T_vis = trans_mat @ T_camera_grasp
        
        # 创建夹爪几何体
        gripper_geometries = self.create_gripper_geometry(T_vis)
        
        # 使用 draw_geometries 显示（阻塞式）
        o3d.visualization.draw_geometries([cloud] + gripper_geometries)

    def run_episode(self):
        """
        执行完整的抓取任务流程，包括机械臂复位、点云采集、策略推理、动作执行等。

        Returns:
            None
        """
        logger.info("开始抓取任务")
        
        # 复位
        self.move_retract()
        self.control_gripper(0.0)
        self.policy.reset()
        
        logger.info("正在获取点云...")

        # 获取图像
        bgr, depth = self.camera.get_rgb_depth()
        # 等待几秒后执行predict
        time.sleep(3)
        # 再次获取图像以确保深度与 RGB 对齐的图像没问题
        bgr, depth = self.camera.get_rgb_depth()

        if bgr is None or depth is None:
                logger.warning("[AnyGraspService] 相机数据无效，返回空列表")
                return []
        
        # 转换为 RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 获取用户提示
        points_prompt, labels_prompt = self.get_user_prompt(rgb)
        if points_prompt is None:
            logger.warning("用户取消或未选择点")
            return [], None

        # 生成点云
        points, colors = self._rgbd_to_pointcloud(rgb, depth, points_prompt, labels_prompt)

        # 检查点云是否为空
        if len(points) == 0:
            logger.warning("[AnyGraspWrapper] 点云为空，无法进行抓取检测")
            return [], None
        
        step_count = 0
        while step_count < 20:
            obs = self.get_robot_state()
            
            # 策略步进
            pc_input = points if self.policy.state == 'DETECTING' else None
            
            action = self.policy.step(obs, point_cloud=pc_input)
            
            # 如果刚刚完成了检测，可视化抓取
            if self.policy.state == 'EXECUTING' and pc_input is not None:
                 self.visualize_grasps(points, colors, self.policy.latest_grasps)
            
            if action == 'end_episode':
                logger.info("任务完成")
                break
            
            if action is None:
                time.sleep(0.1)
                continue
                
            # 执行动作
            step_count += 1
            logger.info(f"\n执行动作: {step_count}\n")
            self.move_cartesian(action['arm_pos'], action['arm_quat'])
            
            # 夹爪控制
            gripper_val = 1.0 if action['gripper_pos'][0] > 0.5 else 0.0
            self.control_gripper(gripper_val)
            
            time.sleep(0.5)

def main():
    """
    主程序入口，负责解析连接参数并运行抓取系统。

    Returns:
        None
    """
    args = utilities.parseConnectionArguments()
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        system = GraspSystem(router)
        try:
            system.run_episode()
        except KeyboardInterrupt:
            logger.info("User interrupted.")
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()