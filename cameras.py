
import threading
import time
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
    from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
    from kortex_api.autogen.messages import DeviceConfig_pb2, VisionConfig_pb2
    from robot_controller.gen3.kinova import DeviceConnection
    KORTEX_AVAILABLE = True
except ImportError:
    KORTEX_AVAILABLE = False
    logger.warning("Kortex API not available - Kinova camera will not work")

try:
    from configs.constants import BASE_CAMERA_SERIAL
    CONFIGS_AVAILABLE = True
except ImportError:
    CONFIGS_AVAILABLE = False
    BASE_CAMERA_SERIAL = "unknown"
    logger.warning("configs.constants not available")

class Camera:
    def __init__(self):
        self.image = None
        self.last_read_time = time.time()
        threading.Thread(target=self.camera_worker, daemon=True).start()

    def camera_worker(self):
        # Note: We read frames at 30 fps but not every frame is necessarily
        # saved during teleop or used during policy inference
        while True:
            # Reading new frames too quickly causes latency spikes
            while time.time() - self.last_read_time < 0.0333:  # 30 fps
                time.sleep(0.0001)
            _, bgr_image = self.cap.read()
            self.last_read_time = time.time()
            if bgr_image is not None:
                self.image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)

    def get_image(self):
        return self.image

    def close(self):
        self.cap.release()

class LogitechCamera(Camera):
    def __init__(self, serial, frame_width=640, frame_height=360, focus=0):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focus = focus  # Note: Set this to 100 when using fisheye lens attachment
        self.cap = self.get_cap(serial)
        super().__init__()

    def get_cap(self, serial):
        cap = cv.VideoCapture(f'/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_{serial}-video-index0')
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Important - results in much better latency

        # Disable autofocus
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)

        # Read several frames to let settings (especially gain/exposure) stabilize
        for _ in range(30):
            cap.read()
            cap.set(cv.CAP_PROP_FOCUS, self.focus)  # Fixed focus

        # Check all settings match expected
        assert cap.get(cv.CAP_PROP_FRAME_WIDTH) == self.frame_width
        assert cap.get(cv.CAP_PROP_FRAME_HEIGHT) == self.frame_height
        assert cap.get(cv.CAP_PROP_BUFFERSIZE) == 1
        assert cap.get(cv.CAP_PROP_AUTOFOCUS) == 0
        assert cap.get(cv.CAP_PROP_FOCUS) == self.focus

        return cap

def find_fisheye_center(image):
    # Find contours
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Fit a minimum enclosing circle around all contours
    return cv.minEnclosingCircle(np.vstack(contours))

def check_fisheye_centered(image):
    height, width, _ = image.shape
    center, _ = find_fisheye_center(image)
    if center is None:
        return True
    return abs(width / 2 - center[0]) < 0.05 * width and abs(height / 2 - center[1]) < 0.05 * height

class KinovaCamera(Camera):
    def __init__(self):
        if not KORTEX_AVAILABLE:
            raise RuntimeError("Kortex API is required for KinovaCamera. Please install kortex_api.")
        
        # GStreamer video capture (see https://github.com/Kinovarobotics/kortex/issues/88)
        # Note: max-buffers=1 and drop=true are added to reduce latency spikes
        self.cap = cv.VideoCapture('rtspsrc location=rtsp://192.168.1.10/color latency=0 ! decodebin ! videoconvert ! appsink sync=false max-buffers=1 drop=true', cv.CAP_GSTREAMER)
        # self.cap = cv.VideoCapture('rtsp://192.168.1.10/color', cv.CAP_FFMPEG)  # This stream is high latency but works with pip-installed OpenCV
        assert self.cap.isOpened(), 'Unable to open stream. Please make sure OpenCV was built from source with GStreamer support.'

        # Apply camera settings
        threading.Thread(target=self.apply_camera_settings, daemon=True).start()
        super().__init__()

        # Wait for camera to warm up
        image = None
        while image is None:
            image = self.get_image()

        # Make sure fisheye lens did not accidentally get bumped
        if not check_fisheye_centered(image):
            raise Exception('The fisheye lens on the Kinova wrist camera appears to be off-center')

    def apply_camera_settings(self):
        # Note: This function adds significant camera latency when it is called
        # directly in __init__, so we call it in a separate thread instead
        
        if not KORTEX_AVAILABLE:
            logger.warning("Kortex API not available, skipping camera settings")
            return

        # Use Kortex API to set camera settings
        with DeviceConnection.createTcpConnection() as router:
            device_manager = DeviceManagerClient(router)
            vision_config = VisionConfigClient(router)

            # Get vision device ID
            device_handles = device_manager.ReadAllDevices()
            vision_device_ids = [
                handle.device_identifier for handle in device_handles.device_handle
                if handle.device_type == DeviceConfig_pb2.VISION
            ]
            assert len(vision_device_ids) == 1
            vision_device_id = vision_device_ids[0]

            # Check that resolution, frame rate, and bit rate are correct
            sensor_id = VisionConfig_pb2.SensorIdentifier()
            sensor_id.sensor = VisionConfig_pb2.SENSOR_COLOR
            sensor_settings = vision_config.GetSensorSettings(sensor_id, vision_device_id)
            try:
                assert sensor_settings.resolution == VisionConfig_pb2.RESOLUTION_640x480  # FOV 65 ± 3° (diagonal)
                assert sensor_settings.frame_rate == VisionConfig_pb2.FRAMERATE_30_FPS
                assert sensor_settings.bit_rate == VisionConfig_pb2.BITRATE_10_MBPS
            except:
                sensor_settings.sensor = VisionConfig_pb2.SENSOR_COLOR
                sensor_settings.resolution = VisionConfig_pb2.RESOLUTION_640x480
                sensor_settings.frame_rate = VisionConfig_pb2.FRAMERATE_30_FPS
                sensor_settings.bit_rate = VisionConfig_pb2.BITRATE_10_MBPS
                vision_config.SetSensorSettings(sensor_settings, vision_device_id)
                assert False, 'Incorrect Kinova camera sensor settings detected, please restart the camera to apply new settings'

            # Disable autofocus and set manual focus to infinity
            # Note: This must be called after the OpenCV stream is created,
            # otherwise the camera will still have autofocus enabled
            sensor_focus_action = VisionConfig_pb2.SensorFocusAction()
            sensor_focus_action.sensor = VisionConfig_pb2.SENSOR_COLOR
            sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_SET_MANUAL_FOCUS
            sensor_focus_action.manual_focus.value = 0
            vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)

class RealSenseCamera(Camera):
    """
    RealSense D435i 相机封装类

    功能:
        - 采集 RGB 和深度图像
        - 提供相机内参
        - 支持多线程后台采集
    """
    
    def __init__(self, resolution=(640, 480), fps=30, 
                 enable_infrared=False, device_serial=None):
        """
        初始化 RealSense 相机

        Args:
            resolution (tuple): 分辨率 (width, height)，默认 (640, 480)
            fps (int): 帧率，默认 30
            device_serial (str, optional): 设备序列号
        """

        # 需要将深度图对齐到 RGB 图
        self.align = rs.align(rs.stream.color)
        
        # 保存参数
        self.resolution = resolution
        self.fps = fps
        
        # pyrealsense2 组件
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.depth_scale = None

        # 数据存储
        self.depth = None
        self.intrinsics = None

        # 配置 RGB 流
        width, height = resolution
        self.config.enable_stream(
            rs.stream.color,           # 流类型：彩色
            width, height,             # 分辨率
            rs.format.bgr8,            # 格式：BGR8
            fps                        # 帧率
        )

        # 配置深度流
        self.config.enable_stream(
            rs.stream.depth,           # 流类型：深度
            width, height,             # 分辨率
            rs.format.z16,             # 格式：16位深度
            fps                        # 帧率
        )

        # 如果有指定设备序列号的话
        if device_serial is not None:
            self.config.enable_device(device_serial)

        # 启动pipeline
        profile = self.pipeline.start(self.config)

        # 获取深度传感器和 depth_scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        logger.info(f"[RealSenseCamera] Depth scale: {self.depth_scale}")

        # 获取 RGB 流的相机内参
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics_data = color_stream.as_video_stream_profile().get_intrinsics()

        # 存储内参为字典
        self.intrinsics = {
            'fx': intrinsics_data.fx,
            'fy': intrinsics_data.fy,
            'cx': intrinsics_data.ppx,
            'cy': intrinsics_data.ppy,
            'width': intrinsics_data.width,
            'height': intrinsics_data.height,
            'distortion': list(intrinsics_data.coeffs)  # 畸变系数
        }
        logger.info(f"[RealSenseCamera] 相机内参: fx={self.intrinsics['fx']:.2f}, fy={self.intrinsics['fy']:.2f}")

        super().__init__()

    def camera_worker(self):
        """ 后台线程工作函数，持续采集 RGB-D 数据 """
        
        while True:
            # 控制帧率
            while time.time() - self.last_read_time < 0.0333:
                time.sleep(0.0001)
            
            try:
                # 等待并获取一帧数据
                frames = self.pipeline.wait_for_frames()
                
                # 提取彩色帧
                color_frame = frames.get_color_frame()
                
                # 提取深度帧
                depth_frame = frames.get_depth_frame()
                
                # 检查帧是否有效
                if not color_frame or not depth_frame:
                    continue
                
                # 转换为 NumPy 数组
                bgr_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # 将 BGR 转换为 RGB 
                rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
                
                # 将深度值转换为米
                depth_image = depth_image.astype(np.float32)
                depth_image = depth_image * self.depth_scale
                
                # 更新数据
                self.image = rgb_image
                self.depth = depth_image
                self.last_read_time = time.time()
                
            except Exception as e:
                logger.error(f"[RealSenseCamera] 采集帧失败: {e}")
                time.sleep(0.1)

    def get_depth(self):
        """
        获取最新的深度图
        
        Returns:
            np.ndarray: [H, W] 深度图，float32，单位：米
        """

        return self.depth
    
    def get_intrinsics(self):
        """
        获取相机内参
        
        Returns:
            dict: 包含 fx, fy, cx, cy, width, height, distortion
        """

        return self.intrinsics
    
    def get_rgb_depth(self):
        """
        同时获取 RGB 和深度图
        
        Returns:
            tuple: (rgb, depth)
                - rgb: [H, W, 3] RGB图像，uint8
                - depth: [H, W] 深度图，float32，单位：米
        """

        frames = self.pipeline.wait_for_frames()

        # 对齐深度图到 RGB 图
        aligned_frames = self.align.process(frames)

        # 获取对齐后的帧
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # 转换为 numpy 数组
        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        
        # 将深度值从毫米转换为米
        depth = depth.astype(np.float32) * 0.001
        
        return rgb, depth
    
    def close(self):
        """ 关闭 RealSense 相机并释放资源 """
        
        try:
            self.pipeline.stop()
            logger.info("[RealSenseCamera] 相机已关闭")
        except Exception as e:
            logger.warning(f"[RealSenseCamera] 关闭相机时出现警告: {e}")


if __name__ == '__main__':
    ###############
    # 以下为原代码 #
    ###############
    # base_camera = LogitechCamera(BASE_CAMERA_SERIAL)
    # wrist_camera = KinovaCamera()
    # try:
    #     while True:
    #         base_image = base_camera.get_image()
    #         wrist_image = wrist_camera.get_image()
    #         cv.imshow('base_image', cv.cvtColor(base_image, cv.COLOR_RGB2BGR))
    #         cv.imshow('wrist_image', cv.cvtColor(wrist_image, cv.COLOR_RGB2BGR))
    #         key = cv.waitKey(1)
    #         if key == ord('s'):  # Save image
    #             base_image_path = f'base-image-{int(10 * time.time()) % 100000000}.jpg'
    #             cv.imwrite(base_image_path, cv.cvtColor(base_image, cv.COLOR_RGB2BGR))
    #             print(f'Saved image to {base_image_path}')
    #             wrist_image_path = f'wrist-image-{int(10 * time.time()) % 100000000}.jpg'
    #             cv.imwrite(wrist_image_path, cv.cvtColor(wrist_image, cv.COLOR_RGB2BGR))
    #             print(f'Saved image to {wrist_image_path}')
    # finally:
    #     base_camera.close()
    #     wrist_camera.close()
    #     cv.destroyAllWindows()

    ###########################
    # 以下为D435i相机的测试代码 #
    ###########################

    # 创建相机对象
    d435i = RealSenseCamera()

    try:
        while True:
            # 获取 RGB 图像和深度图
            rgb, depth = d435i.get_rgb_depth()
            
            # 检查数据有效性
            if rgb is None or depth is None:
                time.sleep(0.1)
                continue

            # 深度图可视化
            # 方法1：归一化到0-255显示
            depth_normalized = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

            # 方法2：伪彩色显示
            depth_colormap = cv.applyColorMap(depth_normalized, cv.COLORMAP_JET)

            # 绘制图像
            cv.imshow('RGB_image', cv.cvtColor(rgb, cv.COLOR_RGB2BGR))
            cv.imshow('depth_image_1', depth_normalized)
            cv.imshow('depth_image_2', depth_colormap)

            key = cv.waitKey(1)
            if key == ord('q'):
                break

    finally:
        d435i.close()
        cv.destroyAllWindows()
