# coding=utf-8
import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 0: 使用相机
# 1: 使用API录制好的bag
USE_ROS_BAG = 0
# 0: 彩色图像对齐到深度图;
# 1: 深度图对齐到彩色图像
ALIGN_WAY = 1
# 0: 不保存图像
# 1: 保存图像
USE_SAVE = 1
BAG_PATH = "/home/cuhk/kinova/ZMAI/IS_Bot/policy/test/test.bag"
OUTPUT_PATH = "/home/cuhk/kinova/ZMAI/IS_Bot/policy/test/img/"


def Align_version(frames, align, show_pic=0):
    aligned_frames = align.process(frames)
    depth_frame_aligned = aligned_frames.get_depth_frame()
    color_frame_aligned = aligned_frames.get_color_frame()

    color_image_aligned = np.asanyarray(color_frame_aligned.get_data())
    if USE_ROS_BAG:
        color_image_aligned = cv2.cvtColor(color_image_aligned, cv2.COLOR_BGR2RGB)

    depth_image_aligned = np.asanyarray(depth_frame_aligned.get_data())
    depth_colormap_aligned = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image_aligned, alpha=0.05), cv2.COLORMAP_JET
    )

    if show_pic:
        images_aligned = np.hstack((color_image_aligned, depth_colormap_aligned))
        cv2.imshow("aligned_images", images_aligned)

    return color_image_aligned, depth_image_aligned, depth_colormap_aligned


def Unalign_version(frames, show_pic=0):
    # 未对齐版本
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not USE_ROS_BAG:
        left_frame = frames.get_infrared_frame(1)
        right_frame = frames.get_infrared_frame(2)
        left_image = np.asanyarray(left_frame.get_data())
        right_image = np.asanyarray(right_frame.get_data())
        if show_pic:
            cv2.imshow("left_images", left_image)
            cv2.imshow("right_images", right_image)
    # if not depth_frame or not color_frame:
    #     continue
    color_image = np.asanyarray(color_frame.get_data())
    # print("color:",color_image.shape)
    depth_image = np.asanyarray(depth_frame.get_data())
    # print("depth:",depth_image.shape)

    # 相机API录制的大小rosbag的rgb图像与depth图像不一致，用resize调整到一样大
    if USE_ROS_BAG:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        if ALIGN_WAY:  # 深度图对齐到彩色图像
            depth_image = cv2.resize(
                depth_image, (color_image.shape[1], color_image.shape[0])
            )
        else:  # 彩色图像对齐到深度图
            color_image = cv2.resize(
                color_image, (depth_image.shape[1], depth_image.shape[0])
            )
    # 上色
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET
    )
    # Stack both images horizontally
    images = np.hstack((color_image, depth_colormap))
    if show_pic:
        cv2.imshow("images", images)
    return color_image, depth_image, depth_colormap


if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    if USE_ROS_BAG:
        config.enable_device_from_file(BAG_PATH)  # 这是打开相机API录制的视频
    else:
        config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30
        )  # 10、15或者30可选,20或者25会报错，其他帧率未尝试
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # 左右双目
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

    if ALIGN_WAY:
        way = rs.stream.color
    else:
        way = rs.stream.depth
    align = rs.align(way)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("depth_scale is :", depth_scale)  # 深度比例系数为： 0.0010000000474974513

    idx = 0  # 初始化保存索引

    try:
        while True:
            frames = pipeline.wait_for_frames()

            color_image_aligned, depth_image_aligned, depth_colormap_aligned = (
                Align_version(frames, align, show_pic=0)
            )
            color_image, depth_image, depth_colormap = Unalign_version(
                frames, show_pic=0
            )
            # print(depth_image_aligned*depth_scale)

            # === 保存图像 ===
            if USE_SAVE:
                os.makedirs(OUTPUT_PATH, exist_ok=True)
                cv2.imwrite(
                    os.path.join(OUTPUT_PATH, f"color_{idx:04d}.png"),
                    color_image_aligned,
                )
                cv2.imwrite(
                    os.path.join(OUTPUT_PATH, f"depth_{idx:04d}.png"),
                    depth_image_aligned,
                )

                print(f"Saved image_{idx:04d}.png!")
                test = cv2.imread(
                    OUTPUT_PATH + f"depth_{idx:04d}.png", cv2.IMREAD_UNCHANGED
                )
                print(test)

            # real_dis = depth_image_aligned * depth_scale
            # print(f"dis is {real_dis[240][320]}")

            idx += 1

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
