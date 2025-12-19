(anygrasp39) cuhk@cuhk-System-Product-Name:~/Documents/anygrasp_sdk/grasp_detection$ python ./anygrasp_server.py
WARNING:cameras:Kortex API not available - Kinova camera will not work
WARNING:cameras:configs.constants not available
checking license on /home/cuhk/Documents/anygrasp_sdk/grasp_detection/gsnet.cpython-39-x86_64-linux-gnu.so
[2025-12-19 13:03:02.636] [info] [FlexivLic] public key YiboPeng.public_key & signature YiboPeng.signature are matched
[2025-12-19 13:03:02.637] [info] [FlexivLic] license /home/cuhk/Documents/anygrasp_sdk/grasp_detection/license/YiboPeng.lic check passed.
license passed: True, state: FvrLicenseState.PASSED
WARNING:root:Failed to import ros dependencies in rigid_transforms.py
WARNING:root:autolab_core not installed as catkin package, RigidTransform ros methods will be unavailable
INFO:__main__:[AnyGraspServer] 正在加载模型: /home/cuhk/Documents/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar
INFO:__main__:[AnyGraspService] 正在初始化相机...
INFO:cameras:[RealSenseCamera] Depth scale: 0.0010000000474974513
INFO:cameras:[RealSenseCamera] 相机内参: fx=605.85, fy=605.72
INFO:__main__:[AnyGraspService] 初始化 AnyGrasp 模型...
INFO:anygrasp_wrapper:[AnyGraspWrapper] 模型加载成功: /home/cuhk/Documents/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar
INFO:__main__:[AnyGraspService] 模型与相机初始化完成
INFO:__main__:  相机内参: fx=605.85, fy=605.72, cx=318.98, cy=255.84
INFO:__main__:AnyGrasp RPC 服务器已启动: localhost:50000
INFO:__main__:等待客户端连接...
INFO:__main__:[AnyGraspService] 获取图像: rgb=(480, 640, 3), depth=(480, 640)
ERROR:__main__:[AnyGraspService] 检测失败: ValueError: zero-size array to reduction operation minimum which has no identity
ERROR:__main__:[AnyGraspService] 错误堆栈:
ValueError: zero-size array to reduction operation minimum which has no identity




(isbot) cuhk@cuhk-System-Product-Name:~/ZMAI/IS_Bot$ python ./grasp_client.py
INFO:__main__:[AnyGraspClient] 正在连接到 localhost:50000...
INFO:__main__:[AnyGraspClient] 连接成功
INFO:__main__:正在请求 AnyGrasp 服务器进行抓取检测...
INFO:__main__:[AnyGraspClient] 请求抓取检测...
INFO:__main__:收到 0 个抓取
INFO:__main__:[AnyGraspClient] 关闭连接
INFO:__main__:测试完成