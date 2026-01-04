(isbot) cuhk@cuhk-System-Product-Name:~/ZMAI/IS_Bot$ python ./high_level_controll/run_grasp_gen.py
Logging as admin on device 192.168.8.10
[12:25:50] INFO     正在初始化相机...                                                                                                                               run_grasp_gen.py:61
[12:25:51] INFO     [RealSenseCamera] Depth scale: 0.0010000000474974513                                                                                                 cameras.py:216
           INFO     [RealSenseCamera] 相机内参: fx=605.85, fy=605.72                                                                                                     cameras.py:232
           INFO     正在连接 GraspGen 服务器: http://localhost:60000                                                                                             grasp_gen_policy.py:31
[12:25:52] INFO     Loaded checkpoint sucessfully                                                                                                                      build_sam.py:174
[12:25:53] INFO     开始抓取任务                                                                                                                                   run_grasp_gen.py:493
           INFO                                                                                                                                                    run_grasp_gen.py:304
                    移动至 Retract 预设位置...                                                                                                                                         
                                                                                                                                                                                       
[12:25:56] INFO     策略已重置                                                                                                                                   grasp_gen_policy.py:51
           INFO     正在获取点云...                                                                                                                                run_grasp_gen.py:500
[12:25:59 PM] INFO     请在图像中点击要抓取的物体，按 'q' 确认...                                                                                                  run_grasp_gen.py:120
[12:26:02 PM] INFO     Selected point: 398, 69                                                                                                                      run_grasp_gen.py:46
[12:26:03 PM] INFO     For numpy array image, we assume (HxWxC) format                                                                                      sam2_image_predictor.py:102
              INFO     Computing image embeddings for the provided image...                                                                                 sam2_image_predictor.py:116
              INFO     Image embeddings computed.                                                                                                           sam2_image_predictor.py:129
              INFO     [_rgbd_to_pointcloud] 深度范围: 0.313m ~ 2.149m                                                                                             run_grasp_gen.py:377
              INFO     [_rgbd_to_pointcloud] 点云坐标范围: x=[0.081, 0.154], y=[-0.324, -0.243], z=[0.983, 1.005]                                                  run_grasp_gen.py:394
              INFO     发送点云数据 (1823 points) 到 GraspGen...                                                                                                 grasp_gen_policy.py:88
              INFO     收到 44 个抓取候选                                                                                                                       grasp_gen_policy.py:101
              INFO     [visualize_grasps] 显示最佳抓取，基座坐标系位置: [0.38503251 0.00521566 0.01843972]                                                         run_grasp_gen.py:463
[12:26:16 PM] INFO                                                                                                                                                 run_grasp_gen.py:553
                       执行动作: 1                                                                                                                                                     
                                                                                                                                                                                       
[Errno 32] 断开的管道
[TCPTransport.send] ERROR: None
              ERROR    An error occurred: [Errno 32] 断开的管道                                                                                                    run_grasp_gen.py:577
                       Traceback (most recent call last):                                                                                                                              
                         File "/home/cuhk/ZMAI/IS_Bot/./high_level_controll/run_grasp_gen.py", line 573, in main                                                                       
                           system.run_episode()                                                                                                                                        
                         File "/home/cuhk/ZMAI/IS_Bot/./high_level_controll/run_grasp_gen.py", line 554, in run_episode                                                                
                           self.move_cartesian(action['arm_pos'], action['arm_quat'])                                                                                                  
                         File "/home/cuhk/ZMAI/IS_Bot/./high_level_controll/run_grasp_gen.py", line 231, in move_cartesian                                                             
                           notification_handle = self.base.OnNotificationActionTopic(                                                                                                  
                         File "/home/cuhk/miniconda3/envs/isbot/lib/python3.10/site-packages/kortex_api/autogen/client_stubs/BaseClientRpc.py", line 1016, in                          
                       OnNotificationActionTopic                                                                                                                                       
                           future = self.router.send(reqPayload, 1, BaseFunctionUid.uidOnNotificationActionTopic, deviceId, options)                                                   
                         File "/home/cuhk/miniconda3/envs/isbot/lib/python3.10/site-packages/kortex_api/RouterClient.py", line 70, in send                                             
                           self.transport.send(payloadMsgFrame)                                                                                                                        
                         File "/home/cuhk/miniconda3/envs/isbot/lib/python3.10/site-packages/kortex_api/TCPTransport.py", line 85, in send                                             
                           raise ex                                                                                                                                                    
                         File "/home/cuhk/miniconda3/envs/isbot/lib/python3.10/site-packages/kortex_api/TCPTransport.py", line 80, in send                                             
                           self.sock.sendall(payload)                                                                                                                                  
                       BrokenPipeError: [Errno 32] 断开的管道                                                                                                                          
[Errno 32] 断开的管道
[TCPTransport.send] ERROR: None
[SessionManager.CloseSession] super().CloseSession() failed with: [Errno 32] 断开的管道
Traceback (most recent call last):
  File "/home/cuhk/ZMAI/IS_Bot/./high_level_controll/run_grasp_gen.py", line 580, in <module>
    main()
  File "/home/cuhk/ZMAI/IS_Bot/./high_level_controll/run_grasp_gen.py", line 570, in main
    with utilities.DeviceConnection.createTcpConnection(args) as router:
  File "/home/cuhk/ZMAI/IS_Bot/high_level_controll/utilities.py", line 76, in __exit__
    self.transport.disconnect()
  File "/home/cuhk/miniconda3/envs/isbot/lib/python3.10/site-packages/kortex_api/TCPTransport.py", line 72, in disconnect
    self.sock.shutdown(socket.SHUT_RDWR)
OSError: [Errno 107] 传输端点尚未连接
已中止 (核心已转储)

https://msub.xn--m7r52rosihxm.com/api/v1/client/subscribe?token=36b1d43a3543a98b0c337d3fd0c09a83