(isbot) cuhk@cuhk-System-Product-Name:~/ZMAI/IS_Bot/high_level_controll$ python ./high_level_grasp_controll.py
Logging as admin on device 192.168.8.10
/home/cuhk/ZMAI/IS_Bot/robot_controller/../models/gen3.xml
Model path: /home/cuhk/ZMAI/IS_Bot/models/gen3.xml
INFO:policies.grasp_policy:[GraspPolicy] 正在连接到 AnyGrasp 服务器: localhost:50000
INFO:grasp_client:[AnyGraspClient] 正在连接到 localhost:50000...
INFO:grasp_client:[AnyGraspClient] 连接成功
INFO:policies.grasp_policy:[GraspPolicy] 初始化完成
INFO:__main__:
正在重置机器人...

INFO:__main__:
移动至 Retract 预设位置...

ERROR:__main__:An error occurred: Server error name=ERROR_DEVICE, sub name=INVALID_PARAM => Failed to execute gripper command: Argument Invalid 
Traceback (most recent call last):
  File "/home/cuhk/ZMAI/IS_Bot/high_level_controll/./high_level_grasp_controll.py", line 303, in main
    controller.run_episode()
  File "/home/cuhk/ZMAI/IS_Bot/high_level_controll/./high_level_grasp_controll.py", line 252, in run_episode
    self.send_gripper_command(0.0)
  File "/home/cuhk/ZMAI/IS_Bot/high_level_controll/./high_level_grasp_controll.py", line 148, in send_gripper_command
    self.base.SendGripperCommand(gripper_command)
  File "/home/cuhk/miniconda3/envs/isbot/lib/python3.10/site-packages/kortex_api/autogen/client_stubs/BaseClientRpc.py", line 1220, in SendGripperCommand
    result = future.result(options.getTimeoutInSecond())
  File "/home/cuhk/miniconda3/envs/isbot/lib/python3.10/concurrent/futures/_base.py", line 458, in result
    return self.__get_result()
  File "/home/cuhk/miniconda3/envs/isbot/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
    raise self._exception
kortex_api.Exceptions.KServerException.KServerException: Server error name=ERROR_DEVICE, sub name=INVALID_PARAM => Failed to execute gripper command: Argument Invalid 
https://msub.xn--m7r52rosihxm.com/api/v1/client/subscribe?token=36b1d43a3543a98b0c337d3fd0c09a83