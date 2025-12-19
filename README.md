(isbot) cuhk@cuhk-System-Product-Name:~/ZMAI/IS_Bot$ python ./grasp_client.py
INFO:__main__:[AnyGraspClient] 正在连接到 localhost:50000...
Traceback (most recent call last):
  File "/home/cuhk/ZMAI/IS_Bot/./grasp_client.py", line 77, in <module>
    client = AnyGraspClient(
  File "/home/cuhk/ZMAI/IS_Bot/./grasp_client.py", line 44, in __init__
    self.service = self.manager.AnyGraspService()
  File "/home/cuhk/miniconda3/envs/isbot/lib/python3.10/multiprocessing/managers.py", line 723, in temp
    token, exp = self._create(typeid, *args, **kwds)
  File "/home/cuhk/miniconda3/envs/isbot/lib/python3.10/multiprocessing/managers.py", line 608, in _create
    id, exposed = dispatch(conn, None, 'create', (typeid,)+args, kwds)
  File "/home/cuhk/miniconda3/envs/isbot/lib/python3.10/multiprocessing/managers.py", line 93, in dispatch
    raise convert_to_error(kind, result)
multiprocessing.managers.RemoteError: 
---------------------------------------------------------------------------
TypeError: __init__() missing 2 required positional arguments: 'checkpoint_path' and 'cfgs'
---------------------------------------------------------------------------