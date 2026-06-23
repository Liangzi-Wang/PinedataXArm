(data_record_env) ➜  pine_data python data_recording/record_data_single_camera.py
Traceback (most recent call last):
  File "/home/pine/pine_data/data_recording/record_data_single_camera.py", line 429, in <module>
    main()
  File "/home/pine/pine_data/data_recording/record_data_single_camera.py", line 391, in main
    os.makedirs(data_folder, exist_ok=True)
  File "<frozen os>", line 215, in makedirs
  File "<frozen os>", line 215, in makedirs
  File "<frozen os>", line 215, in makedirs
  File "<frozen os>", line 225, in makedirs
PermissionError: [Errno 13] Permission denied: '/media/mainuser'
(data_record_env) ➜  pine_data 





Enter the task instruction: insertion1215
Warming up camera...
Camera ready!
Traceback (most recent call last):
  File "/home/pine/pine_data/data_recording/record_data_single_camera.py", line 472, in <module>
    main()
  File "/home/pine/pine_data/data_recording/record_data_single_camera.py", line 442, in main
    data = Data()
           ^^^^^^
  File "/home/pine/pine_data/data_recording/record_data_single_camera.py", line 89, in __init__
    self.rtde_r, self.rtde_c, self.gripper = self.initialize_robot()
                                             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pine/pine_data/data_recording/record_data_single_camera.py", line 130, in initialize_robot
    rtde_c = RTDEControlInterface(self.ROBOT_HOST)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: One of the RTDE input registers are already in use! Currently you must disable the EtherNet/IP adapter, PROFINET or any MODBUS unit configured on the robot. This might change in the future.


TCP位姿平均值：[0.2506, -0.2463, 0.3242, 1.1388, -2.9149, -0.0240]
关节角度平均值：[1.9795, -1.4438, 1.3327, -1.4459, -1.5835, 12.2385]