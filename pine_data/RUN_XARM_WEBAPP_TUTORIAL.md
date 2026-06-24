# Pine Data xArm Web UI 运行教程

这份文档说明如何在 `pine_data` 目录下启动当前项目，并解释运行过程中实际调用了哪些接口。

当前这条链路的目标是：

- UI 继续使用 `pine_data/webapp` 这套 DataFoundry 页面
- 相机录制继续使用 `pine_data/webapp/record_multi_camera_npy_web.py`
- SpaceMouse 遥操作继续使用 `pine_data/spacemouse_teleoperation_datafoundry/3DConnexion_UR5_Teleop_Gripper_pine_h5.py`
- 机器人入口改为 `PinedataXArm/test.py` 中的 `XArmController`
- `pine_data/xarm_bridge.py` 负责让现有 UI 和遥操作代码兼容这个控制器

## 1. 目录关系

项目运行时会用到这几个目录：

```text
/home/pine/liangzi/PinedataXArm/
├── test.py
├── pine_data/
│   ├── xarm_bridge.py
│   ├── run_recording_webapp.sh
│   └── webapp/
│       └── run_recording_webapp.sh
└── xArm-Python-SDK/
```

其中：

- `pine_data/webapp/run_recording_webapp.sh` 是实际启动 Web UI 的脚本
- `pine_data/run_recording_webapp.sh` 是一个额外包装入口，会转发到 `webapp/run_recording_webapp.sh`
- `test.py` 提供项目统一使用的 `XArmController`
- `pine_data/xarm_bridge.py` 加载 `test.py`，并适配旧的 RTDE 风格调用
- `xArm-Python-SDK` 是 `test.py` 自身依赖的 Python 包源码

## 2. 运行前准备

确保以下条件满足：

- `pine_data/data_record_env` 已经创建好，并装好了 Web UI、相机和输入设备依赖
- `/home/pine/liangzi/PinedataXArm/test.py` 存在，并定义了 `XArmController`
- `xArm-Python-SDK` 目录存在：
  `/home/pine/liangzi/PinedataXArm/xArm-Python-SDK`
- xArm 机械臂已经联网，并且你知道它的 IP
- Realsense 相机和 SpaceMouse 已经接好
- 系统里有 `tmux`

如果你要确认 Python 环境是否存在，可以看：

```bash
ls /home/pine/liangzi/PinedataXArm/pine_data/data_record_env
```

## 3. 启动方式

标准启动方式是在 `webapp` 目录里运行实际脚本：

```bash
cd /home/pine/liangzi/PinedataXArm/pine_data/webapp
ROBOT_IP=<xarm_ip> ./run_recording_webapp.sh
```

例如：

```bash
cd /home/pine/liangzi/PinedataXArm/pine_data/webapp
ROBOT_IP=192.168.1.10 ./run_recording_webapp.sh
```

如果你想从 `pine_data` 根目录直接敲一条命令，也可以用包装入口：

```bash
cd /home/pine/liangzi/PinedataXArm/pine_data
ROBOT_IP=<xarm_ip> ./run_recording_webapp.sh
```

启动后打开浏览器：

```text
http://127.0.0.1:8000
```

## 4. 启动时做了什么

这条命令背后实际会串起三层：

1. `pine_data/webapp/run_recording_webapp.sh`
   作用：启动 FastAPI Web UI，也把机器人相关环境变量导出给后续 tmux 流程

2. `pine_data/webapp/tmux_spacemouse_record_web.sh`
   作用：在 tmux 里拉起两个 pane
   - 左侧：SpaceMouse 遥操作进程
   - 右侧：相机录制 + 机器人状态录制进程

如果你走的是根目录包装入口，那么在上面两层之前还会先经过：

`pine_data/run_recording_webapp.sh`
作用：设置 `PINE_DIR`、`WEBAPP_DIR`、`ROBOT_BACKEND=xarm`、`XARM_CONTROLLER_PATH`，然后转发到 `webapp/run_recording_webapp.sh`

UI 本身只负责页面和控制命令，不直接操作机械臂。

## 5. UI 里的典型操作流程

页面打开后，一般按这个顺序走：

1. 填写或确认 `Robot IP`
2. 点击 `Initialize`
3. 检查相机预览、机器人状态是否正常
4. 输入任务名
5. 点击 `Start episode`
6. 用 SpaceMouse 控制 xArm 采集
7. 点击 `Stop episode`

数据会写到：

```text
recordings/YYYYMMDD/<instruction>/camera_npy/YYYYMMDDHHMMSS/
```

## 6. 常用环境变量

最常用的是这些：

```bash
ROBOT_IP=<xarm_ip>
ROBOT_BACKEND=xarm
XARM_CONTROLLER_PATH=/home/pine/liangzi/PinedataXArm/test.py
```

如果 `test.py` 没有移动，通常只需要传 `ROBOT_IP`。

相机相关变量也可以在启动前覆盖，例如：

```bash
HAND_SERIAL=...
WRIST_SERIAL=...
EXTERNAL_SERIAL=...
ALLOW_MISSING_HAND=1
ALLOW_MISSING_WRIST=1
ALLOW_MISSING_EXTERNAL=1
```

## 7. 实际调用了哪些项目内部接口

### Web UI 侧

- [webapp/run_recording_webapp.sh](/home/pine/liangzi/PinedataXArm/pine_data/webapp/run_recording_webapp.sh)
  实际启动 FastAPI 的脚本
- [run_recording_webapp.sh](/home/pine/liangzi/PinedataXArm/pine_data/run_recording_webapp.sh)
  可选包装入口，转发到 `webapp/run_recording_webapp.sh`
- [webapp/tmux_spacemouse_record_web.sh](/home/pine/liangzi/PinedataXArm/pine_data/webapp/tmux_spacemouse_record_web.sh)
  启动 tmux 双进程
- [webapp/main.py](/home/pine/liangzi/PinedataXArm/pine_data/webapp/main.py)
  Web API、初始化、状态读取、发送开始/停止/删除命令
- [webapp/record_multi_camera_npy_web.py](/home/pine/liangzi/PinedataXArm/pine_data/webapp/record_multi_camera_npy_web.py)
  相机录制、机器人状态采样、episode 保存
- [spacemouse_teleoperation_datafoundry/3DConnexion_UR5_Teleop_Gripper_pine_h5.py](/home/pine/liangzi/PinedataXArm/pine_data/spacemouse_teleoperation_datafoundry/3DConnexion_UR5_Teleop_Gripper_pine_h5.py)
  SpaceMouse 遥操作主循环
- [xarm_bridge.py](/home/pine/liangzi/PinedataXArm/pine_data/xarm_bridge.py)
  加载 `test.py` 中的 `XArmController`，并适配旧的 RTDE 风格方法名

### 兼容层暴露给旧代码的方法名

这些是旧录制器/旧遥操作脚本实际调用的方法名，调用者不用知道底层已经换成 xArm：

- 控制接口：
  `speedL`
  `speedStop`
  `stopL`
  `stopScript`
  `moveL`
  `moveJ`
  `zeroFtSensor`
  `disconnect`
- 状态接口：
  `getRobotMode`
  `getActualQ`
  `getActualTCPPose`
  `getActualTCPForce`
  `getActualTCPSpeed`
  `getJointTorques`
  `getActualCurrentAsTorque`
  `disconnect`

## 8. 实际调用了哪些机器人接口

`pine_data/xarm_bridge.py` 首先按 `XARM_CONTROLLER_PATH` 加载 `test.py`，然后创建：

```python
XArmController(ip=robot_ip)
```

`test.py` 中已封装的方法会优先直接使用：

- `move_relative(...)`
- `set_gripper(...)`
- `disconnect()`

### 8.1 实时运动控制接口

SpaceMouse 控制和停机主要会调用：

- `motion_enable(True)`
- `set_mode(5)`
- `set_state(0)`
- `vc_set_cartesian_velocity(...)`

其中：

- `set_mode(5)` 表示笛卡尔速度控制模式
- `vc_set_cartesian_velocity(...)` 是 SpaceMouse 实时速度控制的核心接口
- 这些实时接口在当前 `test.py` 中尚未封装，因此通过 `XArmController.arm` 调用

### 8.2 机器人状态采样接口

录制器采样机器人状态时会调用：

- `get_state()`
- `get_servo_angle(is_radian=True)`
- `get_position(is_radian=True)`
- `get_ft_sensor_data()`
- `get_joints_torque()`
- `get_joint_states(is_radian=True, num=3)`

这些状态读取方法在当前 `test.py` 中也尚未封装，因此通过
`XArmController.arm` 读取。

这些数据会被转换后写入：

- `joint_state.npy`
- `eef_pose.npy`
- `tcp_wrench.npy`
- `joint_torque.npy`
- `timestamps_robot.npy`

### 8.3 夹爪接口

如果启用了 xArm 自带夹爪状态/控制，桥接层会调用：

- `XArmController.set_gripper(...)`
- `set_gripper_mode(0)`
- `set_gripper_enable(True)`
- `set_gripper_speed(...)`
- `get_gripper_position()`
- `get_gripper_status()`

这些数据会进入：

- `gripper_position.npy`

## 9. 当前实现里有意保留的限制

这版桥接为了先把主链路跑通，保留了一个安全限制：

- `moveL`
- `moveJ`

默认是禁用的。

原因是当前 SpaceMouse 脚本里那组 reset pose 是按 UR 机械臂写的，不能直接拿去给 xArm 执行。桥接层只有在显式设置下面这个变量时才允许走这些动作：

```bash
XARM_ENABLE_RESET_MOTIONS=1
```

在没有把 reset pose 改成 xArm 安全点位之前，不建议打开。

另外，`test.py` 当前没有提供关节运动方法，所以 `moveJ` 不会执行。即使开启
`XARM_ENABLE_RESET_MOTIONS=1`，需要关节复位的流程仍会明确报错。

## 10. 常见问题

### UI 能打开，但点 Initialize 后机器人没连上

先检查：

- `ROBOT_IP` 是否正确
- `XARM_CONTROLLER_PATH` 是否指向 `/home/pine/liangzi/PinedataXArm/test.py`
- `test.py` 中是否仍然定义了 `XArmController`
- `xArm-Python-SDK` 目录是否存在
- 机械臂是否和这台机器网络互通

### 录制器能运行，但 SpaceMouse 不工作

检查：

- `spnav` 是否已安装到 `data_record_env`
- SpaceMouse 是否被系统识别
- 当前会话是否允许读取输入设备

### 为什么文档里还会出现 RTDE 这个名字

因为这次改造保留了旧代码的函数名和调用习惯，方便最小代价接入。  
也就是说，脚本表面上还是在调用 `RTDEControlInterface/RTDEReceiveInterface`
风格的方法，但机器人对象实际由 `test.py` 中的 `XArmController` 创建。

## 11. 推荐的最小启动命令

如果你只是想尽快跑起来，优先用实际脚本这一条：

```bash
cd /home/pine/liangzi/PinedataXArm/pine_data/webapp
ROBOT_IP=<xarm_ip> ./run_recording_webapp.sh
```

如果你更习惯从 `pine_data` 根目录启动，也可以用包装入口：

```bash
cd /home/pine/liangzi/PinedataXArm/pine_data
ROBOT_IP=<xarm_ip> ./run_recording_webapp.sh
```

如果后面要继续扩展，优先看这几个文件：

- [xarm_bridge.py](/home/pine/liangzi/PinedataXArm/pine_data/xarm_bridge.py)
- [webapp/main.py](/home/pine/liangzi/PinedataXArm/pine_data/webapp/main.py)
- [webapp/record_multi_camera_npy_web.py](/home/pine/liangzi/PinedataXArm/pine_data/webapp/record_multi_camera_npy_web.py)
- [spacemouse_teleoperation_datafoundry/3DConnexion_UR5_Teleop_Gripper_pine_h5.py](/home/pine/liangzi/PinedataXArm/pine_data/spacemouse_teleoperation_datafoundry/3DConnexion_UR5_Teleop_Gripper_pine_h5.py)
