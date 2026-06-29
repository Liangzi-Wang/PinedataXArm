# Pine Data xArm Web UI 运行教程

这份文档说明如何在 `pine_data` 目录下启动当前项目，并解释运行过程中实际调用了哪些接口。

当前这条链路的目标是：

- UI 继续使用 `pine_data/webapp` 这套 DataFoundry 页面
- 相机录制继续使用 `pine_data/webapp/record_multi_camera_npy_web.py`
- SpaceMouse 输入由 `spacemouse_queue_publisher.py` 发布到本机队列
- 机械臂控制由 `xarm_queue_teleop.py` 消费队列并调用 `set_position`
- `pine_data/xarm_bridge.py` 继续为录制器提供机器人状态采样

## 1. 目录关系

项目运行时会用到这几个目录：

```text
/home/pine/liangzi/PinedataXArm/
├── spacemouse_queue_publisher.py
├── xarm_queue_teleop.py
├── pine_data/
│   ├── queue_spacemouse_publisher_bridge.py
│   ├── queue_teleop_supervisor.py
│   ├── xarm_bridge.py
│   ├── run_recording_webapp.sh
│   └── webapp/
│       └── run_recording_webapp.sh
└── xArm-Python-SDK/
```

其中：

- `pine_data/webapp/run_recording_webapp.sh` 是实际启动 Web UI 的脚本
- `pine_data/run_recording_webapp.sh` 是一个额外包装入口，会转发到 `webapp/run_recording_webapp.sh`
- `spacemouse_queue_publisher.py` 读取 SpaceMouse 并发布队列状态
- `xarm_queue_teleop.py` 消费队列并控制机械臂
- `pine_data/queue_teleop_supervisor.py` 同时监管上述两个进程
- `pine_data/queue_spacemouse_publisher_bridge.py` 把 SpaceMouse 状态写给 UI
- `pine_data/xarm_bridge.py` 为录制器提供关节、TCP、力和力矩状态

## 2. 运行前准备

确保以下条件满足：

- `pine_data/data_record_env` 已经创建好，并装好了 Web UI、相机和输入设备依赖
- `/home/pine/liangzi/PinedataXArm/spacemouse_queue_publisher.py` 存在
- `/home/pine/liangzi/PinedataXArm/xarm_queue_teleop.py` 存在
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
   - 左侧：SpaceMouse 队列 publisher + xArm 队列控制器
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
XARM_TELEOP_SPEED=300
XARM_TELEOP_ANGULAR_SPEED=45
XARM_MOVE_ACCELERATION=2000
XARM_COMMAND_PERIOD_S=0.01
XARM_TELEOP_CONTROL_MODE=servo
XARM_COMMAND_TRANSLATION_MAP=-y,x,z
XARM_USE_TOOL_TWIST_AA=1
XARM_TOOL_TWIST_AXIS=-z
SPACEMOUSE_QUEUE_PUBLISH_HZ=200
SPACEMOUSE_RESPONSE_EXPONENT=1.5
XARM_QUEUE_POLL_HZ=250
```

如果项目目录没有移动，通常只需要传 `ROBOT_IP`。

其中：

- `XARM_TELEOP_SPEED` 是 SpaceMouse 最大线速度，单位 `mm/s`
- `XARM_TELEOP_ANGULAR_SPEED` 是 SpaceMouse 最大旋转速度，单位 `deg/s`
- `XARM_MOVE_ACCELERATION` 是笛卡尔运动加速度，单位 `mm/s²`
- `XARM_COMMAND_PERIOD_S=0.01` 表示 servo 目标以 100 Hz 更新
- `XARM_TELEOP_CONTROL_MODE=servo` 使用 `mode=1` 和 `set_servo_cartesian`
- `XARM_COMMAND_TRANSLATION_MAP=-y,x,z` 把 UR 风格动作的 X/Y 平移轴交换并修正方向后再发给 xArm；如果现场方向仍反，可以改成 `y,-x,z`、`y,x,z` 等
- `XARM_USE_TOOL_TWIST_AA=1` 让 SpaceMouse 扭转动作使用 xArm tool 坐标系下的轴角 servo，避免绕夹爪轴旋转时被 RPY 边界卡住
- `XARM_TOOL_TWIST_AXIS=-z` 表示扭转默认绕 tool Z 轴反向；如果实机 tool 轴定义不同，可以改成 `z`、`x` 等
- `SPACEMOUSE_QUEUE_PUBLISH_HZ` 是 SpaceMouse 状态发布频率
- `SPACEMOUSE_RESPONSE_EXPONENT` 控制摇杆响应曲线；越小，中段响应越快
- `XARM_QUEUE_POLL_HZ` 是 xArm consumer 的队列轮询频率

例如把最大线速度提高到 `500 mm/s`：

```bash
cd /home/pine/liangzi/PinedataXArm/pine_data
XARM_TELEOP_SPEED=500 ROBOT_IP=<xarm_ip> ./run_recording_webapp.sh
```

同时调整线速度和旋转速度：

```bash
cd /home/pine/liangzi/PinedataXArm/pine_data
XARM_TELEOP_SPEED=500 XARM_TELEOP_ANGULAR_SPEED=60 ROBOT_IP=<xarm_ip> ./run_recording_webapp.sh
```

修改速度后需要先关闭已有的录制 tmux session，再重新 Initialize，新的
SpaceMouse 进程才会使用新速度。

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
- [queue_teleop_supervisor.py](/home/pine/liangzi/PinedataXArm/pine_data/queue_teleop_supervisor.py)
  启动并监管队列 publisher 与 xArm 控制器
- [queue_spacemouse_publisher_bridge.py](/home/pine/liangzi/PinedataXArm/pine_data/queue_spacemouse_publisher_bridge.py)
  把队列 publisher 状态同步给 Web UI
- [xarm_bridge.py](/home/pine/liangzi/PinedataXArm/pine_data/xarm_bridge.py)
  为录制器适配旧的 RTDE 风格状态读取方法名

### 队列遥操作调用链

```text
SpaceMouse
  -> spacemouse_queue_publisher.py
  -> BaseManager 本机队列
  -> xarm_queue_teleop.py
  -> XArmAPI.set_position(..., wait=False)
  -> xArm
```

录制器仍通过兼容层调用这些状态接口：

- `getRobotMode`
- `getActualQ`
- `getActualTCPPose`
- `getActualTCPForce`
- `getActualTCPSpeed`
- `getJointTorques`
- `getActualCurrentAsTorque`
- `disconnect`

## 8. 实际调用了哪些机器人接口

### 8.1 实时运动控制接口

`xarm_queue_teleop.py` 默认使用 servo 笛卡尔增量模式：

- `get_position()`
- `motion_enable(True)`
- `set_mode(1)`
- `set_state(0)`
- `set_servo_cartesian(..., is_radian=False)`

默认每 `0.01s` 发送一次 servo 目标。SpaceMouse 回到空闲时停止发送新目标，
不会反复执行 `set_state(4)` 和重新使能。

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

如果启用了 xArm 自带夹爪控制，`xarm_queue_teleop.py` 会调用：

- `get_gripper_position()`
- `set_gripper_position(..., speed=..., wait=False)`

录制器的只读状态接口还可能调用：

- `get_gripper_position()`
- `get_gripper_status()`

这些数据会进入：

- `gripper_position.npy`

## 9. 当前实现里有意保留的限制

队列遥操作脚本负责实时移动和夹爪控制，但没有实现旧 UR 脚本中的预设姿态
Reset 流程。不要把旧 UR reset pose 直接用于 xArm。

## 10. 常见问题

### UI 能打开，但点 Initialize 后机器人没连上

先检查：

- `ROBOT_IP` 是否正确
- `xarm_queue_teleop.py` 和 `spacemouse_queue_publisher.py` 是否存在
- `xArm-Python-SDK` 目录是否存在
- 机械臂是否和这台机器网络互通

### 录制器能运行，但 SpaceMouse 不工作

检查：

- `spnav` 是否已安装到 `data_record_env`
- SpaceMouse 是否被系统识别
- 当前会话是否允许读取输入设备
- tmux 左侧是否同时出现 `[SpaceMouse] queue server listening` 和
  `[xArm] connected to SpaceMouse queue`

### 为什么状态代码里还会出现 RTDE 这个名字

录制器为了兼容旧数据采集逻辑，仍保留 `RTDEReceiveInterface` 风格的方法名。
SpaceMouse 控制已经不再使用这套接口，而是走新的队列脚本。

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
- [queue_teleop_supervisor.py](/home/pine/liangzi/PinedataXArm/pine_data/queue_teleop_supervisor.py)
- [queue_spacemouse_publisher_bridge.py](/home/pine/liangzi/PinedataXArm/pine_data/queue_spacemouse_publisher_bridge.py)
- [webapp/main.py](/home/pine/liangzi/PinedataXArm/pine_data/webapp/main.py)
- [webapp/record_multi_camera_npy_web.py](/home/pine/liangzi/PinedataXArm/pine_data/webapp/record_multi_camera_npy_web.py)
