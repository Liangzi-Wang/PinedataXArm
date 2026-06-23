# Pine Data Recording

当前仓库使用的采集管线已经统一到 SpaceMouse + DataFoundry。

## 当前有效组件

- 遥操作: `spacemouse_teleoperation_datafoundry/3DConnexion_UR5_Teleop_Gripper_pine_h5.py`
- CLI 录制器: `data_recording/record_multi_camera_npy.py`
- Web 录制器: `webapp/record_multi_camera_npy_web.py`
- Web UI: `webapp/main.py`
- tmux 启动脚本:
  - `scripts/tmux_spacemouse_record.sh`
  - `webapp/tmux_spacemouse_record_web.sh`

旧的 gello 流程和旧的 `trajs_h5` 辅助脚本已移除，不再是当前支持路径。

## 数据输出

当前录制统一保存到：

```text
recordings/YYYYMMDD/<instruction>/camera_npy/YYYYMMDDHHMMSS/
```

每个 episode 目录按设备情况写入：

- `rgb_hand.npy`
- `depth_hand.npy`
- `timestamps_hand.npy`
- `rgb_external.npy`
- `depth_external.npy`
- `timestamps_external.npy`
- `timestamps_robot.npy`
- `joint_state.npy`
- `eef_pose.npy`
- `tcp_wrench.npy`
- `joint_torque.npy`
- `gripper_position.npy`
- `metadata.json`

当前管线不会生成 `trajs_h5/trajectory_*.h5`。

## CLI 录制

```bash
cd /home/pine/pine_data
source data_record_env/bin/activate
python data_recording/record_multi_camera_npy.py --root /home/pine/pine_data/recordings
```

控制键：

- `i <instruction>` 设置任务名
- `c` 开始录制
- `s` 停止并保存
- `d` 删除最近 episode
- `q` 退出

当前默认相机分配：

```text
hand:     0B5B
external: 0B5C,0B3A,0B3D,0B07
```

录制器初始化时不会因为缺少相机直接退出；只有在开始录制时，才会按 `allow_missing_*` 配置检查必需相机。

## Web 录制

```bash
cd /home/pine/pine_data/webapp
./run_recording_webapp.sh
```

打开：

```text
http://127.0.0.1:8000
```

Web UI 也采用同样的策略：

- 初始化阶段只展示输入状态并尽量提供预览
- `Start episode` 时才强制检查必需相机
- 默认 `ALLOW_MISSING_HAND=0`
- 默认 `ALLOW_MISSING_EXTERNAL=0`

更多说明见 [webapp/readme.md](/home/pine/pine_data/webapp/readme.md)。

## 检查工具

当前保留并适配的检查脚本：

- `python check_timestamp_recordings.py --root /home/pine/pine_data/recordings`
  - 检查每个 episode 内 hand/external/robot 时间戳是否重叠
- `python check_episode_counts.py --root /home/pine/pine_data/recordings`
  - 统计每个 instruction 的总 episode 数和包含机器人状态的 episode 数

## 环境

建议使用：

```bash
/home/pine/pine_data/data_record_env
```

该环境应包含至少：

- `pyrealsense2`
- `numpy`
- `opencv-python`
- `fastapi`
- `pillow`
- `ur_rtde`
- `spnav`

## 备注

- `spacemouse_teleoperation/` 同级旧目录仍存在，但不是当前支持管线的一部分。
- 如果相机被其他进程占用，先关闭旧录制器或重启 tmux 会话。
