# 当前数据采集流程

`data_recording/` 目录下当前保留的主路径是多相机 NPY 录制器。

## 当前推荐流程

### 1. CLI 录制

```bash
cd /home/pine/pine_data
source data_record_env/bin/activate
python data_recording/record_multi_camera_npy.py --root /home/pine/pine_data/recordings
```

### 2. Web 录制

```bash
cd /home/pine/pine_data/webapp
./run_recording_webapp.sh
```

### 3. 检查数据

```bash
cd /home/pine/pine_data
source data_record_env/bin/activate
python check_timestamp_recordings.py --root /home/pine/pine_data/recordings
python check_episode_counts.py --root /home/pine/pine_data/recordings
```

## 当前目录结构

```text
recordings/
└── YYYYMMDD/
    └── instruction/
        └── camera_npy/
            └── YYYYMMDDHHMMSS/
```

每个 episode 目录会按实际输入源写入 hand、external、robot 的 `.npy` 数据和 `metadata.json`。

## 当前行为说明

- 录制器初始化时允许相机缺失，只上报状态和预览。
- 只有在 `c` 开始录制时，才会按 `allow_missing_hand` 和 `allow_missing_external` 决定是否拦截。
- `0B07` 现在统一归到 external 相机分配。
- 当前流程没有 `trajs_h5`，也不需要额外的同步脚本。

## 已移除的旧路径

以下旧工具不再属于当前流程：

- 旧 gello tmux 启动脚本
- `sync_robot_camera_data.py`
- `record_multi_camera_npy_monitor.py`
- `npy2h5.py`

如需查看整体入口和环境说明，参考仓库根目录的 [README.md](/home/pine/pine_data/README.md)。
