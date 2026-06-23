# DataFoundry Recording Webapp

Browser UI for the current SpaceMouse + DataFoundry recording pipeline.

## Current Pipeline

1. `run_recording_webapp.sh` starts the FastAPI app in `webapp/main.py`.
2. Clicking `Initialize` starts or reconnects the tmux SpaceMouse session from `webapp/tmux_spacemouse_record_web.sh`.
3. Pane 0 runs `spacemouse_teleoperation_datafoundry/3DConnexion_UR5_Teleop_Gripper_pine_h5.py`.
4. Pane 1 runs `webapp/record_multi_camera_npy_web.py`.
5. Runtime status is written to `webapp/.runtime/*.json`.
6. Live previews are written to `webapp/.runtime/*_previews/`.
7. Clicking `Start episode` sends the instruction update and then starts a `camera_npy/<timestamp>` episode.

This webapp does not use the old gello flow and does not produce `trajs_h5/trajectory_*.h5`.

## Camera Behavior

- `allow_missing_hand` and `allow_missing_external` default to `false`.
- `Initialize` is intentionally permissive: it does not fail if one or both cameras are missing.
- The UI shows camera input status immediately and streams whatever cameras are currently available.
- Required camera checks only happen when `Start episode` is pressed.

Current auto-assignment:

```text
Hand camera PIDs:      0B5B
External camera PIDs:  0B5B
```

## Start

```bash
cd /home/pine/pine_data/webapp
./run_recording_webapp.sh
```

Open:

```text
http://127.0.0.1:8000
```

Or from another machine:

```text
http://<robot-computer-ip>:8000
```

## Normal Use

1. Open the page.
2. Check instruction, record root, serials, and camera product IDs.
3. Click `Initialize`.
4. Confirm the status panel reflects the currently connected cameras.
5. Teleoperate with the SpaceMouse.
6. Click `Start episode`.
7. Click `Stop and save` when finished.
8. Refresh the episode list to inspect the saved recording.
9. Click `Shutdown` when done.

## Output Layout

```text
recordings/
└── YYYYMMDD/
    └── instruction/
        └── camera_npy/
            └── YYYYMMDDHHMMSS/
                ├── metadata.json
                ├── rgb_external.npy
                ├── depth_external.npy
                ├── timestamps_external.npy
                ├── rgb_hand.npy
                ├── depth_hand.npy
                ├── timestamps_hand.npy
                ├── timestamps_robot.npy
                ├── joint_state.npy
                ├── eef_pose.npy
                ├── tcp_wrench.npy
                ├── joint_torque.npy
                └── gripper_position.npy
```

If a device is optional and unavailable, only the matching files are skipped.

## Useful Environment Variables

```bash
RECORD_ROOT=/home/pine/pine_data/recordings
DATA_DIR=/home/pine/pine_data/recordings
WEBAPP_ENV=/home/pine/pine_data/data_record_env
TMUX_RECORDING_SESSION=pine_spacemouse_record
HAND_SERIAL=218622270687
EXTERNAL_SERIAL=409122274280
ALLOW_MISSING_HAND=0
ALLOW_MISSING_EXTERNAL=0
UR_ROBOT_IP=192.168.1.10
```

## Troubleshooting

- If `Start episode` fails, check the recorder status box; missing required cameras are enforced there.
- If preview is blank after init, the camera may be disconnected or owned by another process.
- If only one camera is connected, either reconnect the missing one or explicitly allow missing input for that run.
- If the tmux session gets stuck, use `Shutdown`, then `Initialize` again.
