import os
import sys
import time
import math

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI


#######################################################
"""
Brake-drop test with variable reach.

For each reach value, drives the arm to (reach, 0, Z_HEIGHT) with the TCP
pointing down, drops motor enable so the brakes have to hold, then re-reads
joint angles and TCP pose to characterize where the arm sagged.

Reports per-joint deflection so you can see which joint is the weak link,
and how that changes as the arm extends further out.
"""
if len(sys.argv) >= 2:
    ip = sys.argv[1]
else:
    try:
        from configparser import ConfigParser
        parser = ConfigParser()
        parser.read('../robot.conf')
        ip = parser.get('xArm', 'ip')
    except:
        ip = input('Please input the xArm ip address:')
        if not ip:
            print('input error, exit')
            sys.exit(1)
########################################################


# ---- Test parameters --------------------------------------------------
REACH_VALUES = [300, 400, 500, 600, 660]   # x distance from base, mm
TRIALS_PER_REACH = 3
Z_HEIGHT = 200                              # constant z (mm) for all poses
ORIENTATION = (180.0, 0.0, 0.0)             # roll, pitch, yaw -- TCP down
BRAKE_HOLD_TIME = 2.0                       # seconds motors stay disabled
SETTLE_TIME = 0.5                           # general settle between steps
HOME_BETWEEN_REACHES = True                 # safer for big pose changes
# -----------------------------------------------------------------------


arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
##arm.move_gohome(wait=True)


def recover():
    """Clear faults and put the arm back in a movable state."""
    arm.clean_error()
    arm.clean_warn()
    arm.set_mode(0)
    arm.set_state(state=0)
    time.sleep(SETTLE_TIME)


records = []  # one dict per trial

for reach in REACH_VALUES:
    print(f"\n=== Reach = {reach} mm ===")
    pose = (reach, 0, Z_HEIGHT, *ORIENTATION)

    if HOME_BETWEEN_REACHES:
        ##arm.move_gohome(wait=True)
        time.sleep(SETTLE_TIME)

    # Pre-position to the test pose once; subsequent trials just re-move there.
    ret = arm.set_position(*pose, wait=True)
    if ret != 0:
        print(f"  ! set_position to {pose} failed (code {ret}); skipping this reach")
        recover()
        continue

    for trial in range(TRIALS_PER_REACH):
        print(f"  Trial {trial + 1}/{TRIALS_PER_REACH}")

        # Make sure we start each trial from the same commanded pose
        ret = arm.set_position(*pose, wait=True)
        if ret != 0:
            print(f"    ! set_position failed (code {ret}); skipping trial")
            recover()
            continue
        time.sleep(SETTLE_TIME)

        # Read state BEFORE the brake event
        _, joints_before = arm.get_servo_angle()       # degrees, length 7
        _, pos_before = arm.get_position()             # [x,y,z,r,p,yaw]
        joints_before = list(joints_before)[:6]

        # Drop motor enable -> brakes hold (or fail to hold) the arm
        arm.motion_enable(False)
        time.sleep(BRAKE_HOLD_TIME)

        # Re-enable so encoders report reliably, then read settled state
        arm.motion_enable(True)
        time.sleep(SETTLE_TIME)
        _, joints_after = arm.get_servo_angle()
        _, pos_after = arm.get_position()
        joints_after = list(joints_after)[:6]

        joints_delta = [a - b for a, b in zip(joints_after, joints_before)]
        tcp_delta = [a - b for a, b in zip(pos_after, pos_before)]

        print("    joint Δ (deg): " +
              " ".join(f"J{i+1}={d:+7.3f}" for i, d in enumerate(joints_delta)))
        print(f"    TCP   Δ (mm/deg): "
              f"dx={tcp_delta[0]:+7.3f} dy={tcp_delta[1]:+7.3f} dz={tcp_delta[2]:+7.3f}")

        records.append({
            'reach': reach,
            'trial': trial,
            'joints_before': joints_before,
            'joints_after': joints_after,
            'joints_delta': joints_delta,
            'tcp_before': list(pos_before),
            'tcp_after': list(pos_after),
            'tcp_delta': tcp_delta,
        })

        # Clear faults from the disable/enable cycle before the next trial
        recover()


# ---- Analysis ---------------------------------------------------------
if not records:
    print("\nNo successful trials — nothing to plot.")
    arm.disconnect()
    sys.exit(1)

reaches = sorted(set(r['reach'] for r in records))
n_reaches = len(reaches)
n_joints = 6
joint_labels = [f'J{i+1}' for i in range(n_joints)]

mean_abs_joint = np.zeros((n_reaches, n_joints))
std_joint = np.zeros((n_reaches, n_joints))
mean_abs_tcp_xyz = np.zeros((n_reaches, 3))

for ri, reach in enumerate(reaches):
    rs = [r for r in records if r['reach'] == reach]
    jd = np.array([r['joints_delta'] for r in rs])    # (trials, 6)
    td = np.array([r['tcp_delta'] for r in rs])       # (trials, 6)
    mean_abs_joint[ri, :] = np.abs(jd).mean(axis=0)
    std_joint[ri, :] = jd.std(axis=0)
    mean_abs_tcp_xyz[ri, :] = np.abs(td[:, :3]).mean(axis=0)

print("\n=== Mean |joint deflection| (deg) ===")
header = f"{'reach (mm)':>12} | " + " ".join(f"{j:>9}" for j in joint_labels)
print(header)
print("-" * len(header))
for ri, reach in enumerate(reaches):
    print(f"{reach:>12} | " +
          " ".join(f"{v:>9.4f}" for v in mean_abs_joint[ri, :]))

print("\n=== Worst joint per reach ===")
for ri, reach in enumerate(reaches):
    worst = int(np.argmax(mean_abs_joint[ri, :]))
    print(f"  reach {reach} mm  ->  J{worst + 1}  "
          f"({mean_abs_joint[ri, worst]:.4f}° avg)")


# ---- Plot -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Top: grouped bars of per-joint |Δ|, grouped by reach
x = np.arange(n_joints)
width = 0.8 / n_reaches
colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_reaches))

for ri, reach in enumerate(reaches):
    offset = (ri - (n_reaches - 1) / 2) * width
    ax1.bar(x + offset, mean_abs_joint[ri, :], width,
            yerr=std_joint[ri, :], capsize=3,
            label=f'{reach} mm', color=colors[ri])
ax1.set_xticks(x)
ax1.set_xticklabels(joint_labels)
ax1.set_xlabel('Joint')
ax1.set_ylabel('Mean |Δ| (degrees)')
ax1.set_title('Per-joint deflection after E-stop, by reach')
ax1.legend(title='Reach')
ax1.grid(True, alpha=0.3, axis='y')

# Bottom: TCP drop magnitude vs reach
tcp_drop_mag = np.linalg.norm(mean_abs_tcp_xyz, axis=1)
ax2.plot(reaches, tcp_drop_mag, 'o-', linewidth=2, markersize=9,
         label='|ΔTCP| (total)')
ax2.plot(reaches, mean_abs_tcp_xyz[:, 0], 's--', label='|Δx|')
ax2.plot(reaches, mean_abs_tcp_xyz[:, 1], '^--', label='|Δy|')
ax2.plot(reaches, mean_abs_tcp_xyz[:, 2], 'v--', label='|Δz|')
ax2.set_xlabel('Reach (mm)')
ax2.set_ylabel('TCP drop (mm)')
ax2.set_title('TCP translation drop vs. reach')
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle(f'xArm brake-drop characterization '
             f'({TRIALS_PER_REACH} trials/reach, z={Z_HEIGHT} mm)')
fig.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
png_path = os.path.join(out_dir, 'brake_test_reach_results.png')
plt.savefig(png_path, dpi=120)
print(f"\nPlot saved to: {png_path}")


# ---- Raw data dump ----------------------------------------------------
csv_path = os.path.join(out_dir, 'brake_test_reach_data.csv')
joint_cols_before = [f'J{i+1}_before' for i in range(6)]
joint_cols_after = [f'J{i+1}_after' for i in range(6)]
joint_cols_delta = [f'J{i+1}_delta' for i in range(6)]
tcp_cols_before = ['x_before', 'y_before', 'z_before',
                   'roll_before', 'pitch_before', 'yaw_before']
tcp_cols_after = ['x_after', 'y_after', 'z_after',
                  'roll_after', 'pitch_after', 'yaw_after']
tcp_cols_delta = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw']
header_cols = (['reach_mm', 'trial']
               + joint_cols_before + joint_cols_after + joint_cols_delta
               + tcp_cols_before + tcp_cols_after + tcp_cols_delta)

with open(csv_path, 'w') as f:
    f.write(','.join(header_cols) + '\n')
    for r in records:
        row = ([r['reach'], r['trial']]
               + r['joints_before'] + r['joints_after'] + r['joints_delta']
               + r['tcp_before'] + r['tcp_after'] + r['tcp_delta'])
        f.write(','.join(f'{v:.6f}' if isinstance(v, float) else str(v)
                         for v in row) + '\n')
print(f"Raw data saved to: {csv_path}")

plt.show()

arm.disconnect()