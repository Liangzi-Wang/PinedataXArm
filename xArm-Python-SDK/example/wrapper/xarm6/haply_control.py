import asyncio, json, os, sys, time, websockets, threading, math
from scipy.spatial.transform import Rotation as R
import pybullet as pb
import pybullet_data
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from xarm.wrapper import XArmAPI

# --- URDF auto-detection ---
# Run setup_urdf.py once to generate manufacturer URDFs with correct joint conventions.
# If not yet generated, fall back to the pybullet_data placeholder.
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_URDF_DIR     = os.path.join(_SCRIPT_DIR, "urdf")
_XARM6_URDF   = os.path.join(_URDF_DIR, "xarm6.urdf")
_XARM6_URDF   = _XARM6_URDF if os.path.isfile(_XARM6_URDF) else "xarm/xarm6_with_gripper.urdf"
_UF850_URDF   = os.path.join(_URDF_DIR, "uf850.urdf")
_UF850_URDF   = _UF850_URDF if os.path.isfile(_UF850_URDF) else _XARM6_URDF  # fall back to xarm6

# --- Robot selection ---
# Can also pass robot type as second arg: python haply_control.py <ip> xarm|850
_ROBOT_CONFIGS = {
    "xarm": {
        "name":           "xArm 6",
        "home_angle":     [0, -40, -15, 0,   55, 0],
        "urdf":           _XARM6_URDF,
        "scale":          1000.0,   # mm of robot travel per metre of Haply travel
        "scale_y":        1.0,      # Y-axis multiplier (relative to scale)
        "box_half_mm":    250.0,    # ±mm X/Z workspace box around home position
        "box_half_mm_y":  250.0,    # ±mm Y (left-right) workspace box
        "tool_yaw_deg":   0,        # CCW rotation applied to Haply XY before driving robot XY
        "gripper_joints": [8, 9, 10, 11, 12, 13],  # drive_joint + finger mimics
    },
    "850": {
        "name":           "UFactory 850 (right — gripper faces left)",
        "home_angle":     [0,  15, -20, 0,  -35, 0],
        "urdf":           _UF850_URDF,
        "scale":          1400.0,   # larger travel to match 850 mm reach vs 700 mm on xArm 6
        "scale_y":        1.8,      # tuned so full Haply Y range ≈ full arm Y reach (~650 mm)
        "box_half_mm":    500.0,    # ±mm X/Z workspace box
        "box_half_mm_y":  700.0,    # ±mm Y workspace — set above physical limit so clamp never bites
        "tool_yaw_deg":   -90,      # gripper faces left: Haply fwd→-Y, Haply right→-X
        "gripper_joints": [],
    },
    "850l": {
        "name":           "UFactory 850 (left — gripper faces right)",
        "home_angle":     [0,  15, -20, 0,  -35, 0],
        "urdf":           _UF850_URDF,
        "scale":          1400.0,
        "scale_y":        1.8,
        "box_half_mm":    500.0,
        "box_half_mm_y":  700.0,
        "tool_yaw_deg":   90,       # gripper faces right: Haply fwd→+Y, Haply right→+X
        "gripper_joints": [],
    },
}

if len(sys.argv) >= 3 and sys.argv[2].lower() in _ROBOT_CONFIGS:
    _robot_key = sys.argv[2].lower()
else:
    print("\nSelect robot:")
    print("  1  xArm 6")
    print("  2  UFactory 850 (right — gripper faces left)")
    print("  3  UFactory 850 (left  — gripper faces right)")
    _choice = input("Enter 1, 2, or 3: ").strip()
    _robot_key = {"1": "xarm", "2": "850", "3": "850l"}.get(_choice, "xarm")

ROBOT_CFG = _ROBOT_CONFIGS[_robot_key]
print(f"Robot: {ROBOT_CFG['name']}")
print(f"URDF:  {ROBOT_CFG['urdf']}")

# --- xArm IP ---
if len(sys.argv) >= 2:
    ip = sys.argv[1]
else:
    try:
        from configparser import ConfigParser
        parser = ConfigParser(); parser.read('../robot.conf')
        ip = parser.get('xArm', 'ip')
    except Exception:
        ip = input('Please input the xArm ip address: ')

# --- xArm bring-up ---
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.clean_warn(); arm.clean_error()
arm.set_mode(0); arm.set_state(0)
arm.set_servo_angle(angle=ROBOT_CFG["home_angle"], is_radian=False, wait=True)

code, xarm_origin = arm.get_position(is_radian=False)  # [x,y,z,roll,pitch,yaw] mm/deg
assert code == 0, f"get_position failed: {code}"
print("xArm origin:", xarm_origin)

# --- Gripper init (Bio Gripper G2 — skipped gracefully if not attached) ---
HAS_GRIPPER = False
try:
    ret = arm.set_gripper_enable(True)
    if ret == 0:
        arm.set_gripper_mode(0)
        arm.set_gripper_speed(3000)
        arm.set_gripper_g2_position(84, speed=225, wait=True)   # open fully on startup
        HAS_GRIPPER = True
        print("Gripper G2 initialized")
    else:
        raise RuntimeError(f"set_gripper_enable returned {ret}")
except Exception as e:
    print(f"Gripper not available ({e}) — continuing without gripper")
    arm.clean_error(); arm.clean_warn()
    arm.motion_enable(enable=True); arm.set_mode(0); arm.set_state(0)

# --- FT sensor init ---
# Do NOT zero here — use UFactory Studio's "Identify Load" (FT sensor page) first,
# then save the result. Zeroing without load identification on a heavy gripper injects
# a large gravity-bias into the haptic feedback and causes instability.
arm.set_ft_sensor_enable(1)
time.sleep(0.3)
print("FT sensor enabled")

# Servo Cartesian mode for streaming targets
arm.set_mode(1); arm.set_state(0)
time.sleep(0.2)

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
SCALE         = ROBOT_CFG["scale"]   # mm of robot travel per metre of Haply travel (lower = less sensitive)
LOOP_HZ       = 250
LOOP_DT       = 1.0 / LOOP_HZ
MAX_STEP_MM   = 6.0      # hard per-tick velocity cap (mm); lower = smoother but more lag
BOX_HALF_MM   = ROBOT_CFG["box_half_mm"]
BOX_HALF_MM_Y = ROBOT_CFG["box_half_mm_y"]

_yaw_rad   = math.radians(ROBOT_CFG.get("tool_yaw_deg", 0))
_YAW_COS   = math.cos(_yaw_rad)
_YAW_SIN   = math.sin(_yaw_rad)

# Position smoothing: EMA on the commanded Cartesian target.
# Robot glides toward the target instead of snapping to it each tick.
# 0.0 = frozen, 1.0 = no smoothing. Start at 0.6 — raise if it feels too sluggish.
POS_ALPHA     = 0.6

# Force feedback: scale factor from FT sensor (N) → Haply cursor force (N).
FORCE_SCALE    = 0.12    # lower = softer feedback
FT_ALPHA       = 0.06    # EMA on force (lower = smoother, more lag)
# Slew-rate limit: max force change per tick (N). Kills the spike on first contact.
MAX_FORCE_SLEW = 0.08    # N per tick at 250 Hz → max 20 N/s ramp rate

# Position-hold spring — keeps the Haply handle from sagging when released.
# The spring anchor tracks the cursor while the handle is moving (zero spring force),
# then freezes when the handle goes still so the spring holds it in place.
# Robot naturally hovers because the cursor returns to its last position.
HOLD_SPRING_K      = 60.0    # N/m  — lateral stiffness (spring fights displacement from hold_point)
HOLD_VEL_THRESHOLD = 0.0001  # m/tick at 250 Hz ≈ 25 mm/s — above this = "moving"

# Gravity compensation — constant upward force applied at all times to cancel the
# weight of the VersaGrip + Inverse3 end-effector.  A spring alone can't hold Z
# position: equilibrium = hold_point - weight/K, so the handle always sags.
# With gravity comp the equilibrium is exactly at hold_point.
# Tune: hold the handle still, let go — raise GRAVITY_COMP_Z until it stops falling,
# lower it until it stops rising.  Typical range: 1.0–3.0 N.
GRAVITY_COMP_Z = 0.5    # N — start here and tune in 0.25 N steps

# Per-axis bias correction (N) — subtract the standing DC offset visible in UFactory Studio
# when the arm is stationary and not touching anything. Read from the FT graphs at rest.
# Tune these until the Haply feels neutral with no contact.
FT_BIAS = [0.07,   # Fx mean offset (N) — measured ~0.067 N at rest
           0.05,   # Fy mean offset (N) — measured ~0.051 N at rest
          -0.27]   # Fz mean offset (N) — measured ~-0.269 N at rest

# Per-axis deadbands (N) — set just above the noise band after bias removal.
# Fx noise ≈ ±0.25 N, Fy noise ≈ ±0.35 N, Fz noise ≈ ±0.25 N (from FT graphs).
FORCE_DEADBAND_X = 0.4
FORCE_DEADBAND_Y = 0.5
FORCE_DEADBAND_Z = 0.5

# Gripper — Bio Gripper G2: position in mm, 84 = fully open, 0 = fully closed
GRIPPER_OPEN  = 84    # mm
GRIPPER_CLOSE = 0     # mm
GRIPPER_SPEED = 225   # mm/s

# Wrist orientation control (VersaGrip stylus rotation → robot RPY)
ROT_SCALE          = 0.5    # 1.0 = 1:1 mapping; reduce if wrist moves too fast
MAX_DELTA_DEG      = 60.0   # safety clamp: max deviation from home orientation per axis
# How quickly the arm tracks the stylus while orientation is enabled.
# Lower = smoother engage/track but more lag; higher = more responsive but jumpier on press.
# At 250 Hz: 0.15 → smooth glide to target in ~0.1 s.
ORIENT_TRACK_ALPHA  = 0.15
# How quickly the arm returns to home orientation after toggle-off.
# At 250 Hz: 0.005 → ~63% home in 0.8 s, fully home in ~3 s. Increase to return faster.
ORIENT_RETURN_ALPHA = 0.005
# Axis remapping: VersaGrip delta euler [x,y,z] → robot [roll, pitch, yaw]
# Tune signs/order if the robot wrist moves in the wrong direction.
# Row = [roll_from, pitch_from, yaw_from] contributions for each robot axis.
ROT_REMAP = [( 0, -1.0),   # robot roll  ← VersaGrip euler[0]  (VG x-axis)
             ( 1, -1.0),   # robot pitch ← VersaGrip euler[1]  (VG y-axis)
             ( 2, -1.0)]   # robot yaw   ← VersaGrip euler[2]  (VG z-axis)

# VersaGrip hall sensor calibration
# Hold the grip fully open and note the hall value → HALL_OPEN
# Squeeze fully and note the hall value → HALL_CLOSED
# (run with debug prints to find your values, typical range is ~10–60)
HALL_OPEN   = 18   # hall reading when grip is relaxed (fully open)
HALL_CLOSED = 45   # hall reading when grip is fully squeezed — tune this!

# Button mapping (a/b/c) — set to True to use a button instead of hall analog
USE_BUTTON_GRIP = True   # button 'a': hold to close gripper, release to open

# Orientation enable: hold button 'b' to activate wrist rotation tracking.
# On press, the stylus's current pose is captured as the new zero — arm stays at
# home orientation (-180, 0, 0) until the button is held.
ORIENT_BUTTON = "b"

# ---------------------------------------------------------------------------
# Workspace box centred on the home pose
# ---------------------------------------------------------------------------
WS_LIMITS = {
    "x": (xarm_origin[0] - BOX_HALF_MM,   xarm_origin[0] + BOX_HALF_MM),
    "y": (xarm_origin[1] - BOX_HALF_MM_Y, xarm_origin[1] + BOX_HALF_MM_Y),
    "z": (xarm_origin[2] - 100.0,          xarm_origin[2] + 300.0),
}
print("Workspace box:", WS_LIMITS)

clamp = lambda v, lo, hi: max(lo, min(hi, v))

# ---------------------------------------------------------------------------
# PyBullet visualisation
# Shared state written by viz_updater (async), read by viz thread
# ---------------------------------------------------------------------------
_viz_joints  = [0.0] * 6    # arm joint angles in radians
_viz_gripper = [0.0]         # [0]=gripper drive angle: 0=open, 0.85=closed
_viz_cursor  = [xarm_origin[0]/1000, xarm_origin[1]/1000,
                xarm_origin[2]/1000]  # robot TCP in metres (for cursor sphere)
_viz_force   = [0.0, 0.0, 0.0]  # raw FT force [Fx,Fy,Fz] in N (tool frame), for force arrow
_viz_stop    = threading.Event()

# Joint indices are identical across all supported URDFs:
#   [0] world_joint (fixed), [1-6] joint1-6 (revolute), [7+] gripper (robot-dependent)
_ARM_JOINTS     = [1, 2, 3, 4, 5, 6]
_GRIPPER_JOINTS = ROBOT_CFG["gripper_joints"]  # [] for 850, [8-13] for xArm 6

def _run_pybullet_viz():
    """Runs entirely in its own thread — never touches asyncio."""
    client = pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.81)
    pb.resetDebugVisualizerCamera(
        cameraDistance=1.2, cameraYaw=45, cameraPitch=-30,
        cameraTargetPosition=[0.3, 0, 0.3])
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)

    pb.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))

    # Switch search path to local urdf/ so relative mesh paths resolve
    local_urdf_dir = os.path.join(_SCRIPT_DIR, "urdf")
    if os.path.isdir(local_urdf_dir) and ROBOT_CFG["urdf"].startswith(local_urdf_dir):
        pb.setAdditionalSearchPath(local_urdf_dir)

    robot = pb.loadURDF(ROBOT_CFG["urdf"], basePosition=[0, 0, 0], useFixedBase=True)
    print(f"PyBullet loaded: {ROBOT_CFG['urdf']}")

    # --- Gripper body (loaded separately, snapped to arm eef each frame) ---
    _GRIPPER_URDF = os.path.join(_SCRIPT_DIR, "urdf", "xarm_gripper.urdf")
    gripper_body  = None
    _GRIPPER_DRIVE_JOINTS = list(range(6))  # drive + 5 mimic joints (all mult=1)
    if os.path.isfile(_GRIPPER_URDF):
        gripper_body = pb.loadURDF(_GRIPPER_URDF, useFixedBase=False,
                                   flags=pb.URDF_IGNORE_COLLISION_SHAPES)
        # Disable all collisions between gripper and arm so they don't interact
        pb.setCollisionFilterGroupMask(gripper_body, -1, 0, 0)
        print("Gripper URDF loaded.")
    else:
        print("xarm_gripper.urdf not found — run setup_urdf.py to enable gripper viz.")

    # --- FT sensor body (snapped between arm eef and gripper each frame) ---
    _FT_URDF = os.path.join(_SCRIPT_DIR, "urdf", "ft_sensor", "ft_sensor.urdf")
    ft_body  = None
    # The Onshape-exported URDF has its root origin 56 mm before the arm-side mounting
    # flange (mesh Z range: 56–118 mm in sensor local frame). Back the body up by that
    # amount so the mount face lands at the eef; gripper goes at the far face (62 mm past).
    FT_MOUNT_OFFSET  = 0.056   # metres — shift sensor root toward arm so flange = eef
    FT_SENSOR_HEIGHT = 0.062   # metres — gripper offset from eef (sensor thickness)
    if os.path.isfile(_FT_URDF):
        # Set search path to ft_sensor/ so relative mesh paths resolve
        pb.setAdditionalSearchPath(os.path.join(_SCRIPT_DIR, "urdf", "ft_sensor"))
        try:
            ft_body = pb.loadURDF(_FT_URDF, useFixedBase=False,
                                  flags=pb.URDF_IGNORE_COLLISION_SHAPES)
            pb.setCollisionFilterGroupMask(ft_body, -1, 0, 0)
            print("FT sensor URDF loaded.")
        except Exception as e:
            print(f"FT sensor URDF failed to load: {e}")
        # Restore arm search path
        if os.path.isdir(local_urdf_dir) and ROBOT_CFG["urdf"].startswith(local_urdf_dir):
            pb.setAdditionalSearchPath(local_urdf_dir)
    else:
        print("ft_sensor/ft_sensor.urdf not found — run setup_urdf.py.")

    # --- Cursor sphere (orange) — commanded TCP target ---
    sph_vis     = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.02,
                                       rgbaColor=[1.0, 0.55, 0.0, 0.75])
    cursor_body = pb.createMultiBody(0, -1, sph_vis, basePosition=_viz_cursor[:])

    # --- Force arrow — persistent debug line updated each frame ---
    # Scale: 1 N → 0.04 m line length; orange colour
    FORCE_SCALE_VIZ = 0.04
    _force_line = pb.addUserDebugLine([0, 0, 0], [0, 0, 0.01],
                                      lineColorRGB=[1.0, 0.4, 0.0],
                                      lineWidth=3.0)

    # The arm's last link index in the URDF (eef link = joint 6 child for 850)
    _EEF_LINK = 6

    print("PyBullet visualisation ready — check the window that just opened.")

    while not _viz_stop.is_set():
        # --- Arm joints (blocking SDK call — fine in this blocking thread) ---
        err, angles = arm.get_servo_angle(is_radian=True)
        if err == 0 and angles and len(angles) >= 6:
            for idx, angle in zip(_ARM_JOINTS, angles[:6]):
                pb.resetJointState(robot, idx, angle)

        # --- Get eef pose once — used by FT sensor, gripper, and force arrow ---
        ls      = pb.getLinkState(robot, _EEF_LINK, computeForwardKinematics=True)
        eef_pos = ls[4]   # worldLinkFramePosition
        eef_orn = ls[5]   # worldLinkFrameOrientation
        m       = pb.getMatrixFromQuaternion(eef_orn)
        # Tool Z axis in world frame (third column of rotation matrix)
        tz_x, tz_y, tz_z = m[2], m[5], m[8]

        # --- FT sensor: root shifted back by FT_MOUNT_OFFSET along tool Z so the
        #     arm-side flange aligns with the eef (mesh Z range starts at 56 mm) ---
        if ft_body is not None:
            ft_pos = (eef_pos[0] - tz_x * FT_MOUNT_OFFSET,
                      eef_pos[1] - tz_y * FT_MOUNT_OFFSET,
                      eef_pos[2] - tz_z * FT_MOUNT_OFFSET)
            pb.resetBasePositionAndOrientation(ft_body, ft_pos, eef_orn)

        # --- Gripper: snap FT_SENSOR_HEIGHT below eef along tool Z ---
        g = _viz_gripper[0]
        if gripper_body is not None:
            gp = (eef_pos[0] + tz_x * FT_SENSOR_HEIGHT,
                  eef_pos[1] + tz_y * FT_SENSOR_HEIGHT,
                  eef_pos[2] + tz_z * FT_SENSOR_HEIGHT)
            pb.resetBasePositionAndOrientation(gripper_body, gp, eef_orn)
            for gi in _GRIPPER_DRIVE_JOINTS:
                pb.resetJointState(gripper_body, gi, g)

        # --- Cursor sphere ---
        pb.resetBasePositionAndOrientation(
            cursor_body, _viz_cursor[:], [0, 0, 0, 1])

        # --- Force arrow: originates at gripper/sensor junction, tool frame → world ---
        arrow_origin = (eef_pos[0] + tz_x * FT_SENSOR_HEIGHT,
                        eef_pos[1] + tz_y * FT_SENSOR_HEIGHT,
                        eef_pos[2] + tz_z * FT_SENSOR_HEIGHT)
        fx, fy, fz = _viz_force
        wfx = m[0]*fx + m[1]*fy + m[2]*fz
        wfy = m[3]*fx + m[4]*fy + m[5]*fz
        wfz = m[6]*fx + m[7]*fy + m[8]*fz
        arrow_end = [arrow_origin[0] + wfx * FORCE_SCALE_VIZ,
                     arrow_origin[1] + wfy * FORCE_SCALE_VIZ,
                     arrow_origin[2] + wfz * FORCE_SCALE_VIZ]
        mag = (wfx**2 + wfy**2 + wfz**2) ** 0.5
        # Colour: green (no force) → red (high force, threshold ~5 N)
        r = min(1.0, mag / 5.0)
        _force_line = pb.addUserDebugLine(
            list(arrow_origin), arrow_end,
            lineColorRGB=[r, 1.0 - r, 0.0],
            lineWidth=4.0,
            replaceItemUniqueId=_force_line)

        time.sleep(0.033)   # ~30 Hz

    pb.disconnect()


async def main():
    async with websockets.connect("ws://localhost:10001") as ws:
        # Scan up to 30 frames (~0.5 s at 60 Hz) to discover both devices.
        # The VersaGrip sometimes appears a few frames after the Inverse3.
        VERSAGRIP_KEYS = {"versagrip", "handle", "wrist_joint",
                          "verse_grip", "wireless_verse_grip", "custom_verse_grip"}
        dev_id = None
        haply_origin = None
        vg_key = None
        vg_id  = None

        for frame_n in range(30):
            msg = json.loads(await ws.recv())

            # Print full structure on frame 0 so we can see all keys
            if frame_n == 0:
                print(">>> Haply WS keys in frame 0:", list(msg.keys()))
                for k, v in msg.items():
                    if isinstance(v, list) and v:
                        print(f"    [{k}] device_id =", v[0].get("device_id", "?"),
                              "| top-level keys:", list(v[0].keys()))

            if "inverse3" in msg and msg["inverse3"] and dev_id is None:
                dev_id       = msg["inverse3"][0]["device_id"]
                haply_origin = msg["inverse3"][0]["state"]["cursor_position"]

            # Try known key names first, then any unknown key that looks like a device list
            candidate_keys = list(VERSAGRIP_KEYS) + [
                k for k in msg if k not in ("inverse3", "session_id", "session", *VERSAGRIP_KEYS)
            ]
            for k in candidate_keys:
                val = msg.get(k)
                if isinstance(val, list) and val and isinstance(val[0], dict) and vg_id is None:
                    vg_key = k
                    vg_id  = val[0].get("device_id")
                    print(f"VersaGrip detected under key '{vg_key}': {vg_id}")
                    break

            if dev_id and vg_id:
                break  # found both

        assert dev_id, "Never received Inverse3 device_id from Haply service"
        if vg_id is None:
            print("VersaGrip not found after 30 frames — gripper control disabled")
            print("  (check the key name printed above and add it to VERSAGRIP_KEYS)")

        print("Haply origin:", haply_origin)

        # --- Centre the Inverse3 before starting ---
        # Applies a spring toward (x=0, y=0, z=haply_origin.z) — the device's
        # mechanical XY centre at its natural height. Release the handle.
        # A command is sent every tick so the service keeps streaming state.
        print("Centering Inverse3 — please release the handle...")
        _SPRING_K   = 12.0   # N/m — must be strong enough to overcome device friction
        _MAX_CF     = 1.2    # N  — clamp per-axis so it doesn't snap violently
        _CENTER_TOL = 0.008  # m — within 8 mm counts as settled
        _TIMEOUT    = 4.0    # s — give up after this long regardless
        # Target: XY centre of the workspace, Z at the device's natural resting height.
        # If the device has a passive base swivel it must be set manually before startup —
        # set_cursor_force only drives the three actuated pantograph joints.
        _target     = {"x": 0.0, "y": 0.0, "z": 0.195}   # 0.195 m ≈ Inverse3 home Z
        _t0         = time.monotonic()
        _cur        = dict(haply_origin)

        while (time.monotonic() - _t0) < _TIMEOUT:
            # recv with a short timeout so we never block if the service goes quiet
            try:
                _raw = await asyncio.wait_for(ws.recv(), timeout=0.05)
                _msg = json.loads(_raw)
                if "inverse3" in _msg and _msg["inverse3"]:
                    _cur = _msg["inverse3"][0]["state"]["cursor_position"]
            except asyncio.TimeoutError:
                pass   # no message — still send a command below

            _dx = _cur["x"] - _target["x"]
            _dy = _cur["y"] - _target["y"]
            _dz = _cur["z"] - _target["z"]
            _dist = (_dx**2 + _dy**2 + _dz**2) ** 0.5

            if _dist < _CENTER_TOL:
                break

            # Clamp so the pull is firm but not violent
            def _cf(v): return max(-_MAX_CF, min(_MAX_CF, -_SPRING_K * v))
            # Always send — keeps the Haply service streaming state back to us
            await ws.send(json.dumps({"inverse3": [{"device_id": dev_id, "commands": {
                "set_cursor_force": {"vector": {
                    "x": _cf(_dx),
                    "y": _cf(_dy),
                    "z": _cf(_dz)
                }}
            }}]}))
            await asyncio.sleep(0.004)

        # Zero force once done
        await ws.send(json.dumps({"inverse3": [{"device_id": dev_id, "commands": {
            "set_cursor_force": {"vector": {"x": 0.0, "y": 0.0, "z": 0.0}}
        }}]}))
        elapsed = time.monotonic() - _t0
        if elapsed >= _TIMEOUT:
            print(f"Centering timed out after {_TIMEOUT:.0f}s — proceeding anyway")
        else:
            print(f"Inverse3 centred in {elapsed:.1f}s")

        # ---------------------------------------------------------------------------
        # Shared state (written by reader/ft_poller, read by controller — all in the
        # same asyncio thread so no locks needed)
        # ---------------------------------------------------------------------------
        cursor        = dict(_cur)   # use the settled position as the cursor starting point
        ft_force      = [0.0, 0.0, 0.0]   # [Fx, Fy, Fz] in N, robot base frame
        grip_button   = False              # button 'a' state
        hall_value    = HALL_OPEN          # analog squeeze sensor
        gripper_state = "open"             # track to avoid spamming the gripper
        vg_orientation  = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}  # live quaternion
        orient_enabled  = False    # toggled by button 'b' press
        orient_ref      = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}  # captured at toggle-on
        stop            = asyncio.Event()
        # Shared position/orientation state — written by controller(), read by viz_updater()
        smooth     = list(xarm_origin[:3])    # EMA-smoothed Cartesian target (mm)
        smooth_rpy = list(xarm_origin[3:6])   # EMA-smoothed orientation (deg)

        # ---------------------------------------------------------------------------
        async def reader():
            """Drain Haply WS messages — updates cursor position and VersaGrip button."""
            nonlocal grip_button, hall_value, vg_orientation, orient_enabled, orient_ref
            prev_orient_btn = False   # track previous button state for edge detection
            try:
                while not stop.is_set():
                    msg = json.loads(await ws.recv())

                    if "inverse3" in msg and msg["inverse3"]:
                        cursor.update(msg["inverse3"][0]["state"]["cursor_position"])

                    if vg_id and vg_key in msg and msg[vg_key]:
                        vg_state = msg[vg_key][0]["state"]
                        hall_value = vg_state.get("hall", hall_value)

                        orient = vg_state.get("orientation")
                        if orient:
                            vg_orientation.update(orient)

                        btns = vg_state.get("buttons", {})

                        # Gripper — button 'a'
                        new_grip = bool(btns.get("a", False))
                        if new_grip != grip_button:
                            print(f"Gripper button: {grip_button} → {new_grip}")
                            grip_button = new_grip

                        # Orientation toggle — button 'b', rising edge only
                        new_orient_btn = bool(btns.get(ORIENT_BUTTON, False))
                        if new_orient_btn and not prev_orient_btn:   # rising edge = one press
                            orient_enabled = not orient_enabled
                            if orient_enabled:
                                orient_ref = dict(vg_orientation)
                                print("Orientation ENABLED — reference captured")
                            else:
                                print("Orientation DISABLED — returning to home RPY")
                        prev_orient_btn = new_orient_btn

            except websockets.ConnectionClosed:
                stop.set()

        # ---------------------------------------------------------------------------
        async def ft_poller():
            """Poll FT sensor at ~50 Hz and cache result in ft_force."""
            _loop = asyncio.get_event_loop()
            while not stop.is_set():
                # run_in_executor keeps the blocking SDK call off the event loop
                # so it doesn't eat into the 4 ms controller tick budget
                err, data = await _loop.run_in_executor(None, lambda: arm.get_ft_sensor_data(is_raw=False))
                if err == 0 and data:
                    ft_force[0] = FT_ALPHA * data[0] + (1 - FT_ALPHA) * ft_force[0]
                    ft_force[1] = FT_ALPHA * data[1] + (1 - FT_ALPHA) * ft_force[1]
                    ft_force[2] = FT_ALPHA * data[2] + (1 - FT_ALPHA) * ft_force[2]
                await asyncio.sleep(0.02)  # 50 Hz is plenty for haptic feedback

        # ---------------------------------------------------------------------------
        async def controller():
            """250 Hz loop: send haptic forces back to Haply, control gripper, move robot."""
            nonlocal gripper_state
            last       = list(xarm_origin[:3])
            prev_hf    = [0.0, 0.0, 0.0]              # previous haptic force (slew limiter)
            home_rpy   = list(xarm_origin[3:6])       # home orientation to return to
            loop       = asyncio.get_event_loop()

            # Yield to reader() so it can drain any buffered frames that arrived between
            # connection and now.  Then re-anchor haply_ref to the *current* cursor position
            # so the first delta is zero and the robot doesn't jump.
            await asyncio.sleep(0.05)
            haply_ref = dict(cursor)
            print("Haply ref (anchored):", haply_ref)
            print("Hold button 'b' on the VersaGrip to enable wrist rotation.")

            await ws.send(json.dumps({"inverse3": [{"device_id": dev_id, "commands": {}}]}))

            # Position-hold spring state
            hold_point  = dict(cursor)   # anchor the spring springs toward
            prev_cursor = [cursor["x"], cursor["y"], cursor["z"]]

            while not stop.is_set():
                t0 = loop.time()

                # --- Position-hold spring ---
                # While the handle is moving, slide hold_point to the cursor so the
                # spring stays near zero and doesn't fight the user.  When the handle
                # goes still, hold_point freezes and the spring holds it in place.
                vel_sq = ((cursor["x"] - prev_cursor[0]) ** 2 +
                          (cursor["y"] - prev_cursor[1]) ** 2 +
                          (cursor["z"] - prev_cursor[2]) ** 2)
                if vel_sq > HOLD_VEL_THRESHOLD ** 2:
                    hold_point["x"] = cursor["x"]
                    hold_point["y"] = cursor["y"]
                    hold_point["z"] = cursor["z"]
                prev_cursor[0] = cursor["x"]
                prev_cursor[1] = cursor["y"]
                prev_cursor[2] = cursor["z"]

                spring_hfx = -HOLD_SPRING_K * (cursor["x"] - hold_point["x"])
                spring_hfy = -HOLD_SPRING_K * (cursor["y"] - hold_point["y"])
                spring_hfz = -HOLD_SPRING_K * (cursor["z"] - hold_point["z"])

                # --- Haptic force feedback ---
                # 1. Subtract per-axis bias (DC offset visible at rest in UFactory Studio)
                # 2. Apply per-axis deadband (kills residual noise after bias removal)
                # 3. Scale and negate (tool frame → Haply frame; gripper-down flips all axes)
                # 4. Slew-rate limit FT force only (prevents spike on first contact)
                # 5. Add position-hold spring after slew limit so it responds instantly
                def ft_axis(raw, bias, db):
                    v = raw - bias
                    return 0.0 if abs(v) < db else v

                raw_hfx = -ft_axis(ft_force[0],  FT_BIAS[0], FORCE_DEADBAND_X) * FORCE_SCALE
                raw_hfy =  ft_axis(ft_force[1],  FT_BIAS[1], FORCE_DEADBAND_Y) * FORCE_SCALE
                raw_hfz = -ft_axis(ft_force[2],  FT_BIAS[2], FORCE_DEADBAND_Z) * FORCE_SCALE
                # Slew-rate limit on FT contact force only — spring bypasses this
                slew_x = clamp(raw_hfx, prev_hf[0] - MAX_FORCE_SLEW, prev_hf[0] + MAX_FORCE_SLEW)
                slew_y = clamp(raw_hfy, prev_hf[1] - MAX_FORCE_SLEW, prev_hf[1] + MAX_FORCE_SLEW)
                slew_z = clamp(raw_hfz, prev_hf[2] - MAX_FORCE_SLEW, prev_hf[2] + MAX_FORCE_SLEW)
                prev_hf[0], prev_hf[1], prev_hf[2] = slew_x, slew_y, slew_z
                hfx = slew_x + spring_hfx
                hfy = slew_y + spring_hfy
                hfz = slew_z + spring_hfz + GRAVITY_COMP_Z

                cmd = {
                    "inverse3": [{
                        "device_id": dev_id,
                        "commands": {
                            "set_cursor_force": {"vector": {"x": hfx, "y": hfy, "z": hfz}}
                        }
                    }]
                }
                if vg_id:
                    cmd[vg_key] = [{"device_id": vg_id, "commands": {}}]

                try:
                    await ws.send(json.dumps(cmd))
                except websockets.ConnectionClosed:
                    stop.set(); break

                # --- Robot motion ---
                dx = cursor["x"] - haply_ref["x"]
                dy = cursor["y"] - haply_ref["y"]
                dz = cursor["z"] - haply_ref["z"]

                rdx = _YAW_COS * dx - _YAW_SIN * dy
                rdy = _YAW_SIN * dx + _YAW_COS * dy
                tx = xarm_origin[0] + rdx * SCALE
                ty = xarm_origin[1] + rdy * SCALE * ROBOT_CFG["scale_y"]
                tz = xarm_origin[2] + ( dz) * SCALE

                tx = clamp(tx, *WS_LIMITS["x"])
                ty = clamp(ty, *WS_LIMITS["y"])
                tz = clamp(tz, *WS_LIMITS["z"])
                tx = clamp(tx, last[0] - MAX_STEP_MM, last[0] + MAX_STEP_MM)
                ty = clamp(ty, last[1] - MAX_STEP_MM, last[1] + MAX_STEP_MM)
                tz = clamp(tz, last[2] - MAX_STEP_MM, last[2] + MAX_STEP_MM)
                # EMA smoothing: glide toward the clamped target instead of snapping
                smooth[0] = POS_ALPHA * tx + (1 - POS_ALPHA) * smooth[0]
                smooth[1] = POS_ALPHA * ty + (1 - POS_ALPHA) * smooth[1]
                smooth[2] = POS_ALPHA * tz + (1 - POS_ALPHA) * smooth[2]
                tx, ty, tz = smooth[0], smooth[1], smooth[2]

                # --- Wrist orientation from VersaGrip ---
                if orient_enabled:
                    q_ref = [orient_ref["x"], orient_ref["y"], orient_ref["z"], orient_ref["w"]]
                    q_cur = [vg_orientation["x"], vg_orientation["y"], vg_orientation["z"], vg_orientation["w"]]
                    r_delta = R.from_quat(q_cur) * R.from_quat(q_ref).inv()
                    delta_euler = r_delta.as_euler("xyz", degrees=True) * ROT_SCALE
                    d_roll  = clamp(ROT_REMAP[0][1] * delta_euler[ROT_REMAP[0][0]], -MAX_DELTA_DEG, MAX_DELTA_DEG)
                    d_pitch = clamp(ROT_REMAP[1][1] * delta_euler[ROT_REMAP[1][0]], -MAX_DELTA_DEG, MAX_DELTA_DEG)
                    d_yaw   = clamp(ROT_REMAP[2][1] * delta_euler[ROT_REMAP[2][0]], -MAX_DELTA_DEG, MAX_DELTA_DEG)
                    target_rpy = [home_rpy[0] + d_roll,
                                  home_rpy[1] + d_pitch,
                                  home_rpy[2] + d_yaw]
                    # EMA toward target — smooths the initial engage jump and
                    # any small discontinuities during tracking
                    smooth_rpy[0] += ORIENT_TRACK_ALPHA * (target_rpy[0] - smooth_rpy[0])
                    smooth_rpy[1] += ORIENT_TRACK_ALPHA * (target_rpy[1] - smooth_rpy[1])
                    smooth_rpy[2] += ORIENT_TRACK_ALPHA * (target_rpy[2] - smooth_rpy[2])
                else:
                    # Toggled off: slowly interpolate back to home orientation
                    smooth_rpy[0] += ORIENT_RETURN_ALPHA * (home_rpy[0] - smooth_rpy[0])
                    smooth_rpy[1] += ORIENT_RETURN_ALPHA * (home_rpy[1] - smooth_rpy[1])
                    smooth_rpy[2] += ORIENT_RETURN_ALPHA * (home_rpy[2] - smooth_rpy[2])

                target = [tx, ty, tz, smooth_rpy[0], smooth_rpy[1], smooth_rpy[2]]
                ret = arm.set_servo_cartesian(target, is_radian=False)
                if ret != 0:
                    print(f"set_servo_cartesian err {ret}; recovering")
                    arm.clean_warn()
                    arm.clean_error()
                    arm.motion_enable(enable=True)   # required after a fault clear
                    arm.set_mode(1)
                    arm.set_state(0)
                    await asyncio.sleep(0.2)         # let the arm settle after re-enable
                    err, here = arm.get_position(is_radian=False)
                    if err == 0:
                        last = list(here[:3])
                    continue

                last = [tx, ty, tz]
                await asyncio.sleep(max(0, LOOP_DT - (loop.time() - t0)))

        # ---------------------------------------------------------------------------
        async def gripper_driver():
            """Dedicated gripper coroutine — no-ops silently if gripper not attached."""
            nonlocal gripper_state
            if not HAS_GRIPPER:
                while not stop.is_set():
                    await asyncio.sleep(0.1)
                return
            _loop = asyncio.get_event_loop()
            last_button = False
            last_pos    = GRIPPER_OPEN
            while not stop.is_set():
                if USE_BUTTON_GRIP:
                    if grip_button != last_button:
                        last_button   = grip_button
                        target        = GRIPPER_CLOSE if grip_button else GRIPPER_OPEN
                        gripper_state = "closed" if grip_button else "open"
                        await _loop.run_in_executor(
                            None,
                            lambda p=target: arm.set_gripper_g2_position(p, speed=GRIPPER_SPEED)
                        )
                else:
                    # Analog hall sensor → Bio Gripper G2 position (84=open, 0=closed)
                    t      = (hall_value - HALL_OPEN) / max(HALL_CLOSED - HALL_OPEN, 1)
                    t      = max(0.0, min(1.0, t))
                    target = int(GRIPPER_OPEN + t * (GRIPPER_CLOSE - GRIPPER_OPEN))
                    if abs(target - last_pos) >= 2:   # 2 mm deadband avoids chatter
                        last_pos      = target
                        gripper_state = "closed" if target < 42 else "open"
                        await _loop.run_in_executor(
                            None,
                            lambda p=target: arm.set_gripper_g2_position(p, speed=GRIPPER_SPEED)
                        )
                await asyncio.sleep(0.01)   # 100 Hz poll; gripper firmware rate-limits itself

        # ---------------------------------------------------------------------------
        async def viz_updater():
            """Push cursor position, gripper state, and FT force to the PyBullet
            thread at 30 Hz.  Joint angles are polled directly inside
            _run_pybullet_viz() so no async/network round-trip sits between
            the data and the display."""
            while not stop.is_set():
                # Gripper: map open/closed → PyBullet drive angle (0=open, 0.85=closed)
                _viz_gripper[0] = 0.0 if gripper_state == "open" else 0.85

                # Cursor sphere: current commanded TCP in metres
                _viz_cursor[0] = smooth[0] / 1000.0
                _viz_cursor[1] = smooth[1] / 1000.0
                _viz_cursor[2] = smooth[2] / 1000.0

                # FT force (tool frame, bias-subtracted EMA) → force arrow
                _viz_force[0] = ft_force[0] - FT_BIAS[0]
                _viz_force[1] = ft_force[1] - FT_BIAS[1]
                _viz_force[2] = ft_force[2] - FT_BIAS[2]

                await asyncio.sleep(0.033)   # match PyBullet display rate (~30 Hz)

        await asyncio.gather(reader(), ft_poller(), controller(), gripper_driver(), viz_updater())


# On macOS, NSWindow (PyBullet GUI) MUST be created on the main thread.
# Solution: run asyncio in a background thread, PyBullet on the main thread.

def _run_async_main():
    """asyncio event loop — runs in a background thread."""
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Async error: {e}")
    finally:
        _viz_stop.set()   # tell PyBullet to close when async finishes

_async_thread = threading.Thread(target=_run_async_main, daemon=True)
_async_thread.start()

try:
    _run_pybullet_viz()   # blocks on main thread until window closed or _viz_stop set
except KeyboardInterrupt:
    pass
finally:
    _viz_stop.set()
    _async_thread.join(timeout=3.0)
    if HAS_GRIPPER:
        arm.set_gripper_g2_position(GRIPPER_OPEN, speed=GRIPPER_SPEED, wait=False)
    arm.set_ft_sensor_enable(0)
    arm.set_mode(0); arm.set_state(0); arm.disconnect()
