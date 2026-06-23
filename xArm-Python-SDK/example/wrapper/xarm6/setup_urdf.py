#!/usr/bin/env python3
"""
setup_urdf.py — Copy and patch the official UFactory URDF files for PyBullet.

The URDFs reference meshes as  package://xarm_description/meshes/...
PyBullet needs plain relative paths.  This script fixes those, copies
everything into urdf/ next to haply_control.py, and you're done.

Usage:
    python setup_urdf.py [path/to/ufactory_usd_urdf_mesh]

If no path is given, the default Downloads location is used.
"""
import os, sys, re, shutil, zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_DIR   = os.path.join(SCRIPT_DIR, "urdf")

# ---------------------------------------------------------------------------
# Locate the source package
# ---------------------------------------------------------------------------
if len(sys.argv) >= 2:
    SRC_ROOT = sys.argv[1]
else:
    SRC_ROOT = os.path.expanduser(
        "~/Downloads/ufactory_usd_urdf_mesh_20250421/ufactory_usd_urdf_mesh")

if not os.path.isdir(SRC_ROOT):
    print(f"ERROR: Source directory not found:\n  {SRC_ROOT}")
    print("Pass the path as an argument:  python setup_urdf.py /path/to/ufactory_usd_urdf_mesh")
    sys.exit(1)

print(f"Source : {SRC_ROOT}")
print(f"Output : {URDF_DIR}\n")

os.makedirs(URDF_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared mesh directory (all robots share xarm_description/meshes/)
# ---------------------------------------------------------------------------
src_meshes = os.path.join(SRC_ROOT, "xarm_description", "meshes")
dst_meshes = os.path.join(URDF_DIR, "meshes")

if not os.path.isdir(src_meshes):
    print(f"ERROR: Mesh directory not found: {src_meshes}")
    sys.exit(1)

if os.path.isdir(dst_meshes):
    print("Removing old meshes/...")
    shutil.rmtree(dst_meshes)

print("Copying meshes...")
shutil.copytree(src_meshes, dst_meshes)
print(f"  → {dst_meshes}/\n")

# ---------------------------------------------------------------------------
# Helper: patch a URDF and write it to urdf/
# ---------------------------------------------------------------------------
def patch_urdf(src_path, out_name):
    """Replace package://xarm_description/meshes/ with meshes/ and save."""
    txt = open(src_path).read()
    # package://xarm_description/meshes/uf850/visual/link1.dae
    #   → meshes/uf850/visual/link1.dae
    txt = re.sub(r'package://xarm_description/meshes/', 'meshes/', txt)
    out_path = os.path.join(URDF_DIR, out_name)
    open(out_path, "w").write(txt)
    print(f"  {src_path}")
    print(f"  → {out_path}")
    return out_path

# ---------------------------------------------------------------------------
# UFactory 850
# ---------------------------------------------------------------------------
print("Processing uf850.urdf...")
uf850_src = os.path.join(SRC_ROOT, "group1_regular_gripper", "uf850", "uf850.urdf")
if not os.path.isfile(uf850_src):
    print(f"  WARNING: not found: {uf850_src}")
else:
    patch_urdf(uf850_src, "uf850.urdf")

# ---------------------------------------------------------------------------
# xArm 6
# ---------------------------------------------------------------------------
print("\nProcessing xarm6.urdf...")
xarm6_src = os.path.join(SRC_ROOT, "group1_regular_gripper", "xarm6", "xarm6.urdf")
if not os.path.isfile(xarm6_src):
    print(f"  WARNING: not found: {xarm6_src}")
else:
    patch_urdf(xarm6_src, "xarm6.urdf")

# ---------------------------------------------------------------------------
# xArm Gripper (Bio Gripper G2 visual stand-in)
# ---------------------------------------------------------------------------
print("\nProcessing xarm_gripper.urdf...")
gripper_src = os.path.join(SRC_ROOT, "group1_regular_gripper", "xarm_gripper", "xarm_gripper.urdf")
if not os.path.isfile(gripper_src):
    print(f"  WARNING: not found: {gripper_src}")
else:
    patch_urdf(gripper_src, "xarm_gripper.urdf")

# ---------------------------------------------------------------------------
# FT Sensor (Onshape export zip — gltf meshes converted to obj for PyBullet)
# ---------------------------------------------------------------------------
FT_ZIP = os.path.expanduser(
    "~/Downloads/触点力矩装配体-带通用转接件.zip")

if not os.path.isfile(FT_ZIP):
    print(f"\nFT sensor zip not found at:\n  {FT_ZIP}\n  Skipping FT sensor URDF.")
else:
    print("\nProcessing FT sensor (Onshape export)...")
    try:
        import trimesh
    except ImportError:
        print("  trimesh not installed — run: pip install trimesh")
        trimesh = None

    if trimesh is not None:
        ft_out = os.path.join(URDF_DIR, "ft_sensor")
        ft_mesh_dir = os.path.join(ft_out, "meshes")
        os.makedirs(ft_mesh_dir, exist_ok=True)

        # Extract zip into a temp staging area
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(FT_ZIP) as zf:
                zf.extractall(tmp)

            src_urdf  = os.path.join(tmp, "pkg_", "urdf",    "pkg_.urdf")
            src_mesh  = os.path.join(tmp, "pkg_", "meshes")

            # Convert every .gltf → .obj
            print("  Converting .gltf meshes to .obj (this may take a moment)...")
            converted = 0
            for fname in os.listdir(src_mesh):
                if not fname.endswith(".gltf"):
                    continue
                src_path = os.path.join(src_mesh, fname)
                out_name = fname.replace(".gltf", ".obj")
                out_path = os.path.join(ft_mesh_dir, out_name)
                try:
                    mesh = trimesh.load(src_path, force="mesh")
                    if hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
                        mesh.export(out_path)
                        converted += 1
                except Exception as e:
                    pass   # skip empty/unloadable meshes silently
            print(f"  Converted {converted} meshes → {ft_mesh_dir}/")

            # Patch the URDF: fix package:// paths and .gltf → .obj
            urdf_txt = open(src_urdf).read()
            urdf_txt = re.sub(r'package://pkg_/meshes/', 'meshes/', urdf_txt)
            urdf_txt = urdf_txt.replace(".gltf", ".obj")
            ft_urdf_path = os.path.join(ft_out, "ft_sensor.urdf")
            open(ft_urdf_path, "w").write(urdf_txt)
            print(f"  → {ft_urdf_path}")

# ---------------------------------------------------------------------------
print("\nDone!  haply_control.py will detect and use these URDFs automatically.")
