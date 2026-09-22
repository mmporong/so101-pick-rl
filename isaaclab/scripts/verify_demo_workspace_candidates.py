"""Offline verification of asset-bound point-IK candidates, NOT collision checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import scipy
from scipy.spatial.transform import Rotation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--robot", type=Path, required=True)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
if digest(args.robot) != "64a877c3b82cdc4a48ab8a1f321a2dd3ef7c55d4b10bce222b58c530d978ae58":
    raise ValueError("robot asset hash mismatch")
if digest(args.dataset) != "f41e3bb0b6fc03f5ff8f899c05a38d24407407a130697dfdb6a802e4cb22beae":
    raise ValueError("dataset hash mismatch")

# Load USD from this SDK without starting Kit or altering installed packages.
sdk_root = Path(sys.executable).resolve().parents[2]
usd_libraries = sorted((sdk_root / "extscache").glob("omni.usd.libs-*"))
if len(usd_libraries) != 1:
    raise RuntimeError("run with the preserved Isaac Sim SDK Python containing one omni.usd.libs extension")
sys.path.insert(0, str(usd_libraries[0]))
dll_handles = []
if sys.platform == "win32":
    for suffix in ("bin", "bin/usd", "bin/deps"):
        directory = usd_libraries[0] / suffix
        if directory.is_dir():
            dll_handles.append(os.add_dll_directory(str(directory)))
from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open(str(args.robot))
names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
transforms, lower, upper = [], [], []
for name in names:
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/so101_new_calib/joints/" + name))
    if joint.GetAxisAttr().Get() != "Z":
        raise ValueError("unexpected joint axis in the asset-bound chain")
    quat = joint.GetLocalRot0Attr().Get()
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix()
    matrix[:3, 3] = joint.GetLocalPos0Attr().Get()
    transforms.append(matrix)
    lower.append(np.deg2rad(joint.GetLowerLimitAttr().Get()))
    upper.append(np.deg2rad(joint.GetUpperLimitAttr().Get()))


def fk(q):
    transform = np.eye(4)
    for frame, angle in zip(transforms, q):
        rotation = np.eye(4)
        rotation[:3, :3] = Rotation.from_rotvec([0, 0, angle]).as_matrix()
        transform = transform @ frame @ rotation
    return transform


held = [(0, 50), (0, 127), (4, 100), (19, 127), (89, 150), (319, 150), (412, 100)]
validation = []
with h5py.File(args.dataset, "r") as source:
    for episode, frame in held:
        group = source[f"data/demo_{episode}"]
        q = group["obs/joint_pos"][frame, :5]
        actual = group["obs/ee_frame_state"][frame]
        transform = fk(q)
        actual_rotation = Rotation.from_quat(np.r_[actual[4:7], actual[3]]).as_matrix()
        position_error = float(np.linalg.norm(transform[:3, 3] - actual[:3]))
        rotation_error = float(Rotation.from_matrix(transform[:3, :3].T @ actual_rotation).magnitude())
        if not (position_error < 1e-5 and rotation_error < 1e-5):
            raise ValueError("forward kinematics does not match the recorded gripper pose")
        validation.append({"episode": episode, "frame": frame, "position_error_m": position_error,
                           "rotation_error_rad": rotation_error})
    root_pose = source["data/demo_0/states/articulation/robot/root_pose"][0]
root_rotation = Rotation.from_quat(np.r_[root_pose[4:7], root_pose[3]]).as_matrix()
grasp_offset = np.array([.008, 0, -.089])
candidate_inputs = [
    ("center", [.58, -.35, .0645], [.753753476, 1.213741643, -1.288484974, 1.085882659, -.000549714]),
    ("near_inside", [.55, -.38, .0645], [.749607513, 1.133964793, -1.288485053, 1.459724950, -.000554168]),
]
candidates = []
for name, target, q in candidate_inputs:
    q = np.array(q)
    if not ((q >= lower) & (q <= upper)).all():
        raise ValueError("candidate violates joint limits")
    transform = fk(q)
    point_root = transform[:3, 3] + transform[:3, :3] @ grasp_offset
    point_world = root_pose[:3] + root_rotation @ point_root
    error = float(np.linalg.norm(point_world - target))
    approach = root_rotation @ transform[:3, :3] @ np.array([0, 0, -1])
    tilt = float(np.degrees(np.arccos(np.clip(-approach[2], -1, 1))))
    if not (error < 1e-7 and approach[2] < 0):
        raise ValueError("candidate does not reach the point with a downward approach")
    candidates.append({"name": name, "target_world_m": target, "joint_position_rad": q.tolist(),
                       "point_error_m": error, "approach_axis_world": approach.tolist(),
                       "tilt_from_world_down_deg": tilt, "within_joint_limits": True})
report = {
    "schema": "so101_pick_rl.offline_workspace_candidates.v1", "status": "verified_point_kinematics_only",
    "robot_sha256": digest(args.robot), "dataset_sha256": digest(args.dataset),
    "script_sha256": digest(__file__), "fk_validation": validation,
    "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
    "maximum_fk_position_error_m": max(row["position_error_m"] for row in validation),
    "maximum_fk_rotation_error_rad": max(row["rotation_error_rad"] for row in validation),
    "assumed_cube_center_in_gripper_m": grasp_offset.tolist(), "candidates": candidates,
    "limitations": ["Fixed grasp-point approximation, not certified grasp/contact geometry",
                    "No collision, path, support, opening, retreat, dynamics, or physical-robot validation",
                    "Does not prove that vertical placement is impossible or these tilts are global minima",
                    "Candidate joint angles are offline data, not commands for a real robot"],
}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x", encoding="utf-8") as output:
    json.dump(report, output, indent=2, allow_nan=False)
print(json.dumps(report, indent=2, allow_nan=False))
