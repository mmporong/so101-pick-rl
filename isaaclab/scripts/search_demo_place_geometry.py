"""Asset-bound conservative convex geometry search, not a physical certificate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "isaaclab"))
from so101_pick_rl.demo_bc import file_sha256, load_contract
from so101_pick_rl.demo_pose_probe import separating_axis_clearance, upright_cube_projected_width, validate_pad_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=32)
    parser.add_argument("--open-rad", type=float, default=1.35)
    parser.add_argument("--clearance-m", type=float, default=.002)
    args = parser.parse_args()
    contract, contract_sha = load_contract(ROOT / "common/demo_box_spec.json")
    gate_path = ROOT / "configs/evaluation/demo_box_replay_gate_v2.json"
    gate = json.loads(gate_path.read_text())
    cube_edge = contract["scene"]["cube_edge_m"]
    if gate["cube_size_m"] != [cube_edge] * 3:
        raise ValueError("gate and cube contract disagree")
    pad_path = ROOT / "configs/isaaclab/demo_box_pad_geometry.json"
    pad_geometry = json.loads(pad_path.read_text())
    validate_pad_geometry(pad_geometry, contract["scene"]["robot_sha256"])
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.starts < 1:
        raise ValueError("starts must be positive")
    if not (.26 <= args.open_rad <= 1.35 and .002 <= args.clearance_m <= .02):
        raise ValueError("invalid opening or search clearance")
    if (file_sha256(args.robot) != contract["scene"]["robot_sha256"] or
            file_sha256(args.dataset) != contract["dataset"]["expected_sha256"]):
        raise ValueError("asset or dataset hash mismatch")
    sdk = Path(sys.executable).resolve().parents[2]
    libs = sorted((sdk / "extscache").glob("omni.usd.libs-*"))
    if len(libs) != 1:
        raise RuntimeError("use the preserved Isaac SDK Python")
    sys.path.insert(0, str(libs[0]))
    handles = []
    if sys.platform == "win32":
        handles = [os.add_dll_directory(str(libs[0] / p)) for p in ("bin", "bin/usd", "bin/deps")
                   if (libs[0] / p).is_dir()]
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(args.robot))
    frames, lower, upper = [], [], []
    for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"):
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/so101_new_calib/joints/" + name))
        if joint.GetAxisAttr().Get() != "Z":
            raise ValueError("unexpected joint axis")
        if (np.linalg.norm(joint.GetLocalPos1Attr().Get()) > 1e-8 or
                abs(joint.GetLocalRot1Attr().Get().GetReal() - 1) > 1e-8):
            raise ValueError("nonidentity child joint frame is not supported")
        quat = joint.GetLocalRot0Attr().Get()
        frame = np.eye(4)
        frame[:3, :3] = Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix()
        frame[:3, 3] = joint.GetLocalPos0Attr().Get()
        frames.append(frame)
        lower.append(np.deg2rad(joint.GetLowerLimitAttr().Get()))
        upper.append(np.deg2rad(joint.GetUpperLimitAttr().Get()))

    def rotation_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def fk(q):
        transform = np.eye(4)
        for frame, angle in zip(frames, q):
            transform = transform @ frame @ rotation_z(angle)
        return transform

    errors = []
    with h5py.File(args.dataset, "r") as data:
        root = data["data/demo_0/initial_state/articulation/robot/root_pose"][:].reshape(7)
        for episode, step in ((0, 50), (0, 127), (4, 100), (19, 127), (89, 150), (319, 150), (412, 100)):
            group = data[f"data/demo_{episode}/obs"]
            transform = fk(group["joint_pos"][step, :5])
            pose = group["ee_frame_state"][step]
            position = np.linalg.norm(transform[:3, 3] - pose[:3])
            rotation = Rotation.from_matrix(transform[:3, :3].T @ Rotation.from_quat(np.r_[pose[4:], pose[3]]).as_matrix()).magnitude()
            if max(position, rotation) > 1e-5:
                raise ValueError("FK validation failed")
            errors.append({"episode": episode, "step": step, "position_m": float(position), "rotation_rad": float(rotation)})
    root_tf = np.eye(4)
    root_tf[:3, :3] = Rotation.from_quat(np.r_[root[4:], root[3]]).as_matrix()
    root_tf[:3, 3] = root[:3]
    body_hulls = {}
    for body in ("gripper", "jaw"):
        prim = stage.GetPrimAtPath("/so101_new_calib/" + body)
        inverse = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        hulls = []
        for mesh_prim in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not mesh_prim.IsA(UsdGeom.Mesh) or "/collisions/" not in str(mesh_prim.GetPath()):
                continue
            relative = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * inverse
            mesh_points = np.array([list(relative.Transform(Gf.Vec3d(*point))) for point in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()])
            hull = ConvexHull(mesh_points)
            normals = np.unique(np.round(hull.equations[:, :3], 7), axis=0)
            # A bounded axis subset is conservative: missing separating axes can
            # reject a good pose but cannot turn an overlap into a positive witness.
            normals = normals[np.unique(np.linspace(0, len(normals)-1, min(64, len(normals))).astype(int))]
            hulls.append((mesh_points[hull.vertices], normals))
        if not hulls:
            raise ValueError("collision mesh missing")
        body_hulls[body] = hulls
    opening = np.linspace(0, args.open_rad, 6)
    hand_hulls = list(body_hulls["gripper"])
    for grip in opening:
        tf = frames[5] @ rotation_z(grip)
        hand_hulls.extend((v @ tf[:3, :3].T + tf[:3, 3], n @ tf[:3, :3].T) for v, n in body_hulls["jaw"])
    fixed_pad, jaw_pad = np.array(pad_geometry["pad_offsets_m"])
    open_jaw_tf = frames[5] @ rotation_z(args.open_rad)
    open_gap_m = float((open_jaw_tf[:3, :3] @ jaw_pad + open_jaw_tf[:3, 3] - fixed_pad)[0])
    # Unchanged runtime wall geometry. Floor screening is conservative over all XY.
    wall_centers = np.array([[.514, -.35, .0815], [.646, -.35, .0815],
                             [.58, -.416, .0815], [.58, -.284, .0815]])
    wall_half = np.array([[.006, .066, .035], [.006, .066, .035],
                          [.06, .006, .035], [.06, .006, .035]])

    def geometry(x):
        tf = root_tf @ fk(x[:5])
        # Upright world-aligned cube; tangent to the fixed pad plane. Still not
        # a force-closure, jaw contact, or carried-object orientation certificate.
        projected_width = upright_cube_projected_width(tf[:3, :3], cube_edge)
        offset = np.array([fixed_pad[0] + projected_width / 2, 0, x[5]])
        center = tf[:3, 3] + tf[:3, :3] @ offset
        separations = [separating_axis_clearance(v, n, tf[:3, :3], tf[:3, 3], wall_centers, wall_half)
                       for v, n in hand_hulls]
        # All hull extreme vertices are used for the floor, not the downsampled cloud.
        floor = min(float((v @ tf[:3, :3].T + tf[:3, 3])[:, 2].min() - .0495) for v, _ in hand_hulls)
        return center, floor, float(np.min(separations)), float(tf[2, 2]), projected_width, offset

    def inequalities(x):
        center, floor, wall, down, width, _ = geometry(x)
        return np.r_[.04 - np.abs(center[:2] - [.58, -.35]), floor - args.clearance_m,
                     wall - args.clearance_m, down - .2, open_gap_m - width - gate["minimum_open_clearance_m"]]

    def objective(x):
        center, _, _, down, _, _ = geometry(x)
        return (1 - down) + 10 * np.sum((center[:2] - [.58, -.35])**2) + 100 * (x[5] + .097)**2

    rng = np.random.default_rng(20260922)
    seed = np.array([.766, 1.186, -1.186, .785, -.249, -.101])
    # Retain at least 5 mm overlap with the authored distal pad for an aligned
    # 3 cm cube; the upright projection only increases its local-Z half extent.
    bounds = list(zip(lower[:5], upper[:5])) + [(-.1044, -.094)]
    best, candidates = [], []
    for index in range(args.starts):
        start = seed.copy()
        if index:
            start[:5] += rng.normal(0, [.13, .25, .25, .3, 1.4])
            start[5] = rng.uniform(-.1044, -.094)
        start = np.clip(start, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
        result = minimize(objective, start, method="SLSQP", bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda x: geometry(x)[0][2] - .0645},
                         {"type": "ineq", "fun": inequalities}],
            options={"maxiter": 160, "ftol": 1e-9})
        center, floor, wall, down, width, offset = geometry(result.x)
        violation = max(abs(center[2] - .0645), max(0, -float(inequalities(result.x).min())))
        row = {"name": f"geometry_{index:02d}", "joint_position_rad": result.x[:5].tolist(),
               "assumed_cube_center_in_gripper_m": offset.tolist(),
               "assumed_cube_orientation_wxyz": [1., 0., 0., 0.],
               "open_gap_m": open_gap_m, "projected_cube_width_m": float(width),
               "target_world_m": center.tolist(), "hull_floor_clearance_m": floor,
               "hull_wall_separation_witness_m": wall, "tilt_from_world_down_deg": float(np.degrees(np.arccos(np.clip(down, -1, 1)))),
               "constraint_violation_m_or_cosine": violation, "solver_success": bool(result.success)}
        best.append(row)
        if violation < 1e-5 and all(np.linalg.norm(result.x[:5] - np.array(c["joint_position_rad"])) > .03 for c in candidates):
            candidates.append(row)
        print(json.dumps({"start": index, "violation": violation, "found": len(candidates)}), flush=True)
    report = {"schema": "so101_pick_rl.convex_geometry_candidates.v1", "status": "convex_geometry_only",
              "robot_sha256": file_sha256(args.robot), "dataset_sha256": file_sha256(args.dataset),
              "task_contract_sha256": contract_sha, "script_sha256": file_sha256(__file__),
              "gate_sha256": file_sha256(gate_path), "pad_geometry_sha256": file_sha256(pad_path),
              "helper_sha256": file_sha256(ROOT / "isaaclab/so101_pick_rl/demo_pose_probe.py"),
              "search": {"seed": 20260922, "starts": args.starts, "open_rad": args.open_rad,
                         "clearance_m": args.clearance_m},
              "fk_validation": errors, "hull_count_including_opening_samples": len(hand_hulls), "opening_samples_rad": opening.tolist(),
              "candidates": candidates, "best_attempts": sorted(best, key=lambda x: x["constraint_violation_m_or_cosine"])[:5],
              "limitations": ["Authored mesh convex hulls, not cooked PhysX hulls or continuous collision detection",
                              "Conservative separating-axis subset may reject feasible poses; opening sampled at six angles",
                              "Only hand geometry screened; no arm/self collision certificate",
                              "Proposed cube grasp transform and upright support center, not contact-validated grasp",
                              "No approach, carried cube, supported release, retreat or task success"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"candidates": candidates, "best_attempts": report["best_attempts"]}), flush=True)


if __name__ == "__main__":
    main()
