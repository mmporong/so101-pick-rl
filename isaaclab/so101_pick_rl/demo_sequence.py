"""Geometric, contact-ordered audit for the separate DemoBox replay experiment."""
from __future__ import annotations

import math
import numpy as np

from .grasp_audit import transform_points
from .demo_replay_metrics import loaded_opposing_sides


def force_sum(contacts, pair):
    rows = contacts[pair]  # Missing contact channels must fail closed.
    values = np.asarray([v for row in rows for v in row["normal_force_n"]], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("invalid normal force")
    return float(values.sum())


def finger_geometry(record, geometry, cube_size_m):
    positions = np.asarray(record["finger_body_position_w_m"], dtype=float)
    quaternions = np.asarray(record["finger_body_quaternion_wxyz"], dtype=float)
    cube = np.asarray(record["cube_pose"], dtype=float)
    if positions.shape != (2, 3) or quaternions.shape != (2, 4) or cube.shape != (7,):
        raise ValueError("invalid finger or cube pose shapes")
    if not all(np.isfinite(v).all() for v in (positions, quaternions, cube)):
        raise ValueError("nonfinite body state")
    pads = np.array([transform_points([geometry["pad_offsets_m"][i]], positions[i], quaternions[i])[0]
                     for i in range(2)])
    normals = np.array([transform_points([geometry["inward_normals"][i]], np.zeros(3), quaternions[i])[0]
                        for i in range(2)])
    cube_axes = transform_points(np.eye(3), np.zeros(3), cube[3:])
    projected_width_m = float(np.abs(cube_axes @ normals[0]) @ np.asarray(cube_size_m))
    gap_m = float(np.dot(pads[1] - pads[0], normals[0]))
    approach = transform_points([geometry["approach_axis_gripper_local"]], np.zeros(3), quaternions[0])[0]
    return {"pads_w_m": pads, "midpoint_w_m": pads.mean(axis=0),
            "distance_m": float(np.linalg.norm(pads.mean(axis=0) - cube[:3])),
            "gap_m": gap_m, "projected_cube_width_m": projected_width_m,
            "cube_half_height_m": float(np.abs(cube_axes[:, 2]) @ np.asarray(cube_size_m) / 2),
            "inside_pad_planes": bool(np.dot(cube[:3] - pads[0], normals[0]) >= 0
                                      and np.dot(cube[:3] - pads[1], normals[1]) >= 0),
            "approach_world_z": float(approach[2])}


def sequence_features(record, geometry, cfg):
    finger = finger_geometry(record, geometry, cfg["cube_size_m"])
    contacts = record["contacts"]
    forces_n = np.array([force_sum(contacts, pair) for pair in ("gripper_cube", "jaw_cube")])
    box_m = np.asarray(record["pre_step_state"]["box_position_w_m"], dtype=float)
    cube_m = np.asarray(record["cube_pose"][:3], dtype=float)
    linear = np.asarray(record["cube_linear_velocity"], dtype=float)
    angular = np.asarray(record["cube_angular_velocity"], dtype=float)
    if linear.shape != (3,) or angular.shape != (3,):
        raise ValueError("velocity must contain three world-frame components")
    linear_m_s = np.linalg.norm(linear)
    angular_rad_s = np.linalg.norm(angular)
    q_rad = np.asarray(record["joint_pos"], dtype=float)
    if (q_rad.shape != (6,) or not np.isfinite(q_rad).all() or box_m.shape != (3,)
            or not np.isfinite(box_m).all() or not math.isfinite(linear_m_s)
            or not math.isfinite(angular_rad_s) or not math.isfinite(record["lift_m"])):
        raise ValueError("nonfinite or malformed sequence sample")
    rest_z_m = box_m[2] + cfg["box_floor_half_height_m"] + finger["cube_half_height_m"]
    above = bool((np.abs(cube_m[:2] - box_m[:2]) <= cfg["box_xy_half_extent_m"]).all())
    at_rest = above and abs(cube_m[2] - rest_z_m) <= cfg["rest_height_tolerance_m"]
    gentle = (linear_m_s <= cfg["maximum_release_linear_speed_m_s"]
              and angular_rad_s <= cfg["maximum_release_angular_speed_rad_s"])
    pairs = ("gripper_cube", "jaw_cube", "gripper_table", "jaw_table", "gripper_box", "jaw_box", "cube_box")
    separations = [v for pair in pairs for row in contacts[pair] for v in row["separation_m"]]
    if not all(math.isfinite(v) for v in separations):
        raise ValueError("invalid contact separation")
    return {**finger, "bilateral": bool((forces_n > cfg["loaded_contact_threshold_n"]).all()),
            "no_contact": bool((forces_n < cfg["release_contact_threshold_n"]).all()),
            "opposing_sides": loaded_opposing_sides(contacts, record["cube_pose"], cfg["cube_size_m"],
                                                    cfg["loaded_contact_threshold_n"]),
            "opened": finger["gap_m"] >= finger["projected_cube_width_m"] + cfg["minimum_open_clearance_m"],
            "above_box": above, "at_rest": at_rest, "gentle": gentle,
            "supported": force_sum(contacts, "cube_box") > cfg["support_contact_threshold_n"],
            "support_force_n": force_sum(contacts, "cube_box"),
            "hand_box_force_n": force_sum(contacts, "gripper_box") + force_sum(contacts, "jaw_box"),
            "hand_box_clear": (force_sum(contacts, "gripper_box") + force_sum(contacts, "jaw_box")
                               <= cfg["maximum_hand_box_force_n"]),
            "penetration_safe": not separations or min(separations) >= -cfg["maximum_penetration_m"],
            "wrist_safe": abs(math.degrees(q_rad[3])) <= cfg["maximum_abs_wrist_flex_deg"],
            "lift_m": record["lift_m"], "cube_position_m": cube_m,
            "linear_speed_m_s": float(linear_m_s), "angular_speed_rad_s": float(angular_rad_s),
            "height_above_rest_m": float(cube_m[2] - rest_z_m),
            "gripper_opening": bool(q_rad[-1] > record["pre_step_state"]["joint_position_rad"][-1] + 1e-5
                                    or record["executed_target"][-1] > record["pre_step_state"]["previous_target_rad"][-1] + 1e-5)}


class DemoSequenceGate:
    """One episode, no smoothing across missed contacts or failure recovery."""
    def __init__(self, cfg, initial_cube_position_m):
        self.cfg = cfg
        self.initial_cube_m = np.asarray(initial_cube_position_m, dtype=float)
        if self.initial_cube_m.shape != (3,) or not np.isfinite(self.initial_cube_m).all():
            raise ValueError("invalid initial cube position")
        self.opened = self.grasped = self.picked = self.released = False
        self.lift_steps = self.stable_steps = 0
        self.previous_release_ready = False
        self.failures = set()
        self.events = {}
        self.samples = 0
        self.required_lift_steps = math.ceil(cfg["minimum_lift_hold_s"] / cfg["physics_dt_s"])
        self.required_stable_steps = math.ceil(cfg["minimum_stable_hold_s"] / cfg["physics_dt_s"])

    def update(self, feature):
        self.samples += 1
        f, cfg = feature, self.cfg
        if not f["penetration_safe"]:
            self.failures.add("penetration_limit")
        if not f["wrist_safe"]:
            self.failures.add("wrist_limit")
        if not f["hand_box_clear"]:
            self.failures.add("loaded_hand_box_collision")
        near = f["distance_m"] <= cfg["pregrasp_maximum_distance_m"]
        if not self.grasped:
            self.opened &= near
        # Arm opening before processing this sample's grasp, never after it.
        if (self.opened and not f["opened"] and f["opposing_sides"] and f["bilateral"]
                and f["inside_pad_planes"] and near):
            self.grasped = True
            self.events.setdefault("grasp_step", self.samples)
        if not self.grasped and f["opened"] and f["no_contact"] and near:
            self.opened = True
            self.events.setdefault("open_approach_step", self.samples)
        if not self.grasped and np.linalg.norm(f["cube_position_m"][:2] - self.initial_cube_m[:2]) > cfg["maximum_pregrasp_cube_xy_motion_m"]:
            self.failures.add("cube_pushed_before_grasp")
        if self.grasped and not self.picked and not f["bilateral"]:
            self.grasped = self.opened = False
            self.lift_steps = 0
        lifted = self.grasped and f["bilateral"] and f["lift_m"] >= cfg["minimum_lift_m"]
        self.lift_steps = self.lift_steps + 1 if lifted else 0
        if self.lift_steps >= self.required_lift_steps:
            self.picked = True
            self.events.setdefault("qualified_pick_step", self.samples)
        if self.picked and not self.released:
            if f["bilateral"] and (not f["inside_pad_planes"] or f["approach_world_z"] >= 0):
                self.failures.add("invalid_carry_geometry")
            if not f["above_box"] and f["lift_m"] < cfg["minimum_carry_clearance_m"]:
                self.failures.add("dragged_outside_box")
            if not f["bilateral"] and not (f["at_rest"] and f["gentle"] and f["supported"]):
                self.failures.add("grasp_lost_before_supported_placement")
            if f["no_contact"] and not self.previous_release_ready:
                self.failures.add("unsupported_or_fast_release")
            if (f["no_contact"] and self.previous_release_ready and f["at_rest"]
                    and f["gentle"] and f["supported"] and f["gripper_opening"]):
                self.released = True
                self.events.setdefault("controlled_release_step", self.samples)
        stable = (self.released and f["no_contact"] and f["at_rest"] and f["gentle"]
                  and f["supported"] and f["opened"] and f["distance_m"] >= cfg["minimum_ee_retreat_m"])
        if self.released and not (f["no_contact"] and f["at_rest"] and f["gentle"] and f["supported"]):
            self.failures.add("post_release_instability")
        self.stable_steps = self.stable_steps + 1 if stable else 0
        self.previous_release_ready = bool(not f["no_contact"] and f["at_rest"] and f["gentle"] and f["supported"])
        passed = self.stable_steps >= self.required_stable_steps and not self.failures
        if passed:
            self.events.setdefault("success_step", self.samples)
        return passed

    def report(self):
        return {"schema": "so101_pick_rl.demo_box_sequence_result.v1", "samples": self.samples,
                "events": dict(self.events), "failures": sorted(self.failures),
                "picked": self.picked, "released": self.released,
                "stable_seconds": self.stable_steps * self.cfg["physics_dt_s"],
                "normal_grasp_gate_pass": "success_step" in self.events and not self.failures,
                "real_robot_validated": False}
