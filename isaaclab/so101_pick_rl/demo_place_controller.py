"""Bounded position-target placement correction; never writes physical state."""
from __future__ import annotations

import numpy as np

from .grasp_audit import transform_points


def skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def midpoint_jacobian(body_quaternions, spatial_jacobians, geometry):
    result = []
    for i in range(2):
        offset_m = transform_points([geometry["pad_offsets_m"][i]], np.zeros(3), body_quaternions[i])[0]
        spatial = np.asarray(spatial_jacobians[i])[:, :5]
        result.append(spatial[:3] - skew(offset_m) @ spatial[3:])
    return (result[0] + result[1]) / 2


def bounded_cartesian_step(jacobian, position_error_m, angular_jacobian, current_quaternion,
                           reference_quaternion, max_cartesian_step_m=.0005, max_joint_step_rad=.006,
                           vertical_axis=False, delta_lower_rad=None, delta_upper_rad=None):
    error_m = np.array(position_error_m, dtype=float, copy=True)
    error_m *= min(1., max_cartesian_step_m / max(np.linalg.norm(error_m), 1e-12))
    current_axes = transform_points(np.eye(3), np.zeros(3), current_quaternion)
    reference_axes = transform_points(np.eye(3), np.zeros(3), reference_quaternion)
    rotation_error_rad = .5 * np.cross(current_axes, reference_axes).sum(axis=0)
    angular = np.asarray(angular_jacobian)[:, :5]
    if vertical_axis:
        # Five arm joints control XYZ and shaft tilt; yaw remains unconstrained.
        shaft = -current_axes[2]
        rotation_error_rad = np.cross(shaft, [0., 0., -1.])
        angular = (np.eye(3) - np.outer(shaft, shaft)) @ angular
    rotation_error_rad *= min(1., .006 / max(np.linalg.norm(rotation_error_rad), 1e-12))
    # Orientation rows use a design lever to express the angular task in meters.
    lever_m = .05
    matrix = np.concatenate((jacobian, lever_m * angular))
    error = np.concatenate((error_m, lever_m * rotation_error_rad))
    if matrix.shape != (6, 5) or not np.isfinite(matrix).all() or not np.isfinite(error).all():
        raise ValueError("invalid placement Jacobian or error")
    lower = np.full(5, -max_joint_step_rad)
    upper = np.full(5, max_joint_step_rad)
    if delta_lower_rad is not None:
        lower = np.maximum(lower, delta_lower_rad)
        upper = np.minimum(upper, delta_upper_rad)
    if (lower.shape != (5,) or upper.shape != (5,) or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(lower > upper)):
        raise ValueError("invalid joint increment bounds")
    # Re-solve around saturated joints. Clipping only after IK would discard
    # wrist motion while executing the other joints' compensating motion.
    delta_rad = np.zeros(5)
    free = np.ones(5, dtype=bool)
    for _ in range(6):
        active_matrix = matrix[:, free]
        residual = error - matrix[:, ~free] @ delta_rad[~free]
        delta_rad[free] = np.linalg.solve(active_matrix.T @ active_matrix + .01**2 * np.eye(free.sum()),
                                          active_matrix.T @ residual)
        violation = np.maximum(lower - delta_rad, delta_rad - upper)
        violation[~free] = -np.inf
        if np.max(violation) <= 0:
            break
        index = int(np.argmax(violation))
        delta_rad[index] = np.clip(delta_rad[index], lower[index], upper[index])
        free[index] = False
        if not free.any():
            break
    return delta_rad


class ControlledPlace:
    """Replace an impending high release with align/lower/support/open/retreat."""
    def __init__(self, geometry, gate_cfg, place_offset_xy_m=(0., 0.), level_cube=False,
                 minimum_support_force_n=None):
        self.geometry, self.cfg = geometry, gate_cfg
        self.level_cube = level_cube
        self.minimum_support_force_n = (gate_cfg["support_contact_threshold_n"] if minimum_support_force_n is None
                                        else minimum_support_force_n)
        self.place_offset_xy_m = np.asarray(place_offset_xy_m, dtype=float)
        if (self.place_offset_xy_m.shape != (2,) or not np.isfinite(self.place_offset_xy_m).all()
                or np.any(np.abs(self.place_offset_xy_m) > gate_cfg["box_xy_half_extent_m"])):
            raise ValueError("placement offset must remain inside the target region")
        self.phase = "recorded_transition"
        self.active = self.done = False
        self.support_started = False
        self.phase_steps = self.stable_steps = 0
        self.history = []

    def _transition(self, phase):
        self.phase, self.phase_steps, self.stable_steps = phase, 0, 0
        self.support_started |= phase == "support_closed"
        self.history.append(phase)

    def action(self, *, proposed_target_rad, previous_target_rad, q_rad, cube_position_m,
               box_position_m, feature, body_quaternions, spatial_jacobians, lower_rad, upper_rad,
               cube_quaternion=None, body_positions_m=None):
        source_rad = np.asarray(proposed_target_rad, dtype=float)
        previous_rad = np.asarray(previous_target_rad, dtype=float)
        cube_m, box_m = np.asarray(cube_position_m), np.asarray(box_position_m)
        goal_xy_m = box_m[:2] + self.place_offset_xy_m
        if not self.active:
            if not (feature is not None and feature["bilateral"]
                    and feature["lift_m"] >= self.cfg["minimum_lift_m"]
                    and source_rad[-1] - previous_rad[-1] > .03):
                return source_rad.copy()
            self.active = True
            self.target_rad = previous_rad.copy()
            self.reference_quaternion = np.asarray(body_quaternions[0]).copy()
            self.align_z_m = float(cube_m[2])
            self.previous_q_rad = np.asarray(q_rad).copy()
            self._transition("align_closed")
        self.phase_steps += 1
        if self.phase in ("align_closed", "lower_closed", "retreat_open"):
            if self.phase == "align_closed":
                goal_cube_m = np.array([*goal_xy_m, self.align_z_m])
                near = np.linalg.norm(cube_m[:2] - goal_xy_m) < .003
                if near:
                    self._transition("align_settle")
                    return self.target_rad.copy()
            elif self.phase == "lower_closed":
                goal_cube_m = box_m + [0, 0, self.cfg["box_floor_half_height_m"] + feature["cube_half_height_m"] - .0001]
                goal_cube_m[:2] = goal_xy_m
                level = (not self.level_cube or
                         transform_points([[0, 0, 1]], np.zeros(3), cube_quaternion)[0, 2] >= .999)
                if (feature["supported"] and feature["at_rest"] and level
                        and feature["support_force_n"] >= self.minimum_support_force_n):
                    self._transition("support_closed")
                    return self.target_rad.copy()
            else:
                goal_cube_m = cube_m + (self.retreat_midpoint_m - feature["midpoint_w_m"])
                if feature["distance_m"] >= .10:
                    self._transition("stable_open")
                    return self.target_rad.copy()
            jacobian = midpoint_jacobian(body_quaternions, spatial_jacobians, self.geometry)
            if self.phase != "retreat_open" and body_positions_m is not None:
                # The controlled translation is the carried cube center, not
                # the CAD pad midpoint. This is a local grasp assumption only;
                # the physics object remains unconstrained and fully audited.
                spatial = np.asarray(spatial_jacobians[0])[:, :5]
                offset_m = cube_m - np.asarray(body_positions_m[0])
                jacobian = spatial[:3] - skew(offset_m) @ spatial[3:]
            orientation = (cube_quaternion if self.level_cube and self.phase != "retreat_open"
                           else body_quaternions[0])
            if orientation is None:
                raise ValueError("cube leveling requires the observed cube orientation")
            delta_rad = bounded_cartesian_step(jacobian, goal_cube_m - cube_m,
                np.asarray(spatial_jacobians[0])[3:, :5], orientation, self.reference_quaternion,
                vertical_axis=True, delta_lower_rad=np.asarray(lower_rad)[:5] - self.target_rad[:5],
                delta_upper_rad=np.asarray(upper_rad)[:5] - self.target_rad[:5])
            # Dampen the target accumulator using measured joint displacement;
            # fixed PD compliance should not create a chasing limit cycle.
            correction_rad = delta_rad - .5 * (np.asarray(q_rad)[:5] - self.previous_q_rad[:5])
            correction_rad /= max(1., np.abs(correction_rad).max() / .006)
            self.target_rad[:5] += correction_rad
            self.target_rad[:5] = np.clip(self.target_rad[:5], np.asarray(q_rad)[:5] - .25,
                                         np.asarray(q_rad)[:5] + .25)
        elif self.phase == "align_settle":
            near = np.linalg.norm(cube_m[:2] - goal_xy_m) < .005
            # This is a closed-grasp motion transition, not release approval.
            # Release still requires supported low raw velocities in the gate.
            self.stable_steps = self.stable_steps + 1 if near else 0
            if self.stable_steps >= 30:
                self._transition("lower_closed")
        elif self.phase == "support_closed":
            enough_support = feature["supported"] and feature["support_force_n"] >= self.minimum_support_force_n
            if not enough_support:
                self._transition("lower_closed")
                return self.target_rad.copy()
            ready = enough_support and feature["at_rest"] and feature["gentle"]
            self.stable_steps = self.stable_steps + 1 if ready else 0
            if self.stable_steps >= 12:
                self._transition("open_supported")
        elif self.phase == "open_supported":
            self.target_rad[-1] = min(1.2, self.target_rad[-1] + .0005)
            ready = feature["no_contact"] and feature["supported"] and feature["gentle"] and feature["opened"]
            self.stable_steps = self.stable_steps + 1 if ready else 0
            if self.stable_steps >= 12:
                self.retreat_midpoint_m = feature["midpoint_w_m"] + [0, 0, .12]
                self._transition("retreat_open")
        elif self.phase == "stable_open":
            if self.phase_steps >= 90:
                self.done = True
        self.target_rad = np.clip(self.target_rad, lower_rad, upper_rad)
        self.previous_q_rad = np.asarray(q_rad).copy()
        return self.target_rad.copy()
