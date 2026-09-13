"""Read-only, policy-rate contact diagnostics; no action or physics setters."""

import json
import numpy as np


def transform_points(points_m, position_m, quaternion_wxyz):
    """Transform link-local points using a unit wxyz quaternion."""
    points = np.asarray(points_m, dtype=float)
    quat = np.asarray(quaternion_wxyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or quat.shape != (4,):
        raise ValueError("Expected Nx3 points and wxyz quaternion")
    if not np.isfinite(points).all() or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1, atol=1e-4):
        raise ValueError("Nonfinite points or non-unit quaternion")
    vector = quat[1:]
    return points + 2 * np.cross(vector, np.cross(vector, points) + quat[0] * points) + position_m


def unpack_contacts(buffers):
    """Exclude unused capacity; preserve shape/filter indices and signed distances."""
    forces, points, normals, separations, counts, starts = [np.asarray(x) for x in buffers]
    if counts.shape != starts.shape:
        raise ValueError("Mismatched contact index shapes")
    rows = []
    for index in np.ndindex(counts.shape):
        count, start = int(counts[index]), int(starts[index])
        if count < 0 or start < 0 or start + count > len(points):
            raise ValueError("Contact buffer bounds exceeded")
        if not count:
            continue
        section = slice(start, start + count)
        values = [value[section] for value in (forces, points, normals, separations)]
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError("Nonfinite active contact data")
        rows.append({"shape_filter_index": list(index), "count": count,
                     "normal_force_n": values[0].reshape(-1).tolist(),
                     "points_w_m": values[1].tolist(), "normals_w": values[2].tolist(),
                     "separation_m": values[3].reshape(-1).tolist()})
    return rows


class GraspAudit:
    """Observe native PhysX contact streams and authored collision mesh vertices."""

    def __init__(self, raw):
        from pxr import Usd, UsdGeom, UsdPhysics
        self.raw = raw
        self.views = {}
        self.meshes = []
        stage = raw.sim.stage
        root = "/World/envs/env_0"
        # Static table filters must resolve to its actual collision prim, not its container.
        table_colliders = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(root + "/Table"))
                           if p.HasAPI(UsdPhysics.CollisionAPI)]
        if not table_colliders:
            raise RuntimeError("No table collision prim found")
        physics_view = raw.scene.sensors["fixed_finger_contact"]._physics_sim_view
        pairs = [("gripper_cube", root + "/Robot/gripper", [root + "/Cube"]),
                 ("jaw_cube", root + "/Robot/jaw", [root + "/Cube"]),
                 ("gripper_table", root + "/Robot/gripper", table_colliders),
                 ("jaw_table", root + "/Robot/jaw", table_colliders),
                 ("cube_table", root + "/Cube", table_colliders)]
        mapping = {}
        for name, sensor, filters in pairs:
            view = physics_view.create_rigid_contact_view(sensor, filter_patterns=filters,
                                                          max_contact_data_count=4096)
            if view.sensor_count < 1 or view.filter_count < 1:
                raise RuntimeError(f"Unresolved contact pair: {name}")
            self.views[name] = view
            mapping[name] = {"sensor_paths": list(view.sensor_paths), "filter_paths": view.filter_paths,
                             "sensor_count": view.sensor_count, "filter_count": view.filter_count,
                             "capacity": view.max_contact_data_count}
        robot = raw.scene["robot"]
        for body in ("gripper", "jaw"):
            body_prim = stage.GetPrimAtPath(root + "/Robot/" + body)
            inverse = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
            for prim in Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies()):
                if "/collisions/" not in str(prim.GetPath()) or not prim.IsA(UsdGeom.Mesh):
                    continue
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * inverse
                local = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix)
                self.meshes.append((str(prim.GetPath()), robot.body_names.index(body), local[:, :3]))
        if not self.meshes:
            raise RuntimeError("No authored finger collision meshes found")
        self.table_center_m = np.asarray(raw.cfg.scene.table.init_state.pos)
        self.table_size_m = np.asarray(raw.cfg.scene.table.spawn.size)
        self.metadata = {"schema": "so101_pick_rl.grasp_contact_audit.v1", "pairs": mapping,
                         "body_names": robot.body_names, "joint_names": robot.joint_names,
                         "soft_joint_limits_rad": robot.data.soft_joint_pos_limits[0].cpu().tolist(),
                         "mesh_paths": [row[0] for row in self.meshes],
                         "table_center_m": self.table_center_m.tolist(), "table_size_m": self.table_size_m.tolist(),
                         "sampling": "30Hz policy boundary, contact stream from last 120Hz physics substep only",
                         "geometry_limit": "Authored mesh vertices, not cooked convex decomposition hulls; vertex clearance is not exact mesh penetration",
                         "contact_limit": "PhysX reported signed separation; negative values are solver-reported overlap, not proof of real-world feasibility"}

    def sample(self, state_index):
        raw = self.raw
        robot = raw.scene["robot"]
        def cpu(value):
            return value.detach().cpu().numpy().copy()
        contacts = {}
        if state_index > 0:
            for name, view in self.views.items():
                buffers = [cpu(value) for value in view.get_contact_data(raw.physics_dt)]
                if int(buffers[4].sum()) >= view.max_contact_data_count:
                    raise RuntimeError(f"Possible contact stream truncation: {name}")
                contacts[name] = unpack_contacts(buffers)
        body_pos_m, body_quat = cpu(robot.data.body_pos_w[0]), cpu(robot.data.body_quat_w[0])
        mesh_clearance = {}
        table_top_m = self.table_center_m[2] + self.table_size_m[2] / 2
        for path, body, local_points in self.meshes:
            world = transform_points(local_points, body_pos_m[body], body_quat[body])
            above = (np.abs(world[:, :2] - self.table_center_m[:2]) <= self.table_size_m[:2] / 2).all(1)
            mesh_clearance[path] = float(world[above, 2].min() - table_top_m) if above.any() else None
        return {"state_index": state_index, "simulation_time_seconds": state_index * raw.step_dt,
                "contacts": contacts, "authored_mesh_vertex_table_clearance_m": mesh_clearance,
                "joint_position_rad": cpu(robot.data.joint_pos[0]).tolist(),
                "body_position_w_m": body_pos_m.tolist(), "body_quaternion_wxyz": body_quat.tolist(),
                "applied_joint_torque_nm": cpu(robot.data.applied_torque[0]).tolist()}

    @staticmethod
    def write_sample(stream, sample):
        stream.write(json.dumps(sample, allow_nan=False, separators=(",", ":")) + "\n")
