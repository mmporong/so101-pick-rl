"""Batched Isaac Lab runtime for the DemoBox absolute-target task.

The caller owns :class:`isaaclab.app.AppLauncher` and must construct this
runtime only after Kit has started.  This module deliberately keeps Isaac Lab,
Isaac Sim, USD, and Torch imports inside ``DemoBoxRuntime.__init__`` so data
and contract tooling remains usable without the simulator environment.

State is written only by :meth:`reset`.  :meth:`step` applies normalized
absolute joint targets and lets PhysX advance the scene; it never overwrites a
robot or object pose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .demo_action_contract import JOINT_NAMES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "common" / "demo_box_spec.json"


class DemoBoxRuntime:
    """Vectorized 60 Hz DemoBox scene using environment-local observations."""

    def __init__(
        self,
        snapshot: Path,
        num_envs: int,
        device: str = "cuda:0",
        *,
        contract_path: Path | None = None,
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty Torch device string")

        snapshot = Path(snapshot).resolve()
        robot_path = snapshot / "assets" / "robots" / "so101_follower.usd"
        scene_path = snapshot / "assets" / "scenes" / "table_with_cube" / "scene.usd"
        if not robot_path.is_file() or not scene_path.is_file():
            raise FileNotFoundError("snapshot must contain the preserved robot and table scene USD assets")

        from .demo_bc import file_sha256, load_contract

        contract, contract_sha256 = load_contract(contract_path or DEFAULT_CONTRACT)
        scene_contract = contract.get("scene", {})
        expected_assets = {
            "robot": (robot_path, scene_contract.get("robot_sha256")),
            "scene": (scene_path, scene_contract.get("scene_sha256")),
        }
        for name, (path, expected_sha256) in expected_assets.items():
            if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
                raise ValueError(f"contract scene.{name}_sha256 must be a SHA-256 string")
            actual_sha256 = file_sha256(path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"{name} asset SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
                )

        # These imports require an already-running Isaac Sim application.
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        action_cfg = contract.get("action", {})
        if tuple(action_cfg.get("joint_names", ())) != JOINT_NAMES:
            raise ValueError("DemoBox contract joint order does not match the source robot")
        if action_cfg.get("dimension") != len(JOINT_NAMES):
            raise ValueError("DemoBox contract action dimension must be six")
        if action_cfg.get("type") != "normalized_absolute_joint_targets":
            raise ValueError("DemoBox runtime requires normalized absolute joint targets")
        observation_cfg = contract.get("observation", {})
        if observation_cfg.get("dimension") != 34:
            raise ValueError("DemoBox runtime requires the 34-dimensional observation contract")

        lower = torch.as_tensor(action_cfg.get("lower_rad"), dtype=torch.float32, device=device)
        upper = torch.as_tensor(action_cfg.get("upper_rad"), dtype=torch.float32, device=device)
        if lower.shape != (6,) or upper.shape != (6,) or not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
            raise ValueError("contract joint limits must be finite six-element arrays")
        if not torch.all(lower < upper):
            raise ValueError("contract joint limits must be strictly ordered")

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(robot_path),
                    activate_contact_sensors=False,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=True,
                        solver_position_iteration_count=4,
                        solver_velocity_iteration_count=4,
                        fix_root_link=True,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.35, -0.64, 0.01),
                    rot=(0.0, 0.0, 0.0, 1.0),
                    joint_pos={name: 0.0 for name in JOINT_NAMES},
                ),
                actuators={
                    "arm": ImplicitActuatorCfg(
                        joint_names_expr=list(JOINT_NAMES[:-1]),
                        effort_limit_sim=10.0,
                        velocity_limit_sim=10.0,
                        stiffness=17.8,
                        damping=0.60,
                    ),
                    "gripper": ImplicitActuatorCfg(
                        joint_names_expr=["gripper"],
                        effort_limit_sim=10.0,
                        velocity_limit_sim=10.0,
                        stiffness=17.8,
                        damping=0.60,
                    ),
                },
                soft_joint_pos_limit_factor=1.0,
            )
            source_scene = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Scene",
                spawn=sim_utils.UsdFileCfg(usd_path=str(scene_path)),
            )
            box_target = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BoxTarget",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.58, -0.35, 0.0455)),
                spawn=sim_utils.CuboidCfg(
                    size=(0.12, 0.12, 0.008),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.45, 0.85)),
                ),
            )
            light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(intensity=1000.0),
            )

        cfg = SceneCfg(num_envs=num_envs, env_spacing=8.0)
        stage_asset = Usd.Stage.Open(str(scene_path))
        if stage_asset is None:
            raise ValueError(f"failed to open source scene USD: {scene_path}")
        rigid_paths: dict[str, str] = {}
        for prim in stage_asset.Traverse():
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            name = prim.GetName()
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            rotation = matrix.ExtractRotationQuat().GetNormalized()
            relative = str(prim.GetPath()).split("/", 2)[-1]
            prim_path = "{ENV_REGEX_NS}/Scene/" + relative
            position = tuple(float(value) for value in matrix.ExtractTranslation())
            quaternion = (float(rotation.GetReal()), *(float(value) for value in rotation.GetImaginary()))
            setattr(
                cfg,
                name,
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=quaternion),
                ),
            )
            rigid_paths[name] = prim_path
        if set(rigid_paths) != {"cube"}:
            raise ValueError(f"unexpected source scene rigid entities: {rigid_paths}")

        for name, size, position in (
            ("Left", (0.012, 0.132, 0.07), (0.514, -0.35, 0.0815)),
            ("Right", (0.012, 0.132, 0.07), (0.646, -0.35, 0.0815)),
            ("Front", (0.12, 0.012, 0.07), (0.58, -0.416, 0.0815)),
            ("Back", (0.12, 0.012, 0.07), (0.58, -0.284, 0.0815)),
        ):
            setattr(
                cfg,
                "box_wall_" + name,
                AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/Box" + name,
                    init_state=AssetBaseCfg.InitialStateCfg(pos=position),
                    spawn=sim_utils.CuboidCfg(
                        size=size,
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    ),
                ),
            )

        sim_cfg = sim_utils.SimulationCfg(dt=1 / 60, render_interval=1, device=device)
        sim_cfg.physx.bounce_threshold_velocity = 0.01
        sim_cfg.physx.solver_type = 1
        sim_cfg.physx.min_position_iteration_count = 1
        sim_cfg.physx.friction_correlation_distance = 0.00625
        sim = sim_utils.SimulationContext(sim_cfg)
        physics_api = PhysxSchema.PhysxSceneAPI(sim.stage.GetPrimAtPath(sim_cfg.physics_prim_path))
        external_force_attr = physics_api.CreateEnableExternalForcesEveryIterationAttr(False)
        if external_force_attr.Get() is not False:
            raise ValueError("source TGS external-force setting was not applied")
        scene = InteractiveScene(cfg)
        sim.reset()
        scene.update(sim_cfg.dt)

        robot = scene["robot"]
        if tuple(robot.joint_names) != JOINT_NAMES:
            raise ValueError(f"source robot joint order mismatch: {robot.joint_names}")
        source_limits = robot.data.soft_joint_pos_limits[0]
        expected_limits = torch.stack((lower, upper), dim=-1)
        if source_limits.shape != expected_limits.shape or not torch.allclose(
            source_limits, expected_limits, atol=1e-6, rtol=0
        ):
            raise ValueError("contract joint limits do not match the source robot USD")

        self.torch = torch
        self.sim = sim
        self.scene = scene
        self.robot = robot
        self.cube = scene["cube"]
        self.box = scene["box_target"]
        self.dt = sim_cfg.dt
        self.num_envs = num_envs
        self.device = torch.device(sim.device)
        self.contract = contract
        self.contract_sha256 = contract_sha256
        self.lower = lower
        self.upper = upper
        self.previous_target = torch.zeros((num_envs, 6), dtype=torch.float32, device=self.device)
        self.source_rigid_paths = rigid_paths
        self._closed = False

    def _matrix(self, value: Any, columns: int, name: str):
        tensor = self.torch.as_tensor(value, dtype=self.torch.float32, device=self.device)
        if tensor.shape != (self.num_envs, columns) or not self.torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be a finite ({self.num_envs}, {columns}) tensor")
        return tensor

    def _state_to_device(self, value: Any, path: str = "initial_state") -> Any:
        if isinstance(value, dict):
            return {key: self._state_to_device(child, f"{path}/{key}") for key, child in value.items()}
        tensor = self.torch.as_tensor(value, device=self.device)
        if tensor.ndim < 1 or tensor.shape[0] != self.num_envs or not self.torch.isfinite(tensor).all():
            raise ValueError(f"{path} must be finite and have leading dimension {self.num_envs}")
        return tensor

    def reset(self, initial_state: dict, previous_target):
        """Reset every environment from recorded state and return ``(N, 34)``."""
        if self._closed:
            raise RuntimeError("DemoBoxRuntime is closed")
        if not isinstance(initial_state, dict):
            raise ValueError("initial_state must be a nested dictionary")
        target = self._matrix(previous_target, 6, "previous_target")
        if self.torch.any(target < self.lower) or self.torch.any(target > self.upper):
            raise ValueError("previous_target is outside the source robot joint limits")
        state = self._state_to_device(initial_state)
        self.scene.reset()
        self.scene.reset_to(state, is_relative=True)
        self.robot.set_joint_velocity_target(self.torch.zeros_like(self.robot.data.joint_vel))
        self.robot.set_joint_position_target(target)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(self.dt)
        self.previous_target.copy_(target)
        return self.observe()

    def observe(self):
        """Return the contract observation in environment-local coordinates."""
        if self._closed:
            raise RuntimeError("DemoBoxRuntime is closed")
        origins = self.scene.env_origins
        cube_pose = self.cube.data.root_state_w[:, :7].clone()
        cube_pose[:, :3] -= origins
        box_position = self.box.data.root_pos_w - origins
        observation = self.torch.cat(
            (
                self.robot.data.joint_pos,
                self.robot.data.joint_vel,
                self.previous_target,
                cube_pose,
                self.cube.data.root_state_w[:, 7:13],
                box_position,
            ),
            dim=-1,
        )
        if observation.shape != (self.num_envs, 34) or not self.torch.isfinite(observation).all():
            raise RuntimeError("DemoBox observation is nonfinite or violates the (N, 34) contract")
        return observation

    def step(self, actions):
        """Advance one 60 Hz physics step from normalized absolute targets."""
        if self._closed:
            raise RuntimeError("DemoBoxRuntime is closed")
        action = self._matrix(actions, 6, "actions")
        if self.torch.any(action < -1) or self.torch.any(action > 1):
            raise ValueError("normalized actions must lie within [-1, 1]")
        target = self.lower + (action + 1) * (self.upper - self.lower) / 2

        # Preserve the source runtime's nearest-rigid-object effort policy.
        objects = list(self.scene.rigid_objects.values())
        positions = self.torch.stack([obj.data.root_pos_w for obj in objects])
        masses = self.torch.stack([obj.data.default_mass[:, 0].to(self.device) for obj in objects])
        nearest = self.torch.linalg.vector_norm(
            positions - self.robot.data.body_pos_w[:, -1].unsqueeze(0), dim=-1
        ).argmin(0)
        env_ids = self.torch.arange(self.num_envs, device=self.device)
        effort = masses[nearest, env_ids] / 0.15
        current = self.robot.data.joint_effort_limits[:, -1]
        effort = self.torch.where((effort - current).abs() > 0.1, effort, current)
        self.robot.write_joint_effort_limit_to_sim(effort.unsqueeze(-1), joint_ids=[5])
        self.robot.set_joint_position_target(target)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.dt)
        self.previous_target.copy_(target)
        return self.observe()

    def close(self) -> None:
        """Release SimulationContext resources; the caller closes its Kit app."""
        if self._closed:
            return
        self._closed = True
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
        self.sim.stop()
