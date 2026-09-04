"""Manager-based SO-101 LiftCube environment for Isaac Lab 2.1.1."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from ...assets import JOINT_ORDER, SO101_FOLLOWER_CFG
from ...task_contract import (
    ACTION_SCALE_RAD,
    DECIMATION,
    EPISODE_SECONDS,
    LIFT_THRESHOLD_M,
    PHYSICS_HZ,
    SUCCESS_HOLD_SECONDS,
)
from . import mdp


CUBE_SIZE_M = 0.04


@configclass
class SO101LiftCubeSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.22, -0.025)),
        spawn=sim_utils.CuboidCfg(
            size=(0.60, 0.60, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.22, 0.12)),
        ),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.30, CUBE_SIZE_M / 2 + 0.002)),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE_M, CUBE_SIZE_M, CUBE_SIZE_M),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.15, 0.05)),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.10)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/jaw",
                name="grasp_center",
                offset=OffsetCfg(pos=(-0.021, -0.070, 0.020)),
            )
        ],
    )

    fixed_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )

    moving_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )


@configclass
class ActionsCfg:
    joint_position_delta = mdp.JointPositionDeltaActionCfg(
        asset_name="robot",
        joint_names=list(JOINT_ORDER),
        scale=ACTION_SCALE_RAD,
        clip={joint_name: (-ACTION_SCALE_RAD, ACTION_SCALE_RAD) for joint_name in JOINT_ORDER},
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_position_rad = ObsTerm(
            func=mdp.joint_position_rad,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True)},
        )
        joint_velocity_rad_s = ObsTerm(
            func=mdp.joint_velocity_rad_s,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True)},
        )
        previous_action_rad = ObsTerm(func=mdp.previous_action_rad)
        end_effector_to_cube_position_m = ObsTerm(func=mdp.end_effector_to_cube_position_m)
        cube_height_relative_to_initial_m = ObsTerm(func=mdp.cube_height_relative_to_initial_m)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_cube = EventTerm(
        func=mdp.reset_cube_pose,
        mode="reset",
        params={
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "z": (0.0, 0.0), "yaw": (-0.35, 0.35)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


@configclass
class RewardsCfg:
    reaching_cube = RewTerm(func=mdp.reaching_cube, weight=2.0, params={"std": 0.12})
    gripper_alignment = RewTerm(
        func=mdp.gripper_cube_alignment,
        weight=1.0,
        params={"xy_std": 0.08, "desired_vertical_gap_m": 0.06, "z_std": 0.08},
    )
    finger_contact = RewTerm(func=mdp.valid_finger_contact, weight=2.0, params={"threshold_n": 0.2})
    lift_progress = RewTerm(
        func=mdp.lift_progress,
        weight=8.0,
        params={"target_delta_z_m": LIFT_THRESHOLD_M},
    )
    lifted = RewTerm(
        func=mdp.object_lifted,
        weight=16.0,
        params={"minimum_delta_z_m": LIFT_THRESHOLD_M},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True)},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    lift_held = DoneTerm(
        func=mdp.lift_held_success,
        params={
            "minimum_delta_z_m": LIFT_THRESHOLD_M,
            "minimum_hold_seconds": SUCCESS_HOLD_SECONDS,
        },
    )
    non_finite = DoneTerm(func=mdp.non_finite_state)
    workspace_exit = DoneTerm(
        func=mdp.workspace_exit,
        params={"xy_limit_m": 0.75, "minimum_z_m": -0.08, "maximum_z_m": 0.80},
    )


@configclass
class SO101LiftCubeEnvCfg(ManagerBasedRLEnvCfg):
    scene: SO101LiftCubeSceneCfg = SO101LiftCubeSceneCfg(num_envs=64, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.decimation = DECIMATION
        self.episode_length_s = EPISODE_SECONDS
        self.sim.dt = 1.0 / PHYSICS_HZ
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.viewer.eye = (0.75, 0.75, 0.55)
        self.viewer.lookat = (0.0, 0.25, 0.15)
