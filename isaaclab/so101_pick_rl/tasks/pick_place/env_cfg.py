"""Full Pick & Place task; retain LiftCube's verified robot and physics configuration."""

import json

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm, EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.utils import configclass

from ...task_contract import REPOSITORY_ROOT
from ..lift_cube.env_cfg import SO101LiftCubeEnvCfg, ObservationsCfg, EventCfg, RewardsCfg, TerminationsCfg
from ..lift_cube.agents.rsl_rl_ppo_cfg import SO101LiftCubePPORunnerCfg
from . import mdp

SPEC = json.loads((REPOSITORY_ROOT / "common" / "pick_place_spec.json").read_text())


@configclass
class PickPlaceObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        cube_to_target_position_m = ObsTerm(func=mdp.cube_to_target)
        cube_linear_angular_velocity = ObsTerm(func=mdp.cube_velocity)
        finger_contact_force_n = ObsTerm(func=mdp.contact_forces)
        episode_phase_state = ObsTerm(func=mdp.phase_state)

    policy: PolicyCfg = PolicyCfg()


@configclass
class PickPlaceEventsCfg(EventCfg):
    reset_cube = EventTerm(func=mdp.reset_pick_place, mode="reset", params={
        "pose_range": {"x": (-0.03, 0.01), "y": (-0.02, 0.02), "z": (0.0, 0.0), "yaw": (-0.2, 0.2)},
        "velocity_range": {}, "asset_cfg": SceneEntityCfg("cube"),
    })


@configclass
class PickPlaceRewardsCfg(RewardsCfg):
    lifted = None
    reaching_cube = RewTerm(func=mdp.staged_reward, weight=2.0, params={"name": "reaching_cube"})
    gripper_alignment = RewTerm(func=mdp.staged_reward, weight=1.0, params={"name": "gripper_alignment"})
    finger_contact = RewTerm(func=mdp.staged_reward, weight=2.0, params={"name": "finger_contact"})
    lift_progress = RewTerm(func=mdp.staged_reward, weight=8.0, params={"name": "lift_progress"})
    transport = RewTerm(func=mdp.staged_reward, weight=12.0, params={"name": "transport"})
    placement = RewTerm(func=mdp.staged_reward, weight=16.0, params={"name": "placement"})
    gripper_opening = RewTerm(func=mdp.staged_reward, weight=24.0, params={"name": "gripper_opening"})
    release_and_retreat = RewTerm(func=mdp.staged_reward, weight=24.0, params={"name": "release_and_retreat"})
    stable_placement = RewTerm(func=mdp.staged_reward, weight=80.0, params={"name": "stable_placement"})
    terminal_success = RewTerm(func=mdp.terminal_success_bonus, weight=SPEC["reward"]["terminal_success_bonus"])

    def __post_init__(self):
        for name, rate in SPEC["reward"]["positive_rates"].items():
            getattr(self, name).weight = rate


@configclass
class PickPlaceTerminationsCfg(TerminationsCfg):
    lift_held = None
    pick_place_success = DoneTerm(func=mdp.pick_place_success)


@configclass
class SO101PickPlaceEnvCfg(SO101LiftCubeEnvCfg):
    observations: PickPlaceObservationsCfg = PickPlaceObservationsCfg()
    events: PickPlaceEventsCfg = PickPlaceEventsCfg()
    rewards: PickPlaceRewardsCfg = PickPlaceRewardsCfg()
    terminations: PickPlaceTerminationsCfg = PickPlaceTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        control = SPEC["control"]
        if control["physics_hz"] % control["policy_hz"]:
            raise ValueError("Pick & Place physics rate must be divisible by policy rate")
        self.decimation = control["physics_hz"] // control["policy_hz"]
        self.sim.dt = 1.0 / control["physics_hz"]
        self.sim.render_interval = self.decimation
        action = self.actions.joint_position_delta
        action.joint_names = list(SPEC["robot"]["joint_order"])
        action.scale = control["action"]["clip_abs_rad"]
        action.clip = {name: (-action.scale, action.scale) for name in action.joint_names}
        self.episode_length_s = SPEC["task"]["episode_seconds"]
        self.scene.target = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Target",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.10, 0.26, 0.001)),
            spawn=sim_utils.CuboidCfg(
                size=(0.06, 0.06, 0.002),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2)),
            ),
        )


@configclass
class SO101PickPlacePPORunnerCfg(SO101LiftCubePPORunnerCfg):
    experiment_name = "so101_pick_place"
    run_name = "scratch"

    def __post_init__(self):
        self.policy.actor_hidden_dims = [256, 128, 64]
        self.policy.critic_hidden_dims = [256, 128, 64]
        self.algorithm.gamma = SPEC["reward"]["discount_gamma"]
