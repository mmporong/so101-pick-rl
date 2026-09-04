# Adapted from LeIsaac v0.1.2 (Apache-2.0).

"""SO-101 follower configuration adapted from LeIsaac v0.1.2.

The robot USD is downloaded separately by ``isaaclab/scripts/prepare_assets.py``.
The binary is not redistributed by this repository.
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from ..task_contract import JOINT_ORDER


def get_so101_asset_path() -> Path:
    """Return the pinned SO-101 USD location without mutating the installation."""
    override = os.environ.get("SO101_RL_ASSET_PATH")
    if override:
        return Path(override).expanduser().resolve()

    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("LOCALAPPDATA is unavailable; set SO101_RL_ASSET_PATH explicitly.")
    return Path(local_app_data) / "so101-pick-rl" / "assets" / "leisaac" / "so101_follower.usd"


SO101_FOLLOWER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(get_so101_asset_path()),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={joint_name: 0.0 for joint_name in JOINT_ORDER},
    ),
    actuators={
        "sts3215_arm": ImplicitActuatorCfg(
            joint_names_expr=list(JOINT_ORDER[:-1]),
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=17.8,
            damping=0.60,
        ),
        "sts3215_gripper": ImplicitActuatorCfg(
            joint_names_expr=[JOINT_ORDER[-1]],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=17.8,
            damping=0.60,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
