"""Register the SO-101 LiftCube task."""

import gymnasium as gym

from ...task_contract import TASK_ID
from . import agents


gym.register(
    id=TASK_ID,
    entry_point=f"{__name__}.environment:SO101LiftCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SO101LiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101LiftCubePPORunnerCfg",
    },
)
