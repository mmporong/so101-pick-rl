"""Register full Pick & Place independently of the historical LiftCube task."""

import gymnasium as gym

gym.register(
    id="SO101-PickPlace-v0",
    entry_point=f"{__name__}.environment:SO101PickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SO101PickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg:SO101PickPlacePPORunnerCfg",
    },
)
