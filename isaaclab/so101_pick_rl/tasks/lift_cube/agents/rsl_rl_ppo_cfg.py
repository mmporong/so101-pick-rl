"""RSL-RL PPO configuration for the privileged-state baseline."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class SO101LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10
    save_interval = 5
    experiment_name = "so101_lift_cube"
    run_name = "smoke"
    logger = "tensorboard"
    empirical_normalization = False
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.7,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
