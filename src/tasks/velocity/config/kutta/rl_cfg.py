"""RL (PPO) runner configuration for the Kutta velocity task.

The Kutta is a lighter robot (total ~2.8 kg) with weaker actuators (10 Nm)
compared to the Go2 (6.9 kg base, up to 45 Nm). The network architecture
is the same but the experiment name is different so logs are kept separate.
Tune learning_rate / num_mini_batches if training is unstable.
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def kutta_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create PPO runner configuration for Kutta velocity task."""
  return RslRlOnPolicyRunnerCfg(
    # Actor network: the policy that runs on-robot.
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=True,
      init_noise_std=1.0,  # Start with high exploration noise.
    ),
    # Critic network: value estimator (privileged info, never deployed).
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=False,
      init_noise_std=1.0,
    ),
    # PPO hyperparameters — identical to Go2 as a safe starting point.
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,         # Encourage exploration early
      num_learning_epochs=5,
      num_mini_batches=8,        # 1024 envs × 24 steps / 8 = 3072 samples/batch → GPU-saturating
      learning_rate=1.0e-3,
      schedule="adaptive",       # Adjusts lr based on KL divergence
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    # Logs go to logs/rsl_rl/kutta_velocity/
    experiment_name="kutta_velocity",
    save_interval=100,           # Save checkpoint every 100 iterations
    num_steps_per_env=24,        # Steps per env per update (matches Go2)
    max_iterations=10001,        # Total PPO iterations (~240k steps/env)
  )
