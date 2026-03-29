"""Register Kutta velocity tasks with the mjlab task registry.

After this module is imported (auto-discovered by src/tasks/__init__.py),
two task IDs become available to train.py:
  - "Kutta-Rough"  : rough terrain with terrain curriculum
  - "Kutta-Flat"   : flat plane (faster for early training)

Usage:
  conda run -n unitree_rl python scripts/train.py Kutta-Flat
  conda run -n unitree_rl python scripts/train.py Kutta-Rough
"""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import kutta_flat_env_cfg, kutta_rough_env_cfg
from .rl_cfg import kutta_ppo_runner_cfg

# Register the rough-terrain task (with terrain curriculum).
register_mjlab_task(
  task_id="Kutta-Rough",
  env_cfg=kutta_rough_env_cfg(),
  play_env_cfg=kutta_rough_env_cfg(play=True),
  rl_cfg=kutta_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Register the flat-terrain task (simpler; good starting point).
register_mjlab_task(
  task_id="Kutta-Flat",
  env_cfg=kutta_flat_env_cfg(),
  play_env_cfg=kutta_flat_env_cfg(play=True),
  rl_cfg=kutta_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
