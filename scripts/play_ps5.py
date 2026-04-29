"""Play an RSL-RL policy while driving the velocity command with a PS5 controller."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

# Ensure the rl_mjlab root (parent of this scripts/ dir) is on the path.
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class Ps5PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  wandb_run_path: str | None = None
  registry_name: str | None = None

  command_name: str = "twist"
  joystick_index: int = 0
  left_x_axis: int = 0
  left_y_axis: int = 1
  right_x_axis: int = 2
  left_x_mode: Literal["strafe", "yaw"] = "strafe"
  max_lin_x: float = 1.0
  max_lin_y: float = 1.0
  max_yaw: float = 1.0
  deadzone: float = 0.08
  invert_left_y: bool = True

  _demo_mode: tyro.conf.Suppress[bool] = False


class Ps5Controller:
  def __init__(self, cfg: Ps5PlayConfig):
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
      import pygame
    except ImportError as exc:
      raise RuntimeError(
        "pygame is required for PS5 controller input. Install it in this env with "
        "`pip install pygame`."
      ) from exc

    self._pygame = pygame
    self._cfg = cfg
    pygame.display.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if count <= cfg.joystick_index:
      pygame.quit()
      raise RuntimeError(
        f"No joystick at index {cfg.joystick_index}. pygame sees {count} joystick(s)."
      )

    self._joystick = pygame.joystick.Joystick(cfg.joystick_index)
    self._joystick.init()
    print(f"[PS5] Using controller: {self._joystick.get_name()}")
    print(
      "[PS5] Left stick: "
      f"Y -> lin_x (+forward), X -> {cfg.left_x_mode}; "
      f"right X -> yaw; "
      f"deadzone={cfg.deadzone}"
    )

  def close(self) -> None:
    self._pygame.quit()

  def read_cmd_vel(self) -> tuple[float, float, float]:
    self._pygame.event.pump()
    left_x = self._axis(self._cfg.left_x_axis)
    left_y = self._axis(self._cfg.left_y_axis)
    right_x = self._axis(self._cfg.right_x_axis)
    left_x = -left_x
    if self._cfg.invert_left_y:
      left_y = -left_y

    lin_x = self._shape(left_y) * self._cfg.max_lin_x
    left_horizontal = self._shape(left_x)
    right_horizontal = self._shape(right_x)
    if self._cfg.left_x_mode == "strafe":
      lin_y = left_horizontal * self._cfg.max_lin_y
      yaw = right_horizontal * self._cfg.max_yaw
    else:
      lin_y = 0.0
      yaw = left_horizontal * self._cfg.max_yaw
    return lin_x, lin_y, yaw

  def _axis(self, axis_id: int) -> float:
    if axis_id >= self._joystick.get_numaxes():
      return 0.0
    return float(self._joystick.get_axis(axis_id))

  def _shape(self, value: float) -> float:
    deadzone = max(0.0, min(self._cfg.deadzone, 0.99))
    if abs(value) <= deadzone:
      return 0.0
    scaled = (abs(value) - deadzone) / (1.0 - deadzone)
    return max(-1.0, min(1.0, scaled)) * (1.0 if value > 0.0 else -1.0)


def configure_manual_velocity_command(command_cfg: Any) -> None:
  if hasattr(command_cfg, "heading_command"):
    command_cfg.heading_command = False
  if hasattr(command_cfg, "ranges") and hasattr(command_cfg.ranges, "heading"):
    command_cfg.ranges.heading = None
  if hasattr(command_cfg, "rel_heading_envs"):
    command_cfg.rel_heading_envs = 0.0
  if hasattr(command_cfg, "rel_standing_envs"):
    command_cfg.rel_standing_envs = 0.0
  if hasattr(command_cfg, "resampling_time_range"):
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)


class Ps5CommandOverride:
  def __init__(self, env: Any, controller: Ps5Controller, command_name: str):
    self._env = env
    self._controller = controller
    self._command_name = command_name
    self._term = env.unwrapped.command_manager.get_term(command_name)

    cfg = getattr(self._term, "cfg", None)
    if cfg is not None:
      configure_manual_velocity_command(cfg)

  def update(self) -> None:
    lin_x, lin_y, yaw = self._controller.read_cmd_vel()
    command = self._term.command
    values = torch.tensor(
      (lin_x, lin_y, yaw), device=command.device, dtype=command.dtype
    )
    with torch.no_grad():
      command[:, :3] = values
      if hasattr(self._term, "time_left"):
        self._term.time_left[:] = 1.0e9
      if hasattr(self._term, "is_heading_env"):
        self._term.is_heading_env[:] = False
      if hasattr(self._term, "is_standing_env"):
        self._term.is_standing_env[:] = False
    self._clear_observation_cache()

  def _clear_observation_cache(self) -> None:
    obs_manager = self._env.unwrapped.observation_manager
    if hasattr(obs_manager, "_obs_buffer"):
      obs_manager._obs_buffer = None


class Ps5CommandEnvWrapper:
  def __init__(self, env: Any, command_override: Ps5CommandOverride):
    self._env = env
    self._command_override = command_override

  def __getattr__(self, name: str) -> Any:
    return getattr(self._env, name)

  def get_observations(self) -> Any:
    self._command_override.update()
    return self._env.get_observations()

  def reset(self) -> Any:
    result = self._env.reset()
    self._command_override.update()
    return result

  def step(self, actions: torch.Tensor) -> Any:
    return self._env.step(actions)

  def close(self) -> None:
    return self._env.close()


def run_play(task_id: str, cfg: Ps5PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  dummy_mode = cfg.agent in {"zero", "random"}
  trained_mode = not dummy_mode

  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  if cfg.command_name not in env_cfg.commands:
    raise ValueError(
      f"Command '{cfg.command_name}' is not in this task. "
      f"Available commands: {tuple(env_cfg.commands.keys())}"
    )

  command_cfg = env_cfg.commands[cfg.command_name]
  configure_manual_velocity_command(command_cfg)

  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )
  if is_tracking_task and cfg._demo_mode:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif dummy_mode and not cfg.registry_name:
      raise ValueError(
        "Tracking tasks require either --motion-file or --registry-name."
      )

  log_dir: Path | None = None
  resume_path: Path | None = None
  if trained_mode:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file).expanduser()
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      run_id = resume_path.parent.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {resume_path.name} "
        f"(run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (trained_mode and cfg.video) else None
  if cfg.video and dummy_mode:
    print("[WARN] Video recording with dummy agents is disabled.")

  controller = Ps5Controller(cfg)
  env: Ps5CommandEnvWrapper | None = None
  try:
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    if trained_mode and cfg.video:
      print("[INFO] Recording videos during play")
      assert log_dir is not None
      base_env = VideoRecorder(
        base_env,
        video_folder=log_dir / "videos" / "play_ps5",
        step_trigger=lambda step: step == 0,
        video_length=cfg.video_length,
        disable_logger=True,
      )

    rsl_env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    command_override = Ps5CommandOverride(rsl_env, controller, cfg.command_name)
    env = Ps5CommandEnvWrapper(rsl_env, command_override)

    if dummy_mode:
      assert env is not None
      action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
      if cfg.agent == "zero":

        class PolicyZero:
          def __call__(self, obs) -> torch.Tensor:
            del obs
            return torch.zeros(action_shape, device=env.unwrapped.device)

        policy = PolicyZero()
      else:

        class PolicyRandom:
          def __call__(self, obs) -> torch.Tensor:
            del obs
            return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

        policy = PolicyRandom()
    else:
      assert resume_path is not None
      runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
      runner = runner_cls(rsl_env, asdict(agent_cfg), device=device)
      runner.load(
        str(resume_path), load_cfg={"actor": True}, strict=False, map_location=device
      )
      policy = runner.get_inference_policy(device=device)
      try:
        import importlib.util as _ilu

        monitor_path = Path(__file__).with_name("Monitor.py")
        spec = _ilu.spec_from_file_location("Monitor", monitor_path)
        mod = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        policy = mod.attach_monitor_hook(policy, env)
      except Exception as monitor_err:
        print(f"[Monitor] Hook not attached: {monitor_err}")

    if cfg.viewer == "auto":
      has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
      resolved_viewer = "native" if has_display else "viser"
    else:
      resolved_viewer = cfg.viewer

    if resolved_viewer == "native":
      NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
      ViserPlayViewer(env, policy).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    controller.close()
    if env is not None:
      env.close()


def main():
  import mjlab
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    Ps5PlayConfig,
    args=remaining_args,
    default=Ps5PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
