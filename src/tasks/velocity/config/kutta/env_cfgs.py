"""Kutta quadruped velocity environment configurations.

This file follows exactly the same pattern as config/go2/env_cfgs.py.
It calls make_velocity_env_cfg() for the generic base, then applies all
Kutta-specific overrides:
  - Robot asset (XML + actuators)
  - Contact sensors (foot geom / body patterns)
  - Pose reward standard deviations per joint group
  - Body/site/geom name references throughout observations, events, rewards

No changes are needed to velocity_env_cfg.py or rewards.py — the Kutta uses
the same revolute1–12 naming convention and the same FL/FR/RL/RR site names.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots.kutta import get_kutta_robot_cfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def kutta_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Kutta rough-terrain velocity tracking configuration.

  Designed to match the Kutta's physical capabilities:
    - 10 Nm actuators  (vs 23–45 Nm on Go2)
    - Symmetric ±3.14 joint ranges softened to 90 % via soft_joint_pos_limit_factor
    - All-zeros standing pose (all joints at 0 = stable stand)

  Returns:
    ManagerBasedRlEnvCfg ready for PPO training.
  """
  cfg = make_velocity_env_cfg()

  # ------------------------------------------------------------------
  # Robot asset
  # ------------------------------------------------------------------
  cfg.scene.entities = {"robot": get_kutta_robot_cfg()}

  # Increase contact buffer for complex terrain.
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  # ------------------------------------------------------------------
  # Raycast terrain sensor – attach frame to Kutta base body.
  # ------------------------------------------------------------------
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      # "base_link" is the root body name in kutta.xml.
      sensor.frame.name = "base_link"

  # ------------------------------------------------------------------
  # Contact sensors
  # Foot geoms: FL_foot, RL_foot, FR_foot, RR_foot (same as Go2).
  # Non-foot collision geoms: <name>_collision (base, hip, thigh, knee).
  # ------------------------------------------------------------------
  # The "feet" are now the knee meshes (knee1, knee2, knee3, knee4).
  foot_names = ("FL", "RL", "FR", "RR")
  site_names = ("FL", "RL", "FR", "RR")
  foot_geoms = ("knee1", "knee2", "knee3", "knee4")

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    # Match each knee mesh against the terrain body.
    primary=ContactMatch(mode="geom", pattern=foot_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Target base, hips, and thighs.
      pattern=r"base_link|hip[1-4]|thigh[1-4]",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  # Enable terrain curriculum progression.
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # ------------------------------------------------------------------
  # Viewer – follow the Kutta base body.
  # ------------------------------------------------------------------
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  # ------------------------------------------------------------------
  # Observations – set foot site names for critic foot_height term.
  # ------------------------------------------------------------------
  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  # ------------------------------------------------------------------
  # Events (domain randomisation)
  # ------------------------------------------------------------------
  # Randomise friction on knee meshes (the feet).
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geoms
  # Shift the centre of mass of the torso only.
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # ------------------------------------------------------------------
  # Rewards
  # ------------------------------------------------------------------
  # Body angular velocity penalty – applied to the base body only.
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  # foot_clearance and foot_slip – reference foot sites.
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  # ── pose reward: speed-dependent tolerances (variable_posture) ──
  #
  # Joint groups by function:
  #   Hip abduction  : revolute1, revolute4 (left)  | revolute7, revolute10 (right, INVERTED)
  #   Thigh flex/ext : revolute2, revolute5 (left)  | revolute8, revolute11 (right)
  #   Knee           : revolute3, revolute6 (left)  | revolute9, revolute12 (right, INVERTED)
  #
  # Kutta has ±3.14 limits, which is much wider than Go2 (±1.05 / ±3.49 / etc.).
  # We use tighter std values while standing to keep the robot upright,
  # and loosen them for walking/running to allow natural gait.
  #
  # ─ Standing (near zero velocity): hold default pose tightly ─
  cfg.rewards["pose"].params["std_standing"] = {
    # Hip abduction joints (all four): very small std → penalise any abduction while still.
    r"revolute(1|4|7|10)": 0.05,
    # Thigh joints: allow tiny flex to absorb weight settling.
    r"revolute(2|5|8|11)": 0.10,
    # Knee joints: similar to thigh.
    r"revolute(3|6|9|12)": 0.10,
  }
  # ─ Walking (slow to moderate speed) ─
  cfg.rewards["pose"].params["std_walking"] = {
    r"revolute(1|4|7|10)": 0.20,   # Moderate abduction allowed
    r"revolute(2|5|8|11)": 0.40,   # Thigh can swing freely
    r"revolute(3|6|9|12)": 0.50,   # Knee bends more when stepping
  }
  # ─ Running (high speed) ─
  cfg.rewards["pose"].params["std_running"] = {
    r"revolute(1|4|7|10)": 0.20,
    r"revolute(2|5|8|11)": 0.45,
    r"revolute(3|6|9|12)": 0.55,
  }

  # Orientation penalty weight – Kutta is lighter, needs stronger upright signal.
  cfg.rewards["flat_orientation_l2"].weight = -2.0

  # ------------------------------------------------------------------
  # Terminations
  # ------------------------------------------------------------------
  # Relax orientation termination: allow the robot to tilt up to 100 degrees.
  import math
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(100.0)

  # End episode if any non-foot geom hits the terrain.
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  # ------------------------------------------------------------------
  # Play-mode overrides (disable noise, push, curriculum, etc.)
  # ------------------------------------------------------------------
  if play:
    cfg.episode_length_s = int(1e9)  # Effectively infinite.
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def kutta_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Kutta flat-terrain velocity tracking configuration.

  Inherits from kutta_rough_env_cfg and switches to a flat plane.
  Removes raycast sensors / height-scan observations (no terrain to scan).
  Useful for early training before introducing rough terrain.
  """
  cfg = kutta_rough_env_cfg(play=play)

  # Flat terrain needs fewer contacts – reduce sim buffers for speed.
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to infinite flat plane.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove the raycast terrain scan sensor (no hills to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  # Remove the height_scan observation term from both actor and critic.
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # No terrain curriculum on a flat plane.
  cfg.curriculum.pop("terrain_levels", None)

  # In play mode, use gentler velocity commands so the robot isn't pushed hard.
  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
