"""Velocity task configuration.

This module provides a factory function to create a base velocity task config.
Robot-specific configurations call the factory and customize as needed. It defines
observations, actions, domain randomization (events), rewards, terminations, and
curriculums required to train a quadruped robot to track velocity commands.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.velocity.mdp as mdp


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration.
  
  Returns:
      ManagerBasedRlEnvCfg: A complete configuration object defining the RL environment
      for velocity tracking, containing all necessary managers (observations, actions, etc.).
  """

  ##
  # Observations
  # Defines what the neural network will "see" at each timestep.
  ##

  # actor_terms defines the observation space for the policy network (the actor).
  actor_terms = {
    # Base angular velocity from the IMU, with simulated noise.
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    # Projected gravity vector to determine the orientation of the base.
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    # The velocity command we want the robot to track (x, y, yaw velocity).
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    # Current relative joint positions (position - default_position).
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Current joint velocities.
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    # The last action taken by the policy.
    "actions": ObservationTermCfg(func=mdp.last_action),
    # Raycast terrain height scan around the robot for obstacle awareness.
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      clip=(-1.0, 1.0),
    ),
  }

  # critic_terms defines the observation space for the value network (the critic in PPO).
  # Critics can typically access privileged information that the actor cannot see.
  critic_terms = {
    # Include all actor terms so the critic sees what the actor sees.
    **actor_terms,
    # Base linear velocity (privileged info, hard to estimate exactly on hardware).
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    # The true terrain scan, without the noise applied to the actor's scan.
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      clip=(-1.0, 1.0),
    ),
    # Foot heights relative to the terrain.
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
    ),
    # Accumulated air time per foot to measure step duration.
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    # Boolean indicator of foot ground contact.
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    # True contact forces on each foot.
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  ##
  # Actions
  # Maps the continuous outputs of the neural network to the robot control.
  ##

  actions: dict[str, ActionTermCfg] = {
    # Network outputs joint target position offsets.
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",), # Targets all actuators.
      scale=0.25,  # Action scale: target = default_pos + action * scale.
      use_default_offset=True, # Actions are relative to default joint positions.
    )
  }

  ##
  # Commands
  # Determines how goals are generated for the agent to follow.
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0), # Resample new command every 3-8 seconds
      rel_standing_envs=0.05, # 5% of envs are purely standing still
      rel_heading_envs=0.25,  # 25% of envs use heading commands instead of ang_vel commands
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 2.0), # Forward/backward velocity range
        lin_vel_y=(-1.0, 1.0), # Lateral velocity range
        ang_vel_z=(-1.0, 1.0), # Yaw velocity range
        heading=(-math.pi, math.pi),
      ),
    )
  }

  ##
  # Events
  # Handles episodic events like resetting states, applying random pushes, and domain randomization.
  ##

  events = {
    # Randomizes robot base pose on episode reset to learn recovery and robust standing.
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05), # Slightly randomize height to drop the robot.
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    # Randomizes joint positions and velocities slightly on reset.
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.5, 0.5),
        "velocity_range": (-0.5, 0.5),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Randomly pushes the robot around every few seconds to teach robustness.
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    # Domain Randomization: Randomizes ground friction at startup for sim-to-real transfer.
    "foot_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    # Domain Randomization: Adds constant biases to joint encoder readings.
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=mdp.randomize_encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    # Domain Randomization: Shifts the center of mass randomly to simulate payload/hardware mismatch.
    "base_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "field": "body_ipos",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
  }

  ##
  # Rewards
  # Multi-objective reward function describing what a 'good' behavior looks like.
  ##

  rewards = {
    # Reward tracking the desired x and y linear velocity command.
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    # Reward tracking the desired yaw velocity command.
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    # Penalty for tilting the body (maintains a flat base).
    "flat_orientation_l2": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-5.0),
    # Encourages the robot to adopt different default postures based on speed
    # (e.g. straight legs when running, crouched when walking).
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.1,
        "running_threshold": 1.5,
      },
    ),
    # Penalty for non-z-axis angular velocity (prevents wobbling).
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,  # Override per-robot
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    # Penalty to minimize overall angular momentum (leads to more efficient, controlled gaits).
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.025,  # Override per-robot
      params={"sensor_name": "robot/root_angmom"},
    ),
    # Huge penalty for early termination (falling over or crashing).
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    # Penalty on joint accelerations to produce smoother movements.
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    # Penalty for getting too close to joint cinematic limits.
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    # Penalty on high-frequency action changes (promotes smooth control).
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    # Promotes lifting the feet to a target clearance height when stepping.
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-1.0,
      params={
        "target_height": 0.10,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    # Penalty for feet dragging or sliding on the ground while taking a step.
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    # Encourages minimal impact force when feet touch down (smooth walking).
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-3,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
  }

  ##
  # Terminations
  # Determines when an episode ends early or finishes successfully.
  ##

  terminations = {
    # Ends episode on max episode length timeout.
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # Ends episode automatically and applies penalty if pitch or roll exceeds 70 degrees.
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  ##
  # Curriculum
  # Adjusts task difficulty over training epochs depending on agent proficiency.
  ##

  curriculum = {
    # Progressively spawns robot on rougher or steeper terrain as it gets better.
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    # Progressively requests higher command velocities as the training steps increment.
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          # Stage 0: Gentle walking speeds
          {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
          # Stage ~120k steps: unlock full running speeds.
          {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
        ],
      },
    ),
  }

  ##
  # Assemble and return
  # Builds final environment configuration object by packing the constructed blocks.
  ##

  # Configuration for simulating terrain height mapping around the robot base.
  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # Centered at robot base (set per-robot).
    ray_alignment="yaw", # Rays rotate with yaw, but not pitch/roll.
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1), # Density of ray grid.
    max_distance=5.0,
    exclude_parent_body=True,
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(show_normals=True),
  )

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scan,),
      num_envs=4096,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Follow robot base body.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35, # Max MuJoCo contacts
      njmax=1500, # Max MuJoCo constraints
      mujoco=MujocoCfg(
        timestep=0.005, # Sim physics timestep
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4, # Policy runs every 4 sim steps (dt=0.02s, 50Hz)
    episode_length_s=20.0, # Agents reset after 20 real-time seconds.
  )
