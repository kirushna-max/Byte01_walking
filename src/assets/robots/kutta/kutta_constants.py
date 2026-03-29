"""Kutta quadruped robot constants.

This mirrors the structure of unitree_go2/go2_constants.py.
The Kutta robot uses the same joint naming convention as the Go2 training
setup (revolute1–revolute12), so the env config layer barely changes.

Joint layout (same ordering as Go2):
  revolute1  (FL hip abduction  – X axis, +Y side)
  revolute2  (FL thigh          – Y axis)
  revolute3  (FL knee           – -Y axis)
  revolute4  (RL hip abduction  – X axis, +Y side)
  revolute5  (RL thigh          – Y axis)
  revolute6  (RL knee           – -Y axis)
  revolute7  (FR hip abduction  – X axis, -Y side, limits INVERTED vs left legs)
  revolute8  (FR thigh          – -Y axis)
  revolute9  (FR knee           – +Y axis, limits INVERTED vs left knees)
  revolute10 (RR hip abduction  – X axis, -Y side, limits INVERTED vs left legs)
  revolute11 (RR thigh          – -Y axis)
  revolute12 (RR knee           – +Y axis, limits INVERTED vs left knees)
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
# The kutta.xml lives in the kutta/xmls/ directory.
# Its STL meshes are stored in kutta/xmls/assets/.
##

KUTTA_XML: Path = (
  SRC_PATH / "assets" / "robots" / "kutta" / "xmls" / "kutta.xml"
)
assert KUTTA_XML.exists(), f"kutta.xml not found at {KUTTA_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  """Load Kutta STL mesh bytes from the local assets/ directory."""
  assets: dict[str, bytes] = {}
  # Assets live in the 'assets/' folder relative to the XML.
  update_assets(assets, KUTTA_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  """Parse and return MuJoCo spec for the Kutta robot."""
  spec = mujoco.MjSpec.from_file(str(KUTTA_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
# The Kutta XML already declares kp=20, forcerange=±10 Nm on every joint.
# We replicate that here so mjlab's actuator manager takes ownership.
# Stiffness / damping / effort_limit reflect the XML values.
##

# Hip abduction actuators: revolute1 (FL), revolute4 (RL), revolute7 (FR), revolute10 (RR)
KUTTA_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "revolute1", "revolute4", "revolute7", "revolute10",
  ),
  stiffness=20.0,   # kp from XML
  damping=0.5,      # Light damping; tune if robot oscilates at joints
  effort_limit=10.0,  # forcerange from XML (N·m)
  armature=0.005,
)

# Thigh (hip flex/ext) actuators: revolute2 (FL), revolute5 (RL), revolute8 (FR), revolute11 (RR)
KUTTA_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "revolute2", "revolute5", "revolute8", "revolute11",
  ),
  stiffness=20.0,
  damping=0.5,
  effort_limit=10.0,
  armature=0.005,
)

# Knee actuators: revolute3 (FL), revolute6 (RL), revolute9 (FR), revolute12 (RR)
KUTTA_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "revolute3", "revolute6", "revolute9", "revolute12",
  ),
  stiffness=20.0,
  damping=0.5,
  effort_limit=10.0,
  armature=0.005,
)

##
# Keyframes / Initial state.
# All revolutes at 0.0 places the Kutta's feet on the ground (confirmed by
# the user). Spawn height of 0.32 m gives a small clearance.
#
# Joint limits (set in soft_joint_pos_limit_factor below and used for
# reward shaping). Full limits per the user spec:
#   Left legs  (legs 1/2 – FL/RL):
#     hip   revolute1,4  : range (-0.60,  1.79) rad
#     thigh revolute2,5  : range (-1.73,  1.73) rad
#     knee  revolute3,6  : range (-1.73, -0.90) rad  ← negative-only range (axis is -Y)
#   Right legs (legs 3/4 – FR/RR) — INVERTED hip & knee:
#     hip   revolute7,10 : range ( 0.60, -1.79) → stored as (-1.79,  0.60) by MuJoCo
#     thigh revolute8,11 : range (-1.73,  1.73) rad
#     knee  revolute9,12 : range ( 1.73, -0.90) rad  ← positive-only range  (axis is +Y)
# These are enforced via the actuator ctrlrange in the XML and the
# soft_joint_pos_limit_factor factor applied to them.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  # Spawn slightly above ground so the robot settles without tunnelling.
  pos=(0.0, 0.0, 0.2),
  # All zeros = standing pose (verified with MuJoCo viewer).
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

##
# Collision config.
# Foot geoms: FL_foot, RL_foot, FR_foot, RR_foot  (same naming as Go2).
# Non-foot collision geoms follow the pattern: <name>_collision
#   e.g. base_collision, hip1_collision, thigh1_collision, knee1_collision, …
##

# The "feet" are now the knee meshes (knee1, knee2, knee3, knee4).
_foot_regex = r"^knee[1-4]$"

# This disables all collisions except the "feet" (knee meshes).
KUTTA_FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions, excluding self collisions.
# All mesh geoms are now targeted (base_link, hip1-4, thigh1-4, knee1-4).
KUTTA_FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r"base_link|hip[1-4]|thigh[1-4]|knee[1-4]",),
  condim={_foot_regex: 3, r".*": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Articulation info (aggregates actuators & limits).
##

KUTTA_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    KUTTA_ACTUATOR_HIP,
    KUTTA_ACTUATOR_THIGH,
    KUTTA_ACTUATOR_KNEE,
  ),
  # Use 90 % of the XML-defined joint range as soft limits so the
  # joint_pos_limits reward penalises before the hard stop.
  soft_joint_pos_limit_factor=0.9,
)


##
# Factory function.
##

def get_kutta_robot_cfg() -> EntityCfg:
  """Return a fresh Kutta robot EntityCfg instance.

  A new instance is returned each call to prevent config mutation
  when the same config object is consumed by multiple callers.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(KUTTA_FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=KUTTA_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_kutta_robot_cfg())
  viewer.launch(robot.spec.compile())
