"""Domain Randomization helpers for velocity tasks.

This module exposes named wrappers around mjlab's generic ``randomize_field``
and ``randomize_encoder_bias`` functions so that the env config can refer to
them via ``mdp.dr.geom_friction``, ``mdp.dr.body_ipos``, etc.
"""

from __future__ import annotations

import functools

from mjlab.envs.mdp import randomize_encoder_bias, randomize_field


def _randomize_field_for(field_name: str):
    """Return a partial of ``randomize_field`` pre-bound to *field_name*."""

    @functools.wraps(randomize_field)
    def _fn(env, env_ids, **kwargs):
        return randomize_field(env, env_ids, field=field_name, **kwargs)

    _fn.__name__ = field_name
    _fn.__qualname__ = f"dr.{field_name}"
    return _fn


# Friction coefficient of geometry collision surfaces.
geom_friction = _randomize_field_for("geom_friction")

# Inertial position offset of a body (simulates CoM shift / payload variance).
body_ipos = _randomize_field_for("body_ipos")

# Joint encoder bias (simulates calibration error on hardware).
encoder_bias = randomize_encoder_bias
