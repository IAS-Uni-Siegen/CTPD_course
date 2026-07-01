"""Adapted from https://github.com/ExcitingSystems/exciting-environments."""

from copy import deepcopy
from typing import Callable
from scipy.io import loadmat
from pathlib import Path
from enum import Enum

import numpy as np
from scipy.interpolate import griddata

import jax
import jax.numpy as jnp

import equinox as eqx


COMMON_GRID_SIZE = (128, 128)


class StaticParams(eqx.Module):
    p: jax.Array = eqx.field(converter=jnp.asarray)
    r_s: jax.Array = eqx.field(converter=jnp.asarray)
    r_r: jax.Array = eqx.field(converter=jnp.asarray)
    l_m: jax.Array = eqx.field(converter=jnp.asarray)
    l_sigs: jax.Array = eqx.field(converter=jnp.asarray)
    l_sigr: jax.Array = eqx.field(converter=jnp.asarray)
    u_dc: jax.Array = eqx.field(converter=jnp.asarray)
    i_lim: jax.Array = eqx.field(converter=jnp.asarray)


class SaturationParams(eqx.Module):
    k1: float
    k2: float
    k3: float
    k4: float


class MotorParams(eqx.Module):
    static_params: StaticParams
    saturation_params: SaturationParams


DEFAULT = MotorParams(
    static_params=StaticParams(
        p=2,
        r_r=1.355,
        r_s=2.9338,
        l_m=143.75e-3,
        l_sigs=5.87e-3,
        l_sigr=5.87e-3,
        u_dc=560,
        i_lim=20.0,
    ),
    saturation_params=SaturationParams(
        k1=0.1596,
        k2=0.0478,
        k3=39.4442,
        k4=0.4938,
    ),
)


def default_params(name):
    """
    Returns default parameters for specified motor configurations.

    Args:
        name (str): Name of the motor.

    Returns:
        MotorConfig: Configuration containing physical constraints, action constraints, static parameters, and LUT data.
    """
    if name is None:
        return deepcopy(DEFAULT)
    else:
        raise ValueError(f"Motor name {name} is not known.")
