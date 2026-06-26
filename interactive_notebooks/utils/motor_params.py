"""Adapted from https://github.com/ExcitingSystems/exciting-environments."""

import jax
import jax.numpy as jnp

from typing import Callable
from scipy.io import loadmat
from pathlib import Path

import equinox as eqx
import numpy as np
from scipy.interpolate import griddata
from enum import Enum

COMMON_GRID_SIZE = (128, 128)


class StaticParams(eqx.Module):
    p: jax.Array = eqx.field(converter=jnp.asarray)  # Number of pole pairs
    r_s: jax.Array = eqx.field(converter=jnp.asarray)  # Stator resistance
    l_d: jax.Array = eqx.field(converter=jnp.asarray)  # D-axis inductance
    l_q: jax.Array = eqx.field(converter=jnp.asarray)  # Q-axis inductance
    psi_p: jax.Array = eqx.field(converter=jnp.asarray)  # Permanent magnet flux linkage
    u_dc: jax.Array = eqx.field(converter=jnp.asarray)  # DC link voltage
    i_lim: jax.Array = eqx.field(converter=jnp.asarray)


class MotorParams(eqx.Module):
    static_params: StaticParams
    default_soft_constraints: Callable
    lut_grids: dict
    lut_values: dict


def generate_luts(pmsm_lut, target_shape=None):
    saturated_quants = ["L_dd", "L_dq", "L_qd", "L_qq", "Psi_d", "Psi_q"]

    i_d_max = np.max(pmsm_lut["i_d_vec"])
    i_q_max = np.max(pmsm_lut["i_q_vec"])
    i_d_min = np.min(pmsm_lut["i_d_vec"])
    i_q_min = np.min(pmsm_lut["i_q_vec"])
    i_d_stepsize = (i_d_max - i_d_min) / (pmsm_lut["i_d_vec"].shape[1] - 1)
    i_q_stepsize = (i_q_max - i_q_min) / (pmsm_lut["i_q_vec"].shape[1] - 1)

    for q in saturated_quants:
        qmap = pmsm_lut[q]
        xi, yi = np.indices(qmap.shape)
        nan_mask = np.isnan(qmap)
        qmap[nan_mask] = griddata(
            (xi[~nan_mask], yi[~nan_mask]),
            qmap[~nan_mask],
            (xi[nan_mask], yi[nan_mask]),
            method="nearest",
        )
        a = np.vstack([qmap[0, :], qmap, qmap[-1, :]])
        b = np.hstack([a[:, :1], a, a[:, -1:]])
        pmsm_lut[q] = b

    n_grid_points_y, n_grid_points_x = pmsm_lut[saturated_quants[0]].shape
    x = np.linspace(i_d_min - i_d_stepsize, i_d_max + i_d_stepsize, n_grid_points_x)
    y = np.linspace(i_q_min - i_q_stepsize, i_q_max + i_q_stepsize, n_grid_points_y)

    if target_shape is not None:
        target_nx, target_ny = target_shape
        pad_x_left = max(0, (target_nx - n_grid_points_x) // 2)
        pad_x_right = max(0, target_nx - n_grid_points_x - pad_x_left)
        pad_y_left = max(0, (target_ny - n_grid_points_y) // 2)
        pad_y_right = max(0, target_ny - n_grid_points_y - pad_y_left)

        x_left = x[0] - np.arange(pad_x_left, 0, -1) * i_d_stepsize
        x_right = x[-1] + np.arange(1, pad_x_right + 1) * i_d_stepsize
        y_left = y[0] - np.arange(pad_y_left, 0, -1) * i_q_stepsize
        y_right = y[-1] + np.arange(1, pad_y_right + 1) * i_q_stepsize

        x = np.concatenate([x_left, x, x_right])
        y = np.concatenate([y_left, y, y_right])

        for q in saturated_quants:
            pmsm_lut[q] = np.pad(
                pmsm_lut[q],
                ((pad_y_left, pad_y_right), (pad_x_left, pad_x_right)),
                mode="edge",
            )

    LUT_grids = {q: (jnp.array(x), jnp.array(y)) for q in saturated_quants}
    LUT_values = {q: jnp.array(pmsm_lut[q][:, :].T) for q in saturated_quants}

    return LUT_grids, LUT_values


# Predefined motor configurations
def default_soft_constraints(instance, state, action_norm):
    state_norm = instance.normalize_state(state)
    physical_state_norm = state_norm.physical_state
    phys_soft_const = jax.tree.map(lambda x: jax.nn.relu(jnp.abs(x) - 1.0), physical_state_norm)
    return phys_soft_const, None


def _make_brusa_params():
    lut_raw = loadmat(Path(__file__).parent / Path("LUT_BRUSA_jax_grad.mat"))
    lut_grids, lut_values = generate_luts(lut_raw)
    return MotorParams(
        static_params=StaticParams(
            p=jnp.array(3),
            r_s=jnp.array(17.932e-3),
            l_d=jnp.array(0.37e-3),
            l_q=jnp.array(1.2e-3),
            psi_p=jnp.array(65.65e-3),
            u_dc=jnp.array(400),
            i_lim=jnp.array(250),
        ),
        default_soft_constraints=default_soft_constraints,
        lut_grids=lut_grids,
        lut_values=lut_values,
    )


def _make_sew_params():
    lut_raw = loadmat(Path(__file__).parent / Path("LUT_SEW_jax_grad.mat"))
    lut_grids, lut_values = generate_luts(lut_raw)
    return MotorParams(
        static_params=StaticParams(
            p=jnp.array(4),
            r_s=jnp.array(208e-3),
            l_d=jnp.array(1.44e-3),
            l_q=jnp.array(1.44e-3),
            psi_p=jnp.array(122e-3),
            u_dc=jnp.array(550),
            i_lim=jnp.array(16),
        ),
        default_soft_constraints=default_soft_constraints,
        lut_grids=lut_grids,
        lut_values=lut_values,
    )


def _make_default_params():
    # DEFAULT uses BRUSA LUTs, for sake of vmapped envs
    lut_raw = loadmat(Path(__file__).parent / Path("LUT_BRUSA_jax_grad.mat"))
    lut_grids, lut_values = generate_luts(lut_raw)
    return MotorParams(
        static_params=StaticParams(
            p=jnp.array(3),
            r_s=jnp.array(15e-3),
            l_d=jnp.array(0.37e-3),
            l_q=jnp.array(1.2e-3),
            psi_p=jnp.array(65.6e-3),
            u_dc=jnp.array(400),
            i_lim=jnp.array(250),
        ),
        default_soft_constraints=default_soft_constraints,
        lut_grids=lut_grids,
        lut_values=lut_values,
    )


class MotorVariant(Enum):
    DEFAULT = "DEFAULT"
    BRUSA = "BRUSA"
    SEW = "SEW"

    def get_params(self):
        if self is MotorVariant.BRUSA:
            return _make_brusa_params()
        elif self is MotorVariant.SEW:
            return _make_sew_params()
        else:
            return _make_default_params()
