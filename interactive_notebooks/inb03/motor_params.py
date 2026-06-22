"""Adapted from https://github.com/ExcitingSystems/exciting-environments."""

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import griddata

COMMON_GRID_SIZE = (128, 128)


def generate_luts(pmsm_lut, target_shape=COMMON_GRID_SIZE):
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


def lut_interpolate(grid_x, grid_y, values, i_d, i_q):
    ix = jnp.searchsorted(grid_x, i_d) - 1
    iy = jnp.searchsorted(grid_y, i_q) - 1
    ix = jnp.clip(ix, 0, grid_x.shape[0] - 2)
    iy = jnp.clip(iy, 0, grid_y.shape[0] - 2)
    tx = (i_d - grid_x[ix]) / (grid_x[ix + 1] - grid_x[ix])
    ty = (i_q - grid_y[iy]) / (grid_y[iy + 1] - grid_y[iy])
    return (
        values[ix, iy] * (1 - tx) * (1 - ty)
        + values[ix + 1, iy] * tx * (1 - ty)
        + values[ix, iy + 1] * (1 - tx) * ty
        + values[ix + 1, iy + 1] * tx * ty
    )
