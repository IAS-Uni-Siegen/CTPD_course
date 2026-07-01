import numpy as np
import jax.numpy as jnp


# only for alpha/beta -> abc
t32 = jnp.array([[1, 0], [-0.5, 0.5 * jnp.sqrt(3)], [-0.5, -0.5 * jnp.sqrt(3)]])
t23 = 2 / 3 * jnp.array([[1, 0], [-0.5, 0.5 * jnp.sqrt(3)], [-0.5, -0.5 * jnp.sqrt(3)]]).T  # only for abc -> alpha/beta

inverter_t_abc = jnp.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, 0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
    ]
)

ROTATION_MAP = np.ones((2, 2, 2), dtype=np.complex64)
ROTATION_MAP[1, 0, 1] = 0.5 * (1 + np.sqrt(3) * 1j)
ROTATION_MAP[1, 1, 0] = 0.5 * (1 - np.sqrt(3) * 1j)
ROTATION_MAP[0, 1, 0] = 0.5 * (-1 - np.sqrt(3) * 1j)
ROTATION_MAP[0, 1, 1] = -1
ROTATION_MAP[0, 0, 1] = 0.5 * (-1 + np.sqrt(3) * 1j)
ROTATION_MAP = jnp.array(ROTATION_MAP)


def t_dq_alpha_beta(eps):
    """Compute the transformation matrix for converting between DQ and Alpha-Beta reference frames."""
    cos = jnp.cos(eps)
    sin = jnp.sin(eps)
    return jnp.column_stack((cos, sin, -sin, cos)).reshape(2, 2)


def dq2abc(u_dq, eps):
    """Transform voltages from DQ coordinates to ABC (three-phase) coordinates."""
    u_abc = t32 @ dq2albet(u_dq, eps).T
    return u_abc.T


def dq2albet(u_dq, eps):
    """Transform voltages from DQ coordinates to Alpha-Beta coordinates."""
    q = t_dq_alpha_beta(-eps)
    u_alpha_beta = q @ u_dq.T

    return u_alpha_beta.T


def albet2dq(u_albet, eps):
    """Transform voltages from Alpha-Beta coordinates to DQ coordinates."""
    q_inv = t_dq_alpha_beta(eps)
    u_dq = q_inv @ u_albet.T

    return u_dq.T


def abc2dq(u_abc, eps):
    """Transform voltages from ABC (three-phase) coordinates to DQ coordinates."""
    u_alpha_beta = t23 @ u_abc.T
    u_dq = albet2dq(u_alpha_beta.T, eps)
    return u_dq


def step_eps(eps, omega_el, T_s, T_s_scale=1.0):
    """Update the electrical angle over a time step with optional scaling."""
    eps += omega_el * T_s * T_s_scale
    eps %= 2 * jnp.pi
    boolean = eps > jnp.pi
    summation_mask = boolean * -2 * jnp.pi
    eps = eps + summation_mask
    return eps


def apply_hex_constraint(u_albet):
    """Clip voltages in alpha/beta coordinates into the voltage hexagon."""
    u_albet_c = u_albet[0] + 1j * u_albet[1]
    idx = (jnp.sin(jnp.angle(u_albet_c)[..., jnp.newaxis] - 2 / 3 * jnp.pi * jnp.arange(3)) >= 0).astype(int)
    rot_vec = ROTATION_MAP[idx[0], idx[1], idx[2]]
    # rotate sectors upwards
    u_albet_c = jnp.multiply(u_albet_c, rot_vec)
    u_albet_c = jnp.clip(u_albet_c.real, -2 / 3, 2 / 3) + 1j * u_albet_c.imag
    u_albet_c = u_albet_c.real + 1j * jnp.clip(u_albet_c.imag, 0, 2 / 3 * jnp.sqrt(3))
    u_albet_c = jnp.multiply(u_albet_c, jnp.conjugate(rot_vec))  # rotate back
    return jnp.column_stack([u_albet_c.real, u_albet_c.imag])


def clip_in_abc_coordinates(u_dq, u_dc, omega_el, eps, T_s):
    """Clip voltages in ABC (three-phase) coordinates and transform back to DQ coordinates."""
    eps_advanced = step_eps(eps, omega_el, T_s, 0.5)
    u_abc = dq2abc(u_dq, eps_advanced)
    # clip in abc coordinates
    u_abc = jnp.clip(u_abc, -u_dc / 2.0, u_dc / 2.0)
    u_dq = abc2dq(u_abc, eps)
    return u_dq


def clip_voltage_alpha_beta(u_alpha_beta, u_dc, eps, omega, T_s):
    u_albet_norm = u_alpha_beta * (1 / (u_dc / 2))
    u_albet_norm_clip = apply_hex_constraint(u_albet_norm)
    u_albet_clipped = u_albet_norm_clip[0] * (u_dc / 2)
    return u_albet_clipped


def clip_voltage(u_dq, u_dc, eps, omega, T_s):
    # normalize to u_dc/2 for hexagon constraints
    u_dq_norm = u_dq * (1 / (u_dc / 2))
    advanced_angle = step_eps(
        eps,
        omega_el=omega,
        T_s=T_s,
        T_s_scale=1.5,
    )
    u_albet_norm = dq2albet(
        u_dq_norm,
        advanced_angle,
    )
    u_albet_norm_clip = apply_hex_constraint(u_albet_norm)
    u_dq_norm_clip = albet2dq(
        u_albet_norm_clip,
        advanced_angle,
    )

    # denormalize from u_dc/2
    u_dq_clipped = u_dq_norm_clip[0] * (u_dc / 2)
    return u_dq_clipped


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
