from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

t32 = jnp.array([[1, 0], [-0.5, 0.5 * jnp.sqrt(3)], [-0.5, -0.5 * jnp.sqrt(3)]])
t23 = 2 / 3 * jnp.array([[1, 0], [-0.5, 0.5 * jnp.sqrt(3)], [-0.5, -0.5 * jnp.sqrt(3)]]).T  # only for abc -> alpha/beta

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


def clip_in_abc_coordinates(u_dq, u_dc, omega_el, eps, tau):
    eps_advanced = step_eps(eps, omega_el, tau, 0.5)
    u_abc = dq2abc(u_dq, eps_advanced)
    # clip in abc coordinates
    u_abc = jnp.clip(u_abc, -u_dc / 2.0, u_dc / 2.0)
    u_dq = abc2dq(u_abc, eps)
    return u_dq


def step_eps(eps, omega_el, tau, tau_scale=1.0):
    eps += omega_el * tau * tau_scale
    eps %= 2 * jnp.pi
    boolean = eps > jnp.pi
    summation_mask = boolean * -2 * jnp.pi
    eps = eps + summation_mask
    return eps


def aprbs_single_batch(len: int, t_min: float, t_max: float, key: jax.random.PRNGKey) -> jax.Array:
    """Creates an amplitude modulated pseudorandom binary sequence (APRBS) in 1d and for 1 batch,
    i.e. without a batch dimension.

    Args:
        len (int): Length of the signal.
        t_min (float): Minimum hold time of an amplitude.
        t_max (float): Maximum hold time of an amplitude
        key (jax.random.PRNGKey): Random key for JAX random sampling.
    """

    t = 0
    sig = []
    while t < len:
        steps_key, value_key, key = jax.random.split(key, 3)

        t_step = jax.random.randint(steps_key, shape=(1,), minval=t_min, maxval=t_max)

        sig.append(jnp.ones(t_step) * jax.random.uniform(value_key, shape=(1,), minval=-1, maxval=1))
        t += t_step.item()

    return jnp.hstack(sig)[:len]


def aprbs(n_steps: int, batch_size: int, t_min: float, t_max: float, key: jax.random.PRNGKey) -> jax.Array:
    """Creates an amplitude modulated pseudorandom binary sequence (APRBS) in 1d where 'batch_size'
    sequences are created independently next to each other.

    Args:
        n_steps (int): Length of the signal.
        batch_size (int): Number of batches.
        t_min (float): Minimum hold time of an amplitude.
        t_max (float): Maximum hold time of an amplitude
        key (jax.random.PRNGKey): Random key for JAX random sampling.
    """
    actions = []
    for _ in range(batch_size):
        subkey, key = jax.random.split(key)
        actions.append(aprbs_single_batch(n_steps, t_min, t_max, subkey)[..., None])
    return jnp.stack(actions, axis=0)


def build_grid(dim: int, low: float | list, high: float, points_per_dim: int) -> jax.Array:
    """Build a uniform grid of points in the given dimension.

    Args:
        dim (int): Dimensionality of the grid (and feature vector z)
        low (float): The minimum value of the grid in each dimension
        high (float): The maximum value of the grid in each dimension
        points_per_dim (int): The number of grid points per dimension. Always identical for each
            dimension

    Returns:
        The flattened grid as a jax.Array with shape (points_per_dim**dim, dim)
    """
    if isinstance(low, list) and isinstance(high, list):
        assert len(low) == len(high)
        assert len(low) == dim
        xs = [jnp.linspace(low[i], high[i], points_per_dim) for i in range(dim)]

    elif isinstance(low, float) and isinstance(high, float):
        xs = [jnp.linspace(low, high, points_per_dim) for _ in range(dim)]

    z_g = jnp.meshgrid(*xs, indexing="ij")
    z_g = jnp.stack([_x for _x in z_g], axis=-1)
    z_g = z_g.reshape(-1, dim)

    assert z_g.shape[0] == points_per_dim**dim
    return z_g


def estimate_eigendynamics_grid(ode, params, omega, u_dq):
    points_per_dim = 15
    i_dq = build_grid(2, low=[-350, -350], high=[0, 350], points_per_dim=points_per_dim)
    i_dq = i_dq.reshape(int(points_per_dim), int(points_per_dim), 2)[::2, :, ...].reshape(-1, 2)

    didt_map = eqx.filter_vmap(ode, in_axes=(0, None, None, None))(i_dq, u_dq, omega, params)
    x, y = i_dq[:, 0], i_dq[:, 1]
    u, v = didt_map[:, 0], didt_map[:, 1]

    return x, y, u, v


def visualize_eigendynamics(ode, params, n_values):

    fig, axs = plt.subplots(
        1,
        len(n_values),
        figsize=(16 / 5 * len(n_values), 8),
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axs = axs[0]

    quiver_data = []
    max_length_all = 0

    for idx, n in enumerate(n_values):

        axs[idx].title.set_text(f"$n={n}$" + r" $\mathrm{min}^{-1}$")

        x, y, u, v = estimate_eigendynamics_grid(
            ode,
            params,
            omega=jnp.array(params.p * n * 2 * jnp.pi / 60),
            u_dq=jnp.zeros(2),  # eigendynamics -> no voltage applied
        )

        quiver_data.append({"x": x, "y": y, "u": u, "v": v})

        max_length = jnp.max(jnp.linalg.norm(jnp.stack([u, v], axis=-1), axis=-1))
        max_length_all = max_length_all if max_length_all > max_length else max_length

    for idx, data in enumerate(quiver_data):
        axs[idx].quiver(
            data["x"], data["y"], data["u"], data["v"], scale=max_length_all.item() * 5, color="b", alpha=0.5
        )

    for ax in axs:
        ax.grid(True, alpha=0.5)
        ax.set_ylim(-350, 350)
        ax.set_xlim(-350, 10)
        ax.set_xlabel(r"$i_\mathrm{d} \; \mathrm{in} \; \mathrm{A}$")
    axs[0].set_ylabel(r"$i_\mathrm{q} \; \mathrm{in} \; \mathrm{A}$", labelpad=-25)

    return fig, axs


def visualize_trajectories(
    i_dq_sequences: list[jax.Array],
    u_dq_sequences: list[jax.Array],
    T_s_list: list[float],
    omegas: list[float],
    ode: Callable,
    params: eqx.Module,
    labels=None,
):
    colors = plt.rcParams["axes.prop_cycle"]()

    fig = plt.figure(figsize=(9, 7), constrained_layout=True)

    gs = gridspec.GridSpec(
        4,
        2,
        figure=fig,
        width_ratios=[1.2, 1],
    )

    # plots on the left
    ax_left = [fig.add_subplot(gs[0, 0])]
    for i in range(1, 4):
        ax_left.append(fig.add_subplot(gs[i, 0], sharex=ax_left[0]))
    for ax in ax_left[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    # plots on the right
    ax_right_top = fig.add_subplot(gs[0:2, 1])
    ax_right_bot = fig.add_subplot(gs[2:4, 1])
    ax_right_top.set_aspect("equal")
    ax_right_bot.set_aspect("equal")

    ###

    for i_dq_sequence, u_dq_sequence, T_s, omega in zip(i_dq_sequences, u_dq_sequences, T_s_list, omegas):

        color = next(colors)["color"]

        N_datapoints = i_dq_sequence.shape[0]
        t = jnp.linspace(0, (T_s * N_datapoints), N_datapoints)

        for plot_idx, data in enumerate(
            [i_dq_sequence[..., 0], i_dq_sequence[..., 1], u_dq_sequence[..., 0], u_dq_sequence[..., 1]]
        ):
            if t.shape[0] - 1 == data.shape[0]:
                t_plot = t[:-1]
            else:
                t_plot = t

            ax_left[plot_idx].plot(t_plot, data, color=color, alpha=0.7)

        # only shows dynamics if u_dq is constant
        if jnp.unique(u_dq_sequence, axis=0).shape == (1, 2):
            x, y, u, v = estimate_eigendynamics_grid(ode=ode, params=params, omega=omega, u_dq=u_dq_sequence[0])
            ax_right_top.quiver(x, y, u, v, scale=None, color="tab:purple", alpha=0.2)

        ax_right_top.plot(i_dq_sequence[..., 0], i_dq_sequence[..., 1], color=color, alpha=0.7)
        ax_right_bot.plot(u_dq_sequence[..., 0], u_dq_sequence[..., 1], color=color, alpha=0.7)

    ###

    if labels is not None:
        handles = ax_left[0].get_lines()
        fig.legend(
            handles=handles[: len(labels)],
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.07),
            ncol=len(labels),
            frameon=False,
        )

    ax_left[-1].set_xlabel("$t$ in s")
    for ax, label in zip(
        ax_left,
        [
            r"$i_\mathrm{d} \; \mathrm{in} \; \mathrm{A}$",
            r"$i_\mathrm{q} \; \mathrm{in} \; \mathrm{A}$",
            r"$u_\mathrm{d} \; \mathrm{in} \; \mathrm{V}$",
            r"$u_\mathrm{q} \; \mathrm{in} \; \mathrm{V}$",
        ],
    ):
        T_s_min = jnp.min(jnp.array(T_s_list))
        t_min = jnp.linspace(0, (T_s_min * N_datapoints), N_datapoints)

        ax.set_xlim(t_min[0], t_min[-1])
        ax.set_ylabel(label)

    ax_left[0].set_ylim(-350, 350)
    ax_left[1].set_ylim(-350, 350)
    ax_left[2].set_ylim(-400, 400)
    ax_left[3].set_ylim(-400, 400)

    ax_right_top.set_xlabel(r"$i_\mathrm{d} \; \mathrm{in} \; \mathrm{A}$")
    ax_right_top.set_ylabel(r"$i_\mathrm{q} \; \mathrm{in} \; \mathrm{A}$")
    ax_right_bot.set_xlabel(r"$u_\mathrm{d} \; \mathrm{in} \; \mathrm{V}$")
    ax_right_bot.set_ylabel(r"$u_\mathrm{q} \; \mathrm{in} \; \mathrm{V}$")

    ax_right_top.set_xlim(-350, 10)
    ax_right_top.set_ylim(-350, 350)

    ax_right_bot.set_xlim(-400, 400)
    ax_right_bot.set_ylim(-400, 400)

    all_axs = [ax for ax in ax_left] + [ax_right_top, ax_right_bot]

    for ax in all_axs:
        ax.grid(True, alpha=0.5)
        ax.tick_params(which="major", axis="y", direction="in")
        ax.tick_params(which="both", axis="x", direction="in")

    return fig, all_axs


def visualize_flux(lut_raw):

    selected_fontsize = 20

    psi_d = lut_raw["Psi_d"]
    psi_q = lut_raw["Psi_q"]
    i_d = lut_raw["i_d_vec"]
    i_q = lut_raw["i_q_vec"]

    X, Y = np.meshgrid(i_d, i_q)
    R = psi_d
    Z = psi_q

    fig20 = plt.figure(figsize=(8, 8))

    gs = gridspec.GridSpec(
        1,
        2,
        figure=fig20,
        left=0.12,
        bottom=0.18,
        right=1.0,
        top=0.99,
        wspace=0.05,
        hspace=0,
        width_ratios=[2, 2],
    )

    ax20 = fig20.add_subplot(
        gs[0, 0],
        projection="3d",
    )
    ax20.plot_surface(X, Y, R, rstride=1, cstride=1, cmap=cm.winter, linewidth=0, antialiased=False)
    ax20.view_init(35, 240, 0)
    ax20.set_xlabel(r"$i_{\mathrm{d}}\ \mathrm{in \ A}$", fontsize=selected_fontsize, loc="center")
    ax20.set_ylabel(r"$i_{\mathrm{q}}\ \mathrm{in \ A}$", fontsize=selected_fontsize, loc="top")
    ax20.zaxis.set_rotate_label(False)
    ax20.set_zlabel(r"$\psi_{\mathrm{d}}\ \mathrm{in \ Vs}$", fontsize=selected_fontsize, rotation=90)

    ax20.set_xticks([-200, -100, 0])
    ax20.set_yticks([-200, 0, 200])
    ax20.set_zticks([0, 0.03, 0.06])
    ax20.set_zlim([0, 0.065])

    ax20.zaxis.labelpad = 15.0
    ax20.xaxis.labelpad = 15.0
    ax20.yaxis.labelpad = 15.0
    ax20.set_box_aspect(None, zoom=0.85)
    # ax20.set_adjustable("box")
    ax20.tick_params(axis="both", direction="in")

    ax21 = fig20.add_subplot(gs[0, 1], projection="3d")
    ax21.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.winter, linewidth=0, antialiased=False)
    ax21.view_init(35, 240, 0)
    ax21.set_xlabel(r"$i_{\mathrm{d}}\ \mathrm{in \ A}$", fontsize=selected_fontsize)
    ax21.set_ylabel(r"$i_{\mathrm{q}}\ \mathrm{in \ A}$", fontsize=selected_fontsize)
    ax21.zaxis.set_rotate_label(False)
    ax21.set_zlabel(r"$\psi_{\mathrm{q}}\ \mathrm{in \ Vs}$", fontsize=selected_fontsize, rotation=90)
    ax21.set_xticks([-200, -100, 0])
    ax21.set_yticks([-200, 0, 200])
    ax21.set_zticks([-0.1, 0, 0.1])
    ax21.set_zlim([-0.15, 0.15])

    ax21.zaxis.labelpad = 15.0
    ax21.xaxis.labelpad = 15.0
    ax21.yaxis.labelpad = 15.0

    ax21.set_box_aspect(None, zoom=0.85)
    # ax21.set_adjustable("box")
    ax21.tick_params(axis="both", direction="in")

    fig20.set_size_inches(fig20.get_size_inches() + [4, 0])
    return fig20


def visualize_diff_inductance(lut_raw):

    selected_fontsize = 20

    i_d = lut_raw["i_d_vec"]
    i_q = lut_raw["i_q_vec"]

    X, Y = np.meshgrid(i_d, i_q)
    L_dd = lut_raw["L_dd"]
    L_dq = lut_raw["L_dq"]
    L_qd = lut_raw["L_qd"]
    L_qq = lut_raw["L_qq"]

    fig4 = plt.figure(figsize=(8, 8))

    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig4,
        left=0.05,
        bottom=0.05,
        right=0.95,
        top=0.95,
        wspace=0.15,
        hspace=0.15,
        width_ratios=[2, 2],
    )

    ########################################################################
    ## L_dd
    ########################################################################
    ax41 = fig4.add_subplot(
        gs[0, 0],
        projection="3d",
    )
    ax41.plot_surface(X, Y, L_dd * 1000, rstride=1, cstride=1, cmap=cm.winter, linewidth=0, antialiased=False)
    ax41.view_init(35, 240, 0)
    ax41.set_xlabel(r"$i_{\mathrm{d}}\ \mathrm{in \ A}$", fontsize=selected_fontsize, loc="center")
    ax41.set_ylabel(r"$i_{\mathrm{q}}\ \mathrm{in \ A}$", fontsize=selected_fontsize, loc="top")
    ax41.zaxis.set_rotate_label(False)
    ax41.set_zlabel(r"$L_{\mathrm{dd}}\ \mathrm{in \ mH}$", fontsize=selected_fontsize, rotation=90)

    ax41.set_xticks([-200, -100, 0])
    ax41.set_yticks([-200, 0, 200])
    ax41.set_zticks([0.2, 0.3, 0.4, 0.5, 0.6])
    ax41.set_zlim([0.2, 0.5])

    ax41.zaxis.labelpad = 7.0
    ax41.xaxis.labelpad = 12.5
    ax41.yaxis.labelpad = 12.5
    ax41.tick_params(axis="both", direction="in")

    ########################################################################
    ## L_dq
    ########################################################################
    ax42 = fig4.add_subplot(gs[0, 1], projection="3d")
    ax42.plot_surface(X, Y, L_dq * 1000, rstride=1, cstride=1, cmap=cm.winter, linewidth=0, antialiased=False)
    ax42.view_init(35, 240, 0)
    ax42.set_xlabel(r"$i_{\mathrm{d}}\ \mathrm{in \ A}$", fontsize=selected_fontsize)
    ax42.set_ylabel(r"$i_{\mathrm{q}}\ \mathrm{in \ A}$", fontsize=selected_fontsize)
    ax42.zaxis.set_rotate_label(False)
    ax42.set_zlabel(r"$L_{\mathrm{dq}}\ \mathrm{in \ mH}$", fontsize=selected_fontsize, rotation=90)
    ax42.set_xticks([-200, -100, 0])
    ax42.set_yticks([-200, 0, 200])
    ax42.set_zticks([-0.1, 0, 0.1])
    ax42.set_zlim([-0.15, 0.15])
    ax42.zaxis.labelpad = 7.0
    ax42.xaxis.labelpad = 12.5
    ax42.yaxis.labelpad = 12.5

    # ax42.set_box_aspect([1, 1, 1])
    # ax42.set_adjustable("box")
    ax42.tick_params(axis="both", direction="in")

    ########################################################################
    ## L_qd
    ########################################################################
    ax43 = fig4.add_subplot(gs[1, 0], projection="3d")
    ax43.plot_surface(X, Y, L_qd * 1000, rstride=1, cstride=1, cmap=cm.winter, linewidth=0, antialiased=False)
    ax43.view_init(35, 240, 0)
    ax43.set_xlabel(r"$i_{\mathrm{d}}\ \mathrm{in \ A}$", fontsize=selected_fontsize)
    ax43.set_ylabel(r"$i_{\mathrm{q}}\ \mathrm{in \ A}$", fontsize=selected_fontsize)
    ax43.zaxis.set_rotate_label(False)
    ax43.set_zlabel(r"$L_{\mathrm{qd}}\ \mathrm{in \ mH}$", fontsize=selected_fontsize, rotation=90)
    ax43.set_xticks([-200, -100, 0])
    ax43.set_yticks([-200, 0, 200])
    ax43.set_zticks([-0.1, 0, 0.1])
    ax43.set_zlim([-0.15, 0.15])
    ax43.zaxis.labelpad = 7.0
    ax43.xaxis.labelpad = 12.5
    ax43.yaxis.labelpad = 12.5
    ax43.tick_params(axis="both", direction="in")

    # ax43.set_box_aspect([1, 1, 1])
    # ax43.set_adjustable("box")

    ########################################################################
    ## L_qq
    ########################################################################
    ax44 = fig4.add_subplot(gs[1, 1], projection="3d")
    ax44.plot_surface(X, Y, L_qq * 1000, rstride=1, cstride=1, cmap=cm.winter, linewidth=0, antialiased=False)
    ax44.view_init(35, 240, 0)
    ax44.set_xlabel(r"$i_{\mathrm{d}}\ \mathrm{in \ A}$", fontsize=selected_fontsize)
    ax44.set_ylabel(r"$i_{\mathrm{q}}\ \mathrm{in \ A}$", fontsize=selected_fontsize)
    ax44.zaxis.set_rotate_label(False)
    ax44.set_zlabel(r"$L_{\mathrm{qq}}\ \mathrm{in \ mH}$", fontsize=selected_fontsize, rotation=90)
    ax44.set_xticks([-200, -100, 0])
    ax44.set_yticks([-200, 0, 200])
    ax44.set_zticks([0, -0.5, 1])
    ax44.set_zlim([0, 1.5])
    ax44.zaxis.labelpad = 7.0
    ax44.xaxis.labelpad = 12.5
    ax44.yaxis.labelpad = 12.5

    # ax44.set_box_aspect([1, 1, 1])
    # ax44.set_adjustable("box")
    ax44.tick_params(axis="both", direction="in")
    ax44.get_xticklabels()

    fig4.patches.append(
        plt.Rectangle(
            (0, 0),  # x, y in figure coordinates (0-1)
            1,
            1,  # width, height
            transform=fig4.transFigure,
            facecolor="white",
            edgecolor="none",
            zorder=-1,  # behind everything
        )
    )
    # ax44.set_xticks([[-0,-0.1],[-100,0.1],[-1.5,.1]], ['200','100','100'])

    return fig4
