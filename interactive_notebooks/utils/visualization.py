from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jax
import jax.numpy as jnp
import equinox as eqx


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

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)

    linewidth = 2

    gs = gridspec.GridSpec(
        4,
        2,  # 4 rows, 2 columns
        figure=fig,
        width_ratios=[1.2, 1],  # left column 3x wider than right
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

            ax_left[plot_idx].plot(t_plot, data, color=color, zorder=1, linewidth=linewidth, alpha=0.8)

        if ode is not None and params is not None:
            # only shows dynamics if u_dq is constant
            if jnp.unique(u_dq_sequence, axis=0).shape == (1, 2):
                x, y, u, v = estimate_eigendynamics_grid(ode=ode, params=params, omega=omega, u_dq=u_dq_sequence[0])
                ax_right_top.quiver(x, y, u, v, scale=None, color="tab:purple", alpha=0.2)

        ax_right_top.plot(
            i_dq_sequence[..., 0],
            i_dq_sequence[..., 1],
            color=color,
            zorder=1,
            linewidth=linewidth,
        )
        ax_right_bot.plot(
            u_dq_sequence[..., 0],
            u_dq_sequence[..., 1],
            color=color,
            zorder=1,
            linewidth=linewidth,
        )

    ###

    if labels is not None:
        handles = ax_left[0].get_lines()
        fig.legend(
            handles=handles[: len(labels)],
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.07),  # centered, just below the figure
            ncol=len(labels),  # all entries in one row
            frameon=False,  # clean look, no box
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

    i_lim = params.i_lim
    i_d_plot = jnp.linspace(-i_lim, i_lim, 1000)
    i_q_plot = jnp.sqrt(i_lim**2 - i_d_plot**2)
    all_axs[4].plot(i_d_plot, i_q_plot, "r", alpha=0.5)
    all_axs[4].plot(
        i_d_plot,
        -i_q_plot,
        "r",
        alpha=0.5,
    )

    return fig, all_axs


def visualize_trajectories_with_reference(
    i_dq_sequences: list[jax.Array],
    u_dq_sequences: list[jax.Array],
    i_dq_ref_sequence: jax.Array,
    T_s_list: list[float],
    omegas: list[float],
    ode: Callable,
    params: eqx.Module,
    labels=None,
):
    fig, all_axes = visualize_trajectories(
        i_dq_sequences=i_dq_sequences,
        u_dq_sequences=u_dq_sequences,
        T_s_list=T_s_list,
        omegas=omegas,
        ode=ode,
        params=params,
        labels=labels,
    )

    if len(set(T_s_list)) == 1:
        T_s = T_s_list[0]

    N_datapoints = i_dq_ref_sequence.shape[0]
    t = jnp.linspace(0, (T_s * N_datapoints), N_datapoints)

    all_axes[0].plot(
        t,
        i_dq_ref_sequence[..., 0],
        c="k",
        linestyle="dashed",
        zorder=1,
        alpha=0.7,
    )
    all_axes[1].plot(
        t,
        i_dq_ref_sequence[..., 1],
        c="k",
        linestyle="dashed",
        zorder=1,
        alpha=0.7,
    )

    all_axes[4].plot(
        i_dq_ref_sequence[..., 0],
        i_dq_ref_sequence[..., 1],
        c="k",
        linestyle="dashed",
        marker="x",
        zorder=1,
        alpha=0.7,
    )

    return fig, all_axes
