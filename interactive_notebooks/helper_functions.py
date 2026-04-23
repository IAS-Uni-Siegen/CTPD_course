from functools import partial
import ipywidgets as widgets

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp


def simplified_clarke_transformation(x_abc):
    T_abc2albet = jnp.array(
        [
            [2 / 3, -1 / 3, -1 / 3],
            [0, 1 / jnp.sqrt(3), -1 / jnp.sqrt(3)],
        ]
    )
    return T_abc2albet @ x_abc


def compute_pwm_space_vectors():

    all_possible_states = jnp.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
    )
    return jax.vmap(simplified_clarke_transformation)(all_possible_states)


def setup_left_plot(t, c_t, u_dc, compute_three_phase_signals):
    fig_left, axs = plt.subplots(
        5, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1, 1.5]}, constrained_layout=True
    )

    u_ref_t, s_ref_t, s_t = compute_three_phase_signals(m=0.7, u_ref_freq=5, t=t, c_t=c_t, u_dc=u_dc)
    (switching_reference_line_A,) = axs[0].plot(t, s_ref_t[..., 0], color="tab:blue", label="$s^*_\\mathrm{a}$")
    (switching_reference_line_B,) = axs[0].plot(t, s_ref_t[..., 1], color="tab:red", label="$s^*_\\mathrm{b}$")
    (switching_reference_line_C,) = axs[0].plot(t, s_ref_t[..., 2], color="black", label="$s^*_\\mathrm{c}$")
    axs[0].plot(t, c_t, color="grey", alpha=0.7, label="$c$")
    (switching_state_line_A,) = axs[1].step(t, s_t[..., 0], color="tab:blue")
    (switching_state_line_B,) = axs[2].step(t, s_t[..., 1], color="tab:red")
    (switching_state_line_C,) = axs[3].step(t, s_t[..., 2], color="black")

    (diff_line,) = axs[4].step(t, (s_t[..., 0] - s_t[..., 1]) / 2, color="tab:orange")

    position_lines = [ax.axvline(x=t[0], color="black", linestyle="--", linewidth=1.5) for ax in axs]

    for ax in axs:
        ax.grid(alpha=0.5)
        ax.tick_params(which="major", axis="y", direction="in")
        ax.tick_params(which="both", axis="x", direction="in")
        ax.set_xlim(t[0], t[-1])

    axs[0].legend(
        prop={"size": 16},
        framealpha=0.5,
        loc="upper center",
        fancybox=True,
        shadow=False,
        ncols=5,
    )
    axs[0].set_ylim(-1.1, 1.1)

    axs[0].set_ylabel("$s^*_i(t)$,$c(t)$")
    axs[1].set_ylabel("$s_\\mathrm{a}(t)$")
    axs[2].set_ylabel("$s_\\mathrm{b}(t)$")
    axs[3].set_ylabel("$s_\\mathrm{c}(t)$")
    axs[4].set_ylabel("$\\frac{u_\\mathrm{a-b}(t)}{u_\\mathrm{dc}}$")
    axs[-1].set_xlabel("$t$ in $s$")

    lines_left = {
        "switching_reference_line_A": switching_reference_line_A,
        "switching_reference_line_B": switching_reference_line_B,
        "switching_reference_line_C": switching_reference_line_C,
        "switching_state_line_A": switching_state_line_A,
        "switching_state_line_B": switching_state_line_B,
        "switching_state_line_C": switching_state_line_C,
        "diff_line": diff_line,
        "position_lines": position_lines,
    }

    return fig_left, lines_left


def update_left_plot(position, fig_left, lines_left, s_ref_t, s_t):

    for line, data in zip(
        list(lines_left.values())[:6],
        [
            s_ref_t[..., 0],
            s_ref_t[..., 1],
            s_ref_t[..., 2],
            s_t[..., 0],
            s_t[..., 1],
            s_t[..., 2],
        ],
    ):
        line.set_ydata(data)

    lines_left["diff_line"].set_ydata((s_t[..., 0] - s_t[..., 1]) / 2)

    for vline in lines_left["position_lines"]:
        vline.set_xdata([position, position])

    fig_left.canvas.draw()


def setup_right_plot():
    fig_right, axs = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)

    axs.set_xlabel(r"$u_\alpha / \frac{u_\mathrm{dc}}{2}$")
    axs.set_ylabel(r"$u_\beta / \frac{u_\mathrm{dc}}{2}$")
    lim = 1.5
    axs.set_xlim(-lim, lim)
    axs.set_ylim(-lim, lim)
    axs.set_aspect("equal")
    axs.grid(alpha=0.5)

    norm_u_alpha_beta = compute_pwm_space_vectors()

    for i in range(norm_u_alpha_beta.shape[0]):
        axs.annotate(
            "",
            xy=(norm_u_alpha_beta[i, 0], norm_u_alpha_beta[i, 1]),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="lightgrey", lw=1.5),
        )
    axs.axhline(0, color="black", linewidth=0.5)
    axs.axvline(0, color="black", linewidth=0.5)

    active_arrow = axs.annotate(
        "", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2.5)
    )
    reference_arrow = axs.annotate(
        "", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="tab:orange", lw=2.5)
    )

    (active_dot,) = axs.plot(0, 0, "o", color="tab:blue", markersize=8, visible=False)

    lines_right = {"active_arrow": active_arrow, "reference_arrow": reference_arrow, "active_dot": active_dot}

    return fig_right, lines_right


def update_right_plot(position, fig_right, lines_right, t, s_ref_t, s_t):
    idx = jnp.argmin(jnp.abs(t - position))
    norm_u_albet = simplified_clarke_transformation(s_t[idx])

    if jnp.all(norm_u_albet == 0):
        lines_right["active_dot"].set_visible(True)
    else:
        lines_right["active_dot"].set_visible(False)

    lines_right["active_arrow"].xy = (norm_u_albet[0], norm_u_albet[1])

    norm_u_albet_ref = simplified_clarke_transformation(s_ref_t[idx])

    lines_right["reference_arrow"].xy = (norm_u_albet_ref[0], norm_u_albet_ref[1])

    fig_right.canvas.draw()


def _update(
    m, u_ref_freq, position, fig_left, lines_left, fig_right, lines_right, t, c_t, u_dc, compute_three_phase_signals
):
    _, s_ref_t, s_t = compute_three_phase_signals(m, u_ref_freq, t=t, c_t=c_t, u_dc=u_dc)
    update_left_plot(position, fig_left, lines_left, s_ref_t, s_t)
    update_right_plot(position, fig_right, lines_right, t, s_ref_t, s_t)


# def setup_ui(t, c_t, u_dc, compute_three_phase_signals):
#     fig_left, lines_left = setup_left_plot(t, c_t, u_dc, compute_three_phase_signals)
#     fig_right, lines_right = setup_right_plot()

#     update = partial(
#         _update,
#         fig_left=fig_left,
#         lines_left=lines_left,
#         fig_right=fig_right,
#         lines_right=lines_right,
#         t=t,
#         c_t=c_t,
#         u_dc=u_dc,
#         compute_three_phase_signals=compute_three_phase_signals,
#     )

#     fig_left_width_px = fig_left.get_size_inches()[0] * fig_left.dpi
#     m_slider = widgets.FloatSlider(
#         min=0,
#         max=1,
#         value=0.7,
#         step=0.025,
#         description="$m$",
#         layout=widgets.Layout(width=f"{fig_left_width_px * 6/7}px"),
#         readout_format=".3f",
#     )
#     u_ref_freq_slider = widgets.FloatSlider(
#         min=1,
#         max=25,
#         value=2.5,
#         step=0.5,
#         description="$\\omega^*$",
#         layout=widgets.Layout(width=f"{fig_left_width_px * 6/7}px"),
#     )
#     position_slider = widgets.FloatSlider(
#         min=t[0],
#         max=t[-1],
#         value=t[0],
#         step=0.002,
#         description="$t$",
#         readout_format=".3f",
#         layout=widgets.Layout(width=f"{fig_left_width_px * 6/7}px"),
#     )

#     ui = widgets.VBox(
#         [
#             widgets.HBox([fig_left.canvas, fig_right.canvas]),
#             widgets.HBox(
#                 [widgets.VBox([m_slider, u_ref_freq_slider, position_slider])],
#                 layout=widgets.Layout(justify_content="center"),
#             ),
#         ]
#     )

#     out = widgets.interactive_output(
#         update,
#         {
#             "m": m_slider,
#             "u_ref_freq": u_ref_freq_slider,
#             "position": position_slider,
#         },
#     )

#     return ui, out, (fig_left, lines_left, fig_right, lines_right)


class InteractivePWMVisualizer:

    def __init__(self, t, c_t, u_dc, compute_three_phase_signals):
        fig_left, lines_left = setup_left_plot(t, c_t, u_dc, compute_three_phase_signals)
        fig_right, lines_right = setup_right_plot()

        update = partial(
            _update,
            fig_left=fig_left,
            lines_left=lines_left,
            fig_right=fig_right,
            lines_right=lines_right,
            t=t,
            c_t=c_t,
            u_dc=u_dc,
            compute_three_phase_signals=compute_three_phase_signals,
        )

        fig_left_width_px = fig_left.get_size_inches()[0] * fig_left.dpi
        m_slider = widgets.FloatSlider(
            min=0,
            max=1,
            value=0.7,
            step=0.025,
            description="$m$",
            layout=widgets.Layout(width=f"{fig_left_width_px * 6/7}px"),
            readout_format=".3f",
        )
        u_ref_freq_slider = widgets.FloatSlider(
            min=1,
            max=25,
            value=2.5,
            step=0.5,
            description="$\\omega^*$",
            layout=widgets.Layout(width=f"{fig_left_width_px * 6/7}px"),
        )
        position_slider = widgets.FloatSlider(
            min=t[0],
            max=t[-1],
            value=t[0],
            step=0.002,
            description="$t$",
            readout_format=".3f",
            layout=widgets.Layout(width=f"{fig_left_width_px * 6/7}px"),
        )

        self.ui = widgets.VBox(
            [
                widgets.HBox([fig_left.canvas, fig_right.canvas]),
                widgets.HBox(
                    [widgets.VBox([m_slider, u_ref_freq_slider, position_slider])],
                    layout=widgets.Layout(justify_content="center"),
                ),
            ]
        )

        self.out = widgets.interactive_output(
            update,
            {
                "m": m_slider,
                "u_ref_freq": u_ref_freq_slider,
                "position": position_slider,
            },
        )
