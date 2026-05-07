import ipywidgets as widgets

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp


def triangular_signal(t, frequency=1.0, amplitude=1.0, phase=0.0):

    T_p = 1.0 / frequency
    t_normalized = t / T_p
    t_normalized = t_normalized + phase / (2 * jnp.pi)
    t_normalized = t_normalized % 1.0

    triangle = jnp.where(
        t_normalized < 0.5,
        1.0 - 4.0 * t_normalized,
        4.0 * t_normalized - 3.0,
    )

    return amplitude * triangle


def compute_three_phase_signals(m, u_ref_freq, t, c_t, u_dc):
    omega = u_ref_freq * 2 * jnp.pi
    s_ref_t = jnp.array(
        [
            m * jnp.sin(omega * t),
            m * jnp.sin(omega * t - jnp.pi * 2 / 3),
            m * jnp.sin(omega * t + jnp.pi * 2 / 3),
        ]
    ).T
    u_ref_t = u_dc / 2 * s_ref_t

    s_t = jnp.where(s_ref_t > c_t[..., None], 1, -1)

    return u_ref_t, s_ref_t, s_t


def three_phase_plot(t, s_ref_t, c_t, s_t, s_0_t=None):
    fig_left, axs = plt.subplots(
        5, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1, 1.5]}, constrained_layout=True
    )
    axs[0].plot(
        t, s_ref_t[..., 0], color="tab:blue", label="$s^*_\\mathrm{a}$" if s_0_t is None else "$s^{**}_\\mathrm{a}$"
    )
    axs[0].plot(
        t, s_ref_t[..., 1], color="tab:red", label="$s^*_\\mathrm{b}$" if s_0_t is None else "$s^{**}_\\mathrm{b}$"
    )
    axs[0].plot(
        t, s_ref_t[..., 2], color="black", label="$s^*_\\mathrm{c}$" if s_0_t is None else "$s^{**}_\\mathrm{c}$"
    )

    if s_0_t is not None:
        axs[0].plot(t, s_0_t, color="tab:purple", label="$s_0$", linestyle="dashed")

    axs[0].plot(t, c_t, color="grey", alpha=0.7, label="$c$")
    axs[1].step(t, s_t[..., 0], color="tab:blue")
    axs[2].step(t, s_t[..., 1], color="tab:red")
    axs[3].step(t, s_t[..., 2], color="black")
    axs[4].step(t, (s_t[..., 0] - s_t[..., 1]) / 2, color="tab:orange")

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

    axs[0].set_ylabel("$s^*_i(t)$,$c(t)$" if s_0_t is None else "$s^{**}_i(t)$,$c(t)$")
    axs[1].set_ylabel("$s_\\mathrm{a}(t)$")
    axs[2].set_ylabel("$s_\\mathrm{b}(t)$")
    axs[3].set_ylabel("$s_\\mathrm{c}(t)$")
    axs[4].set_ylabel("$\\frac{u_\\mathrm{a-b}(t)}{u_\\mathrm{dc}}$")
    axs[-1].set_xlabel("$t$ in $s$")

    return fig_left, axs


def get_fft_spectrum(x, f_s, N):
    spectrum = jnp.abs(jnp.fft.rfft(x, axis=0))
    freqs = jnp.fft.rfftfreq(N, d=1 / f_s)
    amps = (2 / N) * jnp.abs(spectrum)
    return amps, freqs


def plot_fft_spectrum(x, f_s, N, f_fundamental):

    amps, freqs = get_fft_spectrum(x, f_s, N)

    fig, axs = plt.subplots(4, 1, figsize=(10, 8), constrained_layout=True)

    for ax in axs:
        ax.set_ylabel("$\\hat{s}$")
        ax.grid(alpha=0.5)
        ax.tick_params(which="major", axis="y", direction="in")
        ax.tick_params(which="both", axis="x", direction="in")

    axs[0].bar(freqs[:2500] / f_fundamental, amps[:2500])
    axs[0].set_xlim(0, 300)

    axs[1].bar(freqs[:2500] / f_fundamental, amps[:2500])
    axs[1].set_xlim(85, 115)

    axs[2].bar(freqs[:2500] / f_fundamental, amps[:2500])
    axs[2].set_xlim(185, 215)

    axs[-1].bar(freqs[:100] / f_fundamental, amps[:100])
    axs[-1].set_xlim(0, 15)
    axs[-1].set_yscale("log")
    axs[-1].set_xlabel("$f / f_1$")

    return fig, axs


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


def get_amplitude_at_freq(amps, freqs, target_freq):
    idx = jnp.argmin(jnp.abs(freqs - target_freq))
    return amps[idx]


@jax.jit
def compute_THD(amps, freqs, f_fundamental, max_n=300):
    amp_fundamental = get_amplitude_at_freq(amps, freqs, f_fundamental)

    n = jnp.arange(3, max_n, 2)  # all odd numbers from 3 to max_n

    # parallelizes over the target frequencies but not over the other two inputs:
    target_freqs = n * f_fundamental
    amps_harmonic = jax.vmap(get_amplitude_at_freq, in_axes=(None, None, 0))(amps, freqs, target_freqs)

    return jnp.sqrt(jnp.sum(amps_harmonic**2) / amp_fundamental**2)


@jax.jit
def get_effective_value(x_t):
    return jnp.sqrt(jnp.mean(x_t**2))


class InteractivePWMVisualizer:

    def __init__(
        self,
        t,
        c_t,
        u_dc,
        compute_three_phase_signals,
        compute_three_phase_signals_zsi_minmax,
        compute_three_phase_signals_zsi_clipping,
        head_length=0.25,
    ):
        self.N = t.shape[0]
        self.f_sampling = self.N / (t[-1] - t[0])

        self.head_length = head_length
        self.t = t
        self.c_t = c_t
        self.u_dc = u_dc
        self.compute_three_phase_signals = compute_three_phase_signals
        self.compute_three_phase_signals_zsi_minmax = compute_three_phase_signals_zsi_minmax
        self.compute_three_phase_signals_zsi_clipping = compute_three_phase_signals_zsi_clipping

        self.fig_left, self.lines_left = self.setup_left_plot()
        self.fig_right, self.lines_right = self.setup_right_plot()

        fig_left_width_px = self.fig_left.get_size_inches()[0] * self.fig_left.dpi
        m_slider = widgets.FloatSlider(
            min=0,
            max=4 / 3,
            value=0.7,
            step=0.025,
            description="$m$",
            layout=widgets.Layout(width=f"{fig_left_width_px * 6/7}px"),
            readout_format=".3f",
        )
        u_ref_freq_slider = widgets.FloatSlider(
            min=1,
            max=25,
            value=10,
            step=0.25,
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

        pwm_mode_dropdown = widgets.Dropdown(
            options=["basic", "minmax_ZSI", "clipped_minmax_ZSI"],
            value="basic",
            description="PWM mode:",
            disabled=False,
        )

        self.ui = widgets.VBox(
            [
                widgets.HBox([self.fig_left.canvas, self.fig_right.canvas]),
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                m_slider,
                                u_ref_freq_slider,
                                position_slider,
                                pwm_mode_dropdown,
                            ]
                        ),
                    ],
                    layout=widgets.Layout(justify_content="center"),
                ),
            ]
        )

        self.out = widgets.interactive_output(
            self.update,
            {
                "m": m_slider,
                "u_ref_freq": u_ref_freq_slider,
                "position": position_slider,
                "pwm_mode": pwm_mode_dropdown,
            },
        )

    def setup_left_plot(self):
        t = self.t

        fig_left, axs = plt.subplots(
            5,
            1,
            figsize=(10, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1, 1, 1.5]},
            constrained_layout=True,
        )

        u_ref_t, s_ref_t, s_t = self.compute_three_phase_signals(m=0.7, u_ref_freq=5, t=t, c_t=self.c_t, u_dc=self.u_dc)
        (switching_reference_line_A,) = axs[0].plot(t, s_ref_t[..., 0], color="tab:blue", label="$s^*_\\mathrm{a}$")
        (switching_reference_line_B,) = axs[0].plot(t, s_ref_t[..., 1], color="tab:red", label="$s^*_\\mathrm{b}$")
        (switching_reference_line_C,) = axs[0].plot(t, s_ref_t[..., 2], color="black", label="$s^*_\\mathrm{c}$")
        axs[0].plot(t, self.c_t, color="grey", alpha=0.7, label="$c$")
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

    def setup_right_plot(self):
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
                arrowprops=dict(
                    arrowstyle=f"->, head_length={self.head_length}",
                    color="lightgrey",
                    lw=1.5,
                    shrinkA=0,
                    shrinkB=0,
                ),
            )
        axs.axhline(0, color="black", linewidth=0.5)
        axs.axvline(0, color="black", linewidth=0.5)

        active_arrow = axs.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle=f"->, head_length={self.head_length}",
                color="tab:purple",
                lw=2.5,
                shrinkA=0,
                shrinkB=0,
            ),
        )
        reference_arrow = axs.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle=f"->, head_length={self.head_length}",
                color="tab:cyan",
                lw=2.5,
                shrinkA=0,
                shrinkB=0,
            ),
        )

        (active_dot,) = axs.plot(0, 0, "o", color="tab:purple", markersize=8, visible=False)

        lines_right = {"active_arrow": active_arrow, "reference_arrow": reference_arrow, "active_dot": active_dot}
        axs.legend(
            [Line2D([0], [0], color="tab:purple"), Line2D([0], [0], color="tab:cyan")],
            ["active", "ref"],
            prop={"size": 16},
            framealpha=0.5,
            loc="upper center",
            fancybox=True,
            shadow=False,
            ncols=5,
        )

        return fig_right, lines_right

    def update_left_plot(self, position, fig_left, lines_left, s_ref_t, s_t):
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

    def update_right_plot(self, position, fig_right, lines_right, t, s_ref_t, s_t):
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

    def update(
        self,
        m,
        u_ref_freq,
        position,
        pwm_mode,
    ):
        if pwm_mode == "basic":
            _, s_ref_t, s_t = self.compute_three_phase_signals(m, u_ref_freq, t=self.t, c_t=self.c_t, u_dc=self.u_dc)
        elif pwm_mode == "minmax_ZSI":
            _, s_ref_t, s_t, s_0_t = self.compute_three_phase_signals_zsi_minmax(
                m, u_ref_freq, t=self.t, c_t=self.c_t, u_dc=self.u_dc
            )
        elif pwm_mode == "clipped_minmax_ZSI":
            _, s_ref_t, s_t, s_0_t = self.compute_three_phase_signals_zsi_clipping(
                m, u_ref_freq, t=self.t, c_t=self.c_t, u_dc=self.u_dc
            )

        amps, freqs = get_fft_spectrum(s_t[..., 0] - s_t[..., 1], self.f_sampling, self.N)
        THD = compute_THD(amps, freqs, u_ref_freq)

        effective_value = get_effective_value(s_t[..., 0] - s_t[..., 1])

        print(f"l2l THD:            {THD:.5f}")
        print(f"l2l eff. value:     {effective_value:.5f}")

        self.update_left_plot(position, self.fig_left, self.lines_left, s_ref_t, s_t)
        self.update_right_plot(position, self.fig_right, self.lines_right, self.t, s_ref_t, s_t)
