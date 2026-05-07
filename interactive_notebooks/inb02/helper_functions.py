import matplotlib.pyplot as plt

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


def get_fft_spectrum(x, f_s, N, f_fundamental):
    spectrum = jnp.abs(jnp.fft.rfft(x, axis=0))
    freqs = jnp.fft.rfftfreq(N, d=1 / f_s)
    amps = (2 / N) * jnp.abs(spectrum)
    return amps, freqs


def plot_fft_spectrum(x, f_s, N, f_fundamental):

    amps, freqs = get_fft_spectrum(x, f_s, N, f_fundamental)

    fig, axs = plt.subplots(4, 1, figsize=(10, 8), constrained_layout=True)

    for ax in axs:
        ax.set_ylabel("$\\hat{s}$")
        ax.grid(alpha=0.5)
        ax.tick_params(which="major", axis="y", direction="in")
        ax.tick_params(which="both", axis="x", direction="in")

    axs[0].bar(freqs[:1000] / f_fundamental, amps[:1000])
    axs[0].set_xlim(0, 300)

    axs[1].bar(freqs[:1000] / f_fundamental, amps[:1000])
    axs[1].set_xlim(85, 115)

    axs[2].bar(freqs[:1000] / f_fundamental, amps[:1000])
    axs[2].set_xlim(185, 215)

    axs[-1].bar(freqs[:50] / f_fundamental, amps[:50])
    axs[-1].set_xlim(0, 15)
    axs[-1].set_yscale("log")
    axs[-1].set_xlabel("$f / f^*$")

    return fig, axs
