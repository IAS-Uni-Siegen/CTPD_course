from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

###################################################
# Some parameters #
###################################################

XN = 10000
WT = np.linspace(0, 2 * np.pi, XN)
PULSE_NUMBER = 10
EXAMPLE_MODULATION_INDICES = (1.24, 1.40, 2.35)
PHASE_LABELS = np.array(["a", "b", "c"])


###################################################
# Helper functions / function definitions #
###################################################


def c_saw(wt, pulse_number):
    return 1 - 2 * np.abs(np.modf((wt * pulse_number) / (2 * np.pi))[0])


def c_tri(wt, pulse_number):
    return 4 * (np.abs(np.modf((wt * pulse_number) / (2 * np.pi))[0] - 0.5)) - 1


def d_sin(wt, phi=0.0, amplitude=1.0):
    return amplitude * np.sin(wt - phi)


def clip_to_unit_interval(signal):
    return np.clip(signal, -1.0, 1.0)


def s_comp(reference, carrier):
    return np.where(reference > carrier, 1, -1)


def three_phase_references(wt, modulation_index):
    d_a = d_sin(wt, amplitude=modulation_index)
    d_b = d_sin(wt, phi=2 * np.pi / 3, amplitude=modulation_index)
    d_c = d_sin(wt, phi=4 * np.pi / 3, amplitude=modulation_index)
    return d_a, d_b, d_c


def flat_top_zero_injection(d_a, d_b, d_c):
    d_stack = np.vstack((d_a, d_b, d_c))
    dominant_phase_idx = np.argmax(np.abs(d_stack), axis=0)
    sample_idx = np.arange(d_stack.shape[1])
    dominant_phase = d_stack[dominant_phase_idx, sample_idx]
    dominant_sign = np.where(dominant_phase >= 0.0, 1.0, -1.0)
    d_zero = dominant_phase - dominant_sign
    d_mod = d_stack - d_zero
    return d_mod[0], d_mod[1], d_mod[2], d_zero, dominant_phase_idx


def transition_indices(binary_signal):
    transition_idx = np.where(np.diff(binary_signal) != 0)[0]
    if transition_idx.size == 0:
        return np.array([], dtype=int)
    return np.unique(np.concatenate((transition_idx, transition_idx + 1))).astype(int)


def extrema_indices(signal):
    maxima, _ = find_peaks(signal)
    minima, _ = find_peaks(-signal)
    return np.unique(np.concatenate((maxima, minima))).astype(int)


def reduced_sample_indices(carrier, switching_signals, analog_signals, dominant_phase_idx):
    key_indices = [np.array([0, len(carrier) - 1], dtype=int)]

    upper_peaks, _ = find_peaks(carrier)
    lower_peaks, _ = find_peaks(-carrier)
    key_indices.extend((upper_peaks, lower_peaks))

    for signal in switching_signals:
        key_indices.append(transition_indices(signal))

    for signal in analog_signals:
        saturation_mask = np.abs(signal) >= 1.0 - 1e-12
        key_indices.append(transition_indices(saturation_mask.astype(int)))
        key_indices.append(extrema_indices(signal))

    dominant_changes = np.where(np.diff(dominant_phase_idx) != 0)[0]
    if dominant_changes.size > 0:
        key_indices.append(dominant_changes)
        key_indices.append(dominant_changes + 1)

    return np.unique(np.concatenate(key_indices)).astype(int)


def modulation_tag(modulation_index):
    scaled = int(round(100 * modulation_index))
    if scaled % 100 == 0:
        return f"mod{scaled // 100}"
    return f"mod{scaled}"


def simulate_flat_top_dpwm(modulation_index, wt, carrier):
    d_a, d_b, d_c = three_phase_references(wt, modulation_index)
    d_a_mod, d_b_mod, d_c_mod, d_zero, dominant_phase_idx = flat_top_zero_injection(d_a, d_b, d_c)
    d_a_mod_clip = clip_to_unit_interval(d_a_mod)
    d_b_mod_clip = clip_to_unit_interval(d_b_mod)
    d_c_mod_clip = clip_to_unit_interval(d_c_mod)
    s_a = s_comp(d_a_mod_clip, carrier)
    s_b = s_comp(d_b_mod_clip, carrier)
    s_c = s_comp(d_c_mod_clip, carrier)

    idx = reduced_sample_indices(
        carrier,
        (s_a, s_b, s_c),
        (d_a, d_b, d_c, d_a_mod, d_b_mod, d_c_mod, d_a_mod_clip, d_b_mod_clip, d_c_mod_clip, d_zero),
        dominant_phase_idx,
    )

    return {
        "modulation_index": modulation_index,
        "carrier": carrier,
        "d_a": d_a,
        "d_b": d_b,
        "d_c": d_c,
        "d_a_mod": d_a_mod,
        "d_b_mod": d_b_mod,
        "d_c_mod": d_c_mod,
        "d_a_mod_clip": d_a_mod_clip,
        "d_b_mod_clip": d_b_mod_clip,
        "d_c_mod_clip": d_c_mod_clip,
        "d_zero": d_zero,
        "s_a": s_a,
        "s_b": s_b,
        "s_c": s_c,
        "dominant_phase_idx": dominant_phase_idx,
        "reduced_idx": idx,
    }


def save_to_csv(result, wt, output_directory):
    idx = result["reduced_idx"]
    modulation_index = result["modulation_index"]
    filename = f"PWM_three-phase_flat-top_{modulation_tag(modulation_index)}_example.csv"
    save_path = output_directory / filename

    np.savetxt(
        save_path,
        np.column_stack(
            (
                wt[idx],
                result["s_a"][idx],
                result["s_b"][idx],
                result["s_c"][idx],
                result["d_a"][idx],
                result["d_b"][idx],
                result["d_c"][idx],
                result["d_a_mod"][idx],
                result["d_b_mod"][idx],
                result["d_c_mod"][idx],
                result["d_a_mod_clip"][idx],
                result["d_b_mod_clip"][idx],
                result["d_c_mod_clip"][idx],
                result["d_zero"][idx],
                result["carrier"][idx],
                result["dominant_phase_idx"][idx],
            )
        ),
        delimiter=",",
        header="wt, sa, sb, sc, d_a, d_b, d_c, d_a_mod, d_b_mod, d_c_mod, d_a_mod_clip, d_b_mod_clip, d_c_mod_clip, d_zero, c, dominant_phase_idx",
        comments="",
    )


def region_label(modulation_index):
    if modulation_index <= 2 / np.sqrt(3) + 1e-12:
        return "linear range"
    return "overmodulation"


def plot_results(results, wt):
    fig, axes = plt.subplots(3, len(results), sharex=True, figsize=(5.0 * len(results), 8.0), squeeze=False)

    for column, result in enumerate(results):
        modulation_index = result["modulation_index"]
        title = rf"flat-top DPWM, $m^*={modulation_index:.3f}$ ({region_label(modulation_index)})"

        axes[0, column].plot(wt, result["carrier"], color="black", label=r"$c(t)$")
        axes[0, column].plot(wt, result["d_a"], label=r"$s_\mathrm{a}^*(t)$")
        axes[0, column].plot(wt, result["d_b"], label=r"$s_\mathrm{b}^*(t)$")
        axes[0, column].plot(wt, result["d_c"], label=r"$s_\mathrm{c}^*(t)$")
        axes[0, column].set_ylabel("original")
        axes[0, column].set_title(title)
        axes[0, column].grid()

        axes[1, column].plot(wt, result["carrier"], color="black", label=r"$c(t)$")
        axes[1, column].plot(wt, result["d_a_mod"], linestyle="--", label=r"$s_\mathrm{a}^{**}(t)$")
        axes[1, column].plot(wt, result["d_b_mod"], linestyle="--", label=r"$s_\mathrm{b}^{**}(t)$")
        axes[1, column].plot(wt, result["d_c_mod"], linestyle="--", label=r"$s_\mathrm{c}^{**}(t)$")
        axes[1, column].plot(wt, result["d_a_mod_clip"], linewidth=1.6, label=r"$\operatorname{clip}(s_\mathrm{a}^{**})$")
        axes[1, column].plot(wt, result["d_b_mod_clip"], linewidth=1.6, label=r"$\operatorname{clip}(s_\mathrm{b}^{**})$")
        axes[1, column].plot(wt, result["d_c_mod_clip"], linewidth=1.6, label=r"$\operatorname{clip}(s_\mathrm{c}^{**})$")
        axes[1, column].plot(wt, result["d_zero"], linestyle=":", color="gray", label=r"$s_0(t)$")
        axes[1, column].set_ylabel("modified")
        axes[1, column].grid()

        axes[2, column].plot(wt, result["s_a"], label=r"$s_\mathrm{a}(t)$")
        axes[2, column].plot(wt, result["s_b"], label=r"$s_\mathrm{b}(t)$")
        axes[2, column].plot(wt, result["s_c"], label=r"$s_\mathrm{c}(t)$")
        axes[2, column].set_xlabel(r"$\omega t$")
        axes[2, column].set_ylabel("switching")
        axes[2, column].grid()

    axes[0, -1].legend(loc="upper right", fontsize=9)
    axes[1, -1].legend(loc="upper right", fontsize=8)
    axes[2, -1].legend(loc="upper right", fontsize=9)

    fig.suptitle("Three-phase flat-top DPWM with representative linear and overmodulation cases")
    fig.tight_layout()
    plt.show()


def main():
    carrier = c_tri(WT, PULSE_NUMBER)
    output_directory = Path(__file__).resolve().parent
    results = [simulate_flat_top_dpwm(modulation_index, WT, carrier) for modulation_index in EXAMPLE_MODULATION_INDICES]

    for result in results:
        save_to_csv(result, WT, output_directory)
        dominant_phase = PHASE_LABELS[result["dominant_phase_idx"]]
        unique_phases = ", ".join(np.unique(dominant_phase))
        print(
            f"Saved {modulation_tag(result['modulation_index'])}: dominant clamped phases = {unique_phases}, "
            f"reduced samples = {len(result['reduced_idx'])}"
        )

    plot_results(results, WT)


if __name__ == "__main__":
    main()
