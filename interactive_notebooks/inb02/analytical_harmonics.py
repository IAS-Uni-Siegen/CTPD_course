import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv

FREQUENCY_KEY_DECIMALS = 10
COEFFICIENT_TOLERANCE = 1e-14


def add_harmonic(
    spectrum: dict[float, dict[str, float]],
    normalized_frequency: float,
    *,
    sine_coefficient: float = 0.0,
    cosine_coefficient: float = 0.0,
) -> None:
    if abs(sine_coefficient) < COEFFICIENT_TOLERANCE and abs(cosine_coefficient) < COEFFICIENT_TOLERANCE:
        return

    normalized_frequency = float(normalized_frequency)
    if np.isclose(normalized_frequency, 0.0):
        return

    if normalized_frequency < 0.0:
        normalized_frequency = -normalized_frequency
        sine_coefficient = -sine_coefficient

    frequency_key = round(normalized_frequency, FREQUENCY_KEY_DECIMALS)
    if frequency_key not in spectrum:
        spectrum[frequency_key] = {
            "normalized_frequency": normalized_frequency,
            "sine_coefficient": 0.0,
            "cosine_coefficient": 0.0,
        }

    spectrum[frequency_key]["sine_coefficient"] += float(sine_coefficient)
    spectrum[frequency_key]["cosine_coefficient"] += float(cosine_coefficient)


def build_analytical_spectrum(
    pulse_number: float,
    modulation_index: float,
    max_carrier_group_order: int = 2,
    max_sideband_order: int = 3,
    fundamental_frequency_hz: float = 100,
):
    spectrum: dict[float, dict[str, float]] = {}
    add_harmonic(spectrum, 1.0, sine_coefficient=modulation_index)
    for k in range(max_carrier_group_order + 1):
        odd_group_order = 2 * k + 1
        odd_group_factor = -(4.0 / np.pi) * ((-1) ** k) / odd_group_order
        odd_group_argument = odd_group_order * np.pi * modulation_index / 2.0
        odd_group_center = odd_group_order * pulse_number

        add_harmonic(
            spectrum,
            odd_group_center,
            cosine_coefficient=odd_group_factor * jv(0, odd_group_argument),
        )

        for nu in range(1, max_sideband_order + 1):
            coefficient = odd_group_factor * jv(2 * nu, odd_group_argument)
            sideband_offset = 2.0 * nu
            add_harmonic(spectrum, odd_group_center + sideband_offset, cosine_coefficient=coefficient)
            add_harmonic(spectrum, odd_group_center - sideband_offset, cosine_coefficient=coefficient)

    for k in range(1, max_carrier_group_order + 1):
        even_group_factor = (2.0 / np.pi) * ((-1) ** k) / k
        even_group_argument = k * np.pi * modulation_index
        even_group_center = 2.0 * k * pulse_number

        for nu in range(max_sideband_order + 1):
            coefficient = even_group_factor * jv(2 * nu + 1, even_group_argument)
            sideband_offset = 2.0 * nu + 1.0
            add_harmonic(spectrum, even_group_center + sideband_offset, sine_coefficient=coefficient)
            add_harmonic(spectrum, sideband_offset - even_group_center, sine_coefficient=coefficient)

    rows = []
    for frequency_key in sorted(spectrum):
        entry = spectrum[frequency_key]
        amplitude = float(np.hypot(entry["sine_coefficient"], entry["cosine_coefficient"]))
        if amplitude < COEFFICIENT_TOLERANCE:
            continue

        normalized_frequency = entry["normalized_frequency"]
        rows.append(
            {
                "normalized_frequency": normalized_frequency,
                "frequency_hz": normalized_frequency * fundamental_frequency_hz,
                "amplitude": amplitude,
                "sine_coefficient": entry["sine_coefficient"],
                "cosine_coefficient": entry["cosine_coefficient"],
            }
        )

    return rows


def visualize_analytical_spectrum(spectrum: dict[float, dict[str, float]]):
    fig, axs = plt.subplots(1, 1, figsize=(10, 3), constrained_layout=True)

    frequencies = []
    amplitudes = []

    for freq_dict in spectrum:
        frequencies.append(freq_dict["normalized_frequency"])
        amplitudes.append(freq_dict["amplitude"])

    axs.bar(frequencies, amplitudes)
    axs.set_xlim(0, 300)
    axs.set_yscale("log")
    return fig, axs
