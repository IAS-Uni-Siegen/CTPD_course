from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.special import jv


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from PWM_single_phase_harmonic_spectrum import (
    COEFFICIENT_TOLERANCE,
    DEFAULT_FUNDAMENTAL_FREQUENCY_HZ,
    DEFAULT_MAX_CARRIER_GROUP_ORDER,
    DEFAULT_MAX_SIDEBAND_ORDER,
    DEFAULT_MODULATION_INDEX,
    DEFAULT_PULSE_NUMBER,
    FREQUENCY_KEY_DECIMALS,
    SpectrumConfig,
    add_harmonic,
    compact_value_token,
    validate_config,
    write_spectrum_csv,
)


DEFAULT_FILENAME_STEM = "PWM_3ph_ln_harm"


def default_output_path(script_path: Path, modulation_index: float, pulse_number: float, fundamental_frequency_hz: float) -> Path:
    filename = (
        f"{DEFAULT_FILENAME_STEM}_"
        f"m{compact_value_token(modulation_index)}_"
        f"np{compact_value_token(pulse_number)}_"
        f"f{compact_value_token(fundamental_frequency_hz)}.csv"
    )
    return script_path.with_name(filename)


def parse_args() -> SpectrumConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the analytical harmonic spectrum of a three-phase PWM "
            "line-to-neutral voltage by removing triplen terms from the leg spectrum."
        )
    )
    parser.add_argument("--modulation-index", type=float, default=DEFAULT_MODULATION_INDEX)
    parser.add_argument("--pulse-number", type=float, default=DEFAULT_PULSE_NUMBER)
    parser.add_argument("--fundamental-frequency", type=float, default=DEFAULT_FUNDAMENTAL_FREQUENCY_HZ)
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_CARRIER_GROUP_ORDER)
    parser.add_argument("--max-nu", type=int, default=DEFAULT_MAX_SIDEBAND_ORDER)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = default_output_path(
            Path(__file__),
            args.modulation_index,
            args.pulse_number,
            args.fundamental_frequency,
        )

    return SpectrumConfig(
        modulation_index=args.modulation_index,
        pulse_number=args.pulse_number,
        fundamental_frequency_hz=args.fundamental_frequency,
        max_carrier_group_order=args.max_k,
        max_sideband_order=args.max_nu,
        output_path=output_path,
    )


def is_triplen(order_on_fundamental: int) -> bool:
    return order_on_fundamental % 3 == 0


def build_spectrum(config: SpectrumConfig) -> list[dict[str, float]]:
    pulse_number = config.pulse_number
    modulation_index = config.modulation_index
    spectrum: dict[float, dict[str, float]] = {}

    # The fundamental survives unchanged because its order on omega is 1.
    add_harmonic(spectrum, 1.0, sine_coefficient=modulation_index)

    for k in range(config.max_carrier_group_order + 1):
        odd_group_order = 2 * k + 1
        odd_group_factor = -(4.0 / np.pi) * ((-1) ** k) / odd_group_order
        odd_group_argument = odd_group_order * np.pi * modulation_index / 2.0
        odd_group_center = odd_group_order * pulse_number

        # The carrier-center component has order 0 on omega and is therefore triplen.
        for nu in range(1, config.max_sideband_order + 1):
            sideband_order = 2 * nu
            if is_triplen(sideband_order):
                continue

            coefficient = odd_group_factor * jv(sideband_order, odd_group_argument)
            sideband_offset = float(sideband_order)
            add_harmonic(spectrum, odd_group_center + sideband_offset, cosine_coefficient=coefficient)
            add_harmonic(spectrum, odd_group_center - sideband_offset, cosine_coefficient=coefficient)

    for k in range(1, config.max_carrier_group_order + 1):
        even_group_factor = (2.0 / np.pi) * ((-1) ** k) / k
        even_group_argument = k * np.pi * modulation_index
        even_group_center = 2.0 * k * pulse_number

        for nu in range(config.max_sideband_order + 1):
            sideband_order = 2 * nu + 1
            if is_triplen(sideband_order):
                continue

            coefficient = even_group_factor * jv(sideband_order, even_group_argument)
            sideband_offset = float(sideband_order)
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
                "frequency_hz": normalized_frequency * config.fundamental_frequency_hz,
                "amplitude": amplitude,
                "sine_coefficient": entry["sine_coefficient"],
                "cosine_coefficient": entry["cosine_coefficient"],
            }
        )

    return rows


def main() -> None:
    config = parse_args()
    validate_config(config)
    rows = build_spectrum(config)
    write_spectrum_csv(rows, config.output_path)
    print(
        "Saved "
        f"{len(rows)} line-to-neutral spectral lines to {config.output_path} "
        f"(m={config.modulation_index}, n_p={config.pulse_number}, "
        f"f_1={config.fundamental_frequency_hz} Hz, max k={config.max_carrier_group_order}, "
        f"max nu={config.max_sideband_order})."
    )


if __name__ == "__main__":
    main()