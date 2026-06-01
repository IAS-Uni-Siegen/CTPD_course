#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate PGFPlots .dat files for the inverse OPS current-map surfaces

    i_sd^* / i_s,max = f_id(|psi|/psi_N, T/T_N)
    i_sq^* / i_s,max = f_iq(|psi|/psi_N, T/T_N)

The generated surfaces are plotted as a structured current-domain mesh mapped
forward to the (T, |psi|) plane. This avoids the boundary loss that can occur
when a sparse rectangular (T, |psi|) grid is filled by interpolation only.

Current-domain construction:
  - outer boundary is sampled exactly by the analytic MTPC, MC, and MTPV loci,
  - the interior is filled by a space-filling structured mesh between the
    left boundary (MTPV/MC) and the right boundary (MTPC),
  - positive and negative torque branches are included by signed i_q.

The output files are whitespace-separated PGFPlots tables with columns

    T_norm   psi_norm   i_norm

where i_norm is either i_sd/i_s,max or i_sq/i_s,max.

Dependencies: numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# Same normalized linear IPMSM model as in the OPS slides
# -----------------------------------------------------------------------------
PSI_PM = 1.1
L_D = 0.4
L_Q = 0.8
DELTA_L = L_Q - L_D
K_T = 1.0
I_MAX = 5.0


# -----------------------------------------------------------------------------
# Model equations
# -----------------------------------------------------------------------------
def torque(id_amp: float | np.ndarray, iq_amp: float | np.ndarray) -> float | np.ndarray:
    """Electromagnetic torque."""
    return K_T * (PSI_PM - DELTA_L * id_amp) * iq_amp


def flux_abs(id_amp: float | np.ndarray, iq_amp: float | np.ndarray) -> float | np.ndarray:
    """Stator flux magnitude."""
    psi_d = PSI_PM + L_D * id_amp
    psi_q = L_Q * iq_amp
    return np.sqrt(psi_d * psi_d + psi_q * psi_q)


# -----------------------------------------------------------------------------
# Characteristic OPS curves in normalized current coordinates
# x = i_d/I_MAX, y = i_q/I_MAX
# -----------------------------------------------------------------------------
def x_mtpc_norm(y_abs: float | np.ndarray) -> float | np.ndarray:
    """MTPC locus x_MTPC(|y|)."""
    y_abs = np.asarray(y_abs, dtype=float)
    iq = I_MAX * np.abs(y_abs)
    id_mtpc = PSI_PM / (2.0 * DELTA_L) - np.sqrt((PSI_PM / (2.0 * DELTA_L)) ** 2 + iq * iq)
    return id_mtpc / I_MAX


def x_mc_norm(y_abs: float | np.ndarray) -> float | np.ndarray:
    """Current-limit branch x_MC(|y|) = -sqrt(1-y^2)."""
    y_abs = np.asarray(y_abs, dtype=float)
    return -np.sqrt(np.maximum(0.0, 1.0 - y_abs * y_abs))


def x_mtpv_norm(y_abs: float | np.ndarray) -> float | np.ndarray:
    """MTPV locus x_MTPV(|y|), same negative-d branch as in the slides."""
    y_abs = np.asarray(y_abs, dtype=float)

    numerator_linear = -L_D * PSI_PM * (L_D * I_MAX - DELTA_L * I_MAX)
    denominator = 2.0 * L_D * (L_D * I_MAX) * (DELTA_L * I_MAX)
    quad_factor = L_D * (L_D * I_MAX) * (DELTA_L * I_MAX)
    constant = -L_D * PSI_PM * PSI_PM - DELTA_L * (L_Q * I_MAX) ** 2 * y_abs * y_abs

    discriminant = numerator_linear * numerator_linear - 4.0 * quad_factor * constant
    root = np.where(discriminant >= 0.0, np.sqrt(discriminant), np.nan)

    return -numerator_linear / denominator - root / denominator


def find_transition_points(search_points: int = 20001) -> tuple[float, float, float, float]:
    """
    Return (x_R, y_R, x_M, y_M), where
      R: MTPC intersection with MC/current limit,
      M: MTPV intersection with MC/current limit.
    """
    y = np.linspace(0.0, 1.0, search_points)
    x_mc = x_mc_norm(y)
    x_mtpc = x_mtpc_norm(y)
    x_mtpv = x_mtpv_norm(y)

    idx_R = int(np.nanargmin(np.abs(x_mtpc - x_mc)))
    idx_M = int(np.nanargmin(np.abs(x_mtpv - x_mc)))

    y_R = float(y[idx_R])
    y_M = float(y[idx_M])
    x_R = float(x_mtpc[idx_R])
    x_M = float(x_mc[idx_M])

    return x_R, y_R, x_M, y_M


def normalization_from_R(x_R: float, y_R: float) -> tuple[float, float]:
    """Use the MTPC/MC transition R for T_N and psi_N, as in the slides."""
    id_R = I_MAX * x_R
    iq_R = I_MAX * y_R
    return float(torque(id_R, iq_R)), float(flux_abs(id_R, iq_R))


def x_left_boundary(y_abs: float | np.ndarray, y_M: float) -> float | np.ndarray:
    """
    Left/outer OPS boundary:
      MTPV for 0 <= |y| <= y_M,
      MC   for |y| > y_M.
    """
    y_abs = np.asarray(y_abs, dtype=float)
    return np.where(y_abs <= y_M, x_mtpv_norm(y_abs), x_mc_norm(y_abs))


def x_right_boundary(y_abs: float | np.ndarray) -> float | np.ndarray:
    """Right/inner OPS boundary: MTPC."""
    return x_mtpc_norm(y_abs)


# -----------------------------------------------------------------------------
# Structured current-domain mesh and boundary samples
# -----------------------------------------------------------------------------
def build_parametric_ops_mesh(y_M: float, y_R: float, n_y_per_segment: int, n_s: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a structured mesh of the feasible OPS current region.

    The signed q-axis is split at the characteristic ordinates
    -y_R, -y_M, 0, y_M, y_R. This guarantees that the mesh contains the
    MTPC/MC transition R and the MTPV/MC transition M exactly. Each segment
    uses n_y_per_segment points; duplicate segment endpoints are removed.

    For each |y|, x is sampled between the left boundary (MTPV/MC) and the
    right boundary (MTPC) using a normalized span coordinate s in [0, 1].
    """
    if n_y_per_segment < 2:
        raise ValueError("n_y_per_segment must be >= 2")

    segments = [
        np.linspace(-y_R, -y_M, n_y_per_segment),
        np.linspace(-y_M, 0.0, n_y_per_segment)[1:],
        np.linspace(0.0, y_M, n_y_per_segment)[1:],
        np.linspace(y_M, y_R, n_y_per_segment)[1:],
    ]
    y_vec = np.concatenate(segments)
    s_vec = np.linspace(0.0, 1.0, n_s)

    X = np.empty((len(y_vec), n_s), dtype=float)
    Y = np.empty((len(y_vec), n_s), dtype=float)

    for j, y in enumerate(y_vec):
        ya = abs(float(y))
        x_left = float(x_left_boundary(ya, y_M))
        x_right = float(x_right_boundary(ya))
        X[j, :] = (1.0 - s_vec) * x_left + s_vec * x_right
        Y[j, :] = y

    return X, Y


def sample_named_boundaries(y_M: float, y_R: float, n_boundary: int) -> dict[str, np.ndarray]:
    """Return dense MTPC, MC and MTPV boundary samples for diagnostics."""
    boundaries: dict[str, np.ndarray] = {}

    y_mtpc = np.linspace(-y_R, y_R, 2 * n_boundary + 1)
    boundaries["MTPC"] = np.column_stack((x_mtpc_norm(np.abs(y_mtpc)), y_mtpc))

    # MC exists between |y| = y_M and |y| = y_R for both signs.
    y_mc_pos = np.linspace(y_M, y_R, n_boundary)
    y_mc = np.concatenate((-y_mc_pos[::-1], y_mc_pos))
    boundaries["MC"] = np.column_stack((x_mc_norm(np.abs(y_mc)), y_mc))

    # MTPV exists between |y| = 0 and |y| = y_M for both signs.
    y_mtpv = np.linspace(-y_M, y_M, 2 * n_boundary + 1)
    boundaries["MTPV"] = np.column_stack((x_mtpv_norm(np.abs(y_mtpv)), y_mtpv))

    return boundaries


# -----------------------------------------------------------------------------
# Forward mapping and file output
# -----------------------------------------------------------------------------
def forward_map_mesh(X: np.ndarray, Y: np.ndarray, T_N: float, PSI_N: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map normalized current mesh X,Y to normalized T, psi and normalized currents."""
    id_amp = I_MAX * X
    iq_amp = I_MAX * Y
    T_norm = torque(id_amp, iq_amp) / T_N
    PSI_norm = flux_abs(id_amp, iq_amp) / PSI_N
    return T_norm, PSI_norm, X, Y


def write_surface_dat(path: Path, T_norm: np.ndarray, PSI_norm: np.ndarray, Z: np.ndarray) -> None:
    """Write PGFPlots surface table: T_norm psi_norm z."""
    rows: list[str] = []
    n_y, n_s = Z.shape
    for j in range(n_y):
        for i in range(n_s):
            rows.append(f"{T_norm[j, i]:.6f} {PSI_norm[j, i]:.6f} {Z[j, i]:.6f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def print_range(label: str, arr: np.ndarray) -> None:
    print(f"  {label}: min={np.nanmin(arr): .6f}, max={np.nanmax(arr): .6f}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate inverse OPS current-map .dat files for PGFPlots.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent, help="Directory for output .dat files.")
    parser.add_argument("--id-file", default="lut_inverse_id.dat", help="Output .dat file for i_sd.")
    parser.add_argument("--iq-file", default="lut_inverse_iq.dat", help="Output .dat file for i_sq.")
    parser.add_argument("--n-y-per-segment", type=int, default=17, help="Rows per signed-q segment; total mesh rows are 4*n_y_per_segment-3.")
    parser.add_argument("--n-s", type=int, default=51, help="Number of points between left and right OPS boundaries.")
    parser.add_argument("--n-boundary", type=int, default=800, help="Diagnostic samples per half-boundary branch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    x_R, y_R, x_M, y_M = find_transition_points()
    T_N, PSI_N = normalization_from_R(x_R, y_R)

    boundaries = sample_named_boundaries(y_M=y_M, y_R=y_R, n_boundary=args.n_boundary)
    X, Y = build_parametric_ops_mesh(y_M=y_M, y_R=y_R, n_y_per_segment=args.n_y_per_segment, n_s=args.n_s)
    T_norm, PSI_norm, X_norm, Y_norm = forward_map_mesh(X, Y, T_N=T_N, PSI_N=PSI_N)

    id_path = args.out_dir / args.id_file
    iq_path = args.out_dir / args.iq_file
    write_surface_dat(id_path, T_norm, PSI_norm, X_norm)
    write_surface_dat(iq_path, T_norm, PSI_norm, Y_norm)

    print("Generated inverse OPS current-map surface data")
    print(f"  R = MTPC ∩ MC: x_R={x_R:.6f}, y_R={y_R:.6f}")
    print(f"  M = MTPV ∩ MC: x_M={x_M:.6f}, y_M={y_M:.6f}")
    print(f"  T_N={T_N:.6f}, PSI_N={PSI_N:.6f}")
    print(f"  mesh rows n_y={X.shape[0]}, mesh columns n_s={X.shape[1]}")
    print("Current-domain surface ranges:")
    print_range("i_sd/i_s,max", X_norm)
    print_range("i_sq/i_s,max", Y_norm)
    print("Mapped plot-axis ranges:")
    print_range("T/T_N", T_norm)
    print_range("|psi|/psi_N", PSI_norm)
    print("Boundary ranges used for diagnostics:")
    for name, pts in boundaries.items():
        print(f"  {name}: x=[{pts[:,0].min(): .6f}, {pts[:,0].max(): .6f}], y=[{pts[:,1].min(): .6f}, {pts[:,1].max(): .6f}]")
    print(f"  wrote {id_path}")
    print(f"  wrote {iq_path}")


if __name__ == "__main__":
    main()
