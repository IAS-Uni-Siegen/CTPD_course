"""PMSM current-control simulation with voltage saturation and ARW variants.

This script simulates a continuous-time PMSM plant with a regular-sampling digital
current controller, one-step-ahead current prediction, full-inductance decoupling,
abc duty-cycle clipping, and four anti-reset-windup variants.

Exports one CSV file per ARW case, containing signals up to 10 ms:
    time, reference currents, actual currents, unclipped/clipped dq voltages,
    integrator states, duty-cycle extrema, and saturation residual norm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class PMSMParams:
    Rs: float = 10e-3
    Ldd: float = 0.5e-3
    Lqq: float = 1.0e-3
    Ldq: float = 0.1e-3
    Lqd: float = 0.1e-3
    psi_pm: float = 35e-3
    p: int = 10
    n_rpm: float = 3000.0
    Ts: float = 100e-6
    Vdc: float = 560.0

    @property
    def omega_e(self) -> float:
        return 2.0 * np.pi * self.n_rpm / 60.0 * self.p

    @property
    def L(self) -> np.ndarray:
        return np.array([[self.Ldd, self.Ldq], [self.Lqd, self.Lqq]], dtype=float)

    @property
    def Ldiag(self) -> np.ndarray:
        return np.diag([self.Ldd, self.Lqq])

    @property
    def Linv(self) -> np.ndarray:
        return np.linalg.inv(self.L)


@dataclass
class AxisControllerGains:
    a: float
    b: float
    Kp: float
    KiTs: float
    Kf: float


@dataclass
class ControllerDesign:
    d: float
    rho: float
    r: float
    theta: float
    zr: float
    S_lambda: float
    P_lambda: float
    gains_d: AxisControllerGains
    gains_q: AxisControllerGains


@dataclass
class SimulationResult:
    safe_name: str
    label: str
    t: np.ndarray
    i_dq: np.ndarray
    i_ref: np.ndarray
    u_unclipped_dq: np.ndarray
    u_clipped_dq: np.ndarray
    xI: np.ndarray
    duty_min: np.ndarray
    duty_max: np.ndarray
    delta_v_virtual: np.ndarray
    saturation_norm: np.ndarray


def dq_to_alphabeta(x_dq: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    d, q = x_dq
    return np.array([c * d - s * q, s * d + c * q])


def alphabeta_to_dq(x_ab: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    alpha, beta = x_ab
    return np.array([c * alpha + s * beta, -s * alpha + c * beta])


def alphabeta_to_abc(x_ab: np.ndarray) -> np.ndarray:
    alpha, beta = x_ab
    return np.array([
        alpha,
        -0.5 * alpha + np.sqrt(3.0) / 2.0 * beta,
        -0.5 * alpha - np.sqrt(3.0) / 2.0 * beta,
    ])


def abc_to_alphabeta(x_abc: np.ndarray) -> np.ndarray:
    a, b, c = x_abc
    return np.array([
        2.0 / 3.0 * (a - 0.5 * b - 0.5 * c),
        2.0 / 3.0 * (np.sqrt(3.0) / 2.0 * (b - c)),
    ])


def clip_voltage_abc_from_dq(
    u_dq: np.ndarray, theta: float, Vdc: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform dq voltage to abc, clip duties to [0,1], transform back to dq.

    Duty convention:
        duty_a = 0.5 + u_a / Vdc
    so each phase voltage is clipped to [-Vdc/2, +Vdc/2].
    """
    u_ab = dq_to_alphabeta(u_dq, theta)
    u_abc = alphabeta_to_abc(u_ab)

    duty = 0.5 + u_abc / Vdc
    duty_clip = np.clip(duty, 0.0, 1.0)

    u_abc_clip = (duty_clip - 0.5) * Vdc
    u_ab_clip = abc_to_alphabeta(u_abc_clip)
    u_dq_clip = alphabeta_to_dq(u_ab_clip, theta)

    return u_dq_clip, u_abc, u_abc_clip, duty_clip


def pmsm_rhs(i_dq: np.ndarray, u_dq: np.ndarray, theta: float, pars: PMSMParams) -> np.ndarray:
    """Continuous-time PMSM model in rotor-fixed dq coordinates."""
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    psi = pars.L @ i_dq + np.array([pars.psi_pm, 0.0])
    return pars.Linv @ (u_dq - pars.Rs * i_dq - pars.omega_e * (J @ psi))


def rk4_step(i_dq: np.ndarray, u_dq: np.ndarray, theta0: float, dt: float, pars: PMSMParams) -> np.ndarray:
    """Fourth-order Runge-Kutta step with angle progression in the sub-stages."""
    w = pars.omega_e
    k1 = pmsm_rhs(i_dq, u_dq, theta0, pars)
    k2 = pmsm_rhs(i_dq + 0.5 * dt * k1, u_dq, theta0 + 0.5 * w * dt, pars)
    k3 = pmsm_rhs(i_dq + 0.5 * dt * k2, u_dq, theta0 + 0.5 * w * dt, pars)
    k4 = pmsm_rhs(i_dq + dt * k3, u_dq, theta0 + w * dt, pars)
    return i_dq + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_interval(
    i0: np.ndarray, u_dq: np.ndarray, theta0: float, Ts: float, pars: PMSMParams, substeps: int = 25
) -> np.ndarray:
    """Integrate the continuous-time plant over one sampling interval with ZOH voltage."""
    dt = Ts / substeps
    i = i0.copy()
    theta = theta0
    for _ in range(substeps):
        i = rk4_step(i, u_dq, theta, dt, pars)
        theta += pars.omega_e * dt
    return i


def nominal_predict_one_step(i_k: np.ndarray, u_applied_dq: np.ndarray, theta_k: float, pars: PMSMParams) -> np.ndarray:
    """One-step-ahead prediction using the currently applied clipped PWM voltage."""
    return integrate_interval(i_k, u_applied_dq, theta_k, pars.Ts, pars, substeps=10)


def design_axis(L_axis: float, Rs: float, Ts: float, S_lambda: float, P_lambda: float, zr: float) -> AxisControllerGains:
    a = np.exp(-Rs * Ts / L_axis)
    b = (1.0 - a) / Rs
    Kp = (1.0 + a - S_lambda) / b
    KiTs = (1.0 - S_lambda + P_lambda) / b
    Kf = KiTs / (1.0 - zr) - Kp
    return AxisControllerGains(a=a, b=b, Kp=Kp, KiTs=KiTs, Kf=Kf)


def design_controller(pars: PMSMParams, d: float = 0.9, rho: float = 0.5) -> ControllerDesign:
    kappa = np.sqrt(1.0 - d**2) / d if d < 1.0 else 0.0
    r = np.exp(-rho)
    theta = kappa * rho
    S_lambda = 2.0 * r * np.cos(theta)
    P_lambda = r**2
    zr = r * np.cos(theta) if d < 1.0 else r

    gains_d = design_axis(pars.Ldd, pars.Rs, pars.Ts, S_lambda, P_lambda, zr)
    gains_q = design_axis(pars.Lqq, pars.Rs, pars.Ts, S_lambda, P_lambda, zr)
    return ControllerDesign(d, rho, r, theta, zr, S_lambda, P_lambda, gains_d, gains_q)


def decoupling_voltage(v_virtual_dq: np.ndarray, i_dq: np.ndarray, pars: PMSMParams) -> np.ndarray:
    """Nominal full-inductance decoupling from virtual SISO voltages to physical dq voltage.

    Target virtual dynamics:
        L_diag * di/dt = v_virtual - Rs*i
    """
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    psi = pars.L @ i_dq + np.array([pars.psi_pm, 0.0])
    di_target = np.linalg.inv(pars.Ldiag) @ (v_virtual_dq - pars.Rs * i_dq)
    return pars.Rs * i_dq + pars.L @ di_target + pars.omega_e * (J @ psi)


def physical_residual_to_virtual(delta_u_phys_dq: np.ndarray, pars: PMSMParams) -> np.ndarray:
    """Map physical dq saturation residual back to virtual decoupled controller axes."""
    return pars.Ldiag @ pars.Linv @ delta_u_phys_dq


def reference_profile(t: float) -> np.ndarray:
    """Reference step: selected to induce severe transient voltage saturation."""
    if t < 2e-3:
        return np.array([0.0, 0.0])
    return np.array([-100.0, 100.0])


def simulate_case(pars: PMSMParams, design: ControllerDesign, arw_mode: str, T_end: float = 0.010) -> SimulationResult:
    Ts = pars.Ts
    N = int(np.round(T_end / Ts))
    t = np.arange(N + 1) * Ts

    i_hist = np.zeros((N + 1, 2))
    r_hist = np.zeros((N + 1, 2))
    u_unclip_hist = np.zeros((N + 1, 2))
    u_clip_hist = np.zeros((N + 1, 2))
    xI_hist = np.zeros((N + 1, 2))
    duty_min_hist = np.zeros(N + 1)
    duty_max_hist = np.zeros(N + 1)
    delta_v_hist = np.zeros((N + 1, 2))
    sat_norm_hist = np.zeros(N + 1)

    g_d, g_q = design.gains_d, design.gains_q
    Kp = np.array([g_d.Kp, g_q.Kp])
    KiTs = np.array([g_d.KiTs, g_q.KiTs])
    Kf = np.array([g_d.Kf, g_q.Kf])

    if arw_mode == "none":
        k_aw = np.array([0.0, 0.0])
        label = "no ARW"
        safe_name = "no_arw"
    elif arw_mode == "Kr1":
        # Thesis-style K_R=1: k_aw = KiTs/Kp per axis.
        k_aw = KiTs / Kp
        label = "ARW K_R=1"
        safe_name = "arw_KR1"
    elif arw_mode == "lambda_zr":
        k_aw = np.array([1.0 - design.zr, 1.0 - design.zr])
        label = "ARW lambda_aw=z_r"
        safe_name = "arw_lambda_zr"
    elif arw_mode == "lambda_r":
        k_aw = np.array([1.0 - design.r, 1.0 - design.r])
        label = "ARW lambda_aw=exp(-rho)"
        safe_name = "arw_lambda_exp_minus_rho"
    else:
        raise ValueError(arw_mode)

    i = np.array([0.0, 0.0])
    xI = np.array([0.0, 0.0])
    u_applied_clip = np.array([0.0, 0.0])
    u_next_clip = np.array([0.0, 0.0])
    u_next_unclip = np.array([0.0, 0.0])

    for k in range(N):
        tk = t[k]
        theta_k = pars.omega_e * tk

        i_hist[k] = i
        r_hist[k] = reference_profile(tk)
        xI_hist[k] = xI
        u_clip_hist[k] = u_applied_clip
        u_unclip_hist[k] = u_next_unclip

        # One-step-ahead prediction compensating the digital control delay.
        i_pred = nominal_predict_one_step(i, u_applied_clip, theta_k, pars)

        # Controller computes voltage for the next PWM interval.
        t_next = t[k + 1]
        theta_next = pars.omega_e * t_next
        r_next = reference_profile(t_next)
        e = r_next - i_pred

        v_c = Kp * e + xI
        v_f = Kf * r_next
        v_virtual = v_c + v_f

        u_unclip = decoupling_voltage(v_virtual, i_pred, pars)
        u_clip, _, _, duty_clip = clip_voltage_abc_from_dq(u_unclip, theta_next, pars.Vdc)

        delta_u_phys = u_unclip - u_clip
        delta_v_virtual = physical_residual_to_virtual(delta_u_phys, pars)

        # Feedback ARW update. If k_aw = 0, this is the plain PI integrator.
        xI = xI + KiTs * e - k_aw * delta_v_virtual

        u_next_unclip = u_unclip
        u_next_clip = u_clip
        delta_v_hist[k] = delta_v_virtual
        duty_min_hist[k] = np.min(duty_clip)
        duty_max_hist[k] = np.max(duty_clip)
        sat_norm_hist[k] = np.linalg.norm(delta_u_phys)

        # Plant evolves during the current interval with the previous PWM voltage.
        i = integrate_interval(i, u_applied_clip, theta_k, Ts, pars, substeps=25)

        # Regular-sampling update at the next sampling instant.
        u_applied_clip = u_next_clip

    i_hist[N] = i
    r_hist[N] = reference_profile(t[N])
    xI_hist[N] = xI
    u_clip_hist[N] = u_applied_clip
    u_unclip_hist[N] = u_next_unclip

    return SimulationResult(
        safe_name=safe_name,
        label=label,
        t=t,
        i_dq=i_hist,
        i_ref=r_hist,
        u_unclipped_dq=u_unclip_hist,
        u_clipped_dq=u_clip_hist,
        xI=xI_hist,
        duty_min=duty_min_hist,
        duty_max=duty_max_hist,
        delta_v_virtual=delta_v_hist,
        saturation_norm=sat_norm_hist,
    )


def result_to_dataframe(res: SimulationResult) -> pd.DataFrame:
    return pd.DataFrame({
        "time_s": res.t,
        "time_ms": res.t * 1e3,
        "id_ref_A": res.i_ref[:, 0],
        "iq_ref_A": res.i_ref[:, 1],
        "id_A": res.i_dq[:, 0],
        "iq_A": res.i_dq[:, 1],
        "ud_unclipped_V": res.u_unclipped_dq[:, 0],
        "uq_unclipped_V": res.u_unclipped_dq[:, 1],
        "ud_clipped_V": res.u_clipped_dq[:, 0],
        "uq_clipped_V": res.u_clipped_dq[:, 1],
        "xI_d_V": res.xI[:, 0],
        "xI_q_V": res.xI[:, 1],
        "duty_min": res.duty_min,
        "duty_max": res.duty_max,
        "saturation_norm_V": res.saturation_norm,
    })


def main() -> None:
    pars = PMSMParams()
    design = design_controller(pars, d=0.9, rho=0.5)

    print(f"n = {pars.n_rpm:.0f} 1/min")
    print(f"u_dc = {pars.Vdc:.1f} V")
    print(f"omega_e = {pars.omega_e:.2f} rad/s")
    print(f"d = {design.d:.3f}, rho = {design.rho:.3f}, r = {design.r:.5f}, z_r = {design.zr:.5f}")
    print("d-axis gains:", design.gains_d)
    print("q-axis gains:", design.gains_q)

    out_dir = Path(__file__).resolve().parent 
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = ["none", "Kr1", "lambda_zr", "lambda_r"]
    for case in cases:
        res = simulate_case(pars, design, case, T_end=0.010)
        df = result_to_dataframe(res)
        csv_path = out_dir / f"{res.safe_name}.csv"
        df.to_csv(csv_path, index=False, float_format="%.10g")
        print(f"{res.label:28s} -> {csv_path.name:32s} max |u* - u| = {np.max(res.saturation_norm):.2f} V")


if __name__ == "__main__":
    main()
