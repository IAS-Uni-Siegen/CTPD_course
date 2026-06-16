import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DTCConfig:
    # Direkt normierte Flussgrößen
    psi_outer: float = 1.0
    psi_band_width: float = 0.10

    # Normierter Betrag der aktiven Spannungsvektoren
    u_active: float = 1.0

    # Drehmoment-Hysterese
    T_ref: float = 0.45
    T_band_width: float = 0.12

    # Vereinfachte Rotorfluss-Winkelgeschwindigkeit
    omega_r: float = 0.88

    # Simulation
    initial_angle: float = 0.0
    warmup_rotations: float = 3.0
    collect_rotations: float = 1.0
    max_events: int = 50000

    eps: float = 1e-10
    min_event_dt: float = 1e-6

    # Nullvektorlogik
    use_zero_vector: bool = True
    zero_flux_hold_tol: float = 0.015

    # Priorisierung bei Alternativwahl
    torque_priority: float = 20.0


# DTC-Schalttabelle gemäß Anhang
# Schlüssel: (T_cmd, psi_cmd)
# T_cmd   = +1: Drehmoment erhöhen, -1: Drehmoment senken
# psi_cmd = +1: Fluss erhöhen,      -1: Fluss senken
SWITCH_TABLE = {
    1: {(+1, +1): 2, (+1, -1): 3, (-1, +1): 6, (-1, -1): 5},
    2: {(+1, +1): 3, (+1, -1): 4, (-1, +1): 1, (-1, -1): 6},
    3: {(+1, +1): 4, (+1, -1): 5, (-1, +1): 2, (-1, -1): 1},
    4: {(+1, +1): 5, (+1, -1): 6, (-1, +1): 3, (-1, -1): 2},
    5: {(+1, +1): 6, (+1, -1): 1, (-1, +1): 4, (-1, -1): 3},
    6: {(+1, +1): 1, (+1, -1): 2, (-1, +1): 5, (-1, -1): 4},
}


def get_script_directory() -> Path:
    """
    Gibt das Verzeichnis der aktuell ausgeführten Python-Datei zurück.
    In Notebooks / interaktiven Umgebungen wird das aktuelle Arbeitsverzeichnis verwendet.
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def wrap_to_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def voltage_vectors(cfg: DTCConfig):
    """
    V0 ist Nullvektor.
    V1...V6 sind aktive normierte Spannungsvektoren.
    """
    angles = np.arange(6) * np.pi / 3.0
    active = cfg.u_active * np.column_stack((np.cos(angles), np.sin(angles)))
    vectors = np.vstack((np.array([[0.0, 0.0]]), active))
    labels = ["V0"] + [f"V{k}" for k in range(1, 7)]
    return vectors, labels


def sector_from_angle(theta):
    """
    Sektorierung gemäß angehängter Grafik:
    Sektor 1: [-30°, +30°)
    Sektor 2: [+30°, +90°)
    ...
    """
    return int(
        np.floor(((theta + np.pi / 6.0) % (2.0 * np.pi)) / (np.pi / 3.0))
    ) % 6 + 1


def psi_abs(psi):
    return float(np.linalg.norm(psi))


def torque_proxy(psi, theta_r):
    """
    Vereinfachtes normiertes Drehmomentmodell:

        T = sin(theta_s - theta_r)

    theta_s: Statorflusswinkel
    theta_r: gedachter Rotorflusswinkel
    """
    theta_s = np.arctan2(psi[1], psi[0])
    return float(np.sin(theta_s - theta_r))


def actual_derivatives(psi, theta_r, u, cfg: DTCConfig):
    """
    Tatsächliche lokale Wirkung eines Spannungsvektors im idealisierten Modell.

    d psi / dt = u

    |psi|_dot =
        psi^T u / |psi|

    theta_s_dot =
        cross(psi, u) / |psi|^2

    T_dot =
        cos(theta_s - theta_r) * (theta_s_dot - omega_r)
    """
    rho = np.linalg.norm(psi)
    theta_s = np.arctan2(psi[1], psi[0])
    delta = theta_s - theta_r

    psi_dot = float(np.dot(psi, u) / rho)
    theta_s_dot = float((psi[0] * u[1] - psi[1] * u[0]) / rho**2)
    T_dot = float(np.cos(delta) * (theta_s_dot - cfg.omega_r))

    return psi_dot, theta_s_dot, T_dot


def continuous_theta(psi0, u, t, theta0_unwrapped, theta0_raw):
    psi_t = psi0 + u * t
    theta_raw = np.arctan2(psi_t[1], psi_t[0])
    return theta0_unwrapped + wrap_to_pi(theta_raw - theta0_raw)


def update_hysteresis_latches(psi, theta_r, psi_cmd, T_cmd, cfg: DTCConfig):
    """
    Echte Hysterese-Latches:
    Innerhalb des Bandes bleibt der vorherige Schaltbefehl erhalten.
    """
    psi_inner = cfg.psi_outer - cfg.psi_band_width
    T_low = cfg.T_ref - 0.5 * cfg.T_band_width
    T_high = cfg.T_ref + 0.5 * cfg.T_band_width

    rho = psi_abs(psi)
    T = torque_proxy(psi, theta_r)

    if rho <= psi_inner + cfg.eps:
        psi_cmd = +1
    elif rho >= cfg.psi_outer - cfg.eps:
        psi_cmd = -1

    if T <= T_low + cfg.eps:
        T_cmd = +1
    elif T >= T_high - cfg.eps:
        T_cmd = -1

    return psi_cmd, T_cmd


def choose_vector_dtc(psi, theta_r, psi_cmd, T_cmd, cfg: DTCConfig, vectors, labels):
    """
    Korrigierte DTC-Auswahl:

    1. Tabellenvektor gemäß Schalttabelle bestimmen.
    2. Tatsächliche lokale Wirkung prüfen.
    3. Falls der Tabellenvektor eine Hysteresegrenze weiter verletzt,
       wird ein sicherer Alternativvektor gewählt.
    """
    psi_inner = cfg.psi_outer - cfg.psi_band_width
    psi_ref = 0.5 * (psi_inner + cfg.psi_outer)
    T_low = cfg.T_ref - 0.5 * cfg.T_band_width
    T_high = cfg.T_ref + 0.5 * cfg.T_band_width

    rho = psi_abs(psi)
    T = torque_proxy(psi, theta_r)
    theta_s = np.arctan2(psi[1], psi[0])
    sector = sector_from_angle(theta_s)

    table_idx = SWITCH_TABLE[sector][(T_cmd, psi_cmd)]

    candidates = []

    for idx, u in enumerate(vectors):
        if idx == 0 and not cfg.use_zero_vector:
            continue

        psi_dot, theta_s_dot, T_dot = actual_derivatives(psi, theta_r, u, cfg)

        flux_cmd_ok = psi_cmd * psi_dot > 1e-9
        torque_cmd_ok = T_cmd * T_dot > 1e-9

        # Harte Sicherheit an Flussgrenzen
        flux_safe = True
        if rho <= psi_inner + 1e-6 and psi_dot < -1e-9:
            flux_safe = False
        if rho >= cfg.psi_outer - 1e-6 and psi_dot > 1e-9:
            flux_safe = False

        # Harte Sicherheit an Drehmomentgrenzen
        torque_safe = True
        if T <= T_low + 1e-6 and T_dot < -1e-9:
            torque_safe = False
        if T >= T_high - 1e-6 and T_dot > 1e-9:
            torque_safe = False

        # Nullvektor darf den Fluss nicht aktiv korrigieren.
        if idx == 0:
            flux_cmd_ok = abs(rho - psi_ref) <= cfg.zero_flux_hold_tol
            flux_safe = (rho > psi_inner + 1e-6) and (rho < cfg.psi_outer - 1e-6)

        score = 0.0
        score += 1000.0 if flux_safe else 0.0
        score += 1000.0 if torque_safe else 0.0
        score += cfg.torque_priority * abs(T_dot) if torque_cmd_ok else cfg.torque_priority * (T_cmd * T_dot)
        score += abs(psi_dot) if flux_cmd_ok else 0.1 * (psi_cmd * psi_dot)
        score += 0.25 if idx == table_idx else 0.0

        candidates.append(
            {
                "idx": idx,
                "label": labels[idx],
                "u": u,
                "sector": sector,
                "psi_dot": psi_dot,
                "theta_s_dot": theta_s_dot,
                "T_dot": T_dot,
                "flux_cmd_ok": flux_cmd_ok,
                "torque_cmd_ok": torque_cmd_ok,
                "flux_safe": flux_safe,
                "torque_safe": torque_safe,
                "source": "table" if idx == table_idx else "fallback",
                "score": score,
            }
        )

    candidate_classes = [
        lambda c: c["flux_safe"] and c["torque_safe"] and c["flux_cmd_ok"] and c["torque_cmd_ok"],
        lambda c: c["flux_safe"] and c["torque_safe"] and c["torque_cmd_ok"],
        lambda c: c["flux_safe"] and c["torque_safe"],
        lambda c: c["torque_safe"] and c["torque_cmd_ok"],
    ]

    for predicate in candidate_classes:
        group = [c for c in candidates if predicate(c)]
        if group:
            table_group = [c for c in group if c["idx"] == table_idx]
            if table_group:
                best = table_group[0]
                best["source"] = "table_feasible"
            else:
                best = max(group, key=lambda c: c["score"])
                best["source"] = (
                    "fallback_feasible"
                    if best["flux_cmd_ok"] and best["torque_cmd_ok"]
                    else "fallback_safe"
                )
            return best

    best = max(candidates, key=lambda c: c["score"])
    best["source"] = "saturation"
    return best


def time_to_circle(psi0, u, radius, cfg: DTCConfig):
    a = float(np.dot(u, u))
    if a < cfg.eps:
        return None

    b = 2.0 * float(np.dot(psi0, u))
    c = float(np.dot(psi0, psi0)) - radius**2

    disc = b**2 - 4.0 * a * c
    if disc < -cfg.eps:
        return None

    sqrt_disc = np.sqrt(max(disc, 0.0))

    roots = [
        (-b - sqrt_disc) / (2.0 * a),
        (-b + sqrt_disc) / (2.0 * a),
    ]

    roots = [float(t) for t in roots if t > cfg.min_event_dt]
    return min(roots) if roots else None


def find_root_bisection(f, t_max, cfg: DTCConfig, n_grid=160):
    if t_max is None or t_max <= cfg.min_event_dt:
        return None

    grid = np.linspace(cfg.min_event_dt, t_max, n_grid)

    t_prev = grid[0]
    f_prev = f(t_prev)

    for t in grid[1:]:
        f_now = f(t)

        if abs(f_now) < 1e-9:
            return float(t)

        if f_prev * f_now < 0.0:
            lo, hi = t_prev, t
            f_lo = f_prev

            for _ in range(70):
                mid = 0.5 * (lo + hi)
                f_mid = f(mid)

                if f_lo * f_mid <= 0.0:
                    hi = mid
                else:
                    lo = mid
                    f_lo = f_mid

            return float(0.5 * (lo + hi))

        t_prev = t
        f_prev = f_now

    return None


def time_to_torque_boundary(psi0, theta_r0, u, target_T, cfg: DTCConfig, t_max=None):
    if t_max is None:
        rate_bound = abs(cfg.omega_r) + cfg.u_active / max(psi_abs(psi0), 1e-6)
        t_max = 2.0 * np.pi / max(rate_bound, 1e-6)

    def f(t):
        return torque_proxy(psi0 + u * t, theta_r0 + cfg.omega_r * t) - target_T

    return find_root_bisection(f, t_max, cfg)


def time_to_sector_boundary(psi0, u, theta0_unwrapped, theta0_raw, cfg: DTCConfig, t_max):
    if float(np.dot(u, u)) < cfg.eps:
        return None

    theta_dot0 = (psi0[0] * u[1] - psi0[1] * u[0]) / float(np.dot(psi0, psi0))

    if abs(theta_dot0) < 1e-12:
        return None

    step = np.pi / 3.0
    base = -np.pi / 6.0

    if theta_dot0 > 0.0:
        n = np.floor((theta0_unwrapped - base) / step) + 1.0
        target = base + n * step
        if target <= theta0_unwrapped + abs(theta_dot0) * cfg.min_event_dt:
            target += step
    else:
        n = np.ceil((theta0_unwrapped - base) / step) - 1.0
        target = base + n * step
        if target >= theta0_unwrapped - abs(theta_dot0) * cfg.min_event_dt:
            target -= step

    def f(t):
        return continuous_theta(psi0, u, t, theta0_unwrapped, theta0_raw) - target

    return find_root_bisection(f, t_max, cfg, n_grid=80)


def next_event_time(psi0, theta_r0, theta0_unwrapped, theta0_raw, u, cfg: DTCConfig):
    psi_inner = cfg.psi_outer - cfg.psi_band_width
    T_low = cfg.T_ref - 0.5 * cfg.T_band_width
    T_high = cfg.T_ref + 0.5 * cfg.T_band_width

    candidates = []

    # Physikalische Hysteresegrenzen
    for radius, name in [
        (psi_inner, "psi_inner"),
        (cfg.psi_outer, "psi_outer"),
    ]:
        t = time_to_circle(psi0, u, radius, cfg)
        if t is not None:
            candidates.append((t, name))

    rate_bound = abs(cfg.omega_r) + cfg.u_active / max(psi_abs(psi0), 1e-6)
    t_scan = 2.0 * np.pi / max(rate_bound, 1e-6)

    for target_T, name in [
        (T_low, "T_low"),
        (T_high, "T_high"),
    ]:
        t = time_to_torque_boundary(psi0, theta_r0, u, target_T, cfg, t_max=t_scan)
        if t is not None:
            candidates.append((t, name))

    # Zusätzlich Sektorgrenzen, da die Schalttabelle sektorabhängig ist.
    horizon = min([c[0] for c in candidates], default=t_scan)

    t_sector = time_to_sector_boundary(
        psi0=psi0,
        u=u,
        theta0_unwrapped=theta0_unwrapped,
        theta0_raw=theta0_raw,
        cfg=cfg,
        t_max=horizon,
    )

    if t_sector is not None:
        candidates.append((t_sector, "sector_boundary"))

    if not candidates:
        return 0.01, "fallback"

    return min(candidates, key=lambda x: x[0])


def find_angle_crossing_time(psi0, u, theta0_unwrapped, theta0_raw, dt, target_angle, cfg: DTCConfig):
    def f(t):
        return continuous_theta(psi0, u, t, theta0_unwrapped, theta0_raw) - target_angle

    return find_root_bisection(f, dt, cfg, n_grid=30)


def state_at(psi0, theta_r0, u, t, theta0_unwrapped, theta0_raw, cfg: DTCConfig):
    psi_t = psi0 + u * t
    theta_r_t = theta_r0 + cfg.omega_r * t
    theta_s_t = continuous_theta(psi0, u, t, theta0_unwrapped, theta0_raw)

    return (
        psi_t,
        theta_r_t,
        theta_s_t,
        torque_proxy(psi_t, theta_r_t),
        psi_abs(psi_t),
    )


def make_row(
    kind,
    t_rel,
    t_abs,
    event_type,
    selected,
    psi,
    theta_r,
    theta_s_unwrapped,
    psi_cmd,
    T_cmd,
    result_limits,
):
    T_low, T_ref, T_high, psi_inner, psi_outer, omega_r, u_active = result_limits

    return {
        "kind": kind,
        "t_norm": float(t_rel),
        "t_abs": float(t_abs),
        "event_type": event_type,
        "voltage_vector": selected["label"],
        "sector": selected["sector"],
        "psi_alpha": float(psi[0]),
        "psi_beta": float(psi[1]),
        "psi_abs": float(psi_abs(psi)),
        "T": float(torque_proxy(psi, theta_r)),
        "theta_s_unwrapped": float(theta_s_unwrapped),
        "theta_r": float(theta_r),
        "psi_cmd": int(psi_cmd),
        "T_cmd": int(T_cmd),
        "psi_inner": float(psi_inner),
        "psi_outer": float(psi_outer),
        "T_low": float(T_low),
        "T_ref": float(T_ref),
        "T_high": float(T_high),
        "omega_r": float(omega_r),
        "u_active": float(u_active),
        "source": selected["source"],
        "psi_dot": float(selected["psi_dot"]),
        "T_dot": float(selected["T_dot"]),
        "flux_safe": int(selected["flux_safe"]),
        "torque_safe": int(selected["torque_safe"]),
        "is_T_low": int(event_type == "T_low"),
        "is_T_high": int(event_type == "T_high"),
        "is_psi_inner": int(event_type == "psi_inner"),
        "is_psi_outer": int(event_type == "psi_outer"),
        "is_sector_boundary": int(event_type == "sector_boundary"),
        "is_fallback": int(selected["source"].startswith("fallback")),
    }


def simulate_one_rotation(cfg: DTCConfig):
    psi_inner = cfg.psi_outer - cfg.psi_band_width
    T_low = cfg.T_ref - 0.5 * cfg.T_band_width
    T_high = cfg.T_ref + 0.5 * cfg.T_band_width

    if psi_inner <= 0.0:
        raise ValueError("psi_inner muss positiv sein.")

    if not (-1.0 < T_low < T_high < 1.0):
        raise ValueError("Die Drehmomentgrenzen müssen innerhalb (-1, 1) liegen.")

    vectors, labels = voltage_vectors(cfg)

    theta_s0 = cfg.initial_angle
    psi = psi_inner * np.array([np.cos(theta_s0), np.sin(theta_s0)])
    theta_r = theta_s0 - np.arcsin(np.clip(cfg.T_ref, -0.999, 0.999))

    theta_unwrapped = theta_s0
    theta_raw = theta_s0
    t_abs = 0.0

    psi_cmd = +1
    T_cmd = +1

    start_target = theta_s0 + cfg.warmup_rotations * 2.0 * np.pi
    end_target = start_target + cfg.collect_rotations * 2.0 * np.pi

    collecting = False
    rows = []

    limits = (
        T_low,
        cfg.T_ref,
        T_high,
        psi_inner,
        cfg.psi_outer,
        cfg.omega_r,
        cfg.u_active,
    )

    def crossed(a, b, target):
        return (a - target) * (b - target) <= 0.0 and abs(b - a) > 1e-12

    for _ in range(cfg.max_events):
        psi_cmd, T_cmd = update_hysteresis_latches(
            psi=psi,
            theta_r=theta_r,
            psi_cmd=psi_cmd,
            T_cmd=T_cmd,
            cfg=cfg,
        )

        selected = choose_vector_dtc(
            psi=psi,
            theta_r=theta_r,
            psi_cmd=psi_cmd,
            T_cmd=T_cmd,
            cfg=cfg,
            vectors=vectors,
            labels=labels,
        )

        u = selected["u"]

        dt, event_type = next_event_time(
            psi0=psi,
            theta_r0=theta_r,
            theta0_unwrapped=theta_unwrapped,
            theta0_raw=theta_raw,
            u=u,
            cfg=cfg,
        )

        theta_end_segment = continuous_theta(
            psi,
            u,
            dt,
            theta_unwrapped,
            theta_raw,
        )

        if (not collecting) and crossed(theta_unwrapped, theta_end_segment, start_target):
            t_start = find_angle_crossing_time(
                psi,
                u,
                theta_unwrapped,
                theta_raw,
                dt,
                start_target,
                cfg,
            )

            psi_s, theta_r_s, theta_s_s, T_s, rho_s = state_at(
                psi,
                theta_r,
                u,
                t_start,
                theta_unwrapped,
                theta_raw,
                cfg,
            )

            collecting = True
            t0 = t_abs + t_start

            rows.append(
                make_row(
                    kind="start",
                    t_rel=0.0,
                    t_abs=t0,
                    event_type="start_interp",
                    selected=selected,
                    psi=psi_s,
                    theta_r=theta_r_s,
                    theta_s_unwrapped=theta_s_s,
                    psi_cmd=psi_cmd,
                    T_cmd=T_cmd,
                    result_limits=limits,
                )
            )

        if collecting and crossed(theta_unwrapped, theta_end_segment, end_target):
            t_end = find_angle_crossing_time(
                psi,
                u,
                theta_unwrapped,
                theta_raw,
                dt,
                end_target,
                cfg,
            )

            psi_e, theta_r_e, theta_s_e, T_e, rho_e = state_at(
                psi,
                theta_r,
                u,
                t_end,
                theta_unwrapped,
                theta_raw,
                cfg,
            )

            t_abs_e = t_abs + t_end
            t_rel_e = t_abs_e - rows[0]["t_abs"]

            rows.append(
                make_row(
                    kind="end",
                    t_rel=t_rel_e,
                    t_abs=t_abs_e,
                    event_type="end_interp",
                    selected=selected,
                    psi=psi_e,
                    theta_r=theta_r_e,
                    theta_s_unwrapped=theta_s_e,
                    psi_cmd=psi_cmd,
                    T_cmd=T_cmd,
                    result_limits=limits,
                )
            )
            break

        psi_next = psi + u * dt
        theta_r_next = theta_r + cfg.omega_r * dt

        if event_type == "psi_inner":
            psi_next = psi_inner * psi_next / np.linalg.norm(psi_next)
        elif event_type == "psi_outer":
            psi_next = cfg.psi_outer * psi_next / np.linalg.norm(psi_next)

        theta_next_unwrapped = continuous_theta(
            psi,
            u,
            dt,
            theta_unwrapped,
            theta_raw,
        )

        theta_next_raw = np.arctan2(psi_next[1], psi_next[0])

        if collecting:
            t_abs_event = t_abs + dt
            t_rel_event = t_abs_event - rows[0]["t_abs"]

            rows.append(
                make_row(
                    kind="event",
                    t_rel=t_rel_event,
                    t_abs=t_abs_event,
                    event_type=event_type,
                    selected=selected,
                    psi=psi_next,
                    theta_r=theta_r_next,
                    theta_s_unwrapped=theta_next_unwrapped,
                    psi_cmd=psi_cmd,
                    T_cmd=T_cmd,
                    result_limits=limits,
                )
            )

        psi = psi_next
        theta_r = theta_r_next
        theta_unwrapped = theta_next_unwrapped
        theta_raw = theta_next_raw
        t_abs += dt

    else:
        raise RuntimeError("Die vollständige Rotation wurde innerhalb max_events nicht erreicht.")

    return rows


def save_trajectory_csv(rows, path: Path):
    """
    Speichert nur eine CSV-Datei.
    Diese Datei wird bei jedem Skriptlauf überschrieben.
    """
    fieldnames = [
        "kind",
        "t_norm",
        "t_abs",
        "event_type",
        "voltage_vector",
        "sector",
        "psi_alpha",
        "psi_beta",
        "psi_abs",
        "T",
        "theta_s_unwrapped",
        "theta_r",
        "psi_cmd",
        "T_cmd",
        "psi_inner",
        "psi_outer",
        "T_low",
        "T_ref",
        "T_high",
        "omega_r",
        "u_active",
        "source",
        "psi_dot",
        "T_dot",
        "flux_safe",
        "torque_safe",
        "is_T_low",
        "is_T_high",
        "is_psi_inner",
        "is_psi_outer",
        "is_sector_boundary",
        "is_fallback",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    cfg = DTCConfig(
        omega_r=0.88,
        u_active=1.0,
    )

    rows = simulate_one_rotation(cfg)

    out_dir = get_script_directory()
    csv_path = out_dir / "dtc_trajectory.csv"

    save_trajectory_csv(rows, csv_path)

    print(f"CSV gespeichert: {csv_path}")
    print(f"Anzahl Eckpunkte inkl. Start/Ende: {len(rows)}")
    print(f"Anzahl Schalt-/Sektorereignisse: {max(len(rows) - 2, 0)}")
    print(f"min/max |psi| = {min(r['psi_abs'] for r in rows):.9f}, {max(r['psi_abs'] for r in rows):.9f}")
    print(f"min/max T     = {min(r['T'] for r in rows):.9f}, {max(r['T'] for r in rows):.9f}")
    print(
        "Delta theta_s = "
        f"{rows[-1]['theta_s_unwrapped'] - rows[0]['theta_s_unwrapped']:.9f} rad"
    )


if __name__ == "__main__":
    main()