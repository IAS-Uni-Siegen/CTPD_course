#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import imageio

# ----- figure settings -----
HEIGHT_CM = 7.0
HEIGHT_IN = HEIGHT_CM / 2.54
WIDTH_IN = HEIGHT_IN * 16.0 / 9.0
DPI = 330
FPS = 24
N_FRAMES = 96

mpl.rcParams.update({
    'font.size': 7.0,
    'axes.titlesize': 7.8,
    'axes.labelsize': 6.6,
    'xtick.labelsize': 5.8,
    'ytick.labelsize': 5.8,
    'legend.fontsize': 6.0,
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'dejavusans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

STEP_COLORS = [
    (68/255, 119/255, 170/255),
    (0/255, 153/255, 136/255),
    (204/255, 187/255, 68/255),
    (204/255, 51/255, 17/255),
    (238/255, 102/255, 119/255),
    (170/255, 51/255, 119/255),
]
LD_COLOR = STEP_COLORS[0]
LQ_COLOR = STEP_COLORS[3]
GRID_COLOR = (0.88, 0.89, 0.91)

# ----- simulation settings -----
F_E = 50.0
N_STEPS = 6
ID_LEVELS = -np.arange(1, N_STEPS + 1, dtype=float)
IQ_LEVELS = +np.arange(1, N_STEPS + 1, dtype=float)
T_STEP = 1.0 / F_E
T_END = N_STEPS * T_STEP
SAMPLES_PER_STEP = 120
N_SAMPLES = N_STEPS * SAMPLES_PER_STEP
H_PER_A = 110.0
MU0 = 4.0 * math.pi * 1e-7

LD_LEAK_MH = 4.0
LQ_LEAK_MH = 5.0
LD_MAG0_MH = 31.0
LQ_MAG0_MH = 42.0
H_D_PER_A = 86.0
H_Q_PER_A = 78.0
H_PM_D = 0.0
H_DQ_CROSS = 33.0
H_QD_CROSS = 24.0
MIN_MU_RATIO = 0.12

OUT_DIR = Path(__file__).resolve().parent
MP4_PATH = OUT_DIR / 'PMSM_Saturation_BH_Example.mp4'
PREVIEW_PATH = OUT_DIR / 'PMSM_Saturation_BH_Example.png'

@dataclass(frozen=True)
class JAParameters:
    Ms: float
    a: float
    k: float
    c: float
    alpha: float

JA_PARAMS = JAParameters(Ms=1.28e6, a=135.0, k=255.0, c=0.22, alpha=1.05e-4)
JA_BURN_IN_CYCLES = 8


def dq_to_abc(id_cmd, iq_cmd, theta_e):
    i_alpha = id_cmd * np.cos(theta_e) - iq_cmd * np.sin(theta_e)
    i_beta = id_cmd * np.sin(theta_e) + iq_cmd * np.cos(theta_e)
    ia = i_alpha
    ib = -0.5 * i_alpha + 0.5 * np.sqrt(3.0) * i_beta
    ic = -0.5 * i_alpha - 0.5 * np.sqrt(3.0) * i_beta
    return ia, ib, ic


def langevin(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-4
    out[small] = x[small]/3.0 - x[small]**3/45.0 + 2.0*x[small]**5/945.0
    xb = x[~small]
    out[~small] = 1.0/np.tanh(xb) - 1.0/xb
    return out


def d_langevin_dx(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-4
    out[small] = 1.0/3.0 - x[small]**2/15.0 + 2.0*x[small]**4/189.0
    xb = x[~small]
    out[~small] = 1.0/xb**2 - 1.0/np.sinh(xb)**2
    return out


def anhysteretic_magnetization(h_eff, p=JA_PARAMS):
    return float(p.Ms * langevin(h_eff / p.a))


def dman_dhe(h_eff, p=JA_PARAMS):
    return float((p.Ms / p.a) * d_langevin_dx(h_eff / p.a))


def anhysteretic_magnetization_implicit(H, p=JA_PARAMS):
    H = np.asarray(H, dtype=float)
    M = np.zeros_like(H)
    for _ in range(30):
        H_eff = H + p.alpha * M
        M = 0.55 * M + 0.45 * (p.Ms * langevin(H_eff / p.a))
    return M


def ja_material_incremental_mu_ratio(H, p=JA_PARAMS):
    H = np.asarray(H, dtype=float)
    M_an = anhysteretic_magnetization_implicit(H, p)
    H_eff = H + p.alpha * M_an
    slope_eff = (p.Ms / p.a) * d_langevin_dx(H_eff / p.a)
    dM_dH = slope_eff / np.maximum(1.0 - p.alpha * slope_eff, 1e-3)
    slope0_eff = p.Ms / (3.0 * p.a)
    dM0_dH = slope0_eff / max(1.0 - p.alpha * slope0_eff, 1e-3)
    ratio = (1.0 + dM_dH) / (1.0 + dM0_dH)
    return np.clip(ratio, MIN_MU_RATIO, 1.15)


def simulate_jiles_atherton(H, p=JA_PARAMS, M0=0.0):
    H = np.asarray(H, dtype=float)
    M = np.zeros_like(H)
    M[0] = float(np.clip(M0, -1.02*p.Ms, 1.02*p.Ms))
    for i in range(1, len(H)):
        h_prev, m_prev = float(H[i-1]), float(M[i-1])
        dH = float(H[i] - H[i-1])
        if abs(dH) < 1e-12:
            M[i] = m_prev
            continue
        delta = 1.0 if dH > 0.0 else -1.0
        h_eff = h_prev + p.alpha * m_prev
        man = anhysteretic_magnetization(h_eff, p)
        dman = dman_dhe(h_eff, p)
        mirr = (m_prev - p.c * man) / max(1e-12, 1.0 - p.c)
        drive = man - mirr
        if delta * drive > 0.0:
            denom = p.k * delta - p.alpha * drive
            dmirr_dH = 0.0 if abs(denom) < 1e-9 else drive / denom
        else:
            dmirr_dH = 0.0
        denom_total = 1.0 - p.c * p.alpha * dman
        dM_dH = 0.0 if abs(denom_total) < 1e-9 else (p.c * dman + (1.0 - p.c) * dmirr_dH) / denom_total
        M[i] = np.clip(m_prev + dM_dH * dH, -1.02*p.Ms, 1.02*p.Ms)
    B = MU0 * (H + M)
    return M, B


def ja_steady_state_period(H_one_period, cycles=JA_BURN_IN_CYCLES):
    H_train = np.tile(H_one_period, cycles + 1)
    M_train, B_train = simulate_jiles_atherton(H_train, JA_PARAMS, M0=0.0)
    start = cycles * len(H_one_period)
    stop = start + len(H_one_period)
    return M_train[start:stop], B_train[start:stop], H_train[start:stop]


def inductance_maps(id_grid, iq_grid):
    id_grid = np.asarray(id_grid, dtype=float)
    iq_grid = np.asarray(iq_grid, dtype=float)
    # Literature-style inductance map: the incremental inductances attain their
    # maximum at the current origin and decrease with increasing current magnitude.
    # Therefore, no permanent-magnet bias term is used in the map-generating
    # effective field expressions; only current-induced self- and cross-saturation
    # contributions are retained.
    h_d_eff = np.sqrt((H_D_PER_A * id_grid) ** 2 + (H_DQ_CROSS * iq_grid) ** 2)
    h_q_eff = np.sqrt((H_Q_PER_A * iq_grid) ** 2 + (H_QD_CROSS * id_grid) ** 2)
    mu_d = ja_material_incremental_mu_ratio(h_d_eff)
    mu_q = ja_material_incremental_mu_ratio(h_q_eff)
    ld = LD_LEAK_MH + LD_MAG0_MH * mu_d
    lq = LQ_LEAK_MH + LQ_MAG0_MH * mu_q
    return np.clip(ld, LD_LEAK_MH, None), np.clip(lq, LQ_LEAK_MH, None)


def make_data():
    n = np.arange(SAMPLES_PER_STEP)
    phi = 2*np.pi*n/SAMPLES_PER_STEP
    t_list=[]; th_list=[]; id_list=[]; iq_list=[]; st_list=[]; h_list=[]; b_list=[]
    step_loops={}
    for step, (id_level, iq_level) in enumerate(zip(ID_LEVELS, IQ_LEVELS), start=1):
        local_t = (step-1)*T_STEP + n/SAMPLES_PER_STEP*T_STEP
        theta_e = 2*np.pi*(step-1) + phi
        id_cmd = np.full_like(phi, id_level, dtype=float)
        iq_cmd = np.full_like(phi, iq_level, dtype=float)
        ia, _, _ = dq_to_abc(id_cmd, iq_cmd, theta_e)
        h_input = H_PER_A * ia
        _, b_loop, h_loop = ja_steady_state_period(h_input)
        step_loops[step] = (h_loop.copy(), b_loop.copy())
        t_list.append(local_t); th_list.append(theta_e); id_list.append(id_cmd); iq_list.append(iq_cmd)
        st_list.append(np.full_like(phi, step, dtype=int)); h_list.append(h_loop); b_list.append(b_loop)
    t = np.concatenate(t_list)
    theta_e = np.concatenate(th_list)
    id_cmd = np.concatenate(id_list)
    iq_cmd = np.concatenate(iq_list)
    step = np.concatenate(st_list)
    h_inst = np.concatenate(h_list)
    b_inst = np.concatenate(b_list)
    ia, ib, ic = dq_to_abc(id_cmd, iq_cmd, theta_e)
    return dict(t=t, id=id_cmd, iq=iq_cmd, step=step, ia=ia, ib=ib, ic=ic, h_inst=h_inst, b_inst=b_inst, step_loops=step_loops)


def setup_axes(fig):
    fig.clear()
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.3])
    ax_abc = fig.add_subplot(gs[0,0])
    ax_bh = fig.add_subplot(gs[0,1])
    ax_dq = fig.add_subplot(gs[1,0])
    ax_map = fig.add_subplot(gs[1,1], projection='3d')
    ax_map.set_in_layout(False)
    ax_map.set_position([0.51, 0.02, 0.46, 0.46])
    xticks = np.arange(0.0, T_END + 1e-12, T_STEP)

    ax_abc.set(xlim=(0.0, T_END), ylim=(-9.5, 9.5), xticks=xticks, yticks=[-8,-4,0,4,8],
               xlabel=r'$t\,/\,\mathrm{s}$', ylabel=r'$i_\mathrm{a},\,i_\mathrm{b},\,i_\mathrm{c}\,/\,\mathrm{A}$')
    ax_abc.grid(True, color=GRID_COLOR, linewidth=0.7)

    ax_bh.set(xlim=(-980.0, 980.0), ylim=(-1.75,1.75), xticks=[-800,-400,0,400,800],
              yticks=[-1.5,-1.0,-0.5,0,0.5,1.0,1.5], xlabel=r'$H\,/\,\mathrm{A\,m^{-1}}$', ylabel=r'$B\,/\,\mathrm{T}$')
    ax_bh.grid(True, color=GRID_COLOR, linewidth=0.7)

    ax_dq.set(xlim=(0.0, T_END), ylim=(-6.7,6.7), xticks=xticks, yticks=[-6,-4,-2,0,2,4,6],
              xlabel=r'$t\,/\,\mathrm{s}$', ylabel=r'$i_\mathrm{d},\,i_\mathrm{q}\,/\,\mathrm{A}$')
    ax_dq.grid(True, color=GRID_COLOR, linewidth=0.7)

    ax_map.set_xlim(-6.2, 0.2); ax_map.set_ylim(0.0, 6.2); ax_map.set_zlim(4.0, 36.0)
    ax_map.set_xticks([-6,-4,-2,0]); ax_map.set_yticks([0,2,4,6]); ax_map.set_zticks([5,15,25,35])
    ax_map.set_xlabel(r'$i_\mathrm{d}\,/\,\mathrm{A}$', labelpad=-10)
    ax_map.set_ylabel(r'$i_\mathrm{q}\,/\,\mathrm{A}$', labelpad=-10)
    ax_map.set_zlabel(r'$L\,/\,\mathrm{mH}$', labelpad=-9)
    ax_map.view_init(elev=25, azim=125)
    ax_map.tick_params(axis='both', which='major', pad=-3)
    ax_map.zaxis.set_tick_params(pad=-3)
    ax_map.grid(True)
    for axis in (ax_map.xaxis, ax_map.yaxis, ax_map.zaxis):
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((0.82, 0.82, 0.82, 1.0))
        except Exception:
            pass

    for ax in (ax_abc, ax_dq):
        for s in range(1, N_STEPS):
            ax.axvline(s*T_STEP, color=(0.45,0.45,0.45,0.30), lw=0.7, ls=(0,(2,2)))
    return ax_abc, ax_bh, ax_dq, ax_map


def render_frame(fig, data, k, id_grid, iq_grid, ld_grid, lq_grid):
    ax_abc, ax_bh, ax_dq, ax_map = setup_axes(fig)
    t=data['t']; id_cmd=data['id']; iq_cmd=data['iq']; step_arr=data['step']
    ia=data['ia']; ib=data['ib']; ic=data['ic']; h_inst=data['h_inst']; b_inst=data['b_inst']; step_loops=data['step_loops']
    tk=float(t[k]); idk=float(id_cmd[k]); iqk=float(iq_cmd[k]); stepk=int(step_arr[k])
    current_color = STEP_COLORS[stepk-1]

    for s in range(1, N_STEPS+1):
        color = STEP_COLORS[s-1]
        full_mask = step_arr == s
        hist_mask = step_arr[:k+1] == s
        seg_start = (s-1) * SAMPLES_PER_STEP
        seg_end = s * SAMPLES_PER_STEP
        ext_end = min(seg_end + 1, N_SAMPLES)
        ax_abc.plot(t[seg_start:ext_end], ia[seg_start:ext_end], color=color, lw=0.8, alpha=0.35, ls='-')
        ax_abc.plot(t[seg_start:ext_end], ib[seg_start:ext_end], color=color, lw=0.75, alpha=0.24, ls='--')
        ax_abc.plot(t[seg_start:ext_end], ic[seg_start:ext_end], color=color, lw=0.75, alpha=0.18, ls=':')
        if np.any(hist_mask):
            h_end = seg_end + 1 if (s < N_STEPS and seg_end <= k) else k + 1
            ax_abc.plot(t[seg_start:h_end], ia[seg_start:h_end], color=color, lw=1.00, alpha=1.0, ls='-')
            ax_abc.plot(t[seg_start:h_end], ib[seg_start:h_end], color=color, lw=0.95, alpha=0.75, ls='--')
            ax_abc.plot(t[seg_start:h_end], ic[seg_start:h_end], color=color, lw=0.90, alpha=0.55, ls=':')
    ax_abc.axvline(tk, color='k', lw=0.7, alpha=0.5)
    ax_abc.plot([tk],[ia[k]],'o',ms=2.5,color=current_color)
    ax_abc.plot([tk],[ib[k]],'o',ms=2.5,color=current_color,alpha=0.75)
    ax_abc.plot([tk],[ic[k]],'o',ms=2.5,color=current_color,alpha=0.55)
    abc_handles = [
        Line2D([], [], color='k', lw=1.1, ls='-', label=r'$i_\mathrm{a}$'),
        Line2D([], [], color='k', lw=1.1, ls='--', label=r'$i_\mathrm{b}$'),
        Line2D([], [], color='k', lw=1.1, ls=':', label=r'$i_\mathrm{c}$'),
    ]
    ax_abc.legend(handles=abc_handles, loc='upper left', ncol=3, frameon=False,
                  handlelength=1.8, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.2)

    for s in range(1, N_STEPS+1):
        color=STEP_COLORS[s-1]
        h_loop,b_loop = step_loops[s]
        ax_bh.fill(h_loop,b_loop,color=color,alpha=0.06,zorder=1)
        ax_bh.plot(h_loop,b_loop,color=color,lw=0.75,alpha=0.38,zorder=2)
    for s in range(1, stepk+1):
        mask = step_arr[:k+1] == s
        if np.count_nonzero(mask) >= 2:
            color=STEP_COLORS[s-1]
            ax_bh.plot(h_inst[:k+1][mask], b_inst[:k+1][mask], color=color, lw=1.2 if s==stepk else 0.9,
                       alpha=1.0 if s==stepk else 0.72, zorder=3)
    ax_bh.plot([h_inst[k]], [b_inst[k]], 'o', ms=2.8, color='k', zorder=4)
    info_handles = [
        Line2D([],[],linestyle='none',label=rf'$H = {h_inst[k]:.1f}\,\mathrm{{A\,m^{{-1}}}}$'),
        Line2D([],[],linestyle='none',label=rf'$B = {b_inst[k]:.2f}\,\mathrm{{T}}$'),
        Line2D([],[],linestyle='none',label=rf'$i_\mathrm{{d}} = {idk:.1f}\,\mathrm{{A}}$'),
        Line2D([],[],linestyle='none',label=rf'$i_\mathrm{{q}} = {iqk:.1f}\,\mathrm{{A}}$'),
    ]
    leg=ax_bh.legend(handles=info_handles,loc='lower right',framealpha=0.95,facecolor='white',edgecolor='0.55',
                     handlelength=0,handletextpad=0.2,borderpad=0.35,labelspacing=0.25)
    for h in leg.legend_handles:
        h.set_visible(False)

    for s in range(1, N_STEPS+1):
        color=STEP_COLORS[s-1]
        full_mask = step_arr == s
        hist_mask = step_arr[:k+1] == s
        seg_start = (s-1) * SAMPLES_PER_STEP
        seg_end = s * SAMPLES_PER_STEP
        ext_end = min(seg_end + 1, N_SAMPLES)
        ax_dq.plot(t[seg_start:ext_end], iq_cmd[seg_start:ext_end], color=color, lw=0.8, alpha=0.30, ls='-')
        ax_dq.plot(t[seg_start:ext_end], id_cmd[seg_start:ext_end], color=color, lw=0.8, alpha=0.30, ls='--')
        if np.any(hist_mask):
            h_end = seg_end + 1 if (s < N_STEPS and seg_end <= k) else k + 1
            ax_dq.plot(t[seg_start:h_end], iq_cmd[seg_start:h_end], color=color, lw=1.05, alpha=1.0, ls='-')
            ax_dq.plot(t[seg_start:h_end], id_cmd[seg_start:h_end], color=color, lw=1.05, alpha=1.0, ls='--')
    ax_dq.axvline(tk, color='k', lw=0.7, alpha=0.5)
    ax_dq.plot([tk],[iqk],'o',ms=2.5,color=current_color)
    ax_dq.plot([tk],[idk],'o',ms=2.5,color=current_color)
    dq_handles = [
        Line2D([], [], color='k', lw=1.1, ls='--', label=r'$i_\mathrm{d}$'),
        Line2D([], [], color='k', lw=1.1, ls='-', label=r'$i_\mathrm{q}$'),
    ]
    ax_dq.legend(handles=dq_handles, loc='center right', ncol=1, frameon=False,
                 handlelength=1.8, handletextpad=0.4, borderaxespad=0.3)

    ax_map.plot_wireframe(id_grid, iq_grid, ld_grid, rstride=1, cstride=1, color=LD_COLOR, linewidth=0.35, alpha=0.50)
    ax_map.plot_wireframe(id_grid, iq_grid, lq_grid, rstride=1, cstride=1, color=LQ_COLOR, linewidth=0.35, alpha=0.48)
    ax_map.text(-5.6, 0.4, float(ld_grid[0,0])+1.0, r'$L_\mathrm{dd}$', color=LD_COLOR)
    ax_map.text(-5.0, 0.8, 40.0, r'$L_\mathrm{qq}$', color=LQ_COLOR)
    for s in range(1, stepk+1):
        mask = step_arr[:k+1] == s
        if np.any(mask):
            color=STEP_COLORS[s-1]
            idp=id_cmd[:k+1][mask]; iqp=iq_cmd[:k+1][mask]
            ldp,lqp = inductance_maps(idp,iqp)
            ax_map.plot(idp, iqp, ldp, color=color, lw=0.9, alpha=1.0 if s==stepk else 0.65)
            ax_map.plot(idp, iqp, lqp, color=color, lw=0.9, alpha=1.0 if s==stepk else 0.65)
    ldk,lqk = inductance_maps(np.array([idk]), np.array([iqk]))
    ax_map.plot([idk],[iqk],[float(ldk[0])], 'o', ms=2.8, color=LD_COLOR)
    ax_map.plot([idk],[iqk],[float(lqk[0])], 'o', ms=2.8, color=LQ_COLOR)

    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h,w,4).copy()
    return Image.fromarray(buf)


def main():
    data = make_data()
    id_axis=np.linspace(-6.2,0.2,13)
    iq_axis=np.linspace(0.0,6.2,13)
    id_grid,iq_grid=np.meshgrid(id_axis, iq_axis)
    ld_grid,lq_grid = inductance_maps(id_grid, iq_grid)
    frame_indices = np.linspace(0, N_SAMPLES-1, N_FRAMES).astype(int)
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI, constrained_layout=True)
    try:
        fig.set_constrained_layout_pads(w_pad=2/72, h_pad=2/72, wspace=0.04, hspace=0.05)
    except Exception:
        pass
    raw_frames = []
    for idx in frame_indices:
        raw_frames.append(render_frame(fig, data, int(idx), id_grid, iq_grid, ld_grid, lq_grid))
    mp4_frames = []
    for f in raw_frames:
        arr = np.array(f.convert('RGB'))
        h, w = arr.shape[:2]
        mp4_frames.append(arr[:h - h % 2, :w - w % 2])
    imageio.mimwrite(str(MP4_PATH), mp4_frames, fps=FPS, codec='libx264', pixelformat='yuv420p', quality=None, output_params=['-crf', '8', '-preset', 'slow'], macro_block_size=2)
    render_frame(fig, data, int(frame_indices[len(frame_indices)//2]), id_grid, iq_grid, ld_grid, lq_grid).save(PREVIEW_PATH)
    plt.close(fig)
    print(f'Saved {MP4_PATH}')
    print(f'Saved {PREVIEW_PATH}')

if __name__ == '__main__':
    main()
