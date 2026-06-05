"""First-7-segments response-function averaging.

Walks segments 0, 1, 2, ... chronologically for each configured (colony, size)
case and keeps the first N_TARGET_KEEP segments whose reliable ∇⊥c support
spans the full [-DC_RANGE, +DC_RANGE] window. The per-segment NN response
G(∇⊥c) is masked to that segment's reliable support, then averaged across
the kept segments (mean ± std).

Outputs (per case, into Results/):
  - {tag}_anten.{jpg,eps}      mean ± std response curve
  - {tag}_PR-curve.{jpg,eps}   predicted vs measured tangential acceleration
                                  (orange = PR outliers near origin)
  - {tag}_ver1.9.jpg           two-panel response (wide + zoom)
  - {tag}_prob.{jpg,eps}       spatial density distribution P(ρ > 1)
  - summary.csv                 per-case curve statistics

620L is the representative paper case (set via the COLONIES list below).

Requires All_Data_frames.csv at the project root (one level up from this
script).
"""

import os
import re
import csv
import sys

import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from v2_models  import (                          # noqa: F401  (torch.load needs these)
    PINN_PDE, Mechanism_NN, Mechanism_NN_LS,
)
from v2_helpers import (
    BODY_LENGTH, R_arena, R_wall, eps, CustomDataset,
    load_or_build_numdiff, assemble_segment_cache,
    load_or_build_boundary,
    slope_at_origin,
)


# ============================================================================
# Config
# ============================================================================
device = torch.device('cpu')
dtype  = torch.float32

SCRIPT_DIR  = _SCRIPT_DIR
PROJECT_DIR = _PROJECT_DIR
DATA_PATH   = os.environ.get(
    'V2_DATA_PATH',
    os.path.join(PROJECT_DIR, 'All_Data_frames.csv'))
CACHE_DIR   = os.path.join(PROJECT_DIR, 'cache')
MODEL_DIR   = os.path.join(PROJECT_DIR, 'Model', 'PDE')
OUT_DIR     = os.path.join(SCRIPT_DIR, 'Results')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)

# Cases to process. Edit this list to run other colonies / sizes.
# 620L is the representative paper case.
COLONIES = [
    (620, 'L'),
]

NUM_ANTS = 16
num_mul  = 30
dt       = 1 / 29.97 * num_mul
NUM_T    = int(os.environ.get('V2_NUM_T', 150))

SG_WINDOW    = 7
SG_POLYORDER = 3

# Reliable-∇⊥c-percentile range. Each segment's G(∇⊥c) curve is masked to
# its own [PCT_LO, PCT_HI] data percentiles before averaging, since the
# Mechanism_NN extrapolates unreliably beyond the data support.
DC_PCT_LO = 0.5
DC_PCT_HI = 99.5

# Walk segments chronologically, keep those whose reliable ∇⊥c support
# spans [-DC_RANGE, +DC_RANGE], and stop once N_TARGET_KEEP have been
# collected.
N_TARGET_KEEP = 7
DC_RANGE      = 1.5

# Jitter filter for x-intercept picking on the mean curve.
INTERCEPT_MIN_ABS = 0.1

# PR-curve outlier criterion (highlighted in orange on the response and PR
# plots): |a_pred| <= PR_OUTLIER_X AND |a_real - a_pred| > PR_OUTLIER_Y AND
# |∇⊥c| <= PR_OUTLIER_DC. These are the points sitting near the origin of
# the PR scatter that drift away from the y = x diagonal.
PR_OUTLIER_X  = 0.005
PR_OUTLIER_Y  = 0.005
PR_OUTLIER_DC = 0.005


print(f'model dir : {MODEL_DIR}')
print(f'output dir: {OUT_DIR}')
print(f'NUM_T={NUM_T}  N_TARGET_KEEP={N_TARGET_KEEP}  DC_RANGE=[-{DC_RANGE}, +{DC_RANGE}]')
print(f'cases ({len(COLONIES)}): {[f"{c}{s}" for c, s in COLONIES]}')


# ============================================================================
# Per-segment evaluation
# ============================================================================
def evaluate_segment(nd, seg_slice, t_seg, body_length, decision_boundary,
                     model_pde, model_anten):
    cache = assemble_segment_cache(
        nd, seg_slice, t_seg,
        body_length=body_length, num_ants=NUM_ANTS,
        device_=device, dtype_=dtype,
    )

    # Active mask (movement + wall-following filter).
    with torch.no_grad():
        pos_x = cache['pos_x']; pos_y = cache['pos_y']
        u_mag = cache['u_mag']
        vel_x = cache['vel_x']; vel_y = cache['vel_y']
        dist_c = torch.sqrt(pos_x ** 2 + pos_y ** 2)
        mask_u = u_mag >= decision_boundary
        r_hat_x = pos_x / (dist_c + eps); r_hat_y = pos_y / (dist_c + eps)
        v_radial = vel_x * r_hat_x + vel_y * r_hat_y
        SIN_TOL = np.sin(10 * np.pi / 180)
        wall_proximity = dist_c > (R_wall - 2 * body_length)
        wall_velocity  = v_radial.abs() < SIN_TOL * u_mag
        mask_active = mask_u & (~(wall_proximity & wall_velocity))

    # ∇⊥c at sensor rows via autograd.
    inp = cache['input_pde']
    pred = model_pde(inp)
    dc = autograd.grad(pred, inp, cache['ones_buf'],
                       create_graph=False, retain_graph=False)[0]
    n_body = cache['n_body']; n_sensor = cache['n_sensor']
    dc_dx_s = dc[n_body:n_body + n_sensor, [1]].reshape(-1, NUM_ANTS)
    dc_dy_s = dc[n_body:n_body + n_sensor, [2]].reshape(-1, NUM_ANTS)
    vel_x_u = cache['vel_x_u']; vel_y_u = cache['vel_y_u']
    dc_normal = -vel_y_u * dc_dx_s + vel_x_u * dc_dy_s

    dc_normal_sel = dc_normal[1:-1][mask_active[1:-1]].reshape(-1, 1)
    accel_tan = (cache['accel'][:, :NUM_ANTS] * (-vel_y_u)
                 + cache['accel'][:, NUM_ANTS:] * vel_x_u)
    accel_tan_sel = accel_tan[1:-1][mask_active[1:-1]]
    u_mag_sel     = cache['u_mag'][1:-1][mask_active[1:-1]]
    c_func        = model_anten(dc_normal_sel)

    dc_np = dc_normal_sel.flatten().cpu().detach().numpy()
    c_np  = c_func.flatten().cpu().detach().numpy()
    um_np = u_mag_sel.flatten().cpu().numpy()
    at_np = accel_tan_sel.flatten().cpu().numpy()

    acc_mean = float(np.mean(at_np)) if at_np.size else 0.0
    r2       = (1 - (np.sum((at_np - c_np * um_np) ** 2)
                     / max(np.sum((at_np - acc_mean) ** 2), 1e-12))
                if at_np.size else np.nan)
    corr_tan = (np.corrcoef(at_np, c_np * um_np)[0, 1]
                if at_np.size else np.nan)
    slope0   = slope_at_origin(dc_np, c_np, band=0.5)
    u_mag_mean = float(cache['u_mag'].mean().item())

    return dict(
        dc=dc_np, c=c_np, um=um_np, at=at_np,
        r2=r2, corr_tan=corr_tan, slope0=slope0,
        u_mag_mean=u_mag_mean,
    )


# ============================================================================
# Aggregation
# ============================================================================
all_results = {}

for num_colony, size_ant in COLONIES:
    body_length = BODY_LENGTH[size_ant]
    print(f'\n===== Colony {num_colony}  Size {size_ant}  '
          f'(body={body_length*1000:.1f} mm) =====')

    # Raw xy
    cache_file = os.path.join(CACHE_DIR, f'data_{num_colony}{size_ant}.npz')
    if os.path.exists(cache_file):
        xy = np.load(cache_file)['xy']
    else:
        data = pd.read_csv(DATA_PATH,
                           usecols=['xcor_mm', 'ycor_mm', 'Colony', 'SizeClass'])
        sel  = (data.Colony == num_colony) & (data.SizeClass == size_ant)
        xy   = data.loc[sel, ['xcor_mm', 'ycor_mm']].to_numpy()
        np.savez(cache_file, xy=xy); del data

    pos_x_raw = (xy[:, 0] / 1000).reshape(-1, NUM_ANTS).astype(np.float32)
    pos_y_raw = (xy[:, 1] / 1000).reshape(-1, NUM_ANTS).astype(np.float32)
    data_x_np = pos_x_raw[::num_mul, :]
    data_y_np = pos_y_raw[::num_mul, :]
    dt_raw    = 1 / 29.97

    nd_path = os.path.join(
        CACHE_DIR,
        f'numdiff_{num_colony}{size_ant}_w{SG_WINDOW}_p{SG_POLYORDER}_hr.npz')
    nd = load_or_build_numdiff(nd_path, pos_x_raw, pos_y_raw, dt_raw,
                               SG_WINDOW, SG_POLYORDER, num_mul)

    T_total = data_x_np.shape[0]
    data_t  = torch.linspace(0, dt * (T_total - 1), T_total, dtype=dtype)
    data_x  = torch.tensor(data_x_np); data_y = torch.tensor(data_y_np)
    train_loader = DataLoader(CustomDataset(data_x, data_y, data_t),
                              batch_size=NUM_T, shuffle=False, drop_last=True)

    u_mag_global = np.sqrt(nd['vel_x'] ** 2 + nd['vel_y'] ** 2).flatten()
    b_path = os.path.join(
        CACHE_DIR,
        f'boundary_{num_colony}{size_ant}_w{SG_WINDOW}_p{SG_POLYORDER}_hr.npz')
    decision_boundary = load_or_build_boundary(b_path, u_mag_global)

    # Evaluation grid shared across segments.
    plot_step  = 0.001
    dc_plot    = torch.arange(-5, 5 + plot_step, plot_step).reshape(-1, 1).to(device)
    dc_plot_np = dc_plot.flatten().cpu().numpy()

    index_kept, anten_kept, corr_kept, R2_kept = [], [], [], []
    u_mag_mean_list, a_pred_full, a_real_full, um_full = [], [], [], []
    dc_full, seg_id_full = [], []
    n_missing_model = 0; n_failed_gate = 0

    for index, (t_train, x_train, y_train) in enumerate(train_loader):
        if len(index_kept) >= N_TARGET_KEEP:
            break

        pde_path   = os.path.join(MODEL_DIR,
            f'{num_colony}{size_ant}_step-{NUM_T}s_model_pde_{index}')
        anten_path = os.path.join(MODEL_DIR,
            f'{num_colony}{size_ant}_step-{NUM_T}s_model_anten_{index}')
        if not (os.path.exists(pde_path) and os.path.exists(anten_path)):
            n_missing_model += 1
            print(f'  [step {index:3d}] no saved model in {MODEL_DIR}')
            continue

        model_pde   = torch.load(pde_path,   map_location=device,
                                 weights_only=False).to(device).to(dtype)
        model_anten = torch.load(anten_path, map_location=device,
                                 weights_only=False).to(device).to(dtype)
        model_pde.eval(); model_anten.eval()

        seg_start = index * NUM_T
        seg_slice = slice(seg_start, seg_start + NUM_T)
        t_seg = (t_train.reshape(-1) - t_train[0]).to(device).to(dtype)

        ev = evaluate_segment(
            nd, seg_slice, t_seg, body_length, decision_boundary,
            model_pde, model_anten)

        if ev['dc'].size == 0:
            n_failed_gate += 1
            print(f'  [skip] seg {index} — no active observations'); continue
        lower_dc = float(np.percentile(ev['dc'], DC_PCT_LO))
        upper_dc = float(np.percentile(ev['dc'], DC_PCT_HI))
        if lower_dc > -DC_RANGE or upper_dc < DC_RANGE:
            n_failed_gate += 1
            print(f'  [skip] seg {index} — reliable [{lower_dc:+.2f}, {upper_dc:+.2f}] '
                  f'does not cover ±{DC_RANGE}'); continue

        u_mag_mean_list.append(ev['u_mag_mean'])

        with torch.no_grad():
            anten_curve = model_anten(dc_plot).flatten().cpu().numpy()
        anten_curve = np.where(
            (dc_plot_np >= lower_dc) & (dc_plot_np <= upper_dc),
            anten_curve, np.nan)

        in_band = (ev['dc'] >= -DC_RANGE) & (ev['dc'] <= DC_RANGE)
        a_pred_seg = (ev['c'] * ev['um'])[in_band]
        a_real_seg = ev['at'][in_band]
        um_seg     = ev['um'][in_band]
        dc_seg     = ev['dc'][in_band]

        index_kept.append(index)
        anten_kept.append(anten_curve)
        corr_kept.append(float(ev['corr_tan']))
        R2_kept.append(float(ev['r2']))
        a_pred_full.append(a_pred_seg); a_real_full.append(a_real_seg)
        um_full.append(um_seg); dc_full.append(dc_seg)
        seg_id_full.append(np.full(dc_seg.shape, len(index_kept) - 1,
                                   dtype=np.int64))
        print(f'  [step {index:3d}] keep  COR={ev["corr_tan"]:.3f}  '
              f'R2={ev["r2"]:.3f}  slope0={ev["slope0"]:.2e}  '
              f'reliable=[{lower_dc:.2f}, {upper_dc:.2f}]')

    all_results[(num_colony, size_ant)] = dict(
        index_kept=index_kept,
        anten_kept=np.array(anten_kept) if anten_kept else None,
        corr_kept=corr_kept, R2_kept=R2_kept,
        u_mag_mean=u_mag_mean_list,
        a_pred=a_pred_full, a_real=a_real_full, um=um_full,
        dc_pr=dc_full, seg_id=seg_id_full,
        dc_plot=dc_plot_np,
        data_x=data_x, data_y=data_y,
    )
    last_kept = max(index_kept) if index_kept else -1
    if len(index_kept) < N_TARGET_KEEP:
        print(f'  ** WARN ** only {len(index_kept)}/{N_TARGET_KEEP} '
              f'reliable segments for {num_colony}{size_ant} '
              f'(missing model: {n_missing_model}, '
              f'failed gate: {n_failed_gate})')
    else:
        print(f'  -> {len(index_kept)} kept (last idx = {last_kept}, '
              f'missing model: {n_missing_model}, '
              f'failed gate: {n_failed_gate})')


# ============================================================================
# Plot aggregated response + PR scatter (outliers highlighted in orange)
# ============================================================================
plt.rc('font', size=20, family='Arial')
color_blue   = np.array([  0, 100, 250]) / 255
color_orange = np.array([255, 140,   0]) / 255

summary_stats = {}

for (num_colony, size_ant), res in all_results.items():
    if res['anten_kept'] is None or len(res['index_kept']) == 0:
        print(f'[{num_colony}{size_ant}] no segments loaded — skipping plot.')
        continue

    plot_x = res['dc_plot']
    anten_mean = np.nanmean(res['anten_kept'], axis=0)
    anten_std  = np.nanstd( res['anten_kept'], axis=0)

    base = os.path.join(OUT_DIR, f'{num_colony}{size_ant}')

    if any(p.size for p in res['a_pred']):
        a_pred_cat = np.concatenate(res['a_pred'])
        a_real_cat = np.concatenate(res['a_real'])
        dc_cat     = np.concatenate(res['dc_pr'])
        seg_id_cat = np.concatenate(res['seg_id'])
    else:
        a_pred_cat = np.array([]); a_real_cat = np.array([])
        dc_cat = np.array([]);    seg_id_cat = np.array([], dtype=np.int64)

    outlier_mask = ((np.abs(a_pred_cat) <= PR_OUTLIER_X)
                    & (np.abs(a_real_cat - a_pred_cat) > PR_OUTLIER_Y)
                    & (np.abs(dc_cat) <= PR_OUTLIER_DC))
    dc_outlier     = dc_cat[outlier_mask]
    seg_id_outlier = seg_id_cat[outlier_mask]
    n_outlier      = int(outlier_mask.sum())
    if n_outlier:
        nn_outlier = np.array([
            np.interp(dc_outlier[i], res['dc_plot'], res['anten_kept'][sid])
            for i, sid in enumerate(seg_id_outlier)])
    else:
        nn_outlier = np.array([])

    # (A) two-panel: full + zoom
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for k, xlim in enumerate([(-2, 2), (-DC_RANGE, DC_RANGE)]):
        axs[k].plot(plot_x, anten_mean, label='Mean',
                    color=color_blue, linewidth=2)
        axs[k].fill_between(plot_x,
                            anten_mean - anten_std, anten_mean + anten_std,
                            color='blue', alpha=0.1, label='Standard Deviation')
        if n_outlier:
            axs[k].scatter(dc_outlier, nn_outlier, color=color_orange,
                           s=20, zorder=5,
                           label=f'PR outliers (n={n_outlier})')
        axs[k].set_xlabel(r'$\nabla_\perp c$')
        axs[k].set_ylabel(r'$a_{n}$')
        axs[k].set_xlim(*xlim); axs[k].set_ylim(-2, 2)
        axs[k].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axs[1].axvline(x=-0.1, color='black', linestyle='--', linewidth=1)
    axs[1].axvline(x= 0.1, color='black', linestyle='--', linewidth=1)
    plt.subplots_adjust(wspace=0.5)
    plt.savefig(base + '_ver1.9.jpg', bbox_inches='tight', dpi=300)
    print(f'  saved: {base}_ver1.9.jpg')
    plt.close(fig)

    # (B) single-panel response curve
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(plot_x, anten_mean, label='Mean',
            color=color_blue, linewidth=2)
    ax.fill_between(plot_x,
                    anten_mean - anten_std, anten_mean + anten_std,
                    color='blue', alpha=0.1, label='Standard Deviation')
    if n_outlier:
        ax.scatter(dc_outlier, nn_outlier, color=color_orange,
                   s=20, zorder=5, label=f'PR outliers (n={n_outlier})')
    ax.set_xlabel(r'$\nabla_\perp c$'); ax.set_ylabel(r'$a_{n}$')
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-2, 2)
    plt.savefig(base + '_anten.jpg', bbox_inches='tight', dpi=300)
    plt.savefig(base + '_anten.eps', bbox_inches='tight', format='eps')
    print(f'  saved: {base}_anten.{{jpg,eps}}')
    plt.close(fig)

    # (C) PR scatter
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    guide_x = np.arange(-0.05, 0.05, 0.001)
    inlier = ~outlier_mask
    ax.scatter(a_pred_cat[inlier], a_real_cat[inlier], color=color_blue, s=10)
    if n_outlier:
        ax.scatter(a_pred_cat[outlier_mask], a_real_cat[outlier_mask],
                   color=color_orange, s=10,
                   label=f'|x|≤{PR_OUTLIER_X}, |y-x|>{PR_OUTLIER_Y}, '
                         f'|∇⊥c|≤{PR_OUTLIER_DC} (n={n_outlier})')
    ax.plot(guide_x, guide_x, color='k', linestyle='--', lw=1)
    ax.set_xlim(-0.05, 0.05); ax.set_ylim(-0.05, 0.05)
    plt.savefig(base + '_PR-curve.jpg', bbox_inches='tight', dpi=300)
    plt.savefig(base + '_PR-curve.eps', bbox_inches='tight', format='eps')
    print(f'  saved: {base}_PR-curve.{{jpg,eps}}')
    plt.close(fig)

    # ---- Curve statistics
    valid = ~np.isnan(anten_mean)
    plot_x_v = plot_x[valid]; mean_v = anten_mean[valid]
    zero_crossings = np.where(np.diff(np.signbit(mean_v)))[0]
    x_intercepts = [round((plot_x_v[zc] + plot_x_v[zc + 1]) / 2, 3)
                    for zc in zero_crossings]
    g_max = float(np.nanmax(anten_mean))
    g_min = float(np.nanmin(anten_mean))
    g_peak_abs_mean = (abs(g_max) + abs(g_min)) / 2

    filt = [x for x in x_intercepts if abs(x) >= INTERCEPT_MIN_ABS]
    neg_candidates = [x for x in filt if x < 0]
    pos_candidates = [x for x in filt if x > 0]
    neg_intercept = max(neg_candidates) if neg_candidates else float('nan')
    pos_intercept = min(pos_candidates) if pos_candidates else float('nan')
    intercept_abs_mean = ((abs(neg_intercept) + abs(pos_intercept)) / 2
                          if (np.isfinite(neg_intercept) and np.isfinite(pos_intercept))
                          else float('nan'))

    def _g_mean_in(lo, hi):
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            return float('nan')
        mask = (plot_x >= lo) & (plot_x <= hi)
        return float(np.nanmean(anten_mean[mask])) if mask.any() else float('nan')

    g_mean_neg = _g_mean_in(-DC_RANGE, neg_intercept)
    g_mean_pos = _g_mean_in(pos_intercept, DC_RANGE)
    g_region_abs_mean = ((abs(g_mean_neg) + abs(g_mean_pos)) / 2
                         if (np.isfinite(g_mean_neg) and np.isfinite(g_mean_pos))
                         else float('nan'))

    mean_speed = float(np.mean(res['u_mag_mean']))
    band = 2.0 * anten_std
    std_band_width      = float(np.nanmean(band))
    std_band_width_in_2 = float(np.nanmean(
        band[(plot_x >= -DC_RANGE) & (plot_x <= DC_RANGE)]))

    if res['a_pred'] and any(p.size for p in res['a_pred']):
        um_cat = np.concatenate(res['um'])
        m_um = um_cat > 1e-9
        if m_um.any():
            res_dthetadt = ((a_real_cat[m_um] - a_pred_cat[m_um])
                            / um_cat[m_um])
            sigma_dthetadt = float(np.std(res_dthetadt))
            sigma_at       = float(np.std(a_real_cat - a_pred_cat))
            n_pr_points    = int(m_um.sum())
        else:
            sigma_dthetadt = float('nan'); sigma_at = float('nan'); n_pr_points = 0
    else:
        sigma_dthetadt = float('nan'); sigma_at = float('nan'); n_pr_points = 0

    tag = f'{num_colony}{size_ant}'
    idx_str = ';'.join(str(i) for i in res['index_kept'])
    print(f'[{tag}]  kept ({len(res["index_kept"])}): {idx_str}')
    print(f'[{tag}]  G max/min: {g_max:+.3f} / {g_min:+.3f}  '
          f'|peak| mean: {g_peak_abs_mean:.3f}')
    print(f'[{tag}]  intercepts neg/pos: {neg_intercept:+.3f} / {pos_intercept:+.3f}  '
          f'|intercept| mean: {intercept_abs_mean:.3f}')
    print(f'[{tag}]  G region mean: {g_region_abs_mean:.3f}  '
          f'speed: {mean_speed:.4f} m/s  sigma_dθ/dt: {sigma_dthetadt:.4f}')

    summary_stats[(num_colony, size_ant)] = dict(
        colony=num_colony, size=size_ant,
        n_seg=len(res['index_kept']), index_kept=idx_str,
        G_max=g_max, G_min=g_min, G_peak_abs_mean=g_peak_abs_mean,
        neg_intercept=neg_intercept, pos_intercept=pos_intercept,
        intercept_abs_mean=intercept_abs_mean,
        G_mean_neg=g_mean_neg, G_mean_pos=g_mean_pos,
        G_region_abs_mean=g_region_abs_mean,
        mean_speed=mean_speed,
        std_band_width=std_band_width,
        std_band_width_in_2=std_band_width_in_2,
        sigma_dthetadt=sigma_dthetadt, sigma_at=sigma_at,
        n_pr_points=n_pr_points,
    )


# ============================================================================
# Spatial density P(ρ > 1)
# ============================================================================
density_grain = 0.03

for (num_colony, size_ant), res in all_results.items():
    if not res['index_kept']:
        continue

    data_x_s = res['data_x']; data_y_s = res['data_y']
    density_list = []
    for step in range(data_x_s.shape[0] // NUM_T):
        x_t = data_x_s[NUM_T * step : NUM_T * (step + 1)].numpy()
        y_t = data_y_s[NUM_T * step : NUM_T * (step + 1)].numpy()
        density = np.zeros_like(x_t)
        for t in range(x_t.shape[0]):
            for j in range(x_t.shape[1]):
                d = np.sqrt((x_t[t, j] - x_t[t, :]) ** 2
                            + (y_t[t, j] - y_t[t, :]) ** 2)
                density[t, j] = np.sum(d <= density_grain)
        density_list.append(density.flatten())

    density_array = np.array(density_list)
    selected      = density_array[res['index_kept'], :].flatten()
    bin_edges   = np.arange(-0.5, selected.max() + 1.5, 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    counts, _   = np.histogram(selected, bins=bin_edges, density=True)
    prob_gt_1   = float(np.sum(counts[bin_centers > 1]))

    plt.figure(figsize=(6, 4))
    plt.plot(bin_centers, counts, 'o-', linewidth=1, markersize=5,
             color='tab:green',
             label=f'Prob.$(\\rho > 1)$: {prob_gt_1:.4f}')
    plt.xlabel('Density'); plt.ylabel('Probability')
    plt.xticks(np.arange(0, 16, 1)); plt.yticks(np.arange(0, 1.2, 0.2))
    plt.xlim(0, 8); plt.ylim(0, 0.6)
    plt.legend(loc='upper right', fontsize=14)
    base_d = os.path.join(OUT_DIR, f'{num_colony}{size_ant}_prob')
    plt.savefig(base_d + '.jpg', bbox_inches='tight', dpi=300)
    plt.savefig(base_d + '.eps', bbox_inches='tight', format='eps')
    plt.close()
    print(f'[{num_colony}{size_ant}] P(ρ > 1) = {prob_gt_1:.4f}  '
          f'({base_d}.{{jpg,eps}})')
    if (num_colony, size_ant) in summary_stats:
        summary_stats[(num_colony, size_ant)]['prob_gt_1'] = prob_gt_1


# ============================================================================
# Per-case summary CSV
# ============================================================================
if summary_stats:
    SUMMARY_CSV = os.path.join(OUT_DIR, 'summary.csv')
    cols = ['colony', 'size', 'n_seg', 'index_kept',
            'G_max', 'G_min', 'G_peak_abs_mean',
            'neg_intercept', 'pos_intercept', 'intercept_abs_mean',
            'G_mean_neg', 'G_mean_pos', 'G_region_abs_mean',
            'prob_gt_1', 'mean_speed',
            'std_band_width', 'std_band_width_in_2',
            'sigma_dthetadt', 'sigma_at', 'n_pr_points']
    with open(SUMMARY_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for key in sorted(summary_stats.keys()):
            row = summary_stats[key]
            w.writerow({c: row.get(c, '') for c in cols})
    print(f'\nSummary CSV: {SUMMARY_CSV}  ({len(summary_stats)} rows)')

print('\nDone.')
