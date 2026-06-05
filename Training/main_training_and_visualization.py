"""Main training + visualization for the ant PINN-PDE / Mechanism_NN model.

Pipeline (per segment in TARGETED_SEGMENTS):
  1. Build segment cache from raw trajectory data.
  2. If both saved model files already exist at MODEL_DIR, skip training.
     Otherwise train up to four attempts in the order
       w_theta = 2.0, 2.0, 5.0, 5.0
     The loop stops as soon as any attempt meets the final-save criteria.
     If no attempt meets them, the best run so far (lowest L_ang) is
     saved as the fallback.
     Save criteria (loss_ok):
       total_iter > 200  AND  L_ang < SAVE_THETA_TOL
                         AND  corr_tan > 0.8
                         AND  L_c   < SAVE_CONCENT_TOL
  3. Load the saved model (whether just trained or pre-existing) and run the
     5-artifact visualization into Training/Results/visualization_*/.
     Artifacts: response-curve + PR-scatter (paired plot), trajectory MP4,
     concentration-field MP4, mesh + trajectories static plot, and
     active/inactive coloured trajectory MP4.

Requires: All_Data_frames.csv in the project root (one level up from this
script). Download from https://zenodo.org/records/16797507 .
"""

import os
import time
import datetime

import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
from torch.utils.data import DataLoader
from scipy.optimize import minimize

import matplotlib
matplotlib.use('Agg')


def _resolve_ffmpeg_path():
    """Locate ffmpeg without hard-coding a Windows path.

    Precedence:
      1. V2_FFMPEG_PATH environment variable (if it points to a real file).
      2. imageio_ffmpeg.get_ffmpeg_exe() — bundled with the matplotlib
         install on most setups.
      3. shutil.which('ffmpeg') — ffmpeg on the system PATH.
      4. 'ffmpeg' (matplotlib will then fail and the GIF fallback in
         save_anim() handles it).
    """
    import shutil
    env = os.environ.get('V2_FFMPEG_PATH')
    if env and os.path.isfile(env):
        return env
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.isfile(p):
            return p
    except Exception:
        pass
    p = shutil.which('ffmpeg')
    return p if p else 'ffmpeg'

matplotlib.rcParams['animation.ffmpeg_path'] = _resolve_ffmpeg_path()
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

# Add parent dir to path so we can import v2_helpers / v2_models from
# Final_code/ regardless of where this script is launched from.
import sys
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from v2_models import (                          # noqa: F401  (torch.load needs these)
    PINN_PDE, PINN_PDE_old,
    Mechanism_NN, Mechanism_NN_LS,
)
import v2_helpers
from v2_helpers import (
    BODY_LENGTH, R_arena, R_wall, eps, CustomDataset,
    load_or_build_numdiff, assemble_segment_cache,
    compute_mask_active, attach_loss_constants,
    loss_fn, make_scipy_lbfgs_obj, load_or_build_boundary,
    W_CONCENT, W_IC,
)


# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR  = _SCRIPT_DIR
PROJECT_DIR = _PROJECT_DIR
DATA_PATH   = os.environ.get(
    'V2_DATA_PATH',
    os.path.join(PROJECT_DIR, 'All_Data_frames.csv'))
CACHE_DIR   = os.path.join(PROJECT_DIR, 'cache')
MODEL_DIR   = os.path.join(PROJECT_DIR, 'Model', 'PDE')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'Results')

for d in (CACHE_DIR, MODEL_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)


# ============================================================================
# Targets — (colony, size) -> list of segment indices to process.
# 620L is the representative case used in the paper.
# ============================================================================
TARGETED_SEGMENTS = {
    (620, 'L'): [0, 1, 2, 3, 4, 5, 6, 7],
}


# ============================================================================
# Hyperparameters
# ============================================================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype  = torch.float32

NUM_ANTS      = 16
num_mul       = 30
dt            = 1 / 29.97 * num_mul
num_t         = int(os.environ.get('V2_NUM_T', 150))

SG_WINDOW     = 7
SG_POLYORDER  = 3

epoch_warmup  = 500
epoch_adam    = 250
lbfgs_iter    = 10000
pde_hidden    = 64
pde_depth     = 3
fourier_dim   = 64
fourier_sigma = 0.75

# Save criteria.
SAVE_THETA_TOL     = 8e-6
SAVE_CONCENT_TOL   = 4e-6
SAVE_CORR_TAN_MIN  = 0.8
LBFGS_STOP_THETA   = 5e-6
LBFGS_STOP_CONCENT = 4e-6

MECH_INIT_SCALE = 10.0

# W_THETA attempt schedule. Two attempts at 2.0; if save criteria are
# still not met, two more attempts at 5.0. The loop stops as soon as
# any attempt meets the save criteria.
W_THETA_ATTEMPTS = [2.0, 2.0, 5.0, 5.0]


# ============================================================================
# Training helpers
# ============================================================================
def is_loss_ok(total_iter, l_theta, l_c, corr_tan):
    """Save-FINAL criterion: same as the published manuscript, with the
    slope0 and R^2 conditions removed."""
    return (total_iter > 200
            and l_theta < SAVE_THETA_TOL
            and corr_tan > SAVE_CORR_TAN_MIN
            and l_c       < SAVE_CONCENT_TOL)


def build_segment_cache(num_colony, size_ant, body_length):
    """Returns (nd, decision_boundary, train_loader)."""
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
    dt_raw    = 1 / 29.97

    nd_path = os.path.join(
        CACHE_DIR,
        f'numdiff_{num_colony}{size_ant}_w{SG_WINDOW}_p{SG_POLYORDER}_hr.npz')
    nd = load_or_build_numdiff(nd_path, pos_x_raw, pos_y_raw, dt_raw,
                               SG_WINDOW, SG_POLYORDER, num_mul)

    data_x_np = pos_x_raw[::num_mul, :]
    data_y_np = pos_y_raw[::num_mul, :]
    T_total = data_x_np.shape[0]
    data_t  = torch.linspace(0, dt * (T_total - 1), T_total, dtype=dtype)
    data_x  = torch.tensor(data_x_np); data_y = torch.tensor(data_y_np)

    train_loader = DataLoader(CustomDataset(data_x, data_y, data_t),
                              batch_size=num_t, shuffle=False, drop_last=True)

    u_mag_global = np.sqrt(nd['vel_x'] ** 2 + nd['vel_y'] ** 2).flatten()
    b_path = os.path.join(
        CACHE_DIR,
        f'boundary_{num_colony}{size_ant}_w{SG_WINDOW}_p{SG_POLYORDER}_hr.npz')
    decision_boundary = load_or_build_boundary(b_path, u_mag_global)
    return nd, decision_boundary, train_loader


def train_one_attempt(cache, mask_active, body_length, decision_boundary,
                      w_theta):
    """Single train attempt with the given w_theta. Returns a dict carrying
    the trained models and diagnostic metrics."""
    v2_helpers.W_THETA = float(w_theta)   # LBFGS loss reads this

    model_pde = PINN_PDE(input_num=3, output_num=1, bias=True,
                         body_length=body_length,
                         hidden=pde_hidden, depth=pde_depth,
                         fourier_dim=fourier_dim,
                         fourier_sigma=fourier_sigma
                         ).to(device).to(dtype)
    model_anten = Mechanism_NN_LS(
        input_num=1, output_num=1, bias=True,
        init_scale=MECH_INIT_SCALE, hidden2=16).to(device).to(dtype)
    params = list(model_pde.parameters()) + list(model_anten.parameters())

    attach_loss_constants(cache, mask_active, model_pde, NUM_ANTS)

    # Adam warmup + full Adam
    optimizer_adam = torch.optim.Adam(params, lr=0.001)
    loss_warm = []; loss_adam = []
    for i in range(epoch_warmup + epoch_adam):
        is_warmup = i < epoch_warmup
        model_pde.train(); model_anten.train()
        optimizer_adam.zero_grad()
        l_theta, l_c, _, _, l_ic, _, _, _, _, _, _ = loss_fn(
            model_pde, model_anten, cache, NUM_ANTS)
        if is_warmup:
            lt = W_CONCENT * l_c + W_IC * l_ic
        else:
            lt = w_theta * l_theta + W_CONCENT * l_c + W_IC * l_ic
        if not torch.isfinite(lt):
            break
        lt.backward()
        optimizer_adam.step()
        (loss_warm if is_warmup else loss_adam).append(float(lt))

    # L-BFGS with early-stop when both L_ang and L_c are tight enough.
    concent_lbfgs = []
    def _loss_fn_track_concent(model_pde_, model_anten_, cache_, na_):
        out = loss_fn(model_pde_, model_anten_, cache_, na_)
        concent_lbfgs.append(float(out[1].detach().cpu()))
        return out

    obj, x0, unpack, loss_lbfgs, theta_lbfgs = make_scipy_lbfgs_obj(
        model_pde, model_anten, cache, NUM_ANTS, _loss_fn_track_concent)

    def _early_stop(*_a, **_k):
        if (theta_lbfgs and theta_lbfgs[-1] < LBFGS_STOP_THETA
                and concent_lbfgs and concent_lbfgs[-1] < LBFGS_STOP_CONCENT):
            raise StopIteration

    result = minimize(obj, x0, method='L-BFGS-B', jac=True,
                      callback=_early_stop,
                      options=dict(maxiter=lbfgs_iter,
                                   maxfun=20 * lbfgs_iter,
                                   gtol=1e-8, ftol=1e-11,
                                   maxcor=200, disp=False))
    unpack(result.x)

    # Final losses and PR diagnostics.
    (l_theta, l_concent, l_src, l_nosrc,
     l_ic, l_ic_src, l_ic_nosrc,
     _, _, _, _) = loss_fn(model_pde, model_anten, cache, NUM_ANTS)

    with torch.enable_grad():
        inp_g  = cache['input_pde'].detach().clone().requires_grad_(True)
        ones_g = torch.ones((inp_g.shape[0], 1), device=device, dtype=dtype)
        dcdx = autograd.grad(model_pde(inp_g), inp_g, ones_g,
                             create_graph=False)[0]
    n_b = cache['n_body']; n_s = cache['n_sensor']
    dx_s = dcdx[n_b:n_b + n_s, [1]].reshape(-1, NUM_ANTS)
    dy_s = dcdx[n_b:n_b + n_s, [2]].reshape(-1, NUM_ANTS)
    dc_normal = -cache['vel_y_u'] * dx_s + cache['vel_x_u'] * dy_s
    dc_sel = dc_normal[1:-1][cache['mask_inner']].reshape(-1, 1)
    a_tan  = cache['accel_tan_sel']
    u_sel  = cache['u_mag_sel']
    c_func = model_anten(dc_sel)

    a_pred = (c_func.flatten() * u_sel.flatten()).cpu().detach().numpy()
    a_real = a_tan.flatten().cpu().numpy()
    corr_tan = float(np.corrcoef(a_real, a_pred)[0, 1]) if a_real.size else float('nan')

    total_iter = len(loss_warm) + len(loss_adam) + len(loss_lbfgs)

    model_pde.boundary = decision_boundary

    return dict(
        model_pde=model_pde, model_anten=model_anten,
        l_theta=float(l_theta), l_concent=float(l_concent),
        corr_tan=corr_tan,
        total_iter=total_iter,
        loss_ok=is_loss_ok(total_iter, float(l_theta), float(l_concent),
                           corr_tan),
    )


def save_models(num_colony, size_ant, tg_step, model_pde, model_anten):
    pde_path = os.path.join(
        MODEL_DIR,
        f'{num_colony}{size_ant}_step-{num_t}s_model_pde_{tg_step}')
    anten_path = os.path.join(
        MODEL_DIR,
        f'{num_colony}{size_ant}_step-{num_t}s_model_anten_{tg_step}')
    torch.save(model_pde,   pde_path)
    torch.save(model_anten, anten_path)
    return pde_path, anten_path


def train_or_load(num_colony, size_ant, tg_step,
                  cache, mask_active, body_length, decision_boundary):
    """If both checkpoint files exist at MODEL_DIR, load and return them.
    Else run the two-attempt training and save the best result.
    """
    pde_path = os.path.join(
        MODEL_DIR,
        f'{num_colony}{size_ant}_step-{num_t}s_model_pde_{tg_step}')
    anten_path = os.path.join(
        MODEL_DIR,
        f'{num_colony}{size_ant}_step-{num_t}s_model_anten_{tg_step}')

    if os.path.exists(pde_path) and os.path.exists(anten_path):
        print(f'  [skip-train] saved model exists -> '
              f'visualization only')
        m_pde   = torch.load(pde_path,   map_location=device,
                             weights_only=False).to(device).to(dtype)
        m_anten = torch.load(anten_path, map_location=device,
                             weights_only=False).to(device).to(dtype)
        return m_pde, m_anten

    print(f'  [train] no saved model — running two-attempt training')
    best = None
    final_hit = False
    for attempt_i, w_theta in enumerate(W_THETA_ATTEMPTS):
        tag = f'attempt {attempt_i+1}/{len(W_THETA_ATTEMPTS)} (w_theta={w_theta})'
        print(f'    {tag}')
        out = train_one_attempt(cache, mask_active, body_length,
                                decision_boundary, w_theta)
        print(f'    -> total_iter={out["total_iter"]}  '
              f'L_ang={out["l_theta"]:.2e}  L_c={out["l_concent"]:.2e}  '
              f'corr_tan={out["corr_tan"]:+.3f}  loss_ok={out["loss_ok"]}')
        if out['loss_ok']:
            save_models(num_colony, size_ant, tg_step,
                        out['model_pde'], out['model_anten'])
            print(f'    [save FINAL] criteria met on {tag}')
            best = out; final_hit = True
            break
        # Track best by lowest L_ang.
        if best is None or out['l_theta'] < best['l_theta']:
            best = out

    if not final_hit:
        save_models(num_colony, size_ant, tg_step,
                    best['model_pde'], best['model_anten'])
        print(f'    [save BEST] L_ang={best["l_theta"]:.2e}  '
              f'(criteria not met after {len(W_THETA_ATTEMPTS)} attempts)')

    return best['model_pde'], best['model_anten']


# ============================================================================
# Visualization (7 artifacts)
# ============================================================================
def save_anim(anim, path_noext, fps=10, dpi_mp4=150, dpi_gif=100):
    try:
        writer = animation.FFMpegWriter(fps=fps)
        out = path_noext + '.mp4'
        anim.save(out, writer=writer, dpi=dpi_mp4)
        print(f'    saved: {out}')
        return out
    except Exception as e:
        print(f'    FFMpeg unavailable ({type(e).__name__}: {e}); saving GIF.')
        writer = animation.PillowWriter(fps=fps)
        out = path_noext + '.gif'
        anim.save(out, writer=writer, dpi=dpi_gif)
        print(f'    saved: {out}')
        return out


def lateral_grad_at_sensor(model_pde, cache, num_ants):
    """∇⊥c at sensor rows via autograd."""
    inp = cache['input_pde']
    pred = model_pde(inp)
    dc = autograd.grad(pred, inp, cache['ones_buf'],
                       create_graph=False, retain_graph=False)[0]
    n_b = cache['n_body']; n_s = cache['n_sensor']
    dx_s = dc[n_b:n_b + n_s, [1]].reshape(-1, num_ants)
    dy_s = dc[n_b:n_b + n_s, [2]].reshape(-1, num_ants)
    return -cache['vel_y_u'] * dx_s + cache['vel_x_u'] * dy_s


def visualize(num_colony, size_ant, tg_step, body_length,
              cache, t_train, decision_boundary,
              model_pde, model_anten):
    model_pde.eval(); model_anten.eval()

    out_dir = os.path.join(
        RESULTS_DIR,
        f'visualization_{num_colony}{size_ant}_step-{num_t}s_idx{tg_step}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'    output dir: {out_dir}')

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

    # ---- (1) Response curve (NN line) + PR scatter ------------------------
    # Same two-panel style as the per-segment plots in
    # first7_segments_averaging.py: left = NN curve over [-2, 2] with
    # NaN-masking outside the segment's reliable ∇⊥c percentile band;
    # right = predicted vs measured tangential acceleration scatter.
    print('    [1/5] Response curve + PR scatter...')
    plt.rc('font', size=16, family='Arial')

    dc_lat     = lateral_grad_at_sensor(model_pde, cache, NUM_ANTS)
    dc_lat_sel = dc_lat[1:-1][mask_active[1:-1]].reshape(-1, 1)

    accel_tan = (cache['accel'][:, :NUM_ANTS] * (-cache['vel_y_u'])
                 + cache['accel'][:, NUM_ANTS:] * cache['vel_x_u'])
    accel_tan_sel = accel_tan[1:-1][mask_active[1:-1]]
    u_mag_sel     = cache['u_mag'][1:-1][mask_active[1:-1]]
    c_func        = model_anten(dc_lat_sel)

    dc_np = dc_lat_sel.flatten().detach().cpu().numpy()
    a_pred = (c_func.flatten() * u_mag_sel.flatten()).detach().cpu().numpy()
    a_real = accel_tan_sel.flatten().cpu().numpy()

    # Reliable ∇⊥c support for this segment (matches averaging defaults).
    DC_PCT_LO = 0.5; DC_PCT_HI = 99.5
    if dc_np.size:
        lower_dc = float(np.percentile(dc_np, DC_PCT_LO))
        upper_dc = float(np.percentile(dc_np, DC_PCT_HI))
    else:
        lower_dc, upper_dc = -np.inf, np.inf

    plot_step = 0.001
    dc_grid_t = torch.arange(-5, 5 + plot_step, plot_step,
                             device=device, dtype=dtype).reshape(-1, 1)
    dc_grid_np = dc_grid_t.flatten().cpu().numpy()
    with torch.no_grad():
        anten_curve = model_anten(dc_grid_t).flatten().cpu().numpy()
    anten_curve = np.where(
        (dc_grid_np >= lower_dc) & (dc_grid_np <= upper_dc),
        anten_curve, np.nan)

    color_blue = np.array([0, 100, 250]) / 255
    fig, axs_s = plt.subplots(1, 2, figsize=(12, 4))
    axs_s[0].plot(dc_grid_np, anten_curve, color=color_blue, linewidth=2)
    axs_s[0].set_xlabel(r'$\nabla_\perp c$')
    axs_s[0].set_ylabel(r'$a_{n}$')
    axs_s[0].set_xlim(-2, 2); axs_s[0].set_ylim(-1.7, 1.7)
    axs_s[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axs_s[0].set_title(f'{num_colony}{size_ant} seg {tg_step} — response')

    guide_x = np.arange(-0.05, 0.05, 0.001)
    axs_s[1].scatter(a_pred, a_real, color=color_blue, s=10)
    axs_s[1].plot(guide_x, guide_x, color='k', linestyle='--', lw=1)
    axs_s[1].set_xlabel(r'$a_{predicted}$')
    axs_s[1].set_ylabel(r'$a_{real}$')
    axs_s[1].set_xlim(-0.05, 0.05); axs_s[1].set_ylim(-0.05, 0.05)
    axs_s[1].set_title(f'{num_colony}{size_ant} seg {tg_step} — PR-curve')
    plt.subplots_adjust(wspace=0.35)
    plt.savefig(os.path.join(out_dir, 'scatter_G_vs_dc.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    # ---- (2) Trajectory animation -----------------------------------------
    print('    [2/5] Trajectory animation...')
    plt.rc('font', size=20)
    pred_xy = np.concatenate([cache['pos_x'].cpu().numpy(),
                              cache['pos_y'].cpu().numpy()], axis=1)
    vel_xy  = np.concatenate([cache['vel_x'].cpu().numpy(),
                              cache['vel_y'].cpu().numpy()], axis=1)
    t_np    = t_train.cpu().numpy()
    half    = NUM_ANTS
    num_frames = num_t
    cmap1 = plt.cm.get_cmap('jet', NUM_ANTS); color_idx = np.arange(NUM_ANTS)

    fig2 = plt.figure(figsize=(8, 8)); ax2 = plt.axes()
    def animate_traj(f):
        window = 0.4
        ax2.clear()
        ax2.scatter(pred_xy[f, 0:half], pred_xy[f, half:], c=cmap1(color_idx), s=50)
        origin = np.array([pred_xy[f, 0:half], pred_xy[f, half:]])
        ax2.quiver(*origin, vel_xy[f, 0:half], vel_xy[f, half:],
                   scale_units='xy', angles='xy', scale=1,
                   width=0.003, headwidth=5)
        ax2.add_artist(plt.Circle((0, 0), radius=R_wall, fill=False, ls='--'))
        ax2.set_title(f'Time: {t_np[f]:.2f}')
        ax2.set_xlim(-window, window); ax2.set_ylim(-window, window)
    save_anim(animation.FuncAnimation(fig2, animate_traj, frames=num_frames),
              os.path.join(out_dir, 'trajectory'))
    plt.close(fig2)

    # ---- (3) Concentration field animation --------------------------------
    print('    [3/5] Concentration field animation...')
    plt.rc('font', family='Arial', size=20)
    with torch.no_grad():
        pred_pde_vis = model_pde(cache['input_pde'])
    pred_pde_np  = pred_pde_vis.detach().cpu().numpy()
    input_pde_np = cache['input_pde'].detach().cpu().numpy()
    time_idx_np  = cache['time_idx'].cpu().numpy()

    base_cmap = plt.cm.get_cmap('OrRd')
    cmap_arr  = base_cmap(np.arange(base_cmap.N))
    cmap_arr[:, -1] = 0.8
    cmap = mpl.colors.ListedColormap(cmap_arr)
    vmax = float(pred_pde_vis.max().item())
    norm = mpl.colors.Normalize(vmin=0, vmax=max(vmax, 1e-8))
    cmapping = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig3 = plt.figure(figsize=(9, 8)); ax3 = plt.axes()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig3.colorbar(cmapping, ax=ax3, pad=0.02, fraction=0.046, shrink=1)
    def animate_field(f):
        window = 0.4
        ax3.clear()
        m = time_idx_np == f
        x_plot = input_pde_np[m, 1]; y_plot = input_pde_np[m, 2]
        c_plot = pred_pde_np[m, 0]
        try:
            ax3.tricontourf(x_plot, y_plot, c_plot, 100, cmap=cmap)
        except Exception:
            ax3.scatter(x_plot, y_plot, c=c_plot, s=5, cmap=cmap, norm=norm)
        ax3.scatter(pred_xy[f, 0:half], pred_xy[f, half:], s=100, color='k')
        origin = np.array([pred_xy[f, 0:half], pred_xy[f, half:]])
        ax3.quiver(*origin, vel_xy[f, 0:half], vel_xy[f, half:],
                   scale_units='xy', angles='xy', scale=1,
                   width=0.003, headwidth=5)
        ax3.add_artist(plt.Circle((0, 0), radius=R_wall, fill=False, ls='--'))
        ax3.set_title(f'Time: {t_np[f]:.2f}')
        ax3.set_xlim(-window, window); ax3.set_ylim(-window, window)
    save_anim(animation.FuncAnimation(fig3, animate_field, frames=num_frames),
              os.path.join(out_dir, 'concentration_field'))
    plt.close(fig3)

    # ---- (4) Mesh + trajectories (static) ---------------------------------
    print('    [4/5] Mesh + trajectories static plot...')
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    ax4.scatter(input_pde_np[:, 1], input_pde_np[:, 2], color='tab:cyan', s=10)
    for j in range(half):
        ax4.plot(pred_xy[:, j], pred_xy[:, half + j],
                 linestyle='--', lw=0.25, color='black')
    ax4.set_xlabel('$x$ (m)'); ax4.set_ylabel('$y$ (m)')
    ax4.set_xlim(-0.3, 0.3); ax4.set_ylim(-0.3, 0.3); ax4.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mesh_and_trajectories.png'), dpi=150)
    plt.close(fig4)

    # ---- (5) Active/inactive trajectories animation -----------------------
    print('    [5/5] Active/inactive trajectory animation...')
    color_gray = np.array([150, 150, 150]) / 255
    color_blue = np.array([  0, 100, 250]) / 255

    velocity_mag = np.sqrt(vel_xy[:, :half] ** 2 + vel_xy[:, half:] ** 2)
    dist_center  = np.sqrt(pred_xy[:, :half] ** 2 + pred_xy[:, half:] ** 2)
    mask_u_np    = velocity_mag >= decision_boundary
    r_hat_x_np = pred_xy[:, :half] / (dist_center + eps)
    r_hat_y_np = pred_xy[:, half:] / (dist_center + eps)
    v_radial_np = (vel_xy[:, :half] * r_hat_x_np
                   + vel_xy[:, half:] * r_hat_y_np)
    SIN_TOL = np.sin(10 * np.pi / 180)
    wall_proximity_np = dist_center > (R_wall - 2 * body_length)
    wall_velocity_np  = np.abs(v_radial_np) < SIN_TOL * velocity_mag
    wall_following_np = wall_proximity_np & wall_velocity_np
    mask_active_np = mask_u_np & (~wall_following_np)
    color = np.where(mask_active_np[:, :, np.newaxis], color_blue, color_gray)

    fig5 = plt.figure(figsize=(8, 8)); ax5 = plt.axes()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
    def animate_act(i):
        f = min(i, num_frames - 1); window = 0.3
        ax5.clear()
        tail = 5; s0 = max(f - tail, 0)
        for j in range(half):
            ax5.plot(pred_xy[s0:f + 1, j], pred_xy[s0:f + 1, half + j],
                     linestyle='--', lw=2, alpha=0.5, color=color[f, j])
        ax5.scatter(pred_xy[f, 0:half], pred_xy[f, half:], c=color[f], s=200)
        origin = np.array([pred_xy[f, 0:half], pred_xy[f, half:]])
        ax5.quiver(*origin, vel_xy[f, 0:half], vel_xy[f, half:],
                   scale_units='xy', angles='xy', scale=1,
                   width=0.003, headwidth=5)
        ax5.add_artist(plt.Circle((0, 0), radius=R_wall, fill=False, ls='--'))
        ax5.set_title(f'Time: {t_np[f]:.2f}')
        ax5.set_xlabel('$x$ (m)'); ax5.set_ylabel('$y$ (m)')
        ax5.set_xlim(-window, window); ax5.set_ylim(-window, window)
        ax5.set_aspect('equal')
    save_anim(animation.FuncAnimation(fig5, animate_act, frames=num_frames + 5),
              os.path.join(out_dir, 'trajectories_active'))
    plt.close(fig5)

    print(f'    all artifacts saved in: {out_dir}')


# ============================================================================
# Main loop
# ============================================================================
def main():
    print('Device:', device)
    print(f'  data path : {DATA_PATH}')
    print(f'  model dir : {MODEL_DIR}')
    print(f'  results   : {RESULTS_DIR}')
    print(f'  num_t={num_t}  save criteria: L_ang<{SAVE_THETA_TOL:.0e}  '
          f'corr_tan>{SAVE_CORR_TAN_MIN}  L_c<{SAVE_CONCENT_TOL:.0e}')
    print(f'  w_theta attempts: {W_THETA_ATTEMPTS}')

    tik = time.time()
    print(datetime.datetime.now())

    for (num_colony, size_ant), idx_list in sorted(TARGETED_SEGMENTS.items()):
        body_length = BODY_LENGTH[size_ant]
        print(f'\n========== Colony {num_colony}  Size {size_ant}  '
              f'(body={body_length*1000:.1f} mm) ==========')

        nd, decision_boundary, train_loader = build_segment_cache(
            num_colony, size_ant, body_length)
        print(f'  available segments: {len(train_loader)}')

        target_idx_set = set(idx_list)
        for tg_step, (t_train, x_train, y_train) in enumerate(train_loader):
            if tg_step not in target_idx_set:
                continue
            print(f'\n  ----- segment idx {tg_step} -----')

            seg_start = tg_step * num_t
            seg_slice = slice(seg_start, seg_start + num_t)
            t_seg = (t_train.reshape(-1) - t_train[0]).to(device).to(dtype)
            cache = assemble_segment_cache(
                nd, seg_slice, t_seg,
                body_length=body_length, num_ants=NUM_ANTS,
                device_=device, dtype_=dtype)
            mask_active = compute_mask_active(cache, decision_boundary, body_length)

            model_pde, model_anten = train_or_load(
                num_colony, size_ant, tg_step,
                cache, mask_active, body_length, decision_boundary)

            visualize(num_colony, size_ant, tg_step, body_length,
                      cache, t_train, decision_boundary,
                      model_pde, model_anten)

    print(f'\nTotal time: {time.time() - tik:.1f}s')


if __name__ == '__main__':
    main()
