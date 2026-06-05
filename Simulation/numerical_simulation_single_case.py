"""Single-case numerical simulation of the agent-based ant model.

Same physics, response function, and numerical operators as the Fig 4
contour-sweep file in the paper, but driven by a single set of
parameters chosen to match the colony-mean values:

    I_ave = 1.0       (G_peak_abs_mean colony-mean)
    I_att = 0.1       (G_region_abs_mean colony-mean)
    lambda_att = 0.6
    noise_std  = 0.1

Outputs (saved into Results/, both .jpg and .eps for the still plots):
  - Simulation_response_function_*       response curve f(g)
  - Simulation_Aggregation_Probability_* density distribution P(rho > 1)
  - Simulation_MSD_*                     mean MSD (log-log) with slope-1/2 guides
  - Simulation_trajectory_*.mp4          periodic-wrapped trajectory with
                                          concentration field overlay
  - Simulation_continuous_trajectory_*.mp4   unwrapped trajectory animation

Plot styles follow numerical_simulation/260516_plot_only_from_npz.py .

Requires FFmpeg on the path (or set V2_FFMPEG_PATH env var) for MP4 export.
"""

import os
import sys
import time
import datetime

import numpy as np
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
         _save_anim() handles it).
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
from matplotlib.colors import LinearSegmentedColormap

# Force UTF-8 on stdout/stderr for the lambda / sigma print statements.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


# ============================================================================
# Parameters
# ============================================================================
# Response-function shape constants.
ALPHA   = 2.2
BETA    = 5.0
SQRT_2E = np.sqrt(2.0 * np.e)

# Colony-mean values (from Result/selective_wide5/summary.csv aggregation).
I_AVE     = 1.0
I_ATT     = 0.1
LAM_ATT   = 0.6
NOISE_STD = 0.1   # data-calibrated mean sigma_dθ/dt ≈ 0.108 (Δt_exp = 1s)

# Physical / numerical parameters.
kp             = 0.01
km             = 0.1   # matches the trained PINN_PDE model (v2_models.PINN_PDE)
ph_r           = 0.01
body_length    = ph_r
density_radius = 0.03
L              = 0.5

_dx_target = ph_r / 3.0
N      = int(round(2 * L / _dx_target)) + 1
dx     = 2 * L / (N - 1)
x_grid = np.linspace(-L, L, N)
y_grid = np.linspace(-L, L, N)

n_agents = 65
v0       = 0.02
T_end    = 200.0
dt_sim   = 0.1

save_dt       = 1.0
save_stride   = int(round(save_dt / dt_sim))
T_save        = int(round(T_end / save_dt))
n_total_steps = T_save * save_stride

SEED = 0

# Fixed concentration colorbar range, matching 260516_plot_only_from_npz.py.
C_VMAX_FIXED = 0.08


# ============================================================================
# Precomputed ball footprint for H_sum
# ============================================================================
_rc = int(np.ceil(ph_r / dx)) + 1
_di, _dj = np.mgrid[-_rc:_rc + 1, -_rc:_rc + 1]
_ball_mask = (_di * dx) ** 2 + (_dj * dx) ** 2 <= ph_r ** 2
BALL_I = _di[_ball_mask].astype(np.int64)
BALL_J = _dj[_ball_mask].astype(np.int64)


# ============================================================================
# Core operators
# ============================================================================
def wrap_periodic(vals):
    return (vals + L) % (2 * L) - L


def compute_H_sum(pos_x, pos_y):
    ax = np.floor((pos_x + L) / dx + 0.5).astype(np.int64)
    ay = np.floor((pos_y + L) / dx + 0.5).astype(np.int64)
    ii = np.mod(ax[:, None] + BALL_I[None, :], N).ravel()
    jj = np.mod(ay[:, None] + BALL_J[None, :], N).ravel()
    flat = ii * N + jj
    return np.bincount(flat, minlength=N * N).astype(np.float64).reshape(N, N)


def gradient_periodic(c):
    dc_dx = (np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)) * (0.5 / dx)
    dc_dy = (np.roll(c, -1, axis=1) - np.roll(c, 1, axis=1)) * (0.5 / dx)
    return dc_dx, dc_dy


def bilinear_periodic(field, px, py):
    fx = (px + L) / dx
    fy = (py + L) / dx
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    wx = fx - i0; wy = fy - j0
    i0m = np.mod(i0, N);     j0m = np.mod(j0, N)
    i1m = np.mod(i0 + 1, N); j1m = np.mod(j0 + 1, N)
    return (field[i0m, j0m] * (1 - wx) * (1 - wy)
            + field[i1m, j0m] * wx * (1 - wy)
            + field[i0m, j1m] * (1 - wx) * wy
            + field[i1m, j1m] * wx * wy)


def response_function(g, I_ave, I_att, lam_att):
    abs_g = np.abs(g)
    if lam_att <= 0.0:
        return I_att * np.sign(g) * np.tanh(BETA * abs_g)
    u = ALPHA * g / lam_att
    aver = -I_ave * SQRT_2E * u * np.exp(-u * u)
    ramp = np.maximum(np.tanh(BETA * (abs_g - lam_att)), 0.0)
    attr = I_att * np.sign(g) * ramp
    return aver + attr


def compute_dtheta(theta, pos_x, pos_y, c, lam_att, I_ave, I_att):
    cos_th = np.cos(theta); sin_th = np.sin(theta)
    sx = wrap_periodic(pos_x + body_length * cos_th)
    sy = wrap_periodic(pos_y + body_length * sin_th)
    dc_dx, dc_dy = gradient_periodic(c)
    gx = bilinear_periodic(dc_dx, sx, sy)
    gy = bilinear_periodic(dc_dy, sx, sy)
    g = -sin_th * gx + cos_th * gy
    return response_function(g, I_ave, I_att, lam_att)


# ============================================================================
# Single trial
# ============================================================================
def run_single_trial(seed, lam_att, I_ave, I_att, noise_strength):
    rng = np.random.default_rng(seed)
    pos_x = rng.random(n_agents) * 2 * L - L
    pos_y = rng.random(n_agents) * 2 * L - L
    theta = rng.random(n_agents) * 2 * np.pi
    c     = np.zeros((N, N), dtype=np.float64)

    pos_x_hist = np.empty((T_save, n_agents), dtype=np.float32)
    pos_y_hist = np.empty((T_save, n_agents), dtype=np.float32)
    theta_hist = np.empty((T_save, n_agents), dtype=np.float32)
    c_hist     = np.empty((T_save, N, N),     dtype=np.float32)

    for step in range(n_total_steps):
        if step % save_stride == 0:
            si = step // save_stride
            pos_x_hist[si] = pos_x
            pos_y_hist[si] = pos_y
            theta_hist[si] = theta
            c_hist[si]     = c

        dtheta = compute_dtheta(theta, pos_x, pos_y, c, lam_att, I_ave, I_att)
        dW = rng.standard_normal(n_agents) * np.sqrt(dt_sim)
        pos_x = wrap_periodic(pos_x + v0 * np.cos(theta) * dt_sim)
        pos_y = wrap_periodic(pos_y + v0 * np.sin(theta) * dt_sim)
        theta = theta + dtheta * dt_sim + noise_strength * dW
        c = c + (kp * compute_H_sum(pos_x, pos_y) - km * c) * dt_sim

    return pos_x_hist, pos_y_hist, theta_hist, c_hist


def reconstruct_continuous(pos_hist):
    cont = np.empty_like(pos_hist)
    cont[0] = pos_hist[0]
    for t in range(1, pos_hist.shape[0]):
        d = pos_hist[t] - pos_hist[t - 1]
        d = (d + L) % (2 * L) - L
        cont[t] = cont[t - 1] + d
    return cont


def msd_curve(cx, cy):
    """Mean MSD (and std across agents) and the slope fit at t in [15, 100]."""
    T = cx.shape[0]
    msd_all = np.empty((cx.shape[1], T), dtype=np.float64)
    for k in range(cx.shape[1]):
        x = cx[:, k] - cx[0, k]
        y = cy[:, k] - cy[0, k]
        msd_all[k] = x * x + y * y
    mean = msd_all.mean(axis=0)
    std  = msd_all.std(axis=0)
    t_vals = np.arange(T, dtype=np.float64)
    slope = float(np.polyfit(np.log10(t_vals[15:100]),
                             np.log10(mean[15:100]), 1)[0])
    return t_vals, mean, std, slope


# ============================================================================
# Plot helpers (style from 260516_plot_only_from_npz.py)
# ============================================================================
_WOR_COLORS = [
    np.array([255, 250, 245]) / 255.0,
    np.array([255, 179, 141]) / 255.0,
    np.array([234,  85,  20]) / 255.0,
]
_WOR_STOPS = [0.0, 0.5, 1.0]
WOR_CMAP = LinearSegmentedColormap.from_list(
    'WhiteOrangeRed', list(zip(_WOR_STOPS, _WOR_COLORS)))


def _savefig_both(fig, base_path_no_ext, dpi=300):
    fig.savefig(base_path_no_ext + '.jpg', bbox_inches='tight', dpi=dpi)
    fig.savefig(base_path_no_ext + '.eps', bbox_inches='tight', format='eps')


def plot_response_function(save_base, I_ave, I_att, lam_att):
    plt.rc('font', size=14, family='Arial')
    fig = plt.figure(figsize=(5, 3))
    g = np.arange(-3.0, 3.001, 0.005)
    resp = response_function(g, I_ave, I_att, lam_att)
    plt.plot(g, resp, lw=3)
    plt.xlabel(r'$\nabla_{\perp} c$')
    plt.ylabel(r'$f(\nabla_{\perp} c)$')
    plt.xticks(np.arange(-1.5, 2.0, 0.5))
    plt.xlim(-1.5, 1.5)
    _savefig_both(fig, save_base, dpi=300)
    plt.close(fig)
    print(f'  saved: {save_base}.{{jpg,eps}}')


def plot_density_pdf(save_base, pos_x_hist, pos_y_hist):
    T = pos_x_hist.shape[0]
    dens = np.empty((T, n_agents), dtype=np.int32)
    for t in range(T):
        xt = pos_x_hist[t]; yt = pos_y_hist[t]
        d2 = (xt[:, None] - xt[None, :]) ** 2 + (yt[:, None] - yt[None, :]) ** 2
        dens[t] = (d2 <= density_radius ** 2).sum(axis=1)
    sel = dens.ravel()
    bin_edges = np.arange(-0.5, sel.max() + 1.5, 1.0)
    counts, _ = np.histogram(sel, bins=bin_edges, density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    p_gt1 = float(counts[centers > 1].sum())

    plt.rc('font', size=14, family='Arial')
    fig = plt.figure(figsize=(6, 4))
    plt.plot(centers, counts, 'o-', linewidth=2, markersize=10,
             color='tab:green',
             label=r"Prob.($\rho > 1$) = " + f"{p_gt1:.4f}")
    plt.xlabel('Density'); plt.ylabel('Probability')
    plt.xticks(np.arange(0, 16, 1)); plt.yticks(np.arange(0, 1.2, 0.2))
    plt.xlim(0, max(6, int(centers.max()) + 1)); plt.ylim(0, 1)
    plt.legend(loc='upper right', fontsize=16)
    _savefig_both(fig, save_base, dpi=300)
    plt.close(fig)
    print(f'  saved: {save_base}.{{jpg,eps}}  (P(rho>1)={p_gt1:.4f})')
    return p_gt1


def plot_msd_curve(save_base, cx, cy):
    t_vals, mean, std, slope = msd_curve(cx, cy)
    plt.rc('font', size=14, family='Arial')
    fig = plt.figure(figsize=(6, 4))
    plt.plot(t_vals[1:], mean[1:], lw=3, color='k',
             label=f'Mean MSD, β = {slope:.2f} @15<t<100', zorder=1)
    base_t = t_vals[1]; base_m = mean[1]
    plt.plot(t_vals[1:], base_m * (t_vals[1:] / base_t) ** 2,
             'b--', label='Slope = 2', zorder=2)
    plt.plot(t_vals[1:], base_m * (t_vals[1:] / base_t) ** 1,
             'r--', label='Slope = 1', zorder=2)
    plt.xscale('log'); plt.yscale('log')
    plt.xlim(1, 200); plt.ylim(2e-4, 20)
    plt.xticks([1, 10, 100], [r'$10^0$', r'$10^1$', r'$10^2$'])
    plt.yticks([1e-2, 1e0], [r'$10^{-2}$', r'$10^0$'])
    plt.minorticks_off()
    plt.xlabel('Time (t)'); plt.ylabel('MSD')
    plt.legend(loc='best', fontsize=14)
    _savefig_both(fig, save_base, dpi=300)
    plt.close(fig)
    print(f'  saved: {save_base}.{{jpg,eps}}  (slope={slope:.4f})')
    return slope


def _save_anim(anim, save_path_no_ext, fps=10, dpi_mp4=150, dpi_gif=100):
    """Save as MP4 if FFmpeg works, otherwise fall back to GIF."""
    try:
        writer = animation.FFMpegWriter(fps=fps)
        out = save_path_no_ext + '.mp4'
        anim.save(out, writer=writer, dpi=dpi_mp4)
        return out
    except Exception as e:
        print(f'  FFMpeg unavailable ({type(e).__name__}: {e}); saving GIF.')
        writer = animation.PillowWriter(fps=fps)
        out = save_path_no_ext + '.gif'
        anim.save(out, writer=writer, dpi=dpi_gif)
        return out


def animate_trajectory_with_concentration(save_path_no_ext,
                                          pos_x_hist, pos_y_hist,
                                          theta_hist, c_hist,
                                          sel=0, fps=10, dpi=240,
                                          c_vmax=None):
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid, indexing='ij')
    if c_vmax is None:
        c_vmax = C_VMAX_FIXED
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(c_vmax, 1e-12))
    colormapping = cm.ScalarMappable(norm=norm, cmap=WOR_CMAP)
    t_eval = np.arange(T_save, dtype=np.float64)

    plt.rc('font', size=16, family='Arial')
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))

    def _draw(i):
        if i >= T_save:
            i = T_save - 1
        ax.clear()
        ax.contourf(X_mesh, Y_mesh, c_hist[i], 100, cmap=WOR_CMAP, norm=norm)
        px = pos_x_hist[i]; py = pos_y_hist[i]; th_ = theta_hist[i]
        vx = v0 * np.cos(th_); vy = v0 * np.sin(th_)
        ax.scatter(px, py, s=50, alpha=0.4, color='k')
        ax.quiver(px, py, vx, vy, color='b', alpha=0.4,
                  scale_units='xy', angles='xy', scale=0.25,
                  width=0.005, headwidth=5)
        ax.scatter(pos_x_hist[i, sel], pos_y_hist[i, sel], s=50, color='k')
        ax.scatter(pos_x_hist[:i, sel], pos_y_hist[:i, sel],
                   marker='o', s=10, color='r')
        ax.quiver(pos_x_hist[i, sel], pos_y_hist[i, sel],
                  vx[sel], vy[sel], color='b', alpha=1,
                  scale_units='xy', angles='xy', scale=0.25,
                  width=0.005, headwidth=5)
        ax.set_title(f'Time: {t_eval[i]:.2f}')
        ax.set_xticks(np.arange(-1, 1, 0.5))
        ax.set_yticks(np.arange(-1, 1, 0.5))
        ax.set_xlim(-L, L); ax.set_ylim(-L, L)
        ax.set_aspect('equal')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    cbar = fig.colorbar(colormapping, ax=ax, pad=0.02, fraction=0.046, shrink=1)
    cbar.set_ticks([0.0, c_vmax])
    cbar.set_ticklabels([f'{0.0:.2f}', f'{c_vmax:.2f}'])
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)

    anim = animation.FuncAnimation(fig, _draw, frames=T_save + 3)
    out = _save_anim(anim, save_path_no_ext, fps=fps, dpi_mp4=dpi)
    plt.close(fig)
    print(f'  saved: {out}')


def animate_continuous_trajectory(save_path_no_ext, cx, cy,
                                  sel=0, fps=10, dpi=300, half_window=4.5):
    cmap_agents = plt.get_cmap('tab20', n_agents)
    color_list  = [cmap_agents(i) for i in range(n_agents)]

    plt.rc('font', size=14, family='Arial')
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    tick_step = max(1, int(round(2 * half_window / 4)))

    def _draw(i):
        if i >= T_save:
            i = T_save - 1
        ax.clear()
        for j in range(n_agents):
            ax.plot(cx[:i, j], cy[:i, j], color=color_list[j], lw=2, alpha=0.7)
        ax.scatter(cx[i], cy[i], s=75, color='k', alpha=0.7)
        ax.scatter(cx[i, sel], cy[i, sel], s=100, color='r',
                   edgecolors='black', linewidth=1.5)
        ax.plot(cx[:i, sel], cy[:i, sel], color='r', linewidth=3)
        ax.set_yticks(np.arange(-half_window + 0.1, half_window, tick_step))
        ax.set_xticks(np.arange(-half_window + 0.1, half_window, tick_step))
        ax.set_xlim(-half_window, half_window)
        ax.set_ylim(-half_window, half_window)
        ax.set_aspect('equal')

    fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    anim = animation.FuncAnimation(fig, _draw, frames=T_save + 3)
    out = _save_anim(anim, save_path_no_ext, fps=fps, dpi_mp4=dpi)
    plt.close(fig)
    print(f'  saved: {out}')


# ============================================================================
# Main
# ============================================================================
def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'Results')
    os.makedirs(out_dir, exist_ok=True)

    print(datetime.datetime.now())
    print('Single-case numerical simulation')
    print(f'  I_ave={I_AVE}  I_att={I_ATT}  lambda_att={LAM_ATT}  '
          f'noise_std={NOISE_STD}')
    print(f'  n_agents={n_agents}  T_end={T_end}s  dt_sim={dt_sim}s  '
          f'seed={SEED}')
    print(f'  domain: [{-L}, {L}] x [{-L}, {L}]  grid N={N}  dx={dx:.4f}')

    tag = (f'Iave{I_AVE}_Iatt{I_ATT}_lam{LAM_ATT}_noise{NOISE_STD}_'
           f'seed{SEED}')

    # ---- Response function ------------------------------------------------
    plot_response_function(
        os.path.join(out_dir, f'Simulation_response_function_{tag}'),
        I_AVE, I_ATT, LAM_ATT)

    # ---- Run trial --------------------------------------------------------
    tik = time.time()
    pos_x_hist, pos_y_hist, theta_hist, c_hist = run_single_trial(
        SEED, LAM_ATT, I_AVE, I_ATT, NOISE_STD)
    print(f'  trial run time: {time.time() - tik:.1f}s')

    # ---- Analysis stills --------------------------------------------------
    plot_density_pdf(
        os.path.join(out_dir, f'Simulation_Aggregation_Probability_{tag}'),
        pos_x_hist, pos_y_hist)

    cx = reconstruct_continuous(pos_x_hist)
    cy = reconstruct_continuous(pos_y_hist)
    plot_msd_curve(
        os.path.join(out_dir, f'Simulation_MSD_{tag}'),
        cx, cy)

    # ---- Animations -------------------------------------------------------
    print('  trajectory + concentration animation...')
    animate_trajectory_with_concentration(
        os.path.join(out_dir, f'Simulation_trajectory_{tag}'),
        pos_x_hist, pos_y_hist, theta_hist, c_hist)

    print('  continuous trajectory animation (unwrapped)...')
    animate_continuous_trajectory(
        os.path.join(out_dir, f'Simulation_continuous_trajectory_{tag}'),
        cx, cy)

    print(f'\nAll artifacts saved in: {out_dir}')


if __name__ == '__main__':
    main()
