"""Shared helpers for the v2 footprint + perception-response pipeline.

Physical constants, Savitzky-Golay numdiff, PDE input builder, segment
cache assembly, loss / scipy wrapper, and decision-boundary fitting.

Kept self-contained (no import from v4_numdiff_models) so the 260424_ver2
folder is a standalone module.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset

from scipy.signal import savgol_filter
from scipy.stats import gamma as gamma_dist
from scipy.optimize import brentq


# ============================================================================
# Loss weights (env-overridable; read once at module load)
#   V2_W_THETA   — weight on loss_theta   (mechanism_nn fit; default 1.0)
#   V2_W_CONCENT — weight on loss_concent (PDE residual;    default 200.0)
#   V2_W_IC      — weight on loss_ic      (initial cond.;   default 1.0)
# ============================================================================
W_THETA   = float(os.environ.get('V2_W_THETA',   2.0))
W_CONCENT = float(os.environ.get('V2_W_CONCENT', 200.0))
W_IC      = float(os.environ.get('V2_W_IC',      1.0))


# ============================================================================
# Physical constants
# ============================================================================
BODY_LENGTH = {'S': 0.0081, 'M': 0.0097, 'L': 0.0114}
R_arena     = 0.28
R_wall      = 0.275
eps         = torch.finfo(torch.float32).eps


# ============================================================================
# Dataset
# ============================================================================
class CustomDataset(Dataset):
    def __init__(self, data_x, data_y, data_t):
        self.data_t = data_t
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return len(self.data_t)

    def __getitem__(self, idx):
        return self.data_t[idx], self.data_x[idx], self.data_y[idx]


# ============================================================================
# Savitzky–Golay numdiff (high-res then subsample)
# ============================================================================
def savgol_derivatives_highres(pos_raw, dt_raw, window_target, polyorder, stride):
    window_raw = int(window_target * stride)
    if window_raw % 2 == 0:
        window_raw += 1
    common = dict(window_length=window_raw, polyorder=polyorder,
                  delta=dt_raw, axis=0, mode='interp')
    pos_s = savgol_filter(pos_raw, deriv=0, **common)
    vel   = savgol_filter(pos_raw, deriv=1, **common)
    acc   = savgol_filter(pos_raw, deriv=2, **common)
    return pos_s[::stride], vel[::stride], acc[::stride]


def load_or_build_numdiff(cache_path, pos_x_raw, pos_y_raw, dt_raw,
                          window, polyorder, stride):
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        print(f'  numdiff cache hit:   {cache_path}')
        return {k: z[k] for k in ('pos_x', 'pos_y',
                                  'vel_x', 'vel_y',
                                  'acc_x', 'acc_y')}
    print('  numdiff cache miss:  computing Sav-Gol (high-res)...')
    pos_x_s, vel_x, acc_x = savgol_derivatives_highres(
        pos_x_raw, dt_raw, window, polyorder, stride)
    pos_y_s, vel_y, acc_y = savgol_derivatives_highres(
        pos_y_raw, dt_raw, window, polyorder, stride)
    d = dict(
        pos_x=pos_x_s.astype(np.float32), pos_y=pos_y_s.astype(np.float32),
        vel_x=vel_x.astype(np.float32),   vel_y=vel_y.astype(np.float32),
        acc_x=acc_x.astype(np.float32),   acc_y=acc_y.astype(np.float32),
    )
    np.savez(cache_path, **d)
    print(f'  numdiff cache saved: {cache_path}')
    return d


# ============================================================================
# PDE input builder
#   Layout: [ body cross-product  |  sensor (COM + body_length · v̂) ]
#   No L/R antennas. ∇⊥c at the sensor sub-block via autograd.
# ============================================================================
def build_pde_input(t_vals, pos_x_all, pos_y_all, vel_x_u, vel_y_u,
                    body_length, num_ants):
    device_ = t_vals.device
    N_time  = t_vals.shape[0]

    flat_px = pos_x_all.detach().reshape(-1)
    flat_py = pos_y_all.detach().reshape(-1)
    T_pde, X_pde = torch.meshgrid(t_vals, flat_px, indexing='ij')
    _,     Y_pde = torch.meshgrid(t_vals, flat_py, indexing='ij')
    input_body = torch.vstack([T_pde.flatten(),
                               X_pde.flatten(),
                               Y_pde.flatten()]).T
    idx_body = torch.arange(N_time, device=device_).repeat_interleave(
        N_time * num_ants)

    d_fwd = body_length
    sensor_x = pos_x_all + d_fwd * vel_x_u
    sensor_y = pos_y_all + d_fwd * vel_y_u
    t_ant = t_vals.unsqueeze(1).expand(-1, num_ants).reshape(-1, 1)
    input_sense = torch.cat([t_ant,
                             sensor_x.detach().reshape(-1, 1),
                             sensor_y.detach().reshape(-1, 1)], dim=1)
    idx_sense = torch.arange(N_time, device=device_).repeat_interleave(num_ants)

    input_pde = torch.cat([input_body, input_sense], dim=0)
    time_idx  = torch.cat([idx_body,   idx_sense])
    return input_pde, time_idx, input_body.shape[0], input_sense.shape[0]


# ============================================================================
# Per-segment cache (from Sav-Gol numdiff)
# ============================================================================
def assemble_segment_cache(nd, seg_slice, t_seg, body_length, num_ants,
                           device_, dtype_):
    """Slice cached Sav-Gol numdiff into this segment; build input_pde +
    local_dist_func."""
    def to_t(key):
        return torch.as_tensor(nd[key][seg_slice], dtype=dtype_, device=device_)

    pos_x = to_t('pos_x'); pos_y = to_t('pos_y')
    vel_x = to_t('vel_x'); vel_y = to_t('vel_y')
    acc_x = to_t('acc_x'); acc_y = to_t('acc_y')

    u_mag   = torch.sqrt(vel_x ** 2 + vel_y ** 2)
    vel_x_u = vel_x / (u_mag + eps)
    vel_y_u = vel_y / (u_mag + eps)
    accel   = torch.cat([acc_x, acc_y], dim=1)

    input_pde_raw, time_idx, n_body, n_sensor = build_pde_input(
        t_seg, pos_x, pos_y, vel_x_u, vel_y_u, body_length, num_ants)

    with torch.no_grad():
        pos_x_at = pos_x[time_idx]; pos_y_at = pos_y[time_idx]
        px = input_pde_raw[:, 1:2]; py = input_pde_raw[:, 2:3]
        dist = torch.sqrt((px - pos_x_at) ** 2 + (py - pos_y_at) ** 2)
        dist_mask = dist - body_length
        stable_eps = body_length / 20.0
        local_dist_func = torch.sum(
            torch.where(dist_mask <= 0,
                        dist_mask / (dist_mask - stable_eps),
                        torch.zeros_like(dist_mask)),
            dim=1, keepdim=True,
        )

    input_pde = input_pde_raw.clone().detach().to(device_).requires_grad_(True)
    ones_buf  = torch.ones((input_pde.shape[0], 1),
                           device=device_, dtype=dtype_, requires_grad=False)

    return dict(
        pos_x=pos_x, pos_y=pos_y,
        u_mag=u_mag, vel_x=vel_x, vel_y=vel_y,
        vel_x_u=vel_x_u, vel_y_u=vel_y_u,
        accel=accel,
        input_pde=input_pde, time_idx=time_idx,
        n_body=n_body, n_sensor=n_sensor,
        local_dist_func=local_dist_func,
        ones_buf=ones_buf,
    )


# ============================================================================
# Active-ant mask (u_mag threshold + wall-following exclusion)
# ============================================================================
def compute_mask_active(cache, decision_boundary, body_length):
    with torch.no_grad():
        pos_x = cache['pos_x']; pos_y = cache['pos_y']
        u_mag = cache['u_mag']
        vel_x = cache['vel_x']; vel_y = cache['vel_y']

        dist_c = torch.sqrt(pos_x ** 2 + pos_y ** 2)
        mask_u = u_mag >= decision_boundary

        r_hat_x = pos_x / (dist_c + eps)
        r_hat_y = pos_y / (dist_c + eps)
        v_radial = vel_x * r_hat_x + vel_y * r_hat_y

        SIN_TOL = np.sin(10 * np.pi / 180)   # 10°
        wall_proximity = dist_c > (R_wall - 2 * body_length)
        wall_velocity  = v_radial.abs() < SIN_TOL * u_mag
        wall_following = wall_proximity & wall_velocity
        mask_wall = ~wall_following

    return mask_u & mask_wall


# ============================================================================
# Pre-computed loss constants (run once per segment, after mask_active)
# ============================================================================
def attach_loss_constants(cache, mask_active, model_pde, num_ants):
    with torch.no_grad():
        vel_x_u = cache['vel_x_u']; vel_y_u = cache['vel_y_u']
        accel   = cache['accel']
        accel_tan = (accel[:, :num_ants] * (-vel_y_u)
                     + accel[:, num_ants:] *   vel_x_u)
        mask_inner = mask_active[1:-1]
        cache['mask_inner']    = mask_inner
        cache['accel_tan_sel'] = accel_tan[1:-1][mask_inner]
        cache['u_mag_sel']     = cache['u_mag'][1:-1][mask_inner]
        cache['kp_ldf']        = model_pde.kp * cache['local_dist_func']

        # IC target at t=0: c(0, x, y) = (kp/km) · local_dist_func(0, x, y)
        t0_mask = (cache['time_idx'] == 0)
        cache['t0_mask'] = t0_mask
        cache['c_ss_t0'] = (model_pde.kp / model_pde.km) * cache['local_dist_func'][t0_mask]

        # SRC/NOSRC split mask on local_dist_func > 0
        src_mask = (cache['local_dist_func'].squeeze(-1) > 0)
        cache['src_mask']    = src_mask
        # Same split restricted to t=0 rows (for loss_ic src/nosrc balance)
        cache['t0_src_mask'] = src_mask[t0_mask]


# ============================================================================
# Loss  =  1·loss_theta  +  200·loss_concent  +  1·loss_ic
#   Concentration-related weights bumped 10× (20 → 200, 0.1 → 1.0) because
#   the Fourier-feature PINN_PDE converges quickly enough that mechanism_nn
#   was over-fitting to the residual a_n; rebalancing keeps the mechanism
#   G(∇⊥c) curve physical (smooth sigmoid-flip, not zigzag).
# ============================================================================
def loss_fn(model_pde, model_anten, cache, num_ants):
    loss_func = nn.MSELoss()
    input_pde = cache['input_pde']
    n_body    = cache['n_body']
    n_sensor  = cache['n_sensor']

    pred_pde = model_pde(input_pde)
    dc_dt_dx_dy = autograd.grad(
        pred_pde, input_pde, cache['ones_buf'], create_graph=True)[0]
    dc_dt = dc_dt_dx_dy[:, [0]]

    # PDE residual — SRC/NOSRC split: source-active rows (~1%) and decay rows
    # (~99%) weighted equally so the small source population isn't drowned.
    f = cache['kp_ldf'] - model_pde.km * pred_pde
    resid2 = (dc_dt - f).pow(2).squeeze(-1)
    src_mask = cache['src_mask']
    loss_src   = resid2[src_mask].mean()
    loss_nosrc = resid2[~src_mask].mean()
    loss_concent = loss_src + loss_nosrc

    # ∇⊥c at sensor block via autograd (first n_sensor rows of sense block)
    sensor_start = n_body
    sensor_end   = n_body + n_sensor
    dc_dx_s = dc_dt_dx_dy[sensor_start:sensor_end, [1]].reshape(-1, num_ants)
    dc_dy_s = dc_dt_dx_dy[sensor_start:sensor_end, [2]].reshape(-1, num_ants)
    vel_x_u = cache['vel_x_u']; vel_y_u = cache['vel_y_u']
    dc_normal = -vel_y_u * dc_dx_s + vel_x_u * dc_dy_s
    dc_normal_sel = dc_normal[1:-1][cache['mask_inner']].reshape(-1, 1)

    g = model_anten(dc_normal_sel)
    loss_theta = loss_func(
        cache['accel_tan_sel'].flatten(),
        g.flatten() * cache['u_mag_sel'].flatten(),
    )

    # Initial condition: c(t=0, x, y) = (kp/km) · local_dist_func(0, x, y)
    # SRC/NOSRC split (same reasoning as loss_concent): body-active t=0 points
    # (~1%) vs empty t=0 points (~99%) weighted equally so the small source
    # IC gradient isn't drowned by the empty-region gradient.
    t0_mask       = cache['t0_mask']
    t0_src_mask   = cache['t0_src_mask']
    pred_t0   = pred_pde[t0_mask]
    target_t0 = cache['c_ss_t0']
    resid_ic2 = (pred_t0 - target_t0).pow(2).squeeze(-1)
    loss_ic_src   = resid_ic2[t0_src_mask].mean()
    loss_ic_nosrc = resid_ic2[~t0_src_mask].mean()
    loss_ic = loss_ic_src + loss_ic_nosrc

    # Initial condition on dc/dt: steady-state IC implies dc/dt(t=0) = 0
    # everywhere (follows from c(0) = (kp/km)·ldf(0) + PDE residual = 0).
    # Explicit enforcement — same SRC/NOSRC split as loss_ic.
    dc_dt_t0 = dc_dt[t0_mask].squeeze(-1)
    resid_ic_dt2 = dc_dt_t0.pow(2)
    loss_ic_dt_src   = resid_ic_dt2[t0_src_mask].mean()
    loss_ic_dt_nosrc = resid_ic_dt2[~t0_src_mask].mean()
    loss_ic_dt = loss_ic_dt_src + loss_ic_dt_nosrc

    loss_total = (W_THETA * loss_theta + W_CONCENT * loss_concent
                  + W_IC * loss_ic)
    return (loss_theta, loss_concent, loss_src, loss_nosrc,
            loss_ic, loss_ic_src, loss_ic_nosrc,
            loss_ic_dt, loss_ic_dt_src, loss_ic_dt_nosrc,
            loss_total)


# ============================================================================
# SciPy L-BFGS-B glue — flat numpy ↔ torch parameters
# ============================================================================
def make_scipy_lbfgs_obj(model_pde, model_anten, cache, num_ants, loss_fn_):
    trainable = [p for p in (list(model_pde.parameters())
                             + list(model_anten.parameters()))
                 if p.requires_grad]
    shapes = [p.shape for p in trainable]
    numels = [p.numel() for p in trainable]
    dev_   = trainable[0].device
    dt_    = trainable[0].dtype

    def unpack(x_np):
        x_t = torch.from_numpy(np.asarray(x_np, dtype=np.float32)).to(
            dev_, dtype=dt_)
        offs = 0
        with torch.no_grad():
            for p, n, s in zip(trainable, numels, shapes):
                p.copy_(x_t[offs:offs+n].view(s))
                offs += n

    def pack_current():
        return torch.cat(
            [p.detach().reshape(-1) for p in trainable]
        ).cpu().numpy().astype(np.float64)

    log = []
    theta_log = []

    def obj(x_np):
        unpack(x_np)
        for p in trainable:
            if p.grad is not None:
                p.grad = None
        model_pde.train(); model_anten.train()
        losses = loss_fn_(model_pde, model_anten, cache, num_ants)
        l_theta = losses[0]
        total   = losses[-1]
        if torch.isfinite(total):
            total.backward()
        grad_flat = torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1)
            for p in trainable
        ]).cpu().numpy().astype(np.float64)
        loss_val = float(total.detach().cpu())
        log.append(loss_val)
        theta_log.append(float(l_theta.detach().cpu()))
        return loss_val, grad_flat

    return obj, pack_current(), unpack, log, theta_log


# ============================================================================
# Physical slope check at origin
#   Over the band |∇⊥c| < 0.5, find GLOBAL max and min of G(∇⊥c) and compute
#   slope between those extrema. Ants must turn AWAY from rising lateral c,
#   so slope must be negative.
# ============================================================================
def slope_at_origin(dc_np, g_np, band=0.5):
    mask = (dc_np > -band) & (dc_np < band)
    if mask.sum() < 2:
        return float('nan')
    g_b = g_np[mask]; x_b = dc_np[mask]
    i_max = int(g_b.argmax())
    i_min = int(g_b.argmin())
    x_max = x_b[i_max]; y_max = g_b[i_max]
    x_min = x_b[i_min]; y_min = g_b[i_min]
    return float((y_min - y_max) / (x_min - x_max))


# ============================================================================
# Decision boundary — 2-component Gamma mixture EM + brentq root
# ============================================================================
def fit_gamma_mixture_2(x, max_iter=200, tol=1e-7):
    x = np.asarray(x).flatten()
    x = x[x > 0]
    med = np.median(x)
    params = []
    for subset in (x[x < med], x[x >= med]):
        m = subset.mean(); v = subset.var() + 1e-12
        params.append({'shape': m * m / v, 'scale': v / m})
    pi = np.array([0.5, 0.5])
    ll_prev = -np.inf
    for _ in range(max_iter):
        log_p = np.stack([
            np.log(pi[k] + 1e-12)
            + gamma_dist.logpdf(x, params[k]['shape'],
                                loc=0, scale=params[k]['scale'])
            for k in range(2)
        ], axis=1)
        log_max = log_p.max(axis=1, keepdims=True)
        resp = np.exp(log_p - log_max)
        resp /= resp.sum(axis=1, keepdims=True)
        ll = float(np.sum(log_max.ravel()
                          + np.log(np.exp(log_p - log_max).sum(axis=1))))
        for k in range(2):
            Nk = resp[:, k].sum()
            pi[k] = Nk / len(x)
            mk = (resp[:, k] * x).sum() / Nk
            vk = (resp[:, k] * (x - mk) ** 2).sum() / Nk + 1e-12
            params[k]['shape'] = mk * mk / vk
            params[k]['scale'] = vk / mk
        if abs(ll - ll_prev) < tol:
            break
        ll_prev = ll
    return pi, params


def load_or_build_boundary(cache_path, u_mag_flat):
    if os.path.exists(cache_path):
        b = float(np.load(cache_path)['boundary'].reshape(-1)[0])
        print(f'  boundary cache hit:  {cache_path} -> {b:.4f}')
        return b
    pi, params_gamma = fit_gamma_mixture_2(u_mag_flat.reshape(-1, 1))
    means_k = np.array([p['shape'] * p['scale'] for p in params_gamma])
    larger  = int(np.argmax(means_k))
    smaller = int(np.argmin(means_k))

    def posterior_diff(xv):
        lp = [np.log(pi[k] + 1e-12)
              + gamma_dist.logpdf(xv, params_gamma[k]['shape'],
                                  scale=params_gamma[k]['scale'])
              for k in range(2)]
        return lp[larger] - lp[smaller]

    lo, hi = float(means_k.min() * 1.05), float(means_k.max() * 0.95)
    b = float(brentq(posterior_diff, lo, hi))
    np.savez(cache_path, boundary=np.array([b]))
    print(f'  boundary cache saved:{cache_path} -> {b:.4f}')
    return b
