"""Neural network classes for v2 footprint + perception-response training.

Both main_training.py and visualize.py / selective_averaging.py import from
this module so torch.save/torch.load round-trips cleanly.

PINN_PDE        : current best — random Fourier feature embedding + tanh MLP.
PINN_PDE_old    : original 3-hidden tanh MLP (no Fourier). Kept for reference;
                  legacy saved models pickled as `PINN_PDE` will not load
                  under the new class — they need to be retrained.
Mechanism_NN    : 32-16 sigmoid-sigmoid-linear, fixed pre_scale=5.
Mechanism_NN_LS : Mechanism_NN with configurable pre_scale / hidden widths.
Mechanism_NN_LS_Tanh
                : Mechanism_NN_LS with tanh-tanh-linear activations
                  instead of sigmoid-sigmoid-linear.  Same shape and
                  configurable pre_scale / hidden widths; output stays
                  linear via fc3.  Xavier init switched to tanh gain.
Mechanism_NN_LS_LogStretch
                : Mechanism_NN_LS + asinh log-stretch pre-transform
                  f(x) = asinh(alpha*x).  Non-linear: actually widens
                  point-spacing near zero (slope alpha at origin)
                  while compressing tails (~log) — unlike linear
                  pre_scale rescaling.  C^∞ smooth everywhere.
Mechanism_NN_LS_MultiScale
                : Mechanism_NN_LS with multiple FIXED pre_scales fed
                  in parallel as separate fc1 input channels.  fc1
                  learns to weight channels per segment, so the
                  network self-adapts to wide vs narrow ∇⊥c
                  distributions without per-segment tuning or
                  trainable scales.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PINN_PDE(nn.Module):
    """c(t, x, y) network — random Fourier feature embedding + tanh MLP.

    The (t, x, y) input is normalized (t→(t-50)/50, xy→xy/0.28) and then
    projected through a fixed Gaussian random matrix B of shape
    (input_num, fourier_dim). The MLP sees [sin(2π·B·u), cos(2π·B·u)]
    instead of the raw normalized coords. This lets a small tanh MLP
    represent the sharp body-source indicator and the locally non-smooth
    c-field much more efficiently — standard PINN trick for high-frequency
    targets (Tancik et al. 2020; Wang et al. 2021).

    Hyperparameters (best found 2026-04-25, h=64 d=3 sigma=0.75):
      hidden        : width of each hidden layer (tanh)
      depth         : number of hidden layers (default 3)
      fourier_dim   : number of random Fourier frequencies m (input is 2m)
      fourier_sigma : std of the random projection. Larger → higher freq.

    Physics constants: kp=0.01, km=0.1, ph_r=body_length.
    """
    def __init__(self, input_num, output_num, bias=True, body_length=0.01,
                 hidden=64, depth=3, fourier_dim=64, fourier_sigma=0.75,
                 output_scale=0.1):
        super().__init__()
        self.bias = bias
        self.hidden = hidden
        self.depth  = depth
        self.fourier_dim   = fourier_dim
        self.fourier_sigma = fourier_sigma
        # Fixed random Fourier projection matrix (not trained).
        B = torch.randn(input_num, fourier_dim) * fourier_sigma
        self.register_buffer('B', B)

        in_dim = 2 * fourier_dim
        layers = [nn.Linear(in_dim, hidden, bias=bias)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden, bias=bias))
        self.hidden_layers = nn.ModuleList(layers)
        self.out_layer     = nn.Linear(hidden, output_num, bias=bias)
        self.reset_parameters()
        self.kp = 0.01
        self.km = 0.1
        self.ph_r = body_length
        self.boundary = 0.0
        # Output scaling on softplus(out). Baseline kp/km = 0.01/0.1 = 0.1.
        # For km sweeps, pass kp/km so the softplus output stays ~ldf-scale.
        self.output_scale = output_scale

    def forward(self, x):
        self.input = x
        x_t  = (x[:, 0:1] - 50) / 50
        x_xy = x[:, 1:] / 0.28
        u = torch.cat((x_t, x_xy), dim=1)             # (N, 3) normalized
        proj = (2 * math.pi) * (u @ self.B)           # (N, fourier_dim)
        feat = torch.cat([torch.sin(proj),
                          torch.cos(proj)], dim=1)    # (N, 2·fourier_dim)
        h = feat
        for lin in self.hidden_layers:
            h = torch.tanh(lin(h))
        # getattr fallback keeps backward-compat for models pickled before
        # self.output_scale existed (they default to 0.1, the old hard-coded
        # value).
        return F.softplus(self.out_layer(h)) * getattr(self, 'output_scale', 0.1)

    def reset_parameters(self):
        for lin in self.hidden_layers:
            nn.init.xavier_uniform_(lin.weight,
                                    gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.out_layer.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.bias:
            for lin in list(self.hidden_layers) + [self.out_layer]:
                nn.init.constant_(lin.bias, 0.0)


class PINN_PDE_old(nn.Module):
    """Legacy c(t, x, y) network — 3-hidden tanh MLP, no Fourier embedding.
       Kept only for reference. Replaced by PINN_PDE (Fourier) which fits
       the body-source indicator much more efficiently.
       ph_r = body_length, km = 0.1, kp = 0.01  (v2 physics).
    """
    def __init__(self, input_num, output_num, bias=True, body_length=0.01,
                 hidden=32):
        super().__init__()
        self.bias = bias
        self.hidden = hidden
        self.fc1 = nn.Linear(input_num, hidden, bias=bias)
        self.fc2 = nn.Linear(hidden, hidden, bias=bias)
        self.fc3 = nn.Linear(hidden, hidden, bias=bias)
        self.fc4 = nn.Linear(hidden, output_num, bias=bias)
        self.reset_parameters()
        self.kp = 0.01
        self.km = 0.1
        self.ph_r = body_length
        self.boundary = 0.0

    def forward(self, x):
        self.input = x
        x_t  = (x[:, 0:1] - 50) / 50
        x_xy = x[:, 1:] / 0.28
        x = torch.cat((x_t, x_xy), dim=1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return F.softplus(self.fc4(x)) * 0.1

    def reset_parameters(self):
        for lin in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_uniform_(lin.weight,
                                    gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.fc4.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.bias:
            for lin in (self.fc1, self.fc2, self.fc3, self.fc4):
                nn.init.constant_(lin.bias, 0.0)


class Mechanism_NN(nn.Module):
    """Perception-response G(∇⊥c).  32-16-out, sigmoid-sigmoid-linear,
       pre-scale 5 (doubled from original 8-8 — gives cleaner G shape
       and stronger COR_tan/slope0 with the Fourier PINN_PDE)."""
    def __init__(self, input_num, output_num, bias=True):
        super().__init__()
        self.bias = bias
        self.fc1 = nn.Linear(input_num, 32, bias=bias)
        self.fc2 = nn.Linear(32, 16, bias=bias)
        self.fc3 = nn.Linear(16, output_num, bias=bias)
        self.reset_parameters()

    def forward(self, x):
        self.input = x
        x = self.fc1(x * 5)
        x = torch.sigmoid(x)
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight,
                                gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.fc2.weight,
                                gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.fc3.weight, gain=1)
        if self.bias:
            for lin in (self.fc1, self.fc2, self.fc3):
                nn.init.constant_(lin.bias, 0.0)


class Mechanism_NN_LS(nn.Module):
    """Mechanism_NN with CONFIGURABLE pre-scale and hidden widths.

    Same architecture as Mechanism_NN (sigmoid-sigmoid-linear) but with
    three things made configurable at __init__:
      pre_scale (default 5.0) — input multiplier, stored as buffer (not
                                trainable but saved with the model).
      hidden1   (default 32)  — fc1 output / fc2 input width.
      hidden2   (default 16)  — fc2 output / fc3 input width.

    `pre_scale` can also be overwritten later, e.g.
        model.pre_scale.fill_(3.0)
    """
    def __init__(self, input_num, output_num, bias=True,
                 init_scale=5.0, hidden1=32, hidden2=16):
        super().__init__()
        self.bias = bias
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.register_buffer(
            'pre_scale', torch.tensor(float(init_scale)))
        self.fc1 = nn.Linear(input_num, hidden1, bias=bias)
        self.fc2 = nn.Linear(hidden1,   hidden2, bias=bias)
        self.fc3 = nn.Linear(hidden2,   output_num, bias=bias)
        self.reset_parameters()

    def forward(self, x):
        self.input = x
        x = self.fc1(x * self.pre_scale)
        x = torch.sigmoid(x)
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight,
                                gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.fc2.weight,
                                gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.fc3.weight, gain=1)
        if self.bias:
            for lin in (self.fc1, self.fc2, self.fc3):
                nn.init.constant_(lin.bias, 0.0)


class Mechanism_NN_LS_Tanh(nn.Module):
    """Mechanism_NN_LS with tanh activations instead of sigmoid.

    Identical architecture to Mechanism_NN_LS (configurable pre_scale,
    hidden1, hidden2; fc3 output stays linear) but the two hidden-layer
    nonlinearities are tanh instead of sigmoid.  Xavier init gain
    switches to 'tanh' to match.

    Because tanh is symmetric around 0 (range (-1, 1)) it saturates at
    smaller |x| than sigmoid; pre_scale values that worked well for the
    sigmoid version may be slightly too large here.
    """
    def __init__(self, input_num, output_num, bias=True,
                 init_scale=5.0, hidden1=32, hidden2=16):
        super().__init__()
        self.bias = bias
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.register_buffer(
            'pre_scale', torch.tensor(float(init_scale)))
        self.fc1 = nn.Linear(input_num, hidden1, bias=bias)
        self.fc2 = nn.Linear(hidden1,   hidden2, bias=bias)
        self.fc3 = nn.Linear(hidden2,   output_num, bias=bias)
        self.reset_parameters()

    def forward(self, x):
        self.input = x
        x = self.fc1(x * self.pre_scale)
        x = torch.tanh(x)
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight,
                                gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.fc2.weight,
                                gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.fc3.weight, gain=1)
        if self.bias:
            for lin in (self.fc1, self.fc2, self.fc3):
                nn.init.constant_(lin.bias, 0.0)


class Mechanism_NN_LS_LogStretch(nn.Module):
    """Mechanism_NN_LS with asinh log-stretch pre-transform.

    Forward computes
        f(x) = asinh(alpha * x)
        G(x) = NN( f(x) * pre_scale )

    Unlike a linear pre_scale (Mechanism_NN_LS), the log-stretch
    genuinely changes RELATIVE spacing between input points:

        alpha=10, x ∈ {0.01, 0.1, 1.0}
          f(x)  ≈    {0.10, 0.88, 2.998}     (gap 0.78, 2.12)
        linear:    {0.01, 0.10, 1.00}      (gap 0.09, 0.90 — ratio 1:10)

    Near 0:   f(x) ≈ alpha * x  (slope alpha → center expansion).
    Large |x|: f(x) ≈ sign(x) * log(2*alpha*|x|)  (tail compression).
    asinh is C^∞ smooth at 0 (no gradient blow-up; no sign() needed).

    Use when ∇⊥c is concentrated near zero and you need finer G
    resolution there than a linear pre_scale can give. Higher alpha →
    stronger center expansion; alpha → 0 recovers the linear behavior
    of Mechanism_NN_LS (scaled by pre_scale * alpha).

    init_scale  — pre_scale (multiplier applied to f(x)).
    alpha       — log-stretch strength (stored as buffer).
    hidden1/2   — same as Mechanism_NN_LS.
    """
    def __init__(self, input_num, output_num, bias=True,
                 init_scale=10.0, hidden1=32, hidden2=16,
                 alpha=10.0):
        super().__init__()
        self.bias = bias
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.register_buffer(
            'pre_scale', torch.tensor(float(init_scale)))
        self.register_buffer(
            'alpha',     torch.tensor(float(alpha)))
        self.fc1 = nn.Linear(input_num, hidden1, bias=bias)
        self.fc2 = nn.Linear(hidden1,   hidden2, bias=bias)
        self.fc3 = nn.Linear(hidden2,   output_num, bias=bias)
        self.reset_parameters()

    def forward(self, x):
        self.input = x
        f = torch.asinh(self.alpha * x)
        x = self.fc1(f * self.pre_scale)
        x = torch.sigmoid(x)
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight,
                                gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.fc2.weight,
                                gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.fc3.weight, gain=1)
        if self.bias:
            for lin in (self.fc1, self.fc2, self.fc3):
                nn.init.constant_(lin.bias, 0.0)


class Mechanism_NN_LS_MultiScale(nn.Module):
    """Mechanism_NN_LS with multiple FIXED pre_scales fed in parallel.

    Instead of  fc1(x * pre_scale)  (single scale),
    forwards    fc1([x*s_1, x*s_2, ..., x*s_K])  (concat of K channels).
    fc1 learns to combine the channels per segment via its weights,
    so the same model handles narrow (∇⊥c near 0) and wide (long-
    tail) distributions without per-segment pre_scale tuning.

    Motivation: optimal single pre_scale depends on segment ∇⊥c
    distribution (L: ~5, M: ~10-25, S: ~100). Trying to learn a
    single trainable pre_scale degrades — the gradient landscape is
    rough. Multi-scale fixes the scales (as buffers) and lets the
    LINEAR fc1 weights pick the right mix, which is a well-behaved
    optimization problem. The largest scale always provides 0-near
    high resolution, so the primary objective (sharp G transition
    near 0) is guaranteed regardless of data distribution width.

    Same downstream architecture: fc1 → sigmoid → fc2 → sigmoid →
    fc3 (sigmoid-sigmoid-linear). hidden1/hidden2 widths same as LS.

    scales      — tuple of fixed pre_scales (default (5, 25, 100)).
                  Stored as a non-trainable buffer.
    hidden1/2   — same as Mechanism_NN_LS.
    """
    def __init__(self, input_num, output_num, bias=True,
                 scales=(5.0, 25.0, 100.0),
                 hidden1=32, hidden2=16):
        super().__init__()
        self.bias = bias
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.input_num = input_num
        self.register_buffer(
            'scales', torch.tensor([float(s) for s in scales]))
        in_dim = input_num * len(scales)
        self.fc1 = nn.Linear(in_dim,  hidden1, bias=bias)
        self.fc2 = nn.Linear(hidden1, hidden2, bias=bias)
        self.fc3 = nn.Linear(hidden2, output_num, bias=bias)
        self.reset_parameters()

    def forward(self, x):
        # x: (N, input_num) → broadcast to (N, input_num, K) → flatten
        self.input = x
        x_scaled = x.unsqueeze(-1) * self.scales       # (N, input_num, K)
        x_flat = x_scaled.reshape(x.shape[0], -1)      # (N, input_num*K)
        h = self.fc1(x_flat)
        h = torch.sigmoid(h)
        h = torch.sigmoid(self.fc2(h))
        return self.fc3(h)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight,
                                gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.fc2.weight,
                                gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.fc3.weight, gain=1)
        if self.bias:
            for lin in (self.fc1, self.fc2, self.fc3):
                nn.init.constant_(lin.bias, 0.0)
