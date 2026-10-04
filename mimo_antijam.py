#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5G NR Massive MIMO — RL Beamforming & Anti-Jamming Simulator
=============================================================
* gNB with a 64-element Uniform Linear Array (d = λ/2), n78 band (3.5 GHz)
* 2 mobile legitimate UEs + 1 mobile rogue RF jammer (optionally pursuing UEs)
* Uplink receive beamforming: one complex weight vector w_k ∈ C^64 per UE

RL agent (PyTorch, CUDA if available)
  - Actor network: state (DOA estimates + received powers) -> complex antenna
    weights per UE, parameterised in the signal subspace spanned by the source
    steering vectors (where the SINR-optimal solution provably lives in LoS).
  - Reward: sum-rate  R = Σ_k log2(1 + SINR_k)
  - Training: deterministic policy gradient through a differentiable channel /
    SINR model (model-based RL), experience replay, domain-randomised
    exploration states and Gaussian action noise.
  - Optional model-based planner: each frame the actor's action is refined by
    L-BFGS on the differentiable reward, warm-started from the policy and the
    previous action (policy-guided planning) -> deep jammer nulls.
  - Benchmarks: MVDR/Capon and conventional (matched-filter) beamforming using
    the SAME noisy DOA/power estimates.

Install : pip install pyqt5 pyqtgraph numpy torch
Run     : python mimo_antijam.py
"""

import sys
import math
import time
import collections

import numpy as np
import torch
import torch.nn as nn
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# =============================================================================
# System parameters
# =============================================================================
N_ANT = 64                      # ULA elements
FC_HZ = 3.5e9                   # n78 carrier
WAVELENGTH = 3e8 / FC_HZ        # ~8.57 cm, spacing d = λ/2
N_USERS = 2
N_SRC = N_USERS + 1             # sources = UEs + jammer
JAM = N_USERS                   # jammer index
NOISE_POW = 1.0                 # all powers are normalised to thermal noise
USER_SNR_REF_DB = 10.0          # per-element UE SNR at reference distance
REF_DIST_M = 50.0
JAMMER_SPEED = 18.0             # m/s
AREA_X = (-150.0, 150.0)        # metres (gNB at origin, array on x-axis,
AREA_Y = (15.0, 250.0)          #  serving the y>0 half-plane / 0..180°)
DOA_NOISE_DEG = 0.03            # DOA estimation error (std)
PWR_NOISE_DB = 0.3              # power estimation error (std)
POLAR_FLOOR_DB = -60.0
HIST_LEN = 600
FRAME_MS = 33
ANGLE_GRID = np.linspace(1e-3, np.pi - 1e-3, 1441)
COLORS = ['#00e5ff', '#76ff03', '#ff1744']
NAMES = ['UE-1', 'UE-2', 'JAMMER']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

N_IDX_NP = np.arange(N_ANT) - (N_ANT - 1) / 2.0


# =============================================================================
# DSP helpers (NumPy, float64) — used for ground-truth evaluation & plotting
# =============================================================================
def steering_np(phi):
    """ULA steering vectors a(φ) = exp(jπ n cos φ), φ measured from array axis."""
    phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    return np.exp(1j * np.pi * np.outer(np.cos(phi), N_IDX_NP))


def sinr_np(W, A, p):
    """W: (K,N) weights, A: (S,N) steering, p: (S,) rx powers -> (K,) SINR."""
    K, S = W.shape[0], A.shape[0]
    g = np.abs(W.conj() @ A.T) ** 2                 # (K,S) |w_k^H a_s|^2
    rx = g * p[None, :]
    eye = np.eye(K, S)
    sig = (rx * eye).sum(axis=1)
    interf = (rx * (1.0 - eye)).sum(axis=1)
    noise = NOISE_POW * np.sum(np.abs(W) ** 2, axis=1)
    return sig / (interf + noise + 1e-15)


def mvdr_weights(phi, p):
    """MVDR / Capon: w_k ∝ R^{-1} a_k with R built from estimated DOAs/powers."""
    A = steering_np(phi)
    R = NOISE_POW * np.eye(N_ANT, dtype=np.complex128) + (A.T * p) @ A.conj()
    W = np.linalg.solve(R, A[:N_USERS].T).T
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-15))


# =============================================================================
# DSP helpers (PyTorch, differentiable) — used by the RL agent
# =============================================================================
def abs2(z):
    return z.real ** 2 + z.imag ** 2


def steering_t(phi):
    """phi: (B,S) float -> (B,S,N) complex64 steering vectors."""
    n = torch.arange(N_ANT, device=phi.device, dtype=phi.dtype) - (N_ANT - 1) / 2.0
    ph = math.pi * torch.cos(phi).unsqueeze(-1) * n
    return torch.polar(torch.ones_like(ph), ph)


def state_features(A, logp):
    """Normalised Gram-matrix cross terms (spatial correlation) + log powers."""
    G = torch.einsum('bin,bjn->bij', A.conj(), A) / N_ANT
    g = torch.stack([G[:, 0, 1], G[:, 0, 2], G[:, 1, 2]], dim=-1)
    return torch.cat([g.real, g.imag, (logp - 2.0) / 3.0], dim=-1)   # (B, 9)


def weights_from_coeffs(c, A):
    """c: (B,K,S) complex subspace coefficients -> unit-norm W: (B,K,N)."""
    W = torch.einsum('bks,bsn->bkn', c, A)
    return W / torch.sqrt(abs2(W).sum(-1, keepdim=True) + 1e-12)


def sinr_t(W, A, p):
    K, S = W.shape[1], A.shape[1]
    g = abs2(torch.einsum('bkn,bsn->bks', W.conj(), A))   # (B,K,S)
    rx = g * p.unsqueeze(1)
    eye = torch.eye(K, S, device=W.device, dtype=rx.dtype)
    sig = (rx * eye).sum(-1)
    interf = (rx * (1.0 - eye)).sum(-1)
    noise = NOISE_POW * abs2(W).sum(-1)
    return sig / (interf + noise + 1e-12)


def sum_rate_t(c, A, logp):
    W = weights_from_coeffs(c, A)
    return torch.log2(1.0 + sinr_t(W, A, torch.pow(10.0, logp))).sum(-1)


# =============================================================================
# RF environment: 2 UEs + 1 rogue jammer moving in 2D
# =============================================================================
class RFEnvironment:
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        self.pos = np.array([[-90.0, 110.0], [80.0, 80.0], [20.0, 200.0]])
        self.heading = self.rng.uniform(0, 2 * np.pi, N_SRC)
        self.user_speed = 25.0
        self.jammer_power_db = 35.0
        self.pursuit = True
        self.jam_target = 0
        self.retarget_timer = 0.0

    def step(self, dt):
        # Random-walk mobility
        self.heading += self.rng.normal(0.0, 0.7 * np.sqrt(dt), N_SRC)

        # Smart jammer: chase a UE, orbit it once close (stand-off ~60 m)
        if self.pursuit:
            self.retarget_timer -= dt
            if self.retarget_timer <= 0:
                self.jam_target = int(self.rng.integers(0, N_USERS))
                self.retarget_timer = self.rng.uniform(6.0, 12.0)
            d = self.pos[self.jam_target] - self.pos[JAM]
            desired = math.atan2(d[1], d[0])
            if np.hypot(d[0], d[1]) < 60.0:
                desired += np.pi / 2
            err = (desired - self.heading[JAM] + np.pi) % (2 * np.pi) - np.pi
            self.heading[JAM] += 1.5 * err * dt

        speeds = np.array([self.user_speed] * N_USERS + [JAMMER_SPEED])
        vel = speeds[:, None] * np.stack([np.cos(self.heading), np.sin(self.heading)], axis=1)
        self.pos += vel * dt

        # Reflect at area boundaries
        for i in range(N_SRC):
            if not (AREA_X[0] <= self.pos[i, 0] <= AREA_X[1]):
                self.heading[i] = np.pi - self.heading[i]
                self.pos[i, 0] = np.clip(self.pos[i, 0], *AREA_X)
            if not (AREA_Y[0] <= self.pos[i, 1] <= AREA_Y[1]):
                self.heading[i] = -self.heading[i]
                self.pos[i, 1] = np.clip(self.pos[i, 1], *AREA_Y)
        self.heading %= 2 * np.pi

    def angles(self):
        return np.arctan2(self.pos[:, 1], self.pos[:, 0])

    def ranges(self):
        return np.hypot(self.pos[:, 0], self.pos[:, 1])

    def rx_powers(self):
        """Per-element received power (noise-normalised), free-space 1/d^2."""
        d = self.ranges()
        p = np.empty(N_SRC)
        p[:N_USERS] = 10 ** (USER_SNR_REF_DB / 10) * (REF_DIST_M / d[:N_USERS]) ** 2
        p[JAM] = 10 ** (self.jammer_power_db / 10) * (REF_DIST_M / d[JAM]) ** 2
        return p

    def observe(self):
        """Noisy DOA + power estimates (what the gNB actually measures)."""
        phi = self.angles() + self.rng.normal(0, np.deg2rad(DOA_NOISE_DEG), N_SRC)
        logp = np.log10(self.rx_powers()) + self.rng.normal(0, PWR_NOISE_DB / 10, N_SRC)
        return (np.clip(phi, 0.01, np.pi - 0.01).astype(np.float32),
                logp.astype(np.float32))


# =============================================================================
# RL agent
# =============================================================================
class BeamformingActor(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(9, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, N_USERS * N_SRC * 2),
        )
        # Start from conventional matched-filter beams (c_k = e_k)
        with torch.no_grad():
            self.body[-1].weight.mul_(0.1)
            b = torch.zeros(N_USERS, N_SRC, 2)
            for k in range(N_USERS):
                b[k, k, 0] = 1.0
            self.body[-1].bias.copy_(b.flatten())

    def forward(self, feats):
        out = self.body(feats).view(-1, N_USERS, N_SRC, 2)
        return torch.view_as_complex(out.contiguous())          # (B,K,S)


class RLBeamformingAgent:
    def __init__(self, lr=1e-3, batch=256, buffer_size=50000):
        self.actor = BeamformingActor().to(DEVICE)
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.batch = batch
        self.buf_phi = np.zeros((buffer_size, N_SRC), np.float32)
        self.buf_logp = np.zeros((buffer_size, N_SRC), np.float32)
        self.buf_ptr = 0
        self.buf_size = 0
        self.explore_std = 0.03
        self.updates = 0
        self.reward_ema = float('nan')
        self.prev_c = None
        self.rng = np.random.default_rng(1)

    # ---------------- experience replay ----------------
    def remember(self, phi, logp):
        i = self.buf_ptr
        self.buf_phi[i], self.buf_logp[i] = phi, logp
        self.buf_ptr = (i + 1) % len(self.buf_phi)
        self.buf_size = min(self.buf_size + 1, len(self.buf_phi))

    def _sample_batch(self, cur_phi, cur_logp):
        B, rng = self.batch, self.rng
        n_cur = B // 4
        n_rep = min(self.buf_size, B // 4)
        n_syn = B - n_cur - n_rep
        phis = [cur_phi + rng.normal(0, np.deg2rad(0.5), (n_cur, N_SRC))]
        logps = [cur_logp + rng.normal(0, 0.1, (n_cur, N_SRC))]
        if n_rep:
            idx = rng.integers(0, self.buf_size, n_rep)
            phis.append(self.buf_phi[idx])
            logps.append(self.buf_logp[idx])
        # Domain-randomised exploration states
        syn_phi = rng.uniform(0.05, np.pi - 0.05, (n_syn, N_SRC))
        syn_logp = np.empty((n_syn, N_SRC))
        syn_logp[:, :N_USERS] = rng.uniform(-0.5, 2.5, (n_syn, N_USERS))
        syn_logp[:, JAM] = rng.uniform(-0.5, 8.0, n_syn)
        phis.append(syn_phi)
        logps.append(syn_logp)
        phi = np.clip(np.concatenate(phis), 0.01, np.pi - 0.01)
        logp = np.concatenate(logps)
        return (torch.as_tensor(phi, dtype=torch.float32, device=DEVICE),
                torch.as_tensor(logp, dtype=torch.float32, device=DEVICE))

    # ---------------- policy improvement ----------------
    def train(self, n_updates, cur_phi, cur_logp):
        if n_updates <= 0:
            return
        last = None
        for _ in range(n_updates):
            phi, logp = self._sample_batch(cur_phi, cur_logp)
            A = steering_t(phi)
            c = self.actor(state_features(A, logp))
            noise = torch.randn(*c.shape, 2, device=DEVICE) * self.explore_std
            c = c + torch.view_as_complex(noise)                 # exploration
            r = sum_rate_t(c, A, logp)                           # reward
            loss = -r.mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.opt.step()
            self.updates += 1
            last = r.detach().mean()
        val = float(last.item())
        self.reward_ema = val if math.isnan(self.reward_ema) else 0.95 * self.reward_ema + 0.05 * val

    # ---------------- acting ----------------
    @torch.no_grad()
    def _policy_coeffs(self, phi, logp):
        phi_t = torch.as_tensor(phi[None], dtype=torch.float32, device=DEVICE)
        logp_t = torch.as_tensor(logp[None], dtype=torch.float32, device=DEVICE)
        return self.actor(state_features(steering_t(phi_t), logp_t)).cpu()

    def act(self, phi, logp, plan=True, plan_iters=12):
        """Returns (executed weights, raw actor weights), each (K,N) complex."""
        c_pol = self._policy_coeffs(phi, logp)
        phi_c = torch.as_tensor(phi[None], dtype=torch.float32)
        logp_c = torch.as_tensor(logp[None], dtype=torch.float32)
        A = steering_t(phi_c)
        with torch.no_grad():
            W_pol = weights_from_coeffs(c_pol, A)[0].numpy().astype(np.complex128)
        if not plan:
            self.prev_c = None
            return W_pol, W_pol

        # Warm start: better of (policy action, previous executed action)
        with torch.no_grad():
            init = c_pol
            if self.prev_c is not None:
                if sum_rate_t(self.prev_c, A, logp_c).item() > sum_rate_t(c_pol, A, logp_c).item():
                    init = self.prev_c

        x = torch.view_as_real(init).clone().requires_grad_(True)
        opt = torch.optim.LBFGS([x], lr=1.0, max_iter=plan_iters, history_size=10,
                                line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            loss = -sum_rate_t(torch.view_as_complex(x), A, logp_c).sum()
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except RuntimeError:
            pass

        with torch.no_grad():
            c = torch.view_as_complex(x.detach().clone())
            if not torch.isfinite(x).all():
                c = c_pol
            c = c / torch.sqrt(abs2(c).sum(-1, keepdim=True) + 1e-12)
            self.prev_c = c.clone()
            W = weights_from_coeffs(c, A)[0].numpy().astype(np.complex128)
        return W, W_pol


# =============================================================================
# GUI dashboard
# =============================================================================
class Dashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('5G Massive MIMO (64-ULA, 3.5 GHz) — RL Beamforming & Anti-Jamming')
        self.resize(1600, 960)

        self.env = RFEnvironment()
        self.agent = RLBeamformingAgent()
        self.A_grid = steering_np(ANGLE_GRID)
        self.wall_t = 0.0
        self.fps = 0.0
        self.hist = {k: collections.deque(maxlen=HIST_LEN)
                     for k in ('t', 'rl0', 'rl1', 'mv0', 'mv1')}
        self.trails = [collections.deque(maxlen=120) for _ in range(N_SRC)]

        self._build_ui()
        self.last_tick = time.perf_counter()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(FRAME_MS)

    # ---------------------------------------------------------------- UI build
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        plots = QtWidgets.QHBoxLayout()
        root.addLayout(plots, stretch=1)

        self.map_plot = pg.PlotWidget(title='Geo map (m) — gNB at origin, ULA along x-axis')
        self._setup_map()
        plots.addWidget(self.map_plot, 5)

        right = QtWidgets.QVBoxLayout()
        plots.addLayout(right, 4)
        self.polar_plot = pg.PlotWidget(title='Array radiation pattern |wᴴa(φ)|² (dB, normalised)')
        self._setup_polar()
        right.addWidget(self.polar_plot, 3)
        self.sinr_plot = pg.PlotWidget(title='Post-beamforming SINR (true geometry)')
        self._setup_sinr()
        right.addWidget(self.sinr_plot, 2)

        root.addWidget(self._build_controls())

    def _setup_map(self):
        mp = self.map_plot
        mp.setAspectLocked(True)
        mp.showGrid(x=True, y=True, alpha=0.15)
        mp.setXRange(AREA_X[0] - 10, AREA_X[1] + 10)
        mp.setYRange(-25, AREA_Y[1] + 10)
        mp.setLabel('bottom', 'x (m)')
        mp.setLabel('left', 'y (m)')
        rx = [AREA_X[0], AREA_X[1], AREA_X[1], AREA_X[0], AREA_X[0]]
        ry = [AREA_Y[0], AREA_Y[0], AREA_Y[1], AREA_Y[1], AREA_Y[0]]
        mp.plot(rx, ry, pen=pg.mkPen('#37474f', width=1, style=QtCore.Qt.DashLine))

        ax = np.linspace(-14, 14, 16)      # symbolic array (real aperture ≈ 2.7 m)
        mp.plot(ax, np.zeros_like(ax), pen=None, symbol='s', symbolSize=6,
                symbolBrush='#ffd54f', symbolPen=None)
        t = pg.TextItem('gNB · 64-element ULA', color='#ffd54f', anchor=(0.5, 0))
        t.setPos(0, -4)
        mp.addItem(t)

        self.ray_items, self.trail_items, self.ent_items, self.ent_labels = [], [], [], []
        for i in range(N_SRC):
            style = QtCore.Qt.DashLine if i == JAM else QtCore.Qt.SolidLine
            self.ray_items.append(mp.plot([], [], pen=pg.mkPen(COLORS[i], width=1, style=style)))
            c = QtGui.QColor(COLORS[i])
            c.setAlpha(110)
            self.trail_items.append(mp.plot([], [], pen=pg.mkPen(c, width=2)))
            sym = 'star' if i == JAM else 'o'
            size = 20 if i == JAM else 15
            self.ent_items.append(mp.plot([], [], pen=None, symbol=sym, symbolSize=size,
                                          symbolBrush=COLORS[i], symbolPen=pg.mkPen('w', width=1)))
            lbl = pg.TextItem(NAMES[i], color=COLORS[i], anchor=(-0.15, 1.1))
            mp.addItem(lbl)
            self.ent_labels.append(lbl)

    def _setup_polar(self):
        pp = self.polar_plot
        pp.setAspectLocked(True)
        pp.hideAxis('left')
        pp.hideAxis('bottom')
        pp.setXRange(-1.2, 1.2)
        pp.setYRange(-0.1, 1.2)
        pp.setMouseEnabled(False, False)
        grid_pen = pg.mkPen('#37474f', width=1, style=QtCore.Qt.DotLine)

        for d in range(0, int(POLAR_FLOOR_DB), -10):
            r = (d - POLAR_FLOOR_DB) / (-POLAR_FLOOR_DB)
            pp.plot(r * np.cos(ANGLE_GRID), r * np.sin(ANGLE_GRID), pen=grid_pen)
            lbl = pg.TextItem(f'{d}', color='#607d8b', anchor=(0.5, 0))
            lbl.setPos(r, -0.01)
            pp.addItem(lbl)
        for deg in range(0, 181, 30):
            a = np.deg2rad(deg)
            pp.plot([0, np.cos(a)], [0, np.sin(a)], pen=grid_pen)
            lbl = pg.TextItem(f'{deg}°', color='#90a4ae', anchor=(0.5, 0.5))
            lbl.setPos(1.1 * np.cos(a), 1.1 * np.sin(a))
            pp.addItem(lbl)
        pp.plot([-1, 1], [0, 0], pen=pg.mkPen('#ffd54f', width=2))

        pp.addLegend(offset=(10, 10))
        self.pat_curves = [pp.plot([], [], pen=pg.mkPen(COLORS[k], width=2),
                                   name=f'Beam → {NAMES[k]}') for k in range(N_USERS)]
        self.dir_lines, self.dir_marks = [], []
        for i in range(N_SRC):
            self.dir_lines.append(pp.plot([], [], pen=pg.mkPen(COLORS[i], width=1,
                                                               style=QtCore.Qt.DashLine)))
            self.dir_marks.append(pp.plot([], [], pen=None,
                                          symbol='star' if i == JAM else 'o',
                                          symbolSize=14 if i == JAM else 10,
                                          symbolBrush=COLORS[i], symbolPen=None))

    def _setup_sinr(self):
        sp = self.sinr_plot
        sp.showGrid(x=True, y=True, alpha=0.2)
        sp.setLabel('left', 'SINR (dB)')
        sp.setLabel('bottom', 'time (s)')
        sp.addLegend(offset=(10, 10))
        self.curve_rl = [sp.plot([], [], pen=pg.mkPen(COLORS[k], width=2),
                                 name=f'RL {NAMES[k]}') for k in range(N_USERS)]
        self.curve_mv = [sp.plot([], [], pen=pg.mkPen(COLORS[k], width=1, style=QtCore.Qt.DashLine),
                                 name=f'MVDR {NAMES[k]}') for k in range(N_USERS)]

    def _build_controls(self):
        box = QtWidgets.QGroupBox('Scenario & Agent Controls')
        g = QtWidgets.QGridLayout(box)

        self.lbl_jam = QtWidgets.QLabel()
        self.sld_jam = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_jam.setRange(0, 70)
        self.sld_jam.setValue(int(self.env.jammer_power_db))
        self.sld_jam.valueChanged.connect(self._on_jam)

        self.lbl_spd = QtWidgets.QLabel()
        self.sld_spd = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_spd.setRange(0, 120)
        self.sld_spd.setValue(int(self.env.user_speed))
        self.sld_spd.valueChanged.connect(self._on_spd)

        self.spin_upd = QtWidgets.QSpinBox()
        self.spin_upd.setRange(0, 64)
        self.spin_upd.setValue(6)

        self.chk_plan = QtWidgets.QCheckBox('Model-based planner refinement (L-BFGS)')
        self.chk_plan.setChecked(True)
        self.chk_pursuit = QtWidgets.QCheckBox('Smart jammer (pursues UEs)')
        self.chk_pursuit.setChecked(True)
        self.chk_pursuit.toggled.connect(lambda v: setattr(self.env, 'pursuit', v))
        self.chk_mvdr = QtWidgets.QCheckBox('Show MVDR benchmark')
        self.chk_mvdr.setChecked(True)
        self.chk_mvdr.toggled.connect(lambda v: [c.setVisible(v) for c in self.curve_mv])

        self.btn_freeze = QtWidgets.QPushButton('Freeze motion')
        self.btn_freeze.setCheckable(True)
        self.btn_burst = QtWidgets.QPushButton('Burst-train 2000 updates')
        self.btn_burst.clicked.connect(self._burst_train)
        self.btn_reset = QtWidgets.QPushButton('Reset RL agent')
        self.btn_reset.clicked.connect(self._reset_agent)

        g.addWidget(self.lbl_jam, 0, 0)
        g.addWidget(self.sld_jam, 0, 1)
        g.addWidget(self.lbl_spd, 1, 0)
        g.addWidget(self.sld_spd, 1, 1)
        g.addWidget(QtWidgets.QLabel('RL updates / frame'), 2, 0)
        g.addWidget(self.spin_upd, 2, 1)
        g.addWidget(self.chk_plan, 3, 0)
        g.addWidget(self.chk_pursuit, 3, 1)
        g.addWidget(self.chk_mvdr, 4, 0)
        g.addWidget(self.btn_freeze, 4, 1)
        g.addWidget(self.btn_burst, 5, 0)
        g.addWidget(self.btn_reset, 5, 1)

        self.lbl_stats = QtWidgets.QLabel()
        self.lbl_stats.setTextFormat(QtCore.Qt.RichText)
        self.lbl_stats.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.lbl_stats.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        g.addWidget(self.lbl_stats, 0, 2, 6, 1)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(2, 3)

        self._on_jam(self.sld_jam.value())
        self._on_spd(self.sld_spd.value())
        return box

    # ---------------------------------------------------------------- callbacks
    def _on_jam(self, v):
        self.env.jammer_power_db = float(v)
        self.lbl_jam.setText(f'Jammer power: {v} dB (per-element INR @ {REF_DIST_M:.0f} m)')

    def _on_spd(self, v):
        self.env.user_speed = float(v)
        self.lbl_spd.setText(f'UE speed: {v} m/s ({v * 3.6:.0f} km/h)')

    def _reset_agent(self):
        self.agent = RLBeamformingAgent()

    def _burst_train(self):
        self.timer.stop()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            phi, logp = self.env.observe()
            for _ in range(20):
                self.agent.train(100, phi, logp)
                QtWidgets.QApplication.processEvents()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.last_tick = time.perf_counter()
            self.timer.start(FRAME_MS)

    # ---------------------------------------------------------------- main loop
    def tick(self):
        now = time.perf_counter()
        dt = min(now - self.last_tick, 0.1)
        self.last_tick = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 / dt
        self.wall_t += dt
        if not self.btn_freeze.isChecked():
            self.env.step(dt)

        # Observe -> learn -> act
        obs_phi, obs_logp = self.env.observe()
        self.agent.remember(obs_phi, obs_logp)
        self.agent.train(self.spin_upd.value(), obs_phi, obs_logp)
        W_exec, W_pol = self.agent.act(obs_phi, obs_logp, plan=self.chk_plan.isChecked())

        # Ground-truth evaluation
        true_phi = self.env.angles()
        p_true = self.env.rx_powers()
        A_true = steering_np(true_phi)
        s_exec = sinr_np(W_exec, A_true, p_true)
        s_pol = sinr_np(W_pol, A_true, p_true)
        W_mv = mvdr_weights(obs_phi.astype(np.float64), 10 ** obs_logp.astype(np.float64))
        s_mv = sinr_np(W_mv, A_true, p_true)
        W_cb = steering_np(obs_phi.astype(np.float64))[:N_USERS] / np.sqrt(N_ANT)
        s_cb = sinr_np(W_cb, A_true, p_true)

        # Radiation pattern of executed weights
        af = np.abs(self.A_grid @ W_exec.conj().T) ** 2          # (G,K)
        af_max = af.max(axis=0)
        af_db = db(af / af_max[None, :])
        null_db = db(np.abs(W_exec.conj() @ A_true[JAM]) ** 2 / af_max)

        self._update_map()
        self._update_polar(af_db, true_phi)
        self._update_sinr(s_exec, s_mv)
        self._update_stats(true_phi, p_true, s_exec, s_pol, s_mv, s_cb, null_db)

    # ---------------------------------------------------------------- updates
    def _update_map(self):
        for i in range(N_SRC):
            x, y = self.env.pos[i]
            self.trails[i].append((x, y))
            tr = np.asarray(self.trails[i])
            self.trail_items[i].setData(tr[:, 0], tr[:, 1])
            self.ent_items[i].setData([x], [y])
            self.ent_labels[i].setPos(x, y)
            self.ray_items[i].setData([0, x], [0, y])

    def _update_polar(self, af_db, true_phi):
        cx, cy = np.cos(ANGLE_GRID), np.sin(ANGLE_GRID)
        for k in range(N_USERS):
            r = np.clip((af_db[:, k] - POLAR_FLOOR_DB) / (-POLAR_FLOOR_DB), 0.0, 1.0)
            self.pat_curves[k].setData(r * cx, r * cy)
        for i in range(N_SRC):
            ex, ey = 1.03 * np.cos(true_phi[i]), 1.03 * np.sin(true_phi[i])
            self.dir_lines[i].setData([0, ex], [0, ey])
            self.dir_marks[i].setData([ex], [ey])

    def _update_sinr(self, s_exec, s_mv):
        h = self.hist
        h['t'].append(self.wall_t)
        for k in range(N_USERS):
            h[f'rl{k}'].append(db(s_exec[k]))
            h[f'mv{k}'].append(db(s_mv[k]))
        t = np.asarray(h['t'])
        for k in range(N_USERS):
            self.curve_rl[k].setData(t, np.asarray(h[f'rl{k}']))
            self.curve_mv[k].setData(t, np.asarray(h[f'mv{k}']))

    def _update_stats(self, phi, p, s_exec, s_pol, s_mv, s_cb, null_db):
        dev = (f'CUDA · {torch.cuda.get_device_name(0)}' if DEVICE.type == 'cuda'
               else 'CPU (CUDA not available)')
        rate = lambda s: float(np.sum(np.log2(1 + s)))
        lines = [
            f"<b>Compute</b> {dev} | RL updates {self.agent.updates:,} | "
            f"train reward (EMA) {self.agent.reward_ema:5.2f} b/s/Hz | {self.fps:4.1f} FPS",
            "",
        ]
        for k in range(N_USERS):
            lines.append(
                f"<span style='color:{COLORS[k]}'><b>{NAMES[k]}</b></span>  "
                f"φ={np.degrees(phi[k]):6.1f}°  SINR: RL {db(s_exec[k]):6.1f} | "
                f"actor {db(s_pol[k]):6.1f} | MVDR {db(s_mv[k]):6.1f} | "
                f"CBF {db(s_cb[k]):6.1f} dB   null→J {null_db[k]:6.1f} dB")
        lines.append(
            f"<span style='color:{COLORS[JAM]}'><b>JAMMER</b></span> "
            f"φ={np.degrees(phi[JAM]):6.1f}°  per-element INR {db(p[JAM]):5.1f} dB  "
            f"range {self.env.ranges()[JAM]:5.0f} m  {'(pursuing ' + NAMES[self.env.jam_target] + ')' if self.env.pursuit else ''}")
        lines.append("")
        lines.append(
            f"<b>Sum-rate</b>  RL {rate(s_exec):5.2f} | actor {rate(s_pol):5.2f} | "
            f"MVDR {rate(s_mv):5.2f} | CBF {rate(s_cb):5.2f} b/s/Hz")
        self.lbl_stats.setText("<pre>" + "\n".join(lines) + "</pre>")


# =============================================================================
# Entry point
# =============================================================================
DARK_QSS = """
QWidget { background:#0b1016; color:#cfd8dc; font-size:12px; }
QGroupBox { border:1px solid #263238; border-radius:6px; margin-top:12px; padding-top:6px; }
QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 4px; color:#80cbc4; }
QPushButton { background:#1c2833; border:1px solid #37474f; padding:5px 10px; border-radius:4px; }
QPushButton:hover { background:#263849; }
QPushButton:checked { background:#b71c1c; }
QSlider::groove:horizontal { height:6px; background:#263238; border-radius:3px; }
QSlider::handle:horizontal { background:#00bcd4; width:14px; margin:-5px 0; border-radius:7px; }
QSpinBox { background:#1c2833; border:1px solid #37474f; padding:2px; }
"""


def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    pg.setConfigOptions(antialias=True, background='#0b1016', foreground='#cfd8dc')
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(DARK_QSS)
    win = Dashboard()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
