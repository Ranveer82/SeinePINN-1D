"""
The hybrid LSTM-PINN.

The network maps a place ``x``, a time ``t`` and the boundary forcing to water
level and discharge (or velocity):

* an LSTM reads the hourly forcing of the window (upstream discharge,
  downstream stage, tributaries) and gives a latent state, interpolated in
  time with a smooth spline;
* a sub-tidal MLP gives the slowly varying part, from slow harmonics
  (annual ... weekly) and the context;
* a tidal MLP gives the amplitudes ``a_k, b_k`` of each tidal constituent,
  which vary along the channel and with the forcing:
  ``sum_k a_k cos(w_k t) + b_k sin(w_k t)``;
* an optional residual MLP, starting at zero, catches what the harmonics miss;
* the downstream water level is imposed exactly:
  ``eta = eta_down(t) + (1 - s) * H * N(x, t)`` with ``s = (x - x0) / L``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .tides import HarmonicEmbedding, resolve_constituents, periods_hours
from .utils import catmull_rom, interp1d, soft_depth

ACTIVATIONS = {"Tanh": nn.Tanh, "SiLU": nn.SiLU, "GELU": nn.GELU, "Sin": None}


def _mlp(d_in, d_out, hidden, layers, act, zero_last=False):
    """
    A plain multilayer perceptron.

    Args:
        d_in (int): Number of inputs.
        d_out (int): Number of outputs.
        hidden (int): Width of the hidden layers.
        layers (int): Number of layers (hidden layers + output layer).
        act (str): Activation name.
        zero_last (bool): Start the output layer at zero.

    Returns:
        nn.Sequential: The network.
    """
    Act = ACTIVATIONS.get(act) or nn.SiLU
    mods, d = [], d_in
    for _ in range(max(layers - 1, 1)):
        mods += [nn.Linear(d, hidden), Act()]
        d = hidden
    last = nn.Linear(d, d_out)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    mods.append(last)
    return nn.Sequential(*mods)


@dataclass
class Context:
    """What all points of one window share: the window and its LSTM latent."""

    window: object                # seinepinn_1d.data.Window
    latent: torch.Tensor          # (T, C) LSTM output (inside the graph)


class RiverPINN(nn.Module):
    """The physics-informed network of one river."""

    def __init__(self, cfg, scales, geometry, surrogate, dataset_stats):
        """
        Args:
            cfg (CaseConfig): The case.
            scales (Scales): Characteristic scales.
            geometry (RiverGeometry): Tabulated geometry (bed level).
            surrogate (GeometrySurrogate): Fitted geometry surrogate.
            dataset_stats (dict): Means and standard deviations of the forcing,
                used to normalise the instantaneous forcing features.
        """
        super().__init__()
        self.cfg = cfg
        self.scales = scales
        self.geometry = geometry
        self.surrogate = surrogate

        mc, pc = cfg.model, cfg.physics
        self.x_min = float(geometry.x_min)
        self.x_len = float(geometry.x_max - geometry.x_min)
        self.depth_eps = max(pc.min_depth / 4.0, 1e-3)

        # normalisation of the forcing features
        st = dataset_stats
        for k in ("Q_up_mean", "Q_up_std", "eta_mean", "eta_std",
                  "Q_trib_mean", "Q_trib_std"):
            self.register_buffer(f"_{k}", torch.tensor(float(st[k])))

        # LSTM encoder of the forcing
        self.lstm = nn.LSTM(input_size=3, hidden_size=mc.lstm_hidden,
                            num_layers=mc.lstm_layers, batch_first=True)
        self._lstm_no_cudnn = False
        self._lstm_f32 = bool(getattr(mc, "lstm_float32", False))
        if self._lstm_f32:
            self.lstm.to(torch.float32)
        self.latent_dim = mc.lstm_hidden

        # harmonic embeddings
        self.constituents = resolve_constituents(pc.constituents, pc.tidal)
        self.harm_tide = HarmonicEmbedding(periods_hours(self.constituents))
        self.harm_sub = HarmonicEmbedding(np.asarray(pc.subtidal_periods_h))
        self.n_const = self.harm_tide.n_periods

        ctx_dim = 1 + self.latent_dim + 3          # position, latent, forcing

        # sub-tidal branch: context and slow harmonics
        self.net_sub = _mlp(ctx_dim + self.harm_sub.out_dim, 2,
                            mc.hidden_dim, mc.n_layers, mc.activation)

        # Tidal branch: amplitudes of each constituent for stage and for the
        # second output. It does not see the tidal phase itself.
        if self.n_const:
            self.net_amp = _mlp(ctx_dim, 4 * self.n_const,
                                mc.hidden_dim, mc.n_layers, mc.activation)
            with torch.no_grad():
                self.net_amp[-1].weight.mul_(0.1)
                self.net_amp[-1].bias.mul_(0.0)
        else:
            self.net_amp = None

        # free residual branch, starting at zero
        self.net_res = (
            _mlp(ctx_dim + self.harm_sub.out_dim + self.harm_tide.out_dim, 2,
                 mc.hidden_dim, max(mc.n_layers - 1, 2), mc.activation,
                 zero_last=True)
            if mc.residual_branch else None
        )

        # Manning roughness n(x): control points between the bounds
        nodes = torch.linspace(geometry.x_min, geometry.x_max, pc.manning_nodes,
                               dtype=torch.float64)
        self.register_buffer("n_nodes", nodes)
        self.n_raw = nn.Parameter(torch.zeros(pc.manning_nodes))
        self.n_min, self.n_max = pc.manning_min, pc.manning_max

        # optional learnable width correction (starts at 1, the survey)
        self.geo_tol = float(getattr(pc, "geometry_tolerance", 0.0))
        if self.geo_tol > 0.0:
            gn = torch.linspace(geometry.x_min, geometry.x_max,
                                int(getattr(pc, "geometry_nodes", 16)),
                                dtype=torch.float64)
            self.register_buffer("geo_nodes", gn)
            self.geo_raw = nn.Parameter(torch.zeros(gn.numel()))
            self.geo_log_tol = float(np.log1p(self.geo_tol))

    # ------------------------------------------------------------------ pieces
    def manning_grid(self) -> torch.Tensor:
        """Manning n at the control points."""
        return self.n_min + (self.n_max - self.n_min) * torch.sigmoid(self.n_raw)

    def manning(self, x: torch.Tensor):
        """
        Manning n along the reach.

        Returns:
            tuple: ``(n at x, n at the control points)``.
        """
        grid = self.manning_grid()
        return interp1d(x, self.n_nodes, grid), grid

    def width_grid(self) -> torch.Tensor:
        """Width correction factor at its control points (1 = the survey)."""
        return torch.exp(self.geo_log_tol * torch.tanh(self.geo_raw))

    def width_factor(self, x: torch.Tensor):
        """
        Learned width correction along the reach, bounded by ``geometry_tolerance``.

        Returns:
            tuple: ``(factor at x, factor at the control points)``, or
            ``(None, None)`` when the correction is off.
        """
        if self.geo_tol <= 0.0:
            return None, None
        grid = self.width_grid()
        return interp1d(x, self.geo_nodes, grid), grid

    def encode(self, window) -> torch.Tensor:
        """
        Run the LSTM over the forcing of a window.

        Args:
            window (Window): The window.

        Returns:
            torch.Tensor: Latent state at each hour, ``(T, lstm_hidden)``.
        """
        # restoring a state dict on the GPU breaks the flat weight buffer of
        # the fused LSTM; re-flattening costs nothing when nothing moved
        if self.lstm.weight_ih_l0.is_cuda:
            self.lstm.flatten_parameters()
        out, _ = self._lstm_forward(window.lstm_input)
        return out.squeeze(0)

    def _lstm_forward(self, seq):
        """
        Run the LSTM, falling back to the slower generic kernel if the GPU
        library refuses the fused one (this happens in double precision on
        some GPU and driver combinations). ``model.lstm_float32`` avoids it.
        """
        if self._lstm_f32:
            out, hc = self.lstm(seq.to(torch.float32))
            return out.to(seq.dtype), hc
        if not self._lstm_no_cudnn:
            try:
                return self.lstm(seq)
            except RuntimeError as exc:
                if "CUDNN" not in str(exc).upper():
                    raise
                self._lstm_no_cudnn = True
                t0 = time.perf_counter()
                with torch.backends.cudnn.flags(enabled=False):
                    out = self.lstm(seq)
                if seq.is_cuda:
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1e3
                print(f"\n  WARNING: cuDNN refused the LSTM kernel "
                      f"({str(exc).splitlines()[0]}). Using the generic kernel "
                      f"({ms:.0f} ms per pass over {seq.shape[1]} hours), which "
                      f"is much slower. Set `model.lstm_float32: true` to avoid "
                      f"this.\n", flush=True)
                return out
        with torch.backends.cudnn.flags(enabled=False):
            return self.lstm(seq)

    def context(self, window) -> Context:
        """Encode a window once; all its points then share the result."""
        return Context(window=window, latent=self.encode(window))

    def _forcing_features(self, forcing):
        """Normalised instantaneous forcing: upstream Q, downstream stage, tributaries."""
        q = (forcing["Q_up"] - self._Q_up_mean) / self._Q_up_std
        e = (forcing["eta_down"] - self._eta_mean) / self._eta_std
        tr = forcing["Q_trib"]
        tr = (tr.sum(dim=1, keepdim=True) - self._Q_trib_mean) / self._Q_trib_std \
            if tr.shape[1] else torch.zeros_like(q)
        return torch.cat([q, e, tr], dim=1)

    def _latent_at(self, ctx: Context, t: torch.Tensor) -> torch.Tensor:
        """LSTM latent at any time, by Catmull-Rom interpolation of the hourly latent."""
        pos = (t - ctx.window.t_hourly[0]) / 3600.0
        pos = pos.clamp(0.0, float(ctx.latent.shape[0] - 1))
        return catmull_rom(ctx.latent, pos)

    # ----------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, t: torch.Tensor, ctx: Context,
                want_geometry: bool = True, create_graph: bool = True):
        """
        Water level, velocity, discharge and geometry at points ``(x, t)``.

        Args:
            x (torch.Tensor): Chainage (m), ``(N, 1)``.
            t (torch.Tensor): Time (s since the case epoch), ``(N, 1)``.
            ctx (Context): Encoded window.
            want_geometry (bool): Also return ``A``, ``B``, ``P``, ``Q``, ``R``
                and ``n``. Always done when the network outputs discharge,
                because the velocity then needs the area.
            create_graph (bool): Keep the graph of ``B = dA/d(eta)``.

        Returns:
            dict: Fields of shape ``(N, 1)``: ``eta`` (m), ``u`` (m/s),
            ``depth`` (m), ``z_bed`` (m) and, with geometry, ``A`` (m2),
            ``B`` (m), ``P`` (m), ``Q`` (m3/s), ``R`` (m) and ``n``.
        """
        if self.cfg.model.discharge_output:
            want_geometry = True

        s = (x - self.x_min) / self.x_len
        forcing = ctx.window.forcing(t)
        f_feat = self._forcing_features(forcing)
        latent = self._latent_at(ctx, t)

        ctx_in = torch.cat([2.0 * s - 1.0, latent, f_feat], dim=1)
        sub_feat = self.harm_sub(t)

        out = self.net_sub(torch.cat([ctx_in, sub_feat], dim=1))
        eta_sub, u_sub = out[:, 0:1], out[:, 1:2]
        eta_hat, u_hat = eta_sub, u_sub

        tide_feat = self.harm_tide(t)
        if self.net_amp is not None:
            K = self.n_const
            amp = self.net_amp(ctx_in).view(-1, 4, K)
            cos_t, sin_t = tide_feat[:, :K], tide_feat[:, K:]
            eta_tide = (amp[:, 0] * cos_t + amp[:, 1] * sin_t).sum(1, keepdim=True)
            u_tide = (amp[:, 2] * cos_t + amp[:, 3] * sin_t).sum(1, keepdim=True)
            eta_hat = eta_hat + eta_tide
            u_hat = u_hat + u_tide
        else:
            eta_tide = u_tide = torch.zeros_like(eta_sub)

        if self.net_res is not None:
            res = self.net_res(torch.cat([ctx_in, sub_feat, tide_feat], dim=1))
            eta_hat = eta_hat + res[:, 0:1]
            u_hat = u_hat + res[:, 1:2]

        # hard downstream condition: eta(x_L, t) = eta_down(t) exactly
        eta_raw = forcing["eta_down"] + (1.0 - s) * self.scales.H * eta_hat

        z_bed = self.geometry.bed_elevation(x)
        depth = soft_depth(eta_raw - z_bed, self.depth_eps)
        eta = z_bed + depth

        # Velocity, bounded by a tanh at four times the plausible maximum so
        # it can never run away (the plausible range is enforced by a penalty
        # in the loss; inside it the tanh is practically linear).
        u_cap = 4.0 * self.scales.u_max
        u = u_cap * torch.tanh((self.scales.U / u_cap) * u_hat)

        fields = {"eta": eta, "u": u, "depth": depth, "z_bed": z_bed,
                  "eta_raw": eta_raw, "s": s,
                  "eta_sub": eta_sub, "eta_tide": eta_tide, "forcing": forcing}

        if want_geometry:
            A, P, B = self.surrogate.with_width(x, eta, create_graph=create_graph)
            # optional width correction: A, B and P take the same factor, so
            # B = dA/d(eta) still holds and R = A/P is unchanged
            g, g_grid = self.width_factor(x)
            if g is not None:
                A, B, P = A * g, B * g, P * g
                fields["width_factor"], fields["width_grid"] = g, g_grid
            # Storage/conveyance split: A (all wetted area) stores water and is
            # used in continuity; A_c (the conveying part) carries discharge
            # and momentum. Without the split, A_c is A.
            phi, beta = self.surrogate.storage_split(x, eta)
            A_c = A if phi is None else A * phi
            if self.cfg.model.discharge_output:
                # The second output is the discharge, and u = Q / A_c. Then
                # dQ/dx in continuity is the derivative of one smooth output,
                # and the roughness of the geometry goes into u, which is
                # never differentiated.
                Q_cap = 4.0 * self.scales.Q
                Q = Q_cap * torch.tanh((self.scales.Q / Q_cap) * u_hat)
                u = Q / A_c.clamp_min(1.0)
                fields["u"] = u
            else:
                Q = A_c * u
            fields.update({"A": A, "P": P, "B": B, "Q": Q,
                           "R": A_c / P.clamp_min(1e-3)})
            if phi is not None:
                fields.update({"A_conv": A_c, "beta": beta})
            fields["n"], fields["n_grid"] = self.manning(x)
        return fields

    # ------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def harmonic_profile(self, x: torch.Tensor, ctx: Context, t_ref: torch.Tensor):
        """
        Amplitude and phase of every tidal constituent of stage along the channel.

        Args:
            x (torch.Tensor): Chainage (m), ``(N, 1)``.
            ctx (Context): Encoded window.
            t_ref (torch.Tensor): A representative time in the window, ``(N, 1)``.

        Returns:
            tuple: Amplitude (m) and phase lag (degrees), each ``(N, K)``.
        """
        if self.net_amp is None:
            return None, None
        s = (x - self.x_min) / self.x_len
        forcing = ctx.window.forcing(t_ref)
        ctx_in = torch.cat([2.0 * s - 1.0, self._latent_at(ctx, t_ref),
                            self._forcing_features(forcing)], dim=1)
        amp = self.net_amp(ctx_in).view(-1, 4, self.n_const)
        a, b = amp[:, 0], amp[:, 1]
        scale = self.scales.H * (1.0 - s)
        amplitude = scale * torch.sqrt(a * a + b * b)
        phase = torch.rad2deg(torch.atan2(-b, a)) % 360.0
        return amplitude, phase
