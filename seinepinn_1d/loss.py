"""
Saint-Venant residuals and the loss terms.

The 1-D shallow-water equations for a channel of any cross section, written
with the water-surface elevation ``eta`` (so the bed slope is inside
``d(eta)/dx``):

    continuity:  B d(eta)/dt + dQ/dx = q_lat
    momentum:    dQ/dt + d(Q^2/A)/dx + g A (d(eta)/dx + Sf) = 0

with ``B = dA/d(eta)``, ``Sf = n^2 Q|Q| / (A^2 R^(4/3))`` and ``R = A/P``.
``q_lat`` holds the tributaries only; the upstream discharge enters as the
boundary condition ``Q(x0, t) = Q_up(t)``.

Both residuals are divided by their characteristic scale (see ``scales.py``)
and weighted causally in time inside each window (Wang, Sankaran & Perdikaris
2022), so late times are not fitted before early times are right.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import G

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


class LateralInflow:
    """
    Tributary inflows spread along the reach as Gaussian sources.

    Each Gaussian is renormalised over the reach, so a tributary near either
    end still brings in exactly its discharge.
    """

    def __init__(self, cfg, x_min: float, x_max: float, device):
        """
        Args:
            cfg (CaseConfig): The case (list of tributaries).
            x_min, x_max (float): Ends of the reach (m).
            device: Torch device.
        """
        self.tribs = list(cfg.tributaries)
        self.x_min, self.x_max = float(x_min), float(x_max)
        locs, sigmas, norms = [], [], []
        for t in self.tribs:
            sig = float(t.sigma or cfg.physics.lateral_sigma)
            if not (self.x_min - 3 * sig <= t.x <= self.x_max + 3 * sig):
                raise ValueError(
                    f"tributary '{t.name}' at x={t.x} is outside the reach "
                    f"[{self.x_min}, {self.x_max}]"
                )
            z = 0.5 * (math.erf((self.x_max - t.x) / (sig * _SQRT2))
                       - math.erf((self.x_min - t.x) / (sig * _SQRT2)))
            locs.append(t.x)
            sigmas.append(sig)
            norms.append(max(z, 1e-6))
        self.loc = torch.tensor(locs, device=device) if locs else None
        self.sigma = torch.tensor(sigmas, device=device) if sigmas else None
        self.norm = torch.tensor(norms, device=device) if norms else None

    def __call__(self, x: torch.Tensor, forcing: dict) -> torch.Tensor:
        """
        Lateral inflow per unit length.

        Args:
            x (torch.Tensor): Chainage (m), ``(N, 1)``.
            forcing (dict): Forcing at the same points (needs ``Q_trib``).

        Returns:
            torch.Tensor: ``q_lat`` (m2/s), ``(N, 1)``.
        """
        q = torch.zeros_like(x)
        if self.loc is not None:
            d = (x - self.loc) / self.sigma                   # (N, K)
            kern = torch.exp(-0.5 * d * d) / (self.sigma * _SQRT2PI * self.norm)
            q = q + (forcing["Q_trib"] * kern).sum(dim=1, keepdim=True)
        return q


def _d(y, v, create_graph: bool = True):
    """Derivative of ``y`` with respect to ``v`` (pointwise, by autograd)."""
    return torch.autograd.grad(y, v, torch.ones_like(y),
                               create_graph=create_graph,
                               retain_graph=True)[0]


def residuals(model, x, t, ctx, lateral, fields=None, create_graph: bool = True):
    """
    Non-dimensional continuity and momentum residuals at points ``(x, t)``.

    Args:
        model (RiverPINN): The network.
        x (torch.Tensor): Chainage (m), ``(N, 1)``, with ``requires_grad``.
        t (torch.Tensor): Time (s), ``(N, 1)``, with ``requires_grad``.
        ctx (Context): Encoded window.
        lateral (LateralInflow): Tributary inflows.
        fields (dict, optional): Model output at the points, if already known.
        create_graph (bool): Keep the graph (needed for training; not for
            scoring points only).

    Returns:
        tuple: ``(r_mass, r_mom, fields, Sf)``.
    """
    if fields is None:
        fields = model(x, t, ctx, create_graph=create_graph)

    eta, Q, A, B, P, R = (fields["eta"], fields["Q"], fields["A"],
                          fields["B"], fields["P"], fields["R"])
    u, n = fields["u"], fields["n"]
    # momentum uses the conveying area, continuity the total area; they are
    # the same unless the storage split is on
    A_c = fields.get("A_conv", A)
    beta_m = fields.get("beta", None)

    cg = create_graph
    eta_x, eta_t = _d(eta, x, cg), _d(eta, t, cg)
    Q_x, Q_t = _d(Q, x, cg), _d(Q, t, cg)
    A_cx = _d(A_c, x, cg)

    q_lat = lateral(x, fields["forcing"])

    # continuity: every wetted path stores water
    res_mass = B * eta_t + Q_x - q_lat

    # momentum
    Sf = (n * n * Q * torch.abs(Q)) / (A_c * A_c
                                       * torch.pow(R.clamp_min(1e-3), 4.0 / 3.0))
    flux = 2.0 * Q * Q_x / A_c - (Q / A_c) ** 2 * A_cx        # d(Q^2/A_c)/dx
    if beta_m is not None:
        flux = beta_m * flux + _d(beta_m, x, cg) * Q * Q / A_c
    res_mom = Q_t + flux + G * A_c * (eta_x + Sf)

    beta = model.cfg.physics.lateral_momentum
    if beta:
        res_mom = res_mom - beta * q_lat * u

    nu = model.cfg.physics.eddy_viscosity
    if nu:
        res_mom = res_mom - _d(nu * A_c * _d(u, x, cg), x, cg)

    sc = model.scales
    return res_mass / sc.mass, res_mom / sc.mom, fields, Sf


def causal_weights(tau: torch.Tensor, r2: torch.Tensor, n_bins: int, eps: float,
                   floor: float = 0.05):
    """
    Causal weights in time (Wang, Sankaran & Perdikaris 2022), scale-free.

    The window is cut into ``n_bins`` time bins. The weight of a bin is
    ``exp(-eps * C)``, where ``C`` is the loss of the earlier bins divided by
    the total, so the last bin gets ``exp(-eps)`` of the first whatever the
    loss level. Weights never go below ``floor``.

    Args:
        tau (torch.Tensor): Position of each point in the window, 0 to 1.
        r2 (torch.Tensor): Squared residual at each point.
        n_bins (int): Number of time bins.
        eps (float): Strength (``0`` = no weighting).
        floor (float): Smallest weight.

    Returns:
        tuple: ``(weight per point, mean loss per bin)``.
    """
    if eps <= 0 or n_bins <= 1:
        return torch.ones_like(r2), None

    idx = torch.clamp((tau.reshape(-1) * n_bins).long(), 0, n_bins - 1)
    ones = torch.ones_like(idx, dtype=r2.dtype)
    count = torch.zeros(n_bins, dtype=r2.dtype, device=r2.device).index_add_(0, idx, ones)
    total = torch.zeros(n_bins, dtype=r2.dtype, device=r2.device).index_add_(
        0, idx, r2.reshape(-1))
    bin_mean = total / count.clamp_min(1.0)

    with torch.no_grad():
        cum = torch.cumsum(bin_mean, 0) - bin_mean          # loss of earlier bins
        cum = cum / cum[-1].clamp_min(1e-30)                # scaled to [0, 1]
        w_bin = torch.exp(-eps * cum).clamp_min(floor)
        w_bin = torch.nan_to_num(w_bin, nan=1.0, posinf=1.0, neginf=floor)
    return w_bin[idx].reshape(r2.shape), bin_mean.detach()


def volume_balance_loss(model, ctx, lateral, cfg, n_x: int = 64, n_t: int = 6):
    """
    Volume balance of the whole reach: ``dV/dt = Q(0) - Q(L) + tributaries``.

    This follows from the pointwise continuity equation, but a small residual
    of the same sign everywhere adds up over the reach, and the tidal discharge
    inside the reach is not observed. This term ties the discharge at both
    ends to the change of volume between them.

    Evaluated on a regular grid of ``n_x`` points at ``n_t`` random times of
    the window.

    Args:
        model (RiverPINN): The network.
        ctx (Context): Encoded window.
        lateral (LateralInflow): Tributary inflows.
        cfg (CaseConfig): The case.
        n_x (int): Points along the reach.
        n_t (int): Times per evaluation.

    Returns:
        torch.Tensor: Mean squared imbalance, divided by the discharge scale.
    """
    w = ctx.window
    dev = model.n_nodes.device
    xs = torch.linspace(model.x_min, model.x_min + model.x_len, n_x,
                        dtype=torch.float64, device=dev).unsqueeze(1)
    ts = (w.t0 + (w.t1 - w.t0)
          * torch.rand(n_t, dtype=torch.float64, device=dev))

    err = []
    for tv in ts:
        t_col = tv.reshape(1, 1).expand_as(xs).clone().requires_grad_(True)
        f = model(xs, t_col, ctx)
        # dV/dt = integral over the reach of B * d(eta)/dt
        eta_t = _d(f["eta"], t_col, create_graph=True)
        dVdt = torch.trapz((f["B"] * eta_t).reshape(-1), xs.reshape(-1))
        q = lateral(xs, f["forcing"])
        net = (f["Q"][0] - f["Q"][-1]).reshape(())            \
            + torch.trapz(q.reshape(-1), xs.reshape(-1))
        err.append((dVdt - net) / model.scales.Q)
    return torch.stack(err).pow(2).mean()


def physics_loss(model, x, t, ctx, lateral, cfg, fields=None):
    """
    Causally weighted continuity and momentum losses, plus the bound penalty.

    The bound penalty is zero inside the plausible range (velocity below
    ``u_max``, depth above ``min_depth``) and grows outside it.

    Args:
        model (RiverPINN): The network.
        x, t (torch.Tensor): Collocation points, with ``requires_grad``.
        ctx (Context): Encoded window.
        lateral (LateralInflow): Tributary inflows.
        cfg (CaseConfig): The case.
        fields (dict, optional): Model output at the points.

    Returns:
        tuple: ``(losses, fields, diagnostics, mean loss per time bin)``.
    """
    r_mass, r_mom, fields, Sf = residuals(model, x, t, ctx, lateral, fields)
    w = ctx.window
    tau = (t - w.t0) / max(w.t1 - w.t0, 1.0)

    r2 = r_mass ** 2 + r_mom ** 2
    weights, bin_mean = causal_weights(tau, r2, cfg.train.causal_bins,
                                       cfg.train.causal_eps,
                                       floor=cfg.train.causal_floor)
    denom = weights.sum().clamp_min(1e-12)

    over_u = F.relu(fields["u"].abs() - model.scales.u_max) / model.scales.U
    thin = F.relu(cfg.physics.min_depth - fields["depth"]) / cfg.physics.min_depth

    losses = {
        "mass": (weights * r_mass ** 2).sum() / denom,
        "momentum": (weights * r_mom ** 2).sum() / denom,
        "bounds": (over_u ** 2).mean() + (thin ** 2).mean(),
    }

    diag = {
        "res_mass_rms": float(r_mass.detach().pow(2).mean().sqrt()),
        "res_mom_rms": float(r_mom.detach().pow(2).mean().sqrt()),
        "causal_w": float(weights.detach().mean()),
        "Sf_share": float((G * fields.get("A_conv", fields["A"]) * Sf)
                          .detach().abs().mean()
                          / model.scales.mom),
    }
    return losses, fields, diag, bin_mean


def boundary_and_data_loss(model, ctx, cfg, scales, n_data: int,
                           include_bounds: bool = False, generator=None):
    """
    Upstream boundary, gauge observations and priors.

    Terms:

    - ``bc_up``: upstream discharge ``Q(x0, t) = Q_up(t)`` (measured hours only);
    - ``data``: stage at the gauges;
    - ``bounds``: plausibility penalty (only when the physics loss, which
      also carries it, is not running);
    - ``smooth_n``: smoothness of the roughness profile;
    - ``smooth_geo``: smoothness and closeness to 1 of the width correction.

    Args:
        model (RiverPINN): The network.
        ctx (Context): Encoded window.
        cfg (CaseConfig): The case.
        scales (Scales): Characteristic scales.
        n_data (int): Points drawn per term.
        include_bounds (bool): Add the plausibility penalty.
        generator (torch.Generator, optional): Random generator.

    Returns:
        tuple: ``(losses, diagnostics)``.
    """
    w = ctx.window
    dev = w.t_hourly.device
    losses, diag = {}, {}

    # upstream discharge
    T = w.n_hours
    k = torch.randint(0, T, (min(n_data, T),), device=dev, generator=generator)
    t_u = w.t_hourly[k].reshape(-1, 1)
    x_u = torch.full_like(t_u, model.x_min)
    f_u = model(x_u, t_u, ctx)
    mask = w.Q_up_mask[k].reshape(-1, 1)
    err = (f_u["Q"] - w.Q_up[k].reshape(-1, 1)) / scales.Q
    losses["bc_up"] = (mask * err ** 2).sum() / mask.sum().clamp_min(1.0)
    diag["bc_up_rmse_m3s"] = float(
        ((mask * (f_u["Q"] - w.Q_up[k].reshape(-1, 1)) ** 2).sum()
         / mask.sum().clamp_min(1.0)).sqrt().detach())

    # stage at the gauges
    if w.gauge_t.numel() > 0:
        g = torch.randint(0, w.gauge_t.numel(), (min(n_data, w.gauge_t.numel()),),
                          device=dev, generator=generator)
        x_g = w.gauge_x[g].reshape(-1, 1)
        t_g = w.gauge_t[g].reshape(-1, 1)
        f_g = model(x_g, t_g, ctx, want_geometry=False)
        d_eta = (f_g["eta"] - w.gauge_eta[g].reshape(-1, 1))
        sq = (d_eta / scales.H) ** 2
        if cfg.train.station_weighted_data and w.gauge_w is not None:
            ww = w.gauge_w[g].reshape(-1, 1)
            losses["data"] = (ww * sq).sum() / ww.sum().clamp_min(1e-12)
        else:
            losses["data"] = sq.mean()
        diag["data_rmse_m"] = float(d_eta.pow(2).mean().sqrt().detach())
    else:
        losses["data"] = torch.zeros((), device=dev)

    # plausibility, when the physics loss is not running
    if include_bounds:
        xr = torch.rand((n_data, 1), device=dev) * model.x_len + model.x_min
        tr = torch.rand((n_data, 1), device=dev) * (w.t1 - w.t0) + w.t0
        fr = model(xr, tr, ctx, want_geometry=False)
        over_u = F.relu(fr["u"].abs() - scales.u_max) / scales.U
        thin = F.relu(cfg.physics.min_depth - fr["depth"]) / cfg.physics.min_depth
        losses["bounds"] = (over_u ** 2).mean() + (thin ** 2).mean()

    # smoothness of the roughness profile
    n_grid = model.manning_grid()
    dn = (n_grid[1:] - n_grid[:-1]) / (model.n_max - model.n_min)
    losses["smooth_n"] = (dn ** 2).mean()

    # width correction: smooth along the reach and close to the survey
    _, g_grid = model.width_factor(torch.zeros((1, 1), device=dev,
                                               dtype=torch.float64))
    if g_grid is not None:
        dg = g_grid[1:] - g_grid[:-1]
        losses["smooth_geo"] = (dg ** 2).mean() + 0.1 * ((g_grid - 1.0) ** 2).mean()
        diag["width_factor_mean"] = float(g_grid.mean().detach())
        diag["width_factor_absmax"] = float((g_grid - 1.0).abs().max().detach())

    return losses, diag


def initial_condition_loss(model, ctx, cfg, scales, reference, n_points: int = 512):
    """
    Match the start of a window to the state at the end of the previous one.

    Args:
        model (RiverPINN): The network.
        ctx (Context): Encoded window.
        cfg (CaseConfig): The case.
        scales (Scales): Characteristic scales.
        reference (callable | None): ``x -> (eta, u)``: the previous window's
            final state, or a profile from the gauges for the first window.
            ``u`` may be None.
        n_points (int): Random points along the reach.

    Returns:
        tuple: ``({"ic": loss}, diagnostics)``.
    """
    if reference is None:
        return {"ic": torch.zeros((), device=ctx.window.t_hourly.device)}, {}
    dev = ctx.window.t_hourly.device
    x = (torch.rand((n_points, 1), device=dev)
         * model.x_len + model.x_min)
    t = torch.full_like(x, ctx.window.t0)
    f = model(x, t, ctx, want_geometry=False)
    eta_ref, u_ref = reference(x)
    loss = (((f["eta"] - eta_ref) / scales.H) ** 2).mean()
    if u_ref is not None:
        loss = loss + (((f["u"] - u_ref) / scales.U) ** 2).mean()
    return {"ic": loss}, {"ic_rmse_m": float((f["eta"] - eta_ref).pow(2)
                                             .mean().sqrt().detach())}
