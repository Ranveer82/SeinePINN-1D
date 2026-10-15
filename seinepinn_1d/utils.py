"""
Small numerical helpers: interpolation, a smooth depth floor and skill scores.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------
def interp1d(x_query: torch.Tensor, x_data: torch.Tensor,
             y_data: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation that keeps the gradient with respect to the query points.

    Queries outside the data range are extrapolated linearly from the first
    and last segments.

    Args:
        x_query (torch.Tensor): Positions where values are wanted.
        x_data (torch.Tensor): Sample positions, sorted in increasing order.
        y_data (torch.Tensor): Values at the sample positions.

    Returns:
        torch.Tensor: Interpolated values, with the shape of ``x_query``.
    """
    flat = x_query.reshape(-1)
    idx = torch.searchsorted(x_data.contiguous(), flat.contiguous())
    idx = torch.clamp(idx, 1, x_data.numel() - 1)
    x_lo, x_hi = x_data[idx - 1], x_data[idx]
    y_lo, y_hi = y_data[idx - 1], y_data[idx]
    w = (flat - x_lo) / torch.clamp(x_hi - x_lo, min=1e-9)
    return (y_lo + w * (y_hi - y_lo)).reshape(x_query.shape)


def catmull_rom(seq: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    Smooth (C1) Catmull-Rom interpolation of a regularly sampled sequence.

    The hourly forcing and the LSTM latent are interpolated in time with this
    spline. Linear interpolation would give time derivatives that jump at
    every hour, and the Saint-Venant equations need smooth first derivatives.

    Args:
        seq (torch.Tensor): Sequence of shape ``(T, C)``, sampled at 0, 1, ..., T-1.
        pos (torch.Tensor): Query positions of shape ``(N, 1)``, in sample units.

    Returns:
        torch.Tensor: Interpolated values of shape ``(N, C)``.
    """
    T = seq.shape[0]
    p = pos.reshape(-1)
    i1 = torch.clamp(torch.floor(p).long(), 0, T - 1)
    frac = (p - i1.to(p.dtype)).unsqueeze(1)

    i0 = torch.clamp(i1 - 1, 0, T - 1)
    i2 = torch.clamp(i1 + 1, 0, T - 1)
    i3 = torch.clamp(i1 + 2, 0, T - 1)
    p0, p1, p2, p3 = seq[i0], seq[i1], seq[i2], seq[i3]

    t, t2 = frac, frac * frac
    t3 = t2 * frac
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def soft_depth(y: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Smooth, always positive version of ``max(y, 0)``: ``eps * softplus(y / eps)``.

    The smoothing only acts within a band of width about ``eps`` around zero,
    so normal water depths are returned unchanged.

    Args:
        y (torch.Tensor): Water depth before flooring (m).
        eps (float): Width of the smoothing band (m).

    Returns:
        torch.Tensor: Floored depth (m).
    """
    return eps * torch.nn.functional.softplus(y / eps)


# ---------------------------------------------------------------------------
# Skill scores
# ---------------------------------------------------------------------------
def _clean(sim, obs):
    """Flatten both series and drop the pairs where either is missing."""
    sim, obs = np.asarray(sim, float).ravel(), np.asarray(obs, float).ravel()
    m = np.isfinite(sim) & np.isfinite(obs)
    return sim[m], obs[m]


def rmse(sim, obs) -> float:
    """Root mean square error."""
    sim, obs = _clean(sim, obs)
    return float(np.sqrt(np.mean((sim - obs) ** 2))) if sim.size else float("nan")


def nse(sim, obs) -> float:
    """Nash-Sutcliffe efficiency (1 is perfect, 0 is no better than the mean)."""
    sim, obs = _clean(sim, obs)
    if sim.size < 2:
        return float("nan")
    denom = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - np.sum((sim - obs) ** 2) / denom) if denom > 0 else float("nan")


def kge(sim, obs) -> float:
    """
    Kling-Gupta efficiency (2012 form).

    The bias term is the difference of the means divided by the standard
    deviation of the observations, not the ratio of the means. Stage in a
    national datum can have a mean close to zero, and a ratio would then
    depend on the choice of datum.
    """
    sim, obs = _clean(sim, obs)
    if sim.size < 2:
        return float("nan")
    sd_o = obs.std()
    if sd_o == 0:
        return float("nan")
    r = np.corrcoef(sim, obs)[0, 1]
    r = 0.0 if not np.isfinite(r) else r
    alpha = sim.std() / sd_o
    beta = (sim.mean() - obs.mean()) / sd_o
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + beta ** 2))


def metric_table(sim, obs) -> dict:
    """RMSE, NSE, KGE and the number of values, in one dictionary."""
    return {"rmse": rmse(sim, obs), "nse": nse(sim, obs),
            "kge": kge(sim, obs), "n": int(np.isfinite(np.asarray(sim, float)).sum())}
