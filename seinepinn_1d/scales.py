"""
Characteristic scales of a case, taken from its own data.

Each Saint-Venant residual is divided by the size of its leading term, so both
equations are of order one and the loss weights have no units. Because the
scales come from the case data, the same code runs on a small tidal creek and
on a long lowland river without retuning.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch

from .config import G
from .tides import CONSTITUENTS


@dataclass
class Scales:
    """Characteristic magnitudes of one case."""

    L: float        # reach length (m)
    T: float        # characteristic time (s)
    H: float        # characteristic stage variation (m)
    A: float        # characteristic wetted area (m2)
    B: float        # characteristic top width (m)
    D: float        # characteristic depth (m)
    U: float        # characteristic velocity (m/s)
    Q: float        # characteristic discharge (m3/s)
    mass: float     # scale of the continuity residual (m2/s)
    mom: float      # scale of the momentum residual (m3/s2)
    u_max: float    # largest plausible velocity, used by the bound penalty (m/s)

    def as_dict(self) -> dict:
        """The scales as a plain dictionary (stored in checkpoints)."""
        return asdict(self)

    def summary(self) -> str:
        """One readable line per group of scales, for the log."""
        return (f"L={self.L/1000:.1f} km  T={self.T/3600:.2f} h  H={self.H:.2f} m  "
                f"A={self.A:.0f} m2  B={self.B:.0f} m  D={self.D:.2f} m  "
                f"U={self.U:.2f} m/s  Q={self.Q:.0f} m3/s\n"
                f"    residual scales: mass={self.mass:.3e} m2/s  "
                f"mom={self.mom:.3e} m3/s2")


def derive_scales(cfg, dataset, geometry, library) -> Scales:
    """
    Estimate the characteristic scales from the geometry and the forcing record.

    Args:
        cfg (CaseConfig): The case.
        dataset (RiverDataset): Forcing and observations.
        geometry (RiverGeometry): Tabulated channel geometry.
        library (CrossSectionLibrary): Cross sections of the reach.

    Returns:
        Scales: The scales.
    """
    L = float(library.length)

    # time: one M2 period on a tidal river, one day otherwise
    if cfg.physics.tidal:
        T = CONSTITUENTS["M2"] * 3600.0
    else:
        T = 24.0 * 3600.0

    # stage variation at the downstream boundary and at the gauges
    eta = dataset.eta_down
    H = max(2.0 * float(eta.std()), 0.25 * float(eta.max() - eta.min()))
    if dataset.gauges is not None and len(dataset.gauges):
        g = dataset.gauges
        ref = np.interp(g["t"].to_numpy(), dataset.t_hourly, eta)
        H = max(H, float(np.percentile(np.abs(g["eta"].to_numpy() - ref), 95)))
    H = max(H, 0.25)

    # geometry at the median water level
    eta_ref = float(np.median(eta))
    x_probe = torch.linspace(library.x_min, library.x_max, 200,
                             dtype=torch.float64,
                             device=geometry.x_nodes.device).unsqueeze(1)
    with torch.no_grad():
        # the table may carry two extra channels (storage split); the
        # scales use the first three: A, B, P
        A_p, B_p, P_p = geometry(x_probe,
                                 torch.full_like(x_probe, eta_ref))[:3]
    A = max(float(A_p.mean()), 1.0)
    B = max(float(B_p.mean()), 1.0)
    P = max(float(P_p.mean()), 1.0)
    D = A / B

    # velocity: river part from the upstream discharge, tidal part from the
    # long-wave estimate u ~ (H/D) * sqrt(g D)
    u_river = float(np.percentile(np.abs(dataset.Q_up), 95)) / A
    u_tide = (H / max(D, 0.1)) * np.sqrt(G * max(D, 0.1)) if cfg.physics.tidal else 0.0
    U = max(u_river, 0.5 * u_tide, 0.05)
    Q = A * U

    # residual scales
    # continuity: B * d(eta)/dt ~ B*H/T  and  dQ/dx ~ Q/L
    mass = max(B * H / T, Q / L)
    # momentum: dQ/dt ~ Q/T  and  pressure gradient ~ g*A*H/L
    mom = max(Q / T, G * A * H / L)

    return Scales(L=L, T=T, H=H, A=A, B=B, D=D, U=U, Q=Q,
                  mass=float(mass), mom=float(mom),
                  u_max=float(max(3.0 * U, 1.0)))
