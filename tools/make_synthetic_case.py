#!/usr/bin/env python
"""
Make the synthetic tidal estuary: a case whose true solution is known everywhere.

    python tools/make_synthetic_case.py --out data/synthetic
    python tools/make_synthetic_case.py --out data/synthetic_slr1m --sea-level-shift 1.0

Stage can be measured at gauges, but velocity and discharge rarely are. To
show that a model gets the velocity field right, and not only the gauge stage,
we need a river where the true velocity is known: this twin.

The 1-D Saint-Venant equations are integrated with a standard staggered
finite-volume scheme (stage at cell centres, discharge at faces, implicit
friction) on a 40 km funnel-shaped estuary, 60 m wide at the head and 420 m at
the mouth, forced at the mouth by six tidal constituents and upstream by a
river hydrograph with four flood waves, with two tributaries. The result is
written in the same CSV formats as a real case:

    river_profiles.csv    bathymetry (X_m, d_0..d_N, h_0..h_N)
    upstream_Q.csv        discharge at x = 0
    downstream_stage.csv  stage at x = L
    tributaries.csv       two tributary inflows
    gauges.csv            hourly stage at 5 stations, with noise (and the true
                          velocity there, only to score the model)
    truth_grid.npz        the full true field, to score the model

The solver and the PINN share nothing but these files.

Scenario options change the forcing as a climate projection would:
``--sea-level-shift`` raises the downstream stage, ``--q-scale`` multiplies
the upstream discharge (and the tributaries, unless ``--trib-scale`` is set).
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

G = 9.80665
HOUR = 3600.0

#: Scenario changes of the forcing, set from the command line in main().
SEA_LEVEL_SHIFT = 0.0
Q_SCALE = 1.0
TRIB_SCALE = 1.0

#: Tidal constituents at the mouth: period (h), amplitude (m), phase (deg).
#: M2 and S2 beat into the spring-neap cycle; M4 gives flood/ebb asymmetry.
TIDE = [
    (12.420601, 2.60, 0.0),     # M2
    (12.000000, 0.90, 40.0),    # S2
    (12.658348, 0.50, 300.0),   # N2
    (23.934470, 0.16, 120.0),   # K1
    (25.819342, 0.11, 200.0),   # O1
    (6.210300, 0.30, 60.0),     # M4
]


# ---------------------------------------------------------------------------
# The channel (trapezoidal, so A, B and P are exact)
# ---------------------------------------------------------------------------
class Channel:
    """Geometry and roughness of the synthetic estuary."""

    def __init__(self, length: float, n_cells: int):
        """
        Args:
            length (float): Length of the reach (m).
            n_cells (int): Number of finite-volume cells.
        """
        self.L = length
        self.N = n_cells
        self.dx = length / n_cells
        self.x_face = np.linspace(0.0, length, n_cells + 1)
        self.x_cell = 0.5 * (self.x_face[:-1] + self.x_face[1:])
        self.side_slope = 3.0                       # horizontal : vertical
        self.bank_height = 12.0

    def bed(self, x):
        """Bed level: from +1.0 m at the head to -9.0 m at the mouth, with bumps."""
        s = x / self.L
        return 1.0 - 10.0 * s - 0.8 * np.sin(3.0 * np.pi * s)

    def bed_width(self, x):
        """Bed width: funnel from 60 m at the head to 420 m at the mouth."""
        s = x / self.L
        return 60.0 + 360.0 * s ** 1.6

    def manning(self, x):
        """True Manning roughness: smoother in the deep lower reach."""
        s = x / self.L
        return 0.032 - 0.012 * s + 0.004 * np.sin(2.0 * np.pi * s)

    def props(self, x, eta):
        """Wetted area, top width and wetted perimeter at water level ``eta``."""
        d = np.maximum(eta - self.bed(x), 1e-3)
        d = np.minimum(d, self.bank_height)
        b = self.bed_width(x)
        m = self.side_slope
        A = b * d + m * d * d
        B = b + 2.0 * m * d
        P = b + 2.0 * d * np.sqrt(1.0 + m * m)
        return A, B, P

    def section_points(self, x, n_pts: int = 25):
        """Points of the cross section at ``x``, for the bathymetry CSV."""
        zb = self.bed(x)
        b = self.bed_width(x)
        m = self.side_slope
        H = self.bank_height
        half = 0.5 * b
        top = half + m * H
        n_side = (n_pts - 2) // 2
        left = np.linspace(-top, -half, n_side + 1)[:-1]
        right = np.linspace(half, top, n_side + 1)[1:]
        d = np.concatenate([left, [-half, half], right])
        z = np.where(np.abs(d) <= half, zb,
                     zb + (np.abs(d) - half) / m)
        order = np.argsort(d)
        return d[order] + top, z[order]      # offsets start at 0


# ---------------------------------------------------------------------------
# Forcing
# ---------------------------------------------------------------------------
def downstream_stage(t):
    """Stage at the mouth: mean level, a seasonal swing and the tide (m)."""
    eta = SEA_LEVEL_SHIFT + 0.35 + 0.25 * np.sin(2 * np.pi * t / (180 * 86400.0))
    for period_h, amp, phase in TIDE:
        eta = eta + amp * np.cos(2 * np.pi * t / (period_h * HOUR)
                                 - np.deg2rad(phase))
    return eta


def upstream_discharge(t, rng):
    """River discharge at the head: base flow plus four smooth flood waves (m3/s)."""
    q = 180.0 + 40.0 * np.sin(2 * np.pi * t / (120 * 86400.0))
    for centre_d, peak, width_d in [(12, 620, 4.0), (38, 340, 3.0),
                                    (61, 900, 5.5), (80, 420, 3.5)]:
        q = q + peak * np.exp(-0.5 * ((t / 86400.0 - centre_d) / width_d) ** 2)
    return Q_SCALE * q


def tributary_series(t, phase, base, peak):
    """Discharge of a tributary: base flow, a slow swing and three floods (m3/s)."""
    q = base + 0.25 * base * np.sin(2 * np.pi * t / (90 * 86400.0) + phase)
    for centre_d, amp, width_d in [(13, peak, 2.5), (40, 0.6 * peak, 2.0),
                                   (62, 1.1 * peak, 3.0)]:
        q = q + amp * np.exp(-0.5 * ((t / 86400.0 - centre_d) / width_d) ** 2)
    return TRIB_SCALE * q


def gaussian_kernel(x, loc, sigma, x0, x1):
    """Gaussian spreading of a tributary along the reach, normalised over [x0, x1]."""
    z = 0.5 * (math.erf((x1 - loc) / (sigma * math.sqrt(2)))
               - math.erf((x0 - loc) / (sigma * math.sqrt(2))))
    return np.exp(-0.5 * ((x - loc) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi) * z)


# ---------------------------------------------------------------------------
# Reference solver
# ---------------------------------------------------------------------------
def simulate(ch: Channel, days: float, dt: float, tribs, rng,
             sample_every: float = HOUR, verbose: bool = True):
    """
    Integrate the Saint-Venant equations on a staggered grid.

    Args:
        ch (Channel): The channel.
        days (float): Simulated duration (days).
        dt (float): Time step (s).
        tribs (list[dict]): Tributaries (x, sigma, base, peak, phase).
        rng: Random generator.
        sample_every (float): Output interval (s).
        verbose (bool): Print progress.

    Returns:
        tuple: ``(t, eta at cells, Q at faces)``, sampled every ``sample_every``.
    """
    n_steps = int(round(days * 86400.0 / dt))
    n_out = int(round(days * 86400.0 / sample_every)) + 1
    out_stride = int(round(sample_every / dt))

    xc, xf, dx, N = ch.x_cell, ch.x_face, ch.dx, ch.N
    zb_c = ch.bed(xc)
    n_face = ch.manning(xf)

    # shape of each tributary inflow along the reach (per unit length)
    kern = np.stack([gaussian_kernel(xc, tr["x"], tr["sigma"], 0.0, ch.L)
                     for tr in tribs]) if tribs else np.zeros((0, N))

    eta = np.full(N, downstream_stage(0.0))
    eta = np.maximum(eta, zb_c + 0.5)
    Q = np.zeros(N + 1)

    rec_eta = np.zeros((n_out, N))
    rec_Q = np.zeros((n_out, N + 1))
    rec_t = np.zeros(n_out)
    k = 0

    for step in range(n_steps + 1):
        t = step * dt

        if step % out_stride == 0 and k < n_out:
            rec_eta[k], rec_Q[k], rec_t[k] = eta, Q, t
            k += 1

        eta_dn = downstream_stage(t)
        q_up = upstream_discharge(t, rng)
        q_lat = np.zeros(N)
        for j, tr in enumerate(tribs):
            q_lat += tributary_series(t, tr["phase"], tr["base"], tr["peak"]) * kern[j]

        A_c, B_c, P_c = ch.props(xc, eta)

        # momentum on the interior and downstream faces
        eta_ghost = 2.0 * eta_dn - eta[-1]
        eta_ext = np.concatenate([eta[:1], eta, [eta_ghost]])     # (N+2,)
        eta_f = 0.5 * (eta_ext[:-1] + eta_ext[1:])                # (N+1,)
        A_f, _, P_f = ch.props(xf, eta_f)
        R_f = A_f / np.maximum(P_f, 1e-3)

        Q_c = 0.5 * (Q[:-1] + Q[1:])
        F_c = Q_c * Q_c / np.maximum(A_c, 1e-3)
        F_ext = np.concatenate([F_c[:1], F_c, F_c[-1:]])
        dF = (F_ext[1:] - F_ext[:-1]) / dx                        # (N+1,)
        dEta = (eta_ext[1:] - eta_ext[:-1]) / dx

        rhs = -dF - G * A_f * dEta
        fric = (G * n_face ** 2 * np.abs(Q)
                / (np.maximum(A_f, 1e-3) * np.maximum(R_f, 1e-3) ** (4.0 / 3.0)))
        Q_new = (Q + dt * rhs) / (1.0 + dt * fric)

        Q_new[0] = q_up                                           # upstream condition
        Q = Q_new

        # continuity on the cells
        eta = eta + dt * (-(Q[1:] - Q[:-1]) / dx + q_lat) / np.maximum(B_c, 1e-3)
        eta = np.maximum(eta, zb_c + 0.05)

        if not np.all(np.isfinite(eta)):
            raise FloatingPointError(f"solver blew up at t = {t/3600:.1f} h")
        if verbose and step % max(1, n_steps // 10) == 0:
            print(f"    t = {t/86400:6.2f} d   eta[{N//2}] = {eta[N//2]:+.3f} m   "
                  f"|Q|max = {np.abs(Q).max():7.1f} m3/s")

    return rec_t[:k], rec_eta[:k], rec_Q[:k]


# ---------------------------------------------------------------------------
# Writing the case
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Make the synthetic tidal estuary (a case with a known solution).")
    ap.add_argument("--out", default="data/synthetic",
                    help="output folder (default data/synthetic)")
    ap.add_argument("--days", type=float, default=90.0, help="duration (days)")
    ap.add_argument("--length", type=float, default=40000.0, help="reach length (m)")
    ap.add_argument("--cells", type=int, default=80, help="number of cells")
    ap.add_argument("--dt", type=float, default=15.0, help="time step (s)")
    ap.add_argument("--epoch", default="2020-01-01 00:00:00", help="start date")
    ap.add_argument("--noise", type=float, default=0.015,
                    help="noise on the gauge stage (m)")
    ap.add_argument("--seed", type=int, default=7, help="random seed")
    ap.add_argument("--sea-level-shift", type=float, default=0.0,
                    help="add this many metres to the downstream stage")
    ap.add_argument("--q-scale", type=float, default=1.0,
                    help="multiply the upstream discharge by this factor")
    ap.add_argument("--trib-scale", type=float, default=None,
                    help="multiply the tributaries by this factor "
                         "(default: the same as --q-scale)")
    args = ap.parse_args()

    global SEA_LEVEL_SHIFT, Q_SCALE, TRIB_SCALE
    SEA_LEVEL_SHIFT = args.sea_level_shift
    Q_SCALE = args.q_scale
    TRIB_SCALE = args.q_scale if args.trib_scale is None else args.trib_scale
    print(f"sea-level shift {SEA_LEVEL_SHIFT:+.2f} m   "
          f"upstream Q x{Q_SCALE:g}   tributaries x{TRIB_SCALE:g}")

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ch = Channel(args.length, args.cells)

    tribs = [
        {"name": "trib_upper", "x": 0.22 * ch.L, "sigma": 600.0,
         "base": 22.0, "peak": 95.0, "phase": 0.4},
        {"name": "trib_lower", "x": 0.68 * ch.L, "sigma": 800.0,
         "base": 14.0, "peak": 60.0, "phase": 2.1},
    ]

    print(f"Integrating the reference Saint-Venant solution: "
          f"{args.days:.0f} d, {ch.N} cells, dt = {args.dt:g} s "
          f"({int(args.days*86400/args.dt):,} steps)")
    t, eta, Q = simulate(ch, args.days, args.dt, tribs, rng)
    print(f"  done: {eta.shape[0]} hourly snapshots")

    epoch = pd.to_datetime(args.epoch)
    stamps = epoch + pd.to_timedelta(t, unit="s")

    # everything on the face grid, which is also the chainage of the sections
    A_f = np.stack([ch.props(ch.x_face, np.interp(ch.x_face, ch.x_cell, e))[0]
                    for e in eta])
    U_f = Q / np.maximum(A_f, 1e-3)
    eta_f = np.stack([np.interp(ch.x_face, ch.x_cell, e) for e in eta])
    eta_f[:, -1] = downstream_stage(t)          # exact at the open boundary

    # bathymetry
    rows = []
    for x in ch.x_face:
        d, z = ch.section_points(x)
        row = {"X_m": x}
        row.update({f"d_{i}": v for i, v in enumerate(d)})
        row.update({f"h_{i}": v for i, v in enumerate(z)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{args.out}/river_profiles.csv", index=False)

    # forcing
    pd.DataFrame({"timestamp": stamps,
                  "discharge": upstream_discharge(t, rng)}
                 ).to_csv(f"{args.out}/upstream_Q.csv", index=False)
    pd.DataFrame({"timestamp": stamps, "stage": downstream_stage(t)}
                 ).to_csv(f"{args.out}/downstream_stage.csv", index=False)
    trib_df = {"timestamp": stamps}
    for tr in tribs:
        trib_df[tr["name"]] = tributary_series(t, tr["phase"], tr["base"], tr["peak"])
    pd.DataFrame(trib_df).to_csv(f"{args.out}/tributaries.csv", index=False)

    # gauges: stage with noise, and the true velocity (only to score the model)
    station_x = np.array([0.10, 0.30, 0.50, 0.72, 0.90]) * ch.L
    g_rows = []
    for xs in station_x:
        e = np.array([np.interp(xs, ch.x_face, ef) for ef in eta_f])
        u = np.array([np.interp(xs, ch.x_face, uf) for uf in U_f])
        g_rows.append(pd.DataFrame({
            "timestamp": stamps,
            "X_m": xs,
            "stage_m": e + rng.normal(0.0, args.noise, e.size),
            "velocity_ms": u,
        }))
    pd.concat(g_rows).sort_values("timestamp").to_csv(
        f"{args.out}/gauges.csv", index=False)

    # the full true field
    np.savez_compressed(
        f"{args.out}/truth_grid.npz",
        t=t, x=ch.x_face, eta=eta_f, u=U_f, Q=Q,
        manning=ch.manning(ch.x_face), bed=ch.bed(ch.x_face),
        epoch=str(epoch),
    )

    print(f"\nWrote the case to {args.out}/ ({len(t)} hourly steps, "
          f"{stamps[0]} .. {stamps[-1]})")
    print(f"  tidal range at mouth : {eta_f[:,-1].max()-eta_f[:,-1].min():.2f} m")
    print(f"  tidal range at head  : {eta_f[:,0].max()-eta_f[:,0].min():.2f} m")
    print(f"  |u| max / mean       : {np.abs(U_f).max():.2f} / {np.abs(U_f).mean():.2f} m/s")
    print(f"  |Q| max              : {np.abs(Q).max():.0f} m3/s")


if __name__ == "__main__":
    main()
