"""
Channel geometry: from surveyed cross sections to ``A(x, eta)``, ``P(x, eta)``
and ``B(x, eta)``.

1. The cross sections are read, cleaned (raster no-data fill, flat padding)
   and closed where the survey stops below dry ground.
2. Wetted area ``A``, top width ``B`` and wetted perimeter ``P`` are tabulated
   on a grid of chainage and water level.
3. A small neural network (the surrogate) is fitted to the table. The model
   evaluates ``A`` and ``P`` through it and gets the top width as
   ``B = dA/d(eta)`` by automatic differentiation, so continuity and momentum
   always use the same geometry.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEVICE
from .utils import interp1d


# ---------------------------------------------------------------------------
# Section properties from the surveyed points
# ---------------------------------------------------------------------------
def _segment_properties(d: np.ndarray, z: np.ndarray, eta: float):
    """
    Wetted area, top width and wetted perimeter of each segment of a section.

    Args:
        d (np.ndarray): Transverse offsets of the points (m), increasing.
        z (np.ndarray): Bed elevation of the points (m).
        eta (float): Water-surface elevation (m).

    Returns:
        tuple: ``(A, W, P)`` per segment.
    """
    y = eta - z                                  # depth at each point
    d0, d1 = d[:-1], d[1:]
    y0, y1 = y[:-1], y[1:]
    dd = d1 - d0
    dz = z[1:] - z[:-1]

    A = np.zeros_like(dd)
    W = np.zeros_like(dd)
    P = np.zeros_like(dd)

    both = (y0 > 0) & (y1 > 0)
    A[both] = 0.5 * (y0 + y1)[both] * dd[both]
    W[both] = dd[both]
    P[both] = np.hypot(dd[both], dz[both])

    # partly wet segments: the waterline cuts the segment at fraction f
    left = (y0 > 0) & (y1 <= 0)
    if left.any():
        f = y0[left] / (y0[left] - y1[left])
        A[left] = 0.5 * y0[left] * dd[left] * f
        W[left] = dd[left] * f
        P[left] = np.hypot(dd[left], dz[left]) * f

    right = (y0 <= 0) & (y1 > 0)
    if right.any():
        f = y1[right] / (y1[right] - y0[right])
        A[right] = 0.5 * y1[right] * dd[right] * f
        W[right] = dd[right] * f
        P[right] = np.hypot(dd[right], dz[right]) * f

    return A, W, P


def _split_at_banks(paths, bankfull_idx):
    """
    Cut each wetted path at the banks, for the divided-channel method.

    A path that crosses a bank becomes up to three pieces: left floodplain,
    main channel and right floodplain. Their conveyances are then summed.

    Args:
        paths (list[np.ndarray]): Segment indices of each wetted path.
        bankfull_idx (tuple): First and last segment of the main channel.

    Returns:
        list[np.ndarray]: The pieces.
    """
    lo, hi = bankfull_idx
    if hi < lo:                       # no channel to divide
        return paths
    out = []
    for p in paths:
        for piece in (p[p < lo], p[(p >= lo) & (p <= hi)], p[p > hi]):
            if piece.size:
                out.append(piece)
    return out


def bankfull_segments(d: np.ndarray, z: np.ndarray, z_bank: float):
    """
    The segments of the main channel at bank level ``z_bank``.

    Walks outwards from the deepest point and stops at the first ground above
    the bank level on each side.

    Args:
        d (np.ndarray): Transverse offsets (m).
        z (np.ndarray): Bed elevation (m).
        z_bank (float): Bank elevation (m).

    Returns:
        tuple: First and last segment index of the main channel.
    """
    i0 = int(np.argmin(z))
    lo = i0
    while lo - 1 >= 0 and z[lo - 1] <= z_bank:
        lo -= 1
    hi = i0
    while hi + 1 < z.size and z[hi + 1] <= z_bank:
        hi += 1
    # lo and hi are the outermost points at or below the bank, so the
    # segments inside the channel are lo .. hi-1
    return lo, hi - 1


def section_properties(d: np.ndarray, z: np.ndarray, eta: float,
                       compound: bool = False, main_only: bool = False,
                       storage_threshold: float = 0.0, full: bool = False,
                       z_bank: float | None = None):
    """
    Wetted area, top width and wetted perimeter of a cross section at level ``eta``.

    The wetted part of a section can be several separate flow paths (around
    islands, side channels). Area and top width are always summed over all
    paths, because every path stores water. With ``compound`` the perimeter is
    a conveyance-equivalent one: the total conveyance
    ``sum_i A_i R_i^(2/3)`` is kept, which gives ``R_eq`` and ``P_eq = A / R_eq``.
    Summing the perimeters directly would make ``R = A/P`` far too small when a
    wide shallow side channel is wet.

    With ``storage_threshold > 0`` a path whose share of the total conveyance
    is below the threshold only stores water: it counts in area and top width
    (continuity) but not in the conveying area used by momentum. The path
    with the thalweg always conveys.

    With ``z_bank`` and a water level above it, the paths are also cut at the
    banks so channel and floodplains convey separately (divided-channel method).

    Args:
        d (np.ndarray): Transverse offsets (m), increasing.
        z (np.ndarray): Bed elevation (m).
        eta (float): Water-surface elevation (m).
        compound (bool): Use the conveyance-equivalent perimeter.
        main_only (bool): Keep only the path with the thalweg (used where a
            section cuts a tributary mouth, which already enters as inflow).
        storage_threshold (float): Conveyance share below which a path only
            stores water.
        full (bool): Also return the conveying area and the Boussinesq
            coefficient.
        z_bank (float, optional): Bank level for the divided-channel method.

    Returns:
        tuple: ``(A, W, P)``, or with ``full`` ``(A_total, W_total, P_conv,
        A_conv, beta)`` where ``beta`` is the momentum (Boussinesq)
        coefficient over the conveying paths (1 for a single path).
    """
    if np.all(eta - z <= 0):
        return (0.0, 0.0, 0.0, 0.0, 1.0) if full else (0.0, 0.0, 0.0)

    A, W, P = _segment_properties(d, z, eta)
    wet = A > 0
    if not wet.any():
        return (0.0, 0.0, 0.0, 0.0, 1.0) if full else (0.0, 0.0, 0.0)

    # Split into connected flow paths. Two wet segments belong to the same
    # path only if the point between them is under water.
    y = eta - z
    joined = wet[:-1] & wet[1:] & (y[1:-1] > 0)
    breaks = np.flatnonzero(~joined) + 1
    paths = [p for p in np.split(np.arange(A.size), breaks) if wet[p[0]]]
    paths = [p[wet[p]] for p in paths]
    paths = [p for p in paths if p.size]
    seg_deep = int(np.argmin(np.minimum(z[:-1], z[1:])))

    # divided-channel method, only when the water is over the bank
    if z_bank is not None and eta > z_bank and len(paths) >= 1:
        paths = _split_at_banks(paths, bankfull_segments(d, z, z_bank))

    if main_only and len(paths) > 1:
        paths = [p for p in paths if p[0] <= seg_deep <= p[-1]] or paths[:1]
        keep = np.concatenate(paths)
        A, W, P = A[keep], W[keep], P[keep]
        paths = [np.arange(A.size)]
        seg_deep = 0

    A_tot, W_tot = float(A.sum()), float(W.sum())
    if A_tot <= 0.0:
        return (0.0, 0.0, 0.0, 0.0, 1.0) if full else (0.0, 0.0, 0.0)
    if not (compound or full):
        return A_tot, W_tot, float(P.sum())

    # per path: area, perimeter, conveyance (without the common 1/n)
    parts = []
    for part in paths:
        a, p = float(A[part].sum()), float(P[part].sum())
        if a <= 0 or p <= 0:
            continue
        parts.append((a, p, a * (a / p) ** (2.0 / 3.0),
                      bool(part[0] <= seg_deep <= part[-1])))
    if not parts:
        return (A_tot, W_tot, float(P.sum()), A_tot, 1.0) if full \
            else (A_tot, W_tot, float(P.sum()))

    K_all = sum(k for _, _, k, _ in parts)
    if not full:
        if K_all <= 0:
            return A_tot, W_tot, float(P.sum())
        return A_tot, W_tot, A_tot / max((K_all / A_tot) ** 1.5, 1e-6)

    # the conveying paths
    if storage_threshold > 0.0 and K_all > 0.0 and len(parts) > 1:
        conv = [q for q in parts if q[3] or q[2] / K_all >= storage_threshold]
    else:
        conv = parts
    if not conv:                                   # keep at least one path
        conv = [max(parts, key=lambda q: q[2])]

    A_c = sum(a for a, _, _, _ in conv)
    K_c = sum(k for _, _, k, _ in conv)
    if A_c <= 0 or K_c <= 0:
        return A_tot, W_tot, float(P.sum()), A_tot, 1.0

    if compound:
        P_c = A_c / max((K_c / A_c) ** 1.5, 1e-6)
    else:
        P_c = sum(p for _, p, _, _ in conv)
    # Boussinesq coefficient: u_i ~ K_i/A_i, so beta = A_c sum(K_i^2/A_i) / K_c^2
    beta = A_c * sum(k * k / a for a, _, k, _ in conv) / (K_c * K_c)
    return A_tot, W_tot, P_c, A_c, float(np.clip(beta, 1.0, 3.0))


def _despike_along_x(field: np.ndarray, window: int, tol: float):
    """
    Replace along-channel outliers of a tabulated field by the local median.

    Short blocks of sections where the survey line clips a side arm or a dock
    can double the area. Cells that differ from a long rolling median by more
    than ``tol`` (relative) are replaced; the rest are left alone.

    Args:
        field (np.ndarray): Table of shape ``(n_eta, n_x)``.
        window (int): Rolling-median window (number of sections).
        tol (float): Relative deviation that marks an outlier.

    Returns:
        tuple: ``(corrected field, share of cells replaced)``.
    """
    df = pd.DataFrame(field.T)                      # rows = chainage
    med = df.rolling(window, center=True, min_periods=1).median()
    dev = (df - med).abs() / med.clip(lower=1e-6)
    bad = dev > tol
    out = df.mask(bad, med)
    # a short mean around the repaired cells only, so they join their
    # neighbours smoothly; clean data are returned unchanged
    near = bad | bad.shift(1, fill_value=False) | bad.shift(-1, fill_value=False)
    out = out.where(~near, out.rolling(3, center=True, min_periods=1).mean())
    return out.to_numpy().T, float(bad.to_numpy().mean())


def nodata_mask(z: np.ndarray, min_count: int = 3,
                require_interior: bool = True):
    """
    Find the raster no-data fill inside a transect.

    A transect cut from a DEM carries the no-data value wherever the line
    leaves the raster coverage, usually as a constant above any real ground,
    so it shows up as the transect maximum, repeated. A run of the maximum
    with lower valid ground on both sides cannot be real terrain; that is the
    test. ``require_interior=False`` drops it once a file is known to use a
    fill (then runs at the ends of a transect are the same fill).

    Args:
        z (np.ndarray): Elevations along the transect (m).
        min_count (int): Minimum repeats of the maximum.
        require_interior (bool): Require a run bounded by valid points.

    Returns:
        np.ndarray: Boolean mask of the fill points (all False if none).
    """
    z = np.asarray(z, float)
    if z.size < 3:
        return np.zeros(z.size, dtype=bool)
    s = z == np.nanmax(z)
    if int(s.sum()) < min_count:
        return np.zeros(z.size, dtype=bool)
    if not require_interior:
        return s
    # a run of the maximum with valid points on both sides
    pad = np.concatenate([[False], s, [False]])
    starts = np.flatnonzero(pad[1:] & ~pad[:-1])
    ends = np.flatnonzero(pad[:-1] & ~pad[1:])          # exclusive
    if np.any((starts > 0) & (ends < z.size)):
        return s
    return np.zeros(z.size, dtype=bool)


def bridge_nodata(d: np.ndarray, z: np.ndarray, bad: np.ndarray,
                  max_gap: float = 40.0):
    """
    Interpolate across short interior no-data gaps; leave long gaps flagged.

    Args:
        d (np.ndarray): Transverse offsets (m).
        z (np.ndarray): Elevations (m).
        bad (np.ndarray): No-data mask.
        max_gap (float): Widest gap that is interpolated (m).

    Returns:
        tuple: ``(z with short gaps filled, mask of what is still missing)``.
    """
    z = np.asarray(z, float).copy()
    still = np.asarray(bad, bool).copy()
    if not still.any():
        return z, still
    pad = np.concatenate([[False], still, [False]])
    starts = np.flatnonzero(pad[1:] & ~pad[:-1])
    ends = np.flatnonzero(pad[:-1] & ~pad[1:])          # exclusive
    for a, b in zip(starts, ends):
        if a == 0 or b >= z.size:                       # not interior
            continue
        if d[b] - d[a - 1] > max_gap:
            continue
        z[a:b] = np.interp(d[a:b], [d[a - 1], d[b]], [z[a - 1], z[b]])
        still[a:b] = False
    return z, still


def trace_channel(d: np.ndarray, z: np.ndarray, bad: np.ndarray,
                  min_points: int = 5, min_width: float = 20.0):
    """
    The surveyed channel: the run of valid points that contains the thalweg.

    Walks outwards from the deepest valid point and stops where the survey
    stops. What lies beyond a no-data block is unknown, so it is not used.

    Args:
        d (np.ndarray): Transverse offsets (m).
        z (np.ndarray): Elevations (m).
        bad (np.ndarray): Mask of missing points.
        min_points (int): Fewest points for a valid trace.
        min_width (float): Narrowest valid trace (m).

    Returns:
        tuple | None: Inclusive index range ``(lo, hi)``, or None when the run
        is too short (then the transect is kept as surveyed).
    """
    good = ~bad
    if not good.any():
        return None
    idx = np.flatnonzero(good)
    i0 = int(idx[np.argmin(z[idx])])
    lo = i0
    while lo - 1 >= 0 and good[lo - 1]:
        lo -= 1
    hi = i0
    while hi + 1 < z.size and good[hi + 1]:
        hi += 1
    if hi - lo + 1 < min_points or d[hi] - d[lo] < min_width:
        return None
    return lo, hi


def _trim_flat_ends(d: np.ndarray, z: np.ndarray):
    """
    Drop the constant-elevation padding a DEM clip leaves at both ends.

    Left in place it would act as a wide flat bench and make the top width
    jump when the water reaches it.

    Args:
        d (np.ndarray): Transverse offsets (m).
        z (np.ndarray): Elevations (m).

    Returns:
        tuple: ``(d, z, number of points removed)``.
    """
    n = z.size
    lo = 0
    while lo + 1 < n and abs(z[lo + 1] - z[0]) < 1e-9:
        lo += 1
    hi = n - 1
    while hi - 1 > lo and abs(z[hi - 1] - z[-1]) < 1e-9:
        hi -= 1
    trimmed = lo + (n - 1 - hi)
    if hi - lo < 1:                                   # degenerate: keep as is
        return d, z, 0
    return d[lo:hi + 1], z[lo:hi + 1], trimmed


class CrossSectionLibrary:
    """
    Reads a bathymetry CSV and tabulates ``A``, ``B`` and ``P`` on a grid of
    water level and chainage.

    Expected columns:

    - ``X_m``: chainage from the upstream end of the reach (m, increasing
      downstream);
    - ``d_0 .. d_N``: transverse offset across the section (m);
    - ``h_0 .. h_N``: bed elevation at that offset, in the survey datum (m).

    Rows may have different numbers of valid points; empty cells are dropped.
    """

    def __init__(self, csv_path: str, datum_shift: float = 0.0,
                 wall_height: float = 10.0, x_col: str = "X_m",
                 clean: bool = True, closure_window_m: float = 2000.0,
                 floodplain_slope: float | None = None, compound: bool = True,
                 smooth_window_m: float = 1500.0, despike_tol: float = 0.25,
                 confluences=(), storage_threshold: float = 0.0,
                 lowpass_m: float = 0.0, compound_channel: bool = False,
                 bankfull_smooth_m: float = 5000.0, trace_edges: bool = True,
                 nodata_min_count: int = 3, nodata_bridge_m: float = 40.0,
                 flood_barrier: bool = False,
                 barrier_freeboard: float = 0.0,
                 bankline_window_m: float = 2000.0):
        """
        Args:
            csv_path (str): Bathymetry CSV.
            datum_shift (float): Added to all bed elevations (m).
            wall_height (float): Height of the vertical wall that closes each
                section above the highest level (m).
            x_col (str): Chainage column.
            clean (bool): Trim flat padding and close truncated sections at a
                reach-consistent level.
            closure_window_m (float): Window of the rolling median of the
                section tops used as closure level (m).
            floodplain_slope (float, optional): Horizontal:vertical slope of
                the banks added to truncated sections. ``None`` measures it
                from the survey.
            compound (bool): Use the conveyance-equivalent perimeter.
            smooth_window_m (float): Window of the along-channel despiking (m).
            despike_tol (float): Relative deviation that marks an outlier.
            confluences (list): ``(x, sigma)`` of each tributary mouth.
            storage_threshold (float): See ``section_properties``.
            lowpass_m (float): Along-channel smoothing length of A, P (m).
            compound_channel (bool): Divided-channel method at bank level.
            bankfull_smooth_m (float): Window to make the bank level
                consistent along the reach (m).
            trace_edges (bool): Remove raster no-data fill from transects.
            nodata_min_count (int): See ``nodata_mask``.
            nodata_bridge_m (float): See ``bridge_nodata``.
            flood_barrier (bool): Close truncated sections with a levee on
                the last surveyed point instead of extended banks.
            barrier_freeboard (float): Levee crest above the bank line (m).
            bankline_window_m (float): Window of the bank line median (m).
        """
        if not os.path.exists(csv_path):
            here = os.path.dirname(csv_path)
            tif = sorted(f for f in os.listdir(here) if f.endswith(".tif")) \
                if os.path.isdir(here) else []
            shp = sorted(f for f in os.listdir(here) if f.endswith(".shp")) \
                if os.path.isdir(here) else []
            hint = ""
            if tif and shp:
                hint = (f"\n\nA raster and a centreline are next to it; build the "
                        f"bathymetry file with:\n\n"
                        f"    python tools/raster_to_transects.py \\\n"
                        f"        --raster {os.path.join(here, tif[0])} \\\n"
                        f"        --centerline {os.path.join(here, shp[0])} \\\n"
                        f"        --out {csv_path} --reverse\n")
            raise FileNotFoundError(f"bathymetry file not found: {csv_path}{hint}")
        df = pd.read_csv(csv_path)
        if x_col not in df.columns:
            raise ValueError(f"{csv_path}: missing chainage column '{x_col}'")

        d_cols = sorted([c for c in df.columns if c.startswith("d_")],
                        key=lambda c: int(c.split("_")[1]))
        z_cols = sorted([c for c in df.columns if c.startswith("h_")],
                        key=lambda c: int(c.split("_")[1]))
        if not d_cols or len(d_cols) != len(z_cols):
            raise ValueError(
                f"{csv_path}: expected matching d_0..d_N and h_0..h_N columns, "
                f"found {len(d_cols)} and {len(z_cols)}"
            )

        df = df.sort_values(x_col)
        self.x = df[x_col].to_numpy(float)
        if np.any(np.diff(self.x) <= 0):
            raise ValueError(f"{csv_path}: chainage must be strictly increasing")

        self.compound = compound
        self.storage_threshold = float(storage_threshold)
        self.lowpass_m = float(lowpass_m)
        self.smooth_window_m = smooth_window_m
        self.despike_tol = despike_tol
        self.floodplain_slope = floodplain_slope     # may be measured below
        self.confluences = [(float(cx), float(cs)) for cx, cs in confluences]
        self.trace_edges = bool(trace_edges)
        self.compound_channel = bool(compound_channel)
        self.bankfull_smooth_m = float(bankfull_smooth_m)
        self.z_bankfull = None
        self.flood_barrier = bool(flood_barrier)
        self.barrier_freeboard = float(barrier_freeboard)

        raw: list[tuple[np.ndarray, np.ndarray]] = []
        n_padded = n_fill = n_traced = n_trace_failed = n_bridged = 0
        for _, row in df.iterrows():
            d = row[d_cols].to_numpy(float)
            z = row[z_cols].to_numpy(float) + datum_shift
            m = np.isfinite(d) & np.isfinite(z)
            d, z = d[m], z[m]
            if d.size < 2:
                raise ValueError(f"{csv_path}: section at x={row[x_col]} has <2 points")
            order = np.argsort(d)
            d, z = d[order], z[order]
            if clean:
                d, z, trimmed = _trim_flat_ends(d, z)
                n_padded += trimmed
            raw.append((d, z))

        # Trace the river edges. Pass one counts the transects with a fill run
        # inside them. Only when at least 10 % do is the file taken to use a
        # fill, and pass two also removes fill runs at the transect ends. A
        # survey without a fill is left unchanged.
        if self.trace_edges:
            interior = [nodata_mask(z, nodata_min_count).any() for _, z in raw]
            self.report_fill_fraction = float(np.mean(interior))
            if self.report_fill_fraction >= 0.10:
                for i, (d, z) in enumerate(raw):
                    bad = nodata_mask(z, nodata_min_count, require_interior=False)
                    if not bad.any():
                        continue
                    z, still = bridge_nodata(d, z, bad, nodata_bridge_m)
                    n_bridged += int(bad.sum() - still.sum())
                    span = trace_channel(d, z, still)
                    if span is None:
                        n_trace_failed += 1
                        continue
                    lo, hi = span
                    n_fill += int(bad.sum())
                    n_traced += 1
                    raw[i] = (d[lo:hi + 1], z[lo:hi + 1])
        else:
            self.report_fill_fraction = 0.0

        self.z_bed = np.array([z.min() for _, z in raw])
        z_top_raw = self.z_top_raw = np.array([z.max() for _, z in raw])
        self.width_surveyed = np.array([d[-1] - d[0] for d, _ in raw])

        # Bank slope for truncated sections, measured from the outer 50 m of
        # every surveyed bank (reach median), unless given.
        if self.floodplain_slope is None:
            grad = []
            for d, z in raw:
                if d.size < 6:
                    continue
                for sel, sgn in (((d <= d[0] + 50.0), -1.0),
                                 ((d >= d[-1] - 50.0), +1.0)):
                    if sel.sum() >= 3:
                        grad.append(sgn * np.polyfit(d[sel], z[sel], 1)[0])
            rise = float(np.median([g for g in grad if g > 1e-3])) if grad else 0.0
            self.floodplain_slope = float(np.clip(1.0 / rise, 2.0, 100.0)) \
                if rise > 1e-3 else 25.0
            self.report_slope_measured = True
        else:
            self.floodplain_slope = float(self.floodplain_slope)
            self.report_slope_measured = False

        # bank line: surveyed ground level, smoothed along the reach
        if self.x.size > 2:
            wb = max(int(round(bankline_window_m / max(np.median(np.diff(self.x)), 1.0))), 3)
            self.z_bankline = (pd.Series(z_top_raw).rolling(wb, center=True,
                                                            min_periods=1)
                               .median().to_numpy())
        else:
            self.z_bankline = z_top_raw.copy()

        # Closure level consistent along the reach: a rolling median of the
        # section tops, so a section clipped below dry ground is closed like
        # its neighbours instead of at its own (arbitrary) top.
        if clean:
            win = max(int(round(closure_window_m / max(np.median(np.diff(self.x)), 1.0))), 3)
            envelope = (pd.Series(z_top_raw).rolling(win, center=True, min_periods=1)
                        .median().to_numpy())
            self.z_close = np.maximum(z_top_raw, envelope)
        else:
            self.z_close = z_top_raw
        self.truncated = self.z_close > z_top_raw + 1e-6

        # sections that cut a tributary mouth
        self.at_confluence = np.zeros(len(raw), dtype=bool)
        for cx, cs in self.confluences:
            self.at_confluence |= np.abs(self.x - cx) <= max(2.0 * cs, 150.0)

        self.sections = raw
        self.z_bank = self.z_close
        self.wall_height = wall_height
        self.x_min, self.x_max = float(self.x[0]), float(self.x[-1])
        self.length = self.x_max - self.x_min
        self.report = {
            "sections": len(raw),
            "padded_points_removed": int(n_padded),
            "truncated_sections": int(self.truncated.sum()),
            "confluence_sections": int(self.at_confluence.sum()),
            "closure_raised_by_max_m": float((self.z_close - z_top_raw).max()),
            "floodplain_slope": float(self.floodplain_slope),
            "floodplain_slope_measured": bool(self.report_slope_measured),
        }
        if self.trace_edges:
            self.report.update({
                "nodata_points_removed": int(n_fill),
                "nodata_points_bridged": int(n_bridged),
                "nodata_transect_fraction": self.report_fill_fraction,
                "traced_sections": int(n_traced),
                "trace_rejected_sections": int(n_trace_failed),
                "traced_width_median_m": float(np.median(self.width_surveyed)),
            })
        if self.flood_barrier:
            self.report["barrier_freeboard_m"] = self.barrier_freeboard
            self.report["barrier_crest_median_m"] = float(
                np.median(self.z_bankline + self.barrier_freeboard))

    # --------------------------------------------------------------- bankfull
    def detect_bankfull(self, eta_min: float, eta_max: float, n: int = 56):
        """
        Bank level of each section, from its own width curve.

        The bank is where the top width stops growing with the banks and
        starts growing over the floodplain: the level of the largest relative
        widening per metre. The levels are then made consistent along the reach
        (rolling median over ``bankfull_smooth_m``; detections more than 2 m
        from it are replaced).

        Args:
            eta_min (float): Lowest level considered (m).
            eta_max (float): Highest level considered (m).
            n (int): Number of levels sampled.

        Returns:
            np.ndarray | None: Bank level of each section (m).
        """
        grid = np.linspace(eta_min, eta_max, n)
        raw = np.full(len(self.sections), np.nan)
        for ix, (d, z) in enumerate(self.sections):
            lo = max(float(z.min()), eta_min)
            hi = min(float(np.percentile(z, 75)), eta_max)
            if hi - lo < 1.0:
                continue
            g = np.linspace(lo + 0.05 * (hi - lo), hi, n)
            W = np.array([section_properties(d, z, e)[1] for e in g])
            ok = W > 1.0
            if ok.sum() < 4:
                continue
            g, W = g[ok], W[ok]
            rel = np.diff(W) / np.maximum(W[:-1], 1.0) / np.maximum(np.diff(g), 1e-6)
            raw[ix] = g[int(np.argmax(rel))]

        if np.isnan(raw).all():
            self.z_bankfull = None
            return None
        s = pd.Series(raw).interpolate(limit_direction="both")
        w = max(int(round(self.bankfull_smooth_m
                          / max(np.median(np.diff(self.x)), 1.0))) | 1, 3)
        sm = s.rolling(w, center=True, min_periods=1).median().to_numpy()
        out = np.where(np.abs(s.to_numpy() - sm) > 2.0, sm, s.to_numpy())
        self.z_bankfull = out
        self.report["bankfull_median_m"] = float(np.median(out))
        self.report["bankfull_replaced_frac"] = float(
            np.mean(np.abs(s.to_numpy() - sm) > 2.0))
        return out

    # ------------------------------------------------------------------ shape
    def _extended(self, i: int, eta_top: float):
        """
        Section ``i``, closed above the survey so it can hold any modelled level.

        A section that reaches dry ground above ``eta_top`` is only walled. A
        truncated section gets banks extended at the floodplain slope up to the
        reach closure level (upper bound on storage). With ``flood_barrier`` a
        levee stands on the last surveyed point instead, its crest
        ``barrier_freeboard`` above the bank line (lower bound on storage);
        above the crest the extended banks resume.

        Args:
            i (int): Section index.
            eta_top (float): Highest water level of the table (m).

        Returns:
            tuple: ``(d, z)`` of the closed section.
        """
        d, z = self.sections[i]
        if not hasattr(self, "z_close"):
            return d, z

        # the survey already reaches above the table: just wall it
        if float(z.max()) >= eta_top:
            top = float(z.max()) + self.wall_height
            return (np.concatenate([[d[0]], d, [d[-1]]]),
                    np.concatenate([[top], z, [top]]))

        z_c = min(float(self.z_close[i]), float(eta_top))
        crest = (max(float(self.z_bankline[i]) + self.barrier_freeboard,
                     float(z[0]), float(z[-1]))
                 if self.flood_barrier else float("-inf"))
        top = max(eta_top, z_c, crest if self.flood_barrier else z_c) \
            + self.wall_height
        s = self.floodplain_slope

        left_d, left_z, right_d, right_z = [], [], [], []
        if z[0] < z_c - 1e-9 and z_c > crest + 1e-9:
            # truncated on this side: extend the bank at the floodplain slope,
            # above the levee crest only
            left_d.append(d[0] - s * (z_c - z[0]))
            left_z.append(z_c)
            if crest > z[0] + 1e-9:
                left_d.append(d[0] - s * (crest - z[0]))
                left_z.append(crest)
        if self.flood_barrier:
            left_d.append(d[0])
            left_z.append(max(crest, float(z[0])))
        if z[-1] < z_c - 1e-9 and z_c > crest + 1e-9:
            if crest > z[-1] + 1e-9:
                right_d.append(d[-1] + s * (crest - z[-1]))
                right_z.append(crest)
            right_d.append(d[-1] + s * (z_c - z[-1]))
            right_z.append(z_c)
        if self.flood_barrier:
            right_d.insert(0, d[-1])
            right_z.insert(0, max(crest, float(z[-1])))

        d_ext = np.concatenate([[(left_d or [d[0]])[0]], left_d, d, right_d,
                                [(right_d or [d[-1]])[-1]]])
        z_ext = np.concatenate([[top], left_z, z, right_z, [top]])
        return d_ext, z_ext

    # ------------------------------------------------------------------ table
    def tabulate(self, eta_min: float, eta_max: float, n_eta: int = 192,
                 smooth: bool | None = None):
        """
        Tabulate the section properties on a grid of water level and chainage.

        After tabulation, along-channel outliers of ``A`` and ``P`` are removed
        (median despiking) and, with ``lowpass_m``, the table is smoothed along
        the channel. ``B`` is then recomputed as ``dA/d(eta)`` so that identity
        holds exactly in the table.

        Args:
            eta_min (float): Lowest water level (m).
            eta_max (float): Highest water level (m).
            n_eta (int): Number of water levels.
            smooth (bool, optional): Apply the despiking (default: when
                ``smooth_window_m > 0``).

        Returns:
            tuple: ``(eta_grid, table)`` with ``table`` of shape
            ``(5, n_eta, n_x)``: area, top width, wetted perimeter, conveying
            area and Boussinesq coefficient.
        """
        smooth = self.smooth_window_m > 0 if smooth is None else smooth
        eta_grid = np.linspace(eta_min, eta_max, n_eta)
        table = np.zeros((5, n_eta, self.x.size))
        if self.compound_channel and self.z_bankfull is None:
            self.detect_bankfull(eta_min, eta_max)
        # truncated = the survey does not reach the top of the table
        self.truncated = np.array([float(z.max()) < eta_max
                                   for _, z in self.sections])
        self.report["truncated_sections"] = int(self.truncated.sum())
        self.report["closure_raised_by_max_m"] = float(
            np.max(np.where(self.truncated,
                            np.minimum(self.z_close, eta_max) - self.z_top_raw,
                            0.0))
            if self.truncated.any() else 0.0)
        for ix in range(len(self.sections)):
            d, z = self._extended(ix, eta_max)
            main_only = bool(self.at_confluence[ix])
            z_bank = (float(self.z_bankfull[ix])
                      if self.compound_channel and self.z_bankfull is not None
                      else None)
            for ie, eta in enumerate(eta_grid):
                A, W, P, A_c, beta = section_properties(
                    d, z, eta, compound=self.compound, main_only=main_only,
                    storage_threshold=self.storage_threshold, full=True,
                    z_bank=z_bank)
                table[0, ie, ix] = A
                table[1, ie, ix] = W
                table[2, ie, ix] = P
                table[3, ie, ix] = A_c
                table[4, ie, ix] = beta

        if smooth and self.x.size > 4:
            dx = max(float(np.median(np.diff(self.x))), 1.0)
            win = int(round(self.smooth_window_m / dx))
            win = max(win + (win + 1) % 2, 3)          # odd, at least 3
            for ch, name in ((0, "A"), (2, "P"), (3, "Aconv")):
                table[ch], n_fix = _despike_along_x(table[ch], win,
                                                    self.despike_tol)
                self.report[f"despiked_{name}_frac"] = n_fix
            # A must not decrease with water level
            table[0] = np.maximum.accumulate(table[0], axis=0)
            table[3] = np.maximum.accumulate(table[3], axis=0)
            d_eta = eta_grid[1] - eta_grid[0]
            table[1] = np.gradient(table[0], d_eta, axis=0).clip(min=0.0)
            table[2] = np.maximum(table[2], 1e-3)
            # the conveying area is part of the total area
            table[3] = np.minimum(table[3], table[0])
            table[4] = table[4].clip(1.0, 3.0)

        if self.lowpass_m > 0 and self.x.size > 4:
            # Smooth A, P and the conveying area along the channel at the
            # length scale below which the 1-D equations do not apply anyway.
            # Two passes of a moving mean, close to a Gaussian.
            dx = max(float(np.median(np.diff(self.x))), 1.0)
            w = max(int(round(self.lowpass_m / dx)) | 1, 3)
            var0 = float(np.var(table[0], axis=1).sum())
            for ch in (0, 2, 3):
                df = pd.DataFrame(table[ch].T)
                sm = df.rolling(w, center=True, min_periods=1).mean()
                sm = sm.rolling(w, center=True, min_periods=1).mean()
                table[ch] = sm.to_numpy().T
            table[0] = np.maximum.accumulate(table[0], axis=0)
            table[3] = np.minimum(np.maximum.accumulate(table[3], axis=0), table[0])
            d_eta = eta_grid[1] - eta_grid[0]
            table[1] = np.gradient(table[0], d_eta, axis=0).clip(min=0.0)
            table[2] = np.maximum(table[2], 1e-3)
            var1 = float(np.var(table[0], axis=1).sum())
            self.report["lowpass_m"] = float(self.lowpass_m)
            self.report["lowpass_var_kept"] = var1 / max(var0, 1e-12)
            wetted = table[0] > 1.0
            self.report["lowpass_R_median"] = float(np.median(
                (table[0][wetted] / table[2][wetted]))) if wetted.any() else 0.0

        if self.storage_threshold > 0:
            wet = table[0] > 1.0
            self.report["storage_fraction_mean"] = float(
                (1.0 - table[3][wet] / table[0][wet]).mean()) if wet.any() else 0.0
            self.report["beta_max"] = float(table[4].max())

        return eta_grid, table


# ---------------------------------------------------------------------------
# Lookup in the table (reference for the surrogate)
# ---------------------------------------------------------------------------
class RiverGeometry(nn.Module):
    """Bilinear interpolation in the tabulated geometry; also holds the bed level."""

    def __init__(self, lib: CrossSectionLibrary, eta_grid: np.ndarray,
                 table: np.ndarray):
        """
        Args:
            lib (CrossSectionLibrary): The cross sections.
            eta_grid (np.ndarray): Water levels of the table (m).
            table (np.ndarray): Table from ``CrossSectionLibrary.tabulate``.
        """
        super().__init__()

        def buf(a):
            # an independent copy, so later changes to the arrays do not leak in
            return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64).copy())

        self.register_buffer("table", buf(table))
        self.register_buffer("x_nodes", buf(lib.x))
        self.register_buffer("eta_nodes", buf(eta_grid))
        self.register_buffer("z_bed_nodes", buf(lib.z_bed))
        self.x_min, self.x_max = lib.x_min, lib.x_max
        self.eta_min, self.eta_max = float(eta_grid[0]), float(eta_grid[-1])
        self.n_eta, self.n_x = table.shape[1], table.shape[2]

    def bed_elevation(self, x: torch.Tensor) -> torch.Tensor:
        """Lowest bed elevation (thalweg) at chainage ``x`` (m)."""
        return interp1d(x, self.x_nodes, self.z_bed_nodes)

    def forward(self, x: torch.Tensor, eta: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Chainage (m).
            eta (torch.Tensor): Water level (m).

        Returns:
            tuple: ``(A, B, P, A_conv, beta)`` at each point, shaped like ``x``.
        """
        xi = (x.reshape(-1) - self.x_min) / (self.x_max - self.x_min) * (self.n_x - 1)
        ei = ((eta.reshape(-1).clamp(self.eta_min, self.eta_max) - self.eta_min)
              / (self.eta_max - self.eta_min) * (self.n_eta - 1))
        xi = xi.clamp(0.0, self.n_x - 1.0)

        x0 = torch.clamp(torch.floor(xi), 0, self.n_x - 2).long()
        e0 = torch.clamp(torch.floor(ei), 0, self.n_eta - 2).long()
        wx = (xi - x0.to(xi.dtype)).unsqueeze(0)
        we = (ei - e0.to(ei.dtype)).unsqueeze(0)

        v00 = self.table[:, e0, x0]
        v01 = self.table[:, e0, x0 + 1]
        v10 = self.table[:, e0 + 1, x0]
        v11 = self.table[:, e0 + 1, x0 + 1]
        v = ((v00 * (1 - wx) + v01 * wx) * (1 - we)
             + (v10 * (1 - wx) + v11 * wx) * we).t()
        shape = x.shape
        out = [v[:, i:i + 1].reshape(shape) for i in range(v.shape[1])]
        return tuple(out)


# ---------------------------------------------------------------------------
# Smooth differentiable surrogate of the geometry
# ---------------------------------------------------------------------------
class GeometrySurrogate(nn.Module):
    """
    Neural network ``(x, eta) -> (A, P)``, fitted to the tabulated geometry.

    The top width is not an output: it is ``B = dA/d(eta)`` from automatic
    differentiation, so continuity and momentum share one geometry. ``A`` and
    ``P`` go through a softplus so they stay positive without a hard clamp.

    The network also has a third output, a direct estimate of the top width.
    It is fitted with the others during pre-training but the model does not
    use it; it is kept so that the published checkpoints load and training is
    reproduced exactly.

    With ``storage=True`` two more outputs give the conveying fraction
    ``phi = A_conv / A`` (sigmoid, between 0 and 1) and the Boussinesq
    coefficient ``beta = 1 + softplus(.)`` (at least 1).
    """

    def __init__(self, x_min, x_max, eta_min, eta_max, A_scale, P_scale,
                 hidden_dim: int = 128, n_layers: int = 5, n_fourier: int = 32,
                 max_period_ratio: float = 400.0, storage: bool = False):
        """
        Args:
            x_min, x_max (float): Chainage range (m).
            eta_min, eta_max (float): Water level range (m).
            A_scale, P_scale (float): Typical area (m2) and perimeter (m).
            hidden_dim (int): Width of the hidden layers.
            n_layers (int): Number of layers.
            n_fourier (int): Number of Fourier frequencies in chainage.
            max_period_ratio (float): Longest over shortest wavelength.
            storage (bool): Add the storage-split outputs.
        """
        super().__init__()
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.eta_min, self.eta_max = float(eta_min), float(eta_max)
        self.A_scale, self.P_scale = float(A_scale), float(P_scale)
        self.storage = bool(storage)

        # Fourier features in chainage: a plain MLP of x cannot follow geometry
        # that changes every 100 m along a long reach, so fixed frequencies on
        # a geometric ladder are added as inputs.
        self.n_fourier = int(n_fourier)
        if self.n_fourier > 0:
            k = torch.logspace(0, np.log10(max_period_ratio), self.n_fourier,
                               dtype=torch.float64)
            self.register_buffer("freqs", k * np.pi)
        d_in = 2 + 2 * self.n_fourier

        layers = []
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d_in, hidden_dim), nn.Tanh()]
            d_in = hidden_dim
        layers += [nn.Linear(d_in, 5 if self.storage else 3)]
        self.net = nn.Sequential(*layers)
        if self.storage:
            # start with phi ~ 1 and beta ~ 1, so the split changes nothing
            # until the fit says so
            last = self.net[-1]
            with torch.no_grad():
                last.bias[3] = 4.0                   # sigmoid(4) ~ 0.982
                last.bias[4] = -6.0                  # 1 + softplus(-6) ~ 1.002

    def _norm(self, x, eta):
        """Scale the inputs to [-1, 1] and add the Fourier features."""
        xn = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        en = 2.0 * (eta - self.eta_min) / (self.eta_max - self.eta_min) - 1.0
        feats = [xn, en]
        if self.n_fourier > 0:
            arg = xn * self.freqs
            feats += [torch.sin(arg), torch.cos(arg)]
        return torch.cat(feats, dim=-1)

    def forward(self, x: torch.Tensor, eta: torch.Tensor):
        """Wetted area and wetted perimeter at ``(x, eta)``."""
        A, P, _ = self.heads(x, eta)
        return A, P

    def heads(self, x: torch.Tensor, eta: torch.Tensor):
        """
        All three outputs.

        Returns:
            tuple: ``(A, P, B_direct)``; the model uses ``B = dA/d(eta)``,
            not ``B_direct``.
        """
        out = self.net(self._norm(x, eta))
        A = self.A_scale * F.softplus(out[..., 0:1]) + 1e-3
        P = self.P_scale * F.softplus(out[..., 1:2]) + 1e-2
        B_direct = self.P_scale * F.softplus(out[..., 2:3]) + 1e-3
        return A, P, B_direct

    def storage_split(self, x: torch.Tensor, eta: torch.Tensor):
        """
        Conveying fraction and Boussinesq coefficient.

        Returns:
            tuple: ``(phi, beta)``, or ``(None, None)`` when the split is off.
        """
        if not self.storage:
            return None, None
        out = self.net(self._norm(x, eta))
        phi = torch.sigmoid(out[..., 3:4]).clamp_min(1e-3)
        beta = 1.0 + F.softplus(out[..., 4:5])
        return phi, beta

    def with_width(self, x: torch.Tensor, eta: torch.Tensor,
                   create_graph: bool = True):
        """
        Area, perimeter and top width ``B = dA/d(eta)``.

        Args:
            x (torch.Tensor): Chainage (m).
            eta (torch.Tensor): Water level (m).
            create_graph (bool): Keep the graph of ``B`` for higher derivatives.

        Returns:
            tuple: ``(A, P, B)``.
        """
        if eta.requires_grad:
            A, P, _ = self.heads(x, eta)
            B = torch.autograd.grad(A, eta, torch.ones_like(A),
                                    create_graph=create_graph,
                                    retain_graph=True)[0]
            return A, P, B

        # prediction without a graph: B is still dA/d(eta), computed apart
        A, P, _ = self.heads(x, eta)
        with torch.enable_grad():
            e = eta.detach().requires_grad_(True)
            A_probe, _, _ = self.heads(x.detach(), e)
            B = torch.autograd.grad(A_probe, e, torch.ones_like(A_probe))[0].detach()
        return A, P, B


def pretrain_surrogate(surrogate: GeometrySurrogate, geometry: RiverGeometry,
                       steps: int = 6000, batch: int = 4096, lr: float = 2e-3,
                       verbose: bool = True):
    """
    Fit the surrogate to the tabulated geometry, then freeze it.

    ``A``, ``P`` and ``dA/d(eta)`` are all fitted: fitting ``A`` alone would
    leave its slope, the top width ``B``, poorly constrained, and ``B`` is the
    coefficient of the storage term that carries the tide.

    Args:
        surrogate (GeometrySurrogate): The network to fit.
        geometry (RiverGeometry): The tabulated geometry (target).
        steps (int): Adam steps. ``0`` only freezes the network (its weights
            then come from a checkpoint).
        batch (int): Random points per step.
        lr (float): Learning rate.
        verbose (bool): Print the loss now and then.

    Returns:
        GeometrySurrogate: The fitted, frozen surrogate.
    """
    if steps <= 0:
        for p in surrogate.parameters():
            p.requires_grad_(False)
        return surrogate

    opt = torch.optim.Adam(surrogate.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 1e-2)
    dev = next(surrogate.parameters()).device

    x_lo, x_hi = geometry.x_min, geometry.x_max
    e_lo, e_hi = geometry.eta_min, geometry.eta_max
    A_s, P_s = surrogate.A_scale, surrogate.P_scale

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        x = torch.rand((batch, 1), device=dev) * (x_hi - x_lo) + x_lo
        eta = torch.rand((batch, 1), device=dev) * (e_hi - e_lo) + e_lo
        eta.requires_grad_(True)

        with torch.no_grad():
            target = geometry(x, eta)
        A_t, B_t, P_t = target[0], target[1], target[2]

        A_p, P_p, B_direct = surrogate.heads(x, eta)
        B_p = torch.autograd.grad(A_p, eta, torch.ones_like(A_p),
                                  create_graph=True)[0]

        loss = (((A_p - A_t) / A_s) ** 2).mean()
        loss = loss + (((P_p - P_t) / P_s) ** 2).mean()
        loss = loss + (((B_p - B_t) / max(P_s, 1.0)) ** 2).mean()
        # A must not decrease with water level
        loss = loss + 10.0 * (F.relu(-B_p) / max(P_s, 1.0)).pow(2).mean()
        # the direct top-width output (not used by the model)
        loss = loss + (((B_direct - B_t) / max(P_s, 1.0)) ** 2).mean()

        if surrogate.storage and len(target) >= 5:
            phi_t = target[3] / target[0].clamp_min(1e-6)
            phi_p, beta_p = surrogate.storage_split(x, eta)
            loss = loss + ((phi_p - phi_t.clamp(0.0, 1.0)) ** 2).mean()
            loss = loss + ((beta_p - target[4]) ** 2).mean()

        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % max(1, steps // 6) == 0 or step == steps - 1):
            print(f"    surrogate step {step:6d}  loss {loss.item():.3e}")

    for p in surrogate.parameters():
        p.requires_grad_(False)
    return surrogate


def surrogate_report(surrogate: GeometrySurrogate, geometry: RiverGeometry,
                     n: int = 20000) -> dict:
    """
    Mean relative errors of the fitted surrogate against the table.

    Args:
        surrogate (GeometrySurrogate): The fitted surrogate.
        geometry (RiverGeometry): The table.
        n (int): Number of random test points.

    Returns:
        dict: ``A_rel_err``, ``B_rel_err``, ``P_rel_err`` (and the storage
        errors when the split is on).
    """
    dev = next(surrogate.parameters()).device
    x = torch.rand((n, 1), device=dev) * (geometry.x_max - geometry.x_min) + geometry.x_min
    eta = torch.rand((n, 1), device=dev) * (geometry.eta_max - geometry.eta_min) + geometry.eta_min
    eta.requires_grad_(True)
    with torch.no_grad():
        target = geometry(x, eta)
    A_t, B_t, P_t = target[0], target[1], target[2]
    with torch.enable_grad():
        A_p, P_p = surrogate(x, eta)
        B_p = torch.autograd.grad(A_p, eta, torch.ones_like(A_p))[0]

    def rel(p, t, scale):
        return float((torch.abs(p - t) / (torch.abs(t) + scale)).mean())

    out = {
        "A_rel_err": rel(A_p.detach(), A_t, surrogate.A_scale * 0.05),
        "B_rel_err": rel(B_p.detach(), B_t, surrogate.P_scale * 0.05),
        "P_rel_err": rel(P_p.detach(), P_t, surrogate.P_scale * 0.05),
    }
    if surrogate.storage and len(target) >= 5:
        with torch.no_grad():
            phi_p, beta_p = surrogate.storage_split(x, eta)
        phi_t = (target[3] / target[0].clamp_min(1e-6)).clamp(0.0, 1.0)
        out["phi_rel_err"] = rel(phi_p, phi_t, 0.05)
        out["beta_rel_err"] = rel(beta_p, target[4], 0.05)
    return out


def build_geometry(cfg, verbose: bool = True, stage_range=None):
    """
    Build the geometry of a case: cross sections, table and surrogate.

    Args:
        cfg (CaseConfig): The case.
        verbose (bool): Print a summary.
        stage_range (tuple, optional): ``(min, max)`` water level the model
            will see. The table then covers that band (plus a margin) instead
            of the full height of the survey.

    Returns:
        tuple: ``(library, geometry, surrogate, info)``.
    """
    datum = cfg.datum_shift or 0.0
    confluences = [(t.x, float(t.sigma or cfg.physics.lateral_sigma))
                   for t in cfg.tributaries]
    p = cfg.physics
    storage_threshold = float(p.storage_threshold)
    lib = CrossSectionLibrary(cfg.path(cfg.geometry), datum_shift=datum,
                              storage_threshold=storage_threshold,
                              lowpass_m=float(p.geometry_smooth_m),
                              compound_channel=bool(p.compound_channel),
                              bankfull_smooth_m=float(p.bankfull_smooth_m),
                              trace_edges=bool(p.trace_edges),
                              nodata_min_count=int(p.nodata_min_count),
                              nodata_bridge_m=float(p.nodata_bridge_m),
                              flood_barrier=bool(p.flood_barrier),
                              barrier_freeboard=float(p.barrier_freeboard_m),
                              confluences=confluences)

    eta_min = float(lib.z_bed.min())
    if stage_range is not None:
        lo, hi = float(stage_range[0]), float(stage_range[1])
        margin = max(0.25 * (hi - lo), 2.0)
        eta_min = min(eta_min, lo - 0.5)
        eta_max = min(float(hi + margin), float(lib.z_bank.max() + 5.0))
    else:
        eta_max = float(np.percentile(lib.z_bank, 90) + 2.0)
    eta_max = max(eta_max, eta_min + 1.0)
    eta_grid, table = lib.tabulate(eta_min, eta_max)

    geometry = RiverGeometry(lib, eta_grid, table).to(DEVICE)

    # Typical magnitudes over cells that are clearly wet (more than 1 % of the
    # largest value), so smoothing slivers in dry cells do not pull them down.
    def _scale(ch):
        v = table[ch]
        m = v > 0.01 * v.max() if v.max() > 0 else np.zeros_like(v, dtype=bool)
        return float(np.percentile(v[m], 60)) if m.any() else 1.0

    A_scale, P_scale = _scale(0), _scale(2)
    m = cfg.model
    surrogate = GeometrySurrogate(
        lib.x_min, lib.x_max, eta_min, eta_max, A_scale, P_scale,
        hidden_dim=m.geo_hidden_dim, n_layers=m.geo_layers,
        n_fourier=m.geo_fourier, max_period_ratio=m.geo_max_period_ratio,
        storage=storage_threshold > 0.0,
    ).to(DEVICE)

    if verbose:
        print(f"  geometry: {len(lib.sections)} sections over "
              f"{lib.length/1000:.1f} km, bed {lib.z_bed.min():.2f}..{lib.z_bed.max():.2f} m, "
              f"stage table {eta_min:.2f}..{eta_max:.2f} m")
        r = lib.report
        bits = [f"{r['padded_points_removed']} padded points trimmed"]
        if r.get("nodata_points_removed"):
            bits.append(f"{r['nodata_points_removed']} no-data fill points in "
                        f"{r['traced_sections']} sections "
                        f"({r['nodata_points_bridged']} short gaps bridged, the "
                        f"rest traced out; median traced width "
                        f"{r['traced_width_median_m']:.0f} m)")
        bits.append(f"{r['truncated_sections']} truncated sections closed "
                    f"(up to +{r['closure_raised_by_max_m']:.1f} m) on a "
                    f"{r['floodplain_slope']:.0f}:1 bank slope "
                    f"{'measured from the survey' if r['floodplain_slope_measured'] else 'set by the case'}")
        if r.get("confluence_sections"):
            bits.append(f"{r['confluence_sections']} confluence sections "
                        f"restricted to the main channel")
        if "despiked_A_frac" in r:
            bits.append(f"{r['despiked_A_frac']:.1%} of A despiked")
        print("  cleaning: " + "; ".join(bits))
        if "barrier_crest_median_m" in r:
            print(f"  flood barrier: levee on the traced edge, crest "
                  f"{r['barrier_crest_median_m']:.2f} m (median, "
                  f"freeboard {r['barrier_freeboard_m']:+.2f} m); "
                  f"{'contained' if r['barrier_crest_median_m'] >= eta_max else 'overtopped'} "
                  f"at the top of the stage table ({eta_max:.2f} m)")
        if "lowpass_m" in r:
            print(f"  low-pass: A(x) smoothed at {r['lowpass_m']:.0f} m, "
                  f"{r['lowpass_var_kept']:.1%} of the variance of A kept, "
                  f"median R = A/P {r['lowpass_R_median']:.2f} m")
        if "bankfull_median_m" in r:
            print(f"  compound channel: divided at a measured bankfull of "
                  f"{r['bankfull_median_m']:.2f} m (median), "
                  f"{r['bankfull_replaced_frac']:.0%} of the per-section "
                  f"detections replaced by the reach profile")
        if "storage_fraction_mean" in r:
            print(f"  storage split: {r['storage_fraction_mean']:.1%} of the "
                  f"wetted area is storage (conveys nothing), peak beta "
                  f"{r['beta_max']:.2f}")

    info = {"A_scale": A_scale, "P_scale": P_scale,
            "eta_min": eta_min, "eta_max": eta_max, "cleaning": lib.report}
    return lib, geometry, surrogate, info
