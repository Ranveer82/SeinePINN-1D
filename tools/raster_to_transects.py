#!/usr/bin/env python
"""
Cut cross sections from a DEM along a centreline, for the bathymetry CSV of a case.

    python tools/raster_to_transects.py \\
        --raster data/seine/flood_plain_seine_compressed_DEFLATE2.tif \\
        --centerline data/seine/seine_centerline.shp \\
        --out data/seine/river_profiles_raster.csv \\
        --spacing 100 --half-width 500 --reverse

    # and compare with an older bathymetry file
    python tools/raster_to_transects.py ... --compare old_profiles.csv

Needs ``rasterio`` and ``scipy``; the model itself only reads the CSV.

What this does:

1. reads the centreline (a PolyLine shapefile, or a CSV of x, y);
2. puts transect points every ``--spacing`` metres along it, with chainage
   increasing downstream (``--reverse`` flips a line drawn the other way; the
   bed level at both ends is printed so the direction can be checked);
3. takes the transect direction from a tangent over ``--tangent-window``
   metres, so the small zigzags of a digitised line do not make neighbouring
   transects cut the channel at different angles;
4. samples the DEM along each transect (bilinear), out to ``--half-width`` on
   each side;
5. cuts a transect where it reaches another part of a meander (the nearest
   point of the centreline is then far along the line from the transect's own
   chainage), so the same water is not counted twice;
6. writes ``X_m, d_0..d_N, h_0..h_N``; points outside the raster or on its
   no-data value are left empty.
"""

from __future__ import annotations

import argparse
import os
import struct

import numpy as np
import pandas as pd


def _read_shp_polylines(path: str):
    """
    Read every polyline of an ESRI shapefile.

    The record layout is read directly, so no extra package is needed.
    Handles PolyLine, PolyLineZ and PolyLineM (the Z and M values are ignored);
    multi-part records come back as separate lines.

    Args:
        path (str): The ``.shp`` file.

    Returns:
        list[np.ndarray]: One ``(n, 2)`` array of x, y per line.
    """
    with open(path, "rb") as fh:
        blob = fh.read()
    if len(blob) < 100 or struct.unpack(">i", blob[:4])[0] != 9994:
        raise SystemExit(f"{path}: not an ESRI shapefile (bad magic number)")
    end = struct.unpack(">i", blob[24:28])[0] * 2      # file length, 16-bit words
    out, pos = [], 100
    while pos + 8 <= min(end, len(blob)):
        content = struct.unpack(">i", blob[pos + 4:pos + 8])[0] * 2
        rec, pos = blob[pos + 8:pos + 8 + content], pos + 8 + content
        if len(rec) < 4:
            continue
        shape_type = struct.unpack("<i", rec[:4])[0]
        if shape_type == 0:                            # null shape
            continue
        if shape_type not in (3, 13, 23):
            raise SystemExit(
                f"{path}: shape type {shape_type} is not a polyline. "
                f"Export the centreline as a PolyLine shapefile.")
        n_parts, n_pts = struct.unpack("<ii", rec[36:44])
        parts = np.frombuffer(rec, "<i4", count=n_parts, offset=44)
        xy = np.frombuffer(rec, "<f8", count=2 * n_pts,
                           offset=44 + 4 * n_parts).reshape(-1, 2)
        for a, b in zip(parts, list(parts[1:]) + [n_pts]):
            if b - a >= 2:
                out.append(np.array(xy[a:b], dtype=float))
    if not out:
        raise SystemExit(f"{path}: no polylines found")
    return out


def read_centerline(path: str) -> np.ndarray:
    """
    The centreline as one line of points, in the coordinates of the file.

    Several pieces are joined end to end (each turned to continue the
    previous one).

    Args:
        path (str): A ``.shp`` file, or a CSV with two coordinate columns.

    Returns:
        np.ndarray: ``(n, 2)`` points.
    """
    if path.lower().endswith(".shp"):
        pts = _read_shp_polylines(path)
    else:
        df = pd.read_csv(path)
        cols = [c for c in df.columns
                if c.lower() in ("x", "y", "easting", "northing", "lon", "lat")]
        if len(cols) < 2:
            cols = list(df.columns[:2])
        pts = [df[cols[:2]].to_numpy(float)]
    if len(pts) > 1:
        out = [pts[0]]
        for p in pts[1:]:
            tail = out[-1][-1]
            if np.hypot(*(p[-1] - tail)) < np.hypot(*(p[0] - tail)):
                p = p[::-1]
            out.append(p)
        pts = [np.vstack(out)]
    return pts[0]


def resample(line: np.ndarray, spacing: float):
    """
    Points at equal chainage along a line.

    Args:
        line (np.ndarray): ``(n, 2)`` points.
        spacing (float): Distance between points (m).

    Returns:
        tuple: ``(points, chainage of each point, total length)``.
    """
    seg = np.hypot(*np.diff(line, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    ch = np.arange(0.0, total + 0.5 * spacing, spacing)
    ch = ch[ch <= total]
    return np.column_stack([np.interp(ch, s, line[:, 0]),
                            np.interp(ch, s, line[:, 1])]), ch, total


def tangents(pts: np.ndarray, chain: np.ndarray, window: float):
    """
    Unit tangent at each point, taken over ``window`` metres of the line.

    Args:
        pts (np.ndarray): Points along the line.
        chain (np.ndarray): Their chainage (m).
        window (float): Length of line used for the direction (m).

    Returns:
        np.ndarray: ``(n, 2)`` unit vectors.
    """
    k = max(int(round(0.5 * window / max(np.median(np.diff(chain)), 1.0))), 1)
    lo = np.clip(np.arange(pts.shape[0]) - k, 0, pts.shape[0] - 1)
    hi = np.clip(np.arange(pts.shape[0]) + k, 0, pts.shape[0] - 1)
    t = pts[hi] - pts[lo]
    n = np.hypot(t[:, 0], t[:, 1])
    n[n < 1e-9] = 1.0
    return t / n[:, None]


def bilinear(arr: np.ndarray, transform, xs: np.ndarray, ys: np.ndarray,
             nodata: float):
    """
    Bilinear sample of a raster at map coordinates.

    Args:
        arr (np.ndarray): Raster values.
        transform: Affine transform of the raster (rasterio).
        xs, ys (np.ndarray): Map coordinates.
        nodata (float): No-data value of the raster.

    Returns:
        np.ndarray: Values; NaN outside the raster or on no-data cells.
    """
    inv = ~transform
    col, row = inv * (xs, ys)
    col = np.asarray(col) - 0.5
    row = np.asarray(row) - 0.5
    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    fc = col - c0
    fr = row - r0
    h, w = arr.shape
    out = np.full(xs.shape, np.nan)
    ok = (c0 >= 0) & (c0 < w - 1) & (r0 >= 0) & (r0 < h - 1)
    if not ok.any():
        return out
    c0o, r0o, fco, fro = c0[ok], r0[ok], fc[ok], fr[ok]
    q = np.stack([arr[r0o, c0o], arr[r0o, c0o + 1],
                  arr[r0o + 1, c0o], arr[r0o + 1, c0o + 1]]).astype(np.float64)
    q[q == nodata] = np.nan
    v = ((q[0] * (1 - fco) + q[1] * fco) * (1 - fro)
         + (q[2] * (1 - fco) + q[3] * fco) * fro)
    out[ok] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raster", required=True, help="DEM (GeoTIFF)")
    ap.add_argument("--centerline", required=True,
                    help="centreline (PolyLine .shp, or CSV of x, y)")
    ap.add_argument("--out", required=True, help="bathymetry CSV to write")
    ap.add_argument("--spacing", type=float, default=100.0,
                    help="distance between transects along the river (m)")
    ap.add_argument("--half-width", type=float, default=500.0,
                    help="transect half-width on each side of the centreline (m)")
    ap.add_argument("--step", type=float, default=None,
                    help="distance between points across the river (m); "
                         "default = the raster resolution")
    ap.add_argument("--tangent-window", type=float, default=600.0,
                    help="length of line used for the transect direction (m)")
    ap.add_argument("--meander-guard", type=float, default=None,
                    help="cut a transect where the nearest centreline point is "
                         "more than this far along the line (m); default = 3x "
                         "half-width; 0 = off")
    ap.add_argument("--reverse", action="store_true",
                    help="flip the centreline so chainage increases downstream")
    ap.add_argument("--compare", default=None,
                    help="existing bathymetry CSV to compare the result with")
    args = ap.parse_args()

    import rasterio
    from scipy.spatial import cKDTree

    src = rasterio.open(args.raster)
    arr = src.read(1)
    nodata = src.nodata if src.nodata is not None else np.nan
    step = args.step or float(abs(src.res[0]))
    guard = (3.0 * args.half_width if args.meander_guard is None
             else args.meander_guard)

    line = read_centerline(args.centerline)
    if args.reverse:
        line = line[::-1]
    pts, chain, total = resample(line, args.spacing)
    tan = tangents(pts, chain, args.tangent_window)
    nrm = np.column_stack([-tan[:, 1], tan[:, 0]])       # left-hand normal

    print(f"raster     {src.width} x {src.height} at {src.res[0]:g} m, "
          f"{src.crs}, nodata {nodata:g}")
    print(f"centreline {line.shape[0]} vertices, {total/1000:.2f} km -> "
          f"{pts.shape[0]} transects every {args.spacing:g} m")

    # for each sampled point, the chainage of the nearest centreline point
    dense, dense_ch, _ = resample(line, min(args.spacing, 25.0))
    tree = cKDTree(dense)

    offs = np.arange(-args.half_width, args.half_width + 0.5 * step, step)
    n_pts = offs.size
    X = pts[:, 0][:, None] + nrm[:, 0][:, None] * offs[None, :]
    Y = pts[:, 1][:, None] + nrm[:, 1][:, None] * offs[None, :]
    Z = bilinear(arr, src.transform, X.ravel(), Y.ravel(), nodata) \
        .reshape(X.shape)

    n_meander = 0
    if guard > 0:
        _, idx = tree.query(np.column_stack([X.ravel(), Y.ravel()]))
        owner = dense_ch[idx].reshape(X.shape)
        stray = np.abs(owner - chain[:, None]) > guard
        # cut outwards from the centre: once a transect enters another part
        # of the reach, the rest of it is dropped
        mid = n_pts // 2
        left = np.cumsum(stray[:, mid::-1], axis=1) > 0
        right = np.cumsum(stray[:, mid:], axis=1) > 0
        cut = np.concatenate([left[:, ::-1], right[:, 1:]], axis=1)
        n_meander = int((cut & np.isfinite(Z)).sum())
        Z[cut] = np.nan

    valid = np.isfinite(Z)
    per = valid.sum(axis=1)
    if (per < 3).any():
        print(f"  WARNING: {(per < 3).sum()} transects have fewer than 3 valid "
              f"points and are dropped")
    keep = per >= 3
    print(f"  sampled {n_pts} points per transect at {step:g} m over "
          f"+-{args.half_width:g} m")
    print(f"  valid points {valid.mean():.1%}; per transect p5 {np.percentile(per,5):.0f} "
          f"p50 {np.percentile(per,50):.0f} p95 {np.percentile(per,95):.0f}")
    if guard > 0:
        print(f"  meander guard removed {n_meander} samples that belonged to "
              f"another part of the reach ({n_meander/valid.sum():.1%})")

    wet = Z[valid]
    print(f"  elevation min {wet.min():.2f}  p50 {np.median(wet):.2f}  "
          f"max {wet.max():.2f} m")
    bed = np.nanmin(np.where(valid, Z, np.nan), axis=1)
    print(f"  thalweg at chainage 0 {bed[0]:.2f} m, at "
          f"{chain[keep][-1]/1000:.1f} km {bed[keep][-1]:.2f} m -- chainage "
          f"must increase DOWNSTREAM, so the second should be the deeper")

    out = {"X_m": chain[keep]}
    d = offs - offs[0]                       # offsets start at zero
    for i in range(n_pts):
        out[f"d_{i}"] = np.full(keep.sum(), d[i])
    for i in range(n_pts):
        out[f"h_{i}"] = Z[keep, i]
    pd.DataFrame(out).to_csv(args.out, index=False)
    print(f"\nwritten {keep.sum()} transects x {n_pts} points -> {args.out}")

    if args.compare:
        old = pd.read_csv(args.compare)
        zc = sorted([c for c in old.columns if c.startswith("h_")],
                    key=lambda c: int(c.split("_")[1]))
        dc = sorted([c for c in old.columns if c.startswith("d_")],
                    key=lambda c: int(c.split("_")[1]))
        ox = old["X_m"].to_numpy(float)
        obed = np.nanmin(old[zc].to_numpy(float), axis=1)
        owid = (np.nanmax(old[dc].to_numpy(float), axis=1)
                - np.nanmin(old[dc].to_numpy(float), axis=1))
        nbed = np.interp(ox, chain[keep], bed[keep])
        dz = nbed - obed
        print(f"\ncompared with {os.path.basename(args.compare)} at its "
              f"{ox.size} chainages:")
        print(f"  thalweg difference (new - old): p5 {np.percentile(dz,5):+.2f}  "
              f"p50 {np.percentile(dz,50):+.2f}  p95 {np.percentile(dz,95):+.2f} m")
        print(f"  corridor half-width: old p50 {np.median(owid)/2:.0f} m, "
              f"new {args.half_width:.0f} m")
        med = float(np.median(dz))
        if abs(med) > 0.5:
            print(f"  !! the median thalweg differs by {med:+.2f} m from the "
                  f"old file. A near-constant offset over the whole reach is a "
                  f"difference of vertical datum, not terrain; resolve it "
                  f"before using this file.")
        else:
            print(f"  the two agree within {abs(med):.2f} m at the median, so "
                  f"they share a vertical datum.")


if __name__ == "__main__":
    main()
