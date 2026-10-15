"""
Reading the forcing and the observations.

Column names come from the case file (or are found automatically), tributaries
are listed in the case file, and one vertical datum shift is applied to the
bathymetry, the downstream stage and the gauges alike. All series are put on
one hourly time index, gaps are filled in a way that suits each quantity, and
a mask records which values were measured.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .config import DEVICE, CaseConfig, SeriesSpec
from .utils import catmull_rom, interp1d

#: Robust deviations from an hour's median beyond which a sub-hourly sample
#: is not used to represent that hour (a spike).
_SPIKE_MAD = 6.0
#: Floor on the robust scale, so rounding steps are not taken as spikes.
_SPIKE_FLOOR = 0.02

_STAGE_HINTS = ("stage", "level", "elev", "water", "h", "z", "niveau", "hauteur")
_FLOW_HINTS = ("discharge", "flow", "q", "debit", "débit")

#: Values that hydrological exports write for "no measurement".
_SENTINELS = (-9999.0, -999.0, -99.99, -9.999, 9999.0, 99999.0)
#: Anything larger than this is a typo or a unit error.
_ABSURD = 1e7

#: First fields of a separated date, e.g. "01/02/2007 03:00".
_DATE_FIELDS = re.compile(r"^\s*(\d{1,4})\s*[^\d\s]\s*(\d{1,2})\s*[^\d\s]\s*(\d{1,4})")


def _auto_column(df: pd.DataFrame, hints, exclude=()) -> str:
    """
    Find the value column of a file from a list of likely names.

    Args:
        df (pd.DataFrame): The file.
        hints (tuple): Likely column names, most likely first.
        exclude (tuple): Columns that are not candidates.

    Returns:
        str: Name of the value column.
    """
    cands = [c for c in df.columns if c not in exclude]
    for hint in hints:
        for c in cands:
            if c.lower() == hint:
                return c
    for hint in hints:
        for c in cands:
            if hint in c.lower():
                return c
    numeric = [c for c in cands if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) == 1:
        return numeric[0]
    raise ValueError(
        f"Could not identify the value column (looked for {hints}); "
        f"set 'value_col' explicitly. Available: {list(df.columns)}"
    )


def detect_date_convention(raw: pd.Series) -> dict:
    """
    Find how a column writes its dates (year first, day first or month first).

    A first field above 12 cannot be a month (day first); a second field above
    12 cannot be a month (month first); a four-digit first field is a year.

    Args:
        raw (pd.Series): The date column as read.

    Returns:
        dict: ``{"kind", "reason"}`` with ``kind`` one of ``iso``, ``dayfirst``,
        ``monthfirst``, ``ambiguous`` or ``inconsistent``.
    """
    if pd.api.types.is_datetime64_any_dtype(raw):
        return {"kind": "iso", "reason": "already datetime"}
    s = raw.dropna().astype(str).str.strip()
    if s.empty:
        return {"kind": "ambiguous", "reason": "no values"}
    m = s.str.extract(_DATE_FIELDS)
    if m[0].isna().all():
        return {"kind": "iso", "reason": "no separated date fields found"}
    f1 = pd.to_numeric(m[0], errors="coerce")
    f2 = pd.to_numeric(m[1], errors="coerce")
    if (m[0].str.len() >= 4).any():
        return {"kind": "iso", "reason": "four-digit leading field (year first)"}

    n1 = int((f1 > 12).sum())
    n2 = int((f2 > 12).sum())
    if n1 and n2:
        return {"kind": "inconsistent",
                "reason": (f"{n1} rows have a first field above 12 and {n2} "
                           f"have a second field above 12")}
    if n1:
        return {"kind": "dayfirst",
                "reason": f"{n1} rows have a first field above 12"}
    if n2:
        return {"kind": "monthfirst",
                "reason": f"{n2} rows have a second field above 12"}
    return {"kind": "ambiguous",
            "reason": "no field anywhere exceeds 12"}


def _candidate_formats(sample: str, dayfirst: bool):
    """
    A few explicit date formats to try, built from one sample value.

    An explicit format parses the whole column at once, which is much faster
    than guessing the format of every row.

    Args:
        sample (str): One value of the date column.
        dayfirst (bool): Day/month order of the column.

    Returns:
        list[str]: Formats to try, most likely first.
    """
    out = []
    try:
        from pandas.core.tools.datetimes import guess_datetime_format
        g = guess_datetime_format(str(sample), dayfirst=dayfirst)
        if g:
            out.append(g)
    except (ValueError, TypeError):
        pass
    m = _DATE_FIELDS.match(str(sample))
    if m:
        sep = re.search(r"^\s*\d{1,4}\s*([^\d\s])", str(sample))
        sep = sep.group(1) if sep else "/"
        year = "%Y" if len(m.group(3)) == 4 else "%y"
        d, mo = ("%d", "%m") if dayfirst else ("%m", "%d")
        date = f"{d}{sep}{mo}{sep}{year}"
        out += [f"{date}{t}" for t in ("", " %H:%M", " %H:%M:%S", "T%H:%M",
                                       "T%H:%M:%S", " %H:%M:%S.%f")]
    return out


def _parse_timestamps(raw: pd.Series, dayfirst, where: str):
    """
    Parse a date column, checking the day/month order against the data.

    ISO dates are tried first. Otherwise the day/month order is detected from
    the column; a ``dayfirst`` setting that contradicts the data is an error,
    because reading dates the wrong way swaps day and month for every date up
    to the 12th without changing the length of the record.

    Args:
        raw (pd.Series): The date column.
        dayfirst (bool | None): Setting from the case file.
        where (str): File name, for error messages.

    Returns:
        tuple: ``(parsed timestamps, report dict)``.
    """
    if pd.api.types.is_datetime64_any_dtype(raw):
        return raw, {}

    # ISO dates first: fastest, and the day/month order cannot be wrong.
    try:
        iso = pd.to_datetime(raw, format="ISO8601", errors="coerce")
        if iso.notna().all():
            return iso, {"convention": "ISO 8601"}
    except (ValueError, TypeError):
        pass

    ev = detect_date_convention(raw)
    if ev["kind"] == "inconsistent":
        raise ValueError(
            f"{where}: the date column mixes day-first and month-first rows "
            f"({ev['reason']}). No single convention can read it; the file "
            f"needs fixing at source."
        )
    detected = {"dayfirst": True, "monthfirst": False}.get(ev["kind"])
    if detected is None:
        # cannot be decided from the data: use the case setting
        detected = bool(dayfirst) if dayfirst is not None else False
        ev["reason"] += f"; using dayfirst={detected} from the case file"
    elif dayfirst is not None and bool(dayfirst) != detected:
        raise ValueError(
            f"{where}: the case file says dayfirst={bool(dayfirst)} but the "
            f"data says otherwise ({ev['reason']}). Remove 'dayfirst' for "
            f"this file to let it be detected, or set it to "
            f"{str(detected).lower()}."
        )
    dayfirst = detected

    s = raw.astype(str).str.strip() if raw.dtype == object else raw

    out = None
    for fmt in _candidate_formats(s.iloc[0], dayfirst):
        try:
            cand = pd.to_datetime(s, format=fmt, errors="coerce")
        except (ValueError, TypeError):
            continue
        if cand.notna().all():
            out = cand
            break
    if out is None:
        out = pd.to_datetime(s, dayfirst=dayfirst, errors="coerce")
    n_slow = 0
    bad = out.isna() & s.notna()
    if bad.any():
        # only the rows that failed are parsed one by one
        retry = pd.to_datetime(s[bad], dayfirst=dayfirst, errors="coerce",
                               format="mixed")
        out.loc[bad] = retry
        n_slow = int(bad.sum())

    # A wrong day/month order gives a record that is no longer in time order,
    # while the other order would be. Check that.
    if not out.is_monotonic_increasing:
        flip = pd.to_datetime(s, dayfirst=not dayfirst, errors="coerce")
        if flip.is_monotonic_increasing and flip.notna().all():
            n_swap = int((out.to_numpy() != flip.to_numpy()).sum())
            raise ValueError(
                f"{where}: with dayfirst={dayfirst} the timestamps are out of "
                f"order, and with dayfirst={not dayfirst} they are not. "
                f"{n_swap} of {len(s)} rows ({100 * n_swap / len(s):.0f}%) "
                f"differ between the two readings. Set "
                f"'dayfirst: {str(not dayfirst).lower()}' for this file."
            )
    failed = int((out.isna() & s.notna()).sum())
    if failed:
        sample = list(s[out.isna() & s.notna()].head(3))
        raise ValueError(
            f"{where}: {failed} timestamps could not be parsed, e.g. {sample}. "
            f"Fix them or remove those rows."
        )
    return out, {"convention": ("day-first" if dayfirst else "month-first"),
                 "reason": ev["reason"], "reparsed": n_slow}


def _parse_numeric(raw: pd.Series, where: str, column: str):
    """
    Read a value column, accepting the usual ways exports write numbers.

    Decimal commas, thousands separators and stray units are handled. Missing
    value codes such as -9999 and absurd values become missing (NaN) and are
    filled later like any other gap.

    Args:
        raw (pd.Series): The column as read.
        where (str): File name, for messages.
        column (str): Column name, for messages.

    Returns:
        tuple: ``(values as float, report dict of what was removed)``.
    """
    rep = {}
    if pd.api.types.is_numeric_dtype(raw):
        s = raw.astype(float)
    else:
        t = raw.astype(str).str.strip()
        t = t.str.replace(r"[\s ']", "", regex=True)
        # a comma is a decimal point only where there is no point already
        has_dot = t.str.contains(r"\.", na=False)
        t = t.where(has_dot, t.str.replace(",", ".", regex=False))
        t = t.str.replace(",", "", regex=False)
        t = t.str.replace(r"[^0-9eE+\-.]", "", regex=True)
        s = pd.to_numeric(t, errors="coerce")
        rep["unreadable"] = int((s.isna() & raw.notna()).sum())
    n0 = int(s.isna().sum())
    for v in _SENTINELS:
        s = s.mask(np.isclose(s, v, rtol=0, atol=1e-6))
    s = s.mask(s.abs() > _ABSURD)
    rep["sentinels"] = int(s.isna().sum() - n0)
    return s, {k: v for k, v in rep.items() if v}


def _read_series(path: str, spec: SeriesSpec, hints) -> pd.DataFrame:
    """
    Read one forcing CSV and reduce it to one row per hour.

    With ``subhourly="nearest"`` (default) each hour keeps the sample closest
    to it, after dropping samples that sit far from that hour's median (spikes).
    The value stays instantaneous, which is what the time interpolation of the
    forcing assumes. With ``"mean"`` the hour is averaged.

    Args:
        path (str): File path.
        spec (SeriesSpec): Column settings from the case file.
        hints (tuple): Likely names of the value column.

    Returns:
        pd.DataFrame: Hourly rows indexed by ``timestamp``.
    """
    df = pd.read_csv(path)
    if spec.time_col not in df.columns:
        raise ValueError(
            f"{path}: missing time column '{spec.time_col}'. "
            f"Columns present: {list(df.columns)}"
        )
    t, tinfo = _parse_timestamps(df[spec.time_col], spec.dayfirst, path)
    df = df.assign(**{spec.time_col: t}).dropna(subset=[spec.time_col])

    target = df[spec.time_col].dt.round("h")
    n_dup = int(target.duplicated().sum())
    if n_dup:
        if spec.subhourly == "mean":
            num = df.select_dtypes("number").columns
            df = df.assign(**{spec.time_col: target})
            df = (df.groupby(spec.time_col, as_index=False)[list(num)].mean()
                  if len(num) else df.drop_duplicates(subset=[spec.time_col]))
        else:
            # Nearest sample to the hour, skipping isolated spikes: within each
            # hour, samples more than _SPIKE_MAD robust deviations from the
            # hour's median are not eligible. If all are, the nearest is kept.
            num = df.select_dtypes("number").columns
            probe = spec.value_col if spec.value_col in num else (
                num[0] if len(num) else None)
            work = df.assign(_t=target,
                             _off=(df[spec.time_col] - target).abs())
            if probe is not None:
                grp = work.groupby("_t")[probe]
                med = grp.transform("median")
                dev = (work[probe] - med).abs()
                mad = dev.groupby(work["_t"]).transform("median")
                scale = np.maximum(1.4826 * mad.to_numpy(), _SPIKE_FLOOR)
                work["_bad"] = (dev.to_numpy() > _SPIKE_MAD * scale).astype(int)
                n_spike = int(work["_bad"].sum())
                if n_spike:
                    print(f"  {os.path.basename(path)}: {n_spike:,} sub-hourly "
                          f"outlier(s) skipped when picking the hourly sample")
            else:
                work["_bad"] = 0
            keep = (work.sort_values(["_t", "_bad", "_off"], kind="stable")
                        .drop_duplicates(subset="_t", keep="first"))
            df = (keep.drop(columns=["_off", "_bad", spec.time_col])
                      .rename(columns={"_t": spec.time_col}))
    else:
        df = df.assign(**{spec.time_col: target})
    df = df.set_index(spec.time_col).sort_index()
    df.index.name = "timestamp"
    df.attrs["duplicate_hours_averaged"] = n_dup
    df.attrs["date_convention"] = tinfo.get("convention", "")
    return df


def _observed_flag(df: pd.DataFrame, value_col: str):
    """
    The file's own record of which values were measured, if it has one.

    Looks for a column ``<value>_imputed`` first, then ``imputed`` (0 =
    measured, anything else = filled).

    Args:
        df (pd.DataFrame): The file.
        value_col (str): Name of the value column.

    Returns:
        pd.Series | None: True where the value was measured, or None when the
        file has no flag.
    """
    for name in (f"{value_col}_imputed", "imputed"):
        if name in df.columns:
            flag = pd.to_numeric(df[name], errors="coerce").fillna(1.0)
            return flag == 0.0
    return None


def _tidal_fill(series: pd.Series, periods_h=None,
                residual_taper_h: float = 72.0) -> pd.Series:
    """
    Fill gaps in a tidal record by harmonic analysis of the observed part.

    A least-squares fit of ``a0 + sum_k (a_k cos w_k t + b_k sin w_k t)`` keeps
    the amplitude and phase of each constituent, so a filled gap keeps its
    spring-neap range. The non-tidal part (surge, river) is interpolated
    across the gap and faded to zero over ``residual_taper_h`` hours.

    Args:
        series (pd.Series): Hourly stage with gaps (NaN).
        periods_h (list, optional): Periods (h) to fit. Default: the 16 tidal
            constituents plus the long-period ones.
        residual_taper_h (float): Fading length of the non-tidal part (h).

    Returns:
        pd.Series: The series with gaps filled.
    """
    if not series.isna().any():
        return series
    obs = series.notna().to_numpy()
    if obs.sum() < 64:                       # too little to fit anything
        return series.interpolate(method="time", limit_direction="both")

    t_h = (series.index - series.index[0]).total_seconds().to_numpy() / 3600.0
    if periods_h is None:
        from .tides import CONSTITUENTS, DEFAULT_TIDAL
        periods_h = [CONSTITUENTS[n] for n in DEFAULT_TIDAL] + \
                    [CONSTITUENTS[n] for n in ("MSF", "MF", "MM", "SSA", "SA")]
    w = 2.0 * np.pi / np.asarray(periods_h, float)

    ang = t_h[:, None] * w[None, :]
    design = np.column_stack([np.ones_like(t_h), np.cos(ang), np.sin(ang)])
    coef, *_ = np.linalg.lstsq(design[obs], series.to_numpy(float)[obs],
                               rcond=None)
    harm = design @ coef

    resid = pd.Series(np.where(obs, series.to_numpy(float) - harm, np.nan),
                      index=series.index)
    step_h = max(float(np.median(np.diff(t_h))), 1e-6)
    limit = max(int(round(residual_taper_h / step_h)), 1)
    resid = resid.interpolate(method="time", limit=limit,
                              limit_direction="both").fillna(0.0)
    return series.where(series.notna(), pd.Series(harm + resid.to_numpy(),
                                                  index=series.index))


def _seasonal_climatology(observed: pd.Series,
                          target: pd.DatetimeIndex,
                          smooth_days: int = 15) -> pd.Series:
    """
    Seasonal norm of a discharge: day-of-year median of ``log Q``, smoothed.

    Args:
        observed (pd.Series): Measured discharge.
        target (pd.DatetimeIndex): Times where the norm is wanted.
        smooth_days (int): Half-width of the smoothing window (days).

    Returns:
        pd.Series: ``log Q`` of the seasonal norm at ``target``.
    """
    lg = np.log(observed.clip(lower=1e-6))
    clim = lg.groupby(observed.index.dayofyear).median()
    clim = clim.reindex(range(1, 367)).interpolate(limit_direction="both")
    w = 2 * smooth_days + 1
    wrapped = pd.concat([clim, clim, clim]).rolling(w, center=True,
                                                    min_periods=1).median()
    clim = wrapped.iloc[len(clim):2 * len(clim)]
    clim.index = pd.Index(range(1, 367))
    return pd.Series(clim.reindex(target.dayofyear).to_numpy(), index=target)


def _recession_constant(s: pd.Series, default_h: float = 240.0) -> float:
    """
    Recession time scale of the river (h), from its falling limbs.

    Without rain a hydrograph decays as ``Q0 exp(-t/k)``, so
    ``k = -dt / ln(Q(t+1)/Q(t))`` where the flow is falling. The median over
    the record is used.

    Args:
        s (pd.Series): Measured discharge.
        default_h (float): Value used when the record is too short.

    Returns:
        float: Recession constant ``k`` (h).
    """
    v = s.dropna()
    if v.size < 48:
        return default_h
    dt_h = (v.index.to_series().diff().dt.total_seconds() / 3600.0).to_numpy()[1:]
    ratio = (v.to_numpy()[1:] / np.maximum(v.to_numpy()[:-1], 1e-9))
    ok = (ratio > 0.5) & (ratio < 0.999) & (dt_h > 0) & (dt_h < 25.0)
    if ok.sum() < 24:
        return default_h
    k = -dt_h[ok] / np.log(ratio[ok])
    return float(np.clip(np.median(k), 6.0, 24.0 * 120.0))


def _fluvial_fill(series: pd.Series, short_gap_h: float = 24.0) -> pd.Series:
    """
    Fill gaps in a discharge record.

    The gap is interpolated in log space between the observations on either
    side, and moves towards the seasonal norm as the distance to the nearest
    observation grows (at the river's own recession rate). Short gaps are
    plain interpolation; long gaps and the ends of the record follow the
    seasonal norm.

    Args:
        series (pd.Series): Hourly discharge with gaps (NaN).
        short_gap_h (float): Gaps up to this length are only interpolated (h).

    Returns:
        pd.Series: The series with gaps filled.
    """
    if not series.isna().any():
        return series
    obs = series.notna()
    if obs.sum() < 24:
        return series.interpolate(method="time", limit_direction="both") \
                     .ffill().bfill()

    lg = np.log(series.clip(lower=1e-6))
    interp = lg.interpolate(method="time", limit_direction="both").ffill().bfill()
    clim = _seasonal_climatology(series[obs], series.index)
    clim = clim.fillna(float(interp.mean()))

    # hours to the nearest observation, backwards and forwards
    t_h = (series.index - series.index[0]).total_seconds().to_numpy() / 3600.0
    last = pd.Series(np.where(obs, t_h, np.nan), index=series.index).ffill()
    nxt = pd.Series(np.where(obs, t_h, np.nan), index=series.index).bfill()
    d_back = t_h - last.to_numpy()
    d_fwd = nxt.to_numpy() - t_h
    dist = np.fmin(np.nan_to_num(d_back, nan=np.inf),
                   np.nan_to_num(d_fwd, nan=np.inf))
    dist = np.where(np.isfinite(dist), dist, 0.0)

    k = _recession_constant(series[obs])
    wgt = np.where(dist <= short_gap_h, 0.0,
                   1.0 - np.exp(-(dist - short_gap_h) / k))
    filled = np.exp((1.0 - wgt) * interp.to_numpy() + wgt * clim.to_numpy())
    return series.where(obs, pd.Series(filled, index=series.index))


@dataclass
class Window:
    """One continuous stretch of forcing and observations (training or prediction)."""

    t0: float                     # start, seconds since the case epoch
    t1: float                     # end
    t_hourly: torch.Tensor        # (T,) hourly times
    Q_up: torch.Tensor            # (T,) upstream discharge
    Q_up_mask: torch.Tensor       # (T,) 1 where measured, 0 where filled
    eta_down: torch.Tensor        # (T,) downstream stage
    eta_down_mask: torch.Tensor
    Q_trib: torch.Tensor          # (T, n_trib) tributary discharges
    lstm_input: torch.Tensor      # (1, T, 3) normalised forcing for the LSTM
    gauge_x: torch.Tensor         # (G,) gauge chainage
    gauge_t: torch.Tensor         # (G,) time of each reading
    gauge_eta: torch.Tensor       # (G,) observed stage
    gauge_u: Optional[torch.Tensor]
    #: Weight of each gauge reading when every station counts the same.
    gauge_w: Optional[torch.Tensor] = None

    @property
    def n_hours(self) -> int:
        """Number of hourly forcing samples in the window."""
        return int(self.t_hourly.numel())

    def _interp(self, flat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Interpolate an hourly series to any time with a smooth (C1) spline.

        The downstream stage enters the solution directly through the hard
        boundary condition, so d(eta)/dt, the storage term of continuity,
        takes the smoothness of this interpolation. Linear interpolation would
        make it jump at every hour.

        Args:
            flat (torch.Tensor): Query times (s), flat.
            y (torch.Tensor): Hourly values.

        Returns:
            torch.Tensor: Values at the query times.
        """
        n = y.numel()
        if n < 4:                       # too short for the 4-point spline
            return interp1d(flat, self.t_hourly, y)
        step = (self.t_hourly[-1] - self.t_hourly[0]) / max(n - 1, 1)
        pos = ((flat - self.t_hourly[0]) / step).clamp(0.0, float(n - 1))
        return catmull_rom(y.reshape(-1, 1), pos).reshape(-1)

    def forcing(self, t: torch.Tensor) -> dict:
        """
        Boundary forcing at any times.

        Args:
            t (torch.Tensor): Times (s), shape ``(N, 1)``.

        Returns:
            dict: ``Q_up`` and ``eta_down`` of shape ``(N, 1)``, and ``Q_trib``
            of shape ``(N, n_trib)``.
        """
        flat = t.reshape(-1)
        out = {
            "Q_up": self._interp(flat, self.Q_up).reshape(-1, 1),
            "eta_down": self._interp(flat, self.eta_down).reshape(-1, 1),
        }
        if self.Q_trib.shape[1]:
            out["Q_trib"] = torch.stack(
                [self._interp(flat, self.Q_trib[:, k])
                 for k in range(self.Q_trib.shape[1])], dim=1)
        else:
            out["Q_trib"] = flat.new_zeros((flat.numel(), 0))
        return out


class RiverDataset:
    """Hourly forcing and gauge observations of one case, on one time index."""

    def __init__(self, cfg: CaseConfig, data_dir: Optional[str] = None,
                 verbose: bool = True):
        """
        Read, align and fill all the series of a case.

        Args:
            cfg (CaseConfig): The case.
            data_dir (str, optional): Read the input files from this folder
                instead of the case ``data_dir`` (e.g. new forcing to predict).
            verbose (bool): Print a summary.
        """
        self.cfg = cfg
        self.epoch = pd.to_datetime(cfg.epoch)
        self.datum = cfg.datum_shift or 0.0
        root = data_dir or cfg.data_dir

        def p(rel):
            # same rule as CaseConfig.path, against `root`
            full = rel if os.path.isabs(rel) else os.path.join(root, rel)
            if not os.path.exists(full):
                for suffix in (".gz", ".bz2", ".xz", ".zip"):
                    if os.path.exists(full + suffix):
                        return full + suffix
            return full

        # upstream discharge
        up = _read_series(p(cfg.upstream.file), cfg.upstream, _FLOW_HINTS)
        up_col = cfg.upstream.value_col or _auto_column(up, _FLOW_HINTS)
        up_ser, up_rep = _parse_numeric(up[up_col], p(cfg.upstream.file), up_col)
        up_ser = up_ser * cfg.upstream.scale

        # downstream stage
        dn = _read_series(p(cfg.downstream.file), cfg.downstream, _STAGE_HINTS)
        dn_col = cfg.downstream.value_col or _auto_column(dn, _STAGE_HINTS)
        dn_val, dn_rep = _parse_numeric(dn[dn_col], p(cfg.downstream.file), dn_col)
        dn_ser = (dn_val * cfg.downstream.scale
                  + cfg.downstream.datum_shift + self.datum)

        # tributaries
        trib_ser = {}
        if cfg.tributaries and cfg.tributaries_file is not None:
            tf = _read_series(p(cfg.tributaries_file.file), cfg.tributaries_file,
                              _FLOW_HINTS)
            for trib in cfg.tributaries:
                if trib.column not in tf.columns:
                    raise ValueError(
                        f"{cfg.tributaries_file.file}: missing column "
                        f"'{trib.column}' for tributary '{trib.name}'"
                    )
                trib_ser[trib.name], _ = _parse_numeric(
                    tf[trib.column], p(cfg.tributaries_file.file), trib.column)
        elif cfg.tributaries:
            raise ValueError("tributaries declared but no 'tributaries_file' given")

        # Common hourly index: the period where BOTH boundaries are recorded.
        # Outside it one of the two boundary conditions would be invented.
        span = {"upstream discharge": up_ser, "downstream stage": dn_ser}
        lo = max(s.index.min() for s in span.values())
        hi = min(s.index.max() for s in span.values())
        if hi <= lo:
            detail = "; ".join(f"{k}: {v.index.min()} .. {v.index.max()}"
                               for k, v in span.items())
            raise ValueError(
                f"the upstream and downstream records do not overlap, so there "
                f"is no period with a boundary condition at both ends ({detail})"
            )
        idx = pd.date_range(lo, hi, freq="h")
        self.trimmed = {k: (int((v.index < lo).sum() + (v.index > hi).sum()),
                            v.index.min(), v.index.max())
                        for k, v in span.items()}
        self.parse_report = {"upstream discharge": up_rep,
                             "downstream stage": dn_rep}

        from .tides import CONSTITUENTS, resolve_constituents
        tide_periods = [CONSTITUENTS[n] for n in
                        resolve_constituents(cfg.physics.constituents,
                                             cfg.physics.tidal)]
        tide_periods += list(cfg.physics.subtidal_periods_h)

        for label, frame, col in (("upstream", up, up_col),
                                  ("downstream", dn, dn_col)):
            if _observed_flag(frame, col) is None and verbose:
                print(f"  note: the {label} file has no 'imputed' column, so "
                      f"every value is treated as measured.")

        self.Q_up, self.Q_up_mask = self._align(
            up_ser, idx, "fluvial", observed=_observed_flag(up, up_col))
        self.eta_down, self.eta_down_mask = self._align(
            dn_ser, idx, "tidal" if cfg.physics.tidal else "linear",
            periods_h=tide_periods or None,
            observed=_observed_flag(dn, dn_col))

        trib_cols, trib_masks = [], []
        for trib in cfg.tributaries:
            v, m = self._align(trib_ser[trib.name], idx, "fluvial",
                               observed=_observed_flag(tf, trib.column))
            trib_cols.append(v)
            trib_masks.append(m)
        self.Q_trib = (np.stack(trib_cols, axis=1) if trib_cols
                       else np.zeros((idx.size, 0)))
        self.Q_trib_mask = (np.stack(trib_masks, axis=1) if trib_masks
                            else np.zeros((idx.size, 0)))

        self.index = idx
        self.t_hourly = (idx - self.epoch).total_seconds().to_numpy(float)

        # gauges
        self.gauges = self._load_gauges(p) if cfg.gauges is not None else None

        # contiguous arrays of the gauge columns, sliced by window()
        if self.gauges is not None and len(self.gauges):
            self._g_t = np.ascontiguousarray(self.gauges["t"].to_numpy(float))
            self._g_x = np.ascontiguousarray(self.gauges["x"].to_numpy(float))
            self._g_eta = np.ascontiguousarray(self.gauges["eta"].to_numpy(float))
            self._g_u = (np.ascontiguousarray(self.gauges["u"].to_numpy(float))
                         if "u" in self.gauges.columns else None)
            if not np.all(np.diff(self._g_t) >= 0):
                raise AssertionError("gauge observations are not sorted by time")
        else:
            self._g_t = self._g_x = self._g_eta = self._g_u = None

        # Normalisation statistics of the LSTM input, from measured values
        # only (filled values would pull the mean and shrink the spread).
        def _obs(v, mask):
            m = np.asarray(mask, bool).reshape(v.shape)
            sel = v[m] if m.any() else v
            return float(sel.mean()), float(sel.std() + 1e-9)

        q_mean, q_std = _obs(self.Q_up, self.Q_up_mask)
        e_mean, e_std = _obs(self.eta_down, self.eta_down_mask)
        if self.Q_trib.size:
            t_mean, t_std = _obs(self.Q_trib, self.Q_trib_mask)
        else:
            t_mean, t_std = 0.0, 1.0
        self.stats = {
            "Q_up_mean": q_mean, "Q_up_std": q_std,
            "eta_mean": e_mean, "eta_std": e_std,
            "Q_trib_mean": t_mean, "Q_trib_std": t_std,
        }

        # copies on the compute device
        def dev(a):
            return torch.as_tensor(np.array(a, dtype=float), device=DEVICE)
        self._Q_up = dev(self.Q_up)
        self._Q_up_mask = dev(self.Q_up_mask)
        self._eta_down = dev(self.eta_down)
        self._eta_down_mask = dev(self.eta_down_mask)
        self._Q_trib = dev(self.Q_trib)
        self._t = dev(self.t_hourly)
        self._lstm = torch.as_tensor(self._lstm_features(), device=DEVICE)

        if verbose:
            print(f"  forcing: {idx[0]:%Y-%m-%d} .. {idx[-1]:%Y-%m-%d}  "
                  f"({idx.size} hours), where both boundaries are recorded")
            for name, mask in (("upstream discharge", self.Q_up_mask),
                               ("downstream stage", self.eta_down_mask)):
                cut, s0, s1 = self.trimmed[name]
                extra = (f", {cut} h of its own record fall outside the overlap"
                         if cut else "")
                bad = self.parse_report.get(name) or {}
                flag = ("  [" + ", ".join(f"{v} {k}" for k, v in bad.items())
                        + "]") if bad else ""
                print(f"    {name:20s} {100.0 * mask.mean():5.1f}% observed  "
                      f"(record {s0:%Y-%m-%d} .. {s1:%Y-%m-%d}{extra}){flag}")
            for k, trib in enumerate(cfg.tributaries):
                print(f"    {trib.name:20s} "
                      f"{100.0 * self.Q_trib_mask[:, k].mean():5.1f}% observed")
            for label, when in (("train_start", cfg.train_start),
                                ("train_end", cfg.train_end),
                                ("val_start", cfg.val_start),
                                ("val_end", cfg.val_end)):
                if when is None:
                    continue
                t = pd.to_datetime(when)
                if t < idx[0] or t > idx[-1]:
                    print(f"    WARNING: {label} ({t:%Y-%m-%d}) lies outside the "
                          f"recorded forcing")
            if self.gauges is not None:
                print(f"  gauges : {len(self.gauges)} readings at "
                      f"{self.gauges['x'].nunique()} locations")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _align(series: pd.Series, idx: pd.DatetimeIndex, kind: str = "linear",
               periods_h=None, observed: pd.Series = None):
        """
        Put a series on the hourly index and fill its gaps.

        Args:
            series (pd.Series): The series as read.
            idx (pd.DatetimeIndex): The hourly index of the case.
            kind (str): How gaps are filled: ``"tidal"`` (harmonic analysis),
                ``"fluvial"`` (log-space interpolation towards the seasonal
                norm) or ``"linear"`` (time interpolation).
            periods_h (list, optional): Periods for the tidal fill (h).
            observed (pd.Series, optional): The file's own measured/filled flag.

        Returns:
            tuple: ``(values, mask)`` as numpy arrays; the mask is 1 where the
            value was measured.
        """
        s = series.reindex(idx)
        mask = s.notna().to_numpy(float)
        if observed is not None:
            mask = mask * observed.reindex(idx, fill_value=False).to_numpy(float)
        if not mask.any():
            raise ValueError("series is entirely empty after alignment")
        if kind == "tidal":
            s = _tidal_fill(s, periods_h=periods_h)
        elif kind == "fluvial":
            s = _fluvial_fill(s)
        s = s.interpolate(method="time", limit_direction="both").ffill().bfill()
        if s.isna().any():
            raise ValueError("series is entirely empty after alignment")
        return s.to_numpy(float), mask

    def _gauge_frame(self, df, tcol, vcol):
        """
        Reduce one gauge file to the columns the model needs: t, x, eta (and u).

        With ``gauge_step_h`` set, one reading per station is kept for every
        grid time: the untouched reading nearest to it (thinning, not averaging).

        Args:
            df (pd.DataFrame): The gauge file.
            tcol (str): Time column.
            vcol (str): Stage column.

        Returns:
            pd.DataFrame: Columns ``t`` (s since epoch), ``x`` (m), ``eta`` (m)
            and optionally ``u`` (m/s).
        """
        cfg = self.cfg
        t, _ = _parse_timestamps(df[tcol], cfg.gauges.dayfirst,
                                 f"gauges:{tcol}")
        df = df.assign(**{tcol: t}).dropna(subset=[tcol, vcol, cfg.gauge_x_col])
        out = pd.DataFrame({
            "t": (df[tcol] - self.epoch).dt.total_seconds().to_numpy(float),
            "x": df[cfg.gauge_x_col].to_numpy(float),
            "eta": (df[vcol].to_numpy(float) * cfg.gauges.scale
                    + cfg.gauges.datum_shift + self.datum),
        })
        if cfg.gauge_velocity_col and cfg.gauge_velocity_col in df.columns:
            out["u"] = df[cfg.gauge_velocity_col].to_numpy(float)

        step = cfg.gauge_step_h
        if step:
            step_s = float(step) * 3600.0
            t = out["t"].to_numpy(float)
            cell = np.rint(t / step_s).astype(np.int64)
            off = np.abs(t - cell * step_s)
            order = np.lexsort((off, cell))          # by cell, then by distance
            first = np.ones(order.size, bool)
            first[1:] = cell[order][1:] != cell[order][:-1]
            keep = np.sort(order[first])
            out = out.iloc[keep].reset_index(drop=True)
        return out

    def _load_gauges(self, p):
        """
        Read the gauge observations: one CSV, or a folder with one CSV per station.

        Args:
            p (callable): Resolves a file name against the data folder.

        Returns:
            pd.DataFrame: All readings, sorted by time.
        """
        cfg = self.cfg
        path = p(cfg.gauges.file)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"gauge file '{path}' not found, but the case declares "
                f"'gauges:'. Fix the path, or remove the 'gauges:' block to "
                f"train on boundary conditions and physics alone."
            )
        if os.path.isdir(path):
            # One file per station (index.csv is metadata). Each file is reduced
            # before the next is opened, to keep memory low on long records.
            parts = sorted(f for f in os.listdir(path)
                           if f.endswith((".csv", ".csv.gz", ".csv.bz2",
                                          ".csv.xz", ".csv.zip"))
                           and os.path.basename(f) != "index.csv")
            if not parts:
                raise FileNotFoundError(
                    f"gauge directory '{path}' holds no station CSVs. Expected "
                    f"one file per station, e.g. H3270001.csv.gz.")
            tcol, xcol = cfg.gauges.time_col, cfg.gauge_x_col
            vcol = cfg.gauges.value_col
            chunks, header = [], None
            for f in parts:
                fp = os.path.join(path, f)
                head = pd.read_csv(fp, nrows=0)
                if header is None:
                    header = tuple(head.columns)
                elif tuple(head.columns) != header:
                    raise ValueError(
                        f"gauge files in '{path}' disagree on columns: "
                        f"{header} vs {tuple(head.columns)}")
                v = vcol or _auto_column(head, _STAGE_HINTS,
                                         exclude=(tcol, xcol))
                keep = [tcol, v, xcol]
                if cfg.gauge_velocity_col in head.columns:
                    keep.append(cfg.gauge_velocity_col)
                part = pd.read_csv(fp, usecols=keep)
                chunks.append(self._gauge_frame(part, tcol, v))
                del part
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            print(f"  gauges: {len(parts)} station files from "
                  f"{os.path.basename(path)}/, {len(df):,} observations")
            out = df
        else:
            df = pd.read_csv(path)
            tcol = cfg.gauges.time_col
            vcol = cfg.gauges.value_col or _auto_column(
                df, _STAGE_HINTS, exclude=(tcol, cfg.gauge_x_col))
            out = self._gauge_frame(df, tcol, vcol)
        return out.sort_values("t").reset_index(drop=True)

    def _lstm_features(self) -> np.ndarray:
        """
        The three normalised forcing channels read by the LSTM.

        Returns:
            np.ndarray: ``(T, 3)``: upstream discharge, downstream stage and
            the sum of the tributary discharges, each normalised.
        """
        s = self.stats
        cols = [(self.Q_up - s["Q_up_mean"]) / s["Q_up_std"],
                (self.eta_down - s["eta_mean"]) / s["eta_std"]]
        if self.Q_trib.shape[1]:
            cols.append((self.Q_trib.sum(axis=1) - s["Q_trib_mean"] * self.Q_trib.shape[1])
                        / (s["Q_trib_std"] * max(1, self.Q_trib.shape[1])))
        else:
            cols.append(np.zeros_like(self.Q_up))
        return np.stack(cols, axis=1)

    # -------------------------------------------------------------------- API
    def seconds(self, when) -> float:
        """Date to model time (seconds since the case epoch)."""
        return float((pd.to_datetime(when) - self.epoch).total_seconds())

    def timestamp(self, seconds: float) -> pd.Timestamp:
        """Model time (seconds since the case epoch) to a date."""
        return self.epoch + pd.to_timedelta(float(seconds), unit="s")

    def window(self, start, end) -> Window:
        """
        Cut a window out of the record.

        A window that reaches past the forcing record is clipped to it, with a
        note, because there is no boundary condition outside the record.

        Args:
            start: Start date (or seconds since the epoch).
            end: End date (or seconds since the epoch).

        Returns:
            Window: Forcing and gauge readings between ``start`` and ``end``.
        """
        t0 = self.seconds(start) if not isinstance(start, (int, float)) else float(start)
        t1 = self.seconds(end) if not isinstance(end, (int, float)) else float(end)
        if t1 <= t0:
            raise ValueError("window end must be after window start")

        lo, hi = float(self.t_hourly[0]), float(self.t_hourly[-1])
        if t0 < lo - 1.0 or t1 > hi + 1.0:
            c0, c1 = max(t0, lo), min(t1, hi)
            if c1 <= c0:
                raise ValueError(
                    f"window {start}..{end} lies entirely outside the forcing "
                    f"record {self.timestamp(lo)} .. {self.timestamp(hi)}")
            print(f"  note: window {self.timestamp(t0)} .. {self.timestamp(t1)} "
                  f"clipped to the forcing record: "
                  f"{self.timestamp(c0)} .. {self.timestamp(c1)}")
            t0, t1 = c0, c1

        m = (self.t_hourly >= t0 - 1.0) & (self.t_hourly <= t1 + 1.0)
        if m.sum() < 8:
            raise ValueError(
                f"window {start}..{end} contains only {int(m.sum())} hourly "
                "forcing samples; is it outside the forcing record?"
            )
        sel = torch.as_tensor(np.flatnonzero(m), device=DEVICE)

        g_x = g_t = g_eta = g_u = g_w = None
        if self.gauges is not None:
            # gauges are sorted by time, so the window is one slice
            lo = int(np.searchsorted(self._g_t, t0, side="left"))
            hi = int(np.searchsorted(self._g_t, t1, side="right"))
            g_x = torch.as_tensor(np.array(self._g_x[lo:hi]), device=DEVICE)
            g_t = torch.as_tensor(np.array(self._g_t[lo:hi]), device=DEVICE)
            g_eta = torch.as_tensor(np.array(self._g_eta[lo:hi]), device=DEVICE)
            if self._g_u is not None:
                g_u = torch.as_tensor(np.array(self._g_u[lo:hi]), device=DEVICE)
            # one over the number of readings of each station in the window
            _, inv, cnt = np.unique(self._g_x[lo:hi], return_inverse=True,
                                    return_counts=True)
            g_w = torch.as_tensor(1.0 / cnt[inv], device=DEVICE)
        if g_x is None:
            g_x = g_t = g_eta = torch.zeros(0, device=DEVICE)
            g_w = None

        return Window(
            t0=t0, t1=t1,
            t_hourly=self._t[sel],
            Q_up=self._Q_up[sel], Q_up_mask=self._Q_up_mask[sel],
            eta_down=self._eta_down[sel], eta_down_mask=self._eta_down_mask[sel],
            Q_trib=self._Q_trib[sel],
            lstm_input=self._lstm[sel].unsqueeze(0).contiguous(),
            gauge_x=g_x, gauge_t=g_t, gauge_eta=g_eta, gauge_u=g_u,
            gauge_w=g_w,
        )

    def window_starts(self, start, end, length_s: float, overlap_s: float):
        """
        Start times of consecutive overlapping windows covering a period.

        Args:
            start: Start of the period.
            end: End of the period.
            length_s (float): Window length (s).
            overlap_s (float): Overlap between windows (s).

        Returns:
            list[float]: Window start times (s since the epoch).
        """
        t0, t1 = self.seconds(start), self.seconds(end)
        step = max(length_s - overlap_s, length_s * 0.25)
        starts, t = [], t0
        while t < t1 - 1.0:
            starts.append(t)
            t += step
        return starts
