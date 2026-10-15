"""
Case configuration.

Everything that belongs to one river is described in a case file (YAML) and
read into a :class:`CaseConfig`. The rest of the package never hard-codes a
river name, a datum, a tributary, a date or a physical range.

A case needs at least:

* a bathymetry CSV (cross sections),
* an upstream discharge series,
* a downstream stage series,

and optionally tributary inflows and gauge observations. See
``cases/seine.yaml`` and ``cases/synthetic.yaml`` for complete examples.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field, asdict, fields
from typing import Optional

import torch
import yaml

# ---------------------------------------------------------------------------
# Hardware and precision
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The Saint-Venant residuals are small differences of large terms, so the
# whole model runs in double precision.
torch.set_default_dtype(torch.float64)

#: Gravitational acceleration (m/s2).
G = 9.80665


# ---------------------------------------------------------------------------
# Reading YAML values into typed fields
# ---------------------------------------------------------------------------
# YAML 1.1 reads `1.0e3` (no sign in the exponent) as a string, not a number.
# Every value is therefore converted to the type of its field when the case
# file is read, and a value that cannot be converted is reported by name.
_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


def _cast(value, annotation: str, where: str):
    """
    Convert one YAML value to the type of its field.

    Args:
        value: Value read from the YAML file.
        annotation (str): Type annotation of the field, e.g. ``"float"``.
        where (str): Name of the option, used in error messages.

    Returns:
        The converted value.
    """
    ann = str(annotation).replace(" ", "")
    optional = ann.startswith("Optional[") or "None" in ann
    if ann.startswith("Optional[") and ann.endswith("]"):
        ann = ann[len("Optional["):-1]

    if value is None:
        if optional:
            return None
        raise ValueError(f"case option '{where}' may not be null")

    def fail(kind):
        raise ValueError(
            f"case option '{where}' should be {kind}, got {value!r} "
            f"({type(value).__name__}). Note that YAML needs a signed exponent: "
            f"write 1.0e+3 or 1000.0, not 1.0e3."
        )

    if ann == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in _TRUE | _FALSE:
            return value.lower() in _TRUE
        fail("a boolean")
    if ann == "int":
        if isinstance(value, bool):
            fail("an integer")
        try:
            return int(value)
        except (TypeError, ValueError):
            fail("an integer")
    if ann == "float":
        if isinstance(value, bool):
            fail("a number")
        try:
            return float(value)
        except (TypeError, ValueError):
            fail("a number")
    if ann == "str" and not isinstance(value, str):
        # An unquoted date such as 2010-01-01 00:00:00 is read by YAML as a
        # datetime; keep it as text, the rest of the code parses it.
        if isinstance(value, (dt.datetime, dt.date)):
            return value.isoformat(sep=" ") if isinstance(value, dt.datetime) \
                else value.isoformat()
        fail("a string")
    return value


def _checked(cls, raw: dict, where: str, skip: set | None = None) -> dict:
    """
    Check the option names of one YAML block and convert their values.

    Args:
        cls: Dataclass the block is read into.
        raw (dict): The block as read from YAML.
        where (str): Name of the block, used in error messages.
        skip (set, optional): Keys passed through unchanged.

    Returns:
        dict: Keyword arguments for ``cls``.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"'{where}' should be a mapping, got {type(raw).__name__}")
    types = {f.name: f.type for f in fields(cls)}
    out = {}
    for key, value in raw.items():
        if skip and key in skip:
            out[key] = value
            continue
        if key not in types:
            close = sorted(k for k in types if k.startswith(str(key)[:3]))
            hint = f" Did you mean {close}?" if close else ""
            raise ValueError(
                f"unknown option '{key}' in '{where}'.{hint} "
                f"Valid options: {sorted(types)}"
            )
        out[key] = _cast(value, types[key], f"{where}.{key}")
    return out


# ---------------------------------------------------------------------------
# Parts of a case
# ---------------------------------------------------------------------------
@dataclass
class Tributary:
    """A lateral inflow entering the reach at a fixed chainage."""

    #: Name used in the logs.
    name: str
    #: Column of the tributary file holding its discharge (m3/s).
    column: str
    #: Chainage of the confluence (m).
    x: float
    #: Length over which the inflow is spread along the reach (m). ``None``
    #: uses ``physics.lateral_sigma``.
    sigma: Optional[float] = None


@dataclass
class SeriesSpec:
    """One input CSV (forcing or observations) and the columns to read."""

    #: File name, relative to the case ``data_dir``.
    file: str
    #: Column with the values. ``None`` lets the reader find it.
    value_col: Optional[str] = None
    #: Column with the timestamps.
    time_col: str = "timestamp"
    #: Added to every value on load, to move a local gauge zero to the
    #: vertical datum of the case. Only used for stage series.
    datum_shift: float = 0.0
    #: Multiplied into every value on load (e.g. -1 to flip a sign).
    scale: float = 1.0
    #: Day/month order for dates such as ``01/02/2007``. ``None`` (default)
    #: detects it from the file. ISO dates (``2007-02-01``) need nothing.
    dayfirst: Optional[bool] = None
    #: How samples finer than one hour are reduced to the hourly grid:
    #: ``"nearest"`` keeps the sample closest to each hour (an instantaneous
    #: value), ``"mean"`` averages the hour.
    subhourly: str = "nearest"


@dataclass
class PhysicsConfig:
    """Options of the Saint-Venant equations and of the channel geometry."""

    #: Is the downstream boundary tidal? Controls the harmonic basis.
    tidal: bool = True
    #: Tidal constituents: ``"auto"`` (16 constituents) or a list of names.
    constituents: str | list = "auto"
    #: Slow periods (h) used by the sub-tidal branch: annual, semi-annual,
    #: quarterly, 30 days, spring-neap (MSf) and weekly by default.
    subtidal_periods_h: list = field(
        default_factory=lambda: [8765.81, 4382.91, 2191.45, 720.0, 354.367, 168.0]
    )
    #: Bounds of the Manning roughness n(x) (s/m^(1/3)).
    manning_min: float = 0.012
    manning_max: float = 0.060
    #: Number of control points of the n(x) profile (fewer is smoother).
    manning_nodes: int = 16
    #: Smallest water depth allowed (m). Keeps A, R and the friction finite.
    min_depth: float = 0.05
    #: Share of the conveyance below which a wetted path (side channel, braid,
    #: pocket) only stores water and does not carry flow. ``0`` = every
    #: wetted path conveys.
    storage_threshold: float = 0.0
    #: Along-channel smoothing length of the tabulated area A(x) (m). Survey
    #: noise in A(x) enters continuity through dA/dx; features shorter than
    #: this length are smoothed out. ``0`` = off.
    geometry_smooth_m: float = 0.0
    #: Divided-channel method: conveyance of the main channel and of each
    #: floodplain is computed separately and summed, with the bank level found
    #: from each section's own width curve.
    compound_channel: bool = False
    #: Along-channel window used to make the detected bank level consistent (m).
    bankfull_smooth_m: float = 5000.0
    #: Remove the no-data fill a raster leaves in transects, and keep the run of
    #: valid points that contains the thalweg. Has no effect on clean surveys.
    trace_edges: bool = True
    #: Minimum repeats of the transect maximum before it can be called a fill.
    nodata_min_count: int = 3
    #: Interior no-data gaps narrower than this are interpolated across (m).
    nodata_bridge_m: float = 40.0
    #: Close sections that stop below dry ground with a vertical wall on the
    #: last surveyed point (lower bound on storage) instead of extending the
    #: banks at the floodplain slope (upper bound, the default).
    flood_barrier: bool = False
    #: Learnable correction on the channel width, as a fraction (``0.3`` allows
    #: +/-30 %). ``0`` = off.
    geometry_tolerance: float = 0.0
    #: Number of control points of that width correction.
    geometry_nodes: int = 16
    #: Crest of the flood barrier relative to the reach bank line (m).
    barrier_freeboard_m: float = 0.0
    #: Add the reach-integrated volume balance dV/dt = Q(0) - Q(L) + tributaries
    #: as a loss term. It is implied by continuity, but a small residual of
    #: constant sign adds up over the reach, and this term prevents that.
    volume_constraint: bool = True
    #: Default spreading length of the tributary inflows (m).
    lateral_sigma: float = 500.0
    #: Eddy viscosity in the momentum equation (m2/s). ``0`` = off.
    eddy_viscosity: float = 0.0
    #: Momentum brought by tributaries, as a fraction of the main-channel
    #: velocity (0 = perpendicular inflow).
    lateral_momentum: float = 0.0


@dataclass
class TrainConfig:
    """Training schedule and sampling."""

    #: Length of one training window (days). On a tidal river it should cover
    #: at least one spring-neap cycle.
    window_days: float = 30.0
    #: Overlap between consecutive windows (days).
    window_overlap_days: float = 3.0
    #: Collocation points (where the equations are checked) per step.
    n_collocation: int = 8192
    #: Boundary and observation points per step.
    n_data: int = 2048
    #: Steps of the data-only warm start.
    warmup_steps: int = 400
    #: Steps per window during the march through the training period.
    march_steps: int = 600
    #: Steps of the final pass over randomly chosen windows.
    refine_steps: int = 20000
    #: Learning rate at the start and at the end (cosine schedule).
    lr: float = 1e-3
    lr_final: float = 1e-5
    #: Gradient norm clipping.
    grad_clip: float = 1.0
    #: Strength of the causal weighting in time (Wang et al. 2022). ``0`` = off.
    #: The last part of a window is trained at ``exp(-causal_eps)`` of the first.
    causal_eps: float = 1.0
    #: Number of time bins of the causal weighting.
    causal_bins: int = 32
    #: Smallest causal weight, so no part of a window is ever left out.
    causal_floor: float = 0.05
    #: March over every Nth window only (1 = all windows).
    window_stride: int = 1
    #: Weight the gauge loss so every station counts the same, whatever the
    #: length of its record.
    station_weighted_data: bool = False
    #: Update the adaptive loss weights every N steps. ``0`` = fixed weights.
    balance_every: int = 100
    #: Memory of the adaptive weights (moving average factor).
    balance_alpha: float = 0.9
    #: Share of the collocation points drawn where the residual is large
    #: (residual-adaptive sampling, Wu et al. 2023). ``0`` = off.
    adaptive_fraction: float = 0.5
    #: Size of the candidate pool for that draw, as a multiple of the draw.
    adaptive_pool: int = 4
    #: Exponent on the residual in the sampling density.
    adaptive_power: float = 1.0
    #: First step at which adaptive sampling is used.
    adaptive_after: int = 1000
    #: L-BFGS steps after Adam. ``0`` = off.
    lbfgs_steps: int = 0
    lbfgs_history: int = 50
    #: Divergence guard: a step whose loss is more than this many times its
    #: recent median is rejected and the last good state is restored.
    #: ``0`` = off.
    divergence_factor: float = 1.0e3
    #: Steps between saved good states.
    snapshot_every: int = 200
    #: Learning-rate factor applied after each rollback.
    divergence_lr_decay: float = 0.5
    #: Consecutive rejections after which the guard takes the new loss level
    #: as normal and continues.
    divergence_patience: int = 10
    #: Starting loss weights. They stay fixed when ``balance_every`` is 0.
    weights: dict = field(
        default_factory=lambda: {
            "mass": 1.0,
            "momentum": 1.0,
            "bc_up": 10.0,
            "data": 10.0,
            "ic": 10.0,
            "bounds": 100.0,
            "smooth_n": 0.1,
            "smooth_geo": 0.1,
        }
    )
    #: Random seed.
    seed: int = 0


@dataclass
class ModelConfig:
    """Shape of the networks."""

    #: Width and depth of the sub-tidal, tidal and residual MLPs.
    hidden_dim: int = 128
    n_layers: int = 5
    #: Activation function: ``SiLU``, ``Tanh`` or ``GELU``.
    activation: str = "SiLU"
    #: Run the LSTM in single precision (the rest stays in double). Useful
    #: when the GPU library refuses a double-precision LSTM.
    lstm_float32: bool = False
    #: Size and number of layers of the LSTM encoder of the forcing.
    lstm_hidden: int = 48
    lstm_layers: int = 1
    #: Add a free residual branch on top of the harmonic tidal branch. It
    #: starts at zero, so training starts from the harmonic solution.
    residual_branch: bool = True
    #: The second output of the network is the discharge Q (``true``) or the
    #: velocity u (``false``). With Q, dQ/dx in continuity is the derivative
    #: of one smooth output and u = Q/A follows.
    discharge_output: bool = False
    #: Shape of the geometry surrogate A(x, eta), P(x, eta).
    geo_hidden_dim: int = 128
    geo_layers: int = 5
    #: Number of Fourier frequencies in chainage for the surrogate, and the
    #: ratio of the longest to the shortest wavelength. ``0`` = plain MLP.
    geo_fourier: int = 32
    geo_max_period_ratio: float = 400.0


@dataclass
class CaseConfig:
    """The complete description of one river."""

    #: Name of the case, used in the logs.
    name: str
    #: Folder of the input files (relative to the case file).
    data_dir: str
    #: Folder of the outputs (relative to the case file).
    output_dir: str

    #: Bathymetry CSV, relative to ``data_dir``.
    geometry: str
    #: Discharge at the upstream end (x = 0).
    upstream: SeriesSpec = None
    #: Stage at the downstream end (x = L).
    downstream: SeriesSpec = None
    #: File with the tributary discharges, and the list of tributaries.
    tributaries_file: Optional[SeriesSpec] = None
    tributaries: list = field(default_factory=list)
    #: Stage observations inside the reach: one file, or a folder with one
    #: file per station.
    gauges: Optional[SeriesSpec] = None

    #: Column with the chainage of each gauge reading.
    gauge_x_col: str = "X_m"
    #: Optional observed velocity column. Never used in training; only to
    #: score the velocity of the synthetic case.
    gauge_velocity_col: Optional[str] = None
    #: Keep one gauge reading per station every this many hours (the reading
    #: nearest each grid time). ``None`` keeps every reading.
    gauge_step_h: Optional[float] = None

    #: Offset added to all elevations (bathymetry, stage, gauges) so the
    #: working datum is positive. Results are converted back for output.
    datum_shift: Optional[float] = None

    #: Reference date: model time is in seconds since this date.
    epoch: str = "2000-01-01 00:00:00"
    #: Training, validation and test periods, in this order and not
    #: overlapping. Training is used by the loss, validation for any choice
    #: made by looking at a score, and the test period is scored once.
    train_start: str = None
    train_end: str = None
    val_start: Optional[str] = None
    val_end: Optional[str] = None
    test_start: Optional[str] = None
    test_end: Optional[str] = None

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # ---------------------------------------------------------------- paths
    def path(self, rel: str) -> str:
        """
        Full path of an input file, also finding a compressed copy.

        Args:
            rel (str): File name relative to ``data_dir`` (or an absolute path).

        Returns:
            str: The path, with ``.gz``, ``.bz2``, ``.xz`` or ``.zip`` appended
            when only the compressed file exists.
        """
        full = rel if os.path.isabs(rel) else os.path.join(self.data_dir, rel)
        if not os.path.exists(full):
            for suffix in (".gz", ".bz2", ".xz", ".zip"):
                if os.path.exists(full + suffix):
                    return full + suffix
        return full

    @property
    def model_dir(self) -> str:
        """Folder of the trained model (``model.pt``)."""
        return os.path.join(self.output_dir, "models")

    @property
    def stats_dir(self) -> str:
        """Folder of the training history and the evaluation reports."""
        return os.path.join(self.output_dir, "stats")

    @property
    def export_dir(self) -> str:
        """Folder of the gridded predictions."""
        return os.path.join(self.output_dir, "exports")

    def make_dirs(self) -> None:
        """Create the output folders."""
        for d in (self.output_dir, self.model_dir, self.stats_dir,
                  self.export_dir):
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------ read/write
    @classmethod
    def from_yaml(cls, path: str) -> "CaseConfig":
        """
        Read a case file.

        Args:
            path (str): Path of the YAML case file. ``data_dir`` and
                ``output_dir`` in it are relative to this file.

        Returns:
            CaseConfig: The case.
        """
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return cls.from_dict(raw, base_dir=os.path.dirname(os.path.abspath(path)))

    @classmethod
    def from_dict(cls, raw: dict, base_dir: str = ".") -> "CaseConfig":
        """
        Build a case from a dictionary with the layout of a case file.

        Args:
            raw (dict): Case options.
            base_dir (str): Folder that relative ``data_dir`` and
                ``output_dir`` refer to.

        Returns:
            CaseConfig: The case.
        """
        raw = dict(raw)

        def _series(key, default_value_col=None):
            spec = raw.pop(key, None)
            if spec is None:
                return None
            if isinstance(spec, str):
                spec = {"file": spec}
            spec.setdefault("value_col", default_value_col)
            return SeriesSpec(**_checked(SeriesSpec, spec, key))

        up = _series("upstream")
        down = _series("downstream")
        tribs_file = _series("tributaries_file")
        gauges = _series("gauges")

        tribs = [Tributary(**_checked(Tributary, t, f"tributaries[{i}]"))
                 for i, t in enumerate(raw.pop("tributaries", []))]
        physics = PhysicsConfig(**_checked(PhysicsConfig, raw.pop("physics", {}), "physics"))
        train = TrainConfig(**_checked(TrainConfig, raw.pop("train", {}), "train"))
        model = ModelConfig(**_checked(ModelConfig, raw.pop("model", {}), "model"))

        for key in ("data_dir", "output_dir"):
            if key in raw and not os.path.isabs(raw[key]):
                raw[key] = os.path.normpath(os.path.join(base_dir, raw[key]))

        raw = _checked(cls, raw, "case", skip={
            "upstream", "downstream", "tributaries_file", "gauges",
            "tributaries", "physics", "train", "model"})
        return cls(upstream=up, downstream=down, tributaries_file=tribs_file,
                   gauges=gauges, tributaries=tribs, physics=physics,
                   train=train, model=model, **raw)

    def to_yaml(self, path: str) -> None:
        """Write the case, with every option filled in, to a YAML file."""
        with open(path, "w") as fh:
            yaml.safe_dump(asdict(self), fh, sort_keys=False)

    # ---------------------------------------------------------------- checks
    def validate(self) -> None:
        """
        Check that the case is complete and that the periods are in order.

        Raises:
            ValueError: On a missing boundary, a missing training period,
                overlapping or unordered periods, or invalid physical bounds.
        """
        if self.upstream is None or self.downstream is None:
            raise ValueError(
                "A case needs both an 'upstream' discharge series and a "
                "'downstream' stage series: subcritical 1-D flow needs one "
                "condition at each end of the reach."
            )
        if self.train_start is None or self.train_end is None:
            raise ValueError("train_start and train_end are required.")

        # Train, then validation, then test, without overlap: an overlap would
        # report a fitted score as an independent one.
        import pandas as _pd
        spans = [(n, self.__dict__.get(f"{n}_start"), self.__dict__.get(f"{n}_end"))
                 for n in ("train", "val", "test")]
        spans = [(n, _pd.to_datetime(a), _pd.to_datetime(b))
                 for n, a, b in spans if a and b]
        for n, a, b in spans:
            if b <= a:
                raise ValueError(f"{n}_end must be after {n}_start")
        for (n1, a1, b1), (n2, a2, b2) in zip(spans, spans[1:]):
            if a2 < b1:
                raise ValueError(
                    f"{n2}_start ({a2:%Y-%m-%d}) is inside the {n1} period, "
                    f"which ends {b1:%Y-%m-%d}. The periods must be in order "
                    f"(train, then val, then test) and must not overlap."
                )
        p = self.physics
        if not (0 < p.manning_min < p.manning_max):
            raise ValueError("Require 0 < manning_min < manning_max.")
        if p.min_depth <= 0:
            raise ValueError("min_depth must be > 0.")
        for t in self.tributaries:
            if t.sigma is None:
                t.sigma = p.lateral_sigma


def apply_overrides(cfg: "CaseConfig", assignments, verbose: bool = True):
    """
    Change case options from the command line.

    Args:
        cfg (CaseConfig): The case to change (in place).
        assignments (list[str]): Items like ``"train.refine_steps=2000"`` or
            ``"gauge_step_h=6"``. Lists and dictionaries are written as JSON.
        verbose (bool): Print each change.
    """
    for item in assignments or []:
        key, _, raw = item.partition("=")
        obj, *rest = key.split(".")
        target = getattr(cfg, obj) if rest else cfg
        attr = rest[-1] if rest else obj
        for step in rest[:-1]:
            target = getattr(target, step)
        current = getattr(target, attr)
        # Lists, dictionaries and unset options are read as JSON; the rest
        # are converted to the type of the current value.
        if current is None or isinstance(current, (list, tuple, dict)):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw                       # plain text, e.g. a path
        elif isinstance(current, bool):
            if raw.lower() not in _TRUE | _FALSE:
                raise ValueError(f"{key} needs true or false, got {raw!r}")
            value = raw.lower() in _TRUE
        else:
            try:
                value = type(current)(raw)
            except (TypeError, ValueError):
                value = json.loads(raw)
        setattr(target, attr, value)
        if verbose:
            print(f"  override {key} = {value!r}")
