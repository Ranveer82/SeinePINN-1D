"""
Tidal constituents and the harmonic time embedding.

The tidal branch of the network writes the tide as a sum of constituents with
known periods. This module lists those periods and turns a time into the
``cos`` and ``sin`` of each constituent.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

#: Constituent name -> period in mean solar hours.
CONSTITUENTS: dict[str, float] = {
    # semi-diurnal
    "M2": 12.420601,     # principal lunar
    "S2": 12.000000,     # principal solar (M2 and S2 beat into spring-neap)
    "N2": 12.658348,     # larger lunar elliptic
    "K2": 11.967235,     # lunisolar
    "NU2": 12.626004,    # larger lunar evectional
    "L2": 12.191620,     # smaller lunar elliptic
    # diurnal
    "K1": 23.934470,
    "O1": 25.819342,
    "P1": 24.065890,
    "Q1": 26.868357,
    # shallow-water overtides and compound tides
    "M4": 6.210300,      # first overtide, tidal asymmetry
    "MS4": 6.103339,
    "MN4": 6.269174,
    "M6": 4.140200,      # second overtide
    "MK3": 8.177140,
    "M8": 3.105150,
    "2MS6": 4.092050,
    # long period
    "MSF": 354.367200,   # 14.77 d, the spring-neap cycle
    "MF": 327.859000,    # 13.66 d
    "MM": 661.310000,    # 27.55 d
    "SSA": 4382.905000,  # 182.6 d
    "SA": 8765.810000,   # 365.25 d
}

#: Default set of constituents for a meso- or macrotidal estuary (16).
DEFAULT_TIDAL = [
    "M2", "S2", "N2", "K2", "NU2", "L2",
    "K1", "O1", "P1", "Q1",
    "M4", "MS4", "MN4", "M6", "MK3", "M8",
]

#: Default for a river without tide: no tidal constituents.
DEFAULT_NONTIDAL: list[str] = []


def resolve_constituents(spec, tidal: bool) -> list[str]:
    """
    Turn the case setting into a checked list of constituent names.

    Args:
        spec (str | list): ``"auto"`` for the default set, or a list of names.
        tidal (bool): Whether the downstream boundary is tidal.

    Returns:
        list[str]: Upper-case constituent names.
    """
    if not tidal:
        if spec in (None, "auto"):
            return list(DEFAULT_NONTIDAL)
        names = list(spec)
    elif spec in (None, "auto"):
        names = list(DEFAULT_TIDAL)
    else:
        names = list(spec)

    unknown = [n for n in names if n.upper() not in CONSTITUENTS]
    if unknown:
        raise ValueError(
            f"Unknown tidal constituent(s) {unknown}. "
            f"Known: {sorted(CONSTITUENTS)}"
        )
    return [n.upper() for n in names]


def periods_hours(names) -> np.ndarray:
    """Periods (h) of a list of constituent names."""
    return np.array([CONSTITUENTS[n.upper()] for n in names], dtype=np.float64)


class HarmonicEmbedding(nn.Module):
    """
    Maps time to ``[cos(w_k t), sin(w_k t)]`` for a fixed set of periods.

    The frequencies are fixed (astronomy, not something to learn). The phase is
    reduced to one period before taking the cosine and sine, which keeps the
    argument small over a record of many years without changing the result.
    """

    def __init__(self, periods_h):
        """
        Args:
            periods_h (array-like): Periods in hours.
        """
        super().__init__()
        p = torch.as_tensor(np.asarray(periods_h, dtype=np.float64))
        self.register_buffer("periods_s", p * 3600.0)
        self.n_periods = int(p.numel())

    @property
    def out_dim(self) -> int:
        """Number of output features (two per period)."""
        return 2 * self.n_periods

    def forward(self, t_sec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_sec (torch.Tensor): Time in seconds since the case epoch, ``(N, 1)``.

        Returns:
            torch.Tensor: ``(N, 2K)``, all cosines first, then all sines.
        """
        if self.n_periods == 0:
            return t_sec.new_zeros((t_sec.shape[0], 0))
        u = t_sec / self.periods_s                     # number of revolutions
        phase = 2.0 * np.pi * torch.remainder(u, 1.0)
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)
