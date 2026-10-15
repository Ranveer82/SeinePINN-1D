"""
Tests of SeinePINN_1D.

    pytest tests/

The unit tests take a second. The end-to-end test makes a small synthetic
estuary (20 days), trains for a few dozen steps, evaluates, predicts, and
checks that a saved model gives the same prediction after loading; it takes
one to three minutes on a CPU.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from seinepinn_1d.config import CaseConfig, apply_overrides          # noqa: E402
from seinepinn_1d.geometry import section_properties                 # noqa: E402
from seinepinn_1d.utils import catmull_rom, interp1d, nse            # noqa: E402


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
def test_rectangle_section():
    """A rectangular channel 10 m wide, filled 2 m deep."""
    d = np.array([0.0, 0.0, 10.0, 10.0])
    z = np.array([5.0, 0.0, 0.0, 5.0])
    A, W, P = section_properties(d, z, 2.0)
    assert A == pytest.approx(20.0)
    assert W == pytest.approx(10.0)
    assert P == pytest.approx(14.0)


def test_dry_section():
    """A section above the water is dry."""
    d = np.array([0.0, 5.0, 10.0])
    z = np.array([3.0, 1.0, 3.0])
    assert section_properties(d, z, 0.5) == (0.0, 0.0, 0.0)


def test_catmull_rom_passes_through_samples():
    seq = torch.tensor([[0.0], [1.0], [4.0], [9.0], [16.0]])
    pos = torch.arange(5, dtype=torch.float64).reshape(-1, 1)
    assert torch.allclose(catmull_rom(seq, pos), seq)


def test_interp1d():
    x = torch.tensor([0.0, 1.0, 2.0])
    y = torch.tensor([0.0, 10.0, 0.0])
    q = torch.tensor([0.5, 1.5])
    assert torch.allclose(interp1d(q, x, y), torch.tensor([5.0, 5.0]))


def test_nse_perfect():
    obs = np.array([1.0, 2.0, 3.0])
    assert nse(obs, obs) == pytest.approx(1.0)


def test_both_case_files_parse():
    for name in ("seine", "synthetic"):
        cfg = CaseConfig.from_yaml(os.path.join(HERE, "cases", f"{name}.yaml"))
        assert cfg.train_start and cfg.test_end
        assert cfg.upstream is not None and cfg.downstream is not None


def test_unknown_option_is_refused():
    cfg = CaseConfig.from_yaml(os.path.join(HERE, "cases", "synthetic.yaml"))
    raw = {"name": "x", "data_dir": ".", "output_dir": ".", "geometry": "g.csv",
           "upstream": "u.csv", "downstream": "d.csv",
           "train_start": "2020-01-01", "train_end": "2020-02-01",
           "train": {"refine_stpes": 10}}
    with pytest.raises(ValueError, match="refine_stpes"):
        CaseConfig.from_dict(raw)
    assert cfg is not None


def test_override_boolean_false():
    cfg = CaseConfig.from_yaml(os.path.join(HERE, "cases", "seine.yaml"))
    assert cfg.model.discharge_output is True
    apply_overrides(cfg, ["model.discharge_output=false"], verbose=False)
    assert cfg.model.discharge_output is False


# ---------------------------------------------------------------------------
# End to end on a small synthetic estuary
# ---------------------------------------------------------------------------
def test_train_evaluate_predict(tmp_path):
    from seinepinn_1d.engine import (Case, train, evaluate, predict_grid,
                                     load_checkpoint)

    data = tmp_path / "data"
    subprocess.run([sys.executable, os.path.join(HERE, "tools", "make_synthetic_case.py"),
                    "--out", str(data), "--days", "20", "--cells", "40",
                    "--dt", "30"], check=True, capture_output=True)

    raw = {
        "name": "tiny", "data_dir": str(data), "output_dir": str(tmp_path / "out"),
        "geometry": "river_profiles.csv",
        "upstream": {"file": "upstream_Q.csv", "value_col": "discharge"},
        "downstream": {"file": "downstream_stage.csv", "value_col": "stage"},
        "tributaries_file": {"file": "tributaries.csv"},
        "tributaries": [{"name": "trib_upper", "column": "trib_upper", "x": 8800.0},
                        {"name": "trib_lower", "column": "trib_lower", "x": 27200.0}],
        "gauges": {"file": "gauges.csv", "value_col": "stage_m"},
        "gauge_velocity_col": "velocity_ms",
        "epoch": "2020-01-01 00:00:00",
        "train_start": "2020-01-01 00:00:00", "train_end": "2020-01-11 00:00:00",
        "val_start": "2020-01-11 00:00:00", "val_end": "2020-01-15 00:00:00",
        "test_start": "2020-01-15 00:00:00", "test_end": "2020-01-20 00:00:00",
        "physics": {"constituents": ["M2", "S2", "M4"],
                    "subtidal_periods_h": [354.367, 168.0], "manning_nodes": 6},
        "train": {"window_days": 5.0, "window_overlap_days": 1.0,
                  "n_collocation": 256, "n_data": 128, "warmup_steps": 10,
                  "march_steps": 8, "refine_steps": 10, "balance_every": 5,
                  "adaptive_after": 20, "lbfgs_steps": 4, "snapshot_every": 5},
        "model": {"hidden_dim": 32, "n_layers": 3, "lstm_hidden": 8,
                  "geo_hidden_dim": 32, "geo_layers": 3, "geo_fourier": 0},
    }
    cfg = CaseConfig.from_dict(raw)
    case = Case(cfg, pretrain_steps=100, verbose=False)
    hist = train(case, verbose=False, log_every=5)
    assert len(hist) > 0 and np.isfinite(hist["total"]).all()
    ck = os.path.join(cfg.model_dir, "model.pt")
    assert os.path.exists(ck)

    report = evaluate(case, cfg.val_start, cfg.val_end, n_physics=500,
                      verbose=False)
    for key in ("stage", "physics", "mass_balance", "verdict"):
        assert key in report
    assert np.isfinite(report["stage"]["rmse"])

    before = predict_grid(case, cfg.test_start, cfg.test_end, dx=2000.0)
    assert np.isfinite(before[["stage_m", "velocity_ms", "discharge_m3s"]]
                       .to_numpy()).all()

    # a new case loading the checkpoint gives the same prediction
    case2 = Case(CaseConfig.from_dict(raw), pretrain_steps=0, verbose=False)
    load_checkpoint(case2, ck)
    after = predict_grid(case2, cfg.test_start, cfg.test_end, dx=2000.0)
    assert np.allclose(before["stage_m"], after["stage_m"], atol=1e-10)
    assert np.allclose(before["discharge_m3s"], after["discharge_m3s"], atol=1e-8)
