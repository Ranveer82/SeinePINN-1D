#!/usr/bin/env python
"""
Score a trained model over the validation or the test period.

    python run_evaluate.py cases/seine.yaml --period validation
    python run_evaluate.py cases/seine.yaml --period test
    python run_evaluate.py cases/seine.yaml --start 2018-01-01 --end 2018-03-01
    python run_evaluate.py cases/seine.yaml --period test --checkpoint models/seine_headline.pt

Use the validation period for every choice made by looking at a score (a
setting, a model). Score the test period once, at the end.

What this does:

1. loads the trained model (``models/model.pt`` of the case, or ``--checkpoint``);
2. scores the stage at the gauges, pooled and per station (RMSE, NSE, KGE);
3. computes the Saint-Venant residuals on points never used in training,
   the volume balance of the reach and the plausibility of the fields;
4. gives a verdict against the admissibility gates: a model that fits stage
   but fails a gate does not conserve mass and should not be used;
5. for the synthetic case (when ``truth_grid.npz`` is in the data folder),
   also scores the full stage, velocity and discharge fields, the M2
   amplitude along the reach and the recovered roughness against the truth.

Writes ``stats/<period>_report.json`` in the case ``output_dir`` (and, for the
synthetic case, ``stats/<period>_truth_report.json`` and
``stats/<period>_truth_profiles.csv``).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from seinepinn_1d.config import CaseConfig, DEVICE
from seinepinn_1d.engine import Case, evaluate, load_checkpoint
from seinepinn_1d.utils import metric_table

PERIODS = {"train": ("train_start", "train_end"),
           "validation": ("val_start", "val_end"),
           "test": ("test_start", "test_end")}


def score_against_truth(case, start, end, truth_path):
    """
    Score the model against the known solution of the synthetic case.

    Args:
        case (Case): The case with its trained model.
        start, end: The period.
        truth_path (str): ``truth_grid.npz`` written by
            ``tools/make_synthetic_case.py``.

    Returns:
        tuple: ``(report dict, per-chainage profiles DataFrame)``.
    """
    model, ds = case.model, case.dataset
    tr = np.load(truth_path, allow_pickle=True)
    t_all, x_ref = tr["t"], tr["x"]
    t0, t1 = ds.seconds(start), ds.seconds(end)
    sel = (t_all >= t0) & (t_all <= t1)
    t_sel = t_all[sel]
    eta_ref, u_ref, Q_ref = tr["eta"][sel], tr["u"][sel], tr["Q"][sel]
    n_true = tr["manning"]

    win = ds.window(start, end)
    x = torch.as_tensor(x_ref, device=DEVICE).unsqueeze(1)
    eta_sim = np.zeros_like(eta_ref)
    u_sim = np.zeros_like(u_ref)
    Q_sim = np.zeros_like(Q_ref)
    with torch.no_grad():
        ctx = model.context(win)
        for k, tv in enumerate(t_sel):
            f = model(x, torch.full_like(x, float(tv)), ctx)
            eta_sim[k] = f["eta"].reshape(-1).cpu().numpy()
            u_sim[k] = f["u"].reshape(-1).cpu().numpy()
            Q_sim[k] = f["Q"].reshape(-1).cpu().numpy()
        n_sim = model.manning(x)[0].reshape(-1).cpu().numpy()

    def band(sim, obs, lo, hi):
        m = (x_ref >= lo) & (x_ref < hi)
        return metric_table(sim[:, m].ravel(), obs[:, m].ravel())

    L = x_ref.max()
    thirds = {"upper_0_33pct": (0, 0.33 * L),
              "middle_33_66pct": (0.33 * L, 0.66 * L),
              "lower_66_100pct": (0.66 * L, L + 1)}

    def m2_amplitude(field):
        """M2 amplitude at each chainage, by harmonic projection."""
        w = 2 * np.pi / (12.420601 * 3600.0)
        c, s = np.cos(w * t_sel), np.sin(w * t_sel)
        n = t_sel.size
        a = 2.0 / n * (field * c[:, None]).sum(0)
        b = 2.0 / n * (field * s[:, None]).sum(0)
        return np.hypot(a, b)

    m2_sim, m2_ref = m2_amplitude(eta_sim), m2_amplitude(eta_ref)
    report = {
        "period": f"{start} .. {end}",
        "n_times": int(t_sel.size), "n_locations": int(x_ref.size),
        "stage_field": metric_table(eta_sim.ravel(), eta_ref.ravel()),
        "velocity_field": metric_table(u_sim.ravel(), u_ref.ravel()),
        "discharge_field": metric_table(Q_sim.ravel(), Q_ref.ravel()),
        "stage_by_reach": {k: band(eta_sim, eta_ref, *v) for k, v in thirds.items()},
        "velocity_by_reach": {k: band(u_sim, u_ref, *v) for k, v in thirds.items()},
        "M2_amplitude": {
            "mouth_true_m": float(m2_ref[-1]), "mouth_sim_m": float(m2_sim[-1]),
            "head_true_m": float(m2_ref[0]), "head_sim_m": float(m2_sim[0]),
            "profile_rmse_m": float(np.sqrt(np.mean((m2_sim - m2_ref) ** 2))),
        },
        # The variation ratio says whether the spatial pattern of n(x) was
        # recovered at all; a nearly flat profile can still correlate well.
        "manning": {
            "true_mean": float(n_true.mean()), "sim_mean": float(n_sim.mean()),
            "true_std": float(n_true.std()), "sim_std": float(n_sim.std()),
            "variation_ratio": float(n_sim.std() / max(n_true.std(), 1e-12)),
            "rmse": float(np.sqrt(np.mean((n_sim - n_true) ** 2))),
            "correlation": (float(np.corrcoef(n_sim, n_true)[0, 1])
                            if n_sim.std() > 1e-9 else float("nan")),
        },
    }
    profiles = pd.DataFrame({
        "X_m": x_ref, "manning_true": n_true, "manning_sim": n_sim,
        "M2_amp_true_m": m2_ref, "M2_amp_sim_m": m2_sim,
        "u_rmse_ms": np.sqrt(((u_sim - u_ref) ** 2).mean(0)),
        "stage_rmse_m": np.sqrt(((eta_sim - eta_ref) ** 2).mean(0)),
    })
    return report, profiles


def summary(report: dict) -> str:
    """A few lines with the main numbers of a report."""
    lines = []
    st = report.get("stage")
    if st:
        lines.append(f"  stage at the gauges : RMSE {st['rmse']:.3f} m, "
                     f"NSE {st['nse']:.4f}, KGE {st['kge']:.3f} "
                     f"({st['n']:,} readings)")
    ph = report.get("physics", {})
    if ph:
        lines.append(f"  PDE residuals       : mass {ph['res_mass_rms']:.3f}, "
                     f"momentum {ph['res_mom_rms']:.3f}  (gate 0.25)")
    mb = report.get("mass_balance", {})
    if mb:
        lines.append(f"  volume balance      : {mb['relative']:.3f}  (gate 0.25)")
    v = report.get("verdict", {})
    if v:
        lines.append("  admissible          : "
                     + ("yes" if v["admissible"] else f"NO, fails {v['failed']}"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Score a trained model.")
    ap.add_argument("case", help="case file, e.g. cases/seine.yaml")
    ap.add_argument("--period", choices=list(PERIODS), default="validation",
                    help="period of the case file to score (default validation)")
    ap.add_argument("--start", help="score a custom period instead (with --end)")
    ap.add_argument("--end")
    ap.add_argument("--checkpoint", default=None,
                    help="model file (default: <output_dir>/models/model.pt)")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU threads for torch (0 = torch default)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = CaseConfig.from_yaml(args.case)
    if args.start or args.end:
        if not (args.start and args.end):
            ap.error("--start and --end go together")
        start, end = args.start, args.end
        label = (f"{pd.to_datetime(start):%Y%m%d}_"
                 f"{pd.to_datetime(end):%Y%m%d}")
    else:
        a, b = PERIODS[args.period]
        start, end = getattr(cfg, a), getattr(cfg, b)
        if not (start and end):
            ap.error(f"the case file has no {args.period} period ({a}, {b})")
        label = args.period

    # the surrogate weights come from the checkpoint, so it is not refitted
    case = Case(cfg, pretrain_steps=0)
    ck = args.checkpoint or os.path.join(cfg.model_dir, "model.pt")
    load_checkpoint(case, ck)
    print(f"\n[{cfg.name}] {label}: {start} .. {end}  (model {ck})")

    report = evaluate(case, start, end, verbose=False)
    out = os.path.join(cfg.stats_dir, f"{label}_report.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(summary(report))
    print(f"  written to {out}")

    truth = os.path.join(cfg.data_dir, "truth_grid.npz")
    if os.path.exists(truth):
        rep, prof = score_against_truth(case, start, end, truth)
        out = os.path.join(cfg.stats_dir, f"{label}_truth_report.json")
        with open(out, "w") as fh:
            json.dump(rep, fh, indent=2, default=float)
        prof.to_csv(os.path.join(cfg.stats_dir, f"{label}_truth_profiles.csv"),
                    index=False)
        print(f"  against the known solution: velocity NSE "
              f"{rep['velocity_field']['nse']:.3f}, discharge NSE "
              f"{rep['discharge_field']['nse']:.3f}, M2 profile RMSE "
              f"{rep['M2_amplitude']['profile_rmse_m']:.3f} m")
        print(f"  written to {out}")


if __name__ == "__main__":
    main()
