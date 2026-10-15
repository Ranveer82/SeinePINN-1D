#!/usr/bin/env python
"""
Train the model of a case.

    python run_train.py cases/seine.yaml
    python run_train.py cases/synthetic.yaml --set train.refine_steps=2000

Any option of the case file can be changed from the command line with
``--set``, e.g. ``--set train.refine_steps=2000 model.hidden_dim=64``. This is
handy for a short test run before a full training.

What this does:

1. reads the forcing, the gauges and the bathymetry of the case;
2. builds the channel geometry and fits the geometry surrogate;
3. trains the network over the training period (warm start, march through
   the windows, refinement, optional L-BFGS);
4. writes, in the case ``output_dir``:

   - ``models/model.pt``            the trained model
   - ``stats/training_history.csv`` losses, weights and diagnostics
   - ``stats/config.json``          the full configuration of the run

Then score it with ``run_evaluate.py`` and use it with ``run_predict.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

import torch

from seinepinn_1d.config import CaseConfig, apply_overrides
from seinepinn_1d.engine import Case, train


def main():
    ap = argparse.ArgumentParser(description="Train the model of a case.")
    ap.add_argument("case", help="case file, e.g. cases/seine.yaml")
    ap.add_argument("--set", nargs="*", dest="overrides", default=[],
                    metavar="KEY=VALUE",
                    help="change case options, e.g. train.refine_steps=2000")
    ap.add_argument("--geo-steps", type=int, default=10000,
                    help="steps to fit the geometry surrogate (default 10000)")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU threads for torch (0 = torch default)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = CaseConfig.from_yaml(args.case)
    apply_overrides(cfg, args.overrides)

    case = Case(cfg, pretrain_steps=args.geo_steps)

    # record what produced this run before it produces anything
    with open(os.path.join(cfg.stats_dir, "config.json"), "w") as fh:
        json.dump({"case": args.case, "overrides": args.overrides,
                   "geo_steps": args.geo_steps,
                   "config": dataclasses.asdict(cfg)},
                  fh, indent=2, default=str)

    train(case)
    print(f"\nModel written to {os.path.join(cfg.model_dir, 'model.pt')}")
    print(f"Next: python run_evaluate.py {args.case} --period validation")


if __name__ == "__main__":
    main()
