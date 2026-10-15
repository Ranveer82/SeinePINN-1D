#!/usr/bin/env python
"""
Predict stage, velocity and discharge along the whole reach with a trained model.

    python run_predict.py cases/seine.yaml --start 2018-01-22 --end 2018-02-08
    python run_predict.py cases/seine.yaml --start 2018-01-22 --end 2018-02-08 \
        --checkpoint models/seine_headline.pt --dx 500 --dt-hours 0.5
    python run_predict.py cases/seine.yaml --data my_new_forcing/ \
        --start 2030-01-01 --end 2030-12-31
    python run_predict.py cases/seine.yaml --start 2007-01-01 --end 2025-12-31 \
        --checkpoint models/seine_headline.pt --dx 500 --yearly

The prediction needs only the boundary forcing (upstream discharge,
downstream stage and tributaries); gauge readings are not used. ``--data``
points to a folder with new forcing files in the same format as the case
(e.g. a scenario); gauges are optional there.

What this does:

1. loads the trained model (``models/model.pt`` of the case, or ``--checkpoint``),
   with the normalisation of the training record;
2. encodes the forcing with the LSTM, from ``--spinup-days`` before
   ``--start``: the LSTM starts from rest, and its first day is not reliable.
   The spin-up is not written;
3. evaluates the network on a regular grid of chainage (every ``--dx`` m, plus
   the downstream end) and time (``--dt-hours``), from ``--start`` to the end
   of the ``--end`` day (or to the ``--end`` time, if one is given);
4. writes ``exports/prediction_<start>_<end>.csv.gz`` in the case
   ``output_dir``, or one ``prediction_<year>.csv.gz`` per calendar year with
   ``--yearly``: a gzip-compressed CSV with columns ``timestamp``, ``X_m``,
   ``stage_m`` (in the datum of the input files), ``velocity_ms`` (positive
   seaward), ``discharge_m3s`` and ``depth_m``, rounded to mm, mm/s and
   0.1 m3/s. Read it back with ``pd.read_csv(path, parse_dates=["timestamp"])``.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import torch

from seinepinn_1d.config import CaseConfig
from seinepinn_1d.engine import Case, load_checkpoint, predict_grid

# decimals written per column: the model is not more precise than this, and
# it keeps the files small
DECIMALS = {"stage_m": 3, "velocity_ms": 3, "discharge_m3s": 1, "depth_m": 3}


def write(df: pd.DataFrame, path: str):
    """Round the fields and write them as a gzip-compressed CSV."""
    df = df.copy()
    for col, dec in DECIMALS.items():
        df[col] = df[col].round(dec) + 0.0          # + 0.0 turns -0.0 into 0.0
    df.to_csv(path, index=False, compression="gzip")
    print(f"wrote {path}  ({len(df):,} rows)")


def main():
    ap = argparse.ArgumentParser(description="Predict with a trained model.")
    ap.add_argument("case", help="case file, e.g. cases/seine.yaml")
    ap.add_argument("--start", required=True, help="first date, e.g. 2018-01-22")
    ap.add_argument("--end", required=True,
                    help="last date, e.g. 2018-02-08 (the whole day is predicted)")
    ap.add_argument("--data", default=None,
                    help="folder with new forcing files (default: the case data_dir)")
    ap.add_argument("--checkpoint", default=None,
                    help="model file (default: <output_dir>/models/model.pt)")
    ap.add_argument("--dx", type=float, default=250.0,
                    help="chainage step of the output grid, m (default 250)")
    ap.add_argument("--dt-hours", type=float, default=1.0,
                    help="time step of the output grid, h (default 1)")
    ap.add_argument("--spinup-days", type=float, default=10.0,
                    help="forcing read before --start to spin up the LSTM, "
                         "days (default 10)")
    ap.add_argument("--yearly", action="store_true",
                    help="write one file per calendar year")
    ap.add_argument("--out", default=None,
                    help="output file without extension (.csv.gz is added; "
                         "with --yearly, _<year>.csv.gz)")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU threads for torch (0 = torch default)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if ":" not in args.end:                         # a date: up to its last hour
        end += pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    cfg = CaseConfig.from_yaml(args.case)
    case = Case(cfg, data_dir=args.data, pretrain_steps=0)
    ck = args.checkpoint or os.path.join(cfg.model_dir, "model.pt")
    load_checkpoint(case, ck)
    print(f"loaded {ck}")

    if args.yearly:
        base = args.out or os.path.join(cfg.export_dir, "prediction")
        periods = [(max(start, pd.Timestamp(y, 1, 1)),
                    min(end, pd.Timestamp(y, 12, 31, 23, 59, 59)),
                    f"{base}_{y}.csv.gz")
                   for y in range(start.year, end.year + 1)]
    else:
        base = args.out or os.path.join(
            cfg.export_dir, f"prediction_{start:%Y%m%d}_{end:%Y%m%d}")
        periods = [(start, end, base + ".csv.gz")]
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

    for t0, t1, path in periods:
        df = predict_grid(case, t0, t1, dx=args.dx, dt_hours=args.dt_hours,
                          spinup_days=args.spinup_days)
        write(df, path)


if __name__ == "__main__":
    main()
