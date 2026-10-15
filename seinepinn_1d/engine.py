"""
Training, checkpoints, evaluation and prediction.

Training runs in four phases:

1. a short warm start on the data only, to put the solution in the right range;
2. a march through the training period in overlapping time windows, each
   starting from the state at the end of the previous one, with the physics
   on and causal weighting in time inside each window;
3. a refinement pass over randomly chosen windows, so early windows are not
   forgotten;
4. optionally, an L-BFGS polish.

Working window by window also keeps the LSTM roll-out short, which is what
makes it affordable to train the LSTM together with the rest.
"""

from __future__ import annotations

import copy
import json
import os
import time
from collections import deque

import numpy as np
import pandas as pd
import torch

from .balance import GradientBalancer, NON_SHARED
from .config import DEVICE, CaseConfig
from .data import RiverDataset
from .geometry import build_geometry, pretrain_surrogate, surrogate_report
from .loss import (LateralInflow, boundary_and_data_loss, initial_condition_loss,
                   physics_loss, residuals, volume_balance_loss)
from .models import RiverPINN
from .scales import Scales, derive_scales
from .utils import interp1d, metric_table

DAY = 86400.0


class LossGuard:
    """
    Decides whether a training step diverged or the loss simply changed level.

    A step whose loss is more than ``factor`` times the median of the recent
    accepted losses is rejected, and the caller restores the last good state.
    The reference is forgotten when the objective changes (warm start to
    physics), because the loss level then changes legitimately. After
    ``patience`` rejections in a row the guard takes the new level as normal
    and lets training continue, unless the loss is non-finite or more than
    ``runaway`` times the median, which stops training.

    ``verdict`` returns ``(decision, reason)``, with decision one of
    ``"accept"``, ``"reject"``, ``"rebaseline"`` or ``"fatal"``.
    """

    def __init__(self, factor: float, patience: int, window: int = 50,
                 runaway: float = 1e6):
        """
        Args:
            factor (float): Loss ratio to the running median that is rejected
                (``0`` = guard off).
            patience (int): Rejections in a row before re-baselining.
            window (int): Number of accepted losses in the running median.
            runaway (float): Ratio that is never accepted as a new level.
        """
        self.factor = float(factor or 0.0)
        self.patience = int(patience or 0)
        self.runaway = float(runaway)
        self.recent = deque(maxlen=window)
        self.rejected_run = 0
        self.worst_ratio = 0.0
        self.objective = None

    def verdict(self, value: float, objective):
        """
        Judge one step.

        Args:
            value (float): Total loss of the step.
            objective: Label of the current objective (phase, physics on/off).

        Returns:
            tuple: ``(decision, reason)``.
        """
        if objective != self.objective:      # new objective, new scale
            self.objective = objective
            self.recent.clear()
            self.rejected_run = 0
            self.worst_ratio = 0.0

        if not np.isfinite(value):
            self.rejected_run += 1
            if self.patience and self.rejected_run >= self.patience:
                return "fatal", (f"non-finite loss {self.rejected_run} times "
                                 f"in a row")
            return "reject", "non-finite loss"

        if not (self.factor and len(self.recent) == self.recent.maxlen):
            self.rejected_run = 0
            self.worst_ratio = 0.0
            return "accept", ""

        median = float(np.median(self.recent))
        ratio = value / max(median, 1e-300)
        if value > self.factor * median:
            self.rejected_run += 1
            self.worst_ratio = max(self.worst_ratio, ratio)
            if self.patience and self.rejected_run >= self.patience:
                if self.worst_ratio > self.runaway:
                    return "fatal", (
                        f"loss reached {self.worst_ratio:.2e}x its running "
                        f"median and stayed there for {self.rejected_run} "
                        f"rollbacks; something in the residual is unbounded")
                reason = (f"{self.rejected_run} consecutive rejections at loss "
                          f"~{value:.3e} against a median of {median:.3e}; "
                          f"taking this as the new level")
                self.recent.clear()
                self.rejected_run = 0
                self.worst_ratio = 0.0
                return "rebaseline", reason
            return "reject", (f"loss {value:.3e} exceeds {self.factor:g}x the "
                              f"running median {median:.3e}")

        self.rejected_run = 0
        self.worst_ratio = 0.0
        return "accept", ""

    def accept(self, value: float) -> None:
        """Record the loss of an accepted step (rejected ones are not recorded)."""
        self.recent.append(value)


def _stage_range(dataset):
    """Lowest and highest water level in the forcing record and the gauges."""
    lo, hi = float(dataset.eta_down.min()), float(dataset.eta_down.max())
    if dataset.gauges is not None and len(dataset.gauges):
        g = dataset.gauges["eta"].to_numpy(float)
        lo, hi = min(lo, float(g.min())), max(hi, float(g.max()))
    return lo, hi


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def adaptive_draw(model, lateral, window, x_pool, t_pool, n: int,
                  power: float = 1.0, floor: float = 0.1):
    """
    Residual-adaptive sampling (RAD, Wu et al. 2023).

    Scores a pool of candidate points by their PDE residual and draws ``n``
    of them with probability proportional to ``|residual|^power``. The scores
    are capped at their 99th percentile, so a few points that cannot be fitted
    do not take all the samples, and a share ``floor`` of uniform probability
    keeps every region in play.

    Args:
        model (RiverPINN): The network.
        lateral (LateralInflow): Tributary inflows.
        window (Window): The window.
        x_pool, t_pool (torch.Tensor): Candidate points.
        n (int): Number of points to draw.
        power (float): Exponent on the residual.
        floor (float): Share of uniform probability.

    Returns:
        tuple: Drawn ``(x, t)``.
    """
    x_pool = x_pool.detach().requires_grad_(True)
    t_pool = t_pool.detach().requires_grad_(True)
    ctx = model.context(window)
    r_mass, r_mom, _, _ = residuals(model, x_pool, t_pool, ctx, lateral,
                                    create_graph=False)
    score = (r_mass.detach() ** 2 + r_mom.detach() ** 2).reshape(-1)
    score = score.clamp_min(0).pow(0.5 * power)

    if score.numel() > 100:
        cap = torch.quantile(score, 0.99)
        score = score.clamp_max(cap.clamp_min(1e-30))

    prob = score / score.sum().clamp_min(1e-30)
    prob = (1.0 - floor) * prob + floor / prob.numel()
    n = min(n, prob.numel())
    idx = torch.multinomial(prob, n, replacement=False)
    return x_pool.detach()[idx], t_pool.detach()[idx]


class Case:
    """Everything needed to train or run one river, assembled once."""

    def __init__(self, cfg: CaseConfig, verbose: bool = True,
                 data_dir: str | None = None, pretrain_steps: int | None = None):
        """
        Read the data, build the geometry, fit the geometry surrogate and
        create the network.

        Args:
            cfg (CaseConfig): The case.
            verbose (bool): Print progress.
            data_dir (str, optional): Read the forcing from this folder instead
                of the case ``data_dir`` (e.g. new forcing to predict).
            pretrain_steps (int, optional): Steps to fit the geometry
                surrogate (default 10000). Use ``0`` when a checkpoint will be
                loaded: its surrogate weights replace the fit anyway.
        """
        cfg.validate()
        cfg.make_dirs()
        self.cfg = cfg
        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)

        if verbose:
            print(f"[{cfg.name}] assembling case")
        steps = pretrain_steps if pretrain_steps is not None else 10000

        self.dataset = RiverDataset(cfg, data_dir=data_dir, verbose=verbose)
        self.library, self.geometry, self.surrogate, self.geo_info = \
            build_geometry(cfg, verbose=verbose,
                           stage_range=_stage_range(self.dataset))
        if verbose and steps > 0:
            print(f"  fitting geometry surrogate ({steps} steps)")
        pretrain_surrogate(self.surrogate, self.geometry, steps=steps,
                           verbose=verbose)
        self.surrogate_error = (surrogate_report(self.surrogate, self.geometry)
                                if steps > 0 else {})
        if verbose and self.surrogate_error:
            print("  surrogate: " + ", ".join(
                f"{k}={v:.3%}" for k, v in self.surrogate_error.items()))

        self.scales = derive_scales(cfg, self.dataset, self.geometry, self.library)
        if verbose:
            print("  scales: " + self.scales.summary())

        self.model = RiverPINN(cfg, self.scales, self.geometry, self.surrogate,
                               self.dataset.stats).to(DEVICE)
        self.lateral = LateralInflow(cfg, self.library.x_min, self.library.x_max,
                                     DEVICE)
        #: Filled by ``train()`` or by ``load_checkpoint()``.
        self.train_summary = None

        if verbose:
            self._warn_setup()

    def _warn_setup(self) -> None:
        """Warn about setups that run but should not be trusted."""
        if self.dataset.gauges is None or not len(self.dataset.gauges):
            print("  WARNING: no gauge observations inside the reach: the data "
                  "loss is zero and nothing anchors the solution between the "
                  "two boundaries.")
        b_err = self.surrogate_error.get("B_rel_err")
        if b_err is not None and 0.02 < b_err <= 0.05:
            print(f"  note: geometry surrogate top-width error {b_err:.1%}. "
                  f"Under the 5% limit, but consider more --geo-steps.")
        if b_err is not None and b_err > 0.05:
            print(f"  WARNING: geometry surrogate top-width error {b_err:.1%}. "
                  f"B multiplies the storage term of continuity; above ~5% "
                  f"the mass balance cannot close. Raise --geo-steps, or "
                  f"model.geo_fourier / model.geo_hidden_dim.")
        up = float(self.dataset.Q_up_mask.mean())
        if up < 0.8:
            print(f"  WARNING: upstream discharge only {up:.1%} observed; the "
                  f"boundary loss is switched off for the rest.")

    # --------------------------------------------------------------- helpers
    def collocation(self, window, n: int, sobol, adaptive: bool = False):
        """
        Collocation points (where the equations are checked) inside a window.

        Sobol points in ``(x, t)``; with ``adaptive`` a share of them is drawn
        where the residual is large (see ``adaptive_draw``).

        Args:
            window (Window): The window.
            n (int): Number of points.
            sobol (SobolEngine): Quasi-random generator.
            adaptive (bool): Use residual-adaptive sampling for a share.

        Returns:
            tuple: ``(x, t)`` with ``requires_grad``.
        """
        tc = self.cfg.train
        n_ad = int(n * tc.adaptive_fraction) if adaptive else 0
        n_uni = n - n_ad

        def _draw(count):
            p = sobol.draw(count).to(DEVICE)
            x = self.library.x_min + p[:, 0:1] * self.library.length
            t = window.t0 + p[:, 1:2] * (window.t1 - window.t0)
            return x, t

        x, t = _draw(max(n_uni, 1))
        if n_ad > 0:
            xp, tp = _draw(n_ad * max(tc.adaptive_pool, 2))
            xa, ta = adaptive_draw(self.model, self.lateral, window, xp, tp,
                                   n_ad, tc.adaptive_power)
            x = torch.cat([x, xa], dim=0)
            t = torch.cat([t, ta], dim=0)

        return (x.detach().requires_grad_(True),
                t.detach().requires_grad_(True))

    def snapshot(self, window, t: float, n_x: int = 256):
        """
        The state along the reach at time ``t``, used to start the next window.

        Returns:
            tuple: ``(x, eta, u)`` on ``n_x`` points, detached.
        """
        x = torch.linspace(self.library.x_min, self.library.x_max, n_x,
                           device=DEVICE).unsqueeze(1)
        tt = torch.full_like(x, t)
        with torch.no_grad():
            ctx = self.model.context(window)
            f = self.model(x, tt, ctx, want_geometry=True, create_graph=False)
        return x.reshape(-1), f["eta"].reshape(-1), f["u"].reshape(-1)

    def gauge_reference(self, window):
        """
        Starting state of the first window from the gauge readings, if any.

        Returns:
            callable | None: ``x -> (eta, None)``, interpolating the gauges
            read within one hour of the window start.
        """
        w = window
        if w.gauge_t.numel() == 0:
            return None
        m = (w.gauge_t - w.t0).abs() < 3600.0
        if int(m.sum()) < 2:
            return None
        x = w.gauge_x[m]
        eta = w.gauge_eta[m]
        order = torch.argsort(x)
        xs, es = x[order], eta[order]

        def ref(xq):
            return interp1d(xq, xs, es), None
        return ref

    @staticmethod
    def make_reference(x_nodes, eta_nodes, u_nodes):
        """A reference state ``x -> (eta, u)`` from a snapshot."""
        def ref(xq):
            return (interp1d(xq, x_nodes, eta_nodes),
                    interp1d(xq, x_nodes, u_nodes))
        return ref


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(case: Case, verbose: bool = True, log_every: int = 100):
    """
    Train the model of a case and save it.

    Writes ``models/model.pt`` and ``stats/training_history.csv`` in the case
    output folder.

    Args:
        case (Case): The assembled case.
        verbose (bool): Print progress.
        log_every (int): Steps between log lines (and history rows).

    Returns:
        pd.DataFrame: The training history.
    """
    cfg = case.cfg
    tc = cfg.train
    model, ds = case.model, case.dataset

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=tc.lr)

    win_len = tc.window_days * DAY
    overlap = tc.window_overlap_days * DAY
    starts = ds.window_starts(cfg.train_start, cfg.train_end, win_len, overlap)
    stride = max(int(getattr(tc, "window_stride", 1) or 1), 1)
    if stride > 1:
        # keep the last window: the refinement and the next period start there
        kept = starts[::stride]
        if starts and kept[-1] != starts[-1]:
            kept = kept + [starts[-1]]
        if verbose:
            print(f"  march stride {stride}: {len(kept)} of {len(starts)} "
                  f"windows, evenly spread over the record")
        starts = kept
    if not starts:
        raise ValueError("training period is shorter than one window")

    total_steps = tc.warmup_steps + len(starts) * tc.march_steps + tc.refine_steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(total_steps, 1), eta_min=tc.lr_final)
    balancer = GradientBalancer(tc.weights, anchor=GradientBalancer.PHYSICS,
                                alpha=tc.balance_alpha,
                                fixed=set(NON_SHARED))

    sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=tc.seed)
    history, snapshots = [], {}
    step = 0
    t_start = time.time()

    def window_for(t0):
        return ds.window(t0, min(t0 + win_len, ds.seconds(cfg.train_end)))

    def assemble(window, reference, use_physics, adaptive):
        """All loss terms for one optimiser evaluation."""
        ctx = model.context(window)
        losses, diag = {}, {}

        if use_physics:
            x, t = case.collocation(window, tc.n_collocation, sobol,
                                    adaptive=adaptive)
            pl, _, pdiag, _ = physics_loss(model, x, t, ctx, case.lateral, cfg)
            losses.update(pl)
            diag.update(pdiag)
            if cfg.physics.volume_constraint:
                losses["volume"] = volume_balance_loss(model, ctx, case.lateral,
                                                       cfg)
                diag["vol_rel"] = float(losses["volume"].detach().sqrt())

        bl, bdiag = boundary_and_data_loss(model, ctx, cfg, case.scales, tc.n_data,
                                           include_bounds=not use_physics)
        losses.update(bl)
        diag.update(bdiag)

        if reference is not None:
            il, idiag = initial_condition_loss(model, ctx, cfg, case.scales, reference)
            losses.update(il)
            diag.update(idiag)

        return losses, diag

    # Divergence guard: the state is saved now and then, and a step whose loss
    # leaves its normal range is undone (see LossGuard), with a lower
    # learning rate.
    healthy = {"state": None, "opt": None, "step": -1}
    guard = LossGuard(tc.divergence_factor, tc.divergence_patience)
    rollbacks, accepted = 0, 0
    lr_scale = 1.0

    def snapshot():
        healthy["state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        healthy["opt"] = copy.deepcopy(opt.state_dict())
        healthy["step"] = step

    def rollback(reason):
        nonlocal rollbacks, lr_scale
        rollbacks += 1
        if healthy["state"] is None:
            raise FloatingPointError(
                f"diverged at step {step} before any good state was saved: {reason}")
        model.load_state_dict(healthy["state"])
        opt.load_state_dict(healthy["opt"])
        # the reduction is kept outside the optimiser state, so repeated
        # rollbacks keep lowering the learning rate
        lr_scale *= tc.divergence_lr_decay
        for g in opt.param_groups:
            g["lr"] = max(g["lr"] * lr_scale, tc.lr_final)
        if verbose:
            print(f"  !! step {step}: {reason} -- rolled back to step "
                  f"{healthy['step']}, lr -> {opt.param_groups[0]['lr']:.2e} "
                  f"(rollback #{rollbacks})")

    def run_step(window, phase, reference, use_physics):
        nonlocal step, accepted
        opt.zero_grad(set_to_none=True)
        adaptive = (use_physics and tc.adaptive_fraction > 0
                    and step >= tc.adaptive_after)
        losses, diag = assemble(window, reference, use_physics, adaptive)

        if (use_physics and tc.balance_every
                and step % tc.balance_every == 0 and step > 0):
            balancer.update(losses, params)

        total = balancer.total(losses)
        value = float(total.detach())

        verdict, why = guard.verdict(value, (phase, bool(use_physics)))
        if verdict == "reject":
            rollback(why)
            step += 1
            return
        if verdict == "fatal":
            raise FloatingPointError(
                f"step {step}: {why}; rolling back to step {healthy['step']} "
                f"did not help, so the run cannot recover")
        if verdict == "rebaseline" and verbose:
            print(f"  !! step {step}: {why}")

        total.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
        opt.step()
        sched.step()
        guard.accept(value)
        accepted += 1
        if tc.snapshot_every and step % tc.snapshot_every == 0:
            snapshot()

        if step % log_every == 0:
            rec = {"step": step, "phase": phase, "t0": window.t0,
                   "total": float(total.detach()), "grad_norm": float(gnorm),
                   "lr": sched.get_last_lr()[0]}
            rec.update({f"L_{k}": float(v.detach()) for k, v in losses.items()})
            rec.update({f"w_{k}": balancer[k] for k in losses})
            rec["w_saturated"] = len(balancer.last_saturated)
            rec.update(diag)
            history.append(rec)
            if verbose:
                bits = " ".join(f"{k}={rec[k]:.2e}" for k in
                                ("L_mass", "L_momentum", "L_data", "L_bc_up")
                                if k in rec)
                extra = ""
                if "data_rmse_m" in rec:
                    extra += f" stage_rmse={rec['data_rmse_m']:.3f}m"
                if "bc_up_rmse_m3s" in rec:
                    extra += f" Qup_rmse={rec['bc_up_rmse_m3s']:.1f}"
                now = time.time()
                el = now - t_start
                rate = step / el if el > 0 else float("nan")
                print(f"  [{phase:7s}] step {step:6d} total={rec['total']:.3e} "
                      f"{bits}{extra}"
                      f"  | {time.strftime('%H:%M:%S')} +{el/60:.1f}m "
                      f"{rate:.1f} steps/s")
        step += 1

    # phase 1: warm start on the data only
    if verbose:
        print(f"\n[{cfg.name}] warm start ({tc.warmup_steps} steps, data only)")
    w0 = window_for(starts[0])
    for _ in range(tc.warmup_steps):
        run_step(w0, "warmup", None, use_physics=False)

    # phase 2: march through the training period
    if verbose:
        print(f"\n[{cfg.name}] causal march over {len(starts)} windows "
              f"({tc.window_days:.0f} d each, {tc.march_steps} steps per window)")
    reference = case.gauge_reference(w0)
    for wi, t0 in enumerate(starts):
        window = window_for(t0)
        for _ in range(tc.march_steps):
            run_step(window, "march", reference, use_physics=True)
        snap = case.snapshot(window, min(t0 + win_len - overlap, window.t1))
        snapshots[t0] = snap
        reference = case.make_reference(*snap)
        if verbose:
            print(f"  window {wi+1}/{len(starts)} done "
                  f"({pd.to_datetime(ds.epoch) + pd.Timedelta(seconds=t0)})")

    # phase 3: refinement over random windows
    if verbose:
        print(f"\n[{cfg.name}] global refinement ({tc.refine_steps} steps)")
    rng = np.random.default_rng(tc.seed)
    for _ in range(tc.refine_steps):
        t0 = float(starts[rng.integers(len(starts))])
        run_step(window_for(t0), "refine", None, use_physics=True)

    # phase 4: L-BFGS polish
    if tc.lbfgs_steps > 0:
        _lbfgs_polish(case, assemble, balancer, starts, window_for, history,
                      verbose)

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n[{cfg.name}] trained {step} steps in {elapsed/60:.1f} min, "
              f"{accepted} accepted"
              + (f", {rollbacks} rolled back" if rollbacks else ""))
    if step and accepted < 0.9 * step:
        msg = (f"only {accepted}/{step} steps ({accepted/step:.1%}) were "
               f"accepted; the rest were rolled back by the divergence guard. "
               f"The model is effectively frozen at step {healthy['step']} and "
               f"the checkpoint should not be trusted.")
        print(f"\n  {'!' * 70}\n  WARNING: {msg}\n  {'!' * 70}")

    case.train_summary = {
        "steps": int(step),
        "accepted": int(accepted),
        "accepted_frac": float(accepted / step) if step else 0.0,
        "rollbacks": int(rollbacks),
        "frozen_at": int(healthy["step"]),
        "minutes": float(elapsed / 60.0),
        "balancer_saturation": float(balancer.saturation),
    }

    # A saturated balancer (weights at the ceiling) is harmless when the data
    # terms have simply converged, and a warning sign when the physics
    # residual has a floor it cannot get below. The residual tells which.
    tail = [h for h in history[-20:] if "L_mass" in h]
    floor = max((max(h.get(f"L_{k}", 0.0) for k in balancer.anchor)
                 for h in tail), default=0.0) ** 0.5
    if balancer.saturation > 0.25 and floor < 0.5 * GATES["res_mass_rms"]:
        print(f"\n  note: the loss weights were at their ceiling in "
              f"{balancer.saturation:.0%} of the updates, but the physics "
              f"residual is {floor:.2f} against a limit of "
              f"{GATES['res_mass_rms']:.2f}: the data terms have converged. "
              f"Nothing to act on.")
    elif balancer.saturation > 0.25:
        print(f"\n  {'!' * 70}\n  WARNING: the loss weights were at their "
              f"ceiling in {balancer.saturation:.0%} of the updates. The "
              f"physics residual has a floor this model cannot reach; look "
              f"for its cause (geometry too rough, a forcing that cannot be "
              f"represented) before tuning anything.\n  {'!' * 70}")

    hist = pd.DataFrame(history)
    hist.to_csv(os.path.join(cfg.stats_dir, "training_history.csv"), index=False)
    save_checkpoint(case, os.path.join(cfg.model_dir, "model.pt"),
                    weights=balancer.w)
    return hist


def _lbfgs_polish(case: Case, assemble, balancer, starts, window_for,
                  history, verbose: bool):
    """
    Second-order refinement after Adam, window by window.

    L-BFGS needs a fixed objective, so each window keeps one batch of
    collocation points (no adaptive sampling) for all its inner iterations.
    """
    cfg, tc, model = case.cfg, case.cfg.train, case.model
    params = [p for p in model.parameters() if p.requires_grad]
    if verbose:
        print(f"\n[{cfg.name}] L-BFGS polish ({tc.lbfgs_steps} steps "
              f"over {len(starts)} windows)")

    per_window = max(tc.lbfgs_steps // max(len(starts), 1), 1)
    for wi, t0 in enumerate(starts):
        window = window_for(float(t0))
        opt = torch.optim.LBFGS(
            params, lr=1.0, max_iter=per_window,
            history_size=tc.lbfgs_history, line_search_fn="strong_wolfe",
            tolerance_grad=1e-12, tolerance_change=1e-14,
        )
        state = {"n": 0, "last": float("nan")}

        def closure():
            opt.zero_grad(set_to_none=True)
            losses, _ = assemble(window, None, True, adaptive=False)
            total = balancer.total(losses)
            if not torch.isfinite(total):
                # a large finite value makes the line search step back
                return torch.full_like(total, 1e6)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1e3)
            state["n"] += 1
            state["last"] = float(total.detach())
            return total

        try:
            opt.step(closure)
        except (RuntimeError, FloatingPointError) as exc:
            if verbose:
                print(f"  window {wi+1}: L-BFGS stopped early ({exc})")
            continue
        history.append({"step": -1, "phase": "lbfgs", "t0": float(t0),
                        "total": state["last"], "lbfgs_evals": state["n"]})
        if verbose:
            print(f"  window {wi+1}/{len(starts)}: {state['n']} evaluations, "
                  f"total={state['last']:.3e}")


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
#: Model settings that change what the outputs mean without changing the
#: shape of the weights. They are stored with the weights and win on loading.
FORMULATION = ("discharge_output", "lstm_float32")

#: Normalisation of the geometry surrogate inputs and outputs. These are set
#: from the stage range of the data the case was built on and are not part of
#: the weights, so they are stored separately.
SURROGATE_NORM = ("eta_min", "eta_max", "A_scale", "P_scale")


def surrogate_norm(surrogate) -> dict:
    """The surrogate normalisation as a dictionary."""
    return {k: float(getattr(surrogate, k)) for k in SURROGATE_NORM}


def save_checkpoint(case: Case, path: str, weights: dict | None = None):
    """
    Save the trained model with everything needed to use it again.

    Stored: the weights, the scales, the forcing normalisation, the surrogate
    normalisation and errors, the loss weights, the constituents, the
    formulation flags and the training summary.

    Args:
        case (Case): The trained case.
        path (str): Output file (``.pt``).
        weights (dict, optional): Final loss weights.
    """
    torch.save({
        "model": case.model.state_dict(),
        "scales": case.scales.as_dict(),
        "dataset_stats": case.dataset.stats,
        "surrogate_error": case.surrogate_error,
        "weights": weights or {},
        "constituents": case.model.constituents,
        "formulation": {k: getattr(case.cfg.model, k, None)
                        for k in FORMULATION},
        "surrogate_norm": surrogate_norm(case.model.surrogate),
        "train_summary": case.train_summary,
    }, path)


def restore_surrogate_norm(case: Case, norm: dict, verbose: bool = True):
    """
    Put back the surrogate normalisation the weights were fitted with.

    The normalisation comes from the stage range of the data a case is built
    on. With new forcing (another period, a scenario) that range differs, and
    the same weights would then describe a different channel.

    Args:
        case (Case): The case the checkpoint is loaded into.
        norm (dict): Stored normalisation.
        verbose (bool): Say when it differs from this record's own.
    """
    seen = {}
    for s in (getattr(case, "surrogate", None), case.model.surrogate):
        if s is not None:
            seen[id(s)] = s
    for s in seen.values():
        before = surrogate_norm(s)
        for k in SURROGATE_NORM:
            setattr(s, k, float(norm[k]))
    if verbose and any(abs(before[k] - float(norm[k])) > 1e-9 * max(1.0, abs(before[k]))
                       for k in SURROGATE_NORM):
        print("  geometry: surrogate normalisation restored to the training "
              "values (stage band {:.2f}..{:.2f} m; this record's own would "
              "have been {:.2f}..{:.2f} m)".format(
                  norm["eta_min"], norm["eta_max"],
                  before["eta_min"], before["eta_max"]))


def restore_dataset_stats(dataset, stats: dict):
    """
    Normalise the LSTM input with the statistics of the training record.

    With new forcing (e.g. a sea-level-rise scenario) the record's own mean
    would subtract the change before the LSTM sees it. On the training record
    this changes nothing.

    Args:
        dataset (RiverDataset): The dataset of the case.
        stats (dict): Stored statistics.
    """
    fresh = dict(dataset.stats)
    dataset.stats = dict(stats)
    dataset._lstm = torch.as_tensor(dataset._lstm_features(), device=DEVICE)
    moved = {k: (fresh[k] - stats[k]) / max(abs(stats.get(k.replace("_mean", "_std"), 1.0)), 1e-9)
             for k in ("Q_up_mean", "eta_mean") if k in fresh and k in stats}
    if any(abs(v) > 0.05 for v in moved.values()):
        print("  normalisation: using the training statistics; the forcing here "
              "sits " + ", ".join(f"{k.replace('_mean', '')} {v:+.2f} sd"
                                  for k, v in moved.items())
              + " from the training mean")


def load_checkpoint(case: Case, path: str):
    """
    Load a trained model into a case, with the scales and normalisations it
    was trained with.

    Re-deriving the scales from other data (e.g. new forcing) would silently
    rescale every prediction, so the stored ones are used.

    Args:
        case (Case): The case (built with ``pretrain_steps=0``).
        path (str): Checkpoint file.

    Returns:
        dict: The raw checkpoint.
    """
    ck = torch.load(path, map_location=DEVICE, weights_only=False)

    # formulation flags: the stored value is what the weights mean
    for key, stored in (ck.get("formulation") or {}).items():
        if stored is None or not hasattr(case.cfg.model, key):
            continue
        current = getattr(case.cfg.model, key)
        if bool(stored) != bool(current):
            print(f"  note: checkpoint was trained with model.{key} = "
                  f"{stored!r}, the case says {current!r}; using the "
                  f"checkpoint's value.")
            setattr(case.cfg.model, key, stored)
            if key == "discharge_output":
                case.model.cfg.model.discharge_output = stored

    case.model.load_state_dict(ck["model"])
    if "scales" in ck:
        stored = Scales(**ck["scales"])
        case.scales = stored
        case.model.scales = stored
    case.surrogate_error = ck.get("surrogate_error", case.surrogate_error)
    if ck.get("surrogate_norm"):
        restore_surrogate_norm(case, ck["surrogate_norm"])
    else:
        print("  note: this checkpoint does not store the surrogate "
              "normalisation; it is only right on the training record.")
    if ck.get("dataset_stats"):
        restore_dataset_stats(case.dataset, ck["dataset_stats"])
    if ck.get("train_summary"):
        case.train_summary = dict(ck["train_summary"])
    case.model.eval()
    return ck


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def score_gauges(case: Case, win, ctx=None) -> dict:
    """
    Stage skill at the gauges of a window, pooled and per station.

    Every station of the gauge data is scored. Velocity is also scored
    where it is observed (the synthetic case).

    Args:
        case (Case): The case with its trained model.
        win (Window): The window.
        ctx (Context, optional): Encoded window.

    Returns:
        dict: ``stage``, ``stage_by_station`` (by chainage) and, if
        available, ``velocity``.
    """
    model = case.model
    out: dict = {}
    if win.gauge_t.numel() == 0:
        return out
    if ctx is None:
        with torch.no_grad():
            ctx = model.context(win)
    eta_sim, u_sim = [], []
    for i in range(0, win.gauge_t.numel(), 8192):
        xb = win.gauge_x[i:i + 8192].reshape(-1, 1)
        tb = win.gauge_t[i:i + 8192].reshape(-1, 1)
        with torch.no_grad():
            f = model(xb, tb, ctx, want_geometry=False)
        eta_sim.append(f["eta"].reshape(-1).cpu().numpy())
        u_sim.append(f["u"].reshape(-1).cpu().numpy())
    eta_sim = np.concatenate(eta_sim)
    obs = win.gauge_eta.cpu().numpy()
    xs = win.gauge_x.cpu().numpy()
    out["stage"] = metric_table(eta_sim, obs)
    if win.gauge_u is not None:
        out["velocity"] = metric_table(np.concatenate(u_sim),
                                       win.gauge_u.cpu().numpy())
    per = {}
    for xv in np.unique(xs):
        m = xs == xv
        per[float(xv)] = metric_table(eta_sim[m], obs[m])
    out["stage_by_station"] = per
    return out


def evaluate(case: Case, start, end, n_physics: int = 20000,
             verbose: bool = True) -> dict:
    """
    Score the model over a period: skill, physics and admissibility.

    Reports stage skill at the gauges, the non-dimensional PDE residuals on
    unseen points, the plausibility of the fields, the roughness closure, the
    volume balance of the reach, and a verdict against the admissibility
    gates (``GATES``). A model can fit stage well without conserving mass, so
    skill alone is not enough.

    Args:
        case (Case): The case with its trained model.
        start, end: The period.
        n_physics (int): Points where the residuals are evaluated.
        verbose (bool): Print the report.

    Returns:
        dict: The report.
    """
    cfg, model, ds = case.cfg, case.model, case.dataset
    win = ds.window(start, end)
    with torch.no_grad():
        ctx = model.context(win)
    out: dict = score_gauges(case, win, ctx)

    # PDE residuals on points never used in training
    sob = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=12345)
    p = sob.draw(n_physics).to(DEVICE)
    x = (case.library.x_min + p[:, 0:1] * case.library.length).requires_grad_(True)
    t = (win.t0 + p[:, 1:2] * (win.t1 - win.t0)).requires_grad_(True)
    ctx_g = model.context(win)
    r_mass, r_mom, fields, _ = residuals(model, x, t, ctx_g, case.lateral)
    out["physics"] = {
        "res_mass_rms": float(r_mass.detach().pow(2).mean().sqrt()),
        "res_mom_rms": float(r_mom.detach().pow(2).mean().sqrt()),
    }
    out["fields"] = {
        "depth_min": float(fields["depth"].detach().min()),
        "depth_mean": float(fields["depth"].detach().mean()),
        "u_absmax": float(fields["u"].detach().abs().max()),
        "u_absmean": float(fields["u"].detach().abs().mean()),
        "Q_absmax": float(fields["Q"].detach().abs().max()),
    }

    # the fitted closures, and how many control points sit on a bound
    with torch.no_grad():
        n_grid = model.manning_grid().cpu().numpy()
    lo, hi = model.n_min, model.n_max
    out["closures"] = {"manning": {
        "min": float(n_grid.min()), "median": float(np.median(n_grid)),
        "max": float(n_grid.max()),
        "frac_at_ceiling": float((n_grid > hi - 1e-3 * (hi - lo)).mean()),
        "frac_at_floor": float((n_grid < lo + 1e-3 * (hi - lo)).mean()),
    }}
    if getattr(model, "geo_tol", 0.0) > 0.0:
        with torch.no_grad():
            g = model.width_grid().cpu().numpy()
        out["closures"]["width_factor"] = {
            "min": float(g.min()), "median": float(np.median(g)),
            "max": float(g.max()),
            "frac_at_bound": float((np.abs(np.log(g)) >
                                    0.99 * model.geo_log_tol).mean()),
            "tolerance": float(model.geo_tol),
        }

    out["mass_balance"] = _mass_balance(case, win, ctx)

    if case.train_summary is not None:
        out["training"] = dict(case.train_summary)
    out["verdict"] = _verdict(out, case)

    if verbose:
        print(json.dumps({k: v for k, v in out.items()
                          if k != "stage_by_station"}, indent=2, default=float))
    return out


#: Admissibility gates. A model that fails one is not a hydrodynamic model,
#: whatever its stage skill, and should not be ranked with models that pass.
GATES = {
    #: Share of training steps that were applied (not rolled back).
    "accepted_frac": 0.90,
    #: RMS of (dV/dt - net inflow) over the reach, relative to the mean net
    #: inflow: the sharpest single test of mass conservation.
    "mass_balance": 0.25,
    #: Non-dimensional PDE residuals. Each residual is divided by its leading
    #: term, so 1 would mean nothing cancels; the synthetic case reaches about
    #: 0.1, and 0.25 allows a factor of two above that.
    "res_mass_rms": 0.25,
    "res_mom_rms": 0.25,
}


def _verdict(report: dict, case: Case) -> dict:
    """
    Check a report against the admissibility gates.

    Returns:
        dict: ``admissible`` (bool), ``failed`` (list) and ``checks``
        (value, limit and pass of each gate).
    """
    checks, failed = {}, []

    def check(name, value, limit, worse_is_bigger=True):
        if value is None:
            return
        ok = (value <= limit) if worse_is_bigger else (value >= limit)
        checks[name] = {"value": float(value), "limit": float(limit),
                        "pass": bool(ok)}
        if not ok:
            failed.append(name)

    tr = case.train_summary or {}
    if tr:
        check("accepted_frac", tr.get("accepted_frac"), GATES["accepted_frac"],
              worse_is_bigger=False)
    check("mass_balance", report.get("mass_balance", {}).get("relative"),
          GATES["mass_balance"])
    check("res_mass_rms", report.get("physics", {}).get("res_mass_rms"),
          GATES["res_mass_rms"])
    check("res_mom_rms", report.get("physics", {}).get("res_mom_rms"),
          GATES["res_mom_rms"])

    # plausibility of the fields (same bounds as the training penalty)
    f = report.get("fields", {})
    if "depth_min" in f:
        checks["depth_min"] = {"value": float(f["depth_min"]),
                               "limit": float(case.cfg.physics.min_depth),
                               "pass": f["depth_min"] >= case.cfg.physics.min_depth}
        if not checks["depth_min"]["pass"]:
            failed.append("depth_min")
    if "u_absmax" in f:
        check("u_absmax", f["u_absmax"], case.scales.u_max)

    return {"admissible": not failed, "failed": failed, "checks": checks}


def _mass_balance(case: Case, win, ctx) -> dict:
    """
    Volume balance of the reach: ``dV/dt`` against the net inflow.

    Computed hourly over the first two weeks of the window (at most).

    Returns:
        dict: RMS imbalance (m3/s) and the same relative to the mean net inflow.
    """
    model = case.model
    n_x, n_t = 200, min(win.n_hours, 24 * 14)
    x = torch.linspace(case.library.x_min, case.library.x_max, n_x,
                       device=DEVICE).unsqueeze(1)
    ts = win.t_hourly[:n_t]
    vols, q_in, q_out = [], [], []
    with torch.no_grad():
        for tv in ts:
            tt = torch.full_like(x, float(tv))
            f = model(x, tt, ctx)
            vols.append(float(torch.trapz(f["A"].reshape(-1), x.reshape(-1))))
            q_in.append(float(f["Q"][0]))
            q_out.append(float(f["Q"][-1]))
    vols = np.asarray(vols)
    dt = float(ts[1] - ts[0]) if len(ts) > 1 else 3600.0
    dVdt = np.gradient(vols, dt)
    trib = (win.Q_trib[:n_t].sum(dim=1).detach().cpu().numpy()
            if win.Q_trib.shape[1] else np.zeros(len(q_in)))
    net = np.asarray(q_in) - np.asarray(q_out) + trib[:len(q_in)]
    scale = max(np.abs(net).mean(), 1e-6)
    return {"dVdt_minus_net_rms": float(np.sqrt(np.mean((dVdt - net) ** 2))),
            "relative": float(np.sqrt(np.mean((dVdt - net) ** 2)) / scale)}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_grid(case: Case, start, end, dx: float = 250.0,
                 dt_hours: float = 1.0, spinup_days: float = 0.0) -> pd.DataFrame:
    """
    Stage, velocity and discharge on a regular grid of chainage and time.

    Stage is returned in the vertical datum of the input files (the case
    ``datum_shift`` is removed). The chainage runs every ``dx`` from the
    upstream end, plus the downstream end itself.

    The LSTM starts from rest at the start of its window, and the first day of
    a window is off by up to a metre on the Seine. ``spinup_days`` starts the
    window that much earlier and leaves the spin-up out of the output; two
    days are already enough.

    Args:
        case (Case): The case with its trained model.
        start, end: The period.
        dx (float): Chainage step (m).
        dt_hours (float): Time step (h).
        spinup_days (float): Forcing read before ``start`` (days).

    Returns:
        pd.DataFrame: Columns ``timestamp``, ``X_m``, ``stage_m``,
        ``velocity_ms``, ``discharge_m3s`` and ``depth_m``.
    """
    model, ds = case.model, case.dataset
    datum = float(case.cfg.datum_shift or 0.0)
    t_start = ds.seconds(start)
    win = ds.window(t_start - spinup_days * DAY, ds.seconds(end))
    x0, x1 = case.library.x_min, case.library.x_max
    xs = np.arange(x0, x1 + 1e-6, dx)
    if x1 - xs[-1] > 1e-6:
        xs = np.append(xs, x1)
    ts = np.arange(max(t_start, win.t0), win.t1 + 1.0, dt_hours * 3600.0)

    x = torch.as_tensor(xs, device=DEVICE).unsqueeze(1)
    rows = []
    with torch.no_grad():
        ctx = model.context(win)
        for tv in ts:
            tt = torch.full_like(x, float(tv))
            f = model(x, tt, ctx)
            rows.append(pd.DataFrame({
                "t_sec": tv, "X_m": xs,
                "stage_m": f["eta"].reshape(-1).cpu().numpy() - datum,
                "velocity_ms": f["u"].reshape(-1).cpu().numpy(),
                "discharge_m3s": f["Q"].reshape(-1).cpu().numpy(),
                "depth_m": f["depth"].reshape(-1).cpu().numpy(),
            }))
    df = pd.concat(rows, ignore_index=True)
    df["timestamp"] = ds.epoch + pd.to_timedelta(df["t_sec"], unit="s")
    cols = ["timestamp", "X_m", "stage_m", "velocity_ms", "discharge_m3s",
            "depth_m"]
    return df[cols]
