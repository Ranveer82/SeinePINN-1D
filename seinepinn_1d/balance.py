"""
Adaptive loss weights.

Each data and boundary loss is weighted so that its gradient on the shared
network weights is as large as the gradient of the physics residual (Wang,
Teng & Perdikaris, 2021). The physics terms keep weight 1, so the equations
always lead and no weights have to be tuned for a new river.
"""

from __future__ import annotations

import torch


#: Terms that keep their fixed weight from the case file.
#:
#: These act on small, separate groups of parameters (the roughness profile,
#: the width correction), so their gradients are small by nature; balancing
#: them would push their weights to the ceiling and let a smoothness prior
#: flatten the roughness profile.
NON_SHARED = frozenset({"smooth_n", "smooth_geo", "bounds"})


class GradientBalancer:
    """
    Keeps the loss weights and updates them from the gradient norms.

    The weight of a term is moved towards
    ``|grad(physics)| / |grad(term)|``, smoothed in time, and clipped.
    """

    #: Largest weight a term can get.
    MAX_WEIGHT = 1e2

    #: Terms anchored together at weight 1 and never balanced against each
    #: other. Continuity and momentum are two halves of one system; letting
    #: one be traded against the other would let friction take over the
    #: momentum balance.
    PHYSICS = ("mass", "momentum", "volume")

    def __init__(self, weights: dict, anchor=PHYSICS, alpha: float = 0.9,
                 clip: tuple = (1e-3, MAX_WEIGHT), fixed=NON_SHARED):
        """
        Args:
            weights (dict): Starting weight of each loss term.
            anchor (tuple): Terms held at weight 1 (the physics).
            alpha (float): Memory of the moving average of the weights.
            clip (tuple): Smallest and largest allowed weight.
            fixed (iterable): Terms that keep their starting weight.
        """
        self.w = {k: float(v) for k, v in weights.items()}
        self.anchor = frozenset((anchor,) if isinstance(anchor, str) else anchor)
        self.alpha = alpha
        self.lo, self.hi = clip
        self.fixed = frozenset(fixed)
        self.updates = 0
        self.saturated_frac = 0.0
        self.last_saturated = ()

    @property
    def saturation(self) -> float:
        """
        Average share of the adaptive weights that reached the ceiling.

        A high value means the physics residual has a floor the optimiser
        cannot get below, so every other weight is simply at its maximum and
        the weights no longer carry information.
        """
        return self.saturated_frac / max(self.updates, 1)

    def __getitem__(self, key):
        return self.w.get(key, 1.0)

    def total(self, losses: dict) -> torch.Tensor:
        """
        Weighted sum of the loss terms.

        Args:
            losses (dict): Loss tensors by name.

        Returns:
            torch.Tensor: The total loss.
        """
        out = None
        for k, v in losses.items():
            term = self.w.get(k, 1.0) * v
            out = term if out is None else out + term
        return out

    @torch.no_grad()
    def _norm(self, grads):
        """Euclidean norm of a list of gradients."""
        return torch.sqrt(sum((g * g).sum() for g in grads if g is not None))

    def update(self, losses: dict, params) -> dict:
        """
        Re-estimate the weights from the current gradients.

        This needs one backward pass per loss term, so it runs every
        ``train.balance_every`` steps and not at every step.

        Args:
            losses (dict): Loss tensors by name (with their graphs).
            params (list): Trainable parameters.

        Returns:
            dict: The new weights.
        """
        params = [p for p in params if p.requires_grad]
        anchors = [losses[k] for k in self.anchor
                   if k in losses and losses[k].requires_grad]
        if not anchors:
            return dict(self.w)

        ref = anchors[0]
        for extra in anchors[1:]:
            ref = ref + extra
        g_ref = torch.autograd.grad(ref, params, retain_graph=True,
                                    allow_unused=True)
        n_ref = self._norm(g_ref)
        if not torch.isfinite(n_ref) or n_ref == 0:
            return dict(self.w)

        adaptive, hit = 0, []
        for key, value in losses.items():
            if key in self.anchor or key in self.fixed or not value.requires_grad:
                continue
            g = torch.autograd.grad(value, params, retain_graph=True,
                                    allow_unused=True)
            n = self._norm(g)
            if not torch.isfinite(n) or n == 0:
                continue
            target = float(n_ref / n)
            adaptive += 1
            if target >= self.hi:
                hit.append(key)
            target = min(max(target, self.lo), self.hi)
            self.w[key] = self.alpha * self.w.get(key, 1.0) + (1 - self.alpha) * target
        for key in self.anchor:
            self.w[key] = 1.0

        self.updates += 1
        self.last_saturated = tuple(hit)
        if adaptive:
            self.saturated_frac += len(hit) / adaptive
        return dict(self.w)
