"""B40 (#603): B34's per-edge width criterion, made into a thing that moves.

`Edge.m` (`src/patchworks/graph.py:129`) is frozen -- *"fixed here and never
changes"*. B34 settled what should replace that constant:

    `m_e` is the count of directions whose principal-angle cosine clears
    threshold on the short local cycles through `e`, floored at 1, capped by
    `min` of the two endpoints' warrants, rationed by `Sigma_e m_e <= B`.

This module is that sentence as code, and nothing else. It does not train, does
not choose a route, and does not decide anything -- the four arms in
`b40_routes.py` all read *this* criterion, so that what separates them is the
surface they are read on and not the reading.

**The principal-angle cosine, concretely.** For a cycle `c` through edge `e` the
holonomy `H_c` is `m_e x m_e` and its polar factor `Q_c = U V^T` is the rotation
transport applies going round. `holonomy_read`'s `identification` column is
`||Q - I||_F / sqrt(2m)` off exactly this factor. A direction `v` comes back to
itself to the extent that `cos ang(v, Q_c v)` is near 1, and for a real orthogonal
`Q_c` those cosines are the real parts of its eigenvalues -- so "directions whose
principal-angle cosine clears threshold" is, per cycle, the eigenvalues of
`sym(Q_c) = (Q_c + Q_c^T)/2` at or above the threshold.

**Across several cycles, this uses a stated surrogate.** B34's count is over *the*
short local cycles through `e`, plural, and a direction that closes on one cycle
and not another is not warranted. The exact object is the largest subspace on
which `min_c cos ang(v, Q_c v)` clears threshold, which is not a spectrum of
anything. This takes the spectrum of the **mean** of `sym(Q_c)` over the cycles
through `e` instead. That is an upper bound on the exact count -- averaging can
only let a direction that fails on one cycle be carried by another -- so where
this reports a narrow edge the exact rule reports one at least as narrow. It is
recorded as a surrogate wherever it is used, and it is the same surrogate in every
arm, so an arm comparison is not sensitive to it.
"""

from __future__ import annotations

import numpy as np
import torch

#: A direction counts as returned when its principal-angle cosine clears this.
#: Chosen, not derived -- and the sweep in `b40_routes.py` reports the criterion
#: at several thresholds precisely so that no conclusion rests on the value.
THRESHOLD = 0.9


def polar_sym(h: torch.Tensor | np.ndarray) -> np.ndarray | None:
    """`(Q + Q^T)/2` for the polar factor `Q = U V^T` of `h`, or None if degenerate."""
    arr = np.asarray(h, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    try:
        u, s, vh = np.linalg.svd(arr)
    except np.linalg.LinAlgError:
        return None
    if s.size == 0 or s[0] <= 0.0:
        return None
    q = u @ vh
    return (q + q.T) / 2.0


def warranted_count(
    holonomies: list, threshold: float = THRESHOLD
) -> tuple[int, list[float]]:
    """B34's count for one edge, from the holonomies of the cycles through it.

    Returns `(count, cosines)` -- the number of directions clearing `threshold`
    and the full spectrum behind it, so a caller can re-threshold without
    recomputing. With no cycles the spectrum is empty and the count is 0; the
    floor is applied by :func:`allocate`, not here, because "no evidence" and
    "evidence of one direction" are different states and the record keeps them
    apart.
    """
    syms = [s for s in (polar_sym(h) for h in holonomies) if s is not None]
    if not syms:
        return 0, []
    mean = np.mean(np.stack(syms, axis=0), axis=0)
    cos = np.linalg.eigvalsh(mean)[::-1]
    return int((cos >= threshold).sum()), [float(x) for x in cos]


def allocate(
    dome,
    counts: dict[int, int],
    budget: int | None = None,
    *,
    endpoint_warrant: dict[int, int] | None = None,
) -> dict[int, int]:
    """B34's allocation: floor 1, cap by endpoint warrant, ration to `budget`.

    * **Floor.** Every edge keeps `m_e >= 1` so a pruned edge stays measurable and
      can grow back. B34 required it; score 4 exists because the user did not
      accept it as free.
    * **Cap.** `min` of the two endpoints' warrants, where a cell's warrant is
      `k_v` -- the width it is entitled to carry. Passed in rather than derived so
      the reserve mask (#597) stays the single source of it.
    * **Ration.** `Sigma_e m_e <= B`. Where the raw counts exceed the budget the
      surplus is taken from the widest edges first, which keeps the *ordering* the
      criterion produced and spends the shortfall where a lost direction is
      proportionally cheapest.
    """
    alloc = {e: max(1, int(c)) for e, c in counts.items()}
    if endpoint_warrant:
        for edge_id in list(alloc):
            edge = dome.edges[edge_id]
            cap = min(
                endpoint_warrant.get(edge.u, alloc[edge_id]),
                endpoint_warrant.get(edge.v, alloc[edge_id]),
            )
            alloc[edge_id] = max(1, min(alloc[edge_id], int(cap)))
    if budget is not None:
        # **Per predicting cell, not globally.** `graph.py:440` is explicit that
        # `Sigma_e m_e <= spec.capacity_budget` is "a budget **per predicting
        # cell**", so a global ration would be a different and much looser bound
        # than the one the rig enforces, and would let the criterion allocate a
        # width the dome would refuse to build.
        at_cell: dict[int, list[int]] = {}
        for edge_id in alloc:
            edge = dome.edges[edge_id]
            for cell in (edge.u, edge.v):
                at_cell.setdefault(cell, []).append(edge_id)
        for cell, edge_ids in at_cell.items():
            while sum(alloc[e] for e in edge_ids) > budget:
                widest = max(edge_ids, key=lambda e: (alloc[e], -e))
                if alloc[widest] <= 1:
                    break  # every lane at this cell is at its floor
                alloc[widest] -= 1
    return alloc


def spread(alloc: dict[int, int]) -> dict:
    """Score 3: does the criterion discriminate, or does it carve nothing?

    A route that closes every cycle equally allocates the same width everywhere
    and has no carving pressure -- it passes score 1 and fails here. The headline
    is the **share of edges above the floor**, with the standard deviation and the
    distinct-value count beside it, because a spread of zero is the flat bundle's
    predicted signature and must be legible at a glance.
    """
    vals = np.asarray(sorted(alloc.values()), dtype=float)
    if vals.size == 0:
        return {"n": 0}
    counts = np.bincount(vals.astype(int))
    p = counts[counts > 0] / vals.size
    return {
        "n": int(vals.size),
        "min": float(vals.min()),
        "median": float(np.median(vals)),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        "distinct": int(len({int(v) for v in vals})),
        "above_floor_share": float((vals > 1).mean()),
        "entropy": float(-(p * np.log(p)).sum()),
        "total": int(vals.sum()),
    }
