"""T6 ([B29](#585)): the ladder, checked rather than inherited -- and the tension, proved.

[#585](#585) part 2 states a ladder and says in the same breath that it is **an agent
inference, not a citation**: *trivial* holonomy gives full capacity, *abelian* holonomy
gives a simultaneous eigenbasis so shared eigendirections survive every cycle and
capacity is partial, *non-abelian* holonomy generically leaves one common invariant
direction -- and if that is right, "non-abelian holonomy" and "composed rank 1" are the
same fact. Part 5 states the tension: strict path-independence looks incompatible with
lanes that select, and marks it as the first thing to try to break.

Neither is measured on the trained surface, because neither is a question about the
trained surface. They are questions about the algebra, and this module answers them
with synthetic operators of the same shapes: no sandbox, no training, seconds to run.

## What is checked

**A. Trivial holonomy requires lanes to stop being lanes.** :func:`check_flat_iff_equal`.
Under [ADR-0032](../../../docs/adr/0032-the-maps-learn-isometric-transport-and-a-spectral-floor-expresses-it.md)'s
spectral floor a restriction map has orthonormal rows, so each hop `C = F_out F_in^T` is
a matrix of principal-angle cosines between two lane subspaces of the cell's permitted
block, and `||C||_2 <= 1`. A cycle's holonomy is a product of these. Then::

    H = I  <=>  ||C_i y|| = ||y|| at every hop
           <=>  rowspace(F_in) is carried isometrically into rowspace(F_out) at every cell
           <=>  (closing the cycle, equal widths) every lane subspace around the cycle
                is the *same* subspace of the cell's block.

So the sheaf is flat exactly where **every edge incident to a cell exposes the same
block** -- which is the degeneracy [B12](#556) named in `_assemble` and the ceiling
[B13](#560) found the reserve floor climbing to. The check sweeps the angle between two
lanes from 0 and watches departure leave 0 with it, and confirms the equality case.

**B. Flatness and composed rank are different axes.** :func:`check_rank_vs_flatness`.
The ladder's punchline is that non-abelian holonomy and composed rank 1 are one fact.
They are not: holonomy is a property of a **cycle**, composed rank of an open **chain**,
and [B1](#537) already has composed rank as a function of `(m, k_v, hops)` alone. The
check builds the one construction that settles it -- a sheaf that is **exactly flat by
construction** (every lane the same block, so every holonomy is `I`) -- and reads its
composed rank over seven hops. It is `m`, not 1. And it builds the converse: lanes at a
generic angle, whose holonomy is far from `I` and whose composed rank is still driven by
the widths. Flatness does not buy rank and rank does not imply flatness.

**C. The middle and bottom rungs are wrong, and the top one is right.**
:func:`check_ladder`. For a sheaf of *invertible* maps, `H^0` is the subspace fixed by
every holonomy, so trivial holonomy gives `dim H^0 = d` -- the top rung, which is
[B26](#575)'s citation and survives. The other two do not:

* **abelian does not give partial capacity.** Commuting orthogonal matrices are
  simultaneously block-diagonalisable over the reals into 1x1 blocks (`+-1`) and 2x2
  rotations. A *fixed vector* needs eigenvalue exactly `+1`, and a generic abelian
  subgroup of `SO(d)` -- a maximal torus -- has rotation angles that miss it. The
  common fixed subspace is generically **0** in even `d` and **1** in odd `d`, which is
  what non-abelian gives too. The inference slid from *simultaneous eigenbasis over C*
  to *surviving directions over R*; those are not the same statement.
* **non-abelian does not generically leave one common invariant direction.** For a
  generic subgroup of `O(d)` the fixed subspace is **0**-dimensional. One direction is
  what a *single* rotation of odd order leaves, not what a group shares.

The check draws both cases and counts the fixed subspace directly, so the refutation is
a number rather than an argument.

Usage::

    python prototypes/cold-start/T6/b29_theory.py

This module **asserts nothing about the live surface**. It is about the algebra the
ticket's inferences are made of, and it names which of them survive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

#: A vector counts as fixed by a holonomy when the singular value of `H - I` it sits on
#: is below this. Stated rather than tuned; the sweeps below are nowhere near it.
FIXED_TOL = 1e-8


def _haar(n: int, k: int, rng) -> np.ndarray:
    """`k` orthonormal rows in `R^n`: one lane subspace, uniformly drawn."""
    q, _ = np.linalg.qr(rng.standard_normal((n, k)))
    return q.T


def _identification(h: np.ndarray) -> float:
    """`||UV^T - I||_F / sqrt(2m)`, `holonomy_read.departures`' column verbatim."""
    u, _, vh = np.linalg.svd(h)
    m = h.shape[0]
    return float(np.linalg.norm(u @ vh - np.eye(m), ord="fro") / np.sqrt(2 * m))


def _effective_rank(h: np.ndarray) -> float:
    """`(sum s^2)^2 / sum s^4`, `composed_reads`' participation ratio unchanged."""
    s = np.linalg.svd(h, compute_uv=False)
    return float((s**2).sum() ** 2 / max((s**4).sum(), 1e-300))


# -- A: trivial holonomy requires lanes to stop being lanes --------------------


def _rotation(m: int, phi: float, rng) -> np.ndarray:
    """A rotation of a lane's internal frame by angle `phi` in a random plane."""
    q, _ = np.linalg.qr(rng.standard_normal((m, m)))
    block = np.eye(m)
    block[:2, :2] = [[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]
    return q @ block @ q.T


def check_flat_iff_equal(n: int = 48, m: int = 12, hops: int = 6, seed: int = 0) -> dict:
    """Departure from `I` has **two** independent parts, and both must vanish.

    Walk a cycle whose cells all expose the same block, and perturb it two ways:

    * `theta` -- the **subspace** angle: the outgoing lane is the block tilted by
      `theta` out of it. Because the tilt is orthogonal to the block, the hop is
      exactly `cos(theta) I`, so this axis moves `sigma_max` and leaves the polar
      factor alone. It is the axis that *contracts*.
    * `phi` -- the **frame** angle: the outgoing map picks a different orthonormal
      basis of the *same* subspace, rotated by `phi`. The hop is orthogonal, so
      `sigma_max` and `flatness` are both 1 and only `identification` moves. It is the
      axis that *relabels*.

    So `H = I` needs the lanes to be the same subspace **and** the frames to agree
    around the loop -- ADR-0032's two halves, *metric agreement* and *identification
    agreement*, appearing here as two orthogonal directions of failure rather than as
    one number. `sigma_max` alone cannot see a relabelling and `identification` alone
    cannot see a contraction, which is why the trained surface can read
    `identification` at chance *and* `sigma_max` at `1e-5` without the two being the
    same finding.

    The equality case `(theta, phi) = (0, 0)` is asserted to be exactly the identity.
    """
    rng = np.random.default_rng(seed)
    base = _haar(n, m, rng)  # the common block, the same at every cell

    def walk(theta: float, phi: float, draw) -> np.ndarray:
        composed = np.eye(m)
        for _ in range(hops):
            perp = _haar(n, m, draw)
            perp = perp - (perp @ base.T) @ base  # the part outside the block
            q, _ = np.linalg.qr(perp.T)
            perp = q.T[:m]
            f_in = base
            f_out = _rotation(m, phi, draw) @ (np.cos(theta) * base + np.sin(theta) * perp)
            composed = (f_out @ f_in.T) @ composed
        return composed

    grid = []
    for theta in (0.0, 1e-4, 0.05, 0.2, 0.4, 0.8):
        for phi in (0.0, 1e-4, 0.05, 0.2, 0.8, np.pi / 2):
            h = walk(theta, phi, np.random.default_rng(seed + 1))
            s = np.linalg.svd(h, compute_uv=False)
            grid.append(
                {
                    "theta": float(theta),
                    "phi": float(phi),
                    "identification": _identification(h),
                    "sigma_max": float(s[0]),
                    "flatness": float(s[-1] / max(s[0], 1e-300)),
                }
            )

    exact = grid[0]
    theta_only = [g for g in grid if g["phi"] == 0.0]
    phi_only = [g for g in grid if g["theta"] == 0.0]
    return {
        "n": n,
        "m": m,
        "hops": hops,
        "equality_case_is_identity": bool(
            exact["identification"] < FIXED_TOL and abs(exact["sigma_max"] - 1.0) < 1e-9
        ),
        "theta_moves_identification": bool(
            max(g["identification"] for g in theta_only) < 1e-9
        ),
        "phi_moves_sigma_max": bool(
            max(abs(g["sigma_max"] - 1.0) for g in phi_only) < 1e-9
        ),
        "grid": grid,
        "claim": (
            "H = I iff every lane around the cycle is the same subspace (theta = 0) "
            "AND the frames agree (phi = 0). The two failures are orthogonal: theta "
            "moves sigma_max only, phi moves identification only."
        ),
    }


# -- B: flatness and composed rank are different axes -------------------------


def check_rank_vs_flatness(n: int = 48, m: int = 12, hops: int = 7, seed: int = 0) -> dict:
    """A sheaf that is exactly flat by construction, and what its composed rank is.

    If the ladder's punchline held -- non-abelian holonomy *is* composed rank 1 -- then
    a flat sheaf would have to have full composed rank and a rank-1 chain would have to
    have non-trivial holonomy. The first half is true and trivially so; the second is
    false, and both are read here at the same widths.
    """
    rng = np.random.default_rng(seed)

    # (i) exactly flat: every edge exposes the same block. Every hop is `I_m`.
    base = _haar(n, m, rng)
    flat = np.eye(m)
    for _ in range(hops):
        flat = (base @ base.T) @ flat

    # (ii) generic lanes: a fresh block at every edge, which is `arms.generic_at`'s draw.
    generic = None
    for _ in range(hops):
        f_in = _haar(n, m, rng)
        f_out = _haar(n, m, rng)
        hop = f_out @ f_in.T
        generic = hop if generic is None else hop @ generic

    return {
        "n": n,
        "m": m,
        "hops": hops,
        "flat_by_construction": {
            "composed_er": _effective_rank(flat),
            "identification": _identification(flat),
            "sigma_max": float(np.linalg.svd(flat, compute_uv=False)[0]),
        },
        "generic_lanes": {
            "composed_er": _effective_rank(generic),
            "identification": _identification(generic),
            "sigma_max": float(np.linalg.svd(generic, compute_uv=False)[0]),
        },
        "claim": (
            "A flat sheaf has composed rank m, not 1 -- but it is flat only because "
            "every lane is the same lane, which is the degeneracy B12 named. Holonomy "
            "is a cycle property and composed rank a chain property; they are not one "
            "fact."
        ),
    }


# -- C: the ladder's three rungs ----------------------------------------------


def _fixed_dim(group: list[np.ndarray], tol: float = FIXED_TOL) -> int:
    """`dim` of the subspace every element of `group` fixes: the sheaf's `H^0`."""
    d = group[0].shape[0]
    stacked = np.vstack([g - np.eye(d) for g in group])
    s = np.linalg.svd(stacked, compute_uv=False)
    return int((s < tol * max(s[0], 1.0)).sum() + (d - len(s)))


def _torus(d: int, count: int, rng) -> list[np.ndarray]:
    """A generic abelian subgroup of `SO(d)`: one common frame, independent angles."""
    frame, _ = np.linalg.qr(rng.standard_normal((d, d)))
    out = []
    for _ in range(count):
        blocks = []
        angles = rng.uniform(0, 2 * np.pi, d // 2)
        for a in angles:
            blocks.append(np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]))
        m = np.zeros((d, d))
        for i, b in enumerate(blocks):
            m[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = b
        if d % 2:
            m[-1, -1] = 1.0
        out.append(frame @ m @ frame.T)
    return out


def _generic_orthogonal(d: int, count: int, rng) -> list[np.ndarray]:
    """A generic -- and therefore non-abelian -- subgroup of `SO(d)`."""
    out = []
    for _ in range(count):
        q, r = np.linalg.qr(rng.standard_normal((d, d)))
        q = q * np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        out.append(q)
    return out


def check_ladder(dims=(6, 7, 12, 13), generators: int = 3, trials: int = 8, seed: int = 0) -> dict:
    """Count the common fixed subspace for trivial, abelian and non-abelian holonomy."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in dims:
        trivial = _fixed_dim([np.eye(d) for _ in range(generators)])
        ab, nonab, commutes = [], [], []
        for _ in range(trials):
            torus = _torus(d, generators, rng)
            gen = _generic_orthogonal(d, generators, rng)
            ab.append(_fixed_dim(torus))
            nonab.append(_fixed_dim(gen))
            commutes.append(
                float(
                    np.linalg.norm(torus[0] @ torus[1] - torus[1] @ torus[0])
                    / np.sqrt(2 * d)
                )
            )
        rows.append(
            {
                "d": d,
                "trivial_fixed_dim": trivial,
                "abelian_fixed_dim_median": float(np.median(ab)),
                "abelian_fixed_dim_max": int(max(ab)),
                "abelian_commutator_max": float(max(commutes)),
                "non_abelian_fixed_dim_median": float(np.median(nonab)),
                "non_abelian_fixed_dim_max": int(max(nonab)),
            }
        )
    return {
        "generators": generators,
        "trials": trials,
        "rows": rows,
        "verdict": {
            "rung_trivial": "holds -- dim H^0 = d, and this is B26's citation",
            "rung_abelian": (
                "fails as stated -- a generic maximal torus fixes 0 directions in even "
                "d and 1 in odd d, not a partial subspace. Simultaneous eigenbasis over "
                "C is not surviving directions over R."
            ),
            "rung_non_abelian": (
                "fails as stated -- a generic subgroup fixes 0 directions, not 1. "
                "'One invariant direction' is a property of one odd-dimensional "
                "rotation, not of a group."
            ),
            "punchline": (
                "'non-abelian holonomy' and 'composed rank 1' are therefore NOT the "
                "same fact; see check_rank_vs_flatness."
            ),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=_HERE / "585-theory.json")
    args = p.parse_args()

    record = {
        "issue": 585,
        "reading": "the ladder and the tension, checked on the algebra not the surface",
        "flat_iff_equal": check_flat_iff_equal(),
        "rank_vs_flatness": check_rank_vs_flatness(),
        "ladder": check_ladder(),
    }
    args.out.write_text(json.dumps(record, indent=1))

    a = record["flat_iff_equal"]
    print(
        f"A. equality case is the identity: {a['equality_case_is_identity']}"
        f" | theta leaves identification alone: {a['theta_moves_identification']}"
        f" | phi leaves sigma_max alone: {a['phi_moves_sigma_max']}"
    )
    for row in a["grid"]:
        print(
            f"   theta {row['theta']:.4f} phi {row['phi']:.4f} -> ident"
            f" {row['identification']:.6f}  sigma_max {row['sigma_max']:.6f}"
            f"  flatness {row['flatness']:.6f}"
        )
    b = record["rank_vs_flatness"]
    print(
        f"B. flat-by-construction: ER {b['flat_by_construction']['composed_er']:.3f}"
        f" ident {b['flat_by_construction']['identification']:.2e}"
        f" | generic lanes: ER {b['generic_lanes']['composed_er']:.3f}"
        f" ident {b['generic_lanes']['identification']:.3f}"
    )
    print("C. ladder:")
    for row in record["ladder"]["rows"]:
        print(
            f"   d={row['d']:>3}  trivial {row['trivial_fixed_dim']:>3}"
            f"  abelian {row['abelian_fixed_dim_median']:.1f} (max {row['abelian_fixed_dim_max']})"
            f"  non-abelian {row['non_abelian_fixed_dim_median']:.1f} (max {row['non_abelian_fixed_dim_max']})"
        )
    print(f"wrote {args.out.name}")


if __name__ == "__main__":
    main()
