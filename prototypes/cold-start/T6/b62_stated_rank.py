"""B62 (#635), item 3: does `N(theta)` track the traffic's rank or its alignment?

`#635 <https://github.com/NGL321/patchworks/issues/635>`_, under
`the map <https://github.com/NGL321/patchworks/issues/532>`_.

`B57 (#629) <https://github.com/NGL321/patchworks/issues/629>`_ read
`B52 (#622) <https://github.com/NGL321/patchworks/issues/622>`_'s graded
agreement instrument and got `N(0.25) = 1.0000` on every trained arm, with an
excess of `+1.0000` over a matched-generic null reading `0.0000`. The surprise
ledger (`#520 <https://github.com/NGL321/patchworks/issues/520>`_, row 8) files
two readings it cannot separate:

1. the surviving direction is one the *built* maps genuinely share and generic
   ones do not -- the instrument measures alignment; or
2. `q_i` sits far from 1 wherever the traffic is rank one and near 1 otherwise
   -- the instrument measures traffic rank in another basis, and B52's whole
   construction reduces to `B60 (#633) <https://github.com/NGL321/patchworks/issues/633>`_'s
   object.

**This file is the falsifier and it needs no run.** `q_i` is a function of a
*direction* and the maps alone; the traffic only supplies the directions and
their weights. So synthesise traffic of **stated** rank, in subspaces of stated
alignment, put it through B57's own :func:`b57_agreement.agreement` unmodified,
and read whether `N` follows the rank or the alignment. Reading 2 predicts `N`
is a re-description of the rank and cannot depend on which subspace the traffic
occupies; reading 1 predicts the opposite.

**Four families**, each drawn inside a subspace of `R^columns` chosen off the
built maps:

* ``agreeing`` -- inside `ker(delta_P)`, restricted to what the maps can see
  (`H^0` directions of highest `||G u||`). These are exactly the directions the
  cells already agree about.
* ``generic`` -- an isotropic random subspace. The map's *matched-generic* null
  redraws the maps and holds the traffic; this holds the maps and redraws the
  traffic, which is the other half of the same question.
* ``disagreeing`` -- the top right singular vectors of `delta_P`: the directions
  the cells disagree about most.
* ``invisible`` -- inside `ker(G)`. The instrument marks these unseen and drops
  them; the family is here so the record says what it does rather than assuming.

Plus one **composite**, ``mean_plus_variety``: a single heavy agreeing direction
with a light generic remainder riding on it, which is B57's own account of the
real reading (*"the one thing the room agrees about is the standing mean"*).
It is the row that says whether that account reproduces the real numbers --
uncentered `N = 1`, centred `N = 0` -- from stated ingredients.

**Nothing here is a candidate and nothing here sets a threshold.** Per #635 and
the map's *Plan, don't do*: this builds and reads.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b62_stated_rank.py --ticks 0
    PYTHONPATH=src python prototypes/cold-start/T6/b62_stated_rank.py --ticks 2000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b57 = _load("b57_agreement", _HERE / "b57_agreement.py")

from patchworks.learning import PredictionRule, TransportRule  # noqa: E402

#: The stated ranks. B57's real traffic reads an effective rank of 1.0045
#: uncentered and 1.611 centred at 20k, so the interesting range is small.
RANKS = (1, 2, 4, 8)
#: Rows of synthetic traffic. `T = 1,000` is T0's window, unchanged; a stated
#: rank is exact for any `T >= r`, and the window only sets the sampling noise.
TICKS = b57.WINDOW


# -- the subspaces ------------------------------------------------------------


def subspaces(delta: np.ndarray, gram: np.ndarray, *, rank_needed: int, seed: int) -> dict:
    """One orthonormal basis per family, each `[k, columns]`, `k >= rank_needed`.

    `ker(delta_P)` is 1,800 of 4,800 columns on this surface, so the agreeing
    family is drawn from a large space and the *visible* end of it is selected
    rather than assumed: a direction in `H^0` that no incident map can see has
    `q = nan` and would leave the family reading nothing.
    """
    columns = delta.shape[1]
    rng = np.random.default_rng(seed)

    # `delta_P`'s full right basis, once. `vh[:rank]` spans the row space (what
    # the cells disagree about); `vh[rank:]` spans `H^0` (what they agree about).
    _, s, vh = np.linalg.svd(delta, full_matrices=True)
    tol = max(delta.shape) * float(s[0]) * np.finfo(np.float64).eps
    rank = int((s > tol).sum())
    kernel = vh[rank:]
    rowspace = vh[:rank]

    # Order `H^0` by how much of it the maps can see, so `agreeing` is drawn
    # from directions that are agreed about *and* read.
    seen = gram @ kernel.T                              # [gram_rows, dim H0]
    _, s_seen, vh_seen = np.linalg.svd(seen, full_matrices=False)
    agreeing = (vh_seen @ kernel)                       # visible-first in H^0

    # `ker(G)`: what no incident map reads at all. `G` has more rows than
    # columns, so `full_matrices=False` already returns the whole `[columns,
    # columns]` right basis and the left factor is never formed.
    _, s_g, vh_g = np.linalg.svd(gram, full_matrices=False)
    tol_g = max(gram.shape) * float(s_g[0]) * np.finfo(np.float64).eps
    rank_g = int((s_g > tol_g).sum())
    invisible = vh_g[rank_g:]

    generic = np.linalg.qr(rng.standard_normal((columns, rank_needed)))[0].T

    out = {
        "agreeing": agreeing[:rank_needed],
        "generic": generic[:rank_needed],
        "disagreeing": rowspace[:rank_needed],
        "invisible": invisible[:rank_needed] if invisible.shape[0] else None,
    }
    facts = {
        "columns": int(columns),
        "delta_rank": rank,
        "dim_h0": int(columns - rank),
        "gram_rank": rank_g,
        "dim_ker_gram": int(columns - rank_g),
        "h0_visible_sigma_top": float(s_seen[0]) if s_seen.size else 0.0,
        "h0_visible_sigma_at_8": float(s_seen[7]) if s_seen.size > 7 else 0.0,
    }
    return {"bases": {k: v for k, v in out.items() if v is not None}, "facts": facts}


# -- the synthetic traffic ----------------------------------------------------


def stated_rank_traffic(basis: np.ndarray, rank: int, ticks: int, rng) -> np.ndarray:
    """`X`, `[ticks, columns]`, of exactly `rank` -- equal expected weight.

    Coefficients are i.i.d. standard normal, so the `r` directions carry
    statistically equal weight and the *measured* effective rank lands near the
    stated one. That is the point: a family whose weights decay would confound
    the stated rank with the effective one.
    """
    coeff = rng.standard_normal((ticks, rank))
    return coeff @ basis[:rank]


def mean_plus_variety(
    agreeing: np.ndarray, generic: np.ndarray, rank: int, ticks: int, rng, ratio: float
) -> np.ndarray:
    """A standing agreeing mean with a light generic variety riding on it.

    B57's account of the real reading in one construction: `x_t = mu + eps_t`,
    `mu` in `H^0` and `eps_t` isotropic in a `rank - 1` generic subspace at
    `1/ratio` of the amplitude. Its uncentered moment is dominated by `mu mu^T`;
    its covariance proper is the variety.
    """
    mu = agreeing[0] * ratio
    eps = rng.standard_normal((ticks, max(rank - 1, 1))) @ generic[: max(rank - 1, 1)]
    return mu[None, :] + eps


# -- one row ------------------------------------------------------------------


def read(x: np.ndarray, delta: np.ndarray, gram: np.ndarray, stated: int, family: str) -> dict:
    """B57's instrument, unmodified, on one synthetic `X`. Both forms."""
    reading = b57.both(x, delta, gram)
    row = {"family": family, "stated_rank": stated, "ticks": int(x.shape[0])}
    for form in ("uncentred", "centred"):
        a = reading[form]
        row[form] = {
            "traffic_effective_rank": a["traffic_effective_rank"],
            "directions": a["directions"],
            "unseen_directions": a["unseen_directions"],
            "unseen_weight": a["unseen_weight"],
            "q_weighted_mean": a["q_weighted_mean"],
            "q_quantiles": a["q_quantiles"],
            "headline": a["headline"],
            "profile": a["profile"],
            "top_directions": a["top_directions"],
        }
    return row


def _line(row: dict) -> str:
    u, c = row["uncentred"], row["centred"]

    def ns(entry):
        return " ".join(
            f"N@{t:g} {entry['headline'][f'{t:g}']['N']:7.4f}" for t in b57.THETAS
        )

    return (
        f"  {row['family']:<18} r={row['stated_rank']:<2} "
        f"ER {u['traffic_effective_rank']:7.4f}/{c['traffic_effective_rank']:7.4f} | "
        f"{ns(u)} | cen {ns(c)} | "
        f"q_w {u['q_weighted_mean']:7.4f}/{c['q_weighted_mean']:7.4f} | "
        f"unseen {u['unseen_directions']}"
    )


# -- the stage ----------------------------------------------------------------


def run(condition: str, seed: int, ticks_trained: int, out: Path) -> None:
    started = time.time()
    env, agent, arm, _ = b57.build_arm(condition, seed)
    try:
        layout = b57.Layout(agent.dome)
        chains = None
        record = {
            "issue": 635,
            "item": 3,
            "stage": "stated_rank",
            "condition": condition,
            "seed": seed,
            "trained_ticks": ticks_trained,
            "window": TICKS,
            "surface": b57.t0.surface(),
            "delta_check": b57.check_delta(agent, layout),
            "rows": [],
        }
        print(f"[B62] delta check: {record['delta_check']}", flush=True)

        if ticks_trained:
            bias = PredictionRule(agent.sheaf, operator_rate_ratio=arm["c"])
            transport = TransportRule(agent.sheaf)
            recorder = b57.t2.EdgeRecorder(agent)
            traffic = b57.Traffic(agent, TICKS)
            b57.collect(
                agent, traffic, ticks_trained, seed, recorder, bias, transport, learn=True
            )
            print(f"[B62] trained {ticks_trained} ticks "
                  f"({(time.time() - started) / 60:.1f} min)", flush=True)
            # The real traffic on this very surface, as the row every synthetic
            # row is read against. B49: construction and trained are two
            # surfaces and neither is inherited.
            chains = b57.t2.rim_chains(agent.dome)
            real = b57.joint(agent, layout, chains, traffic.matrix(), "real traffic", seed=seed)
            record["real_traffic"] = real
            print(b57._line("real traffic", real), flush=True)

        maps = agent.sheaf.maps
        delta, gram = layout.delta(maps), layout.gram(maps)

        # B48's joint rule: the two columns that ride with any agreement
        # statistic. They are properties of the *maps*, so they are the same on
        # every synthetic row of one surface -- which is itself worth stating.
        if chains is None:
            chains = b57.t2.rim_chains(agent.dome)
        record["audience_differentiation"] = b57.b50.audience_differentiation(
            agent.dome, maps, chains
        )
        record["exposure"] = b57.exposure(agent.dome, maps, layout)
        print(
            f"[B62] joint columns: aud-diff median "
            f"{record['audience_differentiation']['median']:.4f} | exposure "
            f"{record['exposure']['effective_median']:.2f}/{record['exposure']['n']}",
            flush=True,
        )

        built = subspaces(delta, gram, rank_needed=max(RANKS), seed=seed)
        record["subspace_facts"] = built["facts"]
        print(f"[B62] subspaces: {built['facts']}", flush=True)

        rng = np.random.default_rng(seed + 7)
        for family, basis in built["bases"].items():
            for r in RANKS:
                if basis.shape[0] < r:
                    continue
                x = stated_rank_traffic(basis, r, TICKS, rng)
                row = read(x, delta, gram, r, family)
                record["rows"].append(row)
                print(_line(row), flush=True)
                out.write_text(json.dumps(record, indent=1))

        # The composite, at three amplitude ratios so the row is a curve and
        # not a point. `ratio` is the standing mean's amplitude in units of the
        # variety's; it is a property of the synthesis, not a threshold.
        for ratio in (1.0, 3.0, 10.0):
            for r in (2, 4, 8):
                x = mean_plus_variety(
                    built["bases"]["agreeing"], built["bases"]["generic"], r, TICKS, rng, ratio
                )
                row = read(x, delta, gram, r, f"mean_plus_variety@{ratio:g}")
                record["rows"].append(row)
                print(_line(row), flush=True)
                out.write_text(json.dumps(record, indent=1))

        record["elapsed_minutes"] = (time.time() - started) / 60.0
        out.write_text(json.dumps(record, indent=1))
    finally:
        env.close()
    print(f"[B62] wrote {out.name} ({record['elapsed_minutes']:.1f} min)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", default="baseline", choices=("baseline", "winner", "flat"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticks", type=int, default=0, help="taught ticks before reading")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or _HERE / f"635-stated-{args.condition}-seed{args.seed}-{args.ticks}.json"
    run(args.condition, args.seed, args.ticks, out)


if __name__ == "__main__":
    main()
