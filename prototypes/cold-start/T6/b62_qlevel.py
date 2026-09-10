"""B62 (#635), item 3 proper: `N(theta)` against traffic of stated rank *and* stated level.

`#635 <https://github.com/NGL321/patchworks/issues/635>`_, under
`the map <https://github.com/NGL321/patchworks/issues/532>`_.

This supersedes the ``agreeing`` family in :mod:`b62_stated_rank`, which was
built inside `ker(delta_P)` on the assumption that *agreed about* and *visible*
can be had together. **They cannot on this surface**, and that is the first
finding of this file: `rank(G) = rank(delta_P) = 3000` and `G` restricted to
`ker(delta_P)` has top singular value **7.1e-15**, so

    ker(G) = ker(delta_P), exactly.

Every direction the cells agree about *exactly* is a direction no incident map
can see at all. That is `B48 (#615) <https://github.com/NGL321/patchworks/issues/615>`_'s
*the trained architecture's entire `H^0` is the privacy reserve*, read straight
off the operators with no traffic in it. It also means **exact agreement is
unavailable to any visible direction by construction**, which is B52's own case
for a graded statistic rather than `earned`, arriving from the other side.

**So the levels have to be graded, and they are set by a pencil.** `q` is the
generalized Rayleigh quotient of `(L, D)` with `L = delta^T delta` and
`D = G^T G`, so on the *visible* subspace -- `G`'s row space, where `D` is
positive definite -- the attainable `q` are the eigenvalues of the pencil and
the whole spectrum is a property of the maps alone. Taking `G = U S V^T` and
writing a visible direction as `x = V_r^T S_r^{-1} y`:

    q(x) = ||delta x||^2 / ||G x||^2 = ||A y||^2 / ||y||^2,  A = (delta V_r^T) S_r^{-1}

so `A`'s squared singular values **are** the pencil's spectrum and its right
singular vectors are the directions attaining them, cheapest end first.

**Why that makes a clean falsifier.** `q` is a Rayleigh quotient, so on the span
of the `r` lowest generalized eigenvectors *every* direction has `q <= q_r`, and
on the span of the `r` highest every direction has `q >= q_{N-r+1}`. The bound
holds for whatever basis the instrument's own SVD picks out of the traffic, so a
stated band is a stated band no matter how the synthetic traffic is mixed. That
gives the two-by-two the ledger's row 8 needs:

* **rank 1 at `q ~ 1`** -- reading 2 says `q` sits far from 1 wherever the
  traffic is rank one. A rank-one traffic reading `N(0.25) = 0` kills it.
* **rank 8 at low `q`** -- reading 2 says a high-rank traffic cannot be counted.
  A rank-eight traffic reading `N(0.25) ~ 8` kills it again.

Reading 1 -- the instrument measures alignment -- predicts exactly this pattern:
`N` follows the *band* and its magnitude follows the *rank*, independently.

**Nothing here is a candidate and nothing here sets a threshold.** The `q` bands
are read off the surface, not chosen; the ranks are B57's own measured range.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b62_qlevel.py --ticks 0
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

#: B57's measured range: traffic effective rank 1.0045 uncentered, 1.611 centred.
RANKS = (1, 2, 4, 8)
#: T0's window, unchanged.
TICKS = b57.WINDOW


# -- the pencil ---------------------------------------------------------------


def pencil(delta: np.ndarray, gram: np.ndarray) -> dict:
    """The attainable `q` spectrum and the directions attaining it.

    Returns `q` ascending and `X` with those directions as **rows**, each
    normalised to unit Euclidean length so the synthetic traffic below has a
    sane scale. The `q` a direction carries is scale-free, so normalising costs
    nothing.
    """
    u_g, s_g, vh_g = np.linalg.svd(gram, full_matrices=False)
    tol = max(gram.shape) * float(s_g[0]) * np.finfo(np.float64).eps
    rank = int((s_g > tol).sum())
    v_r, s_r = vh_g[:rank], s_g[:rank]

    a = (delta @ v_r.T) / s_r[None, :]                   # [delta_rows, rank]
    _, s_a, vh_a = np.linalg.svd(a, full_matrices=False)
    q = (s_a[::-1].astype(np.float64)) ** 2              # ascending
    y = vh_a[::-1]                                       # [rank, rank]
    x = (y / s_r[None, :]) @ v_r                         # [rank, columns]
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    # `G` on `ker(delta_P)`, for the containment claim in the header. `kernel`
    # here is `G`'s own kernel; the two coincide iff this is zero to precision.
    kernel = vh_g[rank:]
    on_kernel = float(np.linalg.svd(delta @ kernel.T, compute_uv=False)[0]) if kernel.size else 0.0
    return {
        "q": q,
        "x": x,
        "facts": {
            "columns": int(gram.shape[1]),
            "gram_rank": rank,
            "dim_ker_gram": int(gram.shape[1] - rank),
            "gram_sigma_max": float(s_g[0]),
            "gram_sigma_at_rank": float(s_r[-1]),
            "delta_on_ker_gram_sigma_max": on_kernel,
            "q_min": float(q[0]),
            "q_max": float(q[-1]),
            "q_quantiles": {
                k: float(np.quantile(q, v))
                for k, v in (("p01", 0.01), ("p10", 0.1), ("median", 0.5), ("p90", 0.9), ("p99", 0.99))
            },
            "q_below_0.25": int((q <= 0.25).sum()),
            "q_below_1": int((q <= 1.0).sum()),
        },
    }


def bands(q: np.ndarray, x: np.ndarray, width: int) -> dict:
    """Three bands of `width` directions: the lowest `q`, the nearest 1, the highest."""
    near_one = int(np.argmin(np.abs(q - 1.0)))
    lo = max(0, min(near_one - width // 2, q.size - width))
    return {
        "low_q": (q[:width], x[:width]),
        "mid_q": (q[lo : lo + width], x[lo : lo + width]),
        "high_q": (q[-width:], x[-width:]),
    }


# -- the synthetic traffic ----------------------------------------------------


def traffic_in(basis: np.ndarray, rank: int, ticks: int, rng) -> np.ndarray:
    """`[ticks, columns]` of exactly `rank`, isotropic in the band's own span."""
    return rng.standard_normal((ticks, rank)) @ basis[:rank]


def _line(row: dict) -> str:
    u, c = row["uncentred"], row["centred"]

    def ns(e):
        return " ".join(f"N@{t:g} {e['headline'][f'{t:g}']['N']:7.4f}" for t in b57.THETAS)

    return (
        f"  {row['family']:<22} r={row['stated_rank']:<2} "
        f"q[{row['band_q_min']:.4f},{row['band_q_max']:.4f}] "
        f"ER {u['traffic_effective_rank']:7.4f}/{c['traffic_effective_rank']:7.4f} | "
        f"{ns(u)} | cen {ns(c)} | q_w {u['q_weighted_mean']:7.4f} | "
        f"unseen {u['unseen_directions']}"
    )


def read(x, delta, gram, stated, family, q_band) -> dict:
    reading = b57.both(x, delta, gram)
    row = {
        "family": family,
        "stated_rank": stated,
        "ticks": int(x.shape[0]),
        "band_q_min": float(q_band.min()),
        "band_q_max": float(q_band.max()),
    }
    for form in ("uncentred", "centred"):
        a = reading[form]
        row[form] = {
            "traffic_effective_rank": a["traffic_effective_rank"],
            "directions": a["directions"],
            "unseen_directions": a["unseen_directions"],
            "q_weighted_mean": a["q_weighted_mean"],
            "q_quantiles": a["q_quantiles"],
            "headline": a["headline"],
            "profile": a["profile"],
            "top_directions": a["top_directions"],
        }
    return row


# -- the stage ----------------------------------------------------------------


def run(condition: str, seed: int, ticks_trained: int, out: Path) -> None:
    started = time.time()
    env, agent, arm, _ = b57.build_arm(condition, seed)
    try:
        layout = b57.Layout(agent.dome)
        record = {
            "issue": 635,
            "item": 3,
            "stage": "qlevel",
            "condition": condition,
            "seed": seed,
            "trained_ticks": ticks_trained,
            "window": TICKS,
            "surface": b57.t0.surface(),
            "delta_check": b57.check_delta(agent, layout),
            "rows": [],
        }
        print(f"[B62] delta check: {record['delta_check']}", flush=True)

        traffic = None
        if ticks_trained:
            bias = PredictionRule(agent.sheaf, operator_rate_ratio=arm["c"])
            transport = TransportRule(agent.sheaf)
            recorder = b57.t2.EdgeRecorder(agent)
            traffic = b57.Traffic(agent, TICKS)
            b57.collect(agent, traffic, ticks_trained, seed, recorder, bias, transport, learn=True)
            print(f"[B62] trained {ticks_trained} ticks "
                  f"({(time.time() - started) / 60:.1f} min)", flush=True)

        maps = agent.sheaf.maps
        delta, gram = layout.delta(maps), layout.gram(maps)
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

        if traffic is not None:
            real = b57.joint(agent, layout, chains, traffic.matrix(), "real traffic", seed=seed)
            record["real_traffic"] = real
            print(b57._line("real traffic", real), flush=True)

        p = pencil(delta, gram)
        record["pencil"] = p["facts"]
        print(f"[B62] pencil: {p['facts']}", flush=True)
        out.write_text(json.dumps(record, indent=1))

        rng = np.random.default_rng(seed + 11)
        banded = bands(p["q"], p["x"], max(RANKS))
        for family, (q_band, basis) in banded.items():
            for r in RANKS:
                x = traffic_in(basis, r, TICKS, rng)
                row = read(x, delta, gram, r, family, q_band[:r])
                record["rows"].append(row)
                print(_line(row), flush=True)
                out.write_text(json.dumps(record, indent=1))

        # The composite: a standing low-`q` mean with a light mid-`q` variety on
        # it. B57's account of the real reading, from stated ingredients.
        low_basis, mid_basis = banded["low_q"][1], banded["mid_q"][1]
        for ratio in (1.0, 3.0, 10.0):
            for r in (2, 4, 8):
                eps = rng.standard_normal((TICKS, r - 1)) @ mid_basis[: r - 1]
                x = low_basis[0][None, :] * ratio + eps
                row = read(
                    x, delta, gram, r, f"mean_low+variety_mid@{ratio:g}",
                    np.concatenate([banded["low_q"][0][:1], banded["mid_q"][0][: r - 1]]),
                )
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
    p.add_argument("--ticks", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or _HERE / f"635-qlevel-{args.condition}-seed{args.seed}-{args.ticks}.json"
    run(args.condition, args.seed, args.ticks, out)


if __name__ == "__main__":
    main()
