"""T4 (#537): the generic law composed effective rank obeys, once nothing else matters.

`ablate.py` finds that redrawing every carried subspace as a Haar frame --
keeping only each hop's `(m_in, m_out, k_v)` counts and the chain length --
reproduces the real composed-ER distribution on all three seeds. So composed
effective rank is a function of the dimension counts and the depth, and nothing
else. This maps that function, which is what #532's lever ticket needs in order
to choose between levers rather than argue about them.

Three sweeps, all on Haar frames at the real chain length unless stated:

* **`ratio`** -- ER against `m/k_v`, the one quantity the ablation leaves live,
  swept both ways: `m` up at fixed `k_v` (widening the lane, #539's object) and
  `k_v` down at fixed `m` (narrowing the mask, which is the privacy invariant
  #540 carries). If the two agree at equal ratio, they are one lever with two
  prices, not two levers.
* **`scale_n`** -- ER with `m` and `k_v` scaled **together**, holding the ratio
  at the built 3/18. This is raising `n` at fixed degree, and it is the direct
  test of #533 §4 / #540's point 2, which hold that `m/n` is what governs the
  angles and so that `n` "buys absolute width and nothing". If ER moves along
  this sweep, that premise is false.
* **`depth`** -- ER against hop count at the built `m = 3`, `k_v = 18`. Advisory
  only: #537 does not ask for it, and depth is #540's `degree` in another guise.
* **`bar`** -- the smallest `m` at `k_v = 18` whose **median** chain clears a
  given ER, for the bar values that have been said out loud.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T4/genericity.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: The built surface, read off `angles.py`'s structure histogram on seed 42:
#: interior lanes `m = 3`, the modal relay-cell mask `k_v = 18` (1057 of 1578
#: hops), and 7 hops rim to apex.
BUILT_M, BUILT_KV, BUILT_HOPS = 3, 18, 7


def effective_rank(s: np.ndarray) -> float:
    s2 = s.astype(np.float64) ** 2
    return float(s2.sum() ** 2 / max(float((s2**2).sum()), 1e-300))


def haar(rng, rows: int, cols: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((rows, cols)))
    return q * np.sign(np.diag(r))


def chain_er(rng, m: int, k_v: int, hops: int) -> float:
    """One rim-to-apex chain of Haar hops at `(m, k_v)`, per ADR-0032's co-isometries."""
    composed = None
    for _ in range(hops):
        v_in, v_out = haar(rng, k_v, m), haar(rng, k_v, m)
        hop = haar(rng, m, m) @ (v_out.T @ v_in) @ haar(rng, m, m).T
        composed = hop if composed is None else hop @ composed
    return effective_rank(np.linalg.svd(composed, compute_uv=False))


def sample(rng, m: int, k_v: int, hops: int, draws: int) -> dict:
    if m > k_v:
        return None
    er = np.array([chain_er(rng, m, k_v, hops) for _ in range(draws)])
    return {
        "m": m,
        "k_v": k_v,
        "hops": hops,
        "ratio": m / k_v,
        "er_median": float(np.median(er)),
        "er_mean": float(er.mean()),
        "er_p90": float(np.quantile(er, 0.90)),
        "er_max": float(er.max()),
        "frac_above_1p1": float((er > 1.1).mean()),
        "frac_above_1p5": float((er > 1.5).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draws", type=int, default=256, help="chains per cell, matching the dome's 263")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "537-genericity.json")
    args = p.parse_args()
    rng = np.random.default_rng(537)
    record = {"issue": 537, "built": {"m": BUILT_M, "k_v": BUILT_KV, "hops": BUILT_HOPS}}

    print("[T4] ratio sweep -- m up at k_v = 18, and k_v down at m = 3")
    widen = [sample(rng, m, BUILT_KV, BUILT_HOPS, args.draws) for m in range(1, BUILT_KV + 1)]
    narrow = [
        sample(rng, BUILT_M, k, BUILT_HOPS, args.draws)
        for k in (18, 15, 12, 10, 9, 8, 7, 6, 5, 4, 3)
    ]
    record["widen_m"] = [r for r in widen if r]
    record["narrow_kv"] = [r for r in narrow if r]
    for label, rows in (("widen m", record["widen_m"]), ("narrow k_v", record["narrow_kv"])):
        print(f"  {label}:")
        for r in rows:
            print(
                f"    m={r['m']:>2} k_v={r['k_v']:>2} ratio {r['ratio']:.3f} -> "
                f"ER median {r['er_median']:.4f} mean {r['er_mean']:.4f} "
                f"p90 {r['er_p90']:.4f} | >1.1 {r['frac_above_1p1']:.3f} "
                f"| >1.5 {r['frac_above_1p5']:.3f}"
            )

    print("[T4] n sweep -- m and k_v scaled together at fixed ratio (raising n at fixed degree)")
    record["scale_n"] = [
        sample(rng, BUILT_M * mult, BUILT_KV * mult, BUILT_HOPS, args.draws)
        for mult in (1, 2, 3, 4, 5, 6, 8)
    ]
    for r in record["scale_n"]:
        print(
            f"    n x{r['m'] // BUILT_M}: m={r['m']:>2} k_v={r['k_v']:>3} "
            f"ratio {r['ratio']:.3f} -> ER median {r['er_median']:.4f} "
            f"mean {r['er_mean']:.4f} p90 {r['er_p90']:.4f}"
        )

    print("[T4] depth sweep -- hops at the built (m, k_v) [advisory, #537 does not ask]")
    record["depth"] = [
        sample(rng, BUILT_M, BUILT_KV, h, args.draws) for h in (1, 2, 3, 4, 5, 6, 7, 9, 12)
    ]
    for r in record["depth"]:
        print(f"    hops={r['hops']:>2} -> ER median {r['er_median']:.4f} mean {r['er_mean']:.4f}")

    print("[T4] the bar -- smallest m at k_v = 18, 7 hops, whose median chain clears it")
    record["bar"] = {}
    for bar in (1.05, 1.1, 1.25, 1.5, 2.0):
        hit = next((r["m"] for r in record["widen_m"] if r["er_median"] >= bar), None)
        record["bar"][str(bar)] = hit
        print(f"    ER median >= {bar}: m = {hit}")

    args.out.write_text(json.dumps(record, indent=1))
    print(f"[T4] wrote {args.out.name}")


if __name__ == "__main__":
    main()
