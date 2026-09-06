"""T6 (#534): has ADR-0015's structured-`K` fallback trigger fired?

ADR-0015 pre-registers the fallback from a dense `K` to a structured one on one
observable: *"a standing fight between the gradient and the projection"*.
ADR-0008 restates it as *"the projection and the gradient fight every step"*,
and `learning.PredictionRule.step` names the firing form: *"a projection that
binds every step is instead the observable that calls #138's named fallback"*.

**The literal observable no longer exists.** #433 moved the band's enforcement
out of a post-step projection and into a differentiable forward clamp
(`CellOperators.used`), so nothing fires -- `benchmarks/projection_firing.py`
says so in its own header and declines to report a firing rate rather than
report a zero. The trigger therefore has to be read in the terms the live
surface has, and the restatement is named here *before* it is read.

## The named quantity: band pressure, `Pi`

Per cell, at a checkpoint, with `eta_K = c * eta` the operator rate and the band
`[1/rho_K, 1] = [0.5, 1]`:

* **engagement** `sigma_max(K_raw) > 1` -- the clamp is active, i.e. what the
  cell computes with is not what the gradient wrote. The live analogue of *binds
  every step*; read as a fraction of cells and of checkpoints.
* **drift** `sigma_max(K_raw) - 1` -- how far the unconstrained parameter has
  been carried out of band. The integral of the push, if there is one.
* **band pressure** `Pi = -eta_K * <grad_K, u1 v1^T> / (1 - 1/rho_K)` -- the
  first-order change in `sigma_max(K)` that *this step's own gradient* makes,
  in units of the band's width. `u1, v1` are the leading singular vectors of
  `K_raw`, so `<., u1 v1^T>` is exactly `d sigma_max / dK`. `Pi > 0` is the
  gradient pushing the operator *out* of the band, against the clamp.
* **radial share** `<grad_K, u1 v1^T>^2 / ||grad_K||_F^2` -- how much of the
  gradient's energy is the norm-inflating direction at all. Since #433 the
  clamp is differentiable, so the rule's gradient is taken *through* it; if the
  Jacobian annihilates the scale-inflating component, this share is ~0 and the
  fight is pre-empted by construction rather than standing.
* **persistence** the fraction of checkpoints at which `Pi > 0` for a cell --
  the *standing* half. A fight that alternates sign is not a standing one.

**The verdict rule, pre-registered here.** *Fired* requires engagement ~1 **and**
`Pi > 0` persistently **and** a radial share large enough that the push is a
real part of the gradient rather than numerical residue. *Not fired* is
engagement ~0, or `Pi <= 0`. Anything between is *ambiguous* and is reported as
such (#534's own instruction, and #481's rule that only the falsifying end is a
verdict).

**Scope.** This reads `K` -- the object ADR-0015's fallback names. It says
nothing about transport (`F`), which is ADR-0010/ADR-0032's band and R-A's
top candidate's object. Data: T3's committed full-dome 100k checkpoints, both
arms, seeds 42/43/44. Nothing is re-run.

    python prototypes/cold-start/T6/band_pressure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

T3 = Path(__file__).resolve().parent.parent / "T3"
ARMS = ("baseline", "winner")
SEEDS = (42, 43, 44)
TICKS = 100000
STRATA = ("soma", "vision", "core", "apex")


def read_one(arm: str, seed: int) -> dict:
    stem = T3 / f"524-{arm}-seed{seed}-{TICKS}"
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    arrays = np.load(stem.with_suffix(".npz"))
    eta_k = float(meta["operator_rate"])
    lo, hi = (float(x) for x in meta["band"])
    width = hi - lo
    groups = {name: np.asarray(meta["groups"][name], dtype=int) for name in STRATA}

    rows = []
    for cp in meta["checkpoints"]:
        t = int(cp["ticks"])
        key = f"t{t}_"
        if key + "K_raw" not in arrays:
            continue
        K = arrays[key + "K_raw"].astype(np.float64)
        G = arrays[key + "grad_K"].astype(np.float64)
        u, s, vh = np.linalg.svd(K)
        u1, v1 = u[:, :, 0], vh[:, 0, :]
        # <G, u1 v1^T> = u1^T G v1, the derivative of sigma_max along G.
        radial = np.einsum("ci,cij,cj->c", u1, G, v1)
        gfro2 = np.square(G).sum(axis=(1, 2))
        sigma = s[:, 0]
        rows.append(
            {
                "ticks": t,
                "sigma_raw": sigma,
                "sigma_used": np.clip(sigma, lo, hi),
                "engaged_up": sigma > hi,
                "engaged_dn": sigma < lo,
                # descent is K <- K - eta_K * G, so d sigma = -eta_K * radial
                "pi": -eta_k * radial / width,
                "radial_share": np.square(radial) / np.maximum(gfro2, 1e-300),
                "grad_fro": np.sqrt(gfro2),
            }
        )
    return {"arm": arm, "seed": seed, "eta_k": eta_k, "band": (lo, hi),
            "groups": groups, "rows": rows}


def summarise(run: dict) -> list[dict]:
    out = []
    rows = run["rows"]
    pi_stack = np.stack([r["pi"] for r in rows])  # [checkpoints, cells]
    persistence = (pi_stack > 0).mean(axis=0)
    for r in rows:
        for name, idx in run["groups"].items():
            out.append(
                {
                    "arm": run["arm"],
                    "seed": run["seed"],
                    "ticks": r["ticks"],
                    "stratum": name,
                    "n": int(idx.size),
                    "engaged_up": float(r["engaged_up"][idx].mean()),
                    "engaged_dn": float(r["engaged_dn"][idx].mean()),
                    "drift_med": float(np.median(r["sigma_raw"][idx] - run["band"][1])),
                    "drift_max": float((r["sigma_raw"][idx] - run["band"][1]).max()),
                    "pi_med": float(np.median(r["pi"][idx])),
                    "pi_frac_out": float((r["pi"][idx] > 0).mean()),
                    "pi_absmed": float(np.median(np.abs(r["pi"][idx]))),
                    "radial_share_med": float(np.median(r["radial_share"][idx])),
                    "grad_fro_med": float(np.median(r["grad_fro"][idx])),
                    "persist_med": float(np.median(persistence[idx])),
                }
            )
    return out


def main() -> None:
    records = []
    runs = {}
    for arm in ARMS:
        for seed in SEEDS:
            run = read_one(arm, seed)
            runs[(arm, seed)] = run
            records.extend(summarise(run))

    band = runs[("baseline", 42)]["band"]
    eta_k = {a: runs[(a, 42)]["eta_k"] for a in ARMS}
    print(f"# T6 / #534: band pressure on K. Band {band}, eta_K {eta_k}")
    print(f"# Source: T3 committed full-dome {TICKS}-tick checkpoints, 3 seeds, no re-run.")
    print()

    # -- 1. engagement and drift, by arm and stratum, at the reported horizon --
    print("## Clamp engagement and raw drift (mean over seeds; spread = max-min over seeds)")
    print()
    print("| arm | ticks | stratum | n | engaged sigma>1 | drift median | drift max | sigma_used |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for t in (20000, 100000):
            for st in STRATA:
                sel = [r for r in records if r["arm"] == arm and r["ticks"] == t and r["stratum"] == st]
                if not sel:
                    continue
                eng = np.array([r["engaged_up"] for r in sel])
                dm = np.array([r["drift_med"] for r in sel])
                dx = np.array([r["drift_max"] for r in sel])
                print(
                    f"| {arm} | {t} | {st} | {sel[0]['n']} | {eng.mean():.3f} "
                    f"| {dm.mean():+.4f} +/- {np.ptp(dm):.4f} | {dx.max():+.4f} | 1.000 (clamped) |"
                )
    print()

    # -- 2. band pressure, the fight itself --
    print("## Band pressure Pi = d sigma_max per step / band width, and its sign")
    print()
    print("| arm | ticks | stratum | median Pi | frac cells Pi>0 | median |Pi| | radial share | persistence |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for t in (100, 1000, 20000, 100000):
            for st in STRATA:
                sel = [r for r in records if r["arm"] == arm and r["ticks"] == t and r["stratum"] == st]
                if not sel:
                    continue
                pm = np.array([r["pi_med"] for r in sel])
                fo = np.array([r["pi_frac_out"] for r in sel])
                pa = np.array([r["pi_absmed"] for r in sel])
                rs = np.array([r["radial_share_med"] for r in sel])
                pe = np.array([r["persist_med"] for r in sel])
                print(
                    f"| {arm} | {t} | {st} | {pm.mean():+.3e} | {fo.mean():.3f} "
                    f"| {pa.mean():.3e} | {rs.mean():.3e} | {pe.mean():.3f} |"
                )
    print()

    # -- 3. drift trajectory: is the push integrating, or parked? --
    print("## Raw drift sigma_max(K_raw) across the horizon (median per stratum, mean over seeds)")
    print()
    ts = [r["ticks"] for r in runs[("baseline", 42)]["rows"]]
    print("| arm | stratum | " + " | ".join(str(t) for t in ts) + " |")
    print("|---|---|" + "---|" * len(ts))
    for arm in ARMS:
        for st in STRATA:
            cells = []
            for t in ts:
                sel = [r for r in records if r["arm"] == arm and r["ticks"] == t and r["stratum"] == st]
                cells.append(f"{np.mean([r['drift_med'] for r in sel]) + band[1]:.4f}")
            print(f"| {arm} | {st} | " + " | ".join(cells) + " |")
    print()

    # -- 4. the apex-against-core contrast the ticket asks for --
    print("## Apex against core, at 100k (per seed, both arms)")
    print()
    print("| arm | seed | apex Pi med | core Pi med | apex frac>0 | core frac>0 | apex drift | core drift |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for seed in SEEDS:
            a = next(r for r in records if r["arm"] == arm and r["seed"] == seed
                     and r["ticks"] == 100000 and r["stratum"] == "apex")
            c = next(r for r in records if r["arm"] == arm and r["seed"] == seed
                     and r["ticks"] == 100000 and r["stratum"] == "core")
            print(
                f"| {arm} | {seed} | {a['pi_med']:+.3e} | {c['pi_med']:+.3e} "
                f"| {a['pi_frac_out']:.3f} | {c['pi_frac_out']:.3f} "
                f"| {a['drift_med'] + band[1]:.4f} | {c['drift_med'] + band[1]:.4f} |"
            )
    print()

    # -- 4b. is the gradient radial at all? --
    #
    # Wherever the clamp is engaged, `used(K) = K / sigma(K)` is scale-invariant,
    # so `J(cK) = J(K)` and `<grad J, K> = 0` identically. If that holds
    # numerically then the gradient cannot push on the norm at all, and the
    # drift above is not a push: it is the second-order inflation
    # `d ||K||_F^2 = eta_K^2 ||grad||_F^2` that a step orthogonal to `K` makes
    # unavoidably. Both are tested here.
    print("## Is the gradient radial at all? cos(grad_K, K), and the second-order account of the drift")
    print()
    print("| arm | seed | ticks | median cos(grad,K) | max abs | actual/predicted d||K||_F^2 (med) |")
    print("|---|---|---|---|---|---|")
    for arm in ARMS:
        for seed in SEEDS:
            run = runs[(arm, seed)]
            stem = T3 / f"524-{arm}-seed{seed}-{TICKS}"
            arrays = np.load(stem.with_suffix(".npz"))
            eta = run["eta_k"]
            prev = None
            for t in (100, 20000, 100000):
                K = arrays[f"t{t}_K_raw"].astype(np.float64)
                G = arrays[f"t{t}_grad_K"].astype(np.float64)
                cos = np.einsum("cij,cij->c", G, K) / np.maximum(
                    np.linalg.norm(G, axis=(1, 2)) * np.linalg.norm(K, axis=(1, 2)), 1e-300
                )
                ratio = "-"
                if prev is not None:
                    t0, K0, G0 = prev
                    actual = np.square(K).sum((1, 2)) - np.square(K0).sum((1, 2))
                    pred = (t - t0) * eta**2 * np.sqrt(
                        np.square(G0).sum((1, 2)) * np.square(G).sum((1, 2))
                    )
                    ratio = f"{np.median(actual / np.maximum(pred, 1e-300)):.2f}"
                print(
                    f"| {arm} | {seed} | {t} | {np.median(cos):+.2e} "
                    f"| {np.abs(cos).max():.2e} | {ratio} |"
                )
                prev = (t, K, G)
    print()

    # -- 5. whole-dome verdict inputs --
    print("## Verdict inputs (all 150 cells, 100k, mean over seeds)")
    for arm in ARMS:
        eng, fo, rs, pe, pa, dr = [], [], [], [], [], []
        for seed in SEEDS:
            run = runs[(arm, seed)]
            r = next(x for x in run["rows"] if x["ticks"] == 100000)
            pi_stack = np.stack([q["pi"] for q in run["rows"]])
            eng.append(r["engaged_up"].mean())
            fo.append((r["pi"] > 0).mean())
            rs.append(np.median(r["radial_share"]))
            pe.append(np.median((pi_stack > 0).mean(axis=0)))
            pa.append(np.median(np.abs(r["pi"])))
            dr.append(np.median(r["sigma_raw"]))
        print(
            f"  {arm:9s} engaged={np.mean(eng):.4f}  frac Pi>0={np.mean(fo):.4f}  "
            f"median |Pi|={np.mean(pa):.3e}  radial share={np.mean(rs):.3e}  "
            f"persistence={np.mean(pe):.4f}  median sigma_raw={np.mean(dr):.4f}"
        )

    out = Path(__file__).resolve().parent / "534-band-pressure.json"
    out.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"\nrows -> {out}")


if __name__ == "__main__":
    main()
