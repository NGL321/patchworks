"""#496's two pre-registered readouts, and the un-yoked trajectory half.

Reads whatever ``496-<condition>-...json`` files are present, so it can be run
against a partial sweep while the rest is still going.

**The two readouts, exactly as pre-registered.**

::

    g       = (rho_soma_exogenous - rho_soma_frozen) / (rho_vision  - rho_soma_frozen)   at 5000 and 100k
    g_apex  = (rho_apex_exogenous - rho_apex_fixed)  / (rho_core    - rho_apex_fixed)    at 20000 and 100k

**The aggregation is stated because it does not reproduce #477.** The
pre-registration says *mean over seeds 42/43/44, published with its seed
spread*, so the primary statistic here is the **mean of the per-seed group
medians**, and every per-seed value is printed beside it. The pooled-across-
seeds median is printed too, because it is the convention that comes closest to
#477's quoted figures -- and *neither* reproduces them. On #132's own committed
JSON, under the grouping this file uses, #477's `0.673 / 0.470` somatomotor
reads `0.667 / 0.338` pooled and `0.636 / 0.445` as a mean of medians; its apex
`0.406` at 100k reads `0.409` pooled, which is the closest any pair gets. The
figures are the right shape and are not the same numbers, so this file states
its own convention rather than inheriting one it cannot reconstruct.

**The groups, on the live `(3, 4)` surface.** Cell classes are read from the
rebuilt dome, not from a level index alone -- the map's *measure the graph, not
the shape imposed on it* rule (#181) means the somatomotor class is named by
its column and its boundary adjacency:

======================  =====  ==============================================
group                   cells  what it is
======================  =====  ==============================================
``soma``                6      column ``somatomotor``, L1, boundary-adjacent
``vision``              64     column ``vision``, L1, boundary-adjacent
``apex``                8      L7, boundary- **and** drive-adjacent
``core``                52     column ``core``, L3-L6, adjacent to neither
======================  =====  ==============================================

``vision`` is L1 only, per #481's formula and its stated reason -- *same level,
same kind of cell, differing only in evidence constancy*. #496's power
discussion says "eighty vision cells", which is vision L1 **and** L2; the
formula's ceiling is the 64.

Usage::

    PYTHONPATH=src python prototypes/exogenous-variation-496/readout.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from patchworks.graph import DEFAULT_SPEC, CellKind, build_graph

HERE = Path(__file__).parent
SEEDS = (42, 43, 44)


def groups() -> dict[str, list[int]]:
    dome = build_graph(DEFAULT_SPEC)
    pred = list(dome.predicting)
    boundary, drive = set(dome.boundary), {
        c for c, x in enumerate(dome.cells) if x.kind == CellKind.DRIVE
    }
    col = [str(dome.cells[c].index.column) for c in pred]
    lvl = [int(dome.cells[c].index.level) for c in pred]
    adj_b, adj_d = [], []
    for c in pred:
        nb = set(dome.neighbours(c))
        adj_b.append(bool(nb & boundary))
        adj_d.append(bool(nb & drive))
    n = len(pred)
    return {
        "soma": [i for i in range(n) if col[i] == "somatomotor" and lvl[i] == 1 and adj_b[i]],
        "vision": [i for i in range(n) if col[i] == "vision" and lvl[i] == 1 and adj_b[i]],
        "apex": [i for i in range(n) if lvl[i] == 7 and adj_d[i]],
        "core": [i for i in range(n) if col[i] == "core" and 3 <= lvl[i] <= 6],
    }


def load(condition: str) -> dict[int, dict]:
    out = {}
    for seed in SEEDS:
        hits = sorted(HERE.glob(f"496-{condition}-real-train-seed{seed}-*.json"))
        if hits:
            out[seed] = json.loads(hits[-1].read_text())
    return out


def rho(record: dict, tick: int, index: list[int]) -> np.ndarray | None:
    """The group's per-cell `rho(K)` at a checkpoint, or None if not reached."""
    for entry in record["checkpoints"]:
        if entry["ticks"] == tick:
            return np.array(entry["per_cell"]["rho_K"])[index]
    return None


def per_seed(data: dict[int, dict], tick: int, index: list[int]) -> dict[int, float]:
    out = {}
    for seed, record in data.items():
        v = rho(record, tick, index)
        if v is not None:
            out[seed] = float(np.median(v))
    return out


def pooled(data: dict[int, dict], tick: int, index: list[int]) -> float | None:
    vals = [rho(r, tick, index) for r in data.values()]
    vals = [v for v in vals if v is not None]
    return float(np.median(np.concatenate(vals))) if vals else None


def show(label: str, values: dict[int, float]) -> str:
    if not values:
        return f"{label}: (not reached)"
    v = np.array(list(values.values()))
    spread = f"{v.min():.4f}-{v.max():.4f}" if len(v) > 1 else "single seed"
    seeds = " ".join(f"s{s}={x:.4f}" for s, x in sorted(values.items()))
    return f"{label}: mean {v.mean():.4f}  [{spread}]  {seeds}"


def gap(exo: dict[int, float], floor: dict[int, float],
        ceiling: dict[int, float]) -> tuple[float, list[float]] | None:
    """`g` as a mean over the seeds each arm has, and per-seed where paired.

    Reported as a magnitude with its spread and no threshold on it, per the
    pre-registration. Only `g ~ 0` is a verdict.
    """
    if not (exo and floor and ceiling):
        return None
    shared = sorted(set(exo) & set(floor) & set(ceiling))
    if shared:
        each = [
            (exo[s] - floor[s]) / (ceiling[s] - floor[s])
            for s in shared
            if abs(ceiling[s] - floor[s]) > 1e-9
        ]
    else:
        each = []
    me, mf, mc = (
        float(np.mean(list(exo.values()))),
        float(np.mean(list(floor.values()))),
        float(np.mean(list(ceiling.values()))),
    )
    if abs(mc - mf) < 1e-9:
        return None
    return (me - mf) / (mc - mf), each


def main() -> None:
    G = groups()
    base, arm, drive = load("baseline"), load("arm"), load("drive")
    print("groups (live surface):",
          {k: len(v) for k, v in G.items()})
    for name, data in (("baseline", base), ("arm", arm), ("drive", drive)):
        reached = {s: r["checkpoints"][-1]["ticks"] for s, r in data.items()}
        print(f"  {name:9s} seeds {sorted(data)} reached {reached}")
    if base:
        s = next(iter(base.values()))["surface"]
        print(f"  surface: {s['describe']}  (interior_m={s['interior_m']}, "
              f"boundary_m={s['boundary_m']})")

    print("\n=== group rho(K), baseline: the first reading on the (3,4) surface ===")
    for tick in (5_000, 20_000, 100_000):
        print(f"-- tick {tick}")
        for g in ("soma", "vision", "apex", "core"):
            v = per_seed(base, tick, G[g])
            p = pooled(base, tick, G[g])
            extra = f"  pooled {p:.4f}" if p is not None else ""
            print("   " + show(f"{g:7s}", v) + extra)

    print("\n=== readout 1: the arm.  g = (soma_exo - soma_frozen)/(vision - soma_frozen) ===")
    for tick in (5_000, 100_000):
        r = gap(per_seed(arm, tick, G["soma"]),
                per_seed(base, tick, G["soma"]),
                per_seed(base, tick, G["vision"]))
        if r is None:
            print(f"   tick {tick}: (not reached)")
            continue
        g, each = r
        sp = f"per-seed {['%.3f' % x for x in each]}" if each else "per-seed n/a"
        print(f"   tick {tick:>6}: g = {g:+.3f}   {sp}")

    print("\n=== readout 2: the drive.  g_apex = (apex_exo - apex_fixed)/(core - apex_fixed) ===")
    for tick in (20_000, 100_000):
        r = gap(per_seed(drive, tick, G["apex"]),
                per_seed(base, tick, G["apex"]),
                per_seed(base, tick, G["core"]))
        if r is None:
            print(f"   tick {tick}: (not reached)")
            continue
        g, each = r
        sp = f"per-seed {['%.3f' % x for x in each]}" if each else "per-seed n/a"
        print(f"   tick {tick:>6}: g_apex = {g:+.3f}   {sp}")

    print("\n=== the un-yoked half: travel against the somatomotor front ===")
    print("   Does arm displacement decay on the same time constant as somatomotor")
    print("   rho, and does either cross first?")
    for name, data in (("baseline", base), ("arm", arm)):
        if not data:
            continue
        print(f"-- {name}")
        for seed, record in sorted(data.items()):
            rows = []
            for entry in record["checkpoints"]:
                v = np.array(entry["per_cell"]["rho_K"])[G["soma"]]
                rows.append((entry["ticks"], entry["travel_per_tick"],
                             float(np.median(v))))
            print(f"   seed {seed}: " + "  ".join(
                f"{t}:tr={tr:.2e},rho={rh:.3f}" for t, tr, rh in rows))


if __name__ == "__main__":
    main()
