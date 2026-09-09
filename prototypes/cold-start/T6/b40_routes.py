"""B40 (#603): four routes to earned cycle closure, scored against each other.

B34 (#593) settled *what* carves a topology and deliberately did not settle how
the surface gets into a state where the count means anything. The user ruled that
it should not be settled by preference -- *"we can just choose based on what the
results indicate"* -- so this builds the arms and reports the numbers.

**The arms.**

* `split`   -- arm 1. The holonomy term descends on one half of the short local
               cycles; the criterion is read on the half it never touched. Buys
               structure-from-motion's asymmetry inside a joint objective: the
               edge cannot learn to close the loop it will be judged on, because
               it does not know which loop that is.
* `phased`  -- arm 2. Estimate, then prune. **Its terminator is stated**, which
               the ticket demands, and it is not `identification`: B29 read that
               at chance from construction to horizon, so a phase-1 waiting for
               it to separate never ends. Phase 1 ends when **the criterion's own
               input stops moving** -- see :data:`PHASE1_TOL`.
* `control` -- arm 3. B33's joint form as built: the term descends on all the
               local cycles and the criterion is read on the same ones. If the
               side-channel does not materialise, arms 1 and 2 are unwarranted
               machinery and this arm says so.
* `haar`    -- **the scale-matched control for arm 4, and it is not optional.**
               `flat` installs orthonormal frames where the rig's own
               initialisation sits at `sigma_max ~ 1e-7`, so any advantage it
               shows could be isometry rather than flatness. This installs
               `holonomy_read.flat_maps` -- maps of the same block structure,
               exactly isometric, drawn **independently** per endpoint, with no
               shared per-cell frame. Same scale as `flat`, cycle-consistency
               absent. What separates `flat` from `haar` is flatness alone.
* `flat`    -- arm 4, the falsifier. Grimaldi's flat bundle: one frame per node,
               edge map `R_j^T R_i`, cycle-consistent by construction. If
               holonomy cannot fail, the count reads full width everywhere and
               B34's criterion discriminates nothing.

**The scores**, per the ticket, and *not* ADR-0026's conduction ratio -- B38
(#599) read that moving not at all under a 150x training contrast whose per-edge
amplitudes moved 456x, so an arm comparison scored on it reads every arm failing
identically.

1. `identification` on the **wide** subset -- does cycle closure happen.
2. `sigma_max` and `channel_return` -- is it honest. B32's steganography check.
3. the spread of the per-edge warranted count -- does the criterion discriminate.
   A route that closes every cycle equally passes (1) and fails here.
4. probe-lane transmission -- do the floored edges leak.

Every checkpoint stamps its own `motion.read()`, per B38's standing requirement:
nothing about the horizon is inherited.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b40_routes.py run --arms flat
    PYTHONPATH=src python prototypes/cold-start/T6/b40_routes.py run --ticks 500
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b29 = _load("b40r_b29", _HERE / "b29_holonomy.py")
b33 = _load("b40r_b33", _HERE / "b33_coexist.py")
b40e = _load("b40r_enum", _HERE / "b40_enumerate.py")
crit = _load("b40r_crit", _HERE / "b40_criterion.py")

import holonomy_read as hr  # noqa: E402
from patchworks.restriction import pair_index  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
import construction_grading as cg  # noqa: E402

#: Checkpoints. Short by design -- B33 read its whole contrast inside 2,000 ticks
#: and this is a prototype, not a build.
CHECKPOINTS = (0, 50, 150, 500)

#: Phase 1 of the `phased` arm ends when the mean absolute change in the per-edge
#: warranted count, between consecutive checkpoints, falls below this. **This is
#: the terminator the ticket demands an arm state.** It is deliberately not a
#: threshold on `identification`: B29 read that at chance on every cycle with room
#: from construction to horizon, so a phase-1 waiting on it never ends. What can
#: end is the *estimate* settling -- the criterion's own input ceasing to move --
#: which is exactly what the estimate-then-prune schedule in B32's sources means
#: by "estimate".
PHASE1_TOL = 0.05

#: Thresholds the criterion is reported at, so no conclusion rests on one value.
THRESHOLDS = (0.8, 0.9, 0.95)

ARMS = ("control", "split", "phased", "flat", "haar")


# -- the cycle sets each arm reads and descends on ----------------------------


def cycle_sets(dome) -> dict:
    """Enumerated short local cycles, and the train/held-out split arm 1 needs.

    Enumerated per edge rather than taken off B29's basis -- `b40_enumerate.py`
    records why: a basis is a BFS artifact and B34's criterion is over *the* short
    local cycles through `e`, all of them.
    """
    adj, wide_edges = b40e.wide_adjacency(dome)
    per_edge: dict[int, list] = {}
    seen: dict[frozenset, tuple] = {}
    for edge_id in wide_edges:
        cycles = b40e.cycles_through(adj, dome, edge_id)
        local = [c for c in cycles if b33.local_cycles(dome, [c], b33.RADIUS)]
        per_edge[edge_id] = local
        for c in local:
            seen.setdefault(frozenset(c), c)
    allc = sorted(seen.values(), key=lambda c: (len(c), c))
    train = [c for i, c in enumerate(allc) if i % 2 == 0]
    heldout = [c for i, c in enumerate(allc) if i % 2 == 1]
    train_set = {frozenset(c) for c in train}
    held_set = {frozenset(c) for c in heldout}
    return {
        "wide_edges": wide_edges,
        "per_edge": per_edge,
        "all": allc,
        "train": train,
        "heldout": heldout,
        "per_edge_train": {
            e: [c for c in cs if frozenset(c) in train_set] for e, cs in per_edge.items()
        },
        "per_edge_heldout": {
            e: [c for c in cs if frozenset(c) in held_set] for e, cs in per_edge.items()
        },
    }


# -- score 3's input: the criterion read off a surface -------------------------


def read_criterion(dome, maps, per_edge: dict, threshold: float) -> dict:
    """B34's per-edge count, then B34's allocation, then score 3's spread."""
    counts, spectra, no_evidence = {}, {}, []
    for edge_id, cycles in per_edge.items():
        if not cycles:
            no_evidence.append(int(edge_id))
            counts[edge_id] = 0
            continue
        # **Read at `e`, not at the cycle's own base.** A cycle's holonomy is
        # `m x m` for whichever edge the walk is read from, and B34's count is a
        # count of directions *on this edge*. B29's `_narrowest` rotates to the
        # narrowest edge because it is reading the cycle; here the edge is the
        # subject, so every cycle through `e` is rotated to base at `e` and the
        # holonomies are commensurable by construction.
        holos = [hr.holonomy(dome, maps, b29._base_at(tuple(c), edge_id)) for c in cycles]
        n, cos = crit.warranted_count(holos, threshold)
        counts[edge_id] = n
        spectra[edge_id] = cos
    warrant = {c: int(dome._permitted[c]) for c in dome.predicting}
    alloc = crit.allocate(
        dome, counts, budget=int(dome.spec.capacity_budget), endpoint_warrant=warrant
    )
    out = {
        "threshold": threshold,
        "edges": len(counts),
        "edges_with_no_cycle": len(no_evidence),
        "raw_counts": crit.spread({e: v for e, v in counts.items()}),
        "allocated": crit.spread(alloc),
        "alloc": {str(k): int(v) for k, v in alloc.items()},
    }
    # The criterion restricted to the edges that *have* evidence: on the rest it
    # is floored by arithmetic, and mixing the two hides whether it discriminates.
    evid = {e: v for e, v in alloc.items() if per_edge.get(e)}
    out["allocated_with_evidence"] = crit.spread(evid)
    return out


# -- score 4: do the probe lanes leak? ----------------------------------------


def probe_leak(dome, maps, alloc: dict[int, int]) -> dict:
    """Transmission across the edges the criterion floored to `m_e = 1`.

    B34 floors every edge so a pruned edge stays measurable and can grow back.
    The user's caution on agreeing to it: *"I am concerned about transmission
    across functionally closed directions which are kept open to allow later
    access. So we need to check it."* A functionally-pruned edge held open for
    reversibility is still a live channel.

    What is measured is the **operator norm actually carried on the floored
    lane** -- the top singular value of the edge's restriction map restricted to
    the one surviving direction -- against the same quantity on the edges the
    criterion kept. A floored lane that transmits as strongly as a kept one is
    not inert, and the ratio is the reading.
    """
    floored = [e for e, m in alloc.items() if m <= 1]
    kept = [e for e, m in alloc.items() if m > 1]

    def gain(edge_id: int) -> float:
        edge = dome.edges[edge_id]
        vals = []
        for cell in (edge.u, edge.v):
            try:
                idx = pair_index(edge_id, cg.side_of(dome, edge_id, cell))
            except Exception:
                continue
            with torch.no_grad():
                f = maps.maps[idx][: max(1, 1)]  # the one surviving lane
                s = torch.linalg.svdvals(f.double())
            if s.numel():
                vals.append(float(s[0]))
        return float(np.mean(vals)) if vals else float("nan")

    fg = np.array([gain(e) for e in floored], dtype=float)
    kg = np.array([gain(e) for e in kept], dtype=float)
    fg, kg = fg[np.isfinite(fg)], kg[np.isfinite(kg)]

    def q(a):
        if a.size == 0:
            return {"n": 0}
        return {
            "n": int(a.size),
            "median": float(np.median(a)),
            "q1": float(np.percentile(a, 25)),
            "q3": float(np.percentile(a, 75)),
            "max": float(a.max()),
        }

    out = {"floored": q(fg), "kept": q(kg)}
    if fg.size and kg.size:
        out["ratio_median"] = float(np.median(fg) / max(np.median(kg), 1e-12))
        out["floored_above_kept_median"] = float((fg > np.median(kg)).mean())
    return out


# -- arm 4: the flat bundle ----------------------------------------------------


def install_flat_bundle(dome, maps, seed: int) -> dict:
    """Grimaldi's construction on this rig: one frame per cell, edge map off it.

    Each interior cell `c` gets a Haar-random orthogonal frame `R_c` on its stalk;
    each edge endpoint `(e, c)` is set to `S_e R_c`, the edge's first `m_e` rows
    of that frame. That is *the* flat-bundle construction, and installing it here
    rather than asserting its properties is the point.

    **It is exactly cycle-consistent at any widths**, which is measured, not
    argued: `identification` reads `0.0000` and `sigma_max` `1.000` on every wide
    cycle before anything is applied. The reason is that every edge at a cell
    takes *rows of the same frame*, so a hop is the top-left `m_out x m_in` block
    of `R_c R_c^T = I` -- a rectangular identity -- and the cycle telescopes
    exactly. `m_e = n` is **not** required. (An earlier version of this docstring
    argued it was, on the ground that the cell frame cancels out of
    `F_out F_in^T`; the frame does cancel, and what it leaves is the identity
    block rather than an arbitrary selector overlap.)

    **What breaks it is the dimension mask.** `project()` zeroes the columns
    beyond `k_v`, truncating each row of an orthogonal frame to its first `k_v` of
    `n` entries, and truncated rows are not orthonormal -- `identification` 0.0000
    -> 0.6123 at `k_v = 20`, `n = 32`. ADR-0032's band then repairs most of that
    (-> 0.0386) at three orders of amplitude (1.000 -> 7.7e-04). The caller reads
    the post-projection surface, because that is the surface the architecture
    actually has; both are reported in `READOUT-603.md`.
    """
    gen = torch.Generator().manual_seed(seed + 9001)
    n = dome.shape.n
    frames: dict[int, torch.Tensor] = {}
    for cell in dome.predicting:
        a = torch.randn(n, n, generator=gen, dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        frames[cell] = q
    installed = 0
    with torch.no_grad():
        for edge_id, edge in enumerate(dome.edges):
            for cell in (edge.u, edge.v):
                if cell not in frames:
                    continue
                try:
                    idx = pair_index(edge_id, cg.side_of(dome, edge_id, cell))
                except Exception:
                    continue
                block = frames[cell][: edge.m]
                target = maps.maps[idx]
                rows = min(block.shape[0], target.shape[0])
                cols = min(block.shape[1], target.shape[1])
                target[:rows, :cols] = block[:rows, :cols].to(target.dtype)
                installed += 1
    maps.project()
    widths = [e.m for e in dome.edges]
    return {
        "installed_endpoints": installed,
        "n": int(n),
        "m_min": int(min(widths)),
        "m_max": int(max(widths)),
        "full_width_edges": int(sum(1 for m in widths if m >= n)),
        "note": "flat by construction only where m_e == n; measured, not assumed",
    }


# -- the run -------------------------------------------------------------------


def run_arm(arm: str, seed: int, ticks: int, out: Path) -> dict:
    started = time.time()
    env, agent = b33.arms_mod.build_arm(b33.ARM, seed)
    dome = agent.dome
    sets = cycle_sets(dome)
    bases = b29.cycles_of(dome)
    wide = bases["wide"]

    # Which cycles the term descends on, and which the criterion is read on.
    if arm == "split":
        descend, judge_on = sets["train"], sets["per_edge_heldout"]
    else:
        descend, judge_on = sets["all"], sets["per_edge"]

    flat_info = None
    if arm == "flat":
        flat_info = install_flat_bundle(dome, agent.sheaf.maps, seed)
    elif arm == "haar":
        # The scale-matched null: same shapes, same isometry, drawn independently.
        gen = torch.Generator().manual_seed(seed + 9001)
        fm = hr.flat_maps(dome, gen)
        with torch.no_grad():
            agent.sheaf.maps.maps.copy_(fm.maps.to(agent.sheaf.maps.maps.dtype))
        agent.sheaf.maps.project()
        flat_info = {"note": "independent Haar isometries; no shared cell frame"}

    bias = PredictionRule(agent.sheaf)
    transport = TransportRule(agent.sheaf)
    record = {
        "issue": 603,
        "reading": "which training route buys earned cycle closure",
        "arm": arm,
        "base_arm": b33.ARM,
        "seed": seed,
        "ticks": ticks,
        "radius": b33.RADIUS,
        "thresholds": list(THRESHOLDS),
        "phase1_tol": PHASE1_TOL,
        "cycles": {
            "distinct_local": len(sets["all"]),
            "descended_on": len(descend),
            "wide_basis": len(wide),
            "edges_with_no_cycle": sum(1 for v in sets["per_edge"].values() if not v),
        },
        "flat_bundle": flat_info,
        "checkpoints": [],
    }

    motion = b33.Motion(env)
    phase1_over = arm != "phased"
    prev_counts = None

    def checkpoint(target: int, stats: dict) -> None:
        nonlocal phase1_over, prev_counts
        entry = {
            "ticks": target,
            "elapsed_minutes": (time.time() - started) / 60.0,
            "term_stats": stats,
            "motion": motion.read(),
            "read_window": len(motion.buffer),
            "phase1_over": phase1_over,
        }
        maps = agent.sheaf.maps
        # Scores 1 and 2, off B29's instrument unchanged.
        entry["holonomy"] = b29.surface_read(
            dome, maps, wide, f"{arm} s{seed} @{target} [wide]"
        )
        entry["holonomy_local"] = b29.surface_read(
            dome, maps, sets["all"], f"{arm} s{seed} @{target} [local]"
        )
        # Scores 3 and 4, at every threshold.
        entry["criterion"] = {}
        for th in THRESHOLDS:
            c = read_criterion(dome, maps, judge_on, th)
            alloc = {int(k): v for k, v in c.pop("alloc").items()}
            if th == crit.THRESHOLD:
                entry["probe_leak"] = probe_leak(dome, maps, alloc)
                cur = np.array([alloc[e] for e in sorted(alloc)], dtype=float)
                if prev_counts is not None and not phase1_over:
                    drift = float(np.abs(cur - prev_counts).mean())
                    entry["phase1_drift"] = drift
                    if drift < PHASE1_TOL:
                        phase1_over = True
                        entry["phase1_ended_here"] = True
                prev_counts = cur
            entry["criterion"][str(th)] = c
        record["checkpoints"].append(entry)
        out.with_suffix(".inflight.json").write_text(
            json.dumps(record, indent=1), encoding="utf-8"
        )
        sub = entry["holonomy"]["subsets"]["wide"]
        cr = entry["criterion"][str(crit.THRESHOLD)]
        sp = cr["allocated_with_evidence"]
        pl = entry.get("probe_leak", {})
        mo = entry["motion"]
        print(
            f"  [{arm} s{seed}] {target:>5} "
            f"ident {sub['identification']['median']:.4f} "
            f"chan {sub['channel_return']['median']:.4f} "
            f"sig {sub['sigma_max']['median']:.2e} | "
            f"m_e med {sp.get('median', float('nan')):.1f} "
            f"max {sp.get('max', float('nan')):.0f} "
            f"std {sp.get('std', float('nan')):.2f} "
            f"distinct {sp.get('distinct', 0)} | "
            f"leak {pl.get('ratio_median', float('nan')):.3f} | "
            f"world {mo.get('std_max', float('nan')):.2e}",
            flush=True,
        )

    # The training step: B33's `holo` arm, restricted to this arm's cycle set.
    def extra_step() -> dict:
        stats: dict = {}
        if arm in ("flat", "haar"):
            return stats  # these are constructions, not objectives
        if arm == "phased" and not phase1_over:
            return stats  # phase 1: estimate only, no holonomy term
        tensor = agent.sheaf.maps.maps
        param = tensor.detach().clone().requires_grad_(True)
        hl = b33.holonomy_loss(dome, param, descend)
        if float(hl.detach()) == 0.0:
            return stats
        stats["holonomy_loss"] = float(hl.detach())
        (g,) = torch.autograd.grad(hl, param)
        gn = torch.linalg.norm(g)
        if not bool(torch.isfinite(gn)) or float(gn) <= 0.0:
            return stats
        with torch.no_grad():
            tensor.sub_(transport.learning_rate * b33.LAMBDA_HOLO * (g / gn))
        agent.sheaf.maps.project()
        return stats

    checkpoint(0, {})
    seen = 0
    last: dict = {}
    for target in [c for c in CHECKPOINTS if 0 < c <= ticks]:
        span = target - seen
        motion.reset()
        remaining = span
        for _ in b33.t0.run_ticks(agent, span, seed=seed + seen):
            remaining -= 1
            if remaining < min(b33.READ_WINDOW, span):
                motion.observe()
            bias.step()
            if agent.sheaf.ticks > 1:
                transport.step()
                last = extra_step()
        seen = target
        checkpoint(target, last)

    record["minutes"] = (time.time() - started) / 60.0
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    out.with_suffix(".inflight.json").unlink(missing_ok=True)
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run"])
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticks", type=int, default=500)
    args = p.parse_args()
    for arm in args.arms:
        out = _HERE / f"603-routes-{arm}-seed{args.seed}-{args.ticks}.json"
        if out.exists():
            print(f"[B40] {out.name} already at the horizon, skipping", flush=True)
            continue
        print(f"[B40] {arm} seed {args.seed} -> {out.name}", flush=True)
        run_arm(arm, args.seed, args.ticks, out)


if __name__ == "__main__":
    main()
