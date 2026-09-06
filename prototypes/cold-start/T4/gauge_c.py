"""B9 (#551): ADR-0010's `c` at 1 **in circuit** -- projection *and* gain -- to horizon.

`#551 <https://github.com/NGL321/patchworks/issues/551>`_, under
`the map <https://github.com/NGL321/patchworks/issues/532>`_.

**What this is not.** `angles.py::sweep_c` loads a surface trained at `c = 2`,
overwrites `overlap_target`, calls `project()` once and re-reads the composed
spectrum. That is a **post-hoc re-projection**: it measures what one tightening
pass does to an already-trained surface, and it never touches the reconciliation
gain. `#547 <https://github.com/NGL321/patchworks/issues/547>`_ read `c = 1` at
1.0038 (baseline) and 1.0079 (winner) at 100k off exactly that instrument --
~1474x and ~530x the excess over one that `c >= 2` shows, and the only setting
on this map whose advantage *grows* with training. This ticket asks whether that
survives when `c` is actually in circuit.

**`c` has three readers, not one** (`restriction.gain_denominators`' docstring
says so itself):

* :meth:`patchworks.restriction.RestrictionMaps._push_apart` -- the projection
  cap, via the `overlap_target` buffer, the only path `sweep_c` exercises;
* :func:`patchworks.tick.reconciliation_gain` (line 375);
* :func:`patchworks.bias_selection.fold_margin_check`'s nomination (line 1160).

All three reach `c` through one call, `gain_denominators(dome, rho=...)`, taking
`c` from its **keyword default**. So the honest in-circuit arm is one patch on
that default, applied **before the agent is built** so the `overlap_target`
buffer is registered under it too. Nothing else in the library is touched: a
`GAUGE_C` edit would say the same thing and would also have to be reverted.

**The gain doubles at exactly 150 cells.** `overlap_counts(DEFAULT_SPEC)` puts
414 cells at `c_v` of 1 (262), 2 (150), 3 (the actuator) and 8 (the drive).
Dropping `c` to 1 moves the 150 and nothing else -- verified in
:func:`gain_provenance`, which refuses to run an arm whose patch did not land.
**Both of `overlap_counts`' clamps are kept**: the pigeonhole floor and #228's
`c_v = deg(v)` on a wholly-pinned incidence. They live inside `overlap_counts`,
below the patch, so they are structurally out of reach here -- which is why the
actuator holds at 3 and the drive at 8. Relaxing past them is a different ADR's
question and an arm that did it would be measuring a different constant.

**Ungated, to horizon, both arms.** Ruled by the user on #547 Q3. A destabilised
surface is itself the answer to whether `c = 1` is admissible; a gate firing
early would leave composed rank exactly as unmeasured as it is now. So the loop
is T4's `trained.py` verbatim -- `t0.teaching_read`, T3's two conditions, the
frozen world, T3's checkpoint ladder to 100k, seed 42 -- with no instability
check anywhere in it.

**What else moved is read, not assumed.** The doubled gain reaches the tick, so
the arm reads more than composed rank: the gain vector itself (provenance), the
cap's bite, the per-stratum disagreement energy and map effective rank
(`t2.edge_reads`), and a stability read -- state and error magnitudes, and how
much of the surface has left the reals. That last is the one that answers
admissibility.

Comparators, same rig and the same surface, from #547's reading:
`537-baseline-seed42-100000.json` and `537-winner-seed42-100000.json`.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T4/gauge_c.py --condition baseline winner
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

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_T0, _T1, _T2, _T3 = (_HERE.parent / n for n in ("T0", "T1", "T2", "T3"))
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _T0):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t0 = _load("t0_run", _T0 / "run.py")
t1 = _load("t1_run", _T1 / "run.py")
t2 = _load("t2_run", _T2 / "run.py")
t3 = _load("t3_run", _T3 / "run.py")
angles = _load("t4_angles", _HERE / "angles.py")
trained = _load("t4_trained", _HERE / "trained.py")

from patchworks import restriction as R  # noqa: E402
from patchworks.graph import DEFAULT_SPEC, build_graph  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from patchworks.tick import reconciliation_gain  # noqa: E402
from untrained_fixed_point import build  # noqa: E402


def put_c_in_circuit(c: int) -> None:
    """Set ADR-0010's `c` for **all three** readers, before anything is built.

    `gain_denominators` is the single definition the projection, the tick's gain
    and the fold-margin nomination all call, and each calls it without passing
    `c` -- so its keyword default is the one place that reaches every reader.
    `overlap_counts` is patched alongside it for any direct caller (`sweep_c` is
    one, and it passes `c` explicitly, so it is unaffected).

    Patching the default rather than rebinding `restriction.GAUGE_C` is not a
    style choice: the defaults were bound at definition time, so assigning the
    module attribute would move the constant and leave every reader on 2.
    """
    R.gain_denominators.__kwdefaults__["c"] = int(c)
    R.overlap_counts.__kwdefaults__["c"] = int(c)


def gain_provenance(dome, c: int) -> dict:
    """Proof the patch landed, and the size of what it moved.

    Reads the live gain against `c = 2`'s, cell by cell. An arm whose gain did
    not actually change is the failure mode this ticket exists to avoid, so it
    raises rather than recording a null result as a finding.
    """
    put_c_in_circuit(c)  # idempotent; also makes this callable before a build
    counts = R.overlap_counts(dome).numpy()
    base = reconciliation_gain(dome, gamma=1.0).numpy()
    R.gain_denominators.__kwdefaults__["c"] = 2
    ref = reconciliation_gain(dome, gamma=1.0).numpy()
    ref_counts = R.overlap_counts(dome, c=2).numpy()
    put_c_in_circuit(c)

    ratio = base / ref
    moved = int((~np.isclose(ratio, 1.0)).sum())
    if c != 2 and moved == 0:
        raise AssertionError(
            f"c = {c} left the gain identical at all {len(ratio)} cells -- the patch did not land"
        )
    return {
        "gauge_c": int(c),
        "cells": int(len(counts)),
        "overlap_counts_histogram": {
            str(int(k)): int(v) for k, v in zip(*np.unique(counts, return_counts=True))
        },
        "overlap_counts_histogram_at_c2": {
            str(int(k)): int(v) for k, v in zip(*np.unique(ref_counts, return_counts=True))
        },
        "cells_gain_moved": moved,
        "gain_ratio_vs_c2": {
            str(round(float(k), 6)): int(v)
            for k, v in zip(*np.unique(np.round(ratio, 6), return_counts=True))
        },
        "clamps_held": {
            "actuator_or_pinned_at_deg": int((counts > max(c, 1)).sum()),
            "max_count": float(counts.max()),
        },
    }


@torch.no_grad()
def stability_read(agent, recorder) -> dict:
    """Is the surface still on the reals, and how big has it got?

    The admissibility half of the ticket. `c = 1` doubles the reconciliation gain
    at 150 cells for the whole run, and the arm is deliberately ungated, so a
    divergence is a result to be recorded rather than an error to be caught.
    """
    h, e, _v, ticks = recorder.window()
    h, e = h.double(), e.double()
    hn = h.norm(dim=-1).mean(0).numpy()
    en = e.norm(dim=-1).mean(0).numpy()
    maps = agent.sheaf.maps.maps.detach().double()
    norms = maps.norm(dim=(-2, -1)).numpy()
    return {
        "window_ticks": int(ticks),
        "h_norm": {"median": float(np.median(hn)), "max": float(np.nanmax(hn))},
        "e_norm": {"median": float(np.median(en)), "max": float(np.nanmax(en))},
        "h_nonfinite_cells": int((~np.isfinite(hn)).sum()),
        "e_nonfinite_cells": int((~np.isfinite(en)).sum()),
        "map_frobenius": {
            "median": float(np.median(norms)),
            "min": float(np.nanmin(norms)),
            "max": float(np.nanmax(norms)),
        },
        "maps_nonfinite": int((~np.isfinite(norms)).sum()),
    }


def run_seed(condition: str, seed: int, ticks: int, out: Path, c: int, c_values) -> dict:
    """T4 `trained.py`'s loop, with `c` in circuit and the gain-side reads added."""
    started = time.time()
    inflight = out.with_suffix(".inflight.json")
    arm = t3.CONDITIONS[condition]
    pin, lr_c = arm["pin"], arm["c"]

    put_c_in_circuit(c)
    env, agent = build("real", "train", seed)
    try:
        if pin:
            t1.pin_drive_edges(agent)
        chains = t2.rim_chains(agent.dome)
        strata = t2.edge_strata(agent.dome)
        record = {
            "issue": 551,
            "condition": condition,
            "gauge_c": int(c),
            "arm": {"rho1_drive_edges": bool(pin), "c_learning_rate": float(lr_c)},
            "note": (
                "`gauge_c` is ADR-0010's incoherence constant, in circuit for the whole "
                "run -- projection cap, reconciliation gain and fold-margin nomination "
                "alike. `arm.c_learning_rate` is T3's unrelated `eta_K = c * eta`."
            ),
            "seed": seed,
            "ticks": ticks,
            "gated": False,
            "surface": t0.surface(),
            "k": int(agent.sheaf.operators.shape.k),
            "n": int(agent.dome.shape.n),
            "chains": chains,
            "gain": gain_provenance(agent.dome, c),
            "comparator": f"537-{condition}-seed{seed}-{ticks}.json (c = 2 in circuit, #547)",
            "checkpoints": [],
        }
        recorder = t2.EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=lr_c)
        transport = TransportRule(agent.sheaf)

        record["at_construction"] = {
            "angles": angles.read_surface(agent, chains, f"construction {condition} c={c} seed {seed}"),
            "cap": trained.cap_read(agent),
        }

        ladder = [cp for cp in t3.CHECKPOINTS if cp <= ticks]
        if ticks not in ladder:
            ladder.append(ticks)
        seen = 0
        for target in ladder:
            window = target - seen
            for _ in t0.teaching_read(agent, window, seed + seen, recorder, bias, transport):
                pass
            seen = target

            entry = {"ticks": target}
            entry["angles"] = angles.read_surface(agent, chains, f"{condition} c={c} seed {seed} @{target}")
            entry["cap"] = trained.cap_read(agent)
            entry["stability"] = stability_read(agent, recorder)
            entry["edges_by_stratum"] = t2.edge_reads(agent, recorder, strata)["by_stratum"]
            entry["sweep_c"] = angles.sweep_c(agent, chains, c_values)
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            record["checkpoints"].append(entry)
            inflight.write_text(json.dumps(record, indent=1))

            a, s = entry["angles"], entry["stability"]
            print(
                f"  {condition} c={c} s{seed} @{target:>6}: ER med {a['composed_er']['median']:.6f} "
                f"p90 {a['composed_er']['p90']:.4f} max {a['composed_er']['max']:.4f} "
                f"| cos med {a['cos_all']['median']:.3f} "
                f"| cap max {entry['cap']['ratio_max']:.3f} at-cap {entry['cap']['cells_at_cap']} "
                f"| h {s['h_norm']['median']:.3g}/{s['h_norm']['max']:.3g} "
                f"e {s['e_norm']['median']:.3g} nonfinite {s['h_nonfinite_cells']}/{s['maps_nonfinite']} "
                f"({entry['elapsed_minutes']:.1f} min)",
                flush=True,
            )
        inflight.replace(out)
        return record
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=sorted(t3.CONDITIONS), nargs="+", default=["baseline", "winner"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--ticks", type=int, default=100_000)
    p.add_argument("--gauge-c", type=int, default=1)
    p.add_argument("--c-values", type=int, nargs="+", default=[1, 2])
    args = p.parse_args()

    print(f"[B9] c = {args.gauge_c} in circuit; "
          f"{json.dumps(gain_provenance(build_graph(DEFAULT_SPEC), args.gauge_c)['gain_ratio_vs_c2'])}",
          flush=True)
    for condition in args.condition:
        for seed in args.seeds:
            out = _HERE / f"551-c{args.gauge_c}-{condition}-seed{seed}-{args.ticks}.json"
            if out.exists():
                print(f"[B9] {out.name} already at the horizon, skipping", flush=True)
                continue
            print(f"[B9] {condition} seed {seed}, {args.ticks} ticks -> {out.name}", flush=True)
            run_seed(condition, seed, args.ticks, out, args.gauge_c, args.c_values)
            print(f"  wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
