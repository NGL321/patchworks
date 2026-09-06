"""B9 (#551): which cells breach the cap under `c = 1`, and why the winner arm is where it shows.

The 100k arms read `lambda_max(sum_e F^T F) / (g_v^2 c_v) > 1` in the **winner**
arm from 5k ticks on (1.107-1.120) and never in the baseline. A ratio over one
says the projection is not restoring the bound the reconciliation gain divides
by -- which is exactly the pairing #220 insisted lands together -- so it is worth
naming the cells rather than inferring them.

The suspicion this tests: `t1.pin_drive_edges` is a **rig flag applied to an
already-built agent**, so it pins drive-edge maps *after* `overlap_target` and
`pinned_incidence` were computed. `RestrictionMaps.project` reaches only maps
with scale freedom to spend, so a cell whose incidence the rig has pinned behind
the buffer's back cannot be pushed apart -- and at `c = 2` the slack hid it.

Runs the winner arm to the first checkpoint where the breach appeared and dumps
every offending cell with the facts that would explain it.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T4/b9_cap_breach.py --ticks 5000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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
gauge_c = _load("t4_gauge_c", _HERE / "gauge_c.py")

from patchworks import restriction as R  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from untrained_fixed_point import build  # noqa: E402


@torch.no_grad()
def breaches(agent, tol: float = 1e-6) -> list[dict]:
    dome = agent.dome
    maps = agent.sheaf.maps
    peaks = maps.gram_peaks().numpy()
    target = maps.overlap_target.numpy()
    pinned = R.pinned_incidence(dome)
    counts = R.overlap_counts(dome).numpy()
    out = []
    for cid, (peak, tgt) in enumerate(zip(peaks, target)):
        if tgt <= 0 or peak / tgt <= 1 + tol:
            continue
        cell = dome.cells[cid]
        incident = list(dome.incident[cid])
        out.append(
            {
                "cell": cid,
                "ratio": float(peak / tgt),
                "peak": float(peak),
                "target": float(tgt),
                "degree": int(dome.degrees[cid]),
                "stalk": int(cell.stalk),
                "level": int(cell.index.level),
                "is_boundary": bool(cell.is_boundary),
                "pinned_incidence": bool(pinned[cid]),
                "c_v": float(counts[cid]),
                "incident_edge_kinds": sorted({dome.edges[e].kind.value for e in incident}),
                "maps_pinned_by_restriction": [
                    bool(R.map_is_pinned(dome, e, cid)) for e in incident
                ],
            }
        )
    return sorted(out, key=lambda r: -r["ratio"])


def run(condition: str, c: int, ticks: int, seed: int) -> dict:
    arm = t3.CONDITIONS[condition]
    gauge_c.put_c_in_circuit(c)
    env, agent = build("real", "train", seed)
    try:
        if arm["pin"]:
            t1.pin_drive_edges(agent)
        recorder = t2.EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=arm["c"])
        transport = TransportRule(agent.sheaf)
        strata = t2.edge_strata(agent.dome)
        for _ in t0.teaching_read(agent, ticks, seed, recorder, bias, transport):
            pass
        rows = breaches(agent)
        return {
            "condition": condition,
            "gauge_c": c,
            "ticks": ticks,
            "seed": seed,
            "breaching_cells": len(rows),
            "rows": rows[:20],
            # The matched half of "what else moved": #547's arms never recorded
            # these, so a c = 1 edge read has nothing to be differenced against
            # unless c = 2 is re-run on the same rig at the same horizon.
            "edges_by_stratum": t2.edge_reads(agent, recorder, strata)["by_stratum"],
            "stability": gauge_c.stability_read(agent, recorder),
            "gain": gauge_c.gain_provenance(agent.dome, c),
        }
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticks", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    report = []
    for condition in ("winner", "baseline"):
        for c in (1, 2):
            r = run(condition, c, args.ticks, args.seed)
            print(
                f"{condition:>8} c={c}: {r['breaching_cells']} cells over the cap"
                + (f", worst {r['rows'][0]['ratio']:.4f} at cell {r['rows'][0]['cell']}"
                   f" (deg {r['rows'][0]['degree']}, boundary {r['rows'][0]['is_boundary']},"
                   f" pinned_incidence {r['rows'][0]['pinned_incidence']},"
                   f" kinds {r['rows'][0]['incident_edge_kinds']},"
                   f" maps_pinned {r['rows'][0]['maps_pinned_by_restriction']})" if r["rows"] else ""),
                flush=True,
            )
            report.append(r)
    (_HERE / f"551-cap-breach-seed{args.seed}-{args.ticks}.json").write_text(
        json.dumps(report, indent=1)
    )


if __name__ == "__main__":
    main()
