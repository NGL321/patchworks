"""Have the stalks absorbed the rim's fixed 2x map mismatch (#468, for #469)?

[#469](https://github.com/NGL321/patchworks/issues/469) states a failure the
record has derived but never read. On all 273 **boundary-incident** edges the two
endpoint maps sit permanently at a scale ratio of exactly 2 —
[ADR-0010](https://github.com/NGL321/patchworks/blob/main/docs/adr/0010-restriction-map-scale-is-gauge-fixed.md)'s
exact gauge pins the boundary end at `‖F‖_F = 1`, the band's projection carries
the free end to `ρ = 2` and holds it there, and
[#416](https://github.com/NGL321/patchworks/issues/416) measured all 273 within
0.2% of that face at 100,000 ticks. The projection sits **outside** the
disagreement objective and restores the mismatch every tick, so the maps cannot
null it. The **stalks** can — by the interior end shrinking toward half its
boundary neighbour — and that is the side effect ADR-0010 rejected *the exact
gauge everywhere* to avoid, arriving through the option it chose instead.

Whether that has actually happened is what this rig reads::

    python benchmarks/rim_stalk_scale.py read
    python benchmarks/rim_stalk_scale.py read --dome small --ticks 200 --no-file

**The surface.** Every figure here is taken on the **real dome** (`DEFAULT_SPEC`,
414 cells / 682 edges), `train` split, with **both** rules stepping — the
`PredictionRule` and the `TransportRule` — on `main` at the tick this was run,
which is the surface carrying
[ADR-0031](https://github.com/NGL321/patchworks/blob/main/docs/adr/0031-the-sparsity-pressure-is-deleted.md)'s
deletion of the sparsity pressure and
[ADR-0032](https://github.com/NGL321/patchworks/blob/main/docs/adr/0032-the-maps-learn-isometric-transport-and-a-spectral-floor-expresses-it.md)'s
spectral floor. #127's standing rule: a reading carries the surface it was taken
on wherever the surface is contestable, and this one is — #416's own numbers were
taken before the floor landed.

## The quantity, and its orientation

Per edge, the ratio of the two endpoints' **node stalk** magnitudes,
`‖x_near‖ / ‖x_far‖`, oriented by which end the pressure is supposed to be
pushing on:

- On a **boundary-incident** edge the numerator is the **pinned** end — the
  boundary cell, whose map is nailed at the exact gauge — and the denominator is
  the free interior end whose map sits at `ρ`. #469's failure says this ratio
  drifts toward **2**.
- On an **interior** edge neither end is pinned, so the orientation is by
  **depth to the rim**: shallower over deeper. That is the same orientation the
  sensory and motor populations get, where the pinned end *is* the shallow one,
  which is what makes the interior population a control rather than a different
  measurement.

**The drive population is the one place the two orientations disagree**, and the
rig says so rather than smoothing it: the drive cell is pinned but sits at the
apex, not on the rim, so its 8 edges are oriented **pinned-over-free** like the
rest of the boundary-incident population and are excluded from every depth-graded
table. #469 already carries the drive as amplifying outbound where the sensory
population attenuates inbound, and this is the same asymmetry in the geometry.

## Two columns, because the raw norm is dimension-confounded

#468 asks for `‖x_boundary‖ / ‖x_interior‖`, and that raw ratio is the headline
and the metric. It is also **not** dimensionless: this dome's stalk widths are 48
at a patch cell, 32 at a predicting cell, 6 at the actuator, 2 proprioceptive and
1 at a touch or drive cell. Two vectors of the same per-component scale but
different widths already read a ratio of `√(n_near/n_far)` — **1.22x** on a
sensory edge, before any pressure at all.

So every table carries a second column, `rms`, which is `‖x‖/√n` at each end. It
is the same reading with the width divided out, and it is the one to look at when
asking whether the *scale* moved. This is #416's own convention — that rig
reported the Frobenius ratio as *the* ratio and carried `σ_max`/`σ_min` beside it
— and for the same reason: the rig does not get to pick the quantity a bar is
written on, but it does have to say what else it saw.

## The origin of the drift axis is tick 1, and that is a correction to #468

#468 asks for construction to be reported as the axis's origin "the way
`prototypes/edge-scale-ratio-416/` did", on the ground that `INITIAL_NORM = 1.0`
puts the reading at exactly 1 by construction. **That is true of #416's quantity
and not of this one.** `INITIAL_NORM` pins the *maps*; a fresh `Sheaf`'s **node
stalks are identically zero** — `StalkLayout.empty` hands back a zeroed buffer and
nothing has written to it yet — so at tick 0 this ratio is `0/0` on every edge.
There is no construction reading to report.

So the rig reports tick 0 as **empty**, and the origin of the drift axis is
**tick 1**: the first tick that has stalk content at all. Unlike #416's 1, tick 1
here is a *measurement* — it is whatever one pass of the world's write and one
reconciliation leaves — and it is stated as such rather than dressed as
arithmetic. The whole reading is still about what training does; it simply has no
free zero to subtract.

## Two horizons, because #178's trap is live on this axis

[#178](https://github.com/NGL321/patchworks/issues/178) is that a headline read
at one horizon can reverse at another, and #416 is the worked example **on this
very axis**: its own headline was false at 30,000 ticks and true at 100,000, and
its interior median moved by two orders of magnitude between them. So the ladder
runs to 100,000 and every checkpoint is kept.

## The chaining is the second half and it is not optional

ADR-0010's stated failure is not that one edge's stalks compensate. It is that
*"connectivity chains it into near-uniform stalk scale across the graph"*. So the
rig reads the stalk magnitude **against depth from the rim** as well as at the rim
edge itself: a compensation that stops at the first interior cell is a local
accommodation, and one that grades away with depth is the chained failure
ADR-0010 named.

## The control is the interior population

409 edges whose two ends sit at map-scale ratio 1 by construction, read for the
same oriented stalk ratio. If they show the same asymmetry, the rim's 2x is not
what produced it — stalk magnitudes grading with depth for ordinary reasons is
the alternative explanation, and it is a cheap one to rule out.

## What this read cannot claim

Nothing here is trained: the ticks buy an adapting surface, not a converged one,
which is the fence `benchmarks/detectability.py` and `benchmarks/floor_split.py`
carry in the same words. And this is a **read, not a ruling** — #468 says so in
terms. If the stalks have absorbed it, what follows is a grilling to re-open
`ρ = 1` on this population, not an edit to the gauge.

Like every script here **it asserts nothing** and its exit code does not move.
Pass `--no-file` on any read that is not *the* read.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _part in ("src", "tools", "benchmarks", "tests"):
    _path = str(ROOT / _part)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import loop_length  # noqa: E402
import untrained_fixed_point as ufp  # noqa: E402
from patchworks.agent import run  # noqa: E402
from patchworks.graph import Dome  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402

from cutoff_report import report as report_cutoffs  # noqa: E402

#: #416's long horizon, so this reading sits beside the one that produced the
#: mismatch it is about rather than at a horizon nobody else has run.
TICKS = 100000

#: #416's seeds at its own 30,000-tick horizon, plus its seed 3 at 100,000.
#: A 100,000-tick run contains its own 30,000 checkpoint, so three seeds at 30k
#: and one at 100k is what the ladder below delivers from these.
SEEDS = (0, 1, 2, 3)

#: The horizon each seed runs to. Seeds not named here run to `--ticks`.
HORIZON = {0: 30000, 1: 30000, 2: 30000}


def ladder(ticks: int) -> list[int]:
    """Checkpoints: construction, a decade ladder, then the end.

    `prototypes/edge-scale-ratio-416/read.py`'s ladder, unchanged and for its
    reason: two endpoints cannot tell a monotone drift from a step that happened
    in the first hundred ticks and held, and #178's trap is read off the shape of
    the climb rather than off either end of it.
    """
    points = [0, 1, 10, 100, 300, 1000, 3000, 10000, 30000, 100000]
    return sorted({p for p in points if p <= ticks} | {ticks})


def depth_to_rim(dome: Dome) -> dict[int, int]:
    """`d(c, rim)` for every cell, by `loop_length`'s own sweep.

    Taken off `loop_length.adjacency` and `loop_length.rim_of` rather than
    re-derived, so "position on the channel" here is the same rim #351's rig and
    ADR-0026 measure to. Rim cells are 0; the drive cell is not part of the rim
    and gets its own distance like any other cell — which is why the drive
    population is oriented by pinning and excluded from the depth tables.
    """
    adjacency = loop_length.adjacency(dome)
    rim = loop_length.rim_of(dome)
    distance = {cell: 0 for cell in rim}
    queue = collections.deque(rim)
    while queue:
        cell = queue.popleft()
        for other in adjacency[cell]:
            if other not in distance:
                distance[other] = distance[cell] + 1
                queue.append(other)
    return distance


def geometry(dome: Dome) -> dict[str, object]:
    """Per-cell and per-edge metadata: what the stalk norms are read against.

    The orientation is decided **here**, once, and stored — so the summary cannot
    pick a different one from the tables, and so a later session can see which end
    this rig called the numerator without re-running anything.
    """
    distance = depth_to_rim(dome)
    cells = [
        {
            "cell": cell.id,
            "kind": cell.kind.value,
            "stalk": cell.stalk,
            "pinned": bool(cell.is_boundary),
            "depth": distance.get(cell.id, -1),
        }
        for cell in dome.cells
    ]
    edges = []
    for edge in dome.edges:
        u, v = dome.cells[edge.u], dome.cells[edge.v]
        pinned = sum(1 for c in (u, v) if c.is_boundary)
        if pinned == 1:
            # Boundary-incident: the pinned end is the numerator, whatever its
            # depth. This is the orientation #469's failure is stated in.
            near, far = (u, v) if u.is_boundary else (v, u)
            orientation = "pinned over free"
        else:
            # Interior: no pinning to orient by, so depth does it. An edge whose
            # ends sit at the same depth has no orientation at all and is counted
            # rather than given an arbitrary one.
            du, dv = distance.get(u.id, -1), distance.get(v.id, -1)
            if du == dv:
                near, far, orientation = u, v, "none"
            else:
                near, far = (u, v) if du < dv else (v, u)
                orientation = "shallow over deep"
        edges.append(
            {
                "edge": edge.id,
                "kind": edge.kind.value,
                "m": edge.m,
                "near": near.id,
                "far": far.id,
                "near_kind": near.kind.value,
                "far_kind": far.kind.value,
                "near_stalk": near.stalk,
                "far_stalk": far.stalk,
                "near_depth": distance.get(near.id, -1),
                "far_depth": distance.get(far.id, -1),
                "pinned_ends": pinned,
                "orientation": orientation,
                # What the raw ratio reads on two ends of equal per-component
                # scale. The dimension confound, per edge, stated rather than
                # divided out silently.
                "width_bias": float(np.sqrt(near.stalk / far.stalk)),
            }
        )
    return {"cells": cells, "edges": edges}


@torch.no_grad()
def magnitudes(sheaf) -> dict[str, list[float]]:
    """`[cells]` node stalk norm and per-component RMS, in float64.

    **float64 for `_rank`'s reason**, stated in `patchworks.diagnostics`: the norm
    squares entries on its way to a sum, and reading a scale-invariant quantity in
    float32 can come back with the exponent gone. The cast is one copy of a
    12k-element buffer on the checkpoint cadence, which is nothing.
    """
    flat = sheaf.stalks.detach().to(torch.float64)
    norms, rms = [], []
    for cell in sheaf.layout.dome.cells:
        block = flat[sheaf.layout.slice(cell.id)]
        norm = float(block.norm())
        norms.append(norm)
        rms.append(norm / float(np.sqrt(cell.stalk)))
    return {"norm": norms, "rms": rms}


def stage(target: pathlib.Path) -> pathlib.Path:
    """Where a run in flight writes, and what happens to the last one that died.

    The checkpoint write is inside the tick loop because a 100,000-tick run is
    over an hour on this dome, long enough that a kill part way through is a real
    outcome rather than a hypothetical — and an all-or-nothing write loses every
    checkpoint it already passed.

    That much is not enough on its own, and this rig learned it the expensive
    way: writing each checkpoint straight to the final name means a **re-run**
    truncates the file from its first frame, so a retry killed at tick 300
    destroys a previous attempt that had reached 10,000. The partial record is
    exactly what checkpointing is for, and the retry is exactly when it gets
    thrown away.

    So a run in flight writes to `<name>.inflight.json` and only becomes
    `<name>.json` when it finishes. Anything left in flight by a kill is moved
    aside under a `.killed-<n>.json` name rather than overwritten, so the deepest
    attempt survives however many times the run is retried.
    """
    inflight = target.with_suffix(".inflight.json")
    if inflight.exists():
        index = 0
        while True:
            kept = target.with_suffix(f".killed-{index}.json")
            if not kept.exists():
                break
            index += 1
        inflight.replace(kept)
    return inflight


def one(
    seed: int, ticks: int, split: str, dome_name: str, out: pathlib.Path | None
) -> dict:
    """One seed, writing its record as each checkpoint lands. See :func:`stage`."""
    started = time.time()
    env, agent = ufp.build(dome_name, split, seed)
    record: dict[str, object] = {
        "seed": seed,
        "ticks": ticks,
        "split": split,
        "dome": dome_name,
        "geometry": None,
        "frames": [],
    }
    inflight = stage(out) if out is not None else None
    try:
        dome = agent.dome
        record["geometry"] = geometry(dome)
        wanted = set(ladder(ticks))
        frames: list[dict] = record["frames"]  # type: ignore[assignment]

        def keep(tick: int) -> None:
            frames.append({"tick": tick, **magnitudes(agent.sheaf)})
            record["seconds"] = round(time.time() - started, 1)
            if inflight is not None:
                inflight.write_text(json.dumps(record), encoding="utf-8")

        if 0 in wanted:
            keep(0)
        bias = PredictionRule(agent.sheaf)
        transport = TransportRule(agent.sheaf)
        for index, _outcome in enumerate(run(agent, ticks, seed=seed)):
            bias.step()
            if agent.sheaf.ticks > 1:
                transport.step()
            if (index + 1) in wanted:
                keep(index + 1)
    finally:
        env.close()
    if inflight is not None and inflight.exists():
        # The run reached its horizon, so the record is complete and claims the
        # final name. A kill leaves the `.inflight.json` behind instead, which is
        # what `stage` preserves on the next attempt.
        inflight.replace(out)
    return record


# ---------------------------------------------------------------------------
# reading a record
# ---------------------------------------------------------------------------


def quantiles(values) -> dict[str, float]:
    """p05/p25/median/p75/p95 plus the extremes — this rig's neighbours' convention."""
    flat = np.asarray(values, dtype=float).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {}
    p05, p25, p50, p75, p95 = np.percentile(flat, (5, 25, 50, 75, 95))
    return {
        "min": float(flat.min()),
        "p05": float(p05),
        "p25": float(p25),
        "median": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "max": float(flat.max()),
        "n": int(flat.size),
    }


def edge_ratios(record: dict, frame: dict) -> dict[str, np.ndarray]:
    """Per-edge oriented stalk ratios at one checkpoint, both columns."""
    edges = record["geometry"]["edges"]
    out = {}
    for column in ("norm", "rms"):
        values = np.asarray(frame[column], dtype=float)
        near = np.array([values[e["near"]] for e in edges])
        far = np.array([values[e["far"]] for e in edges])
        out[column] = np.where(far > 0, near / np.where(far > 0, far, 1.0), np.nan)
    return out


def populations(record: dict) -> dict[str, np.ndarray]:
    """Boolean masks over the edge list, in #469's own three-plus-one split."""
    kinds = np.array([e["kind"] for e in record["geometry"]["edges"]])
    return {
        "boundary-incident": kinds != "interior",
        "sensory": kinds == "sensory",
        "drive": kinds == "drive",
        "actuator": kinds == "motor",
        "interior (control)": kinds == "interior",
    }


def by_depth(record: dict, frame: dict) -> list[dict]:
    """Median cell magnitude at each depth from the rim.

    The chaining half. The drive cell is pinned but not on the rim, so it is
    excluded here — it would otherwise put a 1-wide pinned stalk into the apex's
    depth bin and read as a chained effect that is nothing of the kind.
    """
    rows = []
    cells = record["geometry"]["cells"]
    for column in ("norm", "rms"):
        values = np.asarray(frame[column], dtype=float)
        buckets: dict[int, list[float]] = collections.defaultdict(list)
        for cell in cells:
            if cell["kind"] == "drive" or cell["depth"] < 0:
                continue
            buckets[cell["depth"]].append(values[cell["cell"]])
        for depth in sorted(buckets):
            rows.append(
                {
                    "column": column,
                    "depth": depth,
                    "cells": len(buckets[depth]),
                    "median": float(np.median(buckets[depth])),
                }
            )
    return rows


def graded(record: dict, frame: dict) -> list[dict]:
    """The oriented edge ratio at each rung of the depth ladder.

    This is what tells a local accommodation from a chained one: the rim edge is
    the `0 -> 1` rung, and a compensation that stops there leaves every deeper
    rung at 1 while a chained one grades away.
    """
    edges = record["geometry"]["edges"]
    ratios = edge_ratios(record, frame)
    rows = []
    buckets: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for index, edge in enumerate(edges):
        if edge["orientation"] == "none" or edge["kind"] == "drive":
            continue
        buckets[(edge["near_depth"], edge["far_depth"])].append(index)
    for rung in sorted(buckets):
        picked = buckets[rung]
        rows.append(
            {
                "rung": f"{rung[0]} -> {rung[1]}",
                "edges": len(picked),
                "norm": float(np.nanmedian(ratios["norm"][picked])),
                "rms": float(np.nanmedian(ratios["rms"][picked])),
            }
        )
    return rows


def readings(records: list[dict]) -> dict[str, float]:
    """What this read has to offer a `measurement` cutoff, by name.

    `rim_stalk_ratio` is #469's bar and is the **raw** norm ratio the ticket asks
    for, medianed over the 273 boundary-incident edges and then over seeds —
    fleet medians first, then the across-seed median, the way
    `benchmarks/floor_split.py` and `benchmarks/detectability.py` take theirs.

    `rim_stalk_ratio_rms` is the same reading with the stalk-width confound
    divided out. It is offered so that a session moving the bar has the
    dimension-controlled number under the same name discipline, and **not** as a
    substitute: the metric a bar is written on is the problem's to choose.
    """
    per_seed: dict[str, list[float]] = collections.defaultdict(list)
    for record in records:
        if not record["frames"]:
            continue
        frame = record["frames"][-1]
        ratios = edge_ratios(record, frame)
        for name, mask in populations(record).items():
            for column in ("norm", "rms"):
                per_seed[f"{name}|{column}"].append(
                    float(np.nanmedian(ratios[column][mask]))
                )
    if not per_seed:
        return {}

    def across(key: str) -> float:
        return float(np.median(per_seed[key])) if per_seed.get(key) else float("nan")

    return {
        "rim_stalk_ratio": across("boundary-incident|norm"),
        "rim_stalk_ratio_rms": across("boundary-incident|rms"),
        "rim_stalk_ratio_sensory": across("sensory|norm"),
        "rim_stalk_ratio_drive": across("drive|norm"),
        "rim_stalk_ratio_actuator": across("actuator|norm"),
        "interior_stalk_ratio": across("interior (control)|norm"),
        "interior_stalk_ratio_rms": across("interior (control)|rms"),
    }


def show(record: dict) -> None:
    """One seed's tables: the drift ladder, the populations, the chaining."""
    seed, ticks = record["seed"], record["ticks"]
    frames = record["frames"]
    if not frames:
        print(f"\n   seed {seed}: no checkpoint was taken; nothing read.")
        return
    masks = populations(record)
    print(f"\n-- seed {seed}, {ticks} ticks, {record.get('seconds', '?')} s --")

    print("\n   the drift axis, median oriented stalk ratio (raw norm)")
    short = {n: n.split(" ")[0][:9] for n in masks}
    print("   {:>8}".format("tick") + "".join(f"{short[n]:>11}" for n in masks))
    for frame in frames:
        if frame["tick"] == 0:
            # A fresh sheaf's node stalks are identically zero, so every ratio
            # here is 0/0. Said out loud rather than printed as a row of NaN:
            # the module docstring carries why this differs from #416's zero.
            print(
                "   {:>8}".format(0)
                + "   the stalks are identically zero; no ratio exists. "
                "The axis starts at tick 1."
            )
            continue
        ratios = edge_ratios(record, frame)
        row = "   {:>8}".format(frame["tick"])
        for mask in masks.values():
            row += f"{np.nanmedian(ratios['norm'][mask]):>11.4g}"
        print(row + ("   <- the origin, and it is measured" if frame["tick"] == 1 else ""))

    last = frames[-1]
    ratios = edge_ratios(record, last)
    print(f"\n   at tick {last['tick']}, per population")
    print(
        "   {:<20}{:>7}{:>9}{:>9}{:>9}{:>9}{:>9}".format(
            "population", "edges", "median", "p25", "p75", "rms", "width"
        )
    )
    for name, mask in masks.items():
        stats = quantiles(ratios["norm"][mask])
        rms = quantiles(ratios["rms"][mask])
        bias = np.median(
            [e["width_bias"] for e, m in zip(record["geometry"]["edges"], mask) if m]
        )
        print(
            f"   {name:<20}{stats['n']:>7}{stats['median']:>9.4g}"
            f"{stats['p25']:>9.4g}{stats['p75']:>9.4g}{rms['median']:>9.4g}"
            f"{bias:>9.4g}"
        )

    print("\n   the chaining: median cell magnitude against depth from the rim")
    print("   {:>7}{:>8}{:>12}{:>12}".format("depth", "cells", "norm", "rms"))
    rows = by_depth(record, last)
    norms = {r["depth"]: r for r in rows if r["column"] == "norm"}
    rmss = {r["depth"]: r for r in rows if r["column"] == "rms"}
    for depth in sorted(norms):
        print(
            f"   {depth:>7}{norms[depth]['cells']:>8}"
            f"{norms[depth]['median']:>12.4g}{rmss[depth]['median']:>12.4g}"
        )

    print("\n   the chaining: oriented edge ratio at each rung (drive excluded)")
    print("   {:>10}{:>8}{:>12}{:>12}".format("rung", "edges", "norm", "rms"))
    for row in graded(record, last):
        print(
            f"   {row['rung']:>10}{row['edges']:>8}"
            f"{row['norm']:>12.4g}{row['rms']:>12.4g}"
        )


def read(
    seeds: tuple[int, ...],
    ticks: int,
    split: str,
    dome_name: str,
    out: pathlib.Path | None,
    *,
    file: bool = True,
) -> list[dict]:
    print(
        "\n== the rim's 2x, and whether the stalks have taken it =="
        f"\n   dome {dome_name}, split {split}, both rules stepping"
    )
    records = []
    for seed in seeds:
        horizon = min(HORIZON.get(seed, ticks), ticks)
        target = None
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            target = out / f"468-{dome_name}-seed{seed}-{horizon}.json"
        print(f"\n   starting seed {seed} ({horizon} ticks)", flush=True)
        record = one(seed, horizon, split, dome_name, target)
        records.append(record)
        show(record)
        if target is not None:
            print(f"\n   -> {target.name}", flush=True)
    # ASCII, for `cutoff_report.report`'s reason: the report goes to whatever
    # console the rig was run from, and on Windows that is still cp1252.
    print(
        "\n   The numerator is the pinned end on a boundary-incident edge and "
        "the shallower\n   end on an interior one; `rms` is the same ratio with "
        "the stalk width divided\n   out, and `width` is what the raw ratio "
        "reads at equal per-component scale.\n   Tick 0 has no stalk content at "
        "all, so the drift axis starts at tick 1 and its\n   origin is a "
        "measurement rather than construction -- see this module's docstring."
    )
    report_cutoffs("rim_stalk_scale", readings(records), file=file)
    return records


def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("read",))
    parser.add_argument("--dome", choices=("full", "small"), default="full")
    parser.add_argument("--split", default="train")
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--out",
        default=str(ROOT / "prototypes" / "rim-stalk-scale-468"),
        help="where the per-seed records land; `-` writes none",
    )
    parser.add_argument(
        "--no-file",
        action="store_true",
        help=(
            "print the cutoff report and touch the tracker not at all. Pass it "
            "on any read that is not *the* read."
        ),
    )
    arguments = parser.parse_args(argv)
    read(
        tuple(arguments.seeds),
        arguments.ticks,
        arguments.split,
        arguments.dome,
        None if arguments.out == "-" else pathlib.Path(arguments.out),
        file=not arguments.no_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
