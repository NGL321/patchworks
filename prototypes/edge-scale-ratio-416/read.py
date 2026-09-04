"""The edge scale ratio against ADR-0010's admissible band (#416).

    python prototypes/edge-scale-ratio-416/read.py --seeds 0 1 2 --ticks 30000

**What is read.** Per edge, per direction, `σ(F_u) / σ(F_v)` — the ratio of the
two endpoint maps' scales, which ADR-0010 leaves deliberately free and
[#411](https://github.com/NGL321/patchworks/issues/411) gave an opinion about.

**The scale is `‖F‖_F`, and that is not a choice this rig makes.** ADR-0010's
gauge is stated on the Frobenius norm — the band `‖F‖_F ∈ [1/ρ, ρ]` at a
predicting cell, the exact gauge `‖F‖_F = 1` at a boundary cell — so the
quantity the band constrains *is* the Frobenius norm. And under #411's flat
spectrum the two readings coincide exactly: a map with `m` equal singular values
has `σ = ‖F‖_F/√m`, an edge's two ends share one `m` (it is a property of the
edge stalk), so `σ_u/σ_v = ‖F_u‖_F/‖F_v‖_F` identically. The rig therefore
reports the Frobenius ratio as *the* ratio, and carries `σ_max` and `σ_min`
beside it so the reading does not silently assume the flatness #411 has not
bought yet.

**Two admissible bands, not one.** #416 quotes `ρ² = 4`, which is the interior
edge's: both ends carry the band, so the ratio ranges over `[1/ρ², ρ²]`. On a
**boundary-incident** edge one end is pinned at exactly 1 and only the other
moves, so the admissible ratio is `[1/ρ, ρ]` — a factor of 2, not 4. 273 of this
dome's 682 edges are in that position, and pooling them into one band would
report them as using a quarter of their freedom when they are using half of it.

**Untrained is 1 by construction, not by measurement.** `INITIAL_NORM = 1.0`
draws every map at the band's geometric centre, precisely "so no edge starts
with a scale ratio built into it". The rig records the construction reading
anyway — it is the zero of the drift axis — but it is arithmetic, not evidence.

**Per edge, per direction** (#127's standing rule; #181's general form). Nothing
here is averaged over the graph or keyed by level. The ratio is stated in both
directions — they are reciprocals, so the population of `u/v` ratios and the
population of `v/u` ratios are mirror images, and the rig publishes the
**directed** ratio per edge together with `|log₂|`, which is direction-free and
is what a spread is honestly taken over.

**A rig asserts nothing** (`benchmarks/run_reporting.py`). This files no cutoff
and writes to no register. It is a prototype, not a published rig: promoting it
to `benchmarks/` would be its own ticket (#127's standing note).
"""

import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _part in ("src", "tools", "benchmarks", "tests"):
    _path = str(ROOT / _part)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import loop_length  # noqa: E402
import untrained_fixed_point as ufp  # noqa: E402
from patchworks.agent import run  # noqa: E402
from patchworks.graph import Dome  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from patchworks.restriction import GAUGE_RHO, pair_index  # noqa: E402

#: #237's and #393's length, so this reading sits beside theirs rather than at a
#: horizon nobody else has run.
TICKS = 30000


def quantiles(values: np.ndarray) -> dict[str, float]:
    """p05/p25/median/p75/p95 plus the extremes, this rig's neighbours' convention."""
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


def depth_to_rim(dome: Dome) -> dict[int, int]:
    """`d(c, rim)` for every cell, by the same sweep `loop_length` uses.

    Taken off `loop_length.adjacency` and `loop_length.rim_of` rather than
    re-derived, so "position on the channel" here is the same rim #351's rig and
    ADR-0026 measure to. Rim cells are 0; the drive cell is not part of the rim
    and gets its own distance like any other cell.
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


def geometry(dome: Dome) -> list[dict]:
    """Per-edge, per-end metadata: the columns the ratio is read against.

    `admissible` is the edge's own band, not the map's: `ρ²` where both ends
    carry the band, `ρ` where one end is pinned at the exact gauge, and 1 where
    both are (which this dome has none of, and the rig does not assume).
    """
    distance = depth_to_rim(dome)
    rows = []
    for edge in dome.edges:
        ends = (dome.cells[edge.u], dome.cells[edge.v])
        free = sum(0 if cell.is_boundary else 1 for cell in ends)
        rows.append(
            {
                "edge": edge.id,
                "kind": edge.kind.value,
                "m": edge.m,
                "u": edge.u,
                "v": edge.v,
                "u_kind": ends[0].kind.value,
                "v_kind": ends[1].kind.value,
                "u_pinned": bool(ends[0].is_boundary),
                "v_pinned": bool(ends[1].is_boundary),
                "free_ends": free,
                "admissible": float(GAUGE_RHO**free),
                "depth": min(distance.get(edge.u, -1), distance.get(edge.v, -1)),
                "u_depth": distance.get(edge.u, -1),
                "v_depth": distance.get(edge.v, -1),
            }
        )
    return rows


@torch.no_grad()
def scales(sheaf) -> dict[str, np.ndarray]:
    """`[pairs]` Frobenius norm, `σ_max` and `σ_min`, in float64.

    **float64 for `_rank`'s reason**, stated in `patchworks.diagnostics`: the
    norm squares entries on its way to a sum, and a scale-invariant quantity
    taken in float32 can come back `inf`. The cast is one small copy of a
    `1364 × 8 × 48` tensor on the diagnostic cadence.

    **`σ_min` is the `k`-th largest of the padded matrix**, `k = min(m_e, n_v)`,
    which is exact rather than an approximation: the padded rows and masked
    columns contribute singular values of exactly zero, so the padded matrix's
    nonzero singular values *are* the block's, and its `k`-th largest is the
    block's smallest. Same identity `Diagnostics.edge_reading` leans on when it
    takes the participation ratio off the whole padded tensor.
    """
    maps = sheaf.maps
    weights = maps.maps.detach().to(torch.float64)
    norms = weights.flatten(1).norm(dim=-1)
    singular = torch.linalg.svdvals(weights)
    top = singular[:, 0]
    bottom = torch.empty_like(top)
    dome = maps.dome
    for edge in dome.edges:
        for side, cell_id in enumerate((edge.u, edge.v)):
            i = pair_index(edge.id, side)
            k = min(edge.m, dome.cells[cell_id].stalk)
            bottom[i] = singular[i, k - 1]
    return {
        "norm": norms.numpy(),
        "sigma_max": top.numpy(),
        "sigma_min": bottom.numpy(),
    }


def ratios(reading: dict[str, np.ndarray], dome: Dome) -> dict[str, np.ndarray]:
    """Per **edge** ratios, from the per-**map** scales. Directed `u/v`.

    The `v/u` direction is the reciprocal and is not stored twice: the rig
    prints both, and `|log₂|` is the direction-free magnitude every spread is
    taken over.
    """
    left = np.array([2 * e.id for e in dome.edges])
    right = left + 1
    out = {}
    for name, values in reading.items():
        u = values[left]
        v = values[right]
        safe = np.where(v > 0, v, np.nan)
        out[name] = u / safe
        out[f"{name}_u"] = u
        out[f"{name}_v"] = v
    return out


def snapshot(sheaf, dome: Dome, tick: int) -> dict:
    """One checkpoint: the whole per-edge ratio population, kept in full.

    The population is 682 edges; storing it whole rather than pre-reduced is
    what lets `summarise.py` slice by kind and by depth without re-running the
    30,000 ticks, and is what keeps the rig from being the only place a
    reduction is chosen.
    """
    reading = scales(sheaf)
    edge = ratios(reading, dome)
    return {
        "tick": tick,
        "norm_ratio": edge["norm"].tolist(),
        "sigma_max_ratio": edge["sigma_max"].tolist(),
        "sigma_min_ratio": edge["sigma_min"].tolist(),
        "norm_u": edge["norm_u"].tolist(),
        "norm_v": edge["norm_v"].tolist(),
        "map_norms": reading["norm"].tolist(),
        "flatness": (reading["sigma_min"] / reading["sigma_max"]).tolist(),
    }


def ladder(ticks: int) -> list[int]:
    """Checkpoints: construction, then a decade ladder, then the end.

    Drift is the discriminating reading #416 names, and a drift is only visible
    against a ladder — two endpoints cannot tell a monotone climb from a step
    that happened in the first hundred ticks and held.
    """
    points = [0, 1, 10, 100, 300, 1000, 3000, 10000, 30000, 100000]
    return sorted({p for p in points if p <= ticks} | {ticks})


def one(seed: int, ticks: int, split: str, dome_name: str) -> dict:
    started = time.time()
    env, agent = ufp.build(dome_name, split, seed)
    try:
        dome = agent.dome
        checkpoints = ladder(ticks)
        wanted = set(checkpoints)
        frames = []
        if 0 in wanted:
            frames.append(snapshot(agent.sheaf, dome, 0))
        bias = PredictionRule(agent.sheaf)
        transport = TransportRule(agent.sheaf)
        for index, _outcome in enumerate(run(agent, ticks, seed=seed)):
            bias.step()
            if agent.sheaf.ticks > 1:
                transport.step()
            if (index + 1) in wanted:
                frames.append(snapshot(agent.sheaf, dome, index + 1))
    finally:
        env.close()
    return {
        "seed": seed,
        "ticks": ticks,
        "split": split,
        "dome": dome_name,
        "rho": GAUGE_RHO,
        "seconds": round(time.time() - started, 1),
        "geometry": geometry(dome),
        "frames": frames,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dome", default="full")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", default=str(pathlib.Path(__file__).parent))
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        print(f"starting seed={seed} ({args.ticks} ticks)", flush=True)
        record = one(seed, args.ticks, args.split, args.dome)
        name = f"416-{args.dome}-seed{seed}-{args.ticks}.json"
        (out / name).write_text(json.dumps(record), encoding="utf-8")
        last = record["frames"][-1]
        magnitude = np.abs(np.log2(np.array(last["norm_ratio"])))
        print(
            f"seed={seed}: |log2 ratio| median {np.median(magnitude):.4g} "
            f"p95 {np.percentile(magnitude, 95):.4g} max {magnitude.max():.4g}  "
            f"[{record['seconds']}s] -> {name}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
