"""T6 (#568 / B19): what a cell knows against what it can say.

[B17](#565) found that `composed_reads` (`T2/run.py:496-520`) multiplies **learned
restriction maps only** and never touches `agent.sheaf.stalks`, so every composed-rank
reading on [#532](#532) is a property of the transport *operator*. This ticket puts
state and emission on the **same cell over the same window**, which no reading on the
map has done, and asks which of two readings of the collapse holds:

* **the specialist** — a cell holds a narrow, self-consistent perspective and says
  most of it; narrow emission is the architecture working;
* **the bottleneck** — the cell's state is rich and the restriction maps lose it in
  translation; the defect sits where the transport architecture is.

**Three quantities, one window, one cell.** Emission is `y_c = F_c h_c` with `F_c` the
cell's incident maps stacked, so the chain from state to edge factors in two:

    state  --(subspace selection)-->  readable  --(singular-value weighting)-->  emitted

* ``state``    — participation ratio of the node-stalk stream `h_c`, split by mask
  block exactly as `T0/run.py:226-229` splits it (total / private / exposed /
  interior, the last being exposed with the drive direction removed).
* ``readable`` — the same stream projected by `Blocks.interior_rowspace`
  (`T0/excitation.py:111`), the orthogonal projector onto what a cell's interior
  neighbours can read. Unweighted: it isolates *which directions survive* from
  *how they are scaled*.
* ``emitted``  — participation ratio of the **concatenation** of what the cell puts
  on all of its edge ends, `[F_e h_c]_e`, computed as the engine computes it
  (`tick.py:827-828`: `maps.restrict(stalks[layout.pair_positions])`) but on the
  stalks as they stand at the observation instant, so state and emission are
  strictly simultaneous rather than a half-tick apart. Reported over all incident
  edges and over interior edges alone.

Both moments are reported for every one of them: [B18](#567) showed the uncentred
participation ratio reads ~1 for a settled belief varying richly around itself, so
an uncentred-only reading would answer the ticket's question wrong in one direction.
`mean_share = |mu|^2 / (|mu|^2 + tr Cov)` is carried beside them.

Deliberately *not* measured: composed rank, principal angles, joint span across
chains, regional consistency. Those are B20/B22's and B16's; this ticket asks one
question and pays for one answer.

Windowing: `WINDOW = 1000` ticks ending at each checkpoint — `T0`'s own window, and
short enough that the maps `F_c` drift little across it while still being applied at
each tick rather than assumed constant.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b19_state_emitted.py \
        --arms reserve reserve_p16 --ticks 20000
"""

from __future__ import annotations

import argparse
import collections
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
_T0, _T2 = (_HERE.parent / n for n in ("T0", "T2"))
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _T0):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t0 = _load("t0_run", _T0 / "run.py")
arms_mod = _load("t6_arms", _HERE / "arms.py")

from excitation import blocks, participation_ratio  # noqa: E402
from patchworks.graph import EdgeKind  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from patchworks.restriction import pair_index  # noqa: E402

#: `T0`'s window, so the state half of this reading is the one `T0/run.py` takes.
WINDOW = 1_000

#: `B16`/`B18`'s ladder, so the checkpoints line up with `564-*` and `567-*`.
CHECKPOINTS = [100, 300, 1_000, 3_000, 10_000, 20_000, 50_000, 100_000]


# -- the cell's emission layout ------------------------------------------------


class Emission:
    """Which endpoint rows of `broadcast` belong to each predicting cell.

    A cell's edge ends are the pairs `pair_index(e, side)` where `side` picks that
    cell, and `tick.py`'s message-passing phase writes `F_e h_c` into exactly those
    rows. Gathered into `[cells, deg_max, m_max]` with a zero pad row, so a cell of
    lower degree contributes zero columns — which change neither sum in the
    participation ratio, the same reason `excitation_rank` masks by multiplying.
    """

    def __init__(self, dome) -> None:
        self.cells = list(dome.predicting)
        pairs = 2 * len(dome.edges)
        rows: list[list[int]] = []
        interior: list[list[bool]] = []
        for cid in self.cells:
            mine, is_int = [], []
            for eid in dome.incident[cid]:
                edge = dome.edges[eid]
                side = 0 if edge.u == cid else 1
                mine.append(pair_index(edge.id, side))
                is_int.append(edge.kind is EdgeKind.INTERIOR)
            rows.append(mine)
            interior.append(is_int)
        self.degree_max = max(len(r) for r in rows)
        idx = torch.full((len(self.cells), self.degree_max), pairs, dtype=torch.long)
        keep_int = torch.zeros(len(self.cells), self.degree_max, dtype=torch.bool)
        for row, (mine, is_int) in enumerate(zip(rows, interior)):
            idx[row, : len(mine)] = torch.tensor(mine, dtype=torch.long)
            keep_int[row, : len(mine)] = torch.tensor(is_int, dtype=torch.bool)
        #: `[cells, deg_max]` into `broadcast` with a trailing pad row.
        self.index = idx
        #: `[cells, deg_max]`: which of those ends is an interior edge.
        self.interior = keep_int

    @torch.no_grad()
    def gather(self, sheaf) -> torch.Tensor:
        """`[cells, deg_max, m_max]`: what each cell is putting on its edges *now*.

        Recomputed from the current stalks rather than read out of
        `sheaf.broadcast`, which holds the previous message-passing phase's
        `outgoing` — taken before reconciliation subtracted its displacement, so it
        is half a tick away from the state this instrument records beside it.
        """
        maps = sheaf.maps
        gathered = sheaf.stalks[sheaf.layout.pair_positions]
        out = maps.restrict(gathered)
        padded = torch.cat([out, out.new_zeros(1, out.shape[1])], dim=0)
        return padded[self.index]


class Recorder:
    """The last `WINDOW` ticks of state and emission for every predicting cell."""

    def __init__(self, agent, emission: Emission) -> None:
        self.agent = agent
        self.emission = emission
        self.h: list[torch.Tensor] = []
        self.y: list[torch.Tensor] = []

    def reset(self) -> None:
        self.h.clear()
        self.y.clear()

    @torch.no_grad()
    def observe(self) -> None:
        sheaf = self.agent.sheaf
        self.h.append(sheaf.stalks[sheaf.layout.predicting_positions].clone())
        self.y.append(self.emission.gather(sheaf))
        if len(self.h) > WINDOW:
            self.h.pop(0)
            self.y.pop(0)

    def streams(self) -> tuple[torch.Tensor, torch.Tensor]:
        """`([cells, T, n], [cells, T, deg_max * m_max])`."""
        h = torch.stack(self.h, dim=0).transpose(0, 1)
        y = torch.stack(self.y, dim=0).transpose(0, 1)
        return h, y.reshape(y.shape[0], y.shape[1], -1)


# -- the read ------------------------------------------------------------------


@torch.no_grad()
def _moment_share(stream: torch.Tensor) -> torch.Tensor:
    """`|mu|^2 / (|mu|^2 + tr Cov)` per cell — [B18](#567)'s diagnostic."""
    x = stream.double()
    mu = x.mean(dim=1)
    mu_energy = mu.pow(2).sum(dim=1)
    tr_cov = (x - mu.unsqueeze(1)).pow(2).sum(dim=(1, 2)) / x.shape[1]
    return mu_energy / (mu_energy + tr_cov).clamp(min=1e-300)


@torch.no_grad()
def haar_maps(agent, emission: Emission, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Shape-, mask- and band-matched **random** maps, as a null for the learned ones.

    The participation ratio is not monotone under a linear map: re-weighting a
    skewed spectrum can *raise* it, so `emitted > state` is not information gained
    and `emitted < state` is not by itself information lost. The control that makes
    the comparison mean something is a map of the **same shape** with the **same
    mask** and the **same band**, drawn at random — if the learned maps lose more of
    the state than random ones of their own shape, the loss is learned; if they lose
    the same, it is the shape.

    Two nulls, because the learned maps differ from an isometry in two ways:

    * ``haar``        — orthonormal rows per edge (ADR-0032's band, exactly), unit
      scale on every edge;
    * ``haar_scaled`` — the same directions, each edge rescaled to the learned map's
      own mean singular value, so the only thing left differing from the surface is
      **which subspace** each lane selects.

    Returns `[cells, deg_max * m_max, n]` each.
    """
    dome, maps = agent.dome, agent.sheaf.maps
    n = dome.shape.n
    m_max = maps.edge_width
    g = torch.Generator().manual_seed(seed)
    rows = len(emission.cells)
    plain = torch.zeros(rows, emission.degree_max * m_max, n, dtype=torch.float64)
    scaled = torch.zeros_like(plain)
    w = maps.maps.detach().double()
    for row, cid in enumerate(emission.cells):
        k_v = int(dome._permitted[cid])
        for d, eid in enumerate(dome.incident[cid]):
            edge = dome.edges[eid]
            side = 0 if edge.u == cid else 1
            m_e = int(edge.m)
            learned = w[pair_index(edge.id, side), :m_e, :k_v]
            sigma = float(torch.linalg.svdvals(learned).mean()) if k_v else 0.0
            draw = torch.randn(k_v, m_e, generator=g, dtype=torch.float64)
            q = torch.linalg.qr(draw)[0][:, :m_e]  # [k_v, m_e], orthonormal columns
            block = q.T  # [m_e, k_v], orthonormal rows — a partial isometry
            lo = d * m_max
            plain[row, lo : lo + m_e, :k_v] = block
            scaled[row, lo : lo + m_e, :k_v] = sigma * block
    return plain, scaled


@torch.no_grad()
def read(agent, recorder: Recorder, attrs: dict, null_seed: int | None = None) -> dict:
    """Every per-cell number this ticket owes, over one window."""
    h, y = recorder.streams()
    bl = blocks(agent)
    ticks = h.shape[1]

    out: dict[str, torch.Tensor] = {}
    # State side, exactly `T0/run.py:226-229`'s split.
    out.update(t0.excitation_reads(h, bl))
    # The readable part: the state projected onto what interior neighbours can read.
    readable = torch.einsum("ctn,cnm->ctm", h.double(), bl.interior_rowspace.double())
    out["pr_readable"] = participation_ratio(readable)
    out["pr_readable_centred"] = participation_ratio(readable, centred=True)
    # Emitted side: all edge ends, then interior ends alone.
    keep = recorder.emission.interior.to(y.dtype).repeat_interleave(
        y.shape[-1] // recorder.emission.degree_max, dim=1
    )
    y_int = y * keep.unsqueeze(1)
    out["pr_emitted"] = participation_ratio(y)
    out["pr_emitted_centred"] = participation_ratio(y, centred=True)
    out["pr_emitted_interior"] = participation_ratio(y_int)
    out["pr_emitted_interior_centred"] = participation_ratio(y_int, centred=True)
    out["mean_share_state"] = _moment_share(h)
    out["mean_share_emitted"] = _moment_share(y)
    out["energy_state"] = h.double().pow(2).sum(dim=(1, 2)) / ticks
    out["energy_emitted"] = y.double().pow(2).sum(dim=(1, 2)) / ticks
    if null_seed is not None:
        plain, scaled = haar_maps(agent, recorder.emission, null_seed)
        for name, w in (("haar", plain), ("haar_scaled", scaled)):
            y_null = torch.einsum("ctn,cdn->ctd", h.double(), w)
            out[f"pr_emitted_{name}"] = participation_ratio(y_null)
            out[f"pr_emitted_{name}_centred"] = participation_ratio(y_null, centred=True)

    rows = []
    for i, cid in enumerate(recorder.emission.cells):
        row = {"cell": int(cid), **{k: float(v[i]) for k, v in out.items()}}
        row.update(attrs["per_cell"][i])
        for tag, cen in (("", ""), ("_centred", "_centred")):
            st = row[f"pr_total{cen}"]
            row[f"ratio_emitted_over_state{cen}"] = (
                row[f"pr_emitted{cen}"] / st if st > 0 else float("nan")
            )
            row[f"ratio_readable_over_state{cen}"] = (
                row[f"pr_readable{cen}"] / st if st > 0 else float("nan")
            )
            rd = row[f"pr_readable{cen}"]
            row[f"ratio_emitted_over_readable{cen}"] = (
                row[f"pr_emitted_interior{cen}"] / rd if rd > 0 else float("nan")
            )
            for name in ("haar", "haar_scaled"):
                key = f"pr_emitted_{name}{cen}"
                if key in row:
                    row[f"ratio_learned_over_{name}{cen}"] = (
                        row[f"pr_emitted{cen}"] / row[key] if row[key] > 0 else float("nan")
                    )
        rows.append(row)
    return {"window_ticks": int(ticks), "cells": rows}


def _q(values) -> dict:
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"n": 0, "median": float("nan")}
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p10": float(np.quantile(a, 0.10)),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p90": float(np.quantile(a, 0.90)),
        "max": float(a.max()),
    }


#: The columns whose distribution is worth a quantile block at every checkpoint.
SUMMARISED = [
    "pr_total",
    "pr_total_centred",
    "pr_private",
    "pr_private_centred",
    "pr_exposed",
    "pr_exposed_centred",
    "pr_interior",
    "pr_interior_centred",
    "pr_readable",
    "pr_readable_centred",
    "pr_emitted",
    "pr_emitted_centred",
    "pr_emitted_interior",
    "pr_emitted_interior_centred",
    "ratio_emitted_over_state",
    "ratio_emitted_over_state_centred",
    "ratio_readable_over_state",
    "ratio_readable_over_state_centred",
    "ratio_emitted_over_readable",
    "ratio_emitted_over_readable_centred",
    "pr_emitted_haar",
    "pr_emitted_haar_centred",
    "pr_emitted_haar_scaled",
    "pr_emitted_haar_scaled_centred",
    "ratio_learned_over_haar",
    "ratio_learned_over_haar_centred",
    "ratio_learned_over_haar_scaled",
    "ratio_learned_over_haar_scaled_centred",
    "mean_share_state",
    "mean_share_emitted",
    "energy_share_private",
    "energy_share_interior",
    "variance_share_private",
    "variance_share_interior",
]

#: Item 3: does the answer vary by level, degree, or lane width?
GROUPED = [
    "pr_total_centred",
    "pr_emitted_centred",
    "ratio_emitted_over_state_centred",
    "ratio_emitted_over_readable_centred",
]


def summarise(rows: list[dict]) -> dict:
    # The null columns are absent unless `--null` asked for them.
    present = [k for k in SUMMARISED if k in rows[0]]
    out = {"quantiles": {k: _q([r[k] for r in rows]) for k in present}}
    by: dict[str, dict] = {}
    for axis in ("level", "degree", "sum_interior_m"):
        buckets = collections.defaultdict(list)
        for r in rows:
            buckets[r[axis]].append(r)
        by[axis] = {
            str(key): {
                "cells": len(group),
                **{k: float(np.median([g[k] for g in group])) for k in GROUPED},
            }
            for key, group in sorted(buckets.items())
        }
    out["by"] = by
    return out


def cell_attrs(dome) -> dict:
    """Level, degree, lane widths and mask sizes, in `dome.predicting` row order."""
    per_cell = []
    for cid in dome.predicting:
        widths, interior_widths = [], []
        for eid in dome.incident[cid]:
            edge = dome.edges[eid]
            widths.append(int(edge.m))
            if edge.kind is EdgeKind.INTERIOR:
                interior_widths.append(int(edge.m))
        permitted = int(dome._permitted[cid])
        per_cell.append(
            {
                "level": int(dome.cells[cid].index.level),
                "degree": len(widths),
                "interior_degree": len(interior_widths),
                "sum_m": int(sum(widths)),
                "sum_interior_m": int(sum(interior_widths)),
                "max_m": max(widths) if widths else 0,
                "k_v": permitted,
                "private_dim": int(dome.shape.n) - permitted,
            }
        )
    return {"per_cell": per_cell}


# -- the arm -------------------------------------------------------------------


def run_seed(arm: str, seed: int, ticks: int, out: Path, null: bool = False) -> dict:
    started = time.time()
    inflight = out.with_suffix(".inflight.json")
    env, agent = arms_mod.build_arm(arm, seed)
    try:
        dome = agent.dome
        _, reserve_p = arms_mod.ARMS[arm]
        emission = Emission(dome)
        attrs = cell_attrs(dome)
        record = {
            "issue": 568,
            "arm": arm,
            "reserve_p": reserve_p,
            "seed": seed,
            "ticks": ticks,
            "window": WINDOW,
            "null": null,
            "n": int(dome.shape.n),
            "predicting_cells": len(emission.cells),
            "degree_max": emission.degree_max,
            "m_max": int(agent.sheaf.maps.edge_width),
            "question": (
                "For the same cell over the same window: effective rank of its "
                "internal state against effective rank of what it puts on its edges. "
                "Does a cell have little to say, or can it not say what it has?"
            ),
            "privacy": arms_mod.privacy_read(dome, reserve_p),
            "attrs": attrs["per_cell"],
            "checkpoints": [],
        }

        bias = PredictionRule(agent.sheaf)
        transport = TransportRule(agent.sheaf)
        rec = Recorder(agent, emission)

        ladder = [c for c in CHECKPOINTS if c <= ticks]
        if ticks not in ladder:
            ladder.append(ticks)
        seen = 0
        for target in ladder:
            span = target - seen
            rec.reset()
            remaining = span
            for _ in t0.run_ticks(agent, span, seed=seed + seen):
                remaining -= 1
                if remaining < WINDOW:
                    rec.observe()
                bias.step()
                if agent.sheaf.ticks > 1:
                    transport.step()
            seen = target
            entry = {
                "ticks": target,
                **read(agent, rec, attrs, null_seed=seed if null else None),
            }
            entry.update(summarise(entry["cells"]))
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            record["checkpoints"].append(entry)
            inflight.write_text(json.dumps(record, indent=1))
            q = entry["quantiles"]
            print(
                f"  {arm} s{seed} @{target:>6}: "
                f"state {q['pr_total_centred']['median']:.3f} "
                f"(unc {q['pr_total']['median']:.3f}) "
                f"| readable {q['pr_readable_centred']['median']:.3f} "
                f"| emitted {q['pr_emitted_centred']['median']:.3f} "
                f"(unc {q['pr_emitted']['median']:.3f}) "
                f"| ratio {q['ratio_emitted_over_state_centred']['median']:.3f} "
                + (
                    f"| haar {q['pr_emitted_haar_centred']['median']:.3f} "
                    f"/ scaled {q['pr_emitted_haar_scaled_centred']['median']:.3f} "
                    if null
                    else ""
                )
                + f"| {entry['elapsed_minutes']:.1f}m",
                flush=True,
            )

        record["minutes"] = (time.time() - started) / 60.0
        out.write_text(json.dumps(record, indent=1))
        if inflight.exists():
            inflight.unlink()
        return record
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["reserve", "reserve_p16"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ticks", type=int, default=20_000)
    ap.add_argument(
        "--null",
        action="store_true",
        help="also read shape/mask/band-matched random maps on the same window",
    )
    args = ap.parse_args()

    for arm in args.arms:
        tag = "null" if args.null else "state-emitted"
        out = _HERE / f"568-{tag}-{arm}-seed{args.seed}-{args.ticks}.json"
        if out.exists():
            print(f"skip {arm}: {out.name} exists", flush=True)
            continue
        print(f"== {arm} seed {args.seed} to {args.ticks} ==", flush=True)
        run_seed(arm, args.seed, args.ticks, out, null=args.null)


if __name__ == "__main__":
    main()
