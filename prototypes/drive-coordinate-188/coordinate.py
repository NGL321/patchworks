"""#188 Q4: which coordinate of the apex stalk does each drive edge collapse onto?

[#356](https://github.com/NGL321/patchworks/issues/356) found that the transport
rule's sparsity pressure — `normalised_l1(F) = ‖F‖₁ / (√p ‖F‖_F)`, whose strict
global minimum is a **single nonzero entry** — drives every map toward rank 1,
and that it reaches the pinned maps too, because the rule steps every pair and
only the projection knows about pinning, restoring *scale* alone.

A drive edge has no rank to lose: the drive stalk is one-dimensional, so the map
out of it is rank 1 by dimension whatever `m_e` is. What the pressure can still
do there is decide **where** the apex-side map points, and the sharp form of
[#183](https://github.com/NGL321/patchworks/issues/183)'s finding is that it
points at a *coordinate* rather than at a learned direction.

So this reads the eight drive edges' apex-side maps `F_apex` along the teaching
schedule and asks the two things #137's undifferentiated-apex trigger needs:

* **How concentrated is each map** — the fraction of its squared mass in its
  single largest apex-stalk coordinate, and Hoyer's `h = ‖F‖₁/(√p‖F‖_F)`, the
  pressure's own quantity, which runs `1/√p` for a fully concentrated map to `1`
  for a flat one. Smaller is sparser.
* **Do the eight land on the same coordinate or on different ones.** The same
  coordinate is a mechanism-level *undifferentiated apex*: eight cells pulled
  along one shared axis. Eight different ones is differentiation by coordinate —
  differentiation without a learned direction, which is a weaker thing than the
  transport rule building a channel, and is worth naming separately.

Interior edges incident on the same apex cells are read alongside as the
control, so the drive edge's concentration is reported against what the pressure
does everywhere rather than in isolation.

Read per-cell, never as a graph-wide average: #127's Notes forbid the aggregate
that cost the map a 1e14 phantom deficit, and an eight-cell effect is exactly
the size that drowns in one.

Checkpoints follow #356's own table (construction, ~3k, 30k) with two extra
points so the schedule's knee is not read off an endpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
for extra in (str(_ROOT / "src"), str(_ROOT / "benchmarks")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from patchworks.graph import CellKind, EdgeKind  # noqa: E402
from patchworks.restriction import pair_index  # noqa: E402
from untrained_fixed_point import build, teaching  # noqa: E402

CHECKPOINTS = (1_000, 3_000, 10_000, 30_000)


def read_map(sheaf, edge, side):
    """`F` for one edge endpoint, cropped to the mask's live block."""
    pair = pair_index(edge.id, side)
    cell = (edge.u, edge.v)[side]
    stalk = int(sheaf.maps.support[pair].any(dim=0).sum())
    return sheaf.maps.maps[pair, : edge.m, :stalk].detach().clone(), cell


def concentration(F: torch.Tensor) -> tuple[int, float, float]:
    """`(top coordinate, its share of squared mass, Hoyer's h)`.

    The coordinate is the *column* — a direction of the sender's node stalk —
    because that is what the map reads out of the cell it belongs to.
    """
    column_energy = F.pow(2).sum(dim=0)
    total = float(column_energy.sum())
    top = int(column_energy.argmax())
    share = float(column_energy[top]) / total if total > 0 else 0.0
    p = F.numel()
    h = float(F.abs().sum() / (F.norm().clamp_min(1e-30) * p**0.5))
    return top, share, h


def report(sheaf, drive_edges, control_edges, ticks: int) -> None:
    print(f"\n=== {ticks} ticks ===", flush=True)
    rows = []
    for edge, side in drive_edges:
        F, cell = read_map(sheaf, edge, side)
        top, share, h = concentration(F)
        rows.append((cell, top, share, h))
    print("  drive edges, apex-side map F_apex:")
    print("    apex cell   top coord   share of mass   Hoyer h")
    for cell, top, share, h in rows:
        print(f"    {cell:>9}   {top:>9}   {share:>13.4f}   {h:>7.4f}")
    coords = [top for _, top, _, _ in rows]
    distinct = len(set(coords))
    note = "  (UNDIFFERENTIATED: one shared axis)" if distinct == 1 else ""
    print(
        f"    -> {distinct} distinct coordinate(s) across {len(coords)} apex cells{note}"
    )
    shares = torch.tensor([s for _, _, s, _ in rows])
    hs = torch.tensor([h for _, _, _, h in rows])
    print(
        f"    mean share {float(shares.mean()):.4f}   "
        f"mean Hoyer h {float(hs.mean()):.4f}"
    )

    cshare, ch = [], []
    for edge, side in control_edges:
        F, _ = read_map(sheaf, edge, side)
        _, share, h = concentration(F)
        cshare.append(share)
        ch.append(h)
    if cshare:
        print(
            f"  control: {len(cshare)} interior maps on the same apex cells — "
            f"mean share {sum(cshare) / len(cshare):.4f}, "
            f"mean Hoyer h {sum(ch) / len(ch):.4f}",
            flush=True,
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dome", default="full", choices=("small", "full"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ticks", type=int, default=30_000)
    args = parser.parse_args(argv)

    env, agent = build(args.dome, args.split, args.seed)
    try:
        dome = agent.sheaf.dome
        drive_id = next(c.id for c in dome.cells if c.kind == CellKind.DRIVE)
        drive_edges, apex_cells = [], set()
        for edge in dome.edges:
            if edge.kind is not EdgeKind.DRIVE:
                continue
            side = 1 if edge.u == drive_id else 0
            drive_edges.append((edge, side))
            apex_cells.add((edge.u, edge.v)[side])
        control_edges = [
            (edge, side)
            for edge in dome.edges
            if edge.kind is not EdgeKind.DRIVE
            for side, cid in enumerate((edge.u, edge.v))
            if cid in apex_cells
        ]

        stalk = dome.cells[next(iter(apex_cells))].stalk
        print(
            f"{args.dome} dome, split {args.split!r}, seed {args.seed}: "
            f"{len(drive_edges)} drive edges into {len(apex_cells)} apex cells, "
            f"{len(control_edges)} interior maps on those cells as control"
        )
        print(f"drive edge m = {drive_edges[0][0].m}, apex stalk n = {stalk}")

        report(agent.sheaf, drive_edges, control_edges, 0)
        seen = 0
        for _ in teaching(agent, args.ticks, args.seed):
            seen += 1
            if seen in CHECKPOINTS:
                report(agent.sheaf, drive_edges, control_edges, seen)
    finally:
        env.close()


if __name__ == "__main__":
    main()
