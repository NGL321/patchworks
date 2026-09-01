"""#183: does the drive's assertion reach, read along the channel?

#120 read the drive with `sensitivity`: set the drive boundary cell to a
different constant, hold the world still, and print the largest change each
level's node stalks show. That table is where *"the drive's assertion dies two
levels from the apex"* comes from, and it has been quoted since as *the drive
does not reach*.

#142 then showed that the taper's whole per-hop budget had been read off
**isotropic** `randn` nudges against **near-rank-1** maps, so ~31/32 of every
probe was spent on directions the maps structurally discard. Read along the
channel the hop is ~184x what #120 reported. #142 re-read the taper and **did
not re-read the drive**, which is what this script is for.

The drive is not the taper, and the difference matters twice:

1. **There is no direction to choose at the drive's own hop.** The drive stalk
   is one-dimensional and every drive edge is `m = 1`, so the map out of it is
   a scalar and the direction it deposits in an apex cell is whatever
   `F_apexᵀ` is. Nothing was averaged away there, and no re-reading can improve
   it. That hop is *computed*, not probed.
2. **#120's drive read was never a random nudge.** It moved the real assertion
   and watched the real response, so the isotropy that cost the taper 184x does
   not apply to it in the way it applied there.

What *does* apply is the other half of #142's complaint — **read the gain
rather than bounding it**. `sensitivity` reports a **threshold crossing**, not
a gain: it prints an absolute displacement in float32 against node stalks of
order 1-10, and its own footer says anything at 1e-6 is the representation's
floor. A channel whose per-hop gain is 5e-3 leaves a unit assertion under that
floor after three hops **whatever the gain is**, so "dies two levels from the
apex" is a statement about float32 and only incidentally about the graph.

So this script chains the hop the way #142's §3 chained the taper's — two
message-passing phases per hop, because a stalk moves the broadcast in one
phase and the broadcast moves the far stalk in the next through the `t-1`
delay `02-tick-semantics.md` fixes — and **renormalises at every hop**, so the
measured quantity is a product of per-hop gains that no amount of depth can
push under the representation's floor. Three readings per hop, on the same
chain:

* **carried** — the direction the drive actually deposits, propagated. The
  true answer, and the one #183 asks for.
* **best** — the top right-singular vector of the sender's own outgoing map.
  #142's ceiling: what the channel would be worth if alignment across the edge
  were perfect.
* **isotropic** — a `randn` direction. #120's equivalent reading, and the
  control that says how much of any gap is the instrument.

`body` is read along the carried direction too, by perturbing a cell's stalk
and running the inference phase, which is #142's pre-registered falsification 3
(*"the body's aligned `sigma_max`... the one link where #140's ceiling was
substituted for a measurement"*) asked for the taper and answered here for the
drive's own path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
for extra in (str(_ROOT / "src"), str(_ROOT / "benchmarks")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from patchworks.agent import DRIVE_ASSERTION, run  # noqa: E402
from patchworks.graph import EdgeKind  # noqa: E402
from patchworks.restriction import pair_index  # noqa: E402
from untrained_fixed_point import build, restore, snapshot, taught  # noqa: E402


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-12)


def transport(sheaf, base, edge, side, direction, epsilon):
    """One hop's two maps, from a sender's stalk to the receiver's, along `direction`.

    Two phases, read as two separate single-phase measurements and multiplied,
    which is #142's §3 idiom: running them in sequence would let the second
    phase see the first phase's edit to the *sender's own* stalk as well, and
    the product of the two half-hops is the linear quantity actually wanted.

    Returns `(gain, arrived_direction)`, or `(0.0, None)` where nothing survives.
    """
    sender = (edge.u, edge.v)[side]
    receiver = (edge.v, edge.u)[side]
    pair = pair_index(edge.id, side)
    where = sheaf.layout.slice(receiver)

    restore(sheaf, base)
    sheaf.message_passing_phase()
    quiet_stalks = sheaf.stalks.clone()
    quiet_broadcast = sheaf.broadcast.clone()

    # stalk -> broadcast
    restore(sheaf, base)
    with torch.no_grad():
        sheaf.stalks[sheaf.layout.slice(sender)] += direction * epsilon
    sheaf.message_passing_phase()
    left = sheaf.broadcast[pair] - quiet_broadcast[pair]
    scale = float(left.norm()) / epsilon
    if scale <= 0.0:
        return 0.0, None

    # broadcast -> the far stalk, injected along what the sender actually emits
    restore(sheaf, base)
    with torch.no_grad():
        sheaf.broadcast[pair, : edge.m] += unit(left[: edge.m]) * epsilon
    sheaf.message_passing_phase()
    moved = sheaf.stalks[where] - quiet_stalks[where]
    arrived = float(moved.norm()) / epsilon
    if arrived <= 0.0:
        return 0.0, None
    return scale * arrived, unit(moved)


def body_gain(sheaf, base, cell_id, direction, epsilon):
    """What the inference phase does to a change on a cell's own stalk.

    The prediction is written back onto the node stalk, so the cell's own slice
    after `inference_phase` *is* its prediction and no row lookup is needed.
    Read along `direction` rather than off a `randn`, which is the reading
    #142 left owed.
    """
    where = sheaf.layout.slice(cell_id)
    restore(sheaf, base)
    sheaf.inference_phase()
    quiet = sheaf.stalks[where].clone()

    restore(sheaf, base)
    with torch.no_grad():
        sheaf.stalks[where] += direction * epsilon
    sheaf.inference_phase()
    moved = sheaf.stalks[where] - quiet
    gain = float(moved.norm()) / epsilon
    if gain <= 0.0:
        return 0.0, None
    return gain, unit(moved)


def top_right_singular(sheaf, dome, edge, side):
    """The sender's own best input direction on this edge — #142's ceiling."""
    sender = (edge.u, edge.v)[side]
    width = dome.cells[sender].stalk
    block = sheaf.maps.maps[pair_index(edge.id, side), : edge.m, :width]
    return torch.linalg.svd(block, full_matrices=False)[2][0]


def descend(agent, base, start_cells, start_directions, epsilon, generator, label):
    """Chain down the taper from the apex, one level at a time.

    At each step every downward edge out of the current cell is measured and
    the **best** one is carried, which is the existence claim #142 pinned the
    destination to — *"there exist directions that reach the apex"*, not that
    every one does. The direction that arrives is renormalised before the next
    hop, so the product below is a product of gains and never a displacement
    fighting float32.
    """
    sheaf, dome = agent.sheaf, agent.dome
    rows = []
    cell, direction = None, None
    # Pick the apex cell the drive delivers most to, and start there.
    best = max(range(len(start_cells)), key=lambda i: start_directions[i][0])
    cell = start_cells[best]
    direction = start_directions[best][1]

    while True:
        level = dome.cells[cell].index.level
        options = []
        for e_id in dome.incident[cell]:
            edge = dome.edges[e_id]
            other = edge.other(cell)
            if dome.cells[other].index.level >= level:
                continue
            if edge.kind is EdgeKind.DRIVE:
                continue
            side = 0 if edge.u == cell else 1
            options.append((edge, side, other))
        if not options:
            break

        # The body first: the change on this cell's stalk becomes a prediction
        # before it is restricted onto any edge.
        b_gain, b_dir = body_gain(sheaf, base, cell, direction, epsilon)
        if b_dir is None:
            break

        picked = None
        for edge, side, other in options:
            carried, arrived_dir = transport(sheaf, base, edge, side, b_dir, epsilon)
            best_dir = top_right_singular(sheaf, dome, edge, side)
            ceiling, _ = transport(sheaf, base, edge, side, best_dir, epsilon)
            noise = torch.randn(b_dir.shape, generator=generator)
            iso, _ = transport(sheaf, base, edge, side, unit(noise), epsilon)
            row = (edge, side, other, carried, ceiling, iso, arrived_dir)
            if picked is None or carried > picked[3]:
                picked = row

        edge, side, other, carried, ceiling, iso, arrived_dir = picked
        rows.append(
            {
                "from": cell,
                "to": other,
                "level": level,
                "to_level": dome.cells[other].index.level,
                "body": b_gain,
                "carried": carried,
                "ceiling": ceiling,
                "isotropic": iso,
                "hop": b_gain * carried,
                "hop_ceiling": b_gain * ceiling,
                "hop_isotropic": b_gain * iso,
            }
        )
        if arrived_dir is None or dome.cells[other].is_boundary:
            break
        cell, direction = other, arrived_dir

    return rows


def level_cells(dome, level):
    return [c.id for c in dome.cells if c.index.level == level and not c.is_boundary]


def coherent_descend(agent, base, epsilon, generator, apex_level):
    """The same descent, but carrying the **whole level's** displacement at once.

    The single-chain read above follows one cell to one neighbour, which is
    what the taper's own instrument does — and for the taper that is the right
    shape, because a rim perturbation starts at one patch. **The drive does
    not.** It attaches to the apex *entire*: 8 edges, one scalar, every apex
    cell driven with the same sign at the same instant. So its signal arrives
    at level 6 as the sum of 8 coherent contributions, and a probe that
    follows one edge at a time cannot see whether those add or cancel — it
    discards exactly the structure that makes a drive a drive.

    So: inject the whole displacement field over level `l`, run the inference
    phase and two message-passing phases, and read what lands on level `l - 1`
    entire. Renormalise, repeat. The field is zeroed outside the level being
    driven so that each row is one level's transfer and not an accumulation,
    which keeps the product below a product of per-hop gains.

    The isotropic control is the same field with random per-cell directions of
    the same per-cell magnitudes, which is the reading that says how much of
    any gain is coherence rather than direction.
    """
    sheaf, dome = agent.sheaf, agent.dome
    rows = []

    def transfer(field, source, target):
        """One level's whole transfer: inference, then the hop's two phases."""
        restore(sheaf, base)
        sheaf.inference_phase()
        sheaf.message_passing_phase()
        sheaf.message_passing_phase()
        quiet = sheaf.stalks.clone()

        restore(sheaf, base)
        with torch.no_grad():
            for cell_id, vec in zip(source, field):
                sheaf.stalks[sheaf.layout.slice(cell_id)] += vec * epsilon
        sheaf.inference_phase()
        sheaf.message_passing_phase()
        sheaf.message_passing_phase()
        landed = [sheaf.stalks[sheaf.layout.slice(c)] - quiet[sheaf.layout.slice(c)] for c in target]
        norm = float(torch.cat(landed).norm()) / epsilon
        return norm, landed

    # The drive's own hop, coherent by construction: one scalar, 8 edges.
    apex = level_cells(dome, apex_level)
    restore(sheaf, base)
    sheaf.message_passing_phase()
    sheaf.message_passing_phase()
    quiet = sheaf.stalks.clone()
    restore(sheaf, base)
    with torch.no_grad():
        sheaf.stalks[agent._drive_slice] += epsilon
    sheaf.message_passing_phase()
    sheaf.message_passing_phase()
    field = [sheaf.stalks[sheaf.layout.slice(c)] - quiet[sheaf.layout.slice(c)] for c in apex]
    gain0 = float(torch.cat(field).norm()) / epsilon
    rows.append({"level": "drive", "to_level": apex_level, "gain": gain0, "iso": gain0})

    source, level = apex, apex_level
    while True:
        target = level_cells(dome, level - 1)
        if not target:
            break
        scale = torch.cat(field).norm().clamp_min(1e-12)
        carried = [v / scale for v in field]
        gain, landed = transfer(carried, source, target)

        noise = []
        for v in carried:
            r = torch.randn(v.shape, generator=generator)
            noise.append(r / r.norm().clamp_min(1e-12) * v.norm())
        iso, _ = transfer(noise, source, target)

        rows.append({"level": level, "to_level": level - 1, "gain": gain, "iso": iso})
        if gain <= 0.0:
            break
        field, source, level = landed, target, level - 1

    return rows


def drive_hop(agent, base, epsilon):
    """The drive's own hop into the apex — computed, because it has no freedom.

    The drive's node stalk is one number and every drive edge is `m = 1`, so
    the map out of it is a scalar and the direction deposited in an apex cell
    is `F_apexᵀ` and nothing else. There is no isotropic reading of this hop to
    correct and no better direction to find; it is what it is, for every
    reading of the channel.
    """
    sheaf, dome = agent.sheaf, agent.dome
    out = []
    direction = torch.ones(1)
    for e_id in dome.incident[agent.drive_cell]:
        edge = dome.edges[e_id]
        side = 0 if edge.u == agent.drive_cell else 1
        gain, arrived = transport(sheaf, base, edge, side, direction, epsilon)
        out.append((gain, arrived, edge.other(agent.drive_cell)))
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dome", default="full", choices=("small", "full"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--learn",
        type=int,
        default=0,
        help="teach for this many ticks with both rules on, instead of --ticks untrained",
    )
    args = parser.parse_args(argv)

    env, agent = build(args.dome, args.split, args.seed)
    try:
        if args.learn:
            taught(agent, args.learn, args.seed)
            how = f"{args.learn} ticks with both rules on"
        else:
            for _ in run(agent, args.ticks, seed=args.seed):
                pass
            how = f"{args.ticks} ticks, untrained"
        sheaf = agent.sheaf
        base = snapshot(sheaf)
        generator = torch.Generator().manual_seed(args.seed)

        print(f"\n{args.dome} dome, split {args.split!r}, seed {args.seed}: {how}")
        print(f"  drive assertion {DRIVE_ASSERTION:g}, nudge {args.epsilon:g}")

        # -- hop 0: the drive into the apex -----------------------------
        landed = drive_hop(agent, base, args.epsilon)
        gains = np.array([g for g, _, _ in landed])
        print(
            f"\n  the drive's own hop, into {len(landed)} apex cells "
            f"(m = 1, scalar stalk, no direction to choose):"
        )
        print(
            f"    gain per apex cell: mean {gains.mean():.4f}  "
            f"min {gains.min():.4f}  max {gains.max():.4f}"
        )
        print(
            f"    a {DRIVE_ASSERTION:g} -> 10.0 step therefore lands "
            f"{gains.max() * 9.0:.4f} on the best apex cell's stalk"
        )

        # -- the descent ------------------------------------------------
        rows = descend(
            agent,
            base,
            [c for _, _, c in landed],
            [(g, d) for g, d, _ in landed],
            args.epsilon,
            generator,
            "carried",
        )

        print(
            "\n  chained down the taper from that apex cell, renormalised at every "
            "hop.\n  `carried` is the direction the drive actually deposits; "
            "`best` is the sender's\n  own top singular direction (#142's ceiling); "
            "`iso` is a randn nudge (#120's\n  reading). `hop` is body x carried."
        )
        print(
            f"  {'hop':>3s}  {'L':>7s}  {'body':>8s}  {'carried':>9s}  "
            f"{'best':>9s}  {'iso':>9s}  {'hop':>9s}  {'cum':>10s}"
        )
        cumulative = float(gains.max())
        cum_iso = float(gains.max())
        print(
            f"  {0:>3d}  {'drive->7':>7s}  {'-':>8s}  {gains.max():9.4f}  "
            f"{gains.max():9.4f}  {gains.max():9.4f}  {gains.max():9.4f}  "
            f"{cumulative:10.3e}"
        )
        for i, r in enumerate(rows, start=1):
            cumulative *= r["hop"]
            cum_iso *= r["hop_isotropic"]
            print(
                f"  {i:>3d}  {f'{r['level']}->{r['to_level']}':>7s}  "
                f"{r['body']:8.4f}  {r['carried']:9.4f}  {r['ceiling']:9.4f}  "
                f"{r['isotropic']:9.4f}  {r['hop']:9.4f}  {cumulative:10.3e}"
            )

        if rows:
            carried = np.array([r["hop"] for r in rows])
            iso = np.array([r["hop_isotropic"] for r in rows])
            ceiling = np.array([r["hop_ceiling"] for r in rows])
            print(
                f"\n  per-hop mean: carried {carried.mean():.4f}, "
                f"isotropic {iso.mean():.4f}, ceiling {ceiling.mean():.4f}"
            )
            print(
                f"  reading along the channel is worth "
                f"{carried.mean() / max(iso.mean(), 1e-30):.2f}x the isotropic "
                f"reading, per hop;\n  the channel is "
                f"{carried.mean() / max(ceiling.mean(), 1e-30):.1%} of what "
                f"perfect cross-edge alignment would give."
            )
            print(
                f"\n  cumulative, drive to the rim: carried {cumulative:.3e}, "
                f"isotropic {cum_iso:.3e}"
            )
            step = 9.0
            print(
                f"  a {DRIVE_ASSERTION:g} -> 10.0 assertion therefore displaces the "
                f"far end by {cumulative * step:.3e},\n  against float32's ~1e-6 "
                f"floor on node stalks of order 1-10: "
                f"{'ABOVE' if cumulative * step > 1e-6 else 'BELOW'} it."
            )
            floor = 1e-6 / step
            reach, acc = 0, float(gains.max())
            for r in rows:
                if acc < floor:
                    break
                acc *= r["hop"]
                reach += 1
            print(
                f"  levels the assertion is *readable* at in float32: {reach} "
                f"below the apex.\n  levels it has a non-zero gain to: "
                f"{len(rows)} — every one measured, none zero."
            )

        # -- the coherent read ------------------------------------------
        apex_level = max(
            agent.dome.cells[c].index.level
            for c in range(len(agent.dome.cells))
            if not agent.dome.cells[c].is_boundary
        )
        coherent = coherent_descend(agent, base, args.epsilon, generator, apex_level)
        print(
            "\n  the same descent carrying the whole level at once, which is the "
            "shape the drive\n  actually has: 8 edges driven with one sign at one "
            "instant. `iso` randomises each\n  cell's direction while keeping its "
            "magnitude, so the ratio is coherence alone."
        )
        print(
            f"  {'hop':>3s}  {'L':>9s}  {'gain':>9s}  {'iso':>9s}  "
            f"{'worth':>7s}  {'cum':>10s}"
        )
        cum = 1.0
        for i, r in enumerate(coherent):
            cum *= r["gain"]
            worth = r["gain"] / max(r["iso"], 1e-30)
            print(
                f"  {i:>3d}  {f'{r['level']}->{r['to_level']}':>9s}  "
                f"{r['gain']:9.4f}  {r['iso']:9.4f}  {worth:7.2f}x  {cum:10.3e}"
            )
        interior = [r for r in coherent if r["level"] != "drive"]
        if interior:
            g = np.array([r["gain"] for r in interior])
            i_ = np.array([r["iso"] for r in interior])
            print(
                f"\n  per-hop mean, coherent: {g.mean():.4f} against "
                f"{i_.mean():.4f} incoherent — coherence is worth "
                f"{g.mean() / max(i_.mean(), 1e-30):.2f}x per hop."
            )
            print(
                f"  cumulative drive-to-rim, coherent: {cum:.3e}; a "
                f"{DRIVE_ASSERTION:g} -> 10.0 step displaces the rim by "
                f"{cum * 9.0:.3e}."
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
