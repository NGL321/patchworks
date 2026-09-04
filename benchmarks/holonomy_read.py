"""Does the transport rule move cross-edge alignment toward the identity, or away? (#453)

[#315](https://github.com/NGL321/patchworks/issues/315)'s instrument, run with
[#411](https://github.com/NGL321/patchworks/issues/411)'s sign::

    python benchmarks/holonomy_read.py read                 # the full read, ~3.5 h
    python benchmarks/holonomy_read.py read --learn 2000    # a fast shape check
    python benchmarks/holonomy_read.py read --dome small --learn 500 --no-file

**Enumerate the graph's independent cycles, compose the restriction maps around
each, and report the composed operator's departure from the identity.** That is
#315's mechanism unchanged; what is not #315's is the polarity.
`docs/motivating-image.md`'s *non-abelian* was a dictation artifact, order-free
composition **is** path-independence, and #411 ruled accordingly: **departure of
holonomy from the identity is the failure measure, not the prize.**

**What this reads is the half [ADR-0032](../docs/adr/0032-the-maps-learn-isometric-transport-and-a-spectral-floor-expresses-it.md)
declines to claim.** The ADR splits its target in two. *Metric agreement* --
lengths carried undistorted, holonomy inside `O(m)` -- is enforceable per map and
the spectral floor buys it: [#435](https://github.com/NGL321/patchworks/issues/435)
reads effective rank 4.000 = 1.000 of `m` on the floored maps. *Identification
agreement* -- which generator is which, route-free, holonomy `= I` -- is what no
cell can compute, because a cycle is not incident to one cell, and the ADR
asserts it is bought anyway *"by the objective on a non-degenerate trajectory"*.
[#437](https://github.com/NGL321/patchworks/issues/437) struck that assertion as
measured-against and [#454](https://github.com/NGL321/patchworks/issues/454) is
the problem minted from it. **This rig is #454's cutoff**, so the reading has to
be legible as an answer to *does departure from identity fall under the transport
rule?* and not only as a table.

## The two halves are separated, because the ADR separates them

Each cycle yields one holonomy operator `H`, and it is read three ways
(:func:`departures`):

- **`sigma_max`** -- what the loop carries at all. Not the question, and printed
  because a departure measure on an operator that has decayed to `1e-9` is
  arithmetic about the decay.
- **flatness `sigma_min/sigma_max`** -- **metric agreement around the cycle**.
  1 says `H` is orthogonal up to scale, i.e. inside `O(m)`; this is the half the
  floor buys per map, read where the ADR says a cell cannot enforce it.
- **identification departure `‖UVᵀ − I‖_F / sqrt(2m)`** -- **identification
  agreement**, and the number this ticket turns on. `UVᵀ` is `H`'s orthogonal
  polar factor: the rotation the loop applies once its scale and its distortion
  are divided out. **0 is holonomy at the identity, 1 is chance, `sqrt(2)` is
  maximally opposed** -- for `Q` Haar-uniform on `O(m)`, `E‖Q − I‖_F² = 2m`
  exactly, which is what puts the null at 1 analytically as well as empirically
  (the flat-and-independent arm below reads it off a draw).

- **channel return `|<u_1, v_1>|`** -- the same question asked only of the one
  direction the loop actually transmits. [#454](https://github.com/NGL321/patchworks/issues/454)
  is the finding that composed transport is rank 1, and on a near-rank-1
  holonomy the whole-operator columns above are dominated by directions carrying
  nothing; this says whether the channel comes back to *itself*. 1 is returned,
  0 is orthogonal to where it started, and the chance arm reads its null.

**All of these are gauge-invariant, and that is what makes `= I` a statement.**
Re-choosing an edge stalk's frame sends `F_{v,e} -> R_e F_{v,e}` at *both*
endpoints, so every hop goes `H_v -> R_out H_v R_in^T` and the holonomy around a
closed cycle goes `H -> R_e H R_e^T` -- a conjugation. Singular values and
`‖UVᵀ − I‖_F` are both conjugation-invariant, so neither column is reading the
arbitrary basis ADR-0010's gauge leaves free.

**The reconciliation gain is deliberately absent.** It is one positive scalar per
cell; it scales `H` and cancels exactly out of `UVᵀ`, so the identification
column is invariant to it and the flatness column is unaffected. Including it
would move only `sigma_max`, and amplitude is not what
[#242](https://github.com/NGL321/patchworks/issues/242) left the bar reading.

## The cycles

A **fundamental cycle basis** off a breadth-first spanning tree of the
**interior** subgraph: 150 predicting cells and 409 edges give **260**
independent cycles, against the 269 the whole graph carries. The nine that are
lost all route through a boundary cell, and a hop *through* a boundary cell is
not transport -- ADR-0016 has a boundary cell written or read and never both,
which is `floor_price.chain_paths`' and `alignment_read.collect`'s existing rule,
carried here unchanged rather than re-argued.

A cycle basis is not canonical: a different spanning tree gives different cycles.
BFS is chosen because it gives the *shortest* fundamental cycles, and short
cycles are where a per-map surface has the best chance of composing back to the
identity -- so the choice is the one that flatters the hypothesis under test. The
length distribution is printed, and every column is reported **per cycle, never a
graph-wide average** (#127's standing rule, #181's per-edge-not-per-level form):
an aggregate over 260 cycles would drown exactly the effect worth having, so what
is printed is the distribution, the split by cycle length, and the extremes by
name.

## The arms

- **trained, floored** -- the shipped surface, at **30k and 100k on one
  trajectory**. #178 has cost this map the 30k mistake four times, most recently
  on #435 where the control's own recovery changed what the floor's margin was
  worth, so both horizons are read and neither stands alone.
- **trained, unfloored** -- `RestrictionMaps._flatten` disabled and nothing else
  changed, `floor_price.without_floor`'s patch in
  `spectral_floor_read.build`'s form. The floor puts holonomy in `O(m)` per map,
  which changes what departure from `I` even means, so the floored reading needs
  it.
- **untrained, floored and unfloored** -- the constructor's draw, no rule steps.
  Free, so it is taken over eight seeds rather than one. **This is the baseline
  the question needs**: a departure measure with no baseline says nothing about
  direction, and direction is the whole ticket. **The floored half of it is
  `floor_price.flat_maps` exactly** -- an untrained floored draw *is*
  exactly-flat maps drawn independently -- so #436's chance null and this
  baseline are one object here and are run and printed once, under both names.

**One seed carries the trained arms and eight carry the free ones.** A 100k
trained trajectory is ~1.6 h on this box and the ticket asks for two of them;
three seeds would be a day. The trained arms are seed 42, stated as a limit
rather than discovered, and the spread that is affordable is published where it
is affordable.

**Which surface every number was taken on.** The map's 2026-09-04 standing rule:
every arm here is trained *in this process*, on `main` as of this run, and
nothing is differenced against stored JSON. ADR-0031 deleted the sparsity
pressure, so a number measured on the `lambda = 0.4` surface is a number about a
build that no longer exists.

## Boundaries

`docs/research/015-information-cohomology.md` firewalls the vocabulary. There is
**no relation** between information cohomology's `H1` and the `H1` of a cellular
sheaf on this graph -- different site, different coefficients, different
differential, no comparison map in the literature. Everything here is on the
**sheaf** side and is not evidence for the Baudot-Bennequin motivation.
`CONTEXT.md` puts *"topological invariant"* on an `_Avoid_` list and this module
observes it.

This rig **rules on no remedy**. [#396](https://github.com/NGL321/patchworks/issues/396)
is the alignment proposal ADR-0032 superseded and it stays superseded; running
#315's instrument is not adopting #315, which goes `adopted` when an ADR covers
it. Like every script here **it asserts nothing** and its exit code does not
move -- `benchmarks/run_reporting.py` states the rule, and rigs do not run in CI.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import deque

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "benchmarks"))

from patchworks.agent import Agent  # noqa: E402
from patchworks.graph import DEFAULT_SPEC, Dome, DomeSpec, build_graph  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from patchworks.restriction import RestrictionMaps, pair_index  # noqa: E402
from patchworks.sandbox import PlanarPushSandbox  # noqa: E402

import construction_grading as cg  # noqa: E402
from untrained_fixed_point import run as driven  # noqa: E402

sys.path.insert(0, str(_ROOT / "tests"))
from conftest import SMALL  # noqa: E402

IMAGE_SIZE = {DEFAULT_SPEC.patch_grid: 64, SMALL.patch_grid: 16}

#: Where the surface is read, `spectral_floor_read.CHECKPOINTS` unchanged: 30k
#: beside 100k on one trajectory, never instead of it (#178).
CHECKPOINTS = (30_000, 100_000)

#: How many draws the free arms get. Eight, `floor_price.control_section`'s count.
DRAWS = 8

#: Below this, a holonomy operator has not transported anything in its smallest
#: direction and its polar factor is arithmetic about the decay rather than about
#: alignment. Relative to the cycle's own top value, as
#: `floor_price.RELATIVE_ZERO` is, and reported rather than dropped.
RELATIVE_ZERO = 1e-12


# -- the cycles -------------------------------------------------------------


def interior_graph(dome: Dome) -> tuple[list[int], list[int]]:
    """The predicting cells, and the edges with a predicting cell at both ends.

    ADR-0016's rule, in `floor_price.chain_paths`' form: a boundary cell is
    written or read and never both, so a hop *through* one is not transport and
    a cycle that routes through one is not a loop the surface composes around.
    """
    interior = [cell.id for cell in dome.cells if not cell.is_boundary]
    inside = set(interior)
    edges = [edge.id for edge in dome.edges if edge.u in inside and edge.v in inside]
    return interior, edges


def _incidence(dome: Dome, interior, edge_ids):
    incident: dict[int, list[int]] = {cell: [] for cell in interior}
    for edge_id in edge_ids:
        edge = dome.edges[edge_id]
        incident[edge.u].append(edge_id)
        incident[edge.v].append(edge_id)
    return incident


def _components(interior, incident, far) -> int:
    seen: set[int] = set()
    count = 0
    for start in interior:
        if start in seen:
            continue
        count += 1
        seen.add(start)
        stack = [start]
        while stack:
            cell = stack.pop()
            for edge_id in incident[cell]:
                other = far(edge_id, cell)
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
    return count


def cycle_basis(dome: Dome) -> list[tuple[int, ...]]:
    """A fundamental cycle basis of the interior subgraph, as tuples of edge ids.

    Breadth-first spanning forest, then one cycle per non-tree edge: its two
    endpoints' tree routes to their meeting point, closed by the edge itself.
    `edges - cells + components` cycles, which is 260 on `DEFAULT_SPEC`, and the
    count is checked against the rank rather than assumed.

    **BFS rather than DFS on purpose.** A fundamental basis is not canonical and
    the tree chooses it; BFS gives the shortest cycles available this way, and
    short cycles are the ones a per-map surface has the best chance of composing
    back to the identity around. The choice therefore flatters the hypothesis
    under test, which is the direction an instrument should err in.
    """
    interior, edge_ids = interior_graph(dome)
    incident = _incidence(dome, interior, edge_ids)

    def far(edge_id: int, cell: int) -> int:
        edge = dome.edges[edge_id]
        return edge.v if edge.u == cell else edge.u

    parent: dict[int, tuple[int, int]] = {}
    depth: dict[int, int] = {}
    tree_edges: set[int] = set()
    for root in interior:
        if root in depth:
            continue
        depth[root] = 0
        queue = deque([root])
        while queue:
            cell = queue.popleft()
            for edge_id in incident[cell]:
                other = far(edge_id, cell)
                if other in depth:
                    continue
                depth[other] = depth[cell] + 1
                parent[other] = (cell, edge_id)
                tree_edges.add(edge_id)
                queue.append(other)

    def climb(cell: int, target: int) -> tuple[int, list[int]]:
        route: list[int] = []
        while depth[cell] > target:
            up, edge_id = parent[cell]
            route.append(edge_id)
            cell = up
        return cell, route

    cycles: list[tuple[int, ...]] = []
    for edge_id in sorted(set(edge_ids) - tree_edges):
        edge = dome.edges[edge_id]
        shallow = min(depth[edge.u], depth[edge.v])
        left, up_left = climb(edge.u, shallow)
        right, up_right = climb(edge.v, shallow)
        while left != right:
            parent_left, edge_left = parent[left]
            parent_right, edge_right = parent[right]
            up_left.append(edge_left)
            up_right.append(edge_right)
            left, right = parent_left, parent_right
        cycles.append(tuple(up_left + list(reversed(up_right)) + [edge_id]))

    rank = len(edge_ids) - len(interior) + _components(interior, incident, far)
    if len(cycles) != rank:
        raise ValueError(f"{len(cycles)} cycles against a cycle rank of {rank}")
    return cycles


def cycle_hops(dome: Dome, cycle: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """`(edge_in, cell, edge_out)` for every cell of a closed cycle.

    `construction_grading.hops_of` reads an open path; a cycle wraps, so the last
    edge hands back to the first and `k` edges give `k` hops rather than `k - 1`.
    Every cell here is interior by :func:`interior_graph`.
    """
    ring = tuple(cycle) + (cycle[0],)
    return cg.hops_of(dome, ring)


# -- the holonomy -----------------------------------------------------------


def hop_operator(dome: Dome, maps: RestrictionMaps, key) -> torch.Tensor:
    """`F_out . F_in^T` in float64. `floor_price.hop_operator` without the gain.

    The gain is one positive scalar per cell: it cancels out of the polar factor
    and moves only `sigma_max`, and amplitude is not the question (#242).
    """
    edge_in, cell, edge_out = key
    m_in, m_out = dome.edges[edge_in].m, dome.edges[edge_out].m
    with torch.no_grad():
        f_in = maps.maps[pair_index(edge_in, cg.side_of(dome, edge_in, cell))][:m_in]
        f_out = maps.maps[pair_index(edge_out, cg.side_of(dome, edge_out, cell))][
            :m_out
        ]
        return f_out.double() @ f_in.double().T


def holonomy(dome: Dome, maps: RestrictionMaps, cycle: tuple[int, ...]) -> torch.Tensor:
    """Transport around the cycle and back: the composed hops, in order.

    The composition is taken and *then* read, never a product of per-hop
    readings: that would assume every hop's carried subspace is the previous
    hop's, which is #233's composition gap and is false.
    """
    hops = cycle_hops(dome, cycle)
    composed = hop_operator(dome, maps, hops[0])
    for key in hops[1:]:
        composed = hop_operator(dome, maps, key) @ composed
    return composed


def departures(operator: torch.Tensor) -> dict[str, float]:
    """The three columns of one cycle: what it carries, and how far from `I`.

    `identification` is `‖UVᵀ − I‖_F / sqrt(2m)` off the SVD's polar factor --
    0 at the identity, 1 at chance, `sqrt(2)` maximally opposed -- and
    `flatness` is `sigma_min/sigma_max`, which is 1 when the holonomy is inside
    `O(m)` up to scale. Both are invariant under the edge-stalk gauge, which is
    the module docstring's conjugation argument.
    """
    u, values, vh = torch.linalg.svd(operator)
    sigma = values.numpy().astype(np.float64)
    top = float(sigma[0])
    m = int(operator.shape[0])
    out: dict[str, float] = {"sigma_max": top, "m": m}
    if not np.isfinite(top) or top <= 0:
        out["flatness"] = float("nan")
        out["identification"] = float("nan")
        out["degenerate"] = True
        return out
    out["flatness"] = float(sigma[-1] / top)
    rotation = (u @ vh).numpy().astype(np.float64)
    out["identification"] = float(
        np.linalg.norm(rotation - np.eye(m), ord="fro") / np.sqrt(2 * m)
    )
    # `|<u_1, v_1>|`: does the one direction the loop actually transmits come
    # back to *itself*? On a near-rank-1 holonomy the whole-operator columns are
    # dominated by directions carrying nothing, and this is the same question
    # asked only of the channel. 1 is the channel returned; 0 is orthogonal to
    # where it started. The chance arm below reads its null.
    out["channel_return"] = float(
        abs(
            np.dot(u.numpy().astype(np.float64)[:, 0], vh.numpy().astype(np.float64)[0])
        )
    )
    # A cycle whose smallest singular value is arithmetic noise has a polar
    # factor that is arithmetic noise in those directions. Said, not hidden.
    out["degenerate"] = bool(sigma[-1] / top < RELATIVE_ZERO)
    return out


def read_surface(dome: Dome, maps: RestrictionMaps, cycles) -> list[dict[str, float]]:
    """Every cycle's columns off one surface, in basis order."""
    rows = []
    for cycle in cycles:
        row = departures(holonomy(dome, maps, cycle))
        row["length"] = len(cycle)
        rows.append(row)
    return rows


# -- the arms ---------------------------------------------------------------


def build(spec: DomeSpec, split: str, seed: int, *, floor: bool):
    """One agent, with the floor on or off and nothing else changed.

    `spectral_floor_read.build`'s control arm, unchanged: `project` keeps its
    mask, its band and its cap, and the floor alone stops running, so a
    difference between the arms is attributable to the floor and to nothing else.
    """
    env = PlanarPushSandbox(split=split, image_size=IMAGE_SIZE[spec.patch_grid])
    agent = Agent(
        env, dome=build_graph(spec), generator=torch.Generator().manual_seed(seed)
    )
    if not floor:
        agent.sheaf.maps._flatten = lambda: None
    return env, agent


def flat_maps(dome: Dome, generator: torch.Generator) -> RestrictionMaps:
    """Exactly flat maps, drawn independently. `floor_price.flat_maps` unchanged.

    The floor's own projection on a fresh draw and nothing else: no training, no
    band, no cap. Every pair of adjacent carried subspaces is at a chance angle,
    so this is the empirical read of the analytic null at 1.
    """
    maps = RestrictionMaps(dome, generator=generator)
    with torch.no_grad():
        maps.maps.mul_(maps.support)
        maps._flatten()
    return maps


def flat_maps_of(dome: Dome):
    """:func:`flat_maps` bound to a dome, for the arm table."""
    return lambda generator: flat_maps(dome, generator)


def untrained_maps(dome: Dome, generator: torch.Generator, *, floor: bool):
    """The constructor's draw with the mask applied, floored or not.

    The baseline *trained against untrained* needs, and the only difference from
    :func:`flat_maps` is whether the floor ran -- which is what separates *the
    draw is misaligned* from *the floor made it misaligned*.

    **With the floor on, this is `flat_maps` and the same object**: an untrained
    floored draw *is* exactly-flat independently drawn maps, so the baseline the
    ticket asks for and #436's chance null coincide by construction on this
    surface. They are run as one arm and labelled as both rather than printed
    twice, and this function keeps the unfloored half.
    """
    maps = RestrictionMaps(dome, generator=generator)
    with torch.no_grad():
        maps.maps.mul_(maps.support)
        if floor:
            maps._flatten()
    return maps


def trained(spec, split, seed, checkpoints, cycles, floor, sink=None):
    """One driven trajectory, read at every checkpoint on the way past.

    `spectral_floor_read.measure`'s loop, unchanged -- `agent.tick()` then the
    rules, the prediction rule joining on the first tick and the transport rule
    on the second -- and its `sink`, for the same reason: this run is hours long
    on a shared box, and a 100k read that only writes at exit loses the 30k
    reading too when the process is killed at 70k.
    """
    env, agent = build(spec, split, seed, floor=floor)
    dome = agent.dome
    horizon = max(checkpoints)
    reads: dict[int, list[dict[str, float]]] = {}
    try:
        bias = PredictionRule(agent.sheaf)
        transport = TransportRule(agent.sheaf)
        for index, _outcome in enumerate(driven(agent, horizon, seed=seed)):
            bias.step()
            if agent.sheaf.ticks > 1:
                transport.step()
            if index + 1 in checkpoints:
                reads[index + 1] = read_surface(dome, agent.sheaf.maps, cycles)
                print(
                    f"    checkpoint {index + 1} read "
                    f"({'floored' if floor else 'unfloored'}, seed {seed})",
                    flush=True,
                )
                if sink is not None:
                    sink(index + 1, reads[index + 1])
    finally:
        env.close()
    return reads


# -- reporting --------------------------------------------------------------


def column(rows, key) -> np.ndarray:
    return np.asarray(
        [row[key] for row in rows if key in row and np.isfinite(np.float64(row[key]))],
        dtype=np.float64,
    )


def quantiles(values) -> str:
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if values.size == 0:
        return f"{'-':>11}{'-':>11}{'-':>11}"
    return (
        f"{np.percentile(values, 5):11.4g}"
        f"{np.median(values):11.4g}"
        f"{np.percentile(values, 95):11.4g}"
    )


def arm_section(label: str, rows) -> None:
    print(f"\n    {label:<40}{'5th':>11}{'median':>11}{'95th':>11}")
    for key, name in (
        ("identification", "identification departure (0=I, 1=chance)"),
        ("channel_return", "channel return |<u1,v1>| (1 = comes back)"),
        ("flatness", "flatness of H (1 = inside O(m))"),
        ("sigma_max", "sigma_max of H"),
    ):
        print(f"    {name:<40}{quantiles(column(rows, key))}")
    degenerate = sum(1 for row in rows if row.get("degenerate"))
    print(f"    {'cycles below the relative zero':<40}{degenerate:>11d} of {len(rows)}")


def by_length(label: str, rows) -> None:
    """Per cycle length, because a long loop composes more maps than a short one.

    Not a graph-wide average and not per level (#181): length is a property of
    the cycle, and the question is whether departure grows with how far the
    surface has to compose to get back.
    """
    lengths = sorted({int(row["length"]) for row in rows})
    print(f"\n    {label}: by cycle length")
    print(f"    {'length':>8}{'cycles':>8}{'median ident.':>15}{'median flatness':>17}")
    for length in lengths:
        block = [row for row in rows if int(row["length"]) == length]
        print(
            f"    {length:>8d}{len(block):>8d}"
            f"{np.median(column(block, 'identification')):>15.4f}"
            f"{np.median(column(block, 'flatness')):>17.4g}"
        )


def extremes(dome: Dome, cycles, rows, count: int = 3) -> None:
    """The cycles nearest and furthest from the identity, by name.

    #127's standing rule is per cycle and never a graph-wide average; the
    distribution above is the population, and this is what makes the ends of it
    something a later session can open.
    """
    order = sorted(
        (i for i in range(len(rows)) if np.isfinite(rows[i]["identification"])),
        key=lambda i: rows[i]["identification"],
    )
    for name, picks in (
        ("nearest the identity", order[:count]),
        ("furthest from it", order[-count:]),
    ):
        print(f"\n    {name}")
        for i in picks:
            cells = [hop[1] for hop in cycle_hops(dome, cycles[i])]
            route = " -> ".join(str(dome.cells[c].index) for c in cells[:4])
            print(
                f"      cycle {i:>4d} len {int(rows[i]['length'])}  "
                f"ident {rows[i]['identification']:.4f}  "
                f"flat {rows[i]['flatness']:.3g}  {route}"
                + (" ..." if len(cells) > 4 else "")
            )


def report(dome, cycles, arms: dict) -> None:
    interior, edges = interior_graph(dome)
    print("\n=== the cycles ===")
    print(
        f"    {len(dome.cells)} cells and {len(dome.edges)} edges; "
        f"cycle rank {len(dome.edges) - len(dome.cells) + 1}"
    )
    print(
        f"    interior only: {len(interior)} cells and {len(edges)} edges; "
        f"{len(cycles)} independent cycles"
    )
    print("    (a hop through a boundary cell is not transport -- ADR-0016)")
    lengths = np.asarray([len(c) for c in cycles])
    print(
        f"    cycle length: min {lengths.min()}, median {np.median(lengths):.0f}, "
        f"max {lengths.max()}"
    )
    for label, rows in arms.items():
        print(f"\n=== {label} ===")
        arm_section(label, rows)
        by_length(label, rows)
    shipped = None
    for label, rows in arms.items():
        if label.startswith("trained, floored"):
            shipped = rows
    if shipped is not None:
        print("\n=== the ends of the distribution, on the shipped surface ===")
        extremes(dome, cycles, shipped)
    print(
        "\n    identification departure is `‖UVᵀ − I‖_F/sqrt(2m)`: "
        "0 at the identity, 1 at chance, sqrt(2) maximally opposed."
    )
    print(
        "    This rig asserts nothing and rules on no remedy "
        "(#396 stays superseded; #315 stays open)."
    )


# -- the run ----------------------------------------------------------------


def read(name: str, split: str, seed: int, learn, draws: int, out) -> None:
    spec = DEFAULT_SPEC if name == "real" else SMALL
    dome = build_graph(spec)
    cycles = cycle_basis(dome)
    checkpoints = tuple(sorted(learn))
    print(f"dome {name}: {len(dome.cells)} cells, {len(dome.edges)} edges")
    print(f"{len(cycles)} independent interior cycles; checkpoints {checkpoints}")
    arms: dict[str, list[dict[str, float]]] = {}

    def write() -> None:
        if out is None:
            return
        out.write_text(
            json.dumps(
                {
                    "dome": name,
                    "split": split,
                    "seed": seed,
                    "checkpoints": list(checkpoints),
                    "draws": draws,
                    "cycles": [list(c) for c in cycles],
                    "arms": arms,
                },
                indent=1,
            )
        )

    print("\n=== free arms: the draw, and chance ===", flush=True)
    for label, maker in (
        ("untrained floored = flat, independent (chance null)", flat_maps_of(dome)),
        ("untrained, unfloored", lambda g: untrained_maps(dome, g, floor=False)),
    ):
        rows: list[dict[str, float]] = []
        for draw in range(draws):
            maps = maker(torch.Generator().manual_seed(900 + draw))
            rows.extend(read_surface(dome, maps, cycles))
        arms[label] = rows
        print(f"    {label}: {draws} draws x {len(cycles)} cycles", flush=True)
    write()

    for floor in (True, False):
        arm = "floored" if floor else "unfloored"
        print(f"\n=== training {arm}, seed {seed} ===", flush=True)

        def sink(checkpoint, rows, arm=arm):
            arms[f"trained, {arm}, {checkpoint}"] = rows
            write()

        trained(spec, split, seed, checkpoints, cycles, floor, sink=sink)

    report(dome, cycles, arms)
    if out is not None:
        print(f"\nwrote {out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    it = sub.add_parser("read", help="the holonomy read")
    it.add_argument("--dome", default="real", choices=("real", "small"))
    it.add_argument("--split", default="train")
    it.add_argument("--seed", type=int, default=42)
    it.add_argument(
        "--learn",
        type=int,
        nargs="+",
        default=list(CHECKPOINTS),
        help="checkpoints on one trajectory (default 30000 100000)",
    )
    it.add_argument("--draws", type=int, default=DRAWS)
    it.add_argument("--out", default=None, help="where the JSON goes")
    it.add_argument("--no-file", action="store_true")
    args = parser.parse_args(argv)
    out = None
    if not args.no_file:
        out = pathlib.Path(
            args.out or _ROOT / "prototypes" / "holonomy-453" / "read.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
    read(args.dome, args.split, args.seed, args.learn, args.draws, out)


if __name__ == "__main__":
    main()
