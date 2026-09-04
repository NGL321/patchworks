"""What did the spectral floor cost, and what did it buy? (ticket #436)

ADR-0032's **third pre-registration**: the read that prices the constraint the
floor imposes. The ADR books a cost and declines to claim the benefit, and says
in terms that *"the comparison is made properly by the read below or not claimed
at all"*. This is that read::

    python benchmarks/floor_price.py price              # two 30k runs, ~40 min
    python benchmarks/floor_price.py price --learn 2000 # a fast shape check

**The quantity is per-hop gain along the channel, per edge and per direction.**
ADR-0022 defines a hop as an operator norm along a learned channel, and
`benchmarks/alignment_read.py` names the operator::

    M(e_in -> v -> e_out)  =  F_{v,e_out} . gain_v . F_{v,e_in}^T

The **per-direction** gains of that hop are exactly its singular values, so the
whole read is a singular spectrum per hop rather than a scalar per hop. That is
not a nicety. The trade ADR-0032 states honestly is `sqrt(m)` of loss on the
single best direction against `m - 1` directions that currently transmit nothing
at all, so a per-edge scalar that collapsed the directions would reproduce the
near-rank-1 view the floor exists to leave — it would report the cost and hide
the benefit. Never a graph-wide average (#127's standing rule) and never per
level (#181): every aggregate below is over **hops**, and the only grouping is by
`(m_in, m_out)`, which is construction rather than shape.

**The cost and the benefit are put in one unit, which is the whole difficulty.**
ADR-0032 refuses to net `sqrt(m)`-per-hop of gain loss against
[#237](https://github.com/NGL321/patchworks/issues/237)'s 2.8e5x-4.4e5x sheaf
effective resistance ratio, because a resistance ratio and a gain product are not
the same kind of number. They are netted here by **composing the hops** along a
fixed rim-to-apex route and reading the composed operator's own spectrum: the
top singular value is what the channel carries end to end, the rest are what the
other directions carry end to end, and `‖.‖_F` over the whole spectrum is the
total transported energy in one number and one unit. Cost is a fall in the first
column; benefit is a rise in the others; the Frobenius column is the net.

**Before and after are the same rig, the same seeds and the same harness**, taken
in one process, because the ticket asks for a comparison and not for a quotation
against an older reading. The `before` surface is trained with
:meth:`RestrictionMaps._flatten` disabled — see :func:`without_floor`, which
patches rather than flags, deliberately: the floor is `project`'s third step and
a flag would be a change to the constraint under test.

**Three limits, stated rather than discovered.**

- **`M` omits the body's Jacobian**, which sits between `F_in^T` and `F_out`.
  This is `alignment_read`'s own limit, carried unchanged. The read is a property
  of the restriction maps, which is what the floor constrains and what this
  ticket prices; a body that *rotated* the transported deviation would move the
  composed columns and not the per-hop ones. `alignment_read.single_step_section`
  is the instrument for that and is not re-run here.
- **The routes are structural, and fixed across the two runs.** A widest path is
  a measured object and would move between the runs, mixing a change of route
  into a change of gain — so the comparison would stop being like-for-like at
  exactly the place it has to be. Every chain below is the graph's own shortest
  rim-to-apex edge path, identical in both runs. What the floor does to *which*
  route binds is a separate question and this read does not answer it.
- **`floored` names 1339 of 1364 endpoints.** The nine unattainable masks are
  excluded by construction and are not evidence about the constraint; they are
  reported separately and excluded from every aggregate that prices the floor.
"""

import argparse
import sys
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from patchworks.graph import Dome, build_graph
from patchworks.restriction import GAUGE_RHO, RestrictionMaps, pair_index
from patchworks.tick import DEFAULT_GAMMA, reconciliation_gain

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.append(_HERE)
import construction_grading as cg  # noqa: E402
import detectability as det  # noqa: E402
import untrained_fixed_point as ufp  # noqa: E402

#: Below this a singular value is not a direction the surface carries, it is the
#: arithmetic left over from one. The composed rim-to-apex operators run to
#: `1e-30` and smaller in the before run, so the cut is relative to each
#: operator's own top value rather than absolute.
RELATIVE_ZERO = 1e-12


# -- the two surfaces, one harness ------------------------------------------


@contextmanager
def without_floor():
    """Train with ADR-0032's spectral floor removed from `project`.

    A patch rather than a constructor flag, and that is the point: the floor is
    the third of `project`'s four ordered steps, and adding a flag would change
    the very method whose effect is being priced. Patching leaves the shipped
    code exactly as it runs and removes one call, so the `before` surface is the
    surface `main` has today under this harness — the mask, the band and the
    incoherence cap, in that order, with nothing between the band and the cap.
    """
    original = RestrictionMaps._flatten
    RestrictionMaps._flatten = lambda self: None
    try:
        yield
    finally:
        RestrictionMaps._flatten = original


def surface(name: str, split: str, seed: int, learn: int, floored: bool):
    """A trained surface, cast to float64 for the read; `det.prepared` unchanged.

    The cast is #146's and #183's standing requirement and is not a change to the
    architecture — see :func:`detectability.double_precision`. It matters more
    here than usual: the composed rim-to-apex operator's trailing singular values
    are the whole benefit half of the read, and in the before run they are the
    values float32 cannot represent as anything but rounding.
    """
    label = "with the floor" if floored else "without the floor"
    print(f"\n=== training {label} ===", flush=True)
    if floored:
        return det.prepared(name, split, seed, learn)
    with without_floor():
        return det.prepared(name, split, seed, learn)


# -- the hop, and its spectrum ----------------------------------------------


def hops_of_graph(dome: Dome) -> list[tuple[int, int, int]]:
    """Every directed `(edge_in, cell, edge_out)` hop through an interior cell.

    Every hop the graph admits, not a sample and not a path: the per-hop half of
    this read is a property of the trained maps and needs no probe, so there is
    no reason to read a subset of it. Boundary cells are skipped, as they are in
    `alignment_read.collect` — a boundary cell is written or read and never both
    (ADR-0016), so a hop *through* one is not a thing the surface does.
    """
    out: list[tuple[int, int, int]] = []
    for cell in dome.cells:
        if cell.is_boundary:
            continue
        incident = dome.incident[cell.id]
        for edge_in in incident:
            for edge_out in incident:
                if edge_in != edge_out:
                    out.append((edge_in, cell.id, edge_out))
    return out


def hop_operator(dome: Dome, maps: RestrictionMaps, gains: torch.Tensor, key) -> torch.Tensor:
    """`M = F_out . gain_v . F_in^T` in float64. `alignment_read.hop_operator`'s `M`.

    Taken here rather than imported so this script does not depend on that one's
    three-value return, and identical to it: same slicing, same `pair_index`,
    same `side_of`, same cast.
    """
    edge_in, cell, edge_out = key
    m_in, m_out = dome.edges[edge_in].m, dome.edges[edge_out].m
    with torch.no_grad():
        f_in = maps.maps[pair_index(edge_in, cg.side_of(dome, edge_in, cell))][:m_in]
        f_out = maps.maps[pair_index(edge_out, cg.side_of(dome, edge_out, cell))][:m_out]
        return (f_out.double() @ f_in.double().T) * float(gains[cell])


def spectrum(operator: torch.Tensor) -> np.ndarray:
    """The operator's singular values, descending: its gain in each direction."""
    return torch.linalg.svdvals(operator).numpy().astype(np.float64)


def effective_rank(values: np.ndarray) -> float:
    """Participation ratio `(sum s^2)^2 / sum s^4`: 1 for rank-1, `r` for `r` equal.

    `alignment_read.rank_profile`'s definition, unchanged, because the before
    run's number has to be comparable with the 1.02-1.06 the record already
    carries for the maps and with #142's near-rank-1 premise.
    """
    squares = values**2
    total = squares.sum()
    if total <= 0:
        return float("nan")
    return float(total**2 / (squares**2).sum())


def carried(values: np.ndarray) -> int:
    """How many directions this operator transmits at a tenth of its channel or better.

    Relative to the operator's own top value, because the composed chains span
    thirty orders of magnitude between the two runs and an absolute cut would
    count the after run's directions and none of the before run's for a reason
    that is about the level and not about the rank.

    **The cut is a tenth and not machine zero, and that is the difference between
    an instrument and a decoration.** Every operator here is dense, so at a
    machine-relative cut a near-rank-1 hop still reports its full `min(m_in,
    m_out)` — a *numerical* rank, which is 4 both before and after and says
    nothing. The question ADR-0032's trade turns on is how many directions carry
    a usable share, and a tenth of the channel is the coarsest cut that can tell
    them apart. :data:`RELATIVE_ZERO` stays below it as the guard against reading
    a direction off arithmetic noise.
    """
    if values.size == 0 or values[0] <= 0:
        return 0
    return int((values / values[0] >= 0.1).sum())


def off_channel_share(values: np.ndarray) -> float:
    """`1 - sigma_1^2/‖.‖_F^2`: the share of transported energy that is not the channel.

    The single number ADR-0032's trade is stated in. *"`m - 1` directions that
    currently transmit nothing at all"* is the claim that this is near zero
    before; the floor buys nothing unless it is materially larger after, and it
    is scale-free, so it says what the trade did without the level of the run
    getting in the way.
    """
    total = float((values**2).sum())
    if total <= 0:
        return float("nan")
    return float(1.0 - values[0] ** 2 / total)


def channel_over_isotropic(values: np.ndarray, m_in: int) -> float:
    """`sigma_max / (‖M‖_F / sqrt(m_in))`: the channel against the average direction.

    This is #142's ratio in its per-hop form — what reading along the channel
    buys over probing isotropically. **It shrinks as the spectrum flattens, by
    construction**: a flat operator has no preferred direction for the channel to
    ride, and the ratio goes to 1 exactly. ADR-0032 flagged that #142's
    explanatory sentence stops being true of the post-floor surface without
    amending it, and [#240](https://github.com/NGL321/patchworks/issues/240)'s
    gate is what that is for. This column is what says what the ratio became.

    The struck ~1e14 phantom deficit does not come back: the isotropic baseline
    does not move here, only the channel's advantage over it.
    """
    total = float(np.sqrt((values**2).sum()))
    if total <= 0:
        return float("nan")
    return float(values[0] / (total / np.sqrt(m_in)))


# -- the rim-to-apex chains -------------------------------------------------


def chain_paths(dome: Dome) -> dict[int, tuple[int, ...]]:
    """One shortest rim-to-apex edge path per rim cell, as edge ids.

    A breadth-first search over cells from each rim cell to the nearest apex
    cell. **Structural, and the same in both runs** — see the module docstring's
    second limit. Interior only in the middle: a path is not allowed to route
    *through* another boundary cell, since ADR-0016 says a boundary cell is
    written or read and never both, so a transit through one is not transport.
    """
    apex = set(det.apex(dome))
    paths: dict[int, tuple[int, ...]] = {}
    for source in det.rim(dome):
        seen = {source}
        queue = deque([(source, ())])
        while queue:
            cell, route = queue.popleft()
            if cell in apex:
                paths[source] = route
                break
            for edge_id in dome.incident[cell]:
                edge = dome.edges[edge_id]
                far = edge.v if edge.u == cell else edge.u
                if far in seen:
                    continue
                if dome.cells[far].is_boundary and far not in apex:
                    continue
                seen.add(far)
                queue.append((far, route + (edge_id,)))
    return paths


def chain_operator(dome, maps, gains, path: tuple[int, ...]) -> torch.Tensor | None:
    """The hops of `path`, composed. `None` where the path admits no interior hop.

    The composition is where the cost and the benefit finally meet in one unit.
    It is **not** the product of the per-hop top gains: that product would assume
    every hop's channel is the previous hop's channel, which is exactly #233's
    composition gap `C` (median 0.0079x) and is false. Composing the operators
    and taking the spectrum of the composition assumes nothing.
    """
    hops = cg.hops_of(dome, path)
    hops = [h for h in hops if not dome.cells[h[1]].is_boundary]
    if not hops:
        return None
    composed = hop_operator(dome, maps, gains, hops[0])
    for key in hops[1:]:
        composed = hop_operator(dome, maps, gains, key) @ composed
    return composed


# -- reporting --------------------------------------------------------------


def quantiles(series) -> str:
    values = np.asarray([v for v in series if np.isfinite(v)], dtype=np.float64)
    if values.size == 0:
        return f"{'-':>11}{'-':>11}{'-':>11}"
    return (
        f"{np.percentile(values, 5):11.4g}"
        f"{np.median(values):11.4g}"
        f"{np.percentile(values, 95):11.4g}"
    )


def ratio_line(label: str, before, after) -> None:
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    good = np.isfinite(before) & np.isfinite(after) & (before > 0)
    print(f"    {label:<34}{quantiles(after[good] / before[good])}   n={int(good.sum())}")


def reach_section(reached, shapes, flat_before, flat_after) -> None:
    """What the floor reached, and how flat it actually left it.

    The first thing to check, because every number after it is meaningless if the
    floor did not fire. `flatness` is `sigma_min/sigma_max` per map and reads 1
    when flat; `project` orders the floor **before** the incoherence cap, so it
    holds exactly only where the cap does not bite, and this is the residual.

    Takes the read values rather than the agents, because :func:`price` releases
    each surface as soon as it has been read.
    """
    print("\n-- what the floor reached (ADR-0032, `floored`) --")
    print(
        f"  floored endpoints: {int(reached.sum())} of {reached.numel()} "
        f"({reached.numel() - int(reached.sum())} unattainable or m=1, excluded)"
    )
    print(f"  groups by (m_e, k_v): {shapes}")
    print(f"\n    {'flatness sigma_min/sigma_max':<34}{'5th':>11}{'median':>11}{'95th':>11}")
    mask = reached.numpy()
    for label, values in (("before", flat_before), ("after", flat_after)):
        print(f"    {label + ', floored maps only':<34}{quantiles(values[mask])}")


def hop_section(dome: Dome, before: dict, after: dict) -> None:
    """The per-hop half: what one hop's gain did, per edge and per direction.

    The cost is the first line and it is the one the arithmetic already fixed —
    `sigma_max` down by `sqrt(m)` per hop is ADR-0032's booked price, and this
    says whether the surface paid it. Everything under it is the benefit, and it
    is read as the *shape* of the spectrum rather than its level: the trailing
    directions' share of the transported energy is what a near-rank-1 hop has
    none of.
    """
    print("\n-- per hop: gain per direction (ADR-0022's `M`, every directed hop) --")
    keys = sorted(before)
    print(f"    {'ratio after/before':<34}{'5th':>11}{'median':>11}{'95th':>11}")
    ratio_line("sigma_max (the channel)", [before[k][0] for k in keys], [after[k][0] for k in keys])
    ratio_line(
        "‖M‖_F (all directions)",
        [np.sqrt((before[k] ** 2).sum()) for k in keys],
        [np.sqrt((after[k] ** 2).sum()) for k in keys],
    )
    ratio_line(
        "sigma_2 (the best off-channel)",
        [before[k][1] if before[k].size > 1 else np.nan for k in keys],
        [after[k][1] if after[k].size > 1 else np.nan for k in keys],
    )
    ratio_line(
        "sigma_min (the worst direction)",
        [before[k][-1] for k in keys],
        [after[k][-1] for k in keys],
    )

    print(f"\n    {'absolute':<34}{'5th':>11}{'median':>11}{'95th':>11}")
    for label, table in (("before", before), ("after", after)):
        print(f"    {'effective rank, ' + label:<34}{quantiles([effective_rank(table[k]) for k in keys])}")
    for label, table in (("before", before), ("after", after)):
        print(
            f"    {'sigma_1^2 share, ' + label:<34}"
            f"{quantiles([(table[k][0] ** 2) / (table[k] ** 2).sum() for k in keys])}"
        )
    for label, table in (("before", before), ("after", after)):
        print(
            f"    {'off-channel share, ' + label:<34}"
            f"{quantiles([off_channel_share(table[k]) for k in keys])}"
        )
    for label, table in (("before", before), ("after", after)):
        print(
            f"    {'directions >= 0.1 sigma_1, ' + label:<34}"
            f"{quantiles([carried(table[k]) for k in keys])}"
        )


def isotropic_section(dome: Dome, before: dict, after: dict) -> None:
    """#142's channel-versus-isotropic ratio, per hop, before and after.

    ADR-0032 flagged that the sentence explaining why the 1e14 phantom went away
    stops being true of the post-floor surface. This is what it became. The
    ceiling is `sqrt(m_in)` — a rank-1 hop puts all of `‖M‖_F` on one direction —
    and the floor is 1, which is a flat hop with no channel to prefer.
    """
    print("\n-- #142's ratio: the channel against an isotropic probe, per hop --")
    keys = sorted(before)
    print(f"    {'sigma_max / (‖M‖_F / sqrt(m_in))':<34}{'5th':>11}{'median':>11}{'95th':>11}")
    for label, table in (("before", before), ("after", after)):
        print(
            f"    {label:<34}"
            f"{quantiles([channel_over_isotropic(table[k], dome.edges[k[0]].m) for k in keys])}"
        )
    widths = sorted({dome.edges[k[0]].m for k in keys})
    print(f"\n    the ceiling is sqrt(m_in), which is {[f'{w}: {np.sqrt(w):.3f}' for w in widths]}")


def shape_section(dome: Dome, before: dict, after: dict) -> None:
    """The same read grouped by `(m_in, m_out)` — construction, never level.

    #181's rule is that a target or a reading is indexed per edge and not per
    level, because a level is a property of the shape imposed on the graph. The
    lane widths are not: `m` is what construction gave the edge, and the floor's
    booked price is a function of it (`sqrt(4) = 2` interior, `sqrt(8) = 2.83`
    at a boundary map), so this is the grouping that can confirm or refute the
    arithmetic rather than decorate it.
    """
    print("\n-- per hop, grouped by lane width (m_in, m_out): construction, not level --")
    groups = defaultdict(list)
    for key in before:
        groups[(dome.edges[key[0]].m, dome.edges[key[2]].m)].append(key)
    print(
        f"    {'(m_in, m_out)':<16}{'hops':>7}{'sigma_max x':>13}{'‖M‖_F x':>12}"
        f"{'eff rank ->':>13}{'#142 ratio ->':>15}"
    )
    for shape in sorted(groups):
        keys = groups[shape]
        top = np.median([after[k][0] / before[k][0] for k in keys if before[k][0] > 0])
        frob = np.median(
            [
                np.sqrt((after[k] ** 2).sum() / (before[k] ** 2).sum())
                for k in keys
                if (before[k] ** 2).sum() > 0
            ]
        )
        rank_before = np.median([effective_rank(before[k]) for k in keys])
        rank_after = np.median([effective_rank(after[k]) for k in keys])
        iso_before = np.median([channel_over_isotropic(before[k], shape[0]) for k in keys])
        iso_after = np.median([channel_over_isotropic(after[k], shape[0]) for k in keys])
        print(
            f"    {str(shape):<16}{len(keys):>7}{top:>13.4g}{frob:>12.4g}"
            f"{rank_before:>6.2f}->{rank_after:<6.2f}{iso_before:>7.3f}->{iso_after:<7.3f}"
        )


def booked_cost(dome: Dome, path: tuple[int, ...]) -> float:
    """ADR-0032's arithmetic price for this chain: the product of `sqrt(m)` per map.

    The ADR books `sqrt(m)` per map on the ground that the projection preserves
    `‖F‖_F` and so moves `sigma_max` from `~‖F‖_F` — which is where it sits at an
    effective rank of 1.0009 — to `‖F‖_F/sqrt(m)`. A chain's hops carry two maps
    each and the interior ones are shared, so the product runs over the path's
    edges: `sqrt(4) = 2` per interior lane, `sqrt(8) = 2.83` at a boundary one.

    **This is the ceiling on the cost, not a prediction of it**, and the
    difference is the whole reason the read exists. The `sqrt(m)` fall is what a
    map at effective rank 1 loses; a map that already carries two directions
    loses less, because it had less concentrated on one. Printing the booked
    number beside the measured one is what says whether the price the ADR wrote
    down is the price the surface actually paid.
    """
    price_ = 1.0
    for edge_id in path:
        price_ *= np.sqrt(dome.edges[edge_id].m)
    return float(price_)


def booked_section(dome: Dome, chains: dict, before: dict, after: dict) -> None:
    """Measured end-to-end cost against the price ADR-0032 booked for it."""
    keys = sorted(before)
    print(f"\n    {'the booked price against the paid one':<34}{'5th':>11}{'median':>11}{'95th':>11}")
    print(
        f"    {'booked: prod sqrt(m) over the path':<34}"
        f"{quantiles([booked_cost(dome, chains[k]) for k in keys])}"
    )
    print(
        f"    {'paid: sigma_max before/after':<34}"
        f"{quantiles([before[k][0] / after[k][0] for k in keys if after[k][0] > 0])}"
    )
    print(
        f"    {'paid / booked':<34}"
        f"{quantiles([(before[k][0] / after[k][0]) / booked_cost(dome, chains[k]) for k in keys if after[k][0] > 0])}"
    )


def chain_section(dome: Dome, chains: dict, before: dict, after: dict) -> None:
    """The composed rim-to-apex operator: cost and benefit in one unit.

    This is the netting ADR-0032 refused to do in mixed units and asked for here.
    `sigma_max` is what the channel carries end to end and is where the booked
    `2^7 = 128x` should show; `‖.‖_F` is the total over **every** direction and
    is the net; `directions carried` is the `m - 1` the trade is against.

    A rise in the Frobenius column with a fall in the `sigma_max` column is the
    trade going through — energy moved off the one direction and onto the rest.
    A fall in both is the constraint costing without buying, and it is the
    outcome ADR-0032 left open.
    """
    print("\n-- composed rim-to-apex chains: the two halves in one unit --")
    keys = sorted(before)
    lengths = Counter(len(chains[k]) for k in keys)
    print(f"  {len(keys)} chains; hop counts {dict(sorted(Counter(len(chains[k]) - 1 for k in keys).items()))}")
    print(f"  (edge-path lengths {dict(sorted(lengths.items()))})")

    print(f"\n    {'ratio after/before':<34}{'5th':>11}{'median':>11}{'95th':>11}")
    ratio_line("sigma_max end to end", [before[k][0] for k in keys], [after[k][0] for k in keys])
    ratio_line(
        "‖chain‖_F, every direction",
        [np.sqrt((before[k] ** 2).sum()) for k in keys],
        [np.sqrt((after[k] ** 2).sum()) for k in keys],
    )
    ratio_line(
        "sigma_2 end to end",
        [before[k][1] if before[k].size > 1 else np.nan for k in keys],
        [after[k][1] if after[k].size > 1 else np.nan for k in keys],
    )

    print(f"\n    {'absolute':<34}{'5th':>11}{'median':>11}{'95th':>11}")
    for label, table in (("before", before), ("after", after)):
        print(f"    {'sigma_max, ' + label:<34}{quantiles([table[k][0] for k in keys])}")
    for label, table in (("before", before), ("after", after)):
        print(
            f"    {'‖chain‖_F, ' + label:<34}"
            f"{quantiles([np.sqrt((table[k] ** 2).sum()) for k in keys])}"
        )
    for label, table in (("before", before), ("after", after)):
        print(f"    {'effective rank, ' + label:<34}{quantiles([effective_rank(table[k]) for k in keys])}")
    for label, table in (("before", before), ("after", after)):
        print(
            f"    {'off-channel share, ' + label:<34}"
            f"{quantiles([off_channel_share(table[k]) for k in keys])}"
        )
    for label, table in (("before", before), ("after", after)):
        print(
            f"    {'directions >= 0.1 sigma_1, ' + label:<34}"
            f"{quantiles([carried(table[k]) for k in keys])}"
        )

    booked_section(dome, chains, before, after)
    print("\n  by rim kind (construction, not level):")
    kinds = defaultdict(list)
    for key in keys:
        kinds[dome.cells[key].kind.value].append(key)
    print(f"    {'kind':<18}{'chains':>7}{'sigma_max x':>13}{'‖chain‖_F x':>14}{'carried ->':>16}")
    for kind in sorted(kinds):
        group = kinds[kind]
        top = np.median([after[k][0] / before[k][0] for k in group if before[k][0] > 0])
        frob = np.median(
            [
                np.sqrt((after[k] ** 2).sum() / (before[k] ** 2).sum())
                for k in group
                if (before[k] ** 2).sum() > 0
            ]
        )
        car_b = np.median([carried(before[k]) for k in group])
        car_a = np.median([carried(after[k]) for k in group])
        print(
            f"    {kind:<18}{len(group):>7}{top:>13.4g}{frob:>14.4g}"
            f"{car_b:>9.1f}->{car_a:<7.1f}"
        )


# -- the run ----------------------------------------------------------------


def read(agent, dome: Dome, hops, chains) -> tuple[dict, dict]:
    """Every hop's spectrum and every chain's spectrum, off one trained surface."""
    gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=GAUGE_RHO)
    maps = agent.sheaf.maps
    per_hop = {key: spectrum(hop_operator(dome, maps, gains, key)) for key in hops}
    per_chain = {}
    for source, path in chains.items():
        composed = chain_operator(dome, maps, gains, path)
        if composed is not None:
            per_chain[source] = spectrum(composed)
    return per_hop, per_chain


def price(name: str, split: str, seed: int, learn: int) -> None:
    """Train, read, release; then train, read, release. Never both surfaces at once.

    **The two runs are sequential and only their spectra outlive them.** Holding
    both trained agents to read them side by side is the obvious shape and it is
    the wrong one: a surface is the sandbox, the bodies and the sheaf, where what
    this read needs off it is a few thousand arrays of at most eight floats. The
    reads are order-independent — a spectrum is a property of a finished surface
    and nothing here trains against anything read — so releasing each surface
    before training the next costs the comparison nothing and halves the peak.
    """
    # The routes off the graph alone, before any surface exists — which is the
    # module docstring's second limit made structural rather than promised.
    dome = build_graph(ufp.dome_named(name)[0])
    hops = hops_of_graph(dome)
    chains = chain_paths(dome)
    print(f"\nreading {len(hops)} directed hops and {len(chains)} rim-to-apex chains")

    after_agent = surface(name, split, seed, learn, floored=True)[1]
    hop_after, chain_after = read(after_agent, dome, hops, chains)
    flat_after = after_agent.sheaf.maps.flatness().numpy().astype(np.float64)
    reached = after_agent.sheaf.maps.floored.clone()
    shapes = list(after_agent.sheaf.maps.floor_shapes)
    del after_agent

    before_agent = surface(name, split, seed, learn, floored=False)[1]
    hop_before, chain_before = read(before_agent, dome, hops, chains)
    flat_before = before_agent.sheaf.maps.flatness().numpy().astype(np.float64)
    del before_agent

    reach_section(reached, shapes, flat_before, flat_after)
    hop_section(dome, hop_before, hop_after)
    isotropic_section(dome, hop_before, hop_after)
    shape_section(dome, hop_before, hop_after)
    chain_section(dome, chains, chain_before, chain_after)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("price",), nargs="?", default="price")
    parser.add_argument("--dome", default="real")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learn", type=int, default=30_000)
    arguments = parser.parse_args(argv)
    price(arguments.dome, arguments.split, arguments.seed, arguments.learn)


if __name__ == "__main__":
    main()
