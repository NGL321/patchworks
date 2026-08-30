"""The dome as a graph: effective resistance, the gauge's share of the hop, and curvature.

Three decision-free measurements, all graph-side and none needing a training run
or a sandbox. Together they turn
[#120](https://github.com/NGL321/patchworks/issues/120)'s measured ~921x per hop
from an observation into a diagnosis that names *which* factor is dominant and
*which* cell pairs are unreachable.

    .venv/bin/python benchmarks/graph_transmission.py

**resistance.** Di Giovanni, Giusti, Barbero, Luise, Lio & Bronstein (ICML 2023,
arXiv:2302.02941) factorise node-to-node sensitivity into a *model* term and a
*topological* term, and bound the Jacobian obstruction above and below by
commute time `tau(v,u) / 2|E|`, **independent of depth**. The topological term is
therefore computable without running anything, and it has never been computed on
this graph. Effective resistance is also one of Hansen & Ghrist's spectral sheaf
results (arXiv:1808.01513) -- the sheaf Laplacian's effective resistance and the
over-squashing effective resistance are the same object seen from two sides, so
one computation serves both halves of the architecture.

Two weightings are run, because they answer different questions. **Unweighted**
is the literature's object, so Theorem 5.5 applies to it verbatim. **Stalk-
weighted** (`w_e = m_e`) is the graph's own bandwidth, and is the one that knows
a boundary-incident edge is twice an interior one.

**gauge.** `rho = 2` bounds each restriction map's magnitude to within a factor
2, so over the taper's hops the band is worth at most `rho` per map. The measured
attenuation is ~921 per hop, and `2^9 = 512` is the same order of magnitude --
which is either a coincidence or a diagnosis, and has never been checked. The
check is exact rather than statistical: one hop's two map factors are

    restrict_p = |F_p dx| / |dx|          edge_r = gain_r * |F_r^T d| / |d|

and for a direction drawn uniformly the expectation of each is its map's
Frobenius norm divided by the square root of the dimension it acts on. So the
hop factorises into a term the gauge sets (`|F_p| |F_r|`, confined to
`[rho^-2, rho^2]`), a term construction sets (`gain_r / sqrt(d_p * m_e)`), and
the body -- and each can be read off separately. The Monte Carlo below is a
check on that identity, not the source of the numbers.

The `rho` sweep is the part that bears on candidate 3 of
[#142](https://github.com/NGL321/patchworks/issues/142): `rho` appears in the
*numerator* of the hop through the map norms **and in the denominator** through
`gain_v = gamma / max(sum_e m_e, rho^2 deg(v))`, so whether raising it buys
anything is arithmetic, not opinion.

**curvature.** Balanced Forman curvature (Topping, Di Giovanni, Chamberlain,
Dong & Bronstein, ICLR 2022, arXiv:2111.14522, Definition 1), which they prove
identifies the negatively curved edges *"responsible for the over-squashing
issue"*. The dome already distinguishes rim-to-interior edges from interior ones
by stalk width (`m = 8` against `m = 4`); this either justifies that choice or
relocates it.

Nothing here is a decision. The body factor is not recomputed -- it needs a
settled operating point and a sandbox -- and is taken from #120's reading, held
as :data:`BODY_GAIN` and used only to put the map factors on the same scale as
that measurement.
"""

import time
from collections import defaultdict

import numpy as np
import torch

from patchworks.graph import CellKind, build_graph
from patchworks.restriction import GAUGE_RHO, RestrictionMaps, pair_index
from patchworks.tick import DEFAULT_GAMMA, reconciliation_gain

#: `body`, `d|prediction| / d|evidence|`, from #120's untrained reading on the
#: real dome. Not recomputed here: it is a frozen random MLP's Jacobian at a
#: settled operating point, which needs the sandbox. Used only to place the two
#: map factors against that measurement's `one hop, one tick`.
BODY_GAIN = 0.4529

#: #120's measured per-hop transfer, untrained, for the same reason.
MEASURED_HOP = 0.001086

#: Draws per endpoint in the Monte Carlo check of the Frobenius identity.
DRAWS = 256

SEED = 42


# -- effective resistance ---------------------------------------------------


def weighted_adjacency(dome, weight) -> np.ndarray:
    """`[cells, cells]` symmetric weights. `weight` maps an edge to a conductance."""
    a = np.zeros((len(dome.cells), len(dome.cells)))
    for e in dome.edges:
        w = float(weight(e))
        a[e.u, e.v] += w
        a[e.v, e.u] += w
    return a


def effective_resistance(adjacency: np.ndarray) -> np.ndarray:
    """`[cells, cells]` pairwise effective resistance, from the Laplacian's pseudoinverse.

    `R(u,v) = L+_uu + L+_vv - 2 L+_uv`. At 414 nodes the pseudoinverse is the
    cheapest correct route; nothing here needs a solver.
    """
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    pinv = np.linalg.pinv(laplacian, hermitian=True)
    diagonal = np.diag(pinv)
    return diagonal[:, None] + diagonal[None, :] - 2.0 * pinv


def hop_distance(dome, sources: list[int]) -> np.ndarray:
    """`[cells]` shortest-path hop count to the nearest of `sources`, by breadth first.

    The number the taper is usually described by -- eight levels from rim to
    apex -- against which the effective resistance says how much of that depth
    the graph's parallel routes buy back.
    """
    distance = np.full(len(dome.cells), -1)
    distance[sources] = 0
    frontier = list(sources)
    while frontier:
        nxt = []
        for cell_id in frontier:
            for neighbour in dome.neighbours(cell_id):
                if distance[neighbour] < 0:
                    distance[neighbour] = distance[cell_id] + 1
                    nxt.append(neighbour)
        frontier = nxt
    return distance


def group_of(dome, cell_id: int) -> str:
    """The row label a cell reports under: its boundary kind, or its level."""
    cell = dome.cells[cell_id]
    if cell.kind is CellKind.PREDICTING:
        return f"L{cell.index.level}"
    return cell.kind.value


def resistance_section(dome) -> dict:
    """Effective resistance and commute time, both weightings. Returns the readings."""
    apex = [
        c.id for c in dome.cells if c.kind is CellKind.PREDICTING and c.index.level == 7
    ]
    readings = {}
    for label, weight in (("unweighted", lambda e: 1.0), ("stalk m_e", lambda e: e.m)):
        adjacency = weighted_adjacency(dome, weight)
        volume = float(adjacency.sum())  # sum_v deg(v), the commute-time constant
        resistance = effective_resistance(adjacency)
        readings[label] = (adjacency, volume, resistance)

        print(f"\n### effective resistance, {label} (volume {volume:g})\n")
        # The rim-to-apex reading, which is the number the taper is about, and
        # the profile by level beside it so the shape of the climb is visible.
        # `hops` is graph distance; `R / R_edge` is the same climb measured in
        # unit-resistance edges. Where the second is far below the first, the
        # graph's parallel routes have already bought that depth back, and
        # depth is not what the attenuation is.
        distance = hop_distance(dome, apex)
        unit = float(np.mean([1.0 / weight(e) for e in dome.edges]))
        print(
            f"  {'from':>16} {'cells':>6} {'hops':>5} | {'R to apex':>18} | "
            f"{'commute tau':>19} | {'in unit edges':>13}"
        )
        print(
            f"  {'':>16} {'':>6} {'':>5} | {'mean':>8} {'max':>9} | "
            f"{'mean':>9} {'max':>9} | {'mean':>13}"
        )
        rows = defaultdict(list)
        for cell in dome.cells:
            if cell.id in apex:
                continue
            rows[group_of(dome, cell.id)].append(cell.id)
        for name in sorted(rows, key=lambda s: (s[0] != "L", s)):
            ids = rows[name]
            block = resistance[np.ix_(ids, apex)]
            tau = volume * block
            print(
                f"  {name:>16} {len(ids):>6} {distance[ids].mean():5.1f} | "
                f"{block.mean():8.4f} {block.max():9.4f} | "
                f"{tau.mean():9.1f} {tau.max():9.1f} | {block.mean() / unit:13.2f}"
            )

        # The worst pairs in the graph, which is where the pairwise map is
        # actually read: the largest commute times are the unreachable pairs.
        flat = np.triu(resistance, 1)
        order = np.argsort(flat, axis=None)[::-1][:8]
        print("\n  worst pairs by commute time")
        for u, v in zip(*np.unravel_index(order, flat.shape)):
            print(
                f"    {str(dome.cells[u].index):>18} {group_of(dome, u):>14}  <->  "
                f"{str(dome.cells[v].index):<18} {group_of(dome, v):<14} "
                f"R {resistance[u, v]:7.4f}  tau {volume * resistance[u, v]:9.1f}"
            )

        # `w_e * R(u,v)` over an edge is that edge's spanning-tree probability:
        # 1 exactly for a bridge, and low for an edge with many parallel routes.
        # It names the cuts directly rather than by inference from the pairs.
        share = np.array([weight(e) * resistance[e.u, e.v] for e in dome.edges])
        bridges = int((share > 0.999).sum())
        print(
            f"\n  edge cut share `w_e R(u,v)`: mean {share.mean():.4f}, "
            f"{bridges} of {len(dome.edges)} edges are bridges (share 1)"
        )
        by_kind = defaultdict(list)
        for e, s in zip(dome.edges, share):
            by_kind[f"{e.kind.value} (m={e.m})"].append(s)
        for kind in sorted(by_kind):
            values = np.array(by_kind[kind])
            print(
                f"    {kind:>18} {len(values):>4} edges  mean {values.mean():6.4f}  "
                f"median {np.median(values):6.4f}  min {values.min():6.4f}"
            )
    return readings


# -- the gauge's share of the hop -------------------------------------------


def hop_endpoints(dome) -> list[dict]:
    """One row per hop-carrying edge endpoint: the population #120's instrument reads.

    An endpoint whose receiver is a boundary cell is dropped, because what a
    neighbour's belief did to a stalk the world overwrites is not a hop in the
    taper -- the same exclusion `benchmarks/untrained_fixed_point.py` makes.
    """
    rows = []
    for e in dome.edges:
        for side in (0, 1):
            sender, receiver = (e.u, e.v)[side], (e.v, e.u)[side]
            if dome.cells[receiver].is_boundary:
                continue
            rows.append(
                dict(
                    edge=e,
                    sending=pair_index(e.id, side),
                    receiving=pair_index(e.id, side) ^ 1,
                    sender=sender,
                    receiver=receiver,
                    # The sender's permitted width: the directions the mask lets
                    # onto the edge at all. Diluting by the padded stalk instead
                    # would understate a patch cell, whose 48 are all permitted.
                    permitted=int(dome.restriction_mask(e.id, sender).sum()),
                    sender_boundary=dome.cells[sender].is_boundary,
                )
            )
    return rows


def gauge_section(dome) -> None:
    """How much of the per-hop factor is the gauge, and how much is everything else."""
    maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(SEED))
    norms = maps.norms().detach()
    gains = reconciliation_gain(dome)
    generator = torch.Generator().manual_seed(SEED)
    rows = hop_endpoints(dome)
    for row in rows:
        row["gain"] = float(gains[row["receiver"]])
        row["sender_norm"] = float(norms[row["sending"]])
        row["receiver_norm"] = float(norms[row["receiving"]])

    # The identity, checked rather than assumed: for `u` uniform on the sphere,
    # `E|F u|^2 = |F|_F^2 / d`, so `E|F u| ~= |F|_F / sqrt(d)` to within the
    # concentration of the chi distribution. If that check fails the split below
    # is meaningless, so it is printed first.
    predicted_restrict, measured_restrict = [], []
    predicted_edge, measured_edge = [], []
    with torch.no_grad():
        for row in rows:
            e, width = row["edge"], row["permitted"]
            sending = maps.maps[row["sending"], : e.m, :width]
            directions = torch.randn((DRAWS, width), generator=generator)
            directions /= directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            measured_restrict.append(float((directions @ sending.T).norm(dim=-1).mean()))
            predicted_restrict.append(row["sender_norm"] / np.sqrt(width))

            beliefs = torch.randn((DRAWS, e.m), generator=generator)
            beliefs /= beliefs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            back = beliefs @ maps.maps[row["receiving"], : e.m, :]
            measured_edge.append(row["gain"] * float(back.norm(dim=-1).mean()))
            predicted_edge.append(row["gain"] * row["receiver_norm"] / np.sqrt(e.m))

    predicted_restrict = np.array(predicted_restrict)
    measured_restrict = np.array(measured_restrict)
    predicted_edge = np.array(predicted_edge)
    measured_edge = np.array(measured_edge)

    print(f"\n### the gauge's share of one hop, over {len(rows)} hop-carrying endpoints\n")
    print("  the Frobenius identity, checked against a Monte Carlo of unit directions")
    for label, predicted, measured in (
        ("restrict", predicted_restrict, measured_restrict),
        ("edge    ", predicted_edge, measured_edge),
    ):
        ratio = measured / predicted
        print(
            f"    {label}  |F|/sqrt(d) {predicted.mean():9.5f}   drawn "
            f"{measured.mean():9.5f}   ratio mean {ratio.mean():6.4f} "
            f"min {ratio.min():6.4f} max {ratio.max():6.4f}"
        )

    hop = measured_restrict.mean() * measured_edge.mean() * BODY_GAIN
    print(
        f"\n  restrict {measured_restrict.mean():.4f}  edge {measured_edge.mean():.4f}  "
        f"body {BODY_GAIN:.4f} (from #120)  ->  one hop {hop:.4g}"
    )
    print(
        f"  #120 measured {MEASURED_HOP:.4g} on a settled sheaf; "
        f"ratio {hop / MEASURED_HOP:.3f}"
    )

    # The split. Each factor is a multiplicative term in the hop, so the honest
    # accounting is in the log: what share of `log(1 / hop)` each contributes.
    gauge = float(np.mean([r["sender_norm"] * r["receiver_norm"] for r in rows]))
    dilution = float(
        np.mean([1.0 / np.sqrt(r["permitted"] * r["edge"].m) for r in rows])
    )
    gain = float(np.mean([r["gain"] for r in rows]))
    print("\n  the hop's four factors, and what sets each")
    total = np.log(1.0 / hop)
    for label, value, sets in (
        (
            "gauge   |F_p| |F_r|",
            gauge,
            f"the gauge band, in [1/{GAUGE_RHO ** 2:g}, {GAUGE_RHO ** 2:g}]",
        ),
        ("dilute  1/sqrt(d m)", dilution, "the mask width and m_e, at construction"),
        ("gain    gamma/max()", gain, "gamma, m_e and degree, at construction"),
        ("body               ", BODY_GAIN, "a frozen random MLP; bounded by nothing"),
    ):
        share = np.log(1.0 / value) / total * 100.0
        print(f"    {label} {value:9.5f}   {share:6.1f}% of the loss   {sets}")

    # What the band is worth at all: its whole dynamic range, from floor to
    # ceiling, against the ~921x the hop loses.
    print(
        f"\n  the band's entire dynamic range is {GAUGE_RHO ** 2:g}x "
        f"(both maps at 1/rho to both at rho), against a per-hop loss of "
        f"{1.0 / hop:.0f}x"
    )

    # Why the sweep below does what it does. `gain_v = gamma / max(sum_e m_e,
    # rho^2 deg(v))`, and which of the two arms wins is what decides whether
    # `rho` cancels: on the `rho^2 deg` arm the gain falls as `rho^-2` exactly
    # as the two map norms rise as `rho^2`, and the hop is flat in `rho`.
    receivers = sorted({r["receiver"] for r in rows})
    stalk_arm = [
        c for c in receivers if dome.stalk_sums[c] > GAUGE_RHO**2 * dome.degrees[c]
    ]
    print(
        f"\n  which arm of the gain's `max` is live, over {len(receivers)} receiving cells: "
        f"{len(stalk_arm)} on `sum_e m_e`, {len(receivers) - len(stalk_arm)} on "
        f"`rho^2 deg` (ties to the second)"
    )

    # And the sweep that matters: `rho` is in the numerator through the map
    # norms and in the denominator through the gain, so it may buy nothing.
    print("\n  what raising rho buys, with the gain recomputed at that rho")
    print(f"    {'rho':>5} | {'restrict':>9} {'edge':>9} {'hop':>10} | {'vs rho=2':>9}")
    swept_hops = {}
    for rho in (1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0):
        swept_gain = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=rho)
        # Saturated: every unpinned map at `rho`, every boundary cell's own map
        # at the exact gauge of 1, which is what `pinned` marks.
        restrict = float(
            np.mean(
                [
                    (1.0 if r["sender_boundary"] else rho) / np.sqrt(r["permitted"])
                    for r in rows
                ]
            )
        )
        edge = float(
            np.mean(
                [
                    rho * float(swept_gain[r["receiver"]]) / np.sqrt(r["edge"].m)
                    for r in rows
                ]
            )
        )
        swept_hops[rho] = restrict * edge * BODY_GAIN
        against = (
            f"{swept_hops[rho] / swept_hops[GAUGE_RHO]:8.3f}x"
            if GAUGE_RHO in swept_hops
            else f"{'':>9}"
        )
        print(
            f"    {rho:5g} | {restrict:9.5f} {edge:9.5f} {swept_hops[rho]:10.4g} | "
            f"{against}"
        )


# -- what the stalk widths are worth ----------------------------------------


def analytic_hop(dome, rho: float = 1.0) -> float:
    """The per-hop transfer the Frobenius identity predicts, without drawing anything.

    `restrict * edge`, each map at Frobenius `rho` except a boundary cell's own,
    which the exact gauge pins at 1, times :data:`BODY_GAIN`. Checked against
    the Monte Carlo in :func:`gauge_section` on the real dome at `rho = 1`; used
    here to price construction parameters the sheaf would have to be rebuilt to
    measure.
    """
    gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=GAUGE_RHO)
    rows = hop_endpoints(dome)
    restrict = float(
        np.mean(
            [
                (1.0 if r["sender_boundary"] else rho) / np.sqrt(r["permitted"])
                for r in rows
            ]
        )
    )
    edge = float(
        np.mean(
            [rho * float(gains[r["receiver"]]) / np.sqrt(r["edge"].m) for r in rows]
        )
    )
    return restrict * edge * BODY_GAIN


def stalk_section() -> None:
    """What widening an edge stalk does to the hop -- and to `H^0`.

    `m` enters the hop three times and helps in none of them at fixed Frobenius
    norm. It is under a square root in the dilution (`1/sqrt(d m)`: the same map
    norm spread over more rows is less per row), and it is in the gain's
    `sum_e m_e` arm (`gamma / max(sum_e m_e, rho^2 deg)`). Above `m = rho^2` the
    two compose to `m^-3/2`, so a wider edge stalk **costs** per-direction
    transmission.

    That is not an argument against `m = 8` at the rim, because the two are
    buying different things: width buys **rank** — how many of a patch cell's 48
    directions can leave at all — and costs **gain** per direction. It is an
    argument that the trade is priced, and the price is here. The `H^0` column
    is the other side of the same coin: `private = max(0, n - sum_e m_e)`, so
    widening spends the private features `01-cell-and-sheaf.md` makes slow state
    out of, and at `interior_m = 8` there are none left in the graph at all.
    """
    import dataclasses

    from patchworks.graph import DEFAULT_SPEC

    print("\n### what the stalk widths are worth, at the construction level\n")
    print(
        f"  {'interior_m':>10} {'boundary_m':>10} | {'hop':>10} {'vs built':>9} | "
        f"{'private dim':>11} {'chi':>8}"
    )
    as_built = (DEFAULT_SPEC.interior_m, DEFAULT_SPEC.boundary_m)
    built = analytic_hop(build_graph(DEFAULT_SPEC))
    for interior_m, boundary_m in ((2, 8), (4, 4), (4, 8), (4, 16), (6, 8), (8, 8)):
        dome = build_graph(
            dataclasses.replace(
                DEFAULT_SPEC, interior_m=interior_m, boundary_m=boundary_m
            )
        )
        hop = analytic_hop(dome)
        print(
            f"  {interior_m:>10} {boundary_m:>10} | {hop:10.4g} {hop / built:8.3f}x | "
            f"{float(dome.private_dimensions.float().mean()):11.2f} "
            f"{dome.euler_characteristic:8d}"
            + ("   <- as built" if (interior_m, boundary_m) == as_built else "")
        )


# -- balanced Forman curvature ----------------------------------------------


def edge_curvature(neighbours: list[set[int]], i: int, j: int) -> float:
    """Balanced Forman curvature of one edge, Topping et al. (ICLR 2022), Definition 1.

    `Ric(i,j) = 2/d_i + 2/d_j - 2 + 2|T|/max(d_i,d_j) + |T|/min(d_i,d_j)
                + (gamma_max)^-1 (|S_i| + |S_j|)`

    where `T` is the triangles on `(i,j)`, `S_i` the neighbours of `i` that open
    a 4-cycle to `j` without closing a triangle, and `gamma_max` the largest
    number of such 4-cycles through any one of them. Zero where either endpoint
    has degree 1, as the definition requires.

    Takes an adjacency rather than a :class:`~patchworks.graph.Dome` so the
    implementation can be checked against graphs whose curvature the definition
    gives in closed form.
    """
    d_i, d_j = len(neighbours[i]), len(neighbours[j])
    if d_i <= 1 or d_j <= 1:
        return 0.0
    triangles = neighbours[i] & neighbours[j]
    t = len(triangles)
    # A 4-cycle `i ~ k ~ w ~ j` with `k` no neighbour of `j` and `w` no
    # neighbour of `i` -- squares based at the edge, sharing no triangle.
    squares_i = {
        k: len((neighbours[k] & neighbours[j]) - neighbours[i] - {i, j})
        for k in neighbours[i] - triangles - {j}
    }
    squares_j = {
        w: len((neighbours[w] & neighbours[i]) - neighbours[j] - {i, j})
        for w in neighbours[j] - triangles - {i}
    }
    squares_i = {k: n for k, n in squares_i.items() if n}
    squares_j = {w: n for w, n in squares_j.items() if n}
    ric = 2.0 / d_i + 2.0 / d_j - 2.0 + 2.0 * t / max(d_i, d_j) + t / min(d_i, d_j)
    widest = max(list(squares_i.values()) + list(squares_j.values()) + [0])
    if widest:
        ric += (len(squares_i) + len(squares_j)) / widest
    return ric


def balanced_forman(dome) -> np.ndarray:
    """`[edges]` balanced Forman curvature over the whole graph."""
    neighbours = [set(dome.neighbours(c.id)) for c in dome.cells]
    return np.array([edge_curvature(neighbours, e.u, e.v) for e in dome.edges])


def curvature_section(dome, resistance: np.ndarray) -> None:
    curvature = balanced_forman(dome)
    print("\n### balanced Forman curvature, by edge kind and by stalk width\n")
    print(
        f"  {'edges':>20} {'count':>6} | {'mean':>8} {'median':>8} {'min':>8} "
        f"{'max':>8} | {'negative':>9}"
    )
    groups = defaultdict(list)
    for e in dome.edges:
        groups[f"{e.kind.value} (m={e.m})"].append(e.id)
    groups["all"] = [e.id for e in dome.edges]
    for name in sorted(groups):
        ids = groups[name]
        values = curvature[ids]
        share = 100.0 * float((values < 0).mean())
        print(
            f"  {name:>20} {len(ids):>6} | {values.mean():8.4f} "
            f"{np.median(values):8.4f} {values.min():8.4f} {values.max():8.4f} | "
            f"{share:8.1f}%"
        )

    print("\n  most negatively curved edges -- Topping et al.'s over-squashing suspects")
    for edge_id in np.argsort(curvature)[:8]:
        e = dome.edges[int(edge_id)]
        print(
            f"    {str(dome.cells[e.u].index):>18} -> {str(dome.cells[e.v].index):<18} "
            f"{e.kind.value:>8} m={e.m}  Ric {curvature[edge_id]:8.4f}  "
            f"R(u,v) {resistance[e.u, e.v]:7.4f}"
        )

    # The two diagnoses should agree if they are seeing the same thing.
    pairs = np.array([resistance[e.u, e.v] for e in dome.edges])
    correlation = float(np.corrcoef(curvature, pairs)[0, 1])
    print(
        f"\n  curvature against per-edge effective resistance: r = {correlation:+.4f} "
        f"over {len(pairs)} edges"
    )


def main() -> None:
    start = time.perf_counter()
    dome = build_graph()
    print(
        f"the real dome: {len(dome.cells)} cells "
        f"({len(dome.predicting)} predicting, {len(dome.boundary)} boundary), "
        f"{len(dome.edges)} edges, chi = {dome.euler_characteristic}"
    )
    readings = resistance_section(dome)
    gauge_section(dome)
    stalk_section()
    curvature_section(dome, readings["unweighted"][2])
    print(f"\n({time.perf_counter() - start:.1f} s wall)")


if __name__ == "__main__":
    main()
