"""Does construction alone predict #214's per-hop grading? (ticket #233)

[#214](https://github.com/NGL321/patchworks/issues/214) read detectability along
the median binding path and found per-hop attenuation of **240x, 57x, 16x, 25x,
84x, 9x, 29x** — strongly graded, not interchangeable, and unexplained. This
script asks whether that grading is a **construction fact**, predictable from
`DomeSpec` and the graph without running anything, or whether a residual is left
over that needs a different cause::

    python benchmarks/construction_grading.py predict
    python benchmarks/construction_grading.py regress

Nothing here decides anything. It explains a measurement, and the remedy species
this grading belongs to is already closed (#184, #230). Its value is that it tells
the retention recharter whether per-edge variation is a construction fact or a
learned one.

**What a hop is, exactly.** `Sheaf.message_passing_phase` writes
`x_v ← x_v − gain_v · Σ_e F_evᵀ (F_ev x_v − y_e)`, and `broadcast` is read from the
*pre-update* stalk, so a displacement arriving on edge `e_in` at tick `t` reaches
`e_out`'s disagreement at tick `t + 1`, through the body's inference phase on the
way. One hop through cell `v` is therefore the linear operator

    M(e_in → v → e_out)  =  F_{v,e_out} · J_body(v) · gain_v · F_{v,e_in}ᵀ

and its per-direction gain, for an isotropic arriving direction, is
`‖M‖_F / sqrt(m_in)`. That is the same Frobenius identity
`benchmarks/graph_transmission.py` checks by Monte Carlo, one hop of a path
rather than a graph-wide average.

**The construction-only predictor.** At a predicting cell the mask is the same
for every incident edge (`Dome.restriction_mask` reads one `_permitted[cell]`
regardless of the edge), so the incoming and outgoing permitted subspaces
coincide and nothing is lost to a mismatch. Modelling each map as its Frobenius
norm spread over its permitted entries gives

    P(e_in → v → e_out)  =  body · gain_v · ‖F_in‖_F · ‖F_out‖_F / sqrt(m_in · perm_v)

with `‖F‖_F = rho` at a predicting cell and `1` at a boundary cell's pinned maps,
`gain_v = gamma / max(Σ_e m_e, rho² deg(v))`, and `perm_v = min(n, Σ_e m_e)`.
Every term is read off the built graph. `body` is a constant (#120's
:data:`~graph_transmission.BODY_GAIN`), so it sets the scale and contributes
**nothing to the grading**, which is what this ticket is about.

**Three tiers, because they separate three different causes.** The regression is
run against each in turn and the difference between them is the answer:

1. **construction** — the formula above, gauge norms, nothing measured.
2. **realised norms** — the same formula with the *trained* maps' Frobenius
   norms, which prices how far the transport rule moved them off the gauge.
3. **exact operator** — `‖F_out gain_v F_inᵀ‖_F / sqrt(m_in)` from the trained
   maps themselves, which leaves only the body and the **direction** the channel
   actually arrives in. Beside it `sigma_max(M) / ‖M‖_F · sqrt(m_in)` prices the
   headroom between the isotropic reading and the best-aligned one — #184's
   parked inter-endpoint misalignment candidate, read rather than guessed.

**The measured quantity is a ratio of ratios, and that is load-bearing.** #214
reports `r_e = ‖dev_e‖ / ‖floor_e‖` per edge, so the printed per-hop attenuation
is

    r_i / r_{i+1}  =  (dev_i / dev_{i+1}) · (floor_{i+1} / floor_i)

— transport **times** a floor ratio. Only the first factor is a hop; the second is
a property of what the trained sheaf happens to be standing at on two different
edges, and no construction quantity predicts it. This script reports the two
separately, which is the one thing #214's profile could not.

**Per edge, never per level** — the map's standing rule. Nothing below is
aggregated across edges except the regression itself, whose whole content is the
per-edge residual it leaves behind.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from patchworks.graph import CellKind, Dome, build_graph
from patchworks.restriction import GAUGE_RHO, RestrictionMaps, pair_index
from patchworks.tick import DEFAULT_GAMMA, reconciliation_gain

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.append(_HERE)
import detectability as det  # noqa: E402
import untrained_fixed_point as ufp  # noqa: E402
from graph_transmission import BODY_GAIN  # noqa: E402


# -- the construction-only predictor ----------------------------------------


def side_of(dome: Dome, edge_id: int, cell_id: int) -> int:
    """Which end of the edge the cell is: the index `pair_index` wants."""
    edge = dome.edges[edge_id]
    if edge.u == cell_id:
        return 0
    if edge.v == cell_id:
        return 1
    raise ValueError(f"cell {cell_id} is not an endpoint of edge {edge_id}")


def permitted_width(dome: Dome, cell_id: int) -> int:
    """`perm_v`: how many node-stalk directions may reach any incident edge.

    Read through :meth:`Dome.restriction_mask` rather than off `_permitted`, so
    this stays correct if the mask ever becomes edge-dependent — at which point
    the module docstring's claim that the incoming and outgoing permitted
    subspaces coincide would need re-reading too.
    """
    edge_id = dome.incident[cell_id][0]
    return int(dome.restriction_mask(edge_id, cell_id).sum())


def gauge_norm(dome: Dome, cell_id: int, rho: float = GAUGE_RHO) -> float:
    """What construction says a map's Frobenius norm is, before anything is measured.

    A boundary cell's maps carry the exact gauge — an equality at 1, not a band —
    so they are pinned; a predicting cell's are bounded by `rho` and the transport
    rule grows into that band. Taking the band's ceiling is the same convention
    `graph_transmission.analytic_hop` uses for its saturated sweep.
    """
    return 1.0 if dome.cells[cell_id].is_boundary else rho


def predicted_hop(
    dome: Dome,
    gains: torch.Tensor,
    edge_in: int,
    cell: int,
    edge_out: int,
    norms: torch.Tensor | None = None,
) -> float:
    """`P(e_in → v → e_out)`: tier 1 with `norms=None`, tier 2 with the trained norms."""
    m_in = dome.edges[edge_in].m
    if norms is None:
        g_in = g_out = gauge_norm(dome, cell)
    else:
        g_in = float(norms[pair_index(edge_in, side_of(dome, edge_in, cell))])
        g_out = float(norms[pair_index(edge_out, side_of(dome, edge_out, cell))])
    perm = permitted_width(dome, cell)
    return BODY_GAIN * float(gains[cell]) * g_in * g_out / np.sqrt(m_in * perm)


def exact_operator(
    dome: Dome,
    maps: RestrictionMaps,
    gains: torch.Tensor,
    edge_in: int,
    cell: int,
    edge_out: int,
) -> tuple[float, float]:
    """Tier 3: `(‖M‖_F / sqrt(m_in) · body, sigma_max(M) / ‖M‖_F · sqrt(m_in))`.

    The second number is the **alignment headroom** — what the hop is worth to a
    direction chosen to survive it, against what it is worth to an average one.
    Large headroom is the room #184's misalignment candidate would have to live
    in; small headroom means the channel's direction cannot be the story
    whatever the endpoints do.
    """
    m_in, m_out = dome.edges[edge_in].m, dome.edges[edge_out].m
    with torch.no_grad():
        f_in = maps.maps[pair_index(edge_in, side_of(dome, edge_in, cell))][:m_in]
        f_out = maps.maps[pair_index(edge_out, side_of(dome, edge_out, cell))][:m_out]
        operator = (f_out.double() @ f_in.double().T) * float(gains[cell])
        frobenius = float(operator.norm())
        largest = float(torch.linalg.matrix_norm(operator, ord=2))
    isotropic = BODY_GAIN * frobenius / np.sqrt(m_in)
    headroom = (BODY_GAIN * largest) / isotropic if isotropic > 0 else float("inf")
    return isotropic, headroom


def hops_of(dome: Dome, path: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """`(edge_in, cell, edge_out)` for each interior cell of a path of edges."""
    out = []
    for first, second in zip(path, path[1:]):
        shared = {dome.edges[first].u, dome.edges[first].v} & {
            dome.edges[second].u,
            dome.edges[second].v,
        }
        if len(shared) != 1:
            raise ValueError(f"edges {first} and {second} do not share one cell")
        out.append((first, shared.pop(), second))
    return out


# -- `predict`: the whole graph, before anything is run ---------------------


def predict_section(dome: Dome) -> None:
    """The predictor's own spread, over every directed hop the graph admits.

    If construction is to explain a 9x-to-240x grading it must first *have* one,
    and this is the check that it does before any measurement is involved.
    """
    gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=GAUGE_RHO)
    values, rows = [], []
    for cell in dome.cells:
        incident = dome.incident[cell.id]
        # A boundary cell's stalk is overwritten by the world every tick, so
        # nothing relays through it — the same exclusion `hop_endpoints` makes
        # in `graph_transmission`.
        if len(incident) < 2 or cell.is_boundary:
            continue
        for edge_in in incident:
            for edge_out in incident:
                if edge_in == edge_out:
                    continue
                value = predicted_hop(dome, gains, edge_in, cell.id, edge_out)
                values.append(value)
                rows.append((cell.id, edge_in, edge_out, value))
    values = np.array(values)
    attenuation = 1.0 / values
    print(f"\n### the construction predictor, over {len(values)} directed hops\n")
    print(
        f"  attenuation 1/P  —  min {attenuation.min():.1f}x  "
        f"p05 {np.percentile(attenuation, 5):.1f}x  "
        f"median {np.median(attenuation):.1f}x  "
        f"p95 {np.percentile(attenuation, 95):.1f}x  max {attenuation.max():.1f}x"
    )
    print(f"  spread, max over min: {attenuation.max() / attenuation.min():.2f}x")
    print(
        f"  #214 measured a spread of {240 / 9:.1f}x along one path "
        f"(9x to 240x), which is the number this has to cover"
    )

    # What actually varies. `body` and `gamma` are constants, so the predictor's
    # whole dynamic range comes from three construction quantities, and it is
    # worth seeing which one carries it.
    print("\n  what the predictor's spread is made of")
    print(f"    {'quantity':>22} {'distinct':>9} {'min':>10} {'max':>10} {'range':>8}")
    relays = [c.id for c in dome.cells if not c.is_boundary and dome.degrees[c.id] > 1]
    for label, series in (
        ("gain_v", np.array([float(gains[c]) for c in relays])),
        ("perm_v", np.array([permitted_width(dome, c) for c in relays], dtype=float)),
        ("m_in", np.array([e.m for e in dome.edges], dtype=float)),
    ):
        print(
            f"    {label:>22} {len(np.unique(series)):>9} {series.min():10.5g} "
            f"{series.max():10.5g} {series.max() / series.min():7.2f}x"
        )

    # Grouped by the two things a hop is made of, so the shape of the grading is
    # readable rather than only its range.
    print("\n  attenuation by relay kind and incoming stalk width")
    groups = defaultdict(list)
    for cell_id, edge_in, _edge_out, value in rows:
        kind = dome.cells[cell_id].kind.value
        groups[f"{kind} (m_in={dome.edges[edge_in].m})"].append(1.0 / value)
    print(f"    {'relay':>26} {'hops':>6} {'median':>10} {'min':>10} {'max':>10}")
    for name in sorted(groups):
        series = np.array(groups[name])
        print(
            f"    {name:>26} {len(series):>6} {np.median(series):9.1f}x "
            f"{series.min():9.1f}x {series.max():9.1f}x"
        )


# -- `regress`: the predictor against #214's own read -----------------------


def paired(agent, state, quiet, observation, applied, source, deviation, probe, window):
    """`(dev, floor, ratio)`, each `[edges]`: #214's own read, with its quotient undone.

    :func:`detectability.ratios` forms `dev / floor` and returns only the
    quotient, which is the right object for the predicate and the wrong one here:
    the grading this ticket explains is a grading of *transport*, and a ratio of
    ratios carries the floor's variation along with it.

    **The reduction is #214's, not a cousin of it.** `ratio` is the maximum of
    the quotient over the window, per edge — exactly what
    :func:`detectability.ratios` followed by `max` returns — so the widest path
    computed from it is the path that read reports. `dev` and `floor` are then
    taken **at the tick that maximum lands on**, which is what makes
    `dev / floor == ratio` hold edge by edge, and therefore makes
    `transport x floor_ratio` reconstruct the reported per-hop attenuation
    exactly rather than approximately. Peaking the numerator instead would be a
    different reduction and would find a different path.
    """
    moved = det.branch(
        agent, state, observation, applied, window, (source, deviation * probe)
    )
    numerator = (moved - quiet).norm(dim=-1).numpy() / probe
    denominator = quiet.norm(dim=-1).numpy()
    quotient = np.where(
        denominator > 0, numerator / np.maximum(denominator, 1e-300), np.inf
    )
    quotient = np.where(numerator == 0, 0.0, quotient)
    peak = quotient.argmax(axis=0)
    index = np.arange(numerator.shape[1])
    return numerator[peak, index], denominator[peak, index], quotient.max(axis=0)


def collect(
    name: str,
    split: str,
    seed: int,
    learn: int,
    trials: int,
    window: int,
    hold: int,
    probe: float,
):
    """Run #214's read and keep, per binding-path hop, what the ratio threw away."""
    env, agent = det.prepared(name, split, seed, learn)
    dome = agent.dome
    ends = {"rim-to-apex": det.apex(dome), "apex-to-rim": det.rim(dome)}
    picker = np.random.default_rng(seed)
    picks = {d: det.sources(dome, d, picker, trials) for d in ends}
    generator = torch.Generator().manual_seed(seed + 1)
    observations: list[dict] = []
    profiles: list[dict] = []

    for i in range(trials):
        observation, _info = env.reset(seed=seed * 1000 + i)
        agent.observe(observation)
        applied = np.zeros(env.action_space.shape, dtype=np.float64)
        det.hold_still(agent, observation, applied, hold)
        state = ufp.snapshot(agent.sheaf)
        quiet = det.branch(agent, state, observation, applied, window, None)
        for direction, targets in ends.items():
            source = picks[direction][i]
            deviation = det.unit(dome.cells[source].stalk, generator)
            dev, floor, ratio = paired(
                agent,
                state,
                quiet,
                observation,
                applied,
                source,
                deviation,
                probe,
                window,
            )
            ufp.restore(agent.sheaf, state)
            value, _target, _edge, path = det.widest_path(dome, ratio, source, targets)
            profiles.append(
                dict(
                    trial=i,
                    direction=direction,
                    bottleneck=float(value),
                    path=path,
                    ratios=tuple(float(ratio[e]) for e in path),
                )
            )
            for edge_in, cell, edge_out in hops_of(dome, path):
                if dome.cells[cell].is_boundary:
                    continue
                if min(dev[edge_in], dev[edge_out], floor[edge_in], floor[edge_out]) <= 0:
                    continue
                observations.append(
                    dict(
                        trial=i,
                        direction=direction,
                        edge_in=edge_in,
                        cell=cell,
                        edge_out=edge_out,
                        transport=float(dev[edge_out] / dev[edge_in]),
                        floor_ratio=float(floor[edge_in] / floor[edge_out]),
                        reported=float(ratio[edge_out] / ratio[edge_in]),
                    )
                )
        print(f"  trial {i + 1}/{trials}: {len(observations)} hops so far", flush=True)
    return dome, observations, profiles, agent


def median_profile(profiles: list[dict], direction: str) -> dict:
    """The trial #214's `report` would have called the median, same reduction."""
    same = [p for p in profiles if p["direction"] == direction]
    values = np.array([p["bottleneck"] for p in same])
    return same[int(np.argsort(values)[len(values) // 2])]


def regression(x: np.ndarray, y: np.ndarray) -> dict:
    """Ordinary least squares of `y` on `x` in the log, with `R²` and the residual.

    In the log because every factor of a hop is multiplicative, which is the same
    accounting `graph_transmission` uses to split the hop into shares.
    """
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = y - fitted
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((residual**2).sum()) / total if total > 0 else float("nan")
    # The residual of the predictor *as it stands*, with no fitted slope or
    # offset allowed: what is left when the prediction is simply believed.
    raw = y - x
    return dict(
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        residual=residual,
        raw=raw,
        correlation=float(np.corrcoef(x, y)[0, 1]),
    )


def report_regression(label: str, x: np.ndarray, y: np.ndarray) -> dict:
    fit = regression(x, y)
    print(
        f"    {label:>22}  r {fit['correlation']:+.3f}  R² {fit['r2']:6.3f}  "
        f"slope {fit['slope']:+.3f}  |  residual sd {fit['residual'].std():.3f} dex"
        f" ({10 ** fit['residual'].std():.1f}x)  |  unfitted bias "
        f"{fit['raw'].mean():+.2f} dex"
    )
    return fit


def profile_section(dome: Dome, profiles: list[dict]) -> None:
    """#214's own printout, reproduced, so the regression's target is checkable."""
    print("\n### the median trial's path, as #214 prints it\n")
    for direction in ("rim-to-apex", "apex-to-rim"):
        middle = median_profile(profiles, direction)
        edges = "  ".join(f"{r:.3g}" for r in middle["ratios"])
        print(f"  {direction}: bottleneck {middle['bottleneck']:.3g}")
        print(f"    edge by edge from the source — {edges}")
        grading = [
            first / second
            for first, second in zip(middle["ratios"], middle["ratios"][1:])
        ]
        print(
            "    per-hop attenuation — "
            + ", ".join(f"{g:.0f}x" for g in grading)
        )
        print(f"    binding edge: {det.name_edge(dome, middle['path'][-1])}")


def regress_section(dome: Dome, rows: list[dict], agent) -> None:
    gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=GAUGE_RHO)
    norms = agent.sheaf.maps.norms().detach()
    for row in rows:
        key = (row["edge_in"], row["cell"], row["edge_out"])
        row["tier1"] = predicted_hop(dome, gains, *key)
        row["tier2"] = predicted_hop(dome, gains, *key, norms=norms)
        row["tier3"], row["headroom"] = exact_operator(
            dome, agent.sheaf.maps, gains, *key
        )

    transport = np.array([r["transport"] for r in rows])
    floor_ratio = np.array([r["floor_ratio"] for r in rows])
    reported = np.array([r["reported"] for r in rows])

    print(f"\n### what #214's per-hop attenuation is made of, over {len(rows)} hops\n")
    # The split is an identity, not a model, so it is checked rather than
    # asserted in prose: if this drifts from 1 the two factors are not the two
    # halves of the reported number and nothing below means what it says.
    closure = np.abs(transport * floor_ratio / reported - 1.0).max()
    print(f"  the split closes: max |transport x floor / reported - 1| = {closure:.2e}")
    print("  the reported grading is `transport x floor ratio`; only the first is a hop")
    print(f"    {'quantity':>22} {'median':>12} {'p05':>12} {'p95':>12} {'spread':>10}")
    for name, series in (
        ("reported 1/r ratio", 1.0 / reported),
        ("transport 1/hop", 1.0 / transport),
        ("floor ratio", floor_ratio),
    ):
        print(
            f"    {name:>22} {np.median(series):11.1f}x "
            f"{np.percentile(series, 5):11.2f}x {np.percentile(series, 95):11.1f}x "
            f"{np.percentile(series, 95) / np.percentile(series, 5):9.1f}x"
        )
    # A hop three orders past the graded band is not a graded hop, and averaging
    # it in with the rest would be the graph-wide-average mistake at hop scale.
    # Counted rather than trimmed: the count is itself a finding.
    dead = int((1.0 / transport > 1e6).sum())
    print(
        f"\n  {dead} of {len(rows)} hops attenuate by more than 1e6x — past the "
        f"graded band entirely, and not the same phenomenon as a 9x-to-240x hop"
    )

    reported_dex = np.log10(1.0 / reported).std()
    floor_dex = np.log10(floor_ratio).std()
    print(
        f"\n  the floor ratio carries {floor_dex:.2f} dex of the "
        f"{reported_dex:.2f} dex the reported grading spans "
        f"({floor_dex / reported_dex:.0%} by sd)"
    )

    print("\n  the three tiers against measured transport, in log10\n")
    y = np.log10(transport)
    fits = {}
    for tier, label in (
        ("tier1", "construction only"),
        ("tier2", "+ realised norms"),
        ("tier3", "+ exact operator"),
    ):
        fits[tier] = report_regression(
            label, np.log10(np.array([r[tier] for r in rows])), y
        )
    print("\n    and against the reported ratio-of-ratios, for comparison")
    report_regression(
        "construction only",
        np.log10(np.array([r["tier1"] for r in rows])),
        np.log10(reported),
    )

    headroom = np.array([r["headroom"] for r in rows])
    print(
        f"\n  alignment headroom sigma_max/isotropic — median {np.median(headroom):.2f}x,"
        f" p95 {np.percentile(headroom, 95):.2f}x, max {headroom.max():.2f}x"
    )
    print(
        "    the room a direction-based account (#184's parked misalignment "
        "candidate) has to work in, on the upside; the downside is a direction in "
        "the kernel and is unbounded"
    )

    # The two gaps the tiers imply, named and measured rather than left to be
    # inferred from three bias figures. They point in opposite directions, and a
    # reading that quoted only the tier-1 residual would see neither.
    tier2 = np.array([r["tier2"] for r in rows])
    tier3 = np.array([r["tier3"] for r in rows])
    composition = tier3 / tier2
    arriving = transport / tier3
    print("\n  the two gaps between the tiers, per hop\n")
    print(f"    {'gap':>34} {'median':>10} {'p05':>10} {'p95':>10}")
    for label, series in (
        ("composition  exact / independent", composition),
        ("direction    measured / exact", arriving),
    ):
        print(
            f"    {label:>34} {np.median(series):9.3g}x "
            f"{np.percentile(series, 5):9.3g}x {np.percentile(series, 95):9.3g}x"
        )
    print(
        "    composition < 1 means a relay's two incident maps do NOT share their\n"
        "    dominant directions, so what arrives on one edge barely projects onto\n"
        "    what leaves on another — the direct route is weaker than an\n"
        "    independent-maps model, not stronger.\n"
        "    direction > 1 means the measured peak exceeds one application of that\n"
        "    operator to an average direction. A one-tick operator cannot be the\n"
        "    whole of a peak-to-peak ratio taken over a window, so this gap is a\n"
        "    lower-bound artifact plus whatever the graph's parallel routes carry —\n"
        "    the two are separated by the degree reading below, not by assertion."
    )

    # Which of the two the `direction` gap is. If parallel routes carry it, the
    # gap grows with how many routes a relay has; if it is the window's
    # accumulation, the gap is flat in degree and the correlation is nil. Either
    # way this is reported per hop and never as a graph-wide claim.
    degree = np.array([float(dome.degrees[r["cell"]]) for r in rows])
    print("\n  is the `direction` gap the graph's parallel routes, or the window?\n")
    print(
        f"    gap against relay degree: r = "
        f"{float(np.corrcoef(degree, np.log10(arriving))[0, 1]):+.3f} "
        f"over {len(rows)} hops, degrees {degree.min():.0f} to {degree.max():.0f}"
    )
    print(f"    {'relay degree':>14} {'hops':>6} {'median gap':>13}")
    by_degree = defaultdict(list)
    for value, deg in zip(arriving, degree):
        by_degree[int(deg)].append(value)
    for deg in sorted(by_degree):
        series = np.array(by_degree[deg])
        print(f"    {deg:>14} {len(series):>6} {np.median(series):12.4g}x")

    # Per edge, never per level. The residual is the ticket's deliverable, so it
    # is reported at the hop rather than only as a standard deviation.
    print("\n  the worst-predicted hops, by tier-1 residual (measured / predicted)\n")
    order = np.argsort(np.abs(fits["tier1"]["raw"]))[::-1][:10]
    print(
        f"    {'relay':>20} {'m_in':>5} {'m_out':>6} {'measured':>10} "
        f"{'tier1':>10} {'residual':>13}"
    )
    for i in order:
        row = rows[int(i)]
        print(
            f"    {str(dome.cells[row['cell']].index):>20} "
            f"{dome.edges[row['edge_in']].m:>5} {dome.edges[row['edge_out']].m:>6} "
            f"{1.0 / row['transport']:9.1f}x {1.0 / row['tier1']:9.1f}x "
            f"{fits['tier1']['raw'][int(i)]:+9.2f} dex"
        )

    print("\n  the same, by relay level — reported for shape, never as an index\n")
    by_level = defaultdict(list)
    for row, raw in zip(rows, fits["tier1"]["raw"]):
        cell = dome.cells[row["cell"]]
        label = (
            f"L{cell.index.level}"
            if cell.kind is CellKind.PREDICTING
            else cell.kind.value
        )
        by_level[label].append(raw)
    print(f"    {'relay':>16} {'hops':>6} {'mean residual':>15} {'sd':>8}")
    for name in sorted(by_level, key=lambda s: (s[0] != "L", s)):
        series = np.array(by_level[name])
        print(
            f"    {name:>16} {len(series):>6} {series.mean():+14.2f} dex "
            f"{series.std():7.2f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("predict", help="the construction predictor, on its own")
    run = commands.add_parser("regress", help="the predictor against #214's read")
    run.add_argument("--dome", default="full", choices=("small", "full"))
    run.add_argument("--split", default="train")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--learn", type=int, default=30000)
    run.add_argument("--trials", type=int, default=24)
    run.add_argument("--window", type=int, default=det.WINDOW)
    run.add_argument("--hold", type=int, default=det.HOLD)
    run.add_argument("--probe", type=float, default=det.PROBE)
    arguments = parser.parse_args(argv)

    if arguments.command == "predict":
        dome = build_graph()
        print(
            f"the real dome: {len(dome.cells)} cells, {len(dome.edges)} edges, "
            f"chi = {dome.euler_characteristic}"
        )
        predict_section(dome)
        return

    dome, rows, profiles, agent = collect(
        arguments.dome,
        arguments.split,
        arguments.seed,
        arguments.learn,
        arguments.trials,
        arguments.window,
        arguments.hold,
        arguments.probe,
    )
    predict_section(dome)
    profile_section(dome, profiles)
    regress_section(dome, rows, agent)


if __name__ == "__main__":
    main()
