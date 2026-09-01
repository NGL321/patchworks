"""Does what arrives on an edge align worse than average with the next hop? (ticket #244)

[#233](https://github.com/NGL321/patchworks/issues/233) read
[#184](https://github.com/NGL321/patchworks/issues/184)'s three parked candidates for the
per-hop grading and closed two of them. What it could not read is the **unfavourable**
side of the third: alignment headroom prices what the *best* direction buys over an
average one and is at most 2.83x on every hop measured, but what a *badly chosen*
direction costs is unbounded below, and that is the form the candidate would actually
take. This script reads it::

    python benchmarks/alignment_read.py null      # no run; the null's own shape
    python benchmarks/alignment_read.py align     # ~25 min; trains 30k ticks

Nothing here decides anything. It explains a measurement; the remedy species this
grading belongs to is closed (#184, #230). Its value is the same as #233's: it tells the
retention recharter whether per-edge variation is a construction fact or a learned one.

**Why #233 could not read it, and what changed.** #233's measured quantity is a
*windowed peak* — `max` over 64 ticks of a paired ratio — sitting ~480x above the
isotropic single-step prediction. That level shift swamps the sign of a per-hop
direction effect. The confound is the window, and it was checked rather than assumed:
the gap is flat in relay degree (`r = -0.121`, 408x-748x across degrees 5-9), so it is
the window's accumulation and not the graph's parallel routes. A **tick-aligned
single-step** read has no window to accumulate over, so the arriving direction's
placement against the hop operator is directly readable.

**What a hop is, exactly** — unchanged from #233, and the ticket names the operator::

    M(e_in -> v -> e_out)  =  F_{v,e_out} . gain_v . F_{v,e_in}^T

`Sheaf.message_passing_phase` writes `x_v <- x_v - gain_v . sum_e F_ev^T (F_ev x_v - y_e)`
and `broadcast` is read from the *pre-update* stalk, so a displacement arriving on
`e_in` at tick `t` reaches `e_out`'s disagreement at tick `t + 1`. That one-tick offset
is the whole of "tick-aligned": every quantity below pairs `d_in(t)` with `d_out(t + 1)`
and never with `d_out(t)`.

**The statistic, and the trap it exists to avoid.** The ticket asks for

    A  =  ||M d|| / (||M||_F ||d|| / sqrt(m_in))

the measured direction against the isotropic one. Read naively, `A < 1` looks like
"worse than average" — and that reading is **wrong**, because it is not what a random
direction scores. For an isotropic `d`, `E[A^2] = 1` exactly, but `A` itself is skewed
below it, and the skew is worst precisely where this graph lives: against a **rank-1**
operator `A = |<v_1, d>| sqrt(m_in)`, so `A^2 / m_in ~ Beta(1/2, (m_in - 1)/2)` and the
median runs 0.816 at `m_in = 4` down to 0.674 as the width grows — never 1. The map's
own standing correction — the maps are near-rank-1, which is what made #142's isotropic
probe report a 1e14 phantom — applies here in its exact dual. So the verdict is never
read off `A` against 1, and it is not read against a graph-wide null either, since the
bar moves with both the width and the operator's rank. It is read off `A` against **the
null `A` for isotropic directions through the same operator**, per hop, and the reported
statistic is the arriving direction's **percentile within that null**. `null` prints the
null's own shape so the trap is visible before any measurement is involved.

**The decomposition, and why it settles the second question in the same read.** Write
`u = F_in^T d` for what the arriving deviation becomes on the node stalk. Then
`||M d|| = gain_v ||F_out u||` exactly, and with

    a_in   =  (||u|| / ||d||)         / (||F_in||_F  / sqrt(m_in))
    a_out  =  (||F_out u|| / ||u||)   / (||F_out||_F / sqrt(perm_v))
    C      =  (||M||_F / sqrt(m_in))  / (gain_v ||F_in||_F ||F_out||_F / sqrt(m_in perm_v))

the identity `A = a_in . a_out / C` holds to floating point. `C` is exactly #233's
**composition** gap — tier 3 over tier 2, median 0.0079x, the finding that a relay's two
incident maps do not share their dominant directions. So the ticket's "a second finding
this may absorb" is not a judgement call to be argued: `C` is a *factor of* `A`, the
composition gap and the arriving-direction gap are algebraically the same quantity read
at two places, and what remains open is only whether they are **statistically**
independent across hops — which is a correlation, and is reported as one.

**What the body does, and the one limit this read carries.** `M` as the ticket defines
it omits the body's Jacobian, which sits between `F_in^T` and `F_out`. `A` is invariant
to the body's *scale* — it cancels top and bottom — so #233's stated body limit does not
bite on the primary reading. It would bite if the body *rotated* `u`. That is not
assumed either way: :func:`single_step_section` reads the empirical one-tick transport,
`M d_in(t)` against the measured `d_out(t + 1)`, and its cosine is the check on whether
`M` is the right operator. A high cosine says the body is not rotating and `M` is the
hop; a low one says it is, and bounds what `a_out` is worth.

**Per edge, never per level** — the map's standing rule. Every aggregate below is over
hops, and the per-relay-level table at the end is printed for shape and is never an
index.
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
import construction_grading as cg  # noqa: E402
import detectability as det  # noqa: E402
import untrained_fixed_point as ufp  # noqa: E402

#: How many isotropic directions the per-hop null is drawn from. The null is a
#: function of the operator's singular values alone (see :func:`null_sample`), so
#: this is a numpy reduction over a handful of values and not a cost worth
#: economising: 20,000 draws put the 5th and 95th percentiles inside +-1%.
NULL_DRAWS = 20_000

#: Below this the arriving deviation's *direction* is not trustworthy even though
#: its norm is representable: float64 goes subnormal at 2.2e-308 and loses
#: relative precision long before that. Ticks under this floor are dropped and
#: **counted**, because a read that silently discarded most of its samples would
#: be reporting on whichever hops happened to survive.
DIRECTION_FLOOR = 1e-200


# -- the operator, and the null a random direction faces --------------------


def hop_operator(
    dome: Dome,
    maps: RestrictionMaps,
    gains: torch.Tensor,
    edge_in: int,
    cell: int,
    edge_out: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(M, F_in, F_out)` in float64, `M = F_out gain_v F_in^T` as the ticket defines it.

    The two maps come back beside `M` because the decomposition needs them
    separately: `a_in` is a property of `F_in` alone and `a_out` of `F_out`
    alone, and rebuilding them from `M` is not possible.
    """
    m_in, m_out = dome.edges[edge_in].m, dome.edges[edge_out].m
    with torch.no_grad():
        f_in = (
            maps.maps[pair_index(edge_in, cg.side_of(dome, edge_in, cell))][:m_in]
            .detach()
            .double()
        )
        f_out = (
            maps.maps[pair_index(edge_out, cg.side_of(dome, edge_out, cell))][:m_out]
            .detach()
            .double()
        )
        operator = (f_out @ f_in.T) * float(gains[cell])
    return operator, f_in, f_out


def alignment(operator: torch.Tensor, direction: torch.Tensor, m_in: int) -> float:
    """`A = ||M d|| / (||M||_F ||d|| / sqrt(m_in))`, the ticket's own quantity.

    One is what an isotropic direction achieves *in mean square* — never what it
    achieves typically; see :func:`null_sample`, which is the thing `A` is
    actually read against.
    """
    scale = float(operator.norm()) / np.sqrt(m_in)
    if scale <= 0:
        return float("nan")
    norm = float(direction.norm())
    if norm <= 0:
        return float("nan")
    return float((operator @ direction).norm()) / (scale * norm)


def null_sample(
    operator: torch.Tensor, m_in: int, generator: np.random.Generator
) -> np.ndarray:
    """`A` for :data:`NULL_DRAWS` isotropic directions through the same operator.

    Drawn from the singular values rather than by matrix-vector product, which is
    the same distribution exactly and not an approximation of it. For an
    orthonormal right-singular basis `v_i` and `d` uniform on the sphere, the
    coordinates `<v_i, d>` are a normalised Gaussian vector, so

        A^2  =  m_in . sum_i s_i^2 g_i^2 / ( ||g||^2 sum_i s_i^2 )

    with `g` standard normal in `m_in` dimensions. The operator enters only
    through `s`, which is why the null is cheap enough to take per hop instead of
    assuming one shape for the graph.
    """
    values = torch.linalg.svdvals(operator).numpy().astype(np.float64)
    total = float((values**2).sum())
    if total <= 0:
        return np.full(NULL_DRAWS, np.nan)
    gauss = generator.standard_normal((NULL_DRAWS, m_in))
    weights = np.zeros(m_in, dtype=np.float64)
    weights[: len(values)] = values**2
    numerator = (gauss**2) @ weights
    denominator = (gauss**2).sum(axis=1) * total
    return np.sqrt(m_in * numerator / denominator)


def rank_profile(operator: torch.Tensor) -> tuple[float, float]:
    """`(effective rank, sigma_1^2 share)` — how near rank-1 this operator is.

    The null's shape is a function of this and nothing else, so it is reported
    beside the null rather than left to be inferred from it. Effective rank is
    the participation ratio `(sum s^2)^2 / sum s^4`, which is 1 for a rank-1
    operator and `r` for `r` equal singular values.
    """
    values = torch.linalg.svdvals(operator).numpy().astype(np.float64) ** 2
    total = values.sum()
    if total <= 0:
        return float("nan"), float("nan")
    return float(total**2 / (values**2).sum()), float(values.max() / total)


# -- the decomposition, and the identity that ties it to #233 ---------------


def decompose(
    operator: torch.Tensor,
    f_in: torch.Tensor,
    f_out: torch.Tensor,
    gain: float,
    direction: torch.Tensor,
    m_in: int,
    perm: int,
) -> dict:
    """`a_in`, `a_out` and `C`, with `A = a_in . a_out / C` checked rather than claimed.

    `a_in` is the arriving direction against the **inbound** map — how much of
    `d` survives onto the node stalk at all — and is body-independent whatever
    the body does. `a_out` is what the survivor is worth to the **outbound** map,
    and is the half a rotating body would reach. `C` is #233's composition gap,
    the same number their tier-3-over-tier-2 column reports.
    """
    stalk = f_in.T @ direction
    d_norm, u_norm = float(direction.norm()), float(stalk.norm())
    fin_norm, fout_norm = float(f_in.norm()), float(f_out.norm())
    out = dict(
        a_in=float("nan"),
        a_out=float("nan"),
        composition=float("nan"),
        a_in_saturation=float("nan"),
        a_out_over_c=float("nan"),
    )
    if min(d_norm, fin_norm, fout_norm) <= 0:
        return out
    out["a_in"] = (u_norm / d_norm) / (fin_norm / np.sqrt(m_in))
    # How much of `a_in`'s own ceiling the arriving direction takes. The ceiling
    # is `sigma_max(F_in) sqrt(m_in) / ||F_in||_F`, reached only by a direction
    # sitting exactly on the inbound map's leading right-singular vector. At 1
    # the arriving direction has nothing left to gain, and — read beside
    # `rank_in` — the reason is that a rank-1 map leaves it no choice.
    ceiling = float(torch.linalg.matrix_norm(f_in, ord=2)) * np.sqrt(m_in) / fin_norm
    if ceiling > 0:
        out["a_in_saturation"] = out["a_in"] / ceiling
    if u_norm > 0:
        out["a_out"] = (float((f_out @ stalk).norm()) / u_norm) / (
            fout_norm / np.sqrt(perm)
        )
    independent = gain * fin_norm * fout_norm / np.sqrt(m_in * perm)
    if independent > 0:
        out["composition"] = (float(operator.norm()) / np.sqrt(m_in)) / independent
    # The centre of what this read found, as a ratio rather than as two
    # distributions a reader has to compare by eye. When `F_in` is rank 1 the two
    # are **equal by algebra**: `u = F_inᵀ d` is then the same node-stalk
    # direction whatever arrives, so `‖F_out u‖ / ‖u‖` is a constant of the maps,
    # and it is exactly the constant `‖M‖_F` carries. `A = a_in · a_out / C` then
    # collapses to `A = a_in`, and the arriving direction has no purchase at the
    # relay at all. Reported so that claim is measured and not inferred from
    # `a_in` happening to sit at its ceiling.
    if np.isfinite(out["a_out"]) and np.isfinite(out["composition"]):
        if out["composition"] > 0:
            out["a_out_over_c"] = out["a_out"] / out["composition"]
    return out


# -- `null`: the null's own shape, before anything is run -------------------


def null_section(dome: Dome) -> None:
    """What a random direction scores, over every directed hop the graph admits.

    This is the read's own precondition and it needs no training run: if the null
    sat at 1 the ticket's ratio could be read against 1, and it does not. Printed
    first for the same reason #233 printed its predictor's range first — the
    argument that does not depend on a measurement is the one that survives a
    disagreement about the measurement.
    """
    maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
    gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=GAUGE_RHO)
    generator = np.random.default_rng(0)

    print("\n### what an isotropic direction scores, at construction\n")
    print(
        "  A = ||M d|| / (||M||_F ||d|| / sqrt(m_in)). E[A^2] = 1 by construction,\n"
        "  and the median of A is what a typical direction actually gets.\n"
    )
    medians, ranks, shares, widths = [], [], [], []
    for cell_id, cell in enumerate(dome.cells):
        if cell.is_boundary:
            continue
        incident = dome.incident[cell_id]
        for edge_in in incident:
            for edge_out in incident:
                if edge_in == edge_out:
                    continue
                operator, _f_in, _f_out = hop_operator(
                    dome, maps, gains, edge_in, cell_id, edge_out
                )
                m_in = dome.edges[edge_in].m
                sample = null_sample(operator, m_in, generator)
                effective, share = rank_profile(operator)
                medians.append(float(np.median(sample)))
                ranks.append(effective)
                shares.append(share)
                widths.append(m_in)
    medians = np.array(medians)
    ranks, shares, widths = np.array(ranks), np.array(shares), np.array(widths)
    print(f"  over {len(medians)} directed hops the graph admits\n")
    print(f"    {'quantity':>28} {'median':>10} {'p05':>10} {'p95':>10}")
    for label, series in (
        ("null median A", medians),
        ("effective rank of M", ranks),
        ("sigma_1^2 share of ||M||_F^2", shares),
        ("m_in", widths.astype(float)),
    ):
        print(
            f"    {label:>28} {np.median(series):9.3f} "
            f"{np.percentile(series, 5):9.3f} {np.percentile(series, 95):9.3f}"
        )
    print(
        f"\n  a typical direction scores {np.median(medians):.3f}, not 1.000 - so "
        f"`A < 1`\n  is NOT evidence of misalignment, and the read below is against "
        f"this null\n  rather than against 1. The construction maps are near rank-1 "
        f"(share\n  {np.median(shares):.3f}), which is the same fact that made #142's "
        "isotropic probe\n  report a deficit that was not there."
    )


# -- `align`: the tick-aligned read on the binding paths --------------------


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
    """#214's read, keeping the per-tick deviation **vectors** the norms threw away.

    `detectability.branch` already returns `[ticks, edges, m]` of disagreement, so
    the arriving direction is a difference of two traces and not a new
    instrument — which is the ticket's own claim that this read is cheap. The
    path reduction is #214's and #233's unchanged, so the hops read here are the
    hops those two reported on.
    """
    env, agent = det.prepared(name, split, seed, learn)
    dome = agent.dome
    ends = {"rim-to-apex": det.apex(dome), "apex-to-rim": det.rim(dome)}
    picker = np.random.default_rng(seed)
    picks = {d: det.sources(dome, d, picker, trials) for d in ends}
    generator = torch.Generator().manual_seed(seed + 1)
    samples: list[dict] = []
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
            moved = det.branch(
                agent, state, observation, applied, window, (source, deviation * probe)
            )
            # The paired difference, as a vector and at every tick. This is the
            # object #233 reduced to a norm and then to a windowed maximum; both
            # reductions are taken later here, so the same run answers both.
            paired = ((moved - quiet) / probe).detach().double()
            norms = paired.norm(dim=-1).numpy()
            floor = quiet.norm(dim=-1).numpy()
            quotient = np.where(floor > 0, norms / np.maximum(floor, 1e-300), np.inf)
            quotient = np.where(norms == 0, 0.0, quotient)
            ratio = quotient.max(axis=0)
            ufp.restore(agent.sheaf, state)
            value, _target, _edge, path = det.widest_path(dome, ratio, source, targets)
            profiles.append(
                dict(
                    trial=i,
                    direction=direction,
                    bottleneck=float(value),
                    path=path,
                    # Carried so `construction_grading.profile_section` can print
                    # #214's own profile off this run. The paths are #233's by
                    # construction — same seed, same surface, same reduction —
                    # but "by construction" is an argument and the binding edge
                    # is a measurement, and this read corroborates rather than
                    # assumes, as #233 did.
                    ratios=tuple(float(ratio[e]) for e in path),
                )
            )
            for edge_in, cell, edge_out in cg.hops_of(dome, path):
                if dome.cells[cell].is_boundary:
                    continue
                samples.append(
                    dict(
                        trial=i,
                        direction=direction,
                        edge_in=edge_in,
                        cell=cell,
                        edge_out=edge_out,
                        # `[ticks, m_in]` and `[ticks, m_out]`: the arriving
                        # deviation and where it lands one tick later.
                        arriving=paired[:, edge_in, : dome.edges[edge_in].m].clone(),
                        leaving=paired[:, edge_out, : dome.edges[edge_out].m].clone(),
                        peak_tick=int(quotient[:, edge_in].argmax()),
                    )
                )
        print(f"  trial {i + 1}/{trials}: {len(samples)} hops so far", flush=True)
    return dome, samples, profiles, agent


def read_hops(dome: Dome, samples: list[dict], agent) -> tuple[list[dict], dict]:
    """Per hop, the tick-aligned alignment of what arrives against `M`.

    Every tick whose arriving deviation is above :data:`DIRECTION_FLOOR` is read,
    and the hop's own figure is the **median over its ticks** — a hop is one
    sample of the question, not sixty-four, and letting a long-lived hop outvote
    a short-lived one would be the graph-wide-average mistake in the time axis.

    **The identity is checked here and not on the rows, and that is not a detail.**
    `A = a_in · a_out / C` holds tick by tick. A median does not distribute over
    a product, so the four *row* medians do not satisfy it and must not be
    expected to — reading the printed table as though they did would find a
    discrepancy that is a property of the median and not of the arithmetic.
    """
    gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=GAUGE_RHO)
    generator = np.random.default_rng(20244)
    nulls: dict[tuple[int, int, int], np.ndarray] = {}
    rows: list[dict] = []
    dropped = 0
    total = 0
    closure = 0.0

    for sample in samples:
        key = (sample["edge_in"], sample["cell"], sample["edge_out"])
        operator, f_in, f_out = hop_operator(dome, agent.sheaf.maps, gains, *key)
        m_in = dome.edges[key[0]].m
        perm = cg.permitted_width(dome, key[1])
        gain = float(gains[key[1]])
        if key not in nulls:
            nulls[key] = null_sample(operator, m_in, generator)
        null = nulls[key]

        arriving, leaving = sample["arriving"], sample["leaving"]
        per_tick: list[dict] = []
        # `t` against `t + 1`: the one-tick offset is the whole of "tick-aligned",
        # and the last tick has no successor to pair with.
        for tick in range(arriving.shape[0] - 1):
            direction = arriving[tick]
            total += 1
            if float(direction.norm()) < DIRECTION_FLOOR:
                dropped += 1
                continue
            score = alignment(operator, direction, m_in)
            if not np.isfinite(score):
                dropped += 1
                continue
            parts = decompose(
                operator, f_in, f_out, gain, direction, m_in, perm
            )
            landed = leaving[tick + 1]
            predicted = operator @ direction
            cosine, magnitude = float("nan"), float("nan")
            if float(predicted.norm()) > 0 and float(landed.norm()) > 0:
                cosine = abs(
                    float(predicted @ landed)
                    / (float(predicted.norm()) * float(landed.norm()))
                )
                magnitude = float(landed.norm()) / float(predicted.norm())
            # Where the identity actually lives. Checked rather than asserted,
            # the way #233 checks its transport/floor split: if this drifts, the
            # three factors are not the three factors of `A` and the section
            # that reads the composition gap off them means nothing.
            if np.isfinite(parts["composition"]) and parts["composition"] > 0 and score > 0:
                rebuilt = parts["a_in"] * parts["a_out"] / parts["composition"]
                closure = max(closure, abs(rebuilt / score - 1.0))
            per_tick.append(
                dict(
                    tick=tick,
                    alignment=score,
                    percentile=float((null < score).mean() * 100.0),
                    versus_null=score / float(np.median(null)),
                    cosine=cosine,
                    magnitude=magnitude,
                    **parts,
                )
            )
        if not per_tick:
            continue
        row = dict(
            trial=sample["trial"],
            direction=sample["direction"],
            edge_in=key[0],
            cell=key[1],
            edge_out=key[2],
            ticks=len(per_tick),
            null_median=float(np.median(null)),
        )
        # A property of the trained maps, not of any tick — but recorded per hop
        # so it is read against the hop's own alignment rather than as a
        # graph-wide average, which is the Notes' standing rule.
        row["rank_in"], _share_in = rank_profile(f_in)
        row["rank_out"], _share_out = rank_profile(f_out)
        for field in (
            "alignment",
            "percentile",
            "versus_null",
            "a_in",
            "a_out",
            "composition",
            "a_in_saturation",
            "a_out_over_c",
            "cosine",
            "magnitude",
        ):
            series = np.array([t[field] for t in per_tick], dtype=np.float64)
            series = series[np.isfinite(series)]
            row[field] = float(np.median(series)) if len(series) else float("nan")
        # The same read at #233's own reduction, so the two are comparable at the
        # hop rather than only in prose.
        #
        # Looked up **by tick number and not by position**: `per_tick` holds only
        # the ticks that cleared the direction floor, so its indices are not tick
        # indices and positional access would silently report a different tick
        # than #233's — the reduction this row exists to be compared against.
        # Falls back to the nearest surviving tick when the peak itself was
        # dropped, which is a real case at a hop that arrives late.
        peak = min(
            per_tick, key=lambda entry: abs(entry["tick"] - sample["peak_tick"])
        )
        row["peak_tick_offset"] = abs(peak["tick"] - sample["peak_tick"])
        row["alignment_at_peak"] = peak["alignment"]
        row["percentile_at_peak"] = peak["percentile"]
        rows.append(row)

    print(
        f"\n  {len(rows)} hops read over {total} tick-pairs; {dropped} dropped "
        f"below the direction floor ({dropped / max(total, 1):.1%})"
    )
    return rows, dict(identity_closure=closure, ticks=total, dropped=dropped)


def spread(label: str, series: np.ndarray, unit: str = "") -> None:
    series = series[np.isfinite(series)]
    if not len(series):
        print(f"    {label:>34} {'no finite samples':>32}")
        return
    print(
        f"    {label:>34} {np.median(series):9.4g}{unit} "
        f"{np.percentile(series, 5):9.4g}{unit} "
        f"{np.percentile(series, 95):9.4g}{unit}"
    )


def verdict_section(rows: list[dict]) -> None:
    """The ticket's question, answered against the null and not against 1."""
    percentile = np.array([r["percentile"] for r in rows])
    align = np.array([r["alignment"] for r in rows])
    versus = np.array([r["versus_null"] for r in rows])

    print(f"\n### the arriving direction against the next hop's operator\n")
    print(f"    {'quantity':>34} {'median':>10} {'p05':>10} {'p95':>10}")
    spread("A  measured / isotropic", align, "x")
    spread("A / median A of the null", versus, "x")
    spread("percentile within the null", percentile, "%")

    below = float((percentile < 50).mean() * 100.0)
    print(
        f"\n  {below:.1f}% of hops sit below their own null's median. A read that "
        f"used\n  1.0 as the bar would instead have called "
        f"{float((align < 1).mean() * 100.0):.1f}% of them misaligned - the "
        "difference\n  between those two numbers is the trap `null` prints."
    )
    # The null percentile is uniform under the hypothesis that arriving
    # directions are isotropic, so its own mean is the test: 50 is no effect, and
    # a systematic effect moves it. Reported with the sign spelled out, because
    # "worse" and "better" both have consequences and neither is the default.
    mean = float(np.nanmean(percentile))
    print(
        f"\n  mean percentile {mean:.1f}% against 50.0% under isotropy - "
        f"{'worse' if mean < 50 else 'better'} than average by "
        f"{abs(mean - 50):.1f} points"
    )
    peak = np.array([r["percentile_at_peak"] for r in rows])
    print(
        f"  at #233's peak tick instead: mean percentile "
        f"{float(np.nanmean(peak)):.1f}%, median A "
        f"{float(np.nanmedian([r['alignment_at_peak'] for r in rows])):.4g}x"
    )


def decomposition_section(rows: list[dict], checks: dict) -> None:
    """`A = a_in . a_out / C`, and whether the two losses are one phenomenon."""
    a_in = np.array([r["a_in"] for r in rows])
    a_out = np.array([r["a_out"] for r in rows])
    composition = np.array([r["composition"] for r in rows])
    align = np.array([r["alignment"] for r in rows])

    print("\n### where the arriving direction loses it: across the edge, or at the relay\n")
    print(f"    {'quantity':>34} {'median':>10} {'p05':>10} {'p95':>10}")
    spread("a_in   d against the inbound map", a_in, "x")
    spread("a_out  the survivor against F_out", a_out, "x")
    spread("C      #233's composition gap", composition, "x")
    spread("A      = a_in . a_out / C", align, "x")

    # Reported from the per-tick check in `read_hops`, which is where the
    # identity holds. Same discipline as #233's transport/floor split: if it
    # drifts, the three factors are not the three factors of `A` and nothing in
    # this section means what it says.
    print(
        f"\n  the identity closes: max |a_in . a_out / C / A - 1| = "
        f"{checks['identity_closure']:.2e} over {checks['ticks']} tick-pairs"
    )
    print(
        "    checked per tick, where it holds. The four medians above do NOT\n"
        "    satisfy it and are not meant to: a median does not distribute over\n"
        "    a product."
    )

    # Why the arriving direction has no purchase at the relay, measured. A rank-1
    # inbound map sends every arriving direction to the same node-stalk
    # direction, so `a_out` stops being a property of what arrived and becomes a
    # constant of the maps — the same constant `C` reports. Both halves are read
    # here rather than inferred from `a_in` sitting at its ceiling.
    print("\n  why: what the inbound map leaves the arriving direction to choose\n")
    print(f"    {'quantity':>34} {'median':>10} {'p05':>10} {'p95':>10}")
    spread(
        "effective rank of F_in", np.array([r["rank_in"] for r in rows]), ""
    )
    spread(
        "effective rank of F_out", np.array([r["rank_out"] for r in rows]), ""
    )
    spread(
        "a_in / its own ceiling",
        np.array([r["a_in_saturation"] for r in rows]),
        "x",
    )
    spread(
        "a_out / C  (1 = no purchase)",
        np.array([r["a_out_over_c"] for r in rows]),
        "x",
    )
    saturation = np.array([r["a_in_saturation"] for r in rows])
    saturation = saturation[np.isfinite(saturation)]
    ratio = np.array([r["a_out_over_c"] for r in rows])
    ratio = ratio[np.isfinite(ratio)]
    if len(saturation) and len(ratio):
        print(
            f"\n    {float((saturation > 0.99).mean() * 100.0):.1f}% of hops take "
            f"their whole `a_in` ceiling, and {float((np.abs(ratio - 1) < 0.01).mean() * 100.0):.1f}% "
            f"have `a_out = C`\n    to within 1%. Where both hold, `A = a_in` at "
            "its ceiling exactly, and the\n    arriving direction is not choosing "
            "anything — the inbound map is."
        )

    pair = np.isfinite(composition) & np.isfinite(align)
    pair &= (composition > 0) & (align > 0)
    if pair.sum() > 2:
        r = float(
            np.corrcoef(np.log10(composition[pair]), np.log10(align[pair]))[0, 1]
        )
        print(
            f"\n  composition against alignment, in the log: r = {r:+.3f} over "
            f"{int(pair.sum())} hops"
        )
        print(
            "    C is a factor of A, so these are the same quantity read at two\n"
            "    places rather than two candidates. The correlation says whether "
            "a hop\n    whose maps compose badly is also a hop whose arriving "
            "direction lands\n    badly, which is what #244's `may absorb` was "
            "asking."
        )


def single_step_section(rows: list[dict]) -> None:
    """Is `M` the hop? The empirical one-tick transport is the check.

    Read here rather than argued, because the primary statistic is only worth
    what the operator is worth, and #233's 480x window artifact is exactly the
    thing a tick-aligned read has to be shown to have removed.
    """
    cosine = np.array([r["cosine"] for r in rows])
    magnitude = np.array([r["magnitude"] for r in rows])

    print("\n### is `M` the hop? the empirical single step, `M d_in(t)` vs `d_out(t+1)`\n")
    print(f"    {'quantity':>34} {'median':>10} {'p05':>10} {'p95':>10}")
    spread("|cos| predicted against landed", cosine, "")
    spread("||landed|| / ||M d_in||", magnitude, "x")
    print(
        "\n  the magnitude is not expected to be 1: `M` omits the body's scale, "
        "and\n  every other route into e_out contributes to what lands there. What "
        "it is\n  expected NOT to be is #233's 480x, which was the window "
        "accumulating."
    )


def per_hop_section(dome: Dome, rows: list[dict]) -> None:
    """The deliverable, per edge — the map's standing rule, and #233's own format."""
    percentile = np.array([r["percentile"] for r in rows])
    order = np.argsort(percentile)

    for label, picks in (
        ("worst-placed arriving directions", order[:10]),
        ("best-placed arriving directions", order[::-1][:10]),
    ):
        print(f"\n  {label}, by null percentile\n")
        print(
            f"    {'relay':>20} {'m_in':>5} {'m_out':>6} {'A':>9} "
            f"{'null med':>9} {'pctile':>8} {'a_in':>8} {'a_out':>8}"
        )
        for i in picks:
            row = rows[int(i)]
            print(
                f"    {str(dome.cells[row['cell']].index):>20} "
                f"{dome.edges[row['edge_in']].m:>5} "
                f"{dome.edges[row['edge_out']].m:>6} "
                f"{row['alignment']:8.3f}x {row['null_median']:8.3f}x "
                f"{row['percentile']:7.1f}% {row['a_in']:7.3f}x {row['a_out']:7.3f}x"
            )

    print("\n  the same, by relay level — reported for shape, never as an index\n")
    by_level = defaultdict(list)
    for row in rows:
        cell = dome.cells[row["cell"]]
        name = (
            f"L{cell.index.level}"
            if cell.kind is CellKind.PREDICTING
            else cell.kind.value
        )
        by_level[name].append(row["percentile"])
    print(f"    {'relay':>16} {'hops':>6} {'mean pctile':>14} {'sd':>8}")
    for name in sorted(by_level, key=lambda s: (s[0] != "L", s)):
        series = np.array(by_level[name])
        print(
            f"    {name:>16} {len(series):>6} {series.mean():13.1f}% "
            f"{series.std():7.1f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("null", help="what a random direction scores; no run")
    run = commands.add_parser("align", help="the tick-aligned read on the paths")
    run.add_argument("--dome", default="full", choices=("small", "full"))
    run.add_argument("--split", default="train")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--learn", type=int, default=30000)
    run.add_argument("--trials", type=int, default=24)
    run.add_argument("--window", type=int, default=det.WINDOW)
    run.add_argument("--hold", type=int, default=det.HOLD)
    run.add_argument("--probe", type=float, default=det.PROBE)
    arguments = parser.parse_args(argv)

    if arguments.command == "null":
        dome = build_graph()
        print(
            f"the real dome: {len(dome.cells)} cells, {len(dome.edges)} edges, "
            f"chi = {dome.euler_characteristic}"
        )
        null_section(dome)
        return

    dome, samples, profiles, agent = collect(
        arguments.dome,
        arguments.split,
        arguments.seed,
        arguments.learn,
        arguments.trials,
        arguments.window,
        arguments.hold,
        arguments.probe,
    )
    null_section(dome)
    # #233's printout, off this run's own surface: the binding edge and the
    # per-hop profile the alignment below is read along.
    cg.profile_section(dome, profiles)
    rows, checks = read_hops(dome, samples, agent)
    verdict_section(rows)
    decomposition_section(rows, checks)
    single_step_section(rows)
    per_hop_section(dome, rows)


if __name__ == "__main__":
    main()
