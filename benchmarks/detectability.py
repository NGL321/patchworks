"""Rim-to-core transmission: ADR-0026's conduction ratio, and ADR-0021's bottleneck.

**Two predicates, two quantities, and this rig publishes both under their own
names.** ADR-0026's **conduction ratio** `τ̂_c / world_loop(c)` is #127's operative
bar; ADR-0021's **bottleneck ratio** is the sufficient per-edge diagnostic it
demoted but kept. Neither decides anything here — both predicates, the amplitude
convention, the floor, the bar and the loop enumeration were settled before this
file computed either, and this is the read that returns a verdict against them::

    python benchmarks/detectability.py read
    python benchmarks/detectability.py corners
    python benchmarks/detectability.py linearity

**`read` carries the cutoff hook** (#284). Open problems cut on this rig —
[#325](https://github.com/NGL321/patchworks/issues/325) and
[#329](https://github.com/NGL321/patchworks/issues/329) both write
`conduction ratio >= 1`, and
[#341](https://github.com/NGL321/patchworks/issues/341) writes `outbound
conduction ratio >= 1` — because they wait on the same precondition: *a graph
that transmits*. `read` states the verdict against each, records the run on the
problem, and adds `register:overdue` on a crossing. It asserts nothing and its
exit code does not move; `tools/cutoff_report.py` holds the argument, and the
reading it hands over is :func:`readings`.

**What `conduction ratio` names, corrected by
[#379](https://github.com/NGL321/patchworks/issues/379).** It is **ADR-0026's**
`τ̂_c / world_loop(c)` — :func:`conducting_path`, a ratio of times — and it is
*not*
ADR-0021's bottleneck ratio, which this module published under that key until
#379. The two are not close and they fail differently: at #232's best corner the
bottleneck ratio was short of its bar by **4.4e8x** where the conduction ratio is
short by roughly **1.1x-8.5x** (#274). Wiring a cutoff to the first while its own
warrant cites the second is a false FAIL that could hold a problem closed
indefinitely past the point its real precondition is met — the bystander ADR-0026
replaced the amplitude bar to be rid of. ADR-0021's ratio is still published,
under `bottleneck_ratio`; only the name it wore was wrong.

Pass `--no-file` on any read that is not *the* read — a short `--learn`, the
small dome — because filing records a run against every problem that cuts on
this rig, and a toy reading is not one.

**`corners` is #232 and it is the one thing here that chose something.** #214
read one corner of a two-axis space, and #230 found it the corner least
favourable to what the architecture claims: the stimulus is added once and the
rim sources are written boundary cells, so it cannot persist; and it enters at
one stalk, where #230 rules inbound rim influence is *collective*. `corners`
varies both axes and reports all four against #214's published baseline, which
stays reproducible because `read` is untouched. The predicate, the floor, `k`,
the median bar and per-edge indexing are ADR-0021's throughout — what changes is
the **stimulus convention** and nothing else. See :func:`injection` for what
*collective* was taken to mean, which is this file's only open call, and
:func:`branch` for what *sustained* was taken to mean.

**ADR-0026's predicate, the operative bar.** `max` over paths `P`, `min` over
*cells* `c ∈ P`, of `τ̂_c / world_loop(c) ≥ 1`; per trial, the bar at the median
with p05/p25/p75/p95 alongside; stated twice, rim→apex and apex→rim. `τ̂_c` is
the e-fold decay time of the paired deviation **projected onto `c`'s private
features** (:func:`tau_hat`), and `world_loop(c)` is the tick length of the
shortest loop through `c` that leaves through the actuator, crosses the world
and re-enters at another sensory boundary cell, **enumerated from the mask**
(:func:`world_loops`) rather than quoted. Boundary cells hold no private
features and carry no `τ̂`, so they bound no path — ADR-0026's own exclusion,
arrived at from the arithmetic rather than bolted on.

**The divisor is `world_loop(c)` since
[#383](https://github.com/NGL321/patchworks/issues/383)**, and it is imported
from `benchmarks/loop_length.py` rather than re-derived here — one enumeration,
one place. `|loop(c)|`, the graph's own round trip, is **kept and demoted**: it
is still computed by :func:`loop_lengths` and still exact, but it is a
construction-time length and no longer a denominator. The two differ at every
cell, by 1 to 7 ticks on `DEFAULT_SPEC`, and #383 ruled that the loop
ADR-0026's own justification names — *the cell still holds what it sent by the
time the answer gets back* — is the one that crosses the world.

**The published `conduction_ratio` key does not move under that swap**, and
that is not the edit failing to bite. It reads `0` under either divisor because
[#385](https://github.com/NGL321/patchworks/issues/385)'s zero-private-dimension
L1 cells pin the `min`. What moves is the per-cell ratios along the path
:func:`report` prints, which is where the ruling is visible.

**This is the widest-path shape, not the acceptance form.** ADR-0026 states the
predicate twice more — inbound as a swept per-stratum count, outbound as a
universal over L1 and the actuator — and that reading is #99's, not this rig's.
What is here is the shape both of those clauses reduce a trial with.

**ADR-0021's predicate, the sufficient diagnostic.** `max` over rim-to-apex paths
`P`, `min` over edges `e ∈ P`, of `[A₀ · Π_{i ≤ e} hop_i] / floor_e ≥ 1`, with
`A₀ = 1` and `k = 1`; per trial, each trial reduced to the peak ratio at that
edge; the same median bar and quantiles. Kept, published, and no longer what the
map is read against.

**The numerator is measured, not chained.** `A₀ · Π_{i ≤ e} hop_i` is *the
arriving deviation at edge `e`*, and with `A₀ = 1` a unit-norm deviation injected
at the source stalk makes the paired counterfactual read that product directly.
Multiplying seven separately-measured hops would assume the channel is the
product of graph-wide averages, which is the aggregation error #142 paid for.

**The paired counterfactual, and what the pairing buys.** Two branches forked
from one state, ticked against the same held world, differenced at the same tick:

    dev_e(t)   = ‖ d_e(t) | perturbed  −  d_e(t) | unperturbed ‖
    floor_e(t) = ‖ d_e(t) | unperturbed ‖

where `d_e` is :meth:`~patchworks.tick.Sheaf.disagreement`, the edge's own term of
`xᵀLx`. Both are the same object in the same lane units — one the
perturbation's contribution to it, the other what stands on the edge without the
perturbation — which is what makes the ratio well-formed at all, and it is the
defect ADR-0021 exists to repair. The floor is read **at the same tick** as the
numerator, which is where ADR-0021's `k = 1` gets its conservatism: excursions
are in the denominator rather than in an invented margin.

**Held world, because the floor is a quiescent-hold floor.** The numerator is
read under the same condition as the denominator: the world is not stepped, the
same observation is written every tick, and what is left is the graph. That
drains the lag floor, which is the driven part ADR-0007 separates and the part
ADR-0021 excludes — including it would make the perturbation beat a quantity it
partly causes. It also makes the counterfactual exact: both branches receive
byte-identical external writes, so every difference between them is the
perturbation, and two unperturbed branches differ by exactly zero.

**Trials sweep configurations.** `reset()` rearranges the world and never the
agent, so a trial is a new pose on the same surface. ADR-0007's static floor is
positional and one pose reports on one point of the overlap; that obligation has
been standing since ADR-0007 and this is the first read to honour it.

**Per edge, never per level** — the Notes' standing rule and ADR-0021's, one
level down from the rule against graph-wide averages. Nothing here is aggregated
across edges: the path reduction is a `min` along one path and a `max` across
paths, both of which pick out an edge rather than averaging over any set of them,
and the profile printed for the median trial names the edges of one path in
order rather than summarising a level.

**The read is taken in float64, and that is not a detail.** See
:func:`double_precision`. Taken in the float32 the architecture runs in, this
measurement reports its own rounding — on the real dome, `2.5e-4` where the
quantity is `1.1e-9`. `linearity` is the check that separates the two, and it is
worth running before believing any number here.

**So `τ̂` carries #224's gate** (:func:`readable`). #224 ruled float32's
granularity the architecture's **noise floor**: a difference smaller than the
representable granularity at the magnitude the state sits at has not arrived. A
`τ̂` whose `1/e` crossing sits below `eps_f32 · ‖state‖` is a float64 reading of
something the float32 build has no signal for, so it is reported **as unreadable
at runtime precision, alongside the number** rather than dropped — the float64
`τ̂` stays visible next to the fact that the running architecture has nothing
there. ADR-0026's amendment carrying that clause is
[#381](https://github.com/NGL321/patchworks/issues/381)'s, and so is the
constant's definition site; see :data:`EPS_F32`.

**What the widest path is.** Edge values in hand, the predicate is the classical
maximum-bottleneck path, computed exactly by a max-min Dijkstra rather than
sampled. `tests/test_detectability.py` holds it against closed forms.
"""

import argparse
import sys
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import torch

from patchworks.agent import Agent
from patchworks.graph import CellKind, Dome
from patchworks import tick

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.append(_HERE)
# The fork's state list, the build and the training loop are imported rather
# than copied: `_TICK_STATE` is held against `Sheaf.assert_no_tape`'s own list by
# `tests/test_untrained_fixed_point.py`, and a second copy drifting from it would
# leave this script forking a branch that shares state with its own control.
import untrained_fixed_point as ufp  # noqa: E402

# ADR-0026's divisor is enumerated in exactly one place (#383). Importing it
# rather than re-deriving it here is the same rule `untrained_fixed_point` is
# imported under: two copies of a construction-time enumeration can disagree,
# and the one that disagreed would be the one under the operative bar.
import loop_length  # noqa: E402

# The cutoff hook lives in `tools/` and not here, because it shells `gh` and a
# network tool belongs on the far side of the line `tests/test_cli.py` defends.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from cutoff_report import report as report_cutoffs  # noqa: E402

#: How many ticks each branch is ticked for after the fork. The dome is seven
#: levels deep and one hop is one tick, so this is several times the crossing.
#: The deviation does **not** decay to nothing — the two branches settle on
#: neighbouring trajectories rather than reconverging — so a longer window is
#: partly a maximum over more excursions and creeps upward with the horizon.
#: `read` prints the horizon ladder rather than leaving that in this constant.
WINDOW = 64

#: Ticks of held world before the fork, draining the lag floor so what is left is
#: ADR-0007's static and settling pair. `untrained_fixed_point.sensitivity`'s own
#: hold, for the same reason and at the same length.
HOLD = 400

#: The amplitude the paired difference is taken at, divided back out before
#: anything is reported. `1.0` is ADR-0021's convention taken literally, and in
#: float64 it sits inside the linear window (`linearity` measures that window as
#: `1e-2` to `1e1` on the real dome), so nothing is being rescaled from anywhere.
#: It is a knob only so that the window can be measured.
PROBE = 1.0

#: float32's granularity: the architecture's arithmetic noise floor (#224).
#:
#: **Imported, not defined.** #379 held the machine quantity here with a note
#: that the definition site was #381's and that this would read from there once
#: it landed; it has, so this is that read. The site is
#: :data:`patchworks.tick.EPS_F32` — `torch.finfo(torch.float32).eps`, evaluated
#: rather than typed per ADR-0018, in a module `tools/constant_registers.py`
#: scans, which `benchmarks/` is not. Two definitions of one number is ADR-0020's
#: failure exactly, and it is the one the gate could least afford: the register
#: row and the reading that gates on it would be free to drift apart.
#:
#: The alias is kept so this module's own uses read as they did.
EPS_F32 = float(tick.EPS_F32)


def double_precision(root, depth: int = 0, seen: set[int] | None = None) -> int:
    """Cast the trained sheaf to float64 for the read. The measurement needs it.

    **Not a change to the architecture, and not a claim that it should run this
    way.** The surface is trained in float32 exactly as it runs; the cast happens
    after training and before the fork, and it changes the precision the
    *measurement* is done at, not the model being measured.

    The reason is one this map has already paid for twice. #146 struck a depth
    claim on the finding that it located *float32's resolving limit, not the
    graph's*, and #183 re-read a drive hop for the same reason. This quantity is
    squarely in that territory: the deviation arriving at a deep edge is `1e-9`
    of a quiescent disagreement of order `1e-1`, which float32 cannot represent
    as a difference at all. Measured in float32 the two branches still differ —
    they differ by the *rounding* of two nearly-equal trajectories, which does
    not attenuate with depth. So the reading acquires a false floor that is
    largest exactly where the signal is smallest, and it flatters the
    architecture: on the real dome the float32 read returns `2.5e-4` for a
    quantity that is `1.1e-9`, five orders of magnitude of pure arithmetic.

    The tell is amplitude-independence, and it is what `linearity` prints. A
    transported deviation is linear in what was injected, so the bottleneck
    reported at `A₀ = 1` is flat in the amplitude it was measured at; rounding is
    not, so the column falls like `1/A₀`. In float32 on the real dome it falls
    like `1/A₀` across **nine** decades — there is no linear window at all, so no
    choice of amplitude rescues the float32 read. In float64 it is flat to five
    significant figures from `1e-2` to `1e1`.

    A crawl rather than an inventory, and checked rather than trusted: an
    explicit list of tensors is a list that drifts, so this walks what the sheaf
    actually holds, and `linearity`'s flat ladder is the evidence that nothing
    float32 was left behind to quantise the result.
    """
    if seen is None:
        seen = set()
    if id(root) in seen or depth > 6:
        return 0
    seen.add(id(root))
    cast = 0
    if isinstance(root, torch.nn.Module):
        root.double()
    for name in dir(root):
        if name.startswith("__"):
            continue
        try:
            value = getattr(root, name)
        except Exception:
            continue
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.float32:
                try:
                    setattr(root, name, value.to(torch.float64))
                    cast += 1
                except Exception:
                    pass
        elif hasattr(value, "__dict__") and not isinstance(value, (str, bytes, type)):
            cast += double_precision(value, depth + 1, seen)
    return cast


@dataclass(frozen=True)
class Trial:
    """One trial, read against both predicates: ADR-0026's and ADR-0021's.

    The two reductions run over the same fork and the same window — one over
    cells and times, the other over edges and amplitudes — so the trial carries
    both verdicts and nothing has to be re-measured to compare them.
    """

    source: tuple[int, ...]
    """Every cell the deviation was injected into: one for #214, a stratum for #232."""

    kind: str
    """The stratum injected into, so the corners can be broken down by it."""

    bottleneck: float
    target: int
    edge: int
    """The edge that binds the widest path — what the predicate says fails."""

    path: tuple[int, ...]
    """The binding path's edges, source-first: the descent, edge by edge."""

    ratios: np.ndarray
    """`[edges]`: the peak paired ratio at each edge."""

    peak_at: np.ndarray
    """`[edges]`: the tick each edge's peak landed on, for the window check."""

    horizons: tuple[tuple[int, float], ...]
    """`(ticks, bottleneck)`: the same trial read to several horizons."""

    conduction: float
    """ADR-0026's reading: `min` over the path's cells of `τ̂_c / world_loop(c)`."""

    cell: int
    """The cell that binds the conducting path — what ADR-0026's predicate says fails."""

    walk: tuple[int, ...]
    """The conducting path's cells, source-first."""

    tau: np.ndarray
    """`[predicting cells]`: `τ̂_c` in ticks, in `Dome.predicting` row order."""

    censored: np.ndarray
    """`[predicting cells]` bool: the window ended before the `1/e` crossing."""

    resolved: np.ndarray
    """`[predicting cells]` bool: #224's gate — the crossing clears float32's floor."""

    private: np.ndarray
    """`[predicting cells]` bool: the cell has private dimension to hold at all (#385)."""

    floor: float
    """`eps_f32 · ‖state‖` at the median cell: what `tick.PRECISION_FLOOR` records."""

    conduction_horizons: tuple[tuple[int, float], ...]
    """`(ticks, conduction)`: `τ̂` is a decay time, so the window is never silent."""


def rim(dome: Dome) -> tuple[int, ...]:
    """The sensorimotor rim: the cells the world writes and reads.

    The drive is excluded. It is a boundary cell and it sits at the *internal*
    rim by `CONTEXT.md`'s own words — at the apex level, not on the sensorimotor
    boundary — so a path that starts or ends there is not the path the predicate
    is about.
    """
    return tuple(
        c.id for c in dome.cells if c.is_boundary and c.kind is not CellKind.DRIVE
    )


def apex(dome: Dome) -> tuple[int, ...]:
    """The predicting cells at the deepest level."""
    deepest = max(c.index.level for c in dome.cells if not c.is_boundary)
    return tuple(
        c.id for c in dome.cells if not c.is_boundary and c.index.level == deepest
    )


def widest_path(
    dome: Dome, values: np.ndarray, source: int | tuple[int, ...], targets: tuple[int, ...]
) -> tuple[float, int, int, tuple[int, ...]]:
    """`max` over paths, `min` over edges: the widest path, where it lands, what stops it.

    A max-min Dijkstra. The invariant that makes the greedy step correct is that
    a path's value cannot rise by extending it, so the unsettled cell of greatest
    value can never be improved by going through a cell of smaller value.

    The **binding edge** comes back with the value, because the predicate's whole
    content is that what fails is an edge rather than a level, and a verdict that
    could not name the edge would not be that predicate's verdict. So does the
    path itself, so the descent can be printed edge by edge.

    The source's own value is `inf` — a zero-edge path is bounded by no edge —
    which is what makes a target adjacent to the source come back with that
    edge's value rather than with `0`.

    **Several sources are the same search, seeded wider.** A collective
    perturbation (#232) has no single source, and the predicate's `max` is over
    rim-to-apex paths, not over paths from one named cell — so every perturbed
    cell starts at `inf` and the search returns the widest path from any of them.
    That is the same object as a virtual super-source joined to each by an
    unbounded edge, and it degenerates to #214's search when there is one.
    """
    origins = (source,) if isinstance(source, int) else tuple(source)
    best = {s: float("inf") for s in origins}
    binding = {s: -1 for s in origins}
    came: dict[int, int] = {}
    frontier = [(-float("inf"), s) for s in origins]
    settled: set[int] = set()
    reached = tuple(t for t in targets if t not in set(origins))
    while frontier:
        value, cell = heappop(frontier)
        value = -value
        if cell in settled:
            continue
        settled.add(cell)
        if cell in reached:
            walk, at = [], cell
            while at in came:
                walk.append(came[at])
                at = dome.edges[came[at]].other(at)
            return value, cell, binding[cell], tuple(reversed(walk))
        for edge_id in dome.incident[cell]:
            other = dome.edges[edge_id].other(cell)
            if other in settled:
                continue
            crossing = float(values[edge_id])
            through = min(value, crossing)
            if through > best.get(other, -float("inf")):
                best[other] = through
                binding[other] = edge_id if crossing <= value else binding[cell]
                came[other] = edge_id
                heappush(frontier, (-through, other))
    return 0.0, -1, -1, ()


def loop_lengths(dome: Dome) -> dict[int, int]:
    """`|loop(c)|` for every predicting cell: `2 · d(c, rim)`, off the mask.

    ADR-0026's divisor, **enumerated rather than quoted**, which is that ADR's
    own instruction: the ladder it prints is `DEFAULT_SPEC`'s reading, and *"any
    graph change re-derives it, and no session should quote 14 at a dome it has
    not checked."* A breadth-first sweep from :func:`rim` — the sensorimotor rim,
    the drive excluded for the reason it is excluded there — gives `d(c, rim)`,
    and the rim-returning round trip is twice it.

    **A predicting cell the sweep never reaches is absent from the mapping**
    rather than carried with an infinite loop length. *No loop closing* is
    ADR-0026's stated falsification, and it should read as one rather than as a
    very large divisor.

    `|loop(c)| = 2 · level` on the default dome is a fact about that taper and
    not a licence to index by level (#181): this computes per cell from the
    graph, and a changed `DomeSpec` gets a different ladder for free.

    **This is no longer the conduction ratio's divisor** (#383). It is kept and
    published under its own name as what it is — an exact construction-time
    length, the graph's own rim-returning round trip — and :func:`world_loops`
    carries the divisor. The demotion cost the reading nothing: `|loop(c)|` is
    still what ADR-0026 enumerated and still what a changed `DomeSpec` moves.
    """
    distance = {cell: 0 for cell in rim(dome)}
    frontier = list(distance)
    while frontier:
        onward = []
        for cell in frontier:
            for edge_id in dome.incident[cell]:
                other = dome.edges[edge_id].other(cell)
                if other not in distance:
                    distance[other] = distance[cell] + 1
                    onward.append(other)
        frontier = onward
    return {c: 2 * distance[c] for c in dome.predicting if c in distance}


def world_loops(dome: Dome) -> dict[int, int]:
    """`world_loop(c)` for every predicting cell: ADR-0026's divisor since #383.

    `min over (a, p), a an actuator, p any sensory boundary cell, a != p, of
    d(c, a) + w + d(p, c)` — out through the actuator, across the world, back in
    at another boundary cell, where `a != p` is ADR-0016's written-or-read ban
    rather than a modelling choice. This is the loop ADR-0026's justification
    names, and [#383](https://github.com/NGL321/patchworks/issues/383) ruled it
    the divisor in place of `|loop(c)|`, which it kept and demoted.

    **Delegated to `benchmarks/loop_length.py`, not re-derived.** That module
    owns the enumeration and the world tick `w` the sandbox fixes, and a second
    copy here could drift from it in silence — under the operative bar, which is
    the one place a silent disagreement costs the most.

    **A predicting cell no (actuator, sensory) pair reaches is absent** rather
    than carried with an infinite divisor, for :func:`loop_lengths`' reason: *no
    loop closing* is ADR-0026's stated falsification and it should read as one.
    """
    return loop_length.world_loops(dome).lengths


#: The relative slack on `τ̂`'s `1/e` comparison, so a sample landing *on* the
#: threshold counts as having crossed it (#381). Order the double's own
#: resolution — about 4.5e4 times `eps_f64`, which is nothing this instrument can
#: resolve and far more than the 1-ULP disagreement it exists to absorb. Private
#: to this reduction and not a knob: a value large enough to change a `τ̂` would
#: be interpolating between samples, which ADR-0026 forbids.
_CROSSING_SLACK = 1e-12


def tau_hat(deviation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`τ̂` per cell: peak-to-`1/e` in ticks, ADR-0026's reading taken literally.

    `deviation` is `[ticks, cells]` — the norm of the paired deviation projected
    onto each cell's private features. Per cell: find the peak tick, then read
    the ticks from that peak until the deviation first falls to `1/e` of it.
    Comes back as `(τ̂, censored, peak_at)`.

    **Integer ticks, and no interpolation between samples.** ADR-0026 says *"the
    ticks from that peak until it falls to `1/e` of peak"*, and the tick is the
    architecture's own unit — one edge, one tick
    (`01-cell-and-sheaf.md`, *Unit delay*), which is also what `world_loop(c)`
    counts, so the ratio is in the same currency top and bottom. Interpolating a
    crossing between two samples would report a resolution the instrument does
    not have, and at L1, where the divisor is as small as 3, the interpolated
    digit would be doing the deciding.

    **Censoring is flagged, not filled in.** A deviation that never falls to
    `1/e` inside the window has a `τ̂` this window cannot read. It comes back as
    the ticks remaining after the peak — a **lower bound** — with the flag set,
    so a reading resting on one is visible. A lower bound under-reports `τ̂`, so
    it can cost a PASS and can never manufacture one; that asymmetry is why the
    censored value is usable at all.

    **A cell the deviation never reaches gets `τ̂ = 0`**, not a decay time. There
    is nothing to decay, and `0 / |loop|` is the falsification reading rather
    than a small pass.

    **A sample sitting exactly on the threshold has crossed** (#381). The
    comparison carries :data:`_CROSSING_SLACK`, a relative tolerance of order the
    double's own resolution, because a sample landing *on* `peak / e` is a 1-ULP
    question and numpy answers it differently on different hardware: it dispatches
    `exp` on the runner's SIMD capability, and on an exact exponential — which is
    what `tests/test_detectability.py` feeds — the two answers differ. The same
    commit went green on one CI runner and red on another with `τ̂` reading `9`
    where the decay constant is `8`.

    **It fails in the direction that matters, which is why it is fixed rather
    than tolerated.** Missing the crossing reads `τ̂` one tick **long**, and long
    is the direction that manufactures a PASS on a bar of exactly `1` — the same
    asymmetry #224 ruled on, arriving through the crossing test itself. The
    censoring branch above is already careful to fail the safe way; this is that
    care applied to the comparison.

    **It is not interpolation.** ADR-0026's reading stays integer ticks and whole
    samples: the tolerance decides only what *equality* at a sample means, and it
    is far below any difference the instrument could resolve. Nothing moves
    except a boundary case that was previously decided by rounding.
    """
    ticks, cells = deviation.shape
    peak_at = deviation.argmax(axis=0)
    peak = deviation.max(axis=0)
    tau = np.zeros(cells, dtype=float)
    censored = np.zeros(cells, dtype=bool)
    for cell in range(cells):
        start = int(peak_at[cell])
        if not peak[cell] > 0.0:
            continue
        threshold = (peak[cell] / np.e) * (1.0 + _CROSSING_SLACK)
        below = np.nonzero(deviation[start:, cell] <= threshold)[0]
        if below.size:
            tau[cell] = float(below[0])
        else:
            tau[cell] = float(ticks - 1 - start)
            censored[cell] = True
    return tau, censored, peak_at


def readable(
    deviation: np.ndarray, state: np.ndarray, tau: np.ndarray, peak_at: np.ndarray
) -> np.ndarray:
    """#224's gate: is the `1/e` crossing above float32's granularity?

    `[cells]` bool. The projected deviation at the crossing must stand above
    `EPS_F32 · ‖state‖` at that cell and tick, where `state` is `[ticks, cells]`
    of the **unperturbed** branch's node-stalk norm — the magnitude the state
    actually sits at, which is what fixes the granularity.

    **A False is reported alongside the `τ̂`, never in place of it.** The read is
    taken in float64 (:func:`double_precision`); a crossing below the float32
    floor means the number is real but the running architecture has no signal
    there, and #224's ruling is that this is the architecture's **noise floor**
    rather than a defect to engineer away. Dropping the reading would hide the
    pressure that ruling puts on this instrument; printing it beside the `τ̂` is
    what carries it.

    Nothing here is invented: `EPS_F32` is the machine's and `‖state‖` is read.
    """
    ticks = deviation.shape[0]
    crossing = np.minimum(peak_at + tau.astype(int), ticks - 1)
    columns = np.arange(deviation.shape[1])
    return deviation[crossing, columns] > EPS_F32 * state[crossing, columns]


def conducting_path(
    dome: Dome,
    ratio: dict[int, float],
    source: int | tuple[int, ...],
    targets: tuple[int, ...],
) -> tuple[float, int, int, tuple[int, ...]]:
    """`max` over paths, `min` over cells: ADR-0026's predicate, cell-indexed.

    :func:`widest_path`'s max-min Dijkstra with the value moved off the edges and
    onto the cells, which is the whole of the difference between the two
    predicates' shapes. Same invariant makes the greedy step correct: a path's
    value cannot rise by extending it.

    `ratio` carries `τ̂_c / world_loop(c)` for every predicting cell. **A cell absent
    from it is unbounded** — that is the boundary cells, which hold no private
    features, carry no `τ̂` and are excluded from the outbound universal by
    ADR-0026 on exactly this ground. They are transited without binding, so a
    path through one is bounded by the predicting cells at its ends. A predicting
    cell whose loop never closes is present with `0.0`, which is the
    falsification and not an absence.

    The **binding cell** comes back with the value, for ADR-0021's reason one
    level over: the predicate's content is that what fails is a *cell*, and a
    verdict that could not name it would not be the predicate's verdict. So does
    the path, in cells, so the descent can be printed.
    """
    origins = (source,) if isinstance(source, int) else tuple(source)

    def value_at(cell: int) -> float:
        return ratio.get(cell, float("inf"))

    best: dict[int, float] = {}
    binding: dict[int, int] = {}
    came: dict[int, int] = {}
    frontier: list[tuple[float, int]] = []
    for start in origins:
        best[start] = value_at(start)
        binding[start] = start if value_at(start) < float("inf") else -1
        heappush(frontier, (-best[start], start))
    settled: set[int] = set()
    reached = tuple(t for t in targets if t not in set(origins))
    while frontier:
        value, cell = heappop(frontier)
        value = -value
        if cell in settled:
            continue
        settled.add(cell)
        if cell in reached:
            walk, at = [cell], cell
            while at in came:
                at = came[at]
                walk.append(at)
            return value, cell, binding[cell], tuple(reversed(walk))
        for edge_id in dome.incident[cell]:
            other = dome.edges[edge_id].other(cell)
            if other in settled:
                continue
            through = min(value, value_at(other))
            if through > best.get(other, -float("inf")):
                best[other] = through
                binding[other] = other if value_at(other) <= value else binding[cell]
                came[other] = cell
                heappush(frontier, (-through, other))
    return 0.0, -1, -1, ()


def conduction(
    dome: Dome,
    quiet: dict[int, torch.Tensor],
    moved: dict[int, torch.Tensor],
    source: tuple[int, ...],
    targets: tuple[int, ...],
    loops: dict[int, int],
    window: int | None = None,
) -> dict:
    """One trial's conduction reading: `τ̂` per cell, then the widest path.

    `quiet` and `moved` are the two branches' recorded node stalks, `[ticks,
    stalk]` per cell, and every predicting cell must be in both — the reduction
    is over the graph's cells and a missing one would silently shorten the `min`.

    **The projection is the masks', not a choice.** `Dome.private_projection` is
    fixed at construction and invariant under learning, and it keeps exactly the
    directions reconciliation cannot move (ADR-0026, *How it is read*).

    **A cell with no private dimension reads `τ̂ = 0`, and that is a structural
    zero rather than a measurement.** `private_dimensions` is `max(0, n - Σ_e
    m_e)` and it is a gradient (`06-graph-topology.md`); where it is `0` the
    projection is the zero map, so the deviation has nowhere to be read and no
    amount of retention can move the number. This is taken **literally**, because
    ADR-0026's predicate is settled and an exclusion invented here would be a
    second predicate wearing its name — but it is flagged all the way up to
    :func:`report`, because a bar pinned by construction is exactly the bystander
    ADR-0026 replaced the amplitude ratio to be rid of. See
    [#385](https://github.com/NGL321/patchworks/issues/385), which owns it.
    """
    cells = dome.predicting
    projection = dome.private_projection.to(torch.float64)
    rows = {cell: row for row, cell in enumerate(cells)}
    private = np.stack(
        [
            ((moved[c] - quiet[c])[:window] * projection[rows[c]])
            .norm(dim=-1)
            .numpy()
            for c in cells
        ],
        axis=1,
    )
    state = np.stack([quiet[c][:window].norm(dim=-1).numpy() for c in cells], axis=1)
    tau, censored, peak_at = tau_hat(private)
    above = readable(private, state, tau, peak_at)
    ratio = {
        cell: (tau[rows[cell]] / loops[cell]) if cell in loops else 0.0
        for cell in cells
    }
    value, target, cell, walk = conducting_path(dome, ratio, source, targets)
    return {
        "conduction": value,
        "target": target,
        "cell": cell,
        "walk": walk,
        "tau": tau,
        "censored": censored,
        "readable": above,
        "private": (dome.private_dimensions > 0).numpy(),
        "floor": float(np.median(EPS_F32 * state.max(axis=0))),
        "ratio": ratio,
    }


def unit(width: int, generator: torch.Generator) -> torch.Tensor:
    """`A₀ = 1`: a unit-norm deviation, direction drawn and magnitude fixed."""
    direction = torch.randn((width,), generator=generator, dtype=torch.float64)
    return direction / direction.norm().clamp_min(1e-12)


def injection(
    dome: Dome,
    cells: tuple[int, ...],
    generator: torch.Generator,
    coherent: bool = True,
) -> tuple[tuple[int, torch.Tensor], ...]:
    """`A₀ = 1` spread over several rim cells: what *collective* is taken to mean.

    **This convention is #232's, and it is the ticket's one open call.** ADR-0021
    fixes `A₀ = 1` and #230 rules that inbound rim influence is *collective*;
    neither says what a coherent many-cell deviation is, so it is stated here
    with its reasoning, the way #214 stated its three instrument calls.

    Two clauses, and each is load-bearing:

    1. **Coherent means the same deviation in each cell's own coordinates.** The
       cells of one rim stratum are exchangeable copies of a single sensor type
       placed at different points of the surface — 256 patch cells of identical
       stalk width, built alike — so a disturbance that is uniform over the
       surface is, in their own coordinates, the same vector in each. That is
       the only common coordinate they have: there is no chart over the rim in
       which to write "the same direction" more strongly, and constructing one
       would be inventing structure the graph does not carry. The alternative
       drawn against this — an independent direction per cell — is available as
       `coherent=False`, and running both is what makes the choice a measurement
       rather than an assertion.

    2. **The whole injection is unit-norm, not each cell's share.** Each of `K`
       cells receives `1/√K` of the deviation, so `‖A₀‖ = 1` over the collective
       exactly as over #214's single stalk. Without this the collective corner
       would inject `√K = 16x` more and beat the baseline by arithmetic; the
       question is whether *coherence* transmits, not whether *more* does. This
       is the clause that keeps the four corners comparable, and it is also the
       clause that makes the collective corner a hard test.

    **Strata, not the whole rim.** The collective is all cells of one kind. The
    rim's kinds have stalk widths 48, 2 and 1, so there is no common coordinate
    across them and clause 1 has nothing to mean; kinds also die at different
    levels by #120, which is why #214 stratified in the first place. Patch is
    the only stratum that is genuinely many — proprioception and touch have
    three cells each — and the report breaks the corners down by kind so that
    the 256-cell read is never averaged into two 3-cell ones.
    """
    widths = {dome.cells[c].stalk for c in cells}
    if len(widths) != 1:
        raise ValueError(f"a collective needs one stalk width, got {sorted(widths)}")
    width = widths.pop()
    scale = 1.0 / np.sqrt(len(cells))
    if coherent:
        shared = unit(width, generator) * scale
        return tuple((c, shared.clone()) for c in cells)
    return tuple((c, unit(width, generator) * scale) for c in cells)


def branch(
    agent: Agent,
    state: dict,
    observation: dict,
    applied,
    ticks: int,
    nudge,
    sustained: dict[int, torch.Tensor] | None = None,
    record: tuple[int, ...] = (),
):
    """One branch of the fork: restore, optionally nudge, then hold for `ticks`.

    `[ticks, edges, m]` of disagreement comes back — the whole trace rather than
    a running maximum, because the two branches are differenced tick by tick and
    a maximum taken inside a branch would have thrown the pairing away. With
    `record`, the named cells' stalks come back alongside it, read at the end of
    each tick — which is what a later branch clamps against.

    **`nudge` is a sequence of `(cell, deviation)`**, one pair being #214's read.

    **`sustained` is what makes the stimulus persist.** #214's stimulus does not:
    the deviation is added once, and the rim sources are written boundary cells,
    so `agent.write` puts them back to quiet on the next tick and what is
    measured is an impulse response with the source clamped. Passed the quiet
    branch's own recorded stalks, this holds each source at *its counterfactual
    value plus the deviation* at the end of every tick — a constant offset from
    the control, maintained for the whole window.

    Clamping against the recorded quiet trace rather than re-adding the
    deviation is what makes *sustained* mean one thing in both directions. At a
    written rim cell the two coincide, since the write has already restored
    quiet. At an apex cell nothing writes, so re-adding would accumulate into a
    ramp — a different stimulus wearing the same name, and one whose growth
    would be read as transmission.
    """
    ufp.restore(agent.sheaf, state)
    layout = agent.sheaf.layout
    pairs = () if nudge is None else tuple(nudge)
    if pairs:
        with torch.no_grad():
            for cell, deviation in pairs:
                agent.sheaf.stalks[layout.slice(cell)] += deviation
    trace = []
    kept: dict[int, list[torch.Tensor]] = {c: [] for c in record}
    for step in range(ticks):
        agent.sheaf.tick()
        trace.append(agent.sheaf.disagreement().clone())
        # The world's write is the tick's last word and both branches get the
        # same one. Nothing is stepped: this is the hold.
        agent.write(observation, applied)
        if sustained is not None:
            with torch.no_grad():
                for cell, deviation in pairs:
                    agent.sheaf.stalks[layout.slice(cell)] = (
                        sustained[cell][step] + deviation
                    )
        for cell in record:
            kept[cell].append(agent.sheaf.stalks[layout.slice(cell)].clone())
    return torch.stack(trace), {c: torch.stack(v) for c, v in kept.items()}


def ratios(
    agent: Agent,
    state: dict,
    quiet: torch.Tensor,
    observation: dict,
    applied,
    nudge,
    probe: float,
    window: int,
    sustained: dict[int, torch.Tensor] | None = None,
    record: tuple[int, ...] = (),
):
    """`[ticks, edges]` of paired ratio at `A₀ = 1`, and the recorded node stalks.

    The stalks come back because ADR-0026's reduction needs the perturbed
    branch's per-cell trace and re-running the branch to get it would be the same
    arithmetic twice — and worse, a second fork, where the pairing needs one.
    """
    scaled = tuple((cell, deviation * probe) for cell, deviation in nudge)
    moved, kept = branch(
        agent, state, observation, applied, window, scaled,
        sustained=sustained, record=record,
    )
    numerator = (moved - quiet).norm(dim=-1).numpy() / probe
    denominator = quiet.norm(dim=-1).numpy()
    # An edge standing at exactly zero is not a division to guard: it is an edge
    # carrying nothing, which anything at all is distinguishable from. It is
    # reported rather than clamped, because a graph full of them would mean the
    # hold had collapsed and not that transmission had succeeded.
    ratio = np.where(
        denominator > 0, numerator / np.maximum(denominator, 1e-300), np.inf
    )
    return np.where(numerator == 0, 0.0, ratio), kept


def horizons(window: int) -> tuple[int, ...]:
    """The ladder the bottleneck is reported on, so no single window is silent.

    The map's standing warning is against extrapolating a trend to its limit, and
    its dual applies here: the deviation does not decay to nothing, so a peak
    taken over a longer window is partly a maximum over more excursions. Printing
    the ladder is what makes the horizon's contribution visible instead of buried
    in the choice of :data:`WINDOW`.

    **The ladder runs past 256 for #232.** #214's flatness from 32 to 64 was
    measured on a decaying pulse and says nothing about a drive held for the
    whole window: a sustained stimulus has a settling time the ladder has to be
    long enough to show, and inheriting the pulse's flatness would assume the
    answer.
    """
    ladder = [h for h in (8, 16, 32, 64, 128, 256, 512, 1024) if h <= window]
    if window not in ladder:
        ladder.append(window)
    return tuple(ladder)


def trial(
    agent: Agent,
    observation: dict,
    applied,
    source,
    targets: tuple[int, ...],
    generator: torch.Generator,
    window: int,
    probe: float,
    stimulus: str = "impulse",
    coherent: bool = True,
    fork=None,
) -> Trial:
    """One perturbation: fork, inject, difference, reduce against both predicates.

    `source` is a cell or a tuple of them; `stimulus` is `impulse` (#214's, the
    deviation added once) or `sustained` (#232's, the source held at a constant
    offset for the window). `fork` carries the quiet branch and the recorded
    source stalks when several corners share one — a control run per corner
    would be the same arithmetic four times, and the pairing requires the
    branches be forked from one state anyway.
    """
    state = ufp.snapshot(agent.sheaf)
    cells = (source,) if isinstance(source, int) else tuple(source)
    # Every predicting cell is recorded, because ADR-0026's `min` runs over the
    # cells of a path and a cell missing from the trace would silently shorten
    # it — the reduction would then report the widest path *among those
    # recorded*, which is a different predicate wearing the same name.
    record = tuple(dict.fromkeys(cells + agent.dome.predicting))
    if fork is None:
        quiet, held = branch(
            agent, state, observation, applied, window, None, record=record
        )
    else:
        quiet, held = fork
    missing = [c for c in agent.dome.predicting if c not in held]
    if missing:
        raise ValueError(
            f"the fork recorded {len(missing)} predicting cells short of the "
            "graph; ADR-0026's reduction is over all of them"
        )
    if len(cells) == 1:
        nudge = ((cells[0], unit(agent.dome.cells[cells[0]].stalk, generator)),)
    else:
        nudge = injection(agent.dome, cells, generator, coherent)
    ratio, moved = ratios(
        agent,
        state,
        quiet,
        observation,
        applied,
        nudge,
        probe,
        window,
        sustained=held if stimulus == "sustained" else None,
        record=record,
    )
    ufp.restore(agent.sheaf, state)
    peaks = ratio.max(axis=0)
    value, target, edge, path = widest_path(agent.dome, peaks, cells, targets)
    loops = world_loops(agent.dome)
    conducted = conduction(agent.dome, held, moved, cells, targets, loops)
    return Trial(
        source=cells,
        kind=agent.dome.cells[cells[0]].kind.value,
        bottleneck=value,
        target=target,
        edge=edge,
        path=path,
        ratios=peaks,
        peak_at=ratio.argmax(axis=0),
        horizons=tuple(
            (h, widest_path(agent.dome, ratio[:h].max(axis=0), cells, targets)[0])
            for h in horizons(window)
        ),
        conduction=conducted["conduction"],
        cell=conducted["cell"],
        walk=conducted["walk"],
        tau=conducted["tau"],
        censored=conducted["censored"],
        resolved=conducted["readable"],
        private=conducted["private"],
        floor=conducted["floor"],
        conduction_horizons=tuple(
            (
                h,
                conduction(
                    agent.dome, held, moved, cells, targets, loops, window=h
                )["conduction"],
            )
            for h in horizons(window)
        ),
    )


def quantiles(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {
        "n": int(values.size),
        "infinite": int(values.size - finite.size),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
    }


def sources(
    dome: Dome,
    direction: str,
    generator: np.random.Generator,
    count: int,
    collective: bool = False,
):
    """Which stalk the unit deviation is injected into, stratified across kinds.

    Stratified rather than drawn uniformly: 256 of the 263 rim cells are patch
    cells, so a uniform draw would report the render's fate under both columns'
    name, and proprioception — which #120 measured dying at a different level —
    would appear in about one trial in forty.

    With `collective`, a trial's source is the **whole stratum** rather than one
    cell drawn from it — #230's ruling that inbound rim influence is many-to-few,
    read at the same rotation across kinds so the corners are trial-for-trial
    comparable with #214's.
    """
    if direction == "apex-to-rim":
        pool = apex(dome)
        return [int(pool[i % len(pool)]) for i in range(count)]
    strata = rim_strata(dome)
    if collective:
        return [tuple(strata[i % len(strata)]) for i in range(count)]
    return [int(generator.choice(strata[i % len(strata)])) for i in range(count)]


def rim_strata(dome: Dome) -> list[list[int]]:
    """The rim's exchangeable sets: one list of cell ids per sensor kind."""
    kinds = (CellKind.PATCH, CellKind.PROPRIOCEPTIVE, CellKind.TOUCH)
    strata = [[c.id for c in dome.cells if c.kind is kind] for kind in kinds]
    return [s for s in strata if s]


def hold_still(agent: Agent, observation: dict, applied, hold: int) -> None:
    """Hold the world still, so the lag floor drains out of the reading."""
    for _ in range(hold):
        agent.sheaf.tick()
        agent.write(observation, applied)


def prepared(name: str, split: str, seed: int, learn: int):
    """A surface trained in float32, then cast to float64 for the read."""
    env, agent = ufp.build(name, split, seed)
    print(f"dome {name}: {len(agent.dome.cells)} cells, {len(agent.dome.edges)} edges")
    print(f"training {learn} ticks with both rules, in float32...", flush=True)
    if learn:
        ufp.taught(agent, learn, seed)
    else:
        for _ in ufp.run(agent, 1, seed=seed):
            pass
    cast = double_precision(agent.sheaf)
    print(f"cast {cast} tensors to float64 for the read; see `double_precision`")
    return env, agent


def name_edge(dome: Dome, edge_id: int) -> str:
    if edge_id < 0:
        return "none"
    edge = dome.edges[edge_id]
    u, v = dome.cells[edge.u], dome.cells[edge.v]
    return f"#{edge_id} {edge.kind.value} m={edge.m} {u.index} -- {v.index}"


def linearity(
    name: str,
    split: str,
    seed: int,
    learn: int,
    window: int,
    hold: int,
    amplitudes: tuple[float, ...],
    stimulus: str = "impulse",
    collective: bool = False,
) -> None:
    """The read's own validity check: is the response linear, or is it rounding?

    A transported deviation is linear in what was injected, so the bottleneck
    reported *at `A₀ = 1`* is the same number whatever amplitude it was measured
    at — a flat column. Rounding is not, and shows as a column falling like
    `1/A₀`. Run this before believing anything `read` prints.

    **Each corner needs its own run of this**, which is #232's requirement rather
    than a convenience: a sustained drive is held for the whole window instead of
    decaying, so it explores more of `encode`'s nonlinearity and has no claim on
    the flat window #214 measured on a pulse. `--stimulus` and `--collective`
    pick the corner, and the linear window is a property of the corner.

    **The ladder runs for `τ̂` too, and that is #224's corroboration** (ADR-0026,
    *The reading is gated on the runtime precision floor*). `τ̂` is a decay
    *time*, so a transported deviation's is flat in the amplitude it was injected
    at — halving `A₀` halves the whole trace and moves neither the peak tick nor
    the `1/e` crossing. **Rounding is not flat, and it fails upward**: a deviation
    decaying into the rounding floor stops decaying, so `τ̂` climbs as `A₀` falls,
    which is the direction that manufactures a PASS on a bar of exactly `1`. That
    makes the two columns complementary rather than redundant — the bottleneck
    column falls like `1/A₀` under rounding, the conduction column *rises*.

    The gate (:func:`readable`) is printed beside them as the fraction of cells
    whose crossing clears `EPS_F32 · ‖state‖`. It answers a different question
    from the ladder: the gate is per cell and per reading, the ladder is a
    property of the whole column, and a column that only goes flat once the
    resolved fraction collapses is reporting arithmetic either way.
    """
    env, agent = prepared(name, split, seed, learn)
    observation, _info = env.reset(seed=seed * 1000)
    agent.observe(observation)
    applied = np.zeros(env.action_space.shape, dtype=np.float64)
    hold_still(agent, observation, applied, hold)
    state = ufp.snapshot(agent.sheaf)
    cells = pick(agent.dome, seed, collective)
    # Every predicting cell, not just the source: `conduction` reduces over the
    # cells of a path, and a missing one would shorten the `min` — the same
    # requirement `trial` records for.
    record = tuple(dict.fromkeys(cells + agent.dome.predicting))
    quiet, held = branch(
        agent, state, observation, applied, window, None, record=record
    )

    twice, _ = branch(agent, state, observation, applied, window, None)
    print(f"\nthe fork is exact: max |quiet - quiet| = {float((twice - quiet).abs().max())}")

    generator = torch.Generator().manual_seed(seed + 1)
    if len(cells) == 1:
        nudge = ((cells[0], unit(agent.dome.cells[cells[0]].stalk, generator)),)
    else:
        nudge = injection(agent.dome, cells, generator)
    targets = apex(agent.dome)
    kind = agent.dome.cells[cells[0]].kind.value
    print(
        f"corner: {stimulus} x {'collective' if collective else 'single-source'}; "
        f"injected into {len(cells)} {kind} cell(s)"
    )
    loops = world_loops(agent.dome)
    print(
        "\n  A0         bottleneck at A0=1     median edge ratio     "
        "conduction     median tau_hat   resolved"
    )
    for a0 in amplitudes:
        ratio, moved = ratios(
            agent,
            state,
            quiet,
            observation,
            applied,
            nudge,
            a0,
            window,
            sustained=held if stimulus == "sustained" else None,
            record=record,
        )
        peaks = ratio.max(axis=0)
        value, _target, _edge, _path = widest_path(agent.dome, peaks, cells, targets)
        finite = peaks[np.isfinite(peaks)]
        read = conduction(agent.dome, held, moved, cells, targets, loops)
        print(
            f"  {a0:<10.3g} {value:<22.5g} {np.median(finite):<21.5g} "
            f"{read['conduction']:<14.5g} {np.median(read['tau']):<16.5g} "
            f"{read['readable'].mean():.1%}"
        )
    ufp.restore(agent.sheaf, state)


def pick(dome: Dome, seed: int, collective: bool) -> tuple[int, ...]:
    """`linearity`'s one source: #214's first stratified draw, or its whole stratum."""
    drawn = sources(
        dome, "rim-to-apex", np.random.default_rng(seed), 1, collective=collective
    )[0]
    return (drawn,) if isinstance(drawn, int) else tuple(drawn)


def name_cell(dome: Dome, cell_id: int) -> str:
    if cell_id < 0:
        return "none"
    cell = dome.cells[cell_id]
    return f"#{cell_id} {cell.kind.value} {cell.index}"


def report(dome: Dome, direction: str, outcomes: list[Trial], window: int) -> None:
    """Both predicates, in their settled order: the operative bar, then the diagnostic.

    ADR-0026's conduction ratio carries the `HOLDS`/`FAILS`, because it is what
    #127 is read against. ADR-0021's bottleneck ratio is printed under its own
    name directly below it — kept as the **sufficient** per-edge diagnostic,
    where a pass is conclusive and a fail is not. Printing them together is what
    makes the two quantities' different sizes visible in one place, which is the
    confusion [#379](https://github.com/NGL321/patchworks/issues/379) closed.
    """
    conducted = np.array([o.conduction for o in outcomes])
    q = quantiles(conducted)
    verdict = "HOLDS" if q["median"] >= 1.0 else "FAILS"
    print(f"== {direction}: rim-core influence {verdict} ==")
    print(
        "   ADR-0026, the operative bar: conduction ratio = "
        "tau_hat / world_loop (#383)"
    )
    print(
        f"   over {q['n']} trials — "
        f"p05 {q['p05']:.3g}  p25 {q['p25']:.3g}  "
        f"median {q['median']:.3g}  p75 {q['p75']:.3g}  p95 {q['p95']:.3g}"
    )
    if q["median"] > 0:
        print(f"   short of the bar by {1.0 / q['median']:.3g}x at the median")
    middle = outcomes[int(np.argsort(conducted)[len(conducted) // 2])]
    print(f"   the median trial's binding cell: {name_cell(dome, middle.cell)}")
    loops = world_loops(dome)
    rows = {cell: row for row, cell in enumerate(dome.predicting)}
    profile = "  ".join(
        f"{middle.tau[rows[c]] / loops[c]:.3g}" if c in loops else "-"
        for c in middle.walk
        if c in rows
    )
    print(f"   its path, cell by cell from the source — {profile}")
    # A cell with no private dimension has nowhere for `tau_hat` to be read, so
    # its ratio is `0` by construction and no retention work can move it (#385).
    # Stated here rather than folded into the number, because a bar pinned by
    # construction is the bystander ADR-0026 replaced the amplitude ratio to be
    # rid of, and this rig is what would hide it.
    empty = [c for c in middle.walk if c in rows and not middle.private[rows[c]]]
    if empty:
        print(
            f"   {len(empty)} of that path's cells hold no private features, so "
            f"their tau_hat is 0 by construction, not by measurement (#385): "
            + ", ".join(name_cell(dome, c) for c in empty[:4])
            + ("  ..." if len(empty) > 4 else "")
        )
    # #224's gate and the window, reported next to the number rather than
    # folded into it: a float64 tau_hat below float32's granularity is a real
    # number the running architecture has no signal for, and a censored one is a
    # lower bound. Both can only cost a PASS, and both stay visible.
    unresolved = np.mean([1.0 - o.resolved.mean() for o in outcomes])
    censored = np.mean([o.censored.mean() for o in outcomes])
    binding_unresolved = np.mean(
        [
            0.0 if o.cell < 0 else float(not o.resolved[rows[o.cell]])
            for o in outcomes
            if o.cell < 0 or o.cell in rows
        ]
    )
    print(
        f"   unreadable at runtime precision: {unresolved:.1%} of cells "
        f"(#224's gate), and the binding cell in {binding_unresolved:.1%} of trials"
    )
    print(
        f"   tau_hat censored by the window of {window}: {censored:.1%} of cells "
        "— a lower bound, so it can only cost a pass"
    )
    # The floor's *magnitude*, which `tick.PRECISION_FLOOR` records. It is a
    # scale rather than a constant — it moves with the state it is read at — so
    # what is published is the median over a trial's cells with the spread
    # across trials, and never one number pretending to be fixed.
    floors = np.array([o.floor for o in outcomes])
    print(
        f"   the runtime precision floor at these cells: median {np.median(floors):.3g} "
        f"(p05 {np.quantile(floors, 0.05):.3g}, p95 {np.quantile(floors, 0.95):.3g}), "
        f"eps_f32 = {EPS_F32:.3g}"
    )
    ladder = "  ".join(
        f"{h}:{np.median([dict(o.conduction_horizons)[h] for o in outcomes]):.3g}"
        for h, _ in outcomes[0].conduction_horizons
    )
    print(f"   median conduction by horizon (ticks:value) — {ladder}")
    strata = sorted({o.kind for o in outcomes})
    if len(strata) > 1:
        breakdown = "  ".join(
            f"{k}:{np.median([o.conduction for o in outcomes if o.kind == k]):.3g}"
            for k in strata
        )
        # Never averaged into one number: patch is 256 cells and the other two
        # are three each, so a collective read means something different in each.
        print(f"   median by stratum — {breakdown}")

    values = np.array([o.bottleneck for o in outcomes])
    b = quantiles(values)
    print(
        "   -- ADR-0021, the sufficient per-edge diagnostic: bottleneck ratio "
        f"{'HOLDS' if b['median'] >= 1.0 else 'FAILS'} "
        "(a pass is conclusive, a fail is not) --"
    )
    print(
        f"   over {b['n']} trials — "
        f"p05 {b['p05']:.3g}  p25 {b['p25']:.3g}  "
        f"median {b['median']:.3g}  p75 {b['p75']:.3g}  p95 {b['p95']:.3g}"
    )
    at = np.array([o.peak_at[o.edge] for o in outcomes])
    print(
        f"   the binding edge peaks at tick {np.median(at):.0f} "
        f"(median over trials), in a window of {window}"
    )
    amplitude = "  ".join(
        f"{h}:{np.median([dict(o.horizons)[h] for o in outcomes]):.3g}"
        for h, _ in outcomes[0].horizons
    )
    print(f"   median bottleneck by horizon (ticks:value) — {amplitude}")
    edge_middle = outcomes[int(np.argsort(values)[len(values) // 2])]
    print(f"   the median trial's binding edge: {name_edge(dome, edge_middle.edge)}")
    edges = "  ".join(f"{edge_middle.ratios[e]:.3g}" for e in edge_middle.path)
    print(f"   its path, edge by edge from the source — {edges}")
    print()


def readings(results: dict[str, list[Trial]]) -> dict[str, float]:
    """What this read has to offer a `measurement` cutoff, by name.

    **`conduction_ratio` is ADR-0026's `τ̂_c / world_loop(c)`** — the divisor is
    `world_loop(c)` since #383, and the key was ADR-0021's bottleneck ratio until
    [#379](https://github.com/NGL321/patchworks/issues/379). #325 and #329 both
    write `conduction ratio >= 1` and both name ADR-0026/0027 as the warrant, so
    the key they cut on now carries the quantity their warrant names. The two are
    eight orders apart and they fail differently, so this was never cosmetic: a
    cutoff wired to the amplitude ratio waits on a quantity no amount of the work
    in front of it can move, which is the bystander ADR-0026 exists to have
    replaced.

    **#383's divisor swap does not move this key and does not re-file those two
    verdicts.** It reads `0` under `|loop(c)|` and under `world_loop(c)` alike,
    because #385's zero-private-dimension L1 cells pin the `min` either way. A
    reader who sees the key unchanged after the swap is seeing #385, not an edit
    that failed to land; the per-cell ratios :func:`report` prints are where the
    ruling shows.

    **ADR-0021's ratio is still published, under `bottleneck_ratio`.** ADR-0026
    kept it as the sufficient per-edge diagnostic and demoted only its role, so
    dropping it here would lose a reading the record still uses. Any cutoff that
    means the amplitude quantity can now say so, which it could not before.

    **Both are offered per direction and by side.** `inbound` is rim→apex and
    `outbound` is apex→rim, ADR-0026's own words for the two clauses;
    [#341](https://github.com/NGL321/patchworks/issues/341) writes `outbound
    conduction ratio >= 1` and had no key to land on until now, which is why the
    register renders it as never having been reported against.

    **The bare key is the lower of the two directions**, because a graph that
    transmits one way and not the other does not transmit, and taking the minimum
    is the reading that cannot say *holds* while a direction fails.
    """
    def medians(attribute: str) -> dict[str, float]:
        return {
            direction: float(np.median([getattr(o, attribute) for o in outcomes]))
            for direction, outcomes in results.items()
            if outcomes
        }

    quantities = {
        "conduction_ratio": medians("conduction"),
        "bottleneck_ratio": medians("bottleneck"),
    }
    if not quantities["conduction_ratio"]:
        return {}
    found: dict[str, float] = {}
    for name, values in quantities.items():
        for direction, value in values.items():
            found[f"{name}_{direction.replace('-', '_')}"] = value
        found[name] = min(values.values())
        for side, direction in (("inbound", "rim-to-apex"), ("outbound", "apex-to-rim")):
            if direction in values:
                found[f"{side}_{name}"] = values[direction]
    return found


def read(
    name: str,
    split: str,
    seed: int,
    learn: int,
    trials: int,
    window: int,
    hold: int,
    probe: float,
    file_cutoffs: bool = True,
) -> None:
    """The whole read: train once, then sweep configurations, both directions."""
    env, agent = prepared(name, split, seed, learn)
    dome = agent.dome
    ends = {"rim-to-apex": apex(dome), "apex-to-rim": rim(dome)}
    picker = np.random.default_rng(seed)
    picks = {d: sources(dome, d, picker, trials) for d in ends}
    generator = torch.Generator().manual_seed(seed + 1)
    results: dict[str, list[Trial]] = {d: [] for d in ends}

    for i in range(trials):
        observation, _info = env.reset(seed=seed * 1000 + i)
        agent.observe(observation)
        applied = np.zeros(env.action_space.shape, dtype=np.float64)
        hold_still(agent, observation, applied, hold)
        for direction in ends:
            results[direction].append(
                trial(
                    agent,
                    observation,
                    applied,
                    picks[direction][i],
                    ends[direction],
                    generator,
                    window,
                    probe,
                )
            )
        print(
            f"  trial {i + 1}/{trials}: "
            + "  ".join(
                f"{d} conduction {results[d][-1].conduction:.3g} "
                f"(bottleneck {results[d][-1].bottleneck:.3g})"
                for d in ends
            ),
            flush=True,
        )

    print()
    for direction, outcomes in results.items():
        report(dome, direction, outcomes, window)
    # The measurement half of the cutoff mechanism (#284). #325, #329 and #341
    # cut on this rig, and all wait on the same thing: a graph that transmits,
    # which is what this read measures. It states a verdict, files the run, and
    # asserts nothing. What `conduction_ratio` means is ADR-0026's ratio of
    # times, corrected by #379 and divided by `world_loop(c)` since #383; see
    # :func:`readings`.
    report_cutoffs("detectability", readings(results), file=file_cutoffs)


#: The two-axis space #232 reads, inbound. #214 measured the first row's first
#: column and it is the corner least favourable to what the architecture claims:
#: a stimulus that cannot persist, from a single sensor's private disturbance.
CORNERS = (
    ("impulse", "single-source"),
    ("impulse", "collective"),
    ("sustained", "single-source"),
    ("sustained", "collective"),
)

#: The collective corners run twice, coherent and not, so `injection`'s clause 1
#: is measured instead of asserted. Same cells, same total norm, same everything
#: else — the only difference is whether the cells were given one direction or
#: `K` of them, so the gap between the two rows *is* what coherence is worth.
CONTRAST = "collective (incoherent)"


def corners(
    name: str,
    split: str,
    seed: int,
    learn: int,
    trials: int,
    window: int,
    hold: int,
    probe: float,
    contrast: bool,
) -> None:
    """All four corners of the inbound stimulus space, against #214's baseline.

    **One quiet branch per trial, shared by every corner.** The control does not
    depend on the perturbation, and re-running it per corner would be the same
    arithmetic six times over — but more than that, the corners must be forked
    from *one* state to be compared with each other at all. The recorded stalks
    that come back with it are what the sustained corners clamp against.

    **Outbound stays single-source, in all four**, per #230: apex influence on
    the rim is claimed to be individually strong, so one apex cell is the correct
    instrument there and the collective axis has nothing to mean. It is read at
    both stimuli, because the sustained axis is about the stimulus and applies to
    either direction.
    """
    env, agent = prepared(name, split, seed, learn)
    dome = agent.dome
    inward, outward = apex(dome), rim(dome)
    picker = np.random.default_rng(seed)
    singles = sources(dome, "rim-to-apex", picker, trials)
    collectives = sources(
        dome, "rim-to-apex", np.random.default_rng(seed), trials, collective=True
    )
    outbound = sources(dome, "apex-to-rim", picker, trials)
    generator = torch.Generator().manual_seed(seed + 1)

    runs: list[tuple[str, str, str]] = [
        (f"rim-to-apex  {stimulus} x {breadth}", stimulus, breadth)
        for stimulus, breadth in CORNERS
    ]
    if contrast:
        runs += [
            (f"rim-to-apex  {stimulus} x {CONTRAST}", stimulus, CONTRAST)
            for stimulus in ("impulse", "sustained")
        ]
    runs += [
        (f"apex-to-rim  {stimulus} x single-source", stimulus, "outbound")
        for stimulus in ("impulse", "sustained")
    ]
    results: dict[str, list[Trial]] = {label: [] for label, _, _ in runs}

    for i in range(trials):
        observation, _info = env.reset(seed=seed * 1000 + i)
        agent.observe(observation)
        applied = np.zeros(env.action_space.shape, dtype=np.float64)
        hold_still(agent, observation, applied, hold)
        touched = tuple(
            sorted({singles[i], outbound[i], *collectives[i]})
        )
        # Plus every predicting cell: the shared fork is what ADR-0026's
        # reduction reads, and :func:`trial` refuses a fork that is short of one.
        state = ufp.snapshot(agent.sheaf)
        fork = branch(
            agent,
            state,
            observation,
            applied,
            window,
            None,
            record=tuple(dict.fromkeys(touched + agent.dome.predicting)),
        )
        ufp.restore(agent.sheaf, state)
        for label, stimulus, breadth in runs:
            if breadth == "outbound":
                source, targets = outbound[i], outward
            elif breadth == "single-source":
                source, targets = singles[i], inward
            else:
                source, targets = collectives[i], inward
            results[label].append(
                trial(
                    agent,
                    observation,
                    applied,
                    source,
                    targets,
                    generator,
                    window,
                    probe,
                    stimulus=stimulus,
                    coherent=breadth != CONTRAST,
                    fork=fork,
                )
            )
        print(
            f"  trial {i + 1}/{trials}: "
            + "  ".join(
                f"{label.split('  ')[1]} {results[label][-1].conduction:.3g}"
                for label, _, _ in runs
                if label.startswith("rim")
            ),
            flush=True,
        )

    print()
    for label, _, _ in runs:
        report(dome, label, results[label], window)
    table(results, runs)


def table(results: dict[str, list[Trial]], runs) -> None:
    """The 2x2 the ticket asks for, with the baseline's own corner in place.

    **Printed twice, once per predicate.** #232's finding is *the size of the gap
    between corners*, and that is a claim about a quantity: the gap the amplitude
    ratio shows and the gap the conduction ratio shows are different readings of
    the same four corners, and #379 is what this file learnt from collapsing two
    quantities into one name. The bottleneck grid keeps #214's published baseline
    underneath it, because that baseline is a bottleneck number.
    """
    for heading, attribute in (
        ("median conduction ratio (ADR-0026, the operative bar)", "conduction"),
        ("median bottleneck ratio (ADR-0021, the diagnostic)", "bottleneck"),
    ):
        print(f"== the four corners, inbound: {heading} ==")
        print(f"   {'':<24}{'impulse':<16}sustained")
        for breadth in ("single-source", "collective", CONTRAST):
            cells = []
            for stimulus in ("impulse", "sustained"):
                label = f"rim-to-apex  {stimulus} x {breadth}"
                if label in results:
                    cells.append(
                        f"{np.median([getattr(o, attribute) for o in results[label]]):.3g}"
                    )
                else:
                    cells.append("-")
            print(f"   {breadth:<24}{cells[0]:<16}{cells[1]}")
        print()
    print(
        "\n   #214's published baseline is impulse x single-source: 8.7e-10 "
        "rim->apex, 1.3e-8 apex->rim."
    )
    print("   The finding is the size of the gap between corners, not any one of them.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for spoken, help_text in (
        ("read", "the rim-to-core detectability read"),
        ("corners", "all four corners of the inbound stimulus space (#232)"),
        ("linearity", "the read's own validity check"),
    ):
        p = commands.add_parser(spoken, help=help_text)
        p.add_argument("--dome", default="full", choices=("small", "full"))
        p.add_argument("--split", default="train")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--learn", type=int, default=30000)
        p.add_argument("--window", type=int, default=WINDOW)
        p.add_argument("--hold", type=int, default=HOLD)
        if spoken in ("read", "corners"):
            p.add_argument("--trials", type=int, default=24)
            p.add_argument("--probe", type=float, default=PROBE)
        if spoken == "read":
            p.add_argument(
                "--no-file",
                dest="file_cutoffs",
                action="store_false",
                help=(
                    "print the cutoff report and file nothing on the tracker. "
                    "Use it for a read that is not the read -- a short --learn, "
                    "the small dome -- because filing records a run against "
                    "#325 and #329 and a toy reading is not one"
                ),
            )
        if spoken == "corners":
            p.add_argument(
                "--no-contrast",
                dest="contrast",
                action="store_false",
                help="skip the incoherent collective rows",
            )
        if spoken == "linearity":
            p.add_argument(
                "--stimulus", default="impulse", choices=("impulse", "sustained")
            )
            p.add_argument("--collective", action="store_true")
            p.add_argument(
                "--amplitudes",
                nargs="+",
                type=float,
                default=[1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2],
            )
    arguments = parser.parse_args(argv)
    if arguments.command == "linearity":
        linearity(
            arguments.dome,
            arguments.split,
            arguments.seed,
            arguments.learn,
            arguments.window,
            arguments.hold,
            tuple(arguments.amplitudes),
            arguments.stimulus,
            arguments.collective,
        )
        return
    if arguments.command == "corners":
        corners(
            arguments.dome,
            arguments.split,
            arguments.seed,
            arguments.learn,
            arguments.trials,
            arguments.window,
            arguments.hold,
            arguments.probe,
            arguments.contrast,
        )
        return
    read(
        arguments.dome,
        arguments.split,
        arguments.seed,
        arguments.learn,
        arguments.trials,
        arguments.window,
        arguments.hold,
        arguments.probe,
        arguments.file_cutoffs,
    )


if __name__ == "__main__":
    main()
