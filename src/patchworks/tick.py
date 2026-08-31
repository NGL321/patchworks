"""The tick: two phases, unit delay, and the state they run on.

`docs/spec/02-tick-semantics.md` is the whole of what this module implements.
Every tick is exactly two phases, in this order:

1. **Inference phase.** Every predicting cell, simultaneously and
   independently, runs `encode` / `step` / `decode` on its own persisted chart
   and the node stalk the previous message-passing phase left behind. No cell
   reads another cell's state here. Boundary cells run no body and sit this
   phase out.
2. **Message-passing phase.** Every cell restricts its own predicted node stalk
   onto each incident edge and runs **one** local descent step against the
   belief its neighbour restricted onto that same edge one tick ago. The result
   edits the node stalk only, never the chart.

Whatever is outside the sheaf then writes its boundary cells. That is an
**ordering**, not a third phase — nothing here computes during it, and it lives
in :mod:`patchworks.agent`, where the world is.

Three commitments are structural here rather than configurable.

* **An edge costs exactly one tick.** :attr:`Sheaf.broadcast` holds what every
  cell put on every incident edge last tick, and the message-passing phase
  reads it through a partner flip. What a cell reconciles against is always one
  tick stale, so there is no "now" spanning the graph.
* **One step, not a solve.** Exactly one simultaneous local descent step per
  cell per tick — no round count, no convergence check, no early stopping
  (ADR-0002). Every cell reads the same prior round's incoming values and
  updates at once, Jacobi-style, so there is no visiting order to define. The
  phase below is straight-line code with no loop in it at all, which is the
  cheapest possible way to keep that true.
* **The tick carries no tape.** The whole tick runs under `torch.no_grad()`,
  and :func:`assert_no_tape` checks every quantity leaving it. This is not an
  optimisation (`docs/spec/09-the-build-stack.md`, *The locality guard*): the
  tick is a rollout, and every quantity it produces is *data* the cell then
  learns from. A leaked gradient does not crash — it makes the agent work
  better, which looks like the thesis being right — so the check is always on
  rather than a review habit.

**Every tick is read.** :class:`FoldRead` takes the fold margin, the
activation region and the standing offset off the two phases as they run — the
pre-activation and the displacement they had already computed, against one
frozen graph-wide constant. `02-tick-semantics.md`'s bound was checked once at
construction until #160 found both of its sides moving, so **construction
nominates and the run decides**
(`docs/adr/0019-construction-nominates-the-run-decides.md`). It is always on,
for the reason :func:`assert_no_tape` is.

**Node stalks are one flat buffer.** Boundary cells are exempt from `n` and
carry the world's shape instead, so the population's stalks are ragged. They
are stored end to end in a single tensor with a per-cell offset, plus one
trailing slot that is permanently zero: padded gathers read it and padded
scatters land in it, which is what lets the ragged graph run as a handful of
whole-population tensor operations with no loop over cells or edges.
"""

from __future__ import annotations

import torch

from .body import CellBiases, CellBody, CellOperators
from .graph import Dome
from .restriction import GAUGE_RHO, RestrictionMaps, pair_index

__all__ = [
    "DEFAULT_GAMMA",
    "FoldRead",
    "Sheaf",
    "StalkLayout",
    "assert_no_tape",
    "reconciliation_gain",
]

#: @type stipulated
#: @flexibility free in (0, 1] and held at the ceiling because nothing derives a lower value: #206 declined a ramp permanently and withdrew the decline on a permanently lower value, which is #205's to weigh
#: @warrant docs/adr/0019-construction-nominates-the-run-decides.md
#: The single global `γ` of the reconciliation gain, `γ ≤ 1`
#: (`docs/spec/02-tick-semantics.md`, *Reconciliation gain*). One scalar for the
#: whole graph: the per-cell variation is entirely in the denominator, which is
#: a structural quantity.
#:
#: **1.0 is the ceiling `02` permits, and nothing below it is derivable.** The
#: fold margin was once expected to cap this — the `@provisional 85` that stood
#: here — and #140 demoted that bound, #160 moved the check itself off
#: construction. The margin bounds the *standing offset*, not `γ`, and both
#: sides of it move through a run, so there is no construction-time number to
#: read a cap off. #160 ruled that the bound holds **after a burn-in**; #202
#: measured 100,000 ticks and found no such count — not one tick free of a
#: breaching cell, and the density plateaus at ~16 cells in 150 — so #206 struck
#: the clause and replaced it with nothing. The margin-against-offset comparison
#: is an **attribution**, not a verdict, and carries no threshold; the verdict is
#: measured region dwell. A ramp stays declined, aimed at a transient the run
#: does not have; a permanently lower `γ` is no longer declined but is not
#: adopted here either — it is #205's
#: (`docs/adr/0019-construction-nominates-the-run-decides.md`, decision 5).
DEFAULT_GAMMA = 1.0


def _would_be_misread(gamma: object) -> bool:
    """`True` for a value that coerces to a float but means something else.

    `float(True)` is `1.0`, `numpy.complex128(0.5 + 3j)` narrows to `0.5`
    behind a warning, and `float("0.5")` parses. So a config carrying
    `gamma = true`, a `γ` read out of a complex array, and a grid read off a
    CSV as text would each otherwise run a whole sweep point at a number
    nobody chose, or at a rule nobody stated, and say nothing.

    **This is the unwrapping half of the rule.** Python's own `bool` and `str`
    need none; numpy's `bool_`, `str_`, a boolean tensor and the complex boxes
    are the same mistakes wearing this stack's containers, and all of them come
    out through the `item()` every scalar here shares. Anything `item()`
    refuses is not a scalar at all, and the coercion below is the one to say
    so.
    """
    try:
        unwrapped = gamma.item() if hasattr(gamma, "item") else gamma
    except Exception:
        # Whatever a stand-in for a value does when asked for one, it is not
        # this function's to diagnose: fall through, and the rule refuses it in
        # its own words — unless it raises on the way to being a number there
        # too, in which case it leaves as what it raised (`_checked_gamma`).
        return False
    # `float` is not a `complex` subclass — that relation holds in the
    # `numbers` tower, not between the built-in types — so this refuses the
    # complex ones without touching an ordinary `γ`.
    return isinstance(unwrapped, (bool, complex, str, bytes, bytearray))


def _shown(gamma: object) -> str:
    """What a refusal quotes back: the value's `repr`, kept short.

    An integer too large for a float has hundreds of digits and a sweep that
    passed its whole grid instead of a point has hundreds of entries. Neither
    is worth a refusal the reader has to scroll to reach the end of — and past
    4300 digits `repr` refuses outright, which would lose the refusal that
    matters behind one about integer formatting.
    """
    try:
        shown = repr(gamma)
    except Exception:
        # `repr` of an integer over 4300 digits raises, and a config proxy
        # standing in for an absent value can raise anything at all. Neither
        # is allowed to become the exception the caller sees: this is the
        # refusal path, and losing the diagnosis here loses it entirely.
        return f"<a {type(gamma).__name__} that cannot be quoted>"
    return shown if len(shown) <= 60 else f"{shown[:57]}..."


def _checked_gamma(gamma: object) -> float:
    """Refuse a `γ` that is not a scalar in `(0, 1]`, and hand it back if it is.

    **The one place the legal-`γ` rule lives.** `Sheaf` owns what counts as a
    legal `γ` and #106 was deliberate about not giving
    :class:`patchworks.agent.Agent` a second opinion on it; this function is
    where that ownership is written down, so the gain and the construction that
    precedes it read one rule rather than each restating it.

    Being a number at all is checked here alongside the bound because the two
    are the same mistake reached the same way: a driver writing
    `gamma=cfg.gamma` for a config that says `None`. Left to the comparison
    alone that call died two frames down on `'<' not supported between
    instances of 'float' and 'NoneType'`, naming neither `gamma` nor the rule
    it broke (#107).

    It is checked by **coercion** rather than by testing a type, and the
    `float` is what comes back. A sweep indexing its points out of a
    `torch.linspace` grid, or computing them in `Fraction`s, is handing over a
    perfectly good scalar that no type test could enumerate; meanwhile a value
    that is nominally a number but cannot divide a tensor would sail through
    such a test and die at the gain — the same two-frames-down `TypeError`,
    after the same wasted draws. Coercing settles both, and settles them here.

    Every refusal is a `ValueError` rather than the `TypeError` a non-number
    would conventionally earn, because one rule deserves one thing to catch: a
    `γ` sweep reading its points from a config meets `None` and `1.5` the same
    way, and `Agent`'s adjacent refusal of a misplaced `gamma=None` is already
    a `ValueError`.

    **That is a claim about its refusals, not about everything that can leave.**
    #107 wrote the stronger one — nothing but a `ValueError` escapes — and #112
    withdrew it rather than restructure the function around it. Two arrivals
    still leave as what they raised. One is a stand-in for a number that raises
    on its way to being one: the guard below asks `hasattr`, which swallows
    only `AttributeError`, and the coercion catches only the four classes it
    names, so a proxy for an absent config key escapes as its own exception
    from either the lookup or `__float__` — naming, on the lookup half, the
    dunder it probed rather than the key the caller was resolving. The other is
    a 1-element array under `-W error::DeprecationWarning`, a filter set
    nowhere in this repo: a sweep slicing its grid rather than indexing it does
    produce that shape, so what puts the escape out of reach here is the absent
    filter, not the arrival.

    Neither justified the surgery. Wrapping the two lookups is cheap and costs
    no refusal, but closes only the first half; the rest needs a wider `except`
    at the coercion, which would swallow whatever a caller's `__float__` raises
    for reasons of its own. The `__float__`/`__index__` test stays either way —
    it is what refuses `memoryview(b"0.5")`, whose `float()` is `0.5`. What
    stands is the smaller, true claim: **reach a refusal and it is a
    `ValueError`.**
    """
    rule = (
        "gamma is a single global scalar in (0, 1] "
        "(docs/spec/02-tick-semantics.md, Reconciliation gain)"
    )

    def not_one_number() -> ValueError:
        return ValueError(
            f"{rule}; got {_shown(gamma)}, which is not a single real number"
        )

    # `float()` parses text and buffers as well as coercing numbers, and a
    # config that quotes its numbers is the same mistake as one that leaves
    # them out. A number offers `__float__` or `__index__`; text and buffers
    # offer neither, which is the difference stated once rather than as an
    # enumeration of the containers text arrives in.
    if _would_be_misread(gamma) or not (
        hasattr(gamma, "__float__") or hasattr(gamma, "__index__")
    ):
        raise not_one_number()
    try:
        as_float = float(gamma)
    # `RuntimeError` because a tensor on the meta device has no storage to
    # read a number out of, and torch says so with `NotImplementedError` — a
    # subclass of it. A rule with one thing to catch should not let *that*
    # arrival out through its own coercion; the docstring says which ones it
    # still does. (A complex tensor never reaches here; the unwrapping above
    # turns it away first.)
    except (RuntimeError, TypeError, ValueError):
        # **A single real** number: the likeliest arrival here is a sweep that
        # passed its whole grid rather than indexing a point out of it, and
        # what is wrong with that is its arity, not its type.
        raise not_one_number() from None
    except OverflowError:
        raise ValueError(f"{rule}; got {_shown(gamma)}, which no float can hold") from None
    if not 0.0 < as_float <= 1.0:
        # The value the caller wrote, not the coercion of it. A sweep handed
        # `Fraction(3, 2)` and told `got 1.5` is being shown something it
        # never wrote, which is half of what made the old failure useless.
        raise ValueError(f"{rule}; got {_shown(gamma)}")
    return as_float


def reconciliation_gain(
    dome: Dome, *, gamma: float = DEFAULT_GAMMA, rho: float = GAUGE_RHO
) -> torch.Tensor:
    """`[cells]`: `gain_v = γ / max(Σ_{e∋v} m_e, ρ² · deg(v))`.

    The denominator bounds the largest eigenvalue of the cell's local Laplacian
    block provably rather than by proxy: ADR-0010 bounds every incident
    restriction map by `‖F‖_F ≤ ρ`, so
    `λ_max(Σ_e F_evᵀF_ev) ≤ Σ_e ‖F_ev‖_F² ≤ ρ² · deg(v)`. At `ρ = 2` and the
    vertical edges' `m = 4` the two terms are equal; it is written as the max so
    that a later change to `ρ` cannot silently loosen the bound.

    This **equalises** the effective step across the graph — every cell takes
    roughly the same descent on its own local energy regardless of how many
    edges it sits on. It removes a degree artifact. **It is not a timescale
    knob and must not become one**: a gain graded by depth would be the explicit
    per-cell clock divisor ADR-0005 rejected, wearing a different name. What
    keeps that honest is the shape of this function — one global `γ` over a
    denominator read straight off the built graph, with nothing per-cell to set.

    Boundary cells are included on the same formula. Their maps carry the
    tighter exact gauge, `Σ_e ‖F‖_F² = deg(v)`, so `ρ² · deg(v)` is a valid
    bound for them too and merely a looser one.
    """
    gamma = _checked_gamma(gamma)
    stalk_sums = torch.tensor(dome.stalk_sums, dtype=torch.float32)
    degrees = torch.tensor(dome.degrees, dtype=torch.float32)
    return gamma / torch.maximum(stalk_sums, rho * rho * degrees)


def assert_no_tape(**tensors: torch.Tensor) -> None:
    """Refuse to let anything leave the tick carrying a `grad_fn`.

    The cheap, always-on half of the locality guard
    (`docs/spec/09-the-build-stack.md`, *The guarantee is tested, because a leak
    would flatter us*). It inspects the **tape**; #90's perturbation test
    inspects the **update**, and neither subsumes the other — an in-place write
    through a detached view couples two cells while leaving a perfectly clean
    tape, and only observing the update catches that.

    Both halves of the condition are checked. A tensor produced under grad mode
    from a live parameter carries a `grad_fn`; one that merely *requires* grad
    is a leaf that a later operation would put on a tape. Either is a tick that
    has stopped being a rollout.
    """
    for name, tensor in tensors.items():
        if tensor.grad_fn is not None or tensor.requires_grad:
            raise AssertionError(
                f"{name} left the tick on the autograd tape "
                f"(grad_fn={type(tensor.grad_fn).__name__ if tensor.grad_fn else None}, "
                f"requires_grad={tensor.requires_grad}). The tick is a rollout, not a "
                "training pass: everything it produces is data the cell then learns "
                "from. See docs/spec/09-the-build-stack.md, The locality guard."
            )


class FoldRead:
    """The fold margin, the standing offset and region dwell, read live off the tick.

    ADR-0019's *the run decides*. The construction sweep
    (:func:`patchworks.bias_selection.fold_margin_check`) **nominates** a cap
    before anything runs; this reads the same quantities on the run that
    actually happens, which is the only place they can be read, because both
    sides of the bound move:

    * the **standing offset** is dominated at construction by model error,
      which is what learning exists to remove, and falls 144x through a run
      ([#158](https://github.com/NGL321/patchworks/issues/158));
    * the **folds** move too, because their positions are the per-cell biases
      the prediction rule trains — one frozen set of orientations, rigidly
      translated per cell, sliding under the operating point for the length of
      the run.

    **Two readings, and they are not the same reading.** :attr:`dwell` is the
    **verdict**: ADR-0005's timescale mechanism holds only where a cell stays
    in one activation region long against the `τ` that region implies, and
    dwell is that, measured. The margin against the offset is the
    **attribution**: dwell alone cannot say *why* a cell left its region, and
    what ADR-0007 forbids specifically is reconciliation moving it. A cell with
    short dwell and :attr:`reconciliation_reaches` false left its region under
    its own dynamics, which is not this bound's business.

    **No new state and no new time constant**, which is what sank every earlier
    re-derivation proposal (#33, struck by #37). The margin's numerator is the
    pre-activation the forward path already computed and its denominator is one
    graph-wide frozen constant
    (:attr:`patchworks.body.CellBody.fold_gradient_norms`); the offset is the
    norm of the displacement the message-passing phase already formed. The
    counters below are a tick count and a crossing count — nothing here
    averages, so nothing here has a rate to set.
    """

    def __init__(self, cells: int) -> None:
        #: `[predicting cells]`: last tick's margin, `min_i |z_i| / ‖∇z_i‖`.
        self.margin = torch.zeros(cells)
        #: `[predicting cells]`: the length of last tick's reconciliation
        #: displacement, over the node stalk's `n` coordinates alone. The
        #: comparison against :attr:`margin` is **conservative** in that
        #: respect: displacement along a coordinate subspace is never less than
        #: the perpendicular distance the margin measures.
        self.offset = torch.zeros(cells)
        #: `[predicting cells]`: how many times this cell's activation pattern
        #: has changed since the read began.
        self.crossings = torch.zeros(cells)
        #: Ticks observed. Kept as the denominator for rates read off this
        #: instrument, not as a burn-in: #202 measured that no burn-in exists
        #: and #206 struck the clause, leaving the margin-against-offset
        #: comparison an attribution with no threshold to reach.
        self.ticks = 0
        self._region: torch.Tensor | None = None

    def observe_inference(self, pre_activation: torch.Tensor, body: CellBody) -> None:
        """Read the margin and the activation region off `encode`'s pre-activations."""
        self.margin = body.fold_margin(pre_activation)
        here = pre_activation > 0
        if self._region is not None:
            self.crossings += (here != self._region).any(dim=-1).to(self.crossings.dtype)
        self._region = here
        self.ticks += 1

    def observe_reconciliation(self, displacement: torch.Tensor) -> None:
        """Read the standing offset off the displacement reconciliation applied."""
        self.offset = torch.linalg.vector_norm(displacement, dim=-1)

    @property
    def dwell(self) -> torch.Tensor:
        """`[predicting cells]`: mean ticks in one activation region so far.

        The same quantity :attr:`patchworks.bias_selection.Measurement.dwell`
        reports on the construction sweep, so the two are comparable directly —
        which is the whole point of nominating and then deciding.
        """
        return self.ticks / (1.0 + self.crossings)

    @property
    def reconciliation_reaches(self) -> torch.Tensor:
        """`[predicting cells]` bool: the offset is at least the margin.

        The attribution. True means last tick's reconciliation was on its own
        large enough to carry the cell out of its activation region — the
        failure ADR-0007 names, observed rather than bounded. It does not say
        the cell *did* leave: the displacement's direction may point across the
        region rather than out of it, which is why this is read alongside
        :attr:`dwell` and not instead of it.
        """
        return self.offset >= self.margin

    def state(self) -> dict[str, object]:
        """Everything the read carries between ticks, cloned.

        The read is state a tick moves, so anything re-running a graph from a
        saved place has to put it back — `benchmarks/untrained_fixed_point.py`'s
        `sensitivity` runs six variants against one reference, and a crossing
        count left running across them would make the six incomparable.
        """
        return {
            "margin": self.margin.clone(),
            "offset": self.offset.clone(),
            "crossings": self.crossings.clone(),
            "ticks": self.ticks,
            "region": None if self._region is None else self._region.clone(),
        }

    def load(self, state: dict[str, object]) -> None:
        """Put back what :meth:`state` took."""
        margin, offset, crossings, region = (
            state["margin"],
            state["offset"],
            state["crossings"],
            state["region"],
        )
        assert isinstance(margin, torch.Tensor)
        assert isinstance(offset, torch.Tensor)
        assert isinstance(crossings, torch.Tensor)
        self.margin, self.offset, self.crossings = (
            margin.clone(),
            offset.clone(),
            crossings.clone(),
        )
        self.ticks = int(state["ticks"])  # type: ignore[arg-type]
        self._region = None if region is None else region.clone()  # type: ignore[union-attr]

    def __repr__(self) -> str:
        if self.ticks == 0:
            return "FoldRead(nothing read yet)"
        return (
            f"FoldRead({self.ticks} ticks, "
            f"median margin {float(self.margin.median()):.4f}, "
            f"median offset {float(self.offset.median()):.4f}, "
            f"median dwell {float(self.dwell.median()):.1f} ticks, "
            f"reconciliation reaches {int(self.reconciliation_reaches.sum())} "
            f"of {self.margin.numel()} cells)"
        )


class StalkLayout:
    """Where every cell's node stalk lives in the flat buffer, and the indices over it.

    Built once from the dome. Nothing here reads a cell's construction layout —
    every index is a precomputed array, and the tick touches nothing else.
    """

    def __init__(self, dome: Dome, maps: RestrictionMaps) -> None:
        self.dome = dome
        offsets: list[int] = []
        total = 0
        for cell in dome.cells:
            offsets.append(total)
            total += cell.stalk
        self.offsets = tuple(offsets)
        self.total = total
        #: The trailing slot. Permanently zero in a node stalk buffer, and the
        #: bin every padded scatter is aimed at.
        self.pad = total

        n = dome.shape.n
        #: `[predicting cells, n]`: where the population the body runs on lives,
        #: in the row order the biases are indexed by.
        self.predicting_positions = torch.tensor(
            [[offsets[c] + i for i in range(n)] for c in dome.predicting],
            dtype=torch.long,
        )

        #: `[pairs, stalk_max]`: one row per edge endpoint, in the pair order
        #: :mod:`patchworks.restriction` fixes — the flat positions of that
        #: pair's owning cell's node stalk, padded out to the widest stalk in
        #: the graph with the zero slot. Both the gather into the maps and the
        #: scatter back out of them run through it, which is what stops the two
        #: from ever disagreeing about which component is which.
        positions = torch.full((maps.pairs, maps.stalk_width), self.pad, dtype=torch.long)
        for edge in dome.edges:
            for side, cell_id in enumerate((edge.u, edge.v)):
                stalk = dome.cells[cell_id].stalk
                base = offsets[cell_id]
                positions[pair_index(edge.id, side), :stalk] = torch.arange(
                    base, base + stalk
                )
        self.pair_positions = positions

    def slice(self, cell_id: int) -> slice:
        """The flat positions of one cell's node stalk."""
        base = self.offsets[cell_id]
        return slice(base, base + self.dome.cells[cell_id].stalk)

    def empty(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """A zeroed flat node stalk buffer, pad slot included."""
        return torch.zeros(self.total + 1, dtype=dtype)

    def per_component(self, per_cell: torch.Tensor) -> torch.Tensor:
        """`[cells]` in, `[total + 1]` out: one cell's value on each of its components.

        The pad slot gets zero, which is what keeps a padded scatter inert.
        """
        spread = torch.zeros(self.total + 1, dtype=per_cell.dtype)
        for cell in self.dome.cells:
            spread[self.slice(cell.id)] = per_cell[cell.id]
        return spread


class Sheaf:
    """The graph as it runs: charts, node stalks, edge buffers, and the tick.

    Owns the state a tick moves and nothing that belongs to the world. The
    world's half of the ordering — reading the actuator's commanded components
    and writing the boundary cells afterwards — is
    :class:`patchworks.agent.Agent`'s, which is what keeps this class free of
    anything that knows the sandbox exists.
    """

    def __init__(
        self,
        dome: Dome,
        *,
        body: CellBody | None = None,
        biases: CellBiases | None = None,
        operators: CellOperators | None = None,
        maps: RestrictionMaps | None = None,
        gamma: float = DEFAULT_GAMMA,
        generator: torch.Generator | None = None,
    ) -> None:
        # Checked here, before the draws, rather than left to the gain at the
        # bottom of this constructor: the body, the per-cell surface and the
        # maps are the expensive part of building a sheaf, and a `γ` that is going to be
        # refused anyway is not worth drawing them for (#107). The rule is not
        # restated here — the gain reads it from the same one place.
        self.gamma = _checked_gamma(gamma)
        self.dome = dome
        self.body = body if body is not None else CellBody(dome.shape, generator=generator)
        self.biases = (
            biases
            if biases is not None
            else CellBiases(dome.shape, len(dome.predicting), generator=generator)
        )
        #: The cell operators `K`, the other half of the per-cell surface
        #: (#138). Drawn here rather than by `CellBiases`, because they are a
        #: sibling module: `a·I` at construction takes no generator, so unlike
        #: every other piece this one is deterministic given the shape.
        self.operators = (
            operators
            if operators is not None
            else CellOperators(dome.shape, len(dome.predicting))
        )
        self.maps = maps if maps is not None else RestrictionMaps(dome, generator=generator)
        # Refused here rather than at the first tick: the layout below indexes
        # one flat buffer by cell and by edge endpoint, so a surface built for
        # another graph would read the wrong components rather than fail.
        if self.maps.dome is not dome:
            raise ValueError("the restriction maps were built for a different dome")
        if self.biases.cells != len(dome.predicting):
            raise ValueError(
                f"biases for {self.biases.cells} cells against this dome's "
                f"{len(dome.predicting)} predicting cells"
            )
        if self.operators.cells != len(dome.predicting):
            raise ValueError(
                f"operators for {self.operators.cells} cells against this dome's "
                f"{len(dome.predicting)} predicting cells"
            )
        # `generator` seeds whatever this constructor was not handed. Hand it
        # the body, the biases *and* the maps and there was nothing left to
        # draw, so the generator was consumed by nothing — refused rather than
        # accepted in silence, on #106's grounds:
        # `Sheaf(dome, body=b, biases=c, maps=m, generator=g)` is written by
        # someone who believes the run is seeded, and a run that is not
        # reproducible while its author believes it is fails plausibly, in the
        # numbers, long after the fact.
        #
        # Last of the three refusals rather than first, though it is the only
        # one that needs nothing drawn to decide. The two above it are mistakes
        # that actually cost something — a surface built for another graph
        # would read the wrong components rather than fail, and biases sized
        # against another population are wrong about this dome — whereas an
        # inert generator only wastes an argument. A caller who made both
        # should hear the costly one now rather than on a second run. Both
        # orderings are pinned in tests/test_tick.py::TestAnInertGenerator.
        #
        # **#106's rule does not transfer verbatim, and that is worth knowing.**
        # There it reads *nothing in this constructor consumes a construction
        # argument once the thing it constructs is supplied*, which is exact at
        # `Agent` because the generator there feeds one construction: the
        # sheaf. Here it feeds three, drawn independently, so "the thing it
        # constructs" has no single referent — and one or two supplied pieces
        # leave the rest genuinely drawn from the generator, which is why
        # `Sheaf(dome, body=b, generator=g)` is not an error. The rule both
        # levels are instances of is the finer one: **an argument is refused
        # when nothing is left for it to do.** #106's wording is that rule
        # where the argument feeds exactly one thing; the condition here is all
        # three feeds supplied at once.
        #
        # `gamma` needs no companion check: it is consumed on every path, since
        # the gain below is computed from it whatever else was handed in.
        # The operators are not in this conjunction, and deliberately: they are
        # `a·I` and take no generator, so supplying them leaves the generator
        # exactly as much to do as it had before.
        nothing_left_to_draw = body is not None and biases is not None and maps is not None
        if generator is not None and nothing_left_to_draw:
            raise ValueError(
                "generator seeds the body, the biases and the maps, and all three "
                "were supplied already drawn, so it would seed nothing — drop it, or "
                "draw the piece you meant it for here: "
                "Sheaf(dome, body=..., biases=..., generator=g) still draws the "
                "maps from g."
            )
        self.layout = StalkLayout(dome, self.maps)

        self.gain = reconciliation_gain(dome, gamma=self.gamma, rho=self.maps.rho)
        self._gain_per_component = self.layout.per_component(self.gain)

        #: `[total + 1]`: every cell's node stalk, end to end. The cell's public
        #: face — what the inference phase reads, what reconciliation edits, and
        #: what the world writes and reads at the boundary.
        self.stalks = self.layout.empty()
        #: `[predicting cells, k]`: the persisted chart. The cell's private
        #: state; reconciliation never reaches it.
        self.charts = torch.zeros(len(dome.predicting), dome.shape.k)
        #: `[pairs, m_max]`: what every cell put on every incident edge stalk
        #: **this** tick. Read one tick later, which is the unit delay.
        self.broadcast = torch.zeros(self.maps.pairs, self.maps.edge_width)
        #: `[pairs, m_max]`: what each cell reconciled against — its neighbour's
        #: broadcast from `t − 1`. Kept because the transport rule (#89) learns
        #: on it, and it is a tick's own record of what it was told.
        self.incoming = torch.zeros_like(self.broadcast)
        #: `[predicting cells, n]`: what `decode` predicted this tick, before
        #: reconciliation edited it. What the prediction rule's prediction error is
        #: measured against next tick — though the rule never descends on *this*
        #: tensor, which is dead and has no gradient in anything.
        self.prediction = torch.zeros(len(dome.predicting), dome.shape.n)
        #: `[predicting cells, k]` and `[predicting cells, n]`: what the last
        #: inference phase **read** — the persisted chart it advanced from and
        #: the node stalk it took as evidence. Kept for the same reason
        #: :attr:`incoming` is: the prediction rule (#88) re-runs exactly that forward
        #: path, live in the biases and `K`, against the node stalk reconciliation has
        #: since left behind (`docs/spec/09-the-build-stack.md`, *Learning is a
        #: separate phase over detached inputs*).
        self.prior_charts = torch.zeros(len(dome.predicting), dome.shape.k)
        self.prior_evidence = torch.zeros(len(dome.predicting), dome.shape.n)
        #: The live fold read (ADR-0019). Always on rather than opt-in, for
        #: :func:`assert_no_tape`'s reason: a diagnostic that has to be
        #: switched on is one that is off in every run nobody suspected yet,
        #: and this one costs an `abs`, a divide, a `min` and a norm per tick
        #: against a graph of matrix products.
        self.fold_read = FoldRead(len(dome.predicting))
        self.ticks = 0

    # -- the two phases ----------------------------------------------------

    def inference_phase(self) -> None:
        """Every predicting cell advances its own chart and decodes a prediction.

        Simultaneous and independent: one batched `encode` / `step` / `decode`
        over the whole population, reading only each cell's own persisted chart
        and its own node stalk. No cell reads another's state here — which the
        batching makes structural, since there is nothing in the call that could
        carry a neighbour's value.

        The prediction becomes the cell's node stalk. `encode` fusing the
        persisted chart with that stalk is how reconciliation's corrections
        re-enter inference: as *evidence* on the next tick, never as an edit to
        the chart.

        Carries its own `no_grad` and its own guard rather than inheriting
        :meth:`tick`'s. The phases are public — the perturbation test and the
        learning phase both have reason to run one on its own — and a phase
        whose guard lived in its caller would leak silently the first time one
        did. Nesting `no_grad` is free.
        """
        with torch.no_grad():
            evidence = self.evidence()
            # The pair the prediction rule re-runs. Both are already private copies:
            # the advanced chart is *rebound* below rather than written into,
            # and an advanced-index gather returns a fresh tensor rather than a
            # view -- so neither record can be moved out from under the rule by
            # the message-passing phase's in-place edits to the stalk buffer.
            self.prior_charts, self.prior_evidence = self.charts, evidence
            # Split into its two halves rather than one `self.body(...)` call,
            # for the pre-activation in the middle: it is what the fold margin
            # and the activation region are read from, and reading them off
            # this forward path is what stops the live read from measuring a
            # body that is not the one running (ADR-0019). Same arithmetic —
            # `body.forward` is these two calls — so the split costs nothing.
            pre_activation, fused = self.body.encode_parts(
                self.charts, evidence, self.biases
            )
            self.charts, self.prediction = self.body.advance(
                fused, self.biases, self.operators
            )
            self.stalks[self.layout.predicting_positions] = self.prediction
            self.fold_read.observe_inference(pre_activation, self.body)
        self.assert_no_tape()

    def message_passing_phase(self) -> None:
        """One simultaneous local descent step per cell, against a `t − 1` belief.

        Each cell restricts its own predicted node stalk onto every incident
        edge, compares that with what its neighbour restricted onto the same
        edge stalk one tick ago, and takes one step down its own local
        disagreement energy `½ Σ_{e∋v} ‖F_ev x_v − y_e‖²`::

            x_v  ←  x_v − gain_v · Σ_{e∋v} F_evᵀ (F_ev x_v − y_e(t−1))

        Every cell's step is computed from the same pre-phase configuration and
        applied at once, so there is no visiting order — and no cell's update
        can see another's, which is Jacobi rather than Gauss-Seidel and is what
        makes "simultaneously" in the spec mean something.

        The result edits the node stalk only. A private feature comes back from
        `spread` as exactly zero, because the mask that made it private zeroed
        the very columns of `F_ev` this sum runs through — so `H⁰` insulation is
        a property of the maps rather than a second rule applied here.

        What is **not** here is as load-bearing as what is: no round count, no
        residual norm, no convergence check, no early stopping. Any legitimate
        stopping rule would need a read of disagreement across the graph, which
        is the global aggregate graph-locality exists to rule out (ADR-0002).

        Carries its own `no_grad` and its own guard, for the reason
        :meth:`inference_phase` gives.
        """
        with torch.no_grad():
            gathered = self.stalks[self.layout.pair_positions]
            outgoing = self.maps.restrict(gathered)
            # The unit delay, as an index flip: pair `2e` and pair `2e + 1` are
            # the two ends of edge `e`, so the belief a cell reconciles against
            # is its partner's slot in *last* tick's broadcast.
            incoming = (
                self.broadcast.reshape(-1, 2, self.maps.edge_width)
                .flip(1)
                .reshape_as(self.broadcast)
            )
            contribution = self.maps.spread(outgoing - incoming)

            delta = torch.zeros_like(self.stalks)
            positions = self.layout.pair_positions
            delta.index_add_(0, positions.reshape(-1), contribution.reshape(-1))
            delta.mul_(self._gain_per_component)
            # Read before the subtraction, off the gained displacement itself:
            # the standing offset is what reconciliation moves the operating
            # point by, and the predicting cells' `n` coordinates are the whole
            # of what `encode` reads back (ADR-0019). Boundary cells are absent
            # because they run no body and so have no fold margin to compare a
            # displacement against.
            self.fold_read.observe_reconciliation(delta[self.layout.predicting_positions])
            self.stalks.sub_(delta)

            self.broadcast = outgoing
            self.incoming = incoming
        self.assert_no_tape()

    def tick(self) -> None:
        """One tick of the graph: the inference phase, then the message-passing phase.

        The world's write lands after this returns and wins — see
        :meth:`patchworks.agent.Agent.tick`, which is where a whole tick,
        ordering included, actually happens.

        The `no_grad` and the guard here are redundant with the phases' own, and
        kept anyway: the spec's sentence is about the *whole tick*, and a guard
        that costs two microseconds is not worth making conditional on nobody
        ever reordering this method.
        """
        with torch.no_grad():
            self.inference_phase()
            self.message_passing_phase()
        self.ticks += 1
        self.assert_no_tape()

    # -- the guard ---------------------------------------------------------

    def assert_no_tape(self) -> None:
        """Check every quantity this tick produced. Cheap, and always on."""
        assert_no_tape(
            charts=self.charts,
            stalks=self.stalks,
            prediction=self.prediction,
            broadcast=self.broadcast,
            incoming=self.incoming,
            prior_charts=self.prior_charts,
            prior_evidence=self.prior_evidence,
            fold_margin=self.fold_read.margin,
            standing_offset=self.fold_read.offset,
        )

    # -- reading the state -------------------------------------------------

    def stalk(self, cell_id: int) -> torch.Tensor:
        """One cell's node stalk, as a view on the flat buffer."""
        return self.stalks[self.layout.slice(cell_id)]

    def evidence(self) -> torch.Tensor:
        """`[predicting cells, n]`: what every predicting cell now reads in as evidence.

        A gather, so the result is a fresh tensor rather than a view on the
        flat buffer — reconciliation's next in-place edit cannot reach back
        into one of these.

        The inference phase reads it, and the prediction rule takes it as the
        detached target: between the two it is the node stalk reconciliation
        left behind, which is how prediction error carries the neighbours'
        disagreement without the rule reading a neighbour
        (`docs/spec/07-local-learning-rule.md`, *The prediction rule*).
        """
        return self.stalks[self.layout.predicting_positions]

    def disagreement(self) -> torch.Tensor:
        """`[edges, m_max]`: the difference between the two ends' restrictions.

        Derived, never transported — and derived here from what each end
        *broadcast*, so the two terms are the same tick's and the delay does not
        show up inside it. The sign is `u`'s restriction minus `v`'s. Rows past
        an edge's own `m` are zero.
        """
        ends = self.broadcast.reshape(-1, 2, self.maps.edge_width)
        return ends[:, 0] - ends[:, 1]
