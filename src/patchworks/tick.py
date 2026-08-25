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

**Node stalks are one flat buffer.** Boundary cells are exempt from `n` and
carry the world's shape instead, so the population's stalks are ragged. They
are stored end to end in a single tensor with a per-cell offset, plus one
trailing slot that is permanently zero: padded gathers read it and padded
scatters land in it, which is what lets the ragged graph run as a handful of
whole-population tensor operations with no loop over cells or edges.
"""

from __future__ import annotations

import torch

from .body import CellBiases, CellBody
from .graph import Dome
from .restriction import GAUGE_RHO, RestrictionMaps, pair_index

__all__ = [
    "DEFAULT_GAMMA",
    "Sheaf",
    "StalkLayout",
    "assert_no_tape",
    "reconciliation_gain",
]

#: The single global `γ` of the reconciliation gain, `γ ≤ 1`
#: (`docs/spec/02-tick-semantics.md`, *Reconciliation gain*). One scalar for the
#: whole graph: the per-cell variation is entirely in the denominator, which is
#: a structural quantity. `γ` is capped globally by the tightest cell's fold
#: margin, a construction-time check that belongs to bias selection (#85); until
#: that check exists this sits at the bound the spec states, which is the only
#: value in `(0, 1]` that is not an unmotivated constant.
DEFAULT_GAMMA = 1.0


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
    if not 0.0 < gamma <= 1.0:
        raise ValueError(
            "gamma is a single global scalar in (0, 1] "
            f"(docs/spec/02-tick-semantics.md, Reconciliation gain); got {gamma}"
        )
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

        n = dome.spec.n
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
        maps: RestrictionMaps | None = None,
        gamma: float = DEFAULT_GAMMA,
        generator: torch.Generator | None = None,
    ) -> None:
        self.dome = dome
        self.body = body if body is not None else CellBody(dome.shape, generator=generator)
        self.biases = (
            biases
            if biases is not None
            else CellBiases(dome.shape, len(dome.predicting), generator=generator)
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
        nothing_left_to_draw = body is not None and biases is not None and maps is not None
        if generator is not None and nothing_left_to_draw:
            raise ValueError(
                "generator seeds the body, the biases and the maps, and all three "
                "were supplied already drawn, so it would seed nothing — drop it, or "
                "draw the piece you meant it for here: "
                "Sheaf(dome, body=..., biases=..., generator=g) still draws the "
                "maps from g."
            )
        self.gamma = gamma
        self.layout = StalkLayout(dome, self.maps)

        self.gain = reconciliation_gain(dome, gamma=gamma, rho=self.maps.rho)
        self._gain_per_component = self.layout.per_component(self.gain)

        #: `[total + 1]`: every cell's node stalk, end to end. The cell's public
        #: face — what the inference phase reads, what reconciliation edits, and
        #: what the world writes and reads at the boundary.
        self.stalks = self.layout.empty()
        #: `[predicting cells, k]`: the persisted chart. The cell's private
        #: state; reconciliation never reaches it.
        self.charts = torch.zeros(len(dome.predicting), dome.spec.k)
        #: `[pairs, m_max]`: what every cell put on every incident edge stalk
        #: **this** tick. Read one tick later, which is the unit delay.
        self.broadcast = torch.zeros(self.maps.pairs, self.maps.edge_width)
        #: `[pairs, m_max]`: what each cell reconciled against — its neighbour's
        #: broadcast from `t − 1`. Kept because the transport rule (#89) learns
        #: on it, and it is a tick's own record of what it was told.
        self.incoming = torch.zeros_like(self.broadcast)
        #: `[predicting cells, n]`: what `decode` predicted this tick, before
        #: reconciliation edited it. What the bias rule's prediction error is
        #: measured against next tick — though the rule never descends on *this*
        #: tensor, which is dead and has no gradient in anything.
        self.prediction = torch.zeros(len(dome.predicting), dome.spec.n)
        #: `[predicting cells, k]` and `[predicting cells, n]`: what the last
        #: inference phase **read** — the persisted chart it advanced from and
        #: the node stalk it took as evidence. Kept for the same reason
        #: :attr:`incoming` is: the bias rule (#88) re-runs exactly that forward
        #: path, live in the biases, against the node stalk reconciliation has
        #: since left behind (`docs/spec/09-the-build-stack.md`, *Learning is a
        #: separate phase over detached inputs*).
        self.prior_charts = torch.zeros(len(dome.predicting), dome.spec.k)
        self.prior_evidence = torch.zeros(len(dome.predicting), dome.spec.n)
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
            # The pair the bias rule re-runs. Both are already private copies:
            # the advanced chart is *rebound* below rather than written into,
            # and an advanced-index gather returns a fresh tensor rather than a
            # view -- so neither record can be moved out from under the rule by
            # the message-passing phase's in-place edits to the stalk buffer.
            self.prior_charts, self.prior_evidence = self.charts, evidence
            self.charts, self.prediction = self.body(self.charts, evidence, self.biases)
            self.stalks[self.layout.predicting_positions] = self.prediction
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

        The inference phase reads it, and the bias rule takes it as the
        detached target: between the two it is the node stalk reconciliation
        left behind, which is how prediction error carries the neighbours'
        disagreement without the rule reading a neighbour
        (`docs/spec/07-local-learning-rule.md`, *The bias rule*).
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
