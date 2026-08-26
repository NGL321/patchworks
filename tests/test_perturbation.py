"""The perturbation test: move one cell's parameters, and nothing else's update moves.

The load-bearing half of the locality guard (ticket #90).
`docs/spec/09-the-build-stack.md`, *The guarantee is tested, because a leak
would flatter us*, together with `docs/spec/07-local-learning-rule.md`,
*Locality boundary*, and
`docs/adr/0011-the-locality-guarantee-is-enforced-not-inherited.md`.

The thesis' central constraint is **cell-local learning rules only** — no error
signal propagated across the graph. PyTorch does not supply that; it is
enforced, and this file is where it is falsifiable. A leaked gradient does not
crash and does not look like a bug. It makes the agent work *better*, which
looks like the thesis being right, and that asymmetry is the whole argument for
a standing test rather than a review habit.

**Why this is not scaffolding around the cheap check.** The always-on assertion
(:func:`patchworks.tick.assert_no_tape`, exercised in `tests/test_tick.py`)
inspects the **tape**; this file inspects the **update**. Neither subsumes the
other, and the gap is documented rather than hypothetical: `Tensor.detach`
returns a tensor **sharing storage** with the original, so an in-place write
through a detached view couples two cells while leaving a perfectly clean tape
— no `grad_fn` anywhere, and a batched graph that genuinely has no cross-cell
edges. :class:`TestTheSharedStorageCase` builds exactly that leak on purpose,
watches the assertion stay silent on it, and watches these tests catch it.

**And why the `torch.func` transform does not make this redundant.** The
transform closes **parameter reachability**, structurally, and closes nothing
else (`09-the-build-stack.md`, *Written as a function transform*): a
neighbour's parameter is not among the `argnums`, so differentiation never
traverses it. It does not touch the shared-storage class, because that class
never goes near a gradient. This file is what covers the rest.

**What "the update" means here.** The gradient each rule would apply, taken
without applying it — :meth:`~patchworks.learning.BiasRule.gradient` and
:meth:`~patchworks.learning.TransportRule.gradient`. Updates are compared
**bit-identically** (`torch.equal`), which is the only comparison a leak cannot
hide under: a tolerance is a budget, and a coupling small enough to fit inside
one is still a coupling.

**The tick's state is held fixed across a perturbation, deliberately.** The
guard is a claim about the *learning phase* — "a function of that cell's own
adapting surface plus detached arrays" — not about the tick. Ticking between
the two readings would let the perturbation reach a neighbour through
reconciliation, which is the architecture working as specified
(graph-local exchange), and would say nothing about whether the rules are
cell-local. :class:`TestThePermittedChannel` is where the one legitimate
cross-cell path is located and bounded.

**Measured, not assumed: this file was kill-tested against five leaks** planted
in `src/patchworks/learning.py`, and caught all five. In the order they would
occur to someone editing that module:

1. the bias objective mixing cells inside the parameter it differentiates;
2. :meth:`~patchworks.learning.ForwardPath.bias_parameters` handing back a view
   on the live surface and writing through it;
3. the transport objective recomputing a neighbour's belief from that
   neighbour's *current* map instead of taking it from
   :attr:`~patchworks.tick.Sheaf.incoming` — the forgotten `.detach()`, in the
   transform idiom;
4. the sparsity penalty normalising against a global aggregate over the graph
   rather than each map's own magnitude;
5. :meth:`~patchworks.learning.TransportPath.map_parameters` doing what (2)
   does, on the maps.

(2) and (5) are the shared-storage class, and the always-on assertion is silent
on both. A control mutation — the transport objective scaled by two, locality
untouched — left every test here passing, which is the other half of the
reading: this guard is falsifiable rather than merely tight.
"""

import ast
import copy
import locale
import re
import tomllib
from pathlib import Path

import pytest
import torch
import yaml

from patchworks.graph import CellKind, build_graph
from patchworks.learning import (
    MAPS_PARAMETER,
    BiasRule,
    ForwardPath,
    SparsityAnneal,
    TransportPath,
    TransportRule,
)
from patchworks.tick import Sheaf

from conftest import SMALL

# -- the cells these tests name --------------------------------------------
#
# Every cell id below is **read off the dome the spec builds**, never typed.
# The distinction is this file's own failure mode one level down: a cell id
# typed here and a spec edited elsewhere do not disagree loudly, they just
# leave the sweep perturbing some other cell, and a guard aimed at the wrong
# cell goes on passing. Deriving them means the spec cannot move without
# taking them with it, and :class:`TestTheCellsTheseTestsName` pins what each
# one is for, so a derivation that started selecting a cell of the wrong kind
# fails rather than quietly changing what is being guarded.
#
# All of it runs at import, because `parametrize` needs the ids at collection.
# That puts this module's collection -- `TestBothChecksRunInCI` with it -- on
# the far side of `build_graph` succeeding, so the derivations below are
# written to state what went wrong rather than to raise a bare `StopIteration`
# a reader has to work backwards from. Everything they can defer to a real
# assertion, they do: `DRIVE_CELLS` is a tuple here and *how many* there are is
# :class:`TestTheCellsTheseTestsName`'s to say.

_DOME = build_graph(SMALL)


def _cells_of(kind):
    """Every cell of one kind, lowest id first."""
    return tuple(cell.id for cell in _DOME.cells if cell.kind is kind)


def _first(kind):
    """The lowest-numbered cell of one kind."""
    cells = _cells_of(kind)
    if not cells:
        raise AssertionError(
            f"the shared spec (tests/conftest.py) builds no {kind.value} cell, "
            f"which this file names one of; see TestTheCellsTheseTestsName"
        )
    return cells[0]


#: This dome's widest cell — seven edges — and a predicting cell, so the
#: transport rule's per-edge path is exercised where it has most to reach.
#: Several cells tie at the widest degree and `.index` takes the lowest, which
#: is a rule rather than a choice: the same spec always yields the same cell.
WIDEST = _DOME.degrees.index(max(_DOME.degrees))

#: The drive boundary cells, and the one this file perturbs: three edges, each
#: of mask width 1. `CONTEXT.md` has one cell to one drive, so the tuple should
#: hold exactly one — asserted rather than unpacked, so that a dome carrying
#: two fails as a sentence about drives instead of aborting this module.
DRIVE_CELLS = _cells_of(CellKind.DRIVE)
DRIVE_CELL = _first(CellKind.DRIVE)


#: A spread to perturb one at a time: a sensory cell the world writes, a
#: proprioceptive one, the actuator, the widest predicting cell, and the drive.
#: The whole-graph sweep beside it makes the standing claim; this spread is
#: what makes a failure readable, so it wants one cell of each shape rather
#: than every cell.
A_SPREAD = [
    _first(CellKind.PATCH),
    _first(CellKind.PROPRIOCEPTIVE),
    _first(CellKind.ACTUATOR),
    WIDEST,
    DRIVE_CELL,
]


# -- the rows and endpoints these tests name by hand -----------------------
#
# Two of the index sets below are not cell ids and are not derived: the bias
# rows, which number the predicting cells in `dome.predicting` order (see
# :func:`perturb_the_biases_of`), and the edge endpoints, which number the two
# ends of every edge as `2 * edge.id + side` — what the `running` fixture
# writes through and what `endpoint ^ 1` flips. Neither numbering offers
# anything to select on — a bias row is not of a kind and an endpoint is not
# the widest of anything — so these are arbitrary representatives, spread
# across their array and no more meaningful than any other. Picking them by a
# rule would dress a choice up as a derivation, and what each stands in for
# reads the same at every row and every endpoint: the bias rows have the
# whole-population sweep beside them and are parametrised so that a failure
# names one, and the four endpoints are four samples of a claim
# :class:`TestThePermittedChannel` makes about that numbering as a whole.
#
# What typing them costs is that a dome retuned smaller can leave one off the
# end of its array, where it fails as an `IndexError` from inside a helper
# rather than as anything about the dome. They are gathered here rather than
# typed at each use, so that :class:`TestTheCellsTheseTestsName` can bound all
# of them against the arrays they index.

#: A bias row well inside the population, for the tests that want a single row
#: rather than a spread.
A_BIAS_ROW = 7

#: The bias rows perturbed one at a time: on this dome the first two, one
#: mid-population, and the last.
NAMED_BIAS_ROWS = (0, 1, A_BIAS_ROW, 14)

#: The `incoming` rows :class:`TestThePermittedChannel` perturbs directly: two
#: at the start of the numbering and two well into it, of both parities, since
#: an endpoint's parity is which end of its edge it is.
NAMED_ENDPOINTS = (0, 1, 43, 100)

#: How hard a perturbation pushes. Large enough that the perturbed cell's own
#: update moves well clear of float32's last bits -- a perturbation too small
#: to be felt would make every assertion below pass on nothing -- and small
#: enough to stay a perturbation of the surface rather than a different graph.
NUDGE = 0.25


@pytest.fixture
def dome():
    return build_graph(SMALL)


@pytest.fixture
def running(dome):
    """A sheaf a few ticks into a run, with something in every buffer.

    A freshly built sheaf is all zeros and a fresh dome is unstimulated, so
    every update below would be trivially zero and a leak would have nothing to
    carry. Three ticks, because the transport rule refuses to run before two
    have happened where the bias rule needs one: an edge carries a unit delay,
    so on an untouched sheaf the first tick still reconciles against the
    constructor's zeros. Seeding `broadcast` here puts a belief on every edge
    before the first of the three, so `incoming` carries something real from
    tick one — the rule's `ticks < 2` refusal is what fixes the count, not this
    fixture's arithmetic.
    """
    sheaf = Sheaf(dome, generator=torch.Generator().manual_seed(0))
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        sheaf.stalks[: sheaf.layout.total] = torch.randn(
            sheaf.layout.total, generator=generator
        )
        sheaf.charts.normal_(0.0, 1.0, generator=generator)
        for edge in dome.edges:
            for side in (0, 1):
                sheaf.broadcast[2 * edge.id + side, : edge.m].normal_(
                    0.0, 1.0, generator=generator
                )
    for _ in range(3):
        sheaf.tick()
    return sheaf


# -- reading an update -----------------------------------------------------


def bias_update(sheaf, rule=BiasRule):
    """The bias rule's update, keyed by bias name, each `[cells, ·]`."""
    return rule(sheaf).gradient()


def map_update(sheaf, rule=TransportRule):
    """The transport rule's update, `[pairs, m_max, stalk_max]`.

    Taken one step onto the anneal's ceiling rather than at the schedule's
    origin, so the sparsity term is carrying its full pressure while the
    comparison is made. At step zero the pressure is exactly zero and half the
    objective would be untested — a leak living in the penalty would pass.
    """
    stepped = rule(sheaf, anneal=SparsityAnneal(horizon=1))
    stepped.steps = 1
    return stepped.gradient()


def moved_rows(before, after):
    """Which rows of an update are not bit-identical between two readings."""
    return frozenset(
        row for row in range(before.shape[0]) if not torch.equal(before[row], after[row])
    )


def cells_whose_bias_update_moved(before, after):
    """The union over all six bias vectors: which cells' updates moved at all."""
    return frozenset().union(
        *(moved_rows(before[name], after[name]) for name in before)
    )


# -- perturbing one cell ---------------------------------------------------


def perturb_the_biases_of(sheaf, cell, *, seed=11):
    """Move every one of one cell's six bias vectors, and no other cell's.

    `cell` here indexes the **biases' own `[cells, ·]` leading dimension** —
    the predicting cells in `dome.predicting` order — where
    :func:`perturb_the_restriction_maps_of` takes a dome cell id, since every
    cell in the graph holds maps and only predicting cells hold biases. The two
    numberings are kept apart rather than reconciled, because each rule's
    update is indexed the way that rule's parameter is.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in sheaf.biases.parameters():
            parameter[cell] += torch.empty(parameter.shape[1:]).normal_(
                0.0, NUDGE, generator=generator
            )


def owned_endpoints(sheaf, cell):
    """The edge endpoints whose restriction map that cell holds."""
    return frozenset((sheaf.maps.owner == cell).nonzero().flatten().tolist())


def perturb_the_restriction_maps_of(sheaf, cell, *, seed=11):
    """Move every map one cell holds, and no other cell's. Returns its endpoints.

    Masked by the structural mask, because a perturbation that opened a closed
    weight would be a different *graph* rather than a different adapting
    surface, and the mask closes permanently
    (:mod:`patchworks.restriction`).

    Deliberately **not** followed by the gauge projection. Two reasons, and the
    first is the substantive one: the guard has to hold for whatever values a
    cell's parameters take, and a leak does not get to hide behind an
    enforcement step. The second is arithmetic — the projection is a global
    rescale whose fixed point is not bit-exact for a pinned map, so running it
    would move rows the perturbation never touched and the comparison below
    would be reading floating point rather than locality.
    """
    endpoints = sorted(owned_endpoints(sheaf, cell))
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        draw = torch.empty(len(endpoints), *sheaf.maps.maps.shape[1:]).normal_(
            0.0, NUDGE, generator=generator
        )
        sheaf.maps.maps[endpoints] += draw * sheaf.maps.support[endpoints]
    return frozenset(endpoints)


# -- the deliberate leak ---------------------------------------------------
#
# The shared-storage class, built on purpose. Both halves are needed and
# neither is exotic:
#
# * `Tensor.detach` shares storage with the `nn.Parameter` it came from, so the
#   shipped paths **clone** as well as detach. Dropping the clone is a one-word
#   regression that reads like a micro-optimisation.
# * A *write* through that alias is what turns the sharing into a coupling.
#   Both writes below are the shape of a plausible tweak — centre the
#   population's operating point, symmetrise an edge's two ends — rather than a
#   contrivance nobody would type.
#
# Nothing here goes near the tape. The coupling happens on plain detached
# tensors before the transform is ever called, so the batched graph the
# transform then builds genuinely has no cross-cell edge in it, and the
# always-on assertion is silent by construction rather than by luck.


class LeakingForwardPath(ForwardPath):
    """:class:`~patchworks.learning.ForwardPath` with the clone dropped.

    And then one in-place write through the alias: the population's decode
    biases are centred over cells. Every cell's operating point is now a
    function of every other cell's, and the running adapting surface has been
    written to from inside what is supposed to be a read.
    """

    LEAKED = "biases.decode_output_bias"

    def bias_parameters(self):
        parameters = {name: value.detach() for name, value in self.named_parameters()}
        leaked = parameters[self.LEAKED]
        leaked.sub_(leaked.mean(0, keepdim=True))
        return parameters


class LeakingBiasRule(BiasRule):
    def __init__(self, sheaf, **keywords):
        super().__init__(sheaf, **keywords)
        self.path = LeakingForwardPath(sheaf.body, sheaf.biases)


class LeakingTransportPath(TransportPath):
    """The same regression on the transport side: detach without the clone.

    The write symmetrises each edge — an endpoint's map picks up a tenth of its
    partner's, in place. The two ends of an edge belong to *different cells'*
    adapting surfaces, so this is one cell's parameters reaching another's
    update directly, with nothing on the tape to show for it.
    """

    SHARE = 0.1

    def map_parameters(self):
        parameters = {name: value.detach() for name, value in self.named_parameters()}
        maps = parameters[MAPS_PARAMETER]
        partner = maps.clone().reshape(-1, 2, *maps.shape[1:]).flip(1).reshape_as(maps)
        maps.add_(self.SHARE * partner)
        return parameters


class LeakingTransportRule(TransportRule):
    def __init__(self, sheaf, **keywords):
        super().__init__(sheaf, **keywords)
        self.path = LeakingTransportPath(sheaf.maps)


def readings(sheaf, cell, read, perturb, rule):
    """One update before a perturbation and one after, over the same tick state.

    Taken on two copies of the same sheaf rather than on one sheaf twice. The
    shipped rules write nothing and would not care, but a leaking one writes
    through its alias on every call, so reading it twice from one sheaf would
    compare a first write against a second rather than a baseline against a
    perturbation.
    """
    baseline, perturbed = copy.deepcopy(sheaf), copy.deepcopy(sheaf)
    perturb(perturbed, cell)
    return read(baseline, rule), read(perturbed, rule)


class TestTheCellsTheseTestsName:
    """What each derived cell id is *for*, pinned.

    Deriving an index off the shared spec (`tests/conftest.py`) stops it going
    stale, but it does not stop it drifting: `WIDEST` would go on naming some
    cell whatever the spec said, and if that cell stopped being a wide
    predicting one the tests below would still pass while guarding something
    else. So each derivation's premise is asserted here, and the counts the
    shared spec's comment claims with it. This is the class that fails when the
    dome is retuned, which is the point: retuning it should send someone back
    through this file rather than pass quietly.

    The two index sets that are *not* derived — the bias rows and the edge
    endpoints — have no premise to pin, being arbitrary representatives of
    their numberings. What they have is a bound, and that is what
    :meth:`test_the_rows_and_endpoints_named_by_hand_fit_this_dome` holds.
    """

    def test_the_dome_is_the_size_the_shared_spec_claims(self, dome):
        assert (len(dome.cells), len(dome.predicting), len(dome.edges)) == (39, 15, 54)

    def test_the_widest_cell_is_a_predicting_cell_with_seven_edges(self, dome):
        # Both halves are load-bearing. Predicting, because the transport
        # tests that name it are about a cell that runs the body; widest,
        # because it is the cell with the most endpoints for a leak to reach.
        assert dome.cells[WIDEST].kind is CellKind.PREDICTING
        assert dome.degrees[WIDEST] == 7
        assert dome.degrees[WIDEST] == max(dome.degrees)

    def test_there_is_exactly_one_drive_cell(self, dome):
        # `CONTEXT.md`, *Drive boundary cell*: one cell is one drive. Held here
        # rather than in the derivation above so that a second drive is a
        # readable failure and not a collection error.
        assert DRIVE_CELLS == (DRIVE_CELL,)

    def test_the_drive_cell_is_a_boundary_cell_on_three_edges_of_width_one(self, dome):
        # `test_a_flat_endpoint_is_the_objective_not_a_missing_signal` reads
        # the objective's flat point off a one-dimensional edge stalk, so the
        # mask width is the premise of that test rather than a detail.
        assert dome.cells[DRIVE_CELL].kind is CellKind.DRIVE
        assert dome.cells[DRIVE_CELL].is_boundary
        assert dome.degrees[DRIVE_CELL] == 3
        assert [dome.edges[edge].m for edge in dome.incident[DRIVE_CELL]] == [1, 1, 1]

    def test_the_spread_is_five_distinct_cells_of_five_kinds(self, dome):
        # A spread that had collapsed onto one kind would still sweep, and
        # would still pass, having stopped being a spread.
        assert len(set(A_SPREAD)) == len(A_SPREAD)
        kinds = [dome.cells[cell].kind for cell in A_SPREAD]
        assert kinds == [
            CellKind.PATCH,
            CellKind.PROPRIOCEPTIVE,
            CellKind.ACTUATOR,
            CellKind.PREDICTING,
            CellKind.DRIVE,
        ]

    def test_the_rows_and_endpoints_named_by_hand_fit_this_dome(self, running):
        # The two numberings this file types rather than derives (see *the
        # rows and endpoints these tests name by hand*). A dome retuned below
        # either fails here, as a sentence about the dome, beside whatever
        # `IndexError` the tests that name them raise from inside a helper
        # rather than instead of it.
        #
        # Bounded against the arrays those helpers subscript, not against the
        # counts beside them: `perturb_the_biases_of` writes into every one of
        # the six bias parameters, and `CellBiases.subset` sets `cells` in a
        # statement of its own, so a bound taken on `biases.cells` would be
        # reading a number kept in step by hand rather than the rows there are.
        rows = min(parameter.shape[0] for parameter in running.biases.parameters())
        assert max(NAMED_BIAS_ROWS) < rows
        assert max(NAMED_ENDPOINTS) < running.incoming.shape[0]


class TestPerturbingOneCellsBiases:
    """The bias rule's per-cell path. Prediction error is a cell's own quantity
    and crosses no edge, so the claim here admits no exception at all.

    `cell` is a row of the population's biases, not a dome cell id — see
    :func:`perturb_the_biases_of`.
    """

    @pytest.mark.parametrize("cell", NAMED_BIAS_ROWS)
    def test_no_other_cells_update_moves(self, running, cell):
        before, after = readings(
            running, cell, bias_update, perturb_the_biases_of, BiasRule
        )
        assert cells_whose_bias_update_moved(before, after) == {cell}

    def test_every_cell_in_the_graph(self, running):
        # The standing sweep. Parametrising the four above keeps the failure
        # readable; this is the claim the guard actually makes.
        for cell in range(running.biases.cells):
            before, after = readings(
                running, cell, bias_update, perturb_the_biases_of, BiasRule
            )
            assert cells_whose_bias_update_moved(before, after) == {cell}, cell

    def test_the_perturbed_cell_is_felt_on_every_one_of_its_six_biases(self, running):
        # The teeth. Without this the assertions above would pass just as
        # happily on a perturbation nothing could see, and a guard that cannot
        # tell "no leak" from "no signal" is not a guard.
        cell = A_BIAS_ROW
        before, after = readings(
            running, cell, bias_update, perturb_the_biases_of, BiasRule
        )
        for name in before:
            assert moved_rows(before[name], after[name]) == {cell}, name

    def test_the_shared_body_is_not_a_back_channel(self, running):
        # Every cell runs the *same* body, which is the one object a cross-cell
        # gradient could travel through. It cannot, because the body is frozen:
        # its weights are buffers, so they are not in the parameter dict the
        # transform differentiates, and the batched gradient stays a stack
        # rather than becoming an average.
        assert list(running.body.buffers())
        assert list(running.body.parameters()) == []


class TestPerturbingOneCellsRestrictionMaps:
    """The transport rule's per-edge path. Disagreement on an edge is a function
    of **both** its maps and each belongs to a different cell, so this is the
    half where a cross-cell parameter is genuinely in the objective."""

    @pytest.mark.parametrize("cell", A_SPREAD)
    def test_no_endpoint_the_cell_does_not_own_moves(self, running, cell):
        owned = owned_endpoints(running, cell)
        before, after = readings(
            running, cell, map_update, perturb_the_restriction_maps_of, TransportRule
        )
        assert moved_rows(before, after) <= owned

    def test_every_cell_in_the_graph(self, running):
        for cell in range(len(running.dome.cells)):
            owned = owned_endpoints(running, cell)
            before, after = readings(
                running, cell, map_update, perturb_the_restriction_maps_of, TransportRule
            )
            moved = moved_rows(before, after)
            assert moved <= owned, cell
            # The teeth, per cell: the perturbation reached at least one of the
            # cell's own endpoints, so the containment above is not vacuous.
            assert moved, cell

    def test_the_partner_across_the_shared_edge_does_not_move_either(self, running):
        # The case worth naming separately, because it is the one a reader
        # expects to be the exception. A neighbour's map is in the transport
        # objective -- but only as the belief it *already restricted* a tick
        # ago, which the tick has since written into `incoming`. Over a fixed
        # tick state it is unreachable rather than severed, so the partner's
        # update is bit-identical too.
        cell = WIDEST  # a predicting cell, and this dome's widest: seven edges
        owned = owned_endpoints(running, cell)
        partners = frozenset(endpoint ^ 1 for endpoint in owned)
        assert partners.isdisjoint(owned)
        before, after = readings(
            running, cell, map_update, perturb_the_restriction_maps_of, TransportRule
        )
        moved = moved_rows(before, after)
        assert moved.isdisjoint(partners)
        assert moved

    def test_a_flat_endpoint_is_the_objective_not_a_missing_signal(self, running):
        # Why the sweep above says "subset" where the bias sweep says "equals".
        # On a one-dimensional edge stalk whose two ends disagree in sign the
        # relative disagreement is pinned at 1, its worst value, and the
        # objective is exactly flat there -- so that endpoint's update is zero
        # whatever its map does. That is ADR-0007's objective behaving as
        # specified (tests/test_transport_rule.py holds it down directly), not
        # a perturbation nobody could feel, and this test says so out loud so
        # the weaker containment is not read as a weaker claim.
        cell = DRIVE_CELL  # the drive boundary cell: three edges, each of width 1
        before, after = readings(
            running, cell, map_update, perturb_the_restriction_maps_of, TransportRule
        )
        unmoved = owned_endpoints(running, cell) - moved_rows(before, after)
        assert unmoved
        for endpoint in unmoved:
            assert torch.equal(before[endpoint], torch.zeros_like(before[endpoint]))


class TestThePermittedChannel:
    """Where a neighbour's map is allowed to reach a cell's update, and how far.

    Exactly one place: the row of :attr:`~patchworks.tick.Sheaf.incoming` for
    the edge the two cells share — the belief the neighbour restricted onto that
    shared edge stalk a tick ago. That is the only thing the transport rule ever
    learns about a neighbour (`07-local-learning-rule.md`: never a neighbour's
    raw node stalk), it arrives already applied, and it is an ordinary argument
    outside the `argnums`.
    """

    def test_a_perturbed_map_leaves_the_graph_only_on_its_own_edges(self, running):
        # The channel, watched through the real machinery rather than
        # re-derived. Perturbing a cell's maps changes what it broadcasts onto
        # its own edge stalks and nothing else; the delay is an index flip, so
        # what lands in a neighbour's `incoming` next tick is the partner slot
        # of exactly those endpoints.
        cell = WIDEST
        owned = owned_endpoints(running, cell)
        baseline, perturbed = copy.deepcopy(running), copy.deepcopy(running)
        perturb_the_restriction_maps_of(perturbed, cell)
        # Run the phase on both, because a phase run twice off one sheaf is
        # not its own baseline: reconciliation edited the node stalks the
        # second run restricts.
        for sheaf in (baseline, perturbed):
            sheaf.message_passing_phase()
        moved = moved_rows(baseline.broadcast, perturbed.broadcast)
        assert moved == owned

    def test_moving_that_row_moves_one_endpoints_update_and_no_other(self, running):
        # And the channel is one endpoint wide. Everything a neighbour's map
        # could ever do to this cell arrives as a change to this row, so
        # perturbing the row directly is the strongest form of the permitted
        # exception -- and even then it reaches the endpoint whose own term it
        # is, never the cell's other edges and never the partner's update.
        for endpoint in NAMED_ENDPOINTS:
            baseline, perturbed = copy.deepcopy(running), copy.deepcopy(running)
            with torch.no_grad():
                perturbed.incoming[endpoint] += NUDGE
            before, after = map_update(baseline), map_update(perturbed)
            assert moved_rows(before, after) == {endpoint}

    def test_the_bias_rule_has_no_such_channel(self, running):
        # The asymmetry between the two rules, stated rather than left implied:
        # prediction error is cell-owned and temporal, so nothing a neighbour
        # holds is an argument of the bias objective at all. Moving every map
        # in the graph leaves every cell's bias update bit-identical.
        before = bias_update(running)
        with torch.no_grad():
            running.maps.maps.mul_(0.5)
        after = bias_update(running)
        for name in before:
            assert torch.equal(before[name], after[name]), name

    def test_and_neither_rule_reaches_the_other_half_of_the_surface(self, running):
        # The mirror of the test above, and it is not decoration. The adapting
        # surface has two halves and the rules split by parameter group
        # (ADR-0008), so there are two directions a rule could reach across and
        # this file would be half a guard if it watched only one. A transport
        # objective that recomputed a neighbour's belief through that
        # neighbour's forward path -- reaching its biases rather than its map
        # -- is a leak every other test here would pass.
        before = map_update(running)
        with torch.no_grad():
            for parameter in running.biases.parameters():
                parameter.mul_(0.5)
        assert torch.equal(before, map_update(running))


class TestTheSharedStorageCase:
    """The leak the tape assertion cannot see, built deliberately and caught here.

    `Tensor.detach` returns a tensor sharing storage with the original, so an
    in-place write through a detached view couples two cells while leaving a
    perfectly clean tape. The assertion cannot see that class by construction;
    only observing the update catches it — which is the whole reason this file
    is the load-bearing half of the guard rather than scaffolding around the
    cheap check.
    """

    def test_a_detached_view_shares_storage_with_the_adapting_surface(self, running):
        # The premise, stated as the fact about torch that it is.
        for parameter in (running.maps.maps, running.biases.decode_output_bias):
            view = parameter.detach()
            assert view.data_ptr() == parameter.data_ptr()
            assert view.grad_fn is None and not view.requires_grad
            # And the clone the shipped paths take is what breaks the sharing.
            assert view.clone().data_ptr() != parameter.data_ptr()

    def test_the_leak_reaches_the_running_surface_without_touching_the_tape(
        self, running
    ):
        sheaf = copy.deepcopy(running)
        surface = sheaf.biases.decode_output_bias.clone()
        LeakingBiasRule(sheaf).gradient()
        assert not torch.equal(sheaf.biases.decode_output_bias, surface)
        sheaf.assert_no_tape()

    def test_the_tape_assertion_stays_silent_on_both_leaks(self, running):
        # Not "does not catch it" as an aside -- run the always-on guard over a
        # whole step of each leaking rule and watch it pass. `step` calls
        # `assert_no_tape` on the way in and on the way out, so this is the
        # cheap check getting every chance it is ever given.
        for leaking in (LeakingBiasRule, LeakingTransportRule):
            sheaf = copy.deepcopy(running)
            leaking(sheaf).step()
            sheaf.assert_no_tape()

    def test_the_updates_the_leaks_produce_carry_no_grad_fn_either(self, running):
        # The other half of "a perfectly clean tape": the coupling happens on
        # plain detached tensors before the transform is called, so the batched
        # graph it then builds has no cross-cell edge in it and the gradient
        # that comes back is as clean as the shipped one.
        sheaf = copy.deepcopy(running)
        update = map_update(sheaf, rule=LeakingTransportRule)
        assert update.grad_fn is None and not update.requires_grad

    def test_the_perturbation_test_catches_the_bias_leak(self, running):
        # The same comparison the standing tests make, on the same helper, so
        # "this test catches it" is demonstrated rather than asserted.
        cell = A_BIAS_ROW
        before, after = readings(
            running, cell, bias_update, perturb_the_biases_of, LeakingBiasRule
        )
        moved = cells_whose_bias_update_moved(before, after)
        assert moved != {cell}
        assert moved == frozenset(range(running.biases.cells))

    def test_the_perturbation_test_catches_the_transport_leak(self, running):
        cell = WIDEST
        owned = owned_endpoints(running, cell)
        before, after = readings(
            running,
            cell,
            map_update,
            perturb_the_restriction_maps_of,
            LeakingTransportRule,
        )
        moved = moved_rows(before, after)
        assert not moved <= owned
        # And it is the partners across the shared edges that moved -- the
        # cells whose adapting surface the write reached.
        assert moved - owned <= frozenset(endpoint ^ 1 for endpoint in owned)


class _WorkflowLoader(yaml.SafeLoader):
    """`yaml.SafeLoader`, reading a workflow the way GitHub Actions reads one.

    Three departures from `yaml.safe_load`, all made because the default is
    wrong *here* rather than as a preference, and all checked against
    GitHub's own workflow parser — `@actions/workflow-parser`, the package
    `actions/languageservices` publishes — rather than assumed.

    **A bare `on` is the string, not the boolean.** PyYAML implements YAML
    1.1, whose implicit resolver reads `on`, `off`, `yes` and `no` as
    booleans. A workflow's `on:` would therefore parse to the key `True`, and
    :meth:`TestBothChecksRunInCI.test_the_workflow_runs_on_every_push` would
    look for a key that is not in the mapping. GitHub reads it as `on`, which
    is what YAML 1.2's core schema says, so the bool resolver here is narrowed
    to `true` and `false` — that schema, and nothing more clever.

    **A duplicate key is refused rather than resolved.** PyYAML lets the last
    of two same-named keys win, and that is a hole the old text match did not
    have: a narrowing `env:` could hide behind a benign `env:` written after
    it, and the checks would read the benign one. Refusing it is GitHub's own
    answer rather than an extra rule of this file's — its parser reports
    `'env' is already defined` and refuses the file.

    **A merge key is refused rather than flattened**, and this third one is a
    consequence rather than a choice: replacing the `map` constructor drops
    `SafeConstructor`'s `flatten_mapping` with it, so `<<: *anchor` raises
    instead of merging, and an unhashable key raises `TypeError` where PyYAML
    raised `ConstructorError`. It is recorded because it is a departure
    whoever edits this needs to know about, and kept because it agrees with
    GitHub, which refuses a merge key too — `A mapping was not expected`
    (#117). Both spellings fail closed: the file is refused rather than read
    as something GitHub would not run. An anchor and a plain alias are
    untouched by this and are read as GitHub reads them.
    """


_WorkflowLoader.yaml_implicit_resolvers = {
    first: [
        (tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"
    ]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_WorkflowLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def _mapping_without_duplicates(loader, node):
    """Construct a mapping, failing on a key written twice. See `_WorkflowLoader`."""
    mapping = {}
    yield mapping
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        assert key not in mapping, f"{key!r} is already defined"
        mapping[key] = loader.construct_object(value_node, deep=True)


_WorkflowLoader.add_constructor("tag:yaml.org,2002:map", _mapping_without_duplicates)


class TestBothChecksRunInCI:
    """A guard CI quietly stops running is this file's own failure mode, one
    level up: silent, and flattering.

    Both halves ride on one `pytest` invocation — the tape assertion because it
    is always on inside the tick, so every test that ticks exercises it, and
    the perturbation test because it is this file. What is dangerous is not
    someone deleting either, which is a visible edit to a file called
    `test_perturbation.py`, but the invocation or the collection quietly
    narrowing under them while everything stays green.

    **Written as a whitelist, and that is the whole design.** A blacklist of
    ways to switch the suite off cannot be finished — a selection flag, a
    positional path, `|| true`, `PYTEST_ADDOPTS` in the step's environment, a
    `testpaths` narrowed in `pyproject.toml`, a root `pytest.ini` that makes
    that table inert — and an unfinishable blacklist that reads as complete is
    worse than none. So each assertion below pins the small number of shapes
    *known* to run the whole suite on every push: the trigger, the invocation,
    every environment block that reaches it, the workflow's top-level keys, and
    every configuration file pytest would read. Anything else fails here and
    has to be re-argued rather than slipping past.

    **The workflow is parsed rather than read line by line** (#117). It used
    to be read as text, and every review round across #90 and #109 turned up
    another way to spell a key that no text match recognised. Two were closed
    by a clause each — the flow mapping and the quoted key — and five were
    still open when #117 ruled: `env:  # a comment`, `env :`, `!!str env:`,
    `? env` / `:`, and a sequence item's `-  if:` with more than one space
    after the dash. Each of those arrived in a round that followed one which
    had closed the spelling before it, which is the answer to whether a clause
    per spelling finishes the job: it does not, because the ways to spell a
    YAML key are not enumerable in advance. A parser resolves all of them to
    the same key before anything here looks at it, so the checks below read
    keys of a mapping rather than text of a line. `block`,
    `nested_under` and `top_level_keys` are gone with the text, and the
    sequence-item normalisation that lived in `block` — the `- ` strip, and
    the depth it left two columns out — is gone with them rather than fixed.

    **That buys immunity to *spelling*, and to nothing else.** The list of
    shapes pinned is the same list it was; what changed is that a shape now
    has one spelling here however many it has in the file. A route that
    reaches the run without writing a key these checks read is untouched by
    parsing, and the two named next are exactly that.

    **What is outside their reach**, stated because a whitelist that reads as
    complete and is not would be this file's own failure mode: the run never
    starting at all — a runner outage, or the workflow file deleted — neither
    of which is silent; and two shapes that are, both found while closing the
    routes below and both left open by decision rather than oversight (#109,
    #117). A step earlier in the job can write `PYTEST_ADDOPTS` into
    `$GITHUB_ENV`, which reaches the pytest step with no `env:` key anywhere
    for the check below to see. And a `defaults: run: shell:` under `jobs:`
    replaces the shell `run: pytest` is handed to, so that line can be pinned
    exactly here and still never run pytest. Neither of those is a spelling,
    so parsing does not reach either: refusing them means pinning every `run:`
    and every key under `jobs:`, a wider whitelist than #90 argued for and a
    decision about this class rather than a gap in it. The third shape that
    used to stand beside them — the `pyproject.toml` scan's — was a gap rather
    than a decision, and #118 closed it. A different third shape took its
    place on the same ticket, and it is the next paragraph.

    **An installed pytest plugin reaches the run through the dependency set**,
    found by `/code-review` on #118 and run rather than reasoned about. A
    plugin whose `pytest_collection_modifyitems` deselects can be named in
    `[project.optional-dependencies].dev`, which the pinned
    `pip install -e ".[dev]"` step installs; appended to that step's own line;
    or installed by a step of its own. All three were written into a scratch
    copy and all three left every assertion here green — no key any of them
    reads is written by any of them, and
    `test_the_suite_runs_whole_and_unfiltered` asks only that *some* `run:`
    in the job holds the string `pytest`, so a second `run:` is unpinned. That
    pytest honours such a hook was checked the same way: a plugin of this
    shape, discovered by entry point, emptied the item list of a two-test
    suite. **The selective form is the dangerous one.** Deselecting everything
    exits 5 — "no tests ran" — and reddens CI, which is loud; deselecting only
    this file exits 0, which is the silent, flattering failure this guard is
    for.

    **Whether that shape is a decision or a gap is not settled here.** Closing
    it means pinning the dependency set and every `run:` in the job, which is
    the same widening #90 declined for `defaults:` — a ruling this record does
    not carry, so it is escalated in #118 rather than filed beside the two
    above. It is counted below rather than set apart from them, because until
    it is ruled, counting it open is the direction that does not overstate.

    **And `pyproject.toml` is parsed too, since #118.** It was read as text
    until then, because `tomllib` arrives in 3.11 and this project's floor was
    3.10. Four spellings of the *table* were tried against that scan and each
    failed *closed*, which was the weaker claim it supported — but four
    spellings were never a proof about TOML, and #117 found a fifth that did
    **not** fail closed. The scan ended at the first line beginning `[`,
    reading it as the next table header; a value written as a multi-line TOML
    string can put such a line inside itself, and every key after it then went
    unread. A `pythonpath` opened that way over an `addopts = "-k nothing"` is
    valid TOML that pytest honours — checked against pytest itself, which
    collected the suite and reported every test deselected — and it passed all
    five assertions here. That was a live narrowing route, and closing it was
    the same argument #117 made for `ci.yml`: the ways to write a TOML key are
    no more enumerable by looking at a line than the ways to write a YAML one.
    The floor moved to 3.11 for it, ruled on in #118 and cheaper than a second
    test-only dependency, since CI already pins 3.12 and nothing in `src/` is
    version-conditional. `configuration` reads the table now: the four
    spellings of the table resolve to the one path it looks up, and a value
    written across lines is a value rather than a place the reader stops.

    **Which is how the list below should be read.** These five assertions hold
    down every route that writes a key they read, in any spelling of it. That
    is the forty-four named here, each one kill-tested on its own — written
    into a scratch copy of the files these assertions read, as an edit where
    the file is already there and as a new file where it is not, since a rival
    configuration and a `conftest.py` are routes precisely by appearing; the
    five assertions run against that copy, unmutated first and then once per
    route; and the route caught. It is not every route there is: the two
    shapes above are outside it by decision and the plugin shape is outside it
    unruled, so the tally is forty-four caught of forty-five known.

    The forty-four, by where they reach. **The invocation**, five: `-k`, a
    positional path, `--collect-only`, `|| true`, `python -m pytest`. **A gate
    on the step**, four: `continue-on-error` and a step-level `if:`, each as
    an ordinary key *and as a sequence item's first key*, where the leading
    `- ` used to hide it from a naive match.
    **The environment**, four: `PYTEST_ADDOPTS` in the step's `env:`, in a
    job-level `env:` written *after* `steps:`, in a workflow-level `env:`
    above `jobs:`, and in a job-level flow mapping, `env: {PYTEST_ADDOPTS: …}`
    — those last three reach the step as surely as the step's own block, and
    none of them is the first `env:` in the file, the last not even spelled
    `env:`. **The trigger**, two: a `branches:` filter under `push:`, and
    `push:` removed. **A quoted key**, four — the same key written so that no
    text match saw it: the job's `env:` in double quotes and in single, the
    step's `continue-on-error:`, and its `if:`. **A spelling of the key that
    was open until #117**, five: `env:  # a comment`, `env :`, `!!str env:`,
    `? env` / `:`, and `-  if:` with two spaces after the dash, or three, or
    any number. #109 could say only that these parse to the key `env` under
    YAML's rules, and flagged that GitHub might reject two of them — the
    explicit tag and the explicit key — in which case they were never routes.
    #117 ran all five through `@actions/workflow-parser`, the workflow parser
    GitHub publishes from `actions/languageservices`, and it **accepted all
    five**, each one reaching the step as the key it spells. That is GitHub's
    own parser rather than a general YAML one, which is what #109 was missing;
    it is not the runner service, which cannot be exercised without pushing a
    branch, so what is claimed is that no spelling was ruled out rather than
    that each is proven to run. A **tab** after the dash is still not in this
    class: YAML refuses it outright, "found character that cannot start any
    token", so it fails at the parser and loudly rather than here and quietly.
    **The rootdir configuration**, four: an `addopts`, a narrowed `testpaths`,
    and a `norecursedirs` in `pyproject.toml`, and an `addopts` written below
    a multi-line string holding a bracketed line, which the scan read as the
    next table header and stopped at (#118). **A file that outranks or
    rivals that table**, six — `pytest.toml`, `.pytest.toml`, `pytest.ini`,
    `.pytest.ini`, `tox.ini`, `setup.cfg` — each carrying a narrowing
    configuration. And **a `conftest.py` narrowing collection**, ten: in
    `tests/conftest.py`, a `collect_ignore`, a `collect_ignore_glob`, a
    `pytest_ignore_collect` and a `pytest_collection_modifyitems`; that last
    hook again in a sub-directory of `tests`, where it is handed the whole item
    list just the same; either hook reached by `import` rather than by `def`;
    the star import that binds one while naming nothing; a `match` statement's
    capture pattern, which binds the hook name without an assignment anywhere;
    and a root `conftest.py`.

    All forty-four fail here. Four spellings of the `pyproject.toml` *table*
    were run too and are counted apart from them, because they are four ways
    of writing one table rather than four ways of reaching the run:
    `[tool.pytest]` with a dotted `ini_options.addopts`, the same with an
    inline `ini_options = {…}`, a header written `[ tool.pytest.ini_options ]`,
    and an `addopts` opened as a multi-line string. What #118 changed is how
    they are met. Against the scan, three of them left the split with nothing
    to split on and raised, and the fourth failed the whitelist — failing
    closed rather than catching. Parsed, each is read as the keys it writes
    and its `addopts` fails the whitelist like any other, which is why the
    same run that catches the forty-four reports these four as read.

    A fixtures-only `conftest.py` — the one shape of that file which narrows
    nothing — still passes, `tests/conftest.py` as #110 wrote it included. So
    does a harmless `-q` *not*: the cost of the design is that a benign edit to
    the invocation has to come with an edit to this class, which is the
    whitelist working rather than a false positive.
    """

    ROOT = Path(__file__).resolve().parents[1]
    WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

    #: The keys `[tool.pytest.ini_options]` is allowed to carry. Whitelisted
    #: rather than blacklisted for the reason the rest of this class is: it is
    #: not only `addopts` that narrows a run — `testpaths`, `norecursedirs` and
    #: `python_files` all do, and the next release could add another.
    PERMITTED_CONFIGURATION = {"testpaths", "pythonpath"}

    #: What the run is allowed in its environment, wherever the block carrying
    #: it is written. `PYTEST_ADDOPTS` would narrow the run from outside the
    #: invocation entirely; the one variable the suite needs is the software
    #: GL context.
    PERMITTED_ENVIRONMENT = {"MUJOCO_GL": "osmesa"}

    #: The workflow's top-level keys, pinned. An `env:` here is inherited by
    #: every step of every job, so it reaches the run as surely as the step's
    #: own. Whitelisted rather than named, because `defaults:` reaches the run
    #: too, through the shell it wraps `run:` in.
    #:
    #: Compared **sorted** rather than in this order. YAML puts no order on a
    #: mapping's keys, so moving `concurrency:` above `on:` changes nothing
    #: about what runs -- and an ordered comparison would fail that edit while
    #: reporting it as an environment finding. Sorting both sides still catches
    #: a key added and a key removed, which is what this pin is for; a key
    #: written twice never reaches the comparison, because `_WorkflowLoader`
    #: refuses the file.
    PERMITTED_WORKFLOW_KEYS = ("name", "on", "concurrency", "jobs")

    #: The other files pytest reads a rootdir configuration from, in the order
    #: it prefers them -- `pyproject.toml` sits between the fourth and the
    #: fifth. Any one of these outranks or displaces `pyproject.toml`'s table
    #: and would make the whitelist above inert. The dotted spellings are read
    #: exactly as the undotted ones are, and the `.toml` pair outranks
    #: everything here.
    RIVAL_CONFIGURATIONS = (
        "pytest.toml",
        ".pytest.toml",
        "pytest.ini",
        ".pytest.ini",
        "tox.ini",
        "setup.cfg",
    )

    @property
    def workflow(self):
        """`ci.yml`, parsed the way GitHub reads it. See `_WorkflowLoader`."""
        return yaml.load(self.WORKFLOW.read_text(), _WorkflowLoader)

    @property
    def configuration(self):
        """The rootdir table pytest reads out of `pyproject.toml`, parsed.

        `tomllib` rather than a scan over the lines, for the reason
        `ci.yml` is parsed rather than matched: the ways to write a TOML key
        are not enumerable by looking at a line, and the scan this replaced
        proved it by missing one (#118). Every spelling of the table header
        resolves to the same three-key path here, and a value written across
        lines is that value rather than a boundary the reader stops at.

        **What it does instead of answering is raise**, in more ways than one
        and all of them this check going red: `KeyError` where nothing is at
        that path, `TypeError` where something that is not a table is — a
        `pytest = "x"` under `[tool]` is subscripted with a string, and an
        `[[tool.pytest.ini_options]]` array of tables arrives as a list of
        dicts and reaches `set(…)` unhashable — and `TOMLDecodeError` where
        the file is not TOML at all. That is the direction the split this
        replaced failed in too, and the property worth keeping from it. What
        cannot happen any more is the other direction: a table pytest reads
        and this misses, whatever it is spelt like.

        Opened in binary, which is `tomllib`'s own idiom rather than a
        preference: TOML is UTF-8 by specification, and `read_text` would
        decode it in whatever the process locale is instead —
        `test_the_table_is_read_as_utf_8_whatever_the_locale_is` forces that
        locale away from UTF-8 and is what fails if this goes back to text.
        """
        with (self.ROOT / "pyproject.toml").open("rb") as source:
            parsed = tomllib.load(source)
        return parsed["tool"]["pytest"]["ini_options"]

    def mappings(self, document):
        """Every mapping in *document*, itself included, outermost first.

        This is what replaced reading lines and matching them by indent, and
        the difference is that a key is found where it is *written* rather
        than at a depth this class predicted. A job-level `env:` put after
        `steps:` is read exactly as one put before it; a step that writes
        `env:` as its first key is read exactly as one that writes it last,
        which the old reader could not do — it recorded a sequence item's
        depth from the raw indent and then stripped the `- `, leaving the
        item's own siblings looking nested under whatever key came first.
        """
        found = []
        if isinstance(document, dict):
            found.append(document)
            for value in document.values():
                found.extend(self.mappings(value))
        elif isinstance(document, list):
            for item in document:
                found.extend(self.mappings(item))
        return found

    def values_of(self, key, document):
        """What each *key* in *document* is set to, one entry per place it is
        written.

        Every, not the first. `env:` is written in more than one place in an
        ordinary workflow — once on the step, once on the job, once above
        `jobs:` — at different depths and reaching the run by different paths.
        A first-match walk returns whichever comes first in the file and
        stops, which is how a job-level `env:` put *after* `steps:` slips
        through: the step's own block satisfies the check, and the one
        carrying `PYTEST_ADDOPTS` is never looked at. (Two of the same key in
        one mapping is a different thing and not the case this is for — a
        duplicate key is refused outright by `_WorkflowLoader`, as GitHub's
        parser refuses it.)
        """
        return [mapping[key] for mapping in self.mappings(document) if key in mapping]

    def names_bound_by(self, source):
        """Every name a module binds: `def`, `class`, assignment, **import**.

        Import included because pytest reads hooks as attributes of the
        imported module and does not care how they got there: `from _hooks
        import pytest_ignore_collect` binds the attribute as surely as a `def`
        does. At any depth, too — a hook defined inside an `if` is a hook.

        A star import binds whatever the imported module holds, which cannot be
        read off this file, and is recorded as `*` for the caller to refuse.

        **This list is finishable, and that is why it is a list.** The
        spellings of a YAML key are not enumerable in advance, which is why the
        workflow above is parsed rather than matched; the binding forms here
        are Python grammar, which `ast` enumerates for us. So they are all
        named rather than sampled: the three statements above, a `Name` in
        `Store` context — which covers assignment, `for`, `with … as`, and the
        walrus — an import alias, the three capture patterns of a `match`
        statement, and an `except … as`. A capture pattern is the one that
        matters: `case pytest_ignore_collect:` binds that name at module scope
        and pytest reads the hook, which a walk over `Name` nodes never sees,
        because `MatchAs.name` is a bare string rather than a node. The
        `except … as` name is unbound again at the end of its handler and so
        cannot carry a hook; it is refused anyway, because the caller's rule is
        about the prefix rather than about reachability.

        **One form is left out on purpose**: a parameter, `ast.arg`. Not on the
        principle that it is local — local bindings *are* read, and the caller
        refuses a `pytest_tmp` inside a helper body, which the comment there
        owns as a cost. It is the trade that differs. A local assignment could
        in principle be the line that binds a hook; a parameter never reaches
        the module's attributes at all, so reading it would refuse a helper
        that happened to take a `pytest_`-prefixed argument and catch nothing
        in exchange. That is a cost with no matching benefit, which is the one
        kind this whitelist does decline.
        """
        names = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, ast.alias):
                names.add(node.asname or node.name.split(".")[0])
            elif isinstance(node, (ast.MatchAs, ast.MatchStar, ast.ExceptHandler)):
                if node.name:
                    names.add(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                names.add(node.rest)
        return names

    def conftests_under(self, root):
        """Every `conftest.py` pytest would import, given the pinned paths.

        The rootdir's own, and the whole `tests` tree: a
        `pytest_collection_modifyitems` is a session hook wherever it sits, and
        one in a sub-directory is handed the whole item list just the same. So
        the walk is recursive rather than the two obvious paths.

        It stops at that tree deliberately rather than sweeping the repository.
        A checkout holding worktrees would otherwise read other checkouts --
        their conftests are not this run's, and refusing them would make this
        assertion fail for a reason that has nothing to do with CI. The
        `testpaths = ["tests"]` pinned just above is what makes that safe:
        collection never leaves the rootdir and that tree.

        Only files that exist are returned, so the caller does not re-check.
        """
        found = [root / "conftest.py", *sorted((root / "tests").rglob("conftest.py"))]
        return [conftest for conftest in found if conftest.exists()]

    def refuse_narrowing(self, source, where="conftest.py"):
        """Fail unless *source* is a `conftest.py` that narrows nothing.

        A fixtures-only `conftest.py` is fine and this should not be the thing
        standing in its way; what narrows collection is the `collect_ignore`
        pair and pytest's hooks. So the file may bind what it likes except a
        name pytest itself reads: every `pytest_` hook rather than the two that
        narrow collection today, because `pytest_ignore_collect` and
        `pytest_collection_modifyitems` are a list that the next release can
        add to, and `pytest_plugins` loads a file that can carry any of them.

        Fixtures are decorated rather than named, so the shape this permits is
        untouched -- but the prefix is refused at every depth and in every
        binding form, which refuses benign names too: `import pytest_asyncio`,
        or a local `pytest_tmp` inside a helper. That is the whitelist's usual
        cost rather than an oversight. What it cannot reach is a hook bound
        past the syntax -- `globals()[...] = hook` -- which is evasion, and
        this class is built against the quiet edit.
        """
        assert "collect_ignore" not in source, f"collect_ignore in {where}"
        names = self.names_bound_by(source)
        # `from _hooks import *` binds the hook while naming nothing, and a
        # conftest has no benign use for the spelling.
        assert "*" not in names, f"{where} imports *"
        for name in names:
            assert not name.startswith("pytest_"), f"{name} in {where}"

    def test_the_workflow_runs_on_every_push(self):
        # `push` a direct child of `on:`, and nothing nested under it: a
        # `branches:` filter would leave this file running on some pushes and
        # not others, which is not what "on every push" says.
        #
        # `self.workflow["on"]` is the string key rather than YAML 1.1's
        # boolean -- see `_WorkflowLoader`, which is what makes that line read
        # as it looks.
        triggers = self.workflow["on"]
        assert "push" in triggers
        assert triggers["push"] is None

    def test_the_suite_runs_whole_and_unfiltered(self):
        # The invocation is pinned exactly. Every way of narrowing it -- a
        # selection flag, a positional path, a `|| true` that swallows the exit
        # code -- changes this string, and changing it fails here.
        assert "pytest" in self.values_of("run", self.workflow["jobs"])

    def test_nothing_reaches_the_run_through_its_environment(self):
        # pytest reads `PYTEST_ADDOPTS` from the environment, so an entry here
        # narrows the run without touching the invocation at all.
        #
        # Every `env:` in the file, at every depth and in every spelling:
        # GitHub hands a job-level block to every step in the job and a
        # workflow-level one to every job, and YAML is happy to have either
        # written after the block it applies to.
        assert self.values_of("env", self.workflow) == [self.PERMITTED_ENVIRONMENT]
        # And the top-level keys, which guards something the line above cannot:
        # `defaults:` reaches the run through the shell it wraps `run:` in and
        # carries no `env:` at all. Sorted on both sides -- see
        # `PERMITTED_WORKFLOW_KEYS`, which says why the order is not part of
        # the pin.
        assert sorted(self.workflow) == sorted(self.PERMITTED_WORKFLOW_KEYS)

    def test_nothing_lets_a_failing_step_pass(self):
        # `continue-on-error` at either level, and an `if:` that would skip the
        # step or the job without anything going red. A sequence item's first
        # key is that key like any other now the file is parsed, whatever the
        # dash is followed by.
        for mapping in self.mappings(self.workflow["jobs"]):
            assert "continue-on-error" not in mapping
            assert "if" not in mapping

    def test_no_configuration_narrows_what_is_collected(self):
        # The other route, and the quieter one: pytest applies its rootdir
        # configuration on its own, so a table here could deselect this file
        # while `run: pytest` still reads as the whole suite.
        #
        # Parsed rather than scanned (#118), the same way and for the same
        # reason `ci.yml` is: the scan ended the table at the first line
        # beginning `[`, and a value written as a multi-line string can hold
        # such a line and hide every key after it. See `configuration`.
        configuration = self.configuration
        assert set(configuration) == self.PERMITTED_CONFIGURATION
        assert configuration["testpaths"] == ["tests"]
        # And nothing that outranks that table, or reaches collection from a
        # `conftest.py` pytest imports on its own.
        for name in self.RIVAL_CONFIGURATIONS:
            assert not (self.ROOT / name).exists()
        # And every `conftest.py` pytest would import.
        for conftest in self.conftests_under(self.ROOT):
            self.refuse_narrowing(conftest.read_text(), conftest)


class TestTheWorkflowReaderReadsWhatGitHubReads:
    """The reader's own behaviour, against workflows written for the occasion.

    Split out from :class:`TestBothChecksRunInCI` rather than added to it: the
    five assertions there are the whitelist itself, read off this repository's
    own files, and these read a workflow written to exercise one thing. What
    earns a test here is anything that could be changed with the rest of the
    suite still green — which is not hypothetical, and is why this class
    exists: before #117 the braces clause in the old text reader had no such
    cover and could be deleted silently.

    Two of these are `_WorkflowLoader`'s departures from `yaml.safe_load`, and
    both are holes if they go: the YAML 1.1 boolean that would make `on:`
    parse to `True`, and the duplicate key PyYAML resolves last-wins. The rest
    are the spellings — five of them open until #117 — which the parser
    collapses onto one key. They are tested rather than taken on trust because
    "the parser handles it" is exactly the kind of claim this file exists to
    falsify, and because the count in
    :class:`TestBothChecksRunInCI`'s docstring rests on them.
    """

    #: The `env:` key, spelt the six ways a text match did not recognise: the
    #: four that reached #117 still open, plus the two quoted spellings #109
    #: had closed by hand. Written to sit at one indent, with the mapping under
    #: them on the following lines -- see `step_with`.
    ENV_SPELLINGS = (
        "env:  # a trailing comment",
        "env :",
        "!!str env:",
        "? env\n        :",
        '"env":',
        "'env':",
    )

    def reader(self, tmp_path, workflow):
        """A :class:`TestBothChecksRunInCI` reading *workflow*, not `ci.yml`."""
        path = tmp_path / "ci.yml"
        path.write_text(workflow)
        reader = TestBothChecksRunInCI()
        reader.WORKFLOW = path
        return reader

    def step_with(self, spelling):
        """A one-step workflow whose step carries *spelling* over a narrowing
        `PYTEST_ADDOPTS`."""
        return (
            "name: CI\n"
            "on:\n"
            "  push:\n"
            "jobs:\n"
            "  test:\n"
            "    steps:\n"
            "      - name: Run tests\n"
            f"        {spelling}\n"
            "          PYTEST_ADDOPTS: -k nothing\n"
            "        run: pytest\n"
        )

    def test_every_spelling_of_the_key_is_read_as_that_key(self, tmp_path):
        # The point of parsing, and the whole of what it buys. Each of these
        # reaches the step as `env` under GitHub's own parser -- checked, not
        # assumed (#117) -- and each is now the same key here.
        for spelling in self.ENV_SPELLINGS:
            reader = self.reader(tmp_path, self.step_with(spelling))
            assert reader.values_of("env", reader.workflow) == [
                {"PYTEST_ADDOPTS": "-k nothing"}
            ], spelling

    def test_a_flow_mapping_is_the_same_key_too(self, tmp_path):
        # Another spelling, and the one the old reader answered by refusing
        # braces wholesale -- which took GitHub's `${{ }}` expressions with it.
        # It writes the key with its value inline, so the *text* was never
        # `env:`; parsed, it is the job-level `env:` route like any other.
        reader = self.reader(
            tmp_path,
            "name: CI\n"
            "on:\n"
            "  push:\n"
            "jobs:\n"
            "  test:\n"
            "    env: {PYTEST_ADDOPTS: -k nothing}\n",
        )
        assert reader.values_of("env", reader.workflow) == [
            {"PYTEST_ADDOPTS": "-k nothing"}
        ]

    def test_a_quoted_scalar_is_not_a_key(self, tmp_path):
        # The distinction the old reader had to draw by hand and got a clause
        # for: `"env":` is a key and `- "3.12"` is a string that happens to
        # start with a quote. A parser draws it for free, and this holds that
        # down -- refusing the second would be a false positive dressed as a
        # finding.
        reader = self.reader(
            tmp_path,
            "name: CI\n"
            "on:\n"
            "  push:\n"
            "jobs:\n"
            "  test:\n"
            "    with:\n"
            "      python-version:\n"
            '        - "3.12"\n'
            "        - '3.13'\n",
        )
        assert reader.values_of("python-version", reader.workflow) == [
            ["3.12", "3.13"]
        ]

    def test_a_sequence_items_first_key_is_read_at_any_spacing(self, tmp_path):
        # The fifth spelling. The old reader stripped the exact text `- `, so
        # a gate written with two spaces after the dash was left unnormalised
        # and reached the checks in a spelling they did not match. The dash is
        # the parser's business now, at any spacing.
        for dash in ("- ", "-  ", "-   "):
            pad = " " * (6 + len(dash))
            reader = self.reader(
                tmp_path,
                "name: CI\n"
                "on:\n"
                "  push:\n"
                "jobs:\n"
                "  test:\n"
                "    steps:\n"
                f"      {dash}if: false\n"
                f"{pad}name: Run tests\n"
                f"{pad}run: pytest\n",
            )
            gated = [
                mapping
                for mapping in reader.mappings(reader.workflow["jobs"])
                if "if" in mapping
            ]
            assert gated == [
                {"if": False, "name": "Run tests", "run": "pytest"}
            ], dash

    def test_a_step_that_writes_env_first_keeps_its_siblings(self, tmp_path):
        # The other half of the old reader's sequence-item handling, and the
        # one that failed in the benign direction: it recorded the item's depth
        # from the raw indent and then stripped the `- `, so a step that wrote
        # `env:` first had its own `name:` and `run:` read as nested under that
        # `env:` -- a reflow that narrows nothing failing, and blaming the
        # environment. Both halves were one decision about one helper, and the
        # helper is gone (#117).
        reader = self.reader(
            tmp_path,
            "name: CI\n"
            "on:\n"
            "  push:\n"
            "jobs:\n"
            "  test:\n"
            "    steps:\n"
            "      - env:\n"
            "          MUJOCO_GL: osmesa\n"
            "        name: Run tests\n"
            "        run: pytest\n",
        )
        assert reader.values_of("env", reader.workflow) == [{"MUJOCO_GL": "osmesa"}]
        assert reader.values_of("run", reader.workflow) == ["pytest"]

    def test_the_trigger_key_is_the_string_on_and_not_a_boolean(self, tmp_path):
        # `_WorkflowLoader`'s first departure. PyYAML is a YAML 1.1 parser, in
        # which a bare `on` is true -- so without the narrowed resolver this
        # workflow parses to the key `True`, `workflow["on"]` raises, and the
        # trigger check fails for a reason that has nothing to do with CI.
        reader = self.reader(
            tmp_path,
            "name: CI\non:\n  push:\njobs:\n  test:\n    steps: []\n",
        )
        assert list(reader.workflow) == ["name", "on", "jobs"]
        assert reader.workflow["on"] == {"push": None}

    def test_a_duplicate_key_is_refused(self, tmp_path):
        # `_WorkflowLoader`'s second departure, and a hole parsing would open
        # if it were left alone: PyYAML lets the last of two same-named keys
        # win, so a narrowing `env:` could hide behind a benign one written
        # after it and `values_of` would report the benign one. GitHub refuses
        # the file outright -- `'env' is already defined` -- and so does this.
        reader = self.reader(
            tmp_path,
            "name: CI\n"
            "on:\n"
            "  push:\n"
            "jobs:\n"
            "  test:\n"
            "    steps:\n"
            "      - env:\n"
            "          PYTEST_ADDOPTS: -k nothing\n"
            "        env:\n"
            "          MUJOCO_GL: osmesa\n"
            "        run: pytest\n",
        )
        with pytest.raises(AssertionError, match="already defined"):
            reader.workflow

    def test_every_env_is_read_not_the_first(self, tmp_path):
        # `values_of`'s whole point, and the real workflow cannot exercise it:
        # `ci.yml` writes `env:` once. This is route three of "the
        # environment, four" -- the step's own `env:` first in the file, then a
        # job-level one written *after* `steps:`, which GitHub hands to every
        # step in the job. A first-match walk is satisfied by the step's block
        # and never reaches the one carrying `PYTEST_ADDOPTS`. The walk is
        # outermost first, so the job's block is the one reported first here
        # even though the step's is written above it.
        reader = self.reader(
            tmp_path,
            "name: CI\n"
            "on:\n"
            "  push:\n"
            "jobs:\n"
            "  test:\n"
            "    steps:\n"
            "      - name: Run tests\n"
            "        env:\n"
            "          MUJOCO_GL: osmesa\n"
            "        run: pytest\n"
            "    env:\n"
            "      PYTEST_ADDOPTS: -k nothing\n",
        )
        assert reader.values_of("env", reader.workflow) == [
            {"PYTEST_ADDOPTS": "-k nothing"},
            {"MUJOCO_GL": "osmesa"},
        ]

    def test_the_top_level_keys_are_pinned_without_their_order(self, tmp_path):
        # Reordering a mapping changes nothing about what runs, so it is not a
        # finding -- but a key added or removed still is. A key written twice
        # never reaches this comparison: `_WorkflowLoader` refuses the file,
        # which `test_a_duplicate_key_is_refused` holds down.
        keys = TestBothChecksRunInCI.PERMITTED_WORKFLOW_KEYS
        reordered = "concurrency:\n  a: 1\njobs:\n  b: 2\nname: CI\non:\n  push:\n"
        assert sorted(self.reader(tmp_path, reordered).workflow) == sorted(keys)
        added = reordered + "env:\n  C: d\n"
        removed = reordered.replace("name: CI\n", "")
        for broken in (added, removed):
            assert sorted(self.reader(tmp_path, broken).workflow) != sorted(keys)


class TestTheConfigurationReaderReadsWhatPytestReads:
    """`configuration`'s own behaviour, against tables written for the occasion.

    The counterpart to :class:`TestTheWorkflowReaderReadsWhatGitHubReads`, one
    format over and for the same reason: the five assertions in
    :class:`TestBothChecksRunInCI` read this repository's own `pyproject.toml`,
    which writes its table one way, so nothing else in this file exercises what
    the reader does with any other way of writing it.

    Until #118 that table was scanned line by line, and the scan ended at the
    first line beginning `[` — so a value written as a multi-line string could
    hold such a line and hide every key after it, which is the first test
    below. Then the four spellings of the table that
    :class:`TestBothChecksRunInCI`'s docstring names, all four covered here
    rather than three: a header spelt with spaces, a dotted key and an inline
    table, which the scan met by *raising* and which
    `test_the_table_is_that_table_however_it_is_written` holds down together;
    and an `addopts` opened as a multi-line string, which the scan met by
    *reading* — it is the one of the four that failed the whitelist rather
    than the split — and which is
    `test_a_value_written_across_lines_is_that_value`. Then the two directions
    the reader must not go wrong in on its own: another tool's table is not
    this one, and a table that is not there fails closed rather than reading
    as empty.

    **And the name of this class is a claim, so it is checked.** Its YAML
    counterpart ran its spellings through GitHub's own published parser rather
    than trusting that a parser is a parser, and
    `test_the_reader_agrees_with_the_reader_pytest_uses` does the same one
    format over: `_pytest.config.findpaths.load_config_dict_from_file` is the
    function pytest locates this table with, and every spelling here is put to
    both. That import is private and is meant to be — if pytest moves the
    table or reads it differently, this class going red is the notice, which
    is the whole reason to compare against the thing itself rather than
    against a belief about it.
    """

    def reader(self, tmp_path, configuration):
        """A :class:`TestBothChecksRunInCI` reading *configuration* as the
        rootdir's `pyproject.toml`."""
        (tmp_path / "pyproject.toml").write_text(configuration)
        reader = TestBothChecksRunInCI()
        reader.ROOT = tmp_path
        return reader

    def test_a_key_behind_a_multi_line_string_is_read(self, tmp_path):
        # #118's route, and the whole reason the scan is gone. `[benchmarks]`
        # inside a multi-line string is a line beginning `[`, which the scan
        # took for the next table header and stopped at, leaving the `addopts`
        # below it unread while pytest honoured it -- checked against pytest
        # itself, which loads this table and reports every test deselected.
        reader = self.reader(
            tmp_path,
            "[tool.pytest.ini_options]\n"
            'testpaths = ["tests"]\n'
            'pythonpath = """\n'
            "[benchmarks]\n"
            '"""\n'
            'addopts = "-k nothing"\n',
        )
        assert set(reader.configuration) == {"testpaths", "pythonpath", "addopts"}
        assert reader.configuration["addopts"] == "-k nothing"

    def test_the_table_is_that_table_however_it_is_written(self, tmp_path):
        # Three spellings of one table: a header with spaces in it, a dotted
        # key, and an inline table. TOML gives all three the same path and
        # pytest reads each of them, so a reader that recognised only the
        # fourth would be back to enumerating spellings.
        spellings = (
            '[ tool.pytest.ini_options ]\ntestpaths = ["tests"]\n'
            'addopts = "-k nothing"\n',
            "[tool.pytest]\n"
            'ini_options.testpaths = ["tests"]\n'
            'ini_options.addopts = "-k nothing"\n',
            "[tool.pytest]\n"
            'ini_options = { testpaths = ["tests"], addopts = "-k nothing" }\n',
        )
        for spelling in spellings:
            reader = self.reader(tmp_path, spelling)
            assert reader.configuration == {
                "testpaths": ["tests"],
                "addopts": "-k nothing",
            }, spelling

    def test_a_value_written_across_lines_is_that_value(self, tmp_path):
        # The other half of the multi-line string: not only does it stop
        # hiding what follows it, it is read as the value it is. The scan saw
        # `addopts = """` and recorded the key -- which was the one spelling of
        # the four it happened to meet by reading rather than by raising.
        reader = self.reader(
            tmp_path, '[tool.pytest.ini_options]\naddopts = """\n-k nothing\n"""\n'
        )
        assert reader.configuration == {"addopts": "-k nothing\n"}

    def test_another_tools_table_is_not_this_one(self, tmp_path):
        # The false positive the parse must not introduce: pytest reads
        # `tool.pytest.ini_options` and nothing else, so an `addopts` under
        # some other tool's table is that tool's business. A reader that
        # searched the document for the key would fail this.
        reader = self.reader(
            tmp_path,
            "[tool.pytest.ini_options]\n"
            'testpaths = ["tests"]\n'
            "[tool.other]\n"
            'addopts = "-k nothing"\n',
        )
        assert reader.configuration == {"testpaths": ["tests"]}

    def test_a_table_that_is_not_there_fails_closed(self, tmp_path):
        # Every way the reader can decline to answer, and each of them is this
        # check going red rather than passing. Named exhaustively rather than
        # sampled, because the docstring above claims the set: nothing at that
        # path, something that is not a table at it -- a string, which is
        # subscripted with a string, and an array of tables, which arrives as
        # a list of dicts and is unhashable where the caller takes `set(...)`
        # -- and a file TOML refuses outright. The split this replaced failed
        # in the same direction, which is the one property of it worth
        # keeping.
        with pytest.raises(KeyError):
            self.reader(tmp_path, '[project]\nname = "patchworks"\n').configuration
        with pytest.raises(TypeError):
            self.reader(tmp_path, '[tool]\npytest = "x"\n').configuration
        with pytest.raises(TypeError):
            table = self.reader(
                tmp_path, '[[tool.pytest.ini_options]]\naddopts = "-k nothing"\n'
            ).configuration
            set(table)
        with pytest.raises(tomllib.TOMLDecodeError):
            self.reader(tmp_path, "[tool.pytest.ini_options\n").configuration

    def test_the_reader_agrees_with_the_reader_pytest_uses(self, tmp_path):
        # What makes the name of this class a claim rather than a wish. Every
        # spelling above, put to `configuration` and to the function pytest
        # locates this table with, and the two asked to agree on which keys
        # the rootdir table holds. A private import on purpose: if pytest
        # moves the table or reads it another way, this is what goes red.
        from _pytest.config.findpaths import load_config_dict_from_file

        spellings = (
            '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
            'pythonpath = """\n[benchmarks]\n"""\naddopts = "-k nothing"\n',
            '[ tool.pytest.ini_options ]\naddopts = "-k nothing"\n',
            '[tool.pytest]\nini_options.addopts = "-k nothing"\n',
            '[tool.pytest]\nini_options = { addopts = "-k nothing" }\n',
            '[tool.pytest.ini_options]\naddopts = """\n-k nothing\n"""\n',
            '[tool."pytest".ini_options]\naddopts = "-k nothing"\n',
            '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
            '[tool.other]\naddopts = "-k nothing"\n',
        )
        for spelling in spellings:
            reader = self.reader(tmp_path, spelling)
            theirs = load_config_dict_from_file(reader.ROOT / "pyproject.toml")
            # pytest 9 hands each value back wrapped in a `ConfigValue`
            # carrying where it came from; unwrapped, both sides are the table
            # itself. The keys are what the whitelist reads, so they are
            # compared as a set as well as through the values.
            assert set(reader.configuration) == set(theirs), spelling
            assert reader.configuration == {
                key: getattr(value, "value", value) for key, value in theirs.items()
            }, spelling

    def test_the_table_is_read_as_utf_8_whatever_the_locale_is(self, tmp_path):
        # `configuration` opens the file in binary and lets `tomllib` decode
        # it, which is the whole of what that costs and buys: TOML is UTF-8 by
        # specification, and `read_text` would decode it in whatever the
        # process locale happens to be. Forced away from UTF-8 here, because a
        # difference that only shows on someone else's machine is one this
        # file would otherwise leave uncovered -- and the premise is checked
        # rather than assumed, since an interpreter in UTF-8 mode cannot be
        # forced and there would then be nothing to test.
        reader = self.reader(
            tmp_path,
            '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n# a café comment\n',
        )
        path = reader.ROOT / "pyproject.toml"
        previous = locale.setlocale(locale.LC_CTYPE)
        try:
            for name in ("C", "POSIX"):
                try:
                    locale.setlocale(locale.LC_CTYPE, name)
                except locale.Error:
                    continue
                try:
                    path.read_text()
                except UnicodeDecodeError:
                    break
            else:
                pytest.skip("no locale here decodes this file as anything but UTF-8")
            assert reader.configuration == {"testpaths": ["tests"]}
        finally:
            locale.setlocale(locale.LC_CTYPE, previous)


class TestEveryModuleAttributeBindingIsRead:
    """`names_bound_by` reads every form that binds a **module attribute**.

    A companion to :class:`TestTheWorkflowReaderReadsWhatGitHubReads`, and the
    counterpart to the argument that sent the workflow to a parser. The ways
    to spell a YAML key cannot be enumerated in advance; these forms can,
    because `ast` is a closed grammar — so this is a list that can be
    finished, and a test that can check it is.

    A module attribute rather than every name the grammar binds: a parameter
    is a binding and is deliberately not read. Not because it is local —
    local names are read, and refusing them is a documented cost — but because
    a parameter can never become an attribute of the module, so reading it
    would buy nothing for the benign helpers it would refuse.

    The form that carries a hook is the `match` capture pattern:
    `case pytest_ignore_collect:` binds that name at module scope and pytest
    reads the hook off it, while `ast.MatchAs.name` is a bare string that no
    walk over `Name` nodes ever reaches.
    """

    def bound(self, source):
        return TestBothChecksRunInCI().names_bound_by(source)

    def test_a_match_capture_pattern_binds_the_name_it_writes(self):
        # The live route: pytest honours a hook bound this way -- verified by
        # running a `conftest.py` of exactly this shape against a two-file
        # suite and watching one file go uncollected.
        source = (
            "from _hooks import hook\n"
            "match hook:\n"
            "    case pytest_ignore_collect:\n"
            "        pass\n"
        )
        assert "pytest_ignore_collect" in self.bound(source)

    def test_the_other_capture_patterns_bind_too(self):
        # A sequence pattern's rest and a mapping pattern's rest are the same
        # binding written two other ways.
        assert "pytest_a" in self.bound(
            "match x:\n    case [*pytest_a]:\n        pass\n"
        )
        assert "pytest_b" in self.bound(
            "match x:\n    case {1: v, **pytest_b}:\n        pass\n"
        )

    def test_an_except_handler_binds_its_name(self):
        # Unbound again at the end of the handler, so it cannot carry a hook.
        # Read anyway: the caller's rule is about the prefix, not reachability.
        source = "try:\n    pass\nexcept ValueError as pytest_c:\n    pass\n"
        assert "pytest_c" in self.bound(source)

    def test_the_forms_that_were_already_read_still_are(self):
        source = (
            "import pytest_plugin_one\n"
            "from _hooks import pytest_ignore_collect\n"
            "def pytest_collection_modifyitems(config, items): pass\n"
            "class pytest_klass: pass\n"
            "pytest_assigned = 1\n"
            "for pytest_loop in ():\n"
            "    pass\n"
            "with open('x') as pytest_ctx:\n"
            "    pass\n"
        )
        assert {
            "pytest_plugin_one",
            "pytest_ignore_collect",
            "pytest_collection_modifyitems",
            "pytest_klass",
            "pytest_assigned",
            "pytest_loop",
            "pytest_ctx",
        } <= self.bound(source)

    def test_a_parameter_is_not_a_module_attribute(self):
        # Left out on purpose: a parameter cannot be an attribute of the
        # module, so pytest can never read a hook off one, and reading it
        # would refuse a benign helper for the name of its argument.
        source = "def helper(pytest_arg):\n    return pytest_arg\n"
        assert "pytest_arg" not in self.bound(source)

    def test_a_local_binding_is_read_even_though_it_is_local(self):
        # The other half of that trade, and the reason "local" is not the
        # rule: a name bound inside a body is read and the caller refuses it.
        source = "def helper():\n    pytest_tmp = 1\n    return pytest_tmp\n"
        assert "pytest_tmp" in self.bound(source)

    def test_a_star_import_is_recorded_as_a_star(self):
        # It binds whatever the imported module holds, which cannot be read
        # off the file, so it is handed to the caller to refuse by name.
        assert "*" in self.bound("from _hooks import *\n")


class TestTheConftestReaderRefusesWhatNarrows:
    """`conftests_under` and `refuse_narrowing`, against a synthetic tree.

    Neither runs against anything in this branch — there is no `conftest.py`
    in it — so without these the recursive walk and the star-import refusal
    are code the suite never executes, and reverting either stays green. The
    docstring of :class:`TestBothChecksRunInCI` counts a sub-directory hook
    and a star import among the routes it holds down, and this is where that
    is checked rather than asserted.

    `tests/conftest.py` exists on `action` as of #110, which is what makes the
    permitted case worth pinning. The source below is a superset of it rather
    than a copy — #110's file is shared definitions only, with no `import
    pytest` and no fixture in it — so it covers that file and the fixtures the
    name leads a reader to expect.
    """

    def guard(self):
        return TestBothChecksRunInCI()

    def test_the_walk_reaches_a_sub_directory_of_tests(self, tmp_path):
        # The rootdir's own file, and the whole `tests` tree rather than its
        # top level: a `pytest_collection_modifyitems` in a sub-directory is
        # handed the whole item list just the same.
        (tmp_path / "conftest.py").write_text("")
        (tmp_path / "tests" / "sub").mkdir(parents=True)
        (tmp_path / "tests" / "conftest.py").write_text("")
        (tmp_path / "tests" / "sub" / "conftest.py").write_text("")
        found = self.guard().conftests_under(tmp_path)
        assert found == [
            tmp_path / "conftest.py",
            tmp_path / "tests" / "conftest.py",
            tmp_path / "tests" / "sub" / "conftest.py",
        ]

    def test_the_walk_stops_at_that_tree(self, tmp_path):
        # A checkout holding worktrees would otherwise read other checkouts,
        # whose conftests are not this run's.
        (tmp_path / ".claude" / "worktrees" / "other").mkdir(parents=True)
        (tmp_path / ".claude" / "worktrees" / "other" / "conftest.py").write_text(
            "def pytest_ignore_collect(collection_path, config):\n    return True\n"
        )
        (tmp_path / "tests").mkdir()
        assert self.guard().conftests_under(tmp_path) == []

    def test_a_fixtures_only_conftest_is_permitted(self):
        # The one shape of the file that narrows nothing, and the reason the
        # check is not simply "no conftest.py".
        self.guard().refuse_narrowing(
            "import pytest\n"
            "from patchworks.graph import DomeSpec\n"
            "SMALL = DomeSpec(patch_grid=4)\n"
            "@pytest.fixture\n"
            "def dome():\n"
            "    return SMALL\n"
        )

    def test_every_shape_that_narrows_is_refused(self):
        narrowing = [
            'collect_ignore = ["test_tick.py"]',
            'collect_ignore_glob = ["test_t*.py"]',
            "def pytest_ignore_collect(collection_path, config): return True",
            "def pytest_collection_modifyitems(config, items): items[:] = []",
            "from _hooks import pytest_collection_modifyitems",
            "from _hooks import *",
            "match hook:\n    case pytest_ignore_collect:\n        pass",
            'pytest_plugins = ["_hooks"]',
        ]
        for source in narrowing:
            with pytest.raises(AssertionError):
                self.guard().refuse_narrowing(source)
