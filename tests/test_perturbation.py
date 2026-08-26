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
from pathlib import Path

import pytest
import torch

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

    def test_the_other_two_numberings_are_the_sizes_this_file_indexes_into(
        self, running
    ):
        # The bias rows and the edge endpoints are numbered separately from
        # dome cell ids (see `perturb_the_biases_of`), and the tests below
        # index into both by hand. Pinned here so a smaller dome fails as a
        # statement about the dome rather than as an `IndexError` somewhere.
        assert running.biases.cells == len(running.dome.predicting)
        assert running.incoming.shape[0] == 2 * len(running.dome.edges)


class TestPerturbingOneCellsBiases:
    """The bias rule's per-cell path. Prediction error is a cell's own quantity
    and crosses no edge, so the claim here admits no exception at all.

    `cell` is a row of the population's biases, not a dome cell id — see
    :func:`perturb_the_biases_of`.
    """

    @pytest.mark.parametrize("cell", [0, 1, 7, 14])
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
        cell = 7
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
        for endpoint in (0, 1, 43, 100):
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
        cell = 7
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

    **What is outside their reach**, stated because a whitelist that reads as
    complete and is not would be this file's own failure mode: the run never
    starting at all — a runner outage, or the workflow file deleted — neither
    of which is silent; and two shapes that are, both found while closing the
    routes below and both left open rather than chosen alone (#109). A step
    earlier in the job can write `PYTEST_ADDOPTS` into `$GITHUB_ENV`, which
    reaches the pytest step with no `env:` block anywhere for the check below
    to see. And a `defaults: run: shell:` under `jobs:` replaces the shell
    `run: pytest` is handed to, so that line can be pinned exactly here and
    still never run pytest. Refusing either means pinning every `run:` line
    and every key under `jobs:` — a wider whitelist than #90 argued for, and a
    decision about this class rather than a gap in it.

    And these checks read lines rather than parsed YAML, so a spelling of a
    key that no text match recognises gets past them until a clause is added
    for it. Two are closed in `block` — the flow mapping and the quoted key —
    and **five are open, every one of them reproduced**. Four are spellings of
    the key itself. Each of these, written under the job and carrying
    `PYTEST_ADDOPTS: -k nothing`, passes all five assertions here while
    reaching the step:

        env:  # any trailing comment at all
        env :
        !!str env:
        ? env
        :

    The fifth is the whitespace after a sequence item's dash. `block` strips
    the exact text `- `, so a gate written with two or more spaces is left
    unnormalised and reaches the checks in a spelling they do not match:

        -  if: false
           name: Run tests

    Two spaces rather than one — or three, or any number. Each parses to the
    key `if` and skips the step with all five assertions green. A **tab**
    after the dash is *not* in this class: YAML refuses it outright, "found
    character that cannot start any token", so it fails at the parser and
    loudly rather than here and quietly.

    All four of the key spellings parse to the key `env` under a real YAML
    parser, which is what "reaching the step" rests on. For the first two that
    settles it. For `!!str env:` and the explicit key it settles the YAML and
    not GitHub's own workflow parser, which cannot be exercised from this
    repository without pushing a branch — so those two carry that caveat, and
    the escalation should not be read as claiming more.

    They are left open on purpose, and #109 escalates them rather than adding
    a clause for each. Three were found in a single review round, and the
    fifth in the round after that, each round following one that had closed
    the spelling before it. That is the answer to whether reading lines can be
    finished by adding clauses. It cannot: these checks match key text, YAML
    has more ways to spell a key than anyone enumerates in advance, and each
    clause added leaves this file reading more finished than it is. The repair
    is to parse the workflow, which is immune to the whole class at once
    rather than to one spelling of it — and that means a YAML parser, which
    the dev extra does not name (`pytest` is all of it) and which nothing
    installs on its own: neither `torch`, `mujoco`, `gymnasium`, `numpy` nor
    `pytest` pulls one in, so `import yaml` fails in this environment today.
    So it is a real dependency decision rather than a free one, and #109
    carries it rather than this class taking it in passing.

    **Which is how the list below should be read.** These five assertions hold
    down every route that does not turn on a novel spelling of a key. That is
    the thirty-eight named here, each one kill-tested on its own — written
    into this repository, the five assertions run against it, and the route
    caught. It is not every route there is, and the five spellings above are
    the ones known to be outside it.

    The thirty-eight, by where they reach. **The invocation**, five: `-k`, a
    positional path, `--collect-only`, `|| true`, `python -m pytest`. **A gate
    on the step**, four: `continue-on-error` and a step-level `if:`, each as
    an ordinary key *and as a sequence item's first key*, where the leading
    `- ` would hide it from a naive match — with the one space this workflow
    and YAML's own style use, which is the spelling that is closed.
    **The environment**, four: `PYTEST_ADDOPTS` in the step's `env:`, in a
    job-level `env:` written *after* `steps:`, in a workflow-level `env:`
    above `jobs:`, and in a job-level flow mapping, `env: {PYTEST_ADDOPTS: …}`
    — those last three reach the step as surely as the step's own block, and
    none of them is the first `env:` in the file, the last not even spelled
    `env:`. **The trigger**, two: a `branches:` filter under `push:`, and
    `push:` removed.
    **A quoted key**, four — the same key written so that no text match sees
    it: the job's `env:` in double quotes and in single, the step's
    `continue-on-error:`, and its `if:`. **The rootdir configuration**, three:
    an `addopts`, a narrowed `testpaths`, and a `norecursedirs` in
    `pyproject.toml`. **A file that outranks or rivals that table**, six —
    `pytest.toml`, `.pytest.toml`, `pytest.ini`, `.pytest.ini`, `tox.ini`,
    `setup.cfg` — each carrying a narrowing configuration. And **a
    `conftest.py` narrowing collection**, ten: in `tests/conftest.py`, a
    `collect_ignore`, a `collect_ignore_glob`, a `pytest_ignore_collect` and a
    `pytest_collection_modifyitems`; that last hook again in a sub-directory
    of `tests`, where it is handed the whole item list just the same; either
    hook reached by `import` rather than by `def`; the star import that binds
    one while naming nothing; a `match` statement's capture pattern, which
    binds the hook name without an assignment anywhere; and a root
    `conftest.py`.

    All thirty-eight fail here. A fixtures-only `conftest.py` — the one shape
    of that file which narrows nothing — still passes, `tests/conftest.py` as
    #110 wrote it included. So does a harmless `-q` *not*: the cost of the
    design is that a benign edit to the invocation has to come with an edit to
    this class, which is the whitelist working rather than a false positive.
    """

    ROOT = Path(__file__).resolve().parents[1]
    WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

    #: The keys `[tool.pytest.ini_options]` is allowed to carry. Whitelisted
    #: rather than blacklisted for the reason the rest of this class is: it is
    #: not only `addopts` that narrows a run — `testpaths`, `norecursedirs` and
    #: `python_files` all do, and the next release could add another.
    PERMITTED_CONFIGURATION = {"testpaths", "pythonpath"}

    #: What the pytest step is allowed in its environment. `PYTEST_ADDOPTS`
    #: would narrow the run from outside the invocation entirely.
    PERMITTED_ENVIRONMENT = ["MUJOCO_GL: osmesa"]

    #: The workflow's top-level keys, pinned. An `env:` here is inherited by
    #: every step of every job, so it reaches the run as surely as the step's
    #: own -- and it sits above `jobs:`, where a check that walked the job
    #: would never see it. Whitelisted rather than named, because `defaults:`
    #: reaches the run too, through the shell it wraps `run:` in.
    #:
    #: Compared as a **multiset** rather than in this order. YAML puts no
    #: order on a mapping's keys, so moving `concurrency:` above `on:` changes
    #: nothing about what runs -- and an ordered comparison would fail that
    #: edit while reporting it as an environment finding. Sorting both sides
    #: still catches a key added, a key removed, and a key written twice,
    #: which is every bit of the guarding this pin is for.
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
    def lines(self):
        return self.WORKFLOW.read_text().splitlines()

    def block(self, key):
        """The YAML under a top-level `key:`, as `(indent, text)` pairs.

        Whole-line comments are dropped — they are not configuration, and a
        comment that happened to read `run: pytest` should not satisfy anything
        here. A **trailing** comment is not dropped, and that is one of the
        open spellings the class docstring lists rather than a detail: it
        leaves `env:  # a note` recorded with the comment attached, which is
        not the text `env:` that the checks match on. The
        leading `- ` of a sequence item is stripped too: YAML lets any mapping
        key go first in a sequence item, so `- if:` and `if:` are the same key
        and a check that only knew the second spelling would miss the first.
        **Exactly that text, one space.** `-  if:` — two spaces, or more — is
        the same key again and is not normalised here. That is the fifth open
        spelling the class docstring records, and it is open for the reason
        the other four are, not because a wider strip would be hard. A tab
        after the dash is not part of it: YAML rejects that outright, so it
        cannot be a quiet route.

        Stripping the dash also leaves the recorded depth two columns left of
        the item's own siblings, since the depth is the line's raw indent. So
        a step that put `env:` first — `- env:` — would have its `name:` and
        `run:` read as nested under that `env:`, and
        `test_nothing_reaches_the_run_through_its_environment` would fail on a
        reflow that narrows nothing, blaming the environment. It fails closed,
        which is the safe direction, and it is listed here rather than fixed
        because the fix and the strip above are one decision about how this
        helper normalises a sequence item — the decision #109 escalates.

        Two spellings are refused outright rather than parsed, because each
        writes a key the checks below would not recognise as that key -- and
        those checks all match on key text. A **flow mapping**, `env: {A: b}`,
        writes the key with its value inline, so the text recorded here is not
        `env:`; braces are refused wholesale, which takes GitHub's `${{ }}`
        expressions with them. A **quoted key**, `"env":` or `'env':`, is the
        same key in a spelling no `== "env:"` matches, and it hides a step's
        `"continue-on-error":` from a `startswith` just as well. This workflow
        needs neither, so both get the whitelist's usual answer: fail, and be
        re-argued.

        A quoted **key** is what is refused, and not every line that opens with
        a quote: what marks the key is the closing quote with a colon after it,
        so `- "3.12"` under a `python-version:` stays permitted. It is an
        ordinary scalar, it hides no key from anything, and refusing it would
        be a false positive dressed as a finding.
        :class:`TestTheWorkflowReaderRefusesWhatItNames` holds both
        halves of that distinction down.

        **These are a class rather than a list, and reading lines cannot
        finish it.** Five more spellings are open: three found in a single
        review round after the two above were closed, and one more in the
        round after that. The class docstring lists them, and makes the
        argument they add up to: parse the file rather than add a clause here
        for each.
        """
        lines = self.lines
        block = []
        for line in lines[lines.index(f"{key}:") + 1 :]:
            if line and not line.startswith(" "):
                break
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            text = text.removeprefix("- ")
            if text[:1] in ('"', "'"):
                # A quoted *key* -- the closing quote followed by its colon.
                # A quoted scalar is not one: `- "3.12"` under a
                # `python-version:` is an ordinary sequence item, and refusing
                # it would be a false positive dressed as a finding.
                closed = text.find(text[0], 1)
                after = text[closed + 1 :].lstrip() if closed != -1 else ""
                assert not after.startswith(":"), f"quoted key under {key}: {text}"
            assert "{" not in text, (
                f"braces under {key}: {text} — a flow mapping hides the key "
                f"from every check here, and an expression goes with it"
            )
            block.append((len(line) - len(line.lstrip()), text))
        return block

    def nested_under(self, block, key):
        """The entries indented under *every* `key` in a block, one list each.

        Every, not the first. `block` flattens the whole tree, so one name
        appears in it once per place it is written — and `env:` is written in
        more than one place in an ordinary workflow: once on the step, once on
        the job, at different depths and reaching the run by different paths.
        A first-match walk returns whichever comes first in the file and stops,
        which is how a job-level `env:` put *after* `steps:` slips through: the
        step's own block satisfies the check, and the one carrying
        `PYTEST_ADDOPTS` is never looked at. (Two of the same key at the same
        depth is a different thing and not the case this is for — a duplicate
        mapping key is a YAML spec violation that GitHub's parser rejects
        outright, so it is not a quiet route into CI.)
        """
        found = []
        for position, (depth, text) in enumerate(block):
            if text != key:
                continue
            nested = []
            for entry_depth, entry_text in block[position + 1 :]:
                if entry_depth <= depth:
                    break
                nested.append(entry_text)
            found.append(nested)
        return found

    @property
    def top_level_keys(self):
        """The workflow's keys at column zero, in order."""
        return tuple(
            line.split(":", 1)[0]
            for line in self.lines
            if line and not line[0].isspace() and not line.startswith("#")
        )

    def names_bound_by(self, source):
        """Every name a module binds: `def`, `class`, assignment, **import**.

        Import included because pytest reads hooks as attributes of the
        imported module and does not care how they got there: `from _hooks
        import pytest_ignore_collect` binds the attribute as surely as a `def`
        does. At any depth, too — a hook defined inside an `if` is a hook.

        A star import binds whatever the imported module holds, which cannot be
        read off this file, and is recorded as `*` for the caller to refuse.

        **This list is finishable, and that is why it is a list.** The spellings
        `block` refuses are YAML key text, which nothing enumerates in advance;
        the binding forms here are Python grammar, which `ast` enumerates for
        us. So they are all named rather than sampled: the three statements
        above, a `Name` in `Store` context — which covers assignment, `for`,
        `with … as`, and the walrus — an import alias, the three capture
        patterns of a `match` statement, and an `except … as`. A capture
        pattern is the one that matters: `case pytest_ignore_collect:` binds
        that name at module scope and pytest reads the hook, which a walk over
        `Name` nodes never sees, because `MatchAs.name` is a bare string rather
        than a node. The `except … as` name is unbound again at the end of its
        handler and so cannot carry a hook; it is refused anyway, because the
        caller's rule is about the prefix rather than about reachability.

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
        entries = self.block("on")
        indent = min(depth for depth, _ in entries)
        assert (indent, "push:") in entries
        assert self.nested_under(entries, "push:") == [[]]

    def test_the_suite_runs_whole_and_unfiltered(self):
        # The invocation is pinned exactly. Every way of narrowing it -- a
        # selection flag, a positional path, a `|| true` that swallows the exit
        # code -- changes this line, and changing this line fails here.
        runs = [text for _, text in self.block("jobs") if text.startswith("run:")]
        assert "run: pytest" in runs

    def test_nothing_reaches_the_run_through_its_environment(self):
        # pytest reads `PYTEST_ADDOPTS` from the environment, so an entry here
        # narrows the run without touching the invocation at all. The one
        # variable the suite needs is the software GL context.
        #
        # Every `env:` under `jobs:`, not the first one: GitHub hands a
        # job-level block to every step in the job, and YAML is happy to have
        # it written after `steps:`, past the point a first-match check stops.
        jobs = self.block("jobs")
        assert self.nested_under(jobs, "env:") == [self.PERMITTED_ENVIRONMENT]
        # And the level above, which `jobs:` does not contain: a workflow-level
        # `env:` reaches every step of every job. The top-level keys are pinned
        # rather than that one name refused, so it has nowhere to land. Sorted
        # on both sides -- see `PERMITTED_WORKFLOW_KEYS`, which says why the
        # order is not part of the pin.
        assert sorted(self.top_level_keys) == sorted(self.PERMITTED_WORKFLOW_KEYS)

    def test_nothing_lets_a_failing_step_pass(self):
        # `continue-on-error` at either level, and an `if:` that would skip the
        # step or the job without anything going red. `block` has already
        # stripped any leading `- `, so a gate written as a sequence item's
        # first key is caught in the same spelling as an ordinary one.
        for _, text in self.block("jobs"):
            assert not text.startswith(("continue-on-error", "if:"))

    def test_no_configuration_narrows_what_is_collected(self):
        # The other route, and the quieter one: pytest applies its rootdir
        # configuration on its own, so a table here could deselect this file
        # while `run: pytest` still reads as the whole suite.
        configuration = (self.ROOT / "pyproject.toml").read_text()
        table = configuration.split("[tool.pytest.ini_options]", 1)[1]
        keys = set()
        for line in table.splitlines():
            text = line.strip()
            if text.startswith("["):
                break
            if text and not text.startswith("#") and "=" in text:
                keys.add(text.split("=", 1)[0].strip())
        assert keys == self.PERMITTED_CONFIGURATION
        assert 'testpaths = ["tests"]' in table
        # And nothing that outranks that table, or reaches collection from a
        # `conftest.py` pytest imports on its own.
        for name in self.RIVAL_CONFIGURATIONS:
            assert not (self.ROOT / name).exists()
        # And every `conftest.py` pytest would import.
        for conftest in self.conftests_under(self.ROOT):
            self.refuse_narrowing(conftest.read_text(), conftest)


class TestTheWorkflowReaderRefusesWhatItNames:
    """The reader's own clauses, against a synthetic workflow.

    Split out from :class:`TestBothChecksRunInCI` rather than added to it: the
    five assertions there are the whitelist itself, read off this repository's
    own files, and these read a workflow written for the occasion. Each clause
    that refuses something gets a test here, so that deleting the clause goes
    red in this suite rather than in a later review round -- which is not
    hypothetical: the quoted-key clause had that cover and the braces clause
    did not, and the braces clause could be deleted with every other test in
    this file still green.

    The quoted key is the one with a distinction to keep. `"env":` is the key
    `env` written so that no `== "env:"` sees it, and is refused. `- "3.12"` is
    an ordinary string that happens to begin with a quote, hides no key from
    anything, and is permitted -- a clause that refused it too would report a
    benign line as a finding, which is a false positive dressed as a guard.

    The last test is the same concern one level up: the workflow's top-level
    keys are pinned as a multiset, so reordering them is not a finding either.
    """

    def reader(self, tmp_path, workflow):
        """A :class:`TestBothChecksRunInCI` reading *workflow*, not `ci.yml`."""
        path = tmp_path / "ci.yml"
        path.write_text(workflow)
        reader = TestBothChecksRunInCI()
        reader.WORKFLOW = path
        return reader

    def test_a_quoted_scalar_is_not_a_quoted_key(self, tmp_path):
        # The shape this repository's own workflow would reach for if the
        # `python-version:` under `setup-python` ever listed more than one:
        # sequence items that are quoted strings, in either quote.
        reader = self.reader(
            tmp_path,
            'jobs:\n'
            '  test:\n'
            '    with:\n'
            '      python-version:\n'
            '        - "3.12"\n'
            "        - '3.13'\n",
        )
        recorded = [text for _, text in reader.block("jobs")]
        assert '"3.12"' in recorded and "'3.13'" in recorded

    def test_a_quoted_key_is_refused_in_either_quote(self, tmp_path):
        # Each of these is a key the checks in `TestBothChecksRunInCI` match on
        # by text, written so that the text is not the one they match.
        spellings = ('"env":', "'env':", '"continue-on-error": true', "'if': false")
        for spelling in spellings:
            reader = self.reader(tmp_path, f"jobs:\n  test:\n    {spelling}\n")
            with pytest.raises(AssertionError, match="quoted key"):
                reader.block("jobs")

    def test_a_flow_mapping_is_refused(self, tmp_path):
        # The quoted key's sibling clause. It writes the key with its value
        # inline, so the text recorded is not `env:` and every check that
        # matches on key text misses it -- which is the job-level
        # `env: {PYTEST_ADDOPTS: ...}` route, one of the thirty-eight.
        reader = self.reader(
            tmp_path, "jobs:\n  test:\n    env: {PYTEST_ADDOPTS: -k nothing}\n"
        )
        with pytest.raises(AssertionError, match="braces under"):
            reader.block("jobs")

    def test_every_block_of_a_name_is_returned_not_the_first(self, tmp_path):
        # `nested_under`'s whole point, and the real workflow cannot exercise
        # it: `ci.yml` writes `env:` once. This is route three of "the
        # environment, four" -- the step's own `env:` first, then a job-level
        # one written *after* `steps:`, which GitHub hands to every step in the
        # job. Two different keys at two different depths that share a name
        # once `block` has flattened them, not a duplicate mapping key. A
        # first-match walk is satisfied by the step's block and never reaches
        # the one carrying `PYTEST_ADDOPTS`.
        reader = self.reader(
            tmp_path,
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
        assert reader.nested_under(reader.block("jobs"), "env:") == [
            ["MUJOCO_GL: osmesa"],
            ["PYTEST_ADDOPTS: -k nothing"],
        ]

    def test_the_top_level_keys_are_pinned_without_their_order(self, tmp_path):
        # Reordering a mapping changes nothing about what runs, so it is not a
        # finding -- but a key added, removed, or repeated still is.
        keys = TestBothChecksRunInCI.PERMITTED_WORKFLOW_KEYS
        reordered = "concurrency:\n  a: 1\njobs:\n  b: 2\nname: CI\non:\n  push:\n"
        assert sorted(self.reader(tmp_path, reordered).top_level_keys) == sorted(keys)
        added = reordered + "env:\n  C: d\n"
        removed = reordered.replace("name: CI\n", "")
        for broken in (added, removed):
            assert sorted(self.reader(tmp_path, broken).top_level_keys) != sorted(keys)


class TestEveryModuleAttributeBindingIsRead:
    """`names_bound_by` reads every form that binds a **module attribute**.

    A companion to :class:`TestTheWorkflowReaderRefusesWhatItNames`, and the
    counterpart to what that class's docstring escalates. The YAML spellings
    `block` matches cannot be enumerated in advance; these forms can, because
    `ast` is a closed grammar — so this is a list that can be finished, and a
    test that can check it is.

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
