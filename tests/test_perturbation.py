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

from patchworks.graph import DomeSpec, build_graph
from patchworks.learning import (
    MAPS_PARAMETER,
    BiasRule,
    ForwardPath,
    SparsityAnneal,
    TransportPath,
    TransportRule,
)
from patchworks.tick import Sheaf

# The same small dome tests/test_learning.py and tests/test_transport_rule.py
# run on: 39 cells, 15 of them predicting, 54 edges. Small enough to sweep
# every cell in the graph twice, built by the same rules as the real one.
SMALL = DomeSpec(
    patch_grid=4,
    vision_sides=(2,),
    somatomotor_sizes=(4,),
    core_sizes=(4, 3),
    core_degree=4,
    apex_degree=3,
)

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

    @pytest.mark.parametrize("cell", [0, 16, 22, 23, 38])
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
        cell = 23  # a predicting cell, and this dome's widest: seven edges
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
        cell = 38  # the drive boundary cell: three edges, each of mask width 1
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
        cell = 23
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
        cell = 23
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
    the step's environment, and every configuration file pytest would read.
    Anything else fails here and has to be re-argued rather than slipping past.
    The one thing outside their reach is the run never starting at all — a
    runner outage, or the workflow file deleted — and neither of those is
    silent.

    Kill-tested against twenty-six ways of narrowing the run: `-k`, a
    positional path, `--collect-only`, `|| true`, `python -m pytest`,
    `continue-on-error` and a step-level `if:` both as an ordinary key **and as
    a sequence item's first key**, where the leading `- ` would hide them from
    a naive match; `PYTEST_ADDOPTS` in the step's `env:`, in a job-level `env:`
    written *after* `steps:`, and in a workflow-level `env:` above `jobs:` —
    the last two reach the step just as well, and neither is the first `env:`
    in the file; a `branches:` filter under `push:`; `push:` removed; an
    `addopts`, a narrowed `testpaths`, and a `norecursedirs` in
    `pyproject.toml`; each of the six files that outrank or rival that table —
    `pytest.toml`, `.pytest.toml`, `pytest.ini`, `.pytest.ini`, `tox.ini`,
    `setup.cfg` — carrying a narrowing configuration; and a `conftest.py`
    narrowing collection through a `collect_ignore`, a `collect_ignore_glob`, a
    `pytest_ignore_collect`, or a `pytest_collection_modifyitems`. All
    twenty-six fail here, and a fixtures-only `conftest.py` — the one shape of
    that file which narrows nothing — still passes. So does a harmless `-q`
    *not*: the cost of the design is that a benign edit to the invocation has
    to come with an edit to this class, which is the whitelist working rather
    than a false positive.
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

        Comments are dropped — they are not configuration, and a comment that
        happened to read `run: pytest` should not satisfy anything here. The
        leading `- ` of a sequence item is stripped too: YAML lets any mapping
        key go first in a sequence item, so `- if:` and `if:` are the same key
        and a check that only knew the second spelling would miss the first.
        """
        lines = self.lines
        block = []
        for line in lines[lines.index(f"{key}:") + 1 :]:
            if line and not line.startswith(" "):
                break
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            block.append((len(line) - len(line.lstrip()), text.removeprefix("- ")))
        return block

    def nested_under(self, block, key):
        """The entries indented under *every* `key` in a block, one list each.

        Every, not the first: YAML puts no order on a mapping's keys, so a
        second block of the same name is ordinary rather than exotic, and the
        one that narrows the run is the one a first-match check stops short of.
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
        """Every name a module binds, at any depth: `def`, `class`, assignment.

        At any depth because a hook defined inside an `if` is a hook: pytest
        finds it by name on the imported module either way.
        """
        names = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
        return names

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
        assert self.nested_under(self.block("jobs"), "env:") == [self.PERMITTED_ENVIRONMENT]
        # And the level above, which `jobs:` does not contain: a workflow-level
        # `env:` reaches every step of every job. The top-level keys are pinned
        # rather than that one name refused, so it has nowhere to land.
        assert self.top_level_keys == self.PERMITTED_WORKFLOW_KEYS

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
        for directory in (self.ROOT, self.ROOT / "tests"):
            conftest = directory / "conftest.py"
            if conftest.exists():
                # A fixtures-only `conftest.py` is fine and this should not be
                # the thing standing in its way; what narrows collection is the
                # `collect_ignore` pair and pytest's hooks. So the file may
                # bind what it likes except a name pytest itself reads: every
                # `pytest_` hook rather than the two that narrow collection
                # today, because `pytest_ignore_collect` and
                # `pytest_collection_modifyitems` are a list that the next
                # release can add to, and `pytest_plugins` loads a file that
                # can carry any of them. Fixtures are decorated, not named, so
                # nothing about this stands in a benign conftest's way.
                text = conftest.read_text()
                assert "collect_ignore" not in text
                for name in self.names_bound_by(text):
                    assert not name.startswith("pytest_"), name
