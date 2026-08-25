"""The transport rule and the gauge projection (ticket #89).

The other half of :mod:`patchworks.learning` — `tests/test_learning.py` holds
the bias rule's. What these hold down is
`docs/spec/07-local-learning-rule.md`, *The transport rule*, together with
ADR-0007's constraint on the objective, ADR-0010's gauge, and
`docs/spec/09-the-build-stack.md`, *The locality guard*.

Three of them are the ones that catch a leak, and all three are written as real
assertions rather than smoke checks: the objective is invariant under scaling
both of an edge's maps together and worst when one goes to zero, a neighbour's
map receives exactly no gradient, and the batched gradient equals the gradient
taken for one edge endpoint on its own. The first is what excludes the trivial
solution the whole gauge argument rests on; the other two are what make the one
cross-cell parameter in the phase unreachable rather than merely severed.
"""

import contextlib
from unittest import mock

import pytest
import torch

from patchworks.graph import DomeSpec, build_graph
from patchworks.learning import (
    DEFAULT_ANNEAL_HORIZON,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SPARSITY_PRESSURE,
    MAPS_PARAMETER,
    SparsityAnneal,
    TransportPath,
    TransportRule,
    normalised_l1,
    relative_disagreement,
    transport_gradient,
    transport_objective,
)
from patchworks.restriction import GAUGE_RHO, RestrictionMaps
from patchworks.tick import Sheaf

# The same small dome tests/test_tick.py and tests/test_learning.py run on:
# small enough to take a gradient endpoint by endpoint, built by the same rules
# as the real one.
SMALL = DomeSpec(
    patch_grid=4,
    vision_sides=(2,),
    somatomotor_sizes=(4,),
    core_sizes=(4, 3),
    core_degree=4,
    apex_degree=3,
)


@pytest.fixture
def dome():
    return build_graph(SMALL)


@pytest.fixture
def sheaf(dome):
    return Sheaf(dome, generator=torch.Generator().manual_seed(0))


@pytest.fixture
def running(sheaf):
    """A sheaf a few ticks into a run, with something in every buffer.

    A freshly built sheaf is all zeros and a fresh dome is unstimulated, so
    every disagreement below would be trivially zero and half these assertions
    would pass on nothing.
    """
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        sheaf.stalks[: sheaf.layout.total] = torch.randn(
            sheaf.layout.total, generator=generator
        )
        sheaf.charts.normal_(0.0, 1.0, generator=generator)
        for edge in sheaf.dome.edges:
            for side in (0, 1):
                sheaf.broadcast[2 * edge.id + side, : edge.m].normal_(
                    0.0, 1.0, generator=generator
                )
    for _ in range(3):
        sheaf.tick()
    return sheaf


@pytest.fixture
def wide_pairs(running):
    """Every endpoint of an edge whose stalk holds more than one direction.

    On an `m = 1` edge the two ends' beliefs are scalars, so whenever their
    signs differ the relative disagreement is pinned at exactly 1 — its worst
    value — and *flat* there, which is correct and is the subject of its own
    test below. It makes the edge useless for any assertion about a ratio
    moving, or about a map that has something to descend.
    """
    return [
        2 * edge.id + side
        for edge in running.dome.edges
        if edge.m > 1
        for side in (0, 1)
    ]


@pytest.fixture
def narrow_pairs(running):
    """The other kind: endpoints of an edge whose stalk is one-dimensional."""
    return [
        2 * edge.id + side
        for edge in running.dome.edges
        if edge.m == 1
        for side in (0, 1)
    ]


def _permitted(sheaf):
    """`[pairs]`: how many weights each map's structural mask leaves open."""
    return sheaf.maps.support.flatten(1).sum(-1)


@pytest.fixture
def free_pairs(running):
    """Endpoints whose mask leaves more than one weight open."""
    return (_permitted(running) > 1).nonzero().flatten().tolist()


@pytest.fixture
def single_entry_pairs(running):
    """Endpoints whose mask leaves exactly one weight open."""
    return (_permitted(running) == 1).nonzero().flatten().tolist()


def maps_of(sheaf):
    return sheaf.maps.maps.detach().clone()


def outgoing_of(rule):
    """What every endpoint restricts, in the maps as they now stand."""
    gathered, _ = rule.inputs()
    with torch.no_grad():
        return rule.sheaf.maps.restrict(gathered)


def objective_of(rule, *, pressure=None, maps=None, neighbour_beliefs=None):
    """The rule's own objective, with any of its arguments substituted."""
    gathered, incoming = rule.inputs()
    parameters = rule.path.map_parameters()
    if maps is not None:
        parameters = {MAPS_PARAMETER: maps}
    return transport_objective(
        parameters,
        rule.path,
        gathered,
        incoming if neighbour_beliefs is None else neighbour_beliefs,
        rule.permitted,
        rule.pressure if pressure is None else pressure,
    )


def in_precision(rule, dtype):
    """The rule's own arguments, cast to one precision. The maps are cast as a
    parameter dict rather than through `Module.to`, which is in place and would
    convert the sheaf the rest of the test is still running on."""
    gathered, incoming = (tensor.to(dtype) for tensor in rule.inputs())
    parameters = {MAPS_PARAMETER: rule.sheaf.maps.maps.detach().to(dtype)}
    return parameters, (
        rule.path,
        gathered,
        incoming,
        rule.permitted.to(dtype),
        rule.pressure,
    )


def local_gradient(rule, pair, dtype=torch.float64):
    """Endpoint `pair`'s gradient, taken on its own under ambient autograd.

    Deliberately not a slice of the batched transform. One map, one stalk, one
    neighbour belief and plain `.backward()` means nothing in the computation
    *could* span endpoints, so what it produces is the local gradient the
    batched one has to match.
    """
    gathered, incoming = rule.inputs()
    single = rule.sheaf.maps.maps[pair].detach().to(dtype).clone().requires_grad_(True)
    restricted = (single @ gathered[pair].to(dtype)).unsqueeze(0)
    term = (
        relative_disagreement(restricted, incoming[pair].to(dtype).unsqueeze(0))[0]
        + rule.pressure
        * normalised_l1(
            single.unsqueeze(0), rule.permitted[pair].to(dtype).unsqueeze(0)
        )[0]
    )
    term.backward()
    return single.grad


def _live_scalar():
    """A one-element tensor that puts whatever multiplies it onto the tape."""
    return torch.ones(1, requires_grad=True)


class TestTheTransportPath:
    """The counterpart to `ForwardPath`: what `functional_call` is called on,
    holding the maps for the length of a gradient step and owning nothing."""

    def test_it_restricts_exactly_what_the_message_passing_phase_does(self, running):
        gathered, _ = TransportRule(running).inputs()
        path = TransportPath(running.maps)
        with torch.no_grad():
            assert torch.equal(path(gathered), running.maps.restrict(gathered))

    def test_the_one_parameter_is_the_padded_map_tensor(self, running):
        path = TransportPath(running.maps)
        assert set(path.map_parameters()) == {MAPS_PARAMETER}
        assert torch.equal(path.map_parameters()[MAPS_PARAMETER], maps_of(running))

    def test_substituting_the_parameter_is_what_the_transform_runs_on(self, running):
        # `functional_call` replaces the module's ambient parameter, so the
        # objective is a function of what it is handed rather than of what the
        # sheaf currently holds. Without that the maps would be live state read
        # off the module and the transform would be decoration.
        rule = TransportRule(running)
        substitute = maps_of(running) * 3.0
        assert not torch.allclose(
            objective_of(rule, maps=substitute), objective_of(rule), atol=1e-4
        )


class TestTheObjectiveIsRelativeDisagreement:
    """ADR-0007's *relative* objective, in ADR-0010's locally stateless form:
    disagreement over `‖F_u x_u‖ + ‖F_v x_v‖`, and never a zero target."""

    def test_it_is_disagreement_over_the_two_restricted_magnitudes(self, running):
        # Read off the sheaf's own per-cell stalks and per-edge buffers rather
        # than off the padded gather, so the endpoint indexing is part of what
        # is checked rather than assumed.
        rule = TransportRule(running)
        _, incoming = rule.inputs()
        taken = relative_disagreement(outgoing_of(rule), incoming)
        for edge in running.dome.edges:
            for side, cell_id in enumerate((edge.u, edge.v)):
                pair = 2 * edge.id + side
                cell = running.dome.cells[cell_id]
                block = running.maps.maps[pair, : edge.m, : cell.stalk].detach()
                own = block @ running.stalk(cell_id)
                neighbour = incoming[pair, : edge.m]
                expected = torch.linalg.vector_norm(own - neighbour) / (
                    torch.linalg.vector_norm(own)
                    + torch.linalg.vector_norm(neighbour)
                )
                assert taken[pair].item() == pytest.approx(expected.item(), abs=1e-6)

    def test_the_neighbours_half_is_what_it_restricted_a_tick_ago(self, running):
        # The normaliser's other term is the belief the neighbour put on the
        # shared edge stalk, not the neighbour's node stalk and not the
        # neighbour's map: what reaches the rule is already transported.
        # And it is a tick stale, which is the unit delay: what the rule sees
        # on an endpoint is its partner's broadcast from `t − 1`.
        before = running.broadcast.reshape(-1, 2, running.maps.edge_width).clone()
        running.tick()
        _, incoming = TransportRule(running).inputs()
        assert torch.equal(incoming, running.incoming)
        assert torch.equal(incoming, before.flip(1).reshape_as(running.broadcast))

    def test_the_ratio_lies_in_the_unit_interval(self, running):
        rule = TransportRule(running)
        _, incoming = rule.inputs()
        ratios = relative_disagreement(outgoing_of(rule), incoming)
        assert torch.all(ratios >= 0.0) and torch.all(ratios <= 1.0 + 1e-6)

    def test_it_is_recomputed_live_in_the_cells_own_maps(self, running):
        # The whole point of recomputing: a map that has moved since the tick
        # changes the disagreement the rule descends on. The tick's own
        # `disagreement()` is a number carried over dead and does not move at
        # all -- descending on that would be the bug that never announces
        # itself.
        rule = TransportRule(running)
        before, dead = objective_of(rule), running.disagreement().clone()
        with torch.no_grad():
            running.maps.maps.mul_(1.5)
        assert not torch.allclose(objective_of(rule), before, atol=1e-4)
        assert torch.equal(running.disagreement(), dead)

    def test_the_gradient_is_the_gradient_of_the_live_objective(self, running):
        # Central differences on the objective itself. This is what pins the
        # rule to relative disagreement rather than to something that merely
        # moves when the maps do.
        rule = TransportRule(running, anneal=SparsityAnneal(horizon=1))
        rule.steps = 1
        parameters, arguments = in_precision(rule, torch.float64)
        taken = transport_gradient(parameters, *arguments)[MAPS_PARAMETER]

        step = 1e-6
        for pair, row, column in ((0, 0, 0), (5, 0, 1), (11, 1, 2)):
            probe = parameters[MAPS_PARAMETER].clone()
            shifted = {MAPS_PARAMETER: probe}
            probe[pair, row, column] += step
            up = transport_objective(shifted, *arguments)
            probe[pair, row, column] -= 2 * step
            down = transport_objective(shifted, *arguments)
            assert (up - down).item() / (2 * step) == pytest.approx(
                taken[pair, row, column].item(), abs=1e-7
            )

    def test_every_map_on_a_wider_edge_receives_a_gradient(self, running, wide_pairs):
        # The missing-gradient failure the transform is chosen to make loud: a
        # wrong `argnums`, or a map left out of the parameter dict, shows up
        # here as an endpoint that never learns. Read on the edges wider than
        # one dimension, for the reason the next test gives.
        gradient = TransportRule(running).gradient()
        assert gradient.shape == running.maps.maps.shape
        per_pair = gradient.flatten(1).abs().sum(-1)
        assert torch.all(per_pair[wide_pairs] > 0)

    def test_a_one_dimensional_edge_stalk_is_flat_where_it_is_worst(
        self, running, narrow_pairs
    ):
        # The one place the objective has no descent direction, recorded
        # because it looks like the missing gradient above and is not one. On
        # an `m = 1` edge both beliefs are scalars, so opposite signs make
        # `‖a − b‖ = ‖a‖ + ‖b‖` identically in a neighbourhood: the ratio is
        # pinned at 1 and flat there. That is a maximum, not a solution the
        # rule can settle into -- reconciliation moves the node stalk every
        # tick, and once the signs agree the gradient is back.
        rule = TransportRule(running, anneal=SparsityAnneal(pressure=0.0))
        _, incoming = rule.inputs()
        outgoing = outgoing_of(rule)
        gradient = rule.gradient().flatten(1).abs().sum(-1)
        opposed = [
            pair
            for pair in narrow_pairs
            if (outgoing[pair, 0] * incoming[pair, 0]).item() < 0
        ]
        assert opposed, "this fixture has no opposed one-dimensional edge"
        for pair in opposed:
            assert relative_disagreement(outgoing, incoming)[pair].item() == (
                pytest.approx(1.0, abs=1e-6)
            )
            assert gradient[pair] == 0.0
        for pair in set(narrow_pairs) - set(opposed):
            assert gradient[pair] > 0


class TestTheObjectiveExcludesTheTrivialSolution:
    """The load-bearing half of ADR-0010: learning on *change* in disagreement
    would read shrinking the maps as progress, and this objective does not."""

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 2.0, 10.0])
    def test_scaling_both_of_an_edges_maps_together_leaves_it_unchanged(
        self, running, alpha
    ):
        # Both ends of an edge scaled together: the cell's own map, and -- since
        # the neighbour's map reaches the rule already applied -- the belief the
        # neighbour restricted. The ratio is homogeneous of the same degree
        # above and below, so shrinking the edge buys nothing.
        rule = TransportRule(running, anneal=SparsityAnneal(pressure=0.0))
        _, incoming = rule.inputs()
        before = relative_disagreement(outgoing_of(rule), incoming)
        with torch.no_grad():
            running.maps.maps.mul_(alpha)
        after = relative_disagreement(outgoing_of(rule), incoming * alpha)
        assert torch.allclose(before, after, atol=1e-5)

    @pytest.mark.parametrize("alpha", [0.1, 0.5])
    def test_shrinking_an_edge_is_not_progress(self, running, alpha):
        # The same fact stated as the rule sees it: the objective a shrunken
        # sheaf reports is the objective it already had, so there is no descent
        # direction in the collapse.
        rule = TransportRule(running, anneal=SparsityAnneal(pressure=0.0))
        _, incoming = rule.inputs()
        before = objective_of(rule)
        after = objective_of(
            rule,
            maps=running.maps.maps.detach() * alpha,
            neighbour_beliefs=incoming * alpha,
        )
        assert after.item() == pytest.approx(before.item(), abs=1e-4)

    def test_sending_the_cells_own_map_to_zero_sends_it_to_one(self, running):
        # One end down and the other not is a different direction, it is not
        # invariant, and it points away from collapse: `1` is the worst value
        # the rule can reach, not a trivial solution it can fall into.
        rule = TransportRule(running)
        _, incoming = rule.inputs()
        outgoing = outgoing_of(rule)
        collapsed = relative_disagreement(outgoing * 0.0, incoming)
        assert torch.allclose(collapsed, torch.ones_like(collapsed), atol=1e-6)
        for shrink in (1.0, 0.5, 0.1, 0.01):
            assert torch.all(
                relative_disagreement(outgoing * shrink, incoming) <= collapsed + 1e-6
            )

    def test_sending_the_neighbours_map_to_zero_sends_it_to_one(self, running):
        rule = TransportRule(running)
        _, incoming = rule.inputs()
        outgoing = outgoing_of(rule)
        collapsed = relative_disagreement(outgoing, incoming * 0.0)
        assert torch.allclose(collapsed, torch.ones_like(collapsed), atol=1e-6)
        for shrink in (1.0, 0.5, 0.1, 0.01):
            assert torch.all(
                relative_disagreement(outgoing, incoming * shrink) <= collapsed + 1e-6
            )

    def test_one_sided_collapse_is_worse_than_where_the_edge_already_is(
        self, running, wide_pairs
    ):
        # The comparison the two tests above leave implicit, on the edges wide
        # enough for it to be strict: collapsing one end is strictly worse than
        # what that endpoint reports now.
        rule = TransportRule(running)
        _, incoming = rule.inputs()
        outgoing = outgoing_of(rule)
        now = relative_disagreement(outgoing, incoming)[wide_pairs]
        collapsed = relative_disagreement(outgoing * 0.0, incoming)[wide_pairs]
        assert torch.all(now < collapsed)

    def test_a_silent_neighbour_leaves_the_endpoint_flat_at_one(self, running):
        # The third flat spot, and the one reachable in the shipped world: a
        # quiet touch sensor reads all zeros, so its whole stalk and every
        # belief it broadcasts are zero, and the endpoint reading it sits at
        # the objective's worst value with nothing to descend. Correct rather
        # than broken -- a cell whose neighbour has said nothing has no
        # evidence about what basis to transport into -- but it is the
        # per-edge form of what the two-tick guard refuses graph-wide, and it
        # is invisible to that guard.
        rule = TransportRule(running, anneal=SparsityAnneal(pressure=0.0))
        silent = 4
        with torch.no_grad():
            running.incoming[silent].zero_()
        _, incoming = rule.inputs()
        ratios = relative_disagreement(outgoing_of(rule), incoming)
        assert ratios[silent].item() == pytest.approx(1.0, abs=1e-6)
        assert rule.gradient()[silent].abs().sum() == 0.0

        # And flat rather than repelling: the cell's own map may be anything
        # at all and the endpoint still reports exactly 1.
        for scale in (0.1, 1.0, 10.0):
            with torch.no_grad():
                running.maps.maps[silent].mul_(scale)
            _, incoming = rule.inputs()
            moved = relative_disagreement(outgoing_of(rule), incoming)[silent]
            assert moved.item() == pytest.approx(1.0, abs=1e-6)

    def test_the_rule_has_no_fixed_point_at_agreement(self):
        # Recorded because `NORM_FLOOR`'s comment could be read as promising
        # one and does not. The numerator is a norm rather than a squared norm
        # -- which is what ADR-0010's `[0, 1]` triangle-inequality reading
        # needs -- so the gradient does not vanish as the two beliefs meet: an
        # endpoint at agreement keeps taking a full step. That sits with
        # ADR-0007 (never a zero target, and the floor means zero is never
        # reached) rather than against it.
        agreed = torch.randn(1, 4, dtype=torch.float64)
        taken = []
        for gap in (0.0, 1e-9, 1e-3, 1e-1):
            probe = (agreed + gap).clone().requires_grad_(True)
            relative_disagreement(probe, agreed).backward()
            taken.append(probe.grad.norm().item())
        # At the meeting point itself the floor leaves `1e-13` of gradient
        # in double and exactly zero at the precision the rule ships in --
        # either way, nothing. One step off it, a full-size step.
        assert taken[0] < 1e-12
        assert all(value > 0.1 for value in taken[1:])
        # Within a factor of two of each other across eight orders of
        # magnitude of disagreement, which is the whole of the point: there is
        # no basin here for an endpoint to settle into.
        assert max(taken[1:]) < 2 * min(taken[1:])

    def test_the_normaliser_is_the_current_magnitudes_not_a_running_average(
        self, running
    ):
        # ADR-0007 rejects a per-edge baseline with a hand-set time constant,
        # and ADR-0010 says the normaliser is locally stateless for the same
        # reason. Stated as a property: the objective is a function of the
        # state in front of it, so a rule three steps into a run and a rule
        # built fresh report the same thing about the same state.
        rule = TransportRule(running, anneal=SparsityAnneal(pressure=0.0))
        for _ in range(3):
            rule.step()
        fresh = TransportRule(running, anneal=SparsityAnneal(pressure=0.0))
        assert torch.equal(rule.gradient(), fresh.gradient())


class TestTheSparsityPressureComposesInTheSameStep:
    """One additive penalty term inside one descent step, not a second update
    loop running alongside it — and an L1 on the *normalised* map, so it
    redistributes weight across a map's directions rather than removing it."""

    def test_the_objective_is_the_two_terms_added(self, running):
        rule = TransportRule(running)
        _, incoming = rule.inputs()
        pressure = 0.37
        disagreement = relative_disagreement(outgoing_of(rule), incoming).sum()
        penalty = normalised_l1(running.maps.maps.detach(), rule.permitted).sum()
        composed = objective_of(rule, pressure=pressure)
        assert composed.item() == pytest.approx(
            (disagreement + pressure * penalty).item(), rel=1e-6
        )

    def test_the_gradient_is_the_two_gradients_added(self, running):
        # Additivity in the pressure is what "composes in the same step" means
        # mechanically: one gradient of one sum, not two steps taken in turn.
        rule = TransportRule(running, anneal=SparsityAnneal(pressure=0.0))
        gathered, incoming = rule.inputs()
        parameters = rule.path.map_parameters()

        def taken(pressure):
            return transport_gradient(
                parameters, rule.path, gathered, incoming, rule.permitted, pressure
            )[MAPS_PARAMETER]

        transport, both = taken(0.0), taken(0.4)
        penalty = (both - transport) / 0.4
        assert torch.allclose(taken(0.9), transport + 0.9 * penalty, atol=1e-6)
        assert penalty.abs().sum() > 0

    def test_the_step_writes_the_maps_once(self, running):
        # A second update loop would show up here as a map that is not the
        # projection of one descent on one gradient.
        rule = TransportRule(running, learning_rate=0.05)
        rule.steps = DEFAULT_ANNEAL_HORIZON
        before = maps_of(running)
        gradient = rule.step()
        after = maps_of(running)
        with torch.no_grad():
            running.maps.maps.copy_(before - 0.05 * gradient)
        running.maps.project()
        assert torch.equal(maps_of(running), after)

    def _term_norms(self, running):
        """`[pairs]` gradient norms of the two terms, the penalty at `λ = 1`."""
        rule = TransportRule(running)
        gathered, incoming = rule.inputs()
        parameters = rule.path.map_parameters()

        def taken(pressure):
            return transport_gradient(
                parameters, rule.path, gathered, incoming, rule.permitted, pressure
            )[MAPS_PARAMETER]

        transport = taken(0.0)
        return (
            transport.flatten(1).norm(dim=-1),
            (taken(1.0) - transport).flatten(1).norm(dim=-1),
        )

    def test_the_ceiling_leaves_pruning_secondary_to_transport(
        self, running, free_pairs
    ):
        # What `DEFAULT_SPARSITY_PRESSURE` is set by, so that the number and
        # its stated reason cannot drift apart. Read as a median, because the
        # ratio is unbounded wherever the transport term happens to be flat.
        transport, penalty = self._term_norms(running)
        ratio = DEFAULT_SPARSITY_PRESSURE * penalty[free_pairs] / transport[free_pairs]
        assert 0.03 < ratio.median().item() < 0.3

    def test_the_pressure_does_not_grade_with_what_the_mask_leaves_open(
        self, running, free_pairs
    ):
        # The `1/√p` normalisation's whole job (ADR-0010, amended in #89).
        # Without it the term's gradient norm carried a `+0.985` correlation
        # with the open-weight count, so one global `λ` pruned a wide map
        # roughly eightfold harder than a narrow one. With it `p` is gone from
        # the gradient identically, and what correlation survives is noise.
        _, penalty = self._term_norms(running)
        permitted = _permitted(running).float()[free_pairs]
        stacked = torch.stack([permitted, penalty[free_pairs]])
        assert abs(torch.corrcoef(stacked)[0, 1].item()) < 0.2

    def test_the_pressures_gradient_is_free_of_the_mask_size(self):
        # The identity the ruling in #89 turned on, checked directly rather
        # than inferred from a correlation: for `h = ‖F‖₁/(√p‖F‖_F)`,
        # `‖∇h‖ = √(1 − h²)/‖F‖_F`, in which `p` does not appear.
        for permitted in (2, 8, 13, 96, 384):
            weights = torch.randn(permitted, dtype=torch.float64).requires_grad_(True)
            count = torch.tensor([float(permitted)], dtype=torch.float64)
            value = normalised_l1(weights.reshape(1, 1, -1), count)[0]
            (taken,) = torch.autograd.grad(value, weights)
            expected = (1 - value.item() ** 2) ** 0.5 / weights.detach().norm().item()
            assert taken.norm().item() == pytest.approx(expected, rel=1e-9)

    def test_the_penalty_is_blind_to_a_maps_magnitude(self, running):
        maps = running.maps.maps.detach()
        permitted = _permitted(running).float()
        for alpha in (0.25, 4.0):
            assert torch.allclose(
                normalised_l1(maps * alpha, permitted),
                normalised_l1(maps, permitted),
                atol=1e-5,
            )

    def test_at_a_fixed_norm_the_penalty_prefers_the_sparser_map(self):
        # What makes it a *pruning* pressure rather than weight decay: at fixed
        # Frobenius norm the sum of absolute values is smallest when the map's
        # weight sits on fewest directions. The `1/√p` is a constant per map,
        # so it cannot change this -- which is the point of it. Both maps here
        # have the same mask, so both divide by the same root.
        concentrated = torch.tensor([[[2.0, 0.0], [0.0, 0.0]]])
        spread = torch.full((1, 2, 2), 1.0)
        permitted = torch.tensor([4.0])
        assert torch.allclose(
            torch.linalg.matrix_norm(concentrated), torch.linalg.matrix_norm(spread)
        )
        assert (
            normalised_l1(concentrated, permitted).item()
            < normalised_l1(spread, permitted).item()
        )
        # Hoyer's ratio: `1/√p` on one direction, exactly `1` when flat.
        assert normalised_l1(concentrated, permitted).item() == pytest.approx(0.5)
        assert normalised_l1(spread, permitted).item() == pytest.approx(1.0)

    def test_the_penalty_redistributes_rather_than_removes(self, running):
        # "Prunes within the mask; does not shrink the stalk", mechanically:
        # under the pressure alone every map keeps its gauge-fixed norm and
        # its widest entries give way to its narrowest.
        rule = TransportRule(
            running, learning_rate=0.05, anneal=SparsityAnneal(horizon=1)
        )
        rule.steps = 1
        widest = running.maps.maps.detach().abs().flatten(1).max(-1).values
        for _ in range(5):
            rule.step()
        assert torch.all(running.maps.norms() >= 1.0 / GAUGE_RHO)
        now = running.maps.maps.detach().abs().flatten(1).max(-1).values
        assert torch.any(now != widest)


class TestTheAnnealSchedule:
    """The second and last permitted global signal: one scalar, schedule-shaped
    rather than information-shaped, reading nothing about any cell or edge."""

    def test_it_starts_at_nothing_and_reaches_the_ceiling(self):
        anneal = SparsityAnneal(pressure=0.2, horizon=10)
        assert anneal.at(0) == 0.0
        assert anneal.at(5) == pytest.approx(0.1)
        assert anneal.at(10) == pytest.approx(0.2)
        assert anneal.at(1000) == pytest.approx(0.2)

    def test_it_is_a_function_of_the_step_and_nothing_else(self):
        anneal = SparsityAnneal(pressure=0.2, horizon=10)
        assert [anneal.at(step) for step in range(4)] == [
            anneal.at(step) for step in range(4)
        ]

    def test_the_rule_walks_the_schedule_one_step_at_a_time(self, running):
        rule = TransportRule(running, anneal=SparsityAnneal(pressure=0.2, horizon=4))
        seen = []
        for _ in range(6):
            seen.append(rule.pressure)
            rule.step()
        assert seen == pytest.approx([0.0, 0.05, 0.1, 0.15, 0.2, 0.2])

    def test_the_default_ceiling_is_positive_and_small(self):
        assert 0 < DEFAULT_SPARSITY_PRESSURE < 1
        assert DEFAULT_ANNEAL_HORIZON >= 1

    @pytest.mark.parametrize("pressure", [-1e-6, float("nan"), float("inf")])
    def test_a_pressure_that_is_not_a_scalar_is_refused(self, pressure):
        # nan and inf among them, for the reason the learning rate refuses
        # them: a nan pressure poisons every map on the first step.
        with pytest.raises(ValueError, match="global scalar"):
            SparsityAnneal(pressure=pressure)

    @pytest.mark.parametrize(
        "horizon", [0, -1, float("nan"), float("inf"), float("-inf")]
    )
    def test_a_horizon_that_is_not_a_step_count_is_refused(self, horizon):
        # nan and inf among them: `horizon < 1` admits both, and both switch
        # the schedule off rather than failing. A nan horizon puts
        # `min(1.0, nan)` at 1.0, so the full pressure applies from step zero;
        # an infinite one holds it at zero for the length of the run.
        with pytest.raises(ValueError, match="positive step count"):
            SparsityAnneal(horizon=horizon)

    @pytest.mark.parametrize("step", [-1, float("nan"), float("inf")])
    def test_a_position_that_is_not_on_the_schedule_is_refused(self, step):
        with pytest.raises(ValueError, match="starts at step 0"):
            SparsityAnneal().at(step)


class TestTheGaugeProjectionRunsAfterTheStep:
    """ADR-0010, enforced by projection: take the transport step, then project
    the map back into the band. Outside the transform, and not inert."""

    def test_every_interior_map_is_back_in_its_band(self, running):
        rule = TransportRule(running, learning_rate=0.5)
        for _ in range(5):
            rule.step()
        lower, upper = running.maps.gauge_bounds
        norms = running.maps.norms()
        assert torch.all(norms >= lower - 1e-6) and torch.all(norms <= upper + 1e-6)

    def test_a_boundary_cells_maps_carry_the_exact_gauge(self, running):
        rule = TransportRule(running, learning_rate=0.5)
        for _ in range(5):
            rule.step()
        pinned = running.maps.norms()[running.maps.pinned]
        assert pinned.numel() > 0
        assert torch.allclose(pinned, torch.ones_like(pinned), atol=1e-6)

    def test_the_projection_sees_the_maps_the_step_descended(self, running):
        # "After the step" as an ordering rather than as an outcome: what
        # `project` is handed is the descended map, not the map the step
        # started from.
        rule = TransportRule(running, learning_rate=0.5)
        before, seen = maps_of(running), []
        original = RestrictionMaps.project

        def spy(self):
            seen.append(self.maps.detach().clone())
            original(self)

        with mock.patch.object(RestrictionMaps, "project", spy):
            gradient = rule.step()
        assert len(seen) == 1
        assert torch.equal(seen[0], before - 0.5 * gradient)
        assert not torch.equal(seen[0], maps_of(running))

    def test_without_it_the_step_walks_out_of_the_band(self, running):
        # The projection is doing work rather than restating an invariant the
        # descent already respects.
        rule = TransportRule(running, learning_rate=0.5)
        gradient = rule.gradient()
        with torch.no_grad():
            unprojected = running.maps.maps.detach() - 0.5 * gradient
        norms = unprojected.flatten(1).norm(dim=-1)
        lower, upper = running.maps.gauge_bounds
        assert torch.any((norms < lower) | (norms > upper))

    def test_it_is_outside_the_transform(self, running):
        # Nothing differentiates through the projection: evaluating the
        # objective moves no map, and the gradient is the gradient of the
        # unprojected quantity, which the finite differences above already
        # pinned.
        rule = TransportRule(running)
        before = maps_of(running)
        objective_of(rule)
        rule.gradient()
        assert torch.equal(maps_of(running), before)

    def test_the_upper_face_binds_once_a_map_reaches_rho(self, running):
        # ADR-0010's corrected drift direction: the joint scale grows, so the
        # projection is not a guardrail that never fires. Started at the
        # ceiling, every interior map is held there rather than drifting up.
        with torch.no_grad():
            running.maps.maps.mul_(GAUGE_RHO)
        running.maps.project()
        rule = TransportRule(running, learning_rate=0.1)
        for _ in range(3):
            rule.step()
        interior = running.maps.norms()[~running.maps.pinned]
        assert torch.all(interior <= GAUGE_RHO + 1e-6)


class TestTheNeighboursMapEntersDetached:
    """The one cross-cell parameter in the phase. It is an ordinary argument of
    the objective and not among the `argnums`, so it is not differentiated —
    not because it was severed, but because differentiation only ever traverses
    what was named."""

    def test_a_neighbours_map_receives_no_gradient(self, running, wide_pairs):
        # Taken one endpoint at a time, because the batched gradient is a sum
        # over all of them and would hide a cross term inside the partner's own
        # legitimate row -- and taken off the **shipped** objective rather than
        # a re-implementation of it, which would only assert the locality of
        # arithmetic this test had written itself.
        #
        # The isolation is the sum's own additivity: one endpoint's neighbour
        # belief appears in exactly one term, so nudging it moves precisely the
        # rows that term's gradient reaches. Under a correct rule that is the
        # endpoint's own row and nothing else. A partner's map inside the term
        # would move the partner's row along with it.
        rule = TransportRule(running, anneal=SparsityAnneal(horizon=1))
        rule.steps = 1
        gathered, incoming = rule.inputs()
        parameters = rule.path.map_parameters()

        def taken(beliefs):
            return transport_gradient(
                parameters, rule.path, gathered, beliefs, rule.permitted, rule.pressure
            )[MAPS_PARAMETER]

        before = taken(incoming)
        for pair in (wide_pairs[0], wide_pairs[1], wide_pairs[-1]):
            nudged = incoming.clone()
            nudged[pair] += 0.5
            moved = (taken(nudged) - before).flatten(1).abs().sum(-1)
            assert moved[pair] > 0
            assert moved[pair ^ 1] == 0.0
            assert (moved > 0).sum() == 1

    def test_the_gradient_does_not_move_when_a_neighbours_map_does(self, running):
        # The same claim from the outside, and the shape #90 makes standing:
        # move the partner's map and the endpoint's own update does not follow.
        rule = TransportRule(running)
        before = rule.gradient()
        with torch.no_grad():
            running.maps.maps[1::2].mul_(0.5)
        after = rule.gradient()
        assert torch.equal(before[0::2], after[0::2])
        assert not torch.equal(before[1::2], after[1::2])

    def test_the_rule_never_reads_a_neighbours_raw_node_stalk(self, running):
        # What a cell learns about its neighbour is the belief the neighbour
        # already restricted onto the shared edge stalk. A raw neighbour stalk
        # is in the wrong space until the map has done that work, and nothing
        # here reads one.
        rule = TransportRule(running)
        pair = 0
        owner = int(running.maps.owner[pair])
        before = rule.gradient()
        with torch.no_grad():
            for cell in running.dome.cells:
                if cell.id != owner:
                    running.stalks[running.layout.slice(cell.id)].add_(1.0)
        after = rule.gradient()
        assert torch.equal(before[pair], after[pair])
        assert not torch.equal(before, after)

    def test_the_gathered_stalks_are_each_endpoints_owners_own(self, running):
        gathered, _ = TransportRule(running).inputs()
        for pair in range(running.maps.pairs):
            cell = running.dome.cells[int(running.maps.owner[pair])]
            assert torch.equal(gathered[pair, : cell.stalk], running.stalk(cell.id))
            assert torch.all(gathered[pair, cell.stalk :] == 0)


class TestTheBatchedGradientEqualsThePerEndpointLocalGradient:
    """What makes `vmap` unnecessary, and what the batching claim rests on:
    the padded map tensor is one parameter, but no term of the summed objective
    reaches past its own row of it."""

    # The identity is exact arithmetic, so it is read in double as well as at
    # the precision the rule ships in: a padded `[pairs, m, stalk]` bmm and a
    # single `[m, stalk]` matmul differ in the last bits, which is a fact about
    # accumulation order and not about the claim.
    @pytest.mark.parametrize(
        "dtype, atol", [(torch.float64, 1e-12), (torch.float32, 1e-6)]
    )
    def test_every_endpoint(self, running, dtype, atol):
        rule = TransportRule(running, anneal=SparsityAnneal(horizon=1))
        rule.steps = 1
        parameters, arguments = in_precision(rule, dtype)
        batched = transport_gradient(parameters, *arguments)[MAPS_PARAMETER]
        for pair in range(running.maps.pairs):
            assert torch.allclose(
                local_gradient(rule, pair, dtype),
                batched[pair],
                atol=atol,
                rtol=0.0,
            ), pair

    def test_the_shipped_rule_is_that_batched_gradient(self, running):
        rule = TransportRule(running)
        parameters, arguments = in_precision(rule, torch.float32)
        expected = transport_gradient(parameters, *arguments)[MAPS_PARAMETER]
        assert torch.equal(rule.gradient(), expected)

    def test_one_endpoints_gradient_does_not_move_when_another_endpoints_stalk_does(
        self, running
    ):
        rule = TransportRule(running)
        before = rule.gradient()
        moved = int(running.maps.owner[1])
        with torch.no_grad():
            running.stalks[running.layout.slice(moved)].add_(0.5)
        after = rule.gradient()
        for pair in range(running.maps.pairs):
            if int(running.maps.owner[pair]) != moved:
                assert torch.equal(before[pair], after[pair]), pair


class TestTrainingChangesNoShape:
    """`m` is fixed at construction and the mask closes and never re-opens. The
    sparsity pressure prunes *within* the mask; it does not shrink a stalk and
    it removes no edge."""

    @pytest.fixture
    def trained(self, running):
        rule = TransportRule(
            running, learning_rate=0.2, anneal=SparsityAnneal(horizon=1)
        )
        rule.steps = 1
        for _ in range(10):
            rule.step()
        return running

    def test_no_edge_is_removed(self, dome, trained):
        assert len(trained.dome.edges) == len(dome.edges)
        assert trained.maps.pairs == 2 * len(dome.edges)
        assert [edge.m for edge in trained.dome.edges] == [
            edge.m for edge in dome.edges
        ]

    def test_no_stalk_dimension_moves(self, dome, trained):
        assert [cell.stalk for cell in trained.dome.cells] == [
            cell.stalk for cell in dome.cells
        ]
        assert trained.maps.maps.shape == (
            trained.maps.pairs,
            trained.maps.edge_width,
            trained.maps.stalk_width,
        )

    def test_no_masked_feature_re_opens(self, dome, trained):
        support = trained.maps.support
        assert torch.equal(
            support, RestrictionMaps(dome, generator=torch.Generator()).support
        )
        assert torch.all(trained.maps.maps.detach()[~support] == 0)

    def test_training_never_walks_a_map_to_zero(self, trained):
        # "Prunes within the mask; does not shrink the stalk" has teeth only if
        # `F = 0` stays out of reach: the gauge's lower face is what keeps a
        # pruned edge functionally weak rather than absent.
        assert torch.all(trained.maps.norms() >= 1.0 / GAUGE_RHO - 1e-6)


class TestAPhaseSeparateFromTheTick:
    def test_the_tick_alone_trains_no_map(self, running):
        before = maps_of(running)
        running.tick()
        assert torch.equal(running.maps.maps.detach(), before)

    def test_the_rule_alone_moves_no_tick_state(self, running):
        before = {
            name: getattr(running, name).clone()
            for name in ("stalks", "charts", "prediction", "broadcast", "incoming")
        }
        TransportRule(running).step()
        for name, value in before.items():
            assert torch.equal(getattr(running, name), value)

    def test_the_rule_runs_over_state_the_tick_left_detached(self, running):
        for tensor in TransportRule(running).inputs():
            assert tensor.grad_fn is None and not tensor.requires_grad

    def test_a_step_before_the_first_tick_is_refused(self, sheaf):
        # A fresh sheaf's `incoming` is zeros, which are a well-formed record
        # of nothing: descending on them reports a gradient rather than a
        # mistake.
        with pytest.raises(ValueError, match="needs two ticks to learn from"):
            TransportRule(sheaf).gradient()

    def test_the_wrong_order_is_refused_rather_than_silently_wrong(self, sheaf):
        with pytest.raises(ValueError, match="needs two ticks to learn from"):
            TransportRule(sheaf).step()

    def test_one_tick_is_not_enough(self, sheaf):
        # The unit delay, as the rule sees it: the message-passing phase reads
        # the broadcast buffer as it stood *before* the phase, so the first
        # tick reconciles against the constructor's zeros. A rule that took
        # this state would take a silent null step -- no gradient, but a
        # position burned on the anneal schedule and every map re-projected.
        sheaf.tick()
        assert sheaf.incoming.abs().sum() == 0.0
        with pytest.raises(ValueError, match="needs two ticks to learn from"):
            TransportRule(sheaf).gradient()

    def test_two_ticks_are_enough(self, sheaf):
        for _ in range(2):
            sheaf.tick()
        assert sheaf.incoming.abs().sum() > 0.0
        assert TransportRule(sheaf).gradient().abs().sum() > 0.0


class TestOneGlobalLearningRate:
    @pytest.mark.parametrize("learning_rate", [1e-3, 0.03, 0.5])
    def test_the_step_is_the_learning_rate_times_the_gradient_then_projected(
        self, running, learning_rate
    ):
        # One scalar, the same for every cell and every edge, and nothing else
        # in the step but the projection. There is no optimiser here.
        rule = TransportRule(running, learning_rate=learning_rate)
        before = maps_of(running)
        gradient = rule.step()
        after = maps_of(running)
        with torch.no_grad():
            running.maps.maps.copy_(before - learning_rate * gradient)
        running.maps.project()
        assert torch.equal(maps_of(running), after)

    def test_the_step_moves_every_map_the_mask_leaves_free(self, running, free_pairs):
        # The formula above is satisfied by a zero gradient too. Read with the
        # pressure at its ceiling, so that every map moves rather than only the
        # ones whose edge has something to descend this tick.
        rule = TransportRule(running, anneal=SparsityAnneal(horizon=1))
        rule.steps = 1
        before = maps_of(running)
        rule.step()
        moved = (running.maps.maps.detach() - before).flatten(1).abs().sum(-1)
        assert torch.all(moved[free_pairs] > 0)

    def test_a_map_the_mask_leaves_one_entry_is_fixed_by_the_gauge(
        self, running, single_entry_pairs
    ):
        # Recorded because it looks like a map that has stopped learning and is
        # not one. Where the structural mask permits exactly one weight, that
        # weight's magnitude is the map's whole norm, and the gauge fixes the
        # norm -- so the transport step moves it and the projection puts it
        # back. Nothing is left for the rule to identify, which is the
        # unidentified-magnitude argument at its smallest.
        assert single_entry_pairs
        rule = TransportRule(running, anneal=SparsityAnneal(horizon=1))
        rule.steps = 1
        before = maps_of(running)
        rule.step()
        for pair in single_entry_pairs:
            assert torch.allclose(
                running.maps.maps.detach()[pair], before[pair], atol=1e-6
            )

    def test_the_rule_carries_nothing_per_cell_or_per_edge(self, running):
        # No momentum, no running average, no per-edge baseline: two rules
        # stepped from the same state at the same schedule position agree, and
        # a second step of one rule is the plain gradient of the state it now
        # sees.
        anneal = SparsityAnneal(pressure=0.0)
        first = TransportRule(running, learning_rate=0.03, anneal=anneal)
        before = maps_of(running)
        first.step()
        gradient = first.step()
        after_two = maps_of(running)

        with torch.no_grad():
            running.maps.maps.copy_(before)
        second = TransportRule(running, learning_rate=0.03, anneal=anneal)
        second.step()
        assert torch.equal(second.step(), gradient)
        assert torch.equal(maps_of(running), after_two)

    def test_the_only_thing_it_carries_is_the_schedules_position(self, running):
        rule = TransportRule(running)
        assert set(vars(rule)) == {
            "sheaf",
            "learning_rate",
            "anneal",
            "path",
            "permitted",
            "steps",
        }
        assert rule.steps == 0

    def test_the_open_weight_counts_are_the_masks_own_and_never_move(self, running):
        # `permitted` is the one per-edge array the rule holds, and it has to
        # be a structural constant rather than state for the "no per-edge
        # auxiliary variable" constraint to survive it. So: it is read off the
        # mask, and training does not touch it.
        rule = TransportRule(running, learning_rate=0.2)
        assert torch.equal(rule.permitted, _permitted(running).to(rule.permitted.dtype))
        before = rule.permitted.clone()
        for _ in range(5):
            rule.step()
        assert torch.equal(rule.permitted, before)
        assert torch.equal(rule.permitted, _permitted(running).to(rule.permitted.dtype))

    def test_the_maps_gain_no_buffer_of_their_own(self, running):
        # A per-edge auxiliary variable would have to live somewhere, and the
        # maps are the only per-edge object in the design.
        TransportRule(running).step()
        assert {name for name, _ in running.maps.named_buffers()} == {
            "support",
            "owner",
            "pinned",
        }
        assert {name for name, _ in running.maps.named_parameters()} == {"maps"}

    @pytest.mark.parametrize(
        "learning_rate", [0.0, -1e-3, float("nan"), float("inf"), float("-inf")]
    )
    def test_a_non_positive_learning_rate_is_refused(self, sheaf, learning_rate):
        with pytest.raises(ValueError, match="global scalar"):
            TransportRule(sheaf, learning_rate=learning_rate)

    def test_the_default_is_the_same_scalar_the_bias_rule_runs(self, running):
        assert TransportRule(running).learning_rate == DEFAULT_LEARNING_RATE


class TestTheGuard:
    def test_the_gradient_carries_no_tape(self, running):
        gradient = TransportRule(running).gradient()
        assert gradient.grad_fn is None and not gradient.requires_grad

    def test_the_rule_leaves_the_ticks_state_off_the_tape(self, running):
        TransportRule(running).step()
        running.assert_no_tape()
        assert running.maps.maps.grad_fn is None and running.maps.maps.is_leaf

    def test_the_maps_accumulate_nothing_on_dot_grad(self, running):
        # The transform returns gradients as a pytree. Anything landing on
        # `.grad` would mean an ambient `.backward()` had run somewhere, which
        # is the idiom this rule is written to avoid.
        TransportRule(running).step()
        assert running.maps.maps.grad is None

    def test_the_step_needs_its_no_grad(self, running):
        # `no_grad` is a context manager rather than a decorator precisely so
        # this can reach it. With the guard removed the in-place descent on a
        # leaf that requires grad is refused outright -- loud, which is the
        # point.
        with mock.patch("torch.no_grad", contextlib.nullcontext):
            with pytest.raises(RuntimeError, match="leaf Variable"):
                TransportRule(running).step()

    @pytest.mark.parametrize("record", ["stalks", "incoming"])
    def test_the_rule_refuses_an_input_that_is_not_detached(self, running, record):
        setattr(running, record, getattr(running, record) * _live_scalar())
        with pytest.raises(AssertionError, match="autograd tape"):
            TransportRule(running).gradient()

    def test_the_gradient_never_carries_a_nan(self, running):
        # The norms are floored rather than plain for this reason: an edge on
        # which nothing has been said yet has `‖v‖ = 0`, whose gradient is
        # `0/0`, and one `nan` reaches every map in the graph through the
        # projection's rescale within a step.
        with torch.no_grad():
            running.incoming.zero_()
            running.stalks.zero_()
        gradient = TransportRule(running).gradient()
        assert torch.all(torch.isfinite(gradient))

    def test_the_parameters_handed_to_the_transform_are_a_copy(self, running):
        # `Tensor.detach` shares storage, so without the clone an in-place
        # write through this dict reaches the running adapting surface while
        # leaving a clean tape -- the leak class #90 exists to catch and the
        # tape assertion cannot see.
        path = TransportRule(running).path
        handed = path.map_parameters()
        before = maps_of(running)
        handed[MAPS_PARAMETER].add_(1.0)
        assert torch.equal(running.maps.maps.detach(), before)


def test_the_real_dome_trains_inside_its_gauge_and_its_mask():
    # Everything above runs on a dome small enough to iterate over. This is the
    # one that runs on the graph the proof of concept actually builds -- 1364
    # edge endpoints, every mask size the taper produces -- because the gauge,
    # the mask and the padding are properties of the shape, and the shape is
    # what changes here.
    sheaf = Sheaf(build_graph(), generator=torch.Generator().manual_seed(0))
    rule = TransportRule(sheaf, learning_rate=0.1, anneal=SparsityAnneal(horizon=1))
    support = sheaf.maps.support.clone()
    for _ in range(3):
        sheaf.tick()
        if sheaf.ticks > 1:
            rule.step()

    lower, upper = sheaf.maps.gauge_bounds
    norms = sheaf.maps.norms()
    assert torch.all(norms >= lower - 1e-5) and torch.all(norms <= upper + 1e-5)
    pinned = norms[sheaf.maps.pinned]
    assert torch.allclose(pinned, torch.ones_like(pinned), atol=1e-6)
    assert torch.equal(sheaf.maps.support, support)
    assert torch.all(sheaf.maps.maps.detach()[~support] == 0)
    assert torch.all(torch.isfinite(sheaf.maps.maps.detach()))


@pytest.mark.parametrize("seed", [0, 3, 17])
def test_the_real_domes_pressure_is_the_fraction_the_constant_records(seed):
    # `DEFAULT_SPARSITY_PRESSURE`'s docstring quotes a median of 0.12 measured
    # on **this** dome. The window is wide enough to survive a reseed and far
    # too narrow to survive the constant moving, which is the job: a number
    # recorded in a comment and held nowhere drifts away from what it claims.
    dome = build_graph()
    sheaf = Sheaf(dome, generator=torch.Generator().manual_seed(seed))
    generator = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        sheaf.stalks[: sheaf.layout.total] = torch.randn(
            sheaf.layout.total, generator=generator
        )
        sheaf.charts.normal_(0.0, 1.0, generator=generator)
    for _ in range(4):
        sheaf.tick()

    rule = TransportRule(sheaf)
    gathered, incoming = rule.inputs()
    parameters = rule.path.map_parameters()

    def taken(pressure):
        return transport_gradient(
            parameters, rule.path, gathered, incoming, rule.permitted, pressure
        )[MAPS_PARAMETER]

    transport = taken(0.0).flatten(1).norm(dim=-1)
    penalty = (taken(1.0) - taken(0.0)).flatten(1).norm(dim=-1)
    # The flat endpoints are excluded rather than clamped: a zero transport
    # gradient makes the ratio infinite and says nothing about the balance.
    live = (_permitted(sheaf) > 1) & (transport > 0)
    ratio = DEFAULT_SPARSITY_PRESSURE * penalty[live] / transport[live]
    assert 0.08 < ratio.median().item() < 0.18
