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

from patchworks.graph import build_graph
from patchworks.learning import (
    DEFAULT_LEARNING_RATE,
    MAPS_PARAMETER,
    NORM_FLOOR,
    TransportPath,
    TransportRule,
    relative_disagreement,
    transport_gradient,
    transport_objective,
)
from patchworks.restriction import GAUGE_RHO, RestrictionMaps
from patchworks.tick import Sheaf

from conftest import SMALL


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


def objective_of(rule, *, maps=None, neighbour_beliefs=None):
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
    )


def in_precision(rule, dtype):
    """The rule's own arguments, cast to one precision. The maps are cast as a
    parameter dict rather than through `Module.to`, which is in place and would
    convert the sheaf the rest of the test is still running on."""
    gathered, incoming = (tensor.to(dtype) for tensor in rule.inputs())
    parameters = {MAPS_PARAMETER: rule.sheaf.maps.maps.detach().to(dtype)}
    return parameters, (rule.path, gathered, incoming)


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
    term = relative_disagreement(restricted, incoming[pair].to(dtype).unsqueeze(0))[0]
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
        # shared lane, not the neighbour's node stalk and not the
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
        rule = TransportRule(running)
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

    def test_a_one_dimensional_lane_is_flat_where_it_is_worst(
        self, running, narrow_pairs
    ):
        # The one place the objective has no descent direction, recorded
        # because it looks like the missing gradient above and is not one. On
        # an `m = 1` edge both beliefs are scalars, so opposite signs make
        # `‖a − b‖ = ‖a‖ + ‖b‖` identically in a neighbourhood: the ratio is
        # pinned at 1 and flat there. That is a maximum, not a solution the
        # rule can settle into -- reconciliation moves the node stalk every
        # tick, and once the signs agree the gradient is back.
        rule = TransportRule(running)
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
        rule = TransportRule(running)
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
        rule = TransportRule(running)
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
        rule = TransportRule(running)
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

    @pytest.mark.parametrize("scale", [1e-2, 1e-1, 1.0, 10.0, 1e3])
    def test_the_rule_has_no_fixed_point_at_agreement(self, scale):
        # Recorded because `NORM_FLOOR`'s comment could be read as promising
        # one and does not. The numerator is a norm rather than a squared norm
        # -- which is what ADR-0010's `[0, 1]` triangle-inequality reading
        # needs -- so the gradient does not vanish as the two beliefs meet: an
        # endpoint at agreement keeps taking a full step. That sits with
        # ADR-0007 (never a zero target, and the floor means zero is never
        # reached) rather than against it.
        #
        # **Every bound below is per unit of `‖agreed‖`, and the scale is swept
        # rather than drawn** (#111). The objective is exactly invariant under
        # an edge's joint scale (ADR-0010), so scale is the one axis a random
        # draw buys no coverage on -- except that `NORM_FLOOR` breaks that
        # invariance, which is what these bounds have to survive, so it is
        # swept deliberately. The *direction* is still drawn unseeded, which is
        # where the geometry varies.
        #
        # The old form drew scale and direction together and hand-set all three
        # bounds for a typical draw. Each was wrong at one end and none said
        # so: `taken[0] < 1e-12` failed below `‖agreed‖ ≈ 0.5` -- 35 of 4000
        # global RNG states, the flake this came from -- `> 0.1` fails above
        # `≈ 5`, and the factor of two fails below `≈ 0.3` for some directions.
        # A draw of `randn(1, 4)` lands outside `[0.5, 5]` once in 138 --
        # `‖·‖² ~ χ²₄`, so the rate is closed-form, not the `35/4000` above,
        # which was one high sample of it.
        direction = torch.randn(1, 4, dtype=torch.float64)
        agreed = direction / direction.norm() * scale
        norm = agreed.norm().item()
        taken = []
        # Displacements are relative for the same reason the bounds are: a
        # ladder in absolute units means "eight orders of magnitude of
        # disagreement" only for an `agreed` of order one.
        for gap in (0.0, 1e-9, 1e-3, 1e-1):
            probe = (agreed + gap * norm).clone().requires_grad_(True)
            relative_disagreement(probe, agreed).backward()
            taken.append(probe.grad.norm().item())
        # At the meeting point the numerator contributes nothing -- its own
        # gradient is `0/√NORM_FLOOR` -- so the whole residue is the
        # denominator term, `√NORM_FLOOR / (‖probe‖ + ‖agreed‖)²`, which is
        # `√NORM_FLOOR / (2‖agreed‖)²`. Asserted as the equality it is rather
        # than as a ceiling, because a ceiling with headroom passes on any
        # residue below it and so holds down nothing about where the residue
        # comes from. It is an approximation only in dropping the floor inside
        # the denominator's own square roots, which costs a relative
        # `1.5·NORM_FLOOR/‖agreed‖²` -- 1.5e-20 at the smallest scale swept,
        # and reaching the `rel=1e-9` asserted here at `‖agreed‖ ≈ 3.9e-8`.
        #
        # `abs=0.0` matters at the top of the sweep, not the bottom: the
        # residue grows as `agreed` shrinks, so it is `2.5e-9` at `scale =
        # 1e-2` but `2.5e-13` and below from `scale = 1` **upward**, under
        # `approx`'s own default absolute tolerance of `1e-12`. Left at the
        # default, those three scales would accept a residue of exactly zero,
        # which is the fixed point this line exists to deny.
        assert taken[0] == pytest.approx(
            NORM_FLOOR**0.5 / (2 * norm) ** 2, rel=1e-9, abs=0.0
        )
        # `√NORM_FLOOR = 1e-12` is the floor's size in the norm itself, so the
        # residue is `2.5e-13` per unit of `‖agreed‖²`, and the four between
        # that and the bound is the `2` above spent as headroom.
        #
        # **This is a tripwire on the constant, not a second claim.** The line
        # above is `1/4` of this one, so it already implies it; what this adds
        # is that `1e-12` is written out where the other reads `NORM_FLOOR`, so
        # raising the floor by more than 4x turns this red while the equality
        # follows the constant and stays green. It is one-sided: no reduction
        # of the floor trips it.
        #
        # Note this is `NORM_FLOOR`, an epsilon guarding `0/0` in a square
        # root, and **not ADR-0007's disagreement floor** -- the irreducible
        # static, lag and settling parts of an edge's disagreement, which are
        # structural and at the scale of the configuration. The two are
        # unrelated quantities that share a word.
        assert taken[0] < 1e-12 / norm**2
        # One step off it, a full-size step: `1/(‖F x‖ + ‖y‖)` per
        # `NORM_FLOOR`'s own comment, so `1/(2‖agreed‖)` here.
        #
        # The envelope is derived. The gradient is `u/D − (N/D²)w` with `w` a
        # unit vector, so its norm lies in `[(1 − N/D)/D, (1 + N/D)/D]` times
        # `‖u‖`. The largest rung displaces by `0.2‖agreed‖`, putting `D` in
        # `[1.8, 2.2]·‖agreed‖` and `N/D` under `1/9`, for `[0.826, 1.235]` of
        # a full step; measured extremes over 20000 directions are `0.8265`
        # and `1.2338`, so the `0.8` below is just outside a tight bound.
        #
        # **`u` is only a unit vector while the displacement clears the
        # floor.** `u = δ/√(‖δ‖² + NORM_FLOOR)`, so it shrinks by
        # `‖δ‖/√(‖δ‖² + NORM_FLOOR)` and the smallest rung -- `‖δ‖ =
        # 2e-9·‖agreed‖` -- is the first to feel it. That factor is `0.9988` at
        # `scale = 1e-2` and `0.80` at `‖agreed‖ ≈ 6.7e-4`, where this
        # assertion breaks. **The sweep stops at `1e-2` for that reason and
        # cannot be extended downward without shortening the ladder**: covering
        # a smaller `agreed` and a `1e-9` relative rung at once is not
        # something the floor permits.
        full = 1.0 / (2.0 * norm)
        assert all(value > 0.8 * full for value in taken[1:])
        # Within a factor of two of each other across eight orders of
        # magnitude of disagreement, which is the whole of the point: there is
        # no basin here for an endpoint to settle into. The envelope caps the
        # ratio at `1.235/0.826 = 1.494`, so the two is headroom now rather
        # than a bound a small `agreed` could walk through.
        assert max(taken[1:]) < 2 * min(taken[1:])

    def test_the_normaliser_is_the_current_magnitudes_not_a_running_average(
        self, running
    ):
        # ADR-0007 rejects a per-edge baseline with a hand-set time constant,
        # and ADR-0010 says the normaliser is locally stateless for the same
        # reason. Stated as a property: the objective is a function of the
        # state in front of it, so a rule three steps into a run and a rule
        # built fresh report the same thing about the same state.
        rule = TransportRule(running)
        for _ in range(3):
            rule.step()
        fresh = TransportRule(running)
        assert torch.equal(rule.gradient(), fresh.gradient())


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
        rule = TransportRule(running)
        gathered, incoming = rule.inputs()
        parameters = rule.path.map_parameters()

        def taken(beliefs):
            return transport_gradient(
                parameters, rule.path, gathered, beliefs
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
        # already restricted onto the shared lane. A raw neighbour stalk
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
    #
    # **The disagreement is read against each endpoint's own scale since #157**,
    # and the reason is magnitude rather than looseness. This was a flat
    # elementwise `atol` of 1e-6 with `rtol=0`, and the Koopman conversion broke
    # it on CI while it passed locally -- an accumulation-order difference
    # between two BLAS builds, on a bound that was already sitting at 56% of its
    # own tolerance before the conversion and 95% after.
    #
    # Measured on this fixture, worst case over all endpoints, before and after
    # the conversion: absolute 5.588e-7 -> 9.537e-7, but *relative to the
    # endpoint's own largest entry* 1.77e-6 -> 6.40e-7 (float64: 1.25e-15 ->
    # 5.96e-16). So the identity holds **more** exactly after the conversion,
    # not less. What grew is the gradient itself -- endpoint 56's largest entry
    # goes 0.568 -> 1.489, because the converted body transmits more content
    # into the stalks these are computed from -- and an absolute-only bound
    # reads a proportionally smaller error as a failure once the numbers get
    # bigger. Stating the claim the way it was measured fixes that; widening the
    # absolute bound would only move the same cliff further out.
    @pytest.mark.parametrize(
        "dtype, tolerance", [(torch.float64, 1e-13), (torch.float32, 1e-5)]
    )
    def test_every_endpoint(self, running, dtype, tolerance):
        rule = TransportRule(running)
        parameters, arguments = in_precision(rule, dtype)
        batched = transport_gradient(parameters, *arguments)[MAPS_PARAMETER]
        for pair in range(running.maps.pairs):
            reference = batched[pair]
            slack = float((local_gradient(rule, pair, dtype) - reference).abs().max())
            scale = float(reference.abs().max())
            assert slack <= tolerance * scale, (
                f"endpoint {pair}: {slack:.3e} against a scale of {scale:.3e}"
            )

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
    """`m` is fixed at construction and the mask closes and never re-opens.
    Training moves weights inside the mask; it does not shrink a stalk and it
    removes no edge."""

    @pytest.fixture
    def trained(self, running):
        rule = TransportRule(running, learning_rate=0.2)
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
        # this state would take a silent null step -- no gradient, but every
        # map re-projected.
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
        # The formula above is satisfied by a zero gradient too, so this reads
        # that every map the mask leaves free actually moves.
        rule = TransportRule(running)
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
        rule = TransportRule(running)
        before = maps_of(running)
        rule.step()
        for pair in single_entry_pairs:
            assert torch.allclose(
                running.maps.maps.detach()[pair], before[pair], atol=1e-6
            )

    def test_the_rule_carries_nothing_per_cell_or_per_edge(self, running):
        # No momentum, no running average, no per-edge baseline: two rules
        # stepped from the same state agree, and a second step of one rule is
        # the plain gradient of the state it now sees.
        first = TransportRule(running, learning_rate=0.03)
        before = maps_of(running)
        first.step()
        gradient = first.step()
        after_two = maps_of(running)

        with torch.no_grad():
            running.maps.maps.copy_(before)
        second = TransportRule(running, learning_rate=0.03)
        second.step()
        assert torch.equal(second.step(), gradient)
        assert torch.equal(maps_of(running), after_two)

    def test_it_carries_nothing_between_steps(self, running):
        # Since #410 deleted the sparsity anneal, the rule holds no counter and
        # no per-edge array at all: what is left is the sheaf it is built
        # against, the one scalar, and the path. `steps` was the anneal
        # schedule's position and `permitted` was the mask's open-weight count
        # the deleted penalty divided by; nothing reads either now.
        rule = TransportRule(running)
        assert set(vars(rule)) == {"sheaf", "learning_rate", "path"}

    def test_the_maps_gain_no_buffer_of_their_own(self, running):
        # A per-edge auxiliary variable would have to live somewhere, and the
        # maps are the only per-edge object in the design. Every buffer here is
        # a constant of the built graph -- the structural mask, who owns which
        # endpoint, and what the projection holds each cell to (#220) -- so the
        # claim is tested twice: by name, and by the stronger property that a
        # step leaves all of them bit-identical. A rule that started
        # accumulating into one would fail the second even if it reused a name.
        before = {
            name: buffer.clone() for name, buffer in running.maps.named_buffers()
        }
        TransportRule(running).step()
        assert set(before) == {
            "support",
            "owner",
            "pinned",
            "holding",
            "holding_cells",
            "hold_rows",
            "hold_pairs",
            "column_mask",
            "overlap_target",
            # Which maps the spectral floor reaches, and their `(m, k)` groups
            # (ADR-0032). Constants of the built graph like the rest: the shapes
            # are read off the masks once and the membership never moves.
            "floored",
        } | {
            f"floor_group_{group}"
            for group in range(len(running.maps.floor_shapes))
        }
        for name, buffer in running.maps.named_buffers():
            assert torch.equal(buffer, before[name]), name
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
    rule = TransportRule(sheaf, learning_rate=0.1)
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
