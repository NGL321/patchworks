"""The tick's two phases, its delay, and its gain (ticket #86).

`docs/spec/02-tick-semantics.md` is what these hold down: two phases in order,
one Jacobi-style descent step per cell per tick, an edge that costs exactly one
tick, the reconciliation gain as the specified max under one global gamma, and
a tick that carries no tape.

The world's half of the ordering is `tests/test_agent.py`'s.
"""

import contextlib
import itertools
from unittest import mock

import pytest
import torch

from patchworks.body import CellBiases, CellBody
from patchworks.graph import DomeSpec, EdgeKind, build_graph
from patchworks.restriction import GAUGE_RHO, RestrictionMaps, pair_index
from patchworks.tick import (
    DEFAULT_GAMMA,
    Sheaf,
    assert_no_tape,
    reconciliation_gain,
)

# Small enough to iterate over cell by cell, built by the same rules as the
# real dome (see tests/test_restriction.py).
SMALL = DomeSpec(
    patch_grid=4,
    vision_sides=(2,),
    somatomotor_sizes=(4,),
    core_sizes=(4, 3),
    core_degree=4,
    apex_degree=3,
)


class Poisoned:
    """Stands in for a construction layout that must not be consulted."""

    def __getattr__(self, name):
        raise AssertionError(f"the construction layout was consulted at runtime ({name})")


@pytest.fixture
def dome():
    return build_graph(SMALL)


@pytest.fixture
def sheaf(dome):
    return Sheaf(dome, generator=torch.Generator().manual_seed(0))


def stir(sheaf, seed=7):
    """Put something in every stalk, chart and edge buffer, so a zero can't hide.

    A freshly built sheaf is all zeros, and half the properties below are
    trivially true of zero.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        sheaf.stalks[: sheaf.layout.total] = torch.randn(
            sheaf.layout.total, generator=generator
        )
        sheaf.charts.normal_(0.0, 1.0, generator=generator)
        sheaf.broadcast.normal_(0.0, 1.0, generator=generator)
        # A padded row of an edge buffer is never written by the real thing.
        for edge in sheaf.dome.edges:
            for side in (0, 1):
                sheaf.broadcast[pair_index(edge.id, side), edge.m :] = 0.0
    return sheaf


class TestTwoPhases:
    def test_a_tick_is_the_inference_phase_then_the_message_passing_phase(self, dome):
        # In that order, and with nothing else in it. Run the phases by hand on
        # one sheaf and the whole tick on its twin, and the two states agree.
        by_hand = stir(Sheaf(dome, generator=torch.Generator().manual_seed(0)))
        whole = stir(Sheaf(dome, generator=torch.Generator().manual_seed(0)))
        with torch.no_grad():
            by_hand.inference_phase()
            by_hand.message_passing_phase()
        whole.tick()
        assert torch.equal(by_hand.stalks, whole.stalks)
        assert torch.equal(by_hand.charts, whole.charts)
        assert torch.equal(by_hand.broadcast, whole.broadcast)

    def test_the_other_order_is_a_different_tick(self, dome):
        # Guards the assertion above against being vacuous.
        forwards = stir(Sheaf(dome, generator=torch.Generator().manual_seed(0)))
        backwards = stir(Sheaf(dome, generator=torch.Generator().manual_seed(0)))
        forwards.tick()
        with torch.no_grad():
            backwards.message_passing_phase()
            backwards.inference_phase()
        assert not torch.equal(forwards.stalks, backwards.stalks)

    def test_the_inference_phase_leaves_boundary_cells_alone(self, sheaf, dome):
        stir(sheaf)
        before = sheaf.stalks.clone()
        with torch.no_grad():
            sheaf.inference_phase()
        for cell_id in dome.boundary:
            where = sheaf.layout.slice(cell_id)
            assert torch.equal(sheaf.stalks[where], before[where])

    def test_the_prediction_becomes_the_node_stalk(self, sheaf, dome):
        stir(sheaf)
        with torch.no_grad():
            sheaf.inference_phase()
        for row, cell_id in enumerate(dome.predicting):
            assert torch.equal(sheaf.stalk(cell_id), sheaf.prediction[row])

    def test_the_message_passing_phase_never_touches_the_chart(self, sheaf):
        stir(sheaf)
        charts = sheaf.charts.clone()
        with torch.no_grad():
            sheaf.message_passing_phase()
        # Reconciliation edits the node stalk only. The correction reaches the
        # chart next tick, as evidence, through encode.
        assert torch.equal(sheaf.charts, charts)

    def test_the_message_passing_phase_does_edit_the_node_stalk(self, sheaf):
        stir(sheaf)
        before = sheaf.stalks.clone()
        with torch.no_grad():
            sheaf.message_passing_phase()
        assert not torch.equal(sheaf.stalks, before)


class TestUnitDelay:
    def test_a_cell_reconciles_against_the_previous_broadcast(self, sheaf, dome):
        stir(sheaf)
        sheaf.tick()
        previous = sheaf.broadcast.clone()
        sheaf.tick()
        for edge in dome.edges:
            u, v = pair_index(edge.id, 0), pair_index(edge.id, 1)
            # What u reconciled against this tick is what v broadcast last tick.
            assert torch.equal(sheaf.incoming[u], previous[v])
            assert torch.equal(sheaf.incoming[v], previous[u])

    def test_what_a_cell_reconciles_against_is_never_this_tick_s_value(self, sheaf):
        stir(sheaf)
        sheaf.tick()
        sheaf.tick()
        # Zero delay would make these the flip of each other within one tick.
        flipped = (
            sheaf.broadcast.reshape(-1, 2, sheaf.maps.edge_width)
            .flip(1)
            .reshape_as(sheaf.broadcast)
        )
        assert not torch.allclose(sheaf.incoming, flipped)

    def test_the_first_tick_reconciles_against_nothing(self, sheaf):
        # Nothing has been broadcast yet, so the buffers are empty rather than
        # seeded: a cell's first tick disagrees with silence.
        stir(sheaf)
        sheaf.broadcast.zero_()
        sheaf.tick()
        assert torch.all(sheaf.incoming == 0)

    def test_a_broadcast_is_this_tick_s_restriction_of_the_predicted_stalk(self, sheaf):
        stir(sheaf)
        with torch.no_grad():
            sheaf.inference_phase()
            predicted = sheaf.stalks.clone()
            sheaf.message_passing_phase()
        # Computed this same tick, from the stalk before reconciliation edited
        # it -- which is what makes the step Jacobi rather than Gauss-Seidel.
        expected = sheaf.maps.restrict(predicted[sheaf.layout.pair_positions])
        assert torch.allclose(sheaf.broadcast, expected, atol=1e-6)


class TestOneStepNotASolve:
    def test_the_update_is_exactly_one_local_descent_step(self, sheaf, dome):
        stir(sheaf)
        cell_id = dome.predicting[len(dome.predicting) // 2]
        before = sheaf.stalks.clone()
        previous = sheaf.broadcast.clone()
        with torch.no_grad():
            sheaf.message_passing_phase()

        where = sheaf.layout.slice(cell_id)
        x = before[where]
        gradient = torch.zeros_like(x)
        for edge_id in dome.incident[cell_id]:
            edge = dome.edges[edge_id]
            side = 0 if edge.u == cell_id else 1
            pair = pair_index(edge_id, side)
            f = sheaf.maps.maps[pair, : edge.m, : x.numel()].detach()
            neighbour = previous[pair ^ 1, : edge.m]
            gradient += f.T @ (f @ x - neighbour)
        expected = x - sheaf.gain[cell_id] * gradient
        assert torch.allclose(sheaf.stalks[where], expected, atol=1e-6)

    def test_one_step_is_a_descent_that_never_arrives(self, sheaf, dome):
        # Agreement is penalised, not enforced. The step goes down the cell's
        # own local disagreement energy, and one step of it does not reach the
        # bottom -- a solve would have, in the same phase.
        stir(sheaf)
        cell_id = dome.predicting[len(dome.predicting) // 2]
        beliefs = sheaf.broadcast.clone()

        def energy(stalks):
            total = 0.0
            for edge_id in dome.incident[cell_id]:
                edge = dome.edges[edge_id]
                side = 0 if edge.u == cell_id else 1
                pair = pair_index(edge_id, side)
                x = stalks[sheaf.layout.slice(cell_id)]
                f = sheaf.maps.maps[pair, : edge.m, : x.numel()].detach()
                total += 0.5 * float(((f @ x - beliefs[pair ^ 1, : edge.m]) ** 2).sum())
            return total

        before = energy(sheaf.stalks.clone())
        with torch.no_grad():
            sheaf.message_passing_phase()
        after = energy(sheaf.stalks)
        assert 0 < after < before

    def test_every_cell_reads_the_same_prior_round(self, dome):
        # Jacobi, not Gauss-Seidel: no cell's update can see another's, so
        # there is no visiting order to define. Recomputing every cell's step
        # from the pre-phase configuration alone reproduces the phase exactly,
        # which it could not if any cell had seen a neighbour's update.
        forwards = stir(Sheaf(dome, generator=torch.Generator().manual_seed(0)))
        with torch.no_grad():
            forwards.message_passing_phase()
        by_pair = stir(Sheaf(dome, generator=torch.Generator().manual_seed(0)))
        before = by_pair.stalks.clone()
        previous = by_pair.broadcast.clone()
        with torch.no_grad():
            by_pair.message_passing_phase()
        # Recompute every cell's step from the pre-phase configuration alone.
        expected = before.clone()
        for edge in dome.edges:
            for side, cell_id in enumerate((edge.u, edge.v)):
                pair = pair_index(edge.id, side)
                where = by_pair.layout.slice(cell_id)
                f = by_pair.maps.maps[pair, : edge.m, : before[where].numel()].detach()
                residual = f @ before[where] - previous[pair ^ 1, : edge.m]
                expected[where] -= by_pair.gain[cell_id] * (f.T @ residual)
        assert torch.allclose(forwards.stalks, expected, atol=1e-5)


class TestReconciliationGain:
    def test_the_formula(self, dome):
        gain = reconciliation_gain(dome)
        for cell in dome.cells:
            bound = max(
                dome.stalk_sums[cell.id], GAUGE_RHO**2 * dome.degrees[cell.id]
            )
            assert float(gain[cell.id]) == pytest.approx(DEFAULT_GAMMA / bound)

    def test_one_global_gamma_and_nothing_else_per_cell(self, dome):
        # The only per-cell quantity in the gain is a denominator read straight
        # off the built graph. Multiply it back out and every cell returns the
        # same number: there is nowhere for a per-cell knob to hide.
        gain = reconciliation_gain(dome)
        bounds = torch.maximum(
            torch.tensor(dome.stalk_sums, dtype=torch.float32),
            GAUGE_RHO**2 * torch.tensor(dome.degrees, dtype=torch.float32),
        )
        assert torch.allclose(gain * bounds, torch.full_like(gain, DEFAULT_GAMMA))

    def test_the_gain_is_not_graded_by_depth(self):
        # A gain deliberately graded by depth would be the per-cell clock
        # divisor ADR-0005 rejected, wearing a different name. Two cells at
        # different levels with the same degree and the same incident mask
        # width take the same step, and the test is written over the levels so
        # that it fails if depth ever enters. On the real dome, where the taper
        # is deep enough for a structural class to span two of them.
        dome = build_graph()
        gain = reconciliation_gain(dome)
        by_structure: dict[tuple[int, int], set[float]] = {}
        levels: dict[tuple[int, int], set[int]] = {}
        for cell in dome.cells:
            key = (dome.degrees[cell.id], dome.stalk_sums[cell.id])
            by_structure.setdefault(key, set()).add(round(float(gain[cell.id]), 9))
            levels.setdefault(key, set()).add(cell.index.level)
        spanning = [key for key, seen in levels.items() if len(seen) > 1]
        assert spanning, "no structural class spans two levels; the test proves nothing"
        for key in spanning:
            assert len(by_structure[key]) == 1

    def test_the_max_is_what_stops_a_larger_rho_loosening_the_bound(self, dome):
        # Written as the max so that a later change to rho cannot silently
        # loosen it below the eigenvalue it is bounding.
        wide = reconciliation_gain(dome, rho=4.0)
        assert torch.all(wide <= reconciliation_gain(dome) + 1e-9)
        assert not torch.equal(wide, reconciliation_gain(dome))

    @pytest.mark.parametrize("gamma", [0.0, -0.5, 1.5])
    def test_gamma_outside_the_bound_is_refused(self, dome, gamma):
        with pytest.raises(ValueError, match="gamma"):
            reconciliation_gain(dome, gamma=gamma)

    def test_gamma_scales_every_cell_together(self, dome):
        assert torch.allclose(
            reconciliation_gain(dome, gamma=0.5), reconciliation_gain(dome) * 0.5
        )

    def test_the_drive_edges_make_the_apex_slacker_rather_than_tighter(self, dome):
        # An extra incident edge lowers gain_v. The spec puts it at about 6% at
        # the real sizes and declines to lean on it; what is asserted here is
        # only the direction, which is what would flip if the drive were ever
        # given the gain to make itself heard with.
        gain = reconciliation_gain(dome)
        apex_level = max(c.index.level for c in dome.cells if not c.is_boundary)
        driven = [
            c.id for c in dome.cells if not c.is_boundary and c.index.level == apex_level
        ]
        assert driven
        for cell_id in driven:
            drive_edges = [
                dome.edges[e]
                for e in dome.incident[cell_id]
                if dome.edges[e].kind is EdgeKind.DRIVE
            ]
            assert drive_edges
            without = max(
                dome.stalk_sums[cell_id] - sum(e.m for e in drive_edges),
                GAUGE_RHO**2 * (dome.degrees[cell_id] - len(drive_edges)),
            )
            assert float(gain[cell_id]) < DEFAULT_GAMMA / without


class TestPrivateFeatures:
    def test_reconciliation_never_moves_a_private_feature(self, sheaf, dome):
        # A direction masked out on every incident edge participates on no edge,
        # so it lies in H^0 by construction. It comes back from spread as
        # exactly zero rather than as something small.
        stir(sheaf)
        before = sheaf.stalks.clone()
        with torch.no_grad():
            sheaf.message_passing_phase()
        private = dome.private_mask
        for row, cell_id in enumerate(dome.predicting):
            where = sheaf.layout.slice(cell_id)
            moved = (sheaf.stalks[where] != before[where])
            assert not torch.any(moved & private[row])

    def test_some_feature_does_move(self, sheaf, dome):
        stir(sheaf)
        before = sheaf.stalks.clone()
        with torch.no_grad():
            sheaf.message_passing_phase()
        assert torch.any(sheaf.stalks != before)


class TestTheTickCarriesNoTape:
    def test_nothing_leaving_the_tick_carries_a_grad_fn(self, sheaf):
        stir(sheaf)
        sheaf.tick()
        for tensor in (
            sheaf.stalks,
            sheaf.charts,
            sheaf.prediction,
            sheaf.broadcast,
            sheaf.incoming,
        ):
            assert tensor.grad_fn is None and not tensor.requires_grad

    def test_the_assertion_catches_the_tick_losing_its_no_grad(self, sheaf):
        # The failure this exists for is silent and flattering: a leaked
        # gradient does not crash, it makes the agent work better. So the check
        # is exercised by actually removing the guard, not by hand-building a
        # tensor the tick could never produce.
        stir(sheaf)
        with mock.patch("torch.no_grad", contextlib.nullcontext):
            with pytest.raises(AssertionError, match="autograd tape"):
                sheaf.tick()

    def test_the_leak_the_assertion_would_have_missed_is_a_real_one(self, sheaf):
        # Same removal, checked from the other side: with the guard gone the
        # state really does end up on a tape, so the test above is not passing
        # for some unrelated reason.
        stir(sheaf)
        with mock.patch("torch.no_grad", contextlib.nullcontext):
            with contextlib.suppress(AssertionError):
                sheaf.tick()
        assert sheaf.stalks.grad_fn is not None

    @pytest.mark.parametrize("phase", ["inference_phase", "message_passing_phase"])
    def test_a_phase_run_on_its_own_guards_itself(self, dome, phase):
        # The phases are public and #90 and the learning phase both have reason
        # to run one alone. A phase whose no_grad lived in its caller would leak
        # silently the first time one did.
        #
        # A fresh sheaf per phase, not one sheaf run twice: `stir` writes in
        # place and an in-place write does not clear an existing `grad_fn`, so a
        # sheaf the first phase had already tainted would raise on the leftover
        # tape and the second half would pass with no guard at all.
        sheaf = stir(Sheaf(dome, generator=torch.Generator().manual_seed(0)))
        with mock.patch("torch.no_grad", contextlib.nullcontext):
            with pytest.raises(AssertionError, match="autograd tape"):
                getattr(sheaf, phase)()

    @pytest.mark.parametrize("phase", ["inference_phase", "message_passing_phase"])
    def test_the_phase_is_what_taped_it(self, dome, phase):
        # Guards the test above against passing on someone else's leak: run
        # unguarded from a clean sheaf, each phase puts the state on a tape by
        # itself.
        sheaf = stir(Sheaf(dome, generator=torch.Generator().manual_seed(0)))
        with mock.patch("torch.no_grad", contextlib.nullcontext):
            with contextlib.suppress(AssertionError):
                getattr(sheaf, phase)()
        assert sheaf.stalks.grad_fn is not None

    def test_a_phase_run_on_its_own_leaves_no_tape(self, sheaf):
        stir(sheaf)
        sheaf.inference_phase()
        sheaf.message_passing_phase()
        assert sheaf.stalks.grad_fn is None and not sheaf.stalks.requires_grad

    def test_the_assertion_names_the_offender(self):
        with pytest.raises(AssertionError, match="charts"):
            assert_no_tape(charts=torch.zeros(2, requires_grad=True))

    def test_a_leaf_that_merely_requires_grad_is_refused_too(self):
        # It carries no grad_fn yet; the next operation would put it on a tape.
        leaf = torch.zeros(3, requires_grad=True)
        assert leaf.grad_fn is None
        with pytest.raises(AssertionError, match="requires_grad=True"):
            assert_no_tape(stalks=leaf)

    def test_the_adapting_surface_is_still_trainable(self, sheaf):
        # The guard is about what leaves the tick, not about freezing the
        # surface: the biases and the maps are what the learning phase descends
        # on, and they keep their gradients.
        sheaf.tick()
        assert all(p.requires_grad for p in sheaf.biases.parameters())
        assert sheaf.maps.maps.requires_grad


class TestTheStalkBuffer:
    def test_every_cell_has_its_own_stalk_of_the_world_s_width(self, sheaf, dome):
        for cell in dome.cells:
            assert sheaf.stalk(cell.id).numel() == cell.stalk

    def test_the_pad_slot_stays_zero(self, sheaf):
        # Padded gathers read it and padded scatters land in it, so a tick that
        # let it drift would quietly feed junk into every ragged map.
        stir(sheaf)
        for _ in range(3):
            sheaf.tick()
        assert sheaf.stalks[sheaf.layout.pad] == 0

    def test_the_tick_reads_no_construction_index(self, sheaf):
        # Same commitment as the dome's runtime surface: the construction
        # layout generates the indices once, at construction, and has no
        # runtime role after that. Poison every cell's index and tick anyway.
        indices = [cell.index for cell in sheaf.dome.cells]
        stir(sheaf)
        try:
            for cell in sheaf.dome.cells:
                object.__setattr__(cell, "index", Poisoned())
            sheaf.tick()
        finally:
            for cell, index in zip(sheaf.dome.cells, indices):
                object.__setattr__(cell, "index", index)


class TestTheRealDome:
    def test_a_hundred_ticks_of_the_real_graph_stay_finite(self):
        dome = build_graph()
        sheaf = Sheaf(dome, generator=torch.Generator().manual_seed(0))
        for _ in range(100):
            sheaf.tick()
        assert torch.isfinite(sheaf.stalks).all()
        assert torch.isfinite(sheaf.charts).all()
        assert sheaf.ticks == 100

    def test_the_gain_at_the_real_sizes(self):
        dome = build_graph()
        gain = reconciliation_gain(dome)
        # At rho = 2 and the vertical edges' m = 4 the two terms of the max are
        # equal for an interior cell of degree 8; the apex, whose incident mask
        # width falls with depth, takes the larger step.
        apex_level = max(c.index.level for c in dome.cells if not c.is_boundary)
        apex = [
            c.id
            for c in dome.cells
            if c.index.level == apex_level and not c.is_boundary
        ]
        rim = [
            c.id
            for c in dome.cells
            if not c.is_boundary and c.index.level == 1
        ]
        assert min(float(gain[i]) for i in apex) > max(float(gain[i]) for i in rim)


class TestASurfaceBuiltForAnotherGraph:
    """The layout indexes one flat buffer by cell and by edge endpoint.

    A surface built for a different dome would read the wrong components rather
    than fail, so it is refused at construction.
    """

    def test_maps_from_another_dome_are_refused(self, dome):
        other = RestrictionMaps(build_graph(), generator=torch.Generator().manual_seed(0))
        with pytest.raises(ValueError, match="different dome"):
            Sheaf(dome, maps=other)

    def test_biases_for_the_wrong_population_are_refused(self, dome):
        with pytest.raises(ValueError, match="predicting cells"):
            Sheaf(dome, biases=CellBiases(dome.shape, len(dome.predicting) + 1))


def _seeded() -> torch.Generator:
    """A generator of this module's own, so building a stand-in draws nothing
    from the global stream.

    What is *in* a ready-drawn piece is irrelevant to every test below -- they
    ask where the piece ends up, not what it holds -- so drawing it unseeded
    would spend the global RNG purely as a side effect, and shift the draw
    every later test in the suite makes. Cheap to avoid, and it keeps these
    tests from depending on what ran before them.
    """
    return torch.Generator().manual_seed(0)


#: How to supply each piece ready-drawn, and where to read the draw the
#: generator would otherwise have made. Keyed alike, so one set of names is
#: both the call and the list of pieces left over for the generator.
SUPPLY = {
    "body": lambda dome: CellBody(dome.shape, generator=_seeded()),
    "biases": lambda dome: CellBiases(
        dome.shape, len(dome.predicting), generator=_seeded()
    ),
    "maps": lambda dome: RestrictionMaps(dome, generator=_seeded()),
}
DRAWN = {
    "body": lambda sheaf: sheaf.body.encode_hidden_weight,
    "biases": lambda sheaf: sheaf.biases.encode_hidden_bias,
    "maps": lambda sheaf: sheaf.maps.maps,
}
#: Every way of supplying some but not all of the pieces, the empty call
#: included: the branches on which the generator is still doing real work.
#: Derived from `SUPPLY` rather than written out, so a fourth ready-drawn piece
#: would widen the cover instead of quietly leaving it behind.
PARTIAL = [
    supplied
    for size in range(len(SUPPLY))
    for supplied in itertools.combinations(SUPPLY, size)
]


class TestAnInertGenerator:
    """`generator` seeds what it was not handed, so all three handed in is a lie (#108).

    #106 closed the same shape one level up: an argument accepted, ignored, and
    silent about it. The rule it recorded — *nothing consumes a construction
    argument once the thing it constructs is supplied* — does not read verbatim
    here, because the generator feeds three independent draws rather than one
    object. The condition is *nothing left to draw*, which is why every partial
    call below is not an error but a generator doing real work.
    """

    def test_body_biases_and_maps_together_refuse_the_generator(self, dome):
        # Anchored on the leading token, because the refusal has to name the
        # argument it was handed rather than merely mention it in its advice.
        with pytest.raises(ValueError, match=r"^generator seeds") as refusal:
            Sheaf(
                dome,
                **{name: supply(dome) for name, supply in SUPPLY.items()},
                generator=torch.Generator().manual_seed(0),
            )
        # Where it would have been consumed, named -- all three of them, since
        # all three are what a caller has to give up to keep the generator.
        # Read off the clause before the colon, not the whole message: the
        # advice after it spells out a call naming all three anyway, so
        # searching the message entire would pass on the advice alone and the
        # enumeration could be deleted without a test noticing.
        named, _, advice = str(refusal.value).partition(":")
        assert advice
        for piece in SUPPLY:
            assert piece in named

    def test_a_surface_built_for_another_graph_is_still_the_first_thing_said(
        self, dome
    ):
        # Both mistakes at once. The mismatched dome is the one that costs
        # something -- it would read the wrong components rather than fail --
        # so it is what the caller hears, rather than hearing about the
        # generator now and the dome on a second run.
        with pytest.raises(ValueError, match="different dome"):
            Sheaf(
                dome,
                body=SUPPLY["body"](dome),
                biases=SUPPLY["biases"](dome),
                maps=RestrictionMaps(build_graph(), generator=_seeded()),
                generator=torch.Generator().manual_seed(0),
            )

    def test_biases_for_the_wrong_population_are_also_said_first(self, dome):
        # The other refusal the generator has to stay behind. Same reasoning:
        # biases sized against another population are a real mistake with a
        # real cost, and an inert generator is a wasted argument -- so the
        # costly one is what a caller who made both hears about. Pinned
        # separately from the dome case above, because the two checks sit at
        # different points and moving the generator's check up past only this
        # one would leave the dome test green.
        with pytest.raises(ValueError, match="predicting cells"):
            Sheaf(
                dome,
                body=SUPPLY["body"](dome),
                biases=CellBiases(
                    dome.shape, len(dome.predicting) + 1, generator=_seeded()
                ),
                maps=SUPPLY["maps"](dome),
                generator=torch.Generator().manual_seed(0),
            )

    def test_all_three_without_a_generator_is_the_ordinary_prepared_call(self, dome):
        # Nothing is drawn and nothing was asked to be, so there is nothing to
        # refuse: handing over a fully prepared surface stays legal.
        pieces = {name: supply(dome) for name, supply in SUPPLY.items()}
        built = Sheaf(dome, **pieces)
        for name, piece in pieces.items():
            assert getattr(built, name) is piece

    @pytest.mark.parametrize(
        "supplied", PARTIAL, ids=lambda names: "+".join(names) or "none"
    )
    def test_every_partial_call_still_seeds_what_it_draws(self, dome, supplied):
        # The surviving branches, each held down by the only thing that shows a
        # generator was consumed at all: the same seed draws the same numbers
        # and a different seed does not.
        def build(seed):
            return Sheaf(
                dome,
                **{name: SUPPLY[name](dome) for name in supplied},
                generator=torch.Generator().manual_seed(seed),
            )

        one, again, other = build(7), build(7), build(8)
        # Walked over `SUPPLY` rather than `DRAWN`, so that a fourth piece
        # added to the one and not the other is a `KeyError` here. Iterating
        # `DRAWN` would instead widen the parametrisation to cover the new
        # piece and quietly assert nothing about it.
        for name in SUPPLY:
            if name in supplied:
                continue
            read = DRAWN[name]
            assert torch.equal(read(one), read(again))
            assert not torch.equal(read(one), read(other))
