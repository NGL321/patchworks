"""The diagnostics that run on a cadence (ticket #91).

What these hold down is `docs/spec/01-cell-and-sheaf.md`, *Known exposure*
(over-smoothing) and *`H⁰` is the private features*, `docs/spec/06-graph-topology.md`,
*What the cycles do*, and ADR-0010's paired instrument.

Three of them are the ones that matter, and all three are written as real
assertions rather than smoke checks. **The pair is unobtainable by halves** —
there is no route through the module's public surface to a per-edge energy
without its per-edge effective rank, because one reading cannot tell collapse
from a draining lag floor. **No rank floor is imposed** — a fleet driven to
rank 1 reads 1 and nothing clamps, warns or refuses. And **the whole-graph
reading is arithmetically checked**, against `χ` on one side and a least-squares
solve on the other, because a dimension that is merely plausible is a dimension
nobody can act on.

Everything runs on the small dome the tick's and the learning rules' tests run
on. The full dome's whole-graph reading is one `3764 × 3764` eigendecomposition
and belongs to a run, not to a suite.
"""

import ast
from dataclasses import replace
import pathlib

import pytest
import torch

from patchworks.diagnostics import (
    BASELINE_SEED,
    DEFAULT_EVERY,
    DEFAULT_WHOLE_GRAPH_EVERY,
    Condition,
    Diagnostics,
    EdgeReading,
    Reading,
    WholeGraphReading,
    TOPOLOGY_ENERGY_DRAWS,
    topology_only_energy,
    topology_only_h1,
)
from patchworks.graph import build_graph
from patchworks.restriction import pair_index
from patchworks.tick import Sheaf

from conftest import SMALL


@pytest.fixture
def dome():
    return build_graph(SMALL)


@pytest.fixture
def sheaf(dome):
    built = Sheaf(dome, generator=torch.Generator().manual_seed(0))
    # A fresh sheaf holds all-zero node stalks, which is a real configuration
    # and a useless one to measure: every edge agrees exactly, so every energy
    # is zero and the boundary conditions assert nothing. The world's write is
    # what puts content in a boundary stalk, and this stands in for it without
    # dragging MuJoCo into a test of linear algebra. The pad slot stays zero,
    # because a padded gather reads it and a padded scatter lands in it.
    torch.manual_seed(1)
    built.stalks = torch.randn_like(built.stalks)
    built.stalks[built.layout.pad] = 0.0
    return built


@pytest.fixture
def wide_dome():
    """`SMALL` at the lane widths it carried until #474.

    `minimum_energy` is only ever positive while the coboundary is row-rank
    deficient, and at `(interior_m, boundary_m) = (3, 4)` this spec's is onto --
    see `test_the_floor_is_zero_on_this_dome_because_the_coboundary_is_onto`. The
    tests that need the floor to *move* need a dome where it is not identically
    zero, so they take this one rather than asserting nothing.
    """
    return build_graph(replace(SMALL, interior_m=4, boundary_m=8))


@pytest.fixture
def wide_sheaf(wide_dome):
    built = Sheaf(wide_dome, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(1)
    built.stalks = torch.randn_like(built.stalks)
    built.stalks[built.layout.pad] = 0.0
    return built


@pytest.fixture
def diagnostics(sheaf):
    return Diagnostics(sheaf, every=1, whole_graph_every=1)


def assembled_coboundary(sheaf):
    """`δ_P` and `b`, assembled here by a different route than the module's.

    The predicting cells' columns of the coboundary, and the boundary cells'
    fixed contribution, built cell by cell out of the map tensor rather than by
    the module's precomputed block table -- so that a test of the whole-graph
    reading is a second opinion rather than the same arithmetic run twice.
    """
    dome = sheaf.dome
    n = dome.shape.n
    row = {cell_id: i for i, cell_id in enumerate(dome.predicting)}
    rows = sum(edge.m for edge in dome.edges)
    delta = torch.zeros(rows, len(dome.predicting) * n, dtype=torch.float64)
    b = torch.zeros(rows, dtype=torch.float64)
    at = 0
    for edge in dome.edges:
        for side, cell_id in enumerate((edge.u, edge.v)):
            sign = 1.0 if side == 0 else -1.0
            block = sheaf.maps.maps[pair_index(edge.id, side)].detach().double()
            if cell_id in row:
                column = row[cell_id] * n
                delta[at : at + edge.m, column : column + n] += (
                    sign * block[: edge.m, :n]
                )
            else:
                stalk = sheaf.stalks[sheaf.layout.slice(cell_id)].double()
                b[at : at + edge.m] += sign * (block[: edge.m, : stalk.numel()] @ stalk)
        at += edge.m
    return delta, b


class TestThePairIsARealPair:
    """Energy and effective rank are recorded together, never separately."""

    def test_one_call_returns_both(self, diagnostics, dome):
        reading = diagnostics.edge_reading()
        assert isinstance(reading, EdgeReading)
        assert reading.energy.shape == (len(dome.edges),)
        assert reading.effective_rank.shape == (len(dome.edges), 2)

    def test_a_reading_always_carries_the_pair(self, diagnostics):
        reading = diagnostics.read(Condition.DRIVEN)
        assert isinstance(reading.edges, EdgeReading)
        assert len(reading.edges) == len(diagnostics.sheaf.dome.edges)

    def test_half_a_pair_cannot_be_constructed(self):
        with pytest.raises(TypeError):
            EdgeReading(energy=torch.zeros(3))
        with pytest.raises(TypeError):
            EdgeReading(effective_rank=torch.zeros(3, 2))

    def test_two_halves_of_different_edges_are_refused(self):
        with pytest.raises(ValueError, match="same edges"):
            EdgeReading(energy=torch.zeros(3), effective_rank=torch.zeros(4, 2))

    def test_an_effective_rank_without_two_ends_is_refused(self):
        with pytest.raises(ValueError, match="same edges"):
            EdgeReading(energy=torch.zeros(3), effective_rank=torch.zeros(3))

    def test_a_per_edge_energy_that_is_not_per_edge_is_refused(self):
        with pytest.raises(ValueError, match="one value per edge"):
            EdgeReading(energy=torch.zeros(3, 2), effective_rank=torch.zeros(3, 2))

    def test_neither_half_may_arrive_on_a_tape(self):
        live = torch.zeros(3, requires_grad=True)
        with pytest.raises(ValueError, match="autograd tape"):
            EdgeReading(energy=live, effective_rank=torch.zeros(3, 2))
        with pytest.raises(ValueError, match="autograd tape"):
            EdgeReading(
                energy=torch.zeros(3), effective_rank=torch.zeros(3, 2) * live[0]
            )

    def test_nothing_public_hands_out_one_half(self):
        """The prohibition, read off the module's own syntax tree.

        The pairing is meant to be enforced in the API rather than in the
        docstrings, and this is what that means concretely: every public
        callable in the module is checked for a `return` of a bare energy or a
        bare effective rank. The two quantities are computed inside
        `edge_reading`, which returns an `EdgeReading`; anything else that
        returned one of them would be the loophole this ticket exists to close.
        """
        source = (
            pathlib.Path(__file__).resolve().parent.parent
            / "src"
            / "patchworks"
            / "diagnostics.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        public = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        ]
        assert {node.name for node in public} == {
            "topology_only_h1",
            # #363's reference level. It returns a per-edge *energy* and never a
            # bare `energy`, so the loophole check below still binds it; the
            # name is listed here because this set is a deliberate gate on the
            # module's public surface and a new one has to be admitted, not
            # arrive.
            "topology_only_energy",
            "edge_reading",
            "whole_graph",
            "read",
            "observe",
            "report",
        }
        halves = {"energy", "effective_rank", "effective", "gram"}
        for node in public:
            for statement in ast.walk(node):
                if isinstance(statement, ast.Return) and isinstance(
                    statement.value, ast.Name
                ):
                    assert statement.value.id not in halves, (
                        f"{node.name} returns a bare {statement.value.id}"
                    )


class TestEffectiveRank:
    """The participation ratio, and the degenerate limit it is allowed to reach."""

    def test_it_is_the_participation_ratio_of_the_singular_values(
        self, diagnostics, dome
    ):
        reading = diagnostics.edge_reading()
        maps = diagnostics.sheaf.maps
        for edge in dome.edges:
            for side in (0, 1):
                singular = torch.linalg.svdvals(maps.maps[pair_index(edge.id, side)])
                want = singular.pow(2).sum() ** 2 / singular.pow(4).sum()
                assert float(reading.effective_rank[edge.id, side]) == pytest.approx(
                    float(want), rel=1e-4
                )

    def test_a_rank_one_map_reads_one(self, sheaf):
        with torch.no_grad():
            # The first column row 0 of each map is permitted to touch -- the
            # mask is what makes a rank-1 map a legal one rather than a shape
            # the support forbids.
            column = sheaf.maps.support[:, 0].float().argmax(dim=-1)
            sheaf.maps.maps.zero_()
            for pair in range(sheaf.maps.pairs):
                sheaf.maps.maps[pair, 0, int(column[pair])] = 1.0
        reading = Diagnostics(sheaf).edge_reading()
        assert torch.allclose(
            reading.effective_rank, torch.ones_like(reading.effective_rank), atol=1e-5
        )

    def test_a_uniform_map_reads_its_own_width(self, sheaf, dome):
        """`m` for a map transmitting every direction equally (ADR-0010)."""
        with torch.no_grad():
            sheaf.maps.maps.zero_()
            for edge in dome.edges:
                for side in (0, 1):
                    pair = pair_index(edge.id, side)
                    sheaf.maps.maps[pair, : edge.m, : edge.m] = torch.eye(edge.m)
        reading = Diagnostics(sheaf).edge_reading()
        widths = torch.tensor([float(edge.m) for edge in dome.edges])
        assert torch.allclose(
            reading.effective_rank, widths.unsqueeze(-1).expand(-1, 2), atol=1e-4
        )

    def test_the_degenerate_limit_is_reported_and_not_prevented(self, sheaf):
        """No rank floor anywhere: a whole fleet at rank 1 is a reading, not an error."""
        with torch.no_grad():
            sheaf.maps.maps.zero_()
            sheaf.maps.maps[:, 0, 0] = 1.0
            sheaf.maps.maps.mul_(sheaf.maps.support)
        diagnostics = Diagnostics(sheaf)
        reading = diagnostics.edge_reading()
        collapsed = reading.effective_rank[reading.effective_rank > 0]
        assert collapsed.numel()
        assert float(collapsed.max()) == pytest.approx(1.0, abs=1e-5)
        # And the projection *does* restore rank, which is ADR-0032 and is the
        # one thing this assertion used to say it did not. The diagnostic still
        # does not prevent the degenerate limit -- it reported it above, on the
        # unprojected surface -- but the spectral floor is a floor, and lifting
        # a dead direction is exactly what distinguishes it from the incoherence
        # cap, which can only ever rescale a survivor. Every map the floor
        # reaches comes back at its full `m`.
        sheaf.maps.project()
        lifted = diagnostics.edge_reading().effective_rank
        # `effective_rank` is indexed `[edge, side]` and `floored` by pair, and
        # `pair_index` is `2 * edge.id + side`, so the reshape is the identity
        # on the ordering rather than a rearrangement.
        floored = sheaf.maps.floored.reshape(-1, 2)
        widths = torch.tensor(
            [[float(edge.m)] * 2 for edge in sheaf.dome.edges]
        )
        assert torch.allclose(lifted[floored], widths[floored], atol=1e-4)

    @pytest.mark.parametrize("scale", [1e-20, 1e-8, 1e-2, 1e2, 1e8, 1e20])
    def test_it_is_scale_invariant_across_any_magnitude(self, dome, scale):
        """The participation ratio does not depend on `‖F‖_F`, and must not.

        `(Σσᵢ²)² / Σσᵢ⁴` is exactly invariant under `F → cF`, so an instrument
        that reads a different rank for a rescaled fleet is reading the norm.
        **ADR-0010's line is that a map's norm is not a diagnostic**, and the
        pair is meant to be two quantities that still move inside the gauge.

        The failure this pins is not a rounding error. An absolute epsilon in
        the denominator competes with `‖F Fᵀ‖_F²` once the norm is small, so a
        genuinely rank-4 fleet at `‖F‖_F = 1e-8` reads ~0.01 -- *total
        collapse*, in the one case the instrument is pointed at an unprojected
        map tensor. The pair would then say "collapse" about a graph that is
        fine, which is precisely the false reading the pairing exists to stop.
        """
        reference = Diagnostics(
            Sheaf(dome, generator=torch.Generator().manual_seed(0))
        ).edge_reading().effective_rank
        rescaled = Sheaf(dome, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            rescaled.maps.maps.mul_(scale)
        reading = Diagnostics(rescaled).edge_reading().effective_rank
        assert torch.allclose(reading, reference, rtol=1e-5, atol=1e-5)
        assert torch.isfinite(reading).all()

    def test_a_map_transmitting_nothing_reads_zero_rather_than_nan(self, sheaf):
        with torch.no_grad():
            sheaf.maps.maps.zero_()
        reading = Diagnostics(sheaf).edge_reading()
        assert torch.equal(
            reading.effective_rank, torch.zeros_like(reading.effective_rank)
        )


class TestPerEdgeEnergy:
    def test_it_is_the_squared_disagreement_on_that_edge(self, diagnostics, dome):
        sheaf = diagnostics.sheaf
        reading = diagnostics.edge_reading()
        for edge in dome.edges:
            ends = []
            for side, cell_id in enumerate((edge.u, edge.v)):
                block = sheaf.maps.maps[pair_index(edge.id, side)]
                stalk = sheaf.stalks[sheaf.layout.slice(cell_id)]
                ends.append(block[: edge.m, : stalk.numel()] @ stalk)
            want = (ends[0] - ends[1]).pow(2).sum()
            assert float(reading.energy[edge.id]) == pytest.approx(
                float(want), rel=1e-5
            )

    def test_the_sum_over_edges_is_the_dirichlet_energy(self, diagnostics):
        """`Σ_e ‖δ_e x‖² = xᵀLx`, checked against the assembled coboundary."""
        sheaf = diagnostics.sheaf
        delta, b = assembled_coboundary(sheaf)
        free = sheaf.stalks[sheaf.layout.predicting_positions].reshape(-1).double()
        want = float((delta @ free + b).pow(2).sum())
        assert float(diagnostics.edge_reading().energy.sum()) == pytest.approx(
            want, rel=1e-4
        )

    def test_energy_is_of_the_configuration_now_not_of_the_last_broadcast(
        self, dome
    ):
        """The reading follows the node stalks; the tick's own record does not."""
        sheaf = Sheaf(dome, generator=torch.Generator().manual_seed(2))
        sheaf.tick()
        diagnostics = Diagnostics(sheaf)
        before = diagnostics.edge_reading().energy
        with torch.no_grad():
            sheaf.stalks.add_(1.0)
            sheaf.stalks[sheaf.layout.pad] = 0.0
        after = diagnostics.edge_reading().energy
        assert not torch.allclose(before, after)
        assert torch.equal(sheaf.disagreement(), sheaf.disagreement())


class TestTheConditionIsInTheRecord:
    def test_a_reading_carries_the_condition_it_was_taken_under(self, diagnostics):
        assert diagnostics.read(Condition.QUIESCENT).condition is Condition.QUIESCENT
        assert diagnostics.read(Condition.DRIVEN).condition is Condition.DRIVEN

    def test_the_two_conditions_are_told_apart_in_the_record(self, diagnostics):
        sheaf = diagnostics.sheaf
        for tick in range(1, 5):
            sheaf.ticks = tick
            diagnostics.observe(
                Condition.QUIESCENT if tick % 2 else Condition.DRIVEN
            )
        quiescent = [r for r in diagnostics.readings if r.condition is Condition.QUIESCENT]
        driven = [r for r in diagnostics.readings if r.condition is Condition.DRIVEN]
        assert [r.tick for r in quiescent] == [1, 3]
        assert [r.tick for r in driven] == [2, 4]

    def test_a_reading_cannot_be_taken_without_declaring_one(self, diagnostics):
        with pytest.raises(TypeError):
            diagnostics.read()
        with pytest.raises(TypeError):
            diagnostics.observe()

    def test_a_condition_that_is_neither_is_refused(self, diagnostics):
        with pytest.raises(ValueError):
            diagnostics.read("resting")

    def test_the_two_names_are_the_records_own(self):
        assert {c.value for c in Condition} == {"quiescent", "driven"}

    def test_a_hand_built_reading_coerces_its_condition(self, diagnostics):
        """A reading in the record that no group counts is worse than an error.

        `Reading` is public, `report` groups by `is`, and `Condition` is a `str`
        enum -- so a plain `"driven"` would compare equal to `Condition.DRIVEN`
        without being it, leaving a reading that sits in `readings`, appears
        under neither condition, and is silently absent from the report.
        """
        edges = diagnostics.edge_reading()
        reading = Reading(tick=1, condition="driven", edges=edges)
        assert reading.condition is Condition.DRIVEN
        diagnostics.readings.append(reading)
        assert "driven (1 reading)" in diagnostics.report()

    def test_a_hand_built_reading_cannot_carry_a_condition_that_is_neither(
        self, diagnostics
    ):
        with pytest.raises(ValueError):
            Reading(tick=1, condition="resting", edges=diagnostics.edge_reading())


class TestTheCadence:
    def test_off_cadence_ticks_read_nothing(self, sheaf):
        diagnostics = Diagnostics(sheaf, every=5, whole_graph_every=10)
        seen = []
        for tick in range(1, 21):
            sheaf.ticks = tick
            reading = diagnostics.observe(Condition.DRIVEN)
            if reading is not None:
                seen.append(reading.tick)
        assert seen == [5, 10, 15, 20]
        assert [r.tick for r in diagnostics.readings] == seen

    def test_the_whole_graph_runs_on_its_own_longer_cadence(self, sheaf):
        diagnostics = Diagnostics(sheaf, every=2, whole_graph_every=6)
        for tick in range(1, 13):
            sheaf.ticks = tick
            diagnostics.observe(Condition.DRIVEN)
        carried = [r.tick for r in diagnostics.readings if r.whole_graph is not None]
        assert carried == [6, 12]
        assert all(
            isinstance(r.whole_graph, WholeGraphReading)
            for r in diagnostics.readings
            if r.whole_graph is not None
        )

    def test_a_whole_graph_reading_always_lands_on_a_paired_one(self, sheaf):
        """The multiple-of rule, asserted on the tick and not on the type.

        `read` builds every `Reading` with an `EdgeReading` in it, so checking
        that the expensive reading carries a pair is true by construction and
        tests nothing. What the rule actually buys is that the tick it lands on
        is one the *per-edge* cadence also covers, so the expensive numbers can
        be lined up against a cheap reading of the same configuration rather
        than interpolated between two.
        """
        diagnostics = Diagnostics(sheaf, every=3, whole_graph_every=9)
        for tick in range(1, 20):
            sheaf.ticks = tick
            diagnostics.observe(Condition.DRIVEN)
        carried = [r for r in diagnostics.readings if r.whole_graph is not None]
        assert [r.tick for r in carried] == [9, 18]
        for reading in carried:
            assert reading.tick % diagnostics.every == 0

    def test_a_whole_graph_cadence_off_the_paired_one_is_refused(self, sheaf):
        with pytest.raises(ValueError, match="must be a multiple"):
            Diagnostics(sheaf, every=4, whole_graph_every=10)

    @pytest.mark.parametrize("every", [0, -1, 1.5, True, "10"])
    def test_a_cadence_is_a_whole_number_of_ticks(self, sheaf, every):
        with pytest.raises(ValueError, match="cadence in ticks"):
            Diagnostics(sheaf, every=every)

    def test_read_ignores_the_cadence(self, sheaf):
        diagnostics = Diagnostics(sheaf, every=100, whole_graph_every=100)
        sheaf.ticks = 7
        assert diagnostics.observe(Condition.DRIVEN) is None
        assert diagnostics.read(Condition.DRIVEN).tick == 7

    def test_the_cadence_counts_ticks_and_not_calls(self, sheaf):
        """A caller striding off the grid records nothing, and the docstring says so.

        The cadence is `sheaf.ticks % every`, which is what makes the
        multiple-of rule a statement about tick numbers and two runs comparable
        at the same ones. The price is here rather than hidden: strides of ten
        from an odd start never land on a multiple of ten.
        """
        diagnostics = Diagnostics(sheaf, every=10, whole_graph_every=10)
        sheaf.ticks = 3
        for _ in range(6):
            sheaf.ticks += 10
            assert diagnostics.observe(Condition.DRIVEN) is None
        assert diagnostics.readings == []
        # And `read` is the way such a caller takes one anyway.
        assert diagnostics.read(Condition.DRIVEN).tick == sheaf.ticks

    def test_the_whole_graph_half_can_be_forced_off_the_grid(self, sheaf):
        """ADR-0007's quiescent hold sweeps configurations, landing where it lands."""
        diagnostics = Diagnostics(sheaf, every=1, whole_graph_every=100)
        sheaf.ticks = 7
        assert diagnostics.read(Condition.QUIESCENT).whole_graph is None
        forced = diagnostics.read(Condition.QUIESCENT, whole_graph=True)
        assert isinstance(forced.whole_graph, WholeGraphReading)
        assert forced.tick == 7
        # It overrides in both directions, so a sweep can also decline the cost
        # on a tick the longer cadence happens to land on.
        sheaf.ticks = 100
        assert (
            diagnostics.read(Condition.QUIESCENT, whole_graph=False).whole_graph is None
        )

    def test_forcing_it_does_not_let_the_pair_be_forced_off(self, sheaf):
        """The expensive half is optional; the pair never is."""
        diagnostics = Diagnostics(sheaf, every=1, whole_graph_every=1)
        with pytest.raises(TypeError):
            diagnostics.read(Condition.DRIVEN, edges=False)
        assert isinstance(
            diagnostics.read(Condition.DRIVEN, whole_graph=False).edges, EdgeReading
        )

    def test_observe_follows_the_longer_cadence_and_takes_no_override(self, sheaf):
        """The override is `read`'s, so `observe` keeps the multiple-of guarantee."""
        diagnostics = Diagnostics(sheaf, every=2, whole_graph_every=4)
        with pytest.raises(TypeError):
            diagnostics.observe(Condition.DRIVEN, whole_graph=True)
        for tick in range(1, 9):
            sheaf.ticks = tick
            diagnostics.observe(Condition.DRIVEN)
        assert [r.tick for r in diagnostics.readings if r.whole_graph is not None] == [
            4,
            8,
        ]

    def test_the_defaults_are_a_pair_the_constructor_would_accept(self):
        assert DEFAULT_WHOLE_GRAPH_EVERY % DEFAULT_EVERY == 0


class TestTheTopologyOnlyEnergy:
    """`topology_only_h1`'s twin on the energy scale (#363).

    The reference level #156's prototype stood in at `0.05` and recorded as
    owed: *a measured construction quantity*. What is held here is that it is a
    level of the right kind — on ADR-0010's gauge rather than off it, moving
    with the world and not with the maps, and supported on every edge, which is
    the property that ruled out the null-space minimum it might have been.
    """

    def test_it_is_one_value_an_edge_and_none_of_them_negative(self, sheaf):
        level = topology_only_energy(sheaf)
        assert level.shape == (len(sheaf.dome.edges),)
        assert bool((level >= 0).all())

    def test_it_is_supported_on_every_edge(self, sheaf):
        """The property the null-space minimum does not have, and the reason it lost.

        `‖P_null b‖²` is identically zero on every edge with a free end, so it
        is zero on every interior edge and cannot be a per-edge level. This one
        is a Dirichlet energy of the same configuration and is non-zero wherever
        the configuration disagrees at all — which, on a random configuration,
        is everywhere.
        """
        assert bool((topology_only_energy(sheaf) > 0).all())

    def test_it_reads_no_learned_map(self, sheaf):
        """The maps are generic, so collapsing the real ones moves nothing.

        This is the half it shares with `topology_only_h1`, and it is what makes
        the level a *reference* rather than a second reading of the thing it is
        being compared against. It survives the maps being zeroed, which is the
        strongest form of the check.
        """
        before = topology_only_energy(sheaf).clone()
        with torch.no_grad():
            sheaf.maps.maps.zero_()
        assert torch.allclose(topology_only_energy(sheaf), before)

    def test_it_is_drawn_on_the_gauge(self, sheaf):
        """Every generic map is at unit Frobenius norm, which is ADR-0010's centre.

        Read off the level rather than off the draw: for a single-draw energy on
        a one-edge complex the whole quantity is `‖F_u x_u − F_v x_v‖²`, which is
        bounded by `(‖F_u‖‖x_u‖ + ‖F_v‖‖x_v‖)²`. At unit norm that bound is
        computable here; at the `√(m·n) ≈ 20` an unnormalised draw carries it is
        two orders of magnitude larger, which is the failure this pins.
        """
        level = topology_only_energy(sheaf)
        dome = sheaf.dome
        for edge in dome.edges:
            reach = sum(
                float(sheaf.stalks[sheaf.layout.slice(c)].double().norm())
                for c in (edge.u, edge.v)
            )
            assert float(level[edge.id]) <= reach**2 + 1e-9

    def test_it_moves_with_the_world(self, sheaf):
        """Unlike the baseline, it is a reading of a configuration.

        Scaling every stalk by 3 scales every restricted end by 3 and the energy
        is quadratic, so the level scales by 9. A level that ignored the
        configuration would not move at all. To float32 and not to float64: the
        read runs in double but the buffer it reads is the sheaf's own single,
        so the `* 3.0` below is where the digits go.
        """
        before = topology_only_energy(sheaf).clone()
        with torch.no_grad():
            sheaf.stalks *= 3.0
        assert torch.allclose(topology_only_energy(sheaf), before * 9.0, rtol=1e-6)

    def test_averaging_the_draws_is_what_makes_it_a_level(self, sheaf):
        """One draw is a sample; the point of `draws` is that the mean is not.

        Two disjoint single draws disagree per edge by of order the value
        itself, and two disjoint eight-draw means are much closer. Held as an
        ordering rather than a number, `08-the-acceptance-demo.md`'s discipline.
        """

        def spread(draws: int) -> float:
            a, b = (
                topology_only_energy(
                    sheaf, draws=draws, generator=torch.Generator().manual_seed(s)
                )
                for s in (11, 22)
            )
            return float(((a - b).abs() / (a + b)).mean())

        assert spread(TOPOLOGY_ENERGY_DRAWS) < spread(1)

    def test_a_draw_count_below_one_is_refused(self, sheaf):
        with pytest.raises(ValueError, match="count of generic draws"):
            topology_only_energy(sheaf, draws=0)

class TestTheTopologyOnlyBaseline:
    def test_it_is_computed_at_construction_and_stored(self, sheaf):
        diagnostics = Diagnostics(sheaf)
        assert diagnostics.h1_baseline == topology_only_h1(sheaf.dome)
        assert isinstance(diagnostics.h1_baseline, int)

    def test_it_does_not_depend_on_the_generic_draw(self, dome):
        """A generic rank is the same for almost every draw, and this says so."""
        seeds = [torch.Generator().manual_seed(s) for s in (BASELINE_SEED, 17, 4242)]
        answers = {topology_only_h1(dome, generator=g) for g in seeds}
        assert len(answers) == 1

    def test_it_reads_no_learned_parameter(self, sheaf):
        before = Diagnostics(sheaf).h1_baseline
        with torch.no_grad():
            sheaf.maps.maps.zero_()
            sheaf.maps.maps[:, 0, 0] = 1.0
        assert Diagnostics(sheaf).h1_baseline == before

    def test_it_is_a_comparison_the_measured_dimension_can_be_made_against(
        self, diagnostics
    ):
        """`H¹` has two sources; the baseline is the one the maps did not cause.

        At a fresh draw the masked maps are generic within their masks, so the
        measured dimension sits at the baseline and the excess is zero. That is
        the reading the comparison exists to make possible -- against zero it
        would say nothing, because the cycles guarantee the baseline whatever
        the maps do (`docs/spec/06-graph-topology.md`, *What the cycles do*).
        """
        measured = diagnostics.read(Condition.DRIVEN).whole_graph
        assert measured.dim_h1 >= diagnostics.h1_baseline
        assert measured.dim_h1 - diagnostics.h1_baseline == 0

    def test_learned_rank_deficiency_shows_up_as_excess_over_it(self, sheaf):
        """Drive the fleet to rank 1 and the measured dimension leaves the baseline."""
        diagnostics = Diagnostics(sheaf)
        with torch.no_grad():
            sheaf.maps.maps.zero_()
            sheaf.maps.maps[:, 0, 0] = 1.0
            sheaf.maps.maps.mul_(sheaf.maps.support)
        measured = diagnostics.read(Condition.DRIVEN).whole_graph
        assert measured.dim_h1 > diagnostics.h1_baseline


class TestTheWholeGraphReading:
    def test_the_two_dimensions_differ_by_chi(self, diagnostics, dome):
        """`dim H⁰ − dim H¹ = χ`, over the predicting-cell subcomplex.

        The identity `01-cell-and-sheaf.md` states, on the convention
        `06-graph-topology.md` corrects it to.

        **This is a check on the convention, not on the assembly**, and saying
        so is the point of this docstring. Both dimensions are counted off one
        `rank`, so their difference is `columns − rows` for *any* rank at all
        and no arrangement of `δ` could break it. What it does catch is the
        thing it is actually sensitive to: that this module counts rows and
        columns the way `graph.py` counts `χ` -- predicting cells only on the
        node side, every edge on the edge side. Get that convention wrong and
        the identity fails immediately. The assembly of `δ` is checked by
        `test_the_rank_is_the_rank_of_an_independently_assembled_coboundary`
        and by the least-squares test below, both of which build `δ` by another
        route.
        """
        measured = diagnostics.read(Condition.DRIVEN).whole_graph
        assert measured.dim_h0 - measured.dim_h1 == dome.euler_characteristic

    def test_the_rank_is_the_rank_of_an_independently_assembled_coboundary(
        self, diagnostics
    ):
        """The check the χ identity cannot make: `δ` itself, by another route.

        `assembled_coboundary` builds the same matrix cell by cell out of the
        map tensor rather than through the module's precomputed block table, so
        a wrong sign, a wrong row offset or a dropped block moves this rank and
        moves neither dimension's relationship to it.
        """
        delta, _ = assembled_coboundary(diagnostics.sheaf)
        want = int(torch.linalg.matrix_rank(delta, atol=1e-9, rtol=1e-9))
        measured = diagnostics.read(Condition.DRIVEN).whole_graph
        assert measured.rank == want

    def test_the_measured_private_dimension_clears_the_construction_bound(
        self, diagnostics, dome
    ):
        """`dim H⁰ ≥ Σ_v max(0, n − Σ_{e∋v} m_e)`, which holds for any maps."""
        measured = diagnostics.read(Condition.DRIVEN).whole_graph
        assert measured.dim_h0 >= int(dome.private_dimensions.sum())

    def test_learned_rank_deficiency_only_enlarges_the_private_dimension(self, sheaf):
        diagnostics = Diagnostics(sheaf)
        before = diagnostics.read(Condition.DRIVEN).whole_graph.dim_h0
        with torch.no_grad():
            sheaf.maps.maps.zero_()
            sheaf.maps.maps[:, 0, 0] = 1.0
            sheaf.maps.maps.mul_(sheaf.maps.support)
        assert diagnostics.read(Condition.DRIVEN).whole_graph.dim_h0 > before

    def test_the_rank_is_what_both_dimensions_are_counted_from(self, diagnostics, dome):
        measured = diagnostics.read(Condition.DRIVEN).whole_graph
        rows = sum(edge.m for edge in dome.edges)
        columns = len(dome.predicting) * dome.shape.n
        assert measured.dim_h0 == columns - measured.rank
        assert measured.dim_h1 == rows - measured.rank

    def test_the_minimum_energy_is_the_least_squares_minimum(self, diagnostics):
        """Checked against a solve of the same system, by a different route."""
        delta, b = assembled_coboundary(diagnostics.sheaf)
        solution = torch.linalg.lstsq(delta, -b, driver="gelsd").solution
        want = float((delta @ solution + b).pow(2).sum())
        measured = diagnostics.read(Condition.DRIVEN).whole_graph
        assert measured.minimum_energy == pytest.approx(want, rel=1e-6, abs=1e-9)

    def test_the_minimum_energy_is_computed_in_float64_throughout(self, diagnostics):
        """`b` is part of the precision argument, not a bystander.

        The null-space form `Σ_{λ ≤ tol} (vᵀb)²` was chosen over
        `‖b‖² − ‖P_range b‖²` because the second loses most of its significant
        figures. A `b` restricted in the sheaf's own float32 puts a ~1e-7
        relative cap straight back on the answer, and bites hardest exactly
        where the floor is small against `‖b‖²` -- the regime the argument is
        about. Checked against a fully-float64 reference built by the other
        route, at a tolerance no float32 intermediate could reach.
        """
        delta, b = assembled_coboundary(diagnostics.sheaf)
        eigenvalues, vectors = torch.linalg.eigh(delta @ delta.T)
        null = eigenvalues <= (
            eigenvalues.max() * max(delta.shape) * torch.finfo(torch.float64).eps
        )
        want = float((vectors[:, null].T @ b).pow(2).sum())
        measured = diagnostics.read(Condition.DRIVEN).whole_graph
        assert measured.minimum_energy == pytest.approx(want, rel=1e-11)

    def test_the_minimum_is_never_above_what_the_configuration_actually_has(
        self, diagnostics
    ):
        reading = diagnostics.read(Condition.DRIVEN)
        assert reading.whole_graph.minimum_energy <= float(reading.edges.energy.sum())

    def test_the_floor_is_zero_on_this_dome_because_the_coboundary_is_onto(
        self, sheaf
    ):
        """And it was not, until #474 narrowed the lanes. Recorded, not asserted away.

        `minimum_energy` is `b` projected onto the **left** null space of the
        coboundary, so it is strictly positive only while `delta` is row-rank
        deficient. On this dome `delta` was `307 x 480` of rank 291 -- 16 rows
        deficient -- at `(interior_m, boundary_m) = (4, 8)`. At (3, 4) it is
        `181 x 480` of **full row rank 181**, the left null space is empty, and
        the floor is exactly 0: a configuration with zero disagreement exists.

        This is the same arithmetic as the private-dimension floor #474 was
        ruled for -- fewer lane dimensions against unchanged stalk dimensions --
        and it is a fact about *this small spec*, measured here. Whether the
        default dome's coboundary is also onto is **not** measured by this test.
        """
        diagnostics = Diagnostics(sheaf)
        assert diagnostics.read(Condition.DRIVEN).whole_graph.minimum_energy == (
            pytest.approx(0.0, abs=1e-12)
        )

    def test_it_moves_with_the_maps_rather_than_being_a_construction_constant(
        self, wide_sheaf
    ):
        # On a dome whose coboundary is not onto, so the quantity has room to
        # move; see the test above for why the default spec no longer does.
        sheaf = wide_sheaf
        diagnostics = Diagnostics(sheaf)
        before = diagnostics.read(Condition.DRIVEN).whole_graph
        with torch.no_grad():
            sheaf.maps.maps.mul_(-1.0)
            sheaf.maps.maps[:, 0, :] *= 3.0
            sheaf.maps.maps.mul_(sheaf.maps.support)
        after = diagnostics.read(Condition.DRIVEN).whole_graph
        assert after.minimum_energy != before.minimum_energy

    def test_the_boundary_conditions_are_the_world_s_and_are_held(
        self, wide_sheaf, wide_dome
    ):
        """Move a boundary cell's node stalk and the floor moves; the maps did not."""
        sheaf, dome = wide_sheaf, wide_dome
        diagnostics = Diagnostics(sheaf)
        before = diagnostics.read(Condition.DRIVEN).whole_graph.minimum_energy
        with torch.no_grad():
            sheaf.stalks[sheaf.layout.slice(dome.boundary[0])] += 5.0
        after = diagnostics.read(Condition.DRIVEN).whole_graph.minimum_energy
        assert after != pytest.approx(before)

    def test_a_predicting_cell_s_stalk_does_not_move_the_floor(self, sheaf, dome):
        """It is a *minimum over* those, so their current values cannot matter."""
        diagnostics = Diagnostics(sheaf)
        before = diagnostics.read(Condition.DRIVEN).whole_graph.minimum_energy
        with torch.no_grad():
            sheaf.stalks[sheaf.layout.slice(dome.predicting[0])] += 5.0
        after = diagnostics.read(Condition.DRIVEN).whole_graph.minimum_energy
        assert after == pytest.approx(before, rel=1e-9, abs=1e-12)


class TestNothingHereTouchesTheRun:
    def test_a_watched_run_is_bit_identical_to_an_unwatched_one(self, dome):
        def run(watch):
            sheaf = Sheaf(dome, generator=torch.Generator().manual_seed(3))
            diagnostics = Diagnostics(sheaf, every=1, whole_graph_every=1) if watch else None
            for _ in range(4):
                sheaf.tick()
                if diagnostics is not None:
                    diagnostics.observe(Condition.DRIVEN)
            return sheaf

        watched, alone = run(True), run(False)
        assert torch.equal(watched.stalks, alone.stalks)
        assert torch.equal(watched.charts, alone.charts)
        assert torch.equal(watched.broadcast, alone.broadcast)
        assert torch.equal(watched.maps.maps, alone.maps.maps)

    def test_a_reading_leaves_nothing_on_a_tape(self, diagnostics):
        reading = diagnostics.read(Condition.DRIVEN)
        for tensor in (reading.edges.energy, reading.edges.effective_rank):
            assert tensor.grad_fn is None and not tensor.requires_grad
        diagnostics.sheaf.assert_no_tape()

    def test_the_reading_is_a_copy_the_next_tick_cannot_reach_into(self, sheaf):
        diagnostics = Diagnostics(sheaf)
        reading = diagnostics.edge_reading()
        kept = reading.energy.clone()
        sheaf.tick()
        assert torch.equal(reading.energy, kept)

    def test_a_sheaf_on_another_dome_is_refused(self, sheaf):
        """The cached layout is one dome's; re-pointing across domes is refused.

        `__init__` reads the shape of `δ` and the `H¹` baseline off the built
        graph once, and the baseline's own note recommends re-pointing `sheaf`
        rather than rebuilding the instrument per run. Across two domes that
        would lay `b` out by one graph's edges while `δ` came from the other --
        a reading of neither. The same guard `timescale`, `tick` and `agent`
        all carry.
        """
        diagnostics = Diagnostics(sheaf)
        other = build_graph(SMALL)
        assert other is not sheaf.dome
        diagnostics.sheaf = Sheaf(other, generator=torch.Generator().manual_seed(0))
        for reading in (diagnostics.edge_reading, diagnostics.whole_graph):
            with pytest.raises(ValueError, match="different dome"):
                reading()

    def test_another_sheaf_on_the_same_dome_is_allowed(self, sheaf, dome):
        """Which is the sweep the baseline's note actually recommends."""
        diagnostics = Diagnostics(sheaf)
        diagnostics.sheaf = Sheaf(dome, generator=torch.Generator().manual_seed(9))
        assert len(diagnostics.edge_reading()) == len(dome.edges)
        assert diagnostics.whole_graph().dim_h0 - diagnostics.whole_graph().dim_h1 == (
            dome.euler_characteristic
        )

    def test_nothing_the_sheaf_holds_can_reach_a_diagnostics(self, sheaf):
        """The instrument holds the sheaf; the sheaf must not hold the instrument.

        The same shape :mod:`patchworks.timescale` and
        :mod:`patchworks.surface` take, and for the same reason: a cell's
        computation is handed the sheaf, so anything unreachable from the sheaf
        is unreachable from a cell.
        """
        diagnostics = Diagnostics(sheaf)
        seen, stack = set(), [sheaf]
        while stack:
            obj = stack.pop()
            if id(obj) in seen or isinstance(obj, torch.Tensor):
                continue
            seen.add(id(obj))
            assert obj is not diagnostics
            values = obj.values() if isinstance(obj, dict) else None
            if values is None and isinstance(obj, (list, tuple, set)):
                values = obj
            if values is None:
                values = getattr(obj, "__dict__", {}).values()
            stack.extend(values)


class TestTheReport:
    def test_it_groups_the_readings_by_condition(self, diagnostics):
        sheaf = diagnostics.sheaf
        for tick in range(1, 5):
            sheaf.ticks = tick
            diagnostics.observe(
                Condition.QUIESCENT if tick % 2 else Condition.DRIVEN
            )
        report = diagnostics.report()
        assert "quiescent (2 readings)" in report
        assert "driven (2 readings)" in report
        assert "per-edge Dirichlet energy" in report
        assert "per-edge effective rank" in report

    def test_it_states_the_baseline_the_measured_dimension_is_read_against(
        self, diagnostics
    ):
        diagnostics.read(Condition.DRIVEN)
        report = diagnostics.report()
        assert f"topology-only dim H^1 baseline: {diagnostics.h1_baseline}" in report
        assert "= chi" in report

    def test_an_empty_record_reports_that_and_not_a_traceback(self, diagnostics):
        assert "no readings yet" in diagnostics.report()


class TestTheReadingIsARecord:
    def test_a_reading_is_frozen(self, diagnostics):
        reading = diagnostics.read(Condition.DRIVEN)
        with pytest.raises(Exception):
            reading.tick = 5
        with pytest.raises(Exception):
            reading.edges.energy = torch.zeros(1)

    def test_a_reading_off_the_whole_graph_cadence_says_so_rather_than_faking_it(
        self, sheaf
    ):
        diagnostics = Diagnostics(sheaf, every=1, whole_graph_every=4)
        sheaf.ticks = 3
        assert diagnostics.read(Condition.DRIVEN).whole_graph is None
        sheaf.ticks = 4
        assert isinstance(
            diagnostics.read(Condition.DRIVEN).whole_graph, WholeGraphReading
        )
