"""ADR-0004's exclusion procedure, as a rig (ticket #363).

The rig itself asserts nothing — `benchmarks/run_reporting.py` states the rule
for everything in that directory. What stands here is the part of it that is not
a measurement: **the waterfall's arithmetic**, which either partitions an
edge's energy or does not, and the three instruments the four stages are built
out of, each held against a case whose answer is known independently of the
implementation.

Three of them matter.

**The identity is exact, not approximate.** A decomposition whose shares do not
sum to the thing decomposed is a set of numbers, not an attribution, and every
stage of `attribute` is checked against both halves of it on a case built to
route an edge down each of the four paths at once.

**Nothing is booked twice, and the order is the ADR's.** The stages are ordered
by ADR-0004 and the order changes the answer: an edge that is both too narrow to
embed its piece and negatively curved books to self-intersection, because that
is the stage the ADR reads first. Written as a test because it is the one thing
a reader cannot see from the arithmetic.

**The gauge share is read against chance and cannot exceed one.** It is a
projection of a vector into orthogonal complements, so both parts are
non-negative — which is what stopped the first shape of that stage, projecting
the stalks and re-reading the energy, from being usable at all.

The run itself gets one smoke test on the small dome, on the same footing as
every other rig in `benchmarks/`.
"""

import numpy as np
import pytest
import torch

import edge_attribution as ea
from patchworks.graph import build_graph
from patchworks.tick import Sheaf

from conftest import SMALL


@pytest.fixture
def sheaf():
    dome = build_graph(SMALL)
    built = Sheaf(dome, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(1)
    built.stalks = torch.randn_like(built.stalks)
    built.stalks[built.layout.pad] = 0.0
    return built


class TestTheGaugeProjector:
    """`colspan(D)`, and the chance level it has to be read against."""

    def test_it_is_a_projector_of_rank_k(self, sheaf):
        projector = ea.gauge_projector(sheaf.body)
        shape = sheaf.dome.shape
        assert projector.shape == (shape.n, shape.n)
        assert torch.allclose(projector @ projector, projector, atol=1e-10)
        assert torch.allclose(projector, projector.T, atol=1e-10)
        assert float(projector.trace()) == pytest.approx(shape.k, abs=1e-8)

    def test_it_fixes_the_columns_of_d_and_nothing_more(self, sheaf):
        """`P D = D`, which is what makes it the projector onto `colspan(D)`."""
        d = sheaf.body.decode_weight.detach().to(torch.float64)
        assert torch.allclose(ea.gauge_projector(sheaf.body) @ d, d, atol=1e-10)

    def test_chance_is_k_over_n(self, sheaf):
        shape = sheaf.dome.shape
        assert ea.chance_alignment(shape) == shape.k / shape.n

    def test_a_random_direction_sits_at_chance(self, sheaf):
        """The reason the share is read against `k/n` and not against zero.

        Over many random directions the fraction landing in a `k`-dimensional
        subspace of `R^n` averages `k/n`. Held loosely, because it is a
        statement about an average and the point is the *level*, not a digit.
        """
        projector = ea.gauge_projector(sheaf.body)
        n = sheaf.dome.shape.n
        generator = torch.Generator().manual_seed(7)
        directions = torch.empty(400, n, dtype=torch.float64).normal_(
            generator=generator
        )
        inside = (directions @ projector.T).pow(2).sum(-1)
        share = float((inside / directions.pow(2).sum(-1)).mean())
        assert share == pytest.approx(ea.chance_alignment(sheaf.dome.shape), abs=0.03)

    def test_both_halves_are_fractions_and_neither_raises_an_energy(self, sheaf):
        """In `[0, 1]` by construction, which the first shape of this stage was not.

        Projecting the stalks and re-reading the energy raised it on 52 of 54
        edges, because taking content off one end of an edge and not the other
        moves the two ends apart. This shape cannot: it splits one vector into
        two orthogonal parts and reports the ratio.
        """
        inside, share = ea.gauge_reading(sheaf, ea.gauge_projector(sheaf.body))
        assert inside.shape == share.shape == (len(sheaf.dome.edges),)
        for half in (inside, share):
            assert bool(((half >= 0.0) & (half <= 1.0)).all())

    def test_a_pull_lying_wholly_outside_the_gauge_books_the_whole_edge(self, sheaf):
        """The saturating case: nothing of the correction is inside `colspan(D)`.

        Held on :func:`excess_over_chance` rather than on a built sheaf, because
        the direction `Fᵀr` lands in is the row space of `F` and there is no
        configuration that puts it wholly outside `colspan(D)` on demand. This
        is the arithmetic that turns the observed fraction into the share, and
        it is the part a reader has to be able to check.
        """
        chance = 1.0 - ea.chance_alignment(sheaf.dome.shape)
        assert ea.excess_over_chance(1.0, chance) == pytest.approx(1.0)
        assert ea.excess_over_chance(chance, chance) == 0.0
        assert ea.excess_over_chance(0.0, chance) == 0.0
        # Halfway between chance and everything reads halfway.
        assert ea.excess_over_chance(
            chance + (1.0 - chance) / 2, chance
        ) == pytest.approx(0.5)

    def test_the_raw_alignment_is_reported_beside_the_share(self, sheaf):
        """Stage 1's job is to *rule the gauge out*, and a zero share only reads
        as a ruling-out next to the level it was compared against.

        On the built graph the pulled-back residual sits within a few points of
        chance, so the share is near zero — which is ADR-0004's *ruling it out
        costs one projection* actually happening, and would be unreadable if the
        rig reported only the zero.
        """
        inside, _share = ea.gauge_reading(sheaf, ea.gauge_projector(sheaf.body))
        alive = inside > 0
        assert bool(alive.any())
        assert float(inside[alive].mean()) == pytest.approx(
            ea.chance_alignment(sheaf.dome.shape), abs=0.15
        )


class TestTheEmbeddingCriterion:
    """#49's criterion and the estimator that feeds it."""

    @pytest.mark.parametrize(
        "width,dimension,expected",
        [
            (4, 1.0, True),
            (4, 2.0, False),  # strict: the theorem guarantees nothing *at* the bound
            (4, 2.5, False),
            (8, 3.9, True),
            (8, 4.0, False),
        ],
    )
    def test_the_criterion_is_strictly_twice_the_dimension(
        self, width, dimension, expected
    ):
        assert ea.embeds(width, dimension) is expected

    def test_a_line_reads_dimension_one(self):
        """A one-dimensional cloud in a high-dimensional space.

        The estimator's answer has to come from the cloud's own structure and
        not from the space it sits in, which is the whole reason the dimension
        is measured in the `n`-dimensional stalk rather than in the `m`-wide
        edge it is being compared against.
        """
        t = np.linspace(0.0, 1.0, 300)[:, None]
        direction = np.zeros((1, 16))
        direction[0, 3] = 1.0
        direction[0, 11] = 2.0
        assert ea.correlation_dimension(t * direction) == pytest.approx(1.0, abs=0.2)

    def test_a_filled_square_reads_dimension_two(self):
        points = np.random.default_rng(3).uniform(size=(600, 2))
        assert ea.correlation_dimension(points) == pytest.approx(2.0, abs=0.3)

    def test_a_cloud_that_never_moves_reads_zero(self):
        """The honest answer, and the one that keeps the criterion from firing.

        A cell whose stalk is constant carries a piece of dimension zero, and no
        stalk width is too narrow for it.
        """
        assert ea.correlation_dimension(np.ones((200, 12))) == 0.0

    def test_too_few_points_read_zero_rather_than_guessing(self):
        assert ea.correlation_dimension(np.random.default_rng(0).uniform(size=(4, 3))) == 0.0

    def test_an_edge_takes_the_smaller_of_its_predicting_ends(self):
        """And a boundary end contributes nothing, because it owns no piece."""
        dome = build_graph(SMALL)
        row = {c: i for i, c in enumerate(dome.predicting)}
        per_cell = np.arange(1.0, len(row) + 1.0)
        got = ea.edge_dimensions(dome, per_cell)
        for edge in dome.edges:
            ends = [per_cell[row[c]] for c in (edge.u, edge.v) if c in row]
            assert got[edge.id] == (min(ends) if ends else 0.0)


class TestTheWaterfall:
    """The arithmetic. Four edges, one down each path, and one that survives."""

    @staticmethod
    def case():
        #                     0: too narrow   1: negative Ric   2: draining   3: survives
        total = np.array([10.0, 10.0, 10.0, 10.0])
        gauge = np.array([0.10, 0.10, 0.10, 0.10])
        # `held` is the driven-scale energy at the end of the hold, net of its
        # own gauge share. Edge 2 drains almost entirely; the rest stand.
        held = np.array([9.0, 9.0, 0.5, 9.0])
        curvature = np.array([1.0, -1.0, 1.0, 1.0])
        reference = np.array([1.0, 1.0, 1.0, 1.0])
        embedded = np.array([False, True, True, True])
        return total, gauge, held, curvature, reference, embedded

    def test_the_two_identities_are_exact_on_every_edge(self):
        got = ea.attribute(*self.case())
        assert np.allclose(
            got.gauge + got.self_intersection + got.lag + got.standing, got.total
        )
        assert np.allclose(got.curvature + got.reference + got.residue, got.standing)

    def test_each_edge_goes_down_the_path_it_was_built_for(self):
        got = ea.attribute(*self.case())
        assert got.self_intersection[0] == pytest.approx(9.0)
        assert got.curvature[1] == pytest.approx(9.0)
        assert got.lag[2] == pytest.approx(8.5)
        assert got.residue[3] == pytest.approx(8.0)
        assert list(got.surviving) == [False, False, False, True]

    def test_self_intersection_is_read_before_curvature(self):
        """ADR-0004's order, and it changes the answer.

        An edge that is both too narrow to embed its piece **and** negatively
        curved books to self-intersection, because that is the stage the ADR
        reads first. Nothing in the arithmetic shows this; only the order does.
        """
        total, gauge, held, curvature, reference, embedded = self.case()
        curvature[0] = -1.0
        got = ea.attribute(total, gauge, held, curvature, reference, embedded)
        assert got.self_intersection[0] == pytest.approx(9.0)
        assert got.curvature[0] == 0.0

    def test_an_edge_standing_below_the_reference_does_not_survive(self):
        """#156's gate 2, and the prototype found it is not optional.

        *Without it the hold cheerfully halts a converged, healthy cell.* The
        whole standing residual books to the reference share, so the identity
        still closes and the residue is zero.
        """
        total, gauge, held, curvature, reference, embedded = self.case()
        reference[3] = 100.0
        got = ea.attribute(total, gauge, held, curvature, reference, embedded)
        assert not got.surviving[3]
        assert got.residue[3] == 0.0
        assert got.reference[3] == pytest.approx(got.standing[3])

    def test_a_hold_that_raises_the_energy_books_no_lag_and_says_so(self):
        """The one clamp. It books zero and flags the edge rather than going negative.

        Conservative in the direction that matters: the whole driven remainder
        passes on as standing, so a stage that could not measure itself takes
        nothing *out* of the residue.
        """
        total, gauge, held, curvature, reference, embedded = self.case()
        held[3] = 50.0
        got = ea.attribute(total, gauge, held, curvature, reference, embedded)
        assert got.lag[3] == 0.0
        assert bool(got.grew[3])
        assert got.standing[3] == pytest.approx(9.0)
        assert np.allclose(
            got.gauge + got.self_intersection + got.lag + got.standing, got.total
        )

    def test_no_share_is_ever_negative(self):
        got = ea.attribute(*self.case())
        for share in (got.gauge, got.self_intersection, got.lag, got.curvature,
                      got.reference, got.residue):
            assert bool((share >= 0.0).all())

    def test_the_ratio_is_the_residue_over_the_reference_on_the_survivors(self):
        got = ea.attribute(*self.case())
        assert got.ratio() == pytest.approx(8.0)

    def test_nothing_surviving_reads_zero_rather_than_undefined(self):
        """What #333 *not* happening looks like, and it is a reading not a gap."""
        total, gauge, held, curvature, reference, embedded = self.case()
        got = ea.attribute(total, gauge, held, curvature, reference * 100.0, embedded)
        assert not got.surviving.any()
        assert got.ratio() == 0.0


class TestTheReadings:
    """What `tools/cutoff_report.py` is handed, and that the minimum is a minimum."""

    def test_the_headline_is_the_minimum_across_the_sweep(self):
        total, gauge, held, curvature, reference, embedded = TestTheWaterfall.case()
        arms = [
            ea.attribute(total, gauge, held, curvature, reference * k, embedded)
            for k in (1.0, 2.0)
        ]
        assert ea.readings(arms)["residue_over_topology"] == pytest.approx(
            min(a.ratio() for a in arms)
        )

    def test_readings_reports_every_metric_it_owns(self):
        """The three `readings` owns. `residue_over_topology_opening` is the
        fourth in the module's table and is assembled in :func:`read`, because
        it is a second sweep's headline and `readings` reports one sweep."""
        total, gauge, held, curvature, reference, embedded = TestTheWaterfall.case()
        got = ea.readings([ea.attribute(total, gauge, held, curvature, reference, embedded)])
        assert set(got) == {
            "residue_over_topology",
            "surviving_edges",
            "gauge_share",
        }

    def test_a_condition_that_could_not_evaluate_withholds_the_headline(self):
        """`@unevaluated`, not a CLEAR off a division that never happened.

        `docs/agents/registers.md`: *a run that could not evaluate the bar is
        not a run.* Reporting `0.0` here would lift #333 out of the register's
        second loud section while nothing whatever was watching it.
        """
        total, gauge, held, curvature, reference, embedded = TestTheWaterfall.case()
        got = ea.attribute(
            total, gauge, held, curvature, np.zeros_like(reference), embedded
        )
        assert got.surviving.any()
        assert got.ratio() is None
        assert "residue_over_topology" not in ea.readings([got])
        assert "surviving_edges" in ea.readings([got])

    def test_no_survivors_is_a_reading_of_zero_and_not_a_withholding(self):
        """The other zero, and it is a real one: every edge's disagreement was
        booked to a named cause, which is what #333 not happening looks like."""
        total, gauge, held, curvature, reference, embedded = TestTheWaterfall.case()
        got = ea.attribute(total, gauge, held, curvature, reference * 100.0, embedded)
        assert not got.surviving.any()
        assert got.ratio() == 0.0
        assert ea.readings([got])["residue_over_topology"] == 0.0


class TestBenchmark:
    """That the script still runs against the API, on the small dome."""

    def test_read_runs(self, capsys):
        ea.main(
            [
                "read",
                "--dome",
                "small",
                "--learn",
                "2",
                "--configurations",
                "1",
                "--drive",
                "12",
                "--hold",
                "6",
                "--cloud",
                "10",
                "--no-file",
            ]
        )
        printed = capsys.readouterr().out
        assert "residue_over_topology" in printed
        assert "the waterfall" in printed
