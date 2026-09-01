"""`benchmarks/alignment_read.py`: the statistic, the null, and the identity (#244).

#244 explains a measurement and decides nothing, so nothing here pins its
numbers — the alignment it reports is a reading of a graph later tickets are
expected to change, and a test holding today's value would have to be deleted by
whoever changes it. That is `tests/test_construction_grading.py`'s reasoning,
inherited unchanged.

What is worth holding is that the arithmetic is **right**, because this ticket's
whole content is one ratio and the null it is read against, and both have a
plausible wrong form that nothing downstream would catch:

* **The null against a brute-force Monte Carlo.** :func:`null_sample` draws from
  the operator's singular values instead of pushing vectors through the matrix,
  on the claim that the two are the same distribution exactly. That claim is the
  read's foundation — the verdict is a percentile within this null — so it is
  checked against the thing it replaces rather than argued for in a docstring.
* **The instrument reports no effect when there is none.** Feed it genuinely
  isotropic directions and the null percentile must come out uniform, mean 50.
  This is the calibration test, and it is the one that would catch the whole
  ticket being answered by a bias in its own statistic. An instrument that
  answered "worse than average" to random noise would have produced a confident,
  wrong resolution.
* **`A` is scale-free and correctly bounded.** `sqrt(m_in)` for a direction on
  the operator's leading right-singular vector when the operator is rank 1, zero
  in the kernel, and invariant to rescaling either `M` or `d` — the properties
  that make it a *direction* statistic rather than a magnitude one in disguise.
* **The decomposition is an identity.** `A = a_in · a_out / C` holds by
  construction of the four quantities, and a version of `decompose` that used the
  wrong width in either denominator would break it silently while still printing
  numbers. The script checks this at runtime too; this is the check that the
  check is real.
* **`C` is #233's composition gap**, not a cousin of it, so the two reads compose
  and the "may absorb" question is settled on the same quantity both ticket
  bodies name.

And that the script still runs against the API. `null` is graph-side and
affordable whole; `align` needs a sandbox and a trained surface, so it is
smoke-tested on the small dome.
"""

import types

import numpy as np
import pytest
import torch

import alignment_read as ar
import construction_grading as grading
from patchworks.graph import build_graph
from patchworks.restriction import GAUGE_RHO, RestrictionMaps, pair_index
from patchworks.tick import DEFAULT_GAMMA, reconciliation_gain

from conftest import SMALL


def rank_one(m_out: int, m_in: int, scale: float = 3.0) -> tuple[torch.Tensor, ...]:
    """`(M, b)` for `M = scale · a bᵀ` with `a`, `b` unit — the worst case for the null.

    Rank 1 is where the gap between `E[A²] = 1` and the median of `A` is widest,
    so it is the case the read's whole framing rests on being able to see.
    """
    generator = torch.Generator().manual_seed(11)
    a = torch.randn(m_out, generator=generator, dtype=torch.float64)
    b = torch.randn(m_in, generator=generator, dtype=torch.float64)
    a, b = a / a.norm(), b / b.norm()
    return scale * torch.outer(a, b), b


class TestTheStatistic:
    def test_a_leading_direction_scores_the_square_root_of_the_width(self):
        """`A = sqrt(m_in)` on a rank-1 operator's own right-singular vector.

        The ceiling of the statistic, and it is the same ceiling #233's
        alignment headroom is bounded by — so the two reads are on one scale.
        """
        for m_in in (2, 4, 8):
            operator, b = rank_one(5, m_in)
            assert ar.alignment(operator, b, m_in) == pytest.approx(
                np.sqrt(m_in), rel=1e-9
            )

    def test_a_kernel_direction_scores_zero(self):
        operator, b = rank_one(5, 4)
        d = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        d = d - (d @ b) * b
        assert ar.alignment(operator, d, 4) == pytest.approx(0.0, abs=1e-12)

    def test_is_invariant_to_rescaling_the_operator_or_the_direction(self):
        """A direction statistic, not a magnitude one wearing its clothes.

        If this failed, every number in the read would be partly a report of how
        big the maps happen to be — which is #233's tier-2 question and already
        answered, not this ticket's.
        """
        operator, b = rank_one(6, 4)
        d = torch.randn(4, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
        base = ar.alignment(operator, d, 4)
        assert ar.alignment(operator * 17.0, d, 4) == pytest.approx(base, rel=1e-9)
        assert ar.alignment(operator, d * 0.001, 4) == pytest.approx(base, rel=1e-9)

    def test_the_mean_square_over_isotropic_directions_is_one(self):
        """`E[A²] = 1` exactly — the property that makes 1 the *mean-square* bar.

        And, read beside :class:`TestTheNull`, the property that makes 1 the
        wrong bar for a median. Both halves are held, because the ticket's own
        phrasing invites the second mistake.
        """
        generator = torch.Generator().manual_seed(5)
        for m_in in (3, 4, 8):
            operator = torch.randn(6, m_in, generator=generator, dtype=torch.float64)
            draws = torch.randn(40_000, m_in, generator=generator, dtype=torch.float64)
            scores = np.array(
                [ar.alignment(operator, d, m_in) for d in draws[:4000]]
            )
            assert (scores**2).mean() == pytest.approx(1.0, rel=0.06)


class TestTheNull:
    def test_matches_a_brute_force_monte_carlo_through_the_matrix(self):
        """The singular-value draw against pushing vectors through `M` itself.

        :func:`null_sample` is the read's foundation — the verdict is a
        percentile within it — and it takes a shortcut. This is the check that
        the shortcut is an identity and not an approximation, made the way #233
        checked its predictor: against the object it claims to stand for.
        """
        generator = torch.Generator().manual_seed(7)
        rng = np.random.default_rng(7)
        for m_out, m_in in ((5, 4), (3, 8), (6, 6)):
            operator = torch.randn(m_out, m_in, generator=generator, dtype=torch.float64)
            fast = ar.null_sample(operator, m_in, rng)
            draws = torch.randn(20_000, m_in, generator=generator, dtype=torch.float64)
            brute = np.array([ar.alignment(operator, d, m_in) for d in draws])
            for q in (5, 25, 50, 75, 95):
                assert np.percentile(fast, q) == pytest.approx(
                    np.percentile(brute, q), rel=0.03
                )

    def test_a_rank_one_operator_puts_the_median_well_below_one(self):
        """The trap, held down as a number — and the number depends on the width.

        Against a rank-1 operator `A = |<v₁, d>| · sqrt(m_in)`, so
        `A² / m_in ~ Beta(1/2, (m_in - 1)/2)` and the median runs **0.816 at
        `m_in = 4`** down to **0.674** as the width grows. Both ends are pinned
        because quoting the asymptote at this graph's actual widths would
        overstate the trap, and quoting the narrow case would understate it
        wherever the edges are wide. The read never relies on either: it draws
        the null per hop. This is the check that it has to.
        """
        for m_in, expected in ((4, 0.816), (64, 0.674)):
            operator, _b = rank_one(5 if m_in == 4 else 70, m_in)
            sample = ar.null_sample(operator, m_in, np.random.default_rng(1))
            assert float(np.median(sample)) == pytest.approx(expected, abs=0.02)
            assert (sample**2).mean() == pytest.approx(1.0, rel=0.05)

    def test_isotropic_directions_come_out_uniform_in_percentile(self):
        """The calibration test: no effect in, no effect out.

        This is the one that would have caught the ticket being answered by a
        bias in its own statistic. The percentile of a draw within its own null
        is uniform on [0, 100] by definition, so the mean must land at 50 — and
        an instrument that reported 40 here would have reported "arriving
        directions are worse than average" about pure noise.
        """
        generator = torch.Generator().manual_seed(13)
        rng = np.random.default_rng(13)
        for m_out, m_in in ((5, 4), (4, 8)):
            operator = torch.randn(m_out, m_in, generator=generator, dtype=torch.float64)
            null = ar.null_sample(operator, m_in, rng)
            draws = torch.randn(4000, m_in, generator=generator, dtype=torch.float64)
            percentiles = np.array(
                [(null < ar.alignment(operator, d, m_in)).mean() * 100.0 for d in draws]
            )
            assert percentiles.mean() == pytest.approx(50.0, abs=2.0)
            assert np.percentile(percentiles, 25) == pytest.approx(25.0, abs=3.5)

    def test_a_null_median_is_never_above_one(self):
        """`E[A²] = 1` with any spread at all forces the median under 1.

        Reported per hop in `null`, so if this ever came out above 1 the read
        would be quoting a null that cannot exist.
        """
        generator = torch.Generator().manual_seed(17)
        rng = np.random.default_rng(17)
        for _ in range(8):
            operator = torch.randn(6, 4, generator=generator, dtype=torch.float64)
            assert float(np.median(ar.null_sample(operator, 4, rng))) <= 1.0 + 1e-9


class TestRankProfile:
    def test_rank_one_reads_one_and_the_whole_share(self):
        operator, _b = rank_one(5, 4)
        effective, share = ar.rank_profile(operator)
        assert effective == pytest.approx(1.0, rel=1e-9)
        assert share == pytest.approx(1.0, rel=1e-9)

    def test_equal_singular_values_read_their_count(self):
        operator = torch.eye(4, dtype=torch.float64)
        effective, share = ar.rank_profile(operator)
        assert effective == pytest.approx(4.0, rel=1e-9)
        assert share == pytest.approx(0.25, rel=1e-9)


class TestTheDecomposition:
    def test_the_identity_closes_on_the_real_maps(self):
        """`A = a_in · a_out / C`, on the graph rather than on a toy.

        The identity is what makes #233's composition gap a *factor of* this
        ticket's statistic rather than a separate candidate to be argued about,
        so it is checked where it is used.
        """
        dome = build_graph(SMALL)
        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
        gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=GAUGE_RHO)
        generator = torch.Generator().manual_seed(23)
        seen = 0
        for cell in dome.cells:
            if cell.is_boundary or dome.degrees[cell.id] < 2:
                continue
            edge_in, edge_out = dome.incident[cell.id][:2]
            operator, f_in, f_out = ar.hop_operator(
                dome, maps, gains, edge_in, cell.id, edge_out
            )
            m_in = dome.edges[edge_in].m
            perm = grading.permitted_width(dome, cell.id)
            d = torch.randn(m_in, generator=generator, dtype=torch.float64)
            parts = ar.decompose(
                operator, f_in, f_out, float(gains[cell.id]), d, m_in, perm
            )
            assert parts["a_in"] * parts["a_out"] / parts["composition"] == (
                pytest.approx(ar.alignment(operator, d, m_in), rel=1e-9)
            )
            seen += 1
        assert seen > 5

    def test_a_rank_one_inbound_map_makes_a_out_exactly_the_composition_gap(self):
        """The mechanism behind the read's answer, as algebra rather than a plot.

        When `F_in` has rank 1, `u = F_inᵀ d` is the *same* node-stalk direction
        whatever arrives — only its length changes. So `‖F_out u‖ / ‖u‖` is a
        constant of the maps, `a_out` stops being a property of the arriving
        direction, and it equals `C` identically. `A = a_in · a_out / C` then
        collapses to `A = a_in`.

        This is why the measured answer is what it is: the arriving direction has
        no purchase at the relay, so it cannot be misaligned there. Held as a
        test because the run reports the two distributions and their near-equality
        is the finding — an equality that holds by construction should be proved,
        not eyeballed off a table.
        """
        generator = torch.Generator().manual_seed(97)
        for m_in, m_out, perm in ((4, 4, 12), (8, 4, 24), (4, 8, 12)):
            left = torch.randn(m_in, generator=generator, dtype=torch.float64)
            stalk = torch.randn(perm, generator=generator, dtype=torch.float64)
            # Rank 1 by construction, and padded to the node-stalk width the
            # maps really have so nothing depends on the padding.
            f_in = torch.outer(left / left.norm(), stalk / stalk.norm()) * 1.7
            f_out = torch.randn(m_out, perm, generator=generator, dtype=torch.float64)
            gain = 0.31
            operator = (f_out @ f_in.T) * gain
            for _ in range(5):
                d = torch.randn(m_in, generator=generator, dtype=torch.float64)
                parts = ar.decompose(
                    operator, f_in, f_out, gain, d, m_in, perm
                )
                assert parts["a_out"] == pytest.approx(parts["composition"], rel=1e-9)
                assert parts["a_out_over_c"] == pytest.approx(1.0, rel=1e-9)
                assert ar.alignment(operator, d, m_in) == pytest.approx(
                    parts["a_in"], rel=1e-9
                )

    def test_a_full_rank_inbound_map_leaves_a_out_free_of_the_composition_gap(self):
        """The contrast case, so the test above is not vacuous.

        With a well-conditioned `F_in` the arriving direction reaches different
        node-stalk directions, `a_out` varies with it, and `a_out / C` moves off
        1. If this passed as an equality too, the previous test would be proving
        a property of `decompose`'s arithmetic rather than of rank.
        """
        generator = torch.Generator().manual_seed(101)
        m_in, m_out, perm = 4, 4, 8
        f_in = torch.randn(m_in, perm, generator=generator, dtype=torch.float64)
        f_out = torch.randn(m_out, perm, generator=generator, dtype=torch.float64)
        gain = 0.5
        operator = (f_out @ f_in.T) * gain
        ratios = []
        for _ in range(8):
            d = torch.randn(m_in, generator=generator, dtype=torch.float64)
            ratios.append(
                ar.decompose(operator, f_in, f_out, gain, d, m_in, perm)["a_out_over_c"]
            )
        assert np.std(ratios) > 0.01

    def test_the_composition_factor_is_233s_tier_three_over_tier_two(self):
        """`C` is the same number `construction_grading` reports, not a cousin.

        #244's body asks whether it absorbs #233's composition finding. That
        question is only meaningful if the two scripts are naming one quantity,
        which is here rather than in prose.
        """
        dome = build_graph(SMALL)
        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
        gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=GAUGE_RHO)
        norms = maps.norms().detach()
        generator = torch.Generator().manual_seed(29)
        seen = 0
        for cell in dome.cells:
            if cell.is_boundary or dome.degrees[cell.id] < 2:
                continue
            edge_in, edge_out = dome.incident[cell.id][:2]
            key = (edge_in, cell.id, edge_out)
            operator, f_in, f_out = ar.hop_operator(dome, maps, gains, *key)
            m_in = dome.edges[edge_in].m
            perm = grading.permitted_width(dome, cell.id)
            d = torch.randn(m_in, generator=generator, dtype=torch.float64)
            mine = ar.decompose(
                operator, f_in, f_out, float(gains[cell.id]), d, m_in, perm
            )["composition"]
            tier2 = grading.predicted_hop(dome, gains, *key, norms=norms)
            tier3, _headroom = grading.exact_operator(dome, maps, gains, *key)
            # To float32 and not to float64: #233's tier 2 reads the maps'
            # Frobenius norms through `RestrictionMaps.norms()`, which is
            # computed in the dtype the maps are stored in. The agreement is
            # exact in exact arithmetic and this is the precision it survives at.
            assert mine == pytest.approx(tier3 / tier2, rel=1e-6)
            seen += 1
        assert seen > 5


class TestTheOperator:
    def test_agrees_with_233s_exact_operator_on_the_frobenius_reading(self):
        """One operator, two scripts. #244 extends #233's read and must not fork it."""
        from graph_transmission import BODY_GAIN

        dome = build_graph(SMALL)
        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
        gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=GAUGE_RHO)
        seen = 0
        for cell in dome.cells:
            if cell.is_boundary or dome.degrees[cell.id] < 2:
                continue
            edge_in, edge_out = dome.incident[cell.id][:2]
            key = (edge_in, cell.id, edge_out)
            operator, _f_in, _f_out = ar.hop_operator(dome, maps, gains, *key)
            isotropic, _headroom = grading.exact_operator(dome, maps, gains, *key)
            m_in = dome.edges[edge_in].m
            assert BODY_GAIN * float(operator.norm()) / np.sqrt(m_in) == (
                pytest.approx(isotropic, rel=1e-9)
            )
            seen += 1
        assert seen > 5

    def test_the_operator_maps_the_incoming_width_to_the_outgoing_one(self):
        dome = build_graph(SMALL)
        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
        gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=GAUGE_RHO)
        for cell in dome.cells:
            if cell.is_boundary or dome.degrees[cell.id] < 2:
                continue
            edge_in, edge_out = dome.incident[cell.id][:2]
            operator, f_in, f_out = ar.hop_operator(
                dome, maps, gains, edge_in, cell.id, edge_out
            )
            assert operator.shape == (dome.edges[edge_out].m, dome.edges[edge_in].m)
            assert f_in.shape[0] == dome.edges[edge_in].m
            assert f_out.shape[0] == dome.edges[edge_out].m
            assert operator.dtype is torch.float64


class TestThePeakLookup:
    def test_finds_the_peak_by_tick_number_and_not_by_position(self):
        """A dropped tick must not shift which tick `..._at_peak` reports.

        `read_hops` keeps only the ticks that clear the direction floor, so its
        per-tick list is compacted and its indices stop being tick indices. The
        peak-tick columns exist to be compared against #233's own reduction, and
        reporting a *different* tick under that name would make the comparison
        quietly wrong rather than visibly broken — the failure mode this whole
        file is written against.
        """
        dome = build_graph(SMALL)
        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
        gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=GAUGE_RHO)
        agent = types.SimpleNamespace(sheaf=types.SimpleNamespace(maps=maps))

        cell = next(
            c for c in dome.cells if not c.is_boundary and dome.degrees[c.id] >= 2
        )
        edge_in, edge_out = dome.incident[cell.id][:2]
        m_in, m_out = dome.edges[edge_in].m, dome.edges[edge_out].m

        generator = torch.Generator().manual_seed(41)
        ticks = 6
        arriving = torch.randn(ticks, m_in, generator=generator, dtype=torch.float64)
        leaving = torch.randn(ticks, m_out, generator=generator, dtype=torch.float64)
        # Tick 0 is below the floor and is dropped, so position 2 of the
        # surviving list is tick 3 — the off-by-one this test exists to catch.
        arriving[0] = 0.0
        peak_tick = 2

        rows, _checks = ar.read_hops(
            dome,
            [
                dict(
                    trial=0,
                    direction="rim-to-apex",
                    edge_in=edge_in,
                    cell=cell.id,
                    edge_out=edge_out,
                    arriving=arriving,
                    leaving=leaving,
                    peak_tick=peak_tick,
                )
            ],
            agent,
        )
        assert len(rows) == 1
        operator, _f_in, _f_out = ar.hop_operator(
            dome, maps, gains, edge_in, cell.id, edge_out
        )
        expected = ar.alignment(operator, arriving[peak_tick], m_in)
        assert rows[0]["alignment_at_peak"] == pytest.approx(expected, rel=1e-9)
        assert rows[0]["peak_tick_offset"] == 0
        # And the value at the position the old code would have read is
        # genuinely different, so this test would have failed against it.
        assert ar.alignment(operator, arriving[peak_tick + 1], m_in) != (
            pytest.approx(expected, rel=1e-6)
        )


class TestBenchmark:
    """That the script still runs against the API, in the other benchmarks' shape."""

    def test_null_runs_on_the_real_dome(self, capsys):
        """`null` is the half of the read that needs no run, so it is held whole."""
        ar.main(["null"])
        printed = capsys.readouterr().out
        assert "a typical direction scores" in printed
        assert "directed hops the graph admits" in printed

    def test_align_runs(self, capsys):
        """The whole path, on the small dome: sandbox, fork, null, decomposition.

        Short everywhere it can be — the numbers this produces are not the
        ticket's numbers and are not meant to be. What is checked is that
        `collect`, `read_hops` and the four report sections still agree about the
        dictionaries they pass between them, which is what #233's own smoke test
        asks of `regress`.
        """
        ar.main(
            [
                "align",
                "--dome",
                "small",
                "--learn",
                "50",
                "--trials",
                "2",
                "--hold",
                "20",
                "--window",
                "8",
            ]
        )
        printed = capsys.readouterr().out
        assert "the arriving direction against the next hop's operator" in printed
        assert "mean percentile" in printed
        assert "the identity closes" in printed
