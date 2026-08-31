"""Bias selection, the fold margin, and the go/no-go (ticket #85).

What these tests hold down is the **rig**, not the verdict. The verdict is a
measurement of a particular body and it is allowed to move; what may not move
is how it is read — `tau` as quantiles and never moments, measured over a driven
trajectory with the operating point varying, dwell reported alongside, decay
reported as realised `lambda` with `max rho < 1` demoted to the cheap sufficient
check, bands overlapping and assigned by level, and a band no draw reached
reported as a kill rather than filled from its neighbour.

Two of them do pin numbers. `test_selection_is_not_a_draw` pins that a selected
population is narrower than the draw it came from, which is the whole claim of
selection over drawing. And `TestTheMeasuredBody` records what this body
actually measured at the time of the ticket, so that a later change to the body
or the rig announces itself instead of quietly moving the falsification.
"""

import math

import pytest
import timescale_selection
import torch

from patchworks.body import BodyShape, CellBiases, CellBody, CellOperators
from patchworks.graph import DomeSpec, build_graph
from patchworks.sandbox import CONTROL_HZ
from patchworks.bias_selection import (
    DEFAULT_DRIVE_CORRELATION,
    DEFAULT_SAFETY_FACTOR,
    TAU_QUANTILES,
    Band,
    DemoHorizons,
    Measurement,
    TargetRange,
    driven_trajectory,
    fold_margin_check,
    go_no_go,
    measure,
    operator_scale_rule,
    select,
    sweep,
)

SEED = 42

#: A dome small enough to select a body for in a test, with the same shape of
#: taper: vision lattices, a somatomotor column beside them, and a core whose
#: last level is the apex.
SMALL = DomeSpec(
    patch_grid=4,
    vision_sides=(2,),
    somatomotor_sizes=(4,),
    core_sizes=(6, 4),
    core_degree=5,
    apex_degree=3,
)


@pytest.fixture(scope="module")
def body():
    return CellBody(BodyShape(n=32, k=12), generator=torch.Generator().manual_seed(SEED))


@pytest.fixture(scope="module")
def drawn(body):
    return sweep(body, draws=256, ticks=24, generator=torch.Generator().manual_seed(SEED))


@pytest.fixture(scope="module")
def dome():
    return build_graph(SMALL)


class TestTheTargetRange:
    def test_the_derivation_is_fixed_and_the_numbers_are_not(self):
        # `05-timescales.md` deliberately does not fix the numbers: the demo is
        # open. What is fixed is that the fast end resolves the fastest
        # perturbation and the slow end outlasts the longest.
        horizons = DemoHorizons(fastest=3.0, longest=750.0)
        assert horizons.target == TargetRange(fastest=3.0, slowest=750.0)

    def test_a_horizon_ordering_that_is_not_one_is_refused(self):
        with pytest.raises(ValueError, match="must exceed the fastest"):
            DemoHorizons(fastest=14.0, longest=3.0)

    def test_bands_overlap_and_cover_the_range(self):
        bands = TargetRange(3.0, 300.0).bands(7, overlap=0.5)
        assert len(bands) == 7
        assert bands[0].lo == pytest.approx(3.0)
        assert bands[-1].hi == pytest.approx(300.0)
        # Adjacent levels overlap and only their distributions separate: the
        # taper's gradient is continuous, not seven discrete rates.
        for below, above in zip(bands, bands[1:]):
            assert above.lo < below.hi
            assert above.lo > below.lo

    def test_bands_tile_when_the_overlap_is_switched_off(self):
        bands = TargetRange(2.0, 128.0).bands(6, overlap=0.0)
        for below, above in zip(bands, bands[1:]):
            assert above.lo == pytest.approx(below.hi)

    def test_bands_are_log_spaced_because_tau_is_a_rate(self):
        bands = TargetRange(1.0, 1000.0).bands(3, overlap=0.0)
        widths = [math.log(b.hi) - math.log(b.lo) for b in bands]
        assert widths[0] == pytest.approx(widths[-1])

    def test_the_shallowest_level_gets_the_fastest_band(self):
        bands = TargetRange(3.0, 300.0).bands(4)
        assert [b.level for b in bands] == [1, 2, 3, 4]
        assert bands[0].hi < bands[-1].hi


class TestTheDrivenTrajectory:
    def test_the_operating_point_varies_rather_than_being_frozen(self, body):
        biases = CellBiases(body.shape, 8, generator=torch.Generator().manual_seed(1))
        stalks = driven_trajectory(
            body, biases, ticks=32, burn_in=0, generator=torch.Generator().manual_seed(2)
        )
        assert stalks.shape == (32, 8, body.shape.n)
        assert not torch.allclose(stalks[0], stalks[-1])

    def test_a_correlated_drive_moves_less_per_tick_than_an_uncorrelated_one(self, body):
        biases = CellBiases(body.shape, 16, generator=torch.Generator().manual_seed(1))
        step = lambda correlation: float(  # noqa: E731
            driven_trajectory(
                body,
                biases,
                ticks=64,
                burn_in=0,
                drive_correlation=correlation,
                generator=torch.Generator().manual_seed(3),
            )
            .diff(dim=0)
            .abs()
            .mean()
        )
        # The stand-in has to be able to *hold* an operating point, or region
        # dwell is one tick by construction and the precondition cannot be read.
        assert step(64.0) < step(1.0)


class TestWhatIsMeasured:
    def test_tau_is_reported_as_quantiles_never_moments(self, drawn):
        # `tau = -1/ln rho` diverges as `rho -> 1`, so a mean is the tail.
        assert drawn.measurement.tau.shape[-1] == len(TAU_QUANTILES)
        quantiles = drawn.measurement.tau
        assert bool((quantiles.diff(dim=-1) >= 0).all())

    def test_the_effective_timescale_is_the_distributions_centre(self, drawn):
        centre = drawn.measurement.tau[:, TAU_QUANTILES.index(0.5)]
        assert torch.equal(drawn.measurement.effective_timescale, centre)

    def test_region_dwell_is_reported_per_candidate_alongside_tau(self, drawn):
        dwell = drawn.measurement.dwell
        assert dwell.shape == drawn.measurement.effective_timescale.shape
        # Dwell is a run length in ticks: at least one, at most the run.
        assert float(dwell.min()) >= 1.0
        assert float(dwell.max()) <= drawn.measurement.ticks

    def test_an_undriven_cell_never_crosses_a_fold(self, body):
        # Freeze the drive and the chart settles; dwell is then the whole run,
        # which is the no-crossing extreme the precondition wants distance from.
        biases = CellBiases(body.shape, 4, generator=torch.Generator().manual_seed(4))
        still = measure(
            body,
            biases,
            ticks=32,
            burn_in=32,
            drive_correlation=1e6,
            drive_scale=1e-6,
            generator=torch.Generator().manual_seed(5),
        )
        assert float(still.dwell.min()) == 32.0

    def test_decay_is_realised_contraction_not_an_eigenvalue(self, drawn):
        m = drawn.measurement
        # The regional Jacobians are non-normal, so `rho` mis-states the rate on
        # the first ticks. `lambda` is the stability object and is measured on
        # the product; the two are different numbers on the same run.
        implied = -1.0 / m.effective_timescale
        assert not torch.allclose(m.contraction, implied, atol=1e-3)

    def test_the_fold_margin_reads_encode_alone(self, body):
        # `decode` is not on the chart's round trip, so moving its bias moves no
        # margin. Anything that read it would be bounding the wrong loop. Since
        # #138 `encode` is also the *only* map with folds at all -- `K` and
        # `decode` are linear -- so this is now the whole of what the margin can
        # be read from.
        biases = CellBiases(body.shape, 4, generator=torch.Generator().manual_seed(6))
        before = measure(
            body, biases, ticks=8, generator=torch.Generator().manual_seed(7)
        )
        with torch.no_grad():
            biases.decode_output_bias.add_(3.0)
        after = measure(body, biases, ticks=8, generator=torch.Generator().manual_seed(7))
        assert torch.equal(before.margin, after.margin)
        assert torch.equal(before.effective_timescale, after.effective_timescale)

    def test_moving_encodes_biases_does_move_the_margin(self, body):
        # The other half: a margin that never moved would be reading nothing.
        biases = CellBiases(body.shape, 4, generator=torch.Generator().manual_seed(6))
        before = measure(
            body, biases, ticks=8, generator=torch.Generator().manual_seed(7)
        )
        with torch.no_grad():
            biases.encode_hidden_bias.add_(3.0)
        after = measure(body, biases, ticks=8, generator=torch.Generator().manual_seed(7))
        assert not torch.equal(before.margin, after.margin)

    def test_the_operator_scale_multiplies_every_placed_timescale(self, body):
        # #140's coupling, which is why `a` is not free: the recurrence is
        # `K @ J_encode` and at construction `K = a.I`, so `a` scales every
        # eigenvalue and therefore every `tau` the rig places.
        biases = CellBiases(body.shape, 8, generator=torch.Generator().manual_seed(6))
        full = measure(
            body, biases, ticks=8, generator=torch.Generator().manual_seed(7)
        )
        half = measure(
            body,
            biases,
            ticks=8,
            operator_scale=0.5,
            generator=torch.Generator().manual_seed(7),
        )
        assert float(half.rho_median.median()) < float(full.rho_median.median())
        assert float(half.effective_timescale.median()) < float(
            full.effective_timescale.median()
        )
        # The margin is *not* asserted equal, and the reason is worth writing
        # down: `a` scales the advanced chart, and that chart is half of
        # `encode`'s input on the next tick, so `a` moves the operating point
        # the margin is read at. What the margin no longer carries is a second
        # map's folds -- that is the claim above, and it is read there by moving
        # `decode`'s bias rather than by scaling `K`.

    def test_the_rig_follows_the_bodys_dtype(self, body):
        # The body carries `device` and `dtype`; a rig that allocated its own
        # trajectory at the global default would refuse to measure one.
        shape = body.shape
        wide = CellBody(
            shape, generator=torch.Generator().manual_seed(16), dtype=torch.float64
        )
        biases = CellBiases(
            shape, 4, generator=torch.Generator().manual_seed(17), dtype=torch.float64
        )
        m = measure(wide, biases, ticks=8, generator=torch.Generator().manual_seed(18))
        assert m.effective_timescale.dtype == torch.float64

    def test_measurement_is_reproducible_from_its_seed(self, body):
        biases = CellBiases(body.shape, 8, generator=torch.Generator().manual_seed(8))
        runs = [
            measure(body, biases, ticks=16, generator=torch.Generator().manual_seed(9))
            for _ in range(2)
        ]
        assert torch.equal(runs[0].effective_timescale, runs[1].effective_timescale)
        assert torch.equal(runs[0].contraction, runs[1].contraction)


class TestContainmentAndTheSlowCap:
    def _measurement(self, tau, contraction, rho_median=None):
        column = torch.tensor(tau).unsqueeze(-1).expand(-1, len(TAU_QUANTILES))
        zeros = torch.zeros(len(tau))
        return Measurement(
            tau=column.contiguous(),
            dwell=torch.full((len(tau),), 4.0),
            contraction=torch.tensor(contraction),
            rho_median=(
                zeros if rho_median is None else torch.tensor(rho_median)
            ),
            finite=torch.ones(len(tau), dtype=torch.bool),
            rho_max=zeros,
            expansive=zeros,
            margin=zeros,
            ticks=16,
        )

    def test_a_divergent_candidate_is_never_contained(self):
        m = self._measurement([5.0, 5.0], [+0.01, -0.20])
        assert m.contained().tolist() == [False, True]

    def test_containment_is_negative_by_the_safety_factor(self):
        # A candidate whose product decays far slower than its regions imply is
        # being held up by non-normal transients: #27 measured those at 2.6x.
        m = self._measurement([5.0, 5.0], [-1.0 / 12.0, -1.0 / 30.0])
        assert m.contained(safety_factor=2.6).tolist() == [True, False]

    def test_the_slow_cap_is_the_slowest_contained_candidate(self):
        m = self._measurement([2.0, 9.0, 40.0], [-0.5, -0.2, -1e-4])
        # 40 is reachable and not usable; the cap is a lambda question.
        assert m.slow_cap() == pytest.approx(9.0)

    def test_a_body_that_contains_nothing_caps_at_zero(self):
        assert self._measurement([2.0], [+0.5]).slow_cap() == 0.0

    def test_a_safety_factor_below_one_is_refused(self):
        with pytest.raises(ValueError, match="above one"):
            self._measurement([2.0], [-0.5]).contained(safety_factor=0.9)

    def test_a_candidate_expansive_at_the_median_is_not_slow(self):
        # `tau` is clamped at the `rho -> 1` ceiling, so a candidate whose median
        # region is expansive would otherwise pass containment trivially and be
        # reported as the slowest thing the body holds. It is refused on `rho`,
        # which can say what the clamped `tau` cannot.
        m = self._measurement([2.0, 1e6], [-0.5, -1e-9], rho_median=[0.6, 1.4])
        assert m.contained().tolist() == [True, False]
        assert m.slow_cap() == pytest.approx(2.0)

    def test_a_trajectory_that_left_the_reals_is_not_contained(self, body):
        # An overflowed chart makes every pre-activation NaN, which reads
        # downstream as no unit active: a zero Jacobian, rho = 0, and the
        # fastest, most contained candidate in the sweep.
        #
        # The variance is 200 rather than #85's 40 because the conversion took
        # a nonlinear map off the round trip: with `K = a.I` the recurrence is
        # `encode` alone and a body that used to overflow at 40 now runs to
        # `rho ~ 14.9` and stays finite. Measured, not guessed -- 200 is the
        # first of 40/200/1000/5000 at which all 16 candidates leave the reals.
        wild = CellBody(
            body.shape,
            weight_variance=200.0,
            generator=torch.Generator().manual_seed(13),
        )
        biases = CellBiases(body.shape, 16, generator=torch.Generator().manual_seed(14))
        m = measure(wild, biases, ticks=16, generator=torch.Generator().manual_seed(15))
        assert not bool(m.finite.all())
        assert not bool(m.contained()[~m.finite].any())


class TestTheOperatorScaleRule:
    """`a`, the scalar in `K = a.I`, as a rule the rig produces (#140)."""

    def test_a_reachable_target_takes_the_largest_admissible_scale(self, body):
        # The rule as stated: *the largest value in the band for which
        # `slow_cap` still admits the target*. A target the body clears easily
        # is admitted at the ceiling, so the ceiling is what comes back.
        a = operator_scale_rule(
            body,
            target=TargetRange(0.2, 0.5),
            draws=64,
            ticks=8,
            generator=torch.Generator().manual_seed(SEED),
        )
        assert a == 1.0

    def test_an_unreachable_target_falls_back_to_the_ceiling_not_the_floor(
        self, body
    ):
        # The direction matters and is silent if it is wrong. What puts the rule
        # here is cells forgetting *too fast*, and the floor is the fastest `a`
        # in the band -- the opposite of the answer. The go/no-go reports the
        # shortfall; this does not hide it by choosing a small number.
        a = operator_scale_rule(
            body,
            target=TargetRange(3.0, 1000.0),
            draws=64,
            ticks=8,
            generator=torch.Generator().manual_seed(SEED),
        )
        assert a == 1.0

    def test_the_scale_it_returns_is_inside_the_band(self, body):
        for rho_k in (2.0, 4.0):
            a = operator_scale_rule(
                body,
                target=TargetRange(0.2, 0.5),
                rho_k=rho_k,
                draws=32,
                ticks=8,
                steps=4,
                generator=torch.Generator().manual_seed(SEED),
            )
            assert 1.0 / rho_k <= a <= 1.0

    def test_slow_cap_rises_with_the_scale(self, body):
        # What makes the scan's "first admissible from the top" the largest
        # admissible: `a` multiplies every eigenvalue, so it multiplies every
        # placed `tau` and the cap over them rises with it. Measured rather
        # than assumed -- the scan does not rely on it, but the rule reads
        # oddly if it is false.
        caps = [
            sweep(
                body,
                draws=64,
                ticks=8,
                operator_scale=scale,
                generator=torch.Generator().manual_seed(SEED),
            ).measurement.slow_cap()
            for scale in (0.5, 0.75, 1.0)
        ]
        assert caps[0] < caps[1] < caps[2]

    def test_a_band_below_one_is_refused(self, body):
        with pytest.raises(ValueError, match="rho_k >= 1"):
            operator_scale_rule(
                body, target=TargetRange(0.2, 0.5), rho_k=0.5, draws=8, ticks=4
            )


class TestSelection:
    def test_selection_is_not_a_draw(self, body, drawn):
        # The whole claim of selecting over drawing: the kept set is narrower in
        # tau than the draw it came out of. Bands wide enough to fill from this
        # body, so that what is being tested is the selection and not the body.
        target = TargetRange(0.9, 1.3)
        dome = build_graph(SMALL)
        selection = select(dome, drawn, target=target)
        kept = selection.measurement.effective_timescale
        assert selection.biases.cells > 0
        assert float(kept.max()) <= 1.3 and float(kept.min()) >= 0.9
        assert float(kept.std()) < float(drawn.measurement.effective_timescale.std())

    def test_every_kept_candidate_lands_in_its_own_levels_band(self, dome, drawn):
        selection = select(dome, drawn, target=TargetRange(0.6, 1.6))
        bands = {band.level: band for band in selection.bands}
        tau = selection.measurement.effective_timescale
        for level, value in zip(selection.levels, tau):
            assert bands[level].lo <= float(value) <= bands[level].hi

    def test_the_kept_set_is_written_onto_predicting_cells_only(self, dome, drawn):
        selection = select(dome, drawn, target=TargetRange(0.6, 1.6))
        assert set(selection.cells) <= set(dome.predicting)
        assert len(set(selection.cells)) == len(selection.cells)
        assert selection.biases.cells == len(selection.cells)

    def test_no_candidate_is_kept_twice(self, dome, drawn):
        selection = select(dome, drawn, target=TargetRange(0.5, 2.0))
        rows = selection.biases.encode_hidden_bias
        unique = torch.unique(rows, dim=0)
        assert unique.shape[0] == rows.shape[0]

    def test_an_unreachable_band_is_reported_and_not_filled_from_its_neighbour(
        self, dome, drawn
    ):
        # A band no draw reaches is a result. Substituting from a neighbour would
        # manufacture the very gradient the construction is supposed to place.
        selection = select(dome, drawn, target=TargetRange(1.0, 1e6))
        assert not selection.filled
        apex = max(selection.bands, key=lambda band: band.level)
        assert apex.level in selection.shortfall
        assert apex.level not in selection.levels

    def test_the_bands_are_one_per_predicting_level(self, dome, drawn):
        selection = select(dome, drawn, target=TargetRange(0.6, 1.6))
        levels = {dome.cells[c].index.level for c in dome.predicting}
        assert [band.level for band in selection.bands] == sorted(levels)

    def test_a_selected_population_drops_into_the_body(self, body, dome, drawn):
        selection = select(dome, drawn, target=TargetRange(0.6, 1.6))
        cells = selection.biases.cells
        chart = torch.zeros((cells, body.shape.k))
        stalk = torch.zeros((cells, body.shape.n))
        operators = CellOperators(body.shape, cells)
        advanced, predicted = body(chart, stalk, selection.biases, operators)
        assert advanced.shape == (cells, body.shape.k)
        assert predicted.shape == (cells, body.shape.n)

    def test_nothing_records_a_rate_on_the_kept_biases(self, dome, drawn):
        # ADR-0005's prohibition is about runtime: the placement happens once,
        # here, and leaves no rate for anything to consult afterwards.
        selection = select(dome, drawn, target=TargetRange(0.6, 1.6))
        assert not any(
            "tau" in name or "timescale" in name or "band" in name
            for name, _ in selection.biases.named_parameters()
        )
        assert not any(
            "tau" in name or "timescale" in name or "band" in name
            for name, _ in selection.biases.named_buffers()
        )


class TestTheFoldMarginCheck:
    def test_the_bound_tightens_where_total_mask_width_falls(self, dome):
        # `gain_v = gamma / max(sum_e m_e, rho^2 deg(v))` is largest at the apex
        # because `sum_e m_e` falls with depth, so the bound binds hardest
        # exactly where the slow cells are meant to live. Equal margins isolate
        # that structural half from the per-cell draw.
        margins = torch.ones(len(dome.predicting))
        check = fold_margin_check(dome, margins, dome.predicting)
        by_level = {level: median for level, _, median, _ in check.by_level()}
        assert by_level[check.apex_level] < by_level[min(by_level)]
        assert check.apex_binds

    def test_the_tightest_cell_caps_gamma_globally(self, dome):
        margins = torch.ones(len(dome.predicting))
        margins[3] = 1e-4
        check = fold_margin_check(dome, margins, dome.predicting)
        assert check.binding == 3
        assert check.cap == pytest.approx(
            float(check.product_cap.min()), rel=1e-6
        )
        assert check.gamma_cap(1.0) < 1.0

    def test_gamma_is_never_raised_above_one_by_a_slack_margin(self, dome):
        check = fold_margin_check(dome, torch.ones(len(dome.predicting)), dome.predicting)
        assert check.gamma_cap(1e-9) == 1.0

    def test_a_floor_that_is_not_a_floor_is_refused(self, dome):
        check = fold_margin_check(dome, torch.ones(len(dome.predicting)), dome.predicting)
        with pytest.raises(ValueError, match="positive"):
            check.gamma_cap(0.0)

    def test_one_margin_per_cell_is_required(self, dome):
        with pytest.raises(ValueError, match="one fold margin per cell"):
            fold_margin_check(dome, torch.ones(3), dome.predicting)

    def test_the_check_covers_the_whole_taper_on_a_run_that_kept_nothing(
        self, dome, body
    ):
        # The bound binds hardest at the apex, so a check run over only the
        # shallow levels would report a looser cap than the taper permits --
        # and a run that keeps nothing is exactly when that would go unnoticed.
        run = go_no_go(
            dome,
            body,
            horizons=DemoHorizons(fastest=1e4, longest=1e5),
            draws=8,
            ticks=8,
            generator=torch.Generator().manual_seed(12),
        )
        assert run.selection.biases.cells == 0
        assert run.margin_check_on_draws
        assert run.margin_check.cells == dome.predicting
        assert run.margin_check.apex_level in run.margin_check.levels


@pytest.fixture(scope="module")
def run(dome, body):
    return go_no_go(
        dome,
        body,
        horizons=DemoHorizons(fastest=3.0, longest=14.0),
        draws=256,
        ticks=24,
        generator=torch.Generator().manual_seed(SEED),
    )


class TestTheGoNoGo:
    def test_a_band_no_draw_reaches_is_a_kill(self, run):
        # The falsification condition for ADR-0005, and the reason the run is
        # cheap: it can kill the mechanism before anything is trained.
        assert run.kill
        assert run.usable[-1] == 0.0
        assert "KILL" in run.report()

    def test_the_kill_is_reported_rather_than_worked_around(self, run):
        # sigma_w^2 is the lever that would reach the band and it is spoken for:
        # containment only, never spread.
        assert run.selection.shortfall
        assert "sigma_w^2" in run.report()

    def test_acceptance_is_reported_per_band(self, run):
        assert len(run.acceptance) == len(run.selection.bands)
        assert all(0.0 <= rate <= 1.0 for rate in run.acceptance)
        # Usable is acceptance minus what containment refuses, so it is never
        # the larger of the two.
        assert all(u <= a for u, a in zip(run.usable, run.acceptance))

    def test_the_target_range_is_not_narrowed_onto_what_was_reached(self, run):
        # A target narrowed onto the result cannot be missed, which would make
        # the go/no-go unfalsifiable.
        assert run.target.slowest == run.horizons.longest
        assert run.slow_cap < run.target.slowest

    def test_the_margin_check_runs_whatever_the_timescale_arm_says(self, run):
        # Same sweep, same afternoon: `02-tick-semantics.md` needs its answer
        # even on a run that kills the mechanism the sweep was drawn for.
        assert run.margin_check_on_draws
        assert run.margin_check.cap > 0
        assert "gamma x floor < fold margin" in run.report()

    def test_the_report_names_the_precondition_when_dwell_is_short(self, run):
        # Dwell of a tick against a tau of about a tick is not "long against"
        # under any reading, and the report has to say so: a band reached by a
        # cell averaging over unrelated regions is not this mechanism.
        assert float(
            run.sweep.measurement.dwell.median()
        ) < DEFAULT_SAFETY_FACTOR * float(
            run.sweep.measurement.effective_timescale.median()
        )
        assert "The precondition fails" in run.report()

    def test_a_second_reading_reuses_the_sweep_rather_than_redrawing(
        self, dome, body, drawn
    ):
        # What differs between readings of the demo's horizons is the target
        # range; the measurement does not depend on it, so a run may be put to
        # a sweep already taken.
        runs = [
            go_no_go(
                dome, body, horizons=DemoHorizons(3.0, longest), drawn=drawn
            )
            for longest in (14.0, 750.0)
        ]
        assert runs[0].sweep is runs[1].sweep
        assert runs[0].draws == drawn.candidates.cells
        assert runs[0].slow_cap == runs[1].slow_cap
        assert runs[0].target.slowest != runs[1].target.slowest

    def test_the_run_is_reproducible_from_its_seed(self, dome, body):
        runs = [
            go_no_go(
                dome,
                body,
                horizons=DemoHorizons(fastest=3.0, longest=14.0),
                draws=64,
                ticks=16,
                generator=torch.Generator().manual_seed(11),
            )
            for _ in range(2)
        ]
        assert runs[0].report() == runs[1].report()


class TestTheMeasuredBody:
    """What this body measured when #85 ran it, so a later change announces itself.

    Recorded rather than asserted as a requirement: these are measurements of a
    body whose widths and `sigma_w^2` are fixed elsewhere, and the tolerances are
    wide enough that only a real change to the body or the rig trips them.
    """

    def test_the_effective_timescale_sits_around_one_tick(self, drawn):
        tau = drawn.measurement.effective_timescale
        assert 0.7 < float(torch.quantile(tau, 0.5)) < 1.3
        assert float(tau.max()) < 10.0

    def test_the_contraction_the_conversion_left(self, drawn):
        # **Superseded, and the supersession is the point.** The prototype it
        # was promoted from measured a round trip through `encode` *and* a
        # nonlinear `step` -- `selection_sweep.trajectory_lambda([45], [13],
        # 1.2)` reported lam_med = -1.47 -- and #138 took `step` off that loop.
        # The round trip is now `encode` then `a.I`, so it contracts less, and
        # the old window cannot be met by the body that actually runs.
        #
        # Re-measured on this fixture at `a = 1.0`: median -0.794, p05 -0.932,
        # p95 -0.589. Recorded rather than asserted as a requirement, with a
        # window wide enough that only a real change to the body or the rig
        # trips it.
        assert -1.0 < float(drawn.measurement.contraction.median()) < -0.6

    def test_the_median_fold_margin_is_the_recorded_one(self, drawn):
        # `01-cell-and-sheaf.md` records 0.019 for [45]/[13] at sigma_w^2 = 1.2,
        # read with the pre-#206 denominator. Since #206 the margin divides by
        # the hidden row's node stalk block alone, which is 1.183x looser, so
        # the band moves with the definition and the recorded figure does not:
        # rescaling it would publish a number nobody ran.
        assert 0.022 < float(drawn.measurement.margin.median()) < 0.034

    def test_region_dwell_is_one_tick_under_a_plausible_drive(self, drawn):
        assert float(drawn.measurement.dwell.median()) < 1.5

    def test_the_cheap_sufficient_check_holds_where_it_is_read(self, drawn):
        # `05-timescales.md` calls `max rho < 1` sufficient for `lambda < 0`.
        # It is not a theorem — a product of individually contracting non-normal
        # matrices can grow, which is the same non-normality the safety factor
        # is stated for — so this is a measurement of a body rather than an
        # invariant, and it lives here with the other measurements.
        m = drawn.measurement
        assert bool((m.contraction[m.rho_max < 1.0] < 0).all())


class TestTheBand:
    def test_a_band_holds_what_lands_in_it(self):
        band = Band(level=3, lo=2.0, hi=4.0)
        held = band.holds(torch.tensor([1.9, 2.0, 3.0, 4.0, 4.1]))
        assert held.tolist() == [False, True, True, True, False]


class TestTheConstructionRun:
    """`benchmarks/timescale_selection.py`, the run that reports the verdict.

    The sweep itself is minutes of trajectories, so what is asserted here is the
    shape of the run: that both readings of `05-timescales.md`'s derivation are
    put to the body rather than one being chosen quietly, and that each of their
    numbers still traces to what it was read from.
    """

    def test_both_readings_of_the_derivation_are_run(self):
        assert set(timescale_selection.READINGS) == {"onset", "duration"}

    def test_the_fast_end_is_the_reflex_loop_under_both(self):
        # Three ticks, from `06-graph-topology.md` and the arm nudge's expected
        # onset in `08-the-acceptance-demo.md`.
        assert {r.fastest for r in timescale_selection.READINGS.values()} == {3.0}

    def test_the_duration_reading_is_a_measured_task(self):
        # `benchmarks/achievability.py` reports ~15 s of sim to solve, and the
        # sandbox's control rate turns that into ticks.
        longest = timescale_selection.READINGS["duration"].longest
        assert longest == pytest.approx(15.0 * CONTROL_HZ)

    def test_the_two_readings_differ_by_orders_of_magnitude(self):
        # Which is why both are run: a verdict that turned on the choice would
        # be a verdict about the reading rather than about the body.
        onset = timescale_selection.READINGS["onset"].longest
        assert timescale_selection.READINGS["duration"].longest / onset > 50

    def test_the_drive_sweep_starts_from_the_rigs_default(self):
        assert timescale_selection.DRIVES[0][0] == DEFAULT_DRIVE_CORRELATION
        # And walks toward a frozen operating point, which is the most
        # favourable case for a slow effective timescale and not a plausible one.
        assert [scale for _, scale in timescale_selection.DRIVES] == sorted(
            (scale for _, scale in timescale_selection.DRIVES), reverse=True
        )
