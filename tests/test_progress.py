"""`patchworks run`: the headless readout, and what it has to be able to show (#121).

Three of these are the ones that matter.

**Reporting changes no trajectory.** The whole reason a readout may be left on
for an hours-long run is that it cannot alter the run. `TestReportingIsFree`
drives the same seed twice — once with the reporting on and once with no
:class:`~patchworks.progress.Progress` at all — and asserts the two trajectories
are equal *bit for bit*, over the command, the efference copy and every float
the world answered with.

**The pair stays a pair.** Everything the readout says about the graph comes out
of #91's paired instrument, and `TestThePairedInstrumentIsTheOneFrom91` holds
that down at both ends: no energy is computed here, and no line prints an energy
without the effective rank beside it.

**#120 is visible.** The ticket's acceptance is that a human reading this output
on the locked-loop case would see that it is stuck.
`TestTheLockedLoopIsVisible` builds that exact signature — a command frozen to
one part in 10⁵, an arm that does not move, and every edge still disagreeing —
feeds it through the real :class:`~patchworks.progress.Progress`, and asserts the
report says so in numbers and the summary says so in words.

The dome is `tests/conftest.py`'s small one throughout, at a 16×16 render. The
full dome for tens of thousands of ticks is what the subcommand's own defaults
do; this file is in CI on every push.
"""

import random
import signal
import threading

import numpy as np
import pytest
import torch

from patchworks import progress as progress_module
from patchworks.agent import Agent, TickOutcome
from patchworks.diagnostics import Condition
from patchworks.graph import build_graph
from patchworks.progress import (
    WHOLE_GRAPH_OFF,
    Progress,
    Report,
    Summary,
    drive,
    format_report,
    format_summary,
    header,
    stationary_reports,
    stopping_on_interrupt,
    ticking,
)
from patchworks.tick import Sheaf

from conftest import SMALL

IMAGE_SIZE = 16
TICKS = 12
EVERY = 4


@pytest.fixture(scope="module")
def dome():
    return build_graph(SMALL)


@pytest.fixture
def sheaf(dome):
    """A sheaf with content in its stalks, standing in for the world's write.

    The same stand-in `tests/test_diagnostics.py` uses, and for the same reason:
    a fresh sheaf's stalks are all zero, every edge agrees exactly, and a
    readout measured on that says nothing about a readout.
    """
    built = Sheaf(dome, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(1)
    built.stalks = torch.randn_like(built.stalks)
    built.stalks[built.layout.pad] = 0.0
    return built


def an_outcome(command=(0.1, -0.2, 0.3), qpos=(0.0, 0.0, 0.0), applied=None):
    """A `TickOutcome` with nothing in it but what the readout reads."""
    command = np.array(command, dtype=np.float32)
    return TickOutcome(
        command=command,
        applied=command if applied is None else np.array(applied, dtype=np.float32),
        observation={"qpos": np.array(qpos, dtype=np.float64)},
        info={},
    )


def a_report(**overrides):
    """One report, with every field filled and any of them replaceable."""
    fields = dict(
        tick=500,
        since=500,
        seconds=10.0,
        rate=50.0,
        elapsed=10.0,
        energy_mean=1.6615,
        energy_max=4.909,
        disagreeing_edges=682,
        edges=682,
        effective_rank_mean=3.9,
        effective_rank_min=1.0,
        commanded_mean_abs=0.368,
        applied_mean_abs=0.368,
        commanded_sd=(7e-06, 1e-06, 3e-06),
        arm_travel=(0.0, 0.0, 0.0),
        non_finite=(),
    )
    fields.update(overrides)
    return Report(**fields)


def a_summary(**overrides):
    reports = overrides.pop("reports", (a_report(),))
    fields = dict(
        ticks=1500,
        asked_for=1500,
        seed=0,
        split="train",
        control_hz=50.0,
        elapsed=30.0,
        interrupted=False,
        reporting_seconds=0.3,
        setup_seconds=4.0,
        reports=reports,
        non_finite=(),
    )
    fields.update(overrides)
    return Summary(**fields)


# ---------------------------------------------------------------------------


class TestReportingIsFree:
    """The property the whole subcommand rests on: the readout changes nothing."""

    def test_reporting_off_gives_a_bit_identical_trajectory(self, dome):
        """The ticket's acceptance criterion, asserted bit for bit.

        Not `approx`, and not on the command alone. If the readout perturbed the
        run at all — a global RNG draw, a tensor edited in place, a stalk read
        that was really a write — the two runs would part somewhere, and the
        cheapest place to notice it is the first float that differs anywhere.
        """
        trajectories = []
        for reporting in (True, False):
            world = _world()
            try:
                agent = Agent(
                    world, dome=dome, generator=torch.Generator().manual_seed(0)
                )
                watching = (
                    Progress(agent.sheaf, every=EVERY, out=None) if reporting else None
                )
                trajectories.append(
                    [
                        (
                            outcome.command.copy(),
                            outcome.applied.copy(),
                            {
                                key: np.asarray(value).copy()
                                for key, value in outcome.observation.items()
                            },
                        )
                        for outcome in ticking(
                            agent, TICKS, seed=0, progress=watching
                        )
                    ]
                )
            finally:
                world.close()

        reported, silent = trajectories
        assert len(reported) == len(silent) == TICKS
        for tick, (one, two) in enumerate(zip(reported, silent)):
            assert np.array_equal(one[0], two[0]), f"the command parted at tick {tick}"
            assert np.array_equal(one[1], two[1]), f"the applied parted at tick {tick}"
            assert one[2].keys() == two[2].keys()
            for key in one[2]:
                assert np.array_equal(one[2][key], two[2][key]), (
                    f"observation[{key!r}] parted at tick {tick}"
                )

    def test_nothing_is_drawn_from_a_global_rng(self, dome):
        """Running this changes no trajectory of anything run after it (#77).

        Asserted against all three global generators, because the code that must
        not draw does not get to pick which one it would have drawn from — the
        same guard `tests/test_cli.py` puts on `check`.
        """
        before = (
            torch.random.get_rng_state().clone(),
            np.random.get_state(),
            random.getstate(),
        )
        drive(
            ticks=TICKS,
            seed=0,
            report_every=EVERY,
            dome=dome,
            image_size=IMAGE_SIZE,
            out=_nowhere(),
        )
        after = (torch.random.get_rng_state(), np.random.get_state(), random.getstate())
        assert torch.equal(before[0], after[0]), "torch's global generator moved"
        assert before[1][1].tolist() == after[1][1].tolist(), "numpy's global state moved"
        assert before[2] == after[2], "the stdlib's global random state moved"

    def test_reporting_off_builds_no_instrument_at_all(self, dome):
        """`reporting=False` is no `Progress`, not a silenced one.

        The difference is the point: constructing the instrument costs a
        decomposition, and a run that has switched the readout off should not be
        paying for one.
        """
        built = []
        original = progress_module.Progress

        class Counted(original):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                built.append(1)
                super().__init__(*args, **kwargs)

        try:
            progress_module.Progress = Counted
            summary = drive(
                ticks=TICKS,
                seed=0,
                reporting=False,
                dome=dome,
                image_size=IMAGE_SIZE,
                out=_nowhere(),
            )
        finally:
            progress_module.Progress = original
        assert built == []
        assert summary.reports == ()
        assert summary.reporting_seconds == 0.0


class TestThePairedInstrumentIsTheOneFrom91:
    """#91's reading, used rather than re-derived, and never printed by halves."""

    def test_every_graph_number_comes_from_a_reading(self, sheaf):
        """The report's energies and ranks are the reading's, to the last bit."""
        watching = Progress(sheaf, every=1, out=None)
        sheaf.tick()
        report = watching.after(an_outcome())
        assert report is not None
        reading = watching.diagnostics.readings[-1]
        assert reading.condition is Condition.DRIVEN
        assert report.energy_mean == float(reading.edges.energy.mean())
        assert report.energy_max == float(reading.edges.energy.max())
        assert report.effective_rank_mean == float(reading.edges.effective_rank.mean())
        assert report.effective_rank_min == float(reading.edges.effective_rank.min())
        assert report.edges == len(reading.edges)

    def test_no_line_shows_an_energy_without_a_rank(self):
        """The pairing, at the one place it could be broken: the printing.

        A readout that showed the energy falling and left the rank off the line
        would be the one-reading instrument #91 exists to refuse — the reader
        could not tell collapse from a draining lag floor, which is exactly the
        distinction the columns are there to make.
        """
        headings = header()
        for column in ("energy mean", "energy max", "rank mean", "rank min"):
            assert column in headings
        # One line, carrying both halves: the energies and the ranks that were
        # read from the same tick's configuration.
        line = format_report(
            a_report(
                energy_mean=1.25, energy_max=4.5, effective_rank_mean=3.9,
                effective_rank_min=1.75,
            )
        )
        assert "1.25" in line and "4.5" in line
        assert "3.900" in line and "1.750" in line

    def test_the_summary_quotes_both_halves_and_says_how_to_read_them(self):
        printed = format_summary(a_summary())
        assert "per-edge energy, mean" in printed
        assert "effective rank, mean" in printed
        assert "effective rank, min" in printed
        assert "collapse" in printed

    def test_the_columns_and_the_values_cannot_drift_apart(self):
        """Headings and values are laid out by one table, so they line up.

        A readout whose headings have slipped a column is worse than one with no
        headings at all, so the alignment is a property of the code rather than
        of somebody counting spaces.
        """
        headings = header()
        # Including the widest each column has to hold on a real run: a tick
        # count a long run could plausibly reach, and an energy in exponent
        # form at either end of the scale.
        for report in (
            a_report(),
            a_report(tick=10**9, energy_mean=1e-12, energy_max=1.234e12),
            a_report(rate=0.0, disagreeing_edges=0, non_finite=("somewhere",)),
        ):
            line = format_report(report)
            assert len(line) == len(headings), f"the columns slipped: {line}"


class TestTheExpensiveHalfStaysOffTheFastPath:
    """`whole_graph()` is a `3764 x 3764` decomposition; it is opt-in and on a cadence."""

    def test_it_is_off_by_default(self, sheaf):
        watching = Progress(sheaf, every=1, out=None)
        assert watching.whole_graph_every == WHOLE_GRAPH_OFF
        for _ in range(3):
            sheaf.tick()
            report = watching.after(an_outcome())
            assert report is not None and report.whole_graph is None
        assert all(
            reading.whole_graph is None for reading in watching.diagnostics.readings
        )

    def test_asked_for_it_lands_on_its_own_cadence_and_on_a_report(self, sheaf):
        watching = Progress(sheaf, every=2, whole_graph_every=4, out=None)
        taken = {}
        for _ in range(8):
            sheaf.tick()
            report = watching.after(an_outcome())
            if report is not None:
                taken[report.tick] = report.whole_graph is not None
        assert taken == {2: False, 4: True, 6: False, 8: True}

    def test_a_cadence_that_is_not_a_multiple_is_refused(self, sheaf):
        """#91's multiple-of rule, kept: a whole-graph reading must land on a report.

        Otherwise the minimum achievable energy would be quoted against a
        configuration no line on the table ever measured.
        """
        with pytest.raises(ValueError, match="must be a multiple"):
            Progress(sheaf, every=3, whole_graph_every=4)

    @pytest.mark.parametrize("every", [0, -1, 1.5, True])
    def test_a_cadence_that_is_not_a_count_of_ticks_is_refused(self, sheaf, every):
        with pytest.raises(ValueError, match="cadence in ticks"):
            Progress(sheaf, every=every)

    def test_a_negative_whole_graph_cadence_is_refused(self, sheaf):
        with pytest.raises(ValueError, match="cadence in ticks"):
            Progress(sheaf, every=1, whole_graph_every=-2)


class TestTheCadenceIsCountedOnTicks:
    """Reports land on tick numbers, so two runs are comparable at the same ticks."""

    def test_reports_land_on_multiples_of_the_cadence(self, sheaf):
        watching = Progress(sheaf, every=3, out=None)
        ticks = []
        for _ in range(10):
            sheaf.tick()
            report = watching.after(an_outcome())
            if report is not None:
                ticks.append(report.tick)
        assert ticks == [3, 6, 9]
        assert [report.tick for report in watching.reports] == ticks

    def test_a_report_can_be_taken_off_the_cadence(self, sheaf):
        """What an interrupted run needs: a reading at the tick it stopped on."""
        watching = Progress(sheaf, every=100, out=None)
        for _ in range(7):
            sheaf.tick()
            assert watching.after(an_outcome()) is None
        report = watching.report(an_outcome())
        assert report.tick == 7
        assert watching.reports == [report]


class TestWhatTheWindowMeasures:
    """Travel and spread are *since the last report*, and neither loses an interval."""

    def test_travel_is_cumulative_and_resets_each_window(self, sheaf):
        """An arm that swings out and back has travelled; its net move is zero.

        The number the "it is stuck" reading hangs on has to mean travel, or a
        working run reports itself as frozen.
        """
        watching = Progress(sheaf, every=2, out=None)
        poses = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        reports = []
        for pose in poses:
            sheaf.tick()
            report = watching.after(an_outcome(qpos=pose))
            if report is not None:
                reports.append(report)
        first, second = reports
        assert first.arm_travel == pytest.approx((1.0, 0.0, 0.0))
        # 1 -> 0 is the step *between* the windows: counted in the second, not
        # dropped for having straddled a report.
        assert second.arm_travel == pytest.approx((1.0, 0.0, 0.0))
        assert second.travel == pytest.approx(1.0)

    def test_a_frozen_command_reads_zero_spread_and_a_moving_one_does_not(self, sheaf):
        watching = Progress(sheaf, every=4, out=None)
        for _ in range(4):
            sheaf.tick()
            frozen = watching.after(an_outcome(command=(0.5, -0.5, 0.25)))
        assert frozen is not None
        assert frozen.commanded_sd == (0.0, 0.0, 0.0)
        assert frozen.command_spread == 0.0

        for value in (0.1, 0.9, 0.1, 0.9):
            sheaf.tick()
            moving = watching.after(an_outcome(command=(value, 0.0, 0.0)))
        assert moving is not None
        assert moving.commanded_sd[0] == pytest.approx(0.4)

    def test_the_spread_never_comes_back_negative(self, sheaf):
        """The sum-of-squares form goes a few ulp negative exactly where it matters.

        A command that never moves is the case this column is read for, and a
        `sqrt` of `-1e-17` there would print `nan` on the one line somebody was
        relying on.
        """
        watching = Progress(sheaf, every=3, out=None)
        for _ in range(3):
            sheaf.tick()
            report = watching.after(an_outcome(command=(1e8, -1e8, 1e8)))
        assert report is not None
        assert all(value >= 0.0 for value in report.commanded_sd)
        assert all(np.isfinite(value) for value in report.commanded_sd)

    def test_the_torque_is_reported_asked_and_applied(self, sheaf):
        """Pre-clip and post-clip are two quantities, and the gap is the diagnosis."""
        watching = Progress(sheaf, every=1, out=None)
        sheaf.tick()
        report = watching.after(
            an_outcome(command=(2.0, 2.0, 2.0), applied=(1.0, 1.0, 1.0))
        )
        assert report is not None
        assert report.commanded_mean_abs == pytest.approx(2.0)
        assert report.applied_mean_abs == pytest.approx(1.0)


class TestTheNonFiniteSweepFindsAndNames:
    """Where, not whether: a bug report that says where is worth more."""

    def test_a_non_finite_command_is_caught_on_the_tick_it_happens(self, sheaf):
        watching = Progress(sheaf, every=100, out=None)
        sheaf.tick()
        watching.after(an_outcome(command=(0.1, 0.1, 0.1)))
        sheaf.tick()
        watching.after(an_outcome(command=(float("nan"), 0.1, 0.1)))
        assert any("command, first at tick 2" in where for where in watching.non_finite)

    def test_a_non_finite_observation_is_named_by_its_key(self, sheaf):
        watching = Progress(sheaf, every=1, out=None)
        sheaf.tick()
        watching.after(an_outcome(qpos=(float("inf"), 0.0, 0.0)))
        assert any("observation['qpos']" in where for where in watching.non_finite)

    def test_a_non_finite_stalk_is_named_even_though_the_command_is_fine(self, sheaf):
        """The graph can go bad before anything reaches the actuator."""
        watching = Progress(sheaf, every=1, out=None)
        sheaf.stalks[0] = float("nan")
        sheaf.tick()
        watching.after(an_outcome())
        assert any("stalks" in where for where in watching.non_finite)

    def test_a_clean_run_names_nothing(self, sheaf):
        watching = Progress(sheaf, every=1, out=None)
        for _ in range(3):
            sheaf.tick()
            watching.after(an_outcome())
        assert watching.non_finite == ()

    def test_the_summary_says_it_is_a_bug_rather_than_an_untrained_agent(self):
        printed = format_summary(a_summary(non_finite=("command, first at tick 4",)))
        assert "NON-FINITE" in printed
        assert "That is a bug, not an untrained agent" in printed


class TestTheLockedLoopIsVisible:
    """#120's signature, through the real readout, end to end.

    The ticket's last acceptance criterion: run the readout on the locked case
    and confirm a human reading the output would see that it is stuck. Built here
    rather than waited for, because the locked loop takes several hundred ticks of
    the real dome to arrive at and the readout's job is to be legible when it does.
    """

    @pytest.fixture
    def locked(self, sheaf):
        """A frozen command against a pinned arm, with every edge still disagreeing."""
        watching = Progress(sheaf, every=5, out=None)
        for step in range(20):
            sheaf.tick()
            # Constant to one part in 10^5, which is what #120 measured -- not
            # exactly constant, so nothing here can pass by way of a special case
            # for an exactly repeated float.
            jitter = 7e-06 * (step % 2)
            watching.after(
                an_outcome(
                    command=(-0.2341 + jitter, -0.5523, 0.3187),
                    qpos=(-3.142, -2.601, 2.602),
                )
            )
        return watching

    def test_the_columns_a_human_reads_are_the_ones_that_show_it(self, locked):
        last = locked.reports[-1]
        assert last.travel == 0.0, "a pinned arm has to read as zero travel"
        assert last.command_spread < 1e-04, "a frozen command has to read as no spread"
        assert last.disagreeing_edges == last.edges, "and the error signal is still up"
        assert last.energy_mean > 0.0

        line = format_report(last)
        assert "0.000" in line, f"the travel column is not visibly zero: {line}"
        assert f"{last.edges}/{last.edges}" in line

    def test_the_summary_says_stuck_in_words(self, locked):
        printed = format_summary(a_summary(reports=tuple(locked.reports)))
        assert "STUCK" in printed
        assert "#120" in printed
        assert "moved the arm exactly 0 radians" in printed
        assert "disagreeing" in printed

    def test_a_run_that_is_moving_is_not_called_stuck(self, sheaf):
        """The reading that would make the verdict worthless is the false positive."""
        watching = Progress(sheaf, every=2, out=None)
        for step in range(8):
            sheaf.tick()
            watching.after(an_outcome(qpos=(0.1 * step, 0.0, 0.0)))
        printed = format_summary(a_summary(reports=tuple(watching.reports)))
        assert "STUCK" not in printed

    def test_the_stationary_span_is_exact_zero_with_no_threshold_beside_it(self):
        """A tolerance here would be a number invented to make a verdict come out.

        An arm jittering at `1e-9` is a different case from one resting on its
        stop, and reporting it as the same one would be the readout claiming
        something it did not measure.
        """
        moving = a_report(arm_travel=(1e-9, 0.0, 0.0))
        still = a_report(arm_travel=(0.0, 0.0, 0.0))
        assert stationary_reports([still, still, moving]) == ()
        assert stationary_reports([moving, still, still]) == (still, still)
        assert stationary_reports([]) == ()


class TestInterruptionIsClean:
    """Ctrl-C asks the loop to stop; it does not tear it down mid-tick."""

    def test_the_flag_is_set_and_what_was_installed_is_restored(self):
        before = signal.getsignal(signal.SIGINT)
        with stopping_on_interrupt(_nowhere()) as stopping:
            assert stopping() is False
            assert signal.getsignal(signal.SIGINT) is not before
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            assert stopping() is True
        assert signal.getsignal(signal.SIGINT) is before

    def test_a_second_interrupt_is_handed_back_to_python(self):
        """An interruption guard that can itself hang is worse than none."""
        before = signal.getsignal(signal.SIGINT)
        try:
            with stopping_on_interrupt(_nowhere()):
                handler = signal.getsignal(signal.SIGINT)
                handler(signal.SIGINT, None)
                with pytest.raises(KeyboardInterrupt):
                    handler(signal.SIGINT, None)
                assert signal.getsignal(signal.SIGINT) is before
        finally:
            signal.signal(signal.SIGINT, before)

    def test_off_the_main_thread_it_installs_nothing_and_says_so(self):
        """`signal.signal` refuses off the main thread; that is not a crash here."""
        seen = []

        def elsewhere():
            with stopping_on_interrupt(_nowhere()) as stopping:
                seen.append(stopping())

        thread = threading.Thread(target=elsewhere)
        thread.start()
        thread.join()
        assert seen == [False]

    def test_the_loop_stops_at_the_end_of_the_tick_it_is_on(self, sheaf):
        """Not mid-tick: a world stopped halfway through corresponds to no report."""
        stopped = {"after": 3}
        seen = []

        def stopping():
            return len(seen) >= stopped["after"]

        agent = _FakeAgent(sheaf)
        for outcome in ticking(agent, 50, seed=0, stopping=stopping):
            seen.append(outcome)
        assert len(seen) == 3
        assert sheaf.ticks == 3

    def test_an_interrupted_run_still_prints_a_final_report(self, dome, capsys):
        """The reason the flag exists: the numbers survive the stop."""
        stream = _Capturing()
        summary = drive(
            ticks=TICKS,
            seed=0,
            report_every=1000,  # far beyond the run: nothing lands on the cadence
            dome=dome,
            image_size=IMAGE_SIZE,
            out=stream,
        )
        assert len(summary.reports) == 1
        assert summary.reports[0].tick == TICKS
        assert "patchworks run -- finished" in stream.text
        assert "reporting cost" in stream.text

    def test_the_final_report_is_not_printed_twice_for_the_same_tick(
        self, dome
    ):
        stream = _Capturing()
        summary = drive(
            ticks=TICKS,
            seed=0,
            report_every=TICKS,  # the cadence lands exactly on the last tick
            dome=dome,
            image_size=IMAGE_SIZE,
            out=stream,
        )
        assert [report.tick for report in summary.reports] == [TICKS]


class TestWhatTheRunReports:
    """The closing report's own claims, including the one about its cost."""

    def test_the_reporting_overhead_is_measured_and_stated(self, dome):
        """Measured on this run, on this machine — not a number from a README."""
        summary = drive(
            ticks=TICKS,
            seed=0,
            report_every=EVERY,
            dome=dome,
            image_size=IMAGE_SIZE,
            out=_nowhere(),
        )
        assert summary.reporting_seconds > 0.0
        assert summary.reporting_seconds < summary.elapsed
        printed = format_summary(summary)
        assert "reporting cost" in printed
        assert "measured here, not estimated" in printed

    def test_building_the_instrument_is_stated_separately_from_running_it(self, dome):
        """A four-second startup must not be able to hide inside "0.9% of the run".

        The `H^1` decomposition is paid once, before the first tick, and it is
        not part of the per-tick cost — so it is reported as its own line rather
        than pooled with one or left out.
        """
        summary = drive(
            ticks=TICKS,
            seed=0,
            report_every=EVERY,
            dome=dome,
            image_size=IMAGE_SIZE,
            out=_nowhere(),
        )
        assert summary.setup_seconds > 0.0
        assert "+ building it" in format_summary(summary)

    def test_a_run_with_reporting_off_pays_neither_cost(self, dome):
        summary = drive(
            ticks=TICKS,
            seed=0,
            reporting=False,
            dome=dome,
            image_size=IMAGE_SIZE,
            out=_nowhere(),
        )
        assert summary.reporting_seconds == 0.0
        assert summary.setup_seconds == 0.0

    def test_a_run_that_asks_for_no_ticks_reports_that_nothing_was_measured(self):
        printed = format_summary(a_summary(ticks=0, reports=()))
        assert "nothing above was measured" in printed

    def test_an_interrupted_run_is_named_as_interrupted_and_is_not_a_failure(self):
        printed = format_summary(a_summary(interrupted=True))
        assert "interrupted" in printed
        assert "NON-FINITE" not in printed

    def test_the_preamble_says_what_is_running_and_what_the_columns_mean(self, dome):
        stream = _Capturing()
        drive(
            ticks=TICKS,
            seed=0,
            report_every=EVERY,
            dome=dome,
            image_size=IMAGE_SIZE,
            out=stream,
        )
        assert "patchworks run --" in stream.text
        assert "Ctrl-C stops cleanly" in stream.text
        assert header() in stream.text
        assert "per-edge Dirichlet energy" in stream.text

    def test_the_same_seed_gives_the_same_run(self, dome):
        runs = [
            drive(
                ticks=TICKS,
                seed=3,
                report_every=EVERY,
                dome=dome,
                image_size=IMAGE_SIZE,
                out=_nowhere(),
            )
            for _ in range(2)
        ]
        first, second = (summary.reports[-1] for summary in runs)
        assert first.energy_mean == second.energy_mean
        assert first.arm_travel == second.arm_travel
        assert first.commanded_sd == second.commanded_sd


# ---------------------------------------------------------------------------
# stand-ins
# ---------------------------------------------------------------------------


def _world():
    """The small world the whole file runs against."""
    from patchworks.sandbox import PlanarPushSandbox

    return PlanarPushSandbox(split="train", image_size=IMAGE_SIZE)


class _Capturing:
    """A stream that keeps what was written, so a test can read the readout."""

    def __init__(self):
        self.lines: list[str] = []

    def write(self, text):
        self.lines.append(text)
        return len(text)

    def flush(self):
        pass

    @property
    def text(self):
        return "".join(self.lines)


def _nowhere():
    """A stream that keeps nothing, for the runs whose output is not the subject."""

    class Sink:
        def write(self, text):
            return len(text)

        def flush(self):
            pass

    return Sink()


class _FakeAgent:
    """An agent-shaped object that ticks the sheaf and touches no world.

    :func:`~patchworks.progress.ticking` is a loop over
    :func:`~patchworks.agent.run`, and what the stopping test is about is the
    loop rather than the physics.
    """

    def __init__(self, sheaf):
        self.sheaf = sheaf
        self.env = self

    def reset(self, seed=None):
        return {"qpos": np.zeros(3)}, {}

    def observe(self, observation):
        pass

    def tick(self):
        self.sheaf.tick()
        return an_outcome()
