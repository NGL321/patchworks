"""The hermetic half of the rig report's cutoff hook (#284).

`tools/cutoff_report.py` is the *measurement* half of the cutoff mechanism. The
`event` half fires by itself — `constant-provenance.yml` watches
`issues: closed` — and the measurement half cannot, because **rigs do not run in
CI and must not learn to**. So the evaluation moves to the report: when a rig
runs, it states, for each open problem whose `@cutoff` names it, whether the bar
was crossed.

Nothing here shells `gh` and nothing here may learn how, for the reason
`tests/test_problem_registers.py` gives: a register test that talked to GitHub
would be the first in this suite to fail offline. The module is built around the
same seam — everything above :func:`~cutoff_report.file_report` is a pure
function of the rendered register and a dict of readings — and this file holds
that half.

**The reader is held against the renderer.** The rig reads the *rendered*
`docs/registers/open-problems.md`, because that needs no token, which is one of
the reasons the register is a checked-in file rather than a live query. That
makes the render a contract between two modules, so the fixtures here are
produced by `tools/problem_registers.py` itself rather than typed out in
imitation of it: a change to either side breaks here, loudly, instead of quietly
leaving a rig that watches nothing.

**A rig asserting nothing still holds.** Nothing here reaches an exception or an
exit code; a crossing is a report and a label.
"""

import pytest

import cutoff_report as hook
import problem_registers as registers


# ---------------------------------------------------------------------------
# a rendered register to read
# ---------------------------------------------------------------------------


def rendered(*rows, resolved=()):
    """`open-problems.md` as the generator lays it out, with *rows* in it."""
    survey = registers.Registers()
    survey.problems.extend(list(rows) + list(resolved))
    survey.problems.sort(key=lambda p: p.sort_key)
    return registers.render_problems(survey, rigs=frozenset({"sandbox_throughput"}))


def problem(
    number=101,
    title="The sandbox is too slow to train on",
    failure="training stalls when the sandbox falls under a tick a millisecond",
    cutoff="measurement sandbox_throughput ticks_per_second < 300",
    state="OPEN",
    overdue=False,
):
    return registers.Problem(
        number=number,
        title=title,
        url=f"https://github.com/NGL321/patchworks/issues/{number}",
        state=state,
        failure=failure,
        cutoff=registers.read_cutoff(cutoff, f"#{number}"),
        discovered="#230",
        overdue=overdue,
        reports=frozenset(),
    )


class TestTheReaderIsHeldAgainstTheRenderer:
    """*Which problems cut on me* is answered off the checked-in file.

    Reading the render is what makes the hook need no token, and it is one of
    the reasons the register is a file rather than a live query. The cost is
    that the rendered row is a contract between the generator and the rig, so
    it is exercised here against the real renderer.
    """

    def test_a_measurement_cutoff_naming_this_rig_is_found(self):
        found = hook.watching(rendered(problem()), "sandbox_throughput")
        assert [w.number for w in found] == [101]
        assert found[0].threshold == "ticks_per_second < 300"
        assert found[0].failure.startswith("training stalls")

    def test_the_rig_may_be_named_three_ways(self):
        """`benchmarks/x.py`, `x.py` and `x` are one rig, as the register says."""
        text = rendered(
            problem(
                cutoff=(
                    "measurement benchmarks/sandbox_throughput.py "
                    "ticks_per_second < 300"
                )
            )
        )
        assert [
            w.number for w in hook.watching(text, "benchmarks/sandbox_throughput.py")
        ] == [101]
        assert [w.number for w in hook.watching(text, "sandbox_throughput")] == [101]

    def test_another_rigs_cutoff_is_not_mine(self):
        text = rendered(problem(cutoff="measurement detectability offset < 0.2"))
        assert hook.watching(text, "sandbox_throughput") == []

    def test_event_and_uncut_cutoffs_are_not_measurements(self):
        text = rendered(
            problem(number=101, cutoff="event 230"),
            problem(number=102, cutoff="uncut"),
        )
        assert hook.watching(text, "sandbox_throughput") == []

    def test_a_resolved_problem_no_longer_cuts(self):
        """A closed problem stays in the register; it does not stay watched."""
        text = rendered(resolved=(problem(state="CLOSED"),))
        assert hook.watching(text, "sandbox_throughput") == []

    def test_a_register_with_no_problems_watches_nothing(self):
        assert hook.watching(rendered(), "sandbox_throughput") == []


# ---------------------------------------------------------------------------
# the bar
# ---------------------------------------------------------------------------


class TestABarIsAMetricAComparatorAndANumber:
    """The threshold the register admits as prose, the rig has to evaluate.

    `tools/problem_registers.py` takes `<threshold>` as free text on purpose —
    it is a projection and evaluates nothing. The rig does evaluate, so it
    needs a form, and a bar it cannot read is **stated in the report** rather
    than skipped: a cutoff nothing can fire on is exactly the disguise #282's
    second loud section exists to show.
    """

    @pytest.mark.parametrize(
        "threshold,metric,comparator,value",
        [
            ("ticks_per_second < 300", "ticks_per_second", "<", 300.0),
            ("offset <= 0.2", "offset", "<=", 0.2),
            ("median_bottleneck >= 1", "median_bottleneck", ">=", 1.0),
            ("camera_ms>1.5", "camera_ms", ">", 1.5),
            ("drift > 1e-3", "drift", ">", 1e-3),
        ],
    )
    def test_a_readable_bar(self, threshold, metric, comparator, value):
        bar = hook.read_bar(threshold)
        assert (bar.metric, bar.comparator, bar.value) == (metric, comparator, value)

    @pytest.mark.parametrize(
        "threshold",
        [
            "when it gets bad",
            "ticks_per_second falls",
            "ticks_per_second < three hundred",
            "ticks_per_second < 300 sometimes",
            "",
        ],
    )
    def test_an_unreadable_bar_is_none_rather_than_a_guess(self, threshold):
        assert hook.read_bar(threshold) is None


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------


class TestTheVerdictStatesTheBarTheReadingAndWhetherItCrossed:
    def _watch(self, **kwargs):
        return hook.watching(rendered(problem(**kwargs)), "sandbox_throughput")[0]

    def test_a_crossing(self):
        verdict = hook.judge(self._watch(), {"ticks_per_second": 210.0})
        assert verdict.crossed is True
        assert verdict.reading == 210.0

    def test_a_reading_that_stands_clear_of_the_bar(self):
        assert hook.judge(self._watch(), {"ticks_per_second": 412.0}).crossed is False

    def test_the_boundary_belongs_to_the_comparator(self):
        assert hook.judge(self._watch(), {"ticks_per_second": 300.0}).crossed is False

    def test_a_bar_nobody_can_read_does_not_cross_and_says_so(self):
        watch = self._watch(cutoff="measurement sandbox_throughput it gets slow")
        verdict = hook.judge(watch, {"ticks_per_second": 210.0})
        assert verdict.crossed is None
        assert "cannot be read" in verdict.why

    def test_a_metric_this_run_did_not_report_does_not_cross_and_says_so(self):
        """Named, with what the run *did* report beside it, so the gap is fixable."""
        verdict = hook.judge(self._watch(), {"camera_ms": 1.2})
        assert verdict.crossed is None
        assert "ticks_per_second" in verdict.why and "camera_ms" in verdict.why


class TestTheReportIsPrintedWhateverHappened:
    """A crossed cutoff is a report and a label, never a failure and never an exit."""

    def test_every_watched_problem_gets_a_line(self, capsys):
        text = rendered(
            problem(number=101),
            problem(number=102, cutoff="measurement sandbox_throughput camera_ms > 5"),
        )
        verdicts = hook.report(
            "sandbox_throughput",
            {"ticks_per_second": 210.0, "camera_ms": 1.2},
            register=text,
            file=False,
        )
        printed = capsys.readouterr().out
        assert "#101" in printed and "#102" in printed
        assert "CROSSED" in printed and "CLEAR" in printed
        assert [v.crossed for v in verdicts] == [True, False]

    def test_a_rig_nothing_cuts_on_says_so_rather_than_nothing(self, capsys):
        """Silence would read the same as a hook nobody wired up."""
        hook.report("sandbox_throughput", {}, register=rendered(), file=False)
        assert "sandbox_throughput" in capsys.readouterr().out

    def test_an_unreadable_register_is_reported_and_never_raised(self, capsys):
        assert hook.report("sandbox_throughput", {}, register=None, file=False) == []
        assert "cutoff" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# what gets filed
# ---------------------------------------------------------------------------


class TestTheCommentRecordsTheRunTheRegisterCanRead:
    """The comment is what clears *cutoffs naming a rig with no recorded run*.

    `tools/problem_registers.py` answers *has anything ever fired against this
    bar* off a `@rig` field block on a comment on the problem. So the body this
    module files has to parse as one, by that module's own parser — which is
    what is asserted here rather than the text of the body.
    """

    def _body(self, readings):
        watch = hook.watching(rendered(problem()), "sandbox_throughput")[0]
        return hook.comment_body(hook.judge(watch, readings))

    def test_the_body_opens_with_a_rig_field_block(self):
        fields = registers.field_block(self._body({"ticks_per_second": 412.0}))
        assert fields[registers.REPORT_KEY] == ["sandbox_throughput"]

    def test_a_clear_run_is_recorded_too_and_not_only_a_crossing(self):
        """Otherwise a rig that runs and never crosses records nothing.

        `open-problems.md` says so in terms: a rig whose runs leave no trace
        sits in the *no recorded run* section forever, watched on the page and
        firing nothing.
        """
        assert registers.field_block(self._body({"ticks_per_second": 412.0})) is not None

    def test_a_crossing_quotes_the_reading(self):
        body = self._body({"ticks_per_second": 210.0})
        assert "210" in body and "crossed" in body

    def test_the_body_names_what_raised_it(self):
        """The overdue-provenance channel's shape: a report says what filed it."""
        assert "cutoff_report.py" in self._body({"ticks_per_second": 412.0})


class TestFilingIsIdempotentTheWayTheOverdueChannelIs:
    """One report at a time. A report nobody can keep up with is a report nobody reads.

    The rig runs on a machine, repeatedly, and a not-crossed reading it has
    already recorded says nothing new. What is new is the *first* record from a
    rig — the one the register is waiting for — and any change of verdict.
    """

    def _verdict(self, readings):
        watch = hook.watching(rendered(problem()), "sandbox_throughput")[0]
        return hook.judge(watch, readings)

    def test_the_first_run_of_a_rig_is_always_recorded(self):
        assert hook.worth_filing(self._verdict({"ticks_per_second": 412.0}), None)

    def test_the_same_verdict_again_is_not(self):
        clear = self._verdict({"ticks_per_second": 412.0})
        assert not hook.worth_filing(clear, hook.stamp(clear))

    def test_a_change_of_verdict_is(self):
        clear = self._verdict({"ticks_per_second": 412.0})
        crossed = self._verdict({"ticks_per_second": 210.0})
        assert hook.worth_filing(crossed, hook.stamp(clear))
        assert hook.worth_filing(clear, hook.stamp(crossed))

    def test_the_stamp_is_the_verdict_and_not_the_reading(self):
        """Two clear runs differ in their number and say the same thing."""
        assert hook.stamp(self._verdict({"ticks_per_second": 412.0})) == hook.stamp(
            self._verdict({"ticks_per_second": 407.0})
        )
