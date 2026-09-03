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

import json

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
    def test_a_pipe_in_the_prose_does_not_hide_the_row(self):
        """`problem_registers._cell` escapes a pipe so the table survives it, so
        the reader has to split on the unescaped ones. Splitting on every `|`
        gives the row extra cells, the cutoff lands in the wrong column, and the
        row is passed over in silence -- the rig stops watching a problem and
        the register goes on saying it is watched. #335's failure carries a
        literal `|loop(c)|`, so this is the register as checked in, not a
        hypothetical."""
        text = rendered(
            problem(
                failure="the projection can only shorten |loop(c)|, "
                "never lengthen it"
            )
        )
        assert [w.number for w in hook.watching(text, "sandbox_throughput")] == [101]



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

    @pytest.mark.parametrize(
        "threshold",
        [
            "conduction ratio >= 1",
            "conduction-ratio >= 1",
            "conduction_ratio>=1",
            "Conduction Ratio >= 1",
        ],
    )
    def test_a_metric_written_in_words_is_the_same_metric(self, threshold):
        """#325 and #329 both write `conduction ratio >= 1`, and they are real.

        A grammar that refused a space would be a tool telling an author how to
        spell a quantity they named in the prose above it — which
        `problem_registers.rig_name` already declines to do for the rig, in
        those terms.
        """
        bar = hook.read_bar(threshold)
        assert (bar.metric, bar.comparator, bar.value) == ("conduction_ratio", ">=", 1.0)

    def test_the_bar_is_quoted_back_as_the_issue_wrote_it(self):
        assert hook.read_bar("conduction ratio >= 1").text == "conduction ratio >= 1"

    def test_the_rigs_spelling_meets_the_issues(self):
        watch = hook.watching(
            rendered(problem(cutoff="measurement sandbox_throughput conduction ratio >= 1")),
            "sandbox_throughput",
        )[0]
        assert hook.judge(watch, {"Conduction-Ratio": 1.4}).crossed is True

    @pytest.mark.parametrize("threshold", ["offset == 0.2", "offset != 0.2"])
    def test_equality_is_not_a_comparator_a_cutoff_may_use(self, threshold):
        """A cutoff is a direction a reading may go far enough in.

        Equality on a measured float is a bar nothing ever lands on, and its
        negation is one crossed on the first run whatever the reading. Both read
        as cutoffs and neither is one.
        """
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

    def register(self, tmp_path, text):
        """The rendered register as the rig meets it: a file on disk."""
        path = tmp_path / "open-problems.md"
        path.write_text(text, encoding="utf-8")
        return path

    def test_every_watched_problem_gets_a_line(self, tmp_path, capsys):
        text = rendered(
            problem(number=101),
            problem(number=102, cutoff="measurement sandbox_throughput camera_ms > 5"),
        )
        verdicts = hook.report(
            "sandbox_throughput",
            {"ticks_per_second": 210.0, "camera_ms": 1.2},
            register=self.register(tmp_path, text),
            file=False,
        )
        printed = capsys.readouterr().out
        assert "#101" in printed and "#102" in printed
        assert "CROSSED" in printed and "CLEAR" in printed
        assert [v.crossed for v in verdicts] == [True, False]

    def test_a_rig_nothing_cuts_on_says_so_rather_than_nothing(self, tmp_path, capsys):
        """Silence would read the same as a hook nobody wired up."""
        hook.report(
            "sandbox_throughput",
            {},
            register=self.register(tmp_path, rendered()),
            file=False,
        )
        assert "sandbox_throughput" in capsys.readouterr().out

    def test_a_register_that_is_not_there_is_reported_and_never_raised(
        self, tmp_path, capsys
    ):
        """A rig on a machine with no checked-out register still finishes its run."""
        missing = tmp_path / "not-generated-yet.md"
        assert hook.report("sandbox_throughput", {}, register=missing, file=False) == []
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


class TestARunThatEvaluatedNothingIsNotRecordedAsARun:
    """The one hole #284 is told not to close, kept open.

    *"Do not try to close it here: a measurement cutoff on a rig nobody runs
    never fires… that is why #279's design makes the register render cutoffs
    naming a rig with no recorded run as its own loud section."*

    A run that could not read the bar, or found no such metric, has fired
    nothing. Signing that as a run would take the problem out of that loud
    section and leave it reading as watched — the disguise the section exists to
    show, now applied by the very tool that was meant to reveal it. So the
    report is still filed, because only the run can see the gap, and it is filed
    under a key the register does not count.
    """

    def _payload(self, readings, threshold="it gets slow"):
        watch = hook.watching(
            rendered(problem(cutoff=f"measurement sandbox_throughput {threshold}")),
            "sandbox_throughput",
        )[0]
        body = hook.comment_body(hook.judge(watch, readings))
        return {
            "number": 101,
            "title": "A problem",
            "body": (
                "```\n@failure a failure\n"
                "@cutoff measurement sandbox_throughput ticks_per_second < 300\n```\n"
            ),
            "state": "OPEN",
            "url": "https://github.com/NGL321/patchworks/issues/101",
            "labels": [{"name": "register:problem"}],
            "comments": [{"body": body}],
        }

    def test_an_unreadable_bar_leaves_the_problem_in_the_loud_section(self):
        found = registers.read_problem(self._payload({"ticks_per_second": 210.0}))
        assert found.reports == frozenset()
        survey = registers.Registers()
        survey.problems.append(found)
        unwatched = registers.unwatched(survey, frozenset({"sandbox_throughput"}))
        assert [u.reason for u in unwatched] == [registers.NO_RUN]

    def test_a_metric_the_run_did_not_report_leaves_it_there_too(self):
        payload = self._payload({"camera_ms": 1.2}, threshold="ticks_per_second < 300")
        assert registers.read_problem(payload).reports == frozenset()

    def test_an_evaluated_run_does_record_one(self):
        """The contrast: a bar that was actually read leaves `@rig` behind."""
        payload = self._payload(
            {"ticks_per_second": 412.0}, threshold="ticks_per_second < 300"
        )
        assert registers.read_problem(payload).reports == frozenset(
            {"sandbox_throughput"}
        )


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

    def test_what_this_rig_last_said_is_read_back_off_the_comments(self):
        clear = self._verdict({"ticks_per_second": 412.0})
        crossed = self._verdict({"ticks_per_second": 210.0})
        comments = [{"body": hook.comment_body(clear)}]
        assert hook.read_stamps(comments, "sandbox_throughput") == hook.stamp(clear)
        comments.append({"body": hook.comment_body(crossed)})
        assert hook.read_stamps(comments, "sandbox_throughput") == hook.stamp(crossed)

    def test_a_run_that_evaluated_nothing_is_still_a_thing_this_rig_has_said(self):
        """It does not count as a *run*; it does count as already reported."""
        watch = hook.watching(
            rendered(problem(cutoff="measurement sandbox_throughput it gets slow")),
            "sandbox_throughput",
        )[0]
        verdict = hook.judge(watch, {})
        comments = [{"body": hook.comment_body(verdict)}]
        assert hook.read_stamps(comments, "sandbox_throughput") == hook.stamp(verdict)
        assert not hook.worth_filing(verdict, hook.stamp(verdict))

    def test_another_rigs_record_is_not_mine(self):
        clear = self._verdict({"ticks_per_second": 412.0})
        assert hook.read_stamps([{"body": hook.comment_body(clear)}], "detectability") is None

    def test_a_hand_written_rig_note_does_not_clear_the_record(self):
        """A comment carrying `@rig` and no verdict is somebody's note, not a run.

        Treating it as a reset would file a verdict the issue already carries,
        which is the repetition the `overdue-provenance` channel's one-report-at-
        a-time rule exists to prevent.
        """
        clear = self._verdict({"ticks_per_second": 412.0})
        comments = [
            {"body": hook.comment_body(clear)},
            {"body": "```\n@rig sandbox_throughput\n```\n\nI ran this by hand.\n"},
        ]
        assert hook.read_stamps(comments, "sandbox_throughput") == hook.stamp(clear)


class TestTheFilingArmDoesTwoThingsAndOnlyOnACrossingBoth:
    """`gh` stood in for, because nothing in this suite may reach the network.

    What is asserted is the *calls*: which of them are made, in what order, and
    that the label is one of them only when the bar was crossed. The stand-in is
    a function, not a subprocess — `tests/test_cli.py` breaks every route to one
    and this file must stay on the right side of that.
    """

    def _verdict(self, readings):
        watch = hook.watching(rendered(problem()), "sandbox_throughput")[0]
        return hook.judge(watch, readings)

    def _calls(self, monkeypatch, verdict, comments=()):
        made = []

        def stand_in(arguments, stdin=None):
            made.append((arguments, stdin))
            if arguments[:2] == ["issue", "view"]:
                return json.dumps({"comments": list(comments)})
            return ""

        monkeypatch.setattr(hook, "gh", stand_in)
        return made, hook.file_report(verdict)

    def test_a_crossing_files_the_comment_and_the_label(self, monkeypatch):
        made, done = self._calls(monkeypatch, self._verdict({"ticks_per_second": 210}))
        assert [call[0][:2] for call in made] == [
            ["issue", "view"],
            ["issue", "comment"],
            ["issue", "edit"],
        ]
        assert made[1][1] == hook.comment_body(self._verdict({"ticks_per_second": 210}))
        assert registers.OVERDUE_LABEL in made[2][0]
        assert registers.OVERDUE_LABEL in done

    def test_a_clear_reading_files_the_comment_and_no_label(self, monkeypatch):
        made, done = self._calls(monkeypatch, self._verdict({"ticks_per_second": 412}))
        assert [call[0][:2] for call in made] == [
            ["issue", "view"],
            ["issue", "comment"],
        ]
        assert registers.OVERDUE_LABEL not in done

    def test_the_same_verdict_again_files_nothing_at_all(self, monkeypatch):
        clear = self._verdict({"ticks_per_second": 412.0})
        made, done = self._calls(
            monkeypatch, clear, comments=[{"body": hook.comment_body(clear)}]
        )
        assert [call[0][:2] for call in made] == [["issue", "view"]]
        assert "already on record" in done

    def test_gh_being_unreachable_costs_the_run_nothing(self, monkeypatch):
        """Offline, unauthenticated and rate-limited are ordinary states here."""

        def refuse(arguments, stdin=None):
            raise FileNotFoundError("gh")

        monkeypatch.setattr(hook, "gh", refuse)
        assert "not filed" in hook.file_report(self._verdict({"ticks_per_second": 210}))
