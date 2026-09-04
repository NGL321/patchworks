"""The hermetic half of the problem registers (#282).

`tools/problem_registers.py` shells `gh`. Nothing here does, and nothing here
may learn how: `tests/test_cli.py`'s
:class:`TestNothingHereOpensAWindowOrInstallsAnything` breaks every route to a
subprocess, and a register test that talked to GitHub would be the first in
this suite that fails offline -- on the development laptop, which is where the
issues are written from. So the generator is built around a seam: one function
fetches, and everything below it is a pure function of a payload. This file
holds the parser and the renderer against fixture payloads shaped exactly like
`gh issue list --json`'s.

What is asserted here is **refusal as much as reading**. A register that
renders a half-read row is worse than one that stops, because the row looks
like provenance and is not; that is the same argument
`tests/test_constant_registers.py` makes for the constants, and the three
refusals #282 names -- a proposal with no source or no shape, a `failed`
dismissal naming no rig and no reading, a cutoff outside the two admissible
forms -- each have a test that names the failure they exist to prevent.

**One judgement call is asserted *not* to be made.** `@source here` on a body
carrying no argument is a real failure and the generator must not try to catch
it: whether prose argues a mechanism is not a thing a parser can decide, and a
parser that guessed at it would refuse good entries and pass bad ones.
"""

import pathlib

import pytest
import yaml

import problem_registers as registers


# ---------------------------------------------------------------------------
# payload builders -- the shape `gh issue list --json` returns
# ---------------------------------------------------------------------------


def issue(number=1, title="A problem", body="", state="OPEN", labels=(), comments=()):
    """One `gh issue list --json number,title,body,state,labels,url,comments` row."""
    return {
        "number": number,
        "title": title,
        "body": body,
        "state": state,
        "url": f"https://github.com/NGL321/patchworks/issues/{number}",
        "labels": [{"name": name} for name in labels],
        "comments": list(comments),
    }


def comment(body="", number=1, ident=1):
    return {
        "body": body,
        "url": (
            f"https://github.com/NGL321/patchworks/issues/{number}"
            f"#issuecomment-{ident}"
        ),
        "author": {"login": "NGL321"},
    }


def block(*lines):
    """A field block, fenced, at the top of a body."""
    return "```\n" + "\n".join(lines) + "\n```\n"


PROBLEM = block(
    "@failure    rim-to-core detectability fails below a standing offset of X",
    "@cutoff     measurement detectability offset < 0.2",
    "@discovered #230",
)

PROPOSAL = block(
    "@proposal Relay cells",
    "@source   docs/spec/04-action-and-the-boundary.md",
    "@shape    transmission over a graph deeper than the dome",
    "@shape    a carrier for attention",
    "@status   open",
)


# ---------------------------------------------------------------------------
# the field block
# ---------------------------------------------------------------------------


class TestTheFieldBlockIsFencedAndAtTheTop:
    """Everything below the block is prose the generator never reads.

    That is the whole reason a register can be a projection: the argument lives
    in the body, the row carries a link to it, and the two cannot disagree
    because only one of them is parsed.
    """

    def test_fields_are_read_from_the_fence(self):
        found = registers.field_block(PROBLEM)
        assert found["failure"] == [
            "rim-to-core detectability fails below a standing offset of X"
        ]
        assert found["discovered"] == ["#230"]

    def test_prose_below_the_block_is_not_read(self):
        text = PROBLEM + "\n@failure this is prose and must not win\n"
        assert registers.field_block(text)["failure"] == [
            "rim-to-core detectability fails below a standing offset of X"
        ]

    def test_a_repeated_key_accumulates(self):
        """`@shape` is a list, and stating one never binds the proposal."""
        assert registers.field_block(PROPOSAL)["shape"] == [
            "transmission over a graph deeper than the dome",
            "a carrier for attention",
        ]

    def test_leading_blank_lines_do_not_hide_the_block(self):
        assert registers.field_block("\n\n" + PROBLEM) is not None

    def test_a_body_with_no_block_has_none(self):
        assert registers.field_block("Just prose, no fields.") is None

    def test_a_fenced_block_that_is_not_fields_is_not_a_field_block(self):
        """A comment may open with a code sample, and that is not provenance.

        Most comments on a problem issue are discussion. The discriminator is
        the first line: a block whose opening line is not `@key` is code, and
        reading it as fields would turn every quoted traceback into a malformed
        entry.
        """
        assert registers.field_block("```python\nprint('hi')\n```") is None

    def test_a_block_that_opened_as_fields_refuses_a_line_that_is_not_one(self):
        """Having declared itself provenance, it is held to the grammar."""
        with pytest.raises(registers.MalformedProvenance, match="not a field"):
            registers.field_block(block("@failure a thing", "some loose prose"))


# ---------------------------------------------------------------------------
# cutoffs
# ---------------------------------------------------------------------------


class TestOnlyTwoCutoffFormsAreAdmissibleAndUncutIsTheThird:
    """Dates and judgement are refused.

    A cutoff must be checkable by someone who is not its author, and "when it
    becomes a problem" is not a cutoff but the absence of one -- which `uncut`
    already says, loudly and on purpose.
    """

    def test_event_names_the_issue_that_closing_fires_it(self):
        cut = registers.read_cutoff("event #230", "#1")
        assert (cut.kind, cut.issue) == ("event", "#230")

    def test_a_bare_number_is_the_same_event(self):
        assert registers.read_cutoff("event 230", "#1").issue == "#230"

    def test_measurement_names_a_rig_and_a_bar(self):
        cut = registers.read_cutoff("measurement detectability offset < 0.2", "#1")
        assert (cut.kind, cut.rig, cut.threshold) == (
            "measurement", "detectability", "offset < 0.2",
        )

    def test_a_rig_may_be_written_as_its_script(self):
        """`benchmarks/detectability.py` and `detectability` are one rig."""
        cut = registers.read_cutoff(
            "measurement benchmarks/detectability.py offset < 0.2", "#1"
        )
        assert cut.rig == "detectability"

    def test_uncut_is_admitted(self):
        assert registers.read_cutoff("uncut", "#1").kind == "uncut"

    def test_a_date_is_refused(self):
        with pytest.raises(registers.MalformedProvenance, match="@cutoff"):
            registers.read_cutoff("2026-12-01", "#1")

    def test_judgement_is_refused(self):
        with pytest.raises(registers.MalformedProvenance, match="@cutoff"):
            registers.read_cutoff("when it becomes a problem", "#1")

    def test_a_measurement_with_no_threshold_is_refused(self):
        """A rig with no bar cannot be crossed, so nothing would ever fire."""
        with pytest.raises(registers.MalformedProvenance, match="threshold"):
            registers.read_cutoff("measurement detectability", "#1")

    def test_an_event_carrying_judgement_after_its_issue_is_refused(self):
        """`event 230 when it feels bad` must not parse as `#230`.

        Taking the first token and dropping the rest would admit the judgement
        the two-forms rule exists to refuse, and admit it *silently*: the row
        would render as a clean event cutoff with the unenforceable half of it
        deleted, which is worse than refusing, because nobody would know.
        """
        with pytest.raises(registers.MalformedProvenance, match="one issue"):
            registers.read_cutoff("event 230 when it feels bad", "#1")

    def test_an_event_naming_no_issue_is_refused(self):
        with pytest.raises(registers.MalformedProvenance, match="issue"):
            registers.read_cutoff("event", "#1")


# ---------------------------------------------------------------------------
# problems
# ---------------------------------------------------------------------------


class TestAProblemIsReadOffItsIssue:
    """A half-read problem is worse than none, so it is refused instead.

    Each of the three refusals below is a way a row could render looking like
    provenance while carrying none: a problem with no failure is the "the
    disagreement floor might be too high" case the admission rule exists to
    exclude, and a problem with no cutoff is not the same state as `uncut` --
    one is a debt someone stated, the other is a field someone forgot.
    """

    def test_the_three_fields_land(self):
        problem = registers.read_problem(issue(number=300, body=PROBLEM))
        assert problem.number == 300
        assert problem.cutoff.rig == "detectability"
        assert problem.discovered == "#230"

    def test_a_problem_with_no_field_block_is_refused(self):
        with pytest.raises(registers.MalformedProvenance, match="field block"):
            registers.read_problem(issue(number=300, body="prose only"))

    def test_a_problem_with_no_failure_is_refused(self):
        """"The disagreement floor might be too high" is not admissible."""
        body = block("@cutoff uncut", "@discovered #230")
        with pytest.raises(registers.MalformedProvenance, match="@failure"):
            registers.read_problem(issue(number=300, body=body))

    def test_a_problem_with_no_cutoff_is_refused(self):
        """`uncut` is a value, and its absence is not the same as choosing it."""
        body = block("@failure a stated failure")
        with pytest.raises(registers.MalformedProvenance, match="@cutoff"):
            registers.read_problem(issue(number=300, body=body))

    def test_a_closed_problem_stays_a_problem(self):
        """Deleting the row deletes the evidence for the bet the project makes."""
        problem = registers.read_problem(
            issue(number=300, body=PROBLEM, state="CLOSED")
        )
        assert problem.resolved


# ---------------------------------------------------------------------------
# proposals
# ---------------------------------------------------------------------------


class TestAProposalNeedsASourceAndAShape:
    """The two fields without which a proposal cannot be found or checked.

    **Shape is the index.** An agent arrives at this register holding a symptom,
    not the name it was going to give its solution, so a proposal with no shape
    is a row nobody can reach -- which is worse than no row, because it makes
    the register look fuller than it is. **Source is where the argument lives**,
    and a proposal whose argument is nowhere is the folklore the whole
    mechanism exists to keep out.
    """

    def test_a_well_formed_proposal_reads(self):
        found = registers.read_proposal(PROPOSAL, where="#279", url="u", number=279)
        assert found.title == "Relay cells"
        assert len(found.shapes) == 2
        assert found.answers == ()
        assert found.status.kind == "open"

    def test_a_proposal_with_no_source_is_refused(self):
        text = block("@proposal Relay cells", "@shape a shape", "@status open")
        with pytest.raises(registers.MalformedProvenance, match="@source"):
            registers.read_proposal(text, where="#279", url="u", number=279)

    def test_a_proposal_with_no_shape_is_refused(self):
        """Shape is the index. A proposal with none cannot be arrived at."""
        text = block("@proposal Relay cells", "@source here", "@status open")
        with pytest.raises(registers.MalformedProvenance, match="@shape"):
            registers.read_proposal(text, where="#279", url="u", number=279)

    def test_source_here_on_a_body_with_no_argument_is_not_the_parsers_call(self):
        """A judgement call, and explicitly not the generator's to make (#282).

        Whether prose argues a mechanism is not decidable by a parser, and one
        that guessed would refuse good entries and pass bad ones. The rule is
        real and it is enforced by a reader.
        """
        text = block("@proposal Relay cells", "@source here", "@shape a shape")
        assert registers.read_proposal(text, where="#279", url="u", number=279)

    def test_answers_binds_to_problems_and_may_be_a_list(self):
        text = block(
            "@proposal Something",
            "@source here",
            "@shape a shape",
            "@answers #300, #301",
        )
        found = registers.read_proposal(text, where="#279", url="u", number=279)
        assert found.answers == ("#300", "#301")

    def test_when_is_a_cutoff_with_the_opposite_polarity(self):
        text = block(
            "@proposal Relay cells",
            "@source here",
            "@shape a carrier for attention",
            "@when event #99",
        )
        found = registers.read_proposal(text, where="#279", url="u", number=279)
        assert found.when is not None and found.when.issue == "#99"

    def test_when_may_not_be_uncut(self):
        """`@when` carries no obligation, so there is nothing for `uncut` to say."""
        text = block(
            "@proposal X", "@source here", "@shape a shape", "@when uncut"
        )
        with pytest.raises(registers.MalformedProvenance, match="@when"):
            registers.read_proposal(text, where="#279", url="u", number=279)


class TestStatusAndTheDismissalThatMustNameItsReading:
    """`@status` is the field that decides which register a row lands in.

    Read it wrong and a binding exclusion advertises itself as a live proposal,
    which is the loudest way this register could be wrong: an agent would
    propose the thing the project has already excluded, and the register that
    was supposed to stop that would be what told them to.
    """

    def test_adopted_names_its_adr(self):
        found = registers.read_proposal(
            block(
                "@proposal X", "@source here", "@shape s",
                "@status adopted ADR-0029",
            ),
            where="#279", url="u", number=279,
        )
        assert (found.status.kind, found.status.adr) == ("adopted", "ADR-0029")

    def test_refused_needs_nothing_more(self):
        found = registers.read_proposal(
            block("@proposal X", "@source here", "@shape s",
                  "@status dismissed refused"),
            where="#279", url="u", number=279,
        )
        assert found.status.dismissal == "refused"

    def test_failed_names_the_rig_and_the_reading(self):
        found = registers.read_proposal(
            block("@proposal X", "@source here", "@shape s",
                  "@status dismissed failed detectability offset never fell"),
            where="#279", url="u", number=279,
        )
        assert found.status.rig == "detectability"
        assert found.status.reading == "offset never fell"

    def test_failed_with_no_rig_and_no_reading_is_refused(self):
        """"We tried it" without a reading is unfalsifiable folklore.

        The whole pre-registration discipline exists to keep that out, and a
        `failed` row is the one place it would get back in wearing the clothes
        of a measurement.
        """
        with pytest.raises(registers.MalformedProvenance, match="rig and the reading"):
            registers.read_proposal(
                block("@proposal X", "@source here", "@shape s",
                      "@status dismissed failed"),
                where="#279", url="u", number=279,
            )

    def test_failed_with_a_rig_but_no_reading_is_refused(self):
        with pytest.raises(registers.MalformedProvenance, match="rig and the reading"):
            registers.read_proposal(
                block("@proposal X", "@source here", "@shape s",
                      "@status dismissed failed detectability"),
                where="#279", url="u", number=279,
            )

    def test_a_status_outside_the_grammar_is_refused(self):
        with pytest.raises(registers.MalformedProvenance, match="@status"):
            registers.read_proposal(
                block("@proposal X", "@source here", "@shape s",
                      "@status maybe"),
                where="#279", url="u", number=279,
            )

    def test_a_missing_status_reads_as_open(self):
        found = registers.read_proposal(
            block("@proposal X", "@source here", "@shape s"),
            where="#279", url="u", number=279,
        )
        assert found.status.kind == "open"


# ---------------------------------------------------------------------------
# collection: the comment arm is what makes this not a labels-only query
# ---------------------------------------------------------------------------


class TestProposalsAreCollectedFromCommentsAsWellAsIssues:
    """A proposal specific to one problem lives as a comment on it.

    That is what makes a problem ticket a single place to read -- open the
    problem, see every solution anyone has offered for it -- and it is why the
    generator cannot be a labels-only query.
    """

    def test_a_proposal_comment_becomes_a_row(self):
        problems = [issue(number=300, body=PROBLEM, comments=[comment(PROPOSAL, 300)])]
        found = registers.collect(problems, [], [])
        assert [p.title for p in found.proposals] == ["Relay cells"]
        assert found.proposals[0].is_comment

    def test_an_ordinary_comment_is_passed_over_in_silence(self):
        """Most comments are discussion, and discussion is not malformed."""
        problems = [
            issue(number=300, body=PROBLEM,
                  comments=[comment("Agreed, let's watch it.", 300)])
        ]
        assert registers.collect(problems, [], []).proposals == []

    def test_a_comment_that_declared_itself_a_proposal_may_not_omit_its_name(self):
        """Silence here would be the one failure silence is never allowed.

        A comment carrying `@source` and `@shape` has declared itself
        provenance. Dropping it *silently* for want of `@proposal` would leave a
        proposal that exists, is invisible in the register, and told nobody --
        so the next agent invents it again, which is exactly what the shelf is
        for. It is skipped rather than fatal (#354), and the skip is a row.
        """
        orphan = block("@source here", "@shape a shape", "@status open")
        problems = [issue(number=300, body=PROBLEM, comments=[comment(orphan, 300)])]
        found = registers.collect(problems, [], [])
        assert found.proposals == []
        assert [s.url for s in found.skipped] == [comment(orphan, 300)["url"]]
        assert "@proposal" in found.skipped[0].reason

    def test_a_comment_with_a_field_block_of_another_kind_is_still_silent(self):
        """A rig report is not a malformed proposal (#284 will file these)."""
        report = block("@rig detectability", "@reading offset fell to 0.18")
        problems = [issue(number=300, body=PROBLEM, comments=[comment(report, 300)])]
        assert registers.collect(problems, [], []).proposals == []

    def test_a_comment_row_links_to_the_comment_and_not_the_issue(self):
        problems = [issue(number=300, body=PROBLEM, comments=[comment(PROPOSAL, 300)])]
        found = registers.collect(problems, [], [])
        assert "#issuecomment-" in found.proposals[0].url

    def test_an_issue_carrying_both_labels_is_one_row(self):
        """A dismissed proposal keeps `register:proposal` and gains the other."""
        payload = issue(
            number=279,
            body=block("@proposal X", "@source here", "@shape s",
                       "@status dismissed refused"),
            state="CLOSED",
            labels=("register:proposal", "register:dismissal"),
        )
        found = registers.collect([], [payload], [payload])
        assert len(found.proposals) == 1

    def test_a_dismissal_issue_whose_status_is_not_dismissed_is_refused(self):
        payload = issue(
            number=279,
            body=block("@proposal X", "@source here", "@shape s", "@status open"),
            state="CLOSED",
            labels=("register:dismissal",),
        )
        with pytest.raises(registers.MalformedProvenance, match="register:dismissal"):
            registers.collect([], [], [payload])


# ---------------------------------------------------------------------------
# one bad comment is one bad comment (#354)
# ---------------------------------------------------------------------------


#: The comment #354 names: a quoted rig report with a sentence written under
#: it, inside the same fence. Ordinary to write, and it used to take all three
#: registers down.
LOOSE = block("@rig detectability", "and then a sentence under it")

#: This class only ever renders, and only ever looks at the skipped section, so
#: the rig set is a placeholder rather than `RIGS` reached for from below.
ANY_RIG = frozenset({"detectability"})


class TestAMalformedCommentCostsOnlyItself:
    """A malformed comment is a fact about one comment, not about the projection.

    Before #354 the reader let :class:`MalformedProvenance` out of the comment
    loop and `main` caught nothing, so one unparseable comment anywhere in the
    tracker exited the generator and **none of the three registers
    regenerated** -- on a workflow that fires on every comment event, which is
    repo-wide. The trigger is ordinary: quote a filed rig report inside a fence
    and write a sentence under it.

    The comment is skipped, and the skip is a row. Silent skipping is the exact
    failure the field-block mechanism exists to prevent, so the count and the
    comment URL are rendered rather than only raised: a reader of the page can
    see that the file below may be incomplete, and can reach the comment that
    made it so without scanning an issue by hand.
    """

    def test_a_loose_line_no_longer_aborts_the_survey(self):
        problems = [
            issue(number=300, body=PROBLEM,
                  comments=[comment(LOOSE, 300, 1), comment(PROPOSAL, 300, 2)])
        ]
        found = registers.collect(problems, [], [])
        assert [p.title for p in found.proposals] == ["Relay cells"]
        assert [p.title for p in found.problems] == ["A problem"]

    def test_the_skip_names_the_comment_url_and_not_only_the_issue(self):
        """The diagnostic named the issue; a reader then scanned every comment."""
        problems = [issue(number=300, body=PROBLEM, comments=[comment(LOOSE, 300, 7)])]
        found = registers.collect(problems, [], [])
        assert len(found.skipped) == 1
        assert found.skipped[0].url.endswith("#issuecomment-7")
        assert "#300" in found.skipped[0].where

    def test_a_malformed_comment_does_not_hide_a_rig_report_beside_it(self):
        """`_reports` reads the same comments; it must skip, not raise."""
        report = block("@rig detectability", "@reading offset fell to 0.18")
        problems = [
            issue(number=300, body=PROBLEM,
                  comments=[comment(LOOSE, 300, 1), comment(report, 300, 2)])
        ]
        found = registers.collect(problems, [], [])
        assert found.problems[0].reports == frozenset({"detectability"})

    def test_a_comment_proposal_that_cannot_be_read_is_skipped_not_fatal(self):
        """Every comment-level refusal, not only the loose line."""
        bad = block("@proposal X", "@source here", "@shape s", "@answers not-a-number")
        problems = [issue(number=300, body=PROBLEM, comments=[comment(bad, 300)])]
        found = registers.collect(problems, [], [])
        assert found.proposals == []
        assert len(found.skipped) == 1

    def test_a_malformed_issue_body_is_still_fatal(self):
        """A label promises a row exists; skipping the body would drop it.

        The comment is different in kind: nothing promised it was provenance,
        which is why most comments are passed over in silence already.
        """
        payload = issue(number=300, body=block("@failure x", "@cutoff nonsense"),
                        labels=("register:problem",))
        with pytest.raises(registers.MalformedProvenance):
            registers.collect([payload], [], [])

    def test_every_register_says_how_many_comments_it_could_not_read(self):
        problems = [issue(number=300, body=PROBLEM, comments=[comment(LOOSE, 300, 7)])]
        found = registers.collect(problems, [], [])
        for text in registers.generate(found, ANY_RIG).values():
            assert "#issuecomment-7" in text
            assert registers.SKIPPED_HEADING in text

    def test_with_nothing_skipped_the_section_says_so(self):
        found = registers.collect([issue(number=300, body=PROBLEM)], [], [])
        for text in registers.generate(found, ANY_RIG).values():
            assert registers.SKIPPED_HEADING in text
            assert "#issuecomment" not in text

    def test_a_skip_never_falls_back_to_the_issue_url(self):
        """The row would then say *scan every comment by hand*, which is the bug.

        A skip pointing at the issue is indistinguishable from one that named
        the comment, so a payload with no comment URL states the degradation
        instead of hiding it behind a link that reads as an answer.
        """
        entry = comment(LOOSE, 300, 7)
        del entry["url"]
        problems = [issue(number=300, body=PROBLEM, comments=[entry])]
        skipped = registers.collect(problems, [], []).skipped[0]
        assert skipped.url == ""
        assert "no comment URL" in skipped.link
        assert "issues/300" not in skipped.link

    def test_the_skip_is_recorded_once_and_not_once_per_reader(self):
        """`_reports` and `collect` read the same comments through one walk."""
        problems = [
            issue(number=300, body=PROBLEM,
                  comments=[comment(LOOSE, 300, 1), comment(LOOSE, 300, 2)])
        ]
        found = registers.collect(problems, [], [])
        assert len(found.skipped) == 2

    def test_the_skip_also_reaches_a_terminal(self):
        """The workflow's only signal was a red run; now the run stays green."""
        problems = [issue(number=300, body=PROBLEM, comments=[comment(LOOSE, 300, 7)])]
        lines = registers.skip_report(registers.collect(problems, [], []))
        assert len(lines) == 1
        assert "#issuecomment-7" in lines[0]


# ---------------------------------------------------------------------------
# rendering: the two loud sections
# ---------------------------------------------------------------------------


UNCUT = issue(
    number=301,
    title="The disagreement floor",
    body=block("@failure the floor does not fall", "@cutoff uncut"),
)
CUT = issue(
    number=300,
    title="Rim-to-core detectability",
    body=PROBLEM,
)
PHANTOM = issue(
    number=302,
    title="A cutoff nothing will fire",
    body=block("@failure something", "@cutoff measurement no_such_rig bar < 1"),
)

#: The rigs these tests pretend `benchmarks/` holds, passed in rather than read
#: off disk. Reading the real directory would make this file's verdicts depend
#: on which rigs happen to exist today: deleting `benchmarks/detectability.py`
#: would redden a test about *rendering*, and adding a `benchmarks/no_such_rig.py`
#: would quietly stop the phantom-cutoff test from testing anything. A file
#: whose docstring makes hermeticism its argument should not have a foot on the
#: filesystem.
RIGS = frozenset({"detectability", "alignment_read"})


class TestUncutSortsFirstAndIsStatedAsADebt:
    """In the voice `@flexibility unknown` uses.

    *Nobody has said when this stops being tolerable* is a fact, and hiding it
    is worse than showing it: silence collapses *nobody set a cutoff* into
    *somebody set one and the register lost it*, which are opposite states for
    a reader deciding whether the problem is being watched.
    """

    def test_uncut_rows_come_before_cut_ones(self):
        found = registers.collect([CUT, UNCUT], [], [])
        text = registers.render_problems(found)
        table = text.split("## Open problems")[1]
        assert table.index("#301") < table.index("#300")

    def test_the_debt_is_stated_and_not_merely_sorted(self):
        text = registers.render_problems(registers.collect([CUT, UNCUT], [], []), RIGS)
        assert "Uncut" in text
        assert "nobody has said when this stops being tolerable" in text.lower()

    def test_with_no_uncut_problems_the_section_says_so(self):
        text = registers.render_problems(registers.collect([CUT], [], []), RIGS)
        assert "Uncut" in text


class TestACutoffNamingARigWithNoRecordedRunIsLouderThanUncut:
    """`uncut` wearing a disguise, and strictly worse, because it reads as cut.

    Two ways a `measurement` cutoff never fires, and the register must show
    both: the rig has no script, so there is nothing to run at all; or the rig
    is real and nobody has run it. Either way the row says the problem is being
    watched and it is not, which is the one failure `uncut` was made loud to
    prevent -- `uncut` at least reads as a debt, and this does not.

    #284 declines this check in terms — *"do not try to close it here… that is
    why #279's design makes the register render cutoffs naming a rig with no
    recorded run as its own loud section"* — so it belongs here.
    """

    def loud(self, text):
        """The part of the register above the main table: the two loud sections."""
        return text.split("## Open problems")[0]

    def test_a_rig_with_no_script_is_named(self):
        text = registers.render_problems(registers.collect([PHANTOM], [], []), RIGS)
        assert "no_such_rig" in text
        assert "#302" in self.loud(text)
        assert "nothing to run" in self.loud(text)

    def test_a_real_rig_that_has_never_reported_is_named(self):
        """The case that reads as watched and is the whole point of the section."""
        text = registers.render_problems(registers.collect([CUT], [], []), RIGS)
        assert "#300" in self.loud(text)
        assert "nothing has fired" in self.loud(text)

    def test_a_rig_that_has_reported_clears_the_section(self):
        """A `@rig` field block on a comment is the run record (#284)."""
        report = block("@rig detectability", "@reading offset fell to 0.18")
        reported = issue(
            number=300, title="Rim-to-core detectability", body=PROBLEM,
            comments=[comment(report, 300)],
        )
        text = registers.render_problems(registers.collect([reported], [], []), RIGS)
        assert "#300" not in self.loud(text)

    def test_a_crossed_bar_counts_as_a_run(self):
        """`register:overdue` cannot have been applied without the rig running."""
        crossed = issue(
            number=300, title="Rim-to-core detectability", body=PROBLEM,
            labels=("register:problem", "register:overdue"),
        )
        text = registers.render_problems(registers.collect([crossed], [], []), RIGS)
        assert "#300" not in self.loud(text)

    def test_an_event_cutoff_is_never_in_this_section(self):
        """It fires by itself; `issues: closed` is already watched."""
        evented = issue(
            number=304, title="Waits on an issue",
            body=block("@failure a thing", "@cutoff event #99"),
        )
        text = registers.render_problems(registers.collect([evented], [], []), RIGS)
        assert "#304" not in self.loud(text)


class TestTheThreeFilesRender:
    """Three files, and each row in exactly one of them.

    The split is not cosmetic. A dismissal binds and a proposal does not, so a
    dismissed proposal appearing on the shelf would invite exactly the
    re-proposal the dismissed register exists to prevent -- and *do not
    re-propose this* has to be reachable without opening every problem ticket,
    which is the whole reason the dismissals are unioned into a file of their
    own rather than left where they happened.
    """

    def test_every_file_carries_the_do_not_edit_banner(self):
        found = registers.collect([CUT, UNCUT], [], [])
        for text in registers.generate(found, RIGS).values():
            assert "Do not edit by hand" in text

    def test_a_dismissed_proposal_leaves_the_shelf_for_the_dismissal_register(self):
        dismissed = issue(
            number=279,
            body=block("@proposal Doomed", "@source here", "@shape s",
                       "@status dismissed failed detectability it never moved"),
            state="CLOSED",
            labels=("register:proposal", "register:dismissal"),
        )
        found = registers.collect([], [dismissed], [dismissed])
        assert "Doomed" not in registers.render_proposals(found)
        assert "Doomed" in registers.render_dismissals(found)

    def test_a_failed_dismissal_carries_its_rig_and_reading(self):
        dismissed = issue(
            number=279,
            body=block("@proposal Doomed", "@source here", "@shape s",
                       "@status dismissed failed detectability it never moved"),
            state="CLOSED",
            labels=("register:dismissal",),
        )
        text = registers.render_dismissals(registers.collect([], [], [dismissed]))
        assert "detectability" in text and "it never moved" in text

    def test_proposals_are_indexed_by_shape(self):
        """Search by the symptom you have, not by the name you would have given it."""
        payload = issue(number=279, body=PROPOSAL, labels=("register:proposal",))
        text = registers.render_proposals(registers.collect([], [payload], []))
        assert "a carrier for attention" in text

    def test_an_empty_tracker_still_renders_three_files(self):
        """Before the seeding pass there are no rows, and that is a real state."""
        rendered = registers.generate(registers.collect([], [], []), RIGS)
        assert len(rendered) == 3
        for text in rendered.values():
            assert text.endswith("\n")


class TestTheWorkflowKeepsTheRegistersFresh:
    """The networked half, pinned where the hermetic half can reach it (#283).

    Nothing here runs the workflow or asks GitHub anything -- it reads the file
    as YAML and holds it to its shape. **Why each choice is what it is stays in
    the workflow's own header**, which is the definition site; what is here is
    the assertion, because a rationale copied into a test is a second place the
    decision lives.

    The class exists because nothing else can fail. CI cannot check these three
    files for freshness -- it cannot ask GitHub anything offline -- so if the
    trigger set quietly narrows, or the guard loses an arm, or `issues: read`
    is dropped from the permissions whitelist, nothing anywhere goes red. The
    register simply stops being true, and a projection that reads as current
    and is not is the exact failure this whole mechanism exists to prevent.

    The reader is `tests/test_perturbation.py`'s `_WorkflowLoader`, imported
    rather than copied: it reads a workflow the way GitHub Actions reads one --
    a bare `on` is the string and not YAML 1.1's boolean, a duplicate key is
    refused rather than resolved -- and a second copy of those rules would be a
    second place that decision lives.
    """

    ROOT = pathlib.Path(__file__).resolve().parents[1]
    WORKFLOW = ROOT / ".github" / "workflows" / "problem-registers.yml"
    PROVENANCE = ROOT / ".github" / "workflows" / "constant-provenance.yml"

    def load(self, path=None):
        from test_perturbation import _WorkflowLoader

        return yaml.load(
            (path or self.WORKFLOW).read_text(encoding="utf-8"),
            Loader=_WorkflowLoader,
        )

    def steps(self):
        return [
            step
            for job in self.load()["jobs"].values()
            for step in job["steps"]
        ]

    def test_every_event_that_can_change_the_render_is_a_trigger(self):
        """Labels, bodies and comments, because each of the three is read.

        The workflow's own header argues the set; this holds it. Each event
        here has a partner that would be useless without it -- `closed`
        without `reopened` is a one-way door into the resolved section, and
        `created` without `deleted` renders a proposal that no longer exists.
        """
        triggers = self.load()["on"]
        assert set(triggers["issues"]["types"]) == {
            "opened",
            "closed",
            "reopened",
            "edited",
            "labeled",
            "unlabeled",
        }
        assert set(triggers["issue_comment"]["types"]) == {
            "created",
            "edited",
            "deleted",
        }

    def test_an_issue_leaving_a_register_still_reaches_the_job(self):
        """The `if:` guard's second arm, which reads as redundant and is not.

        The guard skips the traffic that cannot move a row, and its first arm
        asks whether the issue carries a `register:` label. On `unlabeled` the
        payload no longer carries the label that was just removed, so that arm
        alone drops the event -- and an issue *leaving* a register is a change
        the render must see. The second arm reads `github.event.label`, which
        `unlabeled` does carry. Nothing else in the repository would notice if
        a tidier deleted it: the register would simply keep rendering a row for
        an issue that is no longer in the query.
        """
        guard = self.load()["jobs"]["render"]["if"]
        assert "github.event.issue.labels" in guard
        assert "github.event.label.name" in guard
        for event in ("schedule", "workflow_dispatch"):
            assert event in guard, f"{event} would be filtered out by the guard"

    def test_a_weekly_net_and_a_handle_catch_what_the_events_missed(self):
        """A run lost to a broken workflow or to an empty minute budget."""
        triggers = self.load()["on"]
        assert triggers["schedule"], "no weekly net"
        assert "workflow_dispatch" in triggers

    def test_a_superseded_run_is_cancelled_and_the_provenance_run_is_not(self):
        """The one place these two workflows deliberately disagree.

        A later reader tidying them into agreement would break one of them, so
        both halves are asserted here rather than only this file's. This one
        renders a pure projection, where the newest run subsumes every older
        one; `constant-provenance.yml` *files issues*, and a cancelled run
        there is a report nobody ever sees.
        """
        assert self.load()["concurrency"]["cancel-in-progress"] is True
        assert self.load(self.PROVENANCE)["concurrency"]["cancel-in-progress"] is False

    def test_the_job_installs_nothing(self):
        """It does not import patchworks, so it does not pay for torch.

        The generator reads `gh`'s JSON with the standard library, exactly as
        `constant_registers.py` reads source with `ast`. An install step here
        would put a ~2.5 GB dependency set behind a question about issue
        bodies, and would pay for it on every comment edited on a register
        ticket.

        Both halves of a step are read, because an install is as easily an
        action as a shell line -- the argument `TestBothChecksRunInCI` makes
        for pinning `uses:` alongside `run:` in `ci.yml`.
        """
        permitted = {"actions/checkout@v4", "actions/setup-python@v5"}
        for step in self.steps():
            assert "pip install" not in step.get("run", "")
            if "uses" in step:
                assert step["uses"] in permitted

    def test_it_regenerates_rather_than_reporting(self):
        """`--check` names staleness; this workflow's whole job is to end it."""
        runs = "\n".join(step.get("run", "") for step in self.steps())
        assert "tools/problem_registers.py" in runs
        assert "--check" not in runs

    def test_it_may_write_the_render_back_and_may_read_the_tracker(self):
        """Both, or the job fails at one end or the other.

        `permissions:` is a whitelist -- naming any key drops every key not
        named to `none` -- so `issues: read` is load-bearing and not decorative:
        without it the `gh issue list` the generator shells fails, and without
        `contents: write` the render is computed and thrown away.
        """
        permissions = self.load()["permissions"]
        assert permissions["contents"] == "write"
        assert permissions["issues"] == "read"
