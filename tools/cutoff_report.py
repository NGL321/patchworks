"""The measurement half of the cutoff mechanism, evaluated by the rig (#284).

A `@cutoff` on an open problem has two admissible forms and they fire by
different means. **`event <issue>` fires by itself**: the
`.github/workflows/constant-provenance.yml` pattern watches `issues: closed`
and reports what was resting on the issue that closed. **`measurement <rig>
<threshold>` cannot**, because *rigs do not run in CI and must not learn to* —
`benchmarks/run_reporting.py` states the rule for every script in that
directory, *"Like every other script here it asserts nothing… a wall-clock
measurement belongs on a machine rather than in a suite."* That rule stands and
this module does not touch it.

So the evaluation moves to the report. **When a rig runs, its report states, for
each open problem whose `@cutoff` names that rig, whether the bar was crossed**:

    from cutoff_report import report

    report("sandbox_throughput", {"ticks_per_second": 412.0})

**A rig asserting nothing still holds.** A crossed cutoff is a report and a
label. It is never a test failure, never a raised exception and never a non-zero
exit — every failure inside this module, up to and including GitHub being
unreachable, is printed as a line in the report and returned from.

## Which problems cut on me

Read off the *rendered* `docs/registers/open-problems.md`, which needs no token
— one of the reasons the register is a checked-in file rather than a live query.
The file may be briefly stale and is never the authority; for this question that
is the right trade, because the alternative is a rig that cannot answer it at
all without credentials.

The cost is that the rendered row is a **contract** between
`tools/problem_registers.py` and this module. `tests/test_cutoff_report.py`
holds the two together by parsing what the generator actually emits, so a change
to either side breaks there rather than quietly leaving a rig watching nothing.

## The bar

`problem_registers.py` takes `<threshold>` as free text, because a projection
evaluates nothing. This module does evaluate, so it needs a form:

    <metric> <comparator> <number>          ticks_per_second < 300

`<metric>` is a key of the readings the rig hands over. The threshold states the
condition under which the problem **stops being tolerable**, so the bar is
crossed exactly when the comparison holds.

Two things are stated in the report rather than skipped, because a cutoff
nothing can fire on is the disguise `open-problems.md`'s second loud section
exists to show: a threshold that is not of that form, and a metric this run did
not report.

## What is filed

On every evaluated problem — not only on a crossing — a comment carrying a
`@rig` field block. That block is how `problem_registers.py` answers *has
anything ever fired against this bar*, and filing only on crossings would leave
a rig that runs regularly and never crosses recording nothing, sitting in *no
recorded run* forever. `open-problems.md` says so in terms.

On a crossing, additionally the `register:overdue` label, and the comment quotes
the reading. The shape is the `overdue-provenance` channel's, reused rather than
reinvented: state the fact, quote the reading, name the mechanism that raised it
— and file one report at a time, because a report nobody can keep up with is a
report nobody reads. Here that means a verdict is filed when it is the rig's
first on the problem, or when it differs from the last one this rig filed.

## The precondition

A problem may carry a `@when` **precondition** over its cutoff (#417): the
condition under which the cutoff is a *readable* number at all. It is not a
second cutoff and it obliges nothing, and two things follow, both of them here.

**A crossing behind a shut precondition is recorded and withholds the label.**
The `@rig` block files as normal — so the row never sits in *no recorded run*
while a rig is in fact reading it — and `register:overdue` is not added. The
verdict is stamped `crossed-withheld` rather than `crossed`, which is what lets
the *next* crossing, once the precondition has opened, read as a change and
stamp the label.

**Whether a precondition has opened is read off the record on the problem**, not
guessed and not recomputed. A rig cannot evaluate another rig's bar — #329's
precondition names `detectability` and its cutoff names `driven_settling` — so
the question this module can answer is *has anything reported that this
precondition opened*, and a `@precondition` field block carrying
`@verdict opened` is that report. A precondition nobody ever records as opened
is not silently assumed shut forever: it sits in `open-problems.md`'s second
loud section under its own arm, which is where that debt is made visible.

**A rig also reports on the preconditions naming it.** For each open problem
whose `@when` names this rig, the report states whether the precondition opened,
filed under `@precondition` — **never** `register:overdue`, because opening a
precondition imposes nothing.

**One thing this deliberately does not close.** A measurement cutoff on a rig
nobody runs never fires, and nothing here can change that. #279's design makes
the register render *cutoffs naming a rig with no recorded run* as its own loud
section, so the hole is made visible rather than automated away.

## Where it sits relative to the suite

It shells `gh`, so it is a network tool and lives beside
`tools/problem_registers.py` on the far side of the line `tests/test_cli.py`
defends. Everything above :func:`file_report` is a pure function of the rendered
register and a dict of readings, which is the seam the tests hold.

    python tools/cutoff_report.py sandbox_throughput ticks_per_second=412
    python tools/cutoff_report.py sandbox_throughput ticks_per_second=210 --no-file
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass

import problem_registers as registers

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The projection a rig reads to ask *which problems cut on me*. A file rather
#: than a query, so the question costs no token; stale rather than authoritative,
#: which is the trade `docs/agents/registers.md` names.
REGISTER = ROOT / "docs" / "registers" / "open-problems.md"

#: The heading whose table is live. The resolved table below it is history: a
#: closed problem stays in the register and stops being watched.
OPEN_HEADING = "## Open problems"

#: A `measurement` cutoff as `problem_registers.Cutoff.text` renders it. The
#: same cell shape carries a `measurement` precondition, because it is the same
#: field grammar with the opposite polarity.
_CUTOFF_CELL = re.compile(r"^measurement\s+`(?P<rig>[^`]+)`\s+—\s+(?P<threshold>.+)$")

#: An `event` bar as `Cutoff.text` renders it. Read only to name the subject of
#: a precondition, so that *has this opened* can be asked about an `event`
#: precondition in the same words as a `measurement` one. Nothing here evaluates
#: it: an event fires by the issue closing, which is
#: `constant-provenance.yml`'s half of the mechanism and not a rig's.
_EVENT_CELL = re.compile(r"^event\s+(?P<issue>#\d+)$")

#: The issue cell, as `Problem.link` renders it.
_ISSUE_CELL = re.compile(r"^\[#(?P<number>\d+)\]\((?P<url>[^)]+)\)$")

#: `<metric> <comparator> <number>`, anchored: a threshold with anything else in
#: it is not read at all rather than read partly. `ticks_per_second < 300
#: sometimes` parses under a loose pattern and silently discards the judgement,
#: which is the admission the two-forms rule exists to refuse.
#:
#: The metric may be **several words**. #325 and #329 both write `conduction
#: ratio >= 1`, and a grammar that refused them would be a tool telling an
#: author how to spell a quantity they named in the prose above it --
#: `problem_registers.rig_name` already declines to do that for the rig, in
#: those terms, and the metric is the same kind of name. :func:`metric_name`
#: settles the spelling instead.
_BAR = re.compile(
    r"^(?P<metric>[A-Za-z_][A-Za-z0-9_ \t-]*?)\s*"
    r"(?P<comparator><=|>=|<|>)\s*"
    r"(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)$"
)

#: Anything that separates the words of a metric's name.
_SEPARATOR = re.compile(r"[\s-]+")

#: The four comparators, and not `==` or `!=`. A cutoff is a *direction* a
#: reading may go far enough in; equality on a measured float is a bar nothing
#: ever lands on, and its negation is one that is crossed on the first run
#: whatever the reading. Both would read as cutoffs and neither would be one.
_COMPARE = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}

#: What a filed comment says its verdict was, so the next run can read it back
#: and not file the same thing twice. Not a key `problem_registers.py` reads,
#: and deliberately not one of its `PROPOSAL_KEYS`: a rig report is a record of
#: a run and must never be collected as a proposal.
VERDICT_KEY = "verdict"

#: What a run that could **not** evaluate the cutoff names itself with, instead
#: of `@rig`. The distinction is load-bearing and it is the one thing #284
#: refuses to paper over: `problem_registers._reports` collects every `@rig`
#: block, and `unwatched()` drops a problem out of *cutoffs naming a
#: firing condition nothing will reach* as soon as one appears. A run that
#: read an unreadable bar, or found no such metric, has fired nothing --
#: recording it as a run would take the row out of the loud section and
#: leave the problem reading as watched,
#: which is the disguise that section exists to show. So the report is filed,
#: because only the run can see the gap, and it is filed under a key the
#: register does not count.
UNEVALUATED_KEY = "unevaluated"

#: The three verdicts, in the one spelling that goes into a comment, a stamp and
#: a report line alike. Three and not two: *nothing could be checked* and *the
#: bar held* are opposite states for a reader, and a `None` collapsed into
#: `False` would file a clean bill of health for a cutoff nothing evaluated.
_WORD = {True: "crossed", False: "clear", None: "not-evaluated"}

#: The same three, for a precondition. A precondition does not *cross*: it
#: **opens**, and the words are different because the states are — a crossed
#: cutoff says a problem is due, and an opened precondition says only that the
#: number below it can now be read.
_GATE_WORD = {True: "opened", False: "shut", None: "not-evaluated"}

#: What a crossing whose precondition is shut stamps instead of `crossed`. It is
#: a distinct word and not a flag on the side, because the stamp is the whole
#: dedupe: a withheld crossing filed as `crossed` would make the first crossing
#: *after* the precondition opened read as the same verdict again, and the label
#: this withholds would then never be added at all.
WITHHELD = "crossed-withheld"

#: The two fields a rig reports against, which is also the pair
#: `problem_registers.Unwatched` names. `cutoff` obliges; `precondition` does
#: not, and that difference is every difference between the two passes.
CUTOFF_FIELD = "cutoff"
PRECONDITION_FIELD = "precondition"


# ---------------------------------------------------------------------------
# which problems cut on me
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Watch:
    """One open problem one of whose `measurement` bars names this rig.

    Either bar: `field` says which, and the row is otherwise read the same way
    because it is the same grammar. What differs is downstream — a crossed
    cutoff obliges and a precondition never does.
    """

    number: int
    url: str
    title: str
    failure: str
    rig: str
    threshold: str
    #: `cutoff` or `precondition`.
    field: str = CUTOFF_FIELD
    #: The problem's precondition cell, exactly as the register rendered it, or
    #: empty where the problem has none. Carried on a **cutoff** watch because
    #: it is what decides whether a crossing may stamp `register:overdue`; it is
    #: not read on a precondition watch, whose own bar is `threshold`.
    precondition: str = ""

    @property
    def ref(self) -> str:
        return f"#{self.number}"

    @property
    def obliges(self) -> bool:
        """Whether crossing this bar can put the problem in debt.

        A cutoff can. A precondition cannot, ever, and that is the one fact this
        whole field exists to carry.
        """
        return self.field == CUTOFF_FIELD


def _rows(text: str) -> list[list[str]]:
    """The cells of the table under *Open problems*, header and rule dropped."""
    found: list[list[str]] = []
    inside = False
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith("## "):
            inside = line.strip() == OPEN_HEADING
            continue
        if not inside or not line.strip().startswith("|"):
            continue
        row = line.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|") and not row.endswith("\\|"):
            row = row[:-1]
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", row)]
        if not cells or set("".join(cells)) <= set("- :"):
            continue
        found.append([cell.replace("\\|", "|") for cell in cells])
    return found[1:] if found else []


def watching(text: str, rig: str, *, field: str = CUTOFF_FIELD) -> list[Watch]:
    """Every open problem in the rendered register whose *field* names *rig*.

    The rig may be written three ways -- `benchmarks/x.py`, `x.py` or `x` --
    and `problem_registers.rig_name` is what settles which is which, here as
    there. Rows this module cannot read are passed over rather than raised on:
    the register is the authority on its own grammar, and a rig is not the place
    a malformed row should surface.

    `field` selects the column: the `cutoff` cell, which is the question *which
    problems cut on me*, or the `precondition` cell, which is *which problems
    are waiting on a reading I take*. One reader for both, because the register
    renders both with `Cutoff.text` and a second parser would be a second place
    the same grammar lives.
    """
    wanted = registers.rig_name(rig)
    found = []
    for cells in _rows(text):
        if len(cells) < 6:
            continue
        title, failure, precondition, cutoff, _discovered, issue = cells[:6]
        cell = precondition if field == PRECONDITION_FIELD else cutoff
        bar = _CUTOFF_CELL.match(cell)
        reference = _ISSUE_CELL.match(issue)
        if bar is None or reference is None:
            continue
        if registers.rig_name(bar.group("rig")) != wanted:
            continue
        found.append(
            Watch(
                number=int(reference.group("number")),
                url=reference.group("url"),
                title=title.removesuffix("**(overdue)**").strip(),
                failure=failure,
                rig=wanted,
                threshold=bar.group("threshold").strip(),
                field=field,
                precondition="" if field == PRECONDITION_FIELD else precondition,
            )
        )
    return found


def gate_subject(cell: str) -> str:
    """What a precondition cell says has to happen, named: a rig, or an issue.

    The key a `@precondition` record is matched against, so that *has this
    opened* is one question over both admissible forms. An empty string means
    the row carries no precondition, or one this module cannot read -- and those
    are the same state here, since a bar nothing can read is not one anything
    can report as opened.
    """
    text = (cell or "").strip()
    if text in {"", "—", "-"}:
        return ""
    bar = _CUTOFF_CELL.match(text)
    if bar is not None:
        return registers.rig_name(bar.group("rig"))
    event = _EVENT_CELL.match(text)
    return event.group("issue") if event is not None else ""


# ---------------------------------------------------------------------------
# the bar, and the verdict against it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bar:
    """`<metric> <comparator> <number>`: the reading at which a problem is due."""

    #: Normalised by :func:`metric_name`, so that the issue's spelling and the
    #: rig's key are the same name.
    metric: str
    comparator: str
    value: float
    #: The threshold as the issue wrote it. Quoted back rather than reformatted:
    #: a report that showed the author a tidied version of their own bar would
    #: make them work out whether the tidying changed anything.
    written: str = ""

    @property
    def text(self) -> str:
        return self.written or f"{self.metric} {self.comparator} {self.value:g}"

    def crossed_by(self, reading: float) -> bool:
        return bool(_COMPARE[self.comparator](reading, self.value))


def metric_name(written: str) -> str:
    """`conduction ratio`, `conduction-ratio` and `conduction_ratio` are one name.

    The same service `problem_registers.rig_name` does for a rig written three
    ways, and for its reason: the register may not tell an author how to spell a
    quantity, and a bar refused over a space would read to its author as the
    mechanism not working.
    """
    return _SEPARATOR.sub("_", written.strip()).lower()


def read_bar(threshold: str) -> Bar | None:
    """The threshold as something evaluable, or `None` if it is not.

    `None` is a reported state and not a skipped one -- see :func:`judge`. A
    parser that guessed at prose would report a crossing that was never stated,
    which is worse than a bar that says out loud it cannot fire.
    """
    match = _BAR.match(threshold.strip())
    if match is None:
        return None
    return Bar(
        metric=metric_name(match.group("metric")),
        comparator=match.group("comparator"),
        value=float(match.group("value")),
        written=threshold.strip(),
    )


@dataclass(frozen=True)
class Verdict:
    """What this run has to say about one problem cutting on this rig."""

    watch: Watch
    bar: Bar | None
    reading: float | None
    #: `True` crossed, `False` clear, `None` not evaluated -- and the third is a
    #: state of its own rather than a quiet `False`, because *nothing could be
    #: checked* and *the bar held* are opposite things for a reader. On a
    #: precondition the same three read *opened*, *shut* and *not evaluated*.
    crossed: bool | None
    why: str

    @property
    def bar_text(self) -> str:
        return self.bar.text if self.bar is not None else self.watch.threshold

    @property
    def word(self) -> str:
        """The verdict, in this field's vocabulary.

        A precondition does not cross; it opens. One word per state per field,
        used by the comment, the stamp and the report line alike, so that what
        the next run reads back is what this one meant.
        """
        return (_WORD if self.watch.obliges else _GATE_WORD)[self.crossed]

    @property
    def line(self) -> str:
        """The report's line for this problem: the bar, the reading, the verdict."""
        named = "bar" if self.watch.obliges else "precondition"
        head = f"  {self.watch.ref:<6} {named} `{self.bar_text}`"
        if self.crossed is None:
            return f"{head}   not evaluated - {self.why}"
        return f"{head}   reading {self.reading:g}   {self.word.upper()}"


def judge(watch: Watch, readings: dict[str, float]) -> Verdict:
    """One problem, against what this run measured.

    The rig's keys are put through :func:`metric_name` too, so that the issue
    and the rig meet in the middle rather than the issue having to guess the
    rig's punctuation.
    """
    readings = {metric_name(key): value for key, value in readings.items()}
    bar = read_bar(watch.threshold)
    if bar is None:
        return Verdict(
            watch,
            None,
            None,
            None,
            "the bar cannot be read as `<metric> <comparator> <number>`, so "
            "nothing can fire on it",
        )
    if bar.metric not in readings:
        reported = ", ".join(sorted(readings)) or "nothing"
        return Verdict(
            watch,
            bar,
            None,
            None,
            f"this run reported no `{bar.metric}` (it reported: {reported})",
        )
    reading = float(readings[bar.metric])
    crossed = bar.crossed_by(reading)
    return Verdict(watch, bar, reading, crossed, "")


# ---------------------------------------------------------------------------
# what gets filed
# ---------------------------------------------------------------------------


def _stamp(rig: str, word: str) -> str:
    """The one place the stamp's shape is known.

    Written by :func:`stamp` and read back by :func:`last_verdict`, so the two
    cannot drift into disagreeing about what "the same verdict again" means.
    """
    return f"{rig}:{word}"


def stamp(verdict: Verdict, *, withheld: bool = False) -> str:
    """What this run would leave on record: the rig and its verdict, not the number.

    Two clear runs differ in their reading and say the same thing. Stamping the
    reading would file a comment on every run of every rig, which is the volume
    that turns a channel into noise.

    **A withheld crossing stamps its own word.** It is a different thing from a
    crossing that was stamped, and the difference has to survive into the
    record: it is what makes the first crossing after the precondition opens
    read as a change rather than as the same verdict again, which is the run
    that adds the label.
    """
    return _stamp(verdict.watch.rig, WITHHELD if withheld else verdict.word)


def worth_filing(verdict: Verdict, prior: str | None, *, withheld: bool = False) -> bool:
    """Whether this verdict says anything the problem does not already record.

    The first record from a rig always does -- it is the one the register's *no
    recorded run* section is waiting for -- and so does a change of verdict,
    including a crossing that was withheld last time and is not now.
    """
    return prior is None or prior != stamp(verdict, withheld=withheld)


def _gate_prose(verdict: Verdict) -> str:
    """What a precondition record says, in the three states it can be in.

    Never the cutoff's words, and never `register:overdue`: opening a
    precondition imposes nothing. What it does is make the bar below it
    readable, and the day it opens is the day somebody has to be told, because
    once the precondition is nobody's `@cutoff` no other report mentions it
    (#417).
    """
    watch = verdict.watch
    if verdict.crossed is True:
        return (
            f"**The precondition on this problem opened.** `{watch.rig}` ran and "
            f"read `{verdict.bar.metric} = {verdict.reading:g}` against "
            f"`{verdict.bar.text}`. The cutoff below it is now a readable "
            "number, and this problem is live in a way it was not before. "
            "Opening a precondition imposes nothing, so this is a report and "
            "not a label."
        )
    if verdict.crossed is False:
        return (
            f"`{watch.rig}` ran and read `{verdict.bar.metric} = "
            f"{verdict.reading:g}` against `{verdict.bar.text}`. **The "
            "precondition is still shut**, so a crossing of the cutoff below it "
            "would be recorded and would carry no obligation."
        )
    return (
        f"`{watch.rig}` ran and **could not evaluate this precondition**: "
        f"{verdict.why}. The precondition as the issue states it is "
        f"`{watch.threshold}`. Recorded rather than passed over, and under a key "
        "the register does not count as a run."
    )


def report_key(watch: Watch, evaluated: bool) -> str:
    """The field-block key this record signs itself with.

    Four, and the fourth is not a flourish. `@rig` and `@precondition` are the
    two things `problem_registers.py` counts, one per field, and their
    `@unevaluated` mirrors are the two it counts as nothing -- a run that could
    not read the bar has fired nothing, and signing it as a run would lift the
    row out of the register's second loud section while nothing whatever was
    watching it. Two mirrors and not one shared mirror, so that a rig which is
    both a problem's cutoff and its precondition cannot have one of its records
    read as the other.
    """
    if watch.obliges:
        return registers.REPORT_KEY if evaluated else UNEVALUATED_KEY
    return (
        registers.PRECONDITION_KEY
        if evaluated
        else registers.UNEVALUATED_PRECONDITION_KEY
    )


def comment_body(verdict: Verdict, *, withheld: bool = False) -> str:
    """The comment this run files on the problem.

    It opens with the field block `problem_registers.py` reads, because that
    block is the whole reason the comment exists: it is how the register knows
    the bar has something watching it. Everything under the fence is prose the
    generator never reads, and the shape of that prose is the
    `overdue-provenance` channel's -- the fact, the reading, and what raised it.
    """
    watch = verdict.watch
    key = report_key(watch, verdict.crossed is not None)
    lines = [
        "```",
        f"@{key} {watch.rig}",
        f"@{VERDICT_KEY} {WITHHELD if withheld else verdict.word}",
        "```",
        "",
    ]
    if not watch.obliges:
        lines.append(_gate_prose(verdict))
        lines += [
            "",
            "Filed by `tools/cutoff_report.py` (#417) from a run of "
            f"`benchmarks/{watch.rig}.py`.",
        ]
        return "\n".join(lines) + "\n"
    if withheld:
        lines.append(
            f"**The cutoff on this problem was crossed, and the obligation is "
            f"withheld.** `{watch.rig}` ran and read "
            f"`{verdict.bar.metric} = {verdict.reading:g}` against the bar "
            f"`{verdict.bar.text}` — but the precondition above it, "
            f"`{watch.precondition}`, has not been recorded as opened, so the "
            f"reading is not yet a meaningful number."
        )
        lines += [
            "",
            f"`{registers.OVERDUE_LABEL}` is **not** added, and this is recorded "
            "rather than passed over so the row does not sit in *no recorded "
            "run* while a rig is in fact reading it "
            "([#417](https://github.com/NGL321/patchworks/issues/417)). Once the "
            "precondition opens, the next crossing stamps the label.",
            "",
            "Filed by `tools/cutoff_report.py` (#284) from a run of "
            f"`benchmarks/{watch.rig}.py`.",
        ]
        return "\n".join(lines) + "\n"
    if verdict.crossed is True:
        lines.append(
            f"**The cutoff on this problem was crossed.** `{watch.rig}` ran and read "
            f"`{verdict.bar.metric} = {verdict.reading:g}` against the bar "
            f"`{verdict.bar.text}`."
        )
        lines += [
            "",
            f"The problem is *{watch.failure}*, and it is now due: "
            f"`{registers.OVERDUE_LABEL}` is on this issue. A rig asserts "
            "nothing, so this is a report and a label — the run did not fail and "
            "nothing is blocked by it.",
        ]
    elif verdict.crossed is False:
        lines.append(
            f"`{watch.rig}` ran and read `{verdict.bar.metric} = "
            f"{verdict.reading:g}` against the bar `{verdict.bar.text}`. "
            "**Not crossed.**"
        )
        lines += [
            "",
            "Recorded because a bar with nothing reporting against it reads as "
            "watched and is not — `docs/registers/open-problems.md`, *Cutoffs "
            "naming a rig with no recorded run*.",
        ]
    else:
        lines.append(
            f"`{watch.rig}` ran and **could not evaluate this cutoff**: "
            f"{verdict.why}. The bar as the issue states it is "
            f"`{watch.threshold}`."
        )
        lines += [
            "",
            "Recorded rather than passed over: a cutoff nothing can fire on reads "
            "as watched and is not, and only the run can see that.",
        ]
    lines += [
        "",
        "Filed by `tools/cutoff_report.py` (#284) from a run of "
        f"`benchmarks/{watch.rig}.py`.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# the networked half
# ---------------------------------------------------------------------------


def gh(arguments: list[str], stdin: str | None = None) -> str:
    """One `gh` call. The only thing here that reaches the network."""
    finished = subprocess.run(
        ["gh", *arguments],
        input=stdin,
        capture_output=True,
        text=True,
        # Named rather than left to the locale, as `problem_registers.fetch`
        # names it: `gh` speaks UTF-8, the comment this files is full of em
        # dashes, and cp1252 cannot carry them in either direction.
        encoding="utf-8",
        check=True,
        cwd=ROOT,
    )
    return finished.stdout


#: The keys a record can sign itself with, per field. Both of a field's keys are
#: read back: a run that evaluated the bar and a run that could not are both
#: things this rig has already said, and only the field's *other* pair is
#: somebody else's record.
_KEYS = {
    CUTOFF_FIELD: (registers.REPORT_KEY, UNEVALUATED_KEY),
    PRECONDITION_FIELD: (
        registers.PRECONDITION_KEY,
        registers.UNEVALUATED_PRECONDITION_KEY,
    ),
}


def read_stamps(
    comments: list[dict], rig: str, *, field: str = CUTOFF_FIELD
) -> str | None:
    """The most recent stamp *rig* left in *comments* against *field*, or `None`.

    Both of the field's keys are read: a run that evaluated the bar signs itself
    `@rig` (or `@precondition`), and a run that could not signs itself
    `@unevaluated` (or `@unevaluated_precondition`), and the second is still a
    thing this rig has already said. A comment carrying the key but no verdict
    is **passed over rather than treated as a reset** -- a `@rig` note somebody
    wrote by hand is not this module's record, and clearing the stamp on one
    would make the next run file a verdict the issue already carries.
    """
    found = None
    for _, fields in _blocks(comments):
        named = _named(fields, _KEYS[field])
        if not named or registers.rig_name(named) != rig:
            continue
        word = (fields.get(VERDICT_KEY) or [""])[0].strip()
        if word:
            found = _stamp(rig, word)
    return found


def _blocks(comments: list[dict]) -> list[tuple[dict, dict[str, list[str]]]]:
    """Every comment carrying a readable field block. Unreadable ones are skipped.

    A malformed comment is `problem_registers.py`'s to surface, in the register's
    own *comments this register could not read* section. A rig is not the place
    it should reach a human, and a rig that stopped on one would cost the run
    its readings.
    """
    found = []
    for entry in comments or []:
        try:
            fields = registers.field_block(entry.get("body") or "")
        except registers.MalformedProvenance:
            continue
        if fields:
            found.append((entry, fields))
    return found


def _named(fields: dict[str, list[str]], keys: tuple[str, ...]) -> str:
    for key in keys:
        values = fields.get(key) or []
        if values:
            return values[0]
    return ""


def precondition_opened(comments: list[dict], subject: str) -> bool:
    """Whether anything has recorded *subject*'s precondition as opened.

    **Read off the record rather than recomputed**, because a rig cannot
    evaluate another rig's bar: #329's precondition names `detectability` and
    its cutoff names `driven_settling`, and the run that has the settling
    readings has none of the detectability ones. What this module can ask is
    whether a `@precondition` record on this problem says the bar opened, and
    that record is filed by the run that *did* have the readings.

    A precondition nobody has ever reported on reads as shut, and that is not a
    silent state: `problem_registers.unwatched` puts exactly that row in the
    register's second loud section, under the precondition arm, naming which of
    the two fields is stuck. The debt is shown rather than guessed away.
    """
    if not subject:
        return True
    opened = False
    for _, fields in _blocks(comments):
        named = _named(fields, (registers.PRECONDITION_KEY,))
        if not named or registers.rig_name(named) != registers.rig_name(subject):
            continue
        word = (fields.get(VERDICT_KEY) or [""])[0].strip()
        if word:
            opened = word == _GATE_WORD[True]
    return opened


def comments_on(number: int) -> list[dict]:
    """The comments the tracker holds on a problem.

    One call, and every reader of the record is handed the result rather than
    fetching its own: the withholding check and the dedupe both read these, and
    two calls would be two chances for them to disagree about what is on record.
    """
    payload = json.loads(gh(["issue", "view", str(number), "--json", "comments"]))
    return payload.get("comments") or []


def last_verdict(number: int, rig: str, *, field: str = CUTOFF_FIELD) -> str | None:
    """:func:`read_stamps`, over the comments the tracker holds.

    Read off the comments rather than kept in a file: the record lives where the
    register reads it, and a second copy on disk is a second place the same fact
    lives.
    """
    return read_stamps(comments_on(number), rig, field=field)


def file_report(verdict: Verdict) -> str:
    """File the comment, and the label on a crossing. Returns what it did.

    Every failure is a returned sentence rather than a raised exception: `gh`
    missing, unauthenticated, rate-limited or offline are all ordinary states
    for a rig running on a laptop, and none of them may cost the run its
    readings.
    """
    watch = verdict.watch
    try:
        comments = comments_on(watch.number)
        # A crossing behind a shut precondition is recorded and carries no
        # obligation (#417). Only a crossing: a clear reading obliges nothing
        # either way, and a precondition's own record never obliges at all.
        withheld = bool(
            watch.obliges
            and verdict.crossed
            and not precondition_opened(comments, gate_subject(watch.precondition))
        )
        prior = read_stamps(comments, watch.rig, field=watch.field)
        if not worth_filing(verdict, prior, withheld=withheld):
            # The verdict, not the stamp taken apart again: `stamp` is the only
            # thing that knows the shape, and the two agreeing is what got us
            # here.
            return f"already on record ({WITHHELD if withheld else verdict.word})"
        gh(
            ["issue", "comment", str(watch.number), "--body-file", "-"],
            stdin=comment_body(verdict, withheld=withheld),
        )
        done = "comment filed"
        if withheld:
            return (
                f"{done}, `{registers.OVERDUE_LABEL}` withheld: the precondition "
                f"`{watch.precondition}` is not on record as opened"
            )
        if verdict.crossed and watch.obliges:
            gh(
                [
                    "issue",
                    "edit",
                    str(watch.number),
                    "--add-label",
                    registers.OVERDUE_LABEL,
                ]
            )
            done += f", `{registers.OVERDUE_LABEL}` added"
        return done
    except FileNotFoundError:
        return "not filed: no `gh` on this machine"
    except subprocess.CalledProcessError as failure:
        detail = (failure.stderr or "").strip().splitlines()
        return f"not filed: gh said {detail[-1] if detail else failure}"
    except (json.JSONDecodeError, OSError) as failure:
        return f"not filed: {failure}"


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


def report(
    rig: str,
    readings: dict[str, float],
    *,
    register: pathlib.Path | None = REGISTER,
    file: bool = True,
) -> list[Verdict]:
    """State, for each open problem cutting on *rig*, whether the bar was crossed.

    This is the whole hook, and a rig calls it once at the end of its run with
    what it measured. It prints, it files, and it returns the verdicts; it does
    not raise and it does not decide the exit code.

    **Two passes, and the second is not a cutoff's** (#417). After the problems
    whose `@cutoff` names this rig come the problems whose `@when` precondition
    does, and the report states for each whether the precondition opened. A rig
    may well have the second and not the first: `detectability` gates #325, #329
    and #341 and cuts on none of them.

    `register` is a path and `None` says there is no register to read -- which
    is the same state a missing file leaves, and is reported rather than raised
    for the same reason everything else here is.
    """
    name = registers.rig_name(rig)
    text = (
        register.read_text(encoding="utf-8")
        if register is not None and register.is_file()
        else None
    )
    print(f"\n== cutoffs naming `{name}` ==")
    if text is None:
        print(
            f"   the open problems register is not readable at "
            f"{register or REGISTER}: no cutoff could be evaluated. Regenerate "
            "it with `python tools/problem_registers.py`."
        )
        return []
    found = watching(text, name)
    if not found:
        print(
            f"   no open problem cuts on `{name}`. Nothing to evaluate, and that "
            "is a statement rather than a silence."
        )
    verdicts = _evaluate(found, readings, file)
    if found:
        print(
            "   A rig asserts nothing: a crossing above is a report and a label, "
            "not a failure."
        )

    gating = watching(text, name, field=PRECONDITION_FIELD)
    if gating:
        # Reported because once a precondition is nobody's `@cutoff`, no other
        # report mentions it, and the day it opens several problems become live
        # with nothing saying so (#417).
        print(f"\n== preconditions naming `{name}` ==")
        verdicts += _evaluate(gating, readings, file)
        print(
            "   Opening a precondition imposes nothing: it makes the cutoff "
            "below it a readable number, and files no label."
        )
    return verdicts


def _evaluate(
    found: list[Watch], readings: dict[str, float], file: bool
) -> list[Verdict]:
    """Judge, print and file one pass. The two passes differ only in what they
    were given, which is the point: same grammar, same bar reader, same record.
    """
    verdicts = [judge(watch, readings) for watch in found]
    for verdict in verdicts:
        line = verdict.line
        if file:
            line += f"   [{file_report(verdict)}]"
        print(line)
        # ASCII, unlike the comment this files: the report goes to whatever
        # console the rig was run from, and on Windows that is still cp1252,
        # which turns an em dash into a replacement character mid-report.
        print(f"         {verdict.watch.title} - {verdict.watch.url}")
    return verdicts


def _reading(pair: str) -> tuple[str, float]:
    metric, _, value = pair.partition("=")
    return metric.strip(), float(value)


def main(argv: list[str] | None = None) -> int:
    """A human at a terminal, evaluating a rig's readings by hand.

    The same arm `problem_registers.py --check` exists for: the hook belongs in
    the rig, and this is how it is exercised without one.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rig", help="the rig whose cutoffs to evaluate")
    parser.add_argument(
        "readings",
        nargs="*",
        type=_reading,
        metavar="metric=value",
        help="what the run measured",
    )
    parser.add_argument(
        "--no-file",
        dest="file",
        action="store_false",
        help="print the report and file nothing",
    )
    parser.add_argument(
        "--register",
        type=pathlib.Path,
        default=REGISTER,
        help=(
            "the rendered open problems register to read (default: the "
            "checked-in one). A path rather than a fixed file so the hook can be "
            "exercised against a register the working tree does not carry"
        ),
    )
    arguments = parser.parse_args(argv)
    report(
        arguments.rig,
        dict(arguments.readings),
        register=arguments.register,
        file=arguments.file,
    )
    # Zero whatever was found, for the reason the whole module exists: a crossed
    # cutoff is a report, and an exit code is an assertion.
    return 0


if __name__ == "__main__":
    sys.exit(main())
