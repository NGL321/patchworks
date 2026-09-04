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

#: A `measurement` cutoff as `problem_registers.Cutoff.text` renders it.
_CUTOFF_CELL = re.compile(r"^measurement\s+`(?P<rig>[^`]+)`\s+—\s+(?P<threshold>.+)$")

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


# ---------------------------------------------------------------------------
# which problems cut on me
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Watch:
    """One open problem whose `measurement` cutoff names this rig."""

    number: int
    url: str
    title: str
    failure: str
    rig: str
    threshold: str

    @property
    def ref(self) -> str:
        return f"#{self.number}"


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


def watching(text: str, rig: str) -> list[Watch]:
    """Every open problem in the rendered register whose cutoff names *rig*.

    The rig may be written three ways -- `benchmarks/x.py`, `x.py` or `x` --
    and `problem_registers.rig_name` is what settles which is which, here as
    there. Rows this module cannot read are passed over rather than raised on:
    the register is the authority on its own grammar, and a rig is not the place
    a malformed row should surface.
    """
    wanted = registers.rig_name(rig)
    found = []
    for cells in _rows(text):
        if len(cells) < 5:
            continue
        title, failure, cutoff, _discovered, issue = cells[:5]
        bar = _CUTOFF_CELL.match(cutoff)
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
            )
        )
    return found


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
    #: checked* and *the bar held* are opposite things for a reader.
    crossed: bool | None
    why: str

    @property
    def bar_text(self) -> str:
        return self.bar.text if self.bar is not None else self.watch.threshold

    @property
    def word(self) -> str:
        return _WORD[self.crossed]

    @property
    def line(self) -> str:
        """The report's line for this problem: the bar, the reading, the verdict."""
        head = f"  {self.watch.ref:<6} bar `{self.bar_text}`"
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


def stamp(verdict: Verdict) -> str:
    """What this run would leave on record: the rig and its verdict, not the number.

    Two clear runs differ in their reading and say the same thing. Stamping the
    reading would file a comment on every run of every rig, which is the volume
    that turns a channel into noise.
    """
    return _stamp(verdict.watch.rig, verdict.word)


def worth_filing(verdict: Verdict, prior: str | None) -> bool:
    """Whether this verdict says anything the problem does not already record.

    The first record from a rig always does -- it is the one the register's *no
    recorded run* section is waiting for -- and so does a change of verdict.
    """
    return prior is None or prior != stamp(verdict)


def comment_body(verdict: Verdict) -> str:
    """The comment this run files on the problem.

    It opens with the field block `problem_registers.py` reads, because that
    block is the whole reason the comment exists: it is how the register knows
    the bar has something watching it. Everything under the fence is prose the
    generator never reads, and the shape of that prose is the
    `overdue-provenance` channel's -- the fact, the reading, and what raised it.
    """
    watch = verdict.watch
    key = registers.REPORT_KEY if verdict.crossed is not None else UNEVALUATED_KEY
    lines = [
        "```",
        f"@{key} {watch.rig}",
        f"@{VERDICT_KEY} {verdict.word}",
        "```",
        "",
    ]
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


def read_stamps(comments: list[dict], rig: str) -> str | None:
    """The most recent stamp *rig* left in *comments*, or `None`.

    Both keys are read: a run that evaluated the bar signs itself `@rig`, and a
    run that could not signs itself `@unevaluated`, and the second is still a
    thing this rig has already said. A comment carrying the key but no verdict
    is **passed over rather than treated as a reset** -- a `@rig` note somebody
    wrote by hand is not this module's record, and clearing the stamp on one
    would make the next run file a verdict the issue already carries.
    """
    found = None
    for entry in comments or []:
        try:
            fields = registers.field_block(entry.get("body") or "")
        except registers.MalformedProvenance:
            continue
        if not fields:
            continue
        named = fields.get(registers.REPORT_KEY) or fields.get(UNEVALUATED_KEY) or []
        if not named or registers.rig_name(named[0]) != rig:
            continue
        word = (fields.get(VERDICT_KEY) or [""])[0].strip()
        if word:
            found = _stamp(rig, word)
    return found


def last_verdict(number: int, rig: str) -> str | None:
    """:func:`read_stamps`, over the comments the tracker holds.

    Read off the comments rather than kept in a file: the record lives where the
    register reads it, and a second copy on disk is a second place the same fact
    lives.
    """
    payload = json.loads(gh(["issue", "view", str(number), "--json", "comments"]))
    return read_stamps(payload.get("comments") or [], rig)


def file_report(verdict: Verdict) -> str:
    """File the comment, and the label on a crossing. Returns what it did.

    Every failure is a returned sentence rather than a raised exception: `gh`
    missing, unauthenticated, rate-limited or offline are all ordinary states
    for a rig running on a laptop, and none of them may cost the run its
    readings.
    """
    watch = verdict.watch
    try:
        prior = last_verdict(watch.number, watch.rig)
        if not worth_filing(verdict, prior):
            # The verdict, not the stamp taken apart again: `stamp` is the only
            # thing that knows the shape, and the two agreeing is what got us
            # here.
            return f"already on record ({verdict.word})"
        gh(
            ["issue", "comment", str(watch.number), "--body-file", "-"],
            stdin=comment_body(verdict),
        )
        done = "comment filed"
        if verdict.crossed:
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
        return []
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
    print(
        "   A rig asserts nothing: a crossing above is a report and a label, "
        "not a failure."
    )
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
