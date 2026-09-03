"""The three problem registers, read off the tracker (#279, #282).

`tools/constant_registers.py` projects the provenance of every constant off its
definition site. This module does the same job for the architecture's open
questions, whose definition site is a GitHub issue: it renders
`docs/registers/open-problems.md`, `proposed-solutions.md` and
`dismissed-solutions.md` from `gh`.

**It is a projection and it holds no argument.** A row carries a link back to
the issue or the comment and nothing more. The argument lives at the definition
site, and a second copy is a second place a decision lives, which is the failure
#180 exists to kill. The rendered files carry the same do-not-edit banner the
constants registers carry, and for the same reason.

## What it reads

Labels, and field blocks. Never prose.

* issues labelled `register:problem`, **open and closed both** -- a closed
  problem stays in the register carrying its ground, because the set of problems
  that dissolved on their own is the direct evidence for the bet this project is
  making;
* issues labelled `register:proposal`;
* issues labelled `register:dismissal`;
* **the comments on every `register:problem` issue**, because a proposal
  specific to one problem lives as a comment there rather than as its own issue.
  That is what makes a problem ticket a single place to read, and it is why this
  cannot be a labels-only query. The comments are also where a rig report
  records a run (#284), which is how the register answers *has anything ever
  fired against this bar*.

The grammar is `docs/agents/registers.md`, *Field blocks*: a fenced block at the
top of the body or comment, one `@key value` per line, everything below it prose
the generator never reads.

## What it refuses

Loudly, as `constant_registers.py` refuses malformed provenance -- a register
that renders a half-read row is worse than one that stops, because the row looks
like provenance and is not.

* a proposal with no `@source` or no `@shape`;
* a `@status dismissed failed` naming no rig and no reading, because "we tried
  it" without a reading is unfalsifiable folklore and the whole pre-registration
  discipline exists to keep that out;
* a `@cutoff` outside `event <issue>`, `measurement <rig> <threshold>` and
  `uncut` -- dates and judgement are not admissible, because a cutoff must be
  checkable by someone who is not its author.

**One rule is deliberately not enforced here.** `@source here` on a body that
carries no argument is a real failure and a judgement call, and not the
generator's to make: whether prose argues a mechanism is not decidable by a
parser, and one that guessed would refuse good entries and pass bad ones.

## Where it sits relative to the suite

It shells `gh`, so it is a **network** tool and lives on the far side of the
line `tests/test_cli.py` defends: the suite never reaches it and must not learn
how. Everything below :func:`fetch` is a pure function of a payload, which is
the seam `tests/test_problem_registers.py` tests against fixtures.

Freshness is a workflow's job (#283). ``--check`` exists for a human at a
terminal; CI never runs it, because CI cannot ask GitHub anything offline.

    python tools/problem_registers.py
    python tools/problem_registers.py --check
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "benchmarks"
OUTPUT = ROOT / "docs" / "registers"

#: The issue *is* an open problem. Open means unresolved, closed means resolved
#: with a ground, and both are queried: a closed problem stays in the register.
PROBLEM_LABEL = "register:problem"

#: The issue *is* an orphaned or multifarious proposal. A proposal specific to
#: one problem carries no label at all, because it is a comment.
PROPOSAL_LABEL = "register:proposal"

#: Terminal. Co-occurs with `register:proposal` when a proposal was dismissed,
#: and stands alone on a dismissal that was never one.
DISMISSAL_LABEL = "register:dismissal"

#: A cutoff that fired. The rig report adds it (#284); this module only reads
#: it, so that a problem the tracker already knows is overdue does not read as
#: quietly watched on the page.
OVERDUE_LABEL = "register:overdue"

#: What `gh issue list` is asked for. `comments` is the expensive field and the
#: necessary one: the comment arm is what makes this not a labels-only query.
FIELDS = "number,title,body,state,url,labels,comments"

#: `gh issue list`'s page. A register silently truncated at its limit is the
#: exact failure this whole mechanism exists to prevent -- a projection that
#: reads as complete and is not -- so :func:`fetch` refuses rather than paging:
#: reaching this number means the register has outgrown one query, and that is a
#: thing for a human to see rather than for the tool to paper over.
LIMIT = 500

#: A field line inside a block: `@key`, then the rest of the line.
_FIELD = re.compile(r"^@(?P<key>\w+)(?:\s+(?P<value>.*))?$")

#: A fence opening or closing a block, with or without an info string.
_FENCE = re.compile(r"^(?:```|~~~)")

#: An issue reference, written `#230` or `230`; both are the same issue.
_ISSUE = re.compile(r"^#?(?P<number>\d+)$")

#: The key a rig report's field block names itself with, on a comment on the
#: problem whose cutoff it evaluated (#284). This module only **reads** it, to
#: answer *has anything ever reported against this cutoff* -- the question
#: #282's second loud section asks and #284 explicitly declines to answer
#: ("do not try to close it here"). A rig that has never reported leaves the
#: problem looking watched while nothing fires.
REPORT_KEY = "rig"

#: The keys that make a field block a proposal's. A comment carrying any of them
#: has declared itself provenance, so a missing `@proposal` is refused rather
#: than dropped -- a row nobody can reach, that nobody was told about, is the
#: silent incompleteness this whole mechanism exists to prevent. A comment
#: carrying none of them (a rig report, a pointer, ordinary discussion) is not a
#: proposal and is passed over without comment.
PROPOSAL_KEYS = frozenset({"source", "shape", "answers", "when", "status"})


class MalformedProvenance(Exception):
    """An issue or comment whose field block cannot be read as provenance."""


# ---------------------------------------------------------------------------
# the field block
# ---------------------------------------------------------------------------


def field_block(text: str) -> dict[str, list[str]] | None:
    """The fenced field block at the top of *text*, or `None` if there is none.

    Two states are distinguished on purpose, because collapsing them is how a
    register starts refusing ordinary conversation. A body or comment with no
    field block at its top **is not provenance**: most comments on a problem
    issue are discussion, and a comment that opens with a quoted traceback is a
    code block rather than a malformed entry. The discriminator is the first
    line inside the fence -- a block that opens `@key` has declared itself
    provenance, and from there it is held to the grammar and a loose line is
    refused.

    Values accumulate, because `@shape` is a list. Everything after the closing
    fence is prose and is never read.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start >= len(lines) or not _FENCE.match(lines[start].strip()):
        return None
    body: list[str] = []
    for line in lines[start + 1:]:
        if _FENCE.match(line.strip()):
            break
        body.append(line)
    inside = [line for line in body if line.strip()]
    if not inside or not _FIELD.match(inside[0].strip()):
        return None
    found: dict[str, list[str]] = {}
    for line in inside:
        match = _FIELD.match(line.strip())
        if match is None:
            raise MalformedProvenance(
                f"a field block opened with `@`, so {line.strip()!r} is not a "
                "field: one `@key value` per line, and prose goes below the fence"
            )
        found.setdefault(match.group("key"), []).append(
            (match.group("value") or "").strip()
        )
    return found


def _one(fields: dict[str, list[str]], key: str) -> str:
    """The first value of a key, or the empty string. Most keys are not lists."""
    values = fields.get(key) or []
    return values[0] if values else ""


# ---------------------------------------------------------------------------
# cutoffs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cutoff:
    """The point at which a problem stops being tolerable.

    Two admissible forms and one admitted absence. `uncut` is a real value and
    the loud one, exactly as `@flexibility unknown` is in the constants
    register: *nobody has said when this stops being tolerable* is a fact, and
    hiding it collapses it into *somebody said and the register lost it*, which
    are opposite states for a reader deciding whether the problem is watched.
    """

    kind: str
    issue: str = ""
    rig: str = ""
    threshold: str = ""

    @property
    def text(self) -> str:
        if self.kind == "event":
            return f"event {self.issue}"
        if self.kind == "measurement":
            return f"measurement `{self.rig}` — {self.threshold}"
        return "`uncut`"


def rig_name(written: str) -> str:
    """`benchmarks/detectability.py`, `detectability.py` and `detectability`.

    One rig written three ways. Normalising here rather than at the definition
    site is deliberate: the register may not tell an author how to spell a path
    they are already reading off their own shell history.
    """
    name = written.strip()
    name = name.removeprefix("benchmarks/").removeprefix("./")
    return name.removesuffix(".py")


def read_cutoff(value: str, where: str) -> Cutoff:
    """One `@cutoff` field, checked. Dates and judgement are refused.

    A cutoff must be checkable by someone who is not its author. "When it
    becomes a problem" is not a cutoff but the absence of one, and `uncut`
    already says that -- in the register's loudest voice, which is the point.
    """
    parts = value.split()
    if not parts:
        raise MalformedProvenance(f"{where}: @cutoff is required, and `uncut` is a value")
    head, rest = parts[0], parts[1:]
    if head == "uncut":
        return Cutoff("uncut")
    if head == "event":
        if not rest:
            raise MalformedProvenance(
                f"{where}: @cutoff event names the issue whose closing fires it"
            )
        if len(rest) > 1:
            # `event 230 when it feels bad` would otherwise parse as `#230` and
            # discard the judgement in silence -- admitting exactly what the
            # two-forms rule exists to refuse, while reading as well-formed.
            raise MalformedProvenance(
                f"{where}: @cutoff event names one issue and nothing else, got "
                f"{' '.join(rest)!r}"
            )
        match = _ISSUE.match(rest[0])
        if match is None:
            raise MalformedProvenance(
                f"{where}: @cutoff event wants an issue number, got {rest[0]!r}"
            )
        return Cutoff("event", issue="#" + match.group("number"))
    if head == "measurement":
        if len(rest) < 2:
            raise MalformedProvenance(
                f"{where}: @cutoff measurement names a rig and a threshold; a rig "
                "with no bar cannot be crossed, so nothing would ever fire"
            )
        return Cutoff("measurement", rig=rig_name(rest[0]), threshold=" ".join(rest[1:]))
    raise MalformedProvenance(
        f"{where}: @cutoff {value!r} is not `event <issue>`, "
        "`measurement <rig> <threshold>` or `uncut` — dates and judgement are "
        "not admissible"
    )


def read_when(value: str, where: str) -> Cutoff:
    """`@when`: a cutoff's mirror, opposite polarity, no obligation attached.

    A cutoff says *this problem stops being tolerable*; `@when` says *this
    proposal starts being relevant*. `uncut` has nothing to say here, because
    there is no obligation for it to be the absence of.
    """
    cutoff = read_cutoff(value, where)
    if cutoff.kind == "uncut":
        raise MalformedProvenance(
            f"{where}: @when is `event <issue>` or `measurement <rig> <threshold>`; "
            "it carries no obligation, so `uncut` says nothing"
        )
    return cutoff


# ---------------------------------------------------------------------------
# problems
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """One open problem, as its issue states it."""

    number: int
    title: str
    url: str
    state: str
    failure: str
    cutoff: Cutoff
    discovered: str = ""
    overdue: bool = False
    #: Every rig that has reported against this problem, read off the `@rig`
    #: field blocks in its comments. Empty is the state that matters: a
    #: `measurement` cutoff nothing has ever reported against is the disguise.
    reports: frozenset[str] = frozenset()

    @property
    def resolved(self) -> bool:
        """A closed problem stays in the register, in a resolved section.

        The set of problems that dissolved on their own is the direct evidence
        for the bet this project is making -- that constraint left out is
        supplied by the architecture rather than specified. Deleting the row
        deletes the evidence.
        """
        return self.state.upper() == "CLOSED"

    @property
    def ref(self) -> str:
        return f"#{self.number}"

    @property
    def link(self) -> str:
        return f"[#{self.number}]({self.url})"

    @property
    def sort_key(self) -> tuple[int, int]:
        """`uncut` first. The debt sorts above the problems being watched."""
        return (0 if self.cutoff.kind == "uncut" else 1, self.number)


def read_problem(payload: dict) -> Problem:
    """One `register:problem` issue, checked."""
    where = f"#{payload['number']}"
    fields = field_block(payload.get("body") or "")
    if fields is None:
        raise MalformedProvenance(
            f"{where}: a problem needs a field block, fenced, at the top of the "
            "body — see docs/agents/registers.md, Field blocks"
        )
    failure = _one(fields, "failure")
    if not failure:
        raise MalformedProvenance(
            f"{where}: @failure is required — a problem is admitted if it has a "
            "statable failure someone could recognise happening"
        )
    return Problem(
        number=int(payload["number"]),
        title=payload.get("title") or "",
        url=payload.get("url") or "",
        state=payload.get("state") or "OPEN",
        failure=failure,
        cutoff=read_cutoff(_one(fields, "cutoff"), where),
        discovered=_one(fields, "discovered"),
        overdue=OVERDUE_LABEL in _labels(payload),
        reports=_reports(payload),
    )


def _labels(payload: dict) -> set[str]:
    return {entry.get("name", "") for entry in payload.get("labels") or []}


def _reports(payload: dict) -> frozenset[str]:
    """Every rig that has reported against this problem, off its comments.

    The rig report (#284) files on the problem issue; a field block naming
    `@rig` is that report. Reading it here is what lets the register answer
    *has anything ever fired against this bar* without a run ledger, which this
    repository does not have -- rig readings live as prose in ADRs, spec
    sections and research docs, none of which a projection may parse.
    """
    found: set[str] = set()
    for entry in payload.get("comments") or []:
        fields = field_block(entry.get("body") or "")
        if fields and REPORT_KEY in fields:
            found.update(rig_name(value) for value in fields[REPORT_KEY] if value)
    return frozenset(found)


# ---------------------------------------------------------------------------
# proposals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Status:
    """`open`, `adopted <ADR>`, `dismissed refused`, `dismissed failed …`.

    Edited in place on the issue as it changes; the comment thread around it is
    the history, and this register never restates the history.
    """

    kind: str
    adr: str = ""
    dismissal: str = ""
    rig: str = ""
    reading: str = ""

    @property
    def text(self) -> str:
        if self.kind == "adopted":
            return f"adopted {self.adr}"
        if self.kind == "dismissed" and self.dismissal == "failed":
            return f"dismissed failed — `{self.rig}`: {self.reading}"
        if self.kind == "dismissed":
            return "dismissed refused"
        return "open"


def read_status(value: str, where: str) -> Status:
    """One `@status` field. A missing one reads as `open`, the resting state."""
    parts = value.split()
    if not parts or parts[0] == "open":
        return Status("open")
    head, rest = parts[0], parts[1:]
    if head == "adopted":
        if not rest:
            raise MalformedProvenance(
                f"{where}: @status adopted names the ADR that now covers it"
            )
        return Status("adopted", adr=" ".join(rest))
    if head == "dismissed":
        if rest and rest[0] == "refused":
            return Status("dismissed", dismissal="refused")
        if rest and rest[0] == "failed":
            if len(rest) < 3:
                raise MalformedProvenance(
                    f"{where}: @status dismissed failed must name the rig and the "
                    "reading — \"we tried it\" without a reading is folklore, and "
                    "the pre-registration discipline exists to keep that out"
                )
            return Status(
                "dismissed",
                dismissal="failed",
                rig=rig_name(rest[1]),
                reading=" ".join(rest[2:]),
            )
    raise MalformedProvenance(
        f"{where}: @status {value!r} is not `open`, `adopted <ADR>`, "
        "`dismissed refused` or `dismissed failed <rig> <reading>`"
    )


@dataclass(frozen=True)
class Proposal:
    """One solution on the shelf, arguing a shape it would answer, binding nothing.

    Its own issue when it is orphaned or answers several problems; a comment on
    the problem issue when it is specific to one. Both are first-class rows.
    """

    title: str
    source: str
    shapes: tuple[str, ...]
    where: str
    url: str
    number: int
    answers: tuple[str, ...] = ()
    when: Cutoff | None = None
    status: Status = Status("open")
    is_comment: bool = False

    @property
    def link(self) -> str:
        return f"[{self.where}]({self.url})"

    @property
    def dismissed(self) -> bool:
        return self.status.kind == "dismissed"

    @property
    def sort_key(self) -> tuple[int, int, str]:
        return (self.number, 1 if self.is_comment else 0, self.url)


def read_proposal(
    text: str,
    *,
    where: str,
    url: str,
    number: int,
    title: str = "",
    is_comment: bool = False,
) -> Proposal:
    """One proposal, from an issue body or a comment body, checked.

    A proposal is admitted if it has a **source** and at least one **shape**.
    Shape is the index -- an agent arrives with a symptom, not with the name it
    was going to give its solution -- so a proposal with none cannot be arrived
    at, and a row nobody can reach is worse than no row.

    What is *not* checked here is whether a `@source here` body actually argues
    its mechanism. That rule is real and it is a reader's; a parser guessing at
    it would refuse good entries and pass bad ones.
    """
    fields = field_block(text)
    if fields is None:
        raise MalformedProvenance(
            f"{where}: a proposal needs a field block, fenced, at the top — see "
            "docs/agents/registers.md, Field blocks"
        )
    named = _one(fields, "proposal") or title
    if not named:
        raise MalformedProvenance(f"{where}: @proposal names the proposal")
    source = _one(fields, "source")
    if not source:
        raise MalformedProvenance(
            f"{where}: @source is required — a citation, a rig reading, a research "
            "doc, a session, or `here`"
        )
    shapes = tuple(s for s in fields.get("shape", []) if s)
    if not shapes:
        raise MalformedProvenance(
            f"{where}: @shape is required — it is the index, so a proposal with "
            "none cannot be arrived at by an agent holding the symptom"
        )
    when_field = _one(fields, "when")
    return Proposal(
        title=named,
        source=source,
        shapes=shapes,
        where=where,
        url=url,
        number=number,
        answers=_answers(fields.get("answers", []), where),
        when=read_when(when_field, where) if when_field else None,
        status=read_status(_one(fields, "status"), where),
        is_comment=is_comment,
    )


def _answers(values: list[str], where: str) -> tuple[str, ...]:
    """`@answers`, repeatable and comma-separated alike, normalised to `#N`.

    Stating a shape never binds a proposal to a problem; this field is what
    binding is, and an orphan simply has none.
    """
    found = []
    for value in values:
        for part in value.replace(",", " ").split():
            match = _ISSUE.match(part)
            if match is None:
                raise MalformedProvenance(
                    f"{where}: @answers wants problem issue numbers, got {part!r}"
                )
            found.append("#" + match.group("number"))
    return tuple(found)


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


@dataclass
class Registers:
    """What one survey found, before it is split three ways."""

    problems: list[Problem] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)

    @property
    def shelved(self) -> list[Proposal]:
        return [p for p in self.proposals if not p.dismissed]

    @property
    def dismissals(self) -> list[Proposal]:
        return [p for p in self.proposals if p.dismissed]


def collect(
    problems: list[dict], proposals: list[dict], dismissals: list[dict]
) -> Registers:
    """The three queries, plus the comments, folded into one survey.

    The comment arm is the whole reason this is not a labels-only query: a
    proposal specific to one problem lives as a comment on that problem's issue,
    which is what makes the problem ticket a single place to read.

    A comment with no field block is passed over in silence. Discussion is not
    malformed provenance, and a register that refused ordinary conversation
    would be abandoned within a week.
    """
    survey = Registers()
    seen: set[tuple[str, object]] = set()

    for payload in problems:
        survey.problems.append(read_problem(payload))
        number = int(payload["number"])
        for entry in payload.get("comments") or []:
            body = entry.get("body") or ""
            fields = field_block(body)
            if fields is None:
                continue
            if "proposal" not in fields:
                if PROPOSAL_KEYS & fields.keys():
                    raise MalformedProvenance(
                        f"#{number}: a comment carrying "
                        f"{', '.join('@' + k for k in sorted(PROPOSAL_KEYS & fields.keys()))} "
                        "has declared itself a proposal and must name one with "
                        "@proposal; dropping it would leave a row nobody can reach"
                    )
                continue
            key = ("comment", entry.get("url") or "")
            if key in seen:
                continue
            seen.add(key)
            survey.proposals.append(
                read_proposal(
                    body,
                    where=f"#{number} (comment)",
                    url=entry.get("url") or payload.get("url") or "",
                    number=number,
                    is_comment=True,
                )
            )

    for payload in list(proposals) + list(dismissals):
        number = int(payload["number"])
        key = ("issue", number)
        if key in seen:
            continue
        seen.add(key)
        found = read_proposal(
            payload.get("body") or "",
            where=f"#{number}",
            url=payload.get("url") or "",
            number=number,
            title=payload.get("title") or "",
        )
        if DISMISSAL_LABEL in _labels(payload) and not found.dismissed:
            # A dismissal that was never a proposal still uses the proposal
            # field block -- it is the only one `docs/agents/registers.md`
            # defines -- and it must state which of the two kinds it is, because
            # `refused` and `failed` behave differently and `failed` owes a rig
            # and a reading. Without `@status` the row would default to `open`
            # and render on the shelf, which is a binding exclusion advertised
            # as a live proposal: the loudest way this register could be wrong.
            raise MalformedProvenance(
                f"#{number}: carries `{DISMISSAL_LABEL}`, which is terminal, while "
                f"@status says {found.status.text!r} — write `@status dismissed "
                "refused` or `@status dismissed failed <rig> <reading>` (see "
                "docs/agents/registers.md, Dismissed solutions)"
            )
        survey.proposals.append(found)

    survey.problems.sort(key=lambda p: p.sort_key)
    survey.proposals.sort(key=lambda p: p.sort_key)
    return survey


def available_rigs() -> frozenset[str]:
    """Every rig with a script under `benchmarks/`."""
    if not BENCHMARKS.is_dir():
        return frozenset()
    return frozenset(path.stem for path in BENCHMARKS.glob("*.py"))


#: Why a `measurement` cutoff will not fire, in the two states the tracker and
#: the working tree can tell apart.
NO_SCRIPT = "names no rig in `benchmarks/`, so there is nothing to run"
NO_RUN = "has never reported against this problem, so nothing has fired"


@dataclass(frozen=True)
class Unwatched:
    """A `measurement` cutoff that reads as watched and is not."""

    problem: Problem
    reason: str


def unwatched(
    survey: Registers, rigs: frozenset[str] | None = None
) -> list[Unwatched]:
    """Cutoffs naming a rig with no recorded run — #282's second loud section.

    `uncut` wearing a disguise, and **strictly worse than `uncut`**, because it
    does not read as a debt: the page says the problem is watched, and nothing
    will ever fire. Two states, told apart because the fix differs -- one is a
    cutoff pointing at a rig that does not exist, the other a real rig nobody
    has run.

    This is the register's job and not the rig report's.
    [#284](https://github.com/NGL321/patchworks/issues/284) declines it in
    terms — *"do not try to close it here… that is why #279's design makes the
    register render cutoffs naming a rig with no recorded run as its own loud
    section"* — so the failure is made visible here rather than automated away
    there.

    A crossing counts as a run: `register:overdue` means the bar was crossed,
    which cannot have happened without the rig running.
    """
    known = available_rigs() if rigs is None else rigs
    found = []
    for problem in survey.problems:
        if problem.cutoff.kind != "measurement":
            continue
        if problem.cutoff.rig not in known:
            found.append(Unwatched(problem, NO_SCRIPT))
        elif problem.cutoff.rig not in problem.reports and not problem.overdue:
            found.append(Unwatched(problem, NO_RUN))
    return found


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

_HEADER = """<!-- Generated by tools/problem_registers.py. Do not edit by hand.

The entry lives on the tracker; this file is a projection of it, so it cannot
disagree with the issue. Edit the field block on the issue or the comment and
regenerate. The checked-in file may be briefly stale and is never the
authority. -->
"""

_CONSULT = """
**Before you implement anything, read all three registers**, and say in your PR
body what you found — `docs/agents/registers.md`, *Before you implement
anything*. "Nothing relevant" is a complete answer and the expected one; silence
is not, because an instruction nobody can tell was skipped decays.
"""


def _cell(text: str) -> str:
    """One table cell: pipes escaped, newlines gone."""
    return text.replace("|", "\\|").replace("\n", " ").strip() or "—"


def render_problems(survey: Registers, rigs: frozenset[str] | None = None) -> str:
    """`open-problems.md`: the debts first, then what is watched, then the resolved."""
    uncut = [p for p in survey.problems if p.cutoff.kind == "uncut" and not p.resolved]
    phantom = [u for u in unwatched(survey, rigs) if not u.problem.resolved]
    out = [_HEADER, "# The open problems register\n"]
    out.append(
        "**An open problem is a stateable failure the architecture is expected to "
        "have, deliberately unresolved, carrying a cutoff.** It is not a ticket: a "
        "ticket is work someone will do, a problem is a known gap nobody is working "
        "on, on purpose, because the bet is that the architecture grows past it.\n"
    )
    out.append(
        "A problem is admitted if it has a **statable failure** — a named way the "
        "architecture is expected to fall short, which someone could recognise "
        "happening. *The disagreement floor might be too high* is not admissible. "
        "**Only a grilling session mints one** "
        "([ADR-0029](../adr/0029-a-problem-is-minted-by-a-human-a-proposal-is-not.md)); "
        "an agent that finds a problem files a `wayfinder:grilling` ticket.\n"
    )
    out.append(_CONSULT)

    out.append("\n## Uncut — nobody has said when this stops being tolerable\n")
    out.append(
        "Admitted, and loud. These sort first and are stated as a debt, in the voice "
        "`@flexibility unknown` uses in the constants registers: *nobody has said "
        "when this stops being tolerable* is a fact, and hiding it is worse than "
        "showing it.\n"
    )
    if uncut:
        # Named here and tabled once below, rather than tabled twice. A second
        # copy of the row is a second place the same fact lives, which is the
        # failure #180 exists to kill -- and it applies to a generated file as
        # much as to a hand-written one, because a reader who sees a row twice
        # has to work out whether the two agree.
        out.append(
            "\n".join(f"* {p.link} — *{_cell(p.failure)}*" for p in uncut) + "\n"
        )
    else:
        out.append("None. Every open problem here carries a cutoff.\n")

    out.append("\n## Cutoffs naming a rig with no recorded run\n")
    out.append(
        "A `measurement` cutoff that reads as watched and is not. This is `uncut` "
        "**wearing a disguise, and strictly worse than `uncut`**, because it does "
        "not read as a debt: the row says the problem is being watched, and nothing "
        "will ever fire. Two states, separated because the fix differs — a cutoff "
        "pointing at a rig that does not exist, and a real rig nobody has run.\n"
    )
    if phantom:
        out.append(
            "\n".join(
                f"* {u.problem.link} — `{u.problem.cutoff.rig}` {u.reason}. "
                f"*{_cell(u.problem.failure)}*"
                for u in phantom
            )
            + "\n"
        )
    else:
        out.append(
            "None. Every `measurement` cutoff names a rig that exists and has "
            "reported.\n"
        )
    out.append(
        "\nA run is recorded by a `@rig` field block on a comment on the problem, "
        "which is what the rig report files "
        "([#284](https://github.com/NGL321/patchworks/issues/284), "
        "`tools/cutoff_report.py`), or by `register:overdue`, since a bar cannot be "
        "crossed without the rig running. The report files **every evaluated run** "
        "and not only a crossing, so a rig that runs regularly and never crosses "
        "leaves this section rather than sitting in it. A row here therefore means "
        "one of two things and not a third: the rig has genuinely not run, or it has "
        "not been given the hook.\n"
    )

    out.append("\n## Open problems\n")
    live = [p for p in survey.problems if not p.resolved]
    if live:
        out.append(_problem_table(live))
    else:
        out.append(
            "None yet. The seeding pass is "
            "[#285](https://github.com/NGL321/patchworks/issues/285), "
            "[#286](https://github.com/NGL321/patchworks/issues/286) and the grilling "
            "session that drains their queue, "
            "[#287](https://github.com/NGL321/patchworks/issues/287).\n"
        )

    out.append("\n## Resolved\n")
    out.append(
        "**A closed problem stays in the register.** The set of problems that "
        "dissolved on their own is the direct evidence for the bet this project is "
        "making — that constraint left out is supplied by the architecture rather "
        "than specified — and deleting the row deletes the evidence.\n"
    )
    resolved = [p for p in survey.problems if p.resolved]
    if resolved:
        out.append(_problem_table(resolved))
    else:
        out.append("None.\n")

    out.append(_PROBLEM_GAPS)
    return "\n".join(out).rstrip() + "\n"


def _problem_table(problems: list[Problem]) -> str:
    rows = ["| problem | failure | cutoff | discovered | issue |", "|---|---|---|---|---|"]
    for p in problems:
        title = _cell(p.title) + (" **(overdue)**" if p.overdue else "")
        rows.append(
            f"| {title} | {_cell(p.failure)} | {_cell(p.cutoff.text)} "
            f"| {_cell(p.discovered)} | {p.link} |"
        )
    return "\n".join(rows) + "\n"


_PROBLEM_GAPS = """
## Stated gaps

One thing this projection cannot reach, named rather than silently omitted.

* **The ground a closed problem closed on.** `docs/agents/registers.md` puts the
  ground — dissolved, solved or withdrawn — in a *closing comment*, as prose,
  with no field block of its own. A projection may not restate prose, so the
  resolved row links to the issue and the ground is one click away rather than
  copied here. Giving the ground a field would change that, and is the design's
  call rather than the generator's.
"""


def render_proposals(survey: Registers) -> str:
    """`proposed-solutions.md`: what is on the shelf, indexed by shape."""
    shelved = survey.shelved
    open_rows = [p for p in shelved if p.status.kind == "open"]
    adopted = [p for p in shelved if p.status.kind == "adopted"]
    out = [_HEADER, "# The proposed solutions register\n"]
    out.append(
        "**A proposal is a solution on the shelf, arguing a shape it would answer, "
        "binding nothing.** A proposal specific to one problem lives as a comment on "
        "that problem's issue; an orphaned or multifarious one gets its own issue. "
        "Both are first-class rows here, and the `where` column says which.\n"
    )
    out.append(
        "A proposal is admitted if it has a **source** and at least one **shape**. "
        "**Search by shape** — the symptom you have — rather than by the name you "
        "were going to give your solution. A research pass may mint a proposal "
        "directly: a proposal binds nothing, and only adoption, dismissal and "
        "problems bind "
        "([ADR-0029](../adr/0029-a-problem-is-minted-by-a-human-a-proposal-is-not.md)).\n"
    )
    out.append(_CONSULT)

    out.append("\n## On the shelf\n")
    out.append(_proposal_table(open_rows) if open_rows else "None yet.\n")

    out.append("\n## Adopted\n")
    out.append(
        "An ADR now covers it. The row stays, because the shelf is also the record "
        "of what was tried and taken.\n"
    )
    out.append(_proposal_table(adopted) if adopted else "None yet.\n")

    out.append("\n## By shape\n")
    out.append(
        "The index. Arriving with a symptom, look for it here — stating a shape "
        "never bound the proposal to a problem, so a shape may be listed against a "
        "proposal that answers nothing yet.\n"
    )
    index: dict[str, list[Proposal]] = {}
    for proposal in shelved:
        for shape in proposal.shapes:
            index.setdefault(shape, []).append(proposal)
    if index:
        out.append(
            "\n".join(
                f"* **{_cell(shape)}** — "
                + ", ".join(f"{_cell(p.title)} ({p.link})" for p in found)
                for shape, found in sorted(index.items())
            )
            + "\n"
        )
    else:
        out.append("None yet.\n")

    out.append(
        "\nDismissed proposals leave the shelf for "
        "[the dismissed-solutions register](dismissed-solutions.md), which is where "
        "*do not re-propose this* is reachable without opening every problem ticket.\n"
    )
    return "\n".join(out).rstrip() + "\n"


def _proposal_table(proposals: list[Proposal]) -> str:
    rows = [
        "| proposal | shape | source | answers | when | status | where |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in proposals:
        rows.append(
            f"| {_cell(p.title)} | {_cell('; '.join(p.shapes))} | {_cell(p.source)} "
            f"| {_cell(', '.join(p.answers))} | {_cell(p.when.text if p.when else '')} "
            f"| {_cell(p.status.text)} | {p.link} |"
        )
    return "\n".join(rows) + "\n"


def render_dismissals(survey: Registers) -> str:
    """`dismissed-solutions.md`: what binds, and why it binds."""
    dismissed = survey.dismissals
    refused = [p for p in dismissed if p.status.dismissal == "refused"]
    failed = [p for p in dismissed if p.status.dismissal == "failed"]
    out = [_HEADER, "# The dismissed solutions register\n"]
    out.append(
        "**A dismissal is a solution excluded, and it binds.** This file unions all "
        "three places a dismissal can live — a closed issue that was never a "
        "proposal, a proposal issue that gained `register:dismissal`, and a "
        "comment-proposal whose field block records it — which is what makes *do not "
        "re-propose this* reachable without opening every problem ticket.\n"
    )
    out.append(_CONSULT)

    out.append("\n## Refused — excluded by what the project is\n")
    out.append(
        "An agent **may not propose these**. Reopening one is an ADR-level act, not "
        "a judgement call inside a ticket.\n"
    )
    out.append(_dismissal_table(refused) if refused else "None yet.\n")

    out.append("\n## Failed — tried, and it did not help\n")
    out.append(
        "Each row **names the rig and the reading that killed it**. That is enforced: "
        "a `failed` entry with no rig and no reading is refused by the generator, "
        "because \"we tried it\" without a reading is unfalsifiable folklore and the "
        "pre-registration discipline exists to keep that out. Proposing one of these "
        "again requires stating what has changed since.\n"
    )
    out.append(_dismissal_table(failed) if failed else "None yet.\n")
    return "\n".join(out).rstrip() + "\n"


def _dismissal_table(proposals: list[Proposal]) -> str:
    rows = [
        "| solution | shape | kind | rig | reading | where |",
        "|---|---|---|---|---|---|",
    ]
    for p in proposals:
        rows.append(
            f"| {_cell(p.title)} | {_cell('; '.join(p.shapes))} | {p.status.dismissal} "
            f"| {_cell(p.status.rig)} | {_cell(p.status.reading)} | {p.link} |"
        )
    return "\n".join(rows) + "\n"


def generate(
    survey: Registers, rigs: frozenset[str] | None = None
) -> dict[pathlib.Path, str]:
    return {
        OUTPUT / "open-problems.md": render_problems(survey, rigs),
        OUTPUT / "proposed-solutions.md": render_proposals(survey),
        OUTPUT / "dismissed-solutions.md": render_dismissals(survey),
    }


# ---------------------------------------------------------------------------
# the network seam
# ---------------------------------------------------------------------------


def fetch(label: str) -> list[dict]:
    """Every issue carrying *label*, open and closed, with its comments.

    **The one function here that touches the network.** Everything below it is a
    pure function of this payload, which is what lets the parser and the renderer
    be tested hermetically while the suite stays on its own side of the line
    `tests/test_cli.py` defends.
    """
    finished = subprocess.run(
        [
            "gh", "issue", "list",
            "--label", label,
            "--state", "all",
            "--limit", str(LIMIT),
            "--json", FIELDS,
        ],
        capture_output=True,
        text=True,
        # Named rather than left to the locale: `gh` returns UTF-8, and on
        # Windows the default is cp1252, which cannot decode an em dash. The
        # register's own prose is full of them, so the generator would die on
        # the first issue that quoted the design it projects.
        encoding="utf-8",
        cwd=ROOT,
    )
    if finished.returncode != 0:
        raise SystemExit(
            f"`gh issue list --label {label}` failed:\n{finished.stderr.strip()}"
        )
    payloads = json.loads(finished.stdout or "[]")
    if len(payloads) >= LIMIT:
        raise SystemExit(
            f"`{label}` returned {LIMIT} issues, which is the page limit: the "
            "register would be truncated and would read as complete. Raise LIMIT "
            "in tools/problem_registers.py, or page."
        )
    return payloads


def survey() -> Registers:
    return collect(
        fetch(PROBLEM_LABEL), fetch(PROPOSAL_LABEL), fetch(DISMISSAL_LABEL)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true",
        help="exit non-zero if the checked-in registers are stale",
    )
    args = parser.parse_args(argv)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stale = []
    for path, text in generate(survey()).items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            continue
        if args.check:
            stale.append(path)
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        names = ", ".join(str(p.relative_to(ROOT)) for p in stale)
        print(
            f"stale: {names}\nrun `python tools/problem_registers.py`", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
