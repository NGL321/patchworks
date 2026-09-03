# The registers

`docs/registers/` holds six generated files. Three project the provenance of every constant the
architecture rests on, read off the definition sites by `tools/constant_registers.py`. Three project
the life of the architecture's open questions — what is known to be wrong, what might fix it, and
what has been excluded — read off GitHub by `tools/problem_registers.py`.

**None of the six is editable by hand.** A register is a projection of its source and cannot disagree
with it. For constants the source is the `#:` block above the definition; for problems it is the
issue. Editing the rendered file puts a decision in two places, which is the failure `#180` exists to
kill.

## Before you implement anything, read the registers

This applies to every agent that writes code in this repo, and to every agent that explores a space
of solutions before writing any.

1. **Read `docs/registers/open-problems.md`.** The thing you are about to build may sit next to a
   known failure, and the failure will not be visible in the code — that is what makes it a problem
   rather than a bug.
2. **Read `docs/registers/proposed-solutions.md`.** Someone may already have argued the mechanism you
   are about to invent. Proposals are indexed by `@shape` — the symptom they answer — so search by
   the symptom you have, not by the name you were going to give your solution.
3. **Read `docs/registers/dismissed-solutions.md`.** A `refused` entry is binding: you may not
   propose it, and reopening it is an ADR-level act. A `failed` entry names the rig and the reading
   that killed it; proposing it again requires stating what has changed since.

**State what you found.** Every PR body, and every issue comment reporting implementation work,
carries a line:

> **Registers consulted:** open-problems (nothing relevant), proposed-solutions (#N, `@shape`
> matched, not taken because …), dismissed-solutions (nothing relevant).

"Nothing relevant" is a complete answer and the expected one. What is not acceptable is silence: an
instruction nobody can tell was skipped decays, and this line is the only place the consultation
becomes visible. It is also where a *new* problem gets caught, because the moment you notice the
register is missing something is the moment you were looking at it.

## The three problem registers

### Open problems

**An open problem is a stateable failure the architecture is expected to have, deliberately
unresolved, carrying a cutoff.** It is not a ticket. Most tickets are not problems: a ticket is work
someone will do, a problem is a known gap nobody is working on, on purpose, because the bet is that
the architecture grows past it.

The admission rule is the constants register's rule in a different domain. There, *a constant is
admitted if it has a warrant: a statable reason it is this value rather than another.* Here, **a
problem is admitted if it has a statable failure: a named way the architecture is expected to fall
short, which someone could recognise happening.** "The disagreement floor might be too high" is not
admissible. "If the standing offset does not fall below X, rim-to-core detectability fails" is. This
is the spec's pre-registration discipline pointed at the map instead of at a run.

**Only a grilling session mints a problem.** Not an implementation agent, not a research pass, not a
triage sweep. A stateable failure and a cutoff are exactly the two things a solo agent will fabricate
plausibly and wrongly, and a register full of plausible problems is worse than no register — see
[ADR-0029](../adr/0029-a-problem-is-minted-by-a-human-a-proposal-is-not.md).

An agent that finds a problem **files a `wayfinder:grilling` ticket** describing what it saw, and
never a `register:problem`. The grilling session either admits it — minting the problem issue,
stating the failure and the cutoff, closing the ticket with a pointer — or declines it. The cost is
real and accepted: there is a queue between finding a problem and its being in the register, and only
a grilling session drains it.

### Proposed solutions

**A proposal is a solution on the shelf, arguing a shape it would answer, binding nothing.**

Where it lives depends on how many problems it touches:

- **Specific to one problem** — a comment on that problem's issue. This is the common case, and it is
  what makes the problem ticket a single place to read: open the problem, see every solution anyone
  has offered for it.
- **Orphaned, or applying to several problems** — its own issue, labelled `register:proposal`. Relay
  cells are the type case: no problem is attached, and two unrelated shapes are — transmission over a
  graph much deeper than the dome, and a carrier for attention if attention ever leaves the fog
  (`docs/spec/04-action-and-the-boundary.md`).

Both forms are first-class rows in the register. The generator scans `register:proposal` issues and
the comments on `register:problem` issues alike.

**A research pass may mint proposals directly**, without a grilling session. A proposal binds
nothing; only adoption, dismissal and problems bind. This is deliberate and it is what
[ADR-0029](../adr/0029-a-problem-is-minted-by-a-human-a-proposal-is-not.md) records.

A proposal is admitted if it has a **source** and at least one **shape**.

- **Source** — a citation, a rig reading, a research doc, a session, or the literal `here`, which
  means *the argument is in this issue body*. `here` is admissible and common: a grilling session
  that reasons out a mechanism has produced the argument even when no paper exists. The toll is that
  the body must actually argue it. A body that names an idea without arguing it is refused.
- **Shape** — the symptom an agent would recognise, stated so that arriving with a problem finds the
  proposal. Shapes are a **list**, and stating one never binds the proposal to a problem; binding is
  what `@answers` does, and an orphan has none.

### Dismissed solutions

**A dismissal is a solution excluded, and it binds.** Two kinds, and they behave differently:

- **`refused`** — excluded by what the project is. An agent may not propose it. Reopening is an
  ADR-level act.
- **`failed`** — tried, and it did not help. The entry **must** name the rig and the reading. A
  `failed` row with no rig and no reading is inadmissible, because "we tried it" without a reading is
  unfalsifiable folklore, and the whole pre-registration discipline exists to keep that out.

A dismissal that was never a proposal is a closed issue carrying `register:dismissal` alone. A
dismissed proposal keeps its `register:proposal` label and gains `register:dismissal`; a dismissed
comment-proposal records it in its own field block. The rendered register unions all three, which is
what makes "do not re-propose this" reachable without opening every problem ticket.

## Labels

Three, namespaced like `wayfinder:*`. **Extraction reads labels and field blocks, never prose.**

| label | on | meaning |
|---|---|---|
| `register:problem` | an issue | this issue *is* an open problem. Open means unresolved; closed means resolved, with a ground. |
| `register:proposal` | an issue | this issue *is* an orphaned or multifarious proposal. |
| `register:dismissal` | a closed issue | terminal state. Co-occurs with `register:proposal` when a proposal was dismissed. |

Note what these labels do **not** do: they do not track a problem's state through the issue it was
discovered in. A problem gets its **own** issue precisely because a live problem is routinely found
inside a ticket that then closes. The originating issue receives a one-line pointer comment —
`Problem extracted to #N: <title>` — and nothing more. One statement, one place.

## Field blocks

Structured fields sit in a fenced block at the top of the issue body, or at the top of a proposal
comment. Everything below the block is prose the generator never reads: the argument lives at the
definition site and the row carries a link, exactly as the constants registers work.

**On a `register:problem` issue:**

```
@failure    <the named way the architecture is expected to fall short>
@cutoff     event <issue> | measurement <rig> <threshold> | uncut
@discovered <issue, session, or rig this came out of>
```

**On a `register:proposal` issue, or a proposal comment:**

```
@proposal <title>
@source   <citation | rig reading | research doc | session | here>
@shape    <symptom it answers>          (repeatable)
@answers  <problem issue numbers>       (absent on an orphan)
@when     event <issue> | measurement <rig> <threshold>   (optional)
@status   open | adopted <ADR> | dismissed refused | dismissed failed <rig> <reading>
```

`@status` is edited in place as it changes; the comment thread around it is the history.

`@when` is the mirror of a cutoff, with the opposite polarity and no obligation attached: a cutoff
says *this problem stops being tolerable*, `@when` says *this proposal starts being relevant*.
`@when event <issue>` on the relay cells' attention shape means that the day attention leaves the
fog, the proposal surfaces on its own rather than being remembered.

## Cutoffs

**A cutoff is the point at which a problem stops being tolerable** — the thing that says *this is not
resolving through emergent properties of the system as it stands, address it now*. Two admissible
forms, and dates and judgement are refused: a cutoff must be checkable by someone who is not its
author, and "when it becomes a problem" is not a cutoff but the absence of one.

- **`event <issue>`** — the named issue closes. Automated already: the `constant-provenance.yml`
  pattern watches `issues: closed` and reports what was resting on it.
- **`measurement <rig> <threshold>`** — a named `benchmarks/` rig crosses a stated bar. **Rigs do not
  run in CI and must not learn to**; `benchmarks/run_reporting.py` states the rule — every script
  there asserts nothing, because a measurement belongs on a machine rather than in a suite. So a
  measurement cutoff is evaluated **by the rig report**: when a rig runs, its report states, for each
  problem cutting on it, whether the bar was crossed. Crossing files a comment on the problem issue
  and adds `register:overdue`.
- **`uncut`** — admitted, and loud. The register sorts these first and states them as a debt, in the
  voice `@flexibility unknown` uses: *nobody has said when this stops being tolerable* is a fact, and
  hiding it is worse than showing it.

The register carries a second loud section: **cutoffs naming a rig with no recorded run.** That is
the state where a problem looks cut but nothing will ever fire — `uncut` wearing a disguise, and
strictly worse than `uncut`, because it does not read as a debt.

## Closing a problem

A problem closes on a **stated ground**, in a closing comment, of exactly one kind:

- **dissolved** — a rig reading shows the failure no longer occurs. Quote the reading.
- **solved** — an ADR now covers it. Link it.
- **withdrawn** — a grilling session judged the failure misstated. Say what was wrong about it.

An agent may close on **solved**. Only a grilling session may close on **dissolved** or **withdrawn**.

**A closed problem stays in the register**, in a resolved section, carrying its ground. The set of
problems that dissolved on their own is the direct evidence for the bet this project is making — that
constraint can be left out and the architecture will supply what a more constrained design would have
had to specify. Deleting the row deletes the evidence.

## Generation

`tools/problem_registers.py` renders `open-problems.md`, `proposed-solutions.md` and
`dismissed-solutions.md` from `gh`. It is a **network** tool and lives on the far side of the line
`tests/test_cli.py` defends: the suite never reaches it, and CI never checks these three files for
staleness, because CI cannot ask GitHub anything offline.

Freshness is a workflow's job, mirroring `.github/workflows/constant-provenance.yml`: triggers on
`issues: [opened, closed, edited, labeled, unlabeled]` and `issue_comment: [created, edited]`, a
weekly net, and `workflow_dispatch`; it regenerates and commits when the render changes. Its
concurrency group sets `cancel-in-progress: true` — unlike the provenance workflow's, because a
superseded run of a pure projection is worth nothing and comment edits fire often.

`--check` still exists, for a human at a terminal.

The checked-in files can therefore be briefly stale, and are never the authority. That is the point:
GitHub is the definition site, and the rendered file exists so an agent can read the register without
a network call or a token.
