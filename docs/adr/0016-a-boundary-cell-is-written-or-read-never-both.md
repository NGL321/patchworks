# ADR-0016: A boundary cell is written or read, never both

**Status:** accepted

## Context

Settled in [#128](https://github.com/NGL321/patchworks/issues/128), the domain-agnostic pass, and
written in the same edit as the Koopman conversion
([#127](https://github.com/NGL321/patchworks/issues/127), stage 1b) — because that conversion already
opens the cell contract, and it is the cheapest moment in the project's life to remove the dome's
assumptions from it.

The pass asked what in the cell contract still assumes a dome and a sandbox. Four suspected
assumptions were **clean**: the 50 Hz tick, the `n`/`k` justification, the sensorimotor rim as the
origin of the abstraction measure, and `04`'s three boundary-cell kinds. The real leaks were found by
asking what the *clean* answers presuppose, and they were all at the rim.

This is one of them, and it did not look like an assumption at all. `04` takes for granted that the
sensory and motor rims are **distinct**, and treats the shortest sensorimotor loop as a design budget.
Nothing said so, because on a dome it could not have been otherwise: eyes and an arm are different
organs. A domain that carries both directions on **one alphabet** — characters heard and characters
spoken — invites the collapse, and a graph designer who merged them would be doing something the spec
never forbade.

## Decision

**A boundary cell is written by the outside or read by it, never both. This is a contract-level ban.**

It belongs in the cell contract rather than being deferred to graph design, because it is not a
preference about topology — **the tick's ordering has no defined meaning otherwise.** The external
write is a tick's *last word* (`02-tick-semantics.md`): whatever is outside the sheaf writes its
boundary cells after the message-passing phase. A cell that is also **read** by the world would
therefore be read *before* the word it was going to say. There is no ordering that fixes this, because
the ordering is doing other load-bearing work — it is what makes ADR-0009's *the assertion stands
forever* true, and what lets a commanded component reach the arm without a per-component exemption.

## Consequences

**A language rim has separate boundary cells for heard and spoken characters.** The loop through them
is **longer** than the sandbox's, not degenerate — which is the right direction: it keeps `04`'s
sensorimotor loop budget exactly what it says it is, rather than quietly halving it.

**No existing graph changes.** The dome already satisfies the ban, which is why it was invisible; the
ban costs nothing today and forbids a specific future mistake.

**It composes with the readback requirement** (`04-action-and-the-boundary.md`, *Readback*): a motor
boundary cell owes a readback of what the world made of its command, and the ban does not touch it —
the readback is written back onto that same cell, as the dome's actuator stalk of 6 carries three
commanded components and three efference ones. What the ban keeps off that cell is the **sensory**
stream: on a language rim, the characters the interlocutor *speaks* land on a heard cell of their own,
never on the cell it reads. Those two facts together are what make turn-taking legible as the body's
refusal.

> **Wording corrected** while writing [#170](https://github.com/NGL321/patchworks/issues/170). This
> paragraph previously said the readback "arrives on a *different* cell rather than being written back
> onto the same stalk", which read literally forbids the dome's own actuator cell and the language
> domain's spoken rim ([`12-the-interlocutor.md`](../spec/12-the-interlocutor.md), *The spoken cell is
> read for its command and written for its readback*). The decision above is unchanged; only this
> consequence was stated too widely.

## Alternatives considered

**Leave it to graph design.** Rejected. The failure it prevents is not a bad graph but an
**undefined tick**, and a rule whose violation makes the tick semantics meaningless belongs with the
tick semantics rather than with whoever draws a particular graph.

**Allow a shared cell and define an ordering for it.** Rejected: any such ordering is a second tick
semantics for one class of cell, and the existing one is relied on by ADR-0009 and by the motor
pathway. A special case there is far more expensive than a ban here.

**Say nothing, since the dome does not violate it.** Rejected on the pass's own terms. The point of a
domain-agnostic pass is to find the assumptions that are true of the dome and unstated, precisely
before a second domain makes them false.
