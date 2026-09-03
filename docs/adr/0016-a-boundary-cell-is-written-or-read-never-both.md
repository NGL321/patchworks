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

**A rim is not the unit this ban applies to.** Amended while writing
[#169](https://github.com/NGL321/patchworks/issues/169), which ruled on the first rim that has an
**interior**: the language wedge's spoken buffer, a 128-slot shift register whose head is read by the
interlocutor and whose tail is written by nothing but the shift
([`11-the-language-graph.md`](../spec/11-the-language-graph.md), *A rim's kind is the slot's, not the
rim's*). A boundary cell's kind is read off **that cell's own edge to the world**, so a rim whose
slots differ in what the world does with them is a rim whose slots differ in kind — and that is not an
exception to anything here. The ban above is per boundary cell, and such a rim satisfies it trivially:
a motor head beside a sensory tail puts no cell on both streams. It is
[ADR-0003](./0003-action-is-prediction-the-world-clears.md)'s *"a cell's role is emergent from which
boundary cells it happens to be near"* applied where it had never been tested. The spoken rim is
therefore **one motor boundary cell plus 127 sensory ones**, and the group reading `11` had declared
is retired there.

**No new ADR was minted for it, deliberately.** A fresh ADR whose Decision restated ADR-0003's own
consequence would put one statement in two places — the failure
[#180](https://github.com/NGL321/patchworks/issues/180) exists to kill — and declaring *how rim
contracts are indexed* would dignify **the rim** as a contract-bearing object, which is exactly what
the group reading did and what the ruling refuses. The amendment lands here because this ADR's subject
is the same one seen a step out.

**The dome is untouched by the amendment, and nobody need re-derive that.** Its actuator cell is read
and written by the world; its sensory rim is written and read by no one. The ruling costs the built
graph nothing and forbids a specific future mistake — the same shape the ban above already had.

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
