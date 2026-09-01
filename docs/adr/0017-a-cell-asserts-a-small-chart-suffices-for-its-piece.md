# ADR-0017: A cell asserts a small chart suffices for its piece

**Status:** accepted

> **Amended by [#151](https://github.com/NGL321/patchworks/issues/151): the word *lift* goes, and the
> evidence burden stays.** This ADR was titled and written over *“a small linear lift”*. That is an
> assertion of an **immersion**, said out loud — and Corollary 4 of the discrete-time extension
> (arXiv:2605.15161v2) reaches assertions whether or not it reaches code. It also flatly contradicted
> [ADR-0023](./0023-the-chart-is-not-a-koopman-lift.md), written later: *“The design has no `k_lift`,
> because it has no lift.”* Two `accepted` ADRs, opposite claims. What the contract actually asserts
> is **sufficiency of a small persisting chart under a frozen `encode`** — a claim about compression
> and memory. Everything below about the assertion being **asserted and not granted**, and about each
> domain owing the measurement, is untouched: it was always this ADR's point, and it survives the
> repair intact.

## Context

Settled in [#128](https://github.com/NGL321/patchworks/issues/128) and written in the same edit as the
Koopman conversion ([#127](https://github.com/NGL321/patchworks/issues/127), stage 1b).

The domain-agnostic pass asked what the cell contract may know about its world, and hoped for a clean
answer: **the cell knows its own dimensions and nothing else.** That answer held for the interface,
for the algorithm, and for `n` and `k`. Everything world-shaped stayed confined to boundary cells and
to the graph, exactly as intended.

**The conversion then imported a commitment that did not exist when the question was asked.** Making
`K` a linear operator on a `k`-dimensional chart is a claim about the cell's piece, not only about the
machinery. It is the one place where the contract stopped being world-independent, and it arrived
under cover of a decision about tractability.

## Decision

**The cell contract asserts that a small persisting chart suffices for a cell's piece, under a frozen
`encode`. The contract asserts the property; it does not grant it. Each domain owes the evidence.**

The honest form of the claim matters, because a loose form would license nothing and a wrong one
would claim something the design has refused. What is asserted is not that a piece linearises, and
not that it immerses into anything: it is that **`k` dimensions are enough** — holding the cell's
memory and re-fused with a fresh node stalk every tick, through a nonlinearity that is *frozen*
rather than fitted — for the one-step prediction the cell owes. That is a claim about **compression
and memory**, and it is quantitative and domain-sensitive in a way "the sheaf is domain-general" is
not.

**No immersion is asserted, and that is load-bearing.** A cell's chart is a function of the whole
history of stalks that wrote it, not of the instantaneous state, so there is no `F` for a
semiconjugacy to be built from — which is exactly the escape
[ADR-0023](./0023-the-chart-is-not-a-koopman-lift.md) rests on. An earlier form of this ADR asserted
one by accident, in one word.

So the assertion is split from the grant:

- **Asserted by the contract**, of a cell's piece, uniformly: a cell is built as though `k`
  dimensions suffice, and a cell that is wrong about this is wrong in a way nothing else in the
  architecture compensates for.
- **Discharged per domain, by measurement.** A verdict on pucks says nothing about characters. A
  second domain re-runs the measurement; it does not inherit the first domain's answer.

## Consequences

**This is the contract's only claim about the world**, and it is now marked as such rather than
sitting unlabelled beside claims that are genuinely world-independent. The table in
`01-cell-and-sheaf.md` states the split explicitly so that a later reader cannot mistake the
architecture's portability claim for a stronger one than it is.

**It sharpens rather than weakens the portability claim.** What ports is *same frozen maps, same
dimensions, same rules, different operators* — and two cells in one domain already have different
operators, so cross-domain difference is the same kind of object as cross-cell difference, one the
architecture already has and already survives. What does **not** port for free is this assertion.

**The evidence burden is a rerun, not a new experiment.** The instrument already exists: the sweep
that measures how well a `k`-wide chart predicts a cell's piece over a boundary stalk. Pointing it at
a language stream is a rerun of a rig, which is the cheapest possible form this burden could take.

**It interacts with the frozen `encode`, and the interaction is the risk.** `encode` compresses — `k`
under `n` — where every published Koopman lift is *larger* than its state. The design has no lift's
guarantee to fall back on, because it buys no width: it is asking a **frozen compression, carrying
memory** to be enough. The evidence a domain owes is evidence about exactly that.

**Why the word was borrowed, recorded so the repair does not lose the intuition.** *Lift* named the
thing that makes small linear operators sufficient — one hard nonlinear problem decomposed into many
small pieces that compose back into an answer to it. That intuition is sound, and
[ADR-0004](./0004-linear-restriction-maps-assume-local-flatness.md) is its actual justification:
locality and local flatness. What makes *lift* the wrong word is that **a lift is a purchase by
width**, and this design never buys width —
[ADR-0023](./0023-the-chart-is-not-a-koopman-lift.md) already has the sentence: *“Linearity was never
bought with width. It is paid for in nonlinearity-in-the-loop and in time.”*

**And the nonlinearity is not in the decomposition.** Recorded because it is the natural next
mis-statement, and it would be a fresh false claim replacing the old one. Every part of the gluing is
linear by construction — restriction maps (ADR-0004), transport, edge stalks — so were `encode`
linear, the entire 150-cell graph would collapse to one linear system. The decomposition contributes
**zero** nonlinearity. It buys local flatness and a small `k`; the nonlinearity is at exactly one
point per cell, re-entered every tick, and that is `encode`.

**It composes with the readout gauge's cost.** [ADR-0014](./0014-the-linear-readout-is-gauge-fixed.md)
confines a cell's predictions to a fixed subspace of its stalk; this ADR asserts the chart is rich
enough for the dynamics. The two failures are distinct and their signatures differ — one is a shared
direction across unrelated edges, the other is prediction error no `K` can remove — which is why
`CONTEXT.md` keeps *readout gauge* and *chart linearity* as separate claims.

## Alternatives considered

**Grant it per domain in the contract** — i.e. let the contract assert linearisability outright.
Rejected: that is the trivially true form, and it would let a domain be adopted with no measurement at
all, which is precisely the failure this ADR exists to prevent.

**Leave it unstated, since the conversion is taken on tractability grounds.** Rejected. The grounds
for taking a decision and the commitments the decision creates are different things, and a commitment
about the world that nobody wrote down is the kind that gets discovered by a failed run.

**Fold it into the linearity-claims vocabulary and add no ADR.** Rejected, narrowly. The vocabulary
entry (*chart linearity*) records what the claim *is*; this records that it is **not granted** and
carries an evidence burden that crosses domains. The second half is a decision, with an alternative
that was really available, and it is what a future domain will need to find.
