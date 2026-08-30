# ADR-0017: A cell asserts its piece admits a small linear lift

**Status:** accepted

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

**The cell contract asserts that a cell's piece admits a small linear lift. The contract asserts the
property; it does not grant it. Each domain owes the evidence.**

The honest form of the claim matters, because the loose form is trivially true and would license
nothing. Koopman theory gives *any* nonlinear system a linear representation in *some* observable
space, so **linearisability is not the assumption** — it is free. What this design needs is
**sufficiency of a small finite lift**: that `k` dimensions are enough, with a *frozen* `encode`
producing them. That is quantitative and domain-sensitive in a way "the sheaf is domain-general" is
not.

So the assertion is split from the grant:

- **Asserted by the contract**, of a cell's piece, uniformly: a cell is built as though its piece
  admits such a lift, and a cell that is wrong about this is wrong in a way nothing else in the
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
that measures how well a cell's piece linearises over a boundary stalk. Pointing it at a language
stream is a rerun of a rig, which is the cheapest possible form this burden could take.

**It interacts with the frozen `encode`, and the interaction is the risk.** `encode` compresses — `k`
under `n` — where every published Koopman lift is *larger* than its state. The design is therefore
asking a compression to do a lift's job, and it is doing so with the map frozen. The evidence a domain
owes is evidence about exactly that.

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
