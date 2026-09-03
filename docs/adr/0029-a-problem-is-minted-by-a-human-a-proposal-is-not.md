# ADR-0029: A problem is minted by a human; a proposal is not

**Status:** accepted

## Context

The architecture's open questions were distributed across the ticket queue, conversations, and
readings taken during experiments, with no place that held them together and no point at which any of
them became due. That distribution is not accidental: many of them are deliberately unresolved,
because the bet the project makes is that constraint left out is supplied by the architecture rather
than specified — over-constraining is the road to a symbolic design, and this one declines to take
it. What the bet needs is not fewer open questions but a **cutoff** on each: the point at which *not
resolving emergently* becomes *resolve this now*.

Three registers follow — open problems, proposed solutions, dismissed solutions — projected from
GitHub the way `docs/registers/`'s existing three are projected from the definition sites of
constants. `docs/agents/registers.md` holds the mechanism.

The question this ADR records is **who may write to them**, and the answer is not uniform across the
three. Two candidate rules were available and both are defensible:

- **Uniform human gating.** Every register entry passes through a grilling session. Consistent, and
  the vocabulary discipline `CONTEXT.md` maintains argues for it.
- **Uniform agent access.** Anything an agent finds, it files. Nothing is lost and nothing waits.

## Decision

**Neither. Entries that bind require a human; entries that merely offer do not.**

- **A problem is minted only in a grilling session.** An implementation agent or research pass that
  finds one files a `wayfinder:grilling` ticket and stops there.
- **A proposal may be minted directly by a research pass**, with no grilling session.
- **Adoption and dismissal require a human**: adoption is an ADR, dismissal binds every later agent.

The line is what an entry *obliges*. A problem carries a cutoff, and a cutoff is a promise to
interrupt the work later — it is load-bearing in the same way a pre-registered falsification is, and
a stateable failure is precisely the thing a solo agent will fabricate plausibly and wrongly. A
register full of plausible problems is worse than no register, because every one of them will
eventually demand a rig. A proposal on the shelf obliges nothing: its cost is one row an agent skips.

The asymmetry pays for itself on the proposal side specifically. Proposals will routinely be drawn
from domains the maintainer does not hold — learning a domain in advance in order to be allowed to
mint a proposal from it is a bottleneck priced far above the entry's worth, and most proposals are
testable without their originating mathematics being understood first.

## Consequences

**There is a queue between finding a problem and its being in the register, and only a grilling
session drains it.** This is the accepted cost. The alternative — letting agents mint problems — buys
promptness with a register nobody trusts, and a register nobody trusts is not consulted, which
defeats the entire purpose of pointing implementation agents at it.

**The proposal register will contain entries no human has vetted.** Guarded by admission rather than
by gating: a proposal needs a `@source` and at least one `@shape`, and an entry that argues nothing
is refused at generation time. `@source here` is admissible — a session that reasons out a mechanism
has produced the argument even where no citation exists — but the body must carry the argument, not
merely the name.

**A future reader will find this asymmetry arbitrary, and it is not.** That is the reason this ADR
exists rather than a paragraph in `docs/agents/registers.md`: the natural instinct on encountering
the rule is to make it uniform in one direction or the other, and both uniform rules were considered
and refused above.

**Nothing here licenses an agent to close a problem on judgement.** An agent may close on *solved*
— an ADR now covers it — because that ground is checkable by reading the ADR. *Dissolved* and
*withdrawn* are grilling-session grounds, for the same reason minting is.
