---
status: accepted
---

# Action is prediction the world clears

Active predictive coding is usually built as two things — a world-modelling upstream and an
action-selection downstream, coupled at defined points. Patchworks builds **one graph with one cell
contract** instead, and locates the opposition in the edge rather than in the cell: on a sensory
edge, disagreement is cleared by the cell changing its belief; on a motor edge, it is cleared by the
world moving, because the far endpoint executes rather than restricting a belief back. "One
algorithm running in opposite directions" is therefore literal, and the modification the action side
needs is zero.

The alternatives both bought their separation by spending uniformity, which is the architecture's
main asset: **two paired populations per site** (state cell and action cell) requires two variants of
the local learning rule, and **two coupled graphs** requires a second sheaf plus a coupling interface
that exists nowhere else in the design. Neither buys anything the edge asymmetry does not already
give.

## Consequences

- No cell anywhere is marked as an action cell, and no region of the graph is structurally the
  action side. A cell's role is emergent from which boundary cells it happens to be near.
- Action emission needs no mechanism: there is no output cell, no consensus rule, no read-out map.
  Where several cells drive the same actuator boundary cell, reconciliation on those edges *is* the
  arbitration.
- Planning cannot be a separate faculty. With no second mode of operation, a goal is realised by a
  standing assertion propagating to the boundary — which means the architecture has **no
  counterfactual evaluation at all**. See
  [route selection](https://github.com/NGL321/patchworks/issues/25).
- The generative/action-selection opposition of the literature survives as a *path* through the
  graph (sensory rim → centre → motor rim, abstraction rising then falling), not as two objects.

*Amended by [ADR-0009](./0009-a-drive-is-a-motor-edge-attached-deep.md).* The sensory/motor split
above is exhaustive over the *world's* edges but not over the graph's: a **drive edge** is written from
outside and cleared by the world moving, which makes it a motor edge attached far from the rim rather
than a third kind. The taxonomy holds; it gains a row. The consequence above originally read "a
*clamped* prediction", which named a mechanism ADR-0009 rejects — the word is retired, the claim is
unchanged.

## The bet, stated plainly

This is the active-inference identity that action is prediction the body fulfils. It is strong for
reflexive control and historically contested for anything requiring lookahead. It is adopted **before
reading**, per the map's deliberate citation sequencing, and it is the single most literature-fragile
commitment in the spec. The citation pass for this area checks it as its headline question, and a
revision sweep through [`docs/spec/04-action-and-the-boundary.md`](../spec/04-action-and-the-boundary.md)
and everything downstream is accepted in advance rather than resisted.
