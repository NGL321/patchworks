# ADR-0019: Construction nominates, the run decides

**Status:** accepted

## Context

Settled in [#160](https://github.com/NGL321/patchworks/issues/160).

`02-tick-semantics.md` has carried a bound since
[ADR-0007](./0007-the-disagreement-floor-is-tolerated-not-represented.md): the displacement
reconciliation leaves on a cell's node stalk must stay below that cell's **fold margin**, its distance
to the nearest activation boundary in `encode`. Cross that boundary and the cell lands in a region
with a different regional spectrum — *timescale separation erased by its own reconciliation*, which is
the failure ADR-0007 exists to forbid.

The bound has been checked at construction, once, on the sampling rig.
[#33](https://github.com/NGL321/patchworks/issues/33) found that it could not stay there — the
transport rule trains the magnitudes the gain's denominator stood in for, so the denominator drifts —
and [#37](https://github.com/NGL321/patchworks/issues/37) put it back, because
[ADR-0010](./0010-restriction-map-scale-is-gauge-fixed.md) fixes those magnitudes and removes the
drift at its source. That exchange is entirely about the **denominator**.

#160 went looking for a third round of the same argument and found something else.

### This ADR exists so the record does not read as self-contradicting

A reader arriving cold finds #37 **striking** a periodic runtime re-derivation, and then finds a
runtime read **reinstated** here, and concludes the project changed its mind without saying so. It did
not, and the reason is one sentence: **#37 was right about the gain's denominator and silent about the
arrangement**, and those are opposite sides of the same inequality. The denominator is a property of
the graph and ADR-0010 froze it. The arrangement is a property of the body's biases and nothing has
ever held it still. Three amendments to three documents would leave that sentence written nowhere.

The second thing this ADR carries has no other home either: **why a projection cannot reach the fold
margin.** That question is asked every time a quantity in this system turns out to move — *why isn't
this banded the way `K` is* — and it will be asked again.

### The bound, geometrically

There is one space, not three: **`encode`'s input space**, `R^44 = chart (R^12) ⊕ node stalk (R^32)`,
which `body.py` forms by concatenation before one hidden layer of 45 ReLU units. The arrangement is
**45 hyperplanes in `R^44`**; an activation region is one convex cell of it. The margin is
`min_i |z_i| / ‖∇z_i‖` with `z_i = w_i·[chart; stalk] + b_i`.

The restriction maps are **not in this space** and define no metric on it. They set the *size* of the
displacement; the arrangement sets how much displacement is affordable. Two objects, one inequality —
and both are lengths in the same space, so [#181](https://github.com/NGL321/patchworks/issues/181)'s
ill-formed comparison does not recur here and no amplitude convention is owed.

Reconciliation displaces the point in the **32 stalk coordinates only**. Distance along a coordinate
subspace is never less than the perpendicular distance, so the comparison is **conservative** in that
respect rather than optimistic.

### Three motions, and the record argues about one

1. **The point moves under reconciliation.** The spec assumed the displacement's size is `γ ×` an
   irreducible constant, hence knowable once.
   [#158](https://github.com/NGL321/patchworks/issues/158) measured it: it is dominated by model
   error and falls **144x** in a run. It is a transient of learning, not a constant of the system.
2. **The folds move, because the bias is trained.** The shared frozen weights fix the *directions* of
   every folding hyperplane; the per-cell biases fix *where each fold sits* — and the biases are the
   body's whole trainable surface, trained by the prediction rule. One frozen set of orientations,
   rigidly translated per cell, sliding under the point for the length of the run. **#33 and #37's
   exchange never reaches this.**
3. **The step size changed.** Under [#190](https://github.com/NGL321/patchworks/issues/190) `gain_v`
   is uniform across the interior.

Two further facts make this operational rather than tidy. ADR-0007's specified read protocol — the
quiescent hold — **yields nothing at the moment it is specified to run**: #158 held an untrained graph
400 ticks and got numbers identical to five decimals. And `tick.py` ships `DEFAULT_GAMMA = 1.0`
against [#178](https://github.com/NGL321/patchworks/issues/178)'s 0.90x permitted at 2,000 ticks, so
**the build spends its early life outside its own bound and nothing looks.**

**And it never stops spending it there.** This ADR first stated the bound as holding after a burn-in
and called that count its only new quantity. [#202](https://github.com/NGL321/patchworks/issues/202)
ran the instrument decision 4 builds and found the count does not exist: over 100,000 ticks, not one
tick was free of a breaching cell, and the breach density plateaus at ~16 cells in 150 rather than
decaying. Decision 5 is written below in the light of that read rather than ahead of it — see
[#206](https://github.com/NGL321/patchworks/issues/206).

## Decision

### 1 — The bound governs the standing offset, not a floor

`02`'s `γ × floor < margin_v` names the wrong divisor. What the fold cares about is the
**displacement, whatever caused it**; a fold does not care why the operating point moved. ADR-0007's
own derivation makes the offset the mechanism of concern and the disagreement floor merely its
*assumed source*. The floor is one contributor and, at construction, not the dominant one.

The term is ADR-0007's own — *a bounded standing offset on the reconciled component of a node stalk* —
so this renames nothing that was not already written down. `CONTEXT.md` carries **Standing offset** as
an entry, and *Disagreement floor* gains a line saying it is not this.

### 2 — The margin is read live, and nothing bands the bias

The read is nearly free. `encode`'s hidden pre-activations are computed by the forward pass anyway,
and `‖∇z_i‖` are the **shared frozen weight rows** — one graph-wide constant, computed once. **No new
state and no new time constant**, which is exactly what sank every earlier re-derivation proposal.

**Banding the bias was weighed against ADR-0010 and
[ADR-0015](./0015-the-cell-operator-band-is-on-the-spectral-norm.md) and fails on shape.** This is the
reusable half of this ADR:

> Those two band pure **parameter** norms — `‖F‖_F ≤ ρ`, `σ_max(K) ∈ [1/ρ_K, 1]` — and a projection
> can reach a parameter norm, because the quantity is a function of the parameter alone. **The fold
> margin is not that shape.** It is a property of the **(bias, operating point) pair**:
> `z_i = w_i·[chart; stalk] + b_i` moves when `b_i` moves *and* when the input moves. Holding a floor
> on it would mean shoving `b_i` against the current input every tick — a bias that tracks its own
> evidence, which `02-tick-semantics.md` rejects because it needs its own time constant, and which is
> the per-edge tracking baseline ADR-0007 already declined, relocated onto the body.

So the mechanism that fixed the denominator is unavailable on this side of the inequality, and that is
structural rather than an oversight.

Conservatism — banding at some safe multiple and never looking — was declined on #178's evidence:
these quantities wander with no trend, so there is no worst case to be conservative against.

### 3 — The depth claim is struck, and not replaced

`02`, ADR-0007's restatement of it, and `FoldMarginCheck`'s docstring all assert the bound binds
hardest at the apex because `Σ_e m_e` falls with depth. Under #190's uniform interior gain it binds
**equally everywhere inside**, on each cell's own margin draw — which `bias_selection.py` already
calls "partly a draw".

**No replacement claim is offered.** #158's offset profile down levels 1–7 is not monotone, and #178
showed the 30k reading was a local high of a quantity that wanders 3.8x. The 12x boundary gain does
not rescue it either: **boundary cells run no body, so they have no fold margin and this check never
applied to them.**

### 4 — Construction nominates, the run decides

[#156](https://github.com/NGL321/patchworks/issues/156)'s discipline, reused rather than reinvented.

The construction check is **demoted from gate to nomination**: it reports the cap a body's draw
permits, before anything runs, and nothing turns on it.

**Live dwell is the verdict.** ADR-0005's mechanism holds only where a cell's region dwell is long
against the `τ` that region implies, and `CONTEXT.md`'s *Region dwell* already says dwell is "measured
at runtime on a driven trajectory". That measurement is now taken.

**The live margin-against-offset comparison is the attribution.** Dwell alone cannot say *whether
reconciliation* moved the cell, which is the thing ADR-0007 forbids — a cell may leave its region
under its own dynamics, and that is not this bound's business. The two readings are one instrument,
`patchworks.tick.FoldRead`, and neither substitutes for the other.

ADR-0005's precondition **re-sources onto measured dwell** rather than onto the margin proxy.
[#41](https://github.com/NGL321/patchworks/issues/41) is honoured, not reopened: the margin still
bounds dwell, only the moment of reading moves.

### 5 — `γ` stays 1.0, and the breach is standing rather than transient

*Rewritten before this ADR merged, by [#206](https://github.com/NGL321/patchworks/issues/206), on the
measurement decision 4 asked for. What stood here — **the bound holds after a burn-in** — was false,
and the route from it to this is [#202](https://github.com/NGL321/patchworks/issues/202)'s read.*

**There is no burn-in.** #202 ran `FoldRead` for 100,000 ticks on the real dome, 150 predicting cells.
Every one of those ticks carried at least one breaching cell; all 150 cells breached at least once;
109 were still breaching after tick 90,000. The density falls 33 → 16 cells and then **plateaus** —
p05 11, median 16, p95 22 across the run's second half. The transient is not a transient, and the
count this decision introduced as its only new quantity has no value to take.

**The clause was verdict language for a reading decision 4 had already demoted.** Decision 4 puts the
verdict on measured region dwell and makes the live margin-against-offset comparison the
**attribution**. An attribution has nothing to pass or fail — it says *why* a cell lost its region,
not *whether* the build is healthy. *The bound holds after a burn-in* handed a threshold back to the
reading that had just stopped being the verdict, and the burn-in was the leftover.

**So the comparison carries no threshold at all.** Not a burn-in, and not a tolerance either: *no more
than `k` cells breach after tick `T`* would read `k` off one run's plateau, and that is a constant
with no derivation behind it — the ground on which decision 2 already declined conservatism.
`reconciliation_reaches` is **reported, never asserted**.

**The two readings do not disagree, and the record must stop implying they can.** *The bound is
breached* and *the mechanism the bound protects is working* were both true in #202's run: not one
clean tick in 100,000, and 130 of 150 cells clearing ADR-0005's dwell precondition at the horizon.
They are readings of **two different quantities**, separated one decision earlier — and where the pass
condition lives is decision 4's answer, measured dwell against `τ`.

**That quantity has the shape this decision reached for a burn-in to describe.** dwell/`τ` is 0.96 at
tick 100, passes 2 somewhere past 2,000 ticks, and reaches 82.7 by 100,000: it **fails early and is
earned over the run**. The original argument was sound — during the transient the offset is
model-error dominated, and **a cell whose region flips at tick 2,000 has no slow content to protect**,
its `H⁰` holding nothing worth keeping — and it was attached to the wrong reading. It is kept here as
the reason the early breaches cost nothing, with no count resting on it.

**`γ` stays at 1.0, and the two alternatives are not simply re-declined.** Both were declined on the
premise that the breach was confined to a window where nothing was at stake, and #202 killed that
premise: a cell breaching at tick 95,000 has had 95,000 ticks to accumulate slow content.

- **Ramping `γ`** is declined permanently, and on stronger ground than
  [`08-the-acceptance-demo.md`](../spec/08-the-acceptance-demo.md)'s pre-registration discipline: the
  density plateaus, so a ramp is aimed at a transient the run says does not exist, and would end with
  the breach still standing.
- **A permanently lower `γ`** loses its decline. It is no longer paying reconciliation speed forever
  for safety in a stakeless window; it is a candidate mechanism against a standing failure, which is
  [#205](https://github.com/NGL321/patchworks/issues/205)'s question rather than this ADR's.
  **Withdrawn as a decline, not renewed** — nothing here rules it out, and nothing here adopts it.

`DEFAULT_GAMMA` accordingly leaves the constant registers' `provisional #85` — the debt was waiting on
a construction-time cap that this decision says cannot exist — and is `stipulated` on this ADR. Its
recorded reason changes with this rewrite: the ceiling is held because **nothing derives a lower
value**, not because both alternatives were declined.

### 6 — A new ADR, not the falsification register

[#147](https://github.com/NGL321/patchworks/issues/147) carries the costs *of the Koopman conversion*,
and the conversion did not cause this. It **loosened** the check: the margin was a min over `encode`'s
folds *and* `step`'s, `step` no longer has any, and dropping terms from a min can only raise it. This
is a pre-existing defect in `02` that #158 surfaced.

## Consequences

**Three documents are amended and one instrument is built.**

- [`02-tick-semantics.md`](../spec/02-tick-semantics.md): the divisor is the standing offset; *The
  check is construction-time, and stays there* is struck; the depth claim is struck; the
  margin-against-offset comparison is stated as an attribution carrying no threshold.
- ADR-0007: the divisor is renamed, the check moves off construction, and *Simultaneous learning does
  not need its own bound* is where #37's half and this one are reconciled.
- [ADR-0005](./0005-timescale-is-persistence-not-a-schedule.md): the precondition re-sources onto
  measured dwell.
- `patchworks.tick.FoldRead`, reading `patchworks.body.CellBody.fold_margin` — the **same** method the
  construction sweep reads, deliberately, because the nomination and the verdict have to be comparable
  numbers rather than two quantities that resemble each other.

**The check keeps its ADR-0005 falsification duty, and the duty travels with the verdict.** It is
measured dwell that can kill the timescale mechanism, not the nomination. What construction loses is
the ability to kill it *cheaply, before anything is trained*; what it gains is that the number it
kills on is the one the run actually has.

**The breach is now a documented standing state of the build rather than an unnoticed one.** Anyone
reading a run — at any tick, not only in its first thousands — should expect `reconciliation_reaches`
to be true on cells that are perfectly healthy. #202's run had no clean tick in 100,000 and 130 of 150
cells clearing ADR-0005's precondition at the horizon, and both of those are the normal picture.

**The per-cell bias is left unpinned, deliberately.** It is the one trainable quantity in the chart's
round trip with nothing bounding it, and decision 2 rules the obvious mechanism unavailable. There is
no second candidate, and nothing to choose between until the live read reports whether the
arrangement's drift ever actually costs a cell its region. Recorded in
[#127](https://github.com/NGL321/patchworks/issues/127)'s *Not yet specified*, not here.

## Alternatives considered

- **A band on the per-cell bias**, matching ADR-0010 and ADR-0015. Rejected on shape, above: the
  margin is a property of the (bias, operating point) pair and a projection reaches only parameters.
  This is the answer to *why isn't this banded the way `K` is*, and it is expected to be asked again.

- **Periodic re-derivation on the anneal schedule** — #33's fix, struck by #37. Not revived: it
  re-derives the **denominator**, which ADR-0010 has since frozen, and would still not see either of
  the two motions this ADR is about. What is reinstated here is a *read*, not a re-derivation: it
  computes nothing the forward pass had not already computed.

- **Ramping `γ` through the transient.** Declined under decision 5, on the ground that #202 measured
  no transient to ramp through.

- **A permanently lower `γ`.** Not declined here. Decision 5 withdrew the decline this ADR first
  entered, because the premise behind it — a stakeless window — did not survive #202. It is
  [#205](https://github.com/NGL321/patchworks/issues/205)'s to weigh as a mechanism.

- **A tolerance in place of the burn-in** — *no more than `k` cells breach after tick `T`*. Declined
  under decision 5: `k` would be read off one run's plateau, and the plateau's height is one draw.

- **Conservatism: band the nomination at a multiple and stop looking.** Declined on #178 — these
  quantities wander 3.8x with no trend, so a multiple is a number chosen against no worst case.

- **Folding this into ADR-0007 as a fourth amendment.** Declined under decision 6: the #37
  reconciliation and the projection argument are the two things a reader needs and neither is an
  amendment to anything.
