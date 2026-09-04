# The local learning rule

What updates a cell's adapting surface, given the cell and sheaf contract fixed in
[`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md) and the tick semantics fixed in
[`02-tick-semantics.md`](./02-tick-semantics.md). Settled in
[patchworks#5](https://github.com/NGL321/patchworks/issues/5); see
[ADR-0008](../adr/0008-the-local-rule-splits-by-parameter-not-by-cell.md), amended by
[#139](https://github.com/NGL321/patchworks/issues/139) when the Koopman conversion widened the first
rule and renamed it.

Terms used here are defined in [`CONTEXT.md`](../../CONTEXT.md).

## Two rules, not one

The adapting surface has two parameter groups doing different jobs — the cell's own inference
parameters run inference, restriction maps run transport (see *The division of labour between the two
adapting surfaces*, [`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md)) — and the local learning rule
respects that split rather than collapsing it. It is **two rules**, each training one parameter group
off one of the two signals already sitting at a cell without any new channel.

The sentence the split is for: **prediction error trains how a cell thinks; disagreement trains how it
talks.**

### The prediction rule

Trains the cell's own inference parameters — its **biases and its operator `K`** — on **prediction
error**: the difference between what `decode` predicted last tick and the node stalk the cell reads in
as evidence this tick. Because reconciliation edits the node stalk between the two ticks, this signal
already carries whatever the neighbours' disagreement did to the cell's belief — without the rule ever
reading a neighbour directly.

> **This was the bias rule until [#139](https://github.com/NGL321/patchworks/issues/139).** The
> Koopman conversion gave each cell a learned operator, and ADR-0008 splits on *which already-local
> signal a parameter group trains on* rather than on structure: `K` is on the same forward path, owned
> by the same cell, trained on the same signal toward the same objective, in the same backward pass, on
> the same cadence. So the rule **widened** rather than gaining a sibling, and it is renamed for its
> signal — which is what the split is actually about, and what makes it pair symmetrically with the
> transport rule. The name "bias rule" was never wrong so much as narrow: what it trained happened to
> be all biases, and that coincidence ended with the conversion. **Prediction error keeps its name**;
> it is the signal, and it is unchanged.

The update is a local gradient step through the cell's own forward path — `encode`, `K`, `decode` —
stopped at the per-cell surface the shared body doesn't own. This is a closed backprop entirely inside
one cell: no different in kind from ordinary backprop-to-input, and no different in cost, since the
body is small and the pass never leaves the cell. `decode` is linear and frozen, so **one backward pass
yields both gradients** — `K` and the surviving biases lie on the same path and there is no second pass
to arrange.

**The band is restored after the step**, not inside the objective: `σ_max(K)` is projected back into
`[1/ρ_K, 1]` exactly as ADR-0010's gauge projection restores a restriction map's scale
([ADR-0015](../adr/0015-the-cell-operator-band-is-on-the-spectral-norm.md)). That placement is what
makes it a projection rather than a second rule — it is not in the objective, it has no gradient, and
it reads nothing the cell did not already own.

**No additive term, and the asymmetry with the transport rule is gone.** The band forbids
non-normal transient amplification while a dense `K` trained on a temporal objective pulls toward it,
so the projection and the gradient fight every step. The tempting fix — a soft penalty, the
sparsity-pressure analogue — is refused: the right template for a stability constraint is a direct
parameterisation rather than a penalty, so the fight is not a defect to damp but the **observable that
triggers the fallback** from a dense `K` to a structured one. The transport rule carried an additive
term where this one carried none, and the asymmetry was on the record with a reason; [#406](https://github.com/NGL321/patchworks/issues/406)
deleted that term, so **neither rule carries a penalty now** — and the reason above still stands as
the reason this one never will.

**What is trained is defined rather than listed.** There is no allowlist: the body's weights are
registered as buffers and the per-cell surface as parameters, so *buffers are the frozen body,
parameters are the adapting surface*, and registering `K` as a parameter **is** the widening. The
alternative was rejected on its failure direction — an implicit target fails loudly, silently
*training* a parameter added for another reason, where an explicit list fails quietly, leaving one
never trained and sitting at its initial value forever.

This is the predictive-coding element of the architecture. It trains inference — the cell's operating
point and its chart's dynamics — and never touches a restriction map.

### The transport rule

Trains restriction maps on **disagreement**: already computed during the message-passing phase, and per
[ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md) never aimed at a zero
target.

Of ADR-0007's two permitted objectives the rule takes the **relative** one, and the choice is
load-bearing rather than stylistic ([ADR-0010](../adr/0010-restriction-map-scale-is-gauge-fixed.md)).
Learning on *change* in disagreement does not exclude the trivial solution: shrinking the maps produces
a negative change every step, which a change-descending rule can read as progress. Learning on
disagreement **relative to the restricted beliefs' own current magnitudes** — `‖F_u x_u‖ + ‖F_v x_v‖` —
does, because the ratio is unchanged when both of an edge's maps scale together, so shrinking the edge
buys nothing. Scaling *one* map is not invariant, and points the other way: the ratio lies in `[0, 1]`
and sending a map to zero sends it to `1`, its worst value. The normaliser is
**locally stateless** and deliberately not a running average of the edge's recent scale: that would be a
per-edge auxiliary variable with a hand-set time constant, the object ADR-0007 rejects under *A per-edge
learned baseline*, and it is the same criterion form the change gate settled on
([`05-timescales.md`](./05-timescales.md)) for the same reason.

The update is a local gradient step on that quantity and on nothing else. It used to be composed **in
the same step** with a sparsity pressure — one additive penalty term, an L1 on the normalised map,
inside one descent step — and [#406](https://github.com/NGL321/patchworks/issues/406) deleted that term outright;
[ADR-0031](../adr/0031-the-sparsity-pressure-is-deleted.md) carries the grounds. There is one term, so there is nothing here
trading against transport.

The one term is blind to a map's overall magnitude, which is why that magnitude is **fixed by
construction rather than learned** (`01-cell-and-sheaf.md`, *Scale is gauge-fixed*): interior maps are
projected back into `‖F‖_F ∈ [1/ρ, ρ]` after each step, boundary maps onto `‖F‖_F = 1`. The projection
is part of the transport rule's step and is as local as the rest of it — a cell owns its own incident
maps and needs nothing from a neighbour to project them. It is **not** inert in practice: an edge's
joint scale grows monotonically under a scale-invariant objective, so the upper face binds essentially
every step once a map reaches `ρ`.

This trains transport: the basis in which a cell's features become comparable to a neighbour's. It
never reads a neighbour's raw node stalk — only the disagreement already derived from it during
reconciliation — because the whole point of the map is to make two cells' features comparable when they
don't yet share a basis; a raw neighbour stalk is in the wrong space until the map has done that work.

## Locality boundary

Strictly the cell. Both rules read only the chart, node stalk, and per-edge disagreement the cell
already has once reconciliation finishes in the same tick. Nothing is read that the architecture wasn't
already routing to the cell through the ordinary message-passing channel.

That boundary is what the rule *is*. What stops an implementation from crossing it accidentally is
specified separately, in [`09-the-build-stack.md`](./09-the-build-stack.md), *The locality guard*: the
tick carries no autograd tape, each cell's update is a function of detached inputs plus its own adapting
surface, and a standing perturbation test asserts that no cell's update moves when another cell's
parameters do ([ADR-0011](../adr/0011-the-locality-guarantee-is-enforced-not-inherited.md)).

## Permitted global signals

**Exactly one**, schedule-shaped rather than information-shaped, so it carries no particular cell's
error across the graph:

- a single global **learning-rate scalar**, mirroring reconciliation's `γ`

**It was two until [#406](https://github.com/NGL321/patchworks/issues/406).** The other was the *sparsity-pressure anneal*,
and deleting the pressure took its schedule with it — the anneal was the only non-learning-rate
broadcast in the architecture, so the count moves from two to one and the locality story gets
simpler rather than differently qualified ([ADR-0031](../adr/0031-the-sparsity-pressure-is-deleted.md)). No value is owed for
the anneal's horizon or its ramp: [#89](https://github.com/NGL321/patchworks/issues/89) escalated the
anneal's *direction* and left the horizon open, and that question **dissolves** with the term rather
than resolving — nothing is looking for a number here.

Nothing else broadcasts. A global loss, confidence readout, or any other signal carrying content about
a specific cell's error is explicitly rejected — that would be backprop across the architecture wearing
a different name.

**`K` descends at `η_K = c · η`, and `c` is a construction constant rather than a third signal.** The
rates should differ for a mechanical reason: `K`'s gradient is scaled by the frozen `‖D‖·‖J_encode‖`
where a bias gradient is not, so equal step sizes do not mean equal steps. What the clause above guards
is that a permitted signal be *schedule-shaped rather than information-shaped*, and a fixed ratio
carries no cell's error anywhere. It is visible and auditable, and sits beside `a` and `ρ_K` as one
more number the build fixes — where a hidden per-group rate buried in an optimiser would be exactly
the thing worth objecting to. Unlike `a` it is a **default with a retune duty rather than a rule**:
nothing gates it at construction, so it inherits `η`'s own status as the thing to retune first once a
run can be measured.

**The cadence is every tick, for `K` as for the biases.** A slower cadence was considered and
rejected: giving a parameter group its own *schedule* is a far stronger claim to separateness than a
rate is, and it would stand up a second timescale mechanism beside the one the conversion just
reopened. Timescale belongs to `K`'s spectrum ([`05-timescales.md`](./05-timescales.md)) and is not
smuggled in here as an update schedule.

## What information cohomology does not supply

Considered and disqualified as the shape of either rule; recorded in
[`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md#two-cohomologies-which-are-not-the-same-cohomology) and
`docs/research/015-information-cohomology.md`. No variational principle, no graph in the theory, and an
`O(2ⁿ)` cost ceiling that would not batch. Retained only as an interpretive lens.

## Stability under simultaneous, cell-local learning

Settled in [#33](https://github.com/NGL321/patchworks/issues/33); ADR-0007 amended. Within a tick,
both rules already take the same shape ADR-0002 requires of reconciliation — a single local step,
applied after that tick's signals are read — so simultaneity across cells adds no new locality
problem. Across ticks the risk splits by parameter group and neither half needed new mechanism:

- The **prediction rule** can leave a mid-depth cell oscillating between activation regions under
  ambiguous evidence rather than settling — the **settling floor**, a third kind alongside static and
  lag in ADR-0007's taxonomy. Bounded by the same `γ` that bounds reconciliation; tolerated, not
  represented, like the other two. The floor **survives the conversion**: it is defined over activation
  regions, and those come from `encode`, which is still ReLU. Whether a learned `K` has an *analogous*
  failure — an operator that never settles because its evidence is sign-flipping — is this decision's
  pre-registered falsification and **is not settled here**: it cannot be answered by argument, because
  it needs charts from a graph that transmits.
- The **transport rule** needed no bound of its own. It was found to make ADR-0007's existing
  `γ × floor < fold margin` check go stale, since it trains the magnitudes the `Σ_e m_e` proxy stands in
  for, and the fix was a per-cell re-derivation on the sparsity anneal's schedule (a schedule since
  deleted with the term itself, [ADR-0031](../adr/0031-the-sparsity-pressure-is-deleted.md)). **[ADR-0010](../adr/0010-restriction-map-scale-is-gauge-fixed.md)
  removed the drift at its source** and that re-derivation is struck: with every incident map bounded,
  `λ_max(Σ_e F_evᵀF_ev) ≤ ρ² · deg(v)`, so the denominator in
  [`02-tick-semantics.md`](./02-tick-semantics.md) is a provable bound rather than a proxy that has to
  be re-checked.

ADR-0008's standing note that *stability under simultaneous, cell-local learning is explicitly not
resolved by this ADR* is unchanged by the widening, and now has a named successor in the paragraph
above rather than standing open with nothing pointing at it.

Neither the **change gate** ([`05-timescales.md`](./05-timescales.md)) nor the probabilistic sheaf
plays a role: confirmed orthogonal, and declined a third time respectively. See ADR-0007,
*Simultaneous learning does not need its own bound*. The gate needs no special case in either rule
either, if it is ever built: the sender computes disagreement from its own current restricted
belief, which gating does not touch, and the receiver trains against whatever the edge buffer holds.
