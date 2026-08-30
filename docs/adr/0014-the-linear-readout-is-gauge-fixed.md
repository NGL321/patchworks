# ADR-0014: The linear readout is gauge-fixed

**Status:** accepted

## Context

Settled in [#138](https://github.com/NGL321/patchworks/issues/138), as part of the Koopman conversion
([#127](https://github.com/NGL321/patchworks/issues/127), stage 1). That conversion replaces the
frozen nonlinear `step` with a per-cell learned linear operator `K`, and it is taken for one stated
reason: `body` is a term in the rim-to-core transmission budget, and today it is a random draw nobody
can set. Under the conversion it becomes a property of `K` — settable, bounded, and the same object as
the stability constraint. One knob, two purposes.

The conversion's own slogan, *all nonlinearity lives in the cell's lift; everything downstream of
`encode` is linear*, was contradicted by the ledger it shipped with, which kept `decode` as a ReLU
MLP. `decode` is downstream of `encode`. One of the two had to give, and the slogan won: `decode`
linearises with `step`.

That immediately raises a question the nonlinear readout never posed. With a linear readout a cell's
prediction is `D K z`. **If both `D` and `K` are learned, the factorisation is not identifiable** —
`D K = (D M)(M⁻¹ K)` for any invertible `M` — so `K` can be rescaled freely and compensated in `D`,
and **`σ_max(K)` is not a well-defined quantity at all**, let alone one worth constraining.

## Decision

**`decode` is a single frozen linear map `D`, shared by every predicting cell. Its per-cell output
bias is retained; its hidden layer and hidden bias are deleted with its nonlinearity.**

This is a **gauge fixing**, not a capacity choice. It is the same move as
[ADR-0010](./0010-restriction-map-scale-is-gauge-fixed.md), *restriction map scale is gauge-fixed*,
applied to the body instead of the sheaf: in both cases an objective leaves a quantity unidentified,
and in both cases the response is to fix it by construction rather than to add a term that pretends to
identify it.

Since a settable, bounded `σ_max(K)` is the *entire* stated reason the conversion is being taken, a
learned `D` would dissolve the knob the change exists to create. Freezing the readout is what makes
[ADR-0015](./0015-the-cell-operator-band-is-on-the-spectral-norm.md)'s subject exist.

### The output bias is retained, and the distinction from `K`'s is principled

[ADR-0004](./0004-linear-restriction-maps-assume-local-flatness.md)'s no-constant-term argument
reaches `K` and not `decode`, and the line is not drawn for convenience:

- an **affine `K`** makes the *dynamics* affine — a drift that compounds every tick, which is exactly
  the persistent offset ADR-0004 refuses to let a linear map launder away;
- a **`decode` output bias** is a *static readout offset* that never accumulates, and is the standard
  constant observable of a dictionary. Without it the prediction is pinned to a subspace **through the
  origin**, and any nonzero mean in the stalk becomes permanently unreachable error.

## Consequences

**The per-cell adapting surface changes shape.** `step`'s two bias vectors go with the map and
`decode`'s hidden bias goes with the hidden layer, so the biases fall from six vectors to three — 146
numbers to 89 — before `K`'s are added beside them, reaching 233. The conversion deletes 38 learned
per-cell numbers before it adds one.

**One backward pass yields both gradients.** With `decode` linear and frozen, `K` and the surviving
biases lie on the same path; there is no second pass to arrange. This is what let the prediction rule
widen rather than split ([ADR-0008](./0008-the-local-rule-splits-by-parameter-not-by-cell.md), as
amended).

**The forward path gets cheaper**, which is a small independent gain: two of the body's three maps
lose their hidden layer, and `K` is one batched matrix multiply.

### The cost, pre-registered as this decision's falsification

**Every cell's predictions are confined to the same fixed `k`-dimensional linear subspace `im(D)` of
its `n`-dimensional node stalk.** Under the old nonlinear `decode` they reached a curved
`k`-manifold. The accommodation moves to the restriction maps, which are learned and whose job is
already to set the stalk's basis.

**If that subspace proves systematically unreachable for real stalks, the gauge is wrong.** The test
needs a graph that transmits, so it cannot be run yet; what can be done in advance is make the failure
*readable*, and [ADR-0004](./0004-linear-restriction-maps-assume-local-flatness.md) now carries it as
a **fourth cause of the static floor**, with a distinguishing signature the other three lack: the
residue of an unreachable `im(D)` is a direction *shared across unrelated edges* and known at
construction, where curvature and self-intersection are per-edge and the lag floor is per-level. It is
the cheapest of the four to rule out, and is read first.

**A second cost, named precisely rather than left vague.** Completeness results for stable Koopman
embeddings — Fan et al., *Learning Stable Koopman Embeddings*
([arXiv:2110.06509](https://arxiv.org/abs/2110.06509)) — prove that every discrete-time contracting
model is representable in their framework, but the result holds *over the embedding and the operator
**jointly***. With `encode` frozen and `decode` now gauge-fixed, only `K` is free, so **that
completeness does not transfer to this design.** The gauge buys a meaningful `σ_max(K)` and pays for
it in expressiveness the literature can quantify. Recorded as a known price, not a hidden one.

**And it removes an escape hatch that did not exist.** Deep Koopman methods route around the 2016
invariant-subspace theorem by way of a *nonlinear decoder*. Liu, Ozay & Sontag (*Automatica* 2025,
doi [`10.1016/j.automatica.2025.112220`](https://doi.org/10.1016/j.automatica.2025.112220)) prove an
obstruction requiring only **continuity** of the encoder: any continuous one-to-one immersion to a
class of systems including linear systems cannot distinguish different ω-limit sets. A nonlinear
`decode` cannot invert a map that has already merged two limit sets, so giving it up costs nothing
against the obstruction it was implicitly defending. That obstruction now lands wholly on `encode`,
which is the body's only remaining nonlinearity — a real open question, and not this ADR's to answer.

## Alternatives considered

**A learned linear `decode`.** Rejected on identifiability, above. It is the option that looks like it
costs nothing and in fact costs the whole reason for the conversion.

**Keeping `decode` nonlinear.** This was the shipped design and the ticket's own ledger. Rejected
because it contradicts the conversion's governing principle — *everything that can be linear should
be linear, and the nonlinearity is frozen* — and because the theorem above shows the expressiveness it
was buying does not defend what it appeared to defend.

**Folding this into ADR-0001.** Rejected: ADR-0001 answers what may adapt and on what schedule, which
is a different question. The identifiability argument deserves its own decision and its own
falsification rather than a paragraph inside one about freezing.
