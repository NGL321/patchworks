# ADR-0022: A hop is an operator norm along a learned channel, not an isotropic average

**Status:** accepted

## Context

Settled in [#142](https://github.com/NGL321/patchworks/issues/142) and made load-bearing by
[#190](https://github.com/NGL321/patchworks/issues/190); written by
[#155](https://github.com/NGL321/patchworks/issues/155).

For most of this project's life the record held that rim-to-core transmission was **structurally
absent** — that a perturbation at the sensory rim decayed to nothing long before the apex, by a factor
the arithmetic put near `1e14` end to end. Three architectural candidates were opened against that
deficit, and one of them proposed to redesign the message-passing step itself.

The deficit was an artifact of how it was measured. Every reading behind it probed the graph with an
**isotropic** signal — a random or uniform deviation, averaged over directions — against restriction
maps whose measured effective rank is **1.02–1.06**. A near-rank-1 map transmits essentially one
direction and annihilates the rest, so an isotropic probe spends almost all of its energy on directions
the map is built to discard, and reports the average direction's fate rather than the channel's. Read
along the direction the maps actually carry, and with the gain read rather than bounded, the chained
hop is **~184x** what the isotropic reading reports, and the deficit closes.

**No mechanism changed to produce that factor.** What changed is which quantity the hop was taken to
be. That makes it a decision about what the architecture claims, and it needs to be written down,
because two live arguments now rest on it and neither is safe while it is implicit.

## Decision

### The hop is the operator norm of the transported step along the channel the maps have learned

A single edge's contribution to transmission is the gain of the composed map **in its top
eigendirection**, not the expected gain over an isotropic input. The composed rim-to-apex hop is the
product of those along a path, and it is a property of a **channel**: the aligned subspace that a chain
of restriction maps and cell operators actually carries.

The channel is **learned, not designed**. Nothing in the construction nominates it. It is what the
transport rule builds when it aligns adjacent maps to reduce disagreement, and its existence is a
measured fact about a trained surface rather than a guarantee of the contract: cross-edge alignment
reads **14.20x taught against 3.66x untrained**, so the channel is largely a thing training makes.

### This is what licenses `gain = γ / λ_max`

The reconciliation gain divides by a bound on `λ_max(Σ_e F_evᵀF_ev)`
([`02-tick-semantics.md`](../spec/02-tick-semantics.md),
[ADR-0010](./0010-restriction-map-scale-is-gauge-fixed.md)). `λ_max` is the **largest** eigenvalue, so
that gain is the largest step that is stable in the **steepest** direction of the cell's local energy —
which is the same top eigendirection the hop is read along. The two are one choice, made twice and now
stated once.

Under the isotropic reading the pairing looks arbitrary: if a hop is an average over directions, then
normalising by an extremum over directions is a mismatch, and per-cell equalisation of an isotropic
local energy is the invariant that would follow instead. That is precisely the invariant #190 struck,
and this ADR is the ground it was struck on. **Fixing the reading fixes both ends of it at once.**

### What the reading costs: every other direction is under-relaxed

This is not free, and the cost is why the ADR exists rather than a footnote to it. A step scaled by
`γ / λ_max` is well-scaled in the top eigendirection **and under-relaxed in every other**, by that
direction's eigenvalue ratio `λ_i / λ_max`. On near-rank-1 maps that ratio is small, so the non-channel
directions are reconciled slowly — they are not reconciled *wrongly*, and the step stays stable
everywhere, which is what a bound on the largest eigenvalue buys.

So the architecture reconciles **fast along the channel and slowly off it**, by construction rather
than by accident. That is a defensible thing to want — the channel is where the signal is — but it is a
claim about behaviour that nothing else in the record states, and anyone reading a slow off-channel
component as a defect should read it here first.

### Detectability is stated along the channel, and inherits this

[ADR-0021](./0021-rim-to-core-detectability-is-a-bottleneck-ratio.md)'s predicate asks whether **there
exists a channel** that carries a rim perturbation to the apex — the max over paths of the min
bottleneck ratio along a path. That existential is this ADR's reading in the transmission predicate's
own words: it asks after a channel because a hop is a channel quantity, and it would be the wrong
predicate if a hop were an average. The two were settled independently, on different tickets, and they
agree; recording the agreement is cheaper than rediscovering the disagreement.

## Consequences

- **Any transmission reading that probes isotropically is measuring the wrong thing**, and this is the
  standing correction to the instrument. A probe must be aligned to the channel, and a chained reading
  needs **two** message-passing phases per hop — a stalk moves the broadcast in one phase and the
  broadcast moves the far stalk in the next, through the one-tick delay — because the single-phase
  version reads exactly zero. The `1e14` deficit and every figure derived from it are **withdrawn**.
- **The 184x is a measurement of a trained surface, not a property of the contract.** It reproduces on
  the instrument, and it is not a guarantee: an untrained graph carries 3.66x of alignment rather than
  14.20x, and [#154](https://github.com/NGL321/patchworks/issues/154) is the open question of what
  supplies the first disagreement when the channel must be learned before it can carry the signal that
  trains it. This ADR does not close that.
- **Off-channel reconciliation is slow by construction**, and that is now a stated behaviour rather
  than a surprise. It is the one thing this reading gives up.
- **`Σ_e m_e`'s equalisation defence is void on this ground alone**, independently of
  [#189](https://github.com/NGL321/patchworks/issues/189)'s measurement that the property was not held.
  Per-cell equalisation of an isotropic local energy was never the quantity worth holding constant
  against near-rank-1 maps.
- **[ADR-0002](./0002-message-passing-is-one-step-not-a-solve.md) is untouched, and it stays that
  way.** The one-step rule was the candidate the deficit was going to be spent on. #142 priced it
  against a measured cost for the first time and found it is not the defect; the defect was the probe.
- **Effective rank stops being only a collapse diagnostic.** ADR-0010 reads it to tell parameter
  collapse from a draining lag floor. It is also what makes the channel narrow enough for the isotropic
  reading to be badly wrong, so a rank rising toward `m` would not be a health signal here — it would
  mean the channel is wider and the two readings are converging.

## Alternatives considered

- **Leave it implicit.** The measurement stands on its own and the instrument is committed. Rejected
  because two decisions rest on it — #190's striking of equalisation and ADR-0021's existential — and
  an implicit premise under two decisions is the shape this record has repeatedly had to go back and
  repair. A reading that moves a headline figure by 184x is not a measurement detail.
- **State it as a property of the contract rather than of a trained surface.** Rejected as false. The
  channel is 3.66x untrained against 14.20x taught: it is largely built by the transport rule, and
  claiming it at construction would promise something the cold-start case does not have. #154 is open
  for exactly that reason.
- **Adopt a per-direction gain**, relaxing each eigendirection by its own eigenvalue rather than
  accepting under-relaxation off the channel. Rejected: it is a per-cell runtime read of the local
  Laplacian block's full spectrum from live parameters, which is the shape ADR-0010 avoids for the
  largest eigenvalue alone — at higher cost, and with no measured benefit to weigh against it.
- **Take the deficit at face value and redesign message passing.** This is what the record was about to
  do. Rejected by measurement rather than by argument, and recorded here because the near miss is this
  ADR's whole justification.
