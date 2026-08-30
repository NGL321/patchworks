# ADR-0008: The local learning rule splits by parameter, not by cell

**Status:** accepted

## Context

The adapting surface has two parameter groups that do different jobs: biases run the predictive-coding
element and set a cell's inference (its operating point in the shared frozen body); restriction maps do
transport — fixing the basis in which a cell's features become comparable to a neighbour's. A single
learning rule serving both would have to pick one target and force the other parameter group to serve
it, or read something not already local to the cell.

Two signals are already sitting at a cell without any new channel: its own **prediction error**
(`decode`'s prediction from last tick against the node stalk it now reads in as evidence — a quantity
already shaped by whatever reconciliation did between the two ticks), and per-edge **disagreement**
(already computed during the message-passing phase, per
[ADR-0007](./0007-the-disagreement-floor-is-tolerated-not-represented.md)).

## Decision

**The local learning rule is two rules, not one, split by parameter group and by which of the two
already-local signals each one trains on:**

- **The prediction rule** trains the cell's own inference parameters — its biases **and** its
  operator `K` — on **prediction error**, by a local gradient step through the cell's own forward path
  (`encode`/`K`/`decode`), followed by a projection restoring `K`'s band. This is a closed backprop
  entirely inside one cell — no different in kind from ordinary backprop-to-input, just stopped at the
  parameters the shared body doesn't own. It is the predictive-coding element, and it trains inference.
  *(Widened and renamed from **the bias rule** by
  [#139](https://github.com/NGL321/patchworks/issues/139); see the amendment below.)*
- **The transport rule** trains restriction maps on **disagreement**, composed in the same gradient step
  with the sparsity pressure already named in `01-cell-and-sheaf.md`. It trains transport, never
  inference, and a neighbour's raw node stalk is explicitly not an input to it — that would defeat the
  map's purpose, which is to make two cells' features comparable when they don't yet share a basis.

**Locality boundary**: strictly the cell. Both rules read only the chart, node stalk, and per-edge
disagreement the cell already has by the time reconciliation finishes in the same tick — nothing crosses
the graph that the architecture wasn't already routing.

**Permitted global signals**: exactly two, both schedule-shaped rather than information-shaped, so
neither smuggles in a global error channel — a single learning-rate scalar (mirroring reconciliation's
`γ`) and the sparsity-pressure anneal already named in `06-graph-topology.md`. Nothing else
broadcasts. **A construction-time ratio is not a third**: see the amendment below.

## Consequences

- `CONTEXT.md` gains **Prediction error**, **Prediction rule** (originally **Bias rule**), and
  **Transport rule** as vocabulary,
  distinguishing prediction error (cell-owned, temporal) from disagreement (edge-owned, spatial) —
  a distinction the existing glossary's `_Avoid_` lists already implied but never named.
- Sparsity pressure is not a separate mechanism or a second update loop on the restriction maps; it is
  one additive term inside the transport rule's single gradient step.
- **Stability under simultaneous, cell-local learning is explicitly not resolved by this ADR.** It
  entangles with change gating ([#20](https://github.com/NGL321/patchworks/issues/20)) and the
  probabilistic sheaf (held in the map's fog), and is carried forward as its own ticket rather than
  forced here.

## Alternatives considered

- **One rule, one signal**, training both parameter groups off a single combined error. Rejected: biases
  and restriction maps are training different tasks (inference vs. transport), and collapsing them to
  one signal would make one parameter group serve an objective that isn't its own — the same shape of
  mistake `01-cell-and-sheaf.md` already rejected for `step` learning to weight a second correction
  stream.
- **Reading a neighbour's raw node stalk** as an input to the transport rule. Rejected: it defeats the
  restriction map's purpose, which is to make a basis change meaningful; a raw neighbour stalk is in the
  wrong metric space until the map itself has done that work.
- **Hebbian or forward-forward-style updates**, needing no backprop through the cell's own forward path.
  Rejected in favour of gradient descent: the predictive-coding framing already implies a comparison
  between a prediction and an observation, which is exactly a gradient target, and the frozen body's
  forward pass is small enough that a local backprop through it costs nothing a Hebbian rule would have
  saved.


## Amended by [#139](https://github.com/NGL321/patchworks/issues/139): the split widens, and the first rule is renamed

The Koopman conversion gave every cell a learned operator `K`
([ADR-0014](./0014-the-linear-readout-is-gauge-fixed.md),
[ADR-0015](./0015-the-cell-operator-band-is-on-the-spectral-norm.md)), and the question this ADR had
to answer again was whether `K` gets its own rule.

**It does not. The first rule widens from *biases* to *the cell's own inference parameters*, `(b, K)`.**

This ADR's principle is *split by parameter group where the objectives differ*, and here they do not:
`K` sits on the same forward path, is owned by the same cell, is trained on the same signal toward the
same objective, in the same backward pass, on the same cadence. The architecture's cleanest sentence
survives intact — **prediction error trains how a cell thinks; disagreement trains how it talks** —
and `K` is squarely on the thinking side.

**The rename is wholesale rather than an alias.** The rule is named for its **signal**, which is what
this ADR genuinely splits on, and it now pairs symmetrically with the transport rule, which was
already named for its job rather than its parameters. "Bias rule" was never wrong so much as narrow:
what it trained happened to be all biases, and that coincidence ended with the conversion. **Prediction
error keeps its name** — it is the signal, and it is unchanged.

### The three grounds for a third rule, each tested separately

| ground | verdict |
|---|---|
| **Different learning rates** | Real, and not a split. Fixed by an explicit ratio, below. |
| **The projection** | Fails outright. **A projection is not an objective.** |
| **Stability under simultaneous learning** | Not a ground for a split at all — an open question about the widened rule, recorded rather than answered. |

**The projection was the weightiest of the three and fails hardest.** ADR-0015 requires `σ_max(K)`
restored to its band after each step, which is structurally unlike a bias step — but this ADR does not
split on *structure*, it splits on **which already-local signal a parameter group trains on**. A
post-step gauge restoration carries no signal whatever: it is not in the objective, it has no
gradient, and it reads nothing the cell did not already own. The transport rule has carried exactly
such a projection since [ADR-0010](./0010-restriction-map-scale-is-gauge-fixed.md) without that making
it two rules, and `K`'s projection takes that placement unchanged.

### The rate: `η_K = c · η`, and why `c` is not a third global signal

The rates *should* differ, for a mechanical reason rather than a philosophical one: `K`'s gradient is
scaled by the frozen `‖D‖·‖J_encode‖` where a bias gradient is not, so equal step sizes do not mean
equal steps. But this ADR permits **exactly two** global signals, and a second `η` would be a third.

**So `c` is a construction-time constant**, and that is a reading of the permitted-globals clause
rather than a stretch of it. What the clause guards is stated in the ADR itself: the permitted signals
are *schedule-shaped rather than information-shaped, so neither smuggles in a global error channel*. A
fixed ratio carries no cell's error anywhere. It is visible, auditable, and sits beside `a` and `ρ_K`
as one more number the build fixes — where a hidden per-group rate buried in an optimiser would be
exactly the thing worth objecting to.

`c` is a **default with a stated retune duty, not a rule.** Unlike `a`, which ADR-0015 made a rule the
selection rig produces because a construction-time go/no-go depends on it, nothing gates `c`; it
inherits `η`'s own status as the thing to retune first once a run can be measured. Deriving a rig rule
for it would be choosing a number and calling it a derivation.

### Cadence, and no additive term

**Every tick, for `K` as for the biases.** A slower cadence was rejected: giving a parameter group its
own *schedule* is a far stronger claim to separateness than a rate is, and it would stand up a second
timescale mechanism beside the one ADR-0005's amendment just reopened.

**And no additive term.** ADR-0015's band forbids non-normal transient amplification while a dense `K`
trained on a temporal objective pulls toward it, so the projection and the gradient fight every step.
The tempting fix — a soft penalty, the sparsity-pressure analogue — is **refused**: the fight is not a
defect to damp but the observable that triggers ADR-0015's named fallback to a structured `K`. A
consequence worth stating plainly: the transport rule carries an additive term and the prediction rule
carries none, and that asymmetry is now on the record with a reason rather than as an accident of what
each rule happened to need.

### What the widening actually costs to implement: nothing

There is no allowlist of trained parameters anywhere, and deliberately so. The body's weights are
registered as **buffers** and the per-cell surface as **parameters**, so *buffers are the frozen body,
parameters are the adapting surface* (ADR-0001, as amended) — and **registering `K` as a per-cell
parameter *is* the widening.** The rule needs no change and neither does the gradient.

An explicit list was rejected on its failure direction. Implicit fails loudly: a parameter added for
another reason is silently *trained*, and behaviour changes immediately. Explicit fails quietly: a
parameter added and forgotten is silently never trained, sitting at its initial value forever — on a
system whose standing diagnosis is *the arm does not move*, indistinguishable from the bug already
being hunted. The louder failure is the right one to keep.

### The standing note, still standing

This ADR's own note that *stability under simultaneous, cell-local learning is explicitly not resolved
by this ADR* is unchanged by the widening. It now has a **named successor**: whether a learned `K` can
fail to settle under ambiguous, sign-flipping evidence — the analogue of `07`'s settling floor in the
new vocabulary. It cannot be settled by argument, because it needs charts from a graph that transmits.
ADR-0007 carries the answer if it turns out yes.

### The locality guarantee, confirmed rather than assumed

Stated because it is the constraint most likely to be assumed broken. `K`'s update reads only the
cell's own chart, its own node stalk, and its own prediction error. Nothing crosses a cell boundary,
and the projection is local too — a cell owns its own `K` outright and needs nothing from a neighbour
to take its norm. ADR-0011's enforcement applies unchanged and needs no new machinery: the objective
stays a plain sum over cells whose graph has no cross-cell edge, so the batched gradient is each
cell's own local gradient exactly.
