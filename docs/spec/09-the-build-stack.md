# The build stack

What this gets built in, and what stops the build from quietly violating the thesis while it does.
Settled in [patchworks#12](https://github.com/NGL321/patchworks/issues/12); see
[ADR-0011](../adr/0011-the-locality-guarantee-is-enforced-not-inherited.md).

Deliberately placed *after* the local learning rule ([`07-local-learning-rule.md`](./07-local-learning-rule.md)):
the rule's shape decides which framework fits, and choosing the framework first would have quietly
constrained the rule. Terms used here are defined in [`CONTEXT.md`](../../CONTEXT.md).

## PyTorch

The candidates were JAX and PyTorch. Two of the arguments that would have picked JAX turned out to
have been dissolved by decisions this spec had already taken:

- **"Many tiny MLPs" is not the shape of this graph.** The cell body is *shared and frozen*
  ([`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md), *The cell body*), so there is one body and a
  `[cells, ·]` leading dimension of biases — not 150 small models. The population's forward pass is a
  handful of `~150 × 32` matmuls in any array library. `vmap` over heterogeneous per-cell parameters,
  JAX's clearest advantage, is machinery this design has no use for.
- **The tick cannot be fused into one compiled program anyway.** The environment is classic MuJoCo on
  the host (below), so every tick round-trips out of the framework and back. Compiling the *tick* is
  available in both stacks; compiling the *loop* is available in neither.

What is left is a real but narrow advantage — see ADR-0011 — against an ecosystem the rest of this
build leans on. The stack is **PyTorch**.

### Measured, not assumed

On the development laptop (i7-8750H, CPU only), at the sizes fixed in
[`06-graph-topology.md`](./06-graph-topology.md) — ~150 predicting cells, ~698 edges, `n = 32`,
`k = 12`:

| | |
| --- | --- |
| `env.step()` — physics + the 64×64 render | **3.18 ms** |
| agent tick, forward (numpy stand-in, body width 128) | **1.04 ms** |
| agent tick, with gradients (×3 estimate) | **~3 ms** |

The environment costs about what the whole agent costs, every tick. A framework difference of 2×
therefore moves total wall-clock by well under a third of something that is not the bottleneck. The
body's hidden width is not yet specified, so the second row is an order-of-magnitude figure, not a
commitment.

## The locality guard

The architecture's central constraint is **local learning rules only** — no error signal propagated
across the graph. PyTorch's autograd is a single global tape, and the sheaf is a connected graph, so
that constraint is not inherited from the framework. It is **enforced**, in three parts, and the
enforcement is part of the spec rather than a coding convention.

### 1. The tick carries no tape

The whole tick — inference phase and message-passing phase, [`02-tick-semantics.md`](./02-tick-semantics.md) —
runs under `torch.no_grad()`. This is not an optimisation. The tick is a rollout, not a training pass:
every quantity it produces is *data* the cell then learns from, and none of it should be differentiable
through. Reconciliation edits the node stalk; `decode` emits a prediction; disagreement is derived per
edge. All of it is plain arrays by the time the phase ends.

### 2. Learning is a separate phase over detached inputs

Both halves of the local learning rule then run as a **separate** phase, as a function of

- that cell's own parameters — its biases and its incident restriction maps, the **adapting surface**, and
- detached arrays: the tick's state as the tick left it — its chart, its node stalk, the neighbour
  contribution to each of its incident edges.

Each cell's update builds a fresh, small graph — the closed backprop through the cell's own frozen
forward path that `07-local-learning-rule.md` specifies for the bias rule, and the per-edge gradient plus
gauge projection for the transport rule — and that graph dies at the end of the step. A neighbour's
parameters are not merely severed from it; they are not in it.

**What is detached is the state, not the quantity being descended on.** Each rule *recomputes* its own
objective live in that cell's own parameters; a number carried over dead from the tick has no gradient
in anything. The bias rule re-runs the cell's own frozen forward path so that prediction error is live
in the biases, against a detached target. The transport rule recomputes the disagreement on each
incident edge so that it is live in that cell's own map — and this is the one place where the sheaf
differs from a feedforward chain in a way that matters here. Disagreement on an edge is a function of
**both** of its maps, and each map belongs to a different cell's adapting surface. So a cell's transport
objective contains a neighbour's *trainable parameter*, which a layer's loss in a feedforward network
never does. **The neighbour's map is what has to enter detached**, and it is the only cross-cell
parameter in the phase.

**This still batches.** Because every coupling term enters as a detached constant, the cells' local
graphs compose into one batched graph *with no cross-cell edges*. The gradient of the sum over cells is
therefore exactly the per-cell local gradient, cell by cell — the batching that
[`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md) calls the design's concrete single-machine argument is
preserved, and it is preserved *because* of the detachment rather than in spite of it.

**And it depends on the body being frozen.** The identity above is the ordinary one, but it has a
condition, and this spec relies on the condition without stating it: a summed batch loss decomposes into
per-item gradients only when the batched items **share no trainable parameter**. Where they do share
one, the sum gives an average rather than a stack, which is the entire reason `torch.func`'s per-sample
gradients and Opacus exist. Patchworks batches predicting cells over a body that is *shared* — so what
saves the identity is that the body is also *frozen*, leaving the adapting surface per-cell biases and
per-edge restriction maps
([ADR-0001](../adr/0001-continual-learning-applies-to-the-adapting-surface.md)). Nothing trainable spans
the batched dimension, and the equivalence is exact rather than approximate.

Worth stating because the freeze is explicitly *not* load-bearing — it is the top rung of
`01-cell-and-sheaf.md`'s flex priority ladder. No rung on that ladder actually breaks this, as it
happens: per-cell adapters (rung 1) and an unfrozen per-cell body (rung 3) are both per-cell, and
heterogeneous bodies (rung 2) stay fixed. The one shape that would break it is a body both **shared and
trainable**, which the ladder never reaches. If a later design arrives there anyway, this paragraph is
what has to be re-argued, and `vmap(grad(·))` over per-cell parameters is the standard repair.

#### Written as a function transform

The phase is written with **`torch.func`**: each rule is a pure function whose parameters arrive as an
explicit argument (`functional_call` replaces the module's ambient parameters), and the gradient is taken
with `grad(·)` scoped by `argnums` to that cell's own adapting surface. The neighbour's map is an
ordinary argument of that function and is not among the `argnums`, so it is not differentiated — not
because it was severed, but because differentiation only ever traverses what was named.

**The reason is the direction in which each idiom fails.** This guard exists at all because a leaked
gradient is silent and flattering (§3). Under ambient autograd, the neighbour's map is a live parameter
and a deleted `.detach()` produces an **extra** gradient: exactly the silent, flattering failure, visible
only to the perturbation test. Under the transform, naming the wrong `argnums` produces a **missing**
gradient — a cell that stops learning, which is loud and immediate. Same criterion as the rest of this
section, applied to how the phase is written rather than to what it computes.

It is cheap here for reasons that are specific to this design rather than general. The update is a plain
local gradient step under a global learning-rate scalar
([`07-local-learning-rule.md`](./07-local-learning-rule.md)), so there is no optimiser state to give up
by receiving gradients as a pytree instead of on `.grad`. Batching is untouched: parameters keep their
`[cells, ·]` leading dimension, the objective is a sum, and `grad` returns the same batched gradient the
paragraph above describes — no `vmap` is needed, for the same reason ADR-0011 gives for not needing it in
the first place. ADR-0010's gauge projection runs after the step, outside the transform. And
`torch.func`'s documented constraint on in-place operations bites on the *tick*, where reconciliation
edits the node stalk in place — which is §1's no-tape phase and is not transformed.

**What it does and does not buy, stated precisely.** It closes **parameter reachability**, structurally,
and it closes nothing else. That is worth having because parameter reachability is the one leak class
§1 cannot cover — parameters live outside the tick, so a no-tape tick says nothing about them. It does
*not* touch the shared-storage class in §3, where cells couple in place with a clean tape and no
transform can see it. Two honest limits carry with it: the property is **inferred** from the
pure-function contract and `argnums` rather than asserted anywhere in the documentation, and **no source
was found using `torch.func` for locality** — this is the field's absence, not its endorsement. So the
guarantee is enforced here too. It is simply enforced by construction on one class instead of by
convention, and the standing test in §3 is what covers the rest.

### 3. The guarantee is tested, because a leak would flatter us

A leaked gradient does not crash and does not look like a bug. It makes the agent work *better* — it
looks like the thesis being right. That asymmetry is why the guard carries a standing test rather than a
review habit:

- **Assertion, cheap and always on**: nothing leaving the tick has a `grad_fn`.
- **The perturbation test**: perturb one cell's parameters, and assert that no other cell's update
  changes. This is the thesis' load-bearing claim written as something CI can falsify.

**The two catch different leaks, and neither subsumes the other.** The assertion inspects the *tape*;
the perturbation test inspects the *update*. The gap between them is documented rather than
hypothetical: `Tensor.detach` returns a tensor sharing storage with the original, so an in-place write
through a detached view couples two cells while leaving a perfectly clean tape — no `grad_fn` anywhere,
and a batched graph that genuinely has no cross-cell edges. The assertion cannot see that class by
construction; only observing the update catches it. This is why the perturbation test in
[ADR-0011](../adr/0011-the-locality-guarantee-is-enforced-not-inherited.md) is the load-bearing half of
the guard rather than scaffolding around the cheap check.

## The environment boundary

**Classic MuJoCo, host-driven loop, numpy at the boundary.** MJX — MuJoCo re-implemented in JAX — is
declined on three counts, and only the third is about the framework choice: it exists for massively
parallel rollouts, which this design has no use for (one continual stream, no episodes,
[`03-the-sandbox.md`](./03-the-sandbox.md)); its contact model differs from classic MuJoCo's on exactly
the contact-rich pushing this sandbox is built around; and classic MuJoCo is the far better-maintained
system, the re-implementation serving a much narrower use case. The 64×64 top-down render — the agent's
only view of the pucks — is also not something MJX renders natively.

The environment therefore stays **framework-agnostic**: the sandbox and its spec survive unchanged if
the stack decision is ever reversed. `prototypes/sandbox` is already numpy + classic MuJoCo and is the
primary source behind `03-the-sandbox.md`.

## The compute target

**CPU is the declared target, and the first build measures rather than assumes.** The standing
constraint is a single consumer GPU, laptop-viable where possible; the numbers above suggest this model
may not need a GPU at all, which is the *stronger* version of that thesis and should be reported as a
result rather than buried as a default.

- **Development laptop** (Intel x86_64 macOS) is a **correctness-only** target on pinned wheels. Both
  frameworks are frozen there by wheel availability — `torch` caps at **2.2.2**, `mujoco` is already
  pinned to **3.10.0** for the same reason. Nothing about the architecture depends on a newer release;
  if something ever does, the laptop stops being the reference machine rather than the decision changing.
- **The AMD desktop is not a GPU target.** The RX 580 is Polaris (`gfx803`), dropped by modern ROCm;
  PyTorch's ROCm builds target `gfx90a` / `gfx1030` and later. That card would run CPU code with extra
  steps.
- **A rented NVIDIA cloud box is the named escape hatch**, and it is *gated on measurement* — reached
  only if tick wall-time on CPU proves insufficient, matching the Notes' existing "paid compute if a
  specific need is demonstrated". Nothing is provisioned until that fires.

## What this file does not decide

- **The snapshot format** for continual-learning restore ([ADR-0001](../adr/0001-continual-learning-applies-to-the-adapting-surface.md)
  makes reproducibility a full-state snapshot). Still open; the stack constrains it but does not choose it.
- **The body's hidden width**, and module layout generally.
