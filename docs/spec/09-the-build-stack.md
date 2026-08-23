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
- detached arrays: its chart, its node stalk, its per-edge disagreements.

Each cell's update builds a fresh, small graph — the closed backprop through the cell's own frozen
forward path that `07-local-learning-rule.md` specifies for the bias rule, and the per-edge gradient plus
gauge projection for the transport rule — and that graph dies at the end of the step. A neighbour's
parameters are not merely severed from it; they are not in it.

**This still batches.** Because every coupling term enters as a detached constant, the cells' local
graphs compose into one batched graph *with no cross-cell edges*. The gradient of the sum over cells is
therefore exactly the per-cell local gradient, cell by cell — the batching that
[`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md) calls the design's concrete single-machine argument is
preserved, and it is preserved *because* of the detachment rather than in spite of it.

### 3. The guarantee is tested, because a leak would flatter us

A leaked gradient does not crash and does not look like a bug. It makes the agent work *better* — it
looks like the thesis being right. That asymmetry is why the guard carries a standing test rather than a
review habit:

- **Assertion, cheap and always on**: nothing leaving the tick has a `grad_fn`.
- **The perturbation test**: perturb one cell's parameters, and assert that no other cell's update
  changes. This is the thesis' load-bearing claim written as something CI can falsify.

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
