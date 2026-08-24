# ADR-0011: The locality guarantee is enforced, not inherited

**Status:** accepted

## Context

Settled in [#12](https://github.com/NGL321/patchworks/issues/12), deliberately sequenced *after* the
local learning rule ([ADR-0008](./0008-the-local-rule-splits-by-parameter-not-by-cell.md)) so that the
rule's shape could decide the framework rather than the reverse.

The architecture's central constraint is **local learning rules only**: no error signal propagated
across the graph. The two candidate frameworks hold that constraint very differently.

- **JAX** has no ambient tape. A cell's update is `grad` of a scalar written by hand over a parameter
  subtree named by hand; a neighbour's parameters are unreachable — not because they were severed, but
  because differentiation only ever traverses what was passed in. The guarantee is *structural*.
- **PyTorch** has one global autograd tape, and the sheaf is a connected graph. A single missing
  `.detach()` at an edge stalk backprops one cell's error into a neighbour. The guarantee, if it exists
  at all, is a *discipline*.

That second bullet was drawn too tightly, and [#66](https://github.com/NGL321/patchworks/issues/66)
corrected it after the fact: `torch.func` reaches the same structural property *inside* PyTorch, since
its transforms differentiate with respect to explicitly named arguments of a pure function. The
asymmetry is real but narrower than stated here — it is the difference between a guarantee that holds
globally and one that holds inside a chosen idiom. See *Alternatives considered*.

That asymmetry is sharpened by the failure mode: a leaked gradient does not crash and does not look
like a bug. It makes the agent perform **better**. It looks like the thesis being right.

**Two arguments that would have picked JAX were dissolved by decisions already taken.** The cell body
is shared and frozen ([ADR-0001](./0001-continual-learning-applies-to-the-adapting-surface.md)), so
there are not many small models to `vmap` over — there is one body and a `[cells, ·]` leading dimension
of biases, and the population's forward pass is a few `~150 × 32` matmuls in any array library. And the
environment is classic MuJoCo on the host, so every tick leaves the framework and comes back: the
*loop* cannot be fused into one compiled program in either stack.

**Measured on the development laptop** (i7-8750H, CPU) at the sizes in
[`06-graph-topology.md`](../spec/06-graph-topology.md): `env.step()` including the 64×64 render is
**3.18 ms**, against an agent tick of **1.04 ms** forward and roughly **3 ms** with gradients. The
environment costs about what the whole agent costs. The performance axis does not decide this.

## Decision

**Build in PyTorch, and enforce the locality guarantee explicitly** rather than inheriting it from the
language. The full mechanism is specified in
[`09-the-build-stack.md`](../spec/09-the-build-stack.md), *The locality guard*; in outline:

1. **The tick carries no tape.** The whole tick runs under `torch.no_grad()` — not as an optimisation,
   but because the tick is a rollout and everything it produces is data the cell then learns *from*.
2. **Learning is a separate phase over detached inputs.** Each cell's update is a function of its own
   adapting surface plus detached arrays, building a fresh small graph that dies at the end of the step.
   A neighbour's parameters are not severed from that graph; they are not in it. Because every coupling
   term enters detached, the cells' graphs still compose into one batched graph with no cross-cell edges,
   so the gradient of the sum is exactly the per-cell local gradient and batching survives. **The phase
   is written as a `torch.func` transform** ([#66](https://github.com/NGL321/patchworks/issues/66)): each
   rule is a pure function, parameters arrive explicitly, and `grad` is scoped by `argnums` to the cell's
   own adapting surface — so the neighbour's restriction map, the phase's only cross-cell parameter, is
   undifferentiated by construction rather than by a remembered `.detach()`.
3. **The guarantee is tested.** An always-on assertion that nothing leaving the tick has a `grad_fn`,
   and a standing **perturbation test**: perturb one cell's parameters, assert no other cell's update
   changes.

The decision being recorded is (1)–(3). PyTorch is the *consequence* — what makes the choice defensible
is that the thesis' central constraint is held by a mechanism and a falsifying test rather than by the
shape of the language.

## Consequences

- **The guarantee weakens from "impossible" to "caught".** This is the real cost, stated plainly. A
  future contributor can delete a `no_grad` decorator; in JAX there would have been nothing to delete.
  The perturbation test is what makes that a caught regression rather than a silent one, and it is
  therefore not optional scaffolding — it is the load-bearing half of this decision. **#66 narrowed this
  cost without removing it.** One leak class — a neighbour's parameter being reachable — is back to
  "impossible" under the transform, and it is the class the no-tape tick could never have covered, since
  parameters live outside the tick. Everything else still reads exactly as above, including the
  shared-storage class the tape cannot see, so the test stays load-bearing.
- **The thesis' central claim becomes falsifiable in CI**, which it was not before. This is a gain JAX
  would not have supplied on its own: a structural guarantee is not a *test*, and nothing about JAX
  would have caught a locality violation introduced deliberately in a rewritten update rule.
- **The ecosystem is available** — `torch.func` supplies `grad` / `vmap` / `jacrev` for the
  regional-Jacobian and effective-rank diagnostics the spec keeps calling for, and debugging into a live
  tick stays ordinary. Since #66 the same library is load-bearing in Decision (2) rather than only in the
  diagnostics, which is a point in favour of the stack this ADR chose: the guard and the instruments are
  the same tool.
- **The environment stays framework-agnostic.** Classic MuJoCo, host loop, numpy at the boundary; the
  sandbox and `03-the-sandbox.md` survive unchanged if this decision is ever reversed.
- **Reversal is bounded.** The tick mathematics is framework-agnostic; what a reversal would have to
  re-argue is this ADR, not the spec's architecture.

## Alternatives considered

- **`torch.func` + `functional_call`** — a PyTorch-native structural guard, **not considered when this
  ADR was written and since adopted** ([#66](https://github.com/NGL321/patchworks/issues/66), from the
  citation pass [#62](https://github.com/NGL321/patchworks/issues/62)). Composable function transforms
  scoped by `argnums` give, inside PyTorch, the property *Context* above attributes to JAX alone. It
  changes how Decision (2) is written and nothing else. Recorded here rather than silently absorbed
  because it corrects this ADR's own framing of the trade-off, and because two limits ride with it: the
  property is inferred from the pure-function contract rather than documented, and no source was found
  using these transforms for locality. One argument for adopting it was raised and **did not survive**:
  that hand-placed detaches would not scale to the graph's 682 edges. They would — the maps are one batched
  parameter store, so the neighbour side is a single gathered `.detach()`, collapsing the same way this
  ADR's own shared-frozen-body argument collapses JAX's `vmap` advantage. What decided it instead was the
  failure direction: a wrong `argnums` yields a *missing* gradient, where a missing `.detach()` yields an
  *extra* one, and this ADR exists because the extra one is silent and flattering.
- **JAX.** Rejected on the balance above, not on its merits — the structural guarantee is real, and it
  remains the one axis where JAX is better, though **by less than this ADR first claimed**: #66 found the
  same property available in PyTorch inside a chosen idiom, so what JAX supplies uniquely is that the
  property holds *globally*, with no ambient tape to step outside of. That is a narrower advantage than
  "straightforwardly better", and it argues for the decision rather than against it. Against it: an
  ecosystem cost paid on every
  remaining section of the build, two of its three technical advantages already dissolved by the shared
  frozen body and the host-side environment, and a performance advantage the measurement does not
  support. The same argument the spec already makes for classic MuJoCo over MJX — prefer the
  better-maintained, more-eyes system where the niche one's advantages do not apply — applies one layer
  up.
- **Convention-only locality in PyTorch** (write the rules locally, review for leaks). Rejected: the
  failure mode is silent *and* flattering, which is exactly the case where review is worth least.
- **MJX**, and with it an on-device environment. Rejected in
  [`09-the-build-stack.md`](../spec/09-the-build-stack.md): built for parallel rollouts this design has
  no use for, a contact model that differs on exactly the contact-rich pushing the sandbox is built
  around, no native path for the 64×64 render, and far narrower maintenance than classic MuJoCo.
- **A GPU-first compute target.** Deferred, not rejected: CPU is the declared target and the first build
  measures tick wall-time, with a rented NVIDIA box as an escape hatch gated on that measurement. The
  local AMD RX 580 is Polaris (`gfx803`) and is not a target for any current ROCm build.
