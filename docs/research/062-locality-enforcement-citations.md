# Citation pass: enforcing locality inside an autograd framework (patchworks#62)

Validates the design closed in [#12](https://github.com/NGL321/patchworks/issues/12)
(`docs/spec/09-the-build-stack.md`, *The locality guard*;
[`ADR-0011`](../adr/0011-the-locality-guarantee-is-enforced-not-inherited.md)).
Citations validate after the fact per the map's sequencing rule; this document does not reopen the
closed design — it records where a source confirms a claim and flags where one threatens it.
Vocabulary follows [`CONTEXT.md`](../../CONTEXT.md): Patchworks' side of every comparison in its own
terms, the prior art's in its field's. Where a source could not be reached, that is stated.

**Out of bounds, deliberately:** the framework choice itself. Nothing below is a JAX-versus-PyTorch
argument; where JAX appears it is because a published local-learning implementation is written in it,
and the fact recorded is what that implementation *does*, not which stack is better.

## Headline verdict, stated plainly

**#12's guard is the mainstream pattern of the field, executed correctly, and its central mechanical
claim is not merely standard but is the identity two production PyTorch libraries are built on. Two
things it says are unavailable are in fact available, and one is a genuine gap in the ADR's
alternatives.** In detail:

- **The guard is standard.** Every PyTorch local-learning implementation read for this pass keeps
  updates local by exactly #12's two moves: cut the graph at the interface with `.detach()` (or never
  build one), then optimise each part against its own objective with its own optimiser. Forward-forward,
  local-error-signal training, Torch2PC and equilibrium propagation all do this, verbatim.
- **A quarter of the field does not use autograd at all.** Millidge/Tschantz/Buckley's predictive-coding
  code hand-derives every gradient — no `.detach()`, no `requires_grad`, no tape. Whittington & Bogacz's
  reference implementation is MATLAB. Where locality is the whole point, leaving the framework is a
  live and used option. #12 does not consider it; nor should it, at this graph's size — but it is the
  reason the field looks less disciplined about detachment than it is.
- **There is a PyTorch-native structural guard, and ADR-0011 does not consider it.** `torch.func` +
  `functional_call` are official, documented, JAX-like composable transforms over *pure* functions with
  parameters passed in explicitly; `grad` differentiates only with respect to the arguments `argnums`
  names. That is the structural property ADR-0011 attributes to JAX alone, available in the chosen
  stack. **This warrants a revision ticket against ADR-0011's *Alternatives considered* — see below —
  and nothing more.** No source was found using `torch.func` for this purpose, so it is an untested
  option, not a better one.
- **Nobody tests locality. Not one repository surveyed.** It is asserted by construction everywhere.
  The perturbation test therefore has **no precedent in local learning** — but its *shape* is standard
  practice two doors down: Karpathy's recipe recommends exactly it for batch-mixing bugs, and Pyro
  ships a CI test that computes a Jacobian and asserts the entries that must be zero are zero. There is
  no tested code to port; there is a test idiom to copy.
- **The batched-graph equivalence argument is standard, and its documented failure mode is one
  Patchworks is exposed to.** The identity is what energy-based predictive coding is built from, and
  Opacus and `torch.func`'s per-sample-gradient tutorial exist *because of the case where it fails*:
  it fails when the batched items share trainable parameters. Patchworks batches cells over a shared
  body — and is saved only by that body being frozen ([ADR-0001](../adr/0001-continual-learning-applies-to-the-adapting-surface.md)).
  The claim in `09-the-build-stack.md` is true and the dependency is unstated. See §4.
- **One structural difference from all the precedent, flagged.** Every implementation surveyed is a
  feedforward chain, where "local" means "layer" and there is exactly one place per layer to cut. The
  sheaf is a connected undirected graph: the number of cut sites scales with **edges**, not depth, and
  every cell has many neighbours rather than one predecessor. No source addresses that regime. The
  discipline #12 accepts is the same discipline the field accepts, applied to a strictly larger
  surface.

---

## 1. How existing local-learning work prevents accidental global gradient flow

Four distinct strategies were found. Patchworks' guard is a combination of the second and third.

### (a) Do not use autograd at all — gradients hand-derived

*Source: Millidge, Tschantz & Buckley, `BerenMillidge/PredictiveCodingBackprop`, `cnn.py`, read at
source on GitHub.*

The reference implementation for "Predictive coding approximates backprop along arbitrary computation
graphs" contains **no `.detach()` and no `requires_grad` anywhere**. Autograd is not used. Prediction
errors are computed by hand (`self.prediction_errors[-1] = -self.loss_fn_deriv(self.outs[-1], self.mus[-1])`),
the inference loop updates each layer's estimate directly (`self.mus[j] -= self.inference_learning_rate * (2*dx_l)`),
and each layer's parameters are updated by a hand-written method taking only that layer's error
(`l.update_weights(self.prediction_errors[i+1], update_weights=True)`). `torch.no_grad()` appears only
around the plain forward pass. Whittington & Bogacz's reference code (`djcrw/Supervised-Predictive-Coding`)
is MATLAB, i.e. the same posture in a language with no tape to guard against.

**Relevance to #12:** this is the strongest possible version of "the tick carries no tape" — there is
no tape in the whole program. It is not an option here, because Patchworks' bias rule is specified as
*a local gradient step through the cell's own frozen forward path* (`07-local-learning-rule.md`) rather
than a closed-form Hebbian product, and hand-deriving through the shared body is exactly the work
autograd exists to avoid. Recorded as context, not as a candidate.

### (b) Detach at the interface, one optimiser per part

*Sources: `mohammadpz/pytorch_forward_forward` (`main.py`); Nøkland & Eidnes (2019), `anokland/local-loss`
(`train.py`). Both read at source.*

The forward-forward reference implementation trains layer by layer and hands the next layer a severed
tensor:

> `return self.forward(x_pos).detach(), self.forward(x_neg).detach()`

with each layer owning its optimiser (`self.opt = Adam(self.parameters(), lr=0.03)`) and the network
loop doing nothing but threading detached activations forward (`h_pos, h_neg = layer.train(h_pos, h_neg)`).
This is Hinton's stated design: in arXiv:2212.13345 each layer has its own objective function, and the
point of the algorithm is that there is "no need to backpropagate through" what precedes it.

Nøkland & Eidnes' local-error-signal training is the same pattern with the cut made explicit and
*optional*: each block runs `self.optimizer.zero_grad()`, `loss.backward(retain_graph = args.no_detach)`,
`self.optimizer.step()`, then `h_return.detach_()`. The `--no_detach` flag is the notable part —
locality is a runtime condition their experiments can switch off and compare against, which is the
nearest thing in the surveyed code to treating locality as a property to be *checked* rather than
assumed. It is an ablation, not a test; it measures the difference in accuracy, it does not assert
there is none in the gradient.

**Relevance to #12:** this is part 2 of the guard, at layer granularity. Confirmed as the field's
default.

### (c) Detach, then re-enter deliberately with a targeted `autograd.grad`

*Sources: Rosenbaum, `RobertRosenbaum/Torch2PC` (`TorchSeq2PC.py`); Laborieux et al. (2021),
`Laborieux-Axel/Equilibrium-Propagation` (`model_utils.py`). Both read at source.*

Torch2PC seeds each layer's state from a severed copy —

> `v[layer]=vhat[layer].clone().detach()`

— keeps the whole state-update loop inside `with torch.no_grad():`, then re-enables differentiation
only where it wants it (`vtemp0.requires_grad=True`) and pulls a single layer's parameter gradient out
by naming both the scalar and the parameter:

> `dtheta=torch.autograd.grad(vtemp1,p,grad_outputs=epsilon[layer+1],allow_unused=True,retain_graph=True)[0]`
> `p.grad = dtheta`

Cross-layer error terms enter as `grad_outputs`, i.e. as **detached constants** — the same move
`09-the-build-stack.md` makes with per-edge disagreement. It also uses `torch.autograd.functional.vjp`
against a single layer's module to get the error propagation term, which is a narrower entry point than
a global `.backward()`.

Laborieux et al.'s equilibrium propagation does the same at the phase boundary: before relaxation,

> `neurons[idx] = neurons[idx].detach()` / `neurons[idx].requires_grad = True`

so the relaxation dynamics are outside the parameter graph, and the weight update comes from a
finite difference of two settled energies (`delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)`, then
`delta_phi.backward()`), never from differentiating through the relaxation. The neuron dynamics
themselves come from `torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=check_thm)`
— `create_graph` is off except when a theorem check is running.

**Relevance to #12:** this is parts 1 and 2 together — a no-tape settling phase, then a separate
learning step over detached state — and it is the closest published analogue to Patchworks' two-phase
tick. Confirmed.

### (d) Move to a functional AD system

*Source: Pinchetti, Salvatori et al. (2024), "Benchmarking Predictive Coding Networks — Made Simple",
arXiv:2407.01163; `liukidar/pcx`.*

PCX, the field's current benchmark library, is JAX. The paper's locality statement is a property of
the *objective*, not the framework — "each state and each parameter is updated using local information
as the gradients depend exclusively on the pre and post-synaptic errors" — and its framework discussion
is about speed, not about guarantees; it reports JIT and `vmap` limitations rather than a locality
argument. **No source was found arguing that a functional AD system is necessary for locality**, which
is worth recording precisely because ADR-0011 treats that asymmetry as the decisive axis. The asymmetry
is real and correctly described; it is simply not something the literature has bothered to state.

### The gap the precedent leaves

Every implementation in (a)–(d) is a **feedforward chain**. "Local" means "one layer", the cut site is
the single tensor handed to the successor, and the failure mode is one missing `.detach()` in one
place. Patchworks' sheaf is a connected undirected graph of ~150 predicting cells and ~698 edges
(`06-graph-topology.md`), so the coupling terms that must enter detached are per **edge**, and a cell's
neighbours are many rather than one. Nothing found addresses locality enforcement at that
granularity — including the sheaf/GNN literature, which trains restriction maps end-to-end against a
global objective and therefore has no locality to protect (see Honest gaps). **The guard is the field's
standard guard applied to a surface the field has not tried it on.** That is an argument for #12's
insistence that the test is load-bearing, not against the guard.

---

## 2. Is there a stronger guard?

### `torch.func` + `functional_call` — yes, structurally, and the ADR does not consider it

*Sources: PyTorch 2.2 docs, `torch.func` whirlwind tour, `func.api`, `func.ux_limitations`; the
per-sample-gradients tutorial. All read at source.*

`torch.func` is documented as "a library for JAX-like composable function transforms in PyTorch". The
relevant properties are all official:

- `grad` "helps computing gradients of `func` with respect to the input(s) specified by `argnums`" —
  differentiation is scoped by what is *named*, not by what is reachable.
- `functional_call` "[p]erforms a functional call on the module by replacing the module parameters and
  buffers with the provided ones" — parameters arrive as an explicit dict argument rather than as
  ambient module state.
- The transforms "work well with pure functions", and functions under them must return their outputs
  explicitly rather than assign to globals.

Composed, those give: a cell's update written as `grad(local_energy)(this_cell_params, detached_stuff)`,
where a neighbour's parameters are not an argument and are therefore not differentiable — the exact
sentence ADR-0011 uses for JAX ("differentiation only ever traverses what was passed in"), inside
PyTorch. ADR-0011 names `torch.func` only under *Consequences*, for the regional-Jacobian and
effective-rank diagnostics, and frames the structural guarantee as something the stack choice gave up.
**That is the finding: an alternative that was not considered, not a decision that was wrong.**

Two honest limits on it, both from the docs:

- **The structural guarantee is inferred, not documented.** No PyTorch page states "transforms cannot
  differentiate through parameters not passed as arguments." It follows from the pure-function contract
  and `argnums`, and it is how the per-sample-gradient tutorial is written
  (`params = {k: v.detach() for k, v in model.named_parameters()}`, then `grad(compute_loss)`), but this
  pass could not find it asserted at source.
- **The transforms constrain the code around them.** `func.ux_limitations` states that a `torch.autograd`
  API "like `torch.autograd.grad` or `torch.autograd.backward` inside of a function being transformed"
  may not be transformable, and that `vmap` "will raise an error if it encounters an unsupported PyTorch
  in-place operation." Reconciliation edits the node stalk in place (`02-tick-semantics.md`); that is
  outside the learning phase and so probably untouched, but it is not free.

### The other structural options, and why they are not stronger

- **`autograd.Function` boundaries.** The contract is documented — `backward` "should return as many
  tensors as there were inputs", and "you can return `None`" for inputs that need no gradient — so a
  custom Function *can* hard-cut a path. But it is one hand-written boundary per site, i.e. the same
  discipline as `.detach()` with more code, and the docs steer away from it: for altering gradients they
  say to "consider registering a tensor or Module hook" instead. **Not stronger.**
- **Parameter-group graph partitioning.** No published mechanism found. `requires_grad` is the
  documented control ("Setting `requires_grad` should be the main way you control which parts of the
  model are part of the gradient computation"), and it is per-tensor state, not a partition — the same
  discipline again.
- **A static check.** **None found**, in PyTorch or anywhere. Nothing in the tooling can tell you at
  import time that an edge stalk was not detached.

**Verdict on the ticket's question:** one genuinely stronger guard exists and is native to the chosen
stack; it is untested for this purpose, and adopting it is a decision for a revision ticket, not for
this document.

---

## 3. Has anyone tested it, and how?

**No.** Not in any repository read for this pass — `PredictiveCodingBackprop`, `Torch2PC`,
`pytorch_forward_forward`, `local-loss`, `Equilibrium-Propagation`, `pcx`. Code search across those
repositories for locality assertions returned nothing; `pcx`'s test suite mentions detachment only in
the sense of parameter/module bookkeeping, not gradient reachability. Laborieux et al.'s `check_thm`
path is the nearest thing, and it checks that EP's update *approximates BPTT* — the opposite question,
correctness of the estimator rather than isolation of it. Nøkland & Eidnes' `--no_detach` is an
accuracy ablation.

**Porting is therefore not available. The idiom is.** The perturbation test's shape — assert a
dependency that must not exist is numerically absent — is established practice in two adjacent places,
both reached at source:

- **Karpathy, "A Recipe for Training Neural Networks" (2019)**, on inadvertent batch mixing:
  > "One way to debug this (and other related problems) is to set the loss to be something trivial like
  > the sum of all outputs of example **i**, run the backward pass all the way to the input, and ensure
  > that you get a non-zero gradient only on the **i-th** input."

  That is #12's perturbation test with the perturbation replaced by a gradient — same claim, one
  backward pass instead of two forward re-runs, and it localises the leak rather than merely detecting
  it.
- **Pyro, `tests/distributions/test_transforms.py`**, on autoregressive structure — a CI test that
  computes the Jacobian and asserts the entries that must vanish do:
  > `lower_sum = torch.sum(torch.tril(nonzero(jacobian), diagonal=-1))`
  > `assert lower_sum == float(0.0)`

  A mainstream library treating a structural independence property as something to be falsified
  numerically on every run, rather than reviewed for.

**Verdict:** the perturbation test is, as far as this pass could establish, **without precedent in
local-learning code**, and #12's judgement that it is the load-bearing half of the decision is
uncorroborated by anyone having felt the same need. Its *form* is not novel at all. The one substantive
finding is that the gradient form of the test (Karpathy's, Pyro's) is strictly cheaper and more
diagnostic than the perturbation form for the part of the claim that concerns the tape — and strictly
weaker for the part that does not, since a gradient test cannot see a leak that travels through shared
storage rather than through `grad_fn` (§4).

---

## 4. Is the batched-graph equivalence argument standard or novel?

**Standard, twice over, and with a documented failure mode that matters here.**

The claim in `09-the-build-stack.md` is: coupling terms enter detached ⇒ the per-cell graphs compose
with no cross-cell edges ⇒ the gradient of the sum over cells is exactly the per-cell local gradient.

**Confirmed as the constitutive fact of energy-based predictive coding.** PCX states it as the reason
PC is local at all — "each state and each parameter is updated using local information as the gradients
depend exclusively on the pre and post-synaptic errors", with the parameter gradient written
`∇θₗ = ½ ∂εₗ²/∂θₗ` (arXiv:2407.01163, §3). The total energy is a sum of per-layer terms and each
parameter appears in exactly one; differentiating the sum yields the per-layer gradient. Torch2PC uses
the same identity operationally, assigning `p.grad` layer by layer from a scalar that only that layer's
error touches.

**Confirmed, from the other direction, by the machinery built for the case where it fails.** The
PyTorch per-sample-gradients tutorial exists because a summed batch loss gives "an 'average' gradient
of the entire mini-batch" rather than per-item gradients, and Opacus exists to recover the per-item
gradients efficiently. The separating condition is precisely **whether the batched items share
trainable parameters**. Where they do not, the summed gradient *is* the stack of per-item gradients and
nothing special is needed; where they do, it is not, and you need `vmap(grad(...))` or microbatching.

**The flag.** Patchworks batches predicting cells over a **shared** body. The argument survives only
because that body is *frozen* — the adapting surface is per-cell biases and per-cell restriction maps
([ADR-0001](../adr/0001-continual-learning-applies-to-the-adapting-surface.md)), so no trainable
parameter is shared across the batched dimension and the equivalence holds exactly. `09-the-build-stack.md`
states the conclusion without stating that dependency. It is not wrong; it is one decision away from
being wrong, and the decision it depends on lives in a different ADR. Low-priority revision ticket
below.

**A second documented failure mode, and one the `grad_fn` assertion cannot see.** `Tensor.detach`'s
own documentation:

> "Returned Tensor shares the same storage with the original one. In-place modifications on either of
> them will be seen, and may trigger errors in correctness checks."

Detachment severs the *gradient* path, not the *data* path. A cell writing in place into a detached
view still couples cells, and neither of #12's assertions catches it by construction — the tensor has
no `grad_fn` and the batched graph genuinely has no cross-cell edges. Only the perturbation test would
see it, because only the perturbation test observes the update rather than the graph. This is direct
support for ADR-0011's *Consequences* section: it is a concrete leak class that the cheap assertion
misses and the expensive test catches, which is the ADR's own argument for why the test is not optional
scaffolding.

---

## Revision tickets recommended (recommendations only — not created)

1. **ADR-0011, *Alternatives considered* — `torch.func` + `functional_call` is a PyTorch-native
   structural guard and is not listed.** The ADR's framing is that the structural guarantee was the one
   axis on which the rejected candidate was straightforwardly better; §2 finds that the same
   property — differentiation scoped to explicitly named arguments of a pure function — is available in
   the chosen stack, from an official, documented API the ADR already cites for a different purpose. The
   ticket should record it as an alternative that was not considered and decide whether the learning
   phase adopts it, noting that (a) no source was found using it for locality, (b) the guarantee is
   inferred from the pure-function contract rather than asserted in the docs, and (c) `func.ux_limitations`
   constrains in-place operations and nested `torch.autograd` calls inside transformed functions. **Not
   a reason to reverse anything** — the ADR's decision is (1)–(3), and this changes at most how (2) is
   written.
2. **Low priority — `09-the-build-stack.md`, *The locality guard* §2, "This still batches": state the
   dependency.** The batched-sum equivalence holds because no *trainable* parameter is shared across the
   batched cell dimension, which is true only while the body is frozen (ADR-0001). Opacus and
   `torch.func`'s per-sample-gradient machinery are the field's evidence that this is the exact condition
   that separates the easy case from the hard one. One clause, cross-referencing ADR-0001.
3. **Low priority — `09-the-build-stack.md` §3: note that the two checks cover different leak classes.**
   The `grad_fn` assertion tests the tape; the perturbation test tests the update, and `Tensor.detach`'s
   shared-storage note is a documented way for cells to couple with a clean tape. Worth one sentence,
   because it sharpens the ADR's existing claim that the perturbation test is load-bearing.

**Not recommended:** adding a Karpathy-style gradient locality test *in place of* the perturbation
test. It is cheaper and more diagnostic for tape leaks, and blind to the storage-aliasing class above.
If both are wanted it is an addition, not a substitution — and that is a build decision, not a finding.

## Honest gaps

- **No sheaf/GNN implementation was found that trains per-node parameters without a global
  objective.** Neural sheaf diffusion (Bodnar et al., arXiv:2202.04579) and the sheaf-GNN work
  downstream of it learn restriction maps end-to-end against a task loss, so there is no locality
  constraint in that code to inspect. Searched and not found; the ticket's expectation that such code
  exists is, on this pass's evidence, not met.
- **Target propagation and difference target propagation were not reached in code.** Named in the
  ticket; four families were read instead and the pattern was already saturating. Ernoult et al.'s
  scaled DTP implementation is the obvious next stop and could sharpen §1(c).
- **Repository reads were partial.** Each repository was read at one or two central files
  (`TorchSeq2PC.py`, `model_utils.py`, `train.py`, `main.py`, `cnn.py`), not swept. The absence of a
  locality test in §3 rests on those reads plus code search across the same repositories; it is a
  strong negative, not an exhaustive one.
- **`func.api` at the pinned-version URL returned a redirect stub** on first fetch; the signatures
  quoted in §2 come from the 2.2 documentation reached on the second attempt, and the whirlwind tour
  did not contain an explicit `torch.func` vs `torch.autograd` comparison to quote.
- **Whittington & Bogacz (2017) was verified only to the level of its reference implementation's
  language** (MATLAB) via the repository; the paper text was not reached, and no claim here turns on it.

## Sources

- Hinton (2022). The forward-forward algorithm: some preliminary investigations. arXiv:2212.13345.
- Millidge, Tschantz & Buckley (2020). Predictive coding approximates backprop along arbitrary
  computation graphs. arXiv:2006.04182. Code: `github.com/BerenMillidge/PredictiveCodingBackprop`.
- Rosenbaum (2022). On the relationship between predictive coding and backpropagation. *PLOS ONE*
  17(3): e0266102; arXiv:2106.13082. Code: `github.com/RobertRosenbaum/Torch2PC`.
- Whittington & Bogacz (2017). An approximation of the error backpropagation algorithm in a predictive
  coding network with local Hebbian synaptic plasticity. *Neural Computation* 29(5), 1229–1262. Code:
  `github.com/djcrw/Supervised-Predictive-Coding` (MATLAB).
- Laborieux, Ernoult, Scellier, Bengio, Grollier & Querlioz (2021). Scaling equilibrium propagation to
  deep ConvNets by drastically reducing its gradient estimator bias. *Frontiers in Neuroscience* 15.
  Code: `github.com/Laborieux-Axel/Equilibrium-Propagation`.
- Nøkland & Eidnes (2019). Training neural networks with local error signals. ICML; arXiv:1901.06656.
  Code: `github.com/anokland/local-loss`.
- Pinchetti, Qi, Lokshyn, Salvatori et al. (2024). Benchmarking predictive coding networks — made
  simple. arXiv:2407.01163. Code: `github.com/liukidar/pcx`.
- `mohammadpz/pytorch_forward_forward` — the reference PyTorch forward-forward implementation.
- PyTorch 2.2 documentation: *Autograd mechanics*; `torch.Tensor.detach`; *Extending PyTorch*;
  `torch.func` whirlwind tour, API reference, and UX limitations; *Per-sample-gradients* tutorial.
- Opacus, `opacus/utils/per_sample_gradients_utils.py` — `compute_microbatch_grad_sample`,
  `check_per_sample_gradients_are_correct`.
- Pyro, `tests/distributions/test_transforms.py` — `_test_jacobian`.
- Karpathy (2019). A recipe for training neural networks.
- Bodnar, Di Giovanni, Chamberlain, Liò & Bronstein (2022). Neural sheaf diffusion. NeurIPS;
  arXiv:2202.04579. (Negative result only — see Honest gaps.)
