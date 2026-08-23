# Citation pass: the local learning rule (patchworks#34)

Validates the design closed in patchworks#5 (`docs/spec/07-local-learning-rule.md`, ADR-0008),
against the four candidates named in #34's body. Citations validate after the fact per the map's
Notes; this does not revise the closed design. Vocabulary follows `CONTEXT.md`: Patchworks' side
of every comparison is described in its own terms (bias rule, transport rule, prediction error,
disagreement); prior art's side is described in its own field's terms.

## 1. Predictive coding as local gradient descent on prediction error

**Sources:**
- Rao, R.P.N. & Ballard, D.H. (1999). "Predictive coding in the visual cortex: a functional
  interpretation of some extra-classical receptive-field effects." *Nature Neuroscience* 2(1),
  79–87.
- Whittington, J.C.R. & Bogacz, R. (2017). "An approximation of the error backpropagation
  algorithm in a predictive coding network with local Hebbian synaptic plasticity." *Neural
  Computation* 29(5), 1229–1262.
- Millidge, B., Tschantz, A., & Buckley, C.L. (2022). "Predictive coding approximates backprop
  along arbitrary computation graphs." *Neural Computation* 34(6), 1329–1368. arXiv:2006.04182.

Already read in detail for #19 (`docs/research/019-tick-semantics-citations.md`): Rao & Ballard's
representation vector `r` is estimated by literal gradient descent on a combined prediction-error
objective `E` (their Eq. 7), and the same objective's gradient trains the weights `U`. That
finding transfers directly to this ticket's first question: **the bias rule's "local gradient
step through the cell's own frozen forward path, on prediction error" is exactly Rao & Ballard's
mechanism, restricted to the parameters Patchworks lets vary.** Rao & Ballard do not stop the
gradient at a subset of parameters the way the bias rule stops it at biases — in their model `U`
(the generative weights) is also learned by the same gradient — but the *shape* of the update
(gradient descent through a fixed forward computation, driven by the prediction-vs-observation
difference) is the same mechanism ADR-0008 names.

Whittington & Bogacz's result is more specific than "predictive coding approximates backprop" as
loosely stated in the ticket, and the precise conditions matter for this design:

- Exact equivalence requires the **"fixed prediction assumption"** — during the inference
  sub-loop, each layer's top-down prediction is held fixed at its feedforward-pass value rather
  than being updated as the layer below relaxes — and requires the **inference variables to reach
  a fixed point** (`dvℓ = 0`) before the weight update is taken (their Theorem 1; confirmed
  independently via Song et al.'s follow-up showing PC approximates backprop's gradient exactly
  on the network's *first* inference step, and Millidge et al.'s generalization of the same
  fixed-prediction result to arbitrary computation graphs).
- **This is a real, precisely-locatable divergence from Patchworks, not a loose one.** The
  equivalence-to-backprop literature's standard formulation relaxes over several inference
  iterations *per weight update*; ADR-0002 already rejected exactly this "iterate to convergence"
  pattern for reconciliation, and the bias rule inherits the same one-step-per-tick discipline —
  it takes one gradient step through the frozen forward path per tick, never an inner relaxation
  loop. So the bias rule is a **single-step instance** of the predictive-coding-as-gradient-
  descent family, not the literature's standard multi-iteration relaxation that Whittington &
  Bogacz's exact-equivalence theorem is proved for. Song et al.'s single-inference-step result is
  the closest match to what the bias rule actually does, and it is presented in that literature as
  an *approximation* to the converged case, not as the primary object of study. This does not
  contradict ADR-0008 — Patchworks never claimed exact equivalence to backprop, only that the
  update is "no different in kind from ordinary backprop-to-input" — but the ticket's own framing
  ("does the bias rule... match how predictive-coding networks actually train") should be read as
  "matches the mechanism family, not the specific converged-equivalence result."
- Nothing in this literature bears directly on the stability question carried into #33. The
  predictive-coding-as-backprop papers analyze convergence of the *inference* relaxation within a
  static set of weights; they do not analyze what happens when the weights themselves are also
  being updated continually and simultaneously across every unit, which is #33's open question,
  not this one's.

## 2. Two local signals training two parameter groups — precedent or novel combination

No single source was found using Patchworks' exact split (a cell-owned, temporal signal for one
parameter group; an edge-owned, spatial signal for another). Two real precedents share its
*shape* — same-architecture, two parameter groups, two different local training signals — but
neither shares its *axis of division*, which supports flagging the combination as novel while the
underlying pattern (split-by-parameter-group with distinct local objectives) is well established.

**Source:** Hinton, G.E., Dayan, P., Frey, B.J., & Neal, R.M. (1995). "The 'wake-sleep' algorithm
for unsupervised neural networks." *Science* 268(5214), 1158–1161.

The wake-sleep algorithm holds two weight sets between every pair of layers — **recognition**
(bottom-up) weights and **generative** (top-down) weights — and trains them with two different
local rules on two different phases: in the "wake" phase, units are driven bottom-up by the
recognition weights and the *generative* weights are adapted, by a purely local delta rule, to
better reconstruct the layer below; in the "sleep" phase, units are driven top-down by the
generative weights (fantasizing data) and the *recognition* weights are adapted, again by a local
rule using only locally available information, to better reconstruct the layer above. This is a
genuine precedent for "two parameter groups, two local objectives, neither reading a global error
signal" — the same *shape* of move ADR-0008 makes. It differs from Patchworks on the axis of
split (feedforward-direction vs. feedback-direction weights, not temporal-error vs.
spatial-disagreement) and on timing (two alternating global phases per training cycle, not one
simultaneous update per tick within a single running system) — Patchworks explicitly rejected an
analogous phase split for message-passing (ADR-0002) and the local rule inherits that same
same-tick, no-phases discipline.

**Source:** Ororbia, A. & Kifer, D. (2022) as surveyed alongside Salvatori, T. et al.,
"Predictive coding, precision and natural gradients" (arXiv:2111.06942). Hierarchical predictive
coding networks with **learnable precision** train two parameter classes locally at each layer —
representation-mapping weights and precision (inverse-variance) weights — each with its own
layer-local update derived from the same free-energy functional, described as inducing a local
Fisher-metric preconditioning. This is closer in spirit to a "two parameter groups, two rules"
split inside one predictive-coding layer, but the two objectives here are not qualitatively
different signals in Patchworks' sense — both are gradients of the *same* energy functional with
respect to different arguments, not two structurally distinct local quantities (one temporal, one
spatial) the way prediction error and disagreement are.

**Assessment for the ticket's question:** the pattern "split parameter groups, split local
objectives" is precedented (wake-sleep is the clearest case). The **specific combination** —
cell-owned temporal prediction error training inference parameters, edge-owned spatial
disagreement training transport parameters, both updated simultaneously every tick with no
alternating phase — was not found anywhere in the literature searched. This should be read, per
the ticket's own framing, as a **novel combination of precedented parts**, not as recovered prior
art. Worth stating explicitly in the spec's provenance if that matters to the project (it does not
currently claim otherwise).

## 3. Infomorphic networks, read against an actual design

**Source:** Makkeh, A., Graetz, M., Schneider, A.C., Ehrlich, D.A., Priesemann, V., & Wibral, M.
(2025). "A general framework for interpretable neural learning based on local information-
theoretic goal functions." *PNAS* 122(10). Also arXiv:2306.02149 (earlier preprint title:
"Infomorphic networks: locally learning neural networks derived from partial information
decomposition"). Read directly from the PNAS-published PDF (sections 2–4), not a secondary
description.

Confirmed structure: each infomorphic neuron has **two input classes**, receptive (`X_R`) and
contextual (`X_C`), combined through separate weight vectors `w_R` and `w_C` into an information-
theoretic goal function `G` built from a Partial Information Decomposition of the neuron's output
entropy (their Eq. 5, 7). Both weight vectors are updated by gradient ascent on `G` (their Eq. 8,
9) using only a locally estimated joint probability distribution over the neuron's own inputs and
output. No global loss, no backpropagated signal, and weight updates are computed from local
minibatch statistics only.

**Where the earlier read (in #5, from #15) holds up:** it remains the closest published thing to
"a per-unit local learning rule with no global error signal, applicable across supervised,
unsupervised and memory tasks," and the earlier disqualification of information cohomology as the
*mathematical vehicle* for the rule is unaffected by this closer read — infomorphic networks do
not use cohomology; they use a PID-based goal function optimized by ordinary gradient ascent,
which is a different and unrelated mathematical object.

**Where a closer read, now that #5 has committed to a design, changes the comparison — two points
worth flagging:**

- **Infomorphic networks are not an instance of "two rules, two signals, two objectives."** Both
  `w_R` and `w_C` are trained by gradient ascent on the *same* goal function `G` (Eq. 8 and 9 are
  the same `G`, differentiated with respect to different weight vectors) — the split is by
  *input class*, not by *objective*. ADR-0008's design is a stronger claim: two distinct
  objectives (prediction error vs. disagreement) training two distinct parameter groups. This is a
  genuine structural difference from the paper that was flagged as "the closest existing thing,"
  not a confirmation of it — the earlier framing in #5's first comment was accurate as an initial
  read but understated this distinction, which only becomes visible once there is a design (two
  rules, two objectives) to compare against.
- **The contextual input `X_C` is, in two of the paper's three worked examples (unsupervised
  feature learning, associative memory), literally the raw output activity of every other neuron
  in the network at the previous time step** ("each neuron... receives the activity of all other
  neurons in the previous time step as contextual input," §4.2; `X_C ∈ {-1,1}^99` sourced directly
  from neighbour activity, §4.3). This is the opposite of the transport rule's explicit locality
  commitment — "it never reads a neighbour's raw node stalk" (`07-local-learning-rule.md`) — on
  exactly the point the spec calls out as load-bearing ("a raw neighbour stalk is in the wrong
  space until the map has done that work"). Infomorphic networks solve a different problem (their
  neurons don't have per-neuron restriction maps changing basis before comparison) and this is not
  a criticism of ADR-0008's choice, but it means the paper is a weaker match on the locality
  question specifically than "closest existing thing" might suggest — it is closest on
  *mechanism* (local, information-theoretic, no global error) and diverges on *what counts as
  local* (all-to-all raw context vs. Patchworks' strictly-mediated-by-restriction-maps context).

Neither point rises to a contradiction of ADR-0008's own decision — the design already explicitly
declines to read a neighbour's raw stalk, for a stated reason, and infomorphic networks were never
adopted as a template, only read for orientation. They sharpen, rather than undermine, the earlier
"closest existing thing, not a template" framing from #5/#15.

## 4. Composing a sparsity penalty into one gradient step vs. alternating pruning

**Sources:**
- Ziyin, L. & Wang, Z. (2023). "*spred*: solving L1 penalty with SGD." *ICML 2023* (PMLR 202).
- Evci, U., Pedregosa, F., Gomez, A.N., & Elsen, E. (2019). "The difficulty of training sparse
  neural networks." arXiv:1906.10732.
- Zhu, M. & Gupta, S. (2018). "To prune, or not to prune: exploring the efficacy of pruning for
  model compression." arXiv:1710.01878 (gradual/iterative magnitude pruning, the standard
  alternating-pass baseline this question is implicitly comparing against).

This is the one area with a documented, specific failure mode, read directly from a primary
source rather than inferred: **plain gradient descent does not reliably solve an L1-type sparsity
penalty composed into the same step as a task loss, because the penalty is non-differentiable
exactly at the point sparsity is achieved.** Ziyin & Wang state this plainly: "gradient descent
has yet to be shown to work well in solving the L1 penalty because the L1 penalty is not
differentiable at zero, precisely where the model becomes sparse... optimizing a general nonconvex
objective with L1 regularization remains an important open problem." Their fix (a redundant
reparameterization making the composite objective smooth) is offered precisely because naive
joint-step subgradient descent is known not to reach exact zeros reliably — this is why the
mainstream pruning literature almost never solves sparsity by joint gradient descent alone, and
instead uses either **proximal/soft-thresholding operators** (which explicitly separate the
penalty's non-smooth part into its own operation, an alternating step in all but name) or
**explicit alternating magnitude-pruning passes** (Zhu & Gupta's gradual pruning schedule, the
canonical modern baseline) that never ask a single gradient step to do the thresholding.

Evci et al. give a second, independent line of evidence for the same conclusion from a different
angle: training *directly in the sparse subspace from initialization* (the joint-optimization
regime, structurally closest to composing a sparsity term into every step from the start) find
that the optimization landscape has no monotonically-decreasing path between poor and good sparse
solutions *within* the sparse subspace — a path exists only if the optimizer is allowed to
temporarily leave the sparse subspace (traverse the dense subspace) — which is direct evidence
that confining optimization to (or pressuring it toward) a sparse solution set from the start, in
one continuous process, can trap it in reachable-but-bad regions that iterative pruning (dense
training, then sparsify, sometimes with regrowth) avoids by construction.

**This is a real finding that bears on ADR-0008 and should be weighed, though it does not by
itself demand a revision:**

- The transport rule composes sparsity pressure as "one additive term inside one descent step"
  training restriction maps together with disagreement (`07-local-learning-rule.md`). If that
  additive term is (or resembles) an L1-type penalty, the cited failure mode applies directly:
  plain subgradient descent on it is documented not to reliably reach exact sparsity, which is the
  entire point of the sparsity pressure per `06-graph-topology.md` ("sparsity is a property of the
  maps... annealing is a pressure schedule").
- Two things soften this for Patchworks specifically, and are why this is flagged rather than
  filed as an outright contradiction: (1) `06-graph-topology.md` already specifies sparsity as
  zeroing **within a fixed structural mask**, not as driving weights to exact zero from an
  unconstrained start — the mask already does the hard discrete part the L1-penalty literature
  struggles with, and the penalty's job is softer (pressure within an already-sparse-shaped
  space) than the literature's harder problem (finding sparsity from a dense start); (2) the
  design never specified the penalty's exact form (L1 vs. L2 vs. something else), so whether the
  cited non-differentiability applies depends on a choice not yet made.
- What the citation pass does establish cleanly: **the general claim "composing a sparsity penalty
  into one gradient step, rather than alternating with a separate pruning pass, has no known
  failure modes" would be false.** The failure mode is real, specific, and comes from a primary
  source (Ziyin & Wang) explicitly building a method to work around it. Whether it binds *this*
  design depends on the still-unspecified penalty form and on how much the fixed mask already
  does the discrete work — a build-time question, not obviously a design-time one, but worth a
  named flag rather than silence.

## Assessment: does anything warrant a revision ticket against #5?

Three findings from this pass, weighed together:

1. **Predictive coding correspondence (item 1) is solid** — no revision warranted. The one
   nuance (single-step vs. converged-equilibrium equivalence to backprop) sharpens the spec's own
   already-careful "no different in kind... no different in cost" phrasing rather than
   contradicting it.
2. **The two-rules-two-signals split (item 2) has real partial precedent (wake-sleep) but no exact
   match** — the ticket asked to determine precedent-or-novel, and the answer is genuinely both:
   precedented pattern, novel combination. This is informational, not a design defect; no revision
   warranted, but worth naming as the design's own citation-honest self-description if the project
   ever writes up provenance.
3. **The infomorphic-networks re-read (item 3) sharpens rather than undermines** #5/#15's existing
   framing — it was already "closest existing thing, not a template," and this pass adds the
   precise reasons why it is not a template (same-objective split, not same-objective; raw-context
   reads that Patchworks' locality boundary explicitly forbids). No revision warranted against
   ADR-0008 itself.
4. **The sparsity-composition failure mode (item 4) is the one finding with teeth**, but it lands
   on an **unspecified detail** (the sparsity penalty's exact functional form), not on a committed
   claim in `07-local-learning-rule.md` or ADR-0008 — neither document asserts the penalty is L1
   or asserts joint-step composition is failure-mode-free. Because the risk is real, documented,
   and specifically activated by a choice not yet made (what the penalty's functional form is),
   this is flagged as build-time guidance rather than filed as a revision ticket against a
   closed design: **whoever specifies the sparsity pressure's exact form should read this file
   first**, and should prefer a form that degrades gracefully under joint-step subgradient descent
   (or use a proximal step) precisely because the mask already does the hard discrete part and a
   soft graceful penalty is what the remaining job needs. This does not contradict or need to
   reopen ADR-0008's "composed in the same step, not a second loop" decision — it is a note on
   *which* additive term satisfies that decision safely, which the ADR left open on purpose.

**No revision ticket filed.** Nothing found contradicts a specific, committed claim in the closed
design; the one real risk (item 4) attaches to a parameter the design deliberately left
unspecified, and is recorded here for whoever specifies it rather than as a defect in what is
already decided.

## Sources

- Rao, R.P.N. & Ballard, D.H. (1999). Predictive coding in the visual cortex: a functional
  interpretation of some extra-classical receptive-field effects. *Nature Neuroscience* 2(1),
  79–87. https://www.nature.com/articles/nn0199_79
- Whittington, J.C.R. & Bogacz, R. (2017). An approximation of the error backpropagation algorithm
  in a predictive coding network with local Hebbian synaptic plasticity. *Neural Computation*
  29(5), 1229–1262.
- Millidge, B., Tschantz, A., & Buckley, C.L. (2022). Predictive coding approximates backprop
  along arbitrary computation graphs. *Neural Computation* 34(6), 1329–1368. arXiv:2006.04182.
- Song, Y., Lukasiewicz, T., Xu, Z., & Bogacz, R. (2020). Can the brain do backpropagation? —
  exact implementation of backpropagation in predictive coding networks. *NeurIPS 2020*.
- Rosenbaum, R. (2022). On the relationship between predictive coding and backpropagation. *PLOS
  ONE* 17(3), e0266102.
- Hinton, G.E., Dayan, P., Frey, B.J., & Neal, R.M. (1995). The "wake-sleep" algorithm for
  unsupervised neural networks. *Science* 268(5214), 1158–1161.
- Ororbia, A. & Kifer, D., as surveyed in Salvatori, T. et al. context on precision-weighted
  predictive coding; primary technical source consulted: arXiv:2111.06942, "Predictive coding,
  precision and natural gradients."
- Makkeh, A., Graetz, M., Schneider, A.C., Ehrlich, D.A., Priesemann, V., & Wibral, M. (2025). A
  general framework for interpretable neural learning based on local information-theoretic goal
  functions. *PNAS* 122(10), e2408125122. arXiv:2306.02149.
- Ziyin, L. & Wang, Z. (2023). *spred*: solving L1 penalty with SGD. *Proceedings of the 40th
  International Conference on Machine Learning* (PMLR 202).
- Evci, U., Pedregosa, F., Gomez, A.N., & Elsen, E. (2019). The difficulty of training sparse
  neural networks. arXiv:1906.10732.
- Zhu, M. & Gupta, S. (2018). To prune, or not to prune: exploring the efficacy of pruning for
  model compression. arXiv:1710.01878.
