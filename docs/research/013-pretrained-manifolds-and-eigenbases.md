# Prior art: pretrained low-dimensional manifolds and designated eigenbases

Issue #13. Ground-truths the vocabulary ticket #14 borrows — "designated eigenbasis," "pretrained
generic manifold," "cup-shaped landscape," "bias-only adaptation," "grokking edge-of-chaos window" —
against primary sources before #14 is decided. Five verdicts, then a dedicated Xin Li verdict, then
flags for #14.

## 1. Does "eigenbasis" in the NTK literature mean a fixed, pretrained, reusable manifold?

**No.** The NTK eigenbasis is the eigendecomposition of a kernel operator *derived* from
architecture and input data distribution in the infinite-width limit — it is discovered, not
designated, and it is specific to one data distribution, not a portable object copied across
tasks.

Jacot, Gabriel, and Hongler introduced the NTK, showing that in the infinite-width limit a
network's training dynamics are governed by a deterministic kernel that stays constant during
training ("Neural Tangent Kernel: Convergence and Generalization in Neural Networks," NeurIPS
2018, [arXiv:1806.07572](https://arxiv.org/abs/1806.07572)). "Frozen" here means frozen *during
one training run relative to that run's own initialization statistics* — not pretrained externally
and reused.

The eigenbasis is Mercer/RKHS eigenfunctions of that kernel, and its spectrum explains **spectral
bias**: low-frequency eigenfunctions (large eigenvalues) are learned fast, high-frequency ones
slowly. Bietti and Mairal derived the RKHS spectral decay for two-layer ReLU NTKs (eigenvalues
decay as k^-d in frequency k) and showed the resulting function class coincides with the Laplace
kernel's RKHS ("On the Inductive Bias of Neural Tangent Kernels," NeurIPS 2019,
[arXiv:1905.12173](https://arxiv.org/abs/1905.12173)). Canatar, Bordelon, and Pehlevan generalize
this to any kernel/data distribution via statistical mechanics, introducing "task-model alignment"
— how much of a target function's power sits in the top eigenfunctions — as the thing that
determines sample efficiency ("Spectral bias and task-model alignment explain generalization in
kernel regression and infinitely wide neural networks," Nature Communications 12, 2914 (2021),
[arXiv:2006.13198](https://arxiv.org/abs/2006.13198)). Rahaman et al. showed the same low-frequency-first
bias empirically via Fourier analysis ("On the Spectral Bias of Neural Networks," ICML 2019,
[arXiv:1806.08734](https://arxiv.org/abs/1806.08734)); Basri et al. extended it to non-uniform input
density, where learning speed also tracks local sample density.

The critical point for #14: the eigenbasis is a property of *a specific* (architecture, input
distribution) pair at infinite width, not a compressed generic manifold you train once on
synthetic data and copy into unrelated downstream contexts. It also isn't literally "chosen" —
nobody designates it; it falls out of the limit. The word choice in #14 ("designated eigenbasis")
does not have a real referent in this literature.

## 2. Is there precedent for pretraining a generic manifold on synthetic data, then freezing it for reuse?

**Yes, and it predates deep learning.** Two independent, well-established traditions match this
shape closely enough to matter for #14.

**Reservoir computing / echo state networks** are the strongest structural precedent for "frozen
internals, only a thin outer layer adapts." An ESN's recurrent reservoir is fixed at
initialization (large, sparse, random, tuned only for the echo-state property — spectral radius
< 1); only a **linear readout** is ever trained (Jaeger, "The 'echo state' approach to analysing
and training recurrent neural networks," GMD Report 148, 2001; Jaeger and Haas, "Harnessing
nonlinearity: predicting chaotic systems and saving energy in wireless communication," Science
304(5667):78–80, 2004). This is closer in spirit to #14 than the ticket's own citations (BitFit,
sparse fine-tuning) — the reservoir is never pretrained on a task, its structure is just
non-degenerate random dynamics, and the entire adaptive burden sits in the readout. #14's frozen
body is not random, though — it's pretrained on synthetic data to have a specific shape — so the
match is structural, not exact.

**TabPFN** is the closer match for "pretrained on synthetic data specifically, then frozen and
reused across arbitrary downstream contexts." It is a transformer pretrained exclusively on
millions of synthetic tabular datasets generated from structural causal models, then frozen and
applied zero-shot to real tabular tasks via one forward pass of in-context learning — no
gradient-based adaptation at deployment at all (Hollmann et al., "Accurate predictions on small
data with a tabular foundation model," Nature 637:319–326, 2025,
[doi:10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6)). The mismatch with
#14: TabPFN's downstream "adaptation" is in-context conditioning on a prompt, not a persistent
per-instance parameter (bias) update — TabPFN never modifies its own weights per deployment, while
#14's biases are meant to keep adapting continually per cell.

Both precedents are real and citable as grounding for "frozen generic body + thin adapting layer";
neither is a perfect match for #14's specific mechanism (per-cell bias vectors, continually
updated, never reset).

## 3. Is a "cup" — a single global basin with descent everywhere — a real, named, trainable object?

**No such named trainable object exists in the ML literature.** The closest analogues are three
separate things, none of which is "a manifold shape you pretrain a network into and dial by
compression level."

**Star-convexity / one-point convexity** is the nearest match and is a real, used term: a function
is star-convex if it's convex along every ray from some (unknown a priori) global minimizer.
Kleinberg-style analyses show SGD trajectories are empirically star-convex for most of training,
which is used to *prove* convergence to a global minimum, and follow-on work treats star-convexity
as an explicit constraint for reshaping a landscape ("SGD Converges to Global Minimum in Deep
Learning via Star-convex Path," [arXiv:1901.00451](https://arxiv.org/abs/1901.00451); "Do Deep
Neural Network Solutions Form a Star Domain?," [arXiv:2403.07968](https://arxiv.org/abs/2403.07968)).
This is a *diagnosed* property of trained-network loss landscapes under certain conditions, not a
constructive recipe for building one at a chosen compression level.

**Folding funnels**, from protein energy-landscape theory, are the actual named, well-established
"single dominant basin" object — but for a physical free-energy surface over conformation space,
selected for by evolution, not trained by gradient descent in the ML sense (Onuchic,
Luthey-Schulten, Wolynes, "Theory of Protein Folding: The Energy Landscape Perspective," Annual
Review of Physical Chemistry 48:545–600, 1997). The funnel is real, named, and single-basin, but
it lives in a different field and is not a trainable ML object.

**Mode connectivity / loss-landscape visualization** shows that trained solutions from a shared
lineage (same init, or fine-tuned from one pretrained checkpoint) often sit in one basin connected
by low-loss paths (Garipov et al., "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs,"
NeurIPS 2018; Li, Xu, Taylor, Studer, Goldstein, "Visualizing the Loss Landscape of Neural Nets,"
NeurIPS 2018). This describes single-basin structure that *emerges post hoc* among solutions, not
a designed shape imposed on the network body before deployment. The same NeurIPS-2018 paper (Li et
al.) is directly relevant to #14's premise in the other direction: it shows landscape geometry gets
*more* chaotic, not smoother, as depth increases, absent skip connections — the opposite of what a
"cup, always downhill" claim would predict for an arbitrary architecture.

Separately, Pope et al. establish that natural datasets have low intrinsic dimension and that
lower-intrinsic-dimension data is easier to learn and generalizes better ("The Intrinsic Dimension
of Images and Its Impact on Learning," ICLR 2021,
[arXiv:2104.08894](https://arxiv.org/abs/2104.08894)) — relevant to "chart" compression generally,
but says nothing about landscape *shape* at a given compression level, only that lower-dimensional
problems are easier.

**Verdict for #14: no citation exists for "cup" as a defined, trainable object.** If #14 keeps the
shape, it should be framed as a hypothesis to test, not as recovered prior art.

## 4. What does bias-only adaptation of an otherwise-frozen network actually buy?

**Task-dependent, and the direction of the dependence matters for #14: it is closer to full
fine-tuning on easier/smaller-data regimes and the gap widens as task difficulty and data scale
grow.** It is not "nearly free" in general.

BitFit trains only bias terms of a pretrained transformer (under 0.1% of parameters) and is
reported *competitive with, and sometimes better than, full fine-tuning* on GLUE at small-to-medium
data; at larger data the gap opens and BitFit is only competitive with other sparse methods, not
full fine-tuning (Ben Zaken, Ravfogel, Goldberg, "BitFit: Simple Parameter-efficient Fine-tuning
for Transformer-based Masked Language-models," ACL 2022,
[arXiv:2106.10199](https://arxiv.org/abs/2106.10199)). Representative numbers reported: BitFit
reaches 91.4% mean accuracy on QNLI versus 91.7% for full fine-tuning — near-parity on tasks where
it holds, but the paper's own headline claim is bounded to the small/medium-data regime, not
universal.

Two adjacent bias-adjacent results corroborate the "surprisingly strong, but not full parity"
pattern: training only BatchNorm's affine (scale/shift, i.e. bias-like) parameters on an otherwise
**randomly initialized** (never pretrained) CNN reaches 82% on CIFAR-10 and 32% top-5 on ImageNet —
far above chance, but far below a fully trained network (Frankle, Schwab, Morcos, "Training
BatchNorm and Only BatchNorm: On the Expressive Power of Random Features in CNNs," ICLR 2021,
[arXiv:2003.00152](https://arxiv.org/abs/2003.00152)). FiLM-style affine conditioning layers
(scale+shift on feature maps, freezing the rest) are reported as a strong, data-efficient
adaptation mechanism in vision meta-learning, again in the "surprisingly capable, not a full
substitute for weight adaptation" register (Perez et al., "FiLM: Visual Reasoning with a General
Conditioning Layer," AAAI 2018).

**Verdict for #14:** bias-only adaptation is a real, working idea with real citations, but the
literature's own framing is conditional ("small-to-medium data," "competitive... sometimes") —
#14 should not assume bias-only adaptation scales to arbitrarily hard reconciliation dynamics
without treating that as an open risk.

## 5. Is grokking confined to a narrow edge-of-chaos model-size window?

**No paper matching this description was found; the specific claim should be treated as
unsubstantiated until a matching primary source turns up.** A genuine search across grokking
papers and edge-of-chaos papers did not surface a paper that frames grokking's onset/absence as a
function of network width/depth via a mean-field or dynamical-isometry order-chaos transition.

What exists, and how each falls short of the claim:

- Zhang, Feng, Chen, Lai, "Edge of chaos as a guiding principle for modern neural network
  training" ([arXiv:2107.09437](https://arxiv.org/abs/2107.09437), July 2021) is a genuine
  order-chaos-transition paper for NN training generally — but it **predates** Power et al.'s
  naming of "grokking" (2022) and never mentions it. Not a grokking paper.
- Power, Burda, Edwards, Babuschkin, Misra, "Grokking: Generalization Beyond Overfitting on Small
  Algorithmic Datasets" ([arXiv:2201.02177](https://arxiv.org/abs/2201.02177), 2022) is the
  origin paper; it reports data-size and optimization dependence, not a width-driven edge-of-chaos
  framing.
- Nanda, Chan, Lieberum, Smith, Steinhardt, "Progress measures for grokking via mechanistic
  interpretability" (ICLR 2023, [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)) fully
  reverse-engineers a modular-addition grokking circuit but does not address model-size windows.
- Liu, Michaud, Tegmark, "Omnigrok: Grokking Beyond Algorithmic Data" (ICLR 2023,
  [arXiv:2210.01117](https://arxiv.org/abs/2210.01117)) explains grokking via a weight-norm
  mismatch between train/test loss landscapes (the "LU mechanism") — a landscape-geometry account,
  not a chaos/order dynamical one.
- Prieto, Barsbey, Mediano, Birdal, "Grokking at the Edge of Numerical Stability" (ICLR 2025,
  [arXiv:2501.04697](https://arxiv.org/abs/2501.04697)) reports a narrow "Goldilocks Zone," but
  it's a window of **weight norm / softmax numerical stability**, not model width or depth.
- Wang, "Grokking as Dimensional Phase Transition in Neural Networks"
  ([arXiv:2604.04655](https://arxiv.org/abs/2604.04655), 2026) is the closest in *spirit*: it
  frames grokking as a self-organized-criticality transition in the effective dimensionality of
  the gradient field (sub- to super-diffusive), tested across eight model scales. This is
  genuinely a phase-transition account adjacent to "edge of chaos," but the paper does not use that
  term, does not tie the transition to a narrow *range of model sizes* (it reports the transition
  recurring across all eight scales tested, not disappearing outside a band), and is a very recent,
  single-author, not-yet-widely-vetted preprint.

**Verdict for #14:** do not cite "grokking is confined to a narrow edge-of-chaos width range" as an
established result. If this idea matters to #14's design, it needs to be flagged as speculative
and either sourced more precisely or dropped.

## Xin Li's "cup-shaped manifold" — verdict

**Could not verify as cited — treat with skepticism.** Xin Li (Professor, Department of Computer
Science, University at Albany, SUNY; confirmed via matching email `xli48@albany.edu` on both his
CV and his recent arXiv papers) is a real researcher with a real, active, citable body of
information-topology work. But nothing in that published work describes a "cup" — a trainable
surface with a downhill direction everywhere guaranteeing global-minimum descent — as #14
describes it.

Confirmed real and on-topic: "Information Topology" (Li, [arXiv:2210.03850](https://arxiv.org/abs/2210.03850),
v3 2026) proposes cycle closure as the primitive operation of inference, a "Structure-Before-Specificity"
principle, and the "Context-Content Uncertainty Principle" (CCUP) — a framework built from algebraic
topology (homology classes, cycle closure) and information theory, not differential-geometric
basin/convexity language. Related papers checked directly — "On Context-Content Uncertainty
Principle" ([arXiv:2506.20699](https://arxiv.org/abs/2506.20699)) and "The Two Dragons of
Cognition" (Frontiers in Computational Neuroscience, 2026,
[10.3389/fncom.2026.1778902](https://doi.org/10.3389/fncom.2026.1778902)) — build on the same
topological vocabulary (Topological Trinity Transformation, homological parity, Urysohn's Lemma).

Full-text search of these papers for "cup," "bowl," "funnel," "basin," "convex," "gradient
descent," and "global minimum" returned **zero matches in every paper checked**. His earlier
(pre-2023, WVU-era) publication record is almost entirely image/video processing and computer
vision (denoising, super-resolution, deblurring) with no topology or manifold-shape content at
all — the information-topology program is a distinct, recent (2022–2026) pivot, not the bulk of
his output.

There is a plausible, unverified path by which #14's "cup" could be a loose paraphrase of Li's
work rather than an outright fabrication: CCUP's own name is a near-homophone of "cup," and his
papers do describe "measure concentration onto residual invariant manifolds" (mass collapsing to
a narrow tube around a closed cycle) — a convergence-to-a-stable-structure idea that a listener
could reasonably compress into "gradient descent toward a global minimum" in casual retelling. But
that is speculation on this researcher's part, not something found in the text. **Do not cite Li's
work as prior art for a trainable cup-shaped manifold; if #14 wants to keep referencing him, it
should describe CCUP/cycle-closure accurately (a topological stabilization claim) rather than as a
basin-geometry claim.**

## Flags for ticket #14

- **The "cup" has no name and no citation anywhere searched** — not in loss-landscape geometry, not
  in Xin Li's papers. If #14 keeps it, present it as this project's own hypothesis, not recovered
  prior art. Star-convexity is the one real, citable, nearby concept worth borrowing vocabulary
  from if #14 wants formal footing.
- **"Designated eigenbasis" oversells the NTK connection.** The NTK eigenbasis is derived from a
  specific (architecture, data distribution) pair at infinite width, not a portable object chosen
  by a designer and reused across contexts. If #14 wants an NTK-flavored justification for a
  shared frozen body, it needs a different argument — the eigenbasis framing doesn't carry the
  weight #14 wants from it.
- **Reservoir computing is a better-grounded structural precedent than #14 currently cites**, and
  it's older and simpler: fixed random internal dynamics, train only a thin readout. It weakens
  #14's case in one way, though — ESN reservoirs are *not pretrained on synthetic data to have a
  specific useful shape*, they're just non-degenerate. If #14's mechanism depends on the pretraining
  step doing real work (imposing a cup, not just being "generic enough"), reservoir computing
  doesn't validate that step; it only validates "frozen internals + thin adapting layer" as a
  category.
- **TabPFN is a strong precedent for "synthetic pretraining, frozen reuse," but its adaptation
  mechanism (in-context conditioning) is structurally different from #14's (persistent per-cell
  bias updates).** Citing TabPFN as support for the freeze should not imply it also supports
  bias-only adaptation as the update mechanism — those are two separate claims in #14 that need two
  separate justifications.
- **Bias-only adaptation's own literature is conditional, not unconditional.** BitFit's headline
  result is bounded to small/medium data; BatchNorm-only and FiLM results show real but partial
  capacity. #14 should treat "biases alone are enough" as a bet with known regimes where it's known
  to lose ground to full adaptation, not a settled result.
- **The grokking/edge-of-chaos claim, if it's load-bearing anywhere in #14's reasoning about why a
  compressed frozen body should be trainable/well-behaved, is currently unsupported** and should
  either be sourced to a specific paper (none found here) or removed.

## References

- Jacot, A., Gabriel, F., Hongler, C. "Neural Tangent Kernel: Convergence and Generalization in
  Neural Networks." NeurIPS 2018. https://arxiv.org/abs/1806.07572
- Bietti, A., Mairal, J. "On the Inductive Bias of Neural Tangent Kernels." NeurIPS 2019.
  https://arxiv.org/abs/1905.12173
- Canatar, A., Bordelon, B., Pehlevan, C. "Spectral bias and task-model alignment explain
  generalization in kernel regression and infinitely wide neural networks." Nature Communications
  12, 2914 (2021). https://arxiv.org/abs/2006.13198
- Rahaman, N. et al. "On the Spectral Bias of Neural Networks." ICML 2019.
  https://arxiv.org/abs/1806.08734
- Basri, R. et al. "Frequency Bias in Neural Networks for Input of Non-Uniform Density." ICML 2020.
- Jaeger, H. "The 'echo state' approach to analysing and training recurrent neural networks." GMD
  Report 148, German National Research Center for Information Technology, 2001.
- Jaeger, H., Haas, H. "Harnessing nonlinearity: predicting chaotic systems and saving energy in
  wireless communication." Science 304(5667):78–80, 2004.
- Hollmann, N. et al. "Accurate predictions on small data with a tabular foundation model." Nature
  637:319–326, 2025. https://doi.org/10.1038/s41586-024-08328-6
- Kleinberg, B. et al. "SGD Converges to Global Minimum in Deep Learning via Star-convex Path."
  https://arxiv.org/abs/1901.00451
- "Do Deep Neural Network Solutions Form a Star Domain?" https://arxiv.org/abs/2403.07968
- Onuchic, J.N., Luthey-Schulten, Z., Wolynes, P.G. "Theory of Protein Folding: The Energy
  Landscape Perspective." Annual Review of Physical Chemistry 48:545–600, 1997.
- Garipov, T. et al. "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs." NeurIPS 2018.
- Li, H., Xu, Z., Taylor, G., Studer, C., Goldstein, T. "Visualizing the Loss Landscape of Neural
  Nets." NeurIPS 2018.
- Pope, P., Zhu, C., Abdelkader, A., Goldblum, M., Goldstein, T. "The Intrinsic Dimension of Images
  and Its Impact on Learning." ICLR 2021. https://arxiv.org/abs/2104.08894
- Ben Zaken, E., Ravfogel, S., Goldberg, Y. "BitFit: Simple Parameter-efficient Fine-tuning for
  Transformer-based Masked Language-models." ACL 2022. https://arxiv.org/abs/2106.10199
- Frankle, J., Schwab, D.J., Morcos, A.S. "Training BatchNorm and Only BatchNorm: On the Expressive
  Power of Random Features in CNNs." ICLR 2021. https://arxiv.org/abs/2003.00152
- Perez, E. et al. "FiLM: Visual Reasoning with a General Conditioning Layer." AAAI 2018.
- Power, A., Burda, Y., Edwards, H., Babuschkin, I., Misra, V. "Grokking: Generalization Beyond
  Overfitting on Small Algorithmic Datasets." 2022. https://arxiv.org/abs/2201.02177
- Nanda, N., Chan, L., Lieberum, T., Smith, J., Steinhardt, J. "Progress measures for grokking via
  mechanistic interpretability." ICLR 2023. https://arxiv.org/abs/2301.05217
- Liu, Z., Michaud, E.J., Tegmark, M. "Omnigrok: Grokking Beyond Algorithmic Data." ICLR 2023.
  https://arxiv.org/abs/2210.01117
- Prieto, L., Barsbey, M., Mediano, P.A.M., Birdal, T. "Grokking at the Edge of Numerical
  Stability." ICLR 2025. https://arxiv.org/abs/2501.04697
- Wang, P. "Grokking as Dimensional Phase Transition in Neural Networks." 2026.
  https://arxiv.org/abs/2604.04655
- Zhang, L., Feng, L., Chen, K., Lai, C.H. "Edge of chaos as a guiding principle for modern neural
  network training." 2021. https://arxiv.org/abs/2107.09437
- Li, X. "Information Topology." 2022–2026 (v3). https://arxiv.org/abs/2210.03850
- Li, X. "On Context-Content Uncertainty Principle." 2025. https://arxiv.org/abs/2506.20699
- Li, X. "The Two Dragons of Cognition: Recursive Condensation for..." Frontiers in Computational
  Neuroscience 20, 2026. https://doi.org/10.3389/fncom.2026.1778902
- Li, X. Faculty page and CV, University at Albany, SUNY. https://www.albany.edu/computer-science/faculty/xin-li
