# Literature pass: abstraction as neighbourhood breadth, not hierarchical depth (patchworks#590)

Resolves [B31 / #590](https://github.com/NGL321/patchworks/issues/590), opened by
[B27](https://github.com/NGL321/patchworks/issues/576) on the parent map
[#532](https://github.com/NGL321/patchworks/issues/532). The question is whether the prior art
supports the user's candidate replacement for the dome: **a cell is abstract because it integrates
over a broad, disparate neighbourhood, not because it sits deep in a taper** — and whether a
region's intrinsic timescale tracks connectivity *breadth* rather than hierarchical *position*.

This is a reading pass, not a decision. It closes nothing on #532 and licenses no bar.

**Provenance discipline.** Every claim below is tagged. `[SOURCE]` = quoted or paraphrased from the
paper's own words, read at source. `[INFERENCE]` = mine, drawn from sources, not stated by any of
them. `[ABSENT]` = looked for and not found; absence of evidence, reported as such. The map has been
burned once by a proxy standing in for a measurement ([#538](https://github.com/NGL321/patchworks/issues/538)),
so where a paper measures something *adjacent to* an intrinsic timescale rather than an intrinsic
timescale, that is flagged in the body, not buried.

---

## Headline verdict, stated plainly

**The breadth hypothesis is real, published, and has a mechanism — but the literature does not
support breadth *instead of* depth. In the two best-specified mechanistic models, the depth-aligned
local gradient is the generator of separated timescales and long-range breadth is what *fights* it.
The single strongest empirical result on the question pre-registered exactly the user's hypothesis
in two forms, and one of the two forms failed.** In detail:

- **Breadth-slows-a-unit has a real, non-correlational precedent.** Gollo, Zalesky, Hutchison, van
  den Heuvel & Breakspear (2015) start from *identical* oscillators with the *same* characteristic
  time scales, couple them by the macaque connectome, impose no timescale hierarchy at all, and get
  a dynamical hierarchy out of topology alone. Their stated mechanism is in-degree: *"larger
  in-degree guarantees limited irregularity."* This is the closest thing in the literature to the
  user's claim and it is exactly the shape the user proposed. **But it measures dynamic stability
  and temporal persistence of synchrony, not an autocorrelation-based `τ`** — see §3.1. Read it as
  support with a named proxy gap, not as a measurement of τ against degree.
- **The dissociation the ticket asked for exists, in both directions, and both instances are
  reported at source.** Chaudhuri et al. (2015) name area **8m** — *low* in the hierarchy, long
  timescales, because it sits in the strongly-connected prefrontal core — and area **TEpd** — *high*
  in the hierarchy, fast, because it does not. Broadly-connected-but-shallow and
  deep-but-narrowly-connected are both instantiated, by name, in the same figure.
- **The best empirical test pre-registered the user's hypothesis in two forms and half of it
  failed.** Lurie, Pappas & D'Esposito (2024) pre-registered that both high within-module degree
  *and* high participation coefficient (breadth across communities) would predict longer timescales.
  **Structural** breadth held (SC participation coefficient r = 0.282 / 0.366). **Functional**
  breadth did not — FC participation coefficient was weakly *negative* and never significant, which
  the authors state "runs counter to our initial predictions." Their conclusion is that longer
  timescales reflect *within-community* connectivity, i.e. **topologically local rather than global
  integration**. That is the opposite of "connects broad networks of disparate abstract concepts."
- **The strongest evidence against is a theorem, not a null result.** Li & Wang (PNAS 2022) prove
  that timescale segregation *is* eigenvector localization, and that **"the long-range connections
  tend to delocalize eigenvectors and thus break the timescale hierarchy, but the heterogeneity of
  local recurrent excitation level weakens its effect on eigenvector delocalization in a divisive
  fashion."** In that account breadth is the *destroyer* of separated timescales and the
  depth-aligned local gradient is what protects them. Chaudhuri et al. (2015) reach the same
  asymmetry empirically: remove the local microcircuit gradient and *the range of timescales
  collapses and the relation to hierarchy disappears*; remove long-range projections and *timescale
  simply reflects hierarchical position*.
- **A learned, variable-radius neighbourhood exists in graph learning and is a solved engineering
  problem.** Xu et al. (ICML 2018, JK-Nets) learn a *per-node* effective neighbourhood size; Zhao et
  al. (NeurIPS 2021, ADC) learn a *continuous* radius `t` per layer and per feature channel; Klicpera
  et al. (ICLR 2019, APPNP) show the mechanism by which locality survives an unbounded radius —
  a teleport term. **In sheaf learning specifically, nothing of the kind was found — see §5.4
  `[ABSENT]`.**
- **The one thing nobody has done is the thing #532 would need.** No source read for this pass
  regresses an intrinsic timescale on breadth *and* depth *in the same model* to ask which survives.
  Chaudhuri gets closest, with lesions rather than regression, and answers *both are necessary*.
  `[ABSENT]`

**What this means for the map, stated once and not argued.** The record's finding that the apex is
the *fastest* place in the graph ([B23](https://github.com/NGL321/patchworks/issues/572)) is **not
anomalous in this literature** — Lurie et al. report that "many regions which sit near the top of
canonical information processing hierarchies (e.g., limbic areas) actually exhibit some of the
shortest timescales in the cerebral cortex." But the same literature says the fix is not to swap
depth for breadth: in every model that produces separated timescales, something *heterogeneous and
local* to the unit is doing the separating. `[INFERENCE]`

---

## 1. Does the intrinsic-timescale literature tie timescale to connectivity breadth?

### 1.1 The canonical hierarchy claim, so the baseline is on the record

*Source: Murray, Bernacchia, Freedman, Romo, Wallis, Cai, Padoa-Schioppa, Pasternak, Seo, Lee &
Wang (2014), "A hierarchy of intrinsic timescales across primate cortex", Nature Neuroscience
17(12):1661–1663, doi:10.1038/nn.3862. Read via its abstract and via Chaudhuri et al. (2015)'s
in-text characterisation of it, which I read verbatim in the Chaudhuri PDF; the Nature full text is
paywalled and was not reached.*

The baseline the breadth hypothesis is competing against: timescales of intrinsic spiking
fluctuations were measured across areas and found to be hierarchically ordered, sensory short,
prefrontal long. Chaudhuri et al. state it as: *"an area's timescale was well predicted by its
position in the anatomical hierarchy of Felleman and Van Essen."* `[SOURCE]`

**This is the claim B23 killed on Patchworks' own graph.** It is also the claim the rest of this
section erodes in the prior art itself.

### 1.2 Breadth measures do correlate with timescale — but which breadth measure matters

*Source: Lurie, D. J., Pappas, I., & D'Esposito, M. (2024), "Cortical timescales and the modular
organization of structural and functional brain networks", Human Brain Mapping 45(2),
doi:10.1002/hbm.26587. Open access. Read at source via full-text passages; methods and primary
hypotheses pre-registered on OSF, with an internal replication in a second independent dataset.*

This is the most directly relevant empirical paper found, and the most useful, because it is
**pre-registered** — which makes its failure interpretable in exactly the way this repo treats a
pre-registered falsifier.

Their pre-registered hypothesis, in their words: *"we hypothesized that cortical areas with
extensive within-community connectivity (i.e., high WD) or diverse connectivity across communities
(i.e., high PC) would tend to have longer timescales than less topologically central regions."*
`[SOURCE]`

Results, with the numbers, all partial correlations covarying for ROI volume, corrected for spatial
autocorrelation, discovery / validation:

| Quantity | Discovery | Validation | Pre-registered prediction |
|---|---|---|---|
| Structural connectivity **degree** | r = 0.269 | r = 0.419 | held |
| Functional connectivity **degree** | r = 0.300 (n.s.) | r = 0.124 after tSNR exclusion (n.s.) | **did not hold robustly** |
| **FC within-module degree** (local breadth) | r = 0.368 | r = 0.364 | held |
| **FC participation coefficient** (breadth across communities) | weak *negative*, n.s. | weak *negative*, n.s. | **failed** |
| **SC participation coefficient** (structural breadth across communities) | r = 0.282 | r = 0.366 | held |

`[SOURCE]` for every row.

The authors' own reading of the failure: *"we did not find evidence for a relationship between
timescales and the extent to which each cortical area exhibits diverse functional connectivity
across communities. This finding runs counter to our initial predictions."* And the summary
sentence: *"in functional connectivity networks, the extent of within-community connectivity, but
not the diversity of between-community connectivity, is associated with longer timescales."*
`[SOURCE]`

They generalise it: *"Together, these results suggest that long timescales may reflect topologically
local rather than global integration within functional connectivity networks."* `[SOURCE]`

**Why this matters to #590 specifically.** The user's formulation is *"abstract ones connect broad
networks of disparate abstract concepts."* Disparate-community breadth is precisely the
participation coefficient. In this dataset it holds **structurally** and fails **functionally**, and
the two measures are near-independent in their own data (FC WD vs FC PC correlation 0.085 / 0.128).
So the literature does not give a clean yes: it gives *yes for anatomical breadth, no for functional
breadth, and a positive result for within-neighbourhood density that the breadth story does not
predict.* `[SOURCE]` for the components, `[INFERENCE]` for the reading.

Prior work they replicate, cited by them and not read at source for this pass: Baria et al. (2013),
Fallon et al. (2020), Sethi et al. (2017) — all reporting high-degree hubs with longer timescales.
`[SOURCE, secondary]`

---

## 2. The dissociations the ticket asked for

The ticket asked for a region **broadly connected but shallow**, and one **deep but narrowly
connected**. Both exist, named, in one paper.

*Source: Chaudhuri, R., Knoblauch, K., Gariel, M.-A., Kennedy, H., & Wang, X.-J. (2015), "A
Large-Scale Circuit Mechanism for Hierarchical Dynamical Processing in the Primate Cortex", Neuron
88(2):419–431, doi:10.1016/j.neuron.2015.09.008. Read at source (author PDF, cns.nyu.edu).*

> "Nevertheless, an area's timescales are not entirely determined by its hierarchical position, and
> the plotted timescales do not increase monotonically with hierarchy." `[SOURCE]`

**Broadly connected but shallow — area 8m:**

> "consider area 8m (part of the frontal eye fields), which is low in the hierarchy and would show a
> rapid decay of correlation in the absence of long-range projections … but instead demonstrates
> long timescales in the model (and in the empirical observations of Hasson et al., 2008). … area 8m
> participates in a strongly-connected core of prefrontal and association areas … allowing it to
> show long timescales that emerge from inter-areal excitatory loops" `[SOURCE]`

**Deep but narrowly connected — area TEpd:**

> "Conversely, whereas area TEpd is high in the hierarchy, it does not participate in this core and
> is instead strongly coupled to ventral stream visual areas. Thus, it reflects the faster
> timescales of visual input." `[SOURCE]`

Their three-term summary of what sets a timescale:

> "each area in Figure 2C shows timescales approximately determined by its distance from the
> periphery (hierarchical position), proximity to the central clusters (long-range connectivity),
> and distance from the source of input." `[SOURCE]`

**Read this carefully before using it.** It is a dissociation *within a model*, not an experimental
dissociation, and the model's hierarchical axis is itself parameterised by a *local* quantity
(spine count) — see §3.2. The dissociation is real; it is a dissociation between *anatomical rank*
and *timescale*, mediated by long-range connectivity. It is not evidence that depth is inert.
`[INFERENCE]`

A second, empirical dissociation, from Lurie et al. (2024):

> "we found that many regions which sit near the top of canonical information processing
> hierarchies (e.g., limbic areas [Mesulam, 1998]) actually exhibit some of the shortest timescales
> in the cerebral cortex, and that the majority of sensory and motor regions did not have
> particularly short timescales." `[SOURCE]`

They list six further papers reporting the same divergence of BOLD timescales from the canonical
sensory-association axis (Baria 2013, Fallon 2019, Manea 2022, Shafiei 2020, Shinn 2021, Wengler
2020). `[SOURCE, secondary]`

**This is the closest thing in the prior art to B23's result.** The apex being fast is a published
observation about real cortex, not a Patchworks artefact. `[INFERENCE]`

---

## 3. Is there a *mechanism*, not just a correlation?

### 3.1 Yes — and it is the user's mechanism, with a proxy caveat

*Source: Gollo, L. L., Zalesky, A., Hutchison, R. M., van den Heuvel, M., & Breakspear, M. (2015),
"Dwelling quietly in the rich club: brain network determinants of slow cortical fluctuations",
Philosophical Transactions of the Royal Society B 370(1668):20140165, doi:10.1098/rstb.2014.0165.
Read at source via the authors' arXiv preprint, arXiv:1502.04455. The journal HTML returned 403 and
the PMC copy returned a CAPTCHA; the preprint is the version read.*

The design is exactly the control the breadth hypothesis needs — no imposed timescale gradient, no
depth, no heterogeneity in the units:

> "The objectives of the present study were to study the dynamics that emerge from the interaction
> of **identical** dynamical elements coupled via the structural connectome of primate cortex. That
> is, **we do not impose a hierarchy of time scales**, but rather start with an ensemble of neural
> mass oscillators which all have **the same characteristic time scales**. We then partition their
> dynamics according to the rich club hierarchy to find an emerging dynamical hierarchy." `[SOURCE]`
> (emphasis mine)

The result:

> "the dynamics in rich club nodes are more stable, with greater temporal persistence and dynamic
> stability than peripheral regions" `[SOURCE]`

The mechanism, stated causally and not correlationally:

> "Our analyses suggest that the high in-degree of rich club nodes plays a central role in promoting
> a stable, dynamical core of spontaneous activity in the primate cortex and highly unstable
> dynamical transitions in the periphery." `[SOURCE]`

> "The richness of these brain regions comes from their large in-degree. However, for this very
> reason, their dynamics are consequently more regular because **larger in-degree guarantees limited
> irregularity**. In other words, the autonomous dynamics of connectome hubs are largely enslaved to
> the strong, rhythmic output of the entire connectome." `[SOURCE]`

They also isolate degree from higher-order topology with null models: *"a degree-preserving random
(DPR) network; … uniform in-degree. These benchmark networks allow us to discern which dynamic
properties are due to the heterogeneous node degree, and which owe to higher order properties such
as the motif composition, rich club"*, and conclude that motif arrangement plays *"an additional,
sculpting role"* on top of in-degree. `[SOURCE]`

**The proxy caveat, stated prominently because #538 exists.** Gollo et al.'s dependent variables are
*pair instability*, *dynamic irregularity*, and *temporal persistence of synchrony* — the transverse
stability of pairs of nodes. They explicitly say *"The term 'stability' does not refer here to the
dynamics of each node (which are all chaotic), but rather to the 'transverse stability' of pairs of
nodes."* `[SOURCE]` **They do not fit an autocorrelation decay constant per node and regress it on
degree.** Anyone citing this paper as "degree predicts τ" is substituting a proxy for the
measurement, which is the #538 failure. What it licenses is: *in-degree produces slower, more
persistent, more regular dynamics in a network of identical fast units.* `[INFERENCE]`

A second mechanistic account, named by Lurie et al. and not read at source: Baria et al. (2013)
propose that *"local synaptic integration may act as a low pass filter on these inputs."*
`[SOURCE, secondary]` This is the low-pass argument — breadth as averaging — and it is the
mechanism that would transfer most directly to a transport operator. It was **not** verified at
source in this pass. `[ABSENT — not read]`

### 3.2 And a competing mechanism, which is the depth-aligned one

*Source: Chaudhuri et al. (2015), as above; and Li, S., & Wang, X.-J. (2022), "Hierarchical
timescales in the neocortex: Mathematical mechanism and biological insights", PNAS 119(6):e2110274119,
doi:10.1073/pnas.2110274119. Read at source (author PDF, ins.sjtu.edu.cn).*

Chaudhuri's model has one heterogeneous local parameter — the excitatory gradient, set by dendritic
spine count per pyramidal neuron, which is itself strongly correlated with hierarchical position:

> "the number of basal dendritic spines on layer three pyramidal neurons increases sharply from
> primary sensory to prefrontal areas … Taking spine count as a proxy for excitatory synapses per
> pyramidal cell" `[SOURCE]`

Li & Wang state the same parameter's role formally: *"the gradient of `h_i` parameterizes the
gradient of synaptic excitation across areas in the model, supported by the fact that `h_i` is
proportional to the spine count per pyramidal neuron across areas … in the form of a macroscopic
gradient."* `[SOURCE]`

**The lesion results are the load-bearing ones.** Chaudhuri et al., verbatim:

> "the range of timescales is drastically reduced in the absence of differences in the microcircuit
> across areas. Moreover, there is no longer a relationship to an area's position in the anatomical
> hierarchy. Thus, while differences in long-range inputs and outputs to each area are significant,
> **they are insufficient to account for disparate timescales and local heterogeneity is needed**."
> `[SOURCE]`

> "once long-range projections are removed, an area's time constant simply reflects its position in
> the hierarchy." `[SOURCE]`

> "scrambling almost entirely removes the hierarchy of timescales, further confirming that **a
> gradient of excitation alone is insufficient to separate timescales**." `[SOURCE]`

**Both are necessary; neither is sufficient.** That is the honest summary of the best-specified
model in the field, and it does not license "breadth instead of depth". `[INFERENCE]`

---

## 4. Evidence against, reported prominently as the ticket requires

### 4.1 Breadth actively destroys timescale separation (a theorem, not a null)

Li & Wang (2022) reduce the whole question to eigenvector localization. Their abstract:

> "the segregation of disparate timescales is defined in terms of the localization of eigenvectors
> of the connectivity matrix, which depends on three circuit properties: 1) a macroscopic gradient
> of synaptic excitation, 2) distinct electrophysiological properties between excitatory and
> inhibitory neuronal populations, and 3) a detailed balance between long-range excitatory inputs
> and local inhibitory inputs for each area-to-area pathway." `[SOURCE]`

And the sentence that cuts hardest against #590's hypothesis:

> "the degree of eigenvector localization is determined by the competition between the strength of
> long-range connections encoded in matrix `Σ̄_UL` and the spectral gap of matrix `Λ̄_U`. Therefore,
> **the long-range connections tend to delocalize eigenvectors and thus break the timescale
> hierarchy**, but the heterogeneity of local recurrent excitation level weakens its effect on
> eigenvector delocalization in a divisive fashion." `[SOURCE]`

They also confirm the non-monotonicity from §2, and state the disconnected limit:

> "the slow timescales of each area in this disconnected network are segregated and follow the
> hierarchical order `h_i` as the corresponding eigenvectors are perfectly localized and orthogonal
> to each other." `[SOURCE]`

> "the timescale does not change monotonically with the anatomically defined hierarchy (x axis); the
> precise pattern is sculpted by the measured interareal wiring properties." `[SOURCE]`

**Read together: breadth explains the deviations from the hierarchy; the local gradient explains the
existence of the hierarchy.** That is a supporting role for breadth, not a replacement.
`[INFERENCE]` This is the single most important line of this document for #532, because it is a
statement about a *linear operator's spectrum*, which is the object #532 is arguing about, and it
says that widening reach costs you the very separation you widened for.

### 4.2 Timescale tracks intrinsic cellular properties, and is not fixed at all

*Source: Gao, R., van den Brink, R. L., Pfeffer, T., & Voytek, B. (2020), "Neuronal timescales are
functionally dynamic and shaped by cortical microarchitecture", eLife 9:e61277,
doi:10.7554/eLife.61277. Read via the eLife article page.*

Cellular, not connectional, predictors: timescales correlate with expression of genes encoding NMDA
(GRIN2B) and GABA-A (GABRA3) receptor subunits, voltage-gated sodium (SCN1A) and potassium (KCNA3)
channels, and PVALB; ρ = −0.60 between the timescale gradient and the dominant gene-expression axis;
ρ = −0.47 with the T1w/T2w ratio; ρ = 0.37 with cortical thickness. `[SOURCE]`

And the title's first claim: timescales are ~20% longer during working-memory delay periods, with
longer delay-period timescales in PFC associated with better performance (ρ = 0.75); timescales
shorten with age (ρ = −0.31). `[SOURCE]`

Their reconciliation, which is the fairest statement of the field's position: *"structural
properties may constrain dynamical properties (such as timescale) to a possible range within a
particular brain region … while task requirements, input statistics, short-term synaptic plasticity,
and neuromodulation can then shift timescale within this range."* `[SOURCE]`

**For #532 this is the awkward one.** A timescale that moves 20% with task demand is not a fixed
structural property of a unit's neighbourhood at all, and a bar that reads τ off a resting operator
is reading one point in a range. `[INFERENCE]`

### 4.3 The functional-breadth null (§1.2), restated here because it belongs on this side

Lurie et al.'s FC participation coefficient result is a **pre-registered failure of the breadth
hypothesis in its across-communities form**, replicated in two datasets. It is the cleanest negative
evidence in this document and it is negative evidence about exactly the formulation #590 proposes.
`[SOURCE]`

---

## 5. Learned, variable-radius neighbourhoods in graph learning

### 5.1 Per-node learned radius: JK-Nets

*Source: Xu, K., Li, C., Tian, Y., Sonobe, T., Kawarabayashi, K.-i., & Jegelka, S. (2018),
"Representation Learning on Graphs with Jumping Knowledge Networks", ICML 2018, PMLR 80;
arXiv:1806.03536. Read at source (PMLR PDF).*

The problem statement is the ticket's question in graph-learning vocabulary:

> "the range of 'neighboring' nodes that a node's representation draws from strongly depends on the
> graph structure, analogous to the spread of a random walk. To adapt to local neighborhood
> properties and tasks, we explore an architecture – jumping knowledge (JK) networks – that flexibly
> leverages, for each node, different neighborhood ranges" `[SOURCE]`

They formalise the radius as an **influence distribution** and connect it to random-walk spread:
*"our more formal analysis connects influence distributions with the spread of a random walk at a
given node, a well-understood phenomenon as a function of the graph structure and eigenvalues."*
`[SOURCE]`

The locality argument — why a bigger radius is not free:

> "A too rapid expansion may average too broadly and thereby lose information, while in other parts
> of the graph, a sufficient neighborhood may be needed for stabilizing predictions." `[SOURCE]`

> "Large radii may lead to too much averaging, while small radii may lead to instabilities or
> insufficient information aggregation." `[SOURCE]`

> "Random walks starting inside an expander converge rapidly in `O(log|V|)` steps to an
> almost-uniform distribution … the node representations will be representative of the global graph
> and carry limited information about individual nodes. In contrast, random walks starting at the
> bounded tree-width (almost-tree) part converge slowly, i.e., the features retain more local
> information." `[SOURCE]`

The mechanism: every layer's representation "jumps" to the last layer, and a per-node selector
(concatenation, max-pooling, or LSTM-attention) picks which radius that node uses. *"If this is done
independently for each node, then the model can adapt the effective neighborhood size for each node
as needed."* `[SOURCE]`

**Note what this is and is not.** JK-Net does not learn a continuous radius; it learns a **selection
over a fixed ladder of integer radii** `1..k`. Locality is preserved not by damping but by *keeping
the short-radius representation available and letting the node choose it*. `[INFERENCE]`

### 5.2 Continuous learned radius: ADC

*Source: Zhao, J., Dong, Y., Ding, M., Kharlamov, E., & Tang, J. (2021), "Adaptive Diffusion in
Graph Neural Networks", NeurIPS 2021. Read at source (NeurIPS proceedings PDF).*

This is the closest published object to "a learned variable-radius neighbourhood" as #532 would
mean it. GDC (graph diffusion convolution) already widens the neighbourhood past one hop, but:

> "the neighborhood size in GDC is manually tuned for each graph by conducting grid search over the
> validation set, making its generalization practically limited. To address this issue, we propose
> the adaptive diffusion convolution (ADC) strategy to automatically learn the optimal neighborhood
> size from the data." `[SOURCE]`

The radius is a scalar with a closed form. For the heat kernel `H_t = e^{-(I-T)t}`, they derive
`r_h = Σθ_k k / Σθ_k = t`, and conclude:

> "This suggests that `t` is the neighborhood radius for the heat kernel based GDC, that is, `t`
> becomes a perfect continuous substitute for the hop number in multi-hop models." `[SOURCE]`

`t` is then learned by gradient descent, framed as bilevel optimisation, and — the part that matters
most for a sheaf-style architecture with per-channel structure:

> "we break the conventional assumption that all GNN layers and feature channels (dimensions) should
> use the same neighborhood size for propagation. We design strategies to enable ADC to learn a
> dedicated propagation neighborhood for each GNN layer and each feature channel" `[SOURCE]`

**A per-channel radius is a learned, heterogeneous, continuous locality parameter — precisely the
shape a Patchworks cell would need if breadth were the abstraction axis.** It exists, it is
differentiable, and it is cheap. `[INFERENCE]`

### 5.3 How locality survives a growing radius: the teleport term

*Source: Klicpera (Gasteiger), J., Bojchevski, A., & Günnemann, S. (2019), "Predict then Propagate:
Graph Neural Networks meet Personalized PageRank" (PPNP/APPNP), ICLR 2019; arXiv:1810.05997. Read at
source (arXiv PDF).*

This paper answers the ticket's sub-question — *how is locality preserved when the radius grows* —
in one sentence:

> "The teleport probability allows us to balance the needs of preserving locality (i.e. staying
> close to the root node to avoid oversmoothing) and leveraging the information from a large
> neighborhood. We show that this propagation scheme permits the use of far more (in fact,
> infinitely many) propagation steps without leading to oversmoothing." `[SOURCE]`

And the failure mode it exists to avoid, stated as the reason the radius cannot simply be enlarged:

> "Increasing the size of the neighborhood used by these algorithms, i.e. their range, is not
> trivial since neighborhood aggregation in this scheme is essentially a type of Laplacian smoothing
> and too many layers lead to oversmoothing" `[SOURCE]`

> "aggregation by averaging causes oversmoothing if too many layers are used. It, therefore, loses
> its focus on the local neighborhood" `[SOURCE]`

> "Introducing the teleport vector `i_x` allows us to preserve the node's local neighborhood even in
> the limit of infinitely many propagation steps." `[SOURCE]`

**The published answer to "how does locality survive breadth" is: a per-node restart mass that never
diffuses.** Not a bound on the radius — a permanent anchor at radius zero. That is a structurally
different move from ADR-0011's construction-time locality guard, and it is the one the field
converged on. `[INFERENCE]`

Note the direct collision with §4.1: APPNP's oversmoothing and Li & Wang's eigenvector
delocalization are the same phenomenon in two vocabularies — a widening neighbourhood drives every
unit toward the leading eigenvector and destroys the per-unit distinctions the widening was meant to
buy. Both fields independently name it, and both fix it with a *local* heterogeneity term
(teleport mass; recurrent-excitation gradient). `[INFERENCE]`

### 5.4 Sheaf learning: nothing found

`[ABSENT]` No sheaf-neural-network paper found in this pass defines a learned or variable-radius
neighbourhood. The sheaf line (Hansen & Gebhart's Sheaf Neural Networks; Bodnar et al.'s Neural
Sheaf Diffusion; Barbero et al.'s connection-Laplacian and attention-based variants) learns the
**restriction maps on a fixed edge set** — the stalk-to-stalk geometry — and treats the
neighbourhood itself as given by the input graph. Searches for adaptive-radius or learned-locality
sheaf work returned only the fixed-neighbourhood line plus unrelated computer-vision
"dynamic receptive field" work. **This is a gap in the prior art, not a gap in this search that I
can rule out** — the search was keyword-driven and one pass deep, and the sheaf literature is small
and fast-moving. Treat it as "not found", not as "does not exist".

---

## 6. What is absent from the literature

Stated separately so nothing here is mistaken for a finding.

1. **No study read regresses τ on breadth and depth jointly and reports which survives.**
   Chaudhuri's lesion study is the nearest and it answers *both are necessary*, by ablation rather
   than by partial regression. `[ABSENT]`
2. **No source connects breadth to *abstraction* directly.** Every source here is about
   **timescale**. The step from "broadly connected units are slow" to "broadly connected units are
   abstract" is made nowhere in the material read, and the ticket's framing should not assume it.
   The nearest is the field's own theoretical framing — Hasson's temporal receptive windows, cited by
   Lurie et al. as *"longer timescales in higher-order association regions that integrate information
   over longer periods in order to support more complex information processing (such as the
   construction and maintenance of increasingly abstract representations)"* — which is an *assumed*
   link, stated in a literature review, not a measured one. `[ABSENT]`
3. **No mechanistic account was found in which breadth alone, with homogeneous units, produces
   separated *autocorrelation* timescales.** Gollo et al. do it for stability and persistence; the
   τ-producing models (Chaudhuri, Li & Wang) all require the local gradient. `[ABSENT]`
4. **Baria et al. (2013)'s low-pass-filter mechanism was not read at source.** It is the most
   directly transferable mechanism for a transport operator and it remains unverified here.
   `[ABSENT — not read]`
5. **No variable-radius sheaf work found (§5.4).** `[ABSENT]`

---

## 7. Sources, with what was reached

| Source | Reached how | Status |
|---|---|---|
| Chaudhuri et al. (2015), Neuron 88(2):419–431, doi:10.1016/j.neuron.2015.09.008 | author PDF at cns.nyu.edu, full text | read at source |
| Li & Wang (2022), PNAS 119(6):e2110274119, doi:10.1073/pnas.2110274119 | author PDF at ins.sjtu.edu.cn, full text | read at source |
| Lurie, Pappas & D'Esposito (2024), Hum Brain Mapp 45(2), doi:10.1002/hbm.26587 | full-text passages via Scholar Gateway (Wiley corpus); open access | read at source |
| Gollo et al. (2015), Phil Trans R Soc B 370:20140165, doi:10.1098/rstb.2014.0165 | arXiv:1502.04455 preprint (journal 403, PMC CAPTCHA) | read at source, preprint version |
| Gao, van den Brink, Pfeffer & Voytek (2020), eLife 9:e61277, doi:10.7554/eLife.61277 | eLife article page | read at source |
| Murray et al. (2014), Nat Neurosci 17(12):1661–1663, doi:10.1038/nn.3862 | abstract + Chaudhuri's in-text characterisation | **not read in full** (paywalled) |
| Manea et al. (2022), eLife 11:e75540, doi:10.7554/eLife.75540 | PMC page, fetched summary | **low confidence**, not quoted above |
| Xu et al. (2018), ICML/PMLR 80; arXiv:1806.03536 | PMLR PDF, full text | read at source |
| Zhao et al. (2021), NeurIPS 2021, "Adaptive Diffusion in Graph Neural Networks" | NeurIPS proceedings PDF | read at source |
| Klicpera/Gasteiger et al. (2019), ICLR; arXiv:1810.05997 | arXiv PDF, full text | read at source |
| Baria et al. (2013); Fallon et al. (2020); Sethi et al. (2017); Demirtaş et al. (2019) | cited by Lurie et al. only | **not read** |

Scholar Gateway was used for corpus search over the Wiley full-text corpus; every quotation above is
from a passage returned by it or from a PDF extracted locally. Two literature databases (PubMed,
Europe PMC) and two publisher sites (Nature, Royal Society) were unreachable by the fetcher and are
recorded as such rather than worked around.

---

## 8. Registers consulted

- **open-problems**: [#330](https://github.com/NGL321/patchworks/issues/330) matched directly —
  *"Core cells carry long-range agreement **or** slow state, not both, and which one gives way is
  decided by nothing."* §3 and §4.1 bear on it: the prior art says the two are in genuine tension
  (breadth delocalizes; localization is what makes state slow), which means #330's "decided by
  nothing" has a candidate decider in the literature — the local heterogeneity term. Nothing here
  resolves #330 and this pass mints nothing.
  [#344](https://github.com/NGL321/patchworks/issues/344) (dwell/τ coupling) is adjacent but
  untouched.
- **proposed-solutions**: [#322](https://github.com/NGL321/patchworks/issues/322) matched on
  `@shape` — *"Temporal abstraction with no retention gradient"* and *"at ratio 1 there is no
  hierarchy, only a deep feedforward reflex arc"*. This pass supplies literature relevant to that
  shape but proposes nothing and takes nothing.
- **dismissed-solutions**: nothing relevant. No `refused` or `failed` entry touches neighbourhood
  radius, breadth, or timescale-from-topology.
