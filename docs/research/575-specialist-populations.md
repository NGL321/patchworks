# Literature pass: populations of narrow specialists, and composing across mismatched abstraction (patchworks#575)

Part of map [#532](https://github.com/NGL321/patchworks/issues/532). Opened by
[#565](https://github.com/NGL321/patchworks/issues/565)'s resolution.

**Standing instruction.** *"Look at the literature to see what supports the things we've described,
without constraining on those topics we have not described."* This is not a confirmation pass.
Findings that contradict the project's framing are reported first and loudest in each section.

**Scope.** Four areas from the ticket: (1) narrow units / wide populations and what governs joint
capacity; (2) when low rank is correct versus pathological; (3) composition across mismatched
degrees of abstraction; (4) *reach* — perturbation propagation that stays distinguishable at
distance — as a named measure with instruments.

**This document does not rule on the architecture.** That is B27's job. It reports what the
literature says and separates that from what we might hope it says.

---

## Reading-depth key

#148's key, extended by one qualifier because much of this pass ran through search rather than
full-text fetch.

- **[FULL]** — paper body read (PDF text or HTML extracted).
- **[ABS]** — authoritative abstract / landing page fetched and read.
- **[CITE]** — existence and bibliographic details confirmed; the substantive claim below comes from
  a search-engine synthesis of the source, **not** from reading the source. Treat every [CITE]
  claim as needing re-verification before it is quoted in a design document.

Several PDFs (Gatsby, PNAS, Penn preprints, the Bethge 2002 reprint) were unfetchable or
unparseable in this environment; those are marked and their claims kept correspondingly weak.

---

## 1. Narrow units, wide populations

### 1.1 The result that most directly supports our framing

**Elhage et al. (2022), "Toy Models of Superposition", Transformer Circuits Thread
(arXiv:2209.10652).** [ABS, partial FULL]

What it actually says: a network with `n` dimensions can linearly represent substantially more than
`n` features by placing each feature on its own (nearly-orthogonal) direction and relying on the
features being *sparse* — rarely co-active. The paper measures capacity as "dimensions per
feature", `D* = n / ||W||²_F`, observes a **phase change** from a dense regime (few features, each
getting a full dimension, orthogonal) to a sparse regime (superposition, features arranged in
specific polytope geometries — antipodal pairs, triangles, pentagons), and shows the cost is
**interference** between co-active features which the nonlinearity must filter. It also observes
that under pressure the model **drops** low-importance features entirely rather than degrading all
of them.

Why this matters to us: the picture in the ticket — "each cell is a narrow specialist emitting *how
much of my thing is happening*, and capacity lives in hundreds of chains carrying hundreds of
different directions" — **is** the superposition picture, and superposition is real and well
evidenced. A representation where each feature is one direction and one scalar magnitude is not a
degenerate representation; it is the standard account of how over-complete codes work.

What it does **not** say: that the *channel* between two modules may be rank one. Superposition is a
statement about a code living in a shared `n`-dimensional space. Every feature direction is present
in the same space simultaneously; the space is not rank one. This distinction is the hinge of the
whole pass and is taken up in §5.

Related, and directly on the "population of specialists" reading: **Chaudhari, Nuer & Thorstenson
(2025), "Sparsity and Superposition in Mixture of Experts" (arXiv:2510.23671, preprint, not
peer-reviewed).** [ABS] Finds that MoEs exhibit *more* monosemanticity (less superposition) than
dense models with equal active and total parameters, but that dense models achieve consistently
lower reconstruction loss (gap ~0.03–0.08, narrowing with more experts), and that MoEs "achieve the
same efficiency in packing features as the dense models for the same total parameters". Read
plainly: **specialisation buys interpretability, not capacity.** For a fixed parameter budget the
population-of-specialists arrangement packs no more than the entangled one. That is a mild negative
for the hope that a specialist population is intrinsically higher-capacity.

### 1.2 The results that contradict our framing

**Zhang & Sejnowski (1999), "Neuronal Tuning: To Sharpen or Broaden?", Neural Computation 11(1),
75–84.** [CITE — the Johns Hopkins reprint returned HTTP 403; claims below from multiple
independent secondary syntheses, and the result is standard.]

What it actually says: a general scaling rule for how the Fisher information of a population of
tuned units varies with tuning width `σ`, for a `D`-dimensional encoded variable, independent of
tuning shape, spike distribution, and allowing some correlated noise. **Sharpening tuning increases
Fisher information only for `D < 2`; at `D = 2` width is irrelevant; for `D ≥ 3` sharpening
*reduces* the information the population carries.** The mechanism: narrowing makes each unit look
more informative but reduces the number of simultaneously active units, and in higher dimensions
that second effect dominates.

This is the sharpest counter-evidence in the pass. Our cells are explicitly *not* encoding a scalar:
a schema covers "a slice of dynamics", a perspective outlined by several features. If the variable a
lane must carry is three-dimensional or more, the literature says narrow is the wrong prior and
narrowing is a capacity loss, not a design choice. **We should not assume that "narrow specialist"
is free.**

**Rigotti, Barak, Warden, Wang, Daw, Miller & Fusi (2013), "The importance of mixed selectivity in
complex cognitive tasks", Nature 497(7451), 585–590.** [CITE]

What it actually says: prefrontal neurons are tuned to nonlinear *mixtures* of task variables, not
to single variables; this mixed selectivity is what gives the population a high-dimensional
representation, and the dimensionality is what determines the repertoire of input–output functions a
downstream linear readout can implement. Task aspects remain decodable from the population even
after single-cell selectivity to that aspect is removed. The paper frames mixed selectivity as
offering "a significant computational advantage over specialized responses".

This is a direct challenge to the ticket's central hypothesis in its strong form. The claim is not
that specialists are useless; it is that a population of *pure* (narrow, single-variable) units has
a **lower-dimensional** population geometry and therefore supports fewer readable functions than a
population of the same size with mixed selectivity. "Hundreds of chains carrying hundreds of
different directions" is exactly the pure-selectivity arrangement, and this result says it is the
low-capacity end of the design space.

**Stringer, Pachitariu, Steinmetz, Carandini & Harris (2019), "High-dimensional geometry of
population responses in visual cortex", Nature 571(7765), 361–365.** [CITE]

What it actually says: the population covariance eigenspectrum of mouse V1 responses to natural
images follows a power law, the `n`-th PC variance scaling roughly as `1/n`; and the authors prove
that if the spectrum decayed *more slowly* the code could not be smooth — small input changes would
dominate population activity. So the answer to "how much can a population hold" in a real cortex is
not a single number but a **spectrum with a critical decay exponent**: the code sits at the highest
dimensionality compatible with continuity of the encoding map.

Useful reframing for us: joint capacity is a property of the *eigenspectrum* of the population, not
a count of units and not the rank of any one unit. A population whose spectrum decays like `1/n` is
high capacity even though every individual principal direction after the first is "small".

### 1.3 Capacity theory in the symbolic / distributed-representation line

**Frady, Kleyko & Sommer (2018), "A Theory of Sequence Indexing and Working Memory in Recurrent
Neural Networks", Neural Computation 30(6), 1449–1513.** [ABS] Gives a theory of the retrieval
accuracy and information capacity of superposed high-dimensional vectors under crosstalk noise, with
linear readout for analog data and winner-take-all cleanup for symbolic data; reports that diverse
VSA models have **universal** capacity properties, superior to what earlier analyses predicted.
This is the cleanest existing answer to "what governs how many distinguishable things a population of
narrow units can hold": dimension, the number of superposed items, and the crosstalk-noise-limited
SNR of the readout. *We did not retrieve the closed-form constants; the arXiv landing page did not
carry them and the PDF was not parsed.*

**Smolensky (1990), "Tensor product variable binding and the representation of symbolic structures
in connectionist systems", Artificial Intelligence 46(1), 159–216** [CITE]; **Plate,
"Holographic Reduced Representations", IEEE Trans. Neural Networks 6(3), 1995, and the 2003 CSLI
monograph** [CITE]; **Kleyko, Rachkovskij, Osipov & Rahimi (2022/2023), "A Survey on
Hyperdimensional Computing aka Vector Symbolic Architectures", Parts I and II, ACM Computing
Surveys** [ABS].

Reported trade-off worth recording as a negative: TPR binding is exact and interference-free but its
dimensionality grows **exponentially with structure depth**; HRR fixes the dimension by accepting
lossy retrieval, and while Plate's original theory predicted capacity growing linearly with
dimension, later work found naive HRRs do not achieve this in practice [CITE — this last clause
came from a secondary synthesis and should be checked before use].

### 1.4 What breaks

Failure modes named in this literature, in the order we are most likely to hit them:

- **Interference / crosstalk.** Inherent to superposition; scales with co-activity. Mitigated only by
  sparsity or by a nonlinear cleanup step (Elhage et al. 2022; Frady et al. 2018).
- **Feature dropping.** Under capacity pressure a network does not degrade all features; it stops
  representing the least important ones entirely (Elhage et al. 2022).
- **Routing / expert collapse and dead experts.** In sparse MoE, the gate converges to a few experts
  and the rest die; standard mitigations are auxiliary load-balancing losses (Shazeer et al. 2017)
  and single-expert routing with a balancing loss (Fedus, Zoph & Shazeer, Switch Transformer, JMLR
  2022). [CITE] There is also a distinct **representation collapse** failure in which experts
  converge to similar representations and specialisation is lost (Chi et al., "On the Representation
  Collapse of Sparse Mixture of Experts", NeurIPS 2022, arXiv:2204.09179) [CITE].
- **Dimensionality ceiling from pure selectivity** (Rigotti et al. 2013) — the ceiling nobody exceeds
  by adding more narrow units, because the population dimensionality, not the unit count, is what
  bounds the readable function repertoire.

---

## 2. When narrow is right, and when it is a pathology

**Headline: the field does not have a clean, general, unit-local criterion. It has a task-relative
one, and it has a strong result that the unit is the wrong place to look.**

### 2.1 The unit is probably the wrong level of description

**Kriegeskorte & Wei (2021), "Neural tuning and representational geometry", Nature Reviews
Neuroscience 22, 703–718 (arXiv:2104.09743).** [ABS — arXiv landing page fetched.]

Verbatim from the fetched text: *"the tuning induces the geometry, but different sets of tuned
neurons can induce the same geometry"*, and *"the geometry determines the Fisher information, the
mutual information, and the behavioral performance of an ideal observer."*

What this actually says: single-unit tuning is **not an invariant** of the representation. Two
populations with completely different per-unit tuning profiles can be informationally identical.
What is determined by, and determines, downstream performance is the population geometry.

Consequence for us, and it cuts both ways: our "a cell's own state reads near rank one over long
windows" statistic is measuring a quantity that the literature says is not the thing that bounds
what the system can do. That weakens the alarm about per-cell rank one — **and** it weakens the
proposed defence, because "each cell is a narrow specialist" is likewise a claim about a
non-invariant. The invariant question is what the *joint* geometry across cells is, and neither the
alarm nor the defence has measured that.

### 2.2 The closest thing to a principled criterion: low rank is right when it is *sufficient*

**Papyan, Han & Donoho (2020), "Prevalence of neural collapse during the terminal phase of deep
learning training", PNAS 117(40), 24652–24663.** [CITE] Describes a terminal-phase state in which
within-class variability of last-layer activations collapses to zero, class means go to a simplex
equiangular tight frame, and the classifier collapses onto the class means. This is an extreme
low-rank state (effectively `C−1` dimensions for `C` classes) that arises *because the network has
succeeded*, not because it failed. It is the strongest existence proof that low rank can be the
correct terminal state.

But: **Hui, Belkin & Nakkiran (2022), "Limitations of Neural Collapse for Understanding
Generalization in Deep Learning" (arXiv:2202.08384).** [CITE] Neural collapse occurs on the train
set but not the test set; the authors give simple realistic experiments where **training longer
produces worse last-layer features as measured by transfer to a downstream task**, and report
preliminary evidence of "cascading collapse" into earlier layers.

Put together, this is as close to a principled account as the field has, and it is task-relative:
**low rank is correct exactly when the discarded directions are irrelevant to everything the
representation will be asked to do, and pathological the moment the representation is asked for
anything else.** There is no intrinsic, task-free property of a unit that separates the two. For a
system whose stated goal is several perspectives live at once and whose downstream demands are open,
this criterion leans against tolerating collapse.

Corroborating negative on the neighbouring theory: **Saxe, Bansal, Dapello, Advani, Kolchinsky,
Tracey & Cox (2018), "On the Information Bottleneck Theory of Deep Learning", ICLR 2018 (extended in
J. Stat. Mech. 2019, 124020).** [CITE] Refutes three IB claims: distinct fitting/compression phases,
compression causing generalisation, and compression arising from SGD. Networks that do not compress
still generalise, and vice versa. **So there is no established "compression is good" principle to
appeal to** when arguing that a collapsed representation is the healthy one.

### 2.3 Low rank as a diagnosed pathology, in architectures very close to ours

**Dong, Cordonnier & Loukas (2021), "Attention is not all you need: pure attention loses rank doubly
exponentially with depth", ICML 2021 (PMLR 139).** [CITE] Decomposes a self-attention network into
paths and proves a strong inductive bias toward token uniformity: **without skip connections or
MLPs, the output converges doubly exponentially to a rank-1 matrix.** Skip connections and MLPs stop
the degeneration.

**Oono & Suzuki (2020), "Graph Neural Networks Exponentially Lose Expressive Power for Node
Classification", ICLR 2020 (spotlight).** [CITE] Treats GCN forward propagation as a dynamical
system; when the weights satisfy conditions set by the spectrum of the augmented normalised
Laplacian, the output **approaches exponentially** the set of signals carrying only connected-
component and degree information.

**Jing, Vincent, LeCun & Tian (2022), "Understanding Dimensional Collapse in Contrastive
Self-Supervised Learning", ICLR 2022 (arXiv:2110.09348).** [CITE] Shows dimensional collapse — the
embedding occupying a strict low-dimensional subspace of the available space — occurs in contrastive
as well as non-contrastive SSL, and gives the dynamics that cause it.

These three are the relevant reference class for a **composed operator over seven hops on a graph
that reads rank one**. In all three, that signature is the diagnosed failure, not the design; in all
three there are known structural causes (repeated averaging / trivial edge structure / weight
dynamics) and known structural fixes (residual paths, non-trivial edge maps, decorrelation).

### 2.4 On "the learned carried subspaces are statistically indistinguishable from random frames"

This finding is *less* damning than it looks, and the literature says why. Random near-orthogonal
frames are close to optimal for superposition capacity — the Johnson–Lindenstrauss regime that
Elhage et al. (2022) invoke — so "indistinguishable from random" is not evidence that a subspace is
useless. What it *is* evidence of is that **nothing edge-specific was learned about which directions
two particular cells share**, which is a claim about the lane discovery mechanism, not about the
subspace's carrying capacity. That maps onto a design fact stated in the ticket (the mask is fixed at
construction and identical across a cell's edges) rather than onto a capacity fact.

---

## 3. Composition across mismatched abstraction

**We were least confident of our vocabulary here, and this section is where the search was widest.
The honest finding is that there is no single field-standard name for "composing representations
whose degrees of abstraction do not line up". There are four established literatures that each own
part of it, and one of them is already ours.**

### 3.1 The formal answer: abstraction is a lattice, not a ladder — and this is settled

**Cousot & Cousot (1977), "Abstract interpretation: a unified lattice model...", POPL 4; and Cousot
& Cousot (1979), "Systematic design of program analysis frameworks", POPL 6.** [CITE]

In abstract interpretation, abstractions of a concrete semantics are related to it by **Galois
connections** `(α, γ)`, and the abstractions themselves form a **lattice** — a partial order, not a
chain. Two abstractions can be incomparable: neither is "more abstract" than the other. Composing
them is a named construction, the **reduced product**, which unifies abstract values with the same
concretisation and yields a Galois insertion; and Galois connections **compose**, so soundness
composes.

This is the designer's `G`-as-exponent argument in its rigorous form, and it has been settled since
1979. "A very simple, low-complexity operation remains valuable at highly abstract levels" is just
the observation that the complexity ordering on operations and the abstraction ordering on domains
are **different orders**, and that the abstraction order is not total. The vocabulary we were
missing is: *lattice of abstract domains*, *Galois connection*, *reduced product*, *incomparable
abstractions*. **This supports our framing, and gives it a name and an algebra.**

### 3.2 The answer already inside our own mathematics

A cellular sheaf does not require stalks to have the same dimension. Different cells may carry
representations of genuinely different size and structure, and a restriction map `F_{v ⊴ e}` is
simply a linear map from one to a shared edge space. **Composition across mismatched abstraction is
a first-class citizen of the formalism we already chose.**

**Hansen & Ghrist (2019), "Toward a spectral theory of cellular sheaves", Journal of Applied and
Computational Topology 3(4), 315–358 (arXiv:1808.01513).** [ABS] Lifts the combinatorial graph
Laplacian to a Hodge Laplacian on a cellular sheaf and relates its spectrum to sheaf cohomology and
cell structure, with results on eigenvalue interlacing, sparsification, effective resistance and
sheaf approximation.

**Hansen & Ghrist (2021), "Opinion Dynamics on Discourse Sheaves", SIAM J. Applied Algebra and
Geometry 5(2), 315–342.** [CITE — the Penn preprint PDF would not parse.] Each agent has its own
**opinion space of its own dimension**, whose basis is the set of topics that agent cares about;
restriction maps translate between agents; diffusion under the sheaf Laplacian evolves both opinions
and communications; and the paper treats **controllability, reachability**, bounded confidence and
harmonic extension in that setting. This is our architecture's exact shape, including heterogeneous
per-node dimensionality, and it is where the word *reachability* already appears attached to sheaves.

### 3.3 The cognitive-science answer

**Gentner (1983), "Structure-Mapping: A Theoretical Framework for Analogy", Cognitive Science 7(2),
155–170.** [CITE] Interpretation rules depend only on **syntactic properties of the representation,
not on domain content**; the systematicity principle says a predicate belonging to a mappable system
of interconnecting relations is preferentially imported over an isolated predicate. This is a
mechanism by which a relational structure transfers between domains of wholly different concreteness
(solar system ↔ atom) — which is precisely composition across mismatched abstraction, and it says
the thing that licenses the transfer is *relational systematicity*, not level matching.

### 3.4 The program-synthesis answer, and where it disagrees with us

**Ellis, Wong, Nye, Sablé-Meyer, Morales, Hewitt, Cary, Solar-Lezama & Tenenbaum (2021),
"DreamCoder: bootstrapping inductive program synthesis with wake-sleep library learning", PLDI
2021.** [CITE] Builds a library of reusable abstractions by refactoring solved programs; each
learned function may call earlier ones, forming layered libraries in which low-complexity primitives
remain available and heavily reused at every level.

Partial disagreement worth recording: DreamCoder's abstraction structure is explicitly described as
**progressively deepening / hierarchically layered**. It is a DAG, not a strict ladder, but it is
also not level-agnostic — new abstractions are built *on* old ones. The literature therefore
supports "primitives stay valuable at all levels" more strongly than it supports "levels are not a
meaningful ordering at all".

### 3.5 Neuro-symbolic binding: level-agnostic by construction

VSA/HDC binding (Smolensky 1990; Plate 1995; Kleyko et al. 2022/2023, ACM CSUR, cited in §1.3) binds
a role to a filler by an algebraic operation over fixed-width vectors that is **indifferent to what
the filler is** — an atom, a whole structure, or a function. Nothing in the algebra requires the two
operands to be at comparable levels of abstraction. This is the constructive existence proof that a
representation scheme can compose across mismatched abstraction; the price is the interference and
capacity limits in §1.3.

### 3.6 The heterarchy line, reported with a warning

Searches for "abstraction is not a hierarchy" return the term **heterarchy** — cortical organisation
described as flowing upward, downward and laterally rather than as a strict hierarchy, and critiques
of fixed-level abstraction hierarchies in cognitive systems engineering. The results returned were
review-level and secondary and **none was verified to a primary source in this pass**. The word is
worth having; the citations behind it are not yet established. See §6.

---

## 4. Reach as a measure

**Yes, this is a named thing — several times over, in four literatures, with instruments. This is
the highest-yield section of the pass.**

### 4.1 Over-squashing (graph ML) — the closest match to our setting

**Alon & Yahav (2021), "On the Bottleneck of Graph Neural Networks and its Practical Implications",
ICLR 2021** [CITE]; **Topping, Di Giovanni, Chamberlain, Dong & Bronstein (2022), "Understanding
over-squashing and bottlenecks on graphs via curvature", ICLR 2022** [CITE]; **Di Giovanni, Giusti,
Barbero, Luise, Liò & Bronstein (2023), "On Over-Squashing in Message Passing Neural Networks: The
Impact of Width, Depth, and Topology", ICML 2023 (PMLR 202)** [CITE].

Over-squashing is defined as a node feature being **insensitive to information at distant nodes**,
and it is quantified exactly the way the ticket asks: by the **Jacobian sensitivity**
`∂h_v^{(L)} / ∂x_u` as a function of the graph distance between `u` and `v`. Over-squashing is the
regime in which this decays rapidly with distance. Di Giovanni et al. (2023) report that **width can
mitigate over-squashing but at the cost of making the whole network more sensitive; depth cannot;
and topology dominates — over-squashing occurs between nodes at high commute time**, with effective
resistance and Ricci curvature as the controlling quantities.

Two things follow for us. First, **the instrument the ticket describes already exists and is
standard**: differentiate the far state with respect to the near perturbation and look at how the
resulting Jacobian decays with hop count. Second, this literature also gives the *other* horn: the
field treats over-smoothing (too much mixing, collapse toward a low-dimensional invariant subspace)
and over-squashing (too little propagation) as a **paired tension**, not as independent bugs. A
seven-hop chain collapsing to a single direction is the over-smoothing horn.

### 4.2 Dynamical isometry and signal propagation (deep learning theory)

**Pennington, Schoenholz & Ganguli (2017), "Resurrecting the sigmoid in deep learning through
dynamical isometry: theory and practice", NIPS 2017 (arXiv:1711.04735).** [CITE]

**Dynamical isometry** is defined as the property that **all singular values of the input–output
Jacobian concentrate near 1**, so that every error vector approximately preserves its norm *and all
angles between different error vectors are preserved*. That last clause is the ticket's "stays
distinguishable at distance", stated exactly. The instrument is the **full singular spectrum of the
end-to-end Jacobian**, analysed with free probability / random matrix theory. Reported results:
ReLU networks cannot achieve dynamical isometry; sigmoidal networks can, but only with orthogonal
initialisation; networks that achieve it train orders of magnitude faster.

This is the named concept the project has been circling. **A composed operator whose spectrum
collapses to one non-zero singular value is the maximal violation of dynamical isometry.** The
lineage runs back through mean-field signal-propagation theory (Poole et al., NIPS 2016; Schoenholz
et al., ICLR 2017) [CITE] where there is an ordered phase with a finite correlation depth scale
`ξ_c`, in which all inputs converge to a single fixed point, and a critical point where `ξ_c`
diverges and information propagates arbitrarily deep.

### 4.3 Fisher memory (dynamical systems)

**Ganguli, Huh & Sompolinsky (2008), "Memory traces in dynamical systems", PNAS 105(48),
18970–18975.** [CITE — the PNAS PDF returned HTTP 403.] Applies Fisher information to a dynamical
system driven by a noisy time-dependent signal and defines the **Fisher Memory Curve**: the SNR
embedded in the current state about an input `k` steps back, relative to the input SNR. Its integral
is the **total memory capacity**. The paper reports that memory can be sustained by **transient
non-normal amplification**, i.e. by mechanisms invisible to eigenvalue-based stability analysis.

Directly usable: the FMC is "how distinguishable is a perturbation, at distance `k`" with the
distance being time rather than hops. Its shape is the reach curve. The non-normality point is worth
flagging — a chain of restriction maps is a highly non-normal operator, and eigenvalue intuitions
about it are unreliable.

### 4.4 Control theory and network science

- **Reachability / controllability.** The rank of the controllability (Krylov) matrix
  `[B, AB, A²B, …]` is literally "how many directions a perturbation injected at one place can move
  the state in, after composition". This is the same object as our composed-operator rank, under a
  name with a century of instruments behind it. **Liu, Slotine & Barabási (2011), "Controllability of
  complex networks", Nature 473(7346), 167–173** [CITE] gives the structural-controllability /
  driver-node machinery for large graphs. Note that Hansen & Ghrist (2021) already treat
  controllability and reachability **on sheaves** (§3.2).
- **Damage spreading / Lyapunov.** **Derrida & Pomeau (1986), "Random networks of automata: a simple
  annealed approximation", Europhysics Letters 1(2), 45–49** [CITE] gives a recursion for the
  evolution of the Hamming distance between two perturbed configurations — the canonical
  "does a perturbation stay distinguishable" measure, and the discrete analogue of a Lyapunov
  exponent. **Bertschinger & Natschläger (2004), "Real-time computation at the edge of chaos in
  recurrent neural networks", Neural Computation 16(7), 1413–1436** [CITE] ties computational
  capability in the time-series domain to the largest Lyapunov exponent approaching zero.

**Summary of §4:** reach has at least four names — over-squashing (graph ML), dynamical isometry /
depth scales (deep learning theory), Fisher memory (dynamical systems), reachability &
controllability (control theory) — and one shared instrument family: **the Jacobian of the far state
with respect to the near perturbation, read as a full spectrum rather than a norm.**

---

## 5. What this says about the architecture

Stated as plainly as the evidence allows, with the hedges named.

**5.1 The strong form of the "it is not a failure" reading is not supported, because it conflates a
code with a channel.** The literature that vindicates narrow specialists — superposition, sparse
coding, VSA, population codes — is about a **code**: many nearly-orthogonal feature directions
coexisting in one shared space, each carrying a scalar magnitude. Nothing in that literature says
the **channel** between two modules may be one-dimensional. In our system the composed restriction
map is the channel. A rank-one composed channel means every one of the "hundreds of chains" is a
one-dimensional pipe, and, critically, that the pipes are not independent in the way the defence
needs: they are all measured on the same stalks, and the same collapse mechanism acts on all of them.

**5.2 In our own mathematics there is a capacity bound and it is not the number of chains.** From
Bodnar, Di Giovanni, Chamberlain, Liò & Bronstein (2022), *Neural Sheaf Diffusion*, NeurIPS 2022
(arXiv:2202.04579) [ABS — theorem statements below fetched from the arXiv HTML; numbering may differ
from the published version and should be re-checked before quoting]:

- *Lemma 21:* "Solutions `X(t)` to the diffusion ... converge as `t→∞` to the orthogonal projection
  of `X(0)` onto `ker(Δ_F)`."
- *Lemma 6:* "Let `F` be a discrete `O(d)` bundle over a connected graph `G`. Then `dim(H⁰) ≤ d` and
  `dim(H⁰) = d` if and only if the transport is path-independent."
- *Proposition 12:* "Let `𝒢` be the set of connected graphs with nodes belonging to `C ≥ 3` classes.
  Then for `d ≥ C`, `ℋ_diag^d` has linear separation power over `𝒢`."
- *Proposition 13:* "... for all `d ∈ {2,4}`, `ℋ_orth^d` has linear separation power over `𝒢`" for
  `C ≤ 2d` classes.

Read against our situation: the asymptotic joint content of the whole population is the harmonic
space `ker(Δ_F)`, its dimension is capped by the stalk dimension `d`, and it attains that cap **only
when transport is path-independent**. Composed transport that collapses to one direction over seven
hops is a direct statement that path-dependence has destroyed all but one dimension of the agreement
space — **for the population, not per chain.** And Prop. 12 gives a capacity floor in our own units:
to separate `C` things you need stalk dimension at least on the order of `C`. Hundreds of chains do
not substitute for stalk dimension.

**5.3 The design facts in the ticket line up with the known causes.** The mask is fixed at
construction, identical across all of a cell's edges, and never reopens. Bodnar et al.'s entire
result is that **edge-specific, learned restriction maps** are what buy escape from the collapse; a
GNN with a trivial (shared, fixed) sheaf is precisely the regime that Oono & Suzuki prove collapses
exponentially and that Dong et al. prove collapses doubly exponentially without residual paths. Our
mask is closer to the trivial-sheaf regime than to the non-trivial one. That is not proof, but it is
the mechanism the literature would predict from the construction we described.

**5.4 The instrument objection in the ticket is real but does not rescue the reading.** The ticket
notes that the spectrum instrument never touches the state, so it says nothing about whether signal
travels the retained directions. True. But the whole over-squashing / dynamical-isometry literature
is built on exactly such a linearised operator (the Jacobian), and it is treated as the right object
because the spectrum of the composed linear map *is* what bounds the distinguishability of nearby
states after composition. §4 gives the state-touching version we are missing: perturb, propagate,
and measure the Jacobian spectrum and the decay of the perturbation with hop count.

**5.5 What genuinely supports our framing.** Three things, and they are not small. (a) Heterogeneous
composition across mismatched abstraction is formally settled — the lattice of abstract domains with
Galois connections and reduced products (§3.1) is the designer's `G`-as-exponent argument made
rigorous, and cellular sheaves with heterogeneous stalks (§3.2) are its representational form. (b)
Global agreement is *not* required and *should not* be the goal — the harmonic space is a proper
subspace by design, and Hansen & Ghrist's discourse sheaves make the "no shared worldview" case
first-class. (c) Reach is a legitimate success criterion with a mature instrument set (§4); we do
not have to invent it.

**5.6 The bottom line, unhedged.** The literature says **rank-one specialists as a *code* are
well-founded; rank-one composed transport as a *channel* is a named capacity trap** — over-smoothing
/ rank collapse / loss of dynamical isometry — with known causes that match our construction and
known structural fixes. Our measurement is of the second kind. The defence in the ticket would need
the composed rank to be a property of a decoded scalar readout rather than of the inter-cell
channel, and §5.2 says that even granting that, the population's joint agreement capacity is capped
by stalk dimension and attained only under path-independence, so the chain count does not buy it
back.

---

## 6. What we did not find / open

Reported plainly, because absence is a result.

1. **No paper was found that studies "a large population of very narrow specialists" as a named
   architectural class with its own capacity theory.** The question decomposes into population
   coding, superposition, and MoE, and each answers a piece. If there is a unified treatment, this
   pass did not surface it.

2. **No principled, task-free criterion separating "should be low-rank" from "failed to learn".**
   §2.2 gives the best available answer (task-relative sufficiency) and §2.1 gives the reason a
   unit-local criterion may not exist in principle. The IB route to such a criterion is refuted
   (Saxe et al. 2018). **This is a genuine gap in the field, not a gap in the search.** If we want to
   tell these apart on our own surface, we will have to define the criterion ourselves, and the
   literature says it will have to reference the downstream demands, not the unit.

3. **No standard name for "composition across mismatched abstraction" in the ML/representation
   literature.** The name exists in program analysis (reduced product of incomparable abstract
   domains) and the capability exists in sheaves and VSA, but there is no ML-side term of art. If
   B27 wants to search further, the productive query is not "abstraction" but "**incomparable
   abstract domains**", "**heterogeneous stalk dimension**", or "**multi-resolution / granularity
   mismatch**".

4. **The heterarchy line is unverified.** §3.6's claims came from review-level secondary sources
   only. McCulloch's 1945 "A heterarchy of values determined by the topology of nervous nets"
   (Bulletin of Mathematical Biophysics) is the usual origin cited for the term, but **this pass did
   not verify that paper or its content** and it should not be cited on our authority.

5. **Several primary sources were not reached.** Zhang & Sejnowski (1999) reprint — HTTP 403.
   Pouget/Deneve-line "Narrow versus wide tuning curves: what's best for a population code?" — the
   Gatsby PDF fetched but was encrypted and unreadable, so **its authorship and content are not
   confirmed here**; it is a real and relevant item and B27 should retrieve it. Bethge, Rotermund &
   Pawelzik (2002), "Optimal short-term population coding: when Fisher information fails", Neural
   Computation 14(10), 2317–2351 — bibliographic details confirmed via dblp/PubMed (PMID 12396565);
   the reported headline is that **Fisher information is a valid measure of code precision only if
   the dynamic range is sufficiently large** [CITE], which if it holds would qualify every
   Fisher-information-based capacity argument in §1 and §4.3 for short decoding windows. Not read.
   Ganguli, Huh & Sompolinsky (2008) — HTTP 403. Hansen & Ghrist (2021) preprint — PDF unparseable.

6. **Frady et al.'s closed-form capacity constants were not retrieved.** If B27 wants an actual
   number for "how many things can this population hold", that paper is where it is, and it needs a
   full-text read.

7. **Not searched, and possibly relevant:** modular/mixture-of-experts capacity theory outside the
   transformer setting; the sparse-coding capacity line (Olshausen & Field 1996 onward) and
   compressed-sensing recovery bounds as a capacity theory for narrow-unit populations; Kanerva's
   Sparse Distributed Memory; and the "grandmother cell versus distributed code" debate in its
   primary form (Barlow 1972; Quiroga et al. 2005). Each is adjacent; none was read.

---

## 7. Bibliography

Ordered as cited. Depth tag in brackets.

**§1**
- Elhage, N., et al. (2022). *Toy Models of Superposition*. Transformer Circuits Thread /
  arXiv:2209.10652. [ABS + partial FULL]
- Chaudhari, M., Nuer, J., & Thorstenson, R. (2025). *Sparsity and Superposition in Mixture of
  Experts*. arXiv:2510.23671 (preprint). [ABS]
- Zhang, K., & Sejnowski, T. J. (1999). *Neuronal tuning: to sharpen or broaden?* Neural Computation
  11(1), 75–84. [CITE]
- Rigotti, M., Barak, O., Warden, M. R., Wang, X.-J., Daw, N. D., Miller, E. K., & Fusi, S. (2013).
  *The importance of mixed selectivity in complex cognitive tasks*. Nature 497(7451), 585–590. [CITE]
- Stringer, C., Pachitariu, M., Steinmetz, N., Carandini, M., & Harris, K. D. (2019).
  *High-dimensional geometry of population responses in visual cortex*. Nature 571(7765), 361–365.
  [CITE]
- Frady, E. P., Kleyko, D., & Sommer, F. T. (2018). *A theory of sequence indexing and working memory
  in recurrent neural networks*. Neural Computation 30(6), 1449–1513. [ABS]
- Smolensky, P. (1990). *Tensor product variable binding and the representation of symbolic
  structures in connectionist systems*. Artificial Intelligence 46(1), 159–216. [CITE]
- Plate, T. A. (1995). *Holographic reduced representations*. IEEE Trans. Neural Networks 6(3). [CITE]
- Kleyko, D., Rachkovskij, D., Osipov, E., & Rahimi, A. (2022/2023). *A survey on hyperdimensional
  computing aka vector symbolic architectures*, Parts I & II. ACM Computing Surveys. [ABS]
- Shazeer, N., et al. (2017). *Outrageously large neural networks: the sparsely-gated
  mixture-of-experts layer*. ICLR 2017. [CITE]
- Fedus, W., Zoph, B., & Shazeer, N. (2022). *Switch Transformer*. JMLR 23(120). [CITE]
- Chi, Z., et al. (2022). *On the representation collapse of sparse mixture of experts*. NeurIPS 2022
  / arXiv:2204.09179. [CITE]

**§2**
- Kriegeskorte, N., & Wei, X.-X. (2021). *Neural tuning and representational geometry*. Nature
  Reviews Neuroscience 22, 703–718 / arXiv:2104.09743. [ABS]
- Papyan, V., Han, X. Y., & Donoho, D. L. (2020). *Prevalence of neural collapse during the terminal
  phase of deep learning training*. PNAS 117(40), 24652–24663. [CITE]
- Hui, L., Belkin, M., & Nakkiran, P. (2022). *Limitations of neural collapse for understanding
  generalization in deep learning*. arXiv:2202.08384. [CITE]
- Saxe, A. M., et al. (2018). *On the information bottleneck theory of deep learning*. ICLR 2018;
  J. Stat. Mech. (2019) 124020. [CITE]
- Dong, Y., Cordonnier, J.-B., & Loukas, A. (2021). *Attention is not all you need: pure attention
  loses rank doubly exponentially with depth*. ICML 2021, PMLR 139. [CITE]
- Oono, K., & Suzuki, T. (2020). *Graph neural networks exponentially lose expressive power for node
  classification*. ICLR 2020. [CITE]
- Jing, L., Vincent, P., LeCun, Y., & Tian, Y. (2022). *Understanding dimensional collapse in
  contrastive self-supervised learning*. ICLR 2022 / arXiv:2110.09348. [CITE]

**§3**
- Cousot, P., & Cousot, R. (1977). *Abstract interpretation: a unified lattice model...*. POPL 4.
  [CITE]
- Cousot, P., & Cousot, R. (1979). *Systematic design of program analysis frameworks*. POPL 6. [CITE]
- Hansen, J., & Ghrist, R. (2019). *Toward a spectral theory of cellular sheaves*. J. Applied and
  Computational Topology 3(4), 315–358 / arXiv:1808.01513. [ABS]
- Hansen, J., & Ghrist, R. (2021). *Opinion dynamics on discourse sheaves*. SIAM J. Applied Algebra
  and Geometry 5(2), 315–342. [CITE]
- Gentner, D. (1983). *Structure-mapping: a theoretical framework for analogy*. Cognitive Science
  7(2), 155–170. [CITE]
- Ellis, K., et al. (2021). *DreamCoder: bootstrapping inductive program synthesis with wake-sleep
  library learning*. PLDI 2021. [CITE]

**§4**
- Alon, U., & Yahav, E. (2021). *On the bottleneck of graph neural networks and its practical
  implications*. ICLR 2021. [CITE]
- Topping, J., Di Giovanni, F., Chamberlain, B. P., Dong, X., & Bronstein, M. M. (2022).
  *Understanding over-squashing and bottlenecks on graphs via curvature*. ICLR 2022. [CITE]
- Di Giovanni, F., et al. (2023). *On over-squashing in message passing neural networks: the impact
  of width, depth, and topology*. ICML 2023, PMLR 202. [CITE]
- Pennington, J., Schoenholz, S. S., & Ganguli, S. (2017). *Resurrecting the sigmoid in deep learning
  through dynamical isometry: theory and practice*. NIPS 2017 / arXiv:1711.04735. [CITE]
- Poole, B., Lahiri, S., Raghu, M., Sohl-Dickstein, J., & Ganguli, S. (2016). *Exponential
  expressivity in deep neural networks through transient chaos*. NIPS 2016. [CITE]
- Schoenholz, S. S., Gilmer, J., Ganguli, S., & Sohl-Dickstein, J. (2017). *Deep information
  propagation*. ICLR 2017. [CITE]
- Ganguli, S., Huh, D., & Sompolinsky, H. (2008). *Memory traces in dynamical systems*. PNAS 105(48),
  18970–18975. [CITE]
- Liu, Y.-Y., Slotine, J.-J., & Barabási, A.-L. (2011). *Controllability of complex networks*. Nature
  473(7346), 167–173. [CITE]
- Derrida, B., & Pomeau, Y. (1986). *Random networks of automata: a simple annealed approximation*.
  Europhysics Letters 1(2), 45–49. [CITE]
- Bertschinger, N., & Natschläger, T. (2004). *Real-time computation at the edge of chaos in
  recurrent neural networks*. Neural Computation 16(7), 1413–1436. [CITE]

**§5**
- Bodnar, C., Di Giovanni, F., Chamberlain, B. P., Liò, P., & Bronstein, M. M. (2022). *Neural sheaf
  diffusion: a topological perspective on heterophily and oversmoothing in GNNs*. NeurIPS 2022 /
  arXiv:2202.04579. [ABS, theorem statements quoted from arXiv HTML]

**§6 (named but not read)**
- Bethge, M., Rotermund, D., & Pawelzik, K. (2002). *Optimal short-term population coding: when
  Fisher information fails*. Neural Computation 14(10), 2317–2351. PMID 12396565. [CITE]
- McCulloch, W. S. (1945). *A heterarchy of values determined by the topology of nervous nets*.
  Bulletin of Mathematical Biophysics. **Unverified in this pass.**
