# Citation pass: graph topology (patchworks#31)

Validates the design closed in patchworks#8 (`docs/spec/06-graph-topology.md`,
`docs/adr/0006-boundary-cell-stalks-are-world-shaped.md`). Citations validate after the fact per
the map's citation-sequencing rule; this document does not reopen the design. Where a source
threatens a committed claim it is flagged and a revision ticket is recommended, not opened.
Vocabulary follows `CONTEXT.md`: Patchworks' side of every comparison is described in its own terms
(cell, node stalk, communication lane, restriction map, chart, disagreement, reconciliation,
boundary cell, predicting cell, relay cell, taper, dome); the prior art's side in its own field's
terms. Where a
source could not be reached, that is stated. NTK is not cited, per the ticket.

## Headline verdict, stated plainly

**Six of the seven claims survive. The central one — #8's negative result that "relays solve reach,
and reach is not what is squeezed" — survives its own arithmetic and fails the literature's, because
the over-squashing literature does not measure reach with the diameter that #8 uses to dismiss it.**

Di Giovanni et al. (2023) run exactly the three-way decomposition #8 needs — width, depth, topology
— and conclude that "the graph topology plays the greatest role, since over-squashing occurs
between nodes at **high commute (access) time**" (abstract). Commute time is not hop distance. The
dome's diameter is small (~9 hops) but its *effective resistance* between distant rim cells is
enormous, because every path between two opposite L1 vision cells is forced through the twenty
L2→L3 edges that constitute the entire cut into the core. Relays across L4–L6 add parallel paths and
therefore **lower** commute time — which is, in this literature's own terms, exactly widening the
funnel, not merely shortening the wire. #8's sentence "a relay makes distant things closer without
widening the funnel" is true of hop distance and false of the quantity the literature actually uses.

The engineering decision (build none in the proof of concept) is untouched: the spec pre-specifies
the relay intervention and gates it on measurement, which is the right posture. What does not
survive is the *reason given*. This is the one place a source threatens a committed claim.

## Claim 1 — Long-range transport in graph neural networks

**Is the documented failure mode about distance, channel width, or receptive-field growth?** All
three, in a specific order, and the canonical definition binds them together rather than separating
them.

- Alon & Yahav (2021), the paper that named it: "one of the major problems in training GNNs was
  their struggle to propagate information between distant nodes"; "This bottleneck causes the
  **over-squashing of exponentially growing information into fixed-size vectors**" (abstract).
  Receptive-field growth *and* fixed channel width, indexed by distance — one phenomenon.
- Topping et al. (2022): over-squashing "has been heuristically attributed to graph bottlenecks
  where the number of k-hop neighbors grows rapidly with k"; their result is that "negatively curved
  edges are responsible for the over-squashing issue" (abstract). Structure, not distance alone.
- Di Giovanni et al. (2023) separate the axes: "Neural network width can mitigate over-squashing,
  but at the cost of making the whole network more sensitive"; "Depth cannot help mitigate
  over-squashing"; "The graph topology plays the greatest role, since over-squashing occurs between
  nodes at high commute (access) time" (abstract). Their recommended remedy is **graph rewiring**.

So #8 is right that width matters and that depth does not rescue it. It is wrong that reach and
squeeze are separable: in this literature the topological term *is* the squeeze term, expressed as
commute time / effective resistance, and the standard intervention against it is precisely the
rewiring #8 declines. The dome is a near-worst case for that metric — a single narrow cut with no
alternative routes — so a source-faithful reading predicts relays would help, contra #8.

**Does the analysis survive in a recurrent message-passing system?** Yes, and it has been done.
Arroyo et al. (2025) restate the definition unchanged for the recurrent setting: "Over-squashing
describes the difficulty of propagating information across faraway nodes, as the exponential growth
in a node's receptive field results in many messages being compressed into fixed-size vectors."
Their Theorem 5.1 factors sensitivity into "a term based on the graph topology and a term dependent
on the model dynamics," and "reinterpreting any GNN as a recurrent model enables direct control of
the Jacobian spectrum, which mitigates this issue." Recurrence does not dissolve over-squashing; it
moves the leverage from topology to dynamics. Patchworks is recurrent with unit-delay bidirectional
edges, so that term is available to it — a genuine mitigation the spec does not claim, and a better
argument for declining relays than the one it gives. The vanishing-gradient half of these results is
about backpropagated gradients and does not transfer to a local learning rule; **no source was found
analysing over-squashing under local learning rules.**

**Is over-squashing analysed for cellular sheaf networks specifically?** Recently, yes — and never
as a stalk-width problem.

- Bodnar et al. (2022), the reference neural-sheaf model, does **not** use the term at all; it is
  about over-*smoothing* ("some deeper GNNs produc[e] features that are too smooth to be useful").
- Bamberger et al. (2024), Bundle Neural Networks: message passing "suffers from pathological issues
  such as over-smoothing, over-squashing, and limited node-level expressivity" (abstract); their fix
  is continuous diffusion "operating at larger graph scales," and "discretized BuNNs are special
  cases of Sheaf Neural Networks."
- Cooperative Sheaf Neural Networks (2025): "we characterize the receptive field of CSNN and show it
  allows nodes to selectively attend (listen) to **arbitrarily far nodes** while ignoring all others
  in their path, potentially mitigating oversquashing" (abstract).

Both sheaf-side treatments frame over-squashing as reach. **No source was found stating a
sensitivity bound in terms of lane dimension `m`.** #8's framing — that the squeeze is the
`m`-sized restriction map and not the hop count — has no sheaf-side precedent either supporting or
refuting it. That is an honest gap, and it is where a novel result would live. (Full texts of both
sheaf papers were not reached; abstract-level only.)

## Claim 2 — The convolutional resemblance, and translation equivariance

**Survives.** Elsayed et al. (2020) define the contrast exactly as #8 needs it: "Locally connected
layers, which differ from convolutional layers **only in their lack of spatial invariance**, usually
perform poorly in practice" (abstract). Per-cell restriction maps make the dome locally connected in
this precise sense, so #8's "any result from the convolutional literature that rests on
weight-shared kernels is not ours to borrow" is correct, and the paper's practical warning — locally
connected layers underperform — is a live risk the spec does not record. (Full text not reached; the
abstract carries the definitional identification, not the mechanism proof.)

**But the thing #8 says *does* transfer transfers less than claimed.** Luo et al. (2016): the
effective receptive field "both has a Gaussian distribution and **only occupies a fraction of the
full theoretical receptive field**" (abstract). #8's taper arithmetic and the demo's one-hop /
four-hop depth prediction are nominal receptive-field arithmetic, which the literature finds
overstates real influence; and the derivation assumes a feedforward stack, which the dome's
bidirectional unit-delay edges are not. Read the depth prediction as an upper bound on locality.
Minor threat, worth a footnote rather than a revision.

## Claim 3 — Lateral reconciliation as the load-bearing difference

**Supported, and it is the best-cited claim in the spec.** Prior art makes #8's argument directly,
and comes out for it.

- Linsley et al. (2018) show feedforward CNNs "struggle in recognition tasks where co-dependent
  visual features must be detected over long spatial ranges," and that "a single hGRU layer matches
  or outperforms all tested feedforward hierarchical baselines including state-of-the-art
  architectures which have **orders of magnitude more free parameters**" (abstract). That is #8's
  claim — a CNN can only resolve cross-patch inconsistency by going deeper — demonstrated against
  parameter-matched depth.
- Spoerer, McClure & Kriegeskorte (2017): "recurrent neural networks outperform feedforward control
  models (approximately matched in parametric complexity) at recognizing objects, both in the
  absence of occlusion and in all occlusion conditions"; "at lower levels of occlusion, feedforward
  and lateral connections are sufficient... top-down connections become beneficial when the task
  involves recognizing digits under heavier levels of debris."

One caveat that should be recorded rather than glossed: Spoerer et al.'s **clutter** condition —
overlapping objects, the closest analogue to two patch cells disagreeing about one puck — is where
their recurrent advantage is weakest: "recurrent networks (though still better in absolute
performance) take similar hits to the error rate." The spec stakes the dome's main justification on
lateral reconciliation; the nearest published analogue supports it, but least strongly on the
condition Patchworks cares most about.

## Claim 4 — Cycles are good

**Supported, by a result the spec does not cite and should.** Bodnar et al. (2022) prove, for
orthogonal restriction maps, that "dim(H⁰) ≤ d", with **equality only when transport maps are
path-independent**. Path-independence is trivial holonomy — automatic on a tree, since there are no
independent routes to disagree. So on a tree the harmonic space is maximal: every configuration of
node stalks can be reconciled, and the sheaf tells you nothing the nodes could not have arranged.
That is #8's positive form, stated as a theorem in the sheaf-learning literature.

Hansen & Ghrist (2019) supply the identification: "H⁰(X,ℱ) is naturally isomorphic to Γ(X;ℱ), the
space of global sections," and by discrete Hodge theory "ker(Δᵏ) ≅ Hᵏ(C•)" — so `H⁰` is exactly the
zero-disagreement space, as `05-timescales.md` and `01-cell-and-sheaf.md` use it. Bodnar et al.
gloss it in the spec's own idiom: "those private opinions **x** for which all neighbours agree with
each other in the discourse space."

**Not found:** no source argues that high `β₁` is a *benefit*. The spec's reversal is original — the
literature supports its premise (cycles make disagreement irreducible) without drawing its
conclusion, irreducible disagreement being a nuisance in every cited setting. "Nothing in Patchworks
ever experiences `H¹`" is a design fact, not a citable claim: neither supported nor threatened.

## Claim 5 — Boundary stalks exempt from `n` (ADR-0006)

**Precedent exists, thinly.** The exemption runs against the standard model's stated assumption:
Bodnar et al. assume "all the stalks have a fixed dimension d" across nodes and edges, which is what
lets the sheaf Laplacian be an `nd × nd` matrix — a modelling convenience, not a requirement of
cellular sheaf theory, where stalks are arbitrary vector spaces (Hansen & Ghrist). The nearest
direct precedent is Heterogeneous Sheaf Neural Networks (2024), which handles graphs
"whose nodes and edges can belong to different types and feature spaces" by "assigning type-aware
local feature spaces and learning restriction maps conditioned on node features, node types, and
edge types." That is ADR-0006's move — a distinct kind of node keeps its own stalk and reaches the
rest of the graph through an ordinary restriction map. **Not verified:** whether HetSheaf's
type-aware stalks differ in *dimension* or only in basis; the fetch did not settle it, and the full
text was not reached. Recorded as partial support.

## Claim 6 — Broadcast subspaces

**Unpublished, as suspected; half of it is now published under another name.**

Searches for a Nancy McKenzie publication on "messaging subspaces" / "broadcast subspaces" returned
nothing — no paper, preprint, or abstract. The only McKenzie hit was Sam McKenzie on hippocampal
ripple propagation, a different person and topic. The paired taxonomy "messaging subspace /
broadcast subspace" returned no primary source using those terms. **Record as unpublished
conference-talk provenance.**

The dimensionally-reduced, largely point-to-point half has strong published support under the name
*communication subspace*: Semedo et al. (2019, Neuron 102:249–259) identify a low-dimensional
subspace of source-population fluctuations most predictive of target-population fluctuations, and
propose it as "a general, population-level mechanism by which activity can be selectively routed
across brain areas." (Full text not reached — cell.com 403, CMU PDF unparseable; abstract only, two
attempts.) Kang et al. (2024) extend it to whole-cortex models: "Long-range projections can
selectively route a small number of dimensions of neural dynamics while maintaining others private."

The corticothalamocortical **broadcast** half has **no source found**. Kang et al. explicitly scope
it out — the model "does not include cell types" and is corticocortical only. The spec's first
relay finding ("the core already is the broadcast subspace... a better match than chords across the
surface would be — the thalamus is a deep low-dimensional shared space, not a shortcut wire") is
therefore resting entirely on unpublished provenance. It should be marked as such in the spec; it is
currently written with the same confidence as the cited findings around it.

## Claim 7 — The private-dimension gradient: independent arithmetic check

Recomputed from `06-graph-topology.md` alone, using `dim H⁰ ≥ Σ_v max(0, n − Σ_{e∋v} m_e)` with
`n = 32`, interior `m = 4`, boundary `m = 8`. **The headline holds; three corrections.**

| cell | degree | `Σ_e m_e` recomputed | guaranteed private dim | spec |
|---|---|---|---|---|
| L1 vision (interior) | 4 down + 4 lateral + 1 up = 9 | 4·8 + 4·4 + 4 = **52** | 0 | 0 ✓ |
| L1 vision (lattice corner) | 4 + 2 + 1 = 7 | 32 + 8 + 4 = **44** | 0 | ✓ |
| L2 vision (interior) | 4 + 4 + 1 = 9 | 4·4 + 4·4 + 4 = **36** | 0 | 0 ✓ |
| L2 vision (lattice corner) | 4 + 2 + 1 = 7 | 16 + 8 + 4 = **28** | **4** | 0 ✗ |
| L3–L6 core, degree 6 | 6 | 6·4 = **24** | 8 | ~8 ✓ |
| L7 apex, as tabulated | **4** | **16** | 16 | ~16 ✓ |
| L7 apex, at "uniform degree ~6" | 6 | 24 | **8** | ~16 ✗ |

1. **The apex figure requires degree 4, which contradicts "Core. Uniform degree ~6."** The
   "Connectivity" section commits the core to uniform degree ~6; the private-dimension table gives
   the apex `Σ_e m_e ≈ 16`, i.e. degree 4. At the stated uniform degree the apex gets 8, not 16, and
   the gradient across L3–L7 is flat rather than rising. The claim "rises to about sixteen at the
   apex" is a claim that the apex is *lower* degree than the rest of the core — which is plausible
   (it has no level above it) but is not what the spec says. **Two sections disagree; one must move.**
2. **"Zero at the rim" has exceptions.** The four corner cells of the 4×4 L2 lattice have only two
   lateral neighbours, giving `Σ_e m_e = 28` and a guaranteed private dimension of 4. Small, but the
   spec's "zero at the rim" is stated without qualification.
3. **`χ ≈ +1000` is right, but only under a reading the spec does not state.** Counting edges from
   the spec's own connectivity rules: 263 boundary edges at `m = 8`; 112 L1 lateral; 64 L1→L2
   vision; 24 L2 lateral; 16 L2→L3; ~14 somatomotor; ~170 internal to the 60-cell degree-6 core —
   **~663 edges**, consistent with the spec's "roughly 690". Mean degree over predicting cells is
   (2·400 + 263)/150 ≈ **7.1**, matching "~7" ✓. Then `Σ_v n = 150 · 32 = 4800` and
   `Σ_e m_e = 2104 + 448 + 256 + 96 + 64 + 56 + 680 = 3704`, giving **χ ≈ +1096** ✓. But this
   includes boundary-*incident* edges while excluding boundary *nodes*. Excluding boundary edges as
   well gives χ ≈ +3200, three times the stated figure. The spec says "`χ` must be computed over
   predicting cells only" without saying that this restricts the *node* sum only and that every edge
   in the graph still counts. As written the sentence licenses the wrong computation.

Everything else checks: 16×16 four-neighbour grid `β₁ = 480 − 256 + 1 = 225` ✓; base width
64·64·3 = 256·48 = **12,288** ✓; cell counts 70 + 20 + 60 = 150 predicting and 256 + 3 + 3 + 1 = 263
boundary ✓; a diameter near 9 hops is consistent with two L1 cells routing up-through-core-and-down.

**One figure the spec should add, because it is its own best evidence.** The per-tick capacity of
each cut, computed from the connectivity rules: `12,288 → 2,104 → 280 → 80`. The entire sensory
boundary reaches the core through **80 numbers per tick**, a 154:1 squeeze at a single cut, while the
farthest two predicting cells are only ~9 hops apart. That contrast — not the diameter figure alone
— is the quantitative form of "the taper is the real bottleneck", and it is also precisely the high
effective resistance that Di Giovanni et al. say relays would relieve. Both readings are supported by
the same number; the spec should carry it and own the tension.

## Recommended revision tickets (recommendations only; none opened)

1. **Rewrite the second relay finding against commute time, not diameter.** "Relays solve reach, and
   reach is not what is squeezed" fails on the literature's own metric: over-squashing is indexed by
   commute time / effective resistance (Di Giovanni et al. 2023), the dome's single narrow cut makes
   that quantity large despite a small diameter, and relays lower it. Keep the decision, replace the
   reason — the better available reason is Arroyo et al.'s recurrent-dynamics term, which Patchworks
   has and a feedforward MPNN does not.
2. **Reconcile the apex degree.** "Core. Uniform degree ~6" and "~16 at the apex" cannot both hold.
3. **Restate the `χ` rule.** Say explicitly: node terms over predicting cells only, edge terms over
   every edge in the graph.
4. **Mark the broadcast-subspace finding as unpublished provenance** in `06-graph-topology.md`, and
   cite Semedo et al. (2019) / Kang et al. (2024) for the messaging-subspace half only.
5. *(Optional)* Record Luo et al.'s effective-receptive-field caveat against the acceptance demo's
   one-hop/four-hop depth prediction, and Elsayed et al.'s locally-connected performance warning
   under "Known exposure".

## Sources

Alon & Yahav 2021, ICLR, arXiv:2006.05205 · Topping et al. 2022, ICLR, arXiv:2111.14522 ·
Di Giovanni et al. 2023, ICML, arXiv:2302.02941 · Arroyo et al. 2025, NeurIPS, arXiv:2502.10818 ·
Bodnar et al. 2022, NeurIPS, arXiv:2202.04579 · Hansen & Ghrist 2019, J. Appl. Comput. Topol. 3:315,
arXiv:1808.01513 · Bamberger et al. 2024, arXiv:2405.15540 · Cooperative Sheaf Neural Networks 2025,
arXiv:2507.00647 · Heterogeneous Sheaf Neural Networks 2024, arXiv:2409.08036 · Linsley et al. 2018,
NeurIPS, arXiv:1805.08315 · Spoerer, McClure & Kriegeskorte 2017, Front. Psychol. 8:1551 ·
Luo et al. 2016, NeurIPS, arXiv:1701.04128 · Elsayed et al. 2020, ICML, arXiv:2002.02959 ·
Semedo et al. 2019, Neuron 102:249–259 (abstract only; full text 403) · Kang et al. 2024,
PMC11566003. NTK deliberately not consulted, per the ticket.
