# Citation pass: coherent structures in driven dissipative fields, and the instruments for a sustained one (patchworks#374)

Runs the pass [#374](https://github.com/NGL321/patchworks/issues/374) scopes, opened by
[#144](https://github.com/NGL321/patchworks/issues/144)'s resolution. That resolution adopted the
map's position — **persistence is sustained, not stored**: long-time behaviour is a property of the
driven field, structures continually re-supplied against dissipation, and not of any cell's operator
run forward.

**The position is already taken, and nothing here is a reason it was taken.** It was reached from
[`docs/motivating-image.md`](../motivating-image.md) plus [#274](https://github.com/NGL321/patchworks/issues/274)
and [#166](https://github.com/NGL321/patchworks/issues/166), and per
[#1](https://github.com/NGL321/patchworks/issues/1)'s citation-sequencing rule citations **validate
design and never seed it**. Where a source below agrees with the position, the word used is
*corroborates*; no source below *motivates* anything. [#231](https://github.com/NGL321/patchworks/issues/231)
found six expired grounds in this record where that ordering slipped, and this document is written to
not add a seventh.

**This document does not revise closed design.** Where a source threatens a claim already made it is
flagged, per artifact, in *What this threatens* (§6).

**Vocabulary, and it is load-bearing.** The terms *coherent structure* and *nonlinear wave* appear
nowhere else in this repo; every term here is a fresh import. Following #167's practice, Patchworks'
side is written in `CONTEXT.md`'s terms (cell, chart, channel, node stalk, edge stalk, restriction
map, disagreement, piece) and the field's side in its own (dissipative soliton, discrete breather,
Turing mode, convective instability, non-normality, consistency), and **the two are deliberately not
blurred**. Nothing here enters `CONTEXT.md`; that would need a separate ruling.

**Two lines are kept out, deliberately.** The **thermodynamic-persistence** line owes its own pass
(`docs/motivating-image.md` marks it thin-provenance); §7.1 records the one place this pass touched
it and stops there rather than absorbing it. **Information cohomology** is ruled by
`docs/research/015-information-cohomology.md` an interpretive lens with **no relation** to a cellular
sheaf's `H¹`, and `CONTEXT.md` puts *topological invariant* on its Avoid list; §7.2 records that two
sources in this pass use topological vocabulary and how they are kept apart.

**Honesty about depth.** Several of the most load-bearing sources here were reached only at abstract
depth, and two large PDFs (Flach & Gorbach; Tél & Lai) defeated text extraction entirely. That is
stated at the tag on every source and again in *What could not be reached* (§8). No quotation appears
below that was not read.

## Reading-depth key

Following #148's and #167's key.

- **[FULL]** — paper body read (PDF text or HTML extracted).
- **[ABS]** — authoritative abstract / landing page only.
- **[CITE]** — citation confirmed to exist, text not reached.
- **[UNREACHED]** — existence not confirmed.

One extraction note. Two PDFs in this pass (Flach & Gorbach's *Physics Reports* 467 review, and
Tél & Lai's *Physics Reports* 460 review) were fetched but returned compressed streams that the
extractor could not decode, and one was returned password-protected. **Nothing is quoted from
either.** They appear at [ABS] or [CITE] and the claims attributed to them are held at the strength
their abstracts and indexed summaries support, no further. Where an argument in §5 would have been
stronger with their body text, that is said in place.

---

## Headline verdict, stated plainly

**Four findings, in descending order of what they cost.**

**1. The sharpest threat in the pass, and it lands on a quantity Patchworks has never measured.**
The network-science literature on pattern formation has a specific answer to the question *"can a
driven medium whose homogeneous state is linearly stable — i.e. contracting — still sustain
structure?"*, and the answer is **yes, but the mechanism is non-normality**. Muolo, Asllani, Fanelli,
Maini & Carletti build exactly that case: *"the instability is triggered by the non-normality of the
embedding network. The non-normal character of the dynamics instigates a short time amplification of
the imposed perturbation, thus making the system unstable for a choice of parameters that would yield
stability under the conventional scenario."* Asllani, Lambiotte & Carletti generalise it: *"initial
small disturbances may undergo a transient phase and be strongly amplified in linearly stable
systems"*, and add the warning that in such systems *"eigenvalues may become extremely sensible to
noise, and have a diminished physical meaning."*

#166 measured Patchworks' cell operator `K` at a non-normality of **0.05** — very nearly normal. Read
naively that is a finding *against* the position: the medium is contracting **and** normal, so the
one mechanism the field names for sustaining structure in a contracting medium is absent.

**It is not that simple, and the difference is the pass's most useful output.** In every source
above, the non-normality that does the work is a property of the **coupling structure** — the
network's adjacency or Laplacian — not of the node's own internal operator. Patchworks' analogue of
that object is not `K`. It is the **field-level operator**: the composition of restriction maps and
cell operators along the graph, on a directed rim-to-apex substrate with a measured 1.6–2.1x
apex-vs-rim decay asymmetry (#274) and near-rank-1 maps (`CONTEXT.md`, *channel*). **That operator's
non-normality has never been measured.** So the threat is real, it is not yet decided, and it
converts directly into the pass's strongest recommendation to #375: **measure the departure from
normality of the field-level operator, not the cell's.** §5.2.

**2. The instruments exist, they are discrete, and three of them are cheap.** The network-Turing
literature supplies a detection method that transfers almost without translation, because Patchworks
already owns a sheaf Laplacian ([#237](https://github.com/NGL321/patchworks/issues/237)): Nakao &
Mikhailov project the field onto **Laplacian eigenvectors**, and *"The Laplacian eigenvalue Λ_c of
the critical network mode corresponds to −q_c²"*, with the growth rate obeying *"λ_α = F(ε Λ_α)"* —
a dispersion relation over eigenvalues in place of wavenumbers. The nonlinear-dynamics literature
supplies the test that makes the position's *own claim* falsifiable: **consistency** / the replica
test (Uchida, McAllister & Roy; Lymburn et al.), which measures *"the degree of functional dependency
of a driven nonlinear system to its input"* — precisely the difference between structure the drive
sustains and structure a cell stores. And Butler & Goldenfeld supply the read-out for a
**noise-sustained** structure specifically: the **power spectrum** of fluctuations, with the finding
that noise *"greatly enlarges the region of parameter space where pattern formation occurs"* and
leaves a signature (*"fat tails not present in the mean field case"*) distinguishing noise-driven
structure from mean-field structure. §3 turns these into five concrete observables on Patchworks'
own objects.

**3. There is a hard limit on what #375 can conclude by running longer, and it is a named result.**
Tél & Lai's review of **supertransients** is the standing objection to every long-run persistence
claim in spatially extended systems: transient lifetimes *"grow rapidly with the system size, often
in an exponential fashion"*. A structure that survives 10⁴ ticks on a 414-cell dome is **not**
thereby shown to be sustained; it may be a transient whose lifetime is exponential in the graph size.
**The only discriminator is scaling, not duration** — #375 must run at least two graph sizes, and the
shape-free builder ([#163](https://github.com/NGL321/patchworks/issues/163)) makes that cheap. This
is the single most important methodological constraint the pass returns. §5.5.

**4. The field's own definition of the object contains a requirement Patchworks partly meets and
partly does not, and the shortfall is in a specific place.** Grelu & Akhmediev's definition of a
dissipative soliton — *"localized formations of an electromagnetic field that are balanced through an
energy exchange with the environment in presence of nonlinearity, dispersion and/or diffraction"* —
is a **double** balance: supply against dissipation, **and** a spreading term against a nonlinearity.
Patchworks has the first (drive against contraction, #274) and has a nonlinearity (`encode`, in the
loop every tick, per #167 §1). What it does **not** have, on the measurements the map carries, is
**mode selection**: a near-degenerate within-cell spectrum (eleven of twelve moduli inside ~0.13,
#166) is a nearly flat `F(εΛ)`, and a flat dispersion relation selects no mode. §5.3. This is a
finding against the position at full strength, and it is also, usefully, the thing instrument I in §3
measures directly.

### What this pass did not check

It does not re-open #148's Koopman findings, #166's or #274's measurements, or #167's
recurrence verdicts. It reads no source code and runs nothing. It does not enter the
thermodynamic-persistence line (§7.1) or the information-cohomology line (§7.2). It says nothing
about #357 beyond what #374's own body already says.

---

## 1. What the field means by a coherent structure in a driven dissipative medium

**Verdict: a well-defined object with a definition that is a balance condition, and the balance has
two halves, not one.**

**Grelu & Akhmediev**, *"Dissipative solitons for mode-locked lasers"*, **Nature Photonics 6, 84–92
(2012)**. **[ABS]**

The canonical modern review of the object. Its definition, verbatim from the abstract: *"Dissipative
solitons are localized formations of an electromagnetic field that are balanced through an energy
exchange with the environment in presence of nonlinearity, dispersion and/or diffraction."*

Three things in that sentence are worth separating, because Patchworks' side of the analogy meets
them unequally:

- **Energy exchange with the environment.** Continual supply against continual loss. Patchworks has
  this by construction: a fresh node stalk arrives every tick (`CONTEXT.md`, cell contract) and the
  chart loop is contracting (#274, `ρ_full` 0.708–0.908). This is the half of the balance #144's
  resolution reads as *the dissipation half, working*. **Corroborated**, and no more than
  corroborated — the position was adopted from the image and the measurements, not from here.
- **In presence of nonlinearity.** Patchworks has a nonlinearity, and #167 §1 established it is in an
  unusual and favourable place: `encode` is re-applied **inside** the loop every tick, which is the
  more expressive side of the SSM/RNN line. So this requirement is met, and met in a way this repo
  has already read at source.
- **Dispersion and/or diffraction.** A spreading mechanism for the nonlinearity to balance against.
  This is the term whose Patchworks analogue is least settled. The candidate is transport along the
  graph — restriction maps into edge stalks, and reconciliation — but nothing in the record has
  characterised it as a *spreading* operator with a dispersion relation. §3's instrument I is the
  measurement that would.

**Grelu & Akhmediev is a review of the continuum optical case and its transfer to a sheaf over a
graph is by analogy only.** It is cited here for the definition of the object, which is what #374
asked for, and not for any quantitative result.

A general note on this literature, recorded because it shapes what §3 could recommend: the standard
theoretical vehicle is the **complex Ginzburg–Landau equation** and master-equation variants of it.
That is a PDE on a continuum with a specific cubic-quintic nonlinearity. Its *results* do not
transfer to Patchworks; its *instruments* — dispersion relation, structure factor, localization
measure — do, because those are read off data rather than off the equation. §3 selects only
instruments of that second kind.

---

## 2. The discrete and network analogues, which are the directly relevant part

#374 names this as the most relevant part of the field, and it is right to. Three lines exist.

### 2.1 Network Turing patterns — the line that transfers best

**Nakao & Mikhailov**, *"Turing patterns in network-organized activator–inhibitor systems"*,
**Nature Physics 6, 544–550 (2010)**, arXiv:1005.1986. **[FULL — body read via ar5iv]**

This is the source #375 should build its first plot from, and the reason is that the translation to a
sheaf over a graph costs almost nothing.

The construction, verbatim: *"By introducing the network Laplacian matrix whose elements are given by
L_ij = A_ij − k_i δ_ij, where k_i = Σ_j A_ij is the degree of the node i, diffusive flux of the
species u to node i is expressed as Σ_j L_ij u_j = Σ_j A_ij (u_j − u_i)"*, and *"The eigenvalues Λ_α
and eigenvectors φ^(α) = (φ_1^(α), ⋯, φ_N^(α)) of the Laplacian matrix L_ij are determined by
Σ_j L_ij φ_j^(α) = Λ_α φ_i^(α)"*.

The key move, verbatim: *"The critical ratio σ_c in the networks is the same as in the classical
case. The Laplacian eigenvalue Λ_c of the critical network mode corresponds to −q_c², where q_c is
the wavenumber of the critical mode in the continuous media."* And the dispersion relation itself:
*"the growth rate λ_α of each mode depends only on the combination ε Λ_α of the diffusional mobility
ε and the eigenvalue Λ_α of that mode, i.e. we have λ_α = F(ε Λ_α)."*

**Why this is the right instrument for a sheaf over a graph, in Patchworks' own terms.** The
architecture already has the operator this construction needs, and it is not the graph Laplacian but
the **sheaf** Laplacian: `CONTEXT.md`'s *disagreement* entry states that disagreement collected
across every edge *"is the sheaf's coboundary, and its squared sum is the Dirichlet energy of the
sheaf Laplacian"*, and #237 has already computed with that operator's spectrum (its effective
resistance, off the edge-side Gram). So the eigenvectors φ^(α) that Nakao & Mikhailov expand in
**exist in this codebase already**. What has never been done is project the *field's* fluctuations
onto them and read a dispersion relation.

The paper's second finding is a warning about what the answer will look like, verbatim: *"In such
networks, Turing instability leads to spontaneous differentiation of the network nodes into
activator-rich and activator-low groups, but ordered periodic structures never develop."* **#375
should not go looking for a stripe.** On a graph the signature of a sustained structure is a
*partition* of cells into distinct sustained states, not a periodic spatial pattern; and Nakao &
Mikhailov find the partition is organised by **degree**, with *"a subset of nodes having close
degrees"* differentiating. The dome's degree distribution is not uniform across levels (#150), so
this is a concretely testable prediction on Patchworks' own substrate.

### 2.2 Discrete breathers — the lattice line, and its requirement

**Flach & Gorbach**, *"Discrete breathers — advances in theory and applications"*, **Physics Reports
467, 1–116 (2008)**. **[ABS — full PDF fetched but text extraction failed; nothing quoted from the
body]**

The lattice analogue of a coherent structure: time-periodic, spatially localized excitations of a
nonlinear lattice, generic rather than exceptional. The abstract-level statement carried here is that
these are solutions of **nonlinear** classical Hamiltonian lattices, time-periodic and typically
exponentially localized in space, and that both **nonlinearity and discreteness** are ingredients —
the review poses the question of why discreteness is needed at all once nonlinearity is present.

**Two things are held at abstract strength and no further, and both matter.** First, the existence
theory (MacKay–Aubry, via the anticontinuous limit) turns on a **nonresonance condition** — the
breather frequency and its harmonics must avoid the linear spectrum of the lattice. I could not read
that condition at source in this pass and it is therefore **not** used as a load-bearing argument
below; it is flagged in §8 as the single most valuable unreached item, because a near-degenerate
spectrum (#166) is exactly the configuration in which a nonresonance condition would be hardest to
satisfy. Second, the Hamiltonian case is *not* Patchworks' case; the driven-damped case is.

**Marín, Aubry & Floría**, *"Intrinsic localized modes: Discrete breathers. Existence and linear
stability"*, **Physica D 113, 283–292 (1998)**. **[CITE — abstract-level summary only, body not
reached]**

Recorded because it is the standard reference for **Floquet** stability analysis of a discrete
breather, and Floquet multipliers are the natural discrete-time read on whether a periodic structure
on a lattice is sustained or decaying — the direct analogue of what #375 needs. Its content is not
used as evidence here beyond that it exists and is the standard method.

### 2.3 Noise-sustained structure in a convectively unstable medium — the closest analogue found

This is the sub-line that most resembles Patchworks' situation, and #374's framing did not anticipate
it.

**Deissler**, *"Noise-sustained structure, intermittency, and the Ginzburg–Landau equation"*,
**Journal of Statistical Physics 40, 371–395 (1985)**. **[ABS]**

The concept: in a medium that is **convectively** unstable — a perturbation grows as it is carried
along, but at any fixed location it decays away — no structure survives on its own, and yet a
structure is continuously present, because noise is continuously re-amplified as it is swept
downstream. The abstract-level statement carried here is of *selective spatial amplification of
noise* producing spatially growing waves that form dynamic structures, with microscopic noise playing
an important role in the macroscopic dynamics.

**Huerre & Monkewitz**, *"Local and Global Instabilities in Spatially Developing Flows"*, **Annual
Review of Fluid Mechanics 22, 473–537 (1990)**. **[ABS]**

The taxonomy this sits in: such flows *"behave either as noise amplifiers or as oscillators"*. The
distinction is the one #375 actually has to make. A **noise amplifier** has no dynamics of its own;
its structure is a filtered image of whatever is being fed in, and it dies the instant the feed
stops. An **oscillator** has a self-sustained global mode and persists when the feed stops.

**This is a corroboration of the adopted position and a sharpening of it in the same breath.** The
position — *"structures continually re-supplied against dissipation"*, *"a cell is a filter, and it
is allowed to forget"* — is, in this taxonomy, a claim that Patchworks is a **noise amplifier**, and
the field says that is a real and well-studied way for a medium to hold structure. The sharpening is
that it is not the *only* one, that the field distinguishes the two sharply, and that the
distinguishing test is cheap: **remove the drive.** §3's instrument III is exactly that test, and it
is worth being clear that the position as written commits Patchworks to the *amplifier* answer — the
structure should die at the medium's own `τ` when the drive stops. If #375 finds it does not, the
position is not refuted but it is under-described.

---

## 3. Instruments: five observables for #375, on Patchworks' own objects

#374 asks for concrete, implementable observables over survey breadth, and asks that #375's choice of
what to plot come from here rather than from invention. This section is that answer. Each instrument
names its source, what to compute in `CONTEXT.md`'s terms, and what a positive and a negative read
look like. The first three are the recommended core; IV and V are the ones that make a positive read
trustworthy.

Two general constraints on all five, both from the record rather than from the literature. Every
long-time read in the record today is either per-cell (#274) or a decay (#242, #232, #214), so **all
five must be field-level and none may be a decay curve**. And all five must be read at runtimes far
past any cell's `τ` — #274 puts `τ` at 2.9–10.3 ticks, so "far past" means at minimum 10²–10³ ticks,
and §5.5 argues it must also mean *at two graph sizes*.

### Instrument I — the sheaf-Laplacian dispersion read

**Source: Nakao & Mikhailov [FULL], §2.1.**

**Compute.** Take the sheaf Laplacian whose spectrum #237 already computes. Diagonalise it; take the
eigenvectors φ^(α) with eigenvalues Λ_α. Each tick, project the field's node stalks (or the
disagreement field, which is the coboundary and is already computed) onto each φ^(α). Plot, against
Λ_α: the **time-averaged variance** of the projection over a long driven run, and its **growth rate**
over a window.

**Positive read (structure).** The variance-vs-Λ_α curve has a **peak at a non-extremal Λ_α**. That
is mode selection: the field is concentrating its energy at a particular scale of the graph, and it
is the single cleanest signature of a coherent structure on a network that this pass found.

**Negative read (no structure).** The curve is flat, or monotone in Λ_α. Flat means the field is
filtering white input with no scale preference — a medium with no structure of its own. Monotone
decreasing means the field is simply a low-pass filter of the drive, which is what a contracting
normal medium should look like and is exactly the null hypothesis §5 argues for.

**Why this one first.** It costs one eigendecomposition of an operator the repo already builds, and
its negative read is informative rather than merely null. It also directly measures the quantity §5.3
identifies as the sharpest threat — whether a near-degenerate spectrum leaves any mode selection at
all.

**One caution.** Nakao & Mikhailov's Λ_α is the *graph* Laplacian's, and Patchworks' is a *sheaf*
Laplacian, whose spectrum is not the graph Laplacian's and whose eigenvectors live in the direct sum
of node stalks rather than on nodes. The construction transfers because the argument is about
projecting onto an eigenbasis of the diffusion operator, and the sheaf Laplacian **is** the diffusion
operator here (`CONTEXT.md`, *disagreement*). But the correspondence `Λ_c ↔ −q_c²` is theirs and not
automatically Patchworks'; **#375 should plot against Λ_α and not attempt to convert to a
wavenumber.**

### Instrument II — the replica-consistency test

**Sources: Uchida, McAllister & Roy, *"Consistency of Nonlinear System Response to Complex Drive
Signals"*, Phys. Rev. Lett. 93, 244102 (2004) [ABS]; Lymburn, Khor, Stemler, Corrêa, Small & Jüngling,
*"Consistency in echo-state networks"*, Chaos 29, 023118 (2019) [ABS].**

Consistency is described in the second of these as *"an extension to generalized synchronization
which quantifies the degree of functional dependency of a driven nonlinear system to its input"*,
measured *"through a replica test"* on the high-dimensional response.

**Compute.** Run the whole field twice under the **identical** drive sequence, from two **different**
random initial charts. After a burn-in of many `τ`, correlate the two runs' node stalks cell by cell
and tick by tick. The per-cell correlation is that cell's consistency; the distribution over cells is
the field's consistency profile.

**Positive read for the position.** Consistency → 1 while a structure is present. The structure is a
functional of the drive, reproduced from any initial chart. That is *"persistence is sustained, not
stored"* made into a number: nothing about the structure was carried in the initial condition.

**Read against the position.** Consistency stays well below 1 *and* structure is present. Then part
of the structure is set by the initial chart and is being **stored**, not sustained.

**Why this is the most valuable of the five.** It is the only instrument here that tests the adopted
position's *own* content rather than merely detecting structure, and it is symmetric — it can come
back either way. #374 asks for validation and this is what validation looks like when it is
falsifiable. It is also nearly free: two runs and a correlation, no new machinery.

**One caution, and it is a real one.** In a **contracting** medium (which #274 says Patchworks is),
consistency is close to guaranteed — a contraction forgets its initial condition by construction, and
this is the echo-state property under another name, which #167 already read at source in Jaeger's
GMD 152. So a high consistency reading is **necessary but not sufficient**, and on its own it is
nearly vacuous. It becomes informative only **paired with instrument I**: consistency high *and* a
peaked dispersion curve is the finding; consistency high with a flat curve is just contraction.
**#375 must report the pair, never instrument II alone.**

### Instrument III — the drive-quench test

**Source: the amplifier/oscillator distinction, Huerre & Monkewitz [ABS], §2.3; and the "energy
exchange with the environment" half of Grelu & Akhmediev's definition [ABS], §1.**

**Compute.** Establish a structure under continual drive at t ≫ τ. Then **stop the drive** — hold the
rim's node stalks constant, or zero the arriving stalks — and measure how long the structure's
signature (instrument I's peak, or instrument V's localization) survives, in units of the cell `τ`
that #274 measured.

**Read.** Decay in ~τ means the structure was **sustained**: the field is a noise amplifier, the
structure was the drive's image, and the position is corroborated in its strongest form. Survival for
many τ means the field has a self-sustained mode — an oscillator — which the position does not
predict and does not forbid, and which would be a genuine and reportable finding. Instant collapse
faster than τ would indicate the read-out is measuring the drive rather than the field, and is the
main failure mode to check for.

**Why it belongs in the core three.** It is the only instrument that directly distinguishes *the
field holding shape* from *the drive being visible through the field*, which is the distinction the
position rests on. And it is the cheapest of all five to run.

### Instrument IV — the fluctuation power spectrum over the graph

**Source: Butler & Goldenfeld, *"Robust ecological pattern formation induced by demographic
noise"*, Phys. Rev. E 80, 030902(R) (2009), arXiv:0906.5535 [FULL — abstract read verbatim at
source; body not read].**

Their result, verbatim: *"demographic noise can induce persistent spatial pattern formation and
temporal oscillations"*, and *"demographic noise greatly enlarges the region of parameter space where
pattern formation occurs"*. The instrument is their discriminator, verbatim: *"To distinguish between
patterns generated by fluctuations and those present at the mean field level in real ecosystems, we
calculate the power spectrum in the noise-driven case and predict the presence of fat tails not
present in the mean field case."*

**Compute.** Instrument I gives variance per Laplacian mode; this adds the **temporal** axis. For
each mode α, compute the power spectral density of its projection time series over a long run. Report
the two-dimensional map (Λ_α, frequency) → power.

**Positive read.** A localized bright region in that map: power concentrated at a particular
`(Λ_α, ω)` pair. That is a structure with both a spatial scale on the graph and a timescale, which is
what a sustained coherent structure is. It is also the read that would show a **standing** or
**travelling** structure, which #374's scope explicitly names, since a nonzero ω at a selected Λ_α is
a travelling one.

**Why it matters that this source is about noise.** Butler & Goldenfeld's whole point is that
structure appears in a **stochastic** system across a parameter range where the deterministic
(mean-field) system says there is none. Patchworks' medium is noisy and driven, and the map's
measurements (#274's contraction) are the deterministic reading. **Corroboration, carefully worded:**
this source establishes that a deterministic stability calculation of the kind #274 performed does
not settle whether a noisy driven version of the same system holds structure. It does **not**
establish that Patchworks does. And note the caution — their fat-tail signature is specific to
demographic noise in a birth-death process, and there is no reason to expect the same tail exponent
here; **#375 should use the spectrum, not their tail prediction.**

### Instrument V — localization over cells

**Source: the discrete-breather line, Flach & Gorbach [ABS], §2.2. This instrument is the standard
localization read in that literature; it was NOT verified at source in this pass and is offered on
that basis.**

**Compute.** A participation ratio over cells: with `e_v` the energy (squared node-stalk norm, or
squared disagreement summed over a cell's incident edges) at cell `v`, report
`P = (Σ_v e_v)² / (N Σ_v e_v²)`. `P → 1` is a fully delocalized field; `P → 1/N` is energy on one
cell.

**Positive read.** `P` settles at a value well below 1 and **stays there** across a long driven run,
with the *identity* of the loaded cells stable. That is a localized sustained structure, and it is
the discrete analogue of the object the breather literature studies.

**Distinguish from the trivial cause.** A low `P` could simply reflect the dome's rim-to-apex decay
asymmetry (#274, apex decaying 1.6–2.1x faster than rim) — energy piling at the rim because that is
where it enters. **#375 must report `P` against a drive-only baseline**, e.g. the same graph with the
cell operators replaced by the identity, or the same `P` computed on the arriving stalks alone.
Without that baseline this instrument reads the substrate's geometry, not a structure.

**Held at low confidence.** Of the five, this is the one whose source I could not read. It is
included because it is the only one that speaks to *localization*, which is what "structure" means in
the lattice line, and because it costs nothing to compute. It should not carry a conclusion alone.

### Ranking, and one instrument deliberately not recommended

**If #375 can only do three: I, II, III.** I detects, II tests the position's own claim, III
separates the field from the drive. IV is the best fourth and is the one that would show a
*travelling* structure. V is a cheap addition with a weak provenance.

**Not recommended: dynamic mode decomposition.** DMD (Schmid, *"Dynamic mode decomposition of
numerical and experimental data"*, **J. Fluid Mech. 656, 5–28 (2010)** **[ABS — abstract read
verbatim]**) is the field's default coherent-structure extractor: *"A method is introduced that is
able to extract dynamic information from flow fields"*, whose modes *"can be interpreted as a
generalization of global stability modes"*, and it would give exactly the per-mode growth rates #375
wants. **It is nevertheless the wrong instrument to lead with here, on repo grounds and not on
technical ones.** DMD is the numerical core of Koopman mode decomposition; ADR-0023 moved the design
off the Koopman path and #148 read that literature at length. Putting a DMD plot on the demo surface
would re-introduce, as the headline read on the field, the vocabulary the design deliberately left —
and #144's resolution declined a *diagnostic* on exactly this ground, that *"a diagnostic that reads
a quantity the design says is not there teaches the demo surface to look in the wrong place"*. The
technical distinction is available (a DMD run *on the field's snapshots as a diagnostic* is not the
same thing as a Koopman *lift* of a cell, which is what ADR-0023 rejected), and if #375 wants growth
rates it can have them this way. **But that distinction needs a ruling before it is used, not a
research doc's say-so, and this pass does not make it.** Recorded so nobody re-derives it.

For completeness and not as a recommendation: the continuum field's other standard detector,
**Lagrangian coherent structures** (Haller, *"Lagrangian Coherent Structures"*, **Annu. Rev. Fluid
Mech. 47, 137–162 (2015)** **[ABS]**), is a material-transport construction — it finds *"a robust
skeleton of material surfaces"* behind sensitive tracer patterns. It needs advected particle
trajectories in a spatial continuum. Patchworks has no advected tracer and no continuum. **It does
not transfer**, and it is recorded here so that a later pass does not spend the search again.

---

## 4. Where the field agrees, stated as corroboration and not as ground

Kept deliberately short, because §5 is the section #374 asked to be long, and because this section is
the one most at risk of sliding into motivation.

- **A contracting medium under continual drive is a recognised way to hold structure, and the field
  has a name for it.** Deissler's noise-sustained structure and Huerre & Monkewitz's noise amplifier
  are that concept. **Corroborates** the shape of the adopted position. It did not motivate it: the
  position came from `docs/motivating-image.md`'s field of channels plus #274 and #166, and this pass
  found the concept afterwards.
- **A stability calculation on the deterministic system does not settle the noisy driven case.**
  Butler & Goldenfeld, verbatim above. **Corroborates** #144's reading of #274's contraction as *the
  dissipation half of a balance* rather than as a shortfall — but see §5.1, where the same source cuts
  the other way about *what else* is needed.
- **"The structure is a functional of the drive" is a measurable property with an established test.**
  Uchida et al.'s consistency, Lymburn et al.'s replica test. **Corroborates** that the position is
  the kind of claim that can be checked at all, which is not nothing.
- **The right read on a graph is a partition, not a periodicity.** Nakao & Mikhailov, verbatim in
  §2.1. This corroborates nothing in particular; it is a correction to the *expectation* #375 would
  otherwise carry in from the continuum picture, and it is the most practically useful line in §4.

---

## 5. Who disagrees

#374 asks for this section at real length, and says a source claiming a driven dissipative medium
cannot sustain structure under conditions like Patchworks' is the single most valuable return. Four
such lines were found, and one methodological objection that is stronger than any of them.

The three conditions to test against, restated: the median cell **contracts** (#274, `ρ_full`
0.708–0.908, `τ` 2.9–10.3 ticks); the cell operator `K` is **very nearly normal** (0.05), giving
sequence memory of exactly 1 whatever the dimension (#166, via Ganguli); the within-cell spectrum is
**near-degenerate** (#166, eleven of twelve moduli inside a band of ~0.13).

### 5.1 Contraction: the field does not treat "linearly stable" as sufficient, or as fatal

The naive objection is that a contracting medium cannot hold anything. That objection **does not
survive contact with this literature** — Deissler, Butler & Goldenfeld and Muolo et al. all study
media that are linearly stable and hold structure anyway. So contraction on its own is not the threat.

**What the sources agree on is that contraction requires a *compensating* mechanism, and they name
different ones.** Deissler names convective amplification: the perturbation must grow *as it is
transported*, even though it decays at any fixed point. Muolo et al. name non-normality (§5.2).
Butler & Goldenfeld name the noise itself, but only within an enlarged region around a parameter
range where the mean-field system *does* have a Turing instability — *"Although the model exhibits a
Turing instability in mean field theory, demographic noise greatly enlarges the region of parameter
space where pattern formation occurs"* (emphasis on *enlarges*: it is a widening of an existing
region, not the creation of one from nothing).

**That last clause is the finding against the position in this subsection, and it is sharp.** Butler
& Goldenfeld's noise-sustained patterns live in a *neighbourhood* of a deterministic instability.
Nothing in the record establishes that Patchworks' field is anywhere near such a neighbourhood; #274
measured contraction on all nine seeds with no report of a near-critical mode. **Read strictly, the
source that most corroborates the position in §4 also says the position needs the field to be near
something it has not been shown to be near.** Instrument I is the measurement that would settle it: a
peak in the variance-vs-Λ_α curve, even a shallow one, is evidence of a nearly-unstable mode; a flat
curve is evidence there is none.

### 5.2 Near-normality: the sharpest threat, and it lands on an unmeasured quantity

This is the most important subsection in the document.

**Muolo, Asllani, Fanelli, Maini & Carletti**, *"Patterns of non-normality in networked systems"*,
**Journal of Theoretical Biology (2019)**, arXiv:1812.02514. **[ABS — full abstract read verbatim at
source]**

Their proposal, verbatim: *"a multi-species reaction-diffusion system is studied on a discrete,
network-like support: the instability is triggered by the non-normality of the embedding network. The
non-normal character of the dynamics instigates a short time amplification of the imposed
perturbation, thus making the system unstable for a choice of parameters that would yield stability
under the conventional scenario."*

**Asllani, Lambiotte & Carletti**, *"Structure and dynamical behaviour of non-normal networks"*,
**Science Advances 4(12), eaau9403 (2018)**, arXiv:1803.11542. **[ABS — abstract read at arXiv; the
Science Advances page returned HTTP 403 and the body was not reached]**

Verbatim from the abstract: *"strong non-normality is ubiquitous in network science. Dynamical
processes evolving on non-normal networks exhibit a peculiar behaviour, as initial small disturbances
may undergo a transient phase and be strongly amplified in linearly stable systems."* And a second
clause with teeth for this repo: *"eigenvalues may become extremely sensible to noise, and have a
diminished physical meaning."*

**Trefethen, Trefethen, Reddy & Driscoll**, *"Hydrodynamic Stability Without Eigenvalues"*,
**Science 261, 578–584 (1993)**. **[ABS]**

The founding statement of the mechanism, verbatim from the abstract: *"small perturbations to the
smooth flow may be amplified by factors on the order of 10⁵ by a linear mechanism even though all the
eigenmodes decay monotonically"*, on the basis of *"the pseudospectra of the linearized problem"*.
The last sentence of that abstract explicitly generalises beyond fluids: *"The methods suggested apply
also to other problems in the mathematical sciences that involve nonorthogonal eigenfunctions."*

**Chomaz**, *"Global Instabilities in Spatially Developing Flows: Non-Normality and Nonlinearity"*,
**Annual Review of Fluid Mechanics 37, 357–392 (2005)**. **[CITE — title and venue confirmed; body
not reached, nothing quoted]**. Recorded because its title states the pairing this subsection is
about; **no claim below rests on it.**

**The threat, stated at full strength.** Every one of these sources locates the capacity of a
linearly-stable driven medium to hold structure in **non-normality** — the amplification available to
an operator whose eigenvectors are not orthogonal, invisible to its eigenvalues. Patchworks'
measurements say the cell operator `K` is non-normal at **0.05**, which is to say essentially normal;
and #166's Ganguli-derived consequence, *"the capacity of networks with normal connectivity matrices
is exactly 1"* (quoted in `docs/research/167-linear-recurrence-citations.md` from Ganguli, Huh &
Sompolinsky), is the same fact in memory-capacity form. A normal contracting operator has **no**
transient amplification: its transient response is bounded by its spectral radius from tick one.
**On the cell side, the mechanism the field names is not merely weak. It is absent by construction.**

**Why it is not decided, and why this is the pass's best output.** In every one of these sources the
non-normality that does the work is a property of the **coupling structure** — Muolo et al. say *"the
embedding network"*, Asllani et al. study network adjacency, Trefethen et al. the linearised operator
of the whole flow. **The Patchworks object that corresponds to it is not `K`.** It is the field-level
operator: restriction map into edge stalk, reconciliation, restriction back, cell operator, next
tick — composed along the graph. Three things in the record suggest that operator is **strongly**
non-normal even though `K` is not:

- The substrate is **directed in effect**: #274 measured the apex decaying 1.6–2.1x faster than the
  rim, so the loop is not symmetric between the two ends, and directedness is the canonical source of
  non-normality in Asllani et al.'s taxonomy.
- The restriction maps are **near-rank-1** (`CONTEXT.md`, *channel*: *"narrow because the maps are
  near-rank-1"*). A composition of near-rank-1 maps with different ranges is about as far from normal
  as an operator gets, and the *channel* entry's own language — *"reconciliation is fast along it and
  under-relaxed off it"*, a hop being *"an operator norm along it, never an isotropic average over
  directions"* — is a description of an operator whose behaviour is not given by its eigenvalues.
- **Incoherence count** (`CONTEXT.md`) is measured at 2.42 at the rim and 1.75–1.98 through the core,
  i.e. incident maps are *not* loading the same directions — non-orthogonality across edges, which is
  the structural precondition.

**None of this is a measurement, and this document must not pretend otherwise.** It is a reason to
take the measurement.

**Recommendation to #375, and it is the pass's strongest.** Add a sixth read: the **departure from
normality of the field-level operator**. Concretely, the Henrici departure `‖A*A − AA*‖_F` normalised
by `‖A‖_F²`, or — cheaper and more directly meaningful — the **transient amplification curve**
`max_x ‖A^t x‖ / ‖x‖` against `t`, compared with `ρ(A)^t`. If the two curves coincide, the field is
normal and §5.2's threat lands. If the amplification curve rises above `ρ^t` before falling, the
field has the mechanism, and the position has the thing the literature says it needs — measured on
the object that actually carries it rather than on the cell, which #144's resolution already ruled is
being asked to do the field's job.

**And note the sting in Asllani et al.'s second clause.** If the field-level operator *is* strongly
non-normal, then *"eigenvalues may become extremely sensible to noise, and have a diminished physical
meaning"* — which would put a question over the spectral reads the map already carries, #274's
`ρ_full` among them. That is not a claim that those reads are wrong. It is a claim that they would
need a pseudospectral companion to be complete, and it is flagged in §6.

### 5.3 Near-degeneracy: the dispersion relation has nothing to select with

**Sources: Nakao & Mikhailov [FULL]; and, at [CITE] depth only, Cross & Hohenberg, *"Pattern
formation outside of equilibrium"*, Rev. Mod. Phys. 65, 851–1112 (1993) — landing page only, body not
reached, nothing quoted.**

Nakao & Mikhailov's dispersion relation, verbatim again: *"the growth rate λ_α of each mode depends
only on the combination ε Λ_α ... i.e. we have λ_α = F(ε Λ_α)."* A structure exists in this framework
because `F` has a **maximum** at some Λ_α: one mode grows fastest, and it is the one that shows up.
Pattern formation *is* mode selection.

**The threat.** #166 measured the within-cell spectrum at eleven of twelve moduli inside a band of
~0.13. If the field's analogue of `F` inherits that near-degeneracy, `F` is nearly flat over the
accessible modes, no mode is selected, and what the field produces under drive is not a structure but
**a filtered copy of the drive's own spatial statistics**. The characteristic signature is instrument
I's negative read, and I would expect it: a flat or monotone variance-vs-Λ_α curve.

**How strong is this?** Honestly: it is an argument this pass constructs, not one a source makes about
a system like Patchworks. The step from "the cell's twelve moduli are within 0.13" to "the field's
`F` is flat" is an inference — the field's dispersion relation is a property of the composed
field-level operator including the restriction maps and the Laplacian's own spectrum, not of `K`'s
moduli, and #237 has shown the sheaf Laplacian's spectrum is a nontrivial object in its own right. So
this is stated as a **live and testable threat with a named negative read**, not as a result. Cross &
Hohenberg is the source that would settle how universal the "pattern = mode selection" framing is,
and it was not reached (§8).

**Note the interaction with §5.2, which is the interesting part.** Near-degeneracy and near-normality
compound rather than cancel. A near-degenerate *and* normal operator is the maximally structureless
case: no mode preferred, no transient amplification, response equal to a scalar decay in every
direction. The measurements the map carries put `K` in exactly that corner. **The position's entire
weight therefore rests on the field-level operator being a different animal from the cell operator.**
That is a clean, single statement of what #375 has to show, and it is the pass's summary of the
threat section.

### 5.4 The definitional objection: a driven *linear* medium has no structure of its own

**Source: Grelu & Akhmediev's definition [ABS], §1; and the general framing of the
Ginzburg–Landau/pattern-formation literature.**

Every definition of the object in this literature carries nonlinearity as a **constituent**, not as a
detail: Grelu & Akhmediev's *"in presence of nonlinearity"*; Flach & Gorbach's breathers as solutions
of nonlinear lattices; Huerre & Monkewitz's noise amplifiers and oscillators as flows that *"both
exhibit strong nonlinearities"*. The reason is structural: in a linear driven system the response is a
convolution of the drive with the medium's impulse response, the steady state is unique and globally
attracting, and there is nothing that could be called a structure with an existence separate from the
drive. No bistability, no saturation, no selection, no attractor.

**Patchworks passes this test, and it is worth saying so precisely because it is the one place the
architecture is unambiguously on the right side of a requirement.** #167 §1 established at source that
Patchworks re-applies `encode` **inside** the loop every tick, which is what makes `K·encode` a
nonlinear recurrence, and that this places it on the *more* expressive side of the SSM line. The
per-cell operator is linear; **the field is not**, and the nonlinearity is in the strongest available
position.

**The residual threat is about where the nonlinearity has to bite.** In the dissipative-soliton
literature the nonlinearity's role is specific: it **saturates** the amplification, which is what
stops the structure growing without bound and pins its amplitude — the "balance" in the definition.
`encode`'s role in Patchworks is fusion of arriving evidence with the persisting chart, and nothing in
the record characterises it as a saturating nonlinearity of that kind. **This is not a finding against
the design; it is an unexamined correspondence**, and it is the one I would most want a follow-on pass
to look at if the position becomes more load-bearing.

### 5.5 The methodological objection, which beats all of the above

**Tél & Lai**, *"Chaotic transients in spatially extended systems"*, **Physics Reports 460, 245–275
(2008)**. **[ABS — PDF fetched but text extraction failed; nothing quoted from the body. The
statements below are held at the strength of the abstract and indexed summary.]**

The review's subject is **supertransients**: spatiotemporal behaviour that is genuinely transient, but
whose average lifetime grows with system size *rapidly, often exponentially*. The consequence for
#375 is severe and it is worth stating without hedging:

**A structure that survives 10⁴ ticks on the 414-cell dome is not thereby shown to be sustained.** If
the field is sitting on a chaotic saddle rather than an attractor, its lifetime can be exponential in
the number of cells, in which case *no feasible run length* distinguishes the two. #375's framing —
*"at runtimes far past any cell's `τ`"* — is necessary and it is **not sufficient**, because `τ` is
the wrong yardstick: the relevant comparison is not with the cell's decay time but with the
system-size-dependent transient lifetime, which is unknown a priori.

**The discriminator is scaling, and #375 must build it in.** Run the same construction at several
graph sizes — #163's shape-free builder makes this cheap, and #276 establishes no run has ever had a
timescale gradient placed, so the sizes are comparable — and fit the structure's survival time against
`N`. Exponential in `N` is a supertransient and the structure is **not** sustained, however long any
single run looked. Size-independent survival is the finding the position predicts.

**Two related reads that come with this literature and are cheap.** First, **survival probability**:
run many seeds and plot the fraction still showing structure against time; a chaotic saddle gives an
exponential tail `P(t) ~ e^{-t/⟨τ_s⟩}` with a well-defined escape rate, an attractor gives a plateau.
Nine seeds (#274's count) is thin for that fit but not useless. Second, the review's own noted result
that **weak noise** modifies supertransients — Patchworks' medium is noisy by construction, so a
noise-free calibration is not available.

**I did not read this paper's body and I want that on the record.** The claim that lifetimes scale
exponentially with system size is stated in its abstract and in every indexed summary of it, and I am
confident in it. The specific functional forms, the crossover conditions, and the noise results are
**not** verified at source, and §8 lists this as the most consequential unreached item after the
breather nonresonance condition.

---

## 6. What this threatens, per artifact

Per house style, and per #374: flag, do not revise.

**#144's resolution and the adopted position ("persistence is sustained, not stored").**
*Threatened, not refuted, and the threat is specific.* §5.2: the literature's named mechanism for a
contracting driven medium to hold structure is non-normality, and the operator the map has measured
(`K`, at 0.05) does not have it. §5.3: the near-degenerate spectrum leaves a nearly flat dispersion
relation, which selects no mode. Compounded (§5.3, final paragraph), those two put the *cell* in the
maximally structureless corner. **The position survives if and only if the field-level operator is a
different animal from the cell operator** — which is, encouragingly, exactly what the position itself
asserts. **No revision recommended. The measurement in §5.2's recommendation is what would settle
it.** Note that §5.1's clause — Butler & Goldenfeld's noise-sustained patterns living in an *enlarged
neighbourhood* of a deterministic instability, not in place of one — is a second, independent version
of the same demand.

**#274 (the driven chart-loop spectral read, `ρ_full` 0.708–0.908, `τ` 2.9–10.3).**
*Not threatened as a measurement; threatened as a complete description.* Asllani et al.: *"eigenvalues
may become extremely sensible to noise, and have a diminished physical meaning"* in non-normal
systems. If §5.2's recommended measurement finds the field-level operator strongly non-normal, then
spectral radius is an incomplete summary of the field's transient behaviour and a pseudospectral or
transient-amplification companion would be wanted. **This is a flag on interpretation, not on the
numbers, and it is conditional on a measurement not yet taken.**

**#166 (near-normal `K`, sequence memory 1, near-degenerate spectrum).**
*Not threatened; its readings are what §5.2 and §5.3 are built from.* One thing to record: this pass
found no source that contradicts the Ganguli line #167 already read, and several that independently
converge on the same mechanism from the network-dynamics side rather than the memory-capacity side.
That convergence is worth noting because it means the finding is not an artefact of the
reservoir-computing framing.

**#375 (the field-level prototype).**
*Scope threatened — it needs one thing it does not currently have.* §5.5: runtime alone cannot
distinguish a sustained structure from a supertransient, at any runtime. **#375 must vary the graph
size**, or its central question is not answerable by the experiment as framed. This is the one item
in this document that changes what another open ticket has to do, and #375 should be read against it
before it is taken. §3 additionally recommends it report instruments I and II **as a pair** and never
II alone, and that instruments I and V carry drive-only baselines.

**#357 (whether the operator's shape ever becomes what memory requires).**
*Untouched by this pass, per #374's instruction.* Recorded only: §5.2's finding — that the mechanism
the literature names lives in the coupling structure and not in the node operator — is consistent with
#144's reading that #357 asks the cell to do the field's job. **#357 is not re-scoped from here.**

**ADR-0023 and #148's Koopman line.**
*Not threatened; a collision flagged.* §3's DMD note: the field's default coherent-structure
extractor is the numerical core of Koopman mode decomposition. A field-level DMD *diagnostic* is
arguably not the *lift* ADR-0023 rejected, but that distinction needs a ruling and this document does
not make one. Recorded so it is not re-derived.

**`CONTEXT.md`.**
*Nothing enters it from this pass.* Every term here — coherent structure, dissipative soliton,
discrete breather, Turing mode, non-normality, convective instability, consistency, supertransient —
is a fresh import held in the field's own vocabulary. In particular *coherent structure* must not be
allowed to drift into a synonym for `CONTEXT.md`'s **channel**, which is a *learned* aligned subspace
of the transport, not a dynamical excitation. The two are related in the picture and different in the
formalism, and blurring them would be exactly the failure #167 §1 diagnosed.

---

## 7. Two boundaries, kept

### 7.1 The thermodynamic-persistence line: touched, not entered

The dissipative-structure tradition (Prigogine and successors) is the historical parent of every
source in §1 and §2, and `docs/motivating-image.md` lists Prigogine among its named influences. It is
also the tradition in which "persistent information structures" and "thermodynamic survival" sit —
which `docs/motivating-image.md` marks explicitly as **thin-provenance motivation owing its own
pass**.

**This pass therefore did not read that line, and nothing above rests on it.** The sources here were
selected for the *dynamical* and *observational* content #374 asked for — how a structure is detected,
and how "sustained" is measured — and not for the thermodynamic account of why structures form far
from equilibrium. That account remains unvalidated in this record and this document does not validate
it, absorb it, or borrow its authority. Where §1's "energy exchange with the environment" language
appears, it is Grelu & Akhmediev's optical usage and not a thermodynamic claim about Patchworks.

### 7.2 Information cohomology and topological vocabulary: kept apart

Two sources in this pass use topological or spectral-invariant vocabulary in ways that could be
mistaken for the ruled-out line. Haller's LCS are *material surfaces* in a flow, a differential-geometry
object with no relation to any cohomology of a sheaf. Nakao & Mikhailov's Λ_α are eigenvalues of a
graph Laplacian — a spectral quantity, not an invariant.

`docs/research/015-information-cohomology.md` ruled information cohomology an interpretive lens with
**no relation** to a cellular sheaf's `H¹` — different site, coefficients, differential, and no
comparison map — and `CONTEXT.md` puts *topological invariant* on its Avoid list. **Nothing in this
pass touches that ruling, and no source here supplies a reason to revisit it.** The sheaf Laplacian
that instrument I uses is Patchworks' own operator, entering via #237 and `CONTEXT.md`'s *disagreement*
entry, and it is used here as a **diffusion operator with an eigenbasis** — the same role Nakao &
Mikhailov's graph Laplacian plays — and in no other sense.

---

## 8. What could not be reached

Stated plainly rather than papered over.

- **Flach & Gorbach, Physics Reports 467 (2008) — body not read.** The PDF was fetched (901 KB) and
  text extraction returned undecodable compressed streams. **The single most valuable unreached item
  in this pass** is its statement of the **nonresonance condition** for breather existence — that the
  breather frequency and its harmonics must lie outside the linear spectrum of the lattice. That
  condition would bear directly on #166's near-degenerate spectrum, since a near-degenerate spectrum
  is the configuration in which avoiding resonance with the linear band is hardest. No argument in §5
  rests on it, and §5.3's threat is correspondingly weaker than it could be.
- **Tél & Lai, Physics Reports 460 (2008) — body not read.** PDF fetched (2.7 MB), extraction failed.
  The exponential-lifetime-with-system-size claim in §5.5 is held at abstract strength. The specific
  scaling forms, the crossover conditions, and the weak-noise results are **not** verified at source.
  Since §5.5 is the objection this pass rates highest, this is a real gap.
- **Cross & Hohenberg, Rev. Mod. Phys. 65 (1993) — body not read.** Landing pages only; nothing
  quoted. This is the reference that would establish how universal the "pattern formation is mode
  selection" framing in §5.3 is. §5.3 is stated as an inference partly because of this.
- **Chomaz, Annu. Rev. Fluid Mech. 37 (2005) — [CITE] only.** Publisher returned HTTP 403. Its title
  states the non-normality/nonlinearity pairing §5.2 and §5.4 are about; **no claim rests on it** and
  it is listed so a later pass can go get it.
- **Asllani, Lambiotte & Carletti — Science Advances page returned HTTP 403.** The abstract was read at
  arXiv:1803.11542 and the quotations in §5.2 are from there. The body was not reached, so the
  taxonomy of *which* structural features generate non-normality — which would directly inform whether
  Patchworks' rim-to-apex substrate qualifies — is **not** verified. §5.2's three bullets arguing the
  field-level operator is probably non-normal are therefore this pass's reasoning from `CONTEXT.md`
  and #274, not the source's.
- **Uchida, McAllister & Roy (PRL 2004) and Lymburn et al. (Chaos 2019) — abstracts only.** Both are
  behind publisher paywalls. The definition of consistency and the existence of the replica test are
  solid; the *estimator* — exactly how the correlation is computed and normalised, and what burn-in
  and drive amplitude are appropriate — was **not** read. Instrument II's recipe in §3 is therefore
  the obvious construction and not the papers'. Uchida et al.'s finding that there is *an optimal
  drive amplitude* for consistency (internal noise dominating at small amplitude, nonlinearity
  reducing consistency at large) is carried from the abstract-level summary and #375 should expect to
  have to sweep drive amplitude rather than pick one.
- **Butler & Goldenfeld — abstract read verbatim at arXiv, body not read.** Instrument IV's spectrum
  recipe is the standard construction, not theirs; their specific fat-tail prediction is explicitly
  **not** recommended for transfer (§3, instrument IV).
- **Marín, Aubry & Floría (Physica D 1998) — [CITE].** ScienceDirect abstract-level summary only.
  Recorded as the standard Floquet-stability method for discrete breathers; nothing rests on it.
- **Grelu & Akhmediev, Nature Photonics 6 (2012) — abstract only.** The definition quoted in §1 is
  from the abstract and is the paper's own framing sentence. The review body, including its treatment
  of soliton molecules, pulsations and explosions — all of which are *non-stationary* sustained
  structures and would be relevant to what #375's plots might show — was not read.
- **Not searched at all, and named so the gap is visible:** oscillons in driven granular and fluid
  media; cavity solitons in driven Kerr resonators beyond the definitional citation; chimera states on
  networks (adjacent to §2.1 and possibly the better analogue for a *partition* read); localized
  states and homoclinic snaking; and the spatiotemporal-chaos literature's correlation-length
  estimators. Any of these could plausibly return a better instrument than §3's five, and #374's
  budget did not extend to them.
- **One thing this pass could not do at all:** find a source that treats a **sheaf** over a graph as a
  driven dissipative medium. The search returned nothing, and the negative is reported rather than
  glossed. The instruments in §3 are all transfers by analogy from graph-Laplacian or lattice settings,
  and every one of them carries a stated caution about what the transfer costs.

---

## Sources

Grouped by the section that uses them; depth tag repeated.

**Definition of the object**
- Grelu, P. & Akhmediev, N., *"Dissipative solitons for mode-locked lasers"*, Nature Photonics 6,
  84–92 (2012). doi:10.1038/nphoton.2011.345. **[ABS]**

**Discrete and network analogues**
- Nakao, H. & Mikhailov, A. S., *"Turing patterns in network-organized activator–inhibitor systems"*,
  Nature Physics 6, 544–550 (2010). arXiv:1005.1986. **[FULL — via ar5iv]**
- Flach, S. & Gorbach, A. V., *"Discrete breathers — advances in theory and applications"*, Physics
  Reports 467, 1–116 (2008). **[ABS — body not extractable]**
- Marín, J. L., Aubry, S. & Floría, L. M., *"Intrinsic localized modes: Discrete breathers. Existence
  and linear stability"*, Physica D 113, 283–292 (1998). **[CITE]**
- Deissler, R. J., *"Noise-sustained structure, intermittency, and the Ginzburg–Landau equation"*,
  J. Stat. Phys. 40, 371–395 (1985). doi:10.1007/BF01017180. **[ABS]**
- Huerre, P. & Monkewitz, P. A., *"Local and Global Instabilities in Spatially Developing Flows"*,
  Annu. Rev. Fluid Mech. 22, 473–537 (1990). **[ABS]**

**Instruments**
- Uchida, A., McAllister, R. & Roy, R., *"Consistency of Nonlinear System Response to Complex Drive
  Signals"*, Phys. Rev. Lett. 93, 244102 (2004). **[ABS]**
- Lymburn, T., Khor, A., Stemler, T., Corrêa, D. C., Small, M. & Jüngling, T., *"Consistency in
  echo-state networks"*, Chaos 29, 023118 (2019). doi:10.1063/1.5079686. **[ABS]**
- Butler, T. & Goldenfeld, N., *"Robust ecological pattern formation induced by demographic noise"*,
  Phys. Rev. E 80, 030902(R) (2009). arXiv:0906.5535. **[ABS — abstract verbatim at source]**
- Schmid, P. J., *"Dynamic mode decomposition of numerical and experimental data"*, J. Fluid Mech.
  656, 5–28 (2010). **[ABS — abstract verbatim at source]** *(recorded, not recommended — §3)*
- Haller, G., *"Lagrangian Coherent Structures"*, Annu. Rev. Fluid Mech. 47, 137–162 (2015).
  **[ABS]** *(recorded, does not transfer — §3)*

**Who disagrees**
- Muolo, R., Asllani, M., Fanelli, D., Maini, P. K. & Carletti, T., *"Patterns of non-normality in
  networked systems"*, J. Theor. Biol. (2019). arXiv:1812.02514. **[ABS — abstract verbatim at
  source]**
- Asllani, M., Lambiotte, R. & Carletti, T., *"Structure and dynamical behaviour of non-normal
  networks"*, Science Advances 4(12), eaau9403 (2018). arXiv:1803.11542. **[ABS — via arXiv;
  publisher 403]**
- Trefethen, L. N., Trefethen, A. E., Reddy, S. C. & Driscoll, T. A., *"Hydrodynamic Stability Without
  Eigenvalues"*, Science 261, 578–584 (1993). **[ABS]**
- Chomaz, J.-M., *"Global Instabilities in Spatially Developing Flows: Non-Normality and
  Nonlinearity"*, Annu. Rev. Fluid Mech. 37, 357–392 (2005). **[CITE — publisher 403]**
- Cross, M. C. & Hohenberg, P. C., *"Pattern formation outside of equilibrium"*, Rev. Mod. Phys. 65,
  851–1112 (1993). **[CITE]**
- Tél, T. & Lai, Y.-C., *"Chaotic transients in spatially extended systems"*, Physics Reports 460,
  245–275 (2008). doi:10.1016/j.physrep.2008.01.001. **[ABS — body not extractable]**

**Repo sources leaned on, not re-verified here**
- `docs/motivating-image.md`; `CONTEXT.md`; `docs/research/167-linear-recurrence-citations.md`
  (Ganguli, Huh & Sompolinsky's normal-connectivity capacity result, and the position of the
  nonlinearity); `docs/research/148-local-linear-operator-citations.md`;
  `docs/research/015-information-cohomology.md`; `docs/research/237-the-sheaf-laplacians-effective-resistance.md` (the sheaf Laplacian's
  effective resistance); `docs/adr/0023-the-chart-is-not-a-koopman-lift.md`.

---

*Context: opened by [#144](https://github.com/NGL321/patchworks/issues/144)'s resolution, scoped by
[#374](https://github.com/NGL321/patchworks/issues/374), part of
[#127](https://github.com/NGL321/patchworks/issues/127). Feeds
[#375](https://github.com/NGL321/patchworks/issues/375).*
