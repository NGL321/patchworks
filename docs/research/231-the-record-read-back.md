# The record read back: what this project has already considered, and what else was decided inside the same frame

Research note for [#231](https://github.com/NGL321/patchworks/issues/231). Part of
[#127](https://github.com/NGL321/patchworks/issues/127); blocks
[#230](https://github.com/NGL321/patchworks/issues/230).

**This document rules on nothing.** It is an inventory and an audit. Where a ground has expired it
says so and says when; what follows from that is #230's to decide, or a successor's. A sweep that
starts ruling is a sweep that stops looking.

## Why this exists

[#214](https://github.com/NGL321/patchworks/issues/214) returned the map's first evaluable
transmission verdict and both directions fail: rim→apex at a median bottleneck ratio of **8.7e-10**
(short by **1.15e9x**), apex→rim at **1.3e-8** (short by **7.6e7x**), over 24 trials in float64.
[#184](https://github.com/NGL321/patchworks/issues/184) then closed not one remedy but the **whole
per-hop-multiplier species**, because #214's hops are *graded* — 240x, 57x, 16x, 25x, 84x, 9x, 29x
along the binding path — so a uniform factor is the wrong instrument at any size.

The suspicion this ticket exists to test is that a frame was established early, carried an assumption
— *a hop is a hop* — and shaped decisions nobody has re-read. **The record confirms the suspicion, and
the frame is older and wider than stage 2.**

## Method and coverage

Seven parallel sub-agents, each over a disjoint slice, reading resolution comments in preference to
bodies. Slices: issues #1–#73 (closed map #1); #77–#119 with `src/`; #120–#151; #154–#190; #191–#231;
`docs/spec/` + `docs/adr/` + `CONTEXT.md`; `docs/research/` + `prototypes/` + prototype branches.
Synthesis, and every claim marked *verified here*, was done against the sources directly.

**Coverage is uneven and that is stated rather than smoothed.** In map #1's era only #1, #7, #8, #9,
#20, #27, #28, #31, #33, #36, #37, #41, #42, #47, #48, #50, #53 and #57 were read at source; #14, #49,
#32, #6, #25 and #60 are known only through the map's index and are the highest-priority gaps if this
is ever re-run. In the build era #92–#119 were read by title and state except where they bore on a
job. Fifteen prototype branches were not opened at all. Three findings returned by agents were checked
and **discarded** — see *Three corrections* at the end.

---

# Part 1 — Roads not taken

Live options with their declining grounds, and whether each ground survives #214. Ordered by whether
#184's species closure reaches them, because that is the distinction that decides which are still
available.

## 1. Structural options the species closure does not reach

These change what the graph *is* — width, wiring, or what couples to what — rather than multiplying a
hop. #184 closed uniform per-hop factors; it did not close these.

**1.1 Widening the taper's funnel.** [#8](https://github.com/NGL321/patchworks/issues/8) identified it
and did not fix it: 12,288 numbers at the sensory base reach ~60 core cells of dimension 32 through
edge stalks of dimension 4. Parked in map #1's *Not yet specified* with the ground **"unclear whether
the funnel is a defect or is the compression the architecture exists to perform,"** and the
reactivation condition *"revisit once anything has been trained."* **That condition is met** — #120
ran 100,000 ticks, #214 measured the channel — and the item has never been revisited. Not refuted;
deferred, on a trigger that has since fired.

**1.2 The L2→L3 cut, named as the binding constraint and knowingly left unaddressed.** *Verified here*
in `docs/spec/06-graph-topology.md`: the per-tick capacity of each cut is **`12,288 → 2,120 → 280 →
80`** — "the entire sensory boundary reaches the core through **eighty numbers per tick**, a 154:1
squeeze at a single cut." The spec states that rim-to-rim effective resistance "is dominated by the
single **L2→L3 cut** — twenty edges at `m = 4`," that the pre-specified relays sit across L4–L6
"entirely inside the core, above that cut," and that they therefore **"leave the cut untouched."** The
relays that *would* parallel it are chords across the vision levels, **rejected to protect hop
distance**, "which is the abstraction measure and the acceptance demo's yardstick."

So the record named a structural bottleneck, declined the one intervention that reaches it, and did so
on a ground about *preserving an instrument* rather than about transmission. That ground is untouched
by #214 on its own terms — but it is now weighed against a measured 1.15e9x, which is not the exchange
rate it was set at.

**A caution against over-reading this.** #214's binding edges are `#450 interior m=4 L6/core—L7/core`
(rim→apex) and `#272 interior m=4 L1/vision—L2/vision` (apex→rim). **Neither is the L2→L3 cut.** The
record's long-standing structural prediction is now measurable and the measurement does not match it.
Whether the cut analysis was wrong, or is right about resistance while the bottleneck ratio measures
something else, is open and belongs to #230.

**1.3 Relays were declined on a metric the record itself later abandoned.**
[#8](https://github.com/NGL321/patchworks/issues/8) declined relay cells because "relays solve reach,
and reach is not what is squeezed," reasoning from the dome's small **diameter** (~9 hops).
[#31](https://github.com/NGL321/patchworks/issues/31)'s citation pass found the literature indexes
over-squashing by **commute time / effective resistance, not diameter**, and that by the correct metric
"relays would widen the funnel." [#47](https://github.com/NGL321/patchworks/issues/47) **conceded the
point — "by that quantity the taper *is* the reach problem" — and kept the decision**, on the separate
hop-distance ground in 1.2. **The original ground is dead by the record's own finding**; the
replacement ground has never been weighed against #214.

**1.4 Local Virtual Nodes.** `docs/research/148` §10.3 flags them as "the mildest option found, and the
one most compatible with a dome — worth real consideration for the rim→apex path." **No ticket takes
it up, and no ground for dropping it is recorded.** Note that `docs/research/148-*` **exists only on
the unmerged branch `worktree-citation-pass-148`** and is cited by name from `docs/research/150`, which
is merged — so a merged document cites a file not in the tree.

**1.5 Narrowing interior `m` from 4 to 2 — half of it is structural.** `docs/research/150` §3, *verified
here*: `m = 2` measures a hop at 0.00212 against the built 0.001223, worth **1.734x per hop**, and
**returns private dimension, 10.75 against 3.95**. The gain half is squarely inside the species #184
closed (150 itself says "over seven hops, `1.734^7 = 47x`... a lever, not a solution"). **The private-
dimension half is not a multiplier at all** — it is `H⁰`, the substrate for slow state — and nothing in
the record prices that half against #214. The document states the trade "was never priced, and this is
the price"; no ticket acts on it either way.

**1.6 The change gate.** [#20](https://github.com/NGL321/patchworks/issues/20) specified it in full —
outbound-only attachment, a stateless relative threshold, boundary edges categorically exempt — and
**deliberately did not build it**, because "ADR-0005 already supplies macro/micro another way, so the
gate has no job it is *needed* for." Its recorded trigger is a persistence readout that is "nonzero but
below what planning needs." Its purpose is amplification and commitment rather than raw gain, so its
bearing on #214 is genuinely uncertain — recorded because it is specified, shelved, and its trigger has
never been tested.

**1.7 Relay cells and the core's broadcast subspace, as the mechanism for semi-global reach.** #20
corrected `04-action-and-the-boundary.md`'s pointer and left the object in fog: attention "is not a
per-edge threshold at all but the **semi-global reach** of the objects that have it — the core's
broadcast subspace or relay cells (#8). Neither is established." Unspecified, never ticketed.

**1.8 Coupling through persistence rather than propagation.** #230's own spread item 1, and the record
contains no prior engagement with it — the closest is #25's hippocampal module, held in map #1's fog as
"the escape hatch rather than a step," graduating "if and when #25's stall signature actually fires."
#120's zero-travel-from-tick-5000 is arguably that signature; nothing connects them.

**1.9 The dome is declared abandonable, and a shape-free builder now exists.** #8, *verified here*:
"**Held as explicitly abandonable.** The fallback is a shape-free rule (boundary set, target depth,
per-level degree constraints, connectivity sampled to satisfy them). Every other decision here is
independent of the dome and survives it." #8 also records that **seven-hop depth was chosen to make a
measurement legible** — the vision cone gives only four levels, "nowhere near enough hop depth for
*recovered at the appropriate level* to be measurable" — not because the architecture requires seven.
[#163](https://github.com/NGL321/patchworks/issues/163) has since written the shape-free builder for
the language wedge. #230's spread item 2 asks whether depth itself is the sentence; the fallback it
would need is already argued and already built.

## 2. Options inside the closed species, recorded so none is re-proposed

Listed because #230 should not rediscover them: **#142**'s attention-weighted per-edge gain (the
circuit version declined on circularity — "you cannot use attention to fix transmission if the
attention signal itself depends on transmission to arrive" — while the *local-only* version was
explicitly **deferred, not refuted**, with "it should be tried first"); **#142** §6's cross-edge
coherence term, whose locality ground **#184 showed was a misreading** (ADR-0008 is a gradient
boundary, so no amendment was ever owed) before closing it on scale instead; **#188**'s widening of the
`m = 1` drive edge, never measured, now blocked on #230; **#220**'s denominator swap; and the gain half
of 1.5 above. Also **#150**'s rewiring family — virtual node, fully-adjacent last layer, expander and
curvature rewiring — declined because effective resistance is already near floor (rim to apex is 7 hops
but **1.82 unit-resistance edges**, so rewiring is worth under 2x).

## 3. Grounds that have expired

**3.1 Pre-training waves.** Raised on #120's comment stack (a retinal-wave analogy), and it sits today
in **#127's Out of scope**: "Closed on the arithmetic above, on the sufficiency check #120's own comment
asked for first." **That arithmetic is the isotropic one.** The map's own *standing diagnosis* section,
which carries the 921x/hop and the 267x-at-saturation ceiling the closure rests on, is labelled in the
same map as measured "with an **isotropic** probe against **near-rank-1** maps... and that reading is
now the thing explicitly rejected." The closure has never been re-taken against #142's correction or
#214's measurement. **An out-of-scope ruling is resting on a retracted number.**

**3.2 "The spectrum is already spoken for three times over."** *Verified here* in
`06-graph-topology.md`: Arroyo et al.'s over-squashing mitigation — direct control of the Jacobian
spectrum — is "deliberately not claimed" because the spectrum is spent three ways: the body is shared
and frozen, the regional spectrum is deliberately spread to supply timescales (#27, #42), and `γ` is
fixed for stability. **Two of those three claimants have since changed.** The Koopman conversion made
the body's spectrum `σ_max(K)` — *settable and bounded*, which is the entire stated case for taking the
conversion ("`body` is a term in the transmission budget, and today it is not a design variable") — and
[#143](https://github.com/NGL321/patchworks/issues/143) moved timescale onto `λ(K)`, a learned
retention constant rather than a construction-placed regional spectrum. The decline was correct when
written; the conversion was taken *precisely* to unspend that budget, and nothing has re-read this
paragraph since.

**Re-read and repaired by [#420](https://github.com/NGL321/patchworks/issues/420), with one
correction to the reading above.** The budget **split**; it did not vacate. `σ_max(K)` is still spent,
by [ADR-0015](../adr/0015-the-cell-operator-band-is-on-the-spectral-norm.md), at the
maximal-transmission face of exactly 1 — so no headroom was reclaimed there. What came free is
`ρ(K)`, and the stronger finding is that the mitigation was **never available to decline**: the
conversion *is* Arroyo's remedy, taken in stage 1, as `docs/research/148` §10.2 had already said. `06`
was rewritten from a decline into a claim by
[#421](https://github.com/NGL321/patchworks/issues/421), closing
[#338](https://github.com/NGL321/patchworks/issues/338).

**3.3 "The drive does not reach."** #120's two-levels finding was quoted as settled for a long stretch;
[#183](https://github.com/NGL321/patchworks/issues/183) found it located **float32's resolving limit,
not the graph's**, and that read along the channel the drive's hop is worth 1.84x and the drive
*reaches* at 0.100 per hop. Struck as grounds by #146. The record has now mistaken float32's floor for
the graph's three times (#146, #183, #214) — [#224](https://github.com/NGL321/patchworks/issues/224)'s
subject, still open.

**3.4 #146's own reason for waiting** was #120's float32-epsilon drive reading — struck by #142, "and
not replaced with the opposite claim: nobody knows whether the drive reaches" (later settled by #183).

**3.5 The `Σ_e m_e` denominator.** `src/patchworks/tick.py`'s own docstring, *verified here*, calls the
shipped `γ / max(Σ_e m_e, ρ² deg(v))` **superseded** and its defence **struck** by #190, with the swap
deferred until the gauge projection enforces `GAUGE_C` — which `restriction.py` marks "**Declared, not
yet enforced**... nothing may divide by this constant until it does." The function computing every
cell's transmission gain today is one the record no longer defends. This is #220's, and it is stated
here only because #214's numbers were measured *through* it.

## 4. Specified, owed, or recommended — and never run

Not decisions, but the record's own unfinished business, several items load-bearing for #230.

- **#97–#102 are open with zero comments**: the continual run, the shared-frozen-body falsification,
  the acceptance-demo harness, the live and crossing runs, and the two topology falsification sweeps.
  #98 is explicitly the test of "the shared frozen body is a bet... That conjunction *is* the thesis."
  No ground for skipping is recorded anywhere; they simply have not happened.
- **The falsification register does not exist in the repo.** #147 specified it as an index ADR to be
  authored in #157, which merged without it; #217 could not write its item 3 for that reason. Entry 1's
  fill lives on #147's thread. Owed before #78 spends a budget.
- **`docs/research/032`'s three recommended tickets** are marked "not created" in the source and none
  was later opened — including *"note in `06-graph-topology.md` that `m = 4` is the dimension with the
  least theoretical headroom,"* which bears directly on 1.5.
- **`docs/research/148`'s R7, R8, R9** (the ω-limit-set obstruction record, trivial baselines for the
  falsification register, the delay-dictionary convergence note) show no follow-up; R1–R6 were done or
  explicitly declined.
- **`docs/research/016`'s R3 and R4** (the locality argument "overstates the case"; the LSTM-shaped
  escape hatch is named on the wrong tier) — no evidence the spec sentences were corrected.
- **`docs/research/053`'s R1 and R2**: ADR-0010 states the drift direction backwards (Arora et al.
  prove a scale-invariant parameter's norm is *non-decreasing*), and its "rank concentration becomes a
  priced trade rather than a free lunch" is stronger than the algebra supports (Miyato et al. prove the
  opposite sign). Both are corrections to *reasoning*, not to the decision. **Live relevance:** the
  no-rank-floor choice is what permits the near-rank-1 maps #142 found, and #57 already conceded "the
  budget does not police rank at all." Whether today's maps have drifted toward rank-1 has never been
  measured.
- **`docs/research/150`'s sheaf-Laplacian effective resistance** was explicitly not done — it needs a
  pair of stalk *directions* rather than cells — and remains map #127 fog.
- **The transport rule has no fixed point at agreement** (#89): the disagreement numerator is a norm
  rather than a squared norm, so a converged endpoint keeps taking a full step and maps wander at
  amplitude `~η`. Found, and "documented rather than changed."
- **`docs/research/148`'s rSLDS suggestion** — "neighbouring cells' `K` should be similar," a
  graph-local regulariser — never taken up.

---

# Part 2 — The audit

Decisions resting on an average, a representative element, an interpolation, an isotropic reading, one
unrepeated measurement, or a superseded body. Grouped by defect, with how load-bearing each still is.

## 5. The frame itself: a representative quantity standing for a distribution

This is the shape #230 named, and the sweep finds it is the map's oldest habit rather than stage 2's.

**5.1 The isotropic probe (#142).** The origin. Every pre-#142 transmission figure — 921x/hop, the
0.003747 saturated ceiling, 267x, ~1e17 across seven levels — probed near-rank-1 maps with a random
deviation and read the average direction's fate. Corrected, and the correction is on the map. Still
load-bearing through 3.1, because an out-of-scope ruling rests on the retracted arithmetic.

**5.2 Diameter standing for transmission difficulty (#8 → #31).** Predates #142 by the whole build.
A single graph-wide scalar (~9 hops) stood in for a directional, structural quantity; #31 found the
governing metric is commute time / effective resistance and that "the dome's diameter is small but its
effective resistance between distant rim cells is huge." Same failure, one architectural layer up, and
it is what produced the relay decision in 1.3.

**5.3 An equalisation asserted for years and false when first measured (#189/#190).**
`02-tick-semantics.md` defended `Σ_e m_e` as equalising the effective step across the graph. #189
measured `gain_v · λ_max` for the first time and found **3.57x spread at init, graded by depth, largest
at the apex** — the exact depth-grading `02` invoked ADR-0005 to forbid. #190: "Nothing was spent to buy
transmission... What follows is a correction, not a purchase." **The architecture's own gain rule was
designed to make hops uniform, asserted to have done so, and #214 measured them spanning 9x–240x.**

**5.4 One interpolated hop standing for a path.** ADR-0021's pre-registered expectation reused #158's
30k per-level floors with only the apex corrected, interpolated across #190's two endpoints, and
predicted the apex at ~3e-5, short by ~2,900x. Measured: 8.7e-10, short by 1.15e9x — ~34,000x worse.
#214 names the failed assumption itself: "the assumption that the max-gain path's hops resemble the
graph-wide ones is the one that fails." The expectation was properly labelled indicative, and it is the
closest thing the pre-#214 record has to a transmission estimate.

**5.5 A representative-hop instinct surviving in the code.** *Verified here*: `graph.py`'s module
docstring insists the construction layout "is an index, not an embedding" and a level "has no runtime
role" — true of the mask and the gain, which read precomputed per-cell arrays. But `DomeSpec` is built
from **per-level tuples** (`vision_sides`, `somatomotor_sizes`, `core_sizes`) and `build_graph` assigns
**different degree targets by level** (`core_degree` vs `apex_degree`). The taper's shape — the thing
#214's grading rides on — is authored level by level. `Dome.cut_capacities` and `Dome._group` are
per-level aggregates in the reporting path. None of this violates #181's rule, which governs targets and
pass conditions; it is recorded because the *shape* the rule warns about is structural in the builder.

## 6. Single unrepeated measurements, quoted onward

**6.1 The 30k-tick floor (#158 → #178).** The archetype. #178 read to 100k and found 30k "landed near a
local high" of a quantity that wanders 3.8x with no trend; the apex settles at 0.087, not 0.217, and
#158's mid-depth cap on `γ` was a 30k artifact. Everything downstream of #158 inherited it until #178.

**6.2 `γ_cap` is an order statistic, not a body property (#195).** Four runs at one seed give **four
different binding cells and a 5.8x spread on the cap**, while population statistics hold to 1.2–1.45x.
ADR-0019 now carries the general warning: "a per-cell extremum on this surface is not a reproducible
quantity, which bears on anything in the record quoting one off a long run." **Any single "worst cell"
or "binding cell" figure elsewhere in the record should be read under that warning** — including
#214's own named binding edges, which are the median trial's, not a distribution over trials.

**6.3 Level medians hiding the binding tail (#158).** Level 4's median floor was 0.146 against p95 2.40
and max 3.85; because `γ` is one global scalar, a handful of mid-depth cells set the true cap. #158
flagged it itself — "a verdict read off level medians alone would have missed this" — and then the tail
reading turned out to be a 30k artifact too.

**6.4 The 7.7x timescale spread, measured on a stand-in body and never re-run.**
`docs/research/027`'s own caveat is explicit: the body is "a **stand-in** — iid Gaussian ReLU MLPs at
`k=12, n=32`... This establishes the shape of the answer and the sensitivity of the rig, not the body's
number," and its first listed measurement-to-run is a re-run against the real body. **No re-run
exists.** The figure is nonetheless quoted as operative headroom in map #1 and in #20's resolution, and
#42 re-measured on the same stand-in and got different numbers again (4.5x drawn, 16x selected) without
updating earlier citers. **Compounding this:** 027 measures the spectral radius of a *regional* Jacobian
of a piecewise-linear ReLU `step` — and the conversion made `step` a linear `K`, which has one region
globally. `docs/research/032` carries a Koopman-era amendment; **027 carries none and stands unqualified.**

**6.5 Other single-source figures.** #120's 100k zero-travel behavioural lock is one run per dome, no
seed repeats (its *untrained* characterisation used three). #149's "no impulsive contact anywhere"
generalises from two approach speeds. `DEFAULT_SPARSITY_PRESSURE = 0.4` is pinned to a balance measured
across three seeds on one dome. #142's `bound / λ_max = 5.585` was treated as a uniform multiplier until
#182 found it strongly non-uniform — "the rim is the loosest level in the graph, not the tightest."
`prototypes/route-geometry` reports percentages with no stated sample count or seed.

## 7. Population scope silently narrowed

**7.1 150 of 414 cells treated as "the graph" (#189).** Every table in the record — #142, #158, #178,
#182 — covered only the 150 predicting cells. The other **264 are boundary cells**, carrying the exact
gauge `‖F‖_F = 1` rather than the interior band, so the interior denominator applied to them is loose by
a **permanent 8x**. Exactly one of the 264 has a reconciliation step that survives the tick — **the
actuator**, whose return path into the arm measures at 1/24 where the bound permits 1/3, and which is
the cell **the acceptance demo's instrument actually reads**. A scope convention nobody stated hid the
demo-relevant cell for several ticket-cycles.

**7.2 A binder nobody had counted (#182).** `Σ_e m_e` does not bind only at the rim's 70 cells: it binds
at **142 of 150**, because at `ρ = 2` with vertical `m = 4` the two arguments of the `max` are exactly
equal — which `02:97` says in its own words. #150 had published "80 of 150"; #189 reconciled the two as
an unstated tie-break (70 bind, 72 tie, 8 bind the other arm). **`docs/research/150` still reads "80 of
150"** and carries no correction.

## 8. Figures read on a superseded body, still in the documents

**8.1 `body = 0.4529` is doubly stale.** The map compares #157's post-conversion 0.7138 against 0.4529
"up 1.58x." But 0.4529 is #120's **first** untrained figure, and #120's own second comment corrects it
to **0.4150** — "two numbers in the comment above came from a superseded version of `attenuation` that
measured a hop as two factors instead of three." So the comparison is pre-conversion *and* pre-
correction. `docs/research/150` §2's factor table also carries `body = 0.45290` with no amendment.

**8.2 The dwell evidence is pre-conversion and says so.** `05-timescales.md` states plainly that
#202's and #206's 100,000-tick counts "were read on the pre-conversion body... **The counts do not**
[survive the conversion], and they are marked as read on the body that ran rather than rescaled." Those
counts — median `dwell/τ = 9.49`, 125 of 150 clearing `2.6τ` — remain the evidentiary centrepiece for
ADR-0019's apparatus. The *qualitative* transfer is argued ("one e-fold is one e-fold whatever the
facets are"); the numbers have not been re-measured on the converted body.

**8.3 Sibling documents disagree on the fold-margin cap.** `01-cell-and-sheaf.md` says the cap "rose
from 0.2600 to 0.3502"; `02-tick-semantics.md` records #195's re-measurement in the right space at
**0.1369**. `01` is not updated, so a reader of `01` alone gets a figure superseded twice. `01` also
carries margins read under the pre-#206 denominator, 1.183x too tight, **deliberately not rescaled**
because "a rescale would publish a number nobody ran" — correctly disclosed, still easy to misuse.

**8.4 A stale citation is the same defect applied to provenance.** #180 found `DEFAULT_GAMMA`'s comment
citing #85's fold-margin check as one that "doesn't exist yet" long after #85 closed.

**8.5 #178's own status.** Self-flagged: "One draw. Seed 42, train split, and the pre-#157 body." The
post-conversion re-read is marked **owed** and has not been delivered.

## 9. Precision

Beyond #214: #91 found and fixed a float32 near-resolution defect in the **effective-rank instrument**
itself — an absolute epsilon broke exact scale-invariance, so "a rank-4 fleet at `‖F‖_F = 1e-8` read
0.01," a false collapse in exactly the regime where ADR-0010 says a map's norm is not a diagnostic. Now
computed in float64. The instrument is trusted for the near-rank-1 readings the whole #142 correction
turns on, so it matters that it once failed this way. `reconciliation_gain`'s float32 casts were checked
and are **not** near resolution — small integers — recorded so it is not re-checked.

## 10. Constants and their warrants

Checked and current: `MAP_NORM_BOUND` (= `GAUGE_RHO`, ADR-0018), `IMAGE_SIZE`, `CONTROL_HZ`/`PHYSICS_HZ`
(a genuine catch by #198's new check on its first run), `DEFAULT_DRIVE_SCALE`, `DEFAULT_GAMMA`. Two
notes. **`DEFAULT_GAMMA = 1.0`'s construction-time ground** — that any smaller value would be "the
unmotivated constant ADR-0002 objects to" — never bore on transmission at all; the hard ground came
later (#205: the margin's tail visits zero, so no positive `γ` buys a clean run). **`GAUGE_C = 2`** is a
single global constant chosen because it is "the value levels 4–7 already satisfy untouched," against a
measured overlap range of 1.75–2.42 — one value for a heterogeneous population, in the transmission
chain's own denominator, and still *declared, not enforced*. **`DEFAULT_SAFETY_FACTOR = 2.6`** was
transplanted from a ratio of two times onto a raw duration; #208 and #212 demoted it to reported
headroom.

## 11. Where the discipline held

Recorded so the audit is not read as uniformly damning, and because these are the templates.
**ADR-0021** declined a per-level target on the identical ground #127 later generalised — "the level
structure is an imposed prior... the architecture is measured on the graph, never on the shape imposed
on it" — and declined an invented `k > 1` because "every invented constant in this map's history has
later been found to have none." Its two-directions-stated-separately choice was vindicated exactly by
#214's asymmetry. **#183** is the one isotropic re-check that came back clean, and it deliberately
re-checked against #178's hazard, finding the drive's hop agreeing to four digits between 30k and 100k.
**#212** root-caused one dwell measurement circulating as **three different numbers** (19, 21 and 25 of
150) across three tickets, traced it to an unstated estimator, and pinned the estimator rather than the
number. **#137** declined to derive a ceiling *because* its inputs were under audit. **#111 and #115**
derived test tolerances rather than choosing them — a rigour the production reporting path
(`progress.py`'s graph-wide means, `diagnostics.py`'s `.mean()`-only summary lines) does not match.

---

# Three corrections

Findings returned by the sweep that were checked and **do not stand**. Recorded so they are not
re-derived.

1. **"Three ADRs are numbered 0014."** True of the tree the agents read; **already repaired** by
   [#218](https://github.com/NGL321/patchworks/issues/218) in PR #222, merged 2026-09-01. Current
   `action` carries one 0014, plus 0023 and 0024. The sweep's checkout was one commit behind.
2. **"The gauge band contradicts #214's hop gains."** Rests on reading 240x/57x/… as amplification.
   They are **attenuation** — #214's words are "per-hop attenuation along the path runs 240x, 57x…" —
   so there is no tension with `‖F‖_F ∈ [1/2, 2]`.
3. **"#214 empirically confirms the L2→L3 cut."** It does not. #214's binding edges are L6→L7 inbound
   and L1→L2 outbound. The prediction is now testable and unmatched, which is a finding for #230 rather
   than a confirmation — see 1.2.

# What this hands to #230

Its opening spread said the sweep exists partly to add to the list from the record. The additions are
**1.1** (the funnel, on a trigger that has fired), **1.2/1.3** (a named cut, and relays declined first
on a metric the record abandoned and then to protect an instrument), **1.4** (local virtual nodes,
unactioned and ungrounded), **1.5**'s private-dimension half, **1.6/1.7** (the change gate and
broadcast reach, specified and shelved), and **1.9** — the dome is *already declared abandonable*, its
seven hops were chosen for measurability rather than function, and the shape-free fallback is written.
Spread item 5, what explains the grading, is answered in part by **5.3**: the rule that was supposed to
make hops uniform was asserted, never measured, and false when finally checked.

And **3.1** is the one item that needs a decision before #230 rather than inside it: an option now sits
in *Out of scope* on arithmetic the map itself has retracted.
