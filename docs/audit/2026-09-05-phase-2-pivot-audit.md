# Phase-2 pivot audit: an external read of the record before the implementation pass

*External audit, 2026-09-05, prepared as a senior-consultant read of the repository and the tracker
only: no conversation with the team beyond four clarifying questions, no run of the real model.
Companions: [`2026-09-05-an-outsiders-introduction.md`](./2026-09-05-an-outsiders-introduction.md)
(the reviewer's picture of the project, written to be diffed against the team's) and
[`collapse-mechanism-note.md`](./collapse-mechanism-note.md) (the analytical foundation for the
collapse, a toy, and the reads that would confirm it on the real cells).*

**How to read this.** The headline is first and everything after it is evidence. Every
recommendation carries a flag, `[moves: …]`, naming the commitment, decision or spec section it
would change, because the brief was to recommend freely and say what each recommendation touches.
Claims are tagged **[measured]** (read off the record or this session's runs), **[derived]**
(follows from the code's arithmetic), **[toy]** (shown in the twelve-dimensional toy only) or
**[inferred]** (supported by the record, read on the graph by nobody). The brief's
"rim-to-table" is read as **rim-to-core** (map [#127](https://github.com/NGL321/patchworks/issues/127)'s
destination) and "flatlining" as four symptoms of one loop, §1.

---

## 0. The headline

**The graph has been measured for transmission through a world that nothing varies, and it is
collapsing for lack of variation.** Both local rules are outer-product updates; under constant
evidence they concentrate onto one direction; the operator band turns that concentration into the
loss of every other mode. The agent's untrained command is a world-independent constant, so the
world is static from the first tick; the cells that read the static parts collapse; collapsed cells
emit constants; the arm stays at its stops. Every rim-to-core shortfall in the record — 921x per hop,
1e14, 1.15e9x, 15.4x, 1.1–8.5x, a bar pinned at exactly zero — is that one fact read with a
different instrument. The remedy is upstream of every constraint added since the pivot: **supply
variation at the cold start, then measure.** The record already holds the supplier in four places
and has it filed as fog. Beneath that headline are eight conflicts, a freeze list that costs nothing,
an image with one load-bearing hole, and one experiment set that addresses nine open problems at
once.

---

## 1. The failures the brief names, read as one loop

The brief names rim-to-core communication and flatlining. The record holds them as separate rows;
they are one loop.

| symptom | where the record has it | what it reads |
|---|---|---|
| The arm freeze | [#120](https://github.com/NGL321/patchworks/issues/120) | command constant to 1 part in 10⁵, identical across two worlds; every joint at its stop; zero travel from tick 5000 of 100,000 |
| The cell collapse | [#477](https://github.com/NGL321/patchworks/issues/477), [#482](https://github.com/NGL321/patchworks/issues/482) | 11/6/8 of 150 cells with `modes_retaining = 0` at 100k; `ρ(K)` to 0.058; exactly the apex (constant drive) and the somatomotor column (frozen arm); 0 of 80 vision cells |
| Short, flat retention | [#274](https://github.com/NGL321/patchworks/issues/274), [#235](https://github.com/NGL321/patchworks/issues/235), [#143](https://github.com/NGL321/patchworks/issues/143) | `τ` 2.9–10.3 ticks, no depth gradient, learning making cells *faster* (17.3 → 4.7 over 100k); 7–28 of 150 cells clear their own world loop |
| The composed channel | [#497](https://github.com/NGL321/patchworks/issues/497), [#498](https://github.com/NGL321/patchworks/issues/498) | rim-to-apex transport is effective rank 1.000 before and after the spectral floor; the trajectory is "degenerate" — one direction to identify |
| The drain | [#324](https://github.com/NGL321/patchworks/issues/324) | per-map effective rank 2.9 → 1.0009 under the sparsity pressure; resolved by the floor per map, not end to end |
| Rim-to-core influence | [ADR-0026](../adr/0026-rim-core-influence-is-a-conduction-ratio.md), [#385](https://github.com/NGL321/patchworks/issues/385), [#474](https://github.com/NGL321/patchworks/issues/474) | the conduction ratio read exactly 0 in both directions at every `λ`, every seed, every horizon, untrained included; unpinned by #474 on 2026-09-04; **not yet read on the new graph** |

The loop **[derived + measured]**: constant command → arm at its stops → constant proprioception,
touch, efference → somatomotor `K` collapses to one direction → constant output to the actuator →
constant command. The apex enters the same loop from tick one by design: `DRIVE_ASSERTION = 1.0`
on eight one-dimensional lanes ([#137](https://github.com/NGL321/patchworks/issues/137)). #120 named
the fixed point "saturated" and #154 ruled that "the scarce resource is variety, not disagreement"
(682 of 682 edges disagree, forever). Nobody has yet said that the four rows above are the same
loop, or that its cause is the learning rule's own arithmetic. §2 does.

---

## 2. Root cause **[derived, toy, partly measured]**

The full derivation, the toy, the fit to the real dead cells and five pre-registered predictions are
in the [mechanism note](./collapse-mechanism-note.md). The short form:

1. The prediction rule's gradient in `K` is `(Dᵀe) hᵀ` — an outer product with row space along the
   cell's evidence direction `h` (`learning.py`, `prediction_error`; `body.py`, `advance`/`decode`).
2. Under constant evidence with a persistent error, the updates are coherent: `K → aI + β û hᵀ`
   with `β` growing linearly.
3. The band normalises by `σ_max` (`body.py`; the same used operator before and after
   [#466](https://github.com/NGL321/patchworks/issues/466)), so the used operator tends to
   `(a/β)I + û hᵀ`: `σ_max = 1`, `ρ → 0`, stable rank → 1, non-normality → `√2`. That is #477's
   signature, statistic for statistic, including why the band fires hardest at the apex without
   being the cause ([#335](https://github.com/NGL321/patchworks/issues/335)).
4. **Read on the committed checkpoints this session**: all 25 dead cells across seeds 42/43/44 have
   stable rank within ~0.5 of what the family predicts from their `ρ` alone, non-normality rising to
   the `√2` ceiling as `ρ` falls (seed 44 cell 67: `ρ` 0.058, non-normality 1.386 read, 1.359
   predicted), and `σ_min` below `ρ²` everywhere; live cells sit at stable rank ~9.6 and
   non-normality ~0.03. Consistent with the family, not yet confirmation: the decisive read (does a
   dead cell's `K` row space align with its mean `h`?) needs one 5k-tick run.
5. The same law holds for the maps: the transport gradient is `(…) xᵀ`; a floor keeps `m`
   directions open per map but cannot fill them; unfilled directions compose at chance, which is
   [#436](https://github.com/NGL321/patchworks/issues/436)'s flat-chance null at composed rank
   1.107. **Composed channel rank is bounded by the rank of the variation the trajectory excites.**
   The remedy for composed rank is variation, not a further per-map constraint.

Why vision survives: eight chattering neighbours make `h` turn over. Why the bias does not rescue
the dead cells **[inferred]**: the target moves with the prediction (the error *is* the standing
reconciliation offset), the maps wander at `~η` forever
([#339](https://github.com/NGL321/patchworks/issues/339)), and
[#202](https://github.com/NGL321/patchworks/issues/202) found no clean tick in 100,000 — the
persistent error is regenerated faster than a bias absorbs it.

---

## 3. Major conflicts, ranked

**C1. Locality versus composed acceptance.** Every mechanism is per cell or per map — the gauge
(ADR-0010), the locality guard (ADR-0011), the operator band (ADR-0015), the spectral floor
(ADR-0032). Every pass condition is composed over paths — ADR-0026's `min` over the cells of a
path, #497's composed rank, [#453](https://github.com/NGL321/patchworks/issues/453)'s holonomy.
[#423](https://github.com/NGL321/patchworks/issues/423) already found that per-cell control is not
composed control. §2.5 sharpens it into a law: no local constraint raises composed rank above
excitation rank. The record keeps buying local constraints for composed failures, and #497 is the
receipt. **[measured]**

**C2. "Drive" versus "driven field".** ADR-0009's drive is a constant assertion; the image's
chamber and [#144](https://github.com/NGL321/patchworks/issues/144)/[#375](https://github.com/NGL321/patchworks/issues/375)'s
"driven field" is a forced dissipative medium. The word "driven" has been lending the constant
scalar the content of a forcing term. A constant supplies zero variation. #154's steady-state
account — "the drive pressures the network to act, action changes the world, so the supply of new
directions is never short" — presumes the outbound loop already works, which is the thing that
fails; the ticket's own §2 says a learning system needs external supply "at least at first", and
its Q5 then declined to name a supplier because "supply should not be short in any test this
project runs". #477 has since measured the famine. **[measured + inferred]**

**C3. Where the nonlinearity lives: image against build.** The image: "each node is a linear model
because it must be solvable, and the exchange between them cannot be linear because the thing
modelled is not." The build: the exchange is linear (ADR-0004), the body is frozen (ADR-0001), and
the one nonlinearity nobody trains. Open problem [#333](https://github.com/NGL321/patchworks/issues/333)
states it (uncut); proposal [#319](https://github.com/NGL321/patchworks/issues/319) answers it;
[#173](https://github.com/NGL321/patchworks/issues/173)'s evidence (Sheaf-ADMM's globally-shared maps
8.9% against input-conditioned 99.8%) supports it; the map holds it as fog "until a graph
transmits". This is the largest image/build contradiction in the record. **[measured]**

**C4. Retention: wanted, unplaced, fought, unrewarded.** ADR-0028 makes the depth gradient
learning's job with "nothing guarantees it appears" pre-registered.
[#276](https://github.com/NGL321/patchworks/issues/276) found no run ever placed one. #235 turned
the requirement into a per-cell floor `τ_c ≥ world_loop(c)` that 7–28 of 150 cells clear. #335
(band one-sided) crosses at 20k; #143 and #274 find learning making cells faster;
[#357](https://github.com/NGL321/patchworks/issues/357) finds non-normality and retention are
substitutes under the band. Underneath all of it: a one-step prediction objective never asks a cell
to hold anything — a cell whose evidence is refreshed every tick gains nothing by holding, and a
cell whose evidence is constant is told by §2 to hold one direction only. **[measured + derived]**

**C5. Instrument churn.** The bar was redefined four times in two weeks (Appendix B). Verdicts
flipped on probe direction ([#142](https://github.com/NGL321/patchworks/issues/142)), operator
(#274), horizon ([#178](https://github.com/NGL321/patchworks/issues/178), four times), precision
([#224](https://github.com/NGL321/patchworks/issues/224)), divisor
([#383](https://github.com/NGL321/patchworks/issues/383)), population (#385) and pooling
([#469](https://github.com/NGL321/patchworks/issues/469)). Roughly a third of the last hundred
closed tickets are "write the ruling" tickets. Interventions run on the mechanism: none. The map is
at GitHub's body byte cap and sub-issue cap, and the register grammar grew a precondition field this
week ([#417](https://github.com/NGL321/patchworks/issues/417)). The epistemics has outrun the
mechanics. **[measured]**

**C6. Stale grounds (the #345 class), concrete.** Appendix A lists thirteen sites with line
numbers. The one that matters for direction: **"reducing depth" was closed by
[#230](https://github.com/NGL321/patchworks/issues/230) under ADR-0021's min-over-edges amplitude
predicate ("a shallower dome inherits the same per-hop loss and fails at its own edge 2"), and
ADR-0026 retired that bar.** Under the time bar, depth sets `world_loop(c)` at every cell and so the
floor each cell must clear (15–16 at the apex, 3–9 at L1). The closure has not been re-read.
**[measured]**

**C7. ADR-0032 against ADR-0026 on the budget.** Wide, flat, full-rank transport spends `n`;
private width is what is left of `n`; ADR-0026 reads retention only in private width. #474 relieved
the collision by narrowing lanes to 3/4 at a 12:1 patch compression that `06` never ruled on (it
accepted 6:1 and rejected 24:1). The two premises still pull opposite ways. **[measured]**

**C8. ADR-0002 held on principle against its measured cost.** 52% of per-hop loss is the
reconciliation gain ([`research/150`](../research/150-effective-resistance-and-the-gauge.md)).
The one-step tick is defended on purpose (a cognitive system is not a function solver), uniformity,
and cost; the uniformity ground permits a *uniform* extra sweep count and it is refused on the other
two. Under the time bar this is less central than it was, but the "cross faster" side of #230's
diagnosis is closed entirely while the "hold longer" side is failing. **[measured]**

---

## 4. Abandon, freeze, keep

**Freeze now — costs nothing, buys capacity.**

- **The language domain**: spec `11`, `12`, stage 5, [#330](https://github.com/NGL321/patchworks/issues/330),
  [#331](https://github.com/NGL321/patchworks/issues/331), [#388](https://github.com/NGL321/patchworks/issues/388),
  [#448](https://github.com/NGL321/patchworks/issues/448)/[#463](https://github.com/NGL321/patchworks/issues/463).
  It cannot be interpreted before the sandbox transmits (#331 says so), it mints uncut problems, and
  it doubles the spec's surface. `12` itself says nothing in it assumes a sheaf. `[moves: map #127
  stage 5 row; nothing architectural]`
- **The demo surface and the decorative latency clause**: spec `10` is display-only by its own
  first line; [#334](https://github.com/NGL321/patchworks/issues/334) says the latency ordering
  restates hop count. `[moves: spec 08's second pass condition, already demoted to "reported"]`
- **Growth of the register and cutoff grammar**: #417, [#321](https://github.com/NGL321/patchworks/issues/321),
  [#424](https://github.com/NGL321/patchworks/issues/424). Keep the registers; stop extending the
  language. `[moves: process only]`
- **Holonomy and `H¹` beyond a diagnostic**: [#315](https://github.com/NGL321/patchworks/issues/315)
  stays open as the instrument it is. `[moves: nothing]`
- **Thermodynamic persistence**: already marked thin-provenance; keep it marked. `[moves: nothing]`

**Abandon for the diagnostic phase, revisit afterwards.**

- **The seven-level dome as the target of every mechanism experiment.** Use
  [#163](https://github.com/NGL321/patchworks/issues/163)'s shape-free builder at two or three
  levels: shorter world loops, cheaper runs (100k ticks is ~1.6 h on the full dome), and composed
  rank over two hops instead of seven. The image calls the dome abandonable; C6 says its depth
  closure rests on a retired bar. `[moves: #230's Out-of-scope entry "reducing depth"; spec 06's
  status as the run target; nothing in the cell contract]`
- **The one-sided `σ_max` band as the only control on `K`.** Add a leak toward `aI` (§8). `[moves:
  ADR-0015 gains a second term; ADR-0008's "a projection is not an objective" is untouched, because
  a leak carries no signal]`
- **"No exogenous input at the cold start."** Motor babble as a world-side developmental phase
  (§8). `[moves: ADR-0009's Bootstrapping exposure from "curiosity drive, gated" to "supply, owed";
  spec 04's Route selection; map #127's fog item "what supplies the variation"; map #1's "no
  reward" is untouched — babble carries no error signal, no credit assignment, no satisfaction
  detector, which is [#449](https://github.com/NGL321/patchworks/issues/449)'s own test; ADR-0005's
  "not a schedule" is untouched — an anneal on the world's torque sets no cell's rate]`

**Keep — built, image-true, and not the problem.** The sheaf with heterogeneous masks; the two
local rules; the two-phase tick; boundary cells and the written-or-read ban; the drive as a motor
edge; `H⁰` as private features; retention in `K`'s spectrum; the spectral floor (it keeps lanes from
closing until variation arrives — its job, correctly understood, is defensive); isometric transport
as the maps' target (#453 shows it is reached on the one excited direction, which is the mechanism
working on a starved input).

---

## 5. What matches the motivating image

| image element | build | evidence | verdict |
|---|---|---|---|
| Free-abelian compression via cells holding different generator subsets; a sheaf glues them | masks, lanes, `H⁰` | [#411](https://github.com/NGL321/patchworks/issues/411), ADR-0032 | **matched** |
| Local rules, no gradient across a seam | `learning.py`, `torch.func`, perturbation test | 1,598 tests, kill-tested | **matched** |
| Dissipative chamber; echoes fade | one-step tick; median cell contracts | #274 (median block radius 0.900, [#375](https://github.com/NGL321/patchworks/issues/375)) | **matched** |
| Perturbations enter at a sensory wall; another part of the wall is motor | boundary cells, ADR-0016 | built | **matched** |
| Two branches (model, action) meeting in a core | vision lattice + somatomotor column → core | built; the column is the collapsed one | matched in shape, dead in function |
| Standing assertion in place of reward | drive boundary cell | ADR-0009, #449 | **matched** |
| Commitment as `H⁰` insulation | private features invariant under reconciliation | built; unmeasured in use | matched, untested |
| Stable regions as channels | ADR-0022's learned channel | 14.2x taught vs 3.66x untrained; rank one | matched, one direction only |
| Levels as a shaping prior, not an index | [#181](https://github.com/NGL321/patchworks/issues/181), [#251](https://github.com/NGL321/patchworks/issues/251) | enforced | **matched** |
| A canonical repeated unit | one shared frozen body | ADR-0001 | matched |
| Explicit abandonability of the shape | `06`, #163 | builder written | **matched** |
| Nodes linear, **exchange nonlinear** | exchange linear, node has the only nonlinearity | ADR-0004, #333 | **inverted** |
| "Zoom and the same structure recurs" | a taper; no recursion | — | unbuilt |
| Persistent information structures assembled by ripples | no assembling mechanism; coupling adds no non-normality | #375 §2, twice | **unrealised** |
| "Not doing one thing" — many signals, many channels | one composed channel, one direction | #497 | **unrealised** |
| Heterarchy through lateral edges | lateral edges built; contribution unmeasured | [#327](https://github.com/NGL321/patchworks/issues/327) uncut | untested |
| A found embedding space (the LLM clause) | a random frozen body | ADR-0001 | **contradicted** |

---

## 6. Holes and contradictions in the image

1. **No source of variation or energy.** The chamber is dissipative and "new signals constantly
   enter from the rim" — from what? In the sandbox nothing enters unless the agent moves or a human
   intervenes (`reset()` is called once at start; the demo hands are the only world-side writers).
   The drive is a constant. This is the load-bearing hole: it is the one the build fell into.
2. **Nonlinearity placement** contradicts the build (C3).
3. **"Persistent information structures" replacing free energy** has no operational definition. The
   record's best attempt (#375's five instruments) found the field's coupling contributes nothing
   to the only mechanism the literature offers, and the toy plus §2 say why: the one assembling
   mechanism is map learning, which concentrates under low-rank ripples.
4. **"Decision-making must fall out of the same algorithm"** needs at least two outbound directions
   for the world to select between; the build produces one (#497). Route selection "by the world"
   is unfalsifiable on a rank-one channel.
5. **The LLM embedding-space clause** cuts against the frozen random body: the embedding LLMs
   supply is *found* by learning; here it is drawn once and frozen. ADR-0001 says the freeze is the
   top rung of the flex ladder, non-load-bearing; the image should say the same or drop the clause.
6. **Recursion** ("zoom again and the same structure recurs") is an unbuilt promise, not a
   contradiction.
7. **The free-abelian derivation's "therefores"** ("must be free abelian, therefore heterogeneous,
   therefore a graph with heterogeneous connections") are asserted. The sheaf is a good choice; it
   is not derived.
8. **Scale.** An image of AGI and complex animals driving a world of ~20 numbers through seven levels
   chosen "to make a measurement legible". The funnel fog item exists because the image cannot say
   what compression ratio it wants.
9. **An internal tension the record already owns**: levels shape the ripples (the chamber) yet
   levels are not an earned index (#181). Acknowledged on #251; consistent with treating the dome as
   scaffolding, which §4 recommends.

---

## 7. Higher-level patterns the record is missing

1. **Rank and variation accounting.** No document asks, per cell, how many directions the world
   excites there. #154 §3 named the instrument (participation ratio of the disagreement time series
   against `m_e`) and gated it on the outbound clause passing — which cannot happen without
   variation. It should be the first read, not the last.
2. **The Hebbian-collapse mechanism.** Outer-product rules under low-rank input concentrate (Oja,
   1982, with one input converges to its direction). The record treats the collapse as a cause to be
   found by six 100k runs; it is a property of the rule's arithmetic, testable in one short run.
3. **The constancy death spiral** (§1). Any intervention must break the loop somewhere; the world is
   the cheapest place.
4. **Local-versus-composed as a law**, not a series of surprises (C1).
5. **Objective indifference to retention** (C4). Nothing in `½‖p − target‖²` values holding.
   Retention can only appear if holding predicts better than refreshing, which needs the evidence
   to be intermittent or delayed — the change gate the spec specified and never built is the
   nearest thing on the record to that.
6. **Verdict fragility as a process property.** Seven axes on which a verdict has flipped (C5)
   means every number needs its surface stated — which the team has now ruled
   ([#437](https://github.com/NGL321/patchworks/issues/437)) — but also that the instrument set
   should be frozen (§8).

---

## 8. Mitigations

### 8.1 Already on the record, and useful

- **Exogenous arm / torque babble** — [#496](https://github.com/NGL321/patchworks/issues/496), filed
  as a *test* of constancy. It is the mechanism.
- **Pre-training waves** — map fog, closure withdrawn on #231 §3.1; the biological answer
  (spontaneous activity before sensation) under another name.
- **Exploratory writes at the actuator boundary cell** — map fog.
- **World-side re-supply** (resets, arena churn, unsticking the arm) — map fog, and the map's own
  "candidate with a measured symptom behind it".
- **Curiosity drive** — ADR-0009's named response, gated.
- **Evidence-conditioned maps** — #319, fog; the image's nonlinear exchange.
- **Direct parameterisation of a stable `K`** — [#318](https://github.com/NGL321/patchworks/issues/318),
  adopted as forward normalisation; the leak half not taken.
- **A bound on `‖Δb‖`** — [#317](https://github.com/NGL321/patchworks/issues/317).
- **The change gate and the protected channel** — spec `05`, specified and unbuilt.
- **The shape-free builder** — #163.
- **The excitation-rank read** — #154 §3.
- **Retune `η` and `c`** — `learning.py` flags both as "the first thing to retune once something
  measures a run"; nothing has.
- **Vary the drive as a probe** — [#495](https://github.com/NGL321/patchworks/issues/495).

### 8.2 Obvious, cheap, and not yet on the record

- **(a) Two reads before any new run.** The family fit (done above, §2.4) and the mechanism note's
  predictions 1–3 on one 5k-tick run: alignment of a dead cell's `K` row space with its mean
  evidence; per-cell excitation rank against `modes_retaining`; the persistent error's magnitude and
  direction-stability. About ten minutes of compute. `[moves: nothing; re-sequences #496]`
- **(b) Annealed motor babble as a developmental phase**, world-side, applied through `act()` as
  #481 already specified for the test: uniform torque noise of amplitude `A(t) = A₀·max(0, 1 − t/T_b)`,
  then off. Two invented constants (`A₀`, `T_b`) with a stated retune duty — the map's own rule is
  that invented constants are deferred until necessary, and this is the moment. `[moves: ADR-0009
  Bootstrapping; spec 04 Route selection; map fog item]`
- **(c) A leak on `K`**: `K ← K − μ(K − aI)` after each step. Parameter-local, carries no signal,
  bounds the rank-one part at `η‖Dᵀē‖/μ` and so bounds `ρ` away from zero. One constant. `[moves:
  ADR-0015; the definition site gains a `μ` with provenance]`
- **(d) `c = η_K/η < 1`.** The toy shows a bias that absorbs the persistent error protects the
  operator; giving the biases a head start is the cheapest version. `[moves: DEFAULT_OPERATOR_RATE_RATIO,
  already flagged for retuning]`
- **(e) A shallow diagnostic dome** (§4). `[moves: #230's depth closure]`
- **(f) One frozen canonical table**, reported by every experiment on the same horizon and seeds:
  per-column `ρ(K)` and `modes_retaining`; per-cell excitation rank; composed rim-to-apex rank; the
  conduction ratio inbound and outbound; arm travel per window; disagreement energy paired with
  per-edge effective rank. 100k, seeds 42/43/44, 30k printed beside it. `[moves: process; the minimal
  form of #321]`
- **(g) Stop taking transmission numbers on a frozen world.** Every number so far is a number about
  a rank-one world. `[moves: the standing status of ADR-0026's ledger, which its own text already
  calls a build that no longer exists]`
- **(h) One mechanical ticket for Appendix A.** `[moves: the thirteen sites listed]`

### 8.3 Joint mitigations

● direct, ○ indirect. Columns are the register rows and tickets each mitigation bears on.

| mitigation | #120 freeze | #477/#482 collapse | #497 rank | #154 famine | #341 prior→rim | #375 sustain | #329 K settles | #335 band | #333 nonlinear | #78 bar |
|---|---|---|---|---|---|---|---|---|---|---|
| babble (b) | ● | ● | ● | ● | ○ | ● | ○ | ○ | | ● |
| leak (c) | ○ | ● | | | | ○ | ○ | ● | | ○ |
| `c < 1` (d) | | ● | | | | | ○ | ○ | | |
| shallow dome (e) | ○ | | ● | | ● | ○ | | | | ● |
| frozen table (f) | | | | | | | | | | ○ |
| conditioned maps (#319) | | | ● | | ● | ○ | | | ● | ○ |
| reads (a) | | ● | | ● | | | ● | ● | | |

**Babble + leak + shallow dome + the frozen table is one experiment set** that addresses nine rows at
once and unblocks the bar #78 is cut on. Evidence-conditioned maps are the next lever if the loop
closes and composed rank still does not rise with excitation rank — at which point it is
ADR-0004's linear-exchange assumption that falls, not the sheaf.

---

## 9. Recommended direction, and a sequence for the implementation pass

0. **Freeze** §4's list. One mechanical ticket for Appendix A. Stop redefining the bar: ADR-0026 as
   it stands, on #474's graph, read once at 100k, is the bar until §9.3.
1. **Analytical foundation** (this week): the mechanism note's reads on one 5k-tick run (8.2 a). If
   prediction 1 fails, §2 is wrong and §9.2 still runs, because babble is cheap either way.
2. **Mechanism experiments**, shallow dome first, then full:
   - E1: frozen baseline versus annealed babble (`A₀` at the scripted pusher's torque scale,
     `T_b = 10k`), 30k ticks, seeds 42/43/44, the frozen table. **Pre-registered:** at 30k, arm
     travel per window `> 0` under its own command with babble off; somatomotor and apex
     `modes_retaining ≥ 6` at the median; composed rim-to-apex rank `> 1.5`; inbound reached
     fraction `> 0`. `g > 0` at the somatomotor column and the apex unchanged, since the apex's
     constancy is the drive's (#481, #495).
   - E2: E1's winner + leak `μ ∈ {10⁻⁴, 10⁻³}`; + `c ∈ {1, 0.1}`. **Pre-registered:** collapse
     rate falls in proportion to `c`; `ρ` bounded away from zero by `μ`.
   - E3: the winner on the full dome at 100k.
3. **If the arm keeps moving under its own command after babble is withdrawn**: read ADR-0026 in
   both directions on that surface, then the demo ([#78](https://github.com/NGL321/patchworks/issues/78)).
   Expect the outbound universal to be the hard clause: a world of ~20 numbers bounds the excitation
   rank, and the apex must reach 70 L1 cells on a channel of at most that rank.
4. **If it does not**: #319 on the shallow dome. The question becomes whether linear exchange can
   carry a nonlinear world at all — the image's own claim, tested for the first time.

Everything in the sequence keeps map #1's five commitments (APC, small rigid MLPs, sheaf GNN, local
rules only, two-phase tick) and its no-reward constraint intact. What moves is listed in §4 and §8.

---

## 10. Candidate grilling tickets

Listed for the team to mint under ADR-0029; none is minted here.

- **G1.** Is the collapse the outer-product mechanism? (Mechanism note predictions 1–3, one run.)
- **G2.** Is cold-start supply owed rather than gated, now that #477 has measured the famine? (Re-read
  #154 Q5.)
- **G3.** Does "reducing depth" survive the retirement of the amplitude bar? (#230's closure under
  ADR-0026.)
- **G4.** Should `K` carry a leak, and is a leak an objective under ADR-0008?
- **G5.** Freeze stage 5 until the sandbox transmits?
- **G6.** The canonical table: which numbers, which horizon, which seeds.
- **G7.** Is the apex a constant-evidence cell by ADR-0009's design, and what does that do to "the
  apex as the stable structure the rim's influence is conducted through"?
- **G8.** Is composed rank bounded by excitation rank as a construction-time argument, and does
  that close per-map remedies as a species the way #184 closed per-hop multipliers?

---

## 11. What this audit could not verify

- No run of the real model: this container has no torch (the wheel index is blocked by its proxy),
  so the suite was not run and no live read was taken. Nothing under `docs/audit/` or
  `prototypes/audit-collapse-toy/` is read by any test (`tools/constant_registers.py` scans
  `src/patchworks` only; the README count test globs `docs/adr`, `docs/spec`, `docs/research`).
- The toy omits `encode`'s fusion, reconciliation's feedback into the target, and the chart round
  trip. It shows direction and cause, not rate.
- §2's persistent-error premise at dead cells is inferred from #202 and #339, not read.
- The family fit in §2.4 uses statistics computed for another purpose; consistent-with, not proof.
- Two weeks of record were read in a few hours. The #345 class — a decision resting on a ground
  later retracted — applies to this document too; every figure here names its ticket so it can be
  re-read when the ground moves.

---

## Appendix A. Stale grounds, by site

| # | site | what it says | what is current |
|---|---|---|---|
| 1 | `docs/spec/06-graph-topology.md:231` | the per-cell gain is `γ / Σ_e m_e`, so the apex check is ~6% easier | `gain_v = γ/(g_v²·c_v)` since #190; the apex's `c_v` is unchanged by one more edge, so the figure is false, not stale |
| 2 | `docs/spec/04-action-and-the-boundary.md:344` | "strength is fan-out" argued from the `γ / Σ_e m_e` dilution | the dilution mechanism no longer exists; the conclusion may stand, its argument is gone |
| 3 | `docs/spec/01-cell-and-sheaf.md:386, :434–438, :525`; `CONTEXT.md:101` | `4 × 32`, a 28-dimensional kernel, `m = 8`, `48 → 8`, "3 degrees of freedom of 128" | `3 × 32`, 29-dimensional kernel, `48 → 4`, 3 of 96 since #474 |
| 4 | `docs/spec/11-the-language-graph.md:308` and its tables | "Interior `m = 4`, boundary `m = 8` are the dome's" | 3 and 4 |
| 5 | `docs/adr/0028-…md:242` vs `:107` | Alternatives: "Rejected on measurement, not on argument"; Decision: "not rejected on measurement either" | one document, opposite claims |
| 6 | `docs/adr/0009-…md:161–183` | the width ladder gated on `λ = DEFAULT_SPARSITY_PRESSURE` | annotated as due after ADR-0031; the pre-specified re-read is unrun |
| 7 | `docs/adr/0026-…md` (its own ledger) | 15.4x, 1.1–8.5x, 40–85 of 150 | all read at `m = 4/8`; the first read on #474's graph is owed (#483) |
| 8 | `docs/adr/0002-…md:106–112` | reopens on #237 | #237 fired, onto rank; the ADR does not say so |
| 9 | map #127, Out of scope, "reducing depth" | closed on ADR-0021's min-over-edges amplitude predicate | ADR-0026 retired that bar; unread under the time bar |
| 10 | `src/patchworks/restriction.py:662` and `:697` | `flatness` defined twice, byte-identical | dead code; the second shadows the first |
| 11 | `benchmarks/graph_transmission.py:44, :437, :447`; `benchmarks/untrained_fixed_point.py:63` | `γ / max(Σ_e m_e, ρ²·deg)`; `interior_m = 8` | struck by #190; 3 |
| 12 | `docs/spec/09-the-build-stack.md` | the adapting surface is biases and maps; the with-gradient tick cost is an estimate | `K` trains since #139; no benchmark times the with-gradient tick |
| 13 | `docs/spec/06-graph-topology.md:188, :219, :225` vs `:416, :440` | private dimension ~8 through L3–L6 and ~16 at L7; `16 → 15` when a drive edge is added | the file's own table reads 14 flat across L3–L6 and 19 at the apex after #474 |

## Appendix B. The bar, redefined

~0.37 per hop inherited (struck by #142) → 921x per hop, isotropic (#120) → a 1e14 deficit
(withdrawn by ADR-0022) → the bottleneck ratio `≥ 1` (ADR-0021, [#181](https://github.com/NGL321/patchworks/issues/181))
→ measured 8.7e-10 rim→apex and 1.3e-8 apex→rim, shortfall 1.15e9x ([#214](https://github.com/NGL321/patchworks/issues/214))
→ four corners spanning 7.2x ([#232](https://github.com/NGL321/patchworks/issues/232)) → recharter
onto retention (#230) → the conduction ratio `τ̂/|loop| ≥ 1`, shortfall 15.4x
([#242](https://github.com/NGL321/patchworks/issues/242), ADR-0026) → the stalk relay, `τ` 2.9–10.3,
shortfall 1.1–8.5x (#274) → the float32 gate (#224) → the divisor moves to `world_loop(c)` (#383)
→ pinned at exactly 0 by 70 zero-private L1 cells ([#379](https://github.com/NGL321/patchworks/issues/379), #385)
→ unpinned by narrowing lanes (#474) → unread.

Every step was defensible. The sum is that in two weeks the destination has never been read on a
surface where the world varied.

## Appendix C. What was read

`CONTEXT.md`; `docs/motivating-image.md`; all six registers; map #127 in full (144 KB); the
rulings on #120, #154, #230, #235, #335, #375, #385, #453, #474, #477, #481, #497; the open tickets
#487, #488, #495, #496, #498; `README.md`, `AGENTS.md`, `docs/agents/*`; `learning.py` in full,
`body.py`, `graph.py`, `restriction.py` targeted; three digests prepared for this audit covering the
twelve spec files, the thirty-two ADRs with seven research passes, and the code, twenty benchmarks,
twenty-seven prototypes, the test suite and the git history; the committed 100k checkpoints of
#132 for the fit in §2.4.
