# ADR-0026: Rim-core influence is a conduction ratio, not an arriving amplitude

**Status:** accepted

## Context

Settled in [#242](https://github.com/NGL321/patchworks/issues/242) and written here by
[#257](https://github.com/NGL321/patchworks/issues/257). **This is
[#127](https://github.com/NGL321/patchworks/issues/127)'s operative bar.**
[ADR-0021](./0021-rim-to-core-detectability-is-a-bottleneck-ratio.md) is **kept** as the sufficient
per-edge diagnostic and is no longer what the map is read against; its *What this predicate is for*
section records the demotion.

**Amended by [#224](https://github.com/NGL321/patchworks/issues/224), written by
[#381](https://github.com/NGL321/patchworks/issues/381): the reading is gated on the runtime
precision floor.** The predicate below is untouched; what is added is a **licence** on what a PASS
claims and a **gate** on when `τ̂` is valid at all, because this ADR's instrument reads a quantity
float32 cannot represent. Read *The reading is gated on the runtime precision floor* before reading a
verdict off a `τ̂`.

ADR-0021 asked for this replacement in its own words — *"#242 owes the replacement. It writes the
falsifiable form of the influence reading, or rules that this ratio stays the bar and says what a
stage-3 gate then consists of."* #242 wrote the falsifiable form. This ADR is that form.

The obligation is older than the ticket. [#230](https://github.com/NGL321/patchworks/issues/230)
rechartered the effort onto **retention** and re-scoped ADR-0021 rather than replacing it, which left
#127 carrying an unowned promise: the operative bar stays the bottleneck ratio *until something
falsifiable replaces it*. Nothing was owed to anyone until #242 took it.

### Why the amplitude bar had to move, and it is not because it failed

**The decisive defect is that the gate and the work were different quantities.** Stage 2 is
retention. ADR-0021 reads arriving magnitude. A stage-2 result therefore could not move the stage-3
gate *by construction* — which is why [#232](https://github.com/NGL321/patchworks/issues/232) could
fire a clean FAIL and carry no information about the destination. A bar that no amount of the work in
front of it can move is not a bar; it is a bystander.

Four facts already in the record say the deficit is temporal, and none of them is new measurement.
This ADR needed none:

- **#230 diagnosed the axis.** Seven hops rim to apex, **one tick each**, in a graph where nothing
  holds a value for more than about one tick. Every remedy family the map had opened was spatial.
- **[#143](https://github.com/NGL321/patchworks/issues/143) named what `τ` must beat, in terms:**
  *"`H⁰` is commitment against the graph, `τ` against the cell's **own round trip**."* The bar below
  is that round trip, and nothing else.
- **[`05-timescales.md`](../spec/05-timescales.md) gives the reading today** (*What the live read
  says*): `τ` is **flat at about one tick graph-wide — 0.91 at the apex against 0.99 at the rim**,
  with no depth→timescale gradient at all, and slightly *inverted*. **Read on the chart's *direct*
  round trip**; with the stalk relay included the apex is 1.6 to 13.1 ticks and the inversion is
  larger ([#274](https://github.com/NGL321/patchworks/issues/274)). Flat under either operator,
  which is what this bullet needs; the magnitude is amended under *The shortfall* below.
- **#232 measured that diagnosis directly** rather than inferring it: the horizon ladder is flat from
  32 to 512 and the binding edge peaks at **tick 18** — the stimulus persisted and the arriving
  deviation did not grow.

**The shortfall on this bar is 15.4x**, not 1.3e9x: `τ̂ ≥ 14` at the apex against a measured 0.91.
Same graph, same measurements, different question. The gap only looked unclosable because the bar
read a magnitude where the destination names a loop. #143 further records that `λ = 0.99` gives
`τ ≈ 99.5`, **inside [ADR-0015](./0015-the-cell-operator-band-is-on-the-spectral-norm.md)'s band** —
so the bar is reachable by the parameterisation that already exists. This is the first time the map
has had a visible route rather than a deficit.

> **Amended by [#274](https://github.com/NGL321/patchworks/issues/274): the 15.4x is reading the
> wrong operator.** That 0.91 is `05`'s chart-only `τ` — the chart's *direct* round trip, with the
> relay through `decode`, the node stalk and reconciliation's damped aperture omitted. Read on the
> full loop over nine driven seeds the apex's `τ` is **1.64 to 13.09 ticks**, so the apex conduction
> ratio is **0.12x to 0.93x** and the shortfall is roughly **1.1x to 8.5x**, median across seeds
> ~3.8x rather than 15.4x. **This ADR's decision is untouched**: the predicate is
> `τ̂_c / |loop(c)| ≥ 1` either way and the reading is short on every seed. Only the published
> magnitude moves, and the figure above is kept as read rather than rescaled, on #206's precedent
> for `01`'s recorded margins.
>
> Two fences travel with the amendment, and they matter more than the number. **The defendable
> claim is that 15.4x is reading the wrong operator, not that 3.8x is the shortfall** — the
> corrected reading is a *range* where the old one was a point, and `τ = −1/ln ρ` diverges as
> `ρ → 1`, so the seed spread is the honest object. And **neither number is this ADR's `τ̂`**: the
> quantity below is the e-fold decay of a **paired counterfactual deviation in private features**,
> where `05`'s regional `τ` is `−1/ln ρ` of a per-cell linearised loop that ignores the graph
> coupling and holds neighbours' `y_e` exogenous. They are two instruments, the record has been
> using one as a stand-in for the other, and it should stop; the corrected number is a better
> stand-in and is still a stand-in. #99 owes the real reading.

## Decision

### The predicate

For a cell `c`, fork a perturbed and an unperturbed run from a common state — ADR-0021's **paired
counterfactual, unchanged and not re-derived**. Read the paired deviation restricted to `c`'s
**private features** (`H⁰`, where retained state lives by construction, and the one place
reconciliation cannot move it) and take its e-fold decay time `τ̂_c`. The **conduction ratio** is

> `τ̂_c / |loop(c)|`

where `|loop(c)|` is the tick length of the shortest cycle through `c` that reaches the rim and
returns. The predicate is ADR-0021's own widest-path shape with the quantity swapped:

> `max` over paths `P`, `min` over cells `c ∈ P`, of `τ̂_c / |loop(c)| ≥ 1`

evaluated **per trial**, reported as a **distribution over trials** with the bar at the **median** and
p05 / p25 / p75 / p95 alongside — ADR-0021's precedent, and
[#202](https://github.com/NGL321/patchworks/issues/202)'s reason for it.

It is stated **twice**, and **the two directions do not share a quantifier**:

- **Inbound — a swept single-source count.** For each rim cell `r`, does `r`'s loop close? The
  reading is the **fraction of rim cells whose loop closes**, reported **per stratum** — vision,
  proprioception, touch, #232's own stratification — and **never averaged across strata**. The
  strata's stalk widths are 48, 2 and 1, so an average hides a 256-cell read inside two 3-cell ones.
  **The bar is that every stratum's reaching set is non-empty.** A context blind to touch entirely is
  not global context.
- **Outbound — a universal.** Over **L1 predicting cells and the actuator boundary cell**, every one
  of them. Not a max over paths.

**The falsification**, stated as such: **no loop closing on any path**, or **no counterfactual
dependence at all — a private-feature deviation that does not clear the runtime precision floor**,
`eps_f32 · ‖state‖` at the cell (see *The reading is gated on the runtime precision floor*).

> **Amended by [#224](https://github.com/NGL321/patchworks/issues/224).** The second limb read *a
> private-feature deviation **bit-identical** between the two branches — no counterfactual dependence
> at all*.

*Superseded by #224.* Bit-identity is the one thing float32 rounding reliably **prevents**: two
nearly-equal trajectories differ by their own rounding, so as written the limb could not fire in the
precision the architecture runs in — a falsification clause that is unfalsifiable. The restatement
names the same absence in a **readable** condition, and it is not a weakening: a deviation below the
floor *has not arrived* (#224), which is exactly what the struck limb was reaching for. Quoted rather
than deleted because what changed is the *test*, not the claim.

### The bar is derived, not invented

The bar is `1`, and `1` is not a choice. The conduction ratio compares a cell's measured retention
time against **the loop's own length in ticks**, so the predicate says exactly *the cell still holds
what it sent by the time the answer gets back* and nothing more. There is **no multiplier and no
safety factor**, for the reason ADR-0021 chose `k = 1`: every invented constant in this map's history
has later been found to have none, and
[#142](https://github.com/NGL321/patchworks/issues/142) struck the inherited ~0.37/hop precisely
because it was back-computed with no derivation.

### `|loop(c)|` is computed from the mask, not inherited

**This is the one quantity in the predicate that had never been checked.** #242 recorded it as
unverified in its own provenance: `|loop(apex)| = 14` was taken as *7 hops out and 7 back*, from
#230's seven-hop figure and [`01-cell-and-sheaf.md`](../spec/01-cell-and-sheaf.md)'s *Unit delay*
(*"Every edge costs exactly one tick. Graph distance is literally temporal distance"*). **The shortest
rim-returning cycle had never been enumerated on the actual mask**, and taking a bar from a round
number is how invented constants get in.

**Enumerated on the default dome** (`patchworks.graph.build_graph()`, `DEFAULT_SPEC`): 414 cells, 682
edges, a sensorimotor rim of 263 cells — 256 patch, 3 proprioceptive, 3 touch, 1 actuator. A
breadth-first sweep from that rim gives `d(c, rim)` for all 150 predicting cells, and
`|loop(c)| = 2 · d(c, rim)`:

| level | cells | `d(c, rim)` | `\|loop(c)\|` |
|---|---|---|---|
| L1 | 70 | 1 | **2** |
| L2 | 20 | 2 | **4** |
| L3 | 16 | 3 | **6** |
| L4 | 14 | 4 | **8** |
| L5 | 12 | 5 | **10** |
| L6 | 10 | 6 | **12** |
| L7 (apex) | 8 | 7 | **14** |

**The inherited 14 survives enumeration**, and it is now derived rather than assumed. Two further
facts the enumeration settles, neither of which was safe to assume:

- **`d(c, rim)` is exact, not a minimum with a spread.** Every cell at a level sits at the same
  distance from the rim — the taper's lateral fill opens no shortcut past a level, so there is no
  cell whose loop is shorter than its level implies.
- **14 is not an artifact of allowing the walk to retrace itself.** Read as a *genuine* cycle —
  outbound and return sharing no vertex but the apex — all eight apex cells still close at exactly
  **14 ticks**, each reaching two distinct rim cells by vertex-disjoint seven-hop paths. The round
  trip and the true cycle agree, so the predicate does not turn on which reading is meant.

**`|loop(c)| = 2 · level` is a fact about this graph's wiring and not a licence to index by level.**
The quantity is computed per cell from the graph; that it coincides with twice the construction
layout's level on the default dome is a coincidence of the current taper, and
[#181](https://github.com/NGL321/patchworks/issues/181)'s standing rule holds — per-edge and per-cell
are properties of the graph, per-level is a property of the *shape* imposed on it. **A changed
`DomeSpec` changes these numbers, and they are recomputed rather than quoted**: the enumeration above
is the reading for `DEFAULT_SPEC`, and a graph with a different `core_sizes` has a different ladder.

The **drive boundary cell is not part of the rim** for this purpose. It sits at the internal rim,
attached to all eight apex cells, and its own distance to the sensorimotor rim is 8.

### The quantifiers are asymmetric, and the asymmetry is the architecture's

ADR-0021 stated both directions in the same max-over-paths form — an existential. The destination's
two clauses are not the same kind of claim, and flattening them to one quantifier would misstate the
architecture:

- **Inbound is a count.** Setting global context does not require information from *every* rim cell;
  it requires it from *some*. That is precisely what licenses the apex being **sparsely connected** —
  the taper's decimation is permitted by the inbound claim being a count. An existential is too weak
  (one path is not *collectively*) and a universal is too strong (nothing needs every patch).
- **Outbound is a universal, with no exceptions.** The apex is the **context setter** for the
  network. Global context matters no matter what node you are or what position you occupy, so every
  cell in the outbound population must receive from the apex. This direction admits no aggregate
  form.

### The outbound population, and why the exclusion is about consequence rather than reception

The outbound universal runs over **L1 predicting cells and the actuator boundary cell**. Sensory
boundary cells are excluded, and **the ground is consequence, not reception**:

- **Sensory boundary cells do receive the apex's signal.** They are *written*, and the outside write
  lands **after** the message-passing phase and is always the last word
  ([`CONTEXT.md`](../../CONTEXT.md), *Boundary cell*;
  [ADR-0016](./0016-a-boundary-cell-is-written-or-read-never-both.md)), so what arrives is
  overwritten the same tick. There is no test there to pass or to fail.
- **The actuator boundary cell is included.** It is *read*, not written, so influence arriving there
  is not erased — it is **action**, and it is the acceptance demo's own instrument.
- **The drive boundary cell is excluded.** Attached at the apex, read by nothing, `m = 1`.

This is not a softening of the universal. It is the universal stated over the cells at which arriving
influence has a consequence at all.

### *An apex stable enough to conduct it* is this predicate's subject, not a third reading

`τ̂ ≥ |loop|` at the apex **is** the stability-to-conduct claim. #127's *Done when* names three
things — inbound influence, outbound influence, and an apex stable enough to conduct it — and the
third is **absorbed here** rather than owed as a separate instrument. Written down explicitly so it
is not reopened later as a missing reading.

### The stability tension is named here and priced at #235

Conducting needs `τ̂ ≥ 14`, which puts `λ` near `−1/14`. `05-timescales.md` warns that a cell placed
at `ρ = 0.99` is **one bias update from crossing**, and that the slow-and-stable band is thin: of
20,000 draws at one candidate width, `ρ ∈ [0.98, 1)` holds **0.15%** against **0.53%** at `ρ ≥ 1`.
**Stable enough to conduct** and **stable** pull against each other, and the margin thins as this
predicate approaches PASS.

**Named here, priced at [#235](https://github.com/NGL321/patchworks/issues/235)**, which already
weighs retention against [ADR-0005](./0005-timescale-is-persistence-not-a-schedule.md). It is the
same collision in the stability currency and belongs in one place rather than two. **This ADR does
not resolve it.**

### The fraction threshold is deferred, with a trigger and an owner

The inbound reading publishes a fraction; the bar is only that the set is **non-empty**. No threshold
is set, and the deferral is on the record with its reasons:

- No derivation for one exists.
- The predicate currently reads **zero in every stratum**, so a threshold decides nothing that is
  live.
- **It is owed at the first PASS of the stratum clause.** The ticket that reports the first non-empty
  reaching set opens it. That is the trigger and that is the owner — the same idiom as
  [#8](https://github.com/NGL321/patchworks/issues/8)'s funnel.

**Invented constants are deferred until necessary, not banned outright.** #242's clarification,
recorded here because this ADR's own reasoning leans on it: what is being avoided is a number
invented *before* anything needs it — #142's ~0.37/hop, and the `k = 3` ADR-0021 declined. A constant
that proves **necessary** can be added later. A derived non-zero is simply the more defendable place
to stand while the reading is zero.

### How it is read

- **Impulse source, not sustained.** #232 verified by test that the impulse is erased at the source
  after one tick, so the decay that follows is clean; a sustained clamp confounds decay with drive.
- **Private-feature projection.** The masks give it: `Dome.private_projection` is fixed at
  construction and invariant under learning, and it keeps exactly the directions reconciliation
  cannot move.
- **`τ̂` is peak-to-`1/e` in ticks.** Per trial, project the paired deviation onto the cell's private
  features, find its peak tick, and read `τ̂` as the ticks from that peak until it falls to `1/e` of
  peak.
- **Median over trials**, with the quantiles alongside.

**No new instrument.** [`benchmarks/detectability.py`](../../benchmarks/detectability.py) already
forks paired branches and already keeps **per-tick node stalks** for the cells it records, already
reports the paired ratio as `[ticks, edges]`, and already sweeps the horizon ladder. This is a
**reduction over quantities it computes**, not a new measurement surface. It is on the default branch
as of [#239](https://github.com/NGL321/patchworks/issues/239).

### The reading is gated on the runtime precision floor

Added by [#224](https://github.com/NGL321/patchworks/issues/224), which ruled that **float32's
granularity is the architecture's noise floor** rather than a defect, and found that the ruling lands
on *this ADR's instrument* rather than on the architecture. **The predicate is untouched** — the bar
of `1`, the two quantifiers and the loop enumeration all stand. Two clauses are added and one
Consequence is struck.

**The licence.** `τ̂` is read on the **float64-cast surface**
([`benchmarks/detectability.py`](../../benchmarks/detectability.py)'s `double_precision`, which is
instrument-only and says so). **A PASS is therefore a claim about the graph's conduction, and not
about the float32 build's.** This is a scope on the result rather than a caveat in passing: it is what
stops a later session reading a float64 PASS as a runtime verdict.

**The gate, and it is the operative half.** `τ̂` is valid for a cell only if the **projected paired
deviation stays above `eps_f32 · ‖state‖` at the `1/e` crossing** —
`patchworks.tick.precision_floor`, the one place that quantity is computed. A reading that fails the
gate is reported **alongside** `τ̂` rather than dropped, so the float64 number stays visible next to
the fact that a float32 build has no signal there. **Nothing is invented**: `eps_f32` is the machine's,
held as `torch.finfo(torch.float32).eps` at a definition site in the architecture register rather than
as a typed literal ([ADR-0018](./0018-a-derived-constant-is-derived-where-its-dependency-lives.md),
[ADR-0020](./0020-a-repeated-default-is-a-constant-with-a-definition-site.md)), and `‖state‖` is read.

**Corroborated by amplitude-independence, which costs nothing new.** `double_precision` already
documents the tell — a transported deviation is linear in what was injected, so the reading is flat in
the amplitude it was measured at, and rounding is not — and `linearity` already runs that ladder for
the bottleneck ratio. **The same ladder is required of `τ̂`**, which reuses an instrument already on
the default branch and invents nothing. The gate says whether *this* reading cleared the floor; the
ladder says whether the quantity is a transported deviation at all. `τ̂` has no implementation on the
default branch yet — ADR-0026 promised the reduction and it was never written — so the ladder lands
with the reduction, on [#379](https://github.com/NGL321/patchworks/issues/379), which owns both.

**Why a gate rather than a second architectural bar.** *The arriving deviation must be representable
in float32* was considered and **ruled out on the image**: it would make a numerical parameter part of
the transmission budget, and it would buy nine decades of headroom for a single decaying ripple in an
unsustained fork — precisely the object [`docs/motivating-image.md`](../motivating-image.md) says is
not what persists. Precision is not the lever, and the requirement stays on retention and on sustained
structure under continual drive. A **licence-only** amendment with no gate was also rejected: an
unenforced sentence in *Consequences* is the failure mode
[#345](https://github.com/NGL321/patchworks/issues/345) names — and it is the failure this very
section is repairing.

**The cost is accepted with eyes open.** Naming float32 the noise floor puts the pressure on **this
ADR's instrument**, not on the architecture: the paired counterfactual is a below-the-floor quantity
by construction. The gate is what carries that pressure rather than hiding it.

## Consequences

- **This predicate is necessary, not sufficient, and that is why ADR-0021 is kept.** It says the loop
  **can** close, not that anything distinguishable travels it. The two are complements and neither is
  the other's replacement: ADR-0021 remains the sufficient per-edge diagnostic — a pass there is
  conclusive, a fail is not — and this is the operative bar. **A session reading a PASS here has not
  read a transmission verdict.**
- **ADR-0021 is demoted from the operative bar**, and its *What this predicate is for* section is
  rewritten to say so by this same ticket. Its predicate is untouched: the bottleneck ratio, `A₀ = 1`,
  `k = 1`, the quiescent-hold floor, per-edge indexing, the median bar and the two directions stated
  separately all stand.
- **Stage 3 opens on a PASS in both directions** — the inbound stratum clause and the outbound
  universal. That is what *sufficient* means in #127's stage-3 row.
- **The map gains a defined negative.** #143's registered and never-checked claim — **nothing
  guarantees learning produces the retention gradient at all** — is this map's stated negative
  outcome: if stage 2 establishes that learning cannot produce it and construction must place it
  instead, that **redirects the map** rather than deadlocking it.
- **The gate and the work are now the same quantity.** Stage 2's retention work moves this bar
  directly, which is the property the amplitude reading lacked.
- **Scale-free, so #202's never-settling floor does not touch it** — it is a ratio of times, not a
  ratio against a different quantity in different units. **Float32 does touch it**, and what stood
  here until [#224](https://github.com/NGL321/patchworks/issues/224) answered:

  > **Scale-free, so it touches neither #202's never-settling floor nor #224's float32 problem.**

  *Superseded by #224.* Scale-freedom of the ratio's **units** says nothing about the representability
  of its **numerator**. `τ̂` is the e-fold decay of a paired counterfactual deviation, and **rounding
  does not attenuate; signal does** — a deviation decaying into the rounding floor stops decaying, so
  float32 biases `τ̂` *upward, without bound*. Under ADR-0021's bar the shortfall was 1e9x and no
  rounding artifact could fake a pass; under this one it is 1.1x to 8.5x against a derived bar of
  exactly `1` with **no headroom**, so precision has moved from making a measurement unreadable to
  being **capable of manufacturing a PASS on the operative bar**. Quoted rather than deleted because
  the dismissal is what a reader would otherwise re-derive, and the mechanism is the reverse of the
  one #224 originally feared. What replaces it is *The reading is gated on the runtime precision
  floor*, above.
- **A PASS is read on the float64 surface, and the gate is reported with it.** The licence and the
  gate are the operative half of #224's ruling: `CONTEXT.md` gains the *arithmetic floor* as its own
  object, `eps_f32` gains a definition site in the architecture register, and
  [#379](https://github.com/NGL321/patchworks/issues/379) carries the gate into the reading that
  publishes `τ̂` — written by [#381](https://github.com/NGL321/patchworks/issues/381).
- **The reading is zero everywhere today**, and the shortfall at the apex is **15.4x** as
  published — amended by [#274](https://github.com/NGL321/patchworks/issues/274) to roughly 1.1x to
  8.5x once the stalk relay is included, with the verdict unchanged; see *The shortfall on this
  bar* above.
- **`CONTEXT.md` gains *conduction ratio* and *rim-core influence*, and a gloss on *collectively* as
  counted, not summed.**
- **`|loop(c)|` is a construction-time quantity that moves with `DomeSpec`.** The ladder above is
  `DEFAULT_SPEC`'s. Any graph change re-derives it, and no session should quote 14 at a dome it has
  not checked.

## Alternatives considered

- **Keep the bottleneck ratio as the operative bar.** Rejected: the gate and the work would remain
  different quantities, so no stage-2 result could move it — the defect that let #232 fire a clean
  FAIL carrying no information about the destination. Rejected as *the bar*, not as a diagnostic; it
  is kept as one.
- **A universal inbound, over every rim cell.** Rejected, and this is **the user's own answer** on
  #242 against that ticket's own recommendation: nothing needs every patch, and a universal inbound
  would make the apex's sparse connectivity a defect rather than the licensed decimation it is.
- **An existential outbound — a max over paths, ADR-0021's form.** Rejected: the apex is the context
  setter, and global context that reaches only some positions is not global context.
- **A multiplier `k > 1` on the conduction ratio.** Rejected on provenance, exactly as ADR-0021
  rejected it. The loop length is already the derived quantity; multiplying it would be a constant
  invented before anything needed one.
- **Inheriting `|loop(apex)| = 14` from #230's seven-hop figure.** Rejected: it was never checked
  against the mask, and a bar taken from a round number is the failure mode #142 struck the ~0.37/hop
  for. Enumerated instead — and it came back 14, which is a verification rather than a coincidence to
  lean on next time.
- **Setting a threshold on the inbound fraction now.** Rejected: no derivation exists and the reading
  is zero in every stratum, so the number would decide nothing. Deferred to the first PASS, which is
  its trigger and its owner.
- **Averaging the inbound fraction across strata.** Rejected: stalk widths of 48, 2 and 1 mean an
  average hides a 256-cell read inside two 3-cell ones, and a context blind to touch would pass.
- **A sustained clamp as the source.** Rejected: it confounds decay with drive. The impulse is erased
  at the source after one tick (#232, by test), which is what makes the decay clean.
- **A second architectural bar: the arriving deviation must be representable in float32.** Rejected
  on the image (#224), and recorded so it is not re-litigated: it would make a numerical parameter
  part of the transmission budget, for nine decades of headroom on a single decaying ripple in an
  unsustained fork. The gate above puts the requirement on the *instrument*, which is where #224 ruled
  the pressure belongs.
- **A licence with no gate — one sentence in *Consequences*.** Rejected: an unenforced sentence in
  *Consequences* is exactly what this amendment repairs, which is #345's failure mode.
- **A separate instrument for *an apex stable enough to conduct it*.** Rejected: `τ̂ ≥ |loop|` at the
  apex already is that claim, and a third reading would be a duplicate with its own drift.
