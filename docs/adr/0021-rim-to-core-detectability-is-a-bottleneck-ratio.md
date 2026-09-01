# ADR-0021: Rim-to-core detectability is a bottleneck ratio, not a magnitude

**Status:** accepted

## Context

Settled in [#181](https://github.com/NGL321/patchworks/issues/181). **Re-scoped by
[#230](https://github.com/NGL321/patchworks/issues/230) and carried into
[#127](https://github.com/NGL321/patchworks/issues/127)'s *Done when* by
[#236](https://github.com/NGL321/patchworks/issues/236): read *What this predicate is for*, at the
end of this section, before reading a verdict off anything below it.**

This ADR was derived from a destination the record has since retracted. The sentence it opened with:

> The destination this architecture is being built toward is *a perturbation at the rim measurably
> reaches the apex and returns*.

*Superseded by #230.* The destination is **mutual influence** — rim cells influencing apex cells
**collectively**, many-to-few, and apex cells influencing all rim cells **individually**, few-to-many,
through an apex stable enough to conduct it. **Nothing has to arrive.** #230 ruled that the stable
structures are not destinations but **channels** that direct the ripples — the object
[ADR-0022](./0022-a-hop-is-an-operator-norm-along-a-learned-channel.md) already carries under the name
*learned channel* — and #236 carried the retraction into #127's *Done when*, where *reaches and
returns* is no longer the claim.

The struck sentence is quoted rather than deleted because everything below is a derivation *from* it:
the defect of units it exposes is real, the fix is correct, and a reader who cannot see the premise
cannot check the work. What #230 struck is the **destination**, not the **derivation**.

What was true when this was written and is still true: nothing said what **measurably reaches** means,
and the two quantities the record kept comparing do not live in the same units.

[#142](https://github.com/NGL321/patchworks/issues/142) pinned the per-hop target to
"[ADR-0007](./0007-the-disagreement-floor-is-tolerated-not-represented.md)'s existing disagreement
floor, read per level" and superseded the inherited **~0.37 per hop** as back-computed from an
end-to-end figure with no provenance. Deriving the replacement was made a precondition of the
transmission edit. [#158](https://github.com/NGL321/patchworks/issues/158) was that precondition and
**did not deliver it, by its own account**: the quantity
[`02-tick-semantics.md`](../spec/02-tick-semantics.md)'s bound calls `floor` is not ADR-0007's
disagreement floor at all — it is the standing offset, which is now
[ADR-0019](./0019-construction-nominates-the-run-decides.md)'s subject and carries its own name.

What was left is not a number waiting to be measured. It is a defect of units:

- **A hop is a dimensionless gain** — a ratio, `output / input`.
- **A disagreement floor is a magnitude** — a per-edge disagreement in stalk units that learning
  cannot remove.

`hop ≥ floor` is not a well-formed comparison, and no length of run makes it one. Three decisions
stood between the two, and each is a decision about what the architecture claims rather than a
quantity to read off: **what size the rim perturbation is taken to be**, **which of ADR-0007's three
floors is meant**, and **whether the comparison is per-hop or cumulative**.

### What this predicate is for

**A per-edge diagnostic — sufficient but not necessary — and the operative bar until something
falsifiable replaces it.** #230's re-scope in its own terms:

> ADR-0021 is re-scoped, not replaced. It stays an excellent per-edge diagnostic that correctly
> located where the channel dies, and stops being the destination's *done when* — a sufficient but not
> necessary condition. Deviation above the floor everywhere certainly means transmission; below it
> does not certainly mean none, if the core's retained state moved.

Three consequences, and the first is the one a session reading a result here is most likely to get
wrong:

- **A pass is conclusive; a fail is not.** Clearing the bar on every edge certainly means
  transmission. Failing it does not certainly mean none, because the core's **retained** state may
  have moved without a deviation arriving, and nothing in this predicate reads retained state. Every
  instrument in its family reads arriving magnitude.
- **It is nonetheless still the bar.** #127's *Done when* keeps it in terms — *the operative bar is
  unchanged until something falsifiable replaces it* — because a bad predicate that fires beats a good
  one that cannot be read, and this project's history says soft conditions rot. **Nothing here is
  softened before that replacement exists**, and this ADR is not the place to try.
- **[#242](https://github.com/NGL321/patchworks/issues/242) owes the replacement.** It writes the
  falsifiable form of the influence reading, or rules that this ratio stays the bar and says what a
  stage-3 gate then consists of. Whoever answers #242 rewrites this section; until then it stands.

**The predicate has been read twice, and fired both times.** Recorded as a fact about its use, not as
a re-derivation of it, and neither read is re-opened here:

- [#214](https://github.com/NGL321/patchworks/issues/214), the first direct read: rim→apex arriving at
  `8.7e-10`, short by **1.15e9x**; apex→rim short by **7.6e7x**.
- [#232](https://github.com/NGL321/patchworks/issues/232), all four corners of the sustained ×
  collective stimulus space: the best corner `2.3e-9`, short by **1.3e9x**, with only **7.2x** of
  total spread across the whole space.

So *The pre-registered expectation* below is **discharged, not pending**: it predicted that
falsification condition 1 would fire, and it fired. It is left standing as written, in the
anticipatory voice it was pre-registered in, because striking it would destroy the thing it was for.

## Decision

### The predicate

> `max` over rim-to-apex paths `P`, `min` over edges `e ∈ P`, of
> `[ A₀ · Π_{i ≤ e} hop_i ] / floor_e  ≥  1`

with `A₀ = 1` and `k = 1`, the floor being the **quiescent-hold floor**, evaluated **per trial** with
each trial reduced to the **peak** paired ratio at that edge, reported as a **distribution over
trials** with the bar at the **median**, and stated **twice** — rim→apex and apex→rim as separate
predicates.

A max-min over paths, the standard widest-path form. It says what the architecture actually claims:
*there exists a channel that carries the perturbation*, and the thing that fails is an **edge**, not a
level.

The arriving quantity taken against the floor is the **bottleneck ratio**; the predicate over it is
**rim-to-core detectability**. Both terms are in [`CONTEXT.md`](../../CONTEXT.md), and the bare word
`floor` is deliberately not reused for either — see *The naming constraint is load-bearing* below.

### `A₀ = 1`: a unit-norm deviation on the L1 boundary stalk

Nothing in the spec fixed an amplitude convention. [`03-the-sandbox.md`](../spec/03-the-sandbox.md)
and [`08-the-acceptance-demo.md`](../spec/08-the-acceptance-demo.md) take
`disturb_arm(joint, impulse)` with the impulse a **free parameter**, and the demo reads **onset
latency** — a behavioural quantity, never an arriving magnitude. So this convention was **created
here rather than found**, and it is stated as a convention.

Unit-norm rather than the demo's actual impulse, because **the transmission claim is a property of
the graph and not of how hard someone shoves the arm**. Welding the referent to a demo parameter that
`08` deliberately left free would re-open the sufficiency verdict every time the impulse changed.

The consequence is the whole of the unit fix: with `A₀ = 1` the arriving quantity is in stalk units,
the same units as the floor, and the two are finally comparable.

*Whether the demo's actual impulse is large enough* is a real question and a different one. It is
recorded in [`08-the-acceptance-demo.md`](../spec/08-the-acceptance-demo.md), where the parameter
lives.

### The quiescent-hold floor: static + settling, lag excluded

ADR-0007 names **three** floors — static, lag, settling — and #142 wrote "the disagreement floor" as
though it were one number. The referent here is the pair present at rest: **static + settling**, which
is exactly what ADR-0007's own **quiescent hold** isolates. Three reasons, and the third is the
decisive one:

- The hold is ADR-0007's instrument already, so the predicate **inherits a measurement protocol**
  rather than needing one invented.
- Static and settling are the genuinely irreducible part; the lag floor is the driven part, and drains.
- Including lag would demand that the perturbation beat a quantity **it partly causes**. The
  perturbation is motion, and the lag floor is a function of motion.

### The index is the edge; the gain is cumulative

#142's text read both ways in different paragraphs — a per-hop target `t` and an end-to-end target
`t⁷` are different specifications. **Neither reading was right, because the choice was never between
them: it was about where the index sits, and it sits on the edge.** The target is per-edge; the gain
accumulated along the path up to that edge is cumulative. There is no per-hop number at all.

The "~0.37 per hop" was an artifact of assuming a single floor and a uniform hop, and both are known
false: #158's floors are non-monotonic across levels, and after
[#190](https://github.com/NGL321/patchworks/issues/190) the gain is graded 2.50x at the apex to 12.0x
at the actuator.

**Per-edge, not per-level, and the reason generalises past this decision:**

> **Per-edge is a property of the graph. Per-level is a property of the *shape* of the graph.**

The dome's level structure is an imposed prior — a prediction that connectivity should ultimately
take that form, adopted because training from a dense graph toward sparsity is more than this project
can account for ([`06-graph-topology.md`](../spec/06-graph-topology.md)). Indexing a target by level
would elevate that prior to a relevance it has not earned. The architecture is measured on the graph,
never on the shape imposed on it.

### A ratio, because the floor is not a wall

`arriving_amplitude ≥ floor` reads "above the floor" as a wall to clear. The floor is not a wall — it
is **what is already standing on that edge** — so *arrives above the floor* means the perturbation is
**distinguishable from** it. That makes the quantity a ratio, and the ratio is the only form that
survives the evidence about the floor: it wanders 3.8x with no trend
([#178](https://github.com/NGL321/patchworks/issues/178)), it moved 144x during a run, and per
[#202](https://github.com/NGL321/patchworks/issues/202) it **never settles at all**.

A magnitude target is a target that moves. A ratio is scale-free in exactly the quantity that is
moving.

### `k = 1`, and no safety margin is invented

The bar is 1, not 3, and not any other multiplier. Tick-pairing already does the work a margin would
do — the ratio is formed against the floor **as it stands at that tick**, excursions included — so a
multiplier would double-count the same conservatism.

The decisive argument is provenance: **every invented constant in this map's history has later been
found to have none.** #142 superseded the ~0.37/hop precisely because it was back-computed with no
derivation, and a `k = 3` chosen today would be that number's successor by another route.

### Two predicates, not one fourteen-hop chain

Rim→apex and apex→rim are stated and evaluated **separately**. The two directions do not share a
gain: #190 priced the denominator swap at 2.50x at the apex against 12.0x at the actuator, so the path
is **asymmetric by construction**, and a single chain would hide which half failed.

Forward-only was rejected because it cannot express the claim being made. That claim was *reaches the
apex **and returns*** when this was written and is **mutual influence** after #230, and it is
two-directional under both readings — many-to-few inbound, few-to-many outbound. The retraction moved
this argument's warrant and left its conclusion standing. Forward-only would also leave
[#182](https://github.com/NGL321/patchworks/issues/182)'s return hop, the one the acceptance demo's
instrument actually reads, covered by no predicate at all.

### The instrument

- **The deviation at an edge is a paired counterfactual** — a perturbed run minus an unperturbed run
  forked from a common state, compared at the same tick. Not a raw excursion, which cannot separate
  the perturbation from ordinary dynamics; that is the same aggregation error that cost this effort a
  1e14 phantom deficit at #142. The sandbox supports the fork with no special mode: `reset()`
  rearranges the world and never the agent, the clock is monotonic, and there are no episodes.
- **The distribution is over repeated perturbation trials**, each reduced to the **peak** paired ratio
  at that edge. `disturb_arm` is an impulse, so the deviation is a transient — it rises, peaks,
  decays. A distribution over *ticks* would mix ticks where the perturbation has arrived with ticks
  where it has not, and report mostly the latter. Trials also discharge an obligation ADR-0007 already
  imposes and nothing had honoured: a static floor is **positional**, so one pose reports on one point
  of the overlap and the sweep across configurations is required anyway.
- **The bar is the median trial**, with p05 / p25 / p75 / p95 reported alongside. #202 is the
  precedent and a hard one: `02`'s "the bound holds after a burn-in" demanded a universal condition,
  and the run found it false as written rather than informative. A universal bar on a non-stationary
  quantity writes a condition nothing meets.
- **No settling claim is made, because there is nothing to settle to.** #202 found every one of
  100,000 ticks carried a breaching cell and no clean tick in the final 50,000; the burn-in does not
  exist. Forming the ratio **per trial** needs no burn-in, which is why this predicate acquires no
  dependency on one.

### ADR-0007 gains no new member, and this does not give it one

#142 was firm on this and the constraint is inherited rather than reopened: ADR-0007's content is
that the floor is **tolerated, not represented**, and a named "transmission floor" would make it
represented.

**This predicate adds no floor.** It makes the existing floor a **referent for a diagnostic** — which
ADR-0007 already does itself when it names the quiescent hold as its own instrument. No per-edge state
estimates anything, no channel carries a floor, no cell is told one exists, and nothing is subtracted
from a residual. A future reader will reach for exactly this objection, which is why the answer is
written here rather than left to be re-derived.

### The naming constraint is load-bearing

**The bare word `floor` is not reused for the arriving quantity.** That exact collision —
`02-tick-semantics.md`'s `floor` and ADR-0007's floor being different objects wearing one word — cost
#158 its precondition and a whole ticket to discover. The arriving quantity taken against the floor
is the **bottleneck ratio**; the predicate over it is **rim-to-core detectability**. Both are defined
in `CONTEXT.md`.

## Consequences

- **The transmission edit's falsification condition 1 is restated.** *"The chained aligned hop fails
  to reach the ADR-0007 floor at any level of the taper"* becomes **at any edge on the channel**.
- **The hop of 0.212 is void, not stale.** [#159](https://github.com/NGL321/patchworks/issues/159)'s
  arithmetic was `0.0782 × 3.79 (fold-margin permitted) × 0.7138 (measured body)`. The 3.79x is
  struck twice over: #182 found it had no mechanism — `Σ_e m_e` binds at 142 of 150 cells — and
  [#160](https://github.com/NGL321/patchworks/issues/160) with
  [#195](https://github.com/NGL321/patchworks/issues/195) demoted the fold-margin check from gate to
  diagnostic, so it **permits nothing**. 0.212 lands inside the post-#190 range by coincidence.
- **The hop is level-graded, not a single number.** On a base of `0.0782 × 0.7138 = 0.0558` and
  #190's denominator: **0.140** at the apex, **0.340** at the rim, **0.670** at the actuator.
- **`08-the-acceptance-demo.md` inherits an open question** — whether the demo's `disturb_arm` impulse
  is large enough — which `A₀ = 1` deliberately split off rather than answered.
- **`CONTEXT.md` gains two terms**, and `02-tick-semantics.md` gains a pointer here. `02` owns the
  reconciliation step, not transmission, so it does not carry the predicate.
- **The measurement is owed.** [#214](https://github.com/NGL321/patchworks/issues/214) runs the
  rim-to-core detectability read: paired counterfactual, per edge, over trials, in both directions.

### The pre-registered expectation

Stated **before** the measurement, so that no result is narrated after the fact. **This is indicative
and pre-registered, not a measured verdict**, and it is marked so deliberately: it reuses #158's 30k
per-level floors with only the apex corrected by #178, interpolates across #190's two endpoints, and
has no paired counterfactual, no trials, and no per-edge read.

Against those floors — 0.055 / 0.058 / 0.076 / 0.146 / 0.111 / 0.057 / 0.217 down levels 1–7, apex
settling 0.087 — every hop is below 1, so the cumulative product decays monotonically. A unit rim
deviation clears the first two edges comfortably (5.9x, then 1.4x) and **goes under the floor at the
fourth**, where a cumulative 0.027 meets a floor of 0.146. It arrives at the apex at ~3e-5.

| reading | apex arrival | short by |
|---|---|---|
| every hop at the rim's 6.10x (generous, unphysical) | 5.3e-4 | **164x** |
| #190's endpoints interpolated across the levels | ~3.0e-5 | **~2,900x** |
| every hop at the apex's 2.50x | 1.0e-6 | **~84,000x** |

So the expectation is that **falsification condition 1 fires**. The useful form of the number: at
~0.2 per hop, **each 5x of rim amplitude buys one more hop** — a unit deviation reaches level 3, and
reaching the apex needs `A₀ ≈ 2,900`.

Adopting this as *the* verdict would repeat this effort's characteristic failure a fifth time —
#142's isotropic probe, #158's wrong `floor`, #178's 30k local high, #182's binder, all numbers taken
for something they were not. It goes in as **the expectation the measurement can falsify**, which is
worth more than an unmarked verdict.

## Alternatives considered

- **A magnitude comparison** — `arriving_amplitude ≥ floor`, the reading #142's text invites.
  Rejected: it treats the floor as a wall when the floor is what is already standing on the edge, and
  it makes the target move with a quantity that wanders 3.8x with no trend and never settles (#178,
  #202).
- **A per-level target.** Rejected on the grounds recorded above: the level structure is an imposed
  prior, and indexing the target by it would elevate that prior to a relevance it has not earned. The
  post-#190 gain profile is the inverse of the one the record assumed anyway — largest at the rim,
  smallest at the apex — so a per-level bar would have been written against a shape that no longer
  holds.
- **A safety margin, `k > 1`.** Rejected on provenance and on double-counting; see `k = 1` above.
- **Welding `A₀` to the demo's `disturb_arm` impulse.** Rejected: it makes a claim about the graph
  depend on a free demo parameter, and would re-open the sufficiency verdict every time that parameter
  moved. The question the convention displaces is kept, in `08`.
- **One fourteen-hop chain, or a forward-only predicate.** Rejected: the two directions do not share a
  gain, a single chain hides which half failed, and forward-only cannot express a two-directional
  claim — *and returns* when this was written, **mutual influence** after #230.
- **A distribution over ticks rather than over trials.** Rejected: the deviation is a transient, so a
  per-tick distribution reports mostly the ticks at which the perturbation has not arrived.
- **A universal bar — every trial clears 1 — behind a burn-in.** Rejected: #202 measured that the
  burn-in does not exist, and a universal condition on a non-stationary quantity is a condition
  nothing meets. The median with reported quantiles says the same thing without pretending otherwise.
