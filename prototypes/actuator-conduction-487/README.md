# #487: what #228's `c_v` ruling did to conduction at the actuator

[#228](https://github.com/NGL321/patchworks/issues/228) raised `c_v` at the actuator from 2 to 3,
lowering its reconciliation gain from `0.5γ` to `0.333γ`.
[#487](https://github.com/NGL321/patchworks/issues/487) asks what that does to the conduction ratio
there, and requires the answer be read *before and after, on the same rig and the same seeds, at 30k
and 100k*.

```
python prototypes/actuator-conduction-487/read.py --arm clamped  --seed 0 --ticks 100000
python prototypes/actuator-conduction-487/read.py --arm preclamp --seed 0 --ticks 100000
python prototypes/actuator-conduction-487/summarise.py
```

## The question could not be asked of the instrument as it stands

ADR-0026's **outbound** clause is a universal over *L1 predicting cells **and the actuator boundary
cell***, and the ADR argues the inclusion at length: the actuator is *read*, not written, so
influence arriving there is not erased — *it is action, and it is the acceptance demo's own
instrument*.

Nothing in the rig implements that.

| what | where | the actuator |
|---|---|---|
| `Dome.private_projection` | `graph.py` | `[150, 32]` — predicting cells only |
| `world_loops` / `loop_lengths` | `benchmarks/detectability.py` | 150 entries, no 262 |
| `conduction()`'s reduction | `benchmarks/detectability.py` | `cells = dome.predicting` |

So the actuator is absent from `ratio`, and `conducting_path` treats an absent cell as **unbounded** —
it is transited without ever binding a path. `conducting_path`'s docstring states the ground as
*"boundary cells, which hold no private features, carry no `τ̂` and are excluded from the outbound
universal by ADR-0026 on exactly this ground"*. **ADR-0026 excludes sensory boundary cells and the
drive, and explicitly keeps the actuator.** The blanket phrasing is not the ADR's.

That gap is #487's to report and not this rig's to close; see the ticket for what is owed.

## What this rig reads instead, and why it is exact rather than a stand-in

The actuator **runs no body** — `inference_phase` writes only predicting stalks — and it is never
written whole. So message passing is the entire dynamics of its node stalk, and a paired deviation
obeys, with the neighbours' `y_e` exogenous:

```
Δx  ←  A Δx ,      A = I − g · Σ_e F_eᵀ F_e
```

**Only the commanded components retain.** `Agent.write` sets `stalks[_efference_slice] = applied` at
the end of every tick, so the actuator's `joints = 3` efference components are overwritten from
outside — byte-identically in both branches of a paired fork, which drives the deviation there to
zero each tick. The `commanded = actuator_stalk − joints = 3` components are written by nobody. The
recurrence is therefore `Δx ← P A Δx`, and the retained operator is `A[:3, :3]`.

This is **ADR-0026's own logic arriving at the actuator**, not a new convention. The ADR excludes
sensory boundary cells because *the world's write voids what arrives*; at the actuator that write
voids exactly the efference half, and the commanded half is the part at which arriving influence has
a consequence. Reading `ρ` on the whole 6-dimensional stalk would credit the cell with retention in
directions the world resets — `rho_A_full` is reported beside it so the difference the projection
makes is visible rather than asserted.

**`Δx ← P A Δx` is checked against the running graph, not taken from the source.**
`operator_identity` forks a paired branch, nudges the actuator on its commanded block, runs **one**
tick and compares the measured difference against `P A δ`. One tick is the whole check: the unit
delay means `message_passing_phase` reconciles against the partner's slot in *last* tick's broadcast,
so a deviation introduced at the actuator cannot have reached `y_e` yet, and the first step is
exactly `P A δ`. It agrees to float32 machine precision (`~1e-7`) on every arm and checkpoint.

## #274's trap has nothing to bite on here

#487 warns that *whatever reads this must be the corrected instrument, driven, not the direct one* —
[#274](https://github.com/NGL321/patchworks/issues/274) having found the chart→chart round trip omits
the stalk relay. **That correction is a predicting-cell artifact.** The relay term is
`J_s · A_v · D`: it exists because a predicting cell's `encode` reads a stalk that `inference_phase`
wrote from its own chart. The actuator has no chart, no body and no `encode`, so there is no relay
term to omit and no *direct* reading to be wrong — the recurrence is `A`, entire and verified. This
is the shape [#183](https://github.com/NGL321/patchworks/issues/183) found at the drive, where #142's
184x correction likewise had nothing to bite on.

*Driven* is honoured where it is meaningful: the operating point is a driven, taught one — the real
dome, `split=train`, both rules, the environment attached for the full 30k or 100k ticks. The world
is held still only inside the paired counterfactual, which is ADR-0021's quiescent-hold floor and
the condition the numerator must share with the denominator.

## The two arms

`preclamp` restores `patchworks.restriction.overlap_counts` to its pre-#228 body — the pigeonhole
form with no pinned-incidence branch — **before the graph is built**, so `Sheaf.__init__` reads the
old gain. Every reader reaches the count through `gain_denominators`, which resolves
`overlap_counts` in its own module globals at call time, so the one patch covers the gain, the
fold-margin nomination and the projection alike. `arm_check` records which cells the arm actually
moved; on `DEFAULT_SPEC` the answer is exactly one, 262, and a run that moved a second cell shows up
as that rather than as a number.

`30k` is read as a **prefix of the 100k trajectory** rather than as a separate run.
[#178](https://github.com/NGL321/patchworks/issues/178) has cost this map the 30k mistake four times,
#228's own premise being the latest, and two horizons off one trajectory is what makes them
comparable.

## The reading

`world_loop(actuator) = 3`. `τ` is `−1/ln ρ(A[:3, :3])`; the ratio is `τ / 3`.

| seed | ticks | gain pre | `τ` pre | ratio pre | gain post | `τ` post | ratio post | `τ` factor |
|---|---|---|---|---|---|---|---|---|
| 0 | 30k | 0.5000 | 5.44 | 1.814 | 0.3333 | 6.27 | 2.091 | **1.15x** |
| 0 | 100k | 0.5000 | 5.05 | 1.685 | 0.3333 | 5.46 | 1.820 | **1.08x** |
| 1 | 30k | 0.5000 | 5.29 | 1.765 | 0.3333 | 9.11 | 3.036 | **1.72x** |
| 2 | 30k | 0.5000 | 4.84 | 1.615 | 0.3333 | 14.66 | 4.886 | **3.03x** |

**The direction is unanimous and the magnitude is a range, not a number.** The lowered gain
lengthens retention at the actuator in every pair — which is the direction #487 hypothesised — by
**1.08x to 3.03x**. Both arms clear `≥ 1` in every pair, so nothing crosses the bar in either
direction and the ruling costs the outbound clause nothing.

**The clamp buys its lengthening by moving the cell nearer the band's face.** `preclamp` `τ` is
almost seed-independent (5.44 / 5.29 / 4.84 at 30k) while `clamped` `τ` scatters (6.27 / 9.11 /
14.66). `ρ` moves from 0.813–0.833 to 0.853–0.934, and `τ = −1/ln ρ` diverges as `ρ → 1`, so the
same ruling that lengthens the mean also makes the quantity far more seed-sensitive. That is
ADR-0026's amendment on the spread arriving at this cell, and it is why no point value is quotable.

**`trace(M) = 3.0000` in all eight readings** and `λ_max` sits at 0.741–0.750 throughout. The exact
gauge pins `Σ_e ‖F‖_F² = deg(v)` at a fully-pinned cell, so learning cannot change how much map mass
the actuator has, only how it is spread; every difference above lives in the low end of the spectrum.

**No systematic map compensation.** `λ_min` is *lower* in the clamped arm on seed 0 (0.176 vs 0.222)
and *higher* on seeds 1 and 2 (0.241 vs 0.156; 0.076 vs 0.049). The arms' surfaces differ per seed
rather than in a direction, so the factor's spread is not a mechanism this reading can name.

**The three L1 neighbours show no detectable effect**, under ADR-0026's own `τ̂` on their private
block: 1.50–7.00 ticks with no consistent ordering between arms across four pairs. Reported as a
null at this sample size rather than as an absence.

**`tau_stalk` tracks nothing**, as expected — 2.00–9.00 with no relation to `tau_closed`. It is the
whole-stalk read, which mixes what arrives from the neighbours with what the cell retains, and its
uselessness here is the concrete form of ADR-0026's refusal of that reading site.

## Files

- `read.py` — the rig. Writes one JSON per `(arm, seed, ticks)`, **as each checkpoint lands**, so a
  killed 100k run still leaves its 30k reading behind.
- `summarise.py` — the roll-up table. Runs against whatever JSONs are present.
- `487-<arm>-seed<n>-<ticks>.json` — the readings.

Like every script here **it asserts nothing** and its exit code does not move.
