# B69 (#646) — the flat bundle at horizon: the rule keeps its **construction** null, and says so

**Question.** What is the flat bundle at horizon, and is [B48 (#615)](https://github.com/NGL321/patchworks/issues/615)'s
null still the object the joint scoring rule names?

**Answer.** **No — and not because the null moved, but because there is no trained flat bundle
with the properties the null is supposed to have.** Both halves of B48's null turn out to be
construction-only, and they fail in *different ways*:

- **`earned` does not move with differentiation. It falls off a cliff.** It reads 44 at
  construction and **exactly 0 from tick 50 onward**, on both seeds, unchanged through 9,000
  while differentiation triples. The trained flat bundle is **bit-identical to B48's own
  *trained control* row** on the agreement statistic — `dim H⁰` lands on exactly the private
  reserve, 1808. So the null cannot be re-indexed to any horizon ≥ 50 ticks: at every one of
  them it earns nothing, which is precisely the thing a null for *agreement* must not do.
- **Differentiation climbs and has no measured settle**, even at **60,000 ticks** — double the
  deepest horizon previously run on this map. It reaches **0.2292** and is still rising.

So the joint rule names the **construction** point, explicitly labelled as construction. That is
[#646](https://github.com/NGL321/patchworks/issues/646) item 4's decision, and §4 states it with
its cost.

**And the ticket's own premise is misquoted, in a way that matters.** #646 states the null as
*"the flat bundle's 88 earned dimensions"*. B48's table has three rows, and **88 is B42's
*reserved frame*** — the candidate-side object. **The flat bundle's row is 45.** This ticket reads
**44** on the arm this map runs as the flat bundle, reproducing B48's flat-bundle row and nowhere
near 88. §3 has the correction; the error originates in B48's own §3 prose and #646 inherited it.

**This stands up almost no new rig.** `b69_deep.py` is `b61_horizon.run_one` with three rungs added
to the ladder; `b69_earned.py` is the same runner with `b56_channel.exposure` wrapped so B22's
`whole_graph_split` — which *is* B48's instrument, `b42_earned.py` being three calls to it — rides
along at every rung. Arms, terms, initialisation, seeds and nulls are B61's byte for byte, so every
rung at or below 30,000 is a regression test against [#634](https://github.com/NGL321/patchworks/issues/634)'s
published table, and rungs at or below 6,000 reproduce it to four decimals.

---

## §1 — Where it goes: `s0_baseline` to 60,000, seed 42

Rungs past this run's own stall stamp marked `*` — and see §5, because the honest stamp is **100**
and the `*` marks are drawn from B56's rule, which §6 shows is wrong here.

| ticks | `chan(w)` | `diff` | `k_v` rank | `k_v` part | `rho(used)` | `tau` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0000 | 0.0000 | 12.0 | 11.2 | 1.0000 | inf |
| 2000 | 0.9996 | 0.0803 | 20.0 | 6.9 | 0.9524 | 20.50 |
| 9000 | 0.9984 | 0.1210 | 20.0 | 6.7 | 0.9493 | 19.23 |
| 18000 | 0.9968 | 0.1622 | 20.0 | 6.6 | 0.9475 | 18.54 |
| 30000 | 0.9957 | 0.1931 | 20.0 | 6.9 | 0.9496 | 19.33 |
| 40000 | 0.9937 | 0.2080 | 20.0 | 7.1 | 0.9458 | 17.95 |
| 50000 | 0.9924 | 0.2231 | 20.0 | 7.4 | 0.9435 | 17.20 |
| **60000** | **0.9887** | **0.2292** | 20.0 | 7.6 | 0.9422 | 16.80 |

**Clause 1 never gives.** `channel_return` reads **0.9887** at 60,000. The flat bundle satisfies
B48's clause 1 essentially perfectly for the whole run *while* acquiring 0.23 of differentiation —
which is the standing reason clause 1 alone cannot be a bar.

**Exposure does not pay for the climb.** `k_v` participation sits at **6.5–7.6 across the entire
run** with no trend, and `k_v` rank is pinned at 20.0 throughout. So the differentiation the trained
flat bundle acquires is *not* bought with exposure — the two halves of the joint reading move
independently here, which is itself a caution against quoting them as one paired object.

## §2 — What it is climbing to, and the pre-registered falsifier

The model-free discriminator, per adjacent rung pair — no fit:

| rung pair | 2k→4k | 6k→9k | 13k→18k | 24k→30k | 30k→40k | 40k→50k | **50k→60k** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `diff` per 1k | +0.00743 | +0.00429 | +0.00424 | +0.00342 | +0.00149 | +0.00151 | **+0.00061** |

The rate falls by a factor of **27** from the 1k→2k pair (+0.0164) to the last, and the **final pair
sits below the 0.001-per-1k noise floor** that `b61_analyse.shapes` uses to call a level stopped.
Three-point ratio on the last three rungs: **0.401**, so the rate is falling, and the implied
asymptote is **0.2333** — well below the staggered frame's settled **0.3912**.

**The falsifier, evaluated rather than described.** #646 pre-registered: *the null's climb settles
below the staggered frame's 0.391, on both seeds, and the gap therefore has a positive floor.*

| | seed 42 | seed 43 |
| --- | ---: | ---: |
| horizon reached | 60,000 | 30,000 |
| level | 0.2292 | 0.1914 |
| gap to 0.3912 | **0.1620** | **0.1998** |
| last rate per 1k | +0.00061 | +0.00192 |
| settles, by the criterion applied to the candidate | **no** | **no** |
| reaches 0.3912 | no | no |

**Neither end fires.** The settle test is deliberately the *same* two-adjacent-pairs-under-the-floor
test that declared the staggered arm stopped — a null held to a looser standard than the candidate
would settle by definition — and on seed 42 the last pair passes it while the pair before
(+0.00151) does not. Seed 43 only reached 30,000.

**So the honest split is:** a crossing is **not** in evidence and is not plausibly near — linear
extrapolation at the *current* rate puts one at tick **~327,000**, five times the deepest horizon
run, and the rate is still falling, so that is an upper bound on speed rather than a prediction.
The gap almost certainly has a positive floor around **0.16**. But that floor is reached by
**extrapolation, not by measurement**, and this ticket does not claim otherwise.

**The gap's narrowing is still entirely the null's doing** — the candidate stopped at 0.3912 by
18,000 ([#634](https://github.com/NGL321/patchworks/issues/634) §1) and has not moved since, so
every 0.001 of closure from 18,000 to 60,000 is the flat bundle climbing.
[B50 (#618)](https://github.com/NGL321/patchworks/issues/618)'s *the null is not static* is stronger
still at 60,000 than at the 9,000 that restated it.

**And the 19% seed spread #634 flagged is a transient.** It reproduces at 9,000 (16.5% here) but the
seeds *converge* as the horizon deepens — **1.8%** at 30,000, 0.1879 against 0.1914:

| ticks | 4000 | 9000 | 13000 | 18000 | 24000 | **30000** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed 42 | 0.0952 | 0.1218 | 0.1393 | 0.1574 | 0.1688 | **0.1879** |
| seed 43 | 0.1204 | 0.1458 | 0.1675 | 0.1784 | 0.1799 | **0.1914** |
| spread | 21.0% | 16.5% | 16.8% | 11.8% | 6.2% | **1.8%** |

So the flat bundle's *level* is reproducible after all; #634 read it where the seeds had not yet
converged. (The seed-42 column here is the dedicated 30,000 run; the 60,000 run reads 0.1931 at the
same rung, which is the ~5% fixed-seed jitter #634 documented and this ticket reproduces.)

## §3 — Item 3: `earned` does not move with differentiation, and the null's headline number is misattributed

B22's `whole_graph_split` at every rung, on the same trajectory:

| ticks | `diff` | `dim H⁰` | trivial | **earned** | generic | above generic | `k_v` median |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **0** | 0.0000 | 2812 | 2768 | **44** | 0 | 44 | 12.0 |
| 50 | 0.0289 | 1808 | 1808 | **0** | 0 | 0 | 20.0 |
| 500 | 0.0495 | 1808 | 1808 | **0** | 0 | 0 | 20.0 |
| 2000 | 0.0769 | 1808 | 1808 | **0** | 0 | 0 | 20.0 |
| 9000 | 0.1206 | 1808 | 1808 | **0** | 0 | 0 | 20.0 |

**The mechanism is visible in one column.** At construction the stagger-0 frame leaves each cell
reading only `k_v = 12` of its permitted 20 directions, and the 44 earned dimensions are what that
deficiency leaves unconstrained. Within 50 ticks the maps fill the permitted window — `k_v` 12 → 20,
`rank δ` 1988 → 2992 — and `dim H⁰` lands on **exactly** the private reserve, 1808. `earned` is 0,
`generic` is 0, and `earned_above_generic` is 0: the trained flat bundle is **exactly as coordinated
as a surface in general position**, which is word for word what B48 concluded about the *trained
control*.

**Seed 43 confirms it, and the construction reading is seed-free:** both seeds read 2812 / 2768 /
**44** at tick 0 — identical, because the flat bundle's construction is fixed by the dome and the
mask rather than by the random frame — and both collapse to 0 at tick 50.

**The correction.** B48's table:

| surface | `dim H⁰` | trivial | earned | above generic | exposure per cell |
|---|---|---|---|---|---|
| trained control | 1800 | 1800 | **0** | 0 | 20.0 / 32 |
| **B40's flat bundle** | 2768 | 2723 | **45** | 45 | 13.9 / 32 |
| **B42's reserved frame** | 2668 | 2580 | **88** | 88 | 14.8 / 32 |

**88 is the reserved frame — the candidate. The flat bundle is 45.** This ticket reads **44** on
`s0_baseline`, reproducing the flat-bundle row. The *"5.2 dimensions of exposure per cell"* is
likewise a reserved-frame quantity (20.0 − 14.8 against the trained control), not a flat-bundle one.
B48's §3 prose says *"B42's flat bundle earns 88 … bought with 5.2 dimensions of exposure"*, which
merges the two rows; #646's framing inherited it. **A joint rule whose null is quoted with its
candidate's numbers has a defect prior to any question about horizon**, and it is worth saying that
this is the more consequential of the two findings about B48's rule.

## §4 — Item 4: what the rule states instead. **The construction point, named as such.**

**Decision: the joint scoring rule keeps a construction null and labels it `construction`.** Not a
stated horizon, and not a trajectory. Four reasons, in order of force:

1. **There is no trained level of `earned` to name.** It is 0 at every rung from 50 ticks on. A null
   at `earned = 0` is not a null for an agreement statistic — it is the generic surface, which
   `earned_generic` already controls for, and scoring against it would make every candidate with a
   single earned dimension a pass.
2. **There is no settled trained level of differentiation to name either.** Not at 30,000, and not
   at 60,000. Any number the rule quoted would be a horizon quote, and #634's §5 named quoting
   across horizons as the exact move to stop.
3. **[B42 (#605)](https://github.com/NGL321/patchworks/issues/605)'s derivation is exact only at
   construction.** Collapse is the *only* point at which exact path-independence exists. At
   differentiation 0.2292 the object is not the one the derivation is about, so a trained "flat
   bundle" would be a null with no derivation behind it.
4. **The construction reading is seed-free and horizon-free** (§3: both seeds read 44), which is
   what a null has to be to be quoted in a rule at all.

**What this does *not* do is repeal B50.** *The null is not static* stands, and is stronger than
ever. It is honoured not by moving the null but by a second, separate requirement:

> **Any *gap* claim is read at a shared rung against the trained flat bundle run to that same
> rung.** The construction point defines the null; it does not license comparing a candidate at
> 30,000 against a null at 0.

That is what #634 §3 already did and what §2 above extends, so this is a codification rather than a
new obligation.

**Named ADR cost: none.** This decides how B48's rule *quotes* its null; it changes no ADR, no
construction, and no term. The costs are documentary and land on the tickets that carry the rule:
B48's table needs its flat-bundle/reserved-frame rows disentangled in any text that quotes them, and
the rule's statement of the null needs the word `construction` in it.

**What it is not.** Per [the standing rule](https://github.com/NGL321/patchworks/issues/532), a map
ruling is not a landed change: this is a decision about the scoring rule's wording and nothing on
`main` moves because of it.

## §5 — What a 60,000-tick reading on a body that stopped at 100 is a reading *of*

**The ticket's named risk, and it does not resolve into a clean pass — but it points the same way.**

These arms are the **shipped transport rule**, which reads stalks, so B56's stall exemption does not
apply and this ticket does not claim it. The body's motion falls under threshold after **150** ticks
on seed 42 and **250** on seed 43, so the honest stamp is **100** / **150** and **59,900 of 60,000
ticks are drift under a frozen stimulus.** Stamped per run, inheriting nothing
([B38 (#599)](https://github.com/NGL321/patchworks/issues/599)).

- **What the horizon buys.** The *shape* question — is the climb decelerating — is a question about
  the transport maps' trajectory under a fixed input distribution, and the rungs answer it on their
  own terms. And item 3's answer needs no horizon at all: the `earned` cliff is at **tick 50**, well
  inside the live window on both seeds, so **that finding is not drift** and is the one the decision
  chiefly rests on.
- **What it does not buy.** Nothing about a flat bundle in a live agent. Whether a live body would
  hold, raise or destroy the 0.23 is untouched.

**This strengthens §4 rather than weakening it.** The ticket anticipated the outcome *"what 'the
null' means at horizon is undefined rather than moved, and B48's rule should keep its construction
null and say so explicitly"* — and that is where the evidence lands, by two independent routes: the
trained object is measured only under a frozen stimulus, *and* it earns nothing even there.

## §6 — Instrument notes

**#634 §6's stamping bug replicates on every new run.** `b56_analyse.horizon` takes the **last** rung
clearing `MOVING`, and a late blip drags the stamp to it:

| run | falls under after | late re-crossing | B56's stamp | honest stamp |
| --- | ---: | --- | ---: | ---: |
| `s0` seed 42 → 60,000 | 150 | 40,000 (3.88e-02) | 40,000 | **100** |
| `s0` seed 42 → 30,000 | 150 | 30,000 (8.26e-02) | 30,000 | **100** |
| `s0` seed 43 → 30,000 | 250 | 9,000 (5.39e-02) | 9,000 | **150** |

Three of three. `b61_analyse.stall_diagnostic` reports both rules and substitutes neither; every `*`
in §1 is drawn from B56's rule and is therefore *understated*, not overstated.

**Fixed-seed jitter reproduces at ~5%** (0.1879 / 0.1931 at 30,000 across two seed-42 runs), which is
the precision of every single-run statement here, and is why §2's settle test is applied to rung
*pairs* rather than levels.

**The box.** 62.7 GB committed of a 66.8 GB limit (94%) at launch — the saturation that killed five
of #634's arms. Every run was launched **strictly one at a time**, and none was lost. One run was
killed deliberately and is worth recording: `b61_horizon.py --ticks 60000` **silently stops at
30,000**, because `b56_channel.run_arm` iterates `[c for c in CHECKPOINTS if c <= ticks]` and B61's
ladder ends at 30,000 — it would have written a file named `-60000` holding a 30,000-tick run. That
is why `b69_deep.py` extends the ladder rather than passing a bigger `--ticks`, and it is a trap for
any later ticket taking a deeper reading on this rig.

## §7 — What is owed

- **Seed 43 past 30,000.** The 60,000 rungs — and so the asymptote and the falsifier's near-miss —
  rest on **seed 42 alone**. Seed 43 replicates the trajectory and the level to 30,000 (1.8%) but
  stops short of the rungs where the rate reaches the floor.
- **A settle established rather than extrapolated.** One further doubling to ~120,000 would put two
  adjacent pairs under the noise floor if the ratio holds. Nothing in §4 depends on it — the
  decision rests on the `earned` cliff and on B42's derivation, neither of which is a horizon
  question — so this is worth doing only if some later ticket needs the gap's floor as a *number*.
- **`earned` past 9,000.** It is an exact rank identity pinned at 0 with `dim H⁰` sitting exactly on
  the private reserve, so it is not expected to move; unrun rather than uncertain.
