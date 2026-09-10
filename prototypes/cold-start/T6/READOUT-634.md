# B61 (#634) — the staggered frame's differentiation **settles**, and the null is what closes the gap

**Question.** Does the staggered frame's audience differentiation settle above the flat bundle, or
is 2,000 ticks simply too short to see it arrive there?

**Answer.** It settles. On the arm that reaches the horizon, differentiation stops moving entirely
between rungs 13,000 and 30,000 — **0.4071 → 0.3921 → 0.3921 → 0.3912** — while the decay rate falls
by **two orders of magnitude**, from −0.025 per 1,000 ticks at [B58 (#630)](https://github.com/NGL321/patchworks/issues/630)'s
last measured pair to −0.0001. #630's naive extrapolation — the flat bundle's neighbourhood inside
~15,000 ticks — is **falsified by measurement**, not by argument.

**But the gap is a different question from the level, and it has a different answer.** The flat
bundle is not static and over a long horizon it is the *moving* half: `s0_baseline` **climbs**
0.0000 → 0.0803 (2,000) → **0.1269** (9,000) and is still climbing when the candidate has stopped.
So the gap keeps narrowing — 0.4558 at 2,000, **0.3044** at 9,000 — driven entirely by the null's
rise. [B50 (#618)](https://github.com/NGL321/patchworks/issues/618)'s *the null is not static* is far
stronger at 30,000 than the 2,000-tick reading that coined it, where #630 could reasonably call `s0`
"near-stationary".

**This stands up no new rig.** `b61_horizon.py` wraps `b58_train.py`'s arms byte-identically and
patches exactly one thing: B56's checkpoint ladder, capped at 2,000. Every rung at or below 2,000
reproduces #630's published table and acts as its own regression test — tick 0 reads differentiation
**0.7321** and tick 100 reads **0.9995 / 0.6310 / 14.9**, #630's numbers exactly.

---

## §1 — Where differentiation goes: `s19_baseline`, seed 42, to 30,000

Rungs past this run's own stall stamp marked `*`. **This run stamps at 100**, as #630's did.

| ticks | `chan` | `diff` | `k_v` rank | `k_v` part | `rho(used)` | `tau` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0000 | 0.7321 | 17.0 | 14.3 | 1.0000 | inf |
| 100 | 0.9995 | 0.6310 | 20.0 | 14.9 | 0.9587 | 23.71 |
| 2000\* | 0.9986 | 0.5361 | 20.0 | 13.6 | 0.9591 | 23.96 |
| 4000\* | 0.9969 | 0.4949 | 20.0 | 12.3 | 0.9599 | 24.42 |
| 6000\* | 0.9946 | 0.4713 | 20.0 | 11.6 | 0.9573 | 22.89 |
| 9000\* | 0.9914 | 0.4313 | 20.0 | 10.7 | 0.9570 | 22.76 |
| 13000\* | 0.9873 | 0.4071 | 20.0 | 10.3 | 0.9502 | 19.59 |
| 18000\* | 0.9914 | **0.3921** | 20.0 | 9.9 | 0.9492 | 19.20 |
| 24000\* | 0.9878 | **0.3921** | 20.0 | 9.6 | 0.9519 | 20.29 |
| 30000\* | 0.9819 | **0.3912** | 20.0 | 9.4 | 0.9521 | 20.37 |

**The frame holds.** `channel_return` reads **0.9819** at 30,000 under the shipped transport rule with
nothing holding it there — fifteen times #630's horizon, and the rule still does not walk off the
frame.

## §2 — Decay, or relaxation to a nonzero point? **Relaxation**, and the rungs separate them alone

The model-free discriminator, per adjacent rung pair — no fit, per the ticket's *fit nothing the
checkpoints do not support*:

| rung pair | 1000→2000 | 2000→4000 | 4000→6000 | 6000→9000 | 9000→13000 | 13000→18000 | 18000→24000 | 24000→30000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `diff` per 1k | −0.0251 | −0.0206 | −0.0118 | −0.0133 | −0.0061 | −0.0030 | **−0.00001** | **−0.00014** |

The rate falls monotonically (bar one non-monotone step at 6,000→9,000) and arrives at the noise
floor. Over the last 12,000 ticks the level moves by **0.0009**, against ~5% run-to-run jitter on
this quantity — that is not a rate, and a quotient of two such near-zeros carries no shape
information, which `b61_analyse.shapes` now says in terms rather than reporting a misleading ratio.

Both candidate shapes are reported as arithmetic, neither as a mechanism. The straight line through
the rungs at/after 2,000 runs at −0.00475 per 1k and would reach the null's level at tick ~79,000 —
but it is dominated by the fast early decay and the flattening tail is exactly what refutes it.

## §3 — Against the null's own drift, which is the reading that matters

[B50 (#618)](https://github.com/NGL321/patchworks/issues/618)'s rule: the comparison is the gap at a
rung, never the level against 0.0000.

Seed 42, and seed 43 beneath it — **both seeds, both arms, every shared rung**:

| ticks | 2000 | 4000 | 6000 | 9000 |
| ---: | ---: | ---: | ---: | ---: |
| `s19` diff, seed 42 | 0.5361 | 0.4949 | 0.4713 | 0.4313 |
| `s0` diff (flat bundle), seed 42 | 0.0803 | 0.0952 | 0.1079 | **0.1269** |
| **gap, seed 42** | 0.4558 | 0.3997 | 0.3635 | **0.3044** |
| gap rate per 1k, seed 42 | −0.041 | −0.028 | −0.018 | −0.020 |
| `s19` diff, seed 43 | 0.5436 | 0.5046 | 0.4716 | 0.4397 |
| `s0` diff, seed 43 | 0.0864 | 0.1203 | 0.1313 | **0.1511** |
| **gap, seed 43** | 0.4572 | 0.3842 | 0.3402 | **0.2887** |
| gap rate per 1k, seed 43 | −0.051 | −0.036 | −0.022 | −0.017 |

**The null's *level* is the more seed-sensitive of the two arms, and its *rate* is not.** `s0` reads
0.1269 (seed 42) against 0.1511 (seed 43) at 9,000 — a 19% spread, against the staggered arm's 2%
(0.4313 / 0.4397) — but the climb rate over the last rung pair agrees closely, **+0.0063 and +0.0066
per 1k**. So the flat bundle's *trajectory* is reproducible while its level is not, and a gap quoted
at a rung carries the null's spread. Both seeds give the same qualitative picture: gap ~0.29–0.30 at
9,000, still narrowing, both rates decaying.

**So the honest split is:** the candidate's level settles; the *gap* is still closing at the deepest
shared rung, and closing because the null climbs. The gap's own rate is also falling (−0.041 →
−0.020 on seed 42, −0.051 → −0.017 on seed 43), and against a settled candidate at 0.391 a null
rising at ~0.0065 per 1k would need on the order of 40,000 further ticks to close the remaining 0.29
— with both rates decaying. **A crossing is not in evidence, and neither is a proof that none
occurs.** That is the limit of what these rungs support.

**And it puts a question against the null itself.** [B48 (#615)](https://github.com/NGL321/patchworks/issues/615)'s
null is *the flat bundle's 88 at audience differentiation **0.0000***, and 0.0000 is its defining
property — B42 derived collapse as the *only* point of exact path-independence. Trained under the
shipped rule the flat bundle **does not stay there**: it reads 0.1269 and 0.1511 at 9,000 on the two
seeds and is climbing on both. So the object the joint rule names is a **construction** reading, and
the trained flat bundle is a different object that no ticket on this map has characterised. That is
not a defect in B48's rule — the rule is stated at construction and B42's derivation is exact there —
but a bar carried against "the flat bundle" now has to say **which** flat bundle, and the answer
changes with the horizon it is read at. Ticketed rather than ruled here.

## §4 — `p = 8` against `p = 12`: the ordering **survives**, narrowed and noisy

| ticks | 0 | 2000 | 4000 | 6000 | 9000 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `p8` diff | 0.8036 | 0.5684 | 0.5189 | 0.4796 | 0.4620 |
| `p12` diff | 0.7321 | 0.5361 | 0.4949 | 0.4713 | 0.4313 |
| **p8 − p12** | +0.0714 | +0.0323 | +0.0240 | +0.0083 | **+0.0307** |
| `p8` / `p12` participation | 16.8 / 14.3 | 14.1 / 13.6 | 12.9 / 12.3 | 12.2 / 11.6 | **11.5 / 10.7** |
| `p8` / `p12` `chan` | 1.0000 / 1.0000 | 0.9909 / 0.9986 | 0.9883 / 0.9969 | 0.9849 / 0.9946 | 0.9790 / 0.9914 |

`p = 8` leads at **every** shared rung on differentiation and on participation. The margin narrows
from construction (+0.071) to roughly +0.03 and is **noisy** — the +0.008 at 6,000 is a dip, not a
trend, and at ~5% jitter the rung-to-rung ordering below ~0.03 is not resolvable on one seed. `p = 8`
relaxes on the same shape (rate ratio 0.298, implied asymptote ~0.455 against `p12`'s ~0.39).

**One cost #630 did not emphasise:** `p = 8` holds `channel_return` consistently *worse*
(0.9790 against 0.9914 at 9,000).

**This ticket does not decide `p`, and does not rescope
[B44 (#608)](https://github.com/NGL321/patchworks/issues/608).** #630's §4 case rests on **two**
prices — the trained differentiation *and* B22's graded community band — and only the first is read
here. The construction-side ceiling is integer arithmetic and seed-invariant by derivation; nothing
here touches it. What is added is that the trained price's *margin* is smaller and noisier at horizon
than at 2,000. Advisory for #608, not a body edit.

## §5 — Exposure: #630's headline gain is a **start that decays**

| ticks | 2000 | 6000 | 9000 | 18000 | 30000 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `s19` participation | 13.62 | 11.55 | 10.66 | 9.90 | **9.40** |
| rate per 1k | −0.81 | −0.36 | −0.30 | −0.085 | **−0.032** |

Same shape as the differentiation: a decaying rate, not a collapse. But the level lands at **9.40**
where #630's headline was **13.6, quoted as +4.9 over random init's 8.7**.

**Two comparators, and they say different things — so both are stated separately, per
[B49 (#616)](https://github.com/NGL321/patchworks/issues/616).**

- Against the **flat bundle run to the same horizon**, the advantage plainly persists: 10.66 against
  the null's **6.76** at 9,000, a gap of **+3.90**, and the null's participation is flat at ~6.8 from
  tick 500 on.
- Against **B56's random init**, the comparison **cannot be made at horizon at all**: that arm was
  only ever run to 2,000. Quoting 9.40 against 8.7 would set a 30,000-tick reading beside a
  2,000-tick one, which is the cross-horizon move this ticket exists to stop.

**`k_v` rank reads 20.0 at every rung on every arm**, so B22's construction-time clamp holds
throughout and [B59 (#631)](https://github.com/NGL321/patchworks/issues/631)'s void clause is
satisfied by reading participation rather than rank — which is exactly why the clause says
participation.

## §6 — What a 30,000-tick reading on a body that stopped at 100 is a reading *of*

**The ticket's named risk, and it does not resolve into a clean pass.**

B56's stall exemption covers *a term whose gradient never reads a stalk*. These arms are the
**shipped transport rule**, which does read stalks, so #630 did not claim the exemption and this does
not either. **Every rung past 100 is drift under a frozen stimulus.** A longer horizon makes that
worse, not better.

What the horizon can and cannot buy, stated separately:

- **What it buys.** The *shape* question — is the rate falling — is a question about the trajectory
  of the transport maps under a fixed input distribution, and the rungs answer it on their own terms.
  The rate falls to the noise floor and the level stops. That is a fact about the rule's behaviour on
  this surface, and it falsifies #630's extrapolation, which was the ticket's central ask.
- **What it does not buy.** It says **nothing about a staggered frame in a live agent**. The body
  stopped at 100 and the stimulus has been frozen for 99.7% of the run. So the settle is a settle
  *of drift under a frozen stimulus*, and whether a live body would hold, raise or destroy it is
  untouched — [B38 (#599)](https://github.com/NGL321/patchworks/issues/599)'s territory.

**A stamping bug found here and reported rather than worked around.** `b56_analyse.horizon` returns
the **last** rung clearing the motion threshold. Over 2,000 ticks that is the same as *when did the
body stop*, because the body stops once and stays stopped. Over a long run it is not — the world
**sporadically re-crosses** `MOVING = 1e-2` long after it has died, and a single such rung drags the
reported horizon to it and marks everything before it live:

| run | falls under after | later re-crossings | B56's stamp | honest stamp |
| --- | ---: | --- | ---: | ---: |
| `s19` seed 42 → 30,000 | 150 | 30,000 (3.91e-02) | 30,000 | **100** |
| `s19` seed 43 → 9,000 | 150 | 9,000 (5.48e-01) | 9,000 | **100** |
| `s0` seed 42 → 9,000 | 150 | 6,000 (4.90e-02) | 6,000 | **100** |
| `p8` seed 42 → 9,000 | 150 | none | 100 | **100** |

Three of four runs mis-stamp, and the re-crossing is not always at the final rung, so it is not a
read-window artifact of ending the run. `b61_analyse.stall_diagnostic` reports **both** rules plus the
blips and substitutes neither. **Any long-horizon arm read through `b56_analyse` alone will mis-stamp
its stall this way, and will report drift rungs as live.** This is not a #634 finding about the
architecture; it is an instrument defect that reaches every ticket taking a long reading on B56's rig.

## §7 — Seeds: the decay trajectory replicates, and it is not the object B38 found varying 13×

[B38 (#599)](https://github.com/NGL321/patchworks/issues/599)'s 13× spread is about the **stall
horizon** — when the body stops. That is not the same object as the decay trajectory of the transport
maps, and this ticket finds the latter highly reproducible. Every run stamps its own horizon and
inherits nothing.

`s19_baseline`, seed 42 against seed 43, at every shared rung:

| ticks | 2000 | 4000 | 6000 | 9000 |
| ---: | ---: | ---: | ---: | ---: |
| seed 42 `diff` | 0.5361 | 0.4949 | 0.4713 | 0.4313 |
| seed 43 `diff` | 0.5436 | 0.5046 | 0.4716 | 0.4397 |
| seed 42 participation | 13.6 | 12.3 | 11.6 | 10.66 |
| seed 43 participation | 13.2 | 12.4 | 11.8 | 10.70 |

The levels agree to within the jitter (0.4313 vs 0.4397 at 9,000; participation 10.66 vs 10.70), and
**the decaying rate replicates**:

| rung pair | 1000→2000 | 2000→4000 | 4000→6000 | 6000→9000 |
| --- | ---: | ---: | ---: | ---: |
| seed 42 per 1k | −0.0251 | −0.0206 | −0.0118 | −0.0133 |
| seed 43 per 1k | −0.0338 | −0.0195 | −0.0165 | −0.0106 |

Seed 43's sequence is monotone; seed 42's has one non-monotone step. Both fall by roughly a factor of
three over the range.

The null replicates too, and differently — see §3: `s0`'s **level** carries a 19% seed spread at
9,000 against the staggered arm's 2%, while its **climb rate** agrees to within 5%. The flat bundle
is the noisier arm of the two, which matters because it is the one driving the gap.

**What the second seed does *not* establish.** It reaches 9,000, not 30,000, so it replicates the
**decaying rate** and not the **settle**. The claim that the level stops moving rests on seed 42
alone (§9).

Run-to-run jitter at a *fixed* seed is **~5%** on differentiation at 9,000 (two `s0` attempts read
0.1205 and 0.1269), which is the precision of every single-seed statement above.

## §8 — What landed alongside: [B62 (#635)](https://github.com/NGL321/patchworks/issues/635) names the rule this decay should be attributed to

B62 resolved on #532 while these arms were running, and it bears on the *mechanism* of the decay
without touching the *shape* measured here. Its decomposition of the shipped pair:

- with both rules off past construction, the traffic's rank does not move at all;
- **`PredictionRule` alone** reproduces the rank collapse at **zero cost in exposure or audience
  differentiation**;
- **`TransportRule` alone** *raises* the rank and buys alignment for 2.03 dimensions of exposure;
- **the shipped pair** costs a differentiation column that neither lever costs by itself.

**Every arm in this ticket runs the shipped pair**, so the decay measured in §1–§2 is the pair's, and
B62's reading says the pair is exactly where a differentiation cost appears. **Which of the two rules
drives the settle is therefore not answered here and should not be inferred from these arms** —
B62's own standing constraint is that a candidate names which rule it is aimed at and may not argue
a rule effect from the pair. Running `s19_baseline` under `TransportRule` alone and under
`PredictionRule` alone is the obvious next reading and is cheap on this rig; it is named in the
resolution rather than taken here.

Nothing in B62 contradicts §1–§2: it takes no long-horizon reading and its arms are 5,000 ticks.

## §9 — What the box permitted, and what is consequently owed

The machine was saturated throughout — **62.7 GB committed of a 64.8 GB limit (97%)** at worst, with
44 `claude` processes holding 10.9 GB and other sessions' benchmark processes holding 1.5–1.7 GB
each. The harness's low-memory guard killed **five** arms. Horizons were therefore re-scoped rather
than abandoned: the primary arm keeps its full 30,000, and the panel arms run to 9,000 — 4.5× #630's
horizon — so that every arm shares rungs.

**Owed, and named rather than papered over:**

- The **null past 9,000**, on either seed — which is what a crossing claim would need. An attempt
  reached 18,000 and read `s0` at 0.1429 (13,000) and 0.1638 (18,000); it was **overwritten by a
  retry before the staging fix landed** and is not reproducible from disk. Those two numbers are
  recorded here as observed-then-lost and **no claim rests on them** (they sit close to seed 42's
  trajectory, between the two seeds' 9,000 readings, which is the only thing worth saying about
  them). The gap beyond 9,000 is unmeasured.
- **`p = 8` past 9,000**, and a second seed on it.
- The settle itself is measured on **one seed**, because only one arm reached the rungs where the
  flattening is unambiguous. Seed 43 replicates the decaying rate to 9,000 and stops short of the
  flat region.

**A rig fix landed with it.** B56's `run_arm` writes each rung to `<name>.inflight.json` and renames
at the horizon — which bounds a kill — but it writes from the first frame, so **a retry that dies
early destroys a deeper attempt**. `b61_horizon.run_one` now stages any leftover in-flight file aside
as `<name>.killed-<n>.json` before starting, and `b61_analyse.load` takes the **deepest** attempt on
disk rather than the newest, naming the file and depth it used.
