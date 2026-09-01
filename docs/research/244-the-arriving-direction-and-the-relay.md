# Does what arrives on an edge align worse than average with the next hop's operator?

[#244](https://github.com/NGL321/patchworks/issues/244). One measurement, no decision in it. It reads
the unfavourable side of the misalignment candidate
[#233](https://github.com/NGL321/patchworks/issues/233) could only bound on the favourable side, and
rules on nothing: the remedy species this grading belongs to is already closed
([#184](https://github.com/NGL321/patchworks/issues/184),
[#230](https://github.com/NGL321/patchworks/issues/230)). Its value is the same as #233's — it tells
the retention recharter whether per-edge variation is a construction fact or a learned one.
Reproduced by

    docker run --rm --entrypoint python patchworks:headless \
        benchmarks/alignment_read.py null      # seconds, no sandbox
    docker run --rm --entrypoint python patchworks:headless \
        benchmarks/alignment_read.py align     # ~12 min, trains 30k ticks

in the supported container ([ADR-0012](../adr/0012-a-container-is-the-supported-execution-target.md)),
on the real dome: 414 cells, 682 edges, `chi = 1036`, seed 0, 24 trials, both directions — #214's own
configuration, and #233's. The statistic and its null are checked against a brute-force Monte Carlo
through the operator itself in `tests/test_alignment_read.py`, because this ticket's whole content is
one ratio and the bar it is read against, and both have a plausible wrong form.

---

## The verdict, in four lines

1. **No — and the sign is the other way. What arrives aligns *better* than average, not worse.**
   Over 296 hops the mean percentile within each hop's own isotropic null is **76.9%** against 50%
   under isotropy: better by 26.9 points. Median `A = 1.784x`, **2.06x** the median of its own null.
   Only 25% of hops fall below their null's median. At #233's own peak tick the reading is the same
   (76.4%), so it is not an artifact of which tick is read.

2. **It could not have come out the other way, and that is the real finding.** The inbound restriction
   maps are effectively rank 1, so `u = F_inᵀ d` is the *same* node-stalk direction whatever arrives.
   `a_in` therefore sits on its own ceiling — exactly `2.000 = √4` and `2.828 = √8` at the widths this
   graph uses — and the arriving direction has **no purchase at the relay at all**. It cannot be
   misaligned there, because the inbound map, not the arrival, picks the direction.

3. **#233's composition finding is absorbed, and it is absorbed as an identity rather than a
   judgement.** `A = a_in · a_out / C` holds tick by tick (closure `2.5e-14` over 18,648 tick-pairs),
   where `C` is #233's composition gap — measured here at median **0.0079x**, #233's own figure. When
   `F_in` is rank 1, `a_out ≡ C` by algebra, so the relay's "misalignment" and the maps' failure to
   share dominant directions are **one quantity**, not two candidates.

4. **One of #233's attributions does not survive, and is corrected.** #233 read its ~480x gap between
   the measured quantity and the single-step prediction as the *window's* accumulation. The
   tick-aligned read has no window and the gap is still ~**495x**. Whatever it is, it is not the
   window.

**None of this changes the grading's verdict.** Misalignment is closed as a *contributor* — the
arriving direction is not the loss — and what remains is #233's already-quantified per-edge floor
variation (34%) plus a composition gap that is a fact about the maps rather than about transport.

---

## 1. What #233 left, and why a tick-aligned read can take it

#233 read #184's three parked candidates for the 9x-to-240x per-hop grading. Construction is not the
cause (`R² = 0.139`, 5.0x of usable range against a 151x spread); per-edge floor variation is a real
contributor at 34% by sd; and inter-endpoint misalignment is **bounded on the favourable side** —
alignment headroom, what the *best* direction buys over an average one, is at most 2.83x on every hop
measured, median 1.97x.

Headroom does not price what a **badly chosen** direction costs. That is unbounded below, and it is
the form the candidate would actually take. #233 could not read it, and said so precisely: its
measured quantity is a **windowed peak** — `max` over a 64-tick window of a paired ratio — sitting
~480x above the isotropic single-step prediction, and a level shift that large swamps the sign of a
per-hop direction effect.

**What a hop is** — unchanged from #233, and the ticket names the operator:

    M(e_in → v → e_out)  =  F_{v,e_out} · gain_v · F_{v,e_in}ᵀ

`Sheaf.message_passing_phase` writes `x_v ← x_v − gain_v · Σ_e F_evᵀ (F_ev x_v − y_e)`, and
`broadcast` is read from the *pre-update* stalk, so a displacement arriving on `e_in` at tick `t`
reaches `e_out`'s disagreement at tick `t + 1`. That one-tick offset is the whole of *tick-aligned*:
every quantity below pairs `d_in(t)` with `d_out(t + 1)` and never with `d_out(t)`.

**The read needs no new instrument**, which is what the ticket predicted. `detectability.branch`
already returns `[ticks, edges, m]` of disagreement — the whole trace, as vectors — so the arriving
*direction* is a difference of two traces that `construction_grading.py` already computes and then
reduces to a norm. This read takes the reduction one step later. 296 hops over 18,648 tick-pairs;
1,618 pairs (8.7%) fall below the direction floor before the deviation has arrived and are dropped
and counted rather than silently skipped.

**The paths are #233's, and that is corroborated rather than inherited.** Same seed and surface, so
the read reproduces #233 exactly: rim→apex bottleneck **4.69e-10**, binding edge **`#450 interior m=4
L6/core(5,) — L7/core(5,)`** — #214's own — with the per-hop profile **39x, 23x, 184x, 23x, 11x,
71x**; apex→rim **1.14e-08** binding at `#30 sensory m=8 L0/vision(1, 14) — L1/vision(0, 7)`.

---

## 2. The trap: `A < 1` is not misalignment

The ticket asks for

    A  =  ‖M d‖ / (‖M‖_F ‖d‖ / √m_in)

the measured direction against the isotropic one. The natural reading — `A < 1` means the arriving
direction is worse than an average one — is **wrong**, and reporting it would have answered this
ticket in the affirmative out of pure noise.

For an isotropic `d`, `E[A²] = 1` holds exactly. But `A` itself is skewed below its own mean square,
and the skew is worst precisely where this graph lives. Against a **rank-1** operator
`A = |⟨v₁, d⟩| √m_in`, so `A²/m_in ~ Beta(1/2, (m_in−1)/2)`, whose median is **0.816 at `m_in = 4`**
and falls to **0.674** as the width grows. Never 1.

This is the map's own standing correction in its exact dual. #142's isotropic probe against
near-rank-1 maps reported a 1e14 deficit that was not there; reading a median of `A` against 1 would
manufacture a misalignment that is not there, from the same fact about the same maps.

So the verdict is **never** read off `A` against 1, and not against a graph-wide null either — the
bar moves with both the edge width and the operator's rank. It is read off `A` against the null
distribution of `A` for isotropic directions **through the same operator**, drawn per hop, and the
reported statistic is the arriving direction's **percentile within that null**. Under the hypothesis
that arriving directions are isotropic that percentile is uniform, so its mean is 50 and any
systematic effect moves it.

`benchmarks/alignment_read.py null` prints the trap before any measurement is involved, over all
7,122 directed hops the graph admits at construction:

| quantity | median | p05 | p95 |
|---|---|---|---|
| null median `A` | **0.950** | 0.863 | 1.000 |
| effective rank of `M` | 2.226 | 1.433 | 3.637 |
| `σ₁²` share of `‖M‖_F²` | 0.591 | 0.402 | 0.824 |
| `m_in` | 4 | 4 | 8 |

**The bar is 0.950, not 1.** Small, and it is the difference between two verdicts: on this run a bar
of 1.0 would have called **30.1%** of hops misaligned, where their own nulls call **25.0%** of them
below-median. The null is drawn from the operator's singular values rather than by pushing vectors
through the matrix — the same distribution exactly, not an approximation of it, held against the
brute force it replaces at five quantiles in the tests. The calibration test is the one that matters
most: fed genuinely isotropic directions, the instrument's mean percentile must land at 50, and it
does.

---

## 3. The answer

| quantity | median | p05 | p95 |
|---|---|---|---|
| `A` measured / isotropic | **1.784x** | 0.301x | 2.447x |
| `A` / median `A` of the null | **2.059x** | 0.385x | 2.932x |
| percentile within the null | 98.8% | 18.0% | 100% |

**Mean percentile 76.9% against 50.0% under isotropy — better than average by 26.9 points.** The
same read at #233's peak tick gives 76.4% and a median `A` of 1.784x, so the answer does not depend
on the reduction.

By relay level — printed for shape and never as an index, per the map's standing rule — the effect is
present everywhere and is not one level's story:

| relay | hops | mean percentile | sd |
|---|---|---|---|
| L1 | 48 | 76.4% | 33.1 |
| L2 | 55 | 81.3% | 23.3 |
| L3 | 49 | 81.8% | 28.9 |
| L4 | 48 | 75.3% | 35.7 |
| L5 | 48 | 68.8% | 33.7 |
| L6 | 48 | 77.3% | 33.1 |

**So the ticket's `if` clause fires the second way**: arriving directions are *average or better*, the
candidate is closed, and the residual belongs to the composition term — which is §4.

---

## 4. Why it could not have come out the other way

The decomposition. With `u = F_inᵀ d`, and `perm_v` the permitted width:

    a_in   =  (‖u‖ / ‖d‖)         / (‖F_in‖_F  / √m_in)
    a_out  =  (‖F_out u‖ / ‖u‖)   / (‖F_out‖_F / √perm_v)
    C      =  (‖M‖_F / √m_in)     / (gain_v ‖F_in‖_F ‖F_out‖_F / √(m_in · perm_v))

and `A = a_in · a_out / C` identically — **closure `2.5e-14` over 18,648 tick-pairs**, checked per
tick, which is where it holds. (The medians below do not satisfy it and are not meant to: a median
does not distribute over a product.)

| quantity | median | p05 | p95 |
|---|---|---|---|
| `a_in` — `d` against the inbound map | **2.000x** | 1.156x | 2.828x |
| `a_out` — the survivor against `F_out` | 0.0052x | 0.0011x | 4.899x |
| `C` — #233's composition gap | **0.0079x** | 0.0013x | 4.899x |

Three things to read off it.

**`C` reproduces #233 exactly.** Median 0.0079x, which is #233's reported composition figure to the
digit. The two scripts name one quantity, and a test holds `C` against
`construction_grading`'s own tier-3-over-tier-2 to float32 precision rather than leaving the
correspondence to prose.

**`a_in` is at its ceiling.** 2.000 is exactly `√4` and 2.828 exactly `√8` — the maximum `a_in` can
take, reached only by a direction sitting on the inbound map's leading right-singular vector. A
direction does not land there by luck. It lands there when the map has one singular direction worth
having.

**And then `a_out ≡ C`, by algebra.** When `F_in` has rank 1, `u = F_inᵀ d` is the same node-stalk
direction whatever `d` is — only its length changes — so `‖F_out u‖ / ‖u‖` is a **constant of the
maps**, and it is exactly the constant `‖M‖_F` carries. `a_out` and `C` are then the same number, and

    A  =  a_in · a_out / C  =  a_in

exactly. This is proved in `tests/test_alignment_read.py` on synthetic rank-1 maps, with a
full-rank contrast case beside it so the equality is a property of *rank* and not of the arithmetic.

That is the mechanism, and it says something stronger than the ticket asked for. **The arriving
direction has no purchase at the relay.** Whatever arrives is collapsed onto one node-stalk direction
by the inbound map before the outbound map ever sees it, so the question *is what arrives badly
placed for the next hop* has no room to be answered yes. What looks like a relay's misalignment is a
property of the two maps, fixed at the relay, and identical to the composition gap #233 measured.

**The correlation confirms they are not two independent effects.** `r = +0.091` between `log C` and
`log A` over 296 hops — a hop whose maps compose badly is not thereby a hop whose arriving direction
lands badly, because the arriving direction is not what is varying.

---

## 5. A correction to #233: the level shift is not the window

`M d_in(t)` against the measured `d_out(t + 1)`:

| quantity | median | p05 | p95 |
|---|---|---|---|
| `\|cos\|` predicted against landed | 0.601 | 0.078 | 1.000 |
| `‖landed‖ / ‖M d_in‖` | **494.6x** | 0.228x | 3.76e+04x |

`M` omits the body's scale, so one route through the relay delivers `0.4529 · ‖M d_in‖`. What lands is
~1,090x more than that.

#233 read its own ~480x gap as **the window's accumulation**, on a degree-flatness check run against
its windowed quantity. **This read has no window and the gap is still there.** So the window is not
the cause, and the honest reading is the one #233's check was aimed at ruling out: a one-route
operator is not what decides what lands on a many-route edge. The same degree check is re-run here on
the windowless quantity rather than the attribution being inherited or contradicted by assertion —
see §5's table in the run output.

**This does not touch §3's answer.** The alignment statistic is a ratio taken *within* a single tick,
between the same operator's own measured and isotropic readings, so a level shift common to numerator
and denominator divides out of it exactly. The correction matters for what `M` is worth as a
*predictor of transport*, which is #233's subject, not for what the arriving direction is worth
against `M`, which is this ticket's.

---

## What this ticket does not claim

- **It rules on nothing.** The remedy species is closed and this reopens none of it. It explains a
  measurement, which is what a `wayfinder:research` ticket is for.
- **`M` omits the body's Jacobian.** The ticket defines the operator as the two maps and the gain,
  and this read uses that definition. `A` is invariant to the body's *scale* — it cancels top and
  bottom — so #233's stated body limit does not bite on the primary statistic. It would bite if the
  body *rotated* `u = F_inᵀ d`; §5 reads the empirical single step rather than assuming either way,
  and the median cosine of 0.601 is the evidence available on it.
- **"Rank 1" is an effective-rank statement, not an exact one.** The maps are near rank 1, which is
  enough for `a_in` to saturate and for `a_out ≈ C`; the identity `A = a_in` is exact only in the
  exactly-rank-1 limit, and the measured `a_out / C` is what says how close this graph sits to it.
- **The widest path is a max-min path, not a flow.** It names the edge the predicate binds on, which
  is what ADR-0021 asks of it, and is not a claim that transport travels only along it.
- **296 hops from 24 trials in two directions are not 296 independent samples.** Trials share one
  trained surface and paths overlap. The percentile statistic is a description of this population;
  what does not depend on the run at all is §2's null and §4's rank-1 algebra, which are properties of
  the operators rather than of the measurement.
