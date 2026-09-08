# B33 (#592): can a representation floor and a local holonomy term coexist?

**A prototype to react to, not an implementation.** #592 is a `wayfinder:prototype`
ticket on [#532](https://github.com/NGL321/patchworks/issues/532), whose standing
note is *plan, don't do*. Everything below is built to make the decision concrete;
every surrogate is named as one.

Instruments: `b33_coexist.py` (the two terms and the readout), `b33_motion.py`
(when the world stops), `b33_analyse.py` (the tables). Raw:
`592-coexist-<arm>-seed<n>-2000.json`, `592-motion-reserve_p12-seed42-2000.json`.
**Surface, named per the #455 rule: `main` at `e75c86c`**, `reserve_p12`, `p = 12`
per [B27](https://github.com/NGL321/patchworks/issues/576), built by
`arms.py::build_arm` on the shipped spec, rig in-process.

**And the rig has aged since, which is worth stating rather than discovering
later.** [#597](https://github.com/NGL321/patchworks/pull/597) has landed
[B15](https://github.com/NGL321/patchworks/issues/562)'s write of #556's reserve
mask into `src/`, and on that `main` `DomeSpec` no longer takes `privacy_budget`,
so `arms.py` — which applies the reserve mask to a built `Dome` by
`dataclasses.replace` precisely *because* the ruling was unwritten — does not
import. This branch is therefore based on `e75c86c`, the surface every number here
was measured on, rather than replayed onto current `main`: a replay needs
`arms.py` ported to the shipped mask first, and porting it silently would have
meant shipping numbers no checkout of this branch could reproduce. **The port is
the first thing any successor ticket on this rig owes.**

`b33_coexist.py` loads B29's `b29_holonomy.py` at runtime. That file is not on
`e75c86c`; it landed on `main` in [#588](https://github.com/NGL321/patchworks/pull/588)
and is not duplicated here. Holonomy read by
[B29](https://github.com/NGL321/patchworks/issues/585)'s `surface_read`, unchanged;
the floor read by [B19](https://github.com/NGL321/patchworks/issues/568)'s pairing,
unchanged, with its Haar control.

---

## The answer, in one line

**They coexist, and the ticket's premise is wrong twice over.** Neither term has
the other's collapse — the holonomy term reaches its full effect with a floor
present, on both seeds, without collapsing the channel. But the floor as #592
states it **cannot be built at all** under ADR-0032, and the scale-free floor that
can be built contributes **nothing measurable** once the holonomy term is present.
And the entire separation between arms happens *after* this arm's world has stopped
moving at tick ~125, so on this rig the question cannot yet be answered where it
matters.

---

## 1. The finding that reframes every other number: this arm's world dies at tick ~125

The map's standing note, from [#572](https://github.com/NGL321/patchworks/issues/572)
and [#518](https://github.com/NGL321/patchworks/issues/518), is that the sandbox
stalls between **1,000 and 2,000** ticks. That was measured on `reserve` and
`shipped`. **It does not transfer to `reserve_p12`.** Traced directly
(`b33_motion.py`, disjoint 50-tick windows on MuJoCo's own `qpos`/`qvel`, seed 42):

| tick | 50 | 100 | **150** | 250 | 500 | 600 | 1350 | 2000 |
|---|---|---|---|---|---|---|---|---|
| `std_max` | 1.287 | 1.268 | **6.9e-04** | 1.3e-04 | 4.6e-06 | 6.8e-07 | 1.3e-04 | 4.1e-04 |
| moving share | 0.500 | 0.500 | 0.250 | 0.167 | 0.167 | **0.000** | 0.167 | 0.167 |

A fall of **~1,800x between ticks 100 and 150**, zero moving components by 600,
and the partial recovery after 1,350 sits four orders below live. So the honest
horizon here is **~125 ticks**, not 2,000.

**Consequence, and it is not a caveat but the headline.** Every rung at 250 and
beyond — which is where all of this ticket's separation between arms lives — is
taken against a motionless world. Every arm is marked `*` past 150 in the tables
below and the world's own `std_max` is printed beside every row, so no number here
has to be taken on trust.

---

## 2. The floor as #592 states it cannot be built, and the reason is ADR-0032

#592 asks for "a floor on the representation" against ADR-0032's band, which is a
floor on each *map*. Building it exposed two things.

**(a) The first version floored the mask.** `maps.maps` is `[pairs, m_max, stalk]`
with `m_max = 20` the widest lane in the graph, so a width-1 edge carries 19 padded
rows the structural mask holds at zero. Only **7,722 of 27,280 slots (28.3%)** are
a lane at all. Floored naively, `relu(gamma - 0)^2 = 1` fires on every padded slot:
the loss pinned at 0.049 with gradient **7e-05**, and the floor arm came out
*bit-identical to baseline*. Fixed by masking to live lanes; recorded in the data
as `live_lane_slots`.

**(b) Masked, it is still unreachable — and this is an architectural result.**
On live lanes the emitted per-dimension std reads:

| p5 | p25 | **p50** | p75 | p95 | share >= gamma=1 |
|---|---|---|---|---|---|
| 5.3e-03 | 1.1e-02 | **1.9e-02** | 3.8e-02 | 1.0e-01 | **0.32%** |

against ADR-0032's band pinning map singular values at median **0.302**. The band
is exactly what forbids buying variance with gain, so the only legal move is a
**rotation** of `F` — and a rotation cannot raise an absolute variance whose scale
is set by a stalk that barely fluctuates ([B18](https://github.com/NGL321/patchworks/issues/567):
the mean carries it). The term is therefore *maximally violated and nearly inert at
once*.

**So an absolute variance floor is not a floor this architecture can enforce.**
VICReg's variance term assumes an encoder free to scale its output; ADR-0032
forbids precisely that. This is a conflict the ticket did not anticipate and it is
prior to the coexistence question.

**The corrected form, and it is cheap.** State the floor on the **participation
ratio** of the emitted covariance, `(tr C)^2 / ||C||_F^2` — scale-invariant, exactly
what a rotation can move, and it *subsumes* VICReg's decorrelation term rather than
needing it alongside, since a duplicated direction and a correlated pair both read
as participation 1. That closes [B32](https://github.com/NGL321/patchworks/issues/591)'s
objection (a variance floor alone "is satisfiable by duplicating one direction")
in one term instead of two. Arms `floor_si` / `both_si` carry it.

---

## 3. At their natural scales the two terms are not adversaries — one is ~950x louder

Raw gradient norms, same surface, same tick: holonomy **0.714**, absolute floor
**0.00075**, scale-free floor **0.0012**. Descended as written, the floor is not
outcompeted, it is *invisible*. So every arm below descends **unit-normalised**
gradients under stated weights (`LAMBDA_HOLO = LAMBDA_VAR = 0.05`, equal by
choice), which makes *adversaries* a claim the run can test rather than an artifact
of which term happened to be louder. This is a design call, flagged in
`extra_step`'s docstring, and it is reversible.

---

## 4. The arms

Seed 42, `reserve_p12`, wide cycles only (base width >= 2 — the population where
flatness is not free; [B29](https://github.com/NGL321/patchworks/issues/585) found
250 of 260 basis cycles rank-1 *before a map is drawn*). 40 of the 45 wide cycles
lie inside some cell's 2-hop neighbourhood and are the ones the term acts on.

**At 150 ticks — the last live rung:**

| arm | ident | chan | sigma_max | emit | corr | corr(haar) |
|---|---|---|---|---|---|---|
| baseline | 0.9830 | 0.231 | 1.89e-07 | 2.93 | +0.340 | +0.961 |
| holo | 0.9754 | 0.246 | 1.88e-07 | 2.93 | +0.298 | +0.967 |
| floor | 0.9831 | 0.231 | 1.89e-07 | 2.94 | +0.256 | +0.964 |
| floor_dec | 0.9829 | 0.231 | 1.89e-07 | 2.96 | +0.330 | +0.956 |
| floor_si | 0.9830 | 0.231 | 1.89e-07 | 2.95 | +0.321 | +0.958 |
| both | 0.9751 | 0.252 | 1.88e-07 | 2.89 | +0.361 | +0.962 |
| both_si | 0.9753 | 0.246 | 1.88e-07 | 2.83 | +0.299 | +0.957 |

**Nothing separates.** The largest flatness move is −0.008 and `corr` scatters
±0.06 across arms that differ by nothing. **On a live world, 150 ticks does not
resolve this question.**

**At 2,000 ticks — frozen world, every row:**

| arm | ident | chan | sigma_max | emit | corr |
|---|---|---|---|---|---|
| baseline | 0.9697 | 0.322 | 2.28e-06 | 2.33 | +0.358 |
| **holo** | **0.8591** | **0.745** | 2.44e-06 | 2.39 | +0.350 |
| floor | 0.9746 | 0.355 | 2.28e-06 | 2.43 | +0.243 |
| floor_dec | 0.9800 | 0.351 | 2.30e-06 | 2.57 | +0.331 |
| **floor_si** | 0.9680 | 0.344 | 2.26e-06 | **2.68** | +0.331 |
| both | 0.8629 | 0.748 | 2.46e-06 | 2.48 | +0.330 |
| **both_si** | **0.8622** | **0.738** | 2.46e-06 | **2.39** | +0.239 |

Three readings, in order of what they cost the design.

**(i) The holonomy term works, and not by collapsing the channel.** Identification
0.9697 → **0.8591**, channel return 0.322 → **0.745**, at `sigma_max` *unchanged*
(2.28e-06 → 2.44e-06). Chu et al.'s steganographic failure — satisfy the loop by
hiding the signal in a channel that carries nothing — **does not appear at the
operator level**: the loop closes *and* still carries what it carried. That is the
one place this prototype clears a hazard B32 flagged as live.

**(ii) Neither term blocks the other.** The holonomy term reaches its full effect
with a floor present: `both` 0.8629 and `both_si` 0.8622 against `holo` alone at
0.8591 — a spread of 0.004, well inside the scatter. **The literal fear behind
#592 — that each has the other's collapse — does not materialise as mutual
blocking.**

**(iii) The floor's gain is erased — and the gain itself is not established.** On
seed 42 the scale-free floor's one measurable benefit is holding emitted
participation at **2.68** where baseline decays to 2.33; add the holonomy term and
it goes to **2.39**, *identical to holonomy alone*. **Seed 43 replicates the
erasure and undercuts the gain**: floor_si reads 2.51 against baseline 2.46, a
margin of **+0.05** — inside the record's own two-run drift of 0.06 — and both_si
reads 2.48 against holo's 2.48, a difference of **exactly zero**.

So the honest statement is weaker than one seed suggested and points the same way:
**the floor's benefit is small and not established (+0.35 on one seed, +0.05 on the
other, the second within noise), and in both seeds it goes to precisely zero once
the holonomy term is present.** The two terms are not symmetric adversaries; the
holonomy term wins outright and the floor's contribution is eliminated rather than
reduced.

`corr` is **not** used to carry this: it scatters badly across seeds (baseline at
2,000 reads +0.358 on seed 42 and +0.514 on seed 43) and nothing here should rest
on it.

---

## 5. What this does not measure, said plainly

- **Everything in §4's separation is a frozen-world result.** On the live rungs no
  arm differs from any other. The honest statement is that this rig, as it stands,
  **cannot answer #592's question on a moving world**, and that is a finding about
  the rig rather than about the terms.
- **Two seeds, not more.** §7 has seed 43 on four arms. The record's two-run drift
  at 20k is 0.06, which is above several gaps quoted here — and is exactly why
  §4(iii)'s claim was weakened rather than restated from seed 42.
- **The weights are unswept.** Equal by choice, and the conclusion in (iii) is a
  statement at that ratio. A floor weighted 10x might hold; nothing here says it
  would not, and finding the crossing is a sweep this ticket did not run.
- **ADR-0011 is satisfied only in its weak form.** `local_cycles` establishes that
  *some* cell can see the whole cycle, not that the cell whose map is moved can.
  #396's objection — *"a cycle is not incident to one cell"* — is answered as far
  as a short local cycle can answer it and no further.
- **ADR-0031 is violated as built.** The floor keeps a detached 32-tick window per
  pair. The rule is forbidden momentum, running averages and scale estimates; a
  variance is a statistic over samples and one tick is one sample. This is the
  first thing the map has to price.
- **The training term is a surrogate.** Descended:
  `|| sqrt(m)*H/||H||_F - I ||_F^2`. Measured: B29's polar-factor `identification`,
  unchanged.
- **`benchmarks/detectability.py` was not run.** It is #592's named bar, and the
  reason is §1: detectability is a dynamical reading and this arm's world is dead
  from tick ~150, so a conduction ratio taken at any trained checkpoint would be a
  ratio taken on a motionless world. Reporting one would have been the exact error
  the map's own note warns against. **Composed effective rank is not reported as a
  bar** either, per [B17](https://github.com/NGL321/patchworks/issues/565).

---

## 6. What this suggests for the map, carrying no authority

Advisory, per the map's hand-off rule — these are inferences, not consequences of
a pre-registered branch:

1. **The floor must be stated scale-invariantly.** ADR-0032's band and an absolute
   variance floor are in direct conflict, and the conflict is arithmetic rather
   than empirical. Participation ratio is one form that works and subsumes
   decorrelation.
2. **A rig fix may be prior to the architecture question.** [B33](https://github.com/NGL321/patchworks/issues/592)
   cannot be answered on a world that stops at tick 125. #577's transplant route
   (`b28_gauge.py` / `b28_generalise.py`: save the maps, load them into a freshly
   built agent per arrangement) is the standing way round it and would let these
   same arms be read against a moving world.
3. **The two terms are not co-equal claimants.** If both ship, the holonomy term
   needs an explicit budget cap or the floor is decorative.

---

## 7. Seed 43 — what replicated and what did not

Four arms (`baseline`, `holo`, `floor_si`, `both_si`), same surface, seed 43.

| arm | ident @150 | ident @2000 | chan @2000 | emit @2000 |
|---|---|---|---|---|
| baseline | 0.9858 | 0.9620 | 0.438 | 2.46 |
| holo | 0.9827 | **0.8500** | **0.788** | 2.48 |
| floor_si | 0.9858 | 0.9618 | 0.426 | 2.51 |
| both_si | 0.9827 | **0.8424** | **0.790** | 2.48 |

**Replicated, and firmly:**

- **The holonomy term's effect.** 0.9620 → 0.8500 here against 0.9697 → 0.8591 on
  seed 42; channel return 0.438 → 0.788 against 0.322 → 0.745. Same direction, same
  order of magnitude, on a different draw.
- **No mutual blocking.** `both_si` 0.8424 against `holo` 0.8500 — the term is if
  anything marginally *more* effective with the floor present, and the gap is
  inside the scatter either way. This is the clearest result of the ticket.
- **The erasure.** `both_si` emit 2.48 against `holo` 2.48 — a difference of zero,
  as on seed 42.
- **The live-rung null.** At 150 ticks the arms are again indistinguishable
  (0.9827–0.9858 across all four).

**Did not replicate:**

- **The size of the floor's gain.** +0.35 on seed 42, **+0.05** on seed 43 — the
  latter inside the two-run drift the record already documents. §4(iii) is written
  to this weaker fact rather than to seed 42's number.
- **`corr` is unusable at this sample size.** Baseline at 2,000 reads +0.358 on
  seed 42 and +0.514 on seed 43. No claim here rests on it.

Two seeds is still two seeds, and both were run at the same weights.
