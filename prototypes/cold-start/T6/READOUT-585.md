# B29 (#585): the sheaf is not flat, it is flat exactly where it carries one dimension, and flatness is not the rank lever

**The reading, in one line.** On the 45 cycles where holonomy has more than one dimension
to be non-flat in, departure from the identity sits **at chance on every arm, at every
checkpoint, from construction to horizon** — and the only place path-independence is
achieved is the 250 loops a lateral caps at **rank one**, where the share of loops
returning their carried direction positively climbs from exactly **0.500 at construction**
to **0.69–0.81 by 2,000 ticks**. Across arms, **channel return and composed rank move in
exact opposition**: the arms whose channel comes back have composed rank ≈ 1, and the arms
with composed rank 2.9–4.0 have channel return at chance.

Instruments: `b29_holonomy.py` (the surface), `b29_theory.py` (the algebra),
`b29_nulls.py` (the null per width), `b29_analyse.py` (every table below).
Raw: `585-holonomy-<arm>-baseline-seed42-20000.json`, `585-theory.json`.

---

## 0. What the graph decides before a map is drawn

`benchmarks/holonomy_read.py` has enumerated this cycle basis since #315, but it has never
been run on a surface with **per-edge** lanes. [B8](https://github.com/NGL321/patchworks/issues/548)'s
allocation changes what the instrument is pointed at, and the first reading is a census:

| interior cells | interior edges | width-1 | wider | basis cycles | capped at rank 1 | wide-basis cycles |
|---|---|---|---|---|---|---|
| 150 | 409 | **215** | 194 | 260 | **250** | **45** |

**Every width-1 edge is a lateral (same level at both ends); every wider edge is
cross-level. The split is exact, and identical on all four arms** — it is a property of
the allocation and the graph, not of `p`, not of the seed, not of training.

A cycle's holonomy has rank at most the narrowest lane it passes through. So **250 of the
260 independent interior cycles carry a rank-1 holonomy before a single map is drawn**, and
on those "is the sheaf flat" has no whole-operator content: a 1×1 holonomy is a sign.
`flatness` and `channel_return` are identically 1 there whatever the surface does, and
`identification` can only be 0 or `√2`. **Read from `cycle[0]` as `holonomy_read` does, 151
of the 260 cycles base at a width-1 edge and those constants carry the median.** Every
cycle here is instead based at its **narrowest** edge — the space the loop actually
carries — and the two populations are reported apart.

Dropping the width-1 edges leaves a subgraph that is **still connected and still spans all
150 predicting cells**, on 194 edges with **45 independent cycles**. That is the sheaf's
cross-level transport, and it is the only place on this surface where *trivial*, *abelian*
and *non-abelian* are three different answers.

---

## 1. How far from flat: at chance, everywhere, always

The wide basis at the horizon, against a Haar null drawn on **each arm's own block
structure** and against the same trained maps rewired among endpoints of equal shape:

| arm | p | identification | Haar null | rewired null | σ_max | channel return | Haar null |
|---|---|---|---|---|---|---|---|
| `shipped` | — | 0.9320 | 1.0001 | 0.9799 | 4.238e-03 | **0.9383** | 0.3067 |
| `reserve_p8` | 8 | 0.9599 | 1.0023 | 1.0005 | 4.572e-04 | **0.8665** | 0.2161 |
| `reserve_p16` | 16 | 0.9818 | 0.9985 | 0.9986 | 4.594e-04 | 0.2287 | 0.2117 |
| `reserve_p24` | 24 | 0.9497 | 0.9981 | 1.0092 | 2.638e-03 | 0.3517 | 0.2600 |

`identification` is `‖UVᵀ − I‖_F/√(2m)`: **0 is flat, 1 is chance.** Every arm reads
0.93–0.98 against a null of ~1.00, and the trajectory (below) shows it never moves: it
starts at chance at construction and is still at chance at 20,000 ticks. **Whole-operator
path-independence is not approached on any arm at any point.**

### The 250 one-dimensional loops: the one place flatness is achieved

A 1×1 holonomy poses exactly one path-independence question — **the sign**. The null is a
coin at 0.5, and the construction column confirms it empirically:

| arm | construction | 100 | 500 | 1000 | **2,000** | 5,000 \* | 20,000 \* |
|---|---|---|---|---|---|---|---|
| `shipped` | **0.500** | 0.536 | 0.604 | 0.704 | **0.796** | 0.892 | 0.988 |
| `reserve_p8` | 0.472 | 0.592 | 0.648 | 0.660 | **0.812** | 0.964 | 0.992 |
| `reserve_p16` | **0.500** | 0.560 | 0.584 | 0.644 | **0.692** | 0.856 | 0.960 |
| `reserve_p24` | 0.516 | 0.560 | 0.596 | 0.648 | **0.688** | 0.776 | 0.932 |

`*` = frozen world, see the caveat below. **Monotone on every arm, from exactly the coin.**
The sheaf *does* become path-independent — but only on the loops where transport carries a
single dimension, and it stays at chance on the loops where it carries twelve.

The null is not assumed. Drawn eight times per arm on each arm's own block structure
(2,000 readings each), the Haar share reads **0.493 / 0.487 / 0.502 / 0.495** — a coin, as
the algebra says it must be, since each hop's scalar is a symmetric inner product and a
product of fair signs is a fair sign.

### The construction sweep: chance at every `p`, over eight seeds

Seven arms × eight seeds, untrained. This is the spread the trained arms could not afford:

| arm | p | wide identification (mean ± sd) | 1-D positive share (mean ± sd) | per-chain ER |
|---|---|---|---|---|
| `shipped` | — | 1.0071 ± 0.0128 | 0.5105 ± 0.0242 | 1.1708 |
| `reserve_p0` | 0 | 1.0037 ± 0.0076 | 0.5045 ± 0.0179 | 1.5349 |
| `reserve_p8` | 8 | 1.0018 ± 0.0038 | 0.5085 ± 0.0361 | 1.4477 |
| `reserve_p12` | 12 | 1.0011 ± 0.0047 | 0.5110 ± 0.0381 | 1.3805 |
| `reserve_p16` | 16 | 1.0044 ± 0.0068 | 0.4900 ± 0.0177 | 1.3659 |
| `reserve_p20` | 20 | 0.9999 ± 0.0088 | 0.5065 ± 0.0236 | 1.2758 |
| `reserve_p24` | 24 | 0.9977 ± 0.0103 | 0.4960 ± 0.0267 | 1.1313 |

**At construction the sheaf is at chance on the wide loops and a coin on the narrow ones,
at every `p`, with a seed spread of ~0.01 and ~0.03 respectively.** So the trained
departures in §1 and §4 are departures from a baseline that is measured, tight, and flat
in `p` — not from an assumed null.

---

## 2. Is what remains abelian? No — and the ladder is wrong in its middle and bottom rungs

**On the surface**, the pairwise commutator of the polar factors of loops based at a common
edge (0 = commuting, 1 = chance) never leaves chance:

| arm | pairs | commutator (polar) | Haar null | shared-direction invariance | direction `d_eff` |
|---|---|---|---|---|---|
| `shipped` | 494 | 0.9144 | 0.9907 | **0.9893** | **1.067** |
| `reserve_p8` | 494 | 0.9753 | 0.9962 | **0.9623** | **1.417** |
| `reserve_p16` | 494 | 0.9884 | 0.9930 | 0.4381 | 8.656 |
| `reserve_p24` | 494 | 0.9460 | 0.9927 | 0.2676 | 5.624 |

**Not abelian, on any arm.** But the last two columns split the arms into two regimes:
`shipped` and `p8` have essentially **one common invariant direction** (`d_eff` 1.07 and
1.42, invariance 0.96–0.99), and `p16`/`p24` have none (`d_eff` 5.6–8.7).

**The ladder itself is an agent inference and #585 forbids inheriting it. It does not
survive.** Counting the common fixed subspace directly (`b29_theory.py`):

| d | trivial holonomy | abelian (generic torus) | non-abelian (generic) |
|---|---|---|---|
| 6 | 6 | **0** | 0 |
| 7 | 7 | **1** | 0 |
| 12 | 12 | **0** | 0 |
| 13 | 13 | **1** | 0 |

* **Top rung holds.** Trivial holonomy fixes all `d` directions — this is
  [B26](https://github.com/NGL321/patchworks/issues/575)'s citation and it survives.
* **Middle rung fails.** A generic abelian holonomy does **not** give "partial capacity":
  it fixes 0 directions in even `d` and 1 in odd `d`, the same as non-abelian. The
  inference slid from *simultaneous eigenbasis over ℂ* to *surviving directions over ℝ*.
  Commuting orthogonal matrices share a real block decomposition into 2×2 rotations; a
  fixed **vector** needs eigenvalue exactly `+1`, which a generic torus misses.
* **Bottom rung fails, in the opposite direction from the claim.** A generic non-abelian
  subgroup fixes **0** directions, not one. "One invariant direction" is a property of a
  single odd-dimensional rotation, not of a group.

**So "non-abelian holonomy" and "composed rank 1" are NOT the same fact**, and the
construction that proves it is cheap: a sheaf that is **exactly flat by construction** reads
composed **ER 12.000** over seven hops (identification 1.4e-15), while generic lanes at the
same widths read **ER 2.817** at identification 1.015. Holonomy is a property of a **cycle**
and composed rank of an open **chain**; [B1](https://github.com/NGL321/patchworks/issues/537)
already has the latter as a function of `(m, k_v, hops)` alone.

The association the ladder gestured at *does* appear on the surface — `shipped` and `p8`
have both a common invariant direction and composed rank ≈ 1 — but not for the reason it
gave, and the surface is nowhere near generic.

---

## 3. Does holonomy predict composed rank? No

Spearman ρ, per chain (`n = 263`), per apex, and per edge, on the wide basis:

| arm | ER vs identification | ER vs flatness | ER vs channel return | apex `d_eff` vs ident. | edge ER vs commutator |
|---|---|---|---|---|---|
| `shipped` | +0.097 | +0.132 | +0.013 | −0.204 | +0.238 |
| `reserve_p8` | −0.099 | −0.218 | −0.255 | −0.252 | +0.010 |
| `reserve_p16` | +0.200 | +0.415 | −0.449 | +0.333 | +0.145 |
| `reserve_p24` | −0.017 | +0.012 | −0.155 | +0.024 | −0.074 |

**The sign is not stable across arms**, and within every arm it is not stable across the
trajectory either — `reserve_p16` runs −0.329, −0.137, +0.117, +0.358, +0.363, +0.273,
+0.054, +0.200 across the checkpoint ladder. **This is the ticket's stated point and the
answer is negative: departure from flatness does not predict composed rank**, per chain,
per apex, or per edge. That is what §2's algebra already implied — they are different axes.

---

## 4. Does `p` move it? Yes — and it trades one against the other

This is the finding with consequences. Horizon values, each against **its own arm's chance**:

| arm | p | channel return / null | per-chain ER | joint `d_eff` |
|---|---|---|---|---|
| `shipped` | — | **3.060** | 1.0015 | 1.004 |
| `reserve_p8` | 8 | **4.010** | 1.4577 | 1.331 |
| `reserve_p16` | 16 | 1.081 | **2.8980** | 4.504 |
| `reserve_p24` | 24 | 1.353 | **4.0000** | 6.210 |

**An exact inverse.** The arms whose loops return the direction they carry are the arms
whose composed rank is one; the arms that reach composed rank 2.9–4.0 have channel return
at chance. The trajectory says the same thing dynamically: `shipped` climbs 0.305 → 0.855
(at 2,000) → 0.938, `p8` climbs 0.202 → 0.732 → 0.867, while `p16` and `p24` sit flat at
~0.25 throughout and never leave chance.

**`p`'s gains are not flatness gains in disguise — they are the opposite.** `p` buys
composed rank by spending exactly the path-independence the architecture asked for.

---

## 5. The tension, which does not break — it sharpens into a statement

#585 marks the incompatibility of strict path-independence with lanes that select as *"the
most load-bearing inference in this ticket and the first thing to try to break."* It does
not break. It becomes precise, and it separates into **two orthogonal failures**
(`b29_theory.py`, all three assertions verified):

Under ADR-0032's floor a restriction map has orthonormal rows, so each hop is a matrix of
principal-angle cosines and `‖C‖₂ ≤ 1`. Then `H = I` requires **both**:

* **θ = 0** — every lane subspace around the loop is the *same* subspace. This axis moves
  `σ_max` and **leaves `identification` untouched** (verified). It is the axis that contracts.
* **φ = 0** — the frames agree around the loop. This axis moves `identification` and
  **leaves `σ_max` untouched** (verified). It is the axis that relabels.

These are ADR-0032's *metric agreement* and *identification agreement* appearing as two
independent directions of failure rather than one number — which is why this surface can
read `identification` at chance **and** `σ_max` at `1e-3` without those being one finding.

**The consequence is a scoping statement, not a measurement.** Trivial holonomy requires
every edge incident to a cell to expose the same block — which is precisely the degeneracy
[B12](https://github.com/NGL321/patchworks/issues/556) named in `_assemble` and the ceiling
[B13](https://github.com/NGL321/patchworks/issues/560) found the reserve floor climbing to.
**A flat sheaf on this construction is one whose lanes have stopped being different lanes.**

Contraction is geometric in the hops, and identical in shape to the Haar null — `shipped`
at the horizon, by cycle length: `3.0e-02` (4 hops), `4.2e-03` (6), `8.1e-04` (8),
`2.5e-04` (10), `7.9e-08` (20), `6.5e-10` (24), with `identification` flat at ~0.93
throughout. **The contraction is the geometry, not the training.**

**And the refinement survives, as the thing the surface actually does.** Transport need
only be path-independent *on the content that travels* — and that is exactly the pattern:
`channel_return` at 3–4× chance and one common invariant direction on `shipped`/`p8`,
whole-operator `identification` at chance everywhere. The architecture's reachable target
is a **contextual** sheaf — flat on each coherent region, deliberately not glueable across
them — and this reading is the same measurement as
[B22](https://github.com/NGL321/patchworks/issues/571) from the other side.

---

## What this does **not** measure

Stated as plainly as the rest.

* **One seed (42) for every trained arm**, and **20,000 ticks, not 100,000.** The spread
  that is affordable is in the construction sweep (7 arms × 8 seeds), not in the trained
  arms. So every *trained* number is a single trajectory, and the seed spread quoted is a
  spread of the **untrained** baseline.
* **The world stalls.** `PlanarPushSandbox` motion collapses between 1,000 and 2,000 ticks
  and stays collapsed (#572's `world_std_*`; #518's own `travel_window` records the same
  shape). **Every checkpoint past 2,000 is taken against a world with no exogenous
  variation**, and the tables mark them `*`. The moving-world values are quoted beside the
  horizon everywhere they carry a claim; the 1-D flatness trend continues *and accelerates*
  after the stall, so **the strong version of that number is a frozen-world number** and the
  honest one is 0.69–0.81 at 2,000 ticks.
* **Whether the 1-D lateral path-independence is useful.** A sign agreement on a
  one-dimensional channel is a weak form of order-invariance, and nothing here shows it
  carries information rather than a convention.
* **The live shipped surface.** This is the cold-start rig's per-edge-lane dome.
  `benchmarks/holonomy_read.py` on `DEFAULT_SPEC` at uniform `m = 4` is a different object
  and was not re-run.
* **Whether a different cycle basis changes the picture.** A fundamental basis is not
  canonical; BFS is kept from `holonomy_read` because it gives the shortest cycles, which
  flatters the flatness hypothesis. The narrowest-edge basing choice errs the same way.
* **`rewired` was taken at the horizon only**, not per checkpoint.
* **The user's leads were not pursued.** *Associative symmetry* in human associative
  memory and its vector-symbolic cousin were not verified against primary sources and
  nothing here rests on them.
* **No architecture was changed and no ADR was touched.** This is a reading.
