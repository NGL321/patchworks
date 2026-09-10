# B58 (#630) — the staggered frame, on a trained surface

**The staggered frame is a transport-architecture candidate, and it is the first thing on
[#532](https://github.com/NGL321/patchworks/issues/532) to pass
[B48 (#615)](https://github.com/NGL321/patchworks/issues/615)'s joint rule at a live rung.**
It survives training with **no objective term at all**, and where it holds, clause 1 —
[B56 (#628)](https://github.com/NGL321/patchworks/issues/628)'s surviving candidate — is
**inert**.

Surface `reserve_p12` and `reserve_p8`, seed 42, 2,000 ticks, on
`worktree-b58-stagger-630`. Instruments: `b58_arith.py`, `b58_derive.py`, `b58_curve.py`
(construction, seconds); `b58_train.py`, `b58_analyse.py` (trained, ~2.5 min per arm).
`b58_train.py` **stands up no new rig** — it wraps `b56_channel.run_arm` unchanged and
patches only the initialisation, so every column is B56's.

---

## §1 — Why `{0, 1, 2, 6, 7, 10, 13, 14, 18, 19}`: it is integer arithmetic, and there are no maps in it

`holonomy_read.hop_operator` is `F_out · F_inᵀ`. Under `b42_stagger.build_staggered` both
incident maps at a cell are row-selections of the **same** orthonormal frame `Q_v`, so

```
hop = S_out Q_v Q_vᵀ S_inᵀ = S_out S_inᵀ
```

and **the frame cancels exactly**. `S_out S_inᵀ` is a 0/1 matrix with `(a, b) = 1` iff
`rows_out[a] == rows_in[b]` — a **partial permutation matrix** — and products of partial
permutation matrices are partial permutation matrices. So every cycle's holonomy has
singular values in `{0, 1}`:

> `sigma_max = 1` exactly when the composite is nonzero, and `0` when it is not.

Which is what B56 measured and read as a continuum: **1.000 against 4e-08 of float noise.**
There was never a magnitude here to explain. Exactness is a **survival question about row
indices**, and it is decidable by counting.

`b58_arith.py` decides it. Reproduced, with no torch, no frame drawn and no SVD taken:

| arm | `k_v` | exact staggers | B56's count |
| --- | ---: | --- | ---: |
| `reserve_p8` | 24 | `{0, 1, 2, 22, 23}` | 5 of 24 ✔ |
| `reserve_p12` | 20 | `{0, 1, 2, 6, 7, 10, 13, 14, 18, 19}` | 10 of 20 ✔ |
| `reserve_p16` | 16 | all 16 | 16 of 16 ✔ |

**Identical on seeds 42, 43 and 44**, which is now explained rather than observed: the seed
enters only through `Q_v`, and `Q_v` cancels. That is also why B56 saw `edge_overlap` agree
to four decimals across seeds while `identification` moved — one column is arithmetic and
the other is not.

`b58_curve.py` validates the prediction against B56's measured sweep: **exactness agrees on
20 of 20 staggers, maximum `edge_overlap` error 0.018.**

### What the derivation does *not* give

A **local** closed form. `b58_derive.py` checks the natural candidate — *every hop's two row
intervals intersect* — on every cycle at every stagger at every width:

> **necessary everywhere, sufficient nowhere.**

A cycle can have every hop nonzero and still compose to zero. That is #233's composition gap
in integer form, and it means the family is a **global** survival property, not a per-hop
predicate. The honest statement is that the rule is *decidable in milliseconds* and
*derived from the graph alone*, not that it has a one-line form. My first candidate — a
closure condition `s · D ≡ 0 (mod k_v)` on the accumulated slot shift — was **tested and
failed** as a necessary condition, and is recorded in `b58_arith.py::predicate` as a
falsified attempt rather than deleted.

---

## §2 — Does it survive training? Yes, and clause 1 is inert on it

Four arms, plus B56's own two as random-init controls. Rungs past each run's **own** motion
stamp marked `*`; every run here stamps at **100** ([B38](https://github.com/NGL321/patchworks/issues/599)).

| arm | tick | `chan` | `ident` | `diff` | `k_v` rank | `k_v` part |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B56 baseline (random init) | 100 | 0.2571 | 0.9934 | 0.3585 | 20.0 | 9.1 |
| B56 baseline (random init) | 2000\* | 0.3223 | 0.9697 | 0.3354 | 20.0 | 8.7 |
| B56 channel (random + clause 1) | 2000\* | 0.7899 | 0.9684 | 0.3122 | 20.0 | 8.7 |
| **s0 baseline (flat bundle)** | 2000\* | 0.9996 | 0.2815 | **0.0803** | 20.0 | 6.9 |
| **s19 baseline (staggered, no term)** | 100 | **0.9995** | 0.6839 | **0.6310** | 20.0 | **14.9** |
| **s19 baseline (staggered, no term)** | 2000\* | **0.9984** | 0.6758 | **0.5317** | 20.0 | **13.6** |
| s19 channel (staggered + clause 1) | 2000\* | 0.9985 | 0.6729 | 0.5351 | 20.0 | 13.6 |
| **p8 s22 baseline (staggered, `p=8`)** | 2000\* | 0.9909 | 0.8751 | **0.5697** | 24.0 | **14.2** |

**Three findings.**

**(a) The frame holds.** `channel_return` starts at 1.0000 and reads **0.9984** at 2,000
ticks under the shipped transport rule with nothing holding it there. The transport rule
does *not* walk off it.

**(b) Clause 1 is inert on a staggered start.** `s19_channel` and `s19_baseline` differ by
**+0.0001** in `channel_return` and **+0.0034** in differentiation at 2,000. This is
[B50 (#618)](https://github.com/NGL321/patchworks/issues/618)'s saturation confound arriving
for real — and on the *trained* surface, where B56 netted it out. B56's §5 argument was
*"a saturated quantity cannot be moved 3.9×"*, and that is right about B56's own arm; what
it does not establish is that the quantity has to *start* unsaturated. **The 3.9× rise
clause 1 buys by training is buying back what an initialisation gives for free**, and once
the frame has it, the term has nothing to push on.

**(c) The named risk does not fire.** #630's risk was that a staggered frame *"relaxes toward
stagger 0 under training and has found the null by a longer route"*. Measured, the two do not
converge: `s0` runs **0.0000 → 0.0803** and `s19` runs **0.7321 → 0.5317**, a gap of **0.45**
at 2,000 and widening at the s0 end. The flat bundle is close to stationary; the staggered
frame decays slowly and separately.

### B48's joint rule — agreement up, differentiation nonzero, exposure reported

B56's own finding was that **no arm scores** under this rule. At the **live** rung, against
`B56 baseline`:

| arm | Δ`chan` | Δ`diff` | Δ`k_v` part | |
| --- | ---: | ---: | ---: | --- |
| B56 channel (random + clause 1) | +0.0239 | −0.0000 | −0.02 | fails |
| s0 baseline (flat bundle) | +0.7427 | **−0.3246** | −1.87 | fails |
| **s19 baseline (staggered, no term)** | **+0.7424** | **+0.2725** | **+5.82** | **PASSES** |
| **p8 s22 baseline (staggered, `p=8`)** | **+0.7310** | **+0.3334** | **+6.90** | **PASSES** |

All three columns rise together, and they rise **without an objective term**. The same
ordering holds at 2,000 (Δ`diff` +0.1963 and +0.2343).

**The stall exemption does not cover these arms and is not claimed.** B56's new standing
rule exempts *a term whose gradient never reads a stalk*; these are the **shipped transport
rule**, which does. So the **live rung is the verdict** here, per B33's practice — and the
result is *stronger* at 100 than at 2,000, which is the direction that costs nothing to
argue.

---

## §3 — What it costs in exposure, in B42's own currency

[B42](https://github.com/NGL321/patchworks/issues/605)'s `b42_exposed.py` prices a cell's
exposed dimension as how many distinct frame rows reach a neighbour at all — `min(Σ_e m_e,
k_v)` today, `max_e m_e` at stagger 0. A stagger interpolates them **exactly and for free**,
and `b58_arith.py::exposure` computes it:

| arm | today | flat (s0) | at the ceiling stagger |
| --- | ---: | ---: | ---: |
| `reserve_p8` (s22) | 24.0 | 12.0 | **20.0** |
| `reserve_p12` (s19) | 20.0 | 12.0 | **17.0** |
| `reserve_p16` (s15) | 16.0 | 12.0 | **16.0** |

So at construction the ceiling stagger costs **3 rows of 20** at `p = 12` and **4 of 24** at
`p = 8` — and **nothing** at `p = 16`.

**On the trained surface it costs nothing and pays.** Two instruments, stated separately per
[B49 (#616)](https://github.com/NGL321/patchworks/issues/616):

- **Rank-measured `k_v`** reads 17.0 at construction and **20.0 from tick 50 onward** — the
  construction-time row deficit is gone by the first checkpoint. This is
  [B22 (#571)](https://github.com/NGL321/patchworks/issues/571)'s clamp again, and a
  **fourth** instance of it: `project()` re-applies the construction-fixed mask and the
  window returns to its permitted size regardless.
- **The participation ratio** — the one that actually moves — runs **13.6** on `s19` and
  **14.2** on `p8 s22` against random init's **8.7** and the flat bundle's **6.9**. The
  staggered frame carries **+4.9 to +5.5 dimensions more** effective exposure than the
  shipped initialisation.

B56 §3.1 found its term cost *no* exposure. The staggered construction is different in kind:
it **buys** exposure, and the differentiation it holds is not a reallocation inside a fixed
window but a wider window being used.

---

## §4 — Where `p` wants to sit: both prices point the same way

#630 requires this ticket's curve and [B44 (#608)](https://github.com/NGL321/patchworks/issues/608)'s
to be quoted together or neither is a decision. `b58_curve.py` extends B56's table to every
`p`, predicted rather than swept:

| `p` | `k_v` | exact staggers | **ceiling on differentiation at `chan` 1.0000** | exposed at ceiling |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 28 | 5 of 28 | **0.8036** | 20.0 of 28 |
| 8 | 24 | 5 of 24 | **0.8036** | 20.0 of 24 |
| 12 | 20 | 10 of 20 | 0.7321 | 17.0 of 20 |
| 16 | 16 | 16 of 16 | 0.6072 | 16.0 of 16 |
| 20 | 12 | 12 of 12 | 0.3000 | 12.0 of 12 |
| 24 | 8 | 8 of 8 | 0.3000 | 8.0 of 8 |

**The ceiling saturates at 0.8036 at `p = 8` and gains nothing below it.** And #608 reports,
from B22, that the *graded community band* — the band on which
[B35](https://github.com/NGL321/patchworks/issues/594)'s abstraction-as-membership has any
purchase at all — exists at **`p = 8` alone** (median community 27 cells at `p = 8`, 7 at
`p = 12`, **1** at `p = 16`).

> **The two prices on `p` do not conflict. They agree, and they agree on `p = 8`** — the
> lowest `p` at which this curve has stopped paying, and the only `p` at which #608's band
> exists.

That is worth stating plainly because the map had every reason to expect a trade: B56 framed
`p` as *"a larger reserve buys exactness everywhere and lowers the ceiling"*, a knob with two
opposed prices. It has two prices, and both run the same way.

The **trained** price at `p = 8` is measured here rather than inferred: `p8 s22` reads
`chan` 0.9909 at differentiation **0.5697** and participation **14.2** at 2,000 — better than
`p = 12`'s staggered arm on differentiation and exposure both, at 0.0075 less
`channel_return`.

**This ticket does not decide `p`.** #608 is open, is the ticket that owns the decision, and
is blocked on [B43 (#607)](https://github.com/NGL321/patchworks/issues/607) because a relayed
mask may hold graded communities at a `p` an unrelayed one cannot. What is discharged is
#630's requirement that the two prices be quoted on one axis.

---

## §5 — The ADR amendment it would need

A per-edge row offset is a statement about **where each edge's carried subspace sits inside
the node's permitted window** — placement, not shape, not width, not `k`.

- **ADR-0010 is the primary cost, and the amendment is in its own favour.** Its
  *Incoherence is gauge-fixed too* section makes `c` a cap on the **arrangement of incident
  maps at a cell**, and says in terms that this is *"fixing an unidentified parameter, not
  capping a learned one"*, while conceding that at a cell where the objective drives the maps
  coherent, *"capping the arrangement every tick is capping a learned parameter"*. A staggered
  frame **sets that arrangement at construction** — exactly the form ADR-0010 says is the
  right one — and `edge_overlap` is the quantity `c` bounds. This also speaks directly to
  [#439](https://github.com/NGL321/patchworks/issues/439), *the incoherence cap is assumed but
  never enforced where a cell's incidence is entirely pinned*: a stagger enforces it
  constructively, where a projection cannot reach.
- **ADR-0032 is untouched in substance and needs one recorded sentence.** The band and the
  partial-isometry condition are conditions on each map's *shape*; the stagger constrains
  *which rows* it occupies. The two are orthogonal — every staggered arm here runs under
  ADR-0032 unmodified — but placement stops being a free quantity and that should be written
  down rather than inferred.
- **ADR-0004 is not in this.** #533 established `k` is not a term in the composed object and
  ADR-0004 was exonerated; nothing here touches `k < n`.
- **[B41 (#604)](https://github.com/NGL321/patchworks/issues/604)'s reversibility rule is
  satisfied.** The stagger is an initialisation of a mask B42 already ruled *becomes a
  learned, gated object*; it sets what the gate opens at and deletes nothing.

**Against `main` or against the map's rulings?** Against **`main`**: `graph.py` builds no
frame per cell at all, and this is a change to construction arithmetic. It is *compatible*
with #532's ruling that the mask becomes learned and gated, and does not presuppose it.

---

## What this leaves open

- **The decay has no measured floor.** Differentiation falls 0.7321 → 0.5317 and is still
  falling at ~0.035 per 1,000 ticks between rungs 1,000 and 2,000. Nothing here shows it
  settles rather than continuing to the flat bundle over a much longer run. **A candidate
  resting on the staggered frame needs a horizon far past 2,000**, and this ticket does not
  have it.
- **One seed, one surface, two `p` values.** B38's 13× horizon spread between seeds of one
  arm is reason enough not to read a second seed off this one.
- **Everything past rung 100 is past the stall stamp**, and unlike B56's clause 1 these arms
  do not qualify for the exemption. The live-rung result is the one that carries.
- **`{0,1,2,6,7,10,13,14,18,19}` is derived but not closed-form.** §1's local candidate is
  necessary and not sufficient; whether a global closed form exists is unasked here.
