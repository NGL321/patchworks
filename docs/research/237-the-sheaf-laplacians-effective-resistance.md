# The sheaf Laplacian's effective resistance, rim to apex, along the channel

[#237](https://github.com/NGL321/patchworks/issues/237). One measurement, and it exists to
answer one question: **does [#230](https://github.com/NGL321/patchworks/issues/230)'s closure of
the structural remedy family survive on the sheaf side?** Reproduced by

    python benchmarks/sheaf_resistance.py control                          # ~20 s, no run
    python benchmarks/sheaf_resistance.py read                             # untrained, ~10 s + edges
    python benchmarks/sheaf_resistance.py read --learn 30000 --save maps.pt   # ~30 min
    python benchmarks/sheaf_resistance.py read --load maps.pt              # a re-read costs no run

on the real dome: 414 cells, 682 edges, `dim C⁰ = 17,104`, `dim C¹ = 3,764`. The estimator is
checked against closed forms in `tests/test_sheaf_resistance.py` — the trivial sheaf against
`graph_transmission.effective_resistance`, series and parallel, the scaling exponent, a private
direction's infinity, and the monotonicity the routes section rests on — for the reason
`docs/research/150` gives for its own checks: the whole ticket is arithmetic on an object nobody
can eyeball, and a plausible wrong number would not be caught downstream.

---

## The verdict, in four lines

1. **#150's warning is confirmed, and it is not close.** Read along the channel on a **trained**
   surface, the sheaf resistance rim-to-apex is **~3e5x** the graph resistance of the same pair.
   #150 asked whether it could be *"arbitrarily worse than the graph resistance in specific
   directions"*; the answer is yes, in the channel's own direction, on all three seeds.
2. **It is a rank phenomenon, and training causes it.** Untrained the same reading is **5.3x**,
   with the climb at **1.80 leaf edges against the graph's 1.82** — the graph's own answer. What
   changes between the two is not the graph, which is identical, but the maps: their effective
   rank falls from **2.85 to 1.0009** over 30k ticks.
3. **The topological family nevertheless stays closed, and closes harder.** The parallel routes
   #230's closure rests on are worth **3.80x** graph-side and only **1.39x–1.57x** sheaf-side,
   measured by deleting them. The term the closed remedies move is *smaller* on the sheaf than
   the graph-side number credited it with, not larger.
4. **So the reversal condition fires onto rank, exactly where #237 pre-registered it** — *"not as
   widen the graph but as the rank structure of the maps is the bottleneck"* — and that is a
   different remedy space, touching [ADR-0010](../adr/0010-restriction-map-scale-is-gauge-fixed.md)'s
   decision not to police rank, `docs/research/053`'s standing note, and
   [#57](https://github.com/NGL321/patchworks/issues/57)'s concession.

**The one-line reading:** the sheaf resistance is enormous, the graph resistance is near its
floor, and the gap between them is not somewhere a rewiring can reach.

---

## 1. What was computed, and why it is the object #150 asked for

The sheaf coboundary `δ: C⁰ → C¹` sends a node-stalk assignment to its disagreements,

    (δx)_e  =  F_{e,u} x_u  −  F_{e,v} x_v

and the sheaf Laplacian is `L = δᵀδ` (Hansen & Ghrist, arXiv:1808.01513, the cross-reference
`docs/research/148` §10 flagged as unexploited). Effective resistance between a **direction** `a`
in cell `u`'s stalk and a direction `b` in cell `v`'s is

    R  =  χᵀ L⁺ χ,        χ  =  a@u − b@v

which is the graph's `(e_u − e_v)ᵀ L⁺ (e_u − e_v)` exactly when the sheaf is trivial. #150 said
the sheaf-side object *"needs a pair of stalk directions, not a pair of cells, before 'resistance
between rim and apex' even has a referent"*, and `χ` is the whole of what that means.

**It is read as a current, which is what makes it computable at this size.** `L⁺` on 17,104
dimensions is not formed. Since `L⁺ = δ⁺(δ⁺)ᵀ`,

    R  =  min { ‖y‖² : δᵀ y = χ }

— the energy of the least-norm unit current that supplies `χ`, which is the textbook reading of
resistance and reduces to it exactly. `C¹` is only 3,764 dimensions, so `y* = (δδᵀ)⁺δχ` is
factored once and every pair after that is two matrix-vector products.

**The control.** On the trivial sheaf — one dimension per cell, every map `[1.0]` — this route
returns #150's published patch-to-apex block, **mean 1.8163 and max 2.0524**, and agrees with the
dense pseudoinverse to **2.5e-14** over 64 pairs, with a leak of **4.7e-29** where the true value
is zero. Nothing below is believable without that line, and it is the first thing the script
prints.

### The one thing the graph case does not have

The graph is connected, so `e_u − e_v` is orthogonal to `ker L` and `R` is always finite. The
sheaf's kernel is `H⁰`, and `dim C¹ = 3,764` against `dim C⁰ = 17,104` forces
**`dim H⁰ ≥ 13,340` by construction** — 78% of the node stalks. That is what private features
*are*, and it is exactly #150's objection: *"a map that is rank-deficient in a direction makes
that direction infinitely resistive, and the structural mask creates exactly such deficiency by
construction."*

So a `χ` with a component in `H⁰` has **no finite-energy current at all** and its resistance is
`+∞`. Every pair is therefore two numbers rather than one:

- **`leak`** — `‖χ − δᵀy‖² / ‖χ‖²`, the fraction of the demand no current can supply. The
  infinite part, and a statement about the **rank** of the maps.
- **`R`** — the resistance of the part that can be, rescaled as `R/(1−leak)` to a full unit of
  demand so it is in #150's units.

## 2. The directions are the channel's

[ADR-0022](../adr/0022-a-hop-is-an-operator-norm-along-a-learned-channel.md) governs the reading
and it is a standing ADR rather than a preference: an isotropic probe here would repeat the error
that cost this map a 1e14 phantom deficit. The channel between `u` and `v` is read off the
composed chain operator along the path, the composition `benchmarks/alignment_read.py` uses per
hop:

    C  =  F_{v,e_k}ᵀ · Π_i [ F_{c_i,e_{i+1}} F_{c_i,e_i}ᵀ ] · F_{u,e_1}

`a` is its top right singular vector, `b` its top left, signed so `C a = σ₁ b` with `σ₁ > 0`.
Gains are omitted deliberately: they are positive scalars per cell, so they scale `C` and cannot
move a singular vector.

**`C` is renormalised after every hop, and that is not cosmetic.** On a trained surface the raw
seven-hop product lands at `σ₁ ≈ 4.5e-17` — float64's noise floor against intermediates of order
1 — which would make *along the channel* an empty phrase. Scaling cannot move a singular vector,
so normalising each hop returns the same directions with the intermediates held at order 1. In
the event the two agree to **3.3e-16** even on the trained surface, because near-rank-1 products
preserve direction without cancellation; the normalisation is insurance and is pinned by a test.
`chain` yields the channel's **direction** and never its gain, which is measured rather than
chained (#214, and #142's cost).

Two baselines run beside it, per pair and never pooled: **isotropic**, drawn on the stalk
spheres, which is what the probe ADR-0022 rejects would report; and **public**, drawn inside
`span_e row(F_{e,v})`, the largest subspace not private by construction, which separates the
structural mask's cost from the learned rank-deficiency's.

## 3. The reading, untrained

Seed 0, 16 patch cells and every other rim stratum, against #150's graph-side column:

| from | n | hops | graph R | probe | R/(1−leak) | leak | ratio |
|---|---|---|---|---|---|---|---|
| patch | 16 | 7 | 1.8082 | **channel** | **9.654** | 0.273 | **5.34x** |
| patch | | | | public | 12.637 | 0.385 | 6.99x |
| patch | | | | isotropic | 10.660 | 0.819 | 5.90x |
| proprioceptive | 3 | 7 | 1.9336 | channel | 7.881 | 0.161 | 4.08x |
| touch | 3 | 7 | 1.9458 | channel | 5.361 | 0.132 | 2.75x |
| actuator | 1 | 7 | 1.1822 | channel | 5.729 | 0.147 | 4.85x |
| drive | 1 | 1 | 0.3499 | channel | 1.000 | 0.000 | 2.86x |

**A bounded constant, and the constant sits in the leaf edge rather than the climb.** #150's
finding was that rim-to-apex is 1.82 unit-resistance edges of which ~1.0 is the patch's own
irreducible leaf. Measured in each rim cell's own leaf edge, along that edge's own channel:

| from | leaf R | leaf leak | rim→apex R | **in leaf edges** | graph-side |
|---|---|---|---|---|---|
| patch | 5.222 | 0.009 | 9.654 | **1.80** | **1.82** |
| proprioceptive | 5.709 | 0.035 | 7.881 | 1.38 | 1.94 |
| touch | 6.239 | 0.069 | 5.361 | 0.83 | 1.95 |
| drive | 1.000 | 0.000 | 1.000 | 1.00 | 1.00 |

**1.80 against 1.82.** Untrained, the sheaf's climb is the graph's climb to within a percent. The
whole 5.3x factor is in the leaf edge — the one edge no rewiring removes. On this surface #230's
closure would have been confirmed from the sheaf side, and the answer would have been *"near its
floor"*.

## 4. The reading, trained — and it is a different answer

30,000 ticks of both rules, three seeds, read in float64. #178's hazard is why there are three.

| | patch→apex `R/(1−leak)` | ratio to graph R | leak (median) | climb in leaf edges |
|---|---|---|---|---|
| untrained | 9.654 | **5.3x** | 0.233 | **1.80** |
| taught 30k, seed 0 | 606,586 | **3.36e5x** | 0.322 | 12.36 |
| taught 30k, seed 1 | 493,384 | **2.77e5x** | 0.307 | 8.63 |
| taught 30k, seed 2 | 780,920 | **4.35e5x** | 0.166 | 6.47 |

Five orders of magnitude, on every seed. **This is the ticket's second branch and it is not
marginal.**

**The channel correction does not rescue it.** ADR-0022's directional reading is worth ~184x on
the hop. Here the channel beats the isotropic probe by only **2.1x–2.5x** (seed 0: 3.36e5 against
8.46e5). Reading along the channel is still the right reading — it is the most favourable
direction there is — but on resistance it buys a factor of two, not two orders.

**Per edge, and never as a graph-wide average** (seed 0, along each edge's own channel):

| edges | count | R median | min | max | leak median | graph `w_e R` |
|---|---|---|---|---|---|---|
| sensory (m=8) | 262 | 59,187 | 1.6 | 4,931,591 | 0.062 | 8.0000 |
| interior (m=4) | 409 | 399 | 0.25 | 374,536 | 0.0002 | 1.3836 |
| motor (m=8) | 3 | 56,224 | 2,339 | 197,328 | 0.006 | 3.3465 |
| drive (m=1) | 8 | 36,996 | 36,903 | 36,996 | 0.025 | 0.3347 |

The most resistive edges are the **sensory** ones, at composed `σ₁ ≈ 5e-4` — the two endpoints of
one edge carrying near-orthogonal directions. That is #233's composition finding (*"the direct
route through a relay is 127x weaker than an independent-maps model"*) seen from the resistance
side. It is worth recording plainly that these are **not** #214's binding edges, which were
`L6/core—L7/core` and `L1/vision—L2/vision`: the most resistive edge and the binding edge of a
measured path are different questions, and this document does not claim they agree.

### Why it moved: the maps went rank-1

The graph is identical between §3 and §4. The only thing that changed is the maps:

| | effective rank of the maps (participation ratio) | | |
|---|---|---|---|
| | median | min | max |
| untrained | **2.8535** | 1.0000 | 5.5787 |
| taught 30k | **1.0009** | 1.0000 | 4.8029 |

This is `docs/research/053`'s standing note — *nothing has ever measured whether today's maps
have drifted toward rank-1* — measured, and the answer is that they have, almost exactly. It is
consistent with ADR-0022's own cited 1.02–1.06 rather than a new claim. **The gauge is intact
throughout**: pinned maps at Frobenius exactly 1.0000, banded maps within `[0.6729, 2.0000]`
against `[1/ρ, ρ] = [0.5, 2]`. So this is not a scale collapse and ADR-0010 is doing what it says.
Rank is simply not something it constrains, which is the point.

### It is not a conditioning artifact

The trained `δδᵀ` has condition number **1.08e9**, so where the pseudo-inverse cuts matters and
was checked rather than assumed. On one patch→apex pair:

| `EIGEN_FLOOR` | directions kept | R | leak |
|---|---|---|---|
| 1e-8 | 3,677 / 3,764 | 191,934 | 0.169 |
| 1e-10 | 3,764 / 3,764 | 301,131 | 0.167 |
| **1e-12** | 3,764 / 3,764 | **301,131** | **0.167** |
| 1e-14 | 3,764 / 3,764 | 301,131 | 0.167 |

The reading is on a flat plateau from 1e-10 to 1e-14 and moves only at 1e-8, where the cut is
discarding 87 real directions. The untrained surface's condition number is 1.43e3 and its reading
is identical at every floor. `EIGEN_FLOOR = 1e-12` sits inside the plateau on both.

## 5. What the parallel routes are actually worth — the scope question, measured

#230's closure rests on a **topological** claim, and it is #150 §1's: rim-to-apex is seven hops
but only 1.82 unit-resistance edges, *"because each level is a lattice with many parallel routes
into the next"*, so the depth is already bought back and rewiring has nothing left to buy. That
claim was established graph-side. Whether it holds of the sheaf need not be argued — it is the
difference between the dome and a **spanning tree of the dome**, which keeps every shortest path
and deletes every alternative route, on the same maps and along the same directions. 406 of 682
edges span it.

The comparison uses the **regularised** resistance `χᵀ(L + εI)⁻¹χ` rather than `R`, and that
choice is load-bearing. Deleting an edge shrinks the *suppliable* demand, so plain `R` can
**fall** while transmission gets worse — the first version of this table reported the routes as
worth 0.80x for exactly that reason, which is an artifact and not a finding. The regularised form
is finite for every `χ` and monotone in the PSD order, so deleting edges can only raise it and the
ratio is a real comparison bounded below by 1. Woodbury on the small edge-side Gram makes it
cheap, and `tests/test_sheaf_resistance.py` pins both properties.

**Routes buy, tree / dome, patch → apex:**

| surface | ε=1e-6 | ε=1e-9 | ε=1e-12 |
|---|---|---|---|
| untrained | 1.46x | 1.46x | 1.46x |
| taught 30k, seed 0 | 1.21x | 1.39x | 1.39x |
| taught 30k, seed 1 | 1.32x | 1.49x | 1.52x |
| taught 30k, seed 2 | 1.38x | 1.53x | 1.57x |
| **graph-side, same pairs** | | | **3.80x** |

**The dome's parallel routes are worth 3.80x graph-side and 1.39x–1.57x sheaf-side.** They are
not worthless — the sheaf still gains from them — but they are worth **less than half** what the
graph-side number credits, and the reason is the same rank-1 structure: parallel routes carrying
different directions are not fungible, so a second route is not a second copy of the first.

**This is what keeps the closed family closed.** #150 ruled the topological remedies out because
they attack a term worth under 2x (1.82 reducible to at best ~1.0, the leaf being irreducible).
Sheaf-side that term is *smaller*, not larger: the routes already present buy 1.4x rather than
3.8x, so adding more of them buys less than the graph-side arithmetic promised. Widening the
funnel, paralleling the L2→L3 cut, relay cells, local virtual nodes and expander rewiring all
move a term that the sheaf discounts further. **Nothing in that family reopens.**

## 6. What this leaves

- **The reversal condition in the map's *Out of scope* entry is met on its number and not on its
  family.** The sheaf resistance *is* arbitrarily worse in the channel's direction — 3e5x. The
  remedy family that condition was written to reopen is nevertheless still closed, and §5 is why:
  the resistance is not large for a reason a rewiring reaches.
- **What reopens is rank**, and #237 named it in advance: *"the rank structure of the maps is the
  bottleneck, which is a different remedy space and touches ADR-0010's decision not to police
  rank."* Three things in the record are now live and were not: ADR-0010 bounds a Frobenius norm
  and says nothing about rank; `docs/research/053`'s standing note is discharged in the
  affirmative — the maps *have* drifted to rank-1, median 1.0009; and
  [#57](https://github.com/NGL321/patchworks/issues/57)'s concession that the budget does not
  police rank at all is now attached to a measured consequence.
- **Training is what does it.** Untrained maps sit at effective rank 2.85 and the sheaf's climb is
  the graph's to within a percent. The transport rule drives them to 1.0009. Whether that is the
  rule working — a cell's edges *should* carry different information, which is
  `Not yet specified`'s open question about the right amount of shared direction — or the rule
  destroying the graph's capacity to conduct, is not settled here and is not this ticket's to
  settle.
- **`leak` is the half nobody has priced.** A median of 0.17–0.32 of the channel demand is
  genuinely unsuppliable on a trained surface, against 0.23 untrained. That fraction is `+∞`
  resistance, not a large number, and no per-hop gain reaches it.

## What was not done

- **No remedy is proposed or priced.** This is a measurement ticket and the rank remedy space —
  policing rank in the gauge, an incoherence term with teeth, conditioned maps — is untouched.
- **The apex→rim direction was not read.** #214 states its predicate twice and this reads
  rim→apex only. The rank mechanism is not directional, so the expectation is that it is the same
  reading, but expectation is not measurement.
- **`H¹` was not computed.** The map's fog carries `H¹ = 0` as a precondition on a cell's `K`
  meaning anything, and `δ` is now assembled and factored, so the object is one eigendecomposition
  away. Nothing here needs it, and it belongs to whatever artifact first makes the strong claim.
- **The relationship to #214's binding edges is unresolved**, and §4 says so rather than
  smoothing it: the most resistive edges here are sensory, and #214's binding edges are not.
