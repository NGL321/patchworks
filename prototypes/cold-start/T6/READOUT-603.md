# B40 (#603): which training route buys earned cycle closure, and does the flat bundle end the criterion?

Prototype for [#603](https://github.com/NGL321/patchworks/issues/603), on map
[#532](https://github.com/NGL321/patchworks/issues/532). Surface: `reserve_p12` —
plain `build_graph(DomeSpec())` on B38's port ([#600](https://github.com/NGL321/patchworks/pull/600)),
`n = 32`, 194 wide interior edges, capacity budget 63 per predicting cell.

Everything below is produced by the instruments in this directory:
`b40_census.py`, `b40_enumerate.py`, `b40_stranded.py`, `b40_criterion.py`,
`b40_routes.py`, `b40_table.py`. Nothing is retyped by hand.

---

## 1. The named risk, and the larger one underneath it

The ticket named one place B34's proposal could break on arithmetic: splitting 40
local cycles into train and held-out may leave too few *per edge* to read a
principal-angle spectrum against. **That risk does not fire. A different one does.**

Read off B29's wide cycle **basis** (45 cycles), the picture looks fatal: 90 of 194
wide interior edges lie on no cycle at all, the median covered edge carries one
cycle, and a 20/20 split leaves 29 of 97 covered edges with an empty held-out side.

But a cycle basis is a BFS artifact, and B34's criterion is not defined on a basis —
it counts over **the** short local cycles through `e`, all of them. Enumerating them
directly (`b40_enumerate.py`, every cycle of length ≤ 6, radius-2 locality
unchanged) gives **190 distinct short local cycles** and a very different per-edge
picture:

| | basis (45 cycles) | enumerated (190 cycles) |
|---|---|---|
| edges on no cycle | 90 | **90** |
| median cycles/edge | 1 | **3** |
| q3 | 1 | **8** |
| max | 13 | **25** |
| edges with ≥ 2 | — | **104** |
| edges with ≥ 4 | — | **80** |

**So the split is arithmetically fine.** 104 of 194 edges carry two or more local
cycles and 80 carry four or more; arm 1 is buildable and the ticket's named risk is
answered in the negative.

**The 90 are not an artifact, and not a locality effect.** The same 90 edges are
cycle-less under exhaustive enumeration as under the basis. `b40_stranded.py` runs
Tarjan on the wide subgraph: **all 90 are true bridges** — they lie on no cycle at
*any* length, and 0 of them are stranded by the radius-2 bound.

> **This is the finding that outlives the arm comparison.** On 90 of 194 wide
> interior edges — 46% of the population B34's criterion exists to carve — the
> criterion counts over an empty set and floors to `m_e = 1` **whatever the training
> route does**. No route rescues them, because there is no cycle to close. ADR-0011's
> locality bound is exonerated: widening the radius changes nothing.
> **B34 needs a second rule for acyclic edges, and does not have one.**

---

## 2. The arms and the scores

*(table filled from `b40_table.py`)*

### The fifth arm, which the ticket did not ask for and the comparison needs

Four arms is a floor, not a design — the ticket says so. One more was required to
make arm 4 readable at all.

`flat` installs orthonormal frames onto a surface whose own initialisation sits at
`sigma_max ≈ 1.6e-07`. Any advantage it then shows could be **isometry**, not
flatness — the flat bundle would be winning because its maps are well-scaled, and
that would have nothing to do with cycle consistency. So `haar` installs
`holonomy_read.flat_maps`: the same block structure, exactly isometric, drawn
**independently** per endpoint, with no shared per-cell frame. Same scale as `flat`;
cycle-consistency absent. What separates the two is flatness alone.

It answers cleanly. `haar` reads `identification` **1.0088 → 0.9810** — at chance
throughout — and `sigma_max` **1.10e-07**, the same order as the untouched control
and four orders below `flat`'s **7.72e-04**.

> **The flat bundle's advantage is not scale and not isometry.** Independent
> isometries of identical shape close nothing and carry nothing. Only the *shared
> per-cell frame* buys closure — and with it about **7,000×** the composed amplitude
> that independent isometries lose across seven hops. That loss is
> [#533](https://github.com/NGL321/patchworks/issues/533)'s principal-angle collapse,
> and the flat bundle defeats it **by construction rather than by training**.

---

## 3. Score 4: the probe lanes leak

B34 floors every edge at `m_e = 1` so a pruned edge stays measurable and can grow
back. The user did not accept that as free:

> *"I am concerned about transmission across functionally closed directions which are
> kept open to allow later access. So we need to check it."*

Checked. On the flat bundle at construction, the 90 floored edges — which are exactly
the 90 bridges — carry median lane gain **0.5774** against the kept edges' **0.5164**:
a ratio of **1.118**, with **88.9%** of floored edges transmitting *above* the kept
median. The reading is stable across every checkpoint and every arm that reaches it.

**The reversibility floor is a live channel, and it transmits harder than the lanes
the criterion kept.**

**And the leak is structural, not learned.** The numbers do not move — at every
checkpoint of both seeds, the floored median is `1/√3` and the kept median is
`1/√3.75`, to seven decimal places:

| | floored median | kept median | ratio |
|---|---|---|---|
| observed, all checkpoints, seeds 42 and 43 | 0.5773502 | 0.5163978 | 1.118034 |
| closed form | `1/√3` = 0.5773503 | `1/√3.75` = 0.5163978 | `√1.25` = 1.118034 |

Which gives the mechanism. Under ADR-0032's band a restriction map is a near-isometry,
so its norm is spread across the `m` directions the lane carries. **At `m = 1` all of
it sits on one direction.**

> **The floor does not merely fail to be inert — it leaks *because* it is a floor.**
> Narrowing an edge concentrates the band's norm into fewer directions, so the single
> direction a pruned edge keeps for reversibility is the strongest-transmitting
> direction that edge has. Pruning harder makes the probe lane leak harder, by
> construction, and no training schedule touches it. This is the answer to the user's
> question, and it is worse than the question supposed.

---

## 3a. Three of B34's four clauses are inert

B34's criterion has four parts: the **count**, the **floor** at 1, the **cap** by the
less convinced endpoint, and the **ration** by `Σ_e m_e ≤ B`. On this surface, at
every checkpoint of the only arm where the count is ever above 1:

| ticks | raw total | allocated total | delta | raw max | allocated max |
|---|---|---|---|---|---|
| 0 | 1323 | 1413 | **+90** | 15 | 15 |
| 50 | 1049 | 1139 | **+90** | 13 | 13 |
| 150 | 1018 | 1108 | **+90** | 13 | 13 |
| 500 | 992 | 1082 | **+90** | 13 | 13 |

The delta is exactly 90 every time — the floor lifting the 90 bridges from 0 to 1,
and nothing else. `raw max` equals `allocated max` throughout, so **neither the cap
nor the ration ever clipped a single edge.**

Nor could the cap: `k_v` is 20 at every predicting cell and the largest count
observed anywhere is 15. The ration could in principle bind — the wide subgraph's
max degree is 5, and 5 × 15 = 75 against a budget of 63 — but it never did, because
no degree-5 cell ever had all five lanes at maximum at once.

> **Only the floor does any work, and the floor is the leaking probe lane.** The
> clause B34 added so that a pruned edge stays measurable is simultaneously the
> only clause with an effect and the one carrying the transmission the user asked to
> have checked. The cap and the ration are, on this surface, decoration.

---

## 3b. Arm 1 and arm 3 are not comparable, and the ticket's design hid it

`split` reads `m_e` max 4, std 0.58 at **tick 0** where `control` reads all 1 — same
seed, same untrained maps, nothing trained. The only difference is that `split` reads
the criterion on the held-out **half** of the cycles.

That is not a bug in the arm; it is a property of B34's count. The count is an
intersection over the cycles through an edge — a direction is warranted only if it
clears threshold on all of them — so **adding a cycle can only shrink the warranted
set.** Reading on half the cycles therefore cannot report a narrower edge, and
sometimes reports a wider one. Measured, per edge, on one surface at a time:

| surface | held-out wider | narrower | equal | mean inflation |
|---|---|---|---|---|
| untrained | 16 | **0** | 88 | **+0.327** |
| flat bundle | 11 | 7 | 86 | +0.115 |

On the flat bundle the effect is noise in both directions, because the cycles agree
with each other and the intersection does not shrink. **On a surface that is not
already flat it is a strict one-way bias** — and every surface a training route
actually produces is of the second kind.

> **Arm 1 differs from arm 3 in two ways at once**, and only one was intended: which
> cycles the term descends on, *and* how many cycles the criterion is read over. The
> second inflates arm 1's score-3 numbers on exactly the surfaces where the
> comparison matters. **Any future held-out design must read the criterion over a
> cycle count matched to the control's**, or it is measuring its own sample size.

---

## 4. B38's port moves the stall later; it does not remove it

**The control reproduces B33 closely.** Same seed, and despite descending on 190
enumerated cycles where B33 descended on 40 basis-derived ones:

| ticks | B33 `holo` | B40 `control` |
|---|---|---|
| 0 | 1.0017 | 1.0017 |
| 50 | 0.9827 | 0.9836 |
| 150 | 0.9754 | 0.9776 |

So the descent set's size barely matters, and the instrument is sound.

**What does not reproduce is the stall — and that is the point.** B33's own
`motion.read()` stamps have its world collapsing at tick 150 and never recovering;
that stall is not a discovery here, it is precisely the finding B33 reported and
[B38 (#599)](https://github.com/NGL321/patchworks/issues/599) was opened to fix.
Set the two side by side:

| ticks | 50 | 150 | 500 |
|---|---|---|---|
| B33 `holo`, `world std_max` | 1.29 | **5.7e-04** | 1.9e-04 |
| B40 `control`, `world std_max` | 1.29 | **9.35e-01** | **8.63e-05** |
| B40 `flat`, `world std_max` | 1.09 | 1.10 | **4.12e-01** |

`arms.py` and `graph.py` both changed between B33's branch and `main`, through B38's
port (`b64b4db`) and #597's reserve mask (`5ecaf3e`), so `reserve_p12` is a
different surface than the one B33 read.

> **The port moves the stall later; it does not remove it.** At tick 150 the control's
> world is alive at 0.935 where B33's had already collapsed to 5.7e-04 — so the
> reading B33 could not take *is* now takeable, and score 1's separation between the
> arms at 150 is a live one. But by 500 the control has stalled too (8.63e-05), while
> `flat` is still moving (0.412). The stall is **per run**, which is
> [B38](https://github.com/NGL321/patchworks/issues/599)'s own finding (13× across
> seeds of one arm) rather than a new one, and it is why every row here carries its
> own stamp and none is inherited.
>
> Consequence for this readout: **the 150-tick rows are the ones that compare arms on
> a live world; the 500-tick rows compare `flat` on a live world against trained arms
> on a dying one.** Both are reported. Neither conclusion below rests on a 500-tick
> trained row, because at 150 and at 500 the trained arms read the same thing —
> nothing.

---

## 4a. Why no route wins: the count never leaves the floor

This is the answer to the ticket's question, and it is not a ranking.

On the trained surface the criterion's **raw** count — before the floor is applied —
is zero on every one of the 194 wide edges, at every checkpoint, at the working
threshold. Including the 104 edges that *do* have cycles. Lowering the threshold does
not rescue it:

| ticks | th 0.8 total / max | th 0.9 | th 0.95 |
|---|---|---|---|
| 0 | 1 / 1 | 0 / 0 | 0 / 0 |
| 50 | 3 / 1 | 0 / 0 | 0 / 0 |
| 150 | 4 / 1 | 0 / 0 | 0 / 0 |
| 500 | 12 / 1 | 0 / 0 | 0 / 0 |

At threshold 0.8, across 194 edges and 500 ticks, at most **12 edges** ever acquire a
single warranted direction and **no edge ever acquires a second** — `max` is 1
throughout.

> **B34's criterion, applied to the surface any of these training routes produces,
> allocates `m_e = 1` to every wide edge.** That is the degenerate outcome B34's own §4
> feared — *"a prune criterion that rewards flatness, applied without a floor, prunes
> toward a tree"* — arriving with the floor in place. The floor does not prevent it.
> **The floor is it.**

So the arms are indistinguishable, and **not because they are equally good**. They are
indistinguishable because none of them moves the quantity the comparison was supposed
to rank them on. The 1 → 12 drift at threshold 0.8 is the only sign of life in any
trained arm, and what it says is that the horizon is orders of magnitude too short —
not that held-out cycles, phasing and the joint form differ.

**No preference was needed, and none was exercised.** The ticket asked for a choice
driven by results; the results decline to offer one.

---

## 4b. The flat bundle is *exactly* flat, and the privacy reserve is what breaks it

This is the strongest thing this prototype found, and it was not on the ticket.

The flat bundle as installed reads `identification` 0.0386 and `sigma_max` 7.7e-04 —
already four orders of amplitude above anything else on this map. But that is the
reading **after** `project()`. Before it:

| configuration | `identification` | `channel_return` | `sigma_max` |
|---|---|---|---|
| frames only, nothing applied | **0.0000** | **1.0000** | **1.000** |
| dimension mask only (`k_v = 20`, `n = 32`) | **0.6123** | 0.8645 | 2.91e-01 |
| mask + ADR-0032's band (`project()`) | 0.0386 | 0.9998 | 7.72e-04 |

**Exact path-independence, at unit gain, on every wide cycle.** Not near — `0.0000`
and `1.000`.

And it holds at `m_e < n`, which corrects an argument made earlier in this very
prototype. `install_flat_bundle`'s docstring reasoned that the cell frame cancels out
of `F_out F_inᵀ` so the construction should need `m_e = n`. That is wrong: because
every edge at a cell takes *rows of the same frame*, a hop is the top-left `m_out ×
m_in` block of `R_c R_cᵀ = I` — a rectangular identity — so the cycle telescopes
exactly at any widths. The per-hop deviation from a square identity is large (median
0.72 over 148 hops) and irrelevant; what telescopes is the product.

**What breaks it is the dimension mask, not the band and not the widths.** Zeroing
columns beyond `k_v` truncates each row of an orthogonal frame to its first 20 of 32
entries, and truncated rows are no longer orthonormal, so the telescoping fails —
`identification` 0.0000 → 0.6123. ADR-0032's band then *repairs* most of that
(0.6123 → 0.0386) while costing three orders of amplitude (0.291 → 7.7e-04).

> **Exact path-independence at unit gain is available by construction on this surface,
> and the commitment standing in its way is the privacy reserve `p` — measured here as
> `k_v = 20` against `n = 32`.** ADR-0032 is not the obstacle; on these maps its band
> is repairing damage the mask did. This is the same object
> [B34](https://github.com/NGL321/patchworks/issues/593)'s own resolution pointed at
> when it corrected the record — #320's *"surviving rejection ground is about the
> dimension mask, not `Edge.m`"*.
>
> Note the scope: this is path-independence in B27's retained **coherent-region**
> form, on local cycles. It says nothing about path-independence *at distance*, which
> B27 struck as unavailable, and must not be read as walking that back.

---

## 5. What the flat bundle costs

B25 carried the flat bundle with `O(Bn²)` attached as its price. On this surface that
price is negative:

| | parameters |
|---|---|
| per-edge, `Σ_{e,side} m_e · n` over 1364 endpoints | **247,104** |
| flat bundle, one `n × n` frame per predicting cell (150 × 32²) | **153,600** |
| ratio | **0.62×** |

**The flat bundle is 38% cheaper than what is there now**, so `O(Bn²)` is not an
argument against it here — `n = 32` is small enough that a dense frame per cell costs
less than 1364 sparse blocks.

It is also **more local, not less**. An edge map at `(e, c)` is `S_e R_c`, determined
entirely by the frame of the one cell it is incident on — where B34's criterion needs
cycles out to radius 2. Whatever ADR-0011 demands of a prune decision, a per-cell
frame satisfies it more easily than the criterion it would replace.

**The real price is freedom.** At a degree-9 cell, nine independently learned `m × n`
blocks become nine fixed slices of one frame. The per-edge map stops being an
independent object; what is learned is one orthogonal frame per cell, and the edges
read off it. That is a genuine architectural commitment and it is the thing a
decision has to weigh — not the parameter count, which favours it.

---

## 6. What this does to B34

*(filled once the arms land)*
