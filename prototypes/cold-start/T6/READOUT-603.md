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

**The reversibility floor is a live channel, and it transmits slightly harder than the
lanes the criterion kept.** That is not a subtle leak; it is the floored population
being, on this surface, the structurally central one.

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

## 4. B38's port works, and this is the first trained contrast that shows it

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
| B40 `control`, `world std_max` | 1.29 | **9.35e-01** | 4.12e-01 |

`arms.py` and `graph.py` both changed between B33's branch and `main`, through B38's
port (`b64b4db`) and #597's reserve mask (`5ecaf3e`), so `reserve_p12` is a
different surface than the one B33 read.

> **B38's port delivered what it was built for**, and the rows below are the first
> holonomy-term contrast on this map taken against a world that is still moving at
> the horizon it is read at. B33's own contrast accumulated almost entirely after
> tick 100 on a world four orders quieter; that re-indexing belongs to B33's
> successor, not to this ticket, and is flagged here rather than acted on.

---

## 5. What this does to B34

*(filled once the arms land)*
