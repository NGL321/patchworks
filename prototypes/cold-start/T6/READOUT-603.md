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

*(filled from `b40_table.py`)*

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

## 4. What this does to B34

*(filled once the arms land)*
