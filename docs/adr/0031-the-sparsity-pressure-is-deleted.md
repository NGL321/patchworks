# ADR-0031: The sparsity pressure is deleted from the transport rule

**Status:** accepted

## Context

The transport rule descended two terms: relative disagreement, and an **L1 on the normalised
restriction map** scaled by a global `λ` that annealed up from zero over a fixed horizon. The penalty
was specified in `01-cell-and-sheaf.md` as the local-neuroplasticity analogue — *pruning within what
structure permits* — and `06-graph-topology.md` carried its mechanism: because the map's magnitude is
gauge-fixed ([ADR-0010](./0010-restriction-map-scale-is-gauge-fixed.md)) the pressure could only move
weight *between* a map's directions, and the learned rank-deficiency that produced was said to enlarge
`H⁰`, *"the effect `05-timescales.md` actually wanted from sparsity"*.

[#89](https://github.com/NGL321/patchworks/issues/89) left the constant's value open and
[#324](https://github.com/NGL321/patchworks/issues/324) recorded a suspected drainage.
[#393](https://github.com/NGL321/patchworks/issues/393) swept `λ` and measured what the term buys:
**+1.3%** of `dim H⁰` at the shipped `0.4`, and **exactly zero** below `λ = 0.3`, while the fleet's
effective rank falls from **2.913** at `λ = 0` to **1.002** at the default over 30k ticks.

[#406](https://github.com/NGL321/patchworks/issues/406) was opened to retune the constant. Its
resolution declined the retune as the wrong question and is where the full argument lives; this ADR
records the decision and its grounds, and is deliberately not a second copy of that argument.

## Decision

**The term is deleted from the transport rule, and `H⁰` is a per-cell floor rather than a graph-wide
maximand.** Five grounds, and no one of them is doing the work alone:

1. **The stated warrant is measured false.** +1.3% of `H⁰`, and exactly zero below `λ = 0.3` (#393).
2. **Its purpose was superseded before it was measured.** [#143](https://github.com/NGL321/patchworks/issues/143)
   and [ADR-0028](./0028-a-cell-holds-a-spectrum-of-retention-constants.md) moved the *holding* of slow
   content onto `K`. What `H⁰` still does is **insulate** what `K` holds from reconciliation, which is
   a per-cell floor — enough private width at each cell for retention to survive and be readable — not
   a quantity to grow. The floor itself is [#385](https://github.com/NGL321/patchworks/issues/385)'s.
3. **It is an anti-isometry pressure.** The sheaf exists to carry related features of different cells
   into a common representation, and the hope of holonomy is that each incoming stalk transports
   **isometrically**. That is a demand for a **flat singular spectrum** on the restriction maps. An L1
   on the normalised map has its strict optimum at a single nonzero entry
   ([#356](https://github.com/NGL321/patchworks/issues/356)): it is a maximally *concentrating*
   pressure on exactly the spectrum the design needs flat, and it moves the maps that way
   monotonically in `λ`.
4. **It trivialises the holonomy the sheaf exists to carry.** Rank-1 maps compose to rank ≤ 1 around
   any cycle, so at #393's endpoint the holonomy collapses to scalars — **abelian**, the negation of
   [`docs/motivating-image.md`](../motivating-image.md)'s derivation of the project's central choice.
5. **The job it was meant to do belongs to the mask.** The intent was **selection** — find the minimal
   set of features that overlap. The term implements **concentration**, and `01-cell-and-sheaf.md` had
   already assigned selection elsewhere: *the mask selects — structurally, once, closing and never
   re-opening — and the map compresses, densely and by learning*. For pruning to select, masks would
   have to **open**, which is structural growth and out of scope.

**Deleted, not zeroed.** `λ = 0` and *no term* are not the same edit. A term multiplied by zero is a
constant with no warrant still wearing a value, which `docs/registers/rig.md` refuses to admit.
Deleting takes `DEFAULT_ANNEAL_HORIZON` with it and shrinks *Permitted global signals* from **exactly
two to exactly one**. And `0` is derivable where `0.4` was not, which satisfies
[#127](https://github.com/NGL321/patchworks/issues/127)'s standing note on invented constants rather
than waiving it.

## The reusable lesson: a warrant can be superseded before it is falsified

Worth stating plainly, because the record has no other instance of it written down and the register
disciplines are built to catch a different failure. The constants register admits a constant that has
a **warrant**, and this one had a warrant of the accepted shape — `DEFAULT_SPARSITY_PRESSURE` was
typed `selected` with a *magnitude* justification: at 0.4 the pressure gradient is a median 0.12 of
the transport term's. That row was true when written and stayed true. What it never stated was a
**purpose**, and the purpose it inherited from the spec — enlarging `H⁰` — had been reassigned to `K`
at #143 while the constant sat unchanged and fully documented.

So the term was measured before it was questioned, and the measurement (#393) was taken against a
warrant that no part of the architecture still needed. A magnitude warrant answers *is this the right
size*, never *is there anything for this to do*, and nothing in the pipeline asks the second question
when an adjacent decision moves. **The failure mode is a constant whose provenance is impeccable and
whose job has quietly moved elsewhere** — invisible to the register by construction, because the
register projects the definition site, and the definition site is exactly where the stale purpose is
not written.

## Consequences

**The transport rule's objective is disagreement, whole.** One term, one descent step, and no penalty
trading against transport. `src/patchworks/learning.py` loses `DEFAULT_SPARSITY_PRESSURE`,
`DEFAULT_ANNEAL_HORIZON`, `SparsityAnneal` and `normalised_l1`, and `TransportRule` loses its
open-weight array and its step counter — it now holds **no state whatsoever**, the same sentence
`PredictionRule` carries.

**Exactly one permitted global signal.** The anneal was the only non-learning-rate broadcast, so
`07-local-learning-rule.md`'s count moves from two to one and ADR-0008's two-signal clause is narrowed
to one. The locality story is simpler rather than differently qualified.

**#89's anneal question dissolves rather than resolves.** No value is owed for the horizon or the
ramp; the *direction* #89 escalated has no subject once the term is gone. Recorded here and in `07` so
no session goes looking for a number.

**The gauge projection is untouched.** ADR-0010's band is a separate step, applied after the descent
and outside the transform, and it stays exactly as it was. What ADR-0010 loses is one supporting
paragraph — *the sparsity term gains a rationale it did not have* — and none of its decision rests on
it: the scale-blindness argument for gauge-fixing runs on the relative objective alone, which is what
`01-cell-and-sheaf.md` now says.

**The intent behind the term is preserved and is not refuted.** #127's *Not yet specified* carries
*"An undifferentiated graph, carved by sparsity rather than designed"*, rejected for this build with
openable masks named as its reactivation condition. Nothing here touches that. What was wrong was the
implementation for **this** build, not the intuition.

**Deleting the collapser does not buy isometry, and this ADR does not claim it.** At `λ = 0` the fleet
holds effective rank **2.913** against a mask ceiling of 4 interior and 8 boundary — most of what the
interior mask permits, and nowhere near the flat spectrum ground 3 asks for. This decision stops the
architecture spending what the sheaf wants; it does not purchase it.
[#411](https://github.com/NGL321/patchworks/issues/411) owns what the maps should be learning instead,
and reaches ADR-0010's band (which fixes Frobenius magnitude and says nothing about spread),
[#396](https://github.com/NGL321/patchworks/issues/396)'s holonomy route and
[#315](https://github.com/NGL321/patchworks/issues/315)'s cycle reading.

**Re-adding a pressure later needs a fresh argument, and it is not this one revived.** What ground 3
points at is a **spectral floor** — a different constraint of a different shape, and
[#394](https://github.com/NGL321/patchworks/issues/394) already found it the one remedy family
surviving [ADR-0011](./0011-the-locality-guarantee-is-enforced-not-inherited.md)'s locality.

**Adjacent objects amended rather than ruled.**
[ADR-0009](./0009-a-drive-is-a-motor-edge-attached-deep.md)'s width ladder gated rungs 1 and 3 on
`λ = DEFAULT_SPARSITY_PRESSURE`; that constant no longer exists, and the re-read that ADR
pre-specified is now due against the two ceilings it named.
[#324](https://github.com/NGL321/patchworks/issues/324) stays **`uncut`**: its bar is
`draining_effective_rank < 2` and the reading at `λ = 0` is 2.913, but that is a 30k read of a claim
about drainage **over time**, and [#178](https://github.com/NGL321/patchworks/issues/178) has cost
this map the 30k-horizon mistake three times already.
[#395](https://github.com/NGL321/patchworks/issues/395) is re-pointed rather than cut: the budget it
names is real, but the dial is `m_e` at construction, not `λ` at runtime.
