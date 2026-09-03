# ADR-0030: The Koopman conversion buys a design variable, and the price is booked

**Status:** accepted

## Context

[#147](https://github.com/NGL321/patchworks/issues/147) asked what the conversion costs and what
measurement would show the cost was too high. Its resolution specified the answer as **an index ADR**
— six entries, each naming an owner, a falsification signature, and whether that signature is
readable on a running system — and said the ADR was to be authored in the conversion edit,
[#157](https://github.com/NGL321/patchworks/issues/157).

**#157 merged without it.** `docs/research/231-the-record-read-back.md` §4 found the gap: *"The
falsification register does not exist in the repo... Entry 1's fill lives on #147's thread. Owed
before #78 spends a budget."* [#217](https://github.com/NGL321/patchworks/issues/217) could not write
its item 3 — entry 1's fill — for the same reason, and left it on that ticket. This ADR is the
missing document, written by [#350](https://github.com/NGL321/patchworks/issues/350).

[ADR-0004](./0004-linear-restriction-maps-assume-local-flatness.md) is the model in the strict sense:
adopt a thing for what it buys, then record what it costs, with a test that could show the cost was
too high. ADR-0004 was itself written before its causes were known and has been amended three times,
each time becoming more useful rather than revealing it was written too early. **A register that
waits for certainty is a summary.**

## Decision

### The purchase, stated here because it is homeless elsewhere

Half of ADR-0004's shape had no home. The conversion's *why* lives only on
[#127](https://github.com/NGL321/patchworks/issues/127), which is an index that refuses to be a
store, and none of the four stage-1 tickets books it. So this ADR states the purchase as well as the
price.

**`body` is a term in the transmission budget, and before the conversion it was not a design
variable** — a frozen random MLP's Jacobian, unsettable, and the one factor learning moved *downward*
(`0.4529`, `×0.95` after 30k ticks). Under the conversion that factor is `σ_max(K)`: settable,
bounded by [ADR-0015](./0015-the-cell-operator-band-is-on-the-spectral-norm.md)'s band, and the same
object as the stability constraint. **One knob, two purposes.**

**It does not buy transmission.** The conversion is worth about `2.2×`, against a deficit the record
then put at ~1e14. Nothing in this register should be read as a transmission claim; what the
architecture is now read against is ADR-0026's rim-core influence predicate, not a per-hop number.

### The register is an index, and it re-argues nothing

Each entry names the **owner**, the **falsification signature**, and whether that signature is
**readable on a running system**. The argument lives in the owner. This is the same index discipline
the map runs on, and it is what stops the register going stale each time an owner is amended.

| # | cost | owner | live-readable |
|---|---|---|---|
| 1 | A linear chart cannot hold an either/or | [#151](https://github.com/NGL321/patchworks/issues/151), filled by [#217](https://github.com/NGL321/patchworks/issues/217) | **no — resolved, not a cost** |
| 2 | Open-loop rollout decays into the dominant subspace | [#144](https://github.com/NGL321/patchworks/issues/144) | no — a construction check |
| 3 | Folds retired as a mechanism | [ADR-0014](./0014-the-linear-readout-is-gauge-fixed.md) / [#138](https://github.com/NGL321/patchworks/issues/138) | **no — no signature at all** |
| 4 | A cell that amplifies instead of forgetting | [ADR-0015](./0015-the-cell-operator-band-is-on-the-spectral-norm.md) / [#140](https://github.com/NGL321/patchworks/issues/140) | yes |
| 5 | One global band spanning two regimes | [#146](https://github.com/NGL321/patchworks/issues/146), on [#149](https://github.com/NGL321/patchworks/issues/149)'s ground | yes |
| 6 | The frozen gauge's shared direction | [ADR-0014](./0014-the-linear-readout-is-gauge-fixed.md), signature from [#141](https://github.com/NGL321/patchworks/issues/141) | no — known at construction |

**Two entries moved after #147 specified the table, and both moved toward *fewer* live halts.** That
direction is worth noticing rather than smoothing: a register whose entries only ever grow teeth is
one nobody is checking.

**1 — the either/or. Filled, and it resolves as not a cost against the running architecture.** #147
booked this as the load-bearing entry, live-readable *"yes, in half"*, and held it open for #151.
#217's item 3 fills it: both Brunton et al. (2016)'s invariant-subspace theorem and Liu–Ozay–Sontag
(*Automatica* 2025) bind **immersions**, and this design asserts none — which is why
[ADR-0023](./0023-the-chart-is-not-a-koopman-lift.md)'s escape is re-sourced onto *no cell is
autonomous* and *no semiconjugacy is claimed*, and why ADR-0017 no longer claims a small linear lift.
**No new falsification signature is booked.** The entry points at entry 5's instrument (per-cell
prediction error split by contact state) and at [#166](https://github.com/NGL321/patchworks/issues/166)'s
history-scaling read, rather than double-booking one measurement into a third slot nothing distinct
can fail. #166 has since resolved: twelve dimensions are enough for both of the chart's jobs, and the
memory shortfall it suspected is the operator's **shape**, not its size — so the width reading this
entry pointed at has been taken and does not fire here.

**2 — rollout decay. Demoted from a halt to a construction check, and the capability that would
incur it is declined.** #147 already noted the horizon is computable per cell from the singular-value
gap, so knowable rather than discovered; [#156](https://github.com/NGL321/patchworks/issues/156) took
that to its conclusion — it is read once off `K` and compared against the loop the cell has to serve,
in the build, not in #78's stop condition. At a singular-value gap of `0.5` the horizon is `3.3`
ticks: serves the reflex, fails the visual loop, so cells carrying visual context need a gap above
roughly `0.8`. #144 then declined multi-horizon disagreement outright, on the cell contract rather
than on the arithmetic: `K^h z` is a cell running **without evidence**, which is exactly the trigger
ADR-0023's sharpened escape names, and no cell in this architecture is autonomous. **So the design
currently contains nothing that rolls open-loop.** The entry is kept, not struck: it is what would
have to be re-read if anything ever proposes one.

**3 — folds. A scope reduction with no test, and it says so.** What is written off is narrow:
nothing is *built* on folds any more. `bias_selection.py` keeps running as `K @ J_encode` and stays
region-dependent through `encode`; [#27](https://github.com/NGL321/patchworks/issues/27)'s spectra
study and [#42](https://github.com/NGL321/patchworks/issues/42)'s fold-margin rig keep their
measurements. Folds are **retained as a description of `encode` and retired as a mechanism.** The
entry carries **no falsification signature and states that plainly** — it is a bet you cannot lose
slowly, only regret. Recorded explicitly so no future reader hunts for an instrument that was never
built, and so the list does not grow a fake test in order to look complete.

**4 — amplification.** `σ_max(K)` outside its band is a cell that amplifies instead of forgetting.
ADR-0015 owns the mitigation: the band `[1/ρ_K, 1]`, projected **spectrally** rather than in
Frobenius norm so the body cannot go rank-deficient. The margin is a **designed** quantity, not a
slack one — transmission wants the upper face and stability wants below it. The live reading is the
**pre**-projection `σ_max`: post-projection it is never out of band by construction, so reading it
there says nothing, and the quantity that matters is how often learning has to be pulled back. #156
fixed the form — a dwell and a burn-in, both counts — and the burn-in is legitimate because #138
initialises `K = a·I`, so the fleet's first excursion is the selection rig settling rather than the
cost arriving.

**5 — two regimes, one global band. This cost did not exist when #147 was written.**
[#149](https://github.com/NGL321/patchworks/issues/149) killed the original mechanism outright: the
sandbox's contact is **compliant** (`solref="0.008 1"`, finite force, penetration scaling with
speed), the 50 Hz tick map across it is contracting, and the hybrid-systems objection never reached
us. The switching-dynamics/rSLDS framing is **struck**, along with the void evidence it rested on —
[#126](https://github.com/NGL321/patchworks/issues/126)'s pixel-tile ratios were never a result about
a predicting cell's piece. What replaces it is a consequence of ADR-0015: a single `K`, under a band
that is now **global**, must hold free motion and contact, whose natural rates differ. Signature:
per-cell prediction error split by contact state, read on
[`08-the-acceptance-demo.md`](../spec/08-the-acceptance-demo.md)'s IQR instrument, **plus** #156's
second gate — the gap must *widen* across consecutive windows. That gate carries the entry: a cell
may simply find contact harder, and that is not the cost. The cost is one band failing to hold two
rates, which gets worse. #146 has since ruled out the repair by bilinearity, for every cell rather
than for a boundary set, so this entry has no queued mitigation and is a live cost in the plain
sense.

**6 — the gauge. The one cohomological question the conversion creates.**
[#138](https://github.com/NGL321/patchworks/issues/138) froze `decode`, confining every cell's
predictions to one shared 12-dimensional `im(D)` inside a 32-dimensional stalk. Whether `H⁰` — the
configurations reconciliation cannot move — has useful overlap with it is open, and is the gauge's
pre-registered falsification. [#141](https://github.com/NGL321/patchworks/issues/141) made the
residue ADR-0004's **fourth static-floor cause** and gave it the cheapest signature in the set: a
direction **shared across unrelated edges**, where every other static-floor cause is per-edge or
per-level. Read first, and read forwards — known at construction, so not live. It is minted as open
problem [#325](https://github.com/NGL321/patchworks/issues/325), which currently sits in the open
problems register under *cutoffs naming a rig with no recorded run*: the signature exists, the
instrument exists, and nothing has ever reported against it. **That is this register's one
outstanding debt, and it is a debt of execution rather than of specification.**

### What #78 stops on

The register is a falsification list, and #78's stop condition is **drawn from** it rather than
identical to it. Several entries are read at construction, and a register admitting only mid-run
signals would have to drop them.

The split follows the seam between what the costs are and what the budget is:

- **The identity of the live entries is a property of the costs**, and lives here. As the entries
  stand today that is **entries 4 and 5, and nothing else.** Entry 2 is a construction check to be
  read before the run; entry 6 is read at construction and must be read *first*, because it is the
  cheapest exclusion in ADR-0004's procedure; entries 1 and 3 carry no live signal, one because it
  resolved and one because it never had one.
- **The thresholds are not**, and they are not here. A window, a slope and a floor depend on the
  compute budget and on a graph that transmits. #156 fixed their forms and found that three of the
  four entries it examined need no threshold at all; the only inputs still owed by a run are the
  topology-only baseline and the noise level, both **measured** rather than chosen, which is what
  keeps that a pre-registration rather than a knob.

So this ADR carries the list and #156 carries the thresholds. The ADR does not go stale when a budget
changes, and #78 does not re-derive which signals matter.

### One standing pre-registration, on any claim made *for* the conversion

`docs/research/148-local-linear-operator-citations.md` (R8) records that the Koopman literature is
composed almost entirely of success reports — no multi-lab reproduction, no benchmark paper reporting
a loss — so the absence of failure reports is a fact about publication and not evidence. Two findings
bear on this architecture directly: latent rollout of a learned linear operator drifts and is
routinely fixed by re-encoding (arXiv:2310.15386), and a baseline that copies the most similar
segment of its own context beats leading time-series models on low-dimensional chaos
(arXiv:2505.11349).

**The operational consequence is a requirement, not a finding:** any claim that a learned `K` beats
the frozen random `step` it replaced must be reported alongside a **copy-the-nearest-past-segment**
baseline and a **plain linear-AR** baseline, or it will not be believable. Booked here because the
register is where a claim about the conversion's worth comes to be checked.

### What is not a cost

Kept from #147, because the section exists to stop the list growing by rumour. Two of its lines were
corrected there and the corrections stand:

- ~~"The shared frozen body survives — two of three maps, both wide ones."~~ **Wrong twice after
  #138.** The edit touches **two** of the body's three maps; **one** survives — `encode`, now the
  body's sole nonlinearity. The bet the shared frozen body *is* remains untested on its own terms;
  that is [#98](https://github.com/NGL321/patchworks/issues/98) and
  [`01-cell-and-sheaf.md`](../spec/01-cell-and-sheaf.md)'s *Known exposure*, not this register.
- ~~"`n = 32` and every construction diagnostic built on it survive."~~ `n = 32` survives; the chart
  dimension was contested, and [#145](https://github.com/NGL321/patchworks/issues/145) settled it —
  there is no lift, so no `k_lift`, and #166 then found `k = 12` is not binding on either of the
  chart's jobs.

Unchanged and still load-bearing: batching survives and the conversion is **cheaper** than what it
replaces; the locality guarantee is untouched; the taper's cut capacities are set by `m`, not `k`,
and do not move; **no reward enters anywhere** — the drive is unchanged in kind.

## Consequences

**The register indexes; it does not adjudicate.** An entry whose owner is amended does not need this
file rewritten — only an entry whose *live-readable* column changes does. Two have changed since
#147, both toward fewer live halts, and both changes were made by the owner rather than here.

**#78 is handed four things and no numbers.** The identity of the live entries (4 and 5), the two
construction reads it must take before spending (entry 2's singular-value gap, entry 6's shared
direction), the forms of the halts (#156), and the baseline requirement above. It is not handed a
threshold by this ADR, deliberately.

**Entry 3 is permanently untestable and that is recorded rather than repaired.** Anyone who wants
this register to be uniform will be tempted to invent an instrument for it. There is none, and a
fabricated one would be the worst outcome available: it would make the list look complete while
measuring nothing.

**The register's one execution debt is entry 6.** Its rig exists and has never reported (#325). A
cost with an instrument nobody runs is indistinguishable, from inside, from a cost with no
instrument — which is precisely what the *cutoffs naming a rig with no recorded run* section of the
open problems register exists to make visible.
