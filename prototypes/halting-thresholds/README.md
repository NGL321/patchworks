# PROTOTYPE — halting thresholds for the register's live-readable costs

**Throwaway.** Built to answer [issue #156](https://github.com/NGL321/patchworks/issues/156):
the falsification register ([#147](https://github.com/NGL321/patchworks/issues/147)) gives four
live-readable costs a signature each and no threshold. What window, what slope, and what floor turn
each signature into a halt?

## Run

```bash
python report.py     # numpy only
```

## What it is, and what it is not

The ticket asks for "a cheap rig over recorded error curves" and the judgement *would this have
stopped the run at the right tick*. **There are no recorded error curves.** The conversion edit is
unwritten — no `K` and no `sigma_max` anywhere in `src/` — and the transmission edit
([#155](https://github.com/NGL321/patchworks/issues/155)) is open, so no run has produced the
quantities these rules read. See *The blocker was mis-wired*, below.

So the curves here are **generated, with the cause chosen in advance**, and the question asked of
each rule is the discrimination question rather than the calibration one: *given a residual we
built out of a known cause, does the rule fire for that cause and not for the other three?* That
question does not need a real run, and it turns out to be where nearly all the difficulty was.

Nothing in this directory is a measurement of Patchworks.

## The finding, in one line

**Three of the four entries need no threshold at all, and the fourth needs one constant that is a
ratio of the signal to itself.** Not one live number had to be invented.

`08-the-acceptance-demo.md` sets the discipline this follows — it refuses to pre-register a ratio
threshold because that "would be a number invented before anything was trained", and pre-registers
**orderings** instead. Every rule below is written so its load-bearing constant is a **count** (a
window, a dwell, a budget), a **self-ratio** (before a hold against after it), or a level **already
fixed by an earlier ticket and merely cited**.

| entry | what it needs | kind of constant |
|---|---|---|
| 1 — the either/or ([#151](https://github.com/NGL321/patchworks/issues/151)) | a hold, a budget, a baseline | self-ratio + count |
| 2 — rollout decay ([#144](https://github.com/NGL321/patchworks/issues/144)) | **nothing; it is a construction check** | — |
| 4 — amplification ([#140](https://github.com/NGL321/patchworks/issues/140)) | a dwell and a burn-in | counts |
| 5 — two regimes ([#146](https://github.com/NGL321/patchworks/issues/146)) | `08`'s IQR instrument, reused | ordering |

## Entry 1, which is the one #78 would actually be stopped by

### The question dissolves

The ticket asks it sharply: *say explicitly how long a run must go before "does not fall with
learning" is a claim rather than an impatience.* **It is not a length.**

ADR-0004 already carries the answer and it is an intervention, not patience: hold the world still
and sweep configurations — **lag drains, curvature does not**. You never have to wait long enough to
be sure a residual is irreducible. You interrupt and look. That converts an open-ended waiting game
into a bounded test runnable at any tick, and it is the whole reason entry 1 is affordable.

So the slope only **nominates** and the hold **decides**. A false nomination costs one hold, not a
run, which is why the nomination gate can be loose.

### Four gates, in cost order

1. **Construction.** #49's box-counting criterion and #141's shared-`colspan(D)` direction each
   produce this same signature and are each known *before* the run. An edge carrying either has a
   booked cause that is not entry 1, and is ruled out of this halt for free.
2. **Baseline.** ADR-0004 names **two** comparisons and the hold is only the second: the residual
   must first clear `06-graph-topology.md`'s topology-only baseline. That is a *measured*
   construction quantity, which is precisely how this rule gets a reference level without inventing
   one — and it is not optional. Without it the hold cheerfully halts a converged, healthy cell.
3. **Nomination.** A window in which the error did not fall by `min_decline`.
4. **Confirmation.** A quiescent hold; halt only if `survives` of the level is still standing.

### The budget *is* the threshold

`min_decline` is not chosen. An exponential with time constant `tau` declines by a known amount over
a window, so fixing the slope is identically the statement **"this run will wait for learning with a
time constant up to `tau_max`, and no longer."** `honoured_decline()` inverts it.

That relocates the constant to where [#147](https://github.com/NGL321/patchworks/issues/147) said it
lived all along — *the compute budget* — which is knowable **now**, without a run. Past `tau_max`,
"irreducible" and "slower than we will ever wait" are the same claim, and the register's entry is
honestly a claim about *this* budget.

The one genuinely scale-bound quantity, the noise level, enters only as a **feasibility check** on a
constant already chosen (`detectable()`). If a real run's noise makes the pre-registered `tau_max`
unreadable, the honest response is to shorten `tau_max` or lengthen the window *before* the run — not
to move a threshold during it.

### Results

One seed, and then 200:

```
regime             should  naive          hold-confirmed
learning           run on  ok             ok
curvature          halt    ok             ok
lag-floor          run on  WRONG          ok
slow-affordable    run on  ok             ok
slow-unaffordable  halt    ok             ok
self-intersection  run on  WRONG          ok
gauge-direction    run on  WRONG          ok

over 200 seeds x 7 regimes = 1400 judgements
  naive wrong:            600  (42.9%)
  hold-confirmed wrong:    14  (1.0%)
```

The naive window/slope rule's three failures are **exactly the three causes that are not entry 1**.
It cannot separate them, and `entry_one_sweep()` shows no window length rescues it: a window long
enough to clear slow learning has already spent the run, and no window separates lag from curvature
at any length, because lag is not distinguished by *duration* — only by the hold.

The hold-confirmed rule's residual 1% is entirely `slow-affordable`, whose `tau` of 800 sits against
a `tau_max` of 1000. That is the boundary case behaving like a boundary case, and it argues for
setting `tau_max` with margin rather than for a different rule.

## Four things the prototype got wrong first

Recorded because each is a trap the pre-registration has to be written against, and three of them
looked fine in argument:

- **A raw fractional decline eventually nominates every healthy cell.** As a converging curve
  settles onto its floor its *fractional* decline tends to zero. The decline has to be read on the
  content **above** the baseline.
- **The lever and the reading must be the same quantity.** `honoured_decline` was first derived over
  the window while the estimator measured over half of it, which halted a cell learning comfortably
  inside budget.
- **Scanning every tick is a multiple-comparison problem.** A 3000-tick run offers ~2600 overlapping
  chances to cross any bar, so a 3-sigma test fires on a healthy cell routinely. The halt is
  therefore evaluated on **non-overlapping windows on a fixed schedule** — one decision per window.
- **Entry 4 fires on the architecture working.** #138 initialises `K = aI`, so the fleet's first
  excursion against the band is the selection rig settling. Without a burn-in exclusion the rule
  halts on it. The exclusion is legitimate because the settling length is a construction quantity
  the rig already produces.

## Entry 2 is not a halt

`rollout_horizon()` reads the horizon straight off the singular-value gap of `K`, which #147 already
noted is "knowable rather than discovered". It is a **construction check**: read once, compared
against the loop the cell has to serve — `06-graph-topology.md` puts the somatomotor reflex at three
ticks and a visually-informed correction at four hops out and back.

The arithmetic is unforgiving and worth seeing before anything is built: at a gap of `0.5` the
horizon is 3.3 ticks, which serves the reflex and **fails the visual loop**. Cells that must carry
visual context need a gap above roughly `0.8`. Booking this against #78 as a run-time halt would
have been a mistake — it belongs in the build.

## Entry 4 and entry 5

**Entry 4** reads the **pre**-projection `sigma_max`. Post-projection it is never out of band, by
construction, so reading it says nothing; the live quantity is how often learning has to be pulled
back. The band's upper face is exactly 1 and #140 fixed it, so the only free constants are counts:
a dwell, a fleet fraction, and the burn-in above.

**Entry 5** reuses `08`'s instrument directly: non-overlapping IQRs between the contact and free
populations, **plus** a requirement that the gap widen across consecutive windows. The second gate
matters — a cell may simply find contact harder, and that is not the cost. The cost is one global
band failing to hold two rates, which gets *worse*. No ratio is invented anywhere.

## The blocker was mis-wired

#156 was recorded as blocked by [#142](https://github.com/NGL321/patchworks/issues/142) "since it
needs a transmitting graph to calibrate against". #142 is a **decision** about where transmission
comes from, and closing it did not make any graph transmit — the edit it decided is
[#155](https://github.com/NGL321/patchworks/issues/155), which is open, and stage 1's conversion edit
is unwritten. Closing #142 therefore lifted a gate that was never the real one, and put #156 on the
frontier while the thing it needs is still missing.

What that dependency actually gates is **numbers**, and this prototype found there are almost none to
gate: the forms are fixed, and the only inputs still owed by a run are the topology-only baseline and
the noise level — both **measured**, neither *chosen*, and so neither at risk of being tuned on the
run they gate.
