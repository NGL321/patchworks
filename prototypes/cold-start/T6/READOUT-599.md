# B38 (#599): what surface this map can take a dynamical reading on

Rig work in service of a decision, per the ticket's own note. Four asks, answered
in order. The short version, before the detail:

1. **`arms.py` is ported**, and the port is smaller than the ticket expected —
   #597 shipped the ruling the rig was faking, so the reserve arms stopped being
   a rig construction at all and the rig's own allocator was deleted rather than
   repaired.
2. **The transplant route is adopted**, with a scope the ticket did not state —
   and the ticket's premise is corrected: **the map's own bar does not need it.**
3. **The stall boundary is reported per arm — and the headline is that a per-arm
   number is not a thing that exists.** It moves 13x across seeds of the *same*
   arm and a further 11x with the measurement timescale. Two marks, not one; the
   record's 1,000–2,000 was the wrong mark.
4. **The bar is read, and it FAILS both ways** — conduction 0.433 rim→apex (short
   2.31x) and 0.133 apex→rim (short 7.5x). A 150x training contrast prices what
   the stall costs and finds something sharper: **150x more training moves the
   per-edge amplitude by three orders and the bar not at all**, because the bar is
   a ratio of *times* and training moves *amplitudes*.

---

## 0. The premise this ticket was opened on, corrected

B33 declined to read the bar because *"detectability is a dynamical reading and
this arm's world is dead from tick ~150, so a conduction ratio taken at any
trained checkpoint would be a ratio taken on a motionless world"*. That was the
right call on the information B33 had, and the caution was the right instinct.
**But the premise is false, and it is worth being exact about why.**

`benchmarks/detectability.py` **holds the world by construction.** Its own
`branch()` says so in a comment — *"The world's write is the tick's last word and
both branches get the same one. Nothing is stepped: this is the hold."* The
measurement is a paired fork: two branches restored from one state, one nudged,
both run for `WINDOW = 64` ticks with **the same fixed observation written every
tick**, and differenced. `read()` prepares each trial the same way, with
`hold_still(agent, observation, applied, HOLD)` for 400 ticks on one `env.reset()`
observation. The world is never stepped inside a trial, and it cannot be — the
pairing requires both branches receive identical world input, so a *moving* world
would break the measurement rather than enable it.

The divisor is unaffected for the same kind of reason: `world_loop(c)` is a tick
length enumerated from the mask (`benchmarks/loop_length.py`), a construction
quantity, not a trajectory.

**So the stall does not invalidate detectability's probe. What it corrupts is the
surface underneath it.** `--learn 30000` on an arm whose world stops early is
~29,800 ticks of training against a motionless world, and the operating point the
hold is taken at is whatever pose the stall left. That is a real problem and §4
prices it — but it is a *training* problem, not a *measurement* problem, and the
difference decides which readings need §2's transplant and which do not.

---

## 1. The port (ask 1)

**The fake and the real have swapped places.** #555 wrote `arms.py` while #556's
ruling was still a decision the map had not written, so the rig carried its own
copy of the ruling. #597 landed [B15](https://github.com/NGL321/patchworks/issues/562)'s
write of it into `src/patchworks/graph.py`, and every piece of that copy is now
shipped:

| the rig's copy | what shipped on #597 |
|---|---|
| `apply_reserve(dome, p)` — `dataclasses.replace` onto a built `Dome` | `Dome._assemble` applies it itself, from `spec.private_reserve` |
| `allocate_capped(..., cap=n − p)` | `allocate_lane_widths` caps at `n − spec.private_reserve` natively |
| `spec.privacy_budget` | `spec.capacity_budget`, default `2n − 1 = 63` |
| — | `spec.private_reserve`, default `CHART_DIM = 12` |

So the reserve arms are no longer a rig construction: **`reserve_p<N>` is
`replace(spec, private_reserve=N)` and a plain `build_graph`.**

**The rig's allocator and `check_allocator_matches_shipped` are deleted, not
repaired.** That function existed to assert a rig copy agreed with the shipped
allocator at `cap = n`. Calling the shipped allocator directly is strictly
stronger than asserting a copy of it agrees, so there is nothing left for it to
buy.

**What survives is the inverse of what was there.** The `shipped` and `doubling`
arms run the **union** mask `k_v = min(n, Σ_e m_e)`, which #556 retired and #597
removed from `src/`. They are kept — and only kept — so this map's pre-#556
readings stay reproducible. `apply_union` is now the one `dataclasses.replace` in
the file, and it reconstructs a **retired** surface rather than standing in for an
unwritten ruling. On a union arm `private_reserve = 0` is not a privacy policy: it
is how the shipped allocator is told to cap a lane at `n`, which is what the
pre-#556 allocator did.

**`reserve_p12` is what `main` builds.** `DomeSpec()` with nothing overridden *is*
`reserve_p12`, so that arm is no longer a variant — it is the shipped dome, and
every other arm in the file is now a deliberate departure from it. Per #455 the
surface is named in the module docstring, with each arm marked *live* or
*historical*.

### The port is checked, not asserted

`b38_port_check.py` makes three assertions, and reproduces the record exactly:

```
       shipped [  union] budget 31 k_v med  25.0 priv min   1 zero   0/150 total  914
      doubling [  union] budget 63 k_v med  32.0 priv min   0 zero 104/150 total   54
       reserve [reserve] budget 63 k_v med  24.0 priv min   8 zero   0/150 total 1200
    reserve_p0 [reserve] budget 63 k_v med  32.0 priv min   0 zero 150/150 total    0
    reserve_p4 [reserve] budget 63 k_v med  28.0 priv min   4 zero   0/150 total  600
    reserve_p8 [reserve] budget 63 k_v med  24.0 priv min   8 zero   0/150 total 1200
   reserve_p12 [reserve] budget 63 k_v med  20.0 priv min  12 zero   0/150 total 1800
   reserve_p16 [reserve] budget 63 k_v med  16.0 priv min  16 zero   0/150 total 2400
   reserve_p20 [reserve] budget 63 k_v med  12.0 priv min  20 zero   0/150 total 3000
```

* **`shipped` total 914** — #548's `dim H⁰` floor, to the unit.
* **`doubling` zero-privacy at 104 of 150, total 54** — the exact collision that
  made #548 hold #540's doubling, reproduced from a retired mask.
* **`reserve_p12` total 1800 = 150 × 12** — #556's flat floor, an equality in `p`.
* **`reserve_p12` is byte-identical to `build_graph(DomeSpec())`** — spec, lane
  widths, permitted blocks and private mask all compared. This is the assertion
  the old drift guard was reaching for and could not make, because the rig then
  had no shipped reserve to compare itself to.

Every T6 instrument built on `arms.py` — `b33_motion.py`, `b28_gauge.py`,
`b28_generalise.py`, `b29_*`, `b16_*`, `b18_*`, `b19_*`, `b20_*` — imports again.

---

## 2. The transplant route (ask 2)

**Adopted as the standing way to read a trained surface on a moving world**, with
the scope below. There is no alternative to adopt it against: #577 established
that `reset()` does not revive the world (the sampler may not change the arm's
pose, `sandbox/env.py` `PLACEMENT_ATTEMPTS`), and re-using a body across
arrangements carries the stall into every one of them.

**The route.** `b28_gauge.py` saves `sheaf.maps` at the horizon
(`577-maps-<arm>-seed<seed>-<ticks>.pt`); `b28_generalise.py` loads them into a
**freshly built agent per world arrangement** and reads over a short window while
that fresh body is still moving. The control is the same fresh build with its maps
left at construction, so the two differ in exactly one thing.

### What it carries

`sheaf.maps`, and nothing else. `K`, the biases and the stalks are the fresh
agent's constructor values. Stated as the ticket asked:

**It does carry** — a claim about **transport**: *do the trained lanes agree on
the signal a moving body produces.* That is the object every ruling on #532 is
about, and `composed_reads` multiplies restriction maps and never touches the
stalks ([B17](https://github.com/NGL321/patchworks/issues/565)), so for the
composed object the transplant is lossless in the quantity that matters.

**It does not carry**, and therefore cannot be used to read:

* anything depending on the **trained `K`** or the trained biases — the chart
  operators are the fresh agent's. #526's leak is the live example, and R-A
  already found `K` is not a factor in the composed object, so this costs the
  composed readings nothing and costs a `K` reading everything.
* anything depending on the **trained stalk state** — the fresh sheaf starts at
  constructor zeros, which is why `b28_generalise.py` drops `SETTLE = 10` ticks.
* anything about **training dynamics** — it is a snapshot at one horizon and
  cannot say how the maps got there.
* **a replay of the trained agent's own trajectory.** That is unavailable at any
  horizon, and the transplant is not a substitute for it. This is the difference
  between the reading the transplant gives and the reading the ticket that wanted
  it asked for, and it is stated here rather than buried.

### What it is *not* needed for — the scoping this ticket adds

**A reading that holds the world by construction does not need the transplant.**
`benchmarks/detectability.py` is that reading (§0), and it is this map's own bar.
So the answer to *"is a rig fix prior to the architecture question?"* —
[B33](https://github.com/NGL321/patchworks/issues/592)'s advisory 2 — is **not for
the bar**. The bar could have been read all along; §4 reads it. The transplant is
prior for readings that need the world stepping: edge agreement on held-out
episodes ([B25](https://github.com/NGL321/patchworks/issues/574)'s named test),
and anything measuring what the loop does rather than what the operator is.

### The window is not a constant

`b28_generalise.py` fixes `EVAL_TICKS = 140`, justified by a 100-tick-window
figure (*world `std_max` 1.68 over ticks 0–100, 0.38 over 100–300*). §3 shows why
no fixed value is defensible: the horizon moves by up to 13x across seeds of one
arm, and by a further 11x with the measurement window on one of them. **The window
must be read off the run, at the timescale the reading needs, not off a
constant.**

B28's design already survives this, and that is to its credit: it stamps
`motion.read()` on every row, so a transplant row carries the evidence for its own
validity rather than inheriting it. That stamp is hereby the requirement, and the
constant is demoted to a default.

---

## 3. The stall boundary (ask 3)

The ticket asked for the stall boundary **per live arm**. The honest answer is
that **a per-arm number is not a thing that exists**, and that is the finding.

`b38_stall.py` drives `b33_motion.py`'s `trace()` across four arms x two seeds, on
MuJoCo's own `qpos`/`qvel`, in disjoint 10-tick windows to 2,000 ticks. Two marks,
because *stalled* is two different claims:

* **`t_moving`** — the last tick with `std_max >= 1e-2`: the world is *visibly*
  moving. This is the honest horizon for a reading.
* **`t_dead`** — the last tick with any component above the `1e-6` floor. Past it
  a reading is a ratio of numerical zeros, which is the shape #577 found.

### The table

| arm | `t_moving` s42 | `t_moving` s43 | `t_dead` s42 | `t_dead` s43 |
|---|---|---|---|---|
| `shipped` | 200 | **1240** | 1110 | 2000 |
| `reserve` (p8) | 200 | **60** | 2000 | 230 |
| `reserve_p12` | **70** | 250 | 390 | 430 |
| `reserve_p16` | 110 | **1450** | 2000 | 2000 |

### Three things it says

**(i) The record was reading the wrong mark.** The map's standing note — *the body
stalls between 1,000 and 2,000 ticks*, from #572 and #518 — lands in the `t_dead`
column, and `t_dead` is the last flicker of numerical motion rather than the
window in which the world moves. `t_moving` is **250 or less on six of the eight
cells**. Any reading that took 1,000–2,000 as its working horizon was taking most
of its window on a world that had stopped, whichever arm it ran on.

**(ii) The arm is not the explanatory variable.** Seed variation *within one arm*
reaches **13x** (`reserve_p16`, 110 -> 1450) and is 3.3–6.2x on the other three.
That is larger than any difference between arms. So B33's diagnosis — the figure
*"does not transfer to the arm B27 put the map on"* — was right that it does not
transfer, and the reason is sharper than it looked: **it does not transfer between
runs, not between arms.** Reading a horizon off one arm and applying it to another
was never the error; reading it off one *run* was.

**(iii) There is therefore nothing to inherit.** No constant — not `EVAL_TICKS`,
not this table — is a safe horizon for a future reading. **A reading must stamp
the motion of its own run.** That is now the rule, and `b38_stall.py` and B28's
`motion.read()` are how it is done. This table is evidence for the rule, not a
lookup for anyone to quote.

### Two checks on the instrument

* **Neither mark is window-free, and on one run the window matters more than
  everything else.** Re-run at 50-tick windows on seed 42:

  | arm | `t_moving` w10 | w50 | `t_dead` w10 | w50 |
  |---|---|---|---|---|
  | `shipped` | 200 | 200 | 1110 | 1150 |
  | `reserve` | 200 | 200 | 2000 | 2000 |
  | `reserve_p12` | 70 | 100 | 390 | 2000 |
  | `reserve_p16` | 110 | **1250** | 2000 | 2000 |

  Three arms barely move and `reserve_p16` moves **11x**. The mechanism is
  plain once seen: a 50-tick window aggregates more samples, so slow drift that
  never clears `1e-2` within any 10 ticks does clear it across 50. Both marks
  inflate the same way, `t_dead` on `reserve_p12` (390 -> 2000) and `t_moving`
  on `reserve_p16`.

  **So "the world is moving" is scale-relative, and the horizon is a function of
  (arm, seed, timescale) rather than a number.** This is the strongest form of
  (iii): there is no window-free horizon to inherit even for one run, so a
  reading must measure motion **at the timescale that reading needs**, on its own
  run. B28's per-row stamp does this; a shared constant cannot, whatever value it
  is given.
* **`std_max` peaks are O(1) on every cell** (0.91–4.04), so the `1e-2` threshold
  is two decades below the motion the arm actually makes. It is a generous floor,
  not a tuned one, and moving it would not rescue the 1,000–2,000 reading — the
  fall past `t_moving` is to `1e-4` and below.

---

## 4. The bar, read (ask 4)

**Surface, named per #455:** `main` at `3b82b33` (which contains #597),
`benchmarks/detectability.py read --dome full`, seed 0, 24 trials, `WINDOW = 64`,
`HOLD = 400`. `--dome full` is `dome_named("real")`, and that spec is
`DomeSpec()` exactly — **checked, not assumed** — so the bar's surface *is*
`reserve_p12`, `capacity_budget = 63`, `private_reserve = 12`: the arm B27 put the
map on. `--no-file` on both runs, for the reason in *What was not filed* below.

### The reading

**Both directions FAIL ADR-0026's bar.**

| | 30,000-tick learn | 200-tick learn |
|---|---|---|
| **rim→apex** conduction, median | **0.433** — short by **2.31x** | 0.5 — short by 2x |
| p25–p75 | 0.321–0.466 | 0.333–0.579 |
| binding cell | #411 L7/core(6,) | #330 L1/somatomotor(3,) |
| **apex→rim** conduction, median | **0.133** — short by **7.5x** | 0.188 — short by 5.33x |
| p25–p75 | 0.133–0.2 | 0.133–0.2 |
| binding cell | #410 L7/core(5,) | #411 L7/core(6,) |
| rim→apex bottleneck, median | 8.35e-10 | 1.83e-12 |
| apex→rim bottleneck, median | 1.10e-08 | 7.71e-12 |
| unreadable cells (r→a / a→r) | 92.6% / 84.8% | 94.6% / 90.8% |

Both sit inside the **1.1x–8.5x** band #274 recorded, so nothing here is a
departure from what the map already knew the bar reads at. ADR-0021's bottleneck
ratio fails by ~9–12 orders, which is the split `detectability.py`'s own docstring
predicts: the two ratios fail differently and by wildly different margins.

### What the contrast prices, and it is the finding

The two columns differ in one thing: **150x the training.** Seed 0's own horizon
is `t_moving = 160` (measured, not inherited — §3's rule applied to this ticket's
own reading). So the 200-tick arm trains **80% inside the moving window**; the
30,000-tick arm trains **0.53% inside it**, and is 99.5% dead-world training.

**On the bar, the extra training buys nothing.** 0.5 → 0.433 and 0.188 → 0.133 —
*lower* in both directions, with quartiles that overlap heavily. The honest
statement is that the two are indistinguishable on conduction, and certainly that
150x more training does not improve it.

**But training is not inert — it is inert *on the bar*.** The per-edge bottleneck
ratio moves **456x** (1.83e-12 → 8.35e-10) rim→apex and **1,427x**
(7.71e-12 → 1.10e-08) apex→rim. So 30,000 ticks demonstrably change the surface;
they change it in a quantity the operative bar does not measure.

**The mechanism is arithmetic, and it is the piece the map should take.**
ADR-0026's conduction ratio is `τ̂_c / world_loop(c)` — **a ratio of times**. `τ̂`
is an e-fold decay time; `world_loop` is a construction-time tick count that
training cannot touch at all. Training moves **amplitudes**, which is exactly what
the bottleneck ratio measures and what moved by three orders. It does not move the
**decay timescale**, which is the whole of the numerator. So:

> **The rule that trains transport, as it stands, does not act on the quantity the
> map's bar measures.**

That bears directly on #532's destination, which was widened by B27 from *how
transport is parameterised* to *transport **and the rule that trains it***. It says
the current rule is not a weak lever on detectability — on this evidence it is not
a lever on it at all.

### What this does *not* say

* **It does not say the stall is harmless.** It says the stall does not corrupt
  this measurement (§0) and that the training it corrupts was not buying
  conduction anyway. Whether a *moving* world would let training move `τ̂` is
  untested, and is the natural next question — but answering it needs a world that
  moves for longer, and enriching the sandbox is **out of scope on #532**.
* **It is one seed.** Seed 0, 24 trials, both arms. B33's own note that the
  record's two-run drift exceeds several quoted gaps applies here too.
* **#224's gate is wide open on both arms.** 85–95% of cells are unreadable at
  runtime precision, and the rim→apex binding cell is unreadable in 42–46% of
  trials. This is a standing property of the rig rather than something these runs
  introduced, and it is roughly equal across the two columns — so it does not
  explain the contrast, but it does cap how much weight any single median carries.
* **It is not a verdict on the architecture.** It is a reading of where the bar
  currently sits, which is what the ticket asked for.

### What was not filed

`--no-file` on both runs, deliberately. `read` files a run against #325, #329 and
#341 on the tracker, and the file-worthy read is *the* read. The 30,000-tick run
is at the default learn on the full dome and would ordinarily qualify — but this
same ticket has just found that ~99.5% of that training happens on a motionless
world, and filing a cutoff record against three open problems on a surface whose
training this readout is simultaneously calling degenerate would put a number on
the tracker that the readout beside it disputes. The cutoff readings are reported
here instead: **#341 CLEAR at 0.133; #325 and #329 SHUT at 0.133**, unchanged in
kind from the record.

Whether *the* read should be re-filed on this surface is a decision for the map,
not for a rig ticket, and it is raised as fog rather than taken here.

---

## 5. What this suggests for the map, carrying no authority

Advisory, per the map's hand-off rule. These are inferences from a rig ticket, not
consequences of a pre-registered branch, and none of them is a ruling.

1. **The map's bar and the map's training rule may not be in contact.** §4's
   contrast found 150x training moving the per-edge amplitude by three orders and
   the conduction ratio not at all, and the arithmetic reason — the bar is a ratio
   of *times*, training moves *amplitudes* — is not a property of this run. If
   that holds, then a decision about "the rule that trains transport" cannot be
   argued to reach ADR-0026's bar without something that acts on `τ̂`, and B27's
   widening of the destination has a gap in the middle of it. **This is the one
   worth a ticket.**
2. **`world_loop(c)` is the other half of the ratio and nothing on this map has
   ever moved it.** It is a construction-time tick count from the mask. If
   training cannot raise `τ̂`, the remaining lever on the bar is *topological* —
   shortening the loop — which is a statement about the dome, and B27 demoted the
   dome to scaffold. Worth stating that the map has two levers on its bar and has
   been reasoning about neither.
3. **The stall may be a smaller problem than it looked, and a differently-shaped
   one.** It does not touch the bar; it touches training, and training was not
   moving the bar. What it clearly *does* block is B25's held-out edge-agreement
   test, which is what §2's transplant exists for.
4. **`b38_stall.py` should be run before any new dynamical reading**, not as
   ceremony but because §3 shows the horizon is unpredictable per run. It costs
   ~2.5 minutes for 2,000 ticks.
5. **#224's gate deserves its own look.** 85–95% of cells unreadable at runtime
   precision is the background condition of every reading this map takes, and no
   ticket on #532 has priced it.

---

## Surface and reproduction

* **Base:** `main` at `3b82b33`, which contains #597. `arms.py` is ported to it
  (§1); nothing in `src/` is edited.
* **Instruments (this ticket's own):** `b38_port_check.py`, `b38_imports.py`,
  `b38_stall.py`, `b38_show.py`.
* **Raw:** `599-stall.json` (seed 42, w10), `599-stall-seed43.json` (seed 43,
  w10), `599-stall-w50.json` (seed 42, w50), `599-stall-seed0.json` (the bar's own
  seed), `599-bar-learn30000-seed0.txt`, `599-bar-learn200-seed0.txt`.
* **Commands:**

```
PYTHONPATH=src python prototypes/cold-start/T6/b38_port_check.py
PYTHONPATH=src python prototypes/cold-start/T6/b38_imports.py
PYTHONPATH=src python prototypes/cold-start/T6/b38_stall.py
PYTHONPATH=src python benchmarks/detectability.py read --dome full --learn 30000 --trials 24 --seed 0 --no-file
```
