# B57 (#629): the first reading of the graded agreement instrument

[#629](https://github.com/NGL321/patchworks/issues/629), under
[the map](https://github.com/NGL321/patchworks/issues/532). The instrument is
[B52 (#622)](https://github.com/NGL321/patchworks/issues/622)'s, which specified
it and deliberately took no reading.

**Surface.** `c925866` (2026-09-09, *Merge pull request #627 from
NGL321/worktree-b50-cosine-spectra-618*), `main`. Read on the built dome after
[#548](https://github.com/NGL321/patchworks/issues/548)'s per-edge lane
allocation and [#562](https://github.com/NGL321/patchworks/issues/562)'s
`k_v = n − p`; no figure is inherited from a pre-#548 readout. `n = 32`,
150 predicting cells, `δ_P` is `[3861, 4800]`, seed 42, one seed.

**Rig.** `b57_agreement.py` (instrument and runner), `b57_analyse.py` and
`b57_centred.py` (tables), `b57_run.sh` (the four stages, sequentially).

---

## 0. The instrument is checked against `main` before it is read

`δ_P` is assembled here and held against the one
`patchworks.diagnostics.Diagnostics.whole_graph` already builds by its own
route. Both give **rank 3000**, `dim H⁰` **1800** of 4800 columns, on every arm.
A dropped block, a flipped sign or a shifted offset would move one of them.

`D`'s factor `G` is **not** `δ` unsigned: `δ` gives an edge's two endpoints the
same rows because it differences them, and `D = Σ_{e∋v} F_eᵀF_e` sums per
endpoint, so `G` gives every predicting endpoint its own rows — 6654 of them.
Sharing them would read the denominator as `‖F_u x_u + F_v x_v‖²`.

## 1. The acceptance check passes, and it is not close

> Redraw the traffic, hold every map fixed. A stalk-side statistic must move.

It moves on **every row of every arm**, in both forms, under both redraws. The
baseline arm at its horizon:

| | held | scrambled (permute) | scrambled (isotropic) |
|---|---|---|---|
| traffic ER | 1.0045 | 110.260 | 827.532 |
| `q̄` | 0.1796 | 0.9811 | 1.0008 |
| `N(0.25)` | 1.0000 | 0.0000 | 0.0000 |
| `A(1)` | 0.9997 | 0.5241 | 0.4850 |
| `N(1)` | 1.004 | 32.406 | 401.217 |

and centred, same row: `q̄` **0.7792 → 1.0011**, `A(1)` **0.8536 → 0.4787**.
`earned` cannot move under this at all — it never sees a page. **The instrument
is not rejected; the ticket does not end at item 1.**

The scramble is *one permutation of the held configuration's cells per tick*, so
every page in the room is still a page somebody was holding and only the
arrangement is destroyed. The isotropic redraw is the degenerate end, reported
beside it.

## 2. The window, defended rather than assumed

`T = 1,000` is not invented here: it is `T0/run.py`'s `WINDOW`, the read window
every cold-start figure on this rig has been taken over. Re-read on the same
held configuration at frozen maps:

| `T` | traffic ER, uncentered | `q̄`, uncentered | traffic ER, **centred** | `q̄`, centred |
|---|---|---|---|---|
| 250 | 2.7303 | 0.9411 | 13.014 | 1.0262 |
| 500 | 2.7564 | 0.9405 | 18.465 | 1.0249 |
| 1000 | 2.7656 | 0.9410 | 24.912 | 1.0235 |

A single tick is not a covariance and the whole run is not a state; on B52's
uncentered form 1,000 ticks is stable to the fourth decimal against a quarter of
itself.

> **The defence does not transfer to the centred form, and this is a limit on
> §4 rather than on the window.** Centred traffic ER grows monotonically with
> `T` — 13.0 → 18.5 → 24.9 as the window quadruples — because the centred
> covariance of a slowly-varying configuration keeps admitting directions as the
> window lengthens. **So the centred effective ranks quoted in §4 are not
> window-independent quantities and no architectural conclusion rests on their
> magnitude.** What does not move with the window is the count: centred `N(θ)`
> is **0.0000 at every `θ` at every `T`**, which is the reading §3 actually
> carries. Whether the *trained* centred ranks are window-stable is unmeasured —
> the sweep was run at construction only.

## 3. What it reads — and the two forms disagree about what the room agrees on

[B53's advisory](https://github.com/NGL321/patchworks/issues/629) landed before
this reading and changed it: B52's `C = (1/T) Σ_t x_t x_tᵀ` is a second moment
**about zero**, and `body.py:528–531` says in as many words that *"any nonzero
mean in its stalk is permanently unreachable error"*. Every row below is
therefore read twice, off the same window, at no extra run.

**Baseline arm, `N(θ)` uncentered | centred, against the matched-generic null:**

| ticks | traffic ER | `q̄` | `N(0.25)` | `N_gen(0.25)` | excess |
|---|---|---|---|---|---|
| construction | 2.766 \| 24.912 | 0.9410 \| 1.0235 | 0.0000 \| 0.0000 | 0.0000 \| 0.0000 | +0.0000 \| +0.0000 |
| 500 | 2.982 \| 6.374 | 0.7833 \| 0.9785 | 0.0000 \| 0.0000 | 0.0000 \| 0.0000 | +0.0000 \| +0.0000 |
| 1000 | 1.097 \| 5.701 | 0.2553 \| 1.1379 | 1.0000 \| 0.0000 | 0.0000 \| 0.0000 | **+1.0000** \| +0.0000 |
| 5000 | 1.013 \| 3.245 | 0.1868 \| 0.9133 | 1.0000 \| 0.0000 | 0.0000 \| 0.0000 | **+1.0000** \| +0.0000 |
| 20000 | 1.0045 \| 1.611 | 0.1796 \| 0.7792 | 1.0000 \| 0.0000 | 0.0000 \| 0.0000 | **+1.0000** \| +0.0000 |

`N(0.05)` and `N(0.10)` are **0.0000 on every row of every arm in both forms**.
The profile at the horizon says why — the traffic is one direction, and it sits
just under 0.20:

```
 theta      A          N       dirs
  0.15   0.000000   0.0000      0
  0.20   0.997746   1.0000      1
  0.25   0.997746   1.0000      1
  0.75   0.999546   1.0036      6
  1.00   0.999670   1.0039    343
```

**The one thing the room agrees about is the standing mean.** Uncentered the
count is 1; centred it is 0, on every arm at every checkpoint. Strip the
baseline and the variety riding on it disagrees at essentially the generic
point.

## 4. It moves — and it is the map's first quantity that does

`N(0.25)` goes **0 → 1.000 between 500 and 1,000 taught ticks on all three
arms**, and `q̄` falls monotonically from 0.9410 to 0.1796 on the baseline arm.
[B22 (#571)](https://github.com/NGL321/patchworks/issues/571)'s clamp does not
reach it, exactly as B52's §4 argued: the weights come from what the pages are
carrying, not from what the mask permits.

**What was not predicted is that training drives the ceiling down.** The traffic
is not born at rank one and held there — it *starts* at 2.77 uncentered and
24.91 centred and collapses under the rule:

| arm | uncentered ER, construction → 20k | centred ER, construction → 20k |
|---|---|---|
| baseline | 2.766 → 1.0045 | 24.912 → 1.611 |
| winner | 2.786 → 1.0320 | 27.270 → 9.318 |
| flat | 2.544 → 1.0034 | 2.938 → 2.865 |

Stated as measured and not as mechanism: the collapse **co-occurs with the
training rule running**, and this ticket did not run an arm with the rule off
past construction, so it cannot attribute it.

**And the centred column of that table is window-dependent** (§2): centred ER at
construction reads 13.0 / 18.5 / 24.9 at `T` = 250 / 500 / 1000, so *24.912 →
1.611* is a fall measured at one window rather than a bound on the architecture.
The uncentered column does not have this problem. **The direction of the fall is
robust and its magnitude is not**, and B53's fork — *is centred ER materially
above 1.002?* — is therefore answered only in the weak form: **there is centred
rank at construction that is not there at the horizon**, at `T = 1,000`.

## 5. The flat bundle: B52's §5 prediction is confirmed, in its own words

At construction, the flat bundle installed by `b40_routes.install_flat_bundle`
and read on the post-projection surface the architecture actually has:

| | `N(0.05)` | `N(0.10)` | `N(0.25)` | audience differentiation | effective exposure | `earned` `H⁰` |
|---|---|---|---|---|---|---|
| flat, construction | 0.0000 | 0.0000 | **0.0000** | **0.0000** | 9.73 of 32 (rank 17) | 88 ([B40](https://github.com/NGL321/patchworks/issues/603)) |
| baseline, 20k | 0.0000 | 0.0000 | 1.0000 | 0.4893 | 8.57 of 32 (rank 20) | 0 |

B52 §5, on the record before the reading: *a flat room says one shared thing, so
it agrees about **one** thing, and B42's 88 is a phrasebook fact with no
page-side counterpart.* **The page-side count is 0.0000 — even weaker than the
prediction.** The flat bundle's audience differentiation reads **0.0000**
exactly on its construction row, which is B42's and B48's anchor reproduced on
the moved surface.

**The trained flat arm is a different object and says so.** By 1,000 ticks it
reaches `N(0.25) = 1.0000` like every other arm, at audience differentiation
risen to 0.1542 and 0.2338 by the horizon. Training walks the flat bundle off
flatness; the construction row is the flat bundle, the horizon row is not.

## 6. The joint report — B48's standing constraint, on every line

Agreement never travels alone. Baseline arm:

| ticks | `N(0.25)` | `q̄` | audience differentiation (median) | effective exposure |
|---|---|---|---|---|
| construction | 0.000 | 0.9410 | 0.5307 | 9.05 of 32 |
| 1000 | 1.000 | 0.2553 | 0.5161 | 8.90 |
| 5000 | 1.000 | 0.1868 | 0.4826 | 8.62 |
| 20000 | 1.000 | 0.1796 | 0.4893 | 8.57 |

Winner arm at 20k: `N(0.25)` 1.0017, `q̄` 0.1329, differentiation 0.4779,
exposure 9.05. Flat arm at 20k: 1.0000, 0.1623, 0.2338, 7.24.

Exposure is **rank-measured** (`rank Σ_{e∋v} F_eᵀF_e` and its participation
ratio), not the spec field, per the map's note. Audience differentiation is on
the **transport operator** and the count is on the **pages**; they are tabled
together, which [B49](https://github.com/NGL321/patchworks/issues/616) permits,
and neither is argued from the other.

## 7. Which branch fired

The ticket's second row — *`N ≈ 1` everywhere → B52's §4 prediction confirmed,
not a null result* — fires on the count. **Its `excess ≈ 0` clause does not.**

B52 expected the raw number to be *mostly bookkeeping*, netted out by the
matched-generic null the way
[B50 (#618)](https://github.com/NGL321/patchworks/issues/618)'s forced fraction
nets out on the transport side. It is not: **`N_gen(θ) = 0.0000` at every `θ`,
every arm, every checkpoint, both forms, over three independent redraws.**
Generic phrasebooks on the *same* traffic agree about nothing at all, so the
excess is the whole of the count rather than a small residue of it. That is one
row for the surprise ledger ([#520](https://github.com/NGL321/patchworks/issues/520)),
not a repair to the instrument.

## What this does not establish

- **One seed, one dome.** Seed 42 only; nothing here is a distribution.
- **No threshold is set**, deliberately — B52 §6, and the ticket forbids it.
- **No candidate is scored.** The flat bundle is the map's existing null, not a
  candidate, and reading it is item 3 rather than a scoring.
- **The collapse is not attributed.** §4 measures that traffic rank falls while
  the rule runs; it does not isolate the rule as the cause.
- **Centred effective rank is window-dependent** (§2), so §4's centred column
  reports a direction and not a magnitude. The sweep was run at construction
  only; whether the trained centred ranks are window-stable is unmeasured.
- **`N(θ)` above 1 was never observed**, so nothing here says what the instrument
  would read on a surface whose traffic carries rank — which is exactly
  [B53 (#623)](https://github.com/NGL321/patchworks/issues/623)'s and
  [B60 (#633)](https://github.com/NGL321/patchworks/issues/633)'s question.
