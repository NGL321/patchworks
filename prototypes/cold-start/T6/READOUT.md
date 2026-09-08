# T6 (#555): #556's two arms, built and trained

The rig for [B11](https://github.com/NGL321/patchworks/issues/555) on the map
[*Map: transport whose composed rank exceeds one*](https://github.com/NGL321/patchworks/issues/532).

**The headline: the reserve arm is the first thing on this map that raises
composed rank and privacy together, and the first arm anywhere on the record that
training does not erode.**

## How this ticket changed under it

#555 was opened to build [#540](https://github.com/NGL321/patchworks/issues/540)'s
stack on a rig. While it was being worked,
[B8/#548](https://github.com/NGL321/patchworks/issues/548) **shipped** levers (d)
and (c1) into `src/patchworks/graph.py` and **held** lever (b), the doubled
invariant, because at budget 63 the guaranteed private dimension reads zero at
most cells. [B12/#556](https://github.com/NGL321/patchworks/issues/556) then ruled
that collision, and turned this ticket into **two arms**:

| arm | mask policy | budget | what it is |
|---|---|---|---|
| **doubling** | `k_v = min(n, Σ_e m_e)` | 63 | #540's stack exactly as written |
| **reserve** | `k_v = n − p`, `p = 8` | 63 | #556's ruling |

The first rig (`stack.py`, `555-construction.json`, `555-departure.json`,
`555-b_invariant-*`) predates the shipped allocator and is **superseded**. Its
conclusions survived the rebuild — it read the doubling arm at 1.5933 / 1.1099
where the shipped allocator reads 1.5757 / 1.0936 — but quote the `555-arms-*`
records, not those.

## Files

| file | what it does |
|---|---|
| `arms.py` | builds the three arms on the **shipped** spec. `allocate_capped` is `graph.allocate_lane_widths` with its lane ceiling parameterised; `check_allocator_matches_shipped` asserts they agree at `cap = n` so the rig cannot drift. `apply_reserve` puts #556's one-line mask change on a built `Dome` |
| `trained_arms.py` | the trained ladder, with `cap_read` and `sigma_read` at every checkpoint and `sweep_c` **dropped** ([B9](https://github.com/NGL321/patchworks/issues/551) discredited it) |
| `departure.py` | attributes the gap between the built reading and #540's generic prediction |
| `analyse.py` / `check.py` | the readout tables, and re-derivation of every figure below |
| `stack.py` | **superseded** — the pre-#548 rig, kept because the advisories on #548 quote it |

**Nothing here edits `src/`.** #555's notes say build on the rig, not on the live
surface, and #556's ruling is a decision the map has not yet written.

## 1. The arms at construction, and what each costs

Seed 42, 263 chains, on the shipped spec.

| arm | built ER | p90 | generic | `k_v` med | lanes | private dim min | at zero | **private dim total** |
|---|---|---|---|---|---|---|---|---|
| shipped (today) | 1.1489 | 1.6185 | 1.3029 | 25 | 1–18 | 1 | 0 / 150 | 914 |
| doubling | 1.5757 | 2.0756 | 2.0665 | 32 | 1–32 | **0** | **104 / 150** | **54** |
| reserve | 1.3825 | 2.0306 | **2.2652** | 24 | 1–24 | **8** | 0 / 150 | **1200** |

Zero invariant violations at every arm. The rig **independently reproduces
#556's construction figures** — private dimension 914 / 54 / 1200 and generic
1.304 / 2.073 / 2.231 against 1.3029 / 2.0665 / 2.2652 — from a different reader
than #556's.

## 2. #540's law survives; its instrument's second idealisation does not

#540 named its own falsifier: a surface departing from the generic prediction at
its own per-hop widths. All three arms depart. **None of it is the genericity
claim:**

| arm | generic (Haar `V`, unit `σ`) | real `V`, unit `σ` | real `V` and `σ` | truth | `σ_min/σ_max` |
|---|---|---|---|---|---|
| shipped | 1.3029 | 1.2688 | 1.1489 | 1.1489 | 0.5063 |
| doubling | 2.0665 | 2.0420 | 1.5757 | 1.5757 | 0.3276 |
| reserve | 2.2652 | 2.2203 | 1.3825 | 1.3825 | 0.2544 |

The carried subspaces **are** generic (real `V` within 1–3% of Haar). The whole
gap is **singular-value spread**, which #540's instrument set to 1 — and it
worsens as lanes approach `k_v`, which is why the reserve arm's construction
reading understates it most.

**It is a construction artifact.** ADR-0032's band is a *training-time*
projection: `σ_min/σ_max` reads 1.0000 by 100 ticks, and the reserve arm reads
**2.2903** there — above its own generic value. A construction reading is the
wrong number to rule on, and #540, #548 and #556 all had only construction
readings.

## 3. The trained arms — the reserve arm does not erode

Baseline, both seeds.

| arm | seed | construction | 20k | **erosion** | p90 @20k |
|---|---|---|---|---|---|
| doubling | 42 | 1.5757 | 1.0936 | **×6.2** | 1.6720 |
| doubling | 43 | 1.4847 | 1.0725 | **×6.7** | 1.6336 |
| **reserve** | 42 | 1.3825 | 1.4386 | **×0.9** | 2.3504 |
| **reserve** | 43 | 1.4680 | 1.5307 | **×0.9** | 2.3244 |

The reserve arm at the **100k horizon**, which is what settles floor-versus-rate:

| ticks | 100 | 1,000 | 10,000 | 20,000 | 30,000 | **100,000** |
|---|---|---|---|---|---|---|
| ER median | 2.2903 | 1.8516 | 1.4039 | 1.4444 | 1.5282 | **1.4237** |
| p90 | 2.8215 | 2.5803 | 2.3264 | 2.3393 | 2.3299 | **2.2791** |

Flat from 10k to 100k in the band 1.40–1.53, **above its construction value at
every checkpoint from 10k on**, across the span in which `m = 3` loses a further
12.8× and the doubling arm falls to 1.09. **This is a floor, not a slow rate.**

Every arm ever measured eroded — ×437 (`m=3`), ×269 (6), ×18.0 (10), ×4.1 (14),
×6.2 and ×6.7 (doubling). The reserve arm is the first that does not, and at
**1.4237** it beats the record's previous best trained composed rank
([B6](https://github.com/NGL321/patchworks/issues/546)'s **1.186**) while holding
the `dim H⁰` floor *above* today's (1200 against 914).

## 4. B9's sign constraint, and the one place it needed watching

[B9](https://github.com/NGL321/patchworks/issues/551) showed a lever that pushes a
cell's incident maps apart destroys the alignment composed rank is made of. At
construction **neither arm does**: the Gram cap is flat — p90 0.3851 / 0.3900 /
0.3859, **4 of 414 cells at cap in all three** — while the leading per-hop cosine
rises 0.7184 → 0.9335 (doubling) and → **0.9784** (reserve).

Under training both arms press the cap hard, and the reserve arm hardest: p90
**0.9982** with 13 cells at cap at 100k. That was the live risk, and it **did not
bite** — the leading cosine ends at 0.992, higher than at 20k, and the ER does
not fall. The cap's median stays pinned at 0.2500 throughout, exactly as
[B7](https://github.com/NGL321/patchworks/issues/547) and
[B10](https://github.com/NGL321/patchworks/issues/552) found, so nothing here
re-opens `c`.

## 5. The bar

The map's bar is **generic median ≥ 2.0**.

| | median | p90 | excess over one |
|---|---|---|---|
| bar | 2.0 | — | 1.00 |
| reserve @100k | **1.4237** | **2.2791** | 0.42 |
| doubling @20k | 1.0936 | 1.6720 | 0.09 |

**Neither arm clears the median bar.** The reserve arm delivers 42% of the
required excess against the doubling arm's 9%, and its **p90 sits above the bar**
— so more than a tenth of rim-to-apex chains carry more than two directions at
horizon, which no surface on this map has done before.

## Reproducibility note

Two runs of the same arm at the same seed — the standalone 20k run and the 100k
run's own 20k checkpoint — agree to 3e-6 at 2,000 ticks and drift to 6e-3 by
20,000, with `torch.set_num_threads(1)` set. That is floating-point accumulation
amplified by a chaotic trajectory, not a seeding bug. It is far below every
effect reported here (the arms differ by 0.35) but **a trained median is not a
reproducible fourth decimal**, and `check.py` checks them at a tolerance of 0.02.
