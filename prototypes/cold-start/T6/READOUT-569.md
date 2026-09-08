# B20 (#569): how many distinct directions reach the apex across all chains

*Instrument: `prototypes/cold-start/T6/b20_joint.py`. Raw: `569-joint-construction.json`
(17 arms x 2 seeds, construction) and `569-joint-<arm>-baseline-seed42-20000.json`
(4 arms trained to 20k, read at every checkpoint). Surface: branch
`worktree-b20-joint-span-569` off `main` `13c015b` (B8's per-edge lanes, #558,
shipped); rig `T0`-`T6` taken from `worktree-b16-drive-564`.*

## The answer in one line

**On the live shipped surface the joint span is 1.03 — a genuine collapse under
any reading, and B17's specialist defence does not rescue it. But it is not near
one everywhere: the trained `p = 16` and `p = 24` arms deliver 4.3 to 7.2 distinct
directions, `p = 24` filling 89% of the ceiling the geometry allows. The joint span
and the per-chain median move in the *same* direction with `p`, not opposite ones —
so the map's bar and its candidate have not been in conflict. What the joint reading
adds is a boundary the per-chain reading cannot see: `p` buys rank by spending the
apex's capacity, and at `p = 24` the capacity is nearly gone.**

## 0. What was built, and in whose coordinates

#569's premise was confirmed before anything was built. `composed_reads`
(`T2/run.py:496-520`) returns one operator per chain and `er.append(...)` at line 506
is the entire aggregation; `T4/angles.py:191-195,268-287`, `T6/b16_coherence.py:123-127`
and `T6/b16_junction.py:160-163` are all order statistics over that one list. Nothing
in the rig stacks the composed operators.

Each chain's composed operator `C` runs from the rim edge's lane coordinates to its
**last** edge's lane coordinates. Chains reaching the same apex do not in general
arrive on the same edge, so they are pulled into the apex cell's own stalk with the
adjoint `hop_operator` already uses:

    A = F.T @ C            # (n x m_first), in the apex cell's stalk basis

Three readings are taken over the stack of `A`, because a stack of 84 operators with
different gains has an effective rank set by the loudest of them:

* **`joint_energy`** — effective rank of `[A_1 | ... | A_N]` as it stands.
* **`joint_unit`** — the same at unit Frobenius norm per chain: diversity, no gains.
* **`directions`** — each chain reduced to the one unit direction it delivers,
  summarised by the scale-free `d_eff = 1 / mean_{i≠j} <u_i,u_j>²`, which is `d` for
  directions spread isotropically over a `d`-dimensional subspace and 1 when they
  coincide. It needs no rank tolerance, so it is the number quoted throughout.

Never quoted alone: every `d_eff` sits beside the **lane ceiling** (how many
directions the geometry permits at all) and a **generic control** (Haar directions
drawn inside that same ceiling — a population code working, as a number).

**The pullback was checked, not assumed** (`b20_joint.py check`). At the six apexes
where every chain arrives on one edge, no pullback is needed and the stack can be read
natively in edge coordinates. The two agree: shipped reads 1.790/1.773, 2.201/1.784,
1.148/1.219, 1.264/1.122, 1.227/1.230, 1.520/1.524 (pulled/native), and `p = 16` agrees
as closely even where the map's conditioning reaches 179. The pullback is not
manufacturing the result.

## 1. The graph decides most of it before training starts

263 rim-to-apex chains, all at exactly 7 hops, into **8** apex cells, `n = 32`.

| apex | chains | distinct last edges | ceiling (shipped) | distinct suffixes (1/2/3 edges) |
|---|---|---|---|---|
| 405 | 84 | 2 | 12 | 2 / 3 / 4 |
| 409 | 83 | 2 | 12 | 2 / 3 / 4 |
| 406, 407, 408, 410, 411, 412 | 16 each | 1 | 7 | **1 / 1 / 1** |

Two facts here, and neither has appeared on this map before.

**Six of the eight apex cells receive all sixteen of their chains through a single
edge seven lanes wide.** Whatever those chains carry, at most 7 directions arrive.

**The chains are a tree, not 263 independent paths.** At those six apexes all 16
chains share their *last three edges* — one common three-hop tail. At the two large
apexes, 83 and 84 chains share just **four** distinct three-hop tails. So the premise
#569 set out to test — *"the network's capacity lives in the fact that 256 chains
carry 256 different directions"* — is not merely false on this surface, it is
**unavailable from the graph**, before any operator is looked at. The most the
geometry offers per apex is 7 to 16 directions, not 84.

## 2. The joint span, trained: the headline table

Baseline condition, seed 42, 20k ticks. `per-chain ER` is `composed_reads`' own
median — the number this map has quoted throughout — recomputed here from the
identical product, and it reproduces [B13](https://github.com/NGL321/patchworks/issues/560)'s
values (`p8` 1.4142 vs B13's 1.4386, `p16` 2.8251 vs 2.9171), so the instrument is
reading the same object. `ceiling` and `generic` are medians over the 8 apex
cells, which differ (see §1): shipped is 12 at the two large apexes and 7 at the six
small ones, `p = 8` is 24 and 15, `p = 16` is 16 and 15, `p = 24` is 8 throughout.

| arm | per-chain ER | joint `d_eff` | joint energy ER | joint unit ER | ceiling | generic | energy/ceiling |
|---|---|---|---|---|---|---|---|
| shipped | 1.0017 | **1.029** | 1.005 | 1.049 | 7 | 6.45 | 0.14 |
| `p = 8` | 1.4142 | 1.339 | 1.553 | 2.073 | 15 | 15.78 | 0.10 |
| `p = 16` | 2.8251 | 4.276 | 6.707 | 7.113 | 15 | 16.11 | 0.45 |
| `p = 24` | 4.0000 | 5.083 | **7.123** | 7.204 | 8 | 8.09 | **0.89** |

**The shipped surface is a genuine collapse under every reading.** Joint `d_eff`
1.029, energy ER 1.005: 16 to 84 chains deliver *one shared direction* into each
apex. Per apex it is 1.000 to 1.091. This is the reading #569 was opened to get, and
it answers B17's question in the negative — the chains are not narrow specialists
each broadcasting its own thing, because they are all broadcasting the *same* thing.

**It is not near one everywhere.** `p = 16` and `p = 24` land squarely in #569's
"structured in between": 4.3 to 7.2 directions, well above one and well below the
number of chains, bounded by the geometry rather than by rank collapse.

## 3. How it moves with `p` — the crux, answered

**They move together, not in opposite directions.** #569's crux was that `p` might
lower the joint span while raising the per-chain median, in which case *"the map's
bar and its candidate have been moving in opposite directions the whole time."* On
the trained curve they do not: joint energy ER runs 1.005 → 1.553 → 6.707 → 7.123
across shipped, `p8`, `p16`, `p24`, monotonically, in the same direction as the
per-chain median. The bar has not been misleading in the way the ticket feared.

The `d_eff` column has one non-monotone point (`p8` at 1.339 sits below shipped's
ratio-to-ceiling), but the energy and unit readings are monotone and `p8`'s absolute
values exceed shipped's on all three. It is a soft point on a rising curve, not an
inversion.

**At construction the picture is the mirror of B13's, and consistent with it.**
Across the whole 17-arm ladder, construction joint `d_eff` falls with `p` (2.41 at
`p = 0` to 1.12 at `p = 28`) exactly as construction per-chain ER does, against a
generic control of 16.0 down to 4.1. B13's *"the construction curve and the trained
curve run in opposite directions"* holds for the joint object too, which is one more
instance of B11's *"a construction reading is the wrong number to rule on."*

**What the joint reading adds that the per-chain one cannot see: `p` is spending the
apex's capacity.** The lane ceiling is not fixed across the sweep — it falls with `p`,
because the reserve narrows `k_v` and the allocator caps a lane at `n − p`:

    p ≤ 16  ceiling 15    p = 20  ceiling 12    p = 24  ceiling 8    p = 28  ceiling 4

Read against that, `p = 24` fills **89%** of the directions its apexes can hold, and
`p = 16` fills 45% of a ceiling nearly twice as large. Per-chain ER cannot register
this: at `p = 24` it reads a flat **4.0000**, which is B13's *"degenerate ceiling"*
showing up as a clean number with no indication that the apex behind it is nearly
full. **The joint span is therefore a bound on how far `p` can usefully be pushed,
and it bites well before `p = 28`** — somewhere between 16 and 24 the arm stops
buying directions and starts running out of room to put them.

## 4. Grouped by rim kind

**The grouping #569 asked for is undefined for three of the four kinds, and saying
so is part of the answer.** The rim is 256 `patch` cells against **1** actuator, **3**
proprioceptive and **3** touch. A within-kind subspace dimension over one chain is 1
by construction and over two chains is a single pairwise cosine — `d_eff` values like
47.8 and 305.1 appear in the raw JSON for such groups and mean nothing.

What *is* well defined at any group size is the angle **between** kinds' spans, which
is the claim itself: a healthy population code should put different modalities on
different directions. Median pairwise `|cos|` between delivered directions, at the two
apexes that host more than one kind:

| pair | shipped @20k | `p = 16` @20k |
|---|---|---|
| actuator ∣ patch | **0.995** | 0.637 |
| actuator ∣ proprioceptive | **0.951** | 0.733 |
| actuator ∣ touch | **1.000** | 0.263 |
| patch ∣ proprioceptive | **0.927** / 0.977 | 0.533 / 0.163 |
| patch ∣ touch | **0.994** / 0.976 | 0.135 / 0.187 |
| proprioceptive ∣ touch | **0.950** / 0.980 | 0.109 / 0.270 |

**On the shipped surface the modalities have collapsed onto each other completely** —
every cross-kind median cosine is 0.93 to 1.00, and actuator and touch deliver
literally the same direction (1.000). Where a limb is being commanded and where it is
being touched arrive at the apex as the same vector. **`p = 16` separates them**:
proprioception and touch fall to 0.109, patch and touch to 0.135.

*Caveat on the raw field:* `cos_principal` is 1.0 for every pair involving `patch` and
should not be read as collapse — an 80-chain span is high-dimensional enough to
contain any single vector, so the smallest principal angle is uninformative when one
side is large. `cos_pairwise_median` is the column to read, and it is what is tabled
above.

## 5. The untrained surface

*(#569 item 4.)* Construction, 17 arms x 2 seeds, `569-joint-construction.json`.
Joint `d_eff` sits between **1.04 and 2.65** across the entire ladder and both seeds,
against generic controls of 4.1 to 16.7 — so the joint span is far below a working
population code *at construction too*, on every arm, before training touches it.

The trained/generic comparison the ticket asked for, on the joint object:

| arm | construction `d_eff` | trained `d_eff` @20k | generic |
|---|---|---|---|
| shipped | 1.655 | 1.029 (**eroded**) | 6.45 |
| `p = 8` | 1.252 | 1.339 | 15.78 |
| `p = 16` | 1.672 | 4.276 (**raised ×2.6**) | 16.11 |
| `p = 24` | 1.232 | 5.083 (**raised ×4.1**) | 8.09 |

Training destroys the joint span on the shipped arm and *builds* it on the reserve
arms at `p ≥ 16` — the same sign flip B13 found per chain, reproduced on the joint
object. This is worth stating plainly because it cuts against the intuition the map
has been carrying: **on the arms that work, training is not the enemy of joint
capacity, it is where the joint capacity comes from.** Construction never has it.

## 6. What this does not measure

Stated as plainly as what it does.

* **Still the operator, never the signal.** This is B17's finding and #569 does not
  escape it. `A = Fᵀ C` is built from `agent.sheaf.maps` alone and never touches
  `agent.sheaf.stalks`. A joint span of 7 means seven directions the transport
  *retains*; whether any signal travels them is exactly as unmeasured as before.
  Nothing here settles B17's objection — it re-poses it on a bigger object.
* **One seed on the trained arms.** Construction is two seeds (42, 43) across 17 arms;
  the trained table is **seed 42 only**, 20k ticks, baseline condition. B13's per-chain
  curves held at seed 43 and to 100k, and the per-chain values reproduced here match
  B13's, which is indirect support — but the joint numbers themselves have no second
  seed and no 100k horizon behind them.
* **Four arms trained, not seventeen.** `p` ∈ {shipped, 8, 16, 24}. The boundary this
  readout locates between 16 and 24 is bracketed by two points, not resolved. `p = 20`
  (ceiling 12) is the obvious missing arm and was not run.
* **Only the leading direction per chain, in the `d_eff` column.** Per-chain ER is
  1.0-4.0, so chains deliver a little more than one direction; `joint_energy` and
  `joint_unit` use the full operators and are reported beside it for that reason, but
  `d_eff` itself discards everything below each chain's top singular vector.
* **Rank tolerance.** `RANK_RTOL = 1e-8` on the leading singular value is a stated
  choice, not a derived one; the `rank` fields inherit it. `d_eff` does not — it needs
  no tolerance, which is why it leads.
* **Nothing about the other cells.** Only rim-to-apex chains, only at the 8 apex
  cells `t1_apex` returns. Junctions and relays are untouched.
* **No architecture was changed.** Per #569's notes: an instrument, a reading, a
  readout.
