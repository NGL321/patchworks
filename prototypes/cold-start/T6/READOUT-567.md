# B18 (#567): B16's traffic rank, read centred

**Answer: the artifact is real and larger than expected — the centred excess over
one runs 19–137× the uncentred excess — but it does not overturn B16's
direction. A cell's state is not rank one; it is rank ~1.14–1.18 against an
ambient of 24, and still strikingly concentrated. What does not survive is the
*number*, the *tail*, and the assumption that the two statistics rank cells
alike: they correlate at +0.21 and +0.07.**

Instrument `b18_centred.py`; analysis `b18_analyse.py`; raw
`567-centred-{arm}-seed42-20000.json`, `567-centred-analysis.json`.
Arms `reserve` (`p = 8`) and `reserve_p16`, seed 42, 20k, 142 relay cells,
ambient `k_v` = 24 and 16.

---

## 0. The instrument is B16's, with one line added

`CentredTrafficRecorder` accumulates the **first** moment alongside the second,
over the same tensor (`sh.stalks[layout.slice(c)]` restricted to the cell's
shared structural mask), the same per-checkpoint window, and the same reduction
`(Σλ)²/Σλ²`. `Cov = M/N − μμᵀ`. Nothing else differs except that B16's
principal-angle and composed-rank reads are dropped, which B16 already has and
which is what makes this cheap.

**Replication check, and an honest limit.** The uncentred column reproduces
B16's published medians *exactly* at 100, 300, 1000 and 20000 ticks, and drifts
in the fifth decimal at 3000 (1.0022 against 1.0021) and the third at 10000
(**1.0069 against 1.0054**). The drift is almost certainly because this loop does
not call `recorder.observe()` on B16's `EdgeRecorder`, which perturbs the RNG
stream. Same instrument, same regime, **not bit-identical past a few thousand
ticks** — an earlier claim of exact replication throughout was too strong.

## 1. The correction, per window

`reserve` (`p = 8`), median over 142 relay cells:

| window | er uncentred | er **centred** | excess unc. | excess **cen.** | **ratio** | `mean_share` |
|---|---|---|---|---|---|---|
| 100 | 1.0788 | **2.5235** | 0.0788 | 1.5235 | **19×** | 0.9540 |
| 300 | 1.0021 | **1.2879** | 0.0021 | 0.2879 | **137×** | 0.9986 |
| 1000 | 1.0022 | **1.1386** | 0.0022 | 0.1386 | **63×** | 0.9985 |
| 3000 | 1.0022 | **1.1293** | 0.0022 | 0.1293 | **59×** | 0.9983 |
| 10000 | 1.0069 | **1.1290** | 0.0069 | 0.1290 | **19×** | 0.9958 |
| 20000 | 1.0056 | **1.1785** | 0.0056 | 0.1785 | **32×** | 0.9966 |

`reserve_p16` runs the same shape: 2.2951 / 1.2702 / 1.0922 / 1.0848 / 1.1225 /
**1.1393**, against uncentred 1.0898 / 1.0011 / 1.0022 / 1.0026 / 1.0050 /
1.0051.

**The mean carries 99.6–99.9% of the energy at every window past the first.** So
the uncentred statistic was measuring the baseline, exactly as suspected.

**But the direction survives.** Centred, the state still collapses from 2.52 at
100 ticks to ~1.13 by 3000 and stays there. Against an ambient of 24 that is
very concentrated. *"Rank one"* is wrong as stated; *"concentrated"* is right.

## 2. The two statistics do not rank cells alike

The single most decisive number on this ticket:

    corr(er_uncentred, er_centred)  =  +0.213  (p = 8)
                                       +0.067  (p = 16)

**This is not a bias to correct for. The uncentred reading is a substantially
different measurement**, and any per-cell conclusion drawn from it — which is
what B16's cross-cell correlations are — is drawn from a quantity that does not
track the one intended.

What the uncentred reading actually tracks is the mean/variation ratio:

| | corr with `er_unc` | corr with `er_cen` |
|---|---|---|
| `tr_cov` | **+0.793** | +0.078 |
| `mu_energy` | −0.410 | −0.291 |

## 3. The tail cells are small-baseline cells, not rich cells (item 3)

B16's tail — cells reading 1.5–1.9 uncentred where the median reads 1.002 — has
exactly one thing in common, and it is not dynamics:

| cell | level | degree | `er_unc` | `er_cen` | `mean_share` |
|---|---|---|---|---|---|
| 272 | 1 | 9 | 2.1522 | 1.8059 | **0.553** |
| 311 | 1 | 8 | 1.9380 | 1.6009 | **0.647** |
| 332 | 1 | 7 | 1.6649 | 1.1237 | **0.658** |
| 340 | 2 | 8 | 1.5963 | 1.0263 | **0.750** |
| 325 | 1 | 8 | 1.5949 | 1.9241 | **0.769** |
| 271 | 1 | 8 | 1.5379 | 1.6578 | **0.784** |
| 264 | 1 | 8 | 1.4366 | 1.4277 | **0.808** |
| 320 | 1 | 8 | 1.3831 | 1.0708 | **0.832** |

**Tail `mean_share` 0.725 against 0.991 for every other cell.** They are the
cells with the *smallest baselines* — which is to say, the cells where the
uncentred statistic was least corrupted and therefore came closest to working.
They are not the dome's rich cells: 332 and 340 read 1.12 and **1.03** centred,
below the population median.

All but one sit at **level 1**, the rim-adjacent level, at degree 7–9.

## 4. Level structure is U-shaped, not monotone (item 4)

Centred median by level at 20k:

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| `p = 8` | 1.2128 | 1.1581 | 1.1504 | 1.1071 | 1.0977 | **1.2427** |
| `p = 16` | 1.2075 | 1.0839 | 1.0266 | 1.0372 | 1.1319 | **1.3441** |

State richness falls through the middle of the dome and **rises sharply at the
apex**, in both arms and more strongly at `p = 16` (1.027 → 1.344). This is
invisible to the uncentred statistic, which is flat at 1.001–1.014 across every
level.

It is a **finding for [B23](https://github.com/NGL321/patchworks/issues/572)**: there is level structure, and it does not run the
direction a hierarchy predicts.

Otherwise structure predicts very little about the centred reading — every
correlate is weak: level −0.19/−0.10, degree +0.14/+0.09, `m_sum` −0.23/−0.23,
`m_max` −0.12/−0.00.

## 5. `p = 16` has a *less* rich state than `p = 8`

At 20k the centred median is **1.1393 at `p = 16` against 1.1785 at `p = 8`**,
and it is lower at levels 2, 3 and 4 (1.084/1.027/1.037 against
1.158/1.150/1.107). Over the same arms, composed rim-to-apex operator rank is
**2.9171 against 1.4386** — roughly double.

**So the arm that nearly doubles the operator's rank carries slightly less state
variation through the middle of the dome.** That is independent support for
[B17](https://github.com/NGL321/patchworks/issues/565)'s finding that `p` buys operator rank without buying content, arriving
from the state side rather than the parameter side.

## 6. What this does and does not settle

**Settles**: B16's *number* is wrong by one to two orders of magnitude; its
*tail* is a small-baseline artifact and not a set of rich cells; its per-cell
correlations are computed on a statistic that correlates at +0.07..+0.21 with
the intended one; and B16's qualitative claim — the cell's own dynamics are
strongly concentrated — **survives**, at 1.14–1.18 rather than 1.002.

**Does not settle** the specialist-versus-bottleneck fork, which needs the
emitted rank beside the state rank and is
[B19](https://github.com/NGL321/patchworks/issues/568)'s. A state rank of 1.18
against an ambient of 24 is compatible with both a genuinely narrow cell and a
cell whose richness never reaches its edges.

## 7. Limits

- **One seed.** Seed 42 only, both arms.
- **20k, not 100k.** B13's arms run to 100k; the centred reading past 20k is
  unmeasured.
- **Not bit-identical to B16** past ~3000 ticks (§0).
- The **pooled** figure is reported but is window-noisy (12.6–39.0 across
  checkpoints on one arm) and no conclusion is drawn from it.
- `mean_share` is a scalar summary of a spectral relationship; a cell with a
  large mean *along a direction its variation also uses* is not distinguished
  from one whose mean is orthogonal to its variation. Not needed for this
  ticket's question, but it is not a complete description.
