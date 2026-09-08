# T3 (#524): the canonical table on the full dome

**Surface.** Full dome (`DEFAULT_SPEC`, forward normalisation, `interior_m = 3`,
`boundary_m = 4`), `map/cold-start` with `main` merged, frozen world, no induced
activity at any point. 100k ticks with 30k printed beside it; seeds 42/43/44.
Aggregation: a class figure is the median over its cells per seed; published as the
mean of the per-seed medians ± their standard deviation across seeds.

- **baseline: ρ=1 off, c=1.0** — seeds 42, 43, 44; deepest ticks 100000, 100000, 100000
- **T1's winner: ρ=1 on, c=0.1** — seeds 42, 43, 44; deepest ticks 100000, 100000, 100000

`*` = still in flight.

## The two clauses

Read at **100,000 ticks**, the deepest horizon both arms reached (#178: horizons are never pooled).

### baseline: ρ=1 off, c=1.0 (3 seeds)

- **(1)** apex ρ **0.744 ± 0.021** against core L3–L6 **0.976 ± 0.002** (gap +0.232, larger spread 0.021) → **FAILS**
- **(2a)** arm travel per tick, last window **2.664e-04 ± 2.596e-04**; literally > 0: **yes**; against the frozen baseline's own stopped-arm level 2.664e-04 ± 2.596e-04 (T2's comparator): **not above it — the arm is parked**
- **(2b)** composed rim-to-apex ER **1.000 ± 0.000** (max over chains 1.171) against the bar 1.5 → **FAILS**

### T1's winner: ρ=1 on, c=0.1 (3 seeds)

- **(1)** apex ρ **0.959 ± 0.005** against core L3–L6 **0.992 ± 0.000** (gap +0.033, larger spread 0.005) → **FAILS**
- **(2a)** arm travel per tick, last window **9.347e-05 ± 1.322e-04**; literally > 0: **yes**; against the frozen baseline's own stopped-arm level 2.664e-04 ± 2.596e-04 (T2's comparator): **not above it — the arm is parked**
- **(2b)** composed rim-to-apex ER **1.000 ± 0.000** (max over chains 1.641) against the bar 1.5 → **FAILS**

## Retention: `ρ(K)` and `modes_retaining`, every class

### 30,000 ticks

| class | baseline: ρ=1 off, c=1.0 — ρ used | T1's winner: ρ=1 on, c=0.1 — ρ used | baseline: ρ=1 off, c=1.0 — modes | T1's winner: ρ=1 on, c=0.1 — modes |
| --- | --- | --- | --- | --- |
| apex (core, drive-adjacent; L7, 8) | 0.815 ± 0.008 | 0.973 ± 0.012 | 11.0 ± 0.0 | 12.0 ± 0.0 |
| core L3–L6 (52) | 0.978 ± 0.003 | 0.992 ± 0.001 | 12.0 ± 0.0 | 12.0 ± 0.0 |
| vision L1 (64) | 0.978 ± 0.007 | 0.993 ± 0.001 | 12.0 ± 0.0 | 12.0 ± 0.0 |
| somatomotor L1 boundary-adjacent (6) | 0.973 ± 0.014 | 0.991 ± 0.002 | 12.0 ± 0.0 | 12.0 ± 0.0 |

| reading | baseline: ρ=1 off, c=1.0 | T1's winner: ρ=1 on, c=0.1 |
| --- | --- | --- |
| core L3–L6 − apex gap | 0.163 ± 0.010 | 0.020 ± 0.013 |
| ρ raw, apex | 1.114 ± 0.038 | 1.079 ± 0.033 |
| dead cells (modes = 0) | 0.0 ± 0.0 | 0.0 ± 0.0 |

### 100,000 ticks

| class | baseline: ρ=1 off, c=1.0 — ρ used | T1's winner: ρ=1 on, c=0.1 — ρ used | baseline: ρ=1 off, c=1.0 — modes | T1's winner: ρ=1 on, c=0.1 — modes |
| --- | --- | --- | --- | --- |
| apex (core, drive-adjacent; L7, 8) | 0.744 ± 0.021 | 0.959 ± 0.005 | 11.0 ± 0.0 | 12.0 ± 0.0 |
| core L3–L6 (52) | 0.976 ± 0.002 | 0.992 ± 0.000 | 12.0 ± 0.0 | 12.0 ± 0.0 |
| vision L1 (64) | 0.963 ± 0.008 | 0.985 ± 0.003 | 11.0 ± 0.0 | 12.0 ± 0.0 |
| somatomotor L1 boundary-adjacent (6) | 0.833 ± 0.189 | 0.991 ± 0.002 | 11.0 ± 1.1 | 12.0 ± 0.0 |

| reading | baseline: ρ=1 off, c=1.0 | T1's winner: ρ=1 on, c=0.1 |
| --- | --- | --- |
| core L3–L6 − apex gap | 0.232 ± 0.023 | 0.033 ± 0.006 |
| ρ raw, apex | 1.122 ± 0.010 | 1.085 ± 0.034 |
| dead cells (modes = 0) | 0.0 ± 0.0 | 0.0 ± 0.0 |

## Per-cell excitation rank (participation ratio of the evidence stream)

### 30,000 ticks

| class | baseline: ρ=1 off, c=1.0 — PR | T1's winner: ρ=1 on, c=0.1 — PR | baseline: ρ=1 off, c=1.0 — PR centred | T1's winner: ρ=1 on, c=0.1 — PR centred |
| --- | --- | --- | --- | --- |
| apex (core, drive-adjacent; L7, 8) | 1.001 ± 0.001 | 1.000 ± 0.000 | 1.034 ± 0.007 | 1.808 ± 0.335 |
| core L3–L6 (52) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.487 ± 0.053 | 2.101 ± 0.253 |
| vision L1 (64) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.395 ± 0.078 | 1.462 ± 0.212 |
| somatomotor L1 boundary-adjacent (6) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.553 ± 0.287 | 2.227 ± 0.855 |

### 100,000 ticks

| class | baseline: ρ=1 off, c=1.0 — PR | T1's winner: ρ=1 on, c=0.1 — PR | baseline: ρ=1 off, c=1.0 — PR centred | T1's winner: ρ=1 on, c=0.1 — PR centred |
| --- | --- | --- | --- | --- |
| apex (core, drive-adjacent; L7, 8) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.067 ± 0.038 | 2.236 ± 0.287 |
| core L3–L6 (52) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.987 ± 0.168 | 2.669 ± 0.298 |
| vision L1 (64) | 1.000 ± 0.000 | 1.000 ± 0.000 | 2.177 ± 0.219 | 2.386 ± 0.385 |
| somatomotor L1 boundary-adjacent (6) | 1.001 ± 0.001 | 1.007 ± 0.010 | 2.171 ± 0.892 | 2.944 ± 0.952 |

## Done-when (2): composed rim-to-apex effective rank, and arm travel

Seven hops on the full dome (#436/#497's construction: one chain per sensorimotor rim
cell, the graph's own shortest edge path to an apex cell, composed, participation ratio
of the spectrum). The bar is **> 1.5**. Induced activity is identically zero in
both arms, so *annealed to zero* is the condition throughout, not a phase boundary.

Composed rank **at construction**, before a tick is run, by rim kind:

| rim kind | chains | median | p10 | p90 |
| --- | --- | --- | --- | --- |
| actuator | 1 | 1.074 | 1.074 | 1.074 |
| patch | 256 | 1.025 | 1.000 | 1.309 |
| proprioceptive | 3 | 1.080 | 1.017 | 1.338 |
| touch | 3 | 1.022 | 1.020 | 1.034 |

| reading | baseline: ρ=1 off, c=1.0 | T1's winner: ρ=1 on, c=0.1 |
| --- | --- | --- |
| composed ER, median over chains @ 30,000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| composed ER, median over chains @ 100,000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| composed ER, max over chains @ 30,000 | 1.185 ± 0.111 | 1.429 ± 0.356 |
| composed ER, max over chains @ 100,000 | 1.171 ± 0.188 | 1.641 ± 0.210 |
| arm travel this window @ 30,000 | 0.5042 ± 0.6982 | 0.0000 ± 0.0000 |
| arm travel this window @ 100,000 | 18.6471 ± 18.1700 | 6.5426 ± 9.2527 |
| arm travel per tick @ 30,000 | 0.00005042 ± 0.00006982 | 0.00000000 ± 0.00000000 |
| arm travel per tick @ 100,000 | 0.00026639 ± 0.00025957 | 0.00009347 ± 0.00013218 |

## The persistent error, and P2's separation at 100k

[Ledger row 1](https://github.com/NGL321/patchworks/issues/520) found excitation rank as
pre-registered does not discriminate on this surface (uncentred 1.00 nearly everywhere) while
`log ‖ē‖` reached R² 0.41–0.44 at 20k, and proposed: *read the persistent error beside it as the
coherence variable the mechanism actually names, and re-read P2's separation at T3's 100k, where
column identity itself was 0.63–0.68.* This is that read.

| class | baseline: ρ=1 off, c=1.0 — ‖ē‖ @100k | T1's winner: ρ=1 on, c=0.1 — ‖ē‖ @100k | baseline: ρ=1 off, c=1.0 — ē direction stability @100k | T1's winner: ρ=1 on, c=0.1 — ē direction stability @100k |
| --- | --- | --- | --- | --- |
| apex (core, drive-adjacent; L7, 8) | 0.001126 ± 0.000300 | 0.000390 ± 0.000043 | 0.952 ± 0.020 | 0.890 ± 0.012 |
| core L3–L6 (52) | 0.000112 ± 0.000021 | 0.000211 ± 0.000013 | 0.394 ± 0.089 | 0.396 ± 0.111 |
| vision L1 (64) | 0.000310 ± 0.000029 | 0.000539 ± 0.000064 | 0.724 ± 0.029 | 0.619 ± 0.041 |
| somatomotor L1 boundary-adjacent (6) | 0.001400 ± 0.001802 | 0.000195 ± 0.000066 | 0.414 ± 0.301 | 0.555 ± 0.156 |

R² on `log ρ_used` across all predicting cells. Column identity is #477's `apex + somatomotor`
two dummies, and the 4-way one-hot beside it; every figure is the mean over seeds ± spread.

| design | baseline: ρ=1 off, c=1.0 @30k | T1's winner: ρ=1 on, c=0.1 @30k | baseline: ρ=1 off, c=1.0 @100k | T1's winner: ρ=1 on, c=0.1 @100k |
| --- | --- | --- | --- | --- |
| column (#477: apex + soma) | 0.281 ± 0.031 | 0.080 ± 0.080 | 0.473 ± 0.114 | 0.089 ± 0.072 |
| column (4-way one-hot) | 0.306 ± 0.039 | 0.100 ± 0.076 | 0.507 ± 0.097 | 0.147 ± 0.058 |
| PR total (uncentred) | 0.063 ± 0.086 | 0.006 ± 0.004 | 0.071 ± 0.099 | 0.013 ± 0.013 |
| log PR total, centred | 0.055 ± 0.029 | 0.024 ± 0.018 | 0.149 ± 0.044 | 0.053 ± 0.049 |
| **log ‖ē‖** | 0.356 ± 0.030 | 0.149 ± 0.032 | 0.429 ± 0.046 | 0.223 ± 0.026 |
| ē direction stability | 0.159 ± 0.021 | 0.086 ± 0.048 | 0.161 ± 0.043 | 0.123 ± 0.065 |
| ē stability + log ‖ē‖ | 0.372 ± 0.020 | 0.175 ± 0.050 | 0.433 ± 0.048 | 0.244 ± 0.031 |

## Drive edges, and disagreement energy beside per-edge effective rank

#488's relative form, `|d| / (|a| + |b|)`: a nulled edge reads 0 and a fixed 2x scale
mismatch reads 1/3. This is what the `ρ = 1` arm is for, read on this surface rather
than assumed from the shallow one (where T1 found #488's 1/3 was **not** present, 0.01–0.02).

| reading | baseline: ρ=1 off, c=1.0 | T1's winner: ρ=1 on, c=0.1 |
| --- | --- | --- |
| drive-edge relative disagreement, median @ 30,000 | 0.0101 ± 0.0026 | 0.0183 ± 0.0016 |
| drive-edge relative disagreement, median @ 100,000 | 0.0064 ± 0.0018 | 0.0130 ± 0.0034 |

### Per-edge, by stratum, at 30,000 ticks

| stratum | m | baseline: ρ=1 off, c=1.0 — dis. energy | T1's winner: ρ=1 on, c=0.1 — dis. energy | baseline: ρ=1 off, c=1.0 — map ER | T1's winner: ρ=1 on, c=0.1 — map ER | baseline: ρ=1 off, c=1.0 — PR centred | T1's winner: ρ=1 on, c=0.1 — PR centred |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1-L1 | 3 | 0.000017 ± 0.000009 | 0.000149 ± 0.000159 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.505 ± 0.102 | 1.390 ± 0.102 |
| L1-core | 3 | 0.000029 ± 0.000024 | 0.000179 ± 0.000178 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.593 ± 0.154 | 1.462 ± 0.050 |
| L3-L3 | 3 | 0.000047 ± 0.000041 | 0.000137 ± 0.000094 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.775 ± 0.071 | 1.779 ± 0.216 |
| L3-L4 | 3 | 0.000076 ± 0.000083 | 0.000265 ± 0.000300 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.730 ± 0.196 | 1.804 ± 0.257 |
| L4-L4 | 3 | 0.000074 ± 0.000075 | 0.000436 ± 0.000565 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.735 ± 0.232 | 1.751 ± 0.218 |
| L4-L5 | 3 | 0.000054 ± 0.000048 | 0.000377 ± 0.000464 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.732 ± 0.215 | 1.896 ± 0.205 |
| L5-L5 | 3 | 0.000038 ± 0.000034 | 0.000203 ± 0.000236 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.827 ± 0.192 | 1.906 ± 0.283 |
| L5-L6 | 3 | 0.000026 ± 0.000015 | 0.000203 ± 0.000257 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.651 ± 0.088 | 1.910 ± 0.092 |
| L6-L6 | 3 | 0.000021 ± 0.000009 | 0.000096 ± 0.000091 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.427 ± 0.085 | 1.737 ± 0.147 |
| L6-L7 | 3 | 0.000065 ± 0.000015 | 0.000198 ± 0.000173 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.173 ± 0.008 | 1.764 ± 0.042 |
| L7-L7 | 3 | 0.000087 ± 0.000039 | 0.000237 ± 0.000278 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.158 ± 0.084 | 1.509 ± 0.083 |
| core-apex | 3 | 0.000044 ± 0.000046 | 0.000231 ± 0.000261 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.583 ± 0.216 | 1.631 ± 0.185 |
| core-core | 3 | 0.000071 ± 0.000086 | 0.000246 ± 0.000274 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.769 ± 0.259 | 1.631 ± 0.175 |
| drive | 1 | 0.000986 ± 0.000216 | 0.002894 ± 0.000471 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| motor-rim | 4 | 0.000064 ± 0.000023 | 0.000129 ± 0.000122 | 4.000 ± 0.000 | 4.000 ± 0.000 | 1.329 ± 0.085 | 1.948 ± 0.518 |
| sensory-rim | 4 | 0.000085 ± 0.000018 | 0.000373 ± 0.000230 | 3.987 ± 0.005 | 3.988 ± 0.003 | 1.758 ± 0.088 | 1.805 ± 0.030 |

### Per-edge, by stratum, at 100,000 ticks

| stratum | m | baseline: ρ=1 off, c=1.0 — dis. energy | T1's winner: ρ=1 on, c=0.1 — dis. energy | baseline: ρ=1 off, c=1.0 — map ER | T1's winner: ρ=1 on, c=0.1 — map ER | baseline: ρ=1 off, c=1.0 — PR centred | T1's winner: ρ=1 on, c=0.1 — PR centred |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1-L1 | 3 | 0.000015 ± 0.000000 | 0.000104 ± 0.000047 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.731 ± 0.038 | 1.465 ± 0.040 |
| L1-core | 3 | 0.000018 ± 0.000003 | 0.000122 ± 0.000043 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.774 ± 0.083 | 1.519 ± 0.062 |
| L3-L3 | 3 | 0.000034 ± 0.000022 | 0.000108 ± 0.000037 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.598 ± 0.046 | 1.518 ± 0.154 |
| L3-L4 | 3 | 0.000034 ± 0.000024 | 0.000160 ± 0.000063 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.595 ± 0.073 | 1.615 ± 0.115 |
| L4-L4 | 3 | 0.000054 ± 0.000037 | 0.000210 ± 0.000145 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.623 ± 0.045 | 1.593 ± 0.127 |
| L4-L5 | 3 | 0.000033 ± 0.000021 | 0.000174 ± 0.000095 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.578 ± 0.087 | 1.571 ± 0.061 |
| L5-L5 | 3 | 0.000040 ± 0.000041 | 0.000146 ± 0.000123 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.670 ± 0.085 | 1.665 ± 0.182 |
| L5-L6 | 3 | 0.000029 ± 0.000027 | 0.000093 ± 0.000073 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.421 ± 0.046 | 1.571 ± 0.193 |
| L6-L6 | 3 | 0.000015 ± 0.000009 | 0.000039 ± 0.000026 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.463 ± 0.020 | 1.737 ± 0.179 |
| L6-L7 | 3 | 0.000079 ± 0.000025 | 0.000115 ± 0.000079 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.296 ± 0.125 | 1.891 ± 0.188 |
| L7-L7 | 3 | 0.000027 ± 0.000018 | 0.000078 ± 0.000063 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.149 ± 0.071 | 1.713 ± 0.063 |
| core-apex | 3 | 0.000033 ± 0.000022 | 0.000131 ± 0.000048 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.848 ± 0.113 | 1.470 ± 0.030 |
| core-core | 3 | 0.000026 ± 0.000004 | 0.000157 ± 0.000094 | 3.000 ± 0.000 | 3.000 ± 0.000 | 1.833 ± 0.097 | 1.707 ± 0.124 |
| drive | 1 | 0.000273 ± 0.000051 | 0.001775 ± 0.000373 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| motor-rim | 4 | 0.000308 ± 0.000099 | 0.000284 ± 0.000279 | 4.000 ± 0.000 | 4.000 ± 0.000 | 1.538 ± 0.185 | 1.864 ± 0.433 |
| sensory-rim | 4 | 0.000097 ± 0.000010 | 0.000436 ± 0.000063 | 3.990 ± 0.003 | 3.987 ± 0.004 | 2.044 ± 0.027 | 1.915 ± 0.007 |

## Not read here

**ADR-0026's conduction ratio, inbound and outbound.** It is Done-when (3), the map's
clause for [T4](https://github.com/NGL321/patchworks/issues/525) — which runs on this
same full dome and hands the reading to #127 as the first on a world that varies. T2's
paired-counterfactual fork (`prototypes/cold-start/T2/run.py::reach_fork`) is the
instrument and is reusable as it stands. A scoping call, named rather than silent.

