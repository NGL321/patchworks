# T2 readout — surface `cd44cea 2026-09-05 Build T2's rig (#522): the induced-activity sweep on the baseline build` on `cs/522`
band: forward normalisation in CellOperators.used (#466, PR #513); (interior_m, boundary_m) = (3, 4); shallow dome core_degree 7, 358 cells / 507 edges; **baseline build** (ρ=1 off, c=1). Induced phase T_b, annealed `A(t) = A0 * max(0, 1 - t / T_b)`, then 30000 ticks with the world arranged and the supply off.

## Runs

| condition | T_b | seeds | complete | min/seed | A₀ |
|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 5000 | [42, 43, 44] | [42, 43, 44] | [23.9, 22.7, 23.1] | 1.0000 |
| B — sensory, large, ordered (the wave) | 5000 | [42, 43, 44] | [42, 43, 44] | [23.7, 22.6, 22.8] | 1.0000 |
| C — sensory, large, shuffled | 5000 | [42, 43, 44] | [42, 43, 44] | [23.7, 22.6, 22.9] | 1.0000 |
| D — sensory, small, ordered | 5000 | [42, 43, 44] | [42, 43, 44] | [23.7, 22.6, 22.9] | 0.2916 |
| **Z — no supply at all (control)** | 5000 | [42, 43, 44] | [42, 43, 44] | [22.6, 22.7, 22.8] | 0.0000 |
| A — motor, small, shuffled (today's babble) | 20000 | [42, 43, 44] | [42, 43, 44] | [33.2, 32.9, 41.5] | 1.0000 |
| B — sensory, large, ordered (the wave) | 20000 | [42, 43, 44] | [42, 43, 44] | [32.8, 32.3, 39.5] | 1.0000 |
| C — sensory, large, shuffled | 20000 | [42, 43, 44] | [42, 43, 44] | [32.7, 32.4, 39.9] | 1.0000 |
| D — sensory, small, ordered | 20000 | [42, 43, 44] | [42, 43, 44] | [32.6, 32.2, 40.1] | 0.2916 |
| **Z — no supply at all (control)** | 20000 | [42, 43, 44] | [42, 43, 44] | [37.0, 37.2, 37.3] | 0.0000 |

**Derived constants.** Babble correlation time τ = median `world_loop(c)` over the six L1 somatomotor cells = **3.5** ticks ([3, 4, 3, 4, 3, 4]), so φ = exp(−1/τ); the front advances one patch per **6.0** ticks, the L1 vision cells' median `world_loop(c)` ({'5': 12, '6': 27, '7': 25}). 
The sensory wall's **small** amplitude is calibrated, not chosen: motor babble at the bound moves the render with RMS **0.0799** stalk units, and the front at peak 1 has time-averaged RMS 0.2739, so D's peak is **0.2916** against B and C's 1.0 — D vs A is a structure comparison at matched sensory energy.


---

# T_b = 5000

## Reach — ADR-0026's paired counterfactual, rules off, at the deepest in-phase checkpoint with the supply on

Peak paired private-feature deviation over `eps_f32·‖state‖` at the same tick, median over the class's cells; mean of per-seed medians ± spread. **A ratio above 1 is a deviation that arrived**; the magnitude is the reading.

| condition | A at the fork | apex (core, drive-adjacent; 8) | core (one level; 16) | vision L1 (64) | somatomotor L1 boundary-adjacent (6) |
|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 0.6000 | **3.09** ± 0.75 | **2.64** ± 0.35 | **1.23** ± 0.089 | **1.24e+03** ± 1.4e+03 |
| B — sensory, large, ordered (the wave) | 0.6000 | **90.7** ± 9.9 | **2.01e+03** ± 2.9e+02 | **1.52e+04** ± 1.4e+03 | **16.3** ± 8.9 |
| C — sensory, large, shuffled | 0.6000 | **146** ± 26 | **3.15e+03** ± 4.9e+02 | **2.32e+04** ± 7.9e+02 | **45.7** ± 10 |
| D — sensory, small, ordered | 0.1750 | **38** ± 9.9 | **823** ± 1.3e+02 | **5.92e+03** ± 1.3e+03 | **8.02** ± 3.2 |
| **Z — no supply at all (control)** | — | — | — | — | — |

**Reach through the phase**, apex peak/floor at every in-phase fork. The amplitude falls on the anneal *and* the surface trains, so a fall here is the two together; what it shows is that reach is not a property of the supply alone.

| condition | 100 | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 10.1 | 8.85 | 4.83 | 3.45 | 3.09 |
| B — sensory, large, ordered (the wave) | 50 | 68.9 | 95.5 | 103 | 90.7 |
| C — sensory, large, shuffled | 95.4 | 142 | 181 | 175 | 146 |
| D — sensory, small, ordered | 16.1 | 17.9 | 28.4 | 37.8 | 38 |
| **Z — no supply at all (control)** | — | — | — | — | — |

And the amplitude at each of those forks (the anneal, so the two can be told apart):

| A(t)/A₀ | 0.980 | 0.960 | 0.900 | 0.800 | 0.600 |
|---|---|---|---|---|---|

Fraction of a class's cells whose peak cleared the floor, and the apex's peak in absolute terms:

| condition | apex above-floor fraction | apex peak ‖Δ‖ | apex floor | apex peak tick (of 64) |
|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 1.000 | 2.58e-06 | 7.58e-07 | 39.3 |
| B — sensory, large, ordered (the wave) | 1.000 | 8.13e-05 | 7.61e-07 | 18.7 |
| C — sensory, large, shuffled | 1.000 | 1.36e-04 | 8.03e-07 | 24.0 |
| D — sensory, small, ordered | 1.000 | 3.26e-05 | 7.63e-07 | 23.0 |

## Rank — per-edge excitation rank at the deepest interior edges (core–apex, m_e = 3), during the phase

Median over the 32 core–apex edges' two ends, both forms published (ledger row 1: the uncentred form is DC-dominated on this surface and the centred one is reported beside it, never in place of it).

| condition | uncentred PR | centred PR | against m_e | disagreement energy | map effective rank |
|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 1.001 ± 0.000 | **1.249** ± 0.034 | 3 | 4.18e-04 | 3.000 |
| B — sensory, large, ordered (the wave) | 1.001 ± 0.000 | **1.202** ± 0.017 | 3 | 3.87e-04 | 3.000 |
| C — sensory, large, shuffled | 1.002 ± 0.000 | **1.258** ± 0.020 | 3 | 1.13e-03 | 3.000 |
| D — sensory, small, ordered | 1.001 ± 0.000 | **1.214** ± 0.031 | 3 | 3.29e-04 | 3.000 |
| **Z — no supply at all (control)** | 1.001 ± 0.000 | **1.211** ± 0.035 | 3 | 4.30e-04 | 3.000 |

The same, at the shallower strata, centred (so the taper by depth is visible):

| condition | sensory-rim | L1-L1 | L1-core | core-core | core-apex | drive |
|---|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 1.385 ± 0.021 | 1.218 ± 0.005 | 1.223 ± 0.023 | 1.338 ± 0.044 | 1.249 ± 0.034 | 1.000 ± 0.000 |
| B — sensory, large, ordered (the wave) | 1.506 ± 0.024 | 1.211 ± 0.033 | 1.237 ± 0.034 | 1.260 ± 0.075 | 1.202 ± 0.017 | 1.000 ± 0.000 |
| C — sensory, large, shuffled | 1.363 ± 0.011 | 1.225 ± 0.012 | 1.168 ± 0.021 | 1.285 ± 0.083 | 1.258 ± 0.020 | 1.000 ± 0.000 |
| D — sensory, small, ordered | 1.443 ± 0.023 | 1.251 ± 0.037 | 1.275 ± 0.026 | 1.253 ± 0.026 | 1.214 ± 0.031 | 1.000 ± 0.000 |
| **Z — no supply at all (control)** | 1.407 ± 0.041 | 1.217 ± 0.024 | 1.232 ± 0.034 | 1.294 ± 0.032 | 1.211 ± 0.035 | 1.000 ± 0.000 |

## Priming — composed rim-to-apex effective rank at +30k, world arranged, supply off

Median over the 256 **patch** chains (the graph's own shortest edge path from each rim cell to an apex cell, composed); the other rim strata beside it, never averaged in (#181). The pre-registered comparison is each ordered condition against its shuffled control: **B vs C** and **D vs A**.

| condition | at construction | at end of phase | +20k | **+30k** | spread |
|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 1.2979 | 1.2037 | 1.0878 | **1.0825** | 0.0026 |
| B — sensory, large, ordered (the wave) | 1.2979 | 1.1854 | 1.0830 | **1.0844** | 0.0062 |
| C — sensory, large, shuffled | 1.2979 | 1.3454 | 1.1200 | **1.1161** | 0.0148 |
| D — sensory, small, ordered | 1.2979 | 1.1928 | 1.0901 | **1.0777** | 0.0012 |
| **Z — no supply at all (control)** | 1.2979 | 1.1964 | 1.0909 | **1.0845** | 0.0023 |

**The pre-registered contrasts**, at +30k:

| ordered | control | ordered ER | control ER | Δ | spread | beats control |
|---|---|---|---|---|---|---|
| B — sensory, large, ordered (the wave) | C — sensory, large, shuffled | 1.0844 | 1.1161 | -0.0317 | 0.0148 | **False** |
| D — sensory, small, ordered | A — motor, small, shuffled (today's babble) | 1.0777 | 1.0825 | -0.0048 | 0.0026 | **False** |

**Against the zero-supply control**, which is what says whether *anything* was laid down rather than which member laid down most:

| condition | ER at +30k | Z at +30k | Δ vs Z | spread | above Z |
|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 1.0825 | 1.0845 | -0.0020 | 0.0026 | **False** |
| B — sensory, large, ordered (the wave) | 1.0844 | 1.0845 | -0.0001 | 0.0062 | **False** |
| C — sensory, large, shuffled | 1.1161 | 1.0845 | +0.0316 | 0.0148 | **True** |
| D — sensory, small, ordered | 1.0777 | 1.0845 | -0.0068 | 0.0023 | **False** |

The other rim strata at +30k, for the record:

| condition | patch | proprioceptive | touch | actuator |
|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 1.0825 ± 0.0026 | 1.0744 ± 0.0379 | 1.5757 ± 0.3617 | 1.1475 ± 0.0897 |
| B — sensory, large, ordered (the wave) | 1.0844 ± 0.0062 | 1.1137 ± 0.0647 | 1.5245 ± 0.1089 | 1.1677 ± 0.1760 |
| C — sensory, large, shuffled | 1.1161 ± 0.0148 | 1.1136 ± 0.0657 | 1.4992 ± 0.2671 | 1.0895 ± 0.0598 |
| D — sensory, small, ordered | 1.0777 ± 0.0012 | 1.1589 ± 0.0767 | 1.4300 ± 0.2300 | 1.1382 ± 0.1276 |
| **Z — no supply at all (control)** | 1.0845 ± 0.0023 | 1.1864 ± 0.0667 | 1.4718 ± 0.1693 | 1.1118 ± 0.0231 |

## Travel — arm travel per tick under the graph's own command, post-phase windows

T1's frozen baseline reads **3.48e-04** per tick in its last window (the arm at its stops), which is what *travel > 0* is read against rather than literal zero.

| condition | +1k | +5k | +10k | +20k | **+30k** | spread at +30k | above the frozen baseline |
|---|---|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 6.07e-07 | 1.59e-06 | 1.18e-06 | 2.12e-04 | 3.64e-04 | 2.73e-04 | **False** |
| B — sensory, large, ordered (the wave) | 7.86e-03 | 1.46e-05 | 1.76e-05 | 3.10e-06 | 1.47e-04 | 1.16e-04 | **False** |
| C — sensory, large, shuffled | 7.87e-03 | 2.58e-06 | 3.52e-06 | 3.35e-06 | 2.76e-04 | 2.32e-04 | **False** |
| D — sensory, small, ordered | 8.12e-03 | 2.83e-05 | 2.07e-04 | 5.54e-04 | 4.95e-04 | 5.49e-04 | **False** |
| **Z — no supply at all (control)** | 1.31e-05 | 4.27e-04 | 3.89e-04 | 4.16e-04 | 3.82e-05 | 5.21e-05 | **False** |

## Retention guard — apex and soma ρ(K) at +30k, not below T1's frozen baseline beyond spread

T1's baseline is 30k ticks from construction; a T2 run at +30k has run 35000 in total, so the comparator is matched on **post-phase** ticks and not on total ticks. Stated rather than corrected: no frozen run exists at the longer horizon.

| condition | apex (core, drive-adjacent; 8) | core (one level; 16) | vision L1 (64) | somatomotor L1 boundary-adjacent (6) | guard |
|---|---|---|---|---|---|
| **T1 frozen baseline @30k** | 0.808 ± 0.059 | 0.971 ± 0.008 | 0.977 ± 0.003 | 0.884 ± 0.051 | — |
| A — motor, small, shuffled (today's babble) | 0.795 ± 0.060 (-0.013) | 0.961 ± 0.019 (-0.010) | 0.976 ± 0.001 (-0.001) | 0.818 ± 0.195 (-0.065) | **passes** |
| B — sensory, large, ordered (the wave) | 0.823 ± 0.029 (+0.015) | 0.973 ± 0.003 (+0.003) | 0.974 ± 0.000 (-0.002) | 0.902 ± 0.023 (+0.018) | **passes** |
| C — sensory, large, shuffled | 0.764 ± 0.032 (-0.044) | 0.896 ± 0.023 (-0.075 **FAIL**) | 0.779 ± 0.014 (-0.198 **FAIL**) | 0.907 ± 0.021 (+0.023) | **passes** |
| D — sensory, small, ordered | 0.811 ± 0.041 (+0.002) | 0.957 ± 0.021 (-0.014) | 0.975 ± 0.002 (-0.002) | 0.825 ± 0.120 (-0.059) | **passes** |
| **Z — no supply at all (control)** | 0.795 ± 0.053 (-0.013) | 0.971 ± 0.009 (+0.000) | 0.976 ± 0.002 (-0.001) | 0.909 ± 0.027 (+0.025) | **passes** |

**The same, against the zero-supply control instead of T1.** Z has run the identical number of total ticks, so this comparison carries none of the horizon mismatch above — the difference is the supply's and nothing else. This is the honest per-class read; the T1 table above is the pre-registered one, and both are published.

| condition | apex (core, drive-adjacent; 8) | core (one level; 16) | vision L1 (64) | somatomotor L1 boundary-adjacent (6) |
|---|---|---|---|---|
| **Z — no supply at all** | 0.795 ± 0.053 | 0.971 ± 0.009 | 0.976 ± 0.002 | 0.909 ± 0.027 |
| A — motor, small, shuffled (today's babble) | 0.795 (-0.000) | 0.961 (-0.010) | 0.976 (-0.000) | 0.818 (-0.090) |
| B — sensory, large, ordered (the wave) | 0.823 (+0.028) | 0.973 (+0.002) | 0.974 (-0.002 **below Z**) | 0.902 (-0.007) |
| C — sensory, large, shuffled | 0.764 (-0.031) | 0.896 (-0.075 **below Z**) | 0.779 (-0.198 **below Z**) | 0.907 (-0.001) |
| D — sensory, small, ordered | 0.811 (+0.016) | 0.957 (-0.014) | 0.975 (-0.001) | 0.825 (-0.084) |

Apex ρ(K) through the run, so the phase's own effect is visible beside the post-phase state:

| condition | end of phase | +1k | +5k | +10k | +20k | +30k |
|---|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 0.947 | 0.939 | 0.920 | 0.888 | 0.817 | 0.795 |
| B — sensory, large, ordered (the wave) | 0.948 | 0.942 | 0.920 | 0.894 | 0.848 | 0.823 |
| C — sensory, large, shuffled | 0.931 | 0.921 | 0.883 | 0.848 | 0.793 | 0.764 |
| D — sensory, small, ordered | 0.949 | 0.945 | 0.924 | 0.899 | 0.855 | 0.811 |
| **Z — no supply at all (control)** | 0.950 | 0.942 | 0.923 | 0.893 | 0.834 | 0.795 |

## Mechanism at the apex (T0's reads, +30k) — ledger row 1's coherence variable

| condition | ‖ē‖ | ē direction stability | ē share along the drive lane | dead cells |
|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 2.33e-03 | 0.986 | 0.904 | [0, 0, 0] |
| B — sensory, large, ordered (the wave) | 1.99e-03 | 0.989 | 0.936 | [0, 0, 0] |
| C — sensory, large, shuffled | 2.37e-03 | 0.991 | 0.927 | [0, 0, 0] |
| D — sensory, small, ordered | 2.39e-03 | 0.993 | 0.941 | [0, 1, 0] |
| **Z — no supply at all (control)** | 1.70e-03 | 0.991 | 0.948 | [0, 0, 0] |

## Branch table — which rows fired

| row | reading | consequence |
|---|---|---|
| none travels post-phase | **fired** | wave 3 skipped; T3 replicates T1's winner alone; **ledger row: curiosity drive owed** |
| B beats C beyond spread | did not fire (Δ -0.0317, spread 0.0148) | — |
| C beats A beyond spread and B ≈ C | did not fire (C−A +0.0336 / 0.0148; B−C -0.0317) | — |
| retention guard fails on the winning condition | did not fire (every condition passes) | — |


---

# T_b = 20000

## Reach — ADR-0026's paired counterfactual, rules off, at the deepest in-phase checkpoint with the supply on

Peak paired private-feature deviation over `eps_f32·‖state‖` at the same tick, median over the class's cells; mean of per-seed medians ± spread. **A ratio above 1 is a deviation that arrived**; the magnitude is the reading.

| condition | A at the fork | apex (core, drive-adjacent; 8) | core (one level; 16) | vision L1 (64) | somatomotor L1 boundary-adjacent (6) |
|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 0.5000 | **1.82** ± 1.5 | **1.67** ± 1.2 | **0.818** ± 0.6 | **38.2** ± 36 |
| B — sensory, large, ordered (the wave) | 0.5000 | **402** ± 1.4e+02 | **5.62e+03** ± 1.6e+03 | **5.03e+04** ± 1.2e+04 | **81.7** ± 25 |
| C — sensory, large, shuffled | 0.5000 | **402** ± 1e+02 | **4.2e+03** ± 3.3e+02 | **2.05e+04** ± 2e+03 | **116** ± 25 |
| D — sensory, small, ordered | 0.1458 | **110** ± 36 | **1.76e+03** ± 4.4e+02 | **1.47e+04** ± 3.2e+03 | **27.3** ± 10 |
| **Z — no supply at all (control)** | — | — | — | — | — |

**Reach through the phase**, apex peak/floor at every in-phase fork. The amplitude falls on the anneal *and* the surface trains, so a fall here is the two together; what it shows is that reach is not a property of the supply alone.

| condition | 100 | 200 | 500 | 1000 | 2000 | 5000 | 10000 |
|---|---|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 11.1 | 18.4 | 4.91 | 5.17 | 5.41 | 79.1 | 1.82 |
| B — sensory, large, ordered (the wave) | 56 | 68.5 | 117 | 131 | 125 | 184 | 402 |
| C — sensory, large, shuffled | 116 | 145 | 213 | 243 | 225 | 329 | 402 |
| D — sensory, small, ordered | 14.9 | 19.3 | 31.6 | 47.5 | 52.8 | 59.4 | 110 |
| **Z — no supply at all (control)** | — | — | — | — | — | — | — |

And the amplitude at each of those forks (the anneal, so the two can be told apart):

| A(t)/A₀ | 0.995 | 0.990 | 0.975 | 0.950 | 0.900 | 0.750 | 0.500 |
|---|---|---|---|---|---|---|---|

Fraction of a class's cells whose peak cleared the floor, and the apex's peak in absolute terms:

| condition | apex above-floor fraction | apex peak ‖Δ‖ | apex floor | apex peak tick (of 64) |
|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 0.667 | 1.31e-06 | 7.11e-07 | 23.0 |
| B — sensory, large, ordered (the wave) | 1.000 | 3.08e-04 | 6.84e-07 | 25.0 |
| C — sensory, large, shuffled | 1.000 | 3.62e-04 | 8.60e-07 | 42.3 |
| D — sensory, small, ordered | 1.000 | 7.84e-05 | 7.06e-07 | 28.0 |

## Rank — per-edge excitation rank at the deepest interior edges (core–apex, m_e = 3), during the phase

Median over the 32 core–apex edges' two ends, both forms published (ledger row 1: the uncentred form is DC-dominated on this surface and the centred one is reported beside it, never in place of it).

| condition | uncentred PR | centred PR | against m_e | disagreement energy | map effective rank |
|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 1.000 ± 0.000 | **1.270** ± 0.123 | 3 | 8.67e-05 | 3.000 |
| B — sensory, large, ordered (the wave) | 1.000 ± 0.000 | **1.238** ± 0.085 | 3 | 6.93e-05 | 3.000 |
| C — sensory, large, shuffled | 1.000 ± 0.000 | **1.294** ± 0.159 | 3 | 5.82e-04 | 3.000 |
| D — sensory, small, ordered | 1.000 ± 0.000 | **1.258** ± 0.122 | 3 | 6.75e-05 | 3.000 |
| **Z — no supply at all (control)** | 1.000 ± 0.000 | **1.236** ± 0.084 | 3 | 6.97e-05 | 3.000 |

The same, at the shallower strata, centred (so the taper by depth is visible):

| condition | sensory-rim | L1-L1 | L1-core | core-core | core-apex | drive |
|---|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 1.914 ± 0.049 | 1.451 ± 0.064 | 1.440 ± 0.107 | 1.736 ± 0.036 | 1.270 ± 0.123 | 1.000 ± 0.000 |
| B — sensory, large, ordered (the wave) | 1.356 ± 0.003 | 1.332 ± 0.037 | 1.398 ± 0.023 | 1.466 ± 0.188 | 1.238 ± 0.085 | 1.000 ± 0.000 |
| C — sensory, large, shuffled | 1.558 ± 0.029 | 1.654 ± 0.033 | 1.493 ± 0.053 | 1.427 ± 0.118 | 1.294 ± 0.159 | 1.000 ± 0.000 |
| D — sensory, small, ordered | 1.944 ± 0.010 | 1.398 ± 0.017 | 1.459 ± 0.017 | 1.598 ± 0.153 | 1.258 ± 0.122 | 1.000 ± 0.000 |
| **Z — no supply at all (control)** | 1.910 ± 0.103 | 1.414 ± 0.035 | 1.453 ± 0.025 | 1.743 ± 0.138 | 1.236 ± 0.084 | 1.000 ± 0.000 |

## Priming — composed rim-to-apex effective rank at +30k, world arranged, supply off

Median over the 256 **patch** chains (the graph's own shortest edge path from each rim cell to an apex cell, composed); the other rim strata beside it, never averaged in (#181). The pre-registered comparison is each ordered condition against its shuffled control: **B vs C** and **D vs A**.

| condition | at construction | at end of phase | +20k | **+30k** | spread |
|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 1.2979 | 1.0968 | 1.0782 | **1.0774** | 0.0055 |
| B — sensory, large, ordered (the wave) | 1.2979 | 1.1228 | 1.0902 | **1.0834** | 0.0092 |
| C — sensory, large, shuffled | 1.2979 | 1.2658 | 1.1391 | **1.1126** | 0.0241 |
| D — sensory, small, ordered | 1.2979 | 1.0956 | 1.0801 | **1.0717** | 0.0105 |
| **Z — no supply at all (control)** | 1.2979 | 1.0942 | 1.0846 | **1.0745** | 0.0051 |

**The pre-registered contrasts**, at +30k:

| ordered | control | ordered ER | control ER | Δ | spread | beats control |
|---|---|---|---|---|---|---|
| B — sensory, large, ordered (the wave) | C — sensory, large, shuffled | 1.0834 | 1.1126 | -0.0292 | 0.0241 | **False** |
| D — sensory, small, ordered | A — motor, small, shuffled (today's babble) | 1.0717 | 1.0774 | -0.0058 | 0.0105 | **False** |

**Against the zero-supply control**, which is what says whether *anything* was laid down rather than which member laid down most:

| condition | ER at +30k | Z at +30k | Δ vs Z | spread | above Z |
|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 1.0774 | 1.0745 | +0.0029 | 0.0055 | **False** |
| B — sensory, large, ordered (the wave) | 1.0834 | 1.0745 | +0.0089 | 0.0092 | **False** |
| C — sensory, large, shuffled | 1.1126 | 1.0745 | +0.0381 | 0.0241 | **True** |
| D — sensory, small, ordered | 1.0717 | 1.0745 | -0.0029 | 0.0105 | **False** |

The other rim strata at +30k, for the record:

| condition | patch | proprioceptive | touch | actuator |
|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 1.0774 ± 0.0055 | 1.2046 ± 0.1140 | 1.2988 ± 0.1480 | 1.1293 ± 0.0385 |
| B — sensory, large, ordered (the wave) | 1.0834 ± 0.0092 | 1.1554 ± 0.0583 | 1.2158 ± 0.0689 | 1.0952 ± 0.0045 |
| C — sensory, large, shuffled | 1.1126 ± 0.0241 | 1.1307 ± 0.0623 | 1.2940 ± 0.1628 | 1.1106 ± 0.0874 |
| D — sensory, small, ordered | 1.0717 ± 0.0105 | 1.0909 ± 0.0120 | 1.3733 ± 0.2667 | 1.1357 ± 0.1005 |
| **Z — no supply at all (control)** | 1.0745 ± 0.0051 | 1.1010 ± 0.0457 | 1.5674 ± 0.1239 | 1.1161 ± 0.0479 |

## Travel — arm travel per tick under the graph's own command, post-phase windows

T1's frozen baseline reads **3.48e-04** per tick in its last window (the arm at its stops), which is what *travel > 0* is read against rather than literal zero.

| condition | +1k | +5k | +10k | +20k | **+30k** | spread at +30k | above the frozen baseline |
|---|---|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 4.08e-06 | 3.44e-06 | 3.79e-06 | 3.41e-06 | 1.06e-04 | 1.46e-04 | **False** |
| B — sensory, large, ordered (the wave) | 7.40e-03 | 2.63e-04 | 7.53e-04 | 4.15e-05 | 3.01e-06 | 2.55e-06 | **False** |
| C — sensory, large, shuffled | 7.64e-03 | 4.25e-04 | 1.18e-05 | 8.97e-07 | 1.20e-06 | 9.56e-07 | **False** |
| D — sensory, small, ordered | 7.93e-03 | 1.05e-04 | 2.19e-05 | 1.15e-06 | 1.22e-06 | 1.73e-06 | **False** |
| **Z — no supply at all (control)** | 2.71e-04 | 1.10e-04 | 2.12e-06 | 9.79e-05 | 2.20e-04 | 3.10e-04 | **False** |

## Retention guard — apex and soma ρ(K) at +30k, not below T1's frozen baseline beyond spread

T1's baseline is 30k ticks from construction; a T2 run at +30k has run 50000 in total, so the comparator is matched on **post-phase** ticks and not on total ticks. Stated rather than corrected: no frozen run exists at the longer horizon.

| condition | apex (core, drive-adjacent; 8) | core (one level; 16) | vision L1 (64) | somatomotor L1 boundary-adjacent (6) | guard |
|---|---|---|---|---|---|
| **T1 frozen baseline @30k** | 0.808 ± 0.059 | 0.971 ± 0.008 | 0.977 ± 0.003 | 0.884 ± 0.051 | — |
| A — motor, small, shuffled (today's babble) | 0.761 ± 0.046 (-0.047) | 0.953 ± 0.028 (-0.017) | 0.969 ± 0.002 (-0.007 **FAIL**) | 0.809 ± 0.188 (-0.074) | **passes** |
| B — sensory, large, ordered (the wave) | 0.784 ± 0.012 (-0.024) | 0.931 ± 0.027 (-0.039 **FAIL**) | 0.951 ± 0.002 (-0.025 **FAIL**) | 0.788 ± 0.118 (-0.095) | **passes** |
| C — sensory, large, shuffled | 0.737 ± 0.034 (-0.071 **FAIL**) | 0.819 ± 0.043 (-0.152 **FAIL**) | 0.659 ± 0.020 (-0.318 **FAIL**) | 0.717 ± 0.132 (-0.167 **FAIL**) | **FAILS** |
| D — sensory, small, ordered | 0.776 ± 0.034 (-0.033) | 0.955 ± 0.010 (-0.015 **FAIL**) | 0.973 ± 0.002 (-0.004 **FAIL**) | 0.796 ± 0.160 (-0.088) | **passes** |
| **Z — no supply at all (control)** | 0.779 ± 0.039 (-0.030) | 0.960 ± 0.015 (-0.010) | 0.972 ± 0.002 (-0.005 **FAIL**) | 0.913 ± 0.044 (+0.029) | **passes** |

**The same, against the zero-supply control instead of T1.** Z has run the identical number of total ticks, so this comparison carries none of the horizon mismatch above — the difference is the supply's and nothing else. This is the honest per-class read; the T1 table above is the pre-registered one, and both are published.

| condition | apex (core, drive-adjacent; 8) | core (one level; 16) | vision L1 (64) | somatomotor L1 boundary-adjacent (6) |
|---|---|---|---|---|
| **Z — no supply at all** | 0.779 ± 0.039 | 0.960 ± 0.015 | 0.972 ± 0.002 | 0.913 ± 0.044 |
| A — motor, small, shuffled (today's babble) | 0.761 (-0.018) | 0.953 (-0.007) | 0.969 (-0.002) | 0.809 (-0.103) |
| B — sensory, large, ordered (the wave) | 0.784 (+0.005) | 0.931 (-0.029 **below Z**) | 0.951 (-0.020 **below Z**) | 0.788 (-0.124 **below Z**) |
| C — sensory, large, shuffled | 0.737 (-0.041 **below Z**) | 0.819 (-0.142 **below Z**) | 0.659 (-0.313 **below Z**) | 0.717 (-0.196 **below Z**) |
| D — sensory, small, ordered | 0.776 (-0.003) | 0.955 (-0.005) | 0.973 (+0.001) | 0.796 (-0.116) |

Apex ρ(K) through the run, so the phase's own effect is visible beside the post-phase state:

| condition | end of phase | +1k | +5k | +10k | +20k | +30k |
|---|---|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 0.862 | 0.856 | 0.832 | 0.812 | 0.768 | 0.761 |
| B — sensory, large, ordered (the wave) | 0.866 | 0.860 | 0.827 | 0.809 | 0.793 | 0.784 |
| C — sensory, large, shuffled | 0.797 | 0.793 | 0.770 | 0.751 | 0.712 | 0.737 |
| D — sensory, small, ordered | 0.868 | 0.862 | 0.838 | 0.830 | 0.780 | 0.776 |
| **Z — no supply at all (control)** | 0.867 | 0.862 | 0.839 | 0.821 | 0.789 | 0.779 |

## Mechanism at the apex (T0's reads, +30k) — ledger row 1's coherence variable

| condition | ‖ē‖ | ē direction stability | ē share along the drive lane | dead cells |
|---|---|---|---|---|
| A — motor, small, shuffled (today's babble) | 1.81e-03 | 0.982 | 0.907 | [0, 0, 1] |
| B — sensory, large, ordered (the wave) | 1.68e-03 | 0.984 | 0.875 | [0, 2, 0] |
| C — sensory, large, shuffled | 2.02e-03 | 0.957 | 0.818 | [0, 0, 0] |
| D — sensory, small, ordered | 1.73e-03 | 0.983 | 0.888 | [0, 0, 0] |
| **Z — no supply at all (control)** | 1.32e-03 | 0.969 | 0.872 | [0, 0, 0] |

## Branch table — which rows fired

| row | reading | consequence |
|---|---|---|
| none travels post-phase | **fired** | wave 3 skipped; T3 replicates T1's winner alone; **ledger row: curiosity drive owed** |
| B beats C beyond spread | did not fire (Δ -0.0292, spread 0.0241) | — |
| C beats A beyond spread and B ≈ C | did not fire (C−A +0.0352 / 0.0241; B−C -0.0292) | — |
| retention guard fails on the winning condition | **fired** (fails: ['C']) | the winner is the best condition that passes |

