# The live fold read, re-run against the corrected denominator (#206)

[#202](https://github.com/NGL321/patchworks/issues/202) ran `patchworks.tick.FoldRead` for 100,000
ticks and reported that `02`'s burn-in does not exist.
[#206](https://github.com/NGL321/patchworks/issues/206) then corrected the fold margin's denominator —
`CellBody._gradient_norms` divided by the whole `R^(k+n)` hidden row where reconciliation displaces the
node stalk alone, reporting every margin **1.183x tighter** than it is
([#195](https://github.com/NGL321/patchworks/issues/195)'s finding). That correction moves both
numbers in the comparison ADR-0019 is about, so the read was **re-run rather than carried**.

```
PYTHONPATH=src python prototypes/live-fold-read-206/read.py --ticks 100000
```

Same rig as #202, same run: `real` dome, `split=train`, `seed=42`, both rules on, 150 predicting
cells, 22.4 minutes. Only the output paths differ. Results in `206-read.json` (checkpoint tables) and
`206-per-tick.npz` (per-tick and per-cell arrays).

**Every finding survives the correction, and ADR-0019 decision 5 cites these numbers rather than
#202's** — the ADR should quote the code that ships. #202's figures stand as reported.

## The two reads, side by side

| | #202 (pre-correction) | #206 (corrected) |
|---|---|---|
| ticks with a breaching cell | 100,000 / 100,000 | **100,000 / 100,000** |
| cells that breached at least once | 150 / 150 | **150 / 150** |
| cells still breaching after tick 90,000 | 109 | **103** |
| median cell's last breach | tick 95,976 | **tick 96,452** |
| breach density, first decade → plateau | 33 → 16 | **28 → 15** |
| second half: p05 / median / p95 | 11 / 16 / 22 | **9 / 15 / 22** |
| clean ticks | 0 | **0** |
| within-cell crossing enrichment (median) | 3.66x | **3.93x** |
| … above 1.0x / above 1.5x | 145 / 130 of 150 | **144 / 130 of 150** |
| dwell/τ median at tick 100 | 0.96 | **0.96** |
| dwell/τ median at 100,000 | 82.68 | **86.15** |
| cells clearing `dwell ≥ 2.6 τ` at horizon | 130 / 150 | **131 / 150** |
| cells at dwell ≤ 2 ticks | 15 / 150 | **16 / 150** |

A looser margin breaches slightly less often, which is the whole of the movement. **No finding
changes sign, and none changes character:** the breach is still standing rather than transient, the
density still plateaus rather than decaying, reconciliation still costs cells their regions, and
ADR-0005's precondition is still earned over the run and still met by the large majority of cells at
the horizon.

The differences that are not simply "1.183x looser" are the ones this surface does not reproduce
anyway — #195 measured four runs at one seed giving four different binding cells and 5.8x on the cap,
so a 6-cell move in *which* cells are still breaching at 90,000 is inside that noise.

## Breach density by decade

| decade | mean breaching cells | ticks with any breach |
|---|---|---|
| 0–1,000 | 28.42 | 100% |
| 1,000–2,000 | 25.32 | 100% |
| 2,000–5,000 | 28.89 | 100% |
| 5,000–10,000 | 26.47 | 100% |
| 10,000–20,000 | 22.59 | 100% |
| 20,000–50,000 | 16.44 | 100% |
| 50,000–100,000 | 15.34 | 100% |

## ADR-0005's precondition

| ticks | τ median | window dwell median | dwell/τ median | breaching | ≥ 2.6 τ | dwell ≤ 2 |
|---|---|---|---|---|---|---|
| 100 | 1.213 | 1.01 | 0.96 | 34 | 28 | 114 |
| 1,000 | 1.175 | 1.02 | 1.04 | 20 | 56 | 88 |
| 2,000 | 1.178 | 1.30 | 1.16 | 25 | 63 | 81 |
| 10,000 | 1.079 | 5.79 | 5.78 | 25 | 90 | 50 |
| 30,000 | 1.003 | 46.51 | 42.72 | 19 | 124 | 18 |
| 100,000 | 0.991 | 82.40 | 86.15 | 17 | **131** | **16** |

τ is flat at ~1.0–1.2 ticks for the whole run; dwell grows ~82x. That reading is unchanged by the
correction, and the question of whether a one-tick τ is what construction meant to place remains
[#143](https://github.com/NGL321/patchworks/issues/143)'s.

**The `2.6` is not derived for this use.** It is `DEFAULT_SAFETY_FACTOR`, #27's measured one-tick
non-normal *amplification* — dimensionless — reused as a multiple on a *time*. Reported here because
#202 reported it; whether it survives as the pass condition is
[#208](https://github.com/NGL321/patchworks/issues/208)'s.

## What this does not do

- It does not re-open the burn-in. #206 struck the clause and replaced it with nothing.
- It does not settle the unpinned per-cell bias ([#205](https://github.com/NGL321/patchworks/issues/205))
  or the dwell pass condition ([#208](https://github.com/NGL321/patchworks/issues/208)).
- **One run, one seed, one split**, as #202 was. The plateau's height is one draw; that no tick in
  100,000 is clean is not a marginal call at any plausible seed.
