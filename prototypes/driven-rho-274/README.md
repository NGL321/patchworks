# `rho` of the full chart loop, on the driven rig (#274)

[#271](https://github.com/NGL321/patchworks/issues/271) found that `tau`'s instrument omits a real
term of the chart's recurrence and refused to quote the absolute number it measured, because that
number was read at construction with no environment attached.
[#274](https://github.com/NGL321/patchworks/issues/274) is the re-read on a driven run.

- `read.py` — one run: builds the `real` dome through `benchmarks/untrained_fixed_point.build`, runs
  `teaching` with both rules on, and at #206's checkpoint ladder reads **both** radii per predicting
  cell — the chart-only `rho(K @ J_chart)` the record quotes, and the full loop
  `rho(K @ (J_chart + J_stalk @ A_v @ D))`.
- `summarise.py` — the cross-seed roll-up over every `274-*.json` here.
- `274-<dome>-<split>-seed<n>-<ticks>.json` — one file per run, carrying per-cell radii at every
  checkpoint, the undriven control at tick 0, `p_v`, degree, level and the checks.

Nine seeds: 42 to 100,000 ticks, 43 and 44 to 30,000, and 45-50 to 2,000.

## The two checks, and why they are in the rig rather than in the write-up

The reading is a claim that one existing instrument omits a term, so it is only worth as much as its
attachment to that instrument and to the run:

- **`check_relay_identity`** reconstructs `evidence(t+1)` as `A_v (D z + b) + g_v Sum_e F_ev^T y_e`
  and compares it against the node stalk the run actually left behind. At tick 1 — before
  `TransportRule` joins, so the maps that wrote the stalk are still the maps being read — it agrees
  to **1.6e-7 to 2.2e-7** across all nine seeds, which is float32 machine precision. From tick 2 the
  residual is ~3e-3, and that is the transport step's own size rather than an error in the algebra.
- **`check_chart_only_matches_206`** runs `prototypes/live-fold-read-206/read.py`'s own
  `regional_tau` on the same live state and compares. It agrees to ~1e-5, so the two readings differ
  in the relay term and in nothing else.

## Reproducing

    PYTHONPATH=src python prototypes/driven-rho-274/read.py --ticks 100000 --seed 42
    PYTHONPATH=src python prototypes/driven-rho-274/summarise.py

The 100,000-tick run takes about half an hour on a laptop CPU; the 2,000-tick runs take a minute
each and every conclusion in #274's resolution that does not concern the long horizon is visible in
those.
