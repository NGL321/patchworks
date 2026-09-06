# T6 / #534: band pressure on K. Band (0.5, 1.0), eta_K {'baseline': 0.01, 'winner': 0.001}
# Source: T3 committed full-dome 100000-tick checkpoints, 3 seeds, no re-run.

## Clamp engagement and raw drift (mean over seeds; spread = max-min over seeds)

| arm | ticks | stratum | n | engaged sigma>1 | drift median | drift max | sigma_used |
|---|---|---|---|---|---|---|---|
| baseline | 20000 | soma | 6 | 1.000 | +0.2533 +/- 0.0839 | +0.3672 | 1.000 (clamped) |
| baseline | 20000 | vision | 64 | 1.000 | +0.2808 +/- 0.0375 | +0.6928 | 1.000 (clamped) |
| baseline | 20000 | core | 52 | 1.000 | +0.1854 +/- 0.0282 | +0.3623 | 1.000 (clamped) |
| baseline | 20000 | apex | 8 | 1.000 | +0.3325 +/- 0.0638 | +0.5872 | 1.000 (clamped) |
| baseline | 100000 | soma | 6 | 1.000 | +0.5365 +/- 0.7618 | +2.7807 | 1.000 (clamped) |
| baseline | 100000 | vision | 64 | 1.000 | +0.3562 +/- 0.0535 | +0.8027 | 1.000 (clamped) |
| baseline | 100000 | core | 52 | 1.000 | +0.2290 +/- 0.0299 | +0.4403 | 1.000 (clamped) |
| baseline | 100000 | apex | 8 | 1.000 | +0.5017 +/- 0.0260 | +0.9433 | 1.000 (clamped) |
| winner | 20000 | soma | 6 | 1.000 | +0.1168 +/- 0.0634 | +0.2049 | 1.000 (clamped) |
| winner | 20000 | vision | 64 | 1.000 | +0.1175 +/- 0.0479 | +0.4212 | 1.000 (clamped) |
| winner | 20000 | core | 52 | 1.000 | +0.0527 +/- 0.0290 | +0.2197 | 1.000 (clamped) |
| winner | 20000 | apex | 8 | 1.000 | +0.1001 +/- 0.1015 | +0.2776 | 1.000 (clamped) |
| winner | 100000 | soma | 6 | 1.000 | +0.1335 +/- 0.0629 | +0.3422 | 1.000 (clamped) |
| winner | 100000 | vision | 64 | 1.000 | +0.1916 +/- 0.0212 | +0.5499 | 1.000 (clamped) |
| winner | 100000 | core | 52 | 1.000 | +0.0701 +/- 0.0343 | +0.2387 | 1.000 (clamped) |
| winner | 100000 | apex | 8 | 1.000 | +0.1369 +/- 0.1252 | +0.3447 | 1.000 (clamped) |

## Band pressure Pi = d sigma_max per step / band width, and its sign

| arm | ticks | stratum | median Pi | frac cells Pi>0 | median |Pi| | radial share | persistence |
|---|---|---|---|---|---|---|---|
| baseline | 100 | soma | +1.257e-04 | 0.667 | 2.823e-04 | 3.553e-02 | 0.533 |
| baseline | 100 | vision | +3.674e-05 | 0.562 | 1.910e-04 | 2.520e-02 | 0.567 |
| baseline | 100 | core | -1.507e-05 | 0.487 | 9.616e-05 | 1.946e-02 | 0.500 |
| baseline | 100 | apex | +6.008e-05 | 0.667 | 1.133e-04 | 2.669e-02 | 0.500 |
| baseline | 1000 | soma | +3.317e-05 | 0.722 | 3.900e-05 | 9.717e-03 | 0.533 |
| baseline | 1000 | vision | +2.257e-05 | 0.620 | 7.033e-05 | 1.516e-02 | 0.567 |
| baseline | 1000 | core | +1.337e-05 | 0.583 | 5.285e-05 | 1.697e-02 | 0.500 |
| baseline | 1000 | apex | -3.432e-05 | 0.417 | 5.816e-05 | 1.960e-02 | 0.500 |
| baseline | 20000 | soma | -9.577e-06 | 0.333 | 1.754e-05 | 1.392e-02 | 0.533 |
| baseline | 20000 | vision | +6.219e-06 | 0.583 | 2.635e-05 | 9.039e-03 | 0.567 |
| baseline | 20000 | core | +3.127e-06 | 0.538 | 2.006e-05 | 1.155e-02 | 0.500 |
| baseline | 20000 | apex | -6.067e-06 | 0.625 | 3.507e-05 | 1.274e-02 | 0.500 |
| baseline | 100000 | soma | -4.500e-06 | 0.500 | 4.397e-05 | 5.539e-03 | 0.533 |
| baseline | 100000 | vision | +1.496e-06 | 0.510 | 2.440e-05 | 4.924e-03 | 0.567 |
| baseline | 100000 | core | +9.497e-07 | 0.526 | 1.433e-05 | 1.480e-02 | 0.500 |
| baseline | 100000 | apex | +2.455e-06 | 0.542 | 1.057e-05 | 5.555e-03 | 0.500 |
| winner | 100 | soma | +1.824e-04 | 0.833 | 1.912e-04 | 1.015e-01 | 0.683 |
| winner | 100 | vision | +7.849e-05 | 0.729 | 1.150e-04 | 7.934e-02 | 0.583 |
| winner | 100 | core | +1.906e-05 | 0.590 | 6.135e-05 | 3.750e-02 | 0.550 |
| winner | 100 | apex | +3.194e-06 | 0.542 | 9.383e-05 | 6.403e-02 | 0.483 |
| winner | 1000 | soma | +1.088e-05 | 0.611 | 3.423e-05 | 6.214e-02 | 0.683 |
| winner | 1000 | vision | +2.253e-05 | 0.656 | 4.245e-05 | 6.376e-02 | 0.583 |
| winner | 1000 | core | +6.426e-06 | 0.571 | 2.283e-05 | 4.305e-02 | 0.550 |
| winner | 1000 | apex | -2.789e-06 | 0.458 | 2.724e-05 | 6.520e-02 | 0.483 |
| winner | 20000 | soma | +2.723e-06 | 0.667 | 7.648e-06 | 3.855e-02 | 0.683 |
| winner | 20000 | vision | +9.972e-07 | 0.536 | 1.312e-05 | 3.526e-02 | 0.583 |
| winner | 20000 | core | +1.209e-06 | 0.532 | 1.150e-05 | 4.299e-02 | 0.550 |
| winner | 20000 | apex | -8.761e-06 | 0.292 | 1.704e-05 | 7.940e-02 | 0.483 |
| winner | 100000 | soma | -2.879e-06 | 0.278 | 5.804e-06 | 3.130e-02 | 0.683 |
| winner | 100000 | vision | +1.616e-07 | 0.516 | 5.593e-06 | 1.223e-02 | 0.583 |
| winner | 100000 | core | +1.414e-06 | 0.558 | 7.832e-06 | 4.496e-02 | 0.550 |
| winner | 100000 | apex | +3.196e-06 | 0.458 | 8.927e-06 | 5.948e-02 | 0.483 |

## Raw drift sigma_max(K_raw) across the horizon (median per stratum, mean over seeds)

| arm | stratum | 100 | 200 | 500 | 1000 | 2000 | 5000 | 10000 | 20000 | 30000 | 100000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | soma | 1.1563 | 1.1819 | 1.2065 | 1.2156 | 1.2229 | 1.2330 | 1.2423 | 1.2533 | 1.2596 | 1.5365 |
| baseline | vision | 1.1430 | 1.1525 | 1.1648 | 1.1820 | 1.1997 | 1.2208 | 1.2464 | 1.2808 | 1.3066 | 1.3562 |
| baseline | core | 1.1159 | 1.1207 | 1.1287 | 1.1358 | 1.1439 | 1.1529 | 1.1681 | 1.1854 | 1.1955 | 1.2290 |
| baseline | apex | 1.1413 | 1.1461 | 1.1496 | 1.1530 | 1.1596 | 1.1871 | 1.2548 | 1.3325 | 1.3664 | 1.5017 |
| winner | soma | 1.0341 | 1.0482 | 1.0683 | 1.0848 | 1.0958 | 1.1062 | 1.1118 | 1.1168 | 1.1200 | 1.1335 |
| winner | vision | 1.0338 | 1.0374 | 1.0440 | 1.0518 | 1.0613 | 1.0782 | 1.0957 | 1.1175 | 1.1337 | 1.1916 |
| winner | core | 1.0235 | 1.0249 | 1.0277 | 1.0313 | 1.0345 | 1.0390 | 1.0458 | 1.0527 | 1.0576 | 1.0701 |
| winner | apex | 1.0461 | 1.0518 | 1.0560 | 1.0607 | 1.0644 | 1.0710 | 1.0804 | 1.1001 | 1.1083 | 1.1369 |

## Apex against core, at 100k (per seed, both arms)

| arm | seed | apex Pi med | core Pi med | apex frac>0 | core frac>0 | apex drift | core drift |
|---|---|---|---|---|---|---|---|
| baseline | 42 | -9.014e-07 | -2.814e-06 | 0.375 | 0.481 | 1.4986 | 1.2364 |
| baseline | 43 | -2.470e-06 | +3.122e-06 | 0.500 | 0.538 | 1.4902 | 1.2402 |
| baseline | 44 | +1.074e-05 | +2.541e-06 | 0.750 | 0.558 | 1.5163 | 1.2103 |
| winner | 42 | -5.335e-06 | +6.684e-07 | 0.125 | 0.519 | 1.2000 | 1.0901 |
| winner | 43 | +6.072e-06 | +1.336e-06 | 0.625 | 0.577 | 1.0748 | 1.0643 |
| winner | 44 | +8.851e-06 | +2.238e-06 | 0.625 | 0.577 | 1.1358 | 1.0558 |

## Is the gradient radial at all? cos(grad_K, K), and the second-order account of the drift

| arm | seed | ticks | median cos(grad,K) | max abs | actual/predicted d||K||_F^2 (med) |
|---|---|---|---|---|---|
| baseline | 42 | 100 | -7.39e-10 | 1.37e-07 | - |
| baseline | 42 | 20000 | -5.46e-10 | 1.10e-07 | 2.11 |
| baseline | 42 | 100000 | -1.29e-09 | 2.97e-07 | 4.89 |
| baseline | 43 | 100 | +1.54e-09 | 6.87e-08 | - |
| baseline | 43 | 20000 | -4.68e-10 | 8.83e-08 | 2.31 |
| baseline | 43 | 100000 | -2.80e-09 | 6.76e-08 | 3.48 |
| baseline | 44 | 100 | +6.23e-10 | 1.16e-07 | - |
| baseline | 44 | 20000 | +2.43e-10 | 1.13e-07 | 2.11 |
| baseline | 44 | 100000 | -9.06e-10 | 8.03e-08 | 3.77 |
| winner | 42 | 100 | +1.96e-09 | 1.17e-07 | - |
| winner | 42 | 20000 | +5.99e-10 | 1.76e-07 | 0.82 |
| winner | 42 | 100000 | -3.03e-10 | 1.41e-07 | 2.64 |
| winner | 43 | 100 | +1.83e-09 | 9.71e-08 | - |
| winner | 43 | 20000 | +2.35e-09 | 9.67e-08 | 0.69 |
| winner | 43 | 100000 | +2.10e-09 | 1.38e-07 | 1.48 |
| winner | 44 | 100 | -5.64e-10 | 2.45e-07 | - |
| winner | 44 | 20000 | +8.58e-10 | 1.76e-07 | 0.68 |
| winner | 44 | 100000 | -8.63e-10 | 1.33e-07 | 1.49 |

## Verdict inputs (all 150 cells, 100k, mean over seeds)
  baseline  engaged=1.0000  frac Pi>0=0.5178  median |Pi|=1.848e-05  radial share=7.762e-03  persistence=0.5000  median sigma_raw=1.3108
  winner    engaged=1.0000  frac Pi>0=0.5111  median |Pi|=6.972e-06  radial share=2.711e-02  persistence=0.6000  median sigma_raw=1.1222

rows -> C:\Users\noahl\patchworks\.claude\worktrees\cs-534\prototypes\cold-start\T6\534-band-pressure.json
