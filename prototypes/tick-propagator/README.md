# Is the tick heat or waves? A propagator read on a held world

**Question** (#532, 2026-09-11, the user's): is the system a driven dissipative medium in
which signals "bounce around like ripples through a viscous fluid", and is there a quick
test of whether the dynamics behave that way? `docs/motivating-image.md` holds the
dissipative image as motivation with thin provenance; ADR-0002 refuses the solver reading
of the tick; nothing on the record had read the tick's own dynamics on a still world.

**Prediction going in**, from the equations: the message-passing phase is one Jacobi
gradient step on Dirichlet energy (`src/patchworks/tick.py`, *message_passing_phase*),
first order, no momentum, the predicting stalk overwritten by `decode` before it runs
(`tick.py`, *inference_phase*), the chart advanced by a non-expansive `K`. So: pure
diffusion, nothing oscillates, an impulse's peak arrives later at farther cells and never
changes sign. **That prediction was wrong**, and the reason is recorded below.

Rig: `probe.py`. One seed (42), float64 (`detectability.double_precision`), world held by
re-writing the same observation every tick (`detectability.hold_still`, 100 ticks) — the
same hold every detectability reading uses. Three surfaces: the constructor's draw
(`untrained`), `holonomy_read.flat_maps` installed (`flat`), and
`untrained_fixed_point.taught` at 2,000 ticks (`taught2000`). **Two graphs**, `--graph dome`
and `--graph dense`; one JSON per graph and surface beside this file (`dome-*.json`,
`dense-*.json`). No benchmark CLI `read` is called, so nothing files to GitHub.

The readings below were taken first on the small dome (`tests/conftest.py::SMALL`, 39
cells, 15 predicting, 54 edges). **The user then ruled that prototypes leave the dome**
(B81, #660): the dome is a placeholder the record calls explicitly abandonable, and the
graph the effort is aimed at is a densely connected undifferentiated one carved toward
sparsity afterwards. The dome readings are kept as what they are, labelled, and the whole
read was repeated on that graph — *The same read off the dome* below.

## Readings

### 1. The held world is not at rest; it rings

Per-tick change of the carried state `(stalks, charts, broadcast)` as a fraction of its
norm, over the last 100 of 400 held ticks, and the autocorrelation of the centred orbit.

| surface | change / state | autocorrelation peaks (lags) | period | cells ringing (> 1e-3) | loudest cell: change rms / stalk norm |
|---|---|---|---|---|---|
| untrained | **0.95** | 4, 7, 11, 14, 18 | ≈ 3.5 ticks | 16 / 39 (every predicting cell + the actuator) | cell 37 (L3): 94.8 / 91.7 |
| flat | **0.97** | 4, 7, 11, 14, 18 | ≈ 3.5 ticks | 16 / 39 | cell 37 (L3): 124.3 / 116.3 |
| taught2000 | **0.042** | 2, 4, 6, 8, 10 | 2 ticks | 13 / 39 | cell 34 (L2): 1.6 / 7.2 |

At construction every predicting cell's stalk moves by about its own norm every tick: the
state is not settling toward anything, it is swinging. Installing exactly flat maps changes
nothing about it. Two thousand ticks of training damp it by two orders of magnitude and
shrink the stalks from ~40–90 to ~5–8; what is left is a period-2 alternation concentrated
in one cell.

### 2. Finite-time growth rates along the orbit

Twelve orthonormal perturbations carried through the true tick for 300 ticks with a QR
re-orthonormalisation each tick, reported as per-tick moduli (resolution 1/300 ≈ 0.0033
per tick). *Finite-time rates on this orbit*, not the long-run limit `CONTEXT.md` reserves
the word Lyapunov exponent for.

| surface | top six moduli | growing | neutral within resolution | slowest decaying e-fold |
|---|---|---|---|---|
| untrained | 1.005, 1.002, 0.998, 0.989, 0.992, 0.988 | 1 (at resolution) | 2 | 120 ticks |
| flat | 1.002, 0.997, 0.995, 0.987, 0.994, 0.987 | 0 | 1 | 290 ticks |
| taught2000 | 0.989, 0.988, 0.982, 0.974, 0.964, 0.943 | 0 | 0 | 92 ticks |

The constructions' ring is **sustained**: its leading modes sit at 1 to within resolution
over 300 ticks. Training makes every mode decay; the slowest e-fold is 92 ticks against
`world_loop` of 3–5 on this dome.

### 3. Which loop rings

The held tick with one phase skipped (instrument, never mechanism), on the same snapshot.

| surface | full tick | inference phase alone | reconciliation alone | reconciliation, no unit delay |
|---|---|---|---|---|
| untrained | 0.95, period ≈ 3.5 | **0.94, period ≈ 3.5** | 7.0e-4, monotone (ac ≈ 0.98) | 7.0e-4, monotone |
| flat | 0.97, period ≈ 3.5 | **0.93, period ≈ 3.5** | 7.0e-4, monotone | 7.0e-4, monotone |
| taught2000 | 0.042, period 2 | **0.044, period 2** | 5.8e-5, monotone | 5.8e-5, monotone |

**The ring is each cell's own recurrence** — chart → `K` → `decode` → stalk → `encode` →
chart — with no edge involved. Reconciliation on its own is heat: a monotone drift three
orders of magnitude smaller than the state, and the unit delay on the neighbour term makes
no difference to it. `σ_max(K) ≤ 1` bounds the chart's advance, not the loop through
`encode` and `decode`, and at construction that loop is an undamped oscillator.

### 4. Propagator spectrum at one tick

Central finite differences of the one-tick map on the carried state (3,065 numbers; the
world write included; a closure check confirms the tick reads nothing else). **On the two
constructions the state is not a fixed point, so this is a linearisation on a moving orbit
and its moduli above 1 are not growth rates** — reading 2 is what stands. It is kept for
the angles, which give the ring period directly.

| surface | on the unit circle | moduli > 1 (largest) | complex with modulus > ½ | leading complex mode: modulus, period |
|---|---|---|---|---|
| untrained | 1 | 8 (1.32) | 72 | 1.32, 3.49 ticks |
| flat | 1 | 12 (1.20) | 72 | 1.20, 3.59 ticks |
| taught2000 | 1 | 3 (1.04) | 60 | 1.01, 3.41 ticks |

The one mode exactly on the circle is a stalk component that nothing writes or reconciles
(share: stalks 1.0, private reserve 0.0) — it is not the privacy reserve, which `decode`
overwrites every tick.

### 5. Impulse response

`detectability.branch`, unit impulse at the first rim cell, 256-tick window, the whole
per-cell trace kept. Peak tick by hop distance (median), sign changes of the deviation
projected on its own peak direction, and the dominant period of that projection.

| surface | peak tick at hop 1 / 2 / 3 / 4 | cells with sign changes | median period | energy at 256 / peak |
|---|---|---|---|---|
| untrained | 1 / 15 / 192 / 254 (censored) | 16 / 16 | 3.6 ticks | 9.5e-5 |
| flat | 1 / 21 / 40 / 155 | 16 / 16 | 3.6 ticks | 1.8e-3 |
| taught2000 | 1 / 6 / 12 / 13.5 | 16 / 16 | 3.9 ticks | 2.7e-10 |

On the constructions the deviation does not propagate so much as accumulate: the far cells
are still rising at the end of the window. On the trained surface the peak arrives about
six ticks per hop — front-like, and slower than `world_loop` at every cell — and drains to
nothing. Every trace on every surface carries the ring's period, because the deviation is
riding the cells' own oscillation.

## What this says

- **Not heat, and not a fluid.** At construction the room is fifteen independent
  oscillators, each cell's function ringing at its own full amplitude with period ≈ 3.5
  ticks, coupled by a reconciliation term a thousand times weaker than the swing. The
  motivating image's "ripples" exist, but they are not ripples crossing a medium; they are
  each cell ringing in place.
- **Training damps the cell, not the coupling.** The prediction rule fits `decode ∘ K ∘
  encode` to the reconciled stalk, and a loop fitted to its own output stops swinging. That
  is the same mechanism B62 (#635) read as the traffic-rank collapse under the prediction
  rule alone, seen from the dynamics side — a hypothesis for B68's successors, not a
  finding.
- **The coupling was never the medium.** Alignment of the restriction maps is a property of
  the connection, and this read is consistent with ADR-0032's steer that the remaining gap
  is not alignment: exactly flat maps leave every dynamical reading unchanged.

## The same read off the dome

`dense.py`: the dome's sensorimotor rim cell for cell (the rim is the world's shape, not the
dome's), and in place of levels, columns, covering and taper one population of 15
predicting cells with every pair joined — 133 edges against the dome's 54, degree 15–17
per predicting cell against 3–7. Each sensory cell attaches to one interior cell in a
seeded round-robin, the actuator by the dome's own reflex rule, the drive to the three
least-loaded cells. Interior lane widths are **re-derived, not inherited**: the allocator
pins any same-level edge to `lateral_m = 1` (#540's ruling (c1), which `graph.py` marks
contingent on the dome), so on a one-level graph every lane gets the width the budget
affords evenly, which is 3. The dome's interior lanes are `[1, 15]` — laterals pinned,
verticals wide — and that is the first named confound: the two graphs spend the same
capacity budget on entirely different lane shapes. The privacy reserve is unchanged.

| reading | dome | dense | holds? |
|---|---|---|---|
| ring at construction: change / state (untrained, flat) | 0.95, 0.97 | 0.80, 1.00 | yes |
| ring period (autocorrelation, impulse) | ≈ 3.5 ticks | ≈ 3.5–4.6 ticks | yes |
| cells ringing at construction | 16 / 39 | 16 / 39 | yes |
| leading growth moduli at construction | 1.005, 1.002 | 1.008, 1.002 | yes: sustained |
| inference phase alone (untrained) | 0.94, rings | 0.91, rings | yes |
| reconciliation alone (untrained) | 7.0e-4, monotone | 6.0e-4, monotone | yes |
| reconciliation without the delay | identical | identical | yes |
| exactly flat maps change the dynamics | no | no | yes |
| trained: change / state | 0.042 | **1.8e-4** | damping is ~250× stronger |
| trained: cells still ringing | 13 / 39 | **1 / 39** | |
| trained: impulse peak by hop 1 / 2 / 3 | 1 / 6 / 12 | 1 / 3 / 6 | front twice as fast |
| trained: impulse energy at 256 / peak | 2.7e-10 | 1.8e-3 | dense holds it longer |
| trained: slowest decaying e-fold (300-tick rates) | 92 ticks | 80 ticks | |

**Every qualitative reading holds off the dome.** The ring is the cell's own loop on both
graphs; reconciliation is a whisper of the same size on both — with 2.5× the edges and
3-wide lanes in place of 1-and-15 — because the gain denominator caps `λ_max(Σ_e F_evᵀF_ev)`
at `g_v² c_v` and the gain divides by the same, so the total reconciliation strength at a
cell is bounded by `γ` whatever its degree (`tick.py`, *reconciliation_gain*;
`restriction.py`, *_push_apart*). The coupling is weak by construction, not by shape.

What the dome did confound is the trained surface. On the dense graph two thousand ticks
of the same rules silence the ring almost completely and the impulse crosses the graph
twice as fast, while the perturbation drains more slowly. A hypothesis, not a finding:
with 15–17 neighbours the reconciled target each cell's decoder is fitted to is a wider
average and so steadier, and the prediction rule's damping of the cell's loop is faster
for it. Which of the closed B-series readings this touches is B81's item 4.

## Limits

One seed, float64, world held, small graphs of 15 predicting cells. Nothing here is read
under drive or at the real dome's size. The finite-time rates are 300-tick rates. The
2,000-tick surface is one arm; the reading at 20k, which the bar (#532, *Destination*) is
set at, is not taken. The dense graph's parameters (population, density, covering, drive
attachment, lane widths) are a first reference and are the user's to rule on B81 (#660);
the readings above are posted there.
