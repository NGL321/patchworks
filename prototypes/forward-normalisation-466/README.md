# #433's pre-registered read, as it lands (#466)

`rim τ / apex τ` and median apex `λ(K)` at **100,000 ticks on seeds 0, 1, 2**,
with **both builds re-run** rather than the new one differenced against #422's
stored JSON — the Notes' *a rig's recorded data ages with `main`* rule, which is
why the baseline is here as data and not as a quotation.

* `466-projection-real-train-seed<N>-100000.json` — the **post-hoc projection**
  build, `main` at 18f6781. The surface #422's 0.529 / 0.415 / 0.289 and 12.87
  were taken on, re-run rather than quoted.
* `466-normalised-real-train-seed<N>-100000.json` — the **forward
  normalisation** build, this branch.

One file per seed because they are written as they land: #120's long-run budget
runs the seeds sequentially and checkpoints each, and the first attempt at this
read was killed by the low-memory guard partway through the third seed. Two
finished seeds survived it, which is the whole reason for the shape.

**The baseline's instrument was made leaner mid-read, and it is the same
instrument.** It hoarded one `[cells]` mask tensor per tick; it now keeps a
running sum plus the burn-in's first few masks, so the burn-in is still
subtracted exactly. The firing rate is a mean over the steps past the burn-in
either way. Checked rather than asserted: on a 300-tick run the patched and
unpatched rigs return **bit-identical** `firing` and `radius`, which is what
licenses mixing seeds taken before and after it.

`counted` is 99,986 = 100,000 less the 14-tick burn-in, one apex round trip.

## What the read says

`read-100k-both-builds.txt` is `summarise.py`'s output over the six files.

| | apex λ | apex τ | rim λ | rim τ | rim τ / apex τ |
|---|---|---|---|---|---|
| post-hoc projection | 0.438 / 0.361 / 0.374 | 1.21 / 1.00 / 1.02 | 0.846 / 0.873 / 0.864 | 6.00 / 7.35 / 6.82 | 4.95 / 7.33 / 6.70 |
| forward normalisation | 0.667 / 0.751 / 0.692 | 2.47 / 3.50 / 2.72 | 0.960 / 0.955 / 0.960 | 24.3 / 21.8 / 24.5 | 9.82 / 6.22 / 9.02 |

**Not falsified, and the two clauses part company.** Apex `λ(K)` rises on every
seed and on every one of the 24 apex cell/seed pairs — and on 441 of the 450
cell/seed pairs in the graph. `rim τ / apex τ` does **not** fall against the
re-run baseline; it rises, 6.70 to 9.02 on the medians.

**The baseline does not reproduce #422, and that is why both builds were
re-run.** #422 recorded 12.87 and apex `λ` of 0.529 / 0.415 / 0.289 on this
build; re-run on today's `main` it reads 6.70 and 0.438 / 0.361 / 0.374.
Differencing the new build against #422's stored JSON would have shown the ratio
falling 12.87 → 9.02 and scored the second clause **confirmed**. It is not: the
superseded build already read 6.70. The Notes' *a rig's recorded data ages with
`main`* rule is what catches this, and here it changed the answer.

**Why the second clause reads the way it does.** Retention rises *everywhere*,
and proportionally more at the rim: apex `τ` 1.02 → 2.72 (2.7x), rim `τ` 6.82 →
24.5 (3.6x). The ratio is relative, so a graph-wide lift with a rim-heavy
distribution moves it the wrong way. The forward normalisation is not an
apex-specific remedy.

**ADR-0026's bar is closer and still unmet.** Apex `τ` against the apex's
`world_loop` of 15–16 (#383, #398): short by ~14x before, ~5x now.

**The parameter drifts out of band universally and mildly.** After 100k ticks
all 150 cells sit above the band on every seed — median raw `σ(K)` 1.27–1.32,
max 2.71 — against ~1.1 at 1k ticks. So the normalisation is doing work on every
cell every tick, where the projection it replaced fired intermittently, and the
drift is slow rather than divergent. `σ_max(used)` is in band throughout.
