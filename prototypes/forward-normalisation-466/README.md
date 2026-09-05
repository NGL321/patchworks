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
