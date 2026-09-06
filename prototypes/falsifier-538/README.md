# The pre-registered falsifier, taken (#538)

[#538](https://github.com/NGL321/patchworks/issues/538) reads the quantity
[#142](https://github.com/NGL321/patchworks/issues/142) pre-registered,
[#155](https://github.com/NGL321/patchworks/issues/155) carried as condition 2, and
[#220](https://github.com/NGL321/patchworks/issues/220) closed against a proxy without taking:
**cross-edge alignment, taught and untrained**, baseline **14.20x taught / 3.66x untrained**.

Reproduce:

    PYTHONPATH=src python benchmarks/untrained_fixed_point.py attenuation --dome full --split train --seed 0 --ticks 1500
    PYTHONPATH=src python benchmarks/untrained_fixed_point.py attenuation --dome full --split train --seed 0 --learn 30000

and the same two at `--seed 1`. The four logs are beside this file. A taught arm is ~25 min on this
box; the untrained arms are ~1 min.

## The instrument was never missing

#220's stated reason for not taking the reading was that *"it is #182's instrument and
`prototypes/rim-bound-182/` is not on this branch."* Both halves are wrong, and the record should not
carry them:

- **The instrument is `benchmarks/untrained_fixed_point.py attenuation`**, named in
  [#155](https://github.com/NGL321/patchworks/issues/155)'s own falsification block — *"Run
  `benchmarks/untrained_fixed_point.py attenuation` after the edit"* — which is the text #220
  inherited condition 2 from, and named again by
  [#184](https://github.com/NGL321/patchworks/issues/184) item 2. `prototypes/rim-bound-182/` reads
  the *within-cell* spread factor (`Σσ_max²/λ_max`); it does not compute this quantity at all.
- **Both were on `main` before #220 closed.** The instrument landed 2026-08-29 (`305880e`),
  `prototypes/rim-bound-182/` landed 2026-08-30 (`1a419b1`), and #220 closed 2026-09-01.

## The surface

`origin/main` at `2bce07d`, full dome, `split=train`, seeds 0 and 1, 30,000 ticks with both rules on
for the taught arms and 1,500 untrained ticks for the untrained ones. The `#455` rule wants the
constants the number depends on named, and here they are the point: **`c = 2` is in circuit** — the
incoherence cap is enforced in `RestrictionMaps.project()` and `c_v` is in
`tick.reconciliation_gain`'s denominator, neither of which was true when the baseline was taken —
**ADR-0032's spectral floor is on**, and **ADR-0031 has deleted the sparsity pressure**. #142's
baseline was taken on a surface with none of the three.

## The reading

`one map alone` is `σ_max` mean against the isotropic mean, the sender's own concentration.
`cross-edge` is the chained aligned hop against random nudges — the pre-registered quantity.
`receiver` is `cross-edge / one map alone`, what the *far* map contributes. `% built` is
`log(receiver) / log(one map alone)`, #142's *"roughly 55% built"*. `unspent` is
`one map alone / receiver`, which is #184's 2.15x.

| surface | one map alone | **cross-edge** | receiver | % built | unspent |
|---|---|---|---|---|---|
| #142, untrained | 3.36x | **3.66x** | 1.09x | 7.1% | 3.09x |
| #142, taught 30k | 5.53x | **14.20x** | 2.57x | 55% | **2.15x** |
| now, untrained, seed 0 | 3.847x | **4.04x** | 1.050x | 3.6% | 3.66x |
| now, untrained, seed 1 | 3.842x | **4.08x** | 1.062x | 4.5% | 3.62x |
| now, taught 30k, seed 0 | 3.183x | **3.49x** | 1.096x | 7.9% | **2.90x** |
| now, taught 30k, seed 1 | 3.183x | **3.50x** | 1.100x | 8.2% | **2.90x** |

## The condition fired

**Taught cross-edge alignment reads 3.49x and 3.50x against a pre-registered floor of 14.20x — a
4.06x regression.** The map's standing rule is that *only the falsifying end of a pre-registered
ratio is a verdict*; this is that end, and it is not close.

Three things make it a verdict rather than an artefact:

1. **The untrained arm did not fire.** It reads 4.04x and 4.08x against 3.66x — *up* 11%. Whatever
   moved is not a shift in the instrument or in the graph's construction; both arms run the same
   code on the same dome.
2. **It fired past its own untrained arm.** Taught cross-edge alignment (3.49x) is now **below**
   untrained (4.04x). On #142's surface training multiplied the quantity 3.66x → 14.20x, a factor of
   3.88. On this surface training multiplies it by **0.86 — it takes alignment away.** The channel
   ADR-0022 named as *"largely a thing training makes"* is not a thing training makes any more.
3. **Two seeds agree to the third digit** on every column of the taught arm.

## Where the fall sits, and what this reading does not isolate

The fall splits cleanly across the two factors, and they have different suspects:

- **Sender side**, `one map alone`: 5.53x → 3.18x. This is what **ADR-0032's spectral floor
  predicts**. A scaled co-isometry has a flat spectrum, so `σ_max` against the isotropic mean must
  fall toward `√m`; #142's 5.53x was read on near-rank-1 maps, which ADR-0032 deliberately abolished.
  Nothing here is a surprise and nothing here is `c`'s doing.
- **Receiver side**, `receiver`: 2.57x → 1.10x. **This is the half ADR-0010 pre-registered.** It is
  the alignment *between* a cell's two incident maps, and `c` is the only one of the three surface
  changes that acts on it directly — the incoherence cap's whole job is to push a cell's incident
  maps' top directions apart, and pushing them apart is what condition 2 said could eat the channel.

**This reading does not isolate `c`, and should not be quoted as if it did.** Three things changed at
once and only a control can separate them.
[#537](https://github.com/NGL321/patchworks/issues/537) already owns that sweep. What this reading
establishes is that the pre-registered cost **was incurred**, at 4x, on the factor it was registered
against — not that `c` is the sole cause.

## What it means for #532

[#533](https://github.com/NGL321/patchworks/issues/533) found the composed hop is a product of
principal-angle cosine matrices between each relay cell's two incident carried subspaces. A
`receiver` term of 1.10x is that same statement read on one hop: the far map is very nearly
insensitive to the direction the near map emits, which is `V_outᵀ V_in` sitting near its generic
value. **The falsifier firing and composed rim-to-apex effective rank reading 1.000 are one fact seen
from two instruments**, and this one has a pre-registered baseline attached to it while the other
does not.

## Stated limits

- **Two seeds, one split, one horizon.** 30,000 ticks because that is the baseline's horizon, not
  because anything converges there.
- **`--learn 30000` and `--ticks 1500` are different arms, not a before/after on one trajectory.**
  The untrained arm settles an untrained sheaf; that is how the baseline was taken.
- **No control isolates `c`.** See above; #537 owns it.
- **`one map alone` and `cross-edge` come from the same nudge**, so `receiver` is a ratio of two
  numbers read on one surface and inherits both their noise. Two seeds agreeing to the third digit
  is the whole of the evidence that this is small.
