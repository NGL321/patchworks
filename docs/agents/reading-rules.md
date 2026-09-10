# Reading rules

Standing constraints on how a reading is taken and quoted, promoted off wayfinder map
[#532](https://github.com/NGL321/patchworks/issues/532) on 2026-09-10 so the map can be an index
again. Each rule names the ticket that minted it; the argument lives there and is not restated here.
`docs/agents/domain.md` carries the sibling rule that an ADR quoting a measured figure names its
surface. Nothing here overrides `CONTEXT.md`, the spec or an ADR.

A rule is **standing** until the ticket that minted it is overturned; a rule marked *diagnostic* is a
note a reading should carry, not a condition a candidate must clear.

## Which object a reading is on

- **A reading states which of the two objects it is on, and no number is carried between them.**
  The *transport operator* — products of restriction maps: composed effective rank, principal-angle
  cosines, `channel_return`, `identification`, audience differentiation, earned `H⁰` — never touches
  a node stalk ([B17 #565](https://github.com/NGL321/patchworks/issues/565)). The *node stalks* —
  traffic rank, `N(θ)` with `A(θ)`, `I(P; Δ)`, `τ̂`, impulse amplitude — are what travels. The two
  differ by roughly two orders per hop, and a trade quoted across them is a weld
  ([B49 #616](https://github.com/NGL321/patchworks/issues/616)). In the symbol vocabulary: a
  symbol's *identity* is read on the operator, its *activation* on the stalks.
- **Quote gauge invariants.** The learned sheaf is identified up to `O(k_v) × O(m_e)`; composed
  rank and the principal angles are exact invariants of it (orbit spread 9.1e-07,
  [B28 #577](https://github.com/NGL321/patchworks/issues/577)). A symbol's identity does not depend
  on the frame the maps carry it in, so a frame-dependent number is not a reading (the user's rule,
  2026-09-10).
- **Centred and uncentred are different measurements, not one with a bias.** `corr(er_uncentred,
  er_centred)` is +0.21 / +0.07 across arms; the uncentred statistic tracks the mean-to-variation
  ratio ([B18 #567](https://github.com/NGL321/patchworks/issues/567)). Name which one is quoted.
  The centred traffic rank is also window-bound, growing monotonically with `T`
  ([B57 #629](https://github.com/NGL321/patchworks/issues/629), [B62 #635](https://github.com/NGL321/patchworks/issues/635)).

## Horizons and the stalled world

- **Stamp the stall with `b61_analyse.stall_diagnostic`, never with `b56_analyse.horizon` alone.**
  The last-above rule latches onto sporadic re-crossings after the body has died; three of four long
  runs mis-stamped ([B61 #634](https://github.com/NGL321/patchworks/issues/634)). The stall horizon
  is per-run, not per-arm: 60–1,450 ticks, 13× between seeds of one arm
  ([B38 #599](https://github.com/NGL321/patchworks/issues/599)).
- **Past the stamp, a reading on the shipped rules is drift under a frozen stimulus**, not learning,
  and says nothing about a live agent ([B38](https://github.com/NGL321/patchworks/issues/599),
  [B61](https://github.com/NGL321/patchworks/issues/634)). One exception, the user's ruling on
  [B56 #628](https://github.com/NGL321/patchworks/issues/628): a term whose gradient never reads a
  stalk trains on a motionless body by construction and may be read past the stamp.
- **Two runs identical in seed and schedule diverge past ~3,000 ticks** when a read-only diagnostic
  differs; few-percent differences between arms are not signal
  ([B19 #568](https://github.com/NGL321/patchworks/issues/568)).
- **A difference between two `TransportRule`-carrying arms is quoted against a replicate of one of
  them at the same rung, or it is not quoted**; where a rung reproduces bit-exactly, say so. Such arms
  diverge from the last bit inside one process (ΔER 2.6e-05 at tick 100 with no rule running,
  amplified three orders in 200 ticks by `TransportRule`; up to 0.18 at 20,000), while a separate
  process reproduces exactly ([B68 #645](https://github.com/NGL321/patchworks/issues/645) §4a; the source
  is [B74 #653](https://github.com/NGL321/patchworks/issues/653)'s). B62's and B68's transport-arm
  figures differ by this spread, not by an error.

## Agreement readings

- **`N(θ)` is quoted with `A(θ)` beside it, or it is not a reading.** `N` is a participation ratio
  inside the counted band and reads 1.0000 on a direction carrying 0.4% of the traffic; `A` says how
  much traffic the agreed directions carry ([B62 #635](https://github.com/NGL321/patchworks/issues/635)).
- **An agreement statistic is reported alongside audience differentiation at a cell and the exposure
  it cost.** Agreement rising at differentiation 0.0000 scores nothing: the reserved frame earns 88
  dimensions that way and the flat bundle 45 ([B48 #615](https://github.com/NGL321/patchworks/issues/615),
  [B52 #622](https://github.com/NGL321/patchworks/issues/622); the two rows were merged in B48's prose
  and [B69 #646](https://github.com/NGL321/patchworks/issues/646) disentangled them). **The rule's null
  is the construction point, labelled `construction`** — a trained flat bundle has `earned = 0` from
  tick 50 and no settled differentiation even at 60,000 ticks — and any *gap* claim is read at a
  shared rung against the trained flat bundle run to that rung ([B69](https://github.com/NGL321/patchworks/issues/646),
  restated on [B71 #650](https://github.com/NGL321/patchworks/issues/650)). This guards a *candidate's
  training path*; it is not a scoring column on a reference architecture
  ([B66 #642](https://github.com/NGL321/patchworks/issues/642)).
- **A candidate aimed at the traffic's rank names which rule it is aimed at.** The prediction rule
  alone collapses it at zero exposure cost; the transport rule alone raises it and raises
  differentiation; the shipped pair costs a column neither pays alone
  ([B62 #635](https://github.com/NGL321/patchworks/issues/635)).
- **Exact agreement is unavailable to any visible direction by construction**: `ker(G) = ker(δ_P)`
  exactly, so a bar written on `earned` asks for a measure-zero event ([B62](https://github.com/NGL321/patchworks/issues/635),
  [B48](https://github.com/NGL321/patchworks/issues/615)).

## Dependence readings (the activation gate)

- **Dependence is a gate, not a ranking.** A necessary condition is supposed to be passed by many
  arrangements, the flat bundle among them; nothing separates it from the trained surface on
  `I(P; Δ)` and nothing could ([B66 #642](https://github.com/NGL321/patchworks/issues/642)).
- **Quoted on `patch` alone**, against the **untrained surface and the flat bundle**. Proprioceptive
  and touch read the ceiling exactly at `k = 2, 4, 8`, which is bijectivity to the 3-wide terminus,
  not an alphabet artifact ([B63 #637](https://github.com/NGL321/patchworks/issues/637),
  [B66](https://github.com/NGL321/patchworks/issues/642)); the terminus question is
  [B67 #643](https://github.com/NGL321/patchworks/issues/643)'s.
- **The sweep is the noise model, and `C ≤ 8` is not admissible**: at four situations the untrained
  arm reads 0.87–1.30 bits against 0.20–0.28 at twenty-four ([B63](https://github.com/NGL321/patchworks/issues/637)).
- **The reduction is the whole-window trace, not the peak tick**, for this instrument (the user's
  ruling of 2026-09-10 on B63: 0.857 bits at the peak against 2.810 over the window on patch).
  Amplitude discipline stands: the perturbation is fixed-norm and varies which pattern, never how
  hard ([B43b #609](https://github.com/NGL321/patchworks/issues/609)).
- **Read on the float64 cast with the float32 gate reported beside it, per stratum.** On patch the
  terminus response clears `eps_f32 · ‖state‖` on 0 of 192 trials on every arm, so a patch verdict is
  a float64 claim and says so ([B63](https://github.com/NGL321/patchworks/issues/637);
  ADR-0026's gate).
- **A path quantifier may not be reintroduced to strengthen a dependence claim.** Information only
  decreases along a route; the terminus has already suffered every constriction
  ([B43b #609](https://github.com/NGL321/patchworks/issues/609)).

## Clamps, gains and initialisations

- **A candidate that proposes to train a quantity states which clamp stands between the parameter
  and the reading.** Three construction-time clamps absorb what training moves: the restriction-map
  mask (`project()` re-applies it, so every rank-derived number is invariant to the integer over
  20k ticks, [B22 #571](https://github.com/NGL321/patchworks/issues/571)); the band on `K`
  (`σ(used)` pinned at 1.000 while `ρ(K)` moves on 149 of 150 cells,
  [B44b #610](https://github.com/NGL321/patchworks/issues/610)); and rank-measured `k_v` at 20.0 on
  every arm while differentiation moves ([B56 #628](https://github.com/NGL321/patchworks/issues/628)).
  A fourth sits on the bar itself: `spec.joints = 3` at the terminus
  ([B63](https://github.com/NGL321/patchworks/issues/637)).
- **A gain from an objective term is quoted against its initialisation**, and the same term is
  reported on a frame that already has the quantity. B56's clause 1 moved `channel_return` 3.9×
  from a random start and +0.0001 from a staggered one ([B58 #630](https://github.com/NGL321/patchworks/issues/630)).
- **A dependence claim from a construction is quoted against the untrained surface**, since a
  construction may supply the quantity free ([B58](https://github.com/NGL321/patchworks/issues/630),
  scoped to patch by [B66](https://github.com/NGL321/patchworks/issues/642)).
- **A differentiation reading taken while `k_v` moves unreported is void**, and the quantity to
  report is the participation ratio, not the rank, which recovers to 20.0 by tick 50
  ([B59 #631](https://github.com/NGL321/patchworks/issues/631), amended on B58; ratified 2026-09-10).

## Carving and topology

- **Any carving candidate reports its effect on `world_loop` lengths.** Removing an edge can only
  lengthen a shortest path, so carving raises ADR-0026's divisor monotonically; necessary, not
  sufficient ([B39 #601](https://github.com/NGL321/patchworks/issues/601), the user's amendment).
  Every topological instrument reads the carried graph (`m_e > 0`), not `dome.edges`;
  `benchmarks/loop_length.py` is width-blind ([B41 #604](https://github.com/NGL321/patchworks/issues/604)).
- **A candidate that shortens paths states its amplitude against its own hop count**, not against
  the unrelayed baseline: an unaimed random rewiring passed ADR-0021's amplitude clause by seven
  orders ([B43a #607](https://github.com/NGL321/patchworks/issues/607)). What the null should be is
  [B45 #611](https://github.com/NGL321/patchworks/issues/611)'s.
- **A carve is a reallocation, never a deletion**, and reversibility is graded, "within reason":
  a pruned edge's warrant stays readable, and the node-stalk mask is under the same rule
  ([B41 #604](https://github.com/NGL321/patchworks/issues/604), [B42 #605](https://github.com/NGL321/patchworks/issues/605)).
- **An objective carrying a local holonomy term states what stops its optimum being collapsed
  lanes** and reports audience differentiation beside the holonomy it achieves
  ([B42 #605](https://github.com/NGL321/patchworks/issues/605)). The answer on record: nothing in the
  objective does; the guard is a parameterisation ([B59 #631](https://github.com/NGL321/patchworks/issues/631)).

## The guard's terms (the staggered frame)

Ratified by the user on 2026-09-10 as the terms under which the staggered frame counts as guarded
against collapse. They do not decide candidacy, which is judged on the bar
([B76 #655](https://github.com/NGL321/patchworks/issues/655)).

- **Falsifier:** differentiation settles above collapse with the drift decaying, not linear; read
  at 0.3912 by 30,000 ticks on seed 42 ([B61 #634](https://github.com/NGL321/patchworks/issues/634)).
  The null's own climb is [B69 #646](https://github.com/NGL321/patchworks/issues/646)'s.
- **Exchange rate**, for term-side candidates: differentiation given up per unit `channel_return`
  gained, as excess over a matched baseline at the same rung and seed, at adjacent rung pairs;
  passes if flat or falling. An absolute floor is refused, since the untreated control drifts.
- **Drift form**, for initialisations: drift flat or decaying against the same matched control.
- **Unaided clause 1 is retired**: inert on a staggered start, and the composition that kept it
  alive is empty.

## Diagnostic notes (carried, not binding)

- `τ̂` is blind to the return: at `channel_return` 1.0000 and five orders of bottleneck, paired
  `τ̂` moved within one trial's noise ([B44b #610](https://github.com/NGL321/patchworks/issues/610)).
  Demoted from a standing constraint on 2026-09-10 because `τ̂` left the bar; a candidate arguing
  retention still states its case against the prediction term, which presses `ρ(used)` down.
- Earned `H⁰` and composed effective rank may be reported as diagnostics and never as pass
  conditions ([B27 #576](https://github.com/NGL321/patchworks/issues/576), [B48 #615](https://github.com/NGL321/patchworks/issues/615)).
- Composed rank 1.000 is domination, not annihilation: full rank 3 on the median chain, `σ₂/σ₁`
  0.005 at 20k ([B1 #537](https://github.com/NGL321/patchworks/issues/537)).

## Known instrument defects

- Rigs that share an output path lose tables: B62 overwrote one of its own runs and rebuilt the table
  from the surviving record ([#635](https://github.com/NGL321/patchworks/issues/635)). Give every run
  its own path. The remaining gap between B62's and B68's transport-arm figures is run-to-run spread
  (above), not the overwrite.
- A low-memory guard at 97% killed five arms on B61; the settle there rests on one seed
  ([#634](https://github.com/NGL321/patchworks/issues/634)).
- B50's readout on `main` and its ticket disagree on one number (0.4309 against 0.4278 for the
  winner's differentiation trough at 10k, [#618](https://github.com/NGL321/patchworks/issues/618)).
- `t0.surface()` was dead on `main` between #548 and B50's repair, so #537's filed construction
  figures are stale ([#618](https://github.com/NGL321/patchworks/issues/618)).
