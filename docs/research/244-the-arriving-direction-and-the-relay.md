# Does what arrives on an edge align worse than average with the next hop's operator?

[#244](https://github.com/NGL321/patchworks/issues/244). One measurement, no decision in it. It reads
the unfavourable side of the misalignment candidate
[#233](https://github.com/NGL321/patchworks/issues/233) could only bound on the favourable side, and
rules on nothing: the remedy species this grading belongs to is already closed
([#184](https://github.com/NGL321/patchworks/issues/184),
[#230](https://github.com/NGL321/patchworks/issues/230)). Its value is the same as #233's — it tells
the retention recharter whether per-edge variation is a construction fact or a learned one.
Reproduced by

    docker run --rm --entrypoint python patchworks:headless \
        benchmarks/alignment_read.py null      # 4 s wall, no sandbox
    docker run --rm --entrypoint python patchworks:headless \
        benchmarks/alignment_read.py align     # ~25 min, trains 30k ticks

in the supported container ([ADR-0012](../adr/0012-a-container-is-the-supported-execution-target.md)),
on the real dome: 414 cells, 682 edges, `chi = 1036`, seed 0, 24 trials, both directions — #214's own
configuration, and #233's. The statistic and its null are checked against a brute-force Monte Carlo
through the operator itself in `tests/test_alignment_read.py`, because this ticket's whole content is
one ratio and the bar it is read against, and both have a plausible wrong form.

---

## TODO-VERDICT

---

## 1. What #233 left, and why a tick-aligned read can take it

#233 read #184's three parked candidates for the 9x-to-240x per-hop grading. Construction is not the
cause (`R² = 0.139`, 5.0x of usable range against a 151x spread); per-edge floor variation is a real
contributor at 34% by sd; and inter-endpoint misalignment is **bounded on the favourable side** —
alignment headroom, what the *best* direction buys over an average one, is at most 2.83x on every hop
measured, median 1.97x.

Headroom does not price what a **badly chosen** direction costs. That is unbounded below, and it is
the form the candidate would actually take. #233 could not read it, and said so precisely: its
measured quantity is a **windowed peak** — `max` over a 64-tick window of a paired ratio — sitting
~480x above the isotropic single-step prediction, and a level shift that large swamps the sign of a
per-hop direction effect.

That 480x was not assumed to be the window. #233 checked it against the competing account, the
graph's parallel routes, which predicts the gap grows with relay degree. It is flat: `r = −0.121`,
467.9x / 459.2x / 747.6x / 636.4x / 408.4x at degrees 5 to 9. **The confound is named, so it can be
removed.** A tick-aligned single-step read has no window to accumulate over.

**What a hop is** — unchanged from #233, and the ticket names the operator:

    M(e_in → v → e_out)  =  F_{v,e_out} · gain_v · F_{v,e_in}ᵀ

`Sheaf.message_passing_phase` writes `x_v ← x_v − gain_v · Σ_e F_evᵀ (F_ev x_v − y_e)`, and
`broadcast` is read from the *pre-update* stalk, so a displacement arriving on `e_in` at tick `t`
reaches `e_out`'s disagreement at tick `t + 1`. That one-tick offset is the whole of *tick-aligned*:
every quantity below pairs `d_in(t)` with `d_out(t + 1)` and never with `d_out(t)`.

**The read needs no new instrument**, which is what the ticket predicted. `detectability.branch`
already returns `[ticks, edges, m]` of disagreement — the whole trace, as vectors — so the arriving
*direction* is a difference of two traces that `construction_grading.py` already computes and then
reduces to a norm. This read takes the reduction one step later.

---

## 2. The trap: `A < 1` is not misalignment

The ticket asks for

    A  =  ‖M d‖ / (‖M‖_F ‖d‖ / √m_in)

the measured direction against the isotropic one. The natural reading — `A < 1` means the arriving
direction is worse than an average one — is **wrong**, and reporting it would have answered this
ticket in the affirmative out of pure noise.

For an isotropic `d`, `E[A²] = 1` holds exactly. But `A` itself is skewed below its own mean square,
and the skew is worst precisely where this graph lives. Against a **rank-1** operator
`A = |⟨v₁, d⟩| √m_in`, so `A²/m_in ~ Beta(1/2, (m_in−1)/2)`, whose median is **0.816 at `m_in = 4`**
and falls to **0.674** as the width grows. Never 1.

This is the map's own standing correction in its exact dual. #142's isotropic probe against
near-rank-1 maps reported a 1e14 deficit that was not there; reading a median of `A` against 1 would
manufacture a misalignment that is not there, from the same fact about the same maps.

So the verdict is **never** read off `A` against 1, and not against a graph-wide null either — the
bar moves with both the edge width and the operator's rank. It is read off `A` against the null
distribution of `A` for isotropic directions **through the same operator**, drawn per hop, and the
reported statistic is the arriving direction's **percentile within that null**. Under the hypothesis
that arriving directions are isotropic that percentile is uniform, so its mean is 50 and any
systematic effect moves it.

`benchmarks/alignment_read.py null` prints the trap before any measurement is involved, over every
directed hop the graph admits at construction:

TODO-NULL-TABLE

The null is drawn from the operator's singular values rather than by pushing vectors through the
matrix — the same distribution exactly, not an approximation of it, and
`tests/test_alignment_read.py` holds it against the brute force it replaces at five quantiles. The
calibration test is the one that matters most: fed genuinely isotropic directions, the instrument's
mean percentile must land at 50, and it does.

---

## TODO-SECTIONS

---

## What this ticket does not claim

- **It rules on nothing.** The remedy species is closed and this reopens none of it. It explains a
  measurement, which is what a `wayfinder:research` ticket is for.
- **`M` omits the body's Jacobian.** The ticket defines the operator as the two maps and the gain,
  and this read uses that definition. `A` is invariant to the body's *scale* — it cancels top and
  bottom — so #233's stated body limit does not bite on the primary statistic. It would bite if the
  body *rotated* `u = F_inᵀ d`, and §TODO-BODY-REF reads the empirical single step rather than
  assuming either way.
- **The widest path is a max-min path, not a flow.** It names the edge the predicate binds on, which
  is what ADR-0021 asks of it, and is not a claim that transport travels only along it.
- **Hops from 24 trials in two directions are not independent samples.** Trials share one trained
  surface and paths overlap. The percentile statistic is a description of this population; what does
  not depend on the run at all is §2's null, which is a property of the operators.
