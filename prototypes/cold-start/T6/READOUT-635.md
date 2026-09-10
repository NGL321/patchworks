# B62 (#635): the prediction rule spends the traffic's rank, the transport rule buys the agreement, and the count is not rank in another basis

[#635](https://github.com/NGL321/patchworks/issues/635), under
[the map](https://github.com/NGL321/patchworks/issues/532). Opened by
[B57 (#629)](https://github.com/NGL321/patchworks/issues/629), which built
[B52 (#622)](https://github.com/NGL321/patchworks/issues/622)'s graded agreement
instrument, read it on three arms, and left two things it could not settle: it
measured the rank falling *while the rule ran* and could not attribute it, and
it could not separate its own count from the traffic's rank.

**Surface.** `c925866` (2026-09-09, *Merge pull request #627 from
NGL321/worktree-b50-cosine-spectra-618*), `main` — B57's own surface, so the
comparison against its ladder is row for row and no figure is carried across a
rebuild. `n = 32`, 150 predicting cells, `δ_P` is `[3861, 4800]`, `G` is
`[6654, 4800]`, seed 42, one seed. `δ_P` is held against
`Diagnostics.whole_graph` on every arm before anything is read: **rank 3000,
`dim H⁰` 1800 of 4800 columns**, agreeing to the integer on all four runs.

**Rig.** `b62_frozen.py` (items 1–2: the rules under a switch),
`b62_qlevel.py` (item 3: the pencil and the banded traffic),
`b62_stated_rank.py` (item 3's first pass, superseded in part — see §5),
`b62_table.py` (the four arms side by side), `b62_run.sh` (the stages, strictly
one at a time — and see the warning in its header, which was written after the
fact and at cost).

**One console log is not a record and is kept marked as such.** This box's
low-memory guard killed the runner script three times while its python child
survived and kept writing, so two frozen 20,000-tick arms ran concurrently and
**interleaved their lines into `635-frozen-20k-INTERLEAVED.log`**. The JSON records are
unaffected — each is rewritten whole at every checkpoint, so a file is always one
run — and the surviving 20,000 record is internally consistent with monotonic
elapsed times. Every number in this readout is taken from a JSON record and none
from that log.

**B49's two-objects line.** Everything below is on **node stalks**: `x_t` is
`sheaf.evidence()`, what every page is actually holding. Audience
differentiation rides alongside per B48 and is never argued from.

---

## 0. The control is exact, and it costs no argument

`T0/run.py`'s `teaching_read` is

```python
for outcome in run_ticks(agent, ticks, seed=seed):   # patchworks.agent.run
    recorder.observe()
    bias.step()                                      # PredictionRule
    if agent.sheaf.ticks > 1:
        transport.step()                             # TransportRule
```

so dropping the two rule steps leaves **the same world, the same dome, the same
seed stream and the same stimulus**, with nothing learning. There is no second
code path and nothing is re-implemented. The check that it worked is on every
row: audience differentiation reads **0.5307** and exposure **9.05/32** at every
rung of the frozen arm, bit-stable, because the maps never move.

## 1. The rank holds with the rules off — training spends it

Uncentered traffic effective rank, `T = 1,000`, the headline window, against
B57's own baseline arm at the same rungs on the same surface:

| taught ticks | rules **off** (this ticket) | rules **on** (B57) |
|---|---|---|
| construction | 2.7802 | 2.7656 |
| 100 | 2.7807 | 2.8073 |
| 200 | 2.7808 | 2.8378 |
| 500 | 2.7757 | 2.9825 |
| 1,000 | 2.7787 | **1.0967** |
| 2,000 | 2.7816 | 1.0132 |
| 5,000 | 2.7785 | 1.0130 |
| 10,000 | 2.7821 | 1.0064 |
| 20,000 | **2.7825** | **1.0045** |

The frozen arm is flat to the third decimal across every rung — a spread of
**0.0068** over the full 20,000-tick horizon, smaller than the window-to-window
spread of the same quantity. The rules-on arm *rises* to 2.9825 by 500 and then
falls to 1.0967 by 1,000.

**The frozen column is one arm, and a second one replicates it.** The rungs above
are `635-frozen-baseline-seed42-20000.json` throughout — a single run, so no row
is borrowed from another. An earlier frozen arm to 5,000
(`635-frozen-baseline-seed42-5000.json`, a separate run) reads 2.7798 / 2.7801 /
2.7804 / 2.7750 / 2.7786 / 2.7819 / 2.7812 on its own seven rungs, **spread
0.0069** — the same flatness, independently. Neither arm's spread reaches a
thousandth of the collapse it is being compared against.

**So the rank does not decay on its own, and the collapse is not the body's or
the sandbox's.** It is spent, and it is spent between 500 and 1,000 taught
ticks — the same interval in which B57 measured `N(0.25)` switching 0 → 1.0000.
`N(θ)` stays **0.0000 at every θ at every rung** on the frozen arm: the count
never switches on without training either.

**The branch that fires** is *the rank holds with the rules off → training spends
it, and item 2's split says which rule. This makes the collapse an
objective-design fact and [#633](https://github.com/NGL321/patchworks/issues/633)
inherits it.*

## 2. Which rule spends it — and the two rules pull opposite ways

`PredictionRule` and `TransportRule` separately, on the same dome, seed and
window, against B57's both-on arm. Uncentered ER at `T = 1,000`, with the `q` of
the leading direction and the weight below `θ = 0.25` beside it, because §2a.
**The frozen column here is the through-5,000 arm**, which is the one that runs
to the same horizon as the split arms; §1's is the 20,000 arm. Each table is one
run per column.

| rung | frozen | bias only | transport only | both (B57) |
|---|---|---|---|---|
| construction | 2.7798 | 2.7798 | 2.7798 | 2.7656 |
| 100 | 2.7801 | 2.8069 | 2.8109 | 2.8073 |
| 200 | 2.7804 | 2.8324 | 2.9705 | 2.8378 |
| 500 | 2.7750 | 3.0288 | 3.1619 | 2.9825 |
| 1,000 | 2.7786 | **1.3399** | 3.1837 | **1.0967** |
| 2,000 | 2.7819 | **1.0087** | 3.3307 | 1.0132 |
| 5,000 | 2.7812 | **1.0089** | **3.1347** | 1.0130 |

`q` of the leading direction, same arms and rungs:

| rung | frozen | bias only | transport only | both (B57) |
|---|---|---|---|---|
| construction | 0.8832 | 0.8832 | 0.8832 | 0.8842 |
| 500 | 0.8833 | 0.8808 | 0.1779 | 0.6694 |
| 1,000 | 0.8832 | 0.5550 | 0.1133 | 0.2117 |
| 5,000 | 0.8835 | **0.5495** | **0.0636** | 0.1830 |

**The two rules are not two halves of one movement. They are antagonistic on the
rank.**

- **`PredictionRule` spends the rank and buys no agreement.** It reproduces the
  collapse on its own — 2.7798 → 1.0087 by 2,000, against the both-on arm's
  1.0132 — on the same interval, and its leading direction stops at `q = 0.5495`
  with `A(0.25) = 0.0039`. Nothing is agreed at the end of it.
- **`TransportRule` buys the agreement and *raises* the rank.** It never
  collapses: 2.7798 → 3.1837 at 1,000, peaking at 3.3307 and ending at 3.1347.
  Its leading direction reaches `q = 0.0636` — **better alignment than both rules
  together manage** (0.1830) — at `A(0.25) = 0.5552`.

So the collapse #635 asks about is **`PredictionRule`'s alone**, and the
agreement B52's instrument was built to see is **`TransportRule`'s alone**. The
shipped rule runs both, and the trained surface is their compromise: the rank
ends where the prediction rule puts it and the alignment part-way to where the
transport rule would.

**What each pays with — B48's joint rule, per rule.** `PredictionRule` leaves
exposure and audience differentiation **bit-identical to the frozen arm** at
every rung (9.05/32 and 0.5307), because it updates `K` and never touches a
restriction map; the traffic it collapses is the traffic the body *generates*.
`TransportRule` **raises** differentiation (0.5307 → 0.5505) and spends
**1.96 dimensions of exposure per cell** (9.05 → 7.09). Read against B48's
constraint — *agreement rising while differentiation stays nonzero* — the
transport rule passes the differentiation clause outright and pays in the
exposure column instead. The both-on arm spends differentiation (→ 0.4826) and
less exposure (→ 8.62), which is neither rule's own price.

**This is a reading of the shipped rules and not a candidate.** Per #635 and the
map's *Plan, don't do*: no arm here is scored, and *transport-only reaches better
`q`* is a measurement of the existing rule, not a proposal to ship it alone. What
it does establish is that the two levers are separable in fact, which no reading
on this map had shown.

## 2a. `N(θ)` is not a reading without `A(θ)` — and one of this ticket's own rows proves it

`N(θ) = (Σ_{q≤θ} w)² / Σ_{q≤θ} w²` is a participation ratio **taken inside the
counted band**. It says how many directions dominate *among those below θ*; it
says nothing about how much of the traffic they carry. That is `A(θ) = Σ_{q≤θ} w`,
which B52 specifies alongside it and the profile has carried all along.

The two come apart on a row measured here. The bias-only arm at 5,000 ticks
reads `N(0.25) = 1.0000` — and `A(0.25) = 0.004173`, one direction carrying
**0.4%** of the traffic, while the direction carrying **99.5%** sits at
`q = 0.5493`. Read on `N` alone that arm looks like B57's trained baseline. Read
with `A` it is the opposite of it:

| arm | rung | ER | `A(0.25)` | `N(0.25)` | `q` of leading direction |
|---|---|---|---|---|---|
| frozen | 5,000 | 2.7812 | 0.000000 | 0.0000 | 0.8835 |
| bias-only | 2,000 | 1.0087 | 0.000000 | 0.0000 | 0.5236 |
| bias-only | 5,000 | 1.0094 | **0.004173** | **1.0000** | 0.5493 |
| both-on (B57) | 2,000 | 1.0132 | **0.993434** | 1.0000 | 0.2047 |
| both-on (B57) | 20,000 | 1.0045 | **0.997746** | 1.0000 | 0.1783 |

**B57's own conclusions are untouched** — `A(0.25)` runs 0.9547 → 0.9977 across
its trained ladder, so its `N = 1.0000` is one direction carrying essentially the
whole traffic, exactly as it read it. But the statistic has a vacuous regime and
the map should know where it is.

**Standing constraint, offered:** *an agreement count is quoted with the weight
it counts — `A(θ)` beside `N(θ)` — or it is not a reading.* Cheap: the profile
already carries both, and no run is needed to comply.

## 3. The window, swept at the horizon and not only at construction

#635 asks for this because B57's amendment found the centred form growing with
`T`. It does, and it does not settle. Frozen arm, every rung:

| rung | T=250 | T=500 | T=1,000 | T=2,000 |
|---|---|---|---|---|
| construction | 2.7383 / 13.4027 | 2.7620 / 19.1675 | 2.7798 / 24.1975 | 2.7849 / 31.0925 |
| 1,000 | 2.7347 / 12.9707 | 2.7678 / 18.7758 | 2.7786 / 23.9349 | 2.7906 / 30.4111 |
| 5,000 | 2.7407 / 13.2321 | 2.7687 / 18.8200 | 2.7812 / 24.1460 | 2.7913 / 29.9323 |

Uncentered is stable to ~2% across an eightfold window. Centred **doubles and
keeps climbing**, with no sign of a plateau at `T = 2,000`.

**The centred column is bounded by the window, not by the surface.** A sample
covariance over `T` ticks has rank at most `T` against 4,800 columns, so more
ticks buy more directions and the participation ratio rises with them; what it
reports is how long you looked. So the comparison in §1 is stated **uncentered**,
and **24.9 is correctly no target** — neither is 31.1, which is the same
quantity at a different window on the same untrained surface.

## 4. `ker(G) = ker(δ_P)`, exactly — exact agreement is unavailable to anything visible

Read off the operators with no traffic in it:

- `rank(G)` = `rank(δ_P)` = **3000**, both kernels **1800**-dimensional;
- `‖G‖ = 1.4607`, and **`δ_P` restricted to `ker(G)` has top singular value
  `1.75e-15`** — a relative `1.2e-15`.

The two kernels coincide. **Every direction the cells agree about exactly is a
direction no incident map can see at all**, which is
[B48](https://github.com/NGL321/patchworks/issues/615)'s *the trained
architecture's entire `H⁰` is the privacy reserve* arriving from the operator
side rather than the counting side. It is also B52's own case for a graded
statistic, reached independently: **exact agreement is unavailable to any
visible direction by construction**, so a bar written on `earned` is asking for
a measure-zero event *and* an invisible one.

## 5. The attainable `q` spectrum is a property of the maps alone

`q` is the generalized Rayleigh quotient of `(δ_Pᵀδ_P, GᵀG)`, so on the visible
subspace — `G`'s row space, where the denominator is positive definite — the
attainable levels are the pencil's eigenvalues. Writing `G = U S Vᵀ` and
`x = V_rᵀ S_r⁻¹ y` gives `q(x) = ‖A y‖²/‖y‖²` with `A = (δ_P V_rᵀ) S_r⁻¹`, so
`A`'s squared singular values **are** the spectrum and no traffic is needed to
read it. Over the 3,000 visible directions:

| q_min | p01 | p10 | median | p90 | q_max |
|---|---|---|---|---|---|
| 0.0022 | 0.0195 | 0.1462 | **1.0000** | 1.8525 | 1.9975 |

441 of 3,000 (14.7%) lie at `q ≤ 0.25`; 1,531 (51%) at `q ≤ 1`.

**The median visible direction sits at exactly 1.0000.** B52 argued `q = 1` is
the no-relationship point from the algebra; it is also the centre of this
surface's own spectrum, measured.

**A first pass is superseded here.** `b62_stated_rank.py` built its *agreeing*
family inside `ker(δ_P)` on the assumption that agreed-about and visible can be
had together. §4 says they cannot, so both numerator and denominator of that
family are roundoff and its `q` is not a reading. The proof is internal: read in
`G`'s kernel basis the same 1800-dimensional subspace gives `q̄ ≈ 1.00`, and in
`δ_P`'s it gives `q̄ ≈ 0.06–0.17`. Those rows are kept in the record and are not
quoted. **One instrument note falls out and is filed, not fixed**: B57's `seen`
guard is `den > 1e-24 · max(den)`, relative to the largest denominator *within
the set being read*, so it does not fire on a family whose denominators are
uniformly tiny. Real traffic mixes visible and invisible directions, so **B57's
reading is untouched**; a future synthetic read is not.

## 6. `N(θ)` separates from rank — the ledger's row 8 resolves to its first reading

Traffic of **stated rank** in bands of **stated level**, put through B57's
`agreement()` unmodified. `q` is a Rayleigh quotient, so on the span of the `r`
lowest pencil directions *every* direction has `q ≤ q_r` whatever basis the
instrument's own SVD picks — a stated band stays stated however the traffic is
mixed.

| band | r = 1 | r = 2 | r = 4 | r = 8 |
|---|---|---|---|---|
| **low_q**, `q ∈ [0.0022, 0.0092]` | **1.0000** | **1.9422** | **3.4905** | **6.6410** |
| **mid_q**, `q = 1.0000` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **high_q**, `q ∈ [1.9905, 1.9975]` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

`N(0.25)`, `unseen 0` on every row. **The band decides whether anything is
counted; the rank decides how much.** Both halves of the ledger's second reading
fail outright:

- **a rank-one traffic reads 0.0000** when its direction is generic, against
  that reading's *`q_i` sits far from 1 wherever the traffic is rank one*. The
  isotropic family confirms it independently at every rank: `q̄` = 1.0162 /
  1.0244 / 1.0150 / 1.0130 at `r` = 1 / 2 / 4 / 8, `N(0.25) = 0.0000` throughout;
- **a rank-eight traffic reads 6.6410** when its directions are low-`q`.

`N < r` on the low band because `N` is a participation ratio over sampled
weights at `T = 1,000`, not because anything is dropped.

**Why a generic direction cannot reach the band by luck.** 14.7% of *basis*
directions sit at `q ≤ 0.25`, but a random direction is a weighted mixture of
all 3,000 and its `q` concentrates at the spectrum's mean ≈ 1 — which is what
the isotropic family measures. Reaching `q ≤ 0.25` takes alignment with the
low-`q` end, and that is why B57's matched-generic null reads `N_gen = 0.0000`
at every level, arm, checkpoint and redraw.

## 7. B57's account of the real reading, rebuilt from stated ingredients

B57 said *the one thing the room agrees about is the standing mean*. Synthesise
exactly that — one low-`q` mean at amplitude `ratio`, a mid-`q` variety riding
on it — and the real signature comes back:

| ratio | stated rank | ER unc | ER cen | N(0.25) unc | N(0.25) cen |
|---|---|---|---|---|---|
| 10 | 2 | 1.0215 | 1.0000 | **1.0000** | 0.0000 |
| 10 | 4 | 1.0616 | 2.9708 | **1.0000** | 0.0000 |
| 10 | 8 | 1.1423 | 6.8614 | **1.0000** | 0.0000 |
| 3 | 8 | 2.8451 | 6.8759 | **1.0000** | 0.0000 |
| 1 | 8 | 7.8421 | 6.8736 | 0.0000 | 0.0000 |

against the real trained baseline's ER 1.0045 / 1.611 and `N` 1.0000 / 0.0000.
Uncentered ER runs 1.02 → 1.06 → 1.14 → **2.85** while `N(0.25)` stays pinned at
**1.0000**, because only the mean is low-`q`. That is the count and the rank
moving independently in the one regime the real reading actually occupies, and
it is the last tie between them broken.

## 8. The joint report — B48's standing constraint, on every row

Audience differentiation and exposure ride with every agreement number above.
Exposure (effective median of 32) and differentiation (median), all four arms:

| rung | frozen | bias | transport | both (B57) |
|---|---|---|---|---|
| construction | 9.05 / 0.5307 | 9.05 / 0.5307 | 9.05 / 0.5307 | 9.05 / 0.5307 |
| 500 | 9.05 / 0.5307 | 9.05 / 0.5307 | 7.30 / 0.5262 | 9.04 / 0.5206 |
| 1,000 | 9.05 / 0.5307 | 9.05 / 0.5307 | 6.87 / 0.5282 | 8.90 / 0.5161 |
| 5,000 | 9.05 / 0.5307 | 9.05 / 0.5307 | **7.09 / 0.5505** | 8.62 / 0.4826 |

Three things this column says that no agreement number does:

- **The frozen and bias arms are bit-identical here**, at 9.05 and 0.5307 on
  every rung. For the frozen arm that is the check that the rules are off; for
  the bias arm it is a *finding* — `PredictionRule` collapses the traffic's rank
  by a factor of 2.8 **at exactly zero cost in either column**, because it never
  touches a restriction map.
- **`TransportRule` pays in exposure and *gains* differentiation** — 1.96
  dimensions per cell spent, differentiation ending **above** where it started
  (0.5307 → 0.5505). Read against B48's constraint, *agreement rising while
  differentiation stays nonzero*, the transport rule does not merely pass the
  differentiation clause, it moves that column the right way; its whole price is
  in the column B48 asks to be reported beside it.
- **The both-on arm's price is neither rule's own.** It spends differentiation
  (→ 0.4826, and → 0.4893 by 20,000) and less exposure (→ 8.62) than transport
  alone. The compromise costs a column that neither lever costs by itself.

On §6's synthetic rows both are properties of the *maps*, so they are constant
by construction across every band and rank — worth stating once: **no synthetic
row buys or spends anything, so none of §6's separation is bought with
exposure.**

## 9. Which branch fired

Two of #635's branches fired, and neither is the ambiguous one.

- **"The rank holds with the rules off → training spends it, and item 2's split
  says which rule. This makes the collapse an objective-design fact and
  [#633](https://github.com/NGL321/patchworks/issues/633) inherits it."**
  Fired on §1, split in §2. The collapse is `PredictionRule`'s.
- **"`N(θ)` separates from rank → B52's construction is vindicated on its own
  terms and the ledger's row 8 resolves to its first reading."** Fired on §6.
  The instrument measures alignment; the rank only sets how much of it can be
  counted.

The two branches that did not fire are recorded as refused rather than untested:
*the rank collapses with the rules off* is false at every rung (§1), and
*`N(θ)` tracks stated rank and not alignment* is false in both directions (§6),
so **#633 does not inherit a second name for one quantity** — it inherits a real
one.

**Three findings sit outside the branch table and go to the surprise ledger
([#520](https://github.com/NGL321/patchworks/issues/520)), one row each:**

1. **The two rules are antagonistic on the rank** (§2). `TransportRule` alone
   *raises* traffic ER to 3.0024 while `PredictionRule` alone collapses it to
   1.0089. The map has spoken of *the rule* as one lever throughout; on this
   quantity it is two, pulling opposite ways.
2. **`ker(G) = ker(δ_P)` exactly** (§4), so exact agreement is unavailable to any
   visible direction by construction — B48's `earned = 0` has an operator-side
   proof, not only a counting-side measurement.
3. **`N(θ)` has a vacuous regime** (§2a): it reads 1.0000 on a direction carrying
   0.4% of the traffic. The fix is to quote `A(θ)` beside it and costs nothing.

**What this hands #633.** That ticket asks whether `TransportRule` needs the
traffic's full metric `C⁻¹`. Its whole case rests on how much structure `C` has
— and §2 says the near-rank-one `C` it would whiten is **manufactured by the
prediction rule, not by the transport rule whose update it proposes to change**.
Under `TransportRule` alone the traffic keeps effective rank 3.0. Whether that
relocates the lever is #633's to rule on; this ticket does not rule it.

## What this does not establish

- **One seed, one arm.** Everything is seed 42 on the `baseline` arm. B57 read
  three arms and found the count identical on all three; the *rank* collapse was
  read on all three too, but the rules-off control exists only on `baseline`.
- **No threshold is set and no candidate is scored.** The `q` bands are read off
  the surface rather than chosen, and the amplitude ratios in §7 are properties
  of a synthesis, not proposals.
- **§6 is an instrument result, not a claim about the trained surface.** It says
  what `N` can and cannot distinguish. That the *real* surviving direction is
  low-`q` is B57's reading plus §5's spectrum, not a separate measurement here.
- **The `p` and `k_v` story is untouched.** Nothing here reads the mask.
- **The split arms stop at 5,000 ticks**, not B57's 20,000. The rules-on collapse
  is complete by 1,000 and settled by 2,000, so 5,000 brackets it with two rungs
  to spare — but *the long horizon is unread on the split*, and B38's stamp is
  per-run. Whether the two rules stay antagonistic at 20,000 is not shown, and it
  is [B68 (#645)](https://github.com/NGL321/patchworks/issues/645)'s first item.
  **The frozen arm does carry the full horizon** — 2.7798 → 2.7817 over 20,000
  ticks — so item 1's *the rank does not decay on its own* is read to B57's own
  horizon and not bracketed.
- **The published transport figures were re-taken.** The first transport arm was
  run with the matched-generic null switched off to survive this box's
  low-memory guard, and its output file was then overwritten by the runner
  chain's own transport run before the commit. Everything above is the surviving
  record; the two runs agree in every qualitative respect (ER never below 3.1,
  `q` of the leading direction below 0.07, `A(0.25)` ≈ 0.55) and differ in the
  second decimal. **All three surviving arms do carry the matched-generic null**,
  three redraws each, reading `N_gen = A_gen = 0.0000` — so the excess is
  measured on every arm here rather than inherited from B57.
- **§2's antagonism is measured, not explained.** *Why* `TransportRule` raises
  traffic ER while `PredictionRule` collapses it is not established here. The
  obvious asymmetry — one rule writes `K` and the other writes the maps — is a
  location, not a mechanism.
