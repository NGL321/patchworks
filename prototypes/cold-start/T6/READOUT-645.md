# B68 (#645): the antagonism holds, it is not `K`, and the pair is its own object

[#645](https://github.com/NGL321/patchworks/issues/645), under
[the map (#532)](https://github.com/NGL321/patchworks/issues/532). Opened by
[B62 (#635)](https://github.com/NGL321/patchworks/issues/635), which measured the
antagonism and said in as many words that it had not explained it.

**Surface.** `84a97d9` for the transport arm, `9aa9837` for the bias arm, `5a47540`
for the `both` arms, stall stamp and mechanism runs — the stamp moved only because
this ticket committed its own instruments between runs. `git diff` across that whole
range over `src/`, `prototypes/cold-start/T0`–`T3` and `benchmarks/` is **empty**, so
no trajectory-driving code differs between any two readings below. Condition
`baseline`, seed 42, window 1,000, buffer 2,000 throughout.

**Two objects, kept apart** per [B49 (#616)](https://github.com/NGL321/patchworks/issues/616).
The traffic and the per-cell block readings are **node stalks**; `q`, audience
differentiation and exposure are on the **transport operator**; `ρ(K)`, `σ(K)`,
`ρ(used)` and `τ` are on the **cell operator**, a third object. No number is carried
between them. Item 2's correlations relate two of them across cells, which is the
question B62 named as untaken — not a number quoted as another.

---

## 1. The horizon: the falsifier does not fire

Both split arms to 20,000 on B62's rig unmodified. #645 pre-registered that if
`TransportRule`'s ER fell to the both-on arm's ~1.01, the antagonism was a transient
of the first 5,000 ticks and the ticket's premise was gone.

| ticks | bias ER | bias q_top | bias aud-diff / exp | transport ER | transport q_top | transport aud-diff / exp |
|---|---|---|---|---|---|---|
| 0 | 2.7798 | 0.8832 | 0.5307 / 9.05 | 2.7798 | 0.8832 | 0.5307 / 9.05 |
| 500 | 3.0288 | 0.8808 | 0.5307 / 9.05 | 3.3558 | 0.1955 | 0.5264 / 7.45 |
| 1,000 | 1.3399 | 0.5550 | 0.5307 / 9.05 | **3.4018** | 0.1611 | 0.5225 / 7.18 |
| 5,000 | 1.0090 | 0.5494 | 0.5307 / 9.05 | 3.0024 | 0.0543 | 0.5313 / 7.02 |
| 10,000 | 1.0001 | 0.5929 | 0.5307 / 9.05 | 2.9894 | 0.0709 | 0.5574 / 7.88 |
| 20,000 | **1.0001** | 0.6125 | **0.5307 / 9.05** | **2.9855** | 0.0642 | **0.5868 / 8.70** |

`N(0.25)` is quoted with `A(0.25)` per B62's ledger row 11: at 20,000 the transport arm
reads `N` 1.0000 on `A` **0.5728**, and the bias arm reads `N` 0.0000 on `A` 0.0000 —
the bias arm's earlier `N(0.25) = 1.0000` at 5,000 sits on `A = 0.0039` and means nothing.

**The falsifier does not fire.** `TransportRule` settles at 2.9855, flat from 5,000 on; a
replicate of the same arm reads 3.1637. Both are ~2.0 above the ~1.01 the falsifier was
pre-registered against, which is **11× the arm's own replicate spread at that rung**
(0.178). The antagonism holds at B57's own horizon, and the gap between the two rules —
1.0001 against ≥2.98 — dwarfs anything the noise could supply.

**One weaker claim does not clear its noise at the horizon, and is stated as such.**
*`TransportRule` raises the rank above the frozen arm* is a separation of **0.203** at
20,000 against a spread of **0.178** — about 1.1×, marginal, though both replicates land
above the frozen arm. At **1,000 ticks the same claim is a separation of 0.62 against a
spread of exactly zero**, because the arm reproduces bit-exactly through 2,000. So the
claim is established at the early rungs and is not established at 20,000 on two runs.

Three things past 5,000 that B62 could not see:

- **The transport arm repays its price.** The exposure cost is 2.03 dimensions at 5,000
  and **0.35 (replicate 0.45)** at 20,000 — a fall of ~1.6 against a spread of 0.14, so
  the repayment clears its noise. The alignment holds while the price is repaid: `q_top`
  is 0.0543 at 5,000 and 0.0642 at 20,000, spread 0.012.
  *The direction of the differentiation column is not this ticket's finding* — the map's
  amended B62 entry already carries *raising differentiation rather than spending it* at
  5,000 (0.5505). What is added here is the **horizon**: 0.5868 (replicate 0.5807) at
  20,000, a separation of 0.055 from the frozen arm's flat 0.5307 against a spread of
  0.0061, so it is ~9× its noise and still climbing at the last rung.
- **The bias arm's alignment gets worse as the rank falls further.** `q_top` bottoms at
  0.5232 (2,000) and climbs to 0.6125 while ER goes 1.0087 → 1.0001.
- **The bias arm is free in both joint columns at the full horizon**, not only to 5,000:
  0.5307 and 9.05, bit-identical to frozen at all nine rungs, because it writes `K` and
  never touches a restriction map.

## 2. The mechanism: both halves of the collapse, and it is not `K`

### The collapse is per-cell *and* synchrony, and they move together

The traffic is `sheaf.evidence()`, `[predicting cells, n]` flattened in `dome.predicting`
order, so block `c·n : (c+1)·n` is cell `c`'s own node stalk and the blocks are aligned
with `CellOperators`' own `[cells, ·]` leading dimension. A global ER of 1 can be reached
two ways, and they are separable.

| | block ER median | sync \|cos\| | joint ER |
|---|---|---|---|
| construction | 1.3919 | 0.9764 | 1.082 |
| bias @2,000 | **1.0068** | **0.9998** | **1.003** |
| transport @1,000 | **1.5539** | **0.9680** | **1.123** |
| both @1,000 | 1.0328 | 0.9976 | 1.011 |

**Both move, in the same direction as each other and in opposite directions between the
rules.** `PredictionRule` drives each cell's own stalk traffic onto one direction *and*
locks the cells onto a shared temporal profile; `TransportRule` raises the per-cell
dimensionality and de-synchronises. So *"the loop samples one direction of it"* is
supported at the per-cell level — and it is only half of what happens.

One reading rides with it: **the cells are already strongly synchronised at
construction** (|cos| 0.9764, joint ER 1.082 of a possible 150). The construction ER of
2.78 is carried by the per-cell blocks, not by the cells being independent of one another.

### The candidate account fails, and the control is exact

#645's candidate, off [B44 (#610)](https://github.com/NGL321/patchworks/issues/610): the
rule drives `K` toward something low-rank or strongly non-normal and the world loop
samples one direction of it.

**On the transport arm, `ρ(K)`, `σ(K)`, `ρ(used)` and the non-normality ratio read
exactly 1.0000 on all 150 cells at every rung, and `τ` is infinite everywhere — while
that arm takes the global ER from 2.7798 to 3.4018.** The traffic's rank moves
substantially with `K` bit-identical to construction. `K` is not necessary to move it.

On the arm that does produce the collapse, the correlation has the **wrong sign** and
**decays to nothing** exactly as the collapse completes:

| bias, ticks | r(block ER, ρ_used) | r(block ER, ρ_K) | block ER median |
|---|---|---|---|
| 100 | −0.209 | +0.473 | 1.4021 |
| 500 | −0.167 | +0.306 | 1.6214 |
| 1,000 | −0.029 | +0.336 | 1.2116 |
| 2,000 | **−0.079** | **−0.159** | **1.0068** |

Spearman across cells, `n = 150` at every entry. `τ`'s coefficient is identical to
`ρ(used)`'s, being a monotone function of it. The account predicts a *positive*
r(block ER, ρ_used) — a cell whose `K` is more non-normal should carry the more collapsed
traffic — and the measured sign is negative, weak, and gone by the time the collapse is done.

**So #645's branch *"the rank tracks `ρ(used)` or `τ` per cell"* does not fire, and B44's
adversary constraint is not reached by this route.** B44's own identity does replicate
here on the `baseline` arm: the non-normality column equals `ρ(used)` to four decimals in
every row, so retention is `K`'s departure from normality on this arm as it was on B44's.

## 3. Composition: the pair is not predictable from the solos

Re-run on **this** surface rather than inherited: B62 put its solo arms (read on
`f51667d`) beside B57's both-on baseline (read on `c925866`), and the two do not agree at
construction — 2.7798 against 2.7656, same condition and seed. Item 3's whole question is
whether the pair's outcome is predictable from the solos, so an out-of-interval verdict
must not rest on a surface difference.

`out` is how far the pair lies outside the interval its two solos span, 0 when inside.
Per column, never fused — B49's struck move.

**Audience differentiation**

| ticks | bias | transport | both | out |
|---|---|---|---|---|
| 5,000 | 0.5307 | 0.5313 | 0.5104 | **−0.0203** |
| 10,000 | 0.5307 | 0.5574 | 0.4845 | **−0.0462** |
| 20,000 | 0.5307 | 0.5868 | 0.4771 | **−0.0536** |

**Exposure**

| ticks | bias | transport | both | out |
|---|---|---|---|---|
| 100 | 9.0468 | 8.3662 | 9.1318 | **+0.0850** |
| 200 | 9.0468 | 7.8919 | 9.1265 | **+0.0797** |
| 10,000 | 9.0468 | 7.8756 | 9.0606 | +0.0139 |

**The pair leaves the interval on two columns, in opposite directions, and the
differentiation excursion widens monotonically.** Both solos hold differentiation flat or
raise it; the pair lowers it. On exposure the pair spends *more than either solo* at the
early rungs — on a column `PredictionRule` cannot touch at all, since it never writes a
restriction map and its exposure is 9.0468 at every rung of every arm it runs alone.

**And the divergence is present at tick 100**, before the rank collapse, which happens
between 500 and 1,000. So *"`TransportRule` trained on traffic the prediction rule had
already collapsed"* is not available as the account: at tick 100 the bias arm's traffic ER
is 2.8069 against frozen's 2.7801, and the pair has already parted from both solos on
exposure by +0.085.

### The claim clears its own noise

The `both` arm was replicated — same seed, same command, separate process.

| rung | ΔER | Δq_top | Δaud-diff | Δexposure |
|---|---|---|---|---|
| 0 – 5,000 | **exact** | **exact** | **exact** | **exact** |
| 10,000 | 1.23e-03 | 1.57e-03 | 3.43e-03 | 1.80e-02 |
| 20,000 | 3.32e-04 | 2.56e-03 | 3.94e-03 | 6.74e-02 |

The exposure excursions sit at ticks 100 and 200, where the arm reproduces **bit-exactly**,
so they are not samples at all. The differentiation excursion at 20,000 (−0.0536) is about
**14×** the spread measured on the same arm at the same rung (0.0039).

## 4. Two things this ticket did not go looking for

### The arms that run `TransportRule` do not reproduce, and the null is not why

Re-running B62's `transport` arm did not reproduce B62's stored record: 3.0024 against
**3.1347** at 5,000, already parted at tick 100, on byte-identical trajectory code.
Three short runs separate the candidates.

| | @100 | @200 |
|---|---|---|
| A vs B, **transport** — identical invocations, one process | ΔER **2.30e-02** | ΔER **2.47e-01** |
| A vs B, **frozen** — the same, with no rule at all | ΔER **2.60e-05** | ΔER **2.19e-04** |
| A vs C — differing only by the construction-time null | **0.00e+00** | **0.00e+00** |
| both vs its replicate — identical command, separate process | exact | exact |

So: the **construction-time matched-generic null is exonerated**; an identical command in
a **separate process reproduces exactly**; and a second arm built **inside one process**
does not reproduce the first. The non-reproducibility is an execution-order effect on a
trajectory sensitive at the last bit, not a loose seed — `agent.run` resets the world with
an explicit seed, so the stimulus stream is pinned per call.

**The frozen control locates it, and it is not the rule's.** The divergence is already
there with *nothing learning*, at 2.6e-05 by tick 100 — and `TransportRule` amplifies it
by roughly **three orders in 200 ticks**. So the rule is the amplifier and the shared
execution is the source. On the frozen arm audience differentiation and exposure stay
bit-identical (0.530703, 9.046780), as they must when no map moves: the divergence shows
up only in the stalks.

What remains as the difference between B62's stored record and this re-run is the null
taken **at every checkpoint** rather than only at construction — a diagnostic moving the
trajectory it is measuring. Chasing that is not this ticket's question.

**How loose, per arm.** `|run − replicate|`, same seed, same command, separate processes:

| rung | both: ER / aud-diff / exposure | transport: ER / aud-diff / exposure |
|---|---|---|
| 0 – 2,000 | exact | **exact** |
| 5,000 | exact | 1.17e-01 / 8.6e-03 / 1.4e-01 |
| 10,000 | 1.2e-03 / 3.4e-03 / 1.8e-02 | 1.30e-01 / 7.4e-03 / 5.8e-03 |
| 20,000 | 3.3e-04 / 3.9e-03 / 6.7e-02 | **1.78e-01** / 6.1e-03 / 1.0e-01 |

The `transport` arm is **an order looser in ER than the `both` arm** at the horizon, and
this is why it matters: **the map's own numbers now disagree because of it.** The amended
B62 entry quotes transport at 5,000 as ER 3.1347, `q` 0.0636, 1.96 dimensions and
differentiation 0.5505; this ticket's run of the same arm at the same rung reads 3.0024,
0.0543, 2.03 and 0.5313. Both are honest single runs of the same command.

**Proposed standing constraint:** *a difference between two arms that run `TransportRule`
is quoted against a replicate of one of them at the same rung, or it is not quoted* — and
where a rung reproduces bit-exactly, say so, because that is stronger than any spread.
Every figure in §1–§3 that rests on such a difference has one, and the one that does not
clear its spread is marked as not clearing it.

### The rules keep the body alive, and the stamp cannot say so

Stamped per rule mode, because the rules change the commands the agent issues.

| mode | B38's rule (last above) | first fall | re-crossings | share above | final `std_max` |
|---|---|---|---|---|---|
| frozen | 150 | 160 | **0** | 0.0075 | 2.44e-03 |
| bias | **4,900** | 90 | **70** | 0.0390 | 9.32e-07 |
| transport | **15,790** | 80 | **131** | 0.0690 | 1.32e-04 |
| both | 70 | 80 | **0** | **0.0035** | **2.33e-10** |

`b38_stall.boundary` takes the *last* window above `MOVING`, so it stamps the transport arm
live to 15,790 while its first fall is at 80. Neither number describes the run.
[B61 (#634)](https://github.com/NGL321/patchworks/issues/634) landed this defect on the map
while this ticket ran, as a standing constraint that a long-horizon reading reports both
rules and the re-crossings; `b68_stall.py` was written to that shape independently and
complies.

**One amendment to B61's account, offered rather than assumed.** B61 describes it as *the
world* sporadically re-crossing the threshold after it has died. On these four modes the
frozen arm re-crosses **zero** times while the rule-carrying arms re-cross 70 and 131 — so
on this arm the re-crossing is not the world's, it is **the rules'**. The frozen arm dies
once and stays dead; the rule-carrying arms **re-excite the body**.

Share of moving windows per interval between rungs:

| mode | 0–100 | 200–500 | 500–1k | 1k–2k | 2k–5k | 5k–10k | 10k–20k |
|---|---|---|---|---|---|---|---|
| frozen | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| bias | 0.800 | 0.000 | **0.320** | 0.000 | **0.180** | 0.000 | 0.000 |
| transport | 0.700 | 0.000 | 0.000 | 0.050 | 0.087 | **0.068** | **0.066** |
| both | 0.700 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

The co-occurrence is exact. The bias arm's collapse (500 → 1,000, ER 3.03 → 1.34) falls in
its single largest re-excitation window; the transport arm's sustained rank elevation from
5,000 on rides on sustained intermittency where the frozen arm reads 0.000 throughout.

**And this is a fourth column where the pair leaves the interval its solos span — §3's
verdict in the world rather than in the maps.** Each rule *alone* keeps the body
intermittently alive; **together they kill it deader than running no rule at all**: 0
re-crossings against 70 and 131, a moving share of 0.0035 *below the frozen control's*
0.0075, and a final `std_max` of 2.33e-10 against frozen's 2.44e-03, six orders down.
Neither solo does this and the no-rule control does not either.

**B62's *"the collapse is training's and neither the body's nor the sandbox's"* holds as
attribution** — the frozen arm does not move — but there is a third channel its location
argument did not consider: the rules change what the body does, which changes the stimulus,
which returns in the traffic. **Co-occurrence does not give the direction and none is
claimed here.** It does mean the readings past ~160 ticks are not *"drift under a frozen
stimulus"* on the rule-carrying arms, which is what B38's rule would have to say about them.

---

## Verdicts against #645's branch table

- **`TransportRule`'s rank falls to the both-on arm's by 20,000 →** *did not fire.* Both
  replicates settle ≥2.98 against the ~1.01 pre-registered, 11× the arm's own spread.
- **The antagonism holds →** **fired.** The two rules are separable levers in fact:
  1.0001 against ≥2.98 at 20,000, a gap that dwarfs any noise, with prices that differ in
  kind — the bias arm free in both joint columns at every rung, the transport arm's cost
  repaid to 0.35 dimensions. *Which rule a candidate is aimed at is now a live design
  choice.* **The lever is not picked here**, per the ticket's own instruction; it is
  ticketed.
- **The rank tracks `ρ(used)` or `τ` per cell →** *did not fire, and it is refuted rather
  than merely unsupported.* The transport arm moves the rank 2.7798 → 3.4018 with `K` at
  exactly 1.0000 on all 150 cells, and on the arm that does collapse the correlation has
  the wrong sign and decays to −0.079 as the collapse completes. **B44's adversary
  constraint is not reached by this route.**
- **The pair's outcome is not predictable from the solos →** **fired, and it is the
  headline.** The pair leaves the interval its solos span on **four** columns — audience
  differentiation (−0.0536 at 20,000, ~14× spread), exposure (+0.085 at tick 100, where
  the arm reproduces bit-exactly), the ER trajectory at 1,000 (−0.241), and the body's own
  motion, where the pair is deader than the no-rule control. The divergence is present at
  **tick 100**, before the rank collapse, so *the transport rule trained on already-
  collapsed traffic* is not available as the account. **Composition is its own object and
  this map cannot reason about the shipped rule by reasoning about its parts.**
- **Outside every branch →** two, both filed: the arms running `TransportRule` do not
  reproduce and the map's own B62 numbers already disagree because of it; and the rules,
  not the world, are what re-excite the body past the stall.

## What this ticket did not do

It did not pick a lever, set a threshold, or score a candidate architecture — #645's
closing constraint. It did not explain *why* `PredictionRule` collapses the traffic: it
established what the collapse **is** (per-cell and synchrony together), and ruled out the
one candidate mechanism the ticket named. The positive account is still open.
