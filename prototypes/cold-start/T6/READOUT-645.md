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

**The falsifier does not fire.** `TransportRule` settles at 2.9855, flat from 5,000 on.
The antagonism holds at B57's own horizon. Three things past 5,000 that B62 could not see:

- **The transport arm repays its price.** B62's *"2.03 dimensions of exposure per cell"*
  is a 5,000-tick reading. By 20,000 the cost is **0.35** dimensions, and audience
  differentiation ends at **0.5868**, *above* construction's 0.5307 rather than below.
  The alignment holds while the price is repaid: `q_top` is 0.0543 at 5,000 and 0.0642
  at 20,000.
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

**Proposed standing constraint:** *a difference between two arms that run `TransportRule`
is quoted against a replicate of one of them at the same rung, or it is not quoted.* Every
figure in §1–§3 that rests on such a difference has one.

### The rules keep the body alive, and the stamp cannot say so

Stamped per rule mode, because the rules change the commands the agent issues.

| mode | B38's rule (last above) | first fall | re-crossings | share above |
|---|---|---|---|---|
| frozen | 150 | 160 | **0** | 0.0075 |
| bias | **4,900** | 90 | **70** | 0.0390 |
| transport | **15,790** | 80 | **131** | 0.0690 |

`b38_stall.boundary` takes the *last* window above `MOVING`, so it stamps the transport arm
live to 15,790 while its first fall is at 80. Neither number describes the run. The frozen
arm dies once and stays dead; the rule-carrying arms **re-excite the body**.

Share of moving windows per interval between rungs:

| mode | 0–100 | 200–500 | 500–1k | 1k–2k | 2k–5k | 5k–10k | 10k–20k |
|---|---|---|---|---|---|---|---|
| frozen | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| bias | 0.800 | 0.000 | **0.320** | 0.000 | **0.180** | 0.000 | 0.000 |
| transport | 0.700 | 0.000 | 0.000 | 0.050 | 0.087 | **0.068** | **0.066** |

The co-occurrence is exact. The bias arm's collapse (500 → 1,000, ER 3.03 → 1.34) falls in
its single largest re-excitation window; the transport arm's sustained rank elevation from
5,000 on rides on sustained intermittency where the frozen arm reads 0.000 throughout.

**B62's *"the collapse is training's and neither the body's nor the sandbox's"* holds as
attribution** — the frozen arm does not move — but there is a third channel its location
argument did not consider: the rules change what the body does, which changes the stimulus,
which returns in the traffic. **Co-occurrence does not give the direction and none is
claimed here.** It does mean the readings past ~160 ticks are not *"drift under a frozen
stimulus"* on the rule-carrying arms, which is what B38's rule would have to say about them.
