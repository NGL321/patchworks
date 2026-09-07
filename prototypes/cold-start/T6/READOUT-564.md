# B16 (#564): why training seeks concentration, and what ratio 1 costs

Under [the map](https://github.com/NGL321/patchworks/issues/532). Instruments:
`b16_drive.py`, `b16_junction.py`, `b16_saturation.py`, `b16_coherence.py`.

**The answer in one line.** The transport objective *is* "make the two ends of an
edge agree", the two ends of an edge are the `U` bases of two consecutive hops,
and agreement between them is exactly what makes seven contractions compound
onto one direction. The collapse is the rule's objective being met, not a
pathology it suffers — and because the cell's own traffic is **rank one**, the
rule has evidence about exactly one direction, which is the one it makes
coherent.

## 1. The erosion is not in the angles

Every account on this map so far has been about principal angles. Across `p = 8`'s
existing 100k run (`555-reserve-baseline-seed42-100000.json`):

| ticks | composed ER | `cos_product_er` | cos lead | cos 2nd | σ_min/σ_max |
|---|---|---|---|---|---|
| 100 | 2.2903 | 3.0439 | 0.9795 | 0.9544 | 1.0000 |
| 1000 | 1.8516 | 3.1147 | 0.9836 | 0.9602 | 1.0000 |
| 10000 | 1.4039 | 3.0801 | 0.9921 | 0.9659 | 1.0000 |
| 100000 | 1.4237 | 2.5793 | 0.9915 | 0.9417 | 1.0000 |

**`cos_product_er` is flat while composed ER halves.** The per-hop cosine spectra
— the whole object [B1](https://github.com/NGL321/patchworks/issues/537) and
[#533](https://github.com/NGL321/patchworks/issues/533) reasoned about — barely
move. `angles.py` already names the residue: *"The gap between them is exactly
what the `U`-rotations carry."*

`b16_saturation.py` confirms it from the other side. The obvious mechanism —
training drives each hop's leading cosine to 1 — is parameter-free and
**testable**, and it fails where it matters:

| `p` | k_v | generic | top cosine saturated | predicted ratio | measured | error |
|---|---|---|---|---|---|---|
| 8 | 24 | 2.2332 | 2.1655 | 0.970 | 0.629 | **+0.341** |
| 12 | 20 | 2.4360 | 2.4035 | 0.987 | 0.910 | +0.077 |
| 16 | 16 | 2.8899 | 2.8816 | 0.997 | 1.004 | −0.007 |
| 20 | 12 | 4.0000 | 4.0000 | 1.000 | 1.000 | +0.000 |

Exact at `p = 16` and `p = 20` — because forced overlap has *already* pinned the
top cosine at 1.0000 there, so saturation is a no-op — and badly wrong at `p = 8`,
which is the arm the question is about.

## 2. Where it does live: the junctions

A hop is `U_out (V_outᵀ V_in) U_inᵀ`, carrying lane `e_k` to lane `e_{k+1}`. Take
two consecutive hops. Hop `k`'s `U_out` and hop `k+1`'s `U_in` are orthogonal
bases of **the same lane**, held by the cells at that edge's two ends. The
transport objective is `‖F_v x_v − y_e‖ / (‖F_v x_v‖ + ‖y_e‖)` — *make what this
end sends match what the other end sent.* Bringing those two bases into agreement
is the rule's entire purpose.

`b16_junction.py` tests it by counterfactual: recompose each chain with a random
orthogonal inserted at every junction, **leaving every hop exactly as trained**.

| ticks | `p=8` ER | `p=8` scrambled | `p=16` ER | `p=16` scrambled |
|---|---|---|---|---|
| 100 | 2.2903 | 2.2813 | 2.9691 | 2.8913 |
| 1000 | 1.8709 | 2.3178 | 2.9475 | 2.9073 |
| 10000 | 1.4027 | 2.2656 | 2.8743 | 2.9298 |
| 20000 | 1.4776 | **2.2505** | 2.9089 | 2.9526 |

At `p = 8` scrambling the junctions restores **2.2505 against a generic 2.2652** —
within 0.7% — while the true surface sits at 1.4776. **The whole collapse is
carried by the junctions**; nothing about the individual hops has changed in a
way that matters. At `p = 16` the same operation moves the reading 1.5%.

## 3. Why one direction, and why the fixed point is not zero

`b16_drive.py` reads the quantity the rule actually learns from. The cell's own
node stalk — the `x_v` that is the input-side factor of *every* transport update
both of that cell's incident maps take — has effective rank **1.002**. The
objective has evidence about **one** direction per junction and none at all about
the rest.

That is why the fixed point is a plateau rather than zero, and the ordering makes
it non-trivial. `b16_coherence.py` imposes `r` coherent junction directions on
#540's generic instrument and composes:

| `p` | generic | `r=0` | `r=1` | `r=2` | measured |
|---|---|---|---|---|---|
| 8 | 2.2652 | 1.001 | 0.487 | 0.872 | 0.629 |
| 12 | 2.4305 | 1.015 | 0.541 | 0.864 | 0.910 |
| 16 | 2.9283 | 1.001 | 0.766 | 0.889 | 1.004 |
| 20 | 4.0000 | 1.000 | 1.000 | 1.000 | 1.000 |

`r = 0` reproduces the generic to within 1.5%, which independently validates the
scramble counterfactual. And note the ordering: **perfectly sorted junctions give
`cos_product_er` = 3.07, *above* the random baseline 2.25; training lands at 1.48,
*below* it.** Aligning every direction spreads the spectrum. Aligning exactly one
while the rest stay incoherent is what produces domination — and one is what
rank-one traffic supplies. `p = 8`'s measured 0.629 brackets between `r = 1`
(0.487) and `r = 2` (0.872), i.e. partial coherence on rather more than one
direction, which is what an effective rank of 1.002-and-rising should give.

## 3a. The scalar that moves, and the correction it forces

Measuring the coherence directly took two attempts, and the first was wrong. The
individual maps' leading left factors (`junction_lead`) are **flat** — 0.257 to
0.207 at `p = 8`. What compounds is not `F`'s leading direction but the **hop's**,
whose singular basis folds in the `V_outᵀ V_in` Gram and both `sigma` diagonals.
On that pair (`hop_junction`, `|<P_k[:,0], Q_{k+1}[:,0]>|`):

| ticks | `p=8` coherence | `p=8` 2nd | `p=8` ER | `p=16` coherence | `p=16` ER |
|---|---|---|---|---|---|
| construction | 0.2206 | 0.2499 | 1.3825 | 0.2269 | 1.3433 |
| 1000 | 0.2536 | 0.2046 | 1.8709 | 0.2400 | 2.9472 |
| 3000 | 0.2661 | 0.2030 | 1.5788 | 0.2477 | 2.8788 |
| 10000 | **0.3257** | 0.2236 | **1.4027** | 0.2472 | 2.8771 |
| 20000 | 0.2868 | 0.2388 | 1.4443 | **0.1928** | 2.9326 |

**At `p = 8` the leading junction coherence rises ~48% over construction while
the second direction stays at the random baseline** — one direction going
coherent, exactly as rank-one traffic predicts. And it tracks the outcome
*within* the arm: ER bottoms at 1.4027 where coherence peaks, and eases back to
1.4443 as coherence eases to 0.2868.

**At `p = 16` coherence never builds at all — it falls, 0.2269 to 0.1928.**

## 4. What `p` actually does — a correction to the ticket's own framing

The ticket says `p = 16` works by *"denying training the room to erode, not by
changing what training wants"*. The first half is too weak and the second needs
qualifying. Both arms, 20k ticks, same seed:

| reading | `p = 8` | `p = 16` |
|---|---|---|
| stalk traffic effective rank | 1.006 | 1.006 |
| carried subspace onto traffic top direction | 0.9943 | 0.9967 |
| transport disagreement | 0.00133 | 0.00150 |
| junction coherence, construction → 20k | 0.221 → **0.287** | 0.227 → **0.193** |
| composed ER, 100 → 20k | 2.290 → 1.501 | 2.969 → 2.859 |

**What training wants is indeed identical** — same traffic, same alignment onto
it, same progress on its own objective. But `p = 16` does not merely leave the
drive nowhere to bite: **the junction coherence never forms.** With 67% forced
overlap the agreement objective is *satisfiable without rotating the `U`s at
all*, and disagreement duly falls just as far (0.00150 against 0.00133) with the
junctions left incoherent.

So the sharper statement is: **`p` changes the cheapest route to satisfying the
objective.** That matters for the remedy question, because it means forced
overlap is not the only thing that could work — anything that lets edge-agreement
be reached without junction alignment would do, and forced overlap is merely the
one way this map has found.

## 4a. B6's law is a correlate

Within `p = 8`, at a **fixed** `k_v` where
[B6](https://github.com/NGL321/patchworks/issues/546)'s headroom law is a constant
and can predict nothing, cells with more anisotropic traffic reach higher leading
cosines: `corr(traffic_er, cos_leading)` runs **−0.45** from 3k ticks on. B6's law
describes the **room**; it is not the drive.

## 5. Answering the ticket's four questions

**(1) What is the drive a function of?** The transport rule's own update, and nothing else. Its gradient factors as `∂L/∂F_v = (∂L/∂o) · x_vᵀ` — rank one, with the cell's **own node stalk** on the input side — and a relay cell feeds *both* incident maps from that one stalk. Not ticks: the decay tracks the objective's value (disagreement 0.0126 → 0.0013 alongside ER 2.29 → 1.50), not the tick count. Not prediction-error reduction: `transport_gradient` is `argnums=0` on the map tensor and reads only own-stalk and neighbour belief; the prediction rule is a separate phase. **What sets the fixed point is the traffic's effective rank, 1.002** — one direction of evidence buys one coherent junction direction, so the result is domination (full rank 3, spectrum `[1, 0.113, 0.002]`) and the ratio plateaus at 0.629 rather than falling to zero.

**(2) Is B6's headroom law the mechanism or a correlate?** **A correlate — it measures the room, not the drive.** The drive is *identical* in both arms at 20k: stalk traffic effective rank 1.006 / 1.006, carried subspace onto the traffic's top direction 0.9943 / 0.9967, transport disagreement 0.00133 / 0.00150. Same traffic, same alignment, same objective progress — and `p = 8` erodes 2.290 → 1.501 while `p = 16` does not (2.969 → 2.859). Within `p = 8`, at a **fixed `k_v`** where B6's law is a constant and can predict nothing, `corr(traffic_er, cos_leading)` runs **−0.45** from 3k ticks on. The two make the different prediction B16 said they would: B6's law says only `k_v` can move it; the mechanism says **the traffic's effective rank can too, and nothing on the record has tried that.**

**(3) Can the ratio reach one without forced overlap?** The property a transport rule needs: **the composed object must not be a pure product of learned hops whose junctions an agreement objective can align.** Two ways to have it — raise the rank of the traffic the rule learns from to the rank you want composed, or give the composed object a path not subject to junction alignment.

- **#540 §5's residual `h ← (I+T)h` has the second.** Composed becomes `∏(I + T_k)`, a sum over subsets of hops that includes the **empty product `I`** — full rank, and *not a learned parameter*, so the drive cannot erode it. **That is the mechanism it owed against B1's law**: B1's law governs the product of the `T_k`; the residual makes the composed object a sum whose terms pass through fewer hops. One caveat for whoever takes it: the residual is stated on the stalk, and lanes differ in width (`m_in = 4`, `m_out = 12`), so `I + T` is not well-typed hop-to-hop — it has to live in stalk space, which changes what `composed_reads` measures.
- **The channel-heterogeneous candidate does not have either.** Every term of a sum of Kronecker products still passes through all seven hops, and each channel learns from the *same* rank-one stalk, so the channels align to the same direction and the sum stays dominated. It owes a **decorrelation mechanism across channels**, which nobody has stated.
- **The cheapest untried lever is the traffic's rank**, not the map's shape. It is the only route on the record to ratio 1 that spends neither selectivity nor an ADR on transport's parameterisation.

**(4) Is the drive necessary?** **Yes, for any transport rule whose objective is edge-agreement — which is what a sheaf's transport rule is.** Agreement between an edge's two ends *is* coherence of the two `U` bases, and coherent junctions compound contraction. So **given a pure-product transport learned by an agreement objective, ratio 1 is purchasable in exactly two ways**: kill the contraction (make each hop near-isometric — what `p` does, priced in forced overlap), or add a path the drive cannot erode (the residual). The map should record that the trade is forced *under those premises* — and equally that it is **not** forced in general, because raising traffic rank escapes it and is untested.

