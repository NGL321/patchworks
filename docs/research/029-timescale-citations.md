# Citation pass: timescales and persistence (patchworks#29)

Validates the design closed in [#7](https://github.com/NGL321/patchworks/issues/7)
(`docs/spec/05-timescales.md`, `docs/adr/0005-timescale-is-persistence-not-a-schedule.md`).
Citations validate after the fact per the map's sequencing rule; this document does not reopen the
closed design — it records where a source confirms a claim and flags where one threatens it.
Vocabulary follows [`CONTEXT.md`](../../CONTEXT.md): Patchworks' side of every comparison in its own
terms, the prior art's in its field's. Where a source could not be reached, that is stated. Bullet 5
was already answered by [`027-regional-jacobian-spectra.md`](./027-regional-jacobian-spectra.md);
§5 records only what this pass adds.

## Headline verdict, stated plainly

**The recollection the decision leaned on hardest is correct, and the mechanism the decision
declines is confirmed to be universal in the prior art — but the answer to the ticket's most exposed
question is that no prior system gets timescale separation from persistence alone.** In detail:

- **Rao's APC: confirmed, exactly as recollected.** A higher level's step is a fixed count of
  lower-level steps ("3 macro- and 3 micro-steps"); the options framing is aspirational, with
  termination functions named as *future work* in the paper itself. Nothing to correct.
- **Every multi-timescale recurrent architecture surveyed sets the timescale by a dedicated
  parameter or schedule.** Clockwork RNN: hand-picked exponential periods, admitted arbitrary.
  HM-RNN: a *learned* schedule, but still a discrete binary gate that freezes state. FeUdal: a
  hand-set horizon `c = 10` plus a dilated LSTM that is a modulo clock divisor by another name.
  MTRNN: per-unit time constants `τ = 5` and `τ = 70`, assigned by the designer.
- **No precedent found for timescale separation from persistence alone.** ADR-0005's move —
  timescale as an uninstrumented *side effect* of a bias vector already spent on other jobs, in a
  system with no timescale parameter at all — is, as far as this pass could establish, **without
  precedent**. That is a genuine novelty claim and it is also the risk.
- **One committed claim is threatened.** The spec dismisses depth's slow resonant modes as `~2L`
  ticks, "not steerable," and "an order of magnitude short of task duration." The delay-coupled
  oscillator literature does not characterise them that way: slow rhythms there run to *seconds*
  while the constituent units stay near their intrinsic fast frequency, and their period is set by
  **coupling strength** — slower than the bound and explicitly steerable by a gain. See §3.
- **The protected-subspace claim has a real precedent, in the closest possible field, used for a
  different purpose.** Neural sheaf diffusion engineers `dim ker(Δ_F)` deliberately so information
  survives diffusion, but never as *memory across time*. Patchworks' use is a short, defensible step
  past published work rather than an unsupported leap.

---

## 1. Rao's active predictive coding — confirmed, no correction needed

*Source: Rao, Gklezakos & Sathish (2024), "Active Predictive Coding", Neural Computation 36(1);
arXiv:2210.13461, read via ar5iv (the arXiv PDF returned undecoded binary to the fetcher).*

The ticket asks whether APC's multi-timescale structure is really a **relative action count per
level**, and whether implementations use a **fixed `K`** despite options-like framing. Both hold,
verbatim:

> "The top level runs for T2 steps (referred to as 'macro-steps'). For each macro-step, the bottom
> level runs for T1 'micro-steps'."

That is a relative step count per level and nothing else — no clock, no time constant, no
termination. The counts are small integers fixed per experiment: "we used 3 macro- and 3 micro-steps
(except 4 macro-steps for Omniglot)". The options framing is present — the paper calls its level-2
output a "macro-action (or option) `z_t`" and describes "abstract actions … whose execution executes
a sequence of primitive actions as prescribed by the option's lower level policy" — but the paper is
explicit that the options machinery is not implemented:

> "Future implementations will explore the use of termination functions (Sutton et al., 1999; Eslami
> et al., 2016) to allow a variable number of time steps at each level and for each input. For the
> present paper, we focus on a two-level model."

**Verdict: confirmed.** The single recollection ADR-0005 leaned on hardest survives at source, in
the paper's own words. It also strengthens ADR-0005's rejection of *termination-driven abstract
steps* as "the most faithful to the active-predictive-coding framing and the most expensive": APC's
own authors reach the same cost conclusion and defer it.

One thing the spec does not say: APC's `T1`/`T2` **is** the clock divisor, and it is APC's
mechanism, not its instrument. `05-timescales.md` demotes the divisor to instrument while citing APC
as the source of the goal. A defensible divergence — but a divergence, not a redescription.

---

## 2. Temporal abstraction generally — commitment is formalised; latency is absent

*Sources: Sutton, Precup & Singh (1999), read at source pp. 185–191; Vezhnevets et al. (2017), via
ar5iv.*

The spec claims what planning needs is **ratio in commitment**, not **ratio in latency**. The
options framework supplies the commitment half with unusual precision. An option is

> "a policy π : S × A → [0,1], a termination condition β : S⁺ → [0,1], and an initiation set I ⊆ S"

and, crucially,

> "If the option is taken, then actions are selected according to π until the option terminates
> stochastically according to β."

Theorem 1 makes the commitment structural: "the decision process that selects only among those
options, **executing each to termination**, is an SMDP." The abstract decision is *not revisited*
until `β` fires — exactly "an abstract belief that stands across many primitive steps", and the
options paper's own gloss uses *persist* for it: "some of which persist for a single time step,
others of which are temporally extended". The multi-time model `p^o_{ss'} = Σ_k p(s',k) γ^k` then
makes duration the only thing distinguishing an abstract step from a primitive one.

**Latency does not appear.** Nothing in the options framework, and nothing in FeUdal, models a
decision made on *stale* evidence; `s_{t+c} − s_t` in FeUdal's transition policy gradient is a
horizon, not a lag.

**Verdict on the ticket's question — is the commitment/latency distinction drawn anywhere, and under
what name? No source found drawing it.** The literature has one half (commitment, under the names
*option duration*, *temporal extension*, *executing to termination*) and simply lacks the other,
because the settings it works in are delay-free. The distinction is Patchworks' own, forced by its
being a message-passing graph where hop distance *is* lag. A point in the spec's favour — the
argument is not a restatement of something already known — but it also means **the argument that
killed depth-as-mechanism is uncorroborated**. Nobody has said it is wrong; nobody has said it.

One further note against the spec's framing: FeUdal's Manager emits a goal every timestep, pooled
over `c` (`w_t = φ(Σ_{i=t−c}^{t} g_i)`). Even the canonical "abstract level runs slower"
architecture does not hold one abstract decision across `c` steps; it smooths a stream of them. What
`c` buys is credit-assignment horizon — which `05-timescales.md` correctly disclaims — so FeUdal is
a weaker ally for the commitment claim than its reputation suggests.

---

## 3. Delay is a phase shift, not a decimation — unstated in this setting, and the secondary claim is threatened

*Sources: Gutteridge et al. (2023), abstract read at source; Qie, Martin et al. (2026), HTML read at
source.*

**The elementary claim: no source found stating it about message-passing graphs.** DRew is the only
delayed-message-passing GNN work directly on point, and its delay is a *routing* device — "a delay
mechanism that permits skip connections between nodes depending on the layer and their mutual
distance", introduced precisely because instant rewiring "lose[s] the inductive bias provided by
distance on the graph." It therefore *treats* delay as scheduling when information arrives rather
than what arrives — consistent with the spec — but never states it and never discusses bandwidth or
frequency content. The spec's premise is elementary signal processing and is not in doubt; what is
absent is any prior statement of it as an argument against depth-gives-timescale in a graph.

**The secondary claim is threatened.** `05-timescales.md` writes off depth's slow modes:

> "a loop of length `L` with unit delay has slow resonant modes on the order of `2L` ticks. At the
> sandbox's 50 Hz and a graph of diameter ~12 that is ~0.5 s — an order of magnitude short of task
> duration, not steerable, and an artifact rather than a mechanism."

The delay-coupled network literature supports neither the magnitude nor the "not steerable". Qie et
al. find that in delay-coupled rings "the individual oscillators continue to fire very close to
their intrinsic frequency, yet their relative phases evolve along stable limit cycles" (§1) — the
slow mode lives in *phase differences*, not in a `2L` round-trip resonance — and its period is set
by coupling gain:

> "a small `g` produces very slow phase-difference dynamics. Consequently, to obtain slow patterns
> with periods of a few seconds, we increase `g` greatly" (§6)

Seconds, and tuned by a gain. Both grounds for dismissal are contradicted for this class of system.
Not contradicted: the primary argument (unit delay removes no frequency content, so depth buys
latency and not commitment), and Patchworks' reconciliation gain is bounded for other reasons
(`02-tick-semantics.md`), so the knob Qie et al. turn may not be available at the range they turn
it. **A flag on the wording of a dismissal, not on the decision.** Searched for and **not found**:
any closed form relating loop delay to emergent slow-mode period; any statement in the delayed-GNN
literature that delay is not a low-pass filter.

---

## 4. Slow state in a protected subspace — real precedent, adjacent purpose

*Source: Bodnar et al. (2022), "Neural Sheaf Diffusion", NeurIPS; read via ar5iv.*

The ticket asks whether anyone uses an invariant subspace of a message-passing operator as
**deliberate memory**. The nearest answer is strong on "invariant subspace, deliberately" and weak
on "memory". Neural sheaf diffusion states the invariance directly — "It can be shown that in the
time limit,
each feature channel is projected into `ker(Δ_F)`" (§3), formalised as Lemma 21: "Solutions `X(t)`
to the diffusion in Equation 3 converge as `t→∞` to the orthogonal projection of `X(0)` onto
`ker(Δ_F)`." The *deliberate* part is the paper's whole contribution: because a non-trivial sheaf
makes `dim ker(Δ_F) > 1`, information survives what would otherwise be oversmoothing — "when the
sheaf is non-trivial, discretised parametric diffusion processes have greater control than GNNs over
their asymptotic behaviour", with "sufficient stalk width (i.e., dimension `d`) … needed in order to
solve tasks involving more than two classes". That is a construction quantity governing how much
survives the operator, structurally the same object as `05-timescales.md`'s bound
`dim H⁰ ≥ Σ_v max(0, n − Σ_{e∋v} m_e)`. Same field, same formalism, same design move.

The gap is what the surviving subspace is *for*. Bodnar et al. use `ker(Δ_F)` as the asymptotic
**classification** space — what a diffusion converges to at `t→∞`. Patchworks uses `ker δ` as state
that persists tick to tick under the cell's own dynamics while reconciliation runs alongside. **No
source found using a message-passing operator's kernel as memory across time.** The spec is already
careful about the split: insulation from neighbours is one half, persistence under the cell's own
dynamics the other, and the second half has no analogue in Bodnar et al., who have no per-node
recurrence at all. Their "private opinions" gloss — "sheaf diffusion can be seen as a
'synchronisation' process over the graph, where all the private opinions converge towards global
agreement" — is a coincidence of language, not a cognate: their private opinions are what diffusion
*destroys*; Patchworks' private features are what reconciliation *cannot touch*.

**Verdict: supported, one step out.** The design move is published in the closest possible
neighbourhood; this particular use of it is not.

---

## 5. Timescale from activation-region Jacobians — answered elsewhere

Referred to [`027-regional-jacobian-spectra.md`](./027-regional-jacobian-spectra.md), which carries
the verified primary citations (Schoenholz et al. arXiv:1611.01232 on bias variance shifting the
order-to-chaos regime with weights held fixed; Pennington et al. arXiv:1711.04735 and
arXiv:1802.09979 on Jacobian spectra; S4D's geometric timescale initialisation) and ran the go/no-go
rig. This pass adds one citation and repeats none.

**Bias-only translation selecting an effective time constant is established in gated RNNs.** Tallec
& Ollivier (2018), arXiv:1804.11188, from the abstract at source: "learnable gates in a recurrent
model formally provide quasi-invariance to general time transformations in the input data", and "we
derive a new way of initializing gate biases in LSTMs and GRUs". The chrono initialisation
`b_f ~ log(U([1, T_max − 1]))` and the identification of `1/g(t₀)` as "the local forgetting time of
the network at `t₀`" are **taken from a secondary write-up** — see Honest gaps. The limit that
matters: there, bias reaches timescale through a sigmoid gate that multiplies the state,
analytically; in Patchworks it reaches timescale through *which activation region the cell lands
in* — the step 027 §1 calls unestablished and 027 §6 measured. Nothing here changes that.

---

## 6. The exposed question: does anything get timescale separation from persistence alone?

*Sources: Koutnik et al. (2014) and Chung et al. (2017), via ar5iv; Yamashita & Tani (2008), read at
source. Plus §2's FeUdal and §1's APC; leaky-integrator ESN via 027.*

### Findings, in order of how close each comes

| system | where its timescale comes from | verbatim |
|---|---|---|
| Clockwork RNN | hand-picked schedule; true decimation | "only the output of modules `i` that satisfy `(t mod T_i) = 0` are executed"; "`T_i = 2^(i−1)`"; "The choice of the set of periods … is arbitrary" |
| FeUdal | hand-set horizon + modulo divisor | `c = 10`; dilated LSTM updates one of `r` state groups per step by modulo indexing |
| HM-RNN | *learned* discrete gate; still decimation | "`c_t^l = c_{t−1}^l` … (COPY)"; "our model learns the intrinsic timescales from the data" |
| MTRNN | hand-set per-unit rate; no decimation | "a group of fast context units (τ = 5) and a group of slow context units (τ = 70)" |
| leaky ESN | hand-tuned global leaking rate | via 027 |
| APC | hand-set relative step count | "3 macro- and 3 micro-steps" (§1) |

Two need comment. **HM-RNN is the only one whose schedule is learned**, and its COPY branch is exact
invariance of cell state — structurally the closest thing in this literature to persistence. But it
is conditioned on a discrete control signal the architecture reads and branches on, precisely what
`05-timescales.md` forbids ("Nothing in the architecture reads a cell's timescale"). **MTRNN is the
only one with no decimation at all** — every unit runs every step and is slow purely because it
leaks slowly — and what Yamashita & Tani show *emerges* is the functional hierarchy ("functional
hierarchy can self-organize through multiple timescales in neural activity, without explicit spatial
hierarchical structure"), *given* timescales that are a designer input.

### Verdict, unambiguously

**No precedent found.** Every system surveyed obtains timescale separation from one of three things:
a hand-set schedule (Clockwork, FeUdal, APC's `T1`/`T2`), a learned discrete gate the architecture
branches on (HM-RNN), or a dedicated per-unit rate parameter (MTRNN, leaky ESN, S4D,
chrono-initialised LSTM). ADR-0005 takes none of them. Its position — timescale as an
*uninstrumented side effect* of a per-cell bias vector already committed for unrelated reasons, in
an architecture with no timescale parameter and a standing prohibition on reading timescale — has,
as far as this pass could establish, **no prior system behind it**.

Two honest readings. It is a real novelty claim and the spec is entitled to it. It is also why the
clock divisor is built first: the mechanism is not merely unproven in Patchworks, it is unattested
anywhere, and `05-timescales.md`'s decision to build the fallback first is fully vindicated. The
nearest half-precedents — MTRNN's persistence-not-decimation, chrono-init's bias-sets-timescale —
each supply one half and hand-set the other. Nothing supplies both without a knob.

---

## Revision tickets recommended (recommendations only — not created)

1. **`05-timescales.md`, *Depth does not supply it*, the `~2L` paragraph — the dismissal is
   overstated.** Qie et al. show delay-coupled loops producing second-scale slow rhythms whose
   period is set by coupling gain, contradicting both "an order of magnitude short" and "not
   steerable". The primary argument survives; only the secondary sentence needs weakening, ideally
   to "not steerable *within Patchworks' bounded reconciliation gain*". Also touches ADR-0005
   *Alternatives considered → Emergence from depth*.
2. **`05-timescales.md` *What is actually wanted*, and ADR-0005 *Context* — record that APC's
   `T1`/`T2` is a clock divisor and is APC's mechanism, not its instrument.** The spec cites APC as
   the source of the goal while demoting the divisor to instrument. Defensible, but state it as a
   divergence: a reader who checks the source finds the divisor promoted.
3. **Low priority: state that the phase-shift argument is uncorroborated** (§2, §3). House style is
   to say so where a load-bearing argument is the spec's own.

**Not recommended:** promoting the clock divisor from instrument to mechanism. The ticket
pre-accepts that finding; this pass does not support it. Every weakness ADR-0005 names is confirmed
by Koutnik et al.'s own admission that the period set "is arbitrary" and by Chung et al.'s "finding
proper timescales in the CW-RNN remains as a challenge". The literature runs *away* from fixed
divisors. Keeping it as an already-validated fallback is right.

## Honest gaps

- **Tallec & Ollivier (2018) full text could not be reached.** ar5iv returns an empty rendering
  (PDF-only submission); the yann-ollivier.org PDF is an http↔https redirect loop; OpenReview sits
  behind browser verification. Two attempts each. Abstract read at source; the chrono formula and
  the "local forgetting time" gloss in §5 come from a secondary write-up and are marked as such. No
  verdict here turns on them.
- **APC's planning experiments do not state a micro-step count.** The `3 × 3` figure is from the
  vision experiments; the "fixed `K`" finding is verified there and inferred for planning from the
  absence of any termination mechanism.
- **Only ten sources were read**, deliberately. Not read: Dayan & Hinton (1993), Jaeger et al.
  (2007) leaky-integrator ESN in the original, Precup's thesis on interrupting options, the
  delayed-recurrent-control literature (Stépán, Insperger), and Durstewitz-group PLRNN work on
  multiple timescales. Any of the last three could sharpen §3 or §5.

## Sources

- Rao, Gklezakos & Sathish (2024). Active predictive coding. *Neural Computation* 36(1).
  arXiv:2210.13461 (ar5iv).
- Sutton, Precup & Singh (1999). Between MDPs and semi-MDPs. *Artificial Intelligence* 112, 181–211.
- Vezhnevets et al. (2017). FeUdal networks for hierarchical RL. ICML. arXiv:1703.01161 (ar5iv).
- Gutteridge, Dong, Bronstein & Di Giovanni (2023). DRew: dynamically rewired message passing with
  delay. ICML. arXiv:2305.08018. (abstract-level only)
- Qie, Martin et al. (2026). Dissecting emerging slow rhythms in delay-coupled neural oscillators.
  arXiv:2606.20733.
- Bodnar, Di Giovanni, Chamberlain, Liò & Bronstein (2022). Neural sheaf diffusion. NeurIPS.
  arXiv:2202.04579 (ar5iv).
- Tallec & Ollivier (2018). Can recurrent neural networks warp time? ICLR. arXiv:1804.11188.
  (abstract at source; equations second-hand — see Honest gaps)
- Koutnik, Greff, Gomez & Schmidhuber (2014). A clockwork RNN. ICML. arXiv:1402.3511 (ar5iv).
- Chung, Ahn & Bengio (2017). Hierarchical multiscale RNNs. ICLR. arXiv:1609.01704 (ar5iv).
- Yamashita & Tani (2008). Emergence of functional hierarchy in a multiple timescale neural network
  model. *PLoS Computational Biology* 4(11): e1000220.
