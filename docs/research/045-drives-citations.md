# Citation pass: drives and the dark room problem (patchworks#45)

Validates the design closed in patchworks#9 and revised in patchworks#36
(`docs/adr/0009-a-drive-is-a-motor-edge-attached-deep.md`, `04-action-and-the-boundary.md` *Drives* and
*The internal rim*, `02-tick-semantics.md` *External writes*). Citations validate after the fact per the
map's Notes; this document does not revise the closed design — it flags where a source threatens a claim
already made, and recommends revision tickets without opening them. Vocabulary follows `CONTEXT.md`;
the prior art is described in its own field's terms. Fourteen sources fetched, four recorded unreachable
(*Gaps*). #26 covered the boundary literature and #19 the tick literature; neither is redone here.

## Headline verdict, stated plainly

**The dark room is the safest claim in this area, and the scalar is the most exposed.** The priors-only
resolution ADR-0009 leans on is quoted accurately from its primary source, is still the field's stated
answer, and — contrary to the ADR's own caution — Patchworks needs *less* of it than anyone else: the
paper's burden is explaining where world-expecting priors come from without appealing to epistemic value,
and Patchworks simply writes the prior from outside. There is a working, published, epistemic-term-free
system on that horn (Baltieri & Buckley's PID controller). Two further claims came back better supported
than the design expected: the clamp/nudge distinction ADR-0009 reasons its way to is an already-named
result in energy-based learning (Scellier & Bengio's *weakly clamped* phase, with the same argument for
it), and the external-write ordering rule is the standard discipline in propagation-on-graphs.

The one claim the sources push back on is **valence in one dimension**. Nothing found runs a single
scalar as the sole directional drive of a large graph: the biology runs four distinct diffuse scalars
with four distinct jobs (Doya), hierarchical RL's directional goal is a 16-vector (FeUdal), and scalar
Markov reward provably cannot express some task specifications (Abel et al.). The spec's defence — the
drive is not asked to *specify* anything, because the render does — blunts the proof but produces no
positive precedent. The exposure is correctly named and should stay named.

## (a) The dark room, resolved by priors rather than epistemic value

**Sources.** Friston, Thornton & Clark (2012), "Free-Energy Minimization and the Dark-Room Problem,"
*Front. Psychol.* 3:130, DOI 10.3389/fpsyg.2012.00130. Baltieri & Buckley (2019), "PID Control as a
Process of Active Inference with Linear Generative Models," *Entropy* 21(3):257, DOI 10.3390/e21030257.
Sprevak & Smith (2023), *TopiCS*, DOI 10.1111/tops.12704. Perfors (2024), *TopiCS*, DOI
10.1111/tops.12759.

**The quotation checks out, and the mechanism is definitional.** Friston, Thornton & Clark state the
resolution as a difference in what is minimised: "average surprise or entropy H(s | m) is a function of
sensations and the agent (model) predicting them. Conversely, the entropy H(s) minimized in dark rooms is
only a function of sensory information." The sentence ADR-0009 quotes is in the paper: "a dark room will
afford low levels of surprise if, and only if, the agent has been optimized by evolution (or
neurodevelopment) to predict and inhabit it" — the ADR elides the parenthesis, marked, without changing
the claim. It is grounded in priors: "different surprises rest on different assumptions about the world
that can be cast as prior beliefs and are therefore part of the model," stretched as far as morphology —
"the gross bodily form, biomechanics, and gross initial neural architecture of the agent all form part of
the (initial) 'model.'" **No exploration bonus, curiosity term, or information-gain quantity appears
anywhere in the argument**, which is what ADR-0009 asserts and what this pass confirms.

**Someone has built on it without the epistemic summand.** Baltieri & Buckley derive PID control as
active inference. The setpoint is a prior, full stop: "In essence, an active inference agent implements
set-point regulation by behaving to make its sensations accord with its strong priors/desires,"
implemented as "a Gaussian prior in generalised coordinates encoding desired velocity and acceleration
with means ηx=10 km/h, ηx′=0 km/h²." **No epistemic value, information-gain, or expected-free-energy term
appears in the formulation.** This is the existence proof the ticket asks for. Its scope is narrow — one
regulated variable, a linear model, no hierarchy — so it proves the horn is walkable, not that it scales
to a ~150-cell graph.

**What the sources say against it.** Sprevak & Smith confirm ADR-0009's reading that the epistemic route
is now standard: expected free energy "requires an additional term … [that] drives the agent to seek out
observations expected to improve its model of the world." Perfors registers the live dissatisfaction: "I
am dissatisfied by most responses, which seem to me to be variations on the point that precision error is
always defined relative to a model." Her objection is that predictive processing has no *internal*
account of why the model is a good one — which does not reach an architecture whose answer is stipulated.

### The Sun & Firestone exchange, read in full (gap closed)

This pass originally recorded the *TiCS* exchange as unreachable and named the Seth reply as its soft
spot. The full exchange has since been obtained and read: Van de Cruys, Friston & Clark (2020),
"Controlled Optimism: Reply to Sun and Firestone on the Dark Room Problem," *TiCS* 24(9):680–681, DOI
10.1016/j.tics.2020.05.012; Seth, Millidge, Buckley & Tschantz (2020), "Curious Inferences: Reply to Sun
and Firestone on the Dark Room Problem," *TiCS* 24(9):681–682, DOI 10.1016/j.tics.2020.05.011; and Sun &
Firestone's rejoinder, "Optimism and Pessimism in the Predictive Brain," *TiCS* 24(9):683–684.

**The soft spot resolves, and it resolves in ADR-0009's favour — but it sharpens the price.** Seth et al.
are the epistemic camp's clearest statement, and they place the two routes in exactly the relation
ADR-0009 assumes. On the priors route they write: "One standard response is that Dark Room type
environments are intrinsically surprising, given the homeostatic imperatives of living organisms. One
might worry that this response solves nothing, since it merely redefines what counts as 'surprising' for
an agent. The reply by Van de Cruys and colleagues **relieves us of this worry** by highlighting the
principled role of 'optimistic predictions' in driving actions." So the priors-only horn is not a
fringe position and is not held to be circular — conceded by the authors who decline to stop there.

**What they add, and what Patchworks therefore gives up.** Their own route is expected free energy, and
they are explicit that the epistemic term is what produces exploration: "Minimising expected free energy
entails minimising a (negative) expected information gain term, which rewards sampling those novel
environmental states that are (predicted to) induce a large divergence between prior and posterior
beliefs. This is why long-term free-energy-minimising agents are intrinsically drawn towards novel
experiences (and thus out of Dark Rooms)." Box 1 insists this is not a patch: the epistemic terms "arise
naturally out of the mathematical formalism, instead of being bolted on," and "only arise when performing
inference over temporally extended sequences." **This is precisely the summand ADR-0003 makes
unavailable.** The honest consequence for the spec: Patchworks forfeits a *derived* account of curiosity.
Novelty-seeking cannot fall out of the formalism here — if the agent explores, that is because a drive
was written from outside saying so. ADR-0009 should say this rather than leave it implicit.

**Sun & Firestone's rejoinder is the sharpest attack on the priors route, and it misses Patchworks
specifically.** Their charge is smuggling: "what counts as 'optimistic' *depends on one's desires* … the
only way optimism gets agents out of Dark Rooms and living their lives is by smuggling in desires after
all. (Nor will it do to define optimism relative to evolutionary considerations.)" The force of this is
that predictive processing "aims to replace [beliefs and desires] with a single state: prediction," so
reintroducing desires concedes the central claim. **Patchworks is not making that claim.** It never
asserted that prediction is the only state; it writes the drive in from outside, openly, as a motor edge.
The objection is that the desire arrives undeclared — and here it is declared. This is the strongest
available support for the pass's headline: on this claim Patchworks is *less* exposed than the theories
it borrows from, because it is not defending predictive processing's radical monism.

Their **"Homeostatic Room"** — an IV drip, the right electrolytes, a thermostat, "this arrangement should
be paradise for a surprise-minimizer. Yet, it seems unlikely that you'd stay" — is worth recording as the
sharp form of the objection to *homeostatic* readings specifically. It reinforces the `CONTEXT.md`
_Avoid_ recommendation below: a drive read as a deficit that discharges on satiation walks straight into
this room and stays. Patchworks' drive is a standing assertion, so it does not.

**Verdict: supported.** No revision, beyond a rhetorical note: ADR-0009 calls the compatibility "a
happier coincidence than it had any right to be", when the external write makes it the natural reading.
Add the forfeited-curiosity note above, which is a genuine cost rather than a rhetorical one.

## (b) A goal as a standing assertion, not a clamp and not a reward

**Sources.** Scellier & Bengio (2017), "Equilibrium Propagation," *Front. Comput. Neurosci.* 11:24,
arXiv:1602.05179. Vezhnevets et al. (2017), "FeUdal Networks for Hierarchical Reinforcement Learning,"
arXiv:1703.01161. Nave, Deane, Miller & Clark (2020), *WIREs Cog. Sci.* 11(6):e1542. Baltieri & Buckley
(2019), above.

**The clamp/nudge distinction is an existing, named result.** ADR-0009 rejects the clamp by argument: a
genuine override removes the clamped cell from inference and leaves a hole in the network dynamics.
Equilibrium Propagation names the same distinction and gives the same reason. Verbatim: "**Contrary to
Boltzmann Machines where the visible units are either free or (fully) clamped, here the real-valued
parameter β allows the output units to be weakly clamped.**" And on why weak clamping is productive: the
force "nudges the output units from their free fixed point value in the direction of their target …
the perturbation caused at the output units will propagate in the hidden units as time progresses." A
weakly clamped unit stays in the dynamics and its disagreement propagates; a fully clamped one does not.
That is ADR-0009's argument, arrived at independently, with a decade's head start.

*Divergence.* EqProp's nudge is a *learning* device that alternates with a free phase at β = 0;
Patchworks' drive is not a phase and never switches off. Mechanism corresponds, purpose does not.

**Direction without a target value in the recipient's basis.** FeUdal Networks is the closest match on
the other half. Its Manager emits a goal "directional rather than absolute in nature," for Patchworks'
reason: "We use directions because it is more feasible for the Worker to be able to reliably cause
directional shifts in the latent state than it is to assume that the Worker can take us to (potentially)
arbitrary new absolute locations." The meaning is not imposed — "now g_t acquires a semantic meaning as
an advantageous direction in the latent state space" — matching the spec's "comes instead from the
graph's own learned model of what satisfaction looks like." Their ablation is evidence, not assertion:
absolute goals in place of directional ones leave performance "significantly inferior."

*Divergence.* FeUdal computes an intrinsic reward from cosine similarity between goal direction and
realised state change — a machine-side read of how well the goal is being met, i.e. the satisfaction
detector Patchworks refuses — and its Manager is trained against environment reward. **No source found
combines a persistent external assertion, no target value in the recipient's own basis, and no
satisfaction detector at all;** that combination appears to be the project's own.

**How everyone else makes the assertion resist being explained away.** Nave et al. describe the
homeostatic imperative as "a kind of 'first prior'", and the mechanism stopping the agent from revising
its belief instead of acting is precision: interoceptive signals "are deemed highly precise — they are
given 'a priori hyperprecision of visceral channels' … the system that simply updates its perception of
its body temperature to 50 centigrade, rather than acting to cool down" is not viable. Baltieri &
Buckley's setpoint is likewise a *strong* prior. **This is a real, locatable difference.** The
literature's standing assertion resists erosion by high precision; Patchworks' resists by being written
last (*External writes*) — an ordering rather than a magnitude. The ordering is arguably cleaner, since
precision would need its own representation and time constant, which ADR-0007 refuses independently. But
the field's standard defence against hallucinated satisfaction — precision-weighting the sensory channels
above the drive — is unavailable here, so ADR-0009's *Hallucinating satisfaction* exposure is less
protected than the literature's version of it.

**Verdict: supported, with better precedent than the ADR claims for the clamp rejection, and one gap —
the no-detector reading has no precedent found.**

## (c) Valence without specification: one dimension steering a graph

**Sources.** Doya (2002), *Neural Networks* 15(4–6):495–506. Abel et al. (2021), "On the Expressivity of
Markov Reward," arXiv:2111.00876. Vamplew et al. (2021), arXiv:2112.15422. Vezhnevets et al. (2017).

**The biology is scalar per channel, and there are four channels.** Doya's is the canonical statement
that scalar modulatory signals do directional work: neuromodulators "mediate the global signals that
regulate the distributed learning mechanisms in the brain," projecting "diffusely to the cortex, the
basal ganglia, and the cerebellum from brain stem nuclei." Each carries one number with one job —
"dopamine signals the error in reward prediction, serotonin controls the time scale of reward prediction,
noradrenaline controls the randomness in action selection, and acetylcholine controls the speed of memory
update." **This supports low bandwidth and refutes one dimension** — and supports ADR-0009's *multiple
simultaneous drives compose*: the brain's answer to needing more directional influence is another diffuse
scalar channel, i.e. an additional drive boundary cell, not a wider one.

**The formal result, and why it lands softer than it looks.** Abel et al. prove "there exist instances of
each task type that no Markov reward function can capture," for tasks given as "(1) a set of acceptable
behaviors, (2) a partial ordering over behaviors, or (3) a partial ordering over trajectories." Vamplew
et al. argue the same informally: "scalar rewards are insufficient to account for some aspects of both
biological and computational intelligence." **The spec has a genuine defence and it should be written
down.** Abel's theorem concerns a scalar used as the *task specification*; the spec puts specification in
the render — "It never names a puck or a zone, because the render already does" — so the drive is not the
object quantified over. What survives is weaker and still uncomfortable: no source was found in which one
scalar differentiates behaviour across a large graph, and the nearest comparable object, FeUdal's
directional goal, is a vector — "the dimensionality of the embedding vectors, w, … set as k = 16."

**Verdict: the known exposure is confirmed real and correctly stated.** ADR-0009's escape hatch (a small
learned drive vector) is where the prior art already sits, and k ≈ 16 is a literature-attested width for
this job. **Recommended revision ticket** (not opened here): record the hatch's width and trigger
condition in the exposure, so widening is pre-costed.

## (d) Attachment at the deepest, slowest site

**Sources.** Yin & Chu (2014), "How Basal Ganglia Outputs Generate Behavior," *Adv. Neurosci.*
2014:768313. Vezhnevets et al. (2017); Doya (2002); Nave et al. (2020), above.

Yin & Chu argue depth directly, in hierarchical-control terms: "There are no first-order sensors for
'reward' or 'reward rate' as there are for muscle tension. Rather these are highly abstract variables
constructed from multiple perceptual signals from lower levels … Only at the highest levels can different
transitions be related to each other and only there can such feedback functions be learned, so that the
appropriate actions can be acquired to reach desired goals." On how the reference travels down: "the
error of the outcome control system can reliably set a reference for a lower system that specifies some
action to be performed. The higher level, therefore, recruits a lower one to reduce its error." That is
#36's apex attachment and the spec's *planning is drive propagation*, in a neuroscience register. FeUdal
supplies the timescale half: "The Manager operates at a lower temporal resolution and sets abstract goals
which are conveyed to and enacted by the Worker" — deepest and slowest together, matching ADR-0005's
identity of timescale with persistence.

**Where the sources dissent: depth is an axis, but not the biology's axis for a valence signal.** Doya's
neuromodulators are broadcast diffusely to cortex, basal ganglia and cerebellum at once; Nave et al.'s
interoceptive first prior enters through its own hierarchy, not the top of the exteroceptive one. **No
source was found placing a scalar motivational signal at a single deep site.** #36's rule — one edge to
*each* of the eight apex cells, the level entire — is closer to a bounded broadcast than to a point
attachment, which is the reading these sources support.

**Verdict: supported for depth-as-abstraction; the entire-level fan-out is the part the sources endorse,
and it is already what #36 decided.**

## (e) External writes as the tick's last word

**Sources.** Zhou, Bousquet, Lal, Weston & Schölkopf (2004), "Learning with Local and Global
Consistency," *NIPS* 16:321–328. Scellier & Bengio (2017), above. Zhu & Ghahramani (2002),
CMU-CALD-02-107 — **not reached** (*Gaps*).

There is a standard treatment and it agrees. Propagation-on-graphs has two regimes for a node whose value
is owned from outside: hard clamping — labelled nodes reset to their given values after each propagation
step, and Boltzmann visible units "either free or (fully) clamped" — and soft re-injection, Zhou et al.'s
"F(t+1) = αSF(t) + (1−α)Y", of which they write: "During each iteration of the third step each point
receives the information from its neighbors (first term), and also retains its initial information
(second term)." **Patchworks' rule is the first regime, implemented as an ordering.** The spec's
justification — "Without the ordering, eight apex cells disagreeing with the drive would erode it every
tick" — is precisely the failure both regimes exist to prevent: without re-injection every step the
externally-owned value is diluted by neighbour information until it is gone.

*Divergence.* Both literatures clamp or re-inject a *whole node*; Patchworks states the rule as an
ordering, which is what leaves the actuator's commanded components reconciled while its efference
components are overwritten. No source found does per-component external ownership this way; none
contradicts it either.

**Verdict: supported, standard, low risk.** No revision.

## (f) The internal-rim bans

**Sources.** Ha & Schmidhuber (2018), "World Models," arXiv:1803.10122 (worldmodels.github.io). Dawid &
LeCun (2024), "Introduction to latent variable energy-based models," *J. Stat. Mech.* 104011,
arXiv:2306.02572.

Ha & Schmidhuber is one real precedent, satisfying both bans at once. On the auxiliary module holding no
model of the world: "we deliberately make C as simple and small as possible, and trained separately from
V and M, so that most of our agent's complexity resides in the world model (V and M)." On reaching the
world only through the graph: "C is a simple single layer linear model that maps z_t and h_t directly to
action a_t at each time step" — its only inputs are the world model's outputs; it never sees the render.
**Their justification differs: theirs is trainability, Patchworks' is that a faculty with its own model
imposes something nothing corrects.** Same constraint, different reason — convergent, not borrowed.

LeCun's architecture is the informative contrast. Its intrinsic cost is "hard-wired into the architecture
(i.e., not trainable) and models basic needs like pain, pleasure, hunger" — a faculty holding a constant
and no model, obeying ban one exactly as the drive does. But it then adds "the critic, a trainable module
that predicts future values of the intrinsic cost", an auxiliary module carrying a learned predictive
model of its own alongside the world model; under ban one that is not admissible at the internal rim.
(It also plans by imagined rollout, "akin to model-predictive control", which ADR-0003 refuses anyway.)

**Verdict: the bans are substantially the project's own rule.** One architecture instantiates both
without stating them; one prominent architecture violates the first. **No source was found stating a
general constraint on auxiliary modules attached to a world model** — which makes *The internal rim* a
contribution rather than a restatement, and one nothing external will catch if it is wrong.

## "Drive" as a term of art: a `CONTEXT.md` _Avoid_ recommendation

Two collisions, one of which matters.

1. **Drive-reduction theory (Hull, 1943; refuted through the 1950s).** In psychology a *drive* is an
   aversive tension state produced by a physiological deficit, which behaviour reduces, ending at
   satiation. Close enough to mislead — a standing discomfort cleared by acting — and the differences are
   the design's load-bearing choices: Patchworks' drive is not a state variable of the agent, is not
   reduced by a consummatory act, has no satiation and no detector, and does not decay. A reader
   importing the Hullian sense would hunt for a satiation mechanism and conclude one is missing.
2. **"Drive" in electronics and control** — the signal or stage that drives an actuator; a motor drive is
   the amplifier at the motor. Since a drive edge *is* a motor edge, this reading is actively available
   and points at the wrong end of the graph: a drive edge is the abstract one, far from the rim.

**Recommendation, concrete:** keep the word — it is earned, and no candidate (appetite, imperative, want,
motive) is better. Amend the **Drive** entry in `CONTEXT.md`: extend its _Avoid_ line to `reward, goal
(reserve for the human-set drive), clamp, objective, utility, drive state, drive reduction, tension`, and
add one clause to the definition — that this is not the Hullian drive: no deficit state, no reduction, no
satiation, and the disagreement falls away because the world agrees, not because anything is discharged.
Optionally add the electronics disambiguation to **Drive edge**. No spec claim changes.

## Recommended revision tickets (recommendations only; not opened)

1. **The scalar drive's escape hatch, pre-costed** (`04-action-and-the-boundary.md` *Known exposure*,
   ADR-0009 *Known exposure*). The exposure is confirmed real by (c). Record the hatch's expected width —
   FeUdal's k = 16 is the attested figure for a directional goal in a learned latent space — and the
   observable that would trigger taking it. Small.
2. **`CONTEXT.md` _Avoid_: the Hullian drive**, per the section above. Trivial.
3. *(Optional, framing only)* **ADR-0009's dark-room paragraph understates its own position** — the
   external write frees Patchworks of the phylogenetic burden the priors reading usually carries.
4. **Record the forfeited account of curiosity** (ADR-0009, and ADR-0003 where it declines expected free
   energy). Added after the *TiCS* exchange was read in full. Seth et al. are explicit that the epistemic
   term is what makes long-horizon agents intrinsically novelty-seeking, and that it falls out of the
   formalism rather than being bolted on. Declining it is the right call under ADR-0003, but it has a
   price worth naming: **novelty-seeking cannot be derived here.** If the agent explores, it is because a
   drive was written from outside saying so. This is a real cost, not a rhetorical one, and it belongs
   next to the scalar exposure rather than buried in a citation doc.

## Gaps

- **Friston (2012), *Entropy* 14(11):2100–2121** — MDPI returned 403 on both HTML and PDF. ADR-0009's
  "Friston et al. (2012)" is validated here against the *Frontiers* paper, which carries the quoted line.
- ~~**Sun & Firestone (2020), *TiCS* 24(5):346–348**, and the two published replies~~ — **CLOSED.**
  Obtained via institutional access and read in full; see *The Sun & Firestone exchange, read in full*
  under (a). The named soft spot is resolved, and in ADR-0009's favour. The original *TiCS* 24(5):346–348
  target article is still known only through the three replies that quote it, which is sufficient for
  this pass's question — every claim (a) rests on is now sourced to a primary text rather than to
  Perfors (2024) at one remove.
- **Zhu & Ghahramani (2002), CMU-CALD-02-107** — "clamp the labelled data" attested only second-hand via
  Zhou et al. Two attempts (unextractable Type-3 PDF; connection refused).
- **LeCun (2022), "A Path Towards Autonomous Machine Intelligence"** — OpenReview verification wall, two
  attempts; read through Dawid & LeCun (2024), co-authored by LeCun.
