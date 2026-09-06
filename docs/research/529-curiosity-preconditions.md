# Citation pass: the curiosity drive's preconditions (patchworks#529)

A pre-mortem for the supply ladder's rung 2, run *before* the build, per
[#529](https://github.com/NGL321/patchworks/issues/529) on map
[#517](https://github.com/NGL321/patchworks/issues/517). It asks what the systems where curiosity,
intrinsic motivation and goal babbling **demonstrably work** already had, and which of those this
system lacks. Citations validate after the fact per the map's Notes; **this document proposes no
build** and revises no closed design. Vocabulary follows `CONTEXT.md`; the prior art is described in
its own field's terms. Six primary sources were read in full text (PDF extracted and quoted
verbatim); two were read at abstract/landing-page level only and are marked in *Gaps*.

## Headline verdict, stated plainly

**Three findings, in descending order of consequence.**

**1. The ticket's area-3 hypothesis is false as stated, and what replaces it is worse for this
system.** The ticket asks whether the working cases presupposed a controller that *already produced
varied output*. They did not. The two clearest successes both start from a **degenerate controller of
exactly this system's kind**: Rolf, Steil & Gienger's Goal Babbling starts from an inverse estimate
that is *a constant* — "g(x, θ₀) = const = q^home" — and Der & Martius's DEP starts from "C = 0 and
h = 0, meaning that all actuators of the joints are in their central position." A constant untrained
command (#120) is therefore **not** what disqualifies this system. What both cases require instead is
one rung lower and is not a property of the controller at all: **a body that answers**. Rolf's
learning weight is literally `w_t^eff = ‖x_t − x_{t−1}‖ · ‖q_t − q_{t−1}‖⁻¹`; at zero effector
displacement every example carries weight zero and nothing is learned. DEP: "consider the trivial
case ẏ = 0 where the body is at rest and will stay there as long as there are no extrinsic
perturbations." This system's arm reads **0.000e+00 ± 0.000e+00** travel per tick under T1's winner
(#529). That is the numerator of Rolf's weight, and it is zero. **The precondition this system fails
is not "a varied controller" but "a non-zero, perceivable actuator→outcome response," and it fails it
by measurement, not by argument.** Rung 2 is mis-ordered — but for a different reason than the ticket
guessed, and the corrected reason is harder to fix, because it sits in the world-side famine (P2) that
T2 already failed to relieve.

**2. Every intrinsic-motivation formulation that has scaled is a reward in exactly the sense #5 bars
— except one, and that one's stated precondition is the one this system measurably fails.**
Prediction-error curiosity, learning progress, competence progress, pseudo-counts and RND are all
scalar objectives a policy maximises, all needing credit assignment and a satisfaction detector, all
entering as a term modifying the update (#5's three properties). Empowerment is a utility, and fails a
*different* project bar — ADR-0003's refusal of counterfactual evaluation — before it reaches #5.
**Der & Martius's DEP is the sole reward-free member**: "without having to postulate any higher level
constructs such as intrinsic motivation, curiosity, goal orientation, or a specific reward system." It
is admissible under #5 on inspection. And its escape condition is an extrinsic ẏ ≠ 0, which this
system does not have.

**3. There is a published negative result whose shape matches this system exactly.** In the same DEP
paper, the plain differential Hebbian rule "is not able to depart from the C = 0 condition" and "often
falls into the C = 0 state, which it cannot exit anymore." A rule with a fixed point at zero activity,
started at zero activity, driven by a constant, stays there. That is the documented failure mode, and
this system is sitting in it.

---

## (a) Goal babbling versus motor babbling: what the goal-babbling controllers already had

**Source.** Rolf, Steil & Gienger (2011), "Online Goal Babbling for rapid bootstrapping of inverse
models in high dimensions," *IEEE ICDL-EpiRob*; author preprint, Honda Research Institute Europe,
<https://www.honda-ri.de/pubs/pdf/1732.pdf>. Full text extracted and quoted verbatim below.
Companion: Rolf, Steil & Gienger (2010), "Goal Babbling Permits Direct Learning of Inverse
Kinematics," *IEEE TAMD* 2(3):216–229, DOI 10.1109/TAMD.2010.2062511 (abstract only — *Gaps*).

**Why motor babbling is abandoned there, and it is not for the reason this project's T2 found.** The
paper's stated objection to motor babbling is dimensional, not developmental: "In artificial systems,
exploration is traditionally addressed by 'motor babbling' … This kind of exploration becomes very
inefficient with increasing dimensionality of the sensorimotor space." T2's finding — that babble's
reach *decays* under training, 10.1 → 1.82 by 10k, while the sensory conditions' grows — is a
different failure and has **no counterpart in this literature**. Nothing found reports babble reach
decaying as the learner trains. That asymmetry is itself the signal: in the working systems the
babble amplitude is applied *at the actuator* and its consequence is read *at the sensor*, so there is
no interior taper for it to die in; in this system the deviation must survive two to three decades per
interior hop (#120) and does not arrive at depth (#224).

**Goal babbling's starting controller is a constant — the same degeneracy this system has.**

> "In order to generate examples, Goal Babbling starts with an initial inverse estimate g(x, θ₀) that
> always suggests some comfortable home posture: g(x, θ₀) = const = q^home."

This is the single most important sentence in the pass. The ticket's area-3 conjecture — that
curiosity works only where it *selects among* behaviours a controller already produces — is refuted
here on its own terms. Goal babbling creates variation from a controller that is, at t = 0, exactly
"one world-independent constant."

**What it has instead, and this system does not, is three things.**

*(i) An exploratory perturbation term, declared necessary.*

> "The perturbation term E_t(x) adds exploratory noise in order to discover new positions…"

> "In order to find kinematic solutions for all target positions, it is necessary to consider
> exploratory noise, or rather perturbations of the motor system. Such perturbations arise naturally
> in physical systems and lead to the exploration of new postures that would not be suggested by the
> inverse estimate."

Note "arise naturally in physical systems": in the working case the perturbation is a property of the
body, free of charge. Here it must be manufactured and pushed through the taper, which is what T2
tested and what failed.

*(ii) A learning signal that is identically zero when the effector does not move.* The weighting
scheme is stated as (their Eqns. 4–6):

> `w_t^dir = ½ (1 + cos ∠(x_t − x_{t−1}, x*_t − x*_{t−1}))`
> `w_t^eff = ‖x_t − x_{t−1}‖ · ‖q_t − q_{t−1}‖⁻¹`
> `w_t = w_t^dir · w_t^eff`

> "w_t^dir measures whether the actually observed movement and the intended movement have the same
> direction. w_t^eff measures the kinematic efficiency of the movement and assigns high weight to
> examples that achieve a maximum of effector movement with a minimum of joint movement. For
> learning, each example (x_t, q_t) is weighted by w_t."

**With ‖x_t − x_{t−1}‖ = 0, w_t^eff = 0, so w_t = 0, so every example is discarded.** This system's
arm travel per tick reads 0.000e+00 ± 0.000e+00 under T1's winner on the full dome at 30k (#529
body), below even the frozen baseline's parked 5.042e-05. On the working method's own equation, this
system supplies zero training weight. Further, `w_t^dir` is a *direction* comparison — the same
quantity #517 identified when it found that "constancy was operationalised as magnitude and it is
direction that matters," and that T0 read as an evidence-stream participation ratio of 1.00
everywhere.

*(iii) A task space lower-dimensional than the motor space, and redundant.*

> "Tasks in sensorimotor learning are typically substantially lower-dimensional than the motor systems
> themselves… If there are multiple ways to achieve the same result there is no inherent need to know
> all of them."

Redundancy is what makes goal-directed sampling cheap. This system's composed rim-to-apex transport is
rank-one (#497), read at exactly 1.000 on the full dome under both T3 arms and falling 1.298 → ~1.08
under every T2 condition including the zero-supply control. There is no redundant, multi-directional
outcome space for goals to be drawn from.

**The high-level goal-babbling architectures presuppose a working low-level reaching controller.**

**Source.** Baranes & Oudeyer (2013), "Active Learning of Inverse Models with Intrinsically Motivated
Goal Exploration in Robots," *Robotics and Autonomous Systems* 61(1):49–73, arXiv:1301.4862.

SAGG-RIAC is explicitly two-level, and the lower level is assumed, not derived:

> "Once a goal has been actively chosen at the high-level, the goal directed exploration and learning
> mechanism at the lower level can be carried out in numerous ways… Its main idea is to guide the
> system toward the goal by executing low-level actions which allow progressive exploration."

> "The reaching phase deals with creating a pathway to the current goal position by determining an
> optimal micro-action which would guide the end-effector toward the goal."

> "Once a goal/task is chosen, the system would then try to reach it with a lower-level goal-reaching
> architecture typically based on coupled inverse and forward models."

So the answer to the ticket's "what did their controllers already have" is: **for SAGG-RIAC, a
goal-reaching controller that turns a target into micro-actions with observable consequences; for
Rolf's online goal babbling, not a controller at all but a perturbation term plus a forward map with a
non-zero Jacobian.** This system has neither. It has one constant and a zero Jacobian reading.

---

## (b) The intrinsic-motivation formulations, each against #5's bar

#5's resolution states the test used here. It rules #495's probe admissible "precisely because it is
not a reward: a task-blind schedule has none of the three properties that make one — **it needs no
credit assignment, needs no satisfaction detector, and enters the learning rule as evidence rather
than as a term modifying the update**." Read forward, a formulation *is* a reward when it needs credit
assignment, needs a satisfaction detector, and enters as a term modifying the update. #5's bar for
adopting one is **necessity, not appetite** — "only where a
reward fixes a problem that cannot be fixed without it — or where fixing it without reward is
inconvenient enough to make the trade justifiable."

### Prediction-error curiosity (ICM) — **a reward. Barred.**

**Source.** Pathak, Agrawal, Efros & Darrell (2017), "Curiosity-driven Exploration by Self-supervised
Prediction," *ICML*, arXiv:1705.05363.

> "Our agent is composed of two subsystems: a reward generator that outputs a curiosity-driven
> intrinsic reward signal and a policy that outputs a sequence of actions to maximize that reward
> signal… The policy sub-system is trained to maximize the sum of these two rewards r_t = r_t^i +
> r_t^e."

All three properties present: A3C does the credit assignment, the forward-model error *is* the
satisfaction detector, and it enters as a term in the return. **It is a reward in precisely #5's
sense.**

A second, independent obstruction is worth recording. ICM's feature space is learned by an inverse
dynamics task, and the paper states its consequence:

> "there is no incentive for φ(s_t) to encode any environmental features that can not influence or are
> not influenced by the [agent's actions]"

Under a zero actuator→outcome response the inverse-dynamics task is unlearnable and φ is degenerate by
construction. ICM would not merely be barred here; it would read nothing.

### Learning progress / competence progress (IAC, SAGG-RIAC) — **a reward. Barred.**

**Source.** Oudeyer, Kaplan & Hafner (2007), "Intrinsic Motivation Systems for Autonomous Mental
Development," *IEEE Trans. Evol. Comput.* 11(2):265–286, DOI 10.1109/TEVC.2006.890271.

> "a way to implement an intrinsic motivation system might be to build a mechanism which can evaluate
> operationally the degree of 'novelty,' 'surprise,' 'complexity,' or 'challenge' … and then designing
> an associated reward … Autonomous and active exploratory behavior can then be achieved by acting so
> as to reach situations which maximize this internal reward."

> "For each situation that the robot encounters, it is given an internal reward which is equal to the
> inverse of this difference… The motivation system of the robot is then a system in which the action
> chosen is that for which KGA predicts that it will lead to the greatest decrease of the mean error
> rate of M."

Credit assignment (action selection over predicted futures), satisfaction detector (the error-rate
derivative), term modifying the update. Baranes & Oudeyer's competence progress is the same quantity
on a task space — "The interest of a region is described as the absolute value of the derivative of
local competences inside that region." **Reward. Barred.** It additionally requires a competence
measure, which requires a reachable, populated outcome space this system does not have.

### Count- and novelty-based bonuses (pseudo-counts, RND) — **rewards. Barred.**

**Sources.** Bellemare et al. (2016), "Unifying Count-Based Exploration and Intrinsic Motivation,"
*NeurIPS*, arXiv:1606.01868. Burda, Edwards, Storkey & Klimov (2018), "Exploration by Random Network
Distillation," arXiv:1810.12894.

Bellemare et al. are explicit that the bonus is a term added to the Bellman update — "an exploration
bonus proportional to N(x,a)^{-1/2}" solving "the augmented Bellman equation" — and that the quantity
is the standard one: information gain "is commonly used to quantify novelty or curiosity and
consequently as an intrinsic reward." RND: "We introduce an exploration bonus for deep reinforcement
learning methods… We also introduce a method to flexibly combine intrinsic and extrinsic rewards."
**Rewards, on all three properties. Barred.**

### Empowerment — **a utility, not a reward in #5's sense; barred earlier, by ADR-0003.**

**Sources.** Klyubin, Polani & Nehaniv (2005), "Empowerment: a universal agent-centric measure of
control," *IEEE CEC* (primary unreachable — see *Gaps*). Salge, Glackin & Polani (2014),
"Empowerment — An Introduction," in *Guided Self-Organization: Inception*, arXiv:1310.1863. Full text
extracted and quoted verbatim.

> "the proposed formulation of empowerment defines it via the concept of potential information flow,
> or channel capacity, between an agent's actuator state at earlier times and their sensor state at a
> later time."

> "Empowerment does not consider the learning process or the agent trajectory through the world, but
> instead operates as a pseudo-utility which assigns a value (its empowerment) to each state in the
> world. Highly empowered states are preferred…"

On #5's three properties, empowerment is genuinely unlike ICM/RND: it needs no credit assignment and
no satisfaction detector, and it does not enter a learning rule at all — it scores states. **On #5's
test alone it would not obviously be barred.** It is barred by a different, older ruling:

> "the maximization implies that it is calculated under the assumption that the controller which
> chooses the action A is free to act… Empowerment considers only the potential information flow, so
> the agent will only calculate how it could affect the world, rather than actually carry out its
> potential."

Empowerment is a channel capacity — a maximisation over hypothetical action distributions, computed
with Blahut–Arimoto over p(a). That is counterfactual evaluation of candidate actions before acting,
and ADR-0003 states that "the architecture has **no counterfactual evaluation at all**" — amended
since, but amended into a *position* rather than a cost: "The architecture has no counterfactual
evaluation because it has **no plan to evaluate**." ADR-0009
makes the same refusal for the epistemic-value summand of expected free energy, for the same reason.
**Empowerment is architecturally unavailable here regardless of #5.**

And even were it available, it reads zero:

> "The lowest value for empowerment is 0, which means that an agent has no influence on the world that
> it can perceive. From the empowerment perspective, vanishing empowerment is equivalent to the
> agent's death…"

> "Empowerment is the channel capacity from an agent's actuators to its sensors, and as such, measures
> the efficiency of that channel. Having actuators whose effect on the environment cannot be perceived,
> or sensors which detect no change relevant to the current actions is inefficient…"

Arm travel 0.000e+00 is the zero-empowerment condition, stated in the source's own words.

One further precondition is recorded there and is worth naming because it cuts the other way:

> "for meaningful interaction to emerge from a method such as an empowerment landscape, limitations in
> sensing and acting need to be present."

### Novelty search — **an objective function replacement; requires a population. Not applicable.**

**Source.** Lehman & Stanley (2011), "Abandoning Objectives: Evolution through the Search for Novelty
Alone," *Evolutionary Computation* 19(2):189–223.

This is the one case that **does** match the ticket's area-3 conjecture, and it says so:

> "Evolutionary algorithms like NEAT are well-suited to novelty search because the population of
> genomes that is central to such algorithms naturally covers a wide range of expanding behaviors. In
> fact, tracking novelty requires little change to any evolutionary algorithm aside from replacing the
> fitness function with a novelty metric."

Here novelty genuinely only **selects**; the variation is created by mutation and carried by a
population. Note also that novelty search's celebrated biped result starts from controllers that "fall
down" in different ways — i.e. the substrate already emits distinguishable behaviours. This system has
no population, and its single behaviour is "parked." **Not applicable**, and its applicability
condition is precisely the one the ticket suspected — it is simply not the condition that governs the
goal-babbling and DEP cases.

### Homeokinesis / differential extrinsic plasticity (DEP) — **not a reward. Admissible under #5. Precondition fails.**

**Source.** Der & Martius (2015), "Novel plasticity rule can explain the development of sensorimotor
intelligence," *PNAS* 112(45):E6224–E6232; arXiv:1505.00835. Full text extracted and quoted verbatim.

This is the only formulation found that creates behaviour from a standstill **with no reward and no
objective function of any kind**, and the authors state the negative claim directly:

> "self-determined development can be grounded in this synaptic dynamics, without having to postulate
> any higher level constructs such as intrinsic motivation, curiosity, goal orientation, or a specific
> reward system."

On #5's three properties: no credit assignment (the rule is local and synaptic, `τ Ċ_ij = ỹ_i ẋ_j −
C_ij`), no satisfaction detector, and it enters as plasticity driven by observed sensor and motor
velocities — **evidence, not a term modifying an update toward a goal**. It is also started from
exactly this system's degeneracy:

> "we always start our system under definite initial conditions… by choosing C = 0 and h = 0, meaning
> that all actuators of the joints are in their central position."

And its controller is trivially simple — "This controller network may appear utterly oversimplified…
a simple one-layer network" — which again refutes the "already varied controller" reading of area 3.

**Its precondition is stated in one sentence, and it is the one this system fails:**

> "For a demonstration, consider the trivial case ẏ = 0 where the body is at rest and will stay there
> as long as there are no extrinsic perturbations. However, if the body is being kicked by some
> external force, x, ẋ and hence ỹ may vary so that C changes and the system is driven out of the
> global attractor if κ is sufficiently large. In fact we observe in the experiments how an initial
> kick acts like a dynamical germ for the starting of an individual behavior development."

The mechanism by which the humanoid bootstraps is entirely physical:

> "The first contact with the ground and the gravitation exert forces on the joints that lead to
> non-zero sensor readings. This creates a first learning signal and leads to small movements which get
> more and more amplified and shaped by the body-environment interaction."

And the world must be able to answer back:

> "this effect depends crucially on the mass of the wheel — it must be large enough so that, by its
> inertia, it really can give a feedback to the robotic system."

> "All the robot has is the physical answer of the environment (the wheel) by the reactive forces."

**This is the decisive comparison for #529.** The one reward-free curiosity-like mechanism in the
literature works because gravity, contact and inertia supply the variation the controller does not.
This system's arena supplies none: the world is rank-one, the arm sits at its stops, and enriching the
sandbox is out of scope by #127.

---

## (c) The negative result whose shape matches this system

Same source (Der & Martius, arXiv:1505.00835, §5.3), comparing DEP against the plain differential
Hebbian rule (DHL) on identical hardware:

> "Since DHL is not able to depart from the C = 0 condition, we copy the synaptic weights (C) of the
> DEP run to the DHL experiment after 10 sec."

> "For DHL the matrix reduces to have only a single non-zero eigenvalue. This in turn means that all
> future sensor values are projected onto the corresponding eigenvector and the learning dynamics
> cannot depart from that."

> "For a different initialization DHL (with normalization) may also produce continuous motion patterns
> but it is generally much less sensitive to the embodiment and perturbations and often falls into the
> C = 0 state, which it cannot exit anymore."

Three coincidences with this project's readings, none of them loose:

1. **The zero fixed point that cannot be exited** — this system's arm travel is 0.000e+00 ± 0.000e+00,
   with a world-independent constant command (#120), and no T2 condition put it above the frozen
   baseline.
2. **"Only a single non-zero eigenvalue… cannot depart from that"** — this system's composed rim-to-apex
   effective rank is exactly 1.000 on the full dome under both T3 arms, ~1.08 under every T2 condition
   including the zero-supply control, and per-cell excitation participation ratio is 1.00 everywhere
   (T0, ledger row 1). Rank one, in the literature's own diagnostic, is the non-exitable state.
3. **"much less sensitive to the embodiment"** — this system's untrained taper of two to three decades
   per interior hop (#120) and #224's arithmetic-floor finding say the embodiment's answer, if any,
   does not reach the deep cells.

The literature's proposed exit from the DHL trap is DEP's extrinsic term ỹ — i.e. **more embodiment
feedback, not more intrinsic drive**. That is a finding about ordering, and it is the ticket's
question answered: rung 2 does not stand on rung 1's failure; it stands on a precondition that rung 1
was the attempt to establish.

### Developmental-robotics cases with a comparably degenerate starting controller (area 4)

**They exist, they succeed, and none of them succeed for a reason available here.**

| case | starting controller | how variation gets in | available here? |
|---|---|---|---|
| Rolf et al. 2011 online goal babbling | `g(x, θ₀) = const = q^home` — a constant | perturbation term `E_t(x)`, "arise naturally in physical systems"; weighted by observed effector displacement | no — `‖Δx‖ = 0` zeroes the weight |
| Der & Martius 2015 DEP | `C = 0, h = 0`, all actuators centred | extrinsic ẏ from gravity/contact/inertia; "an initial kick acts like a dynamical germ" | no — no physical answer; ẏ ≡ 0 |
| Der & Martius 2015 DHL control | `C = 0` | none; documented as unable to leave `C = 0` | this is the state the system is in |
| Lehman & Stanley 2011 novelty search | population of small NEAT networks | mutation + population; novelty only selects | no population |
| Baranes & Oudeyer 2013 SAGG-RIAC | assumed low-level reaching controller | micro-actions with observable consequences | no reaching controller; #120's constant |

**No case was found in which an intrinsic drive created variation in a system whose actuator response
was measured at zero.** This is a genuine negative: the search was run against goal babbling, motor
babbling, homeokinesis, empowerment, learning progress, novelty search and bonus-based exploration,
and the working cases uniformly obtain their first learning signal from a source outside the
controller.

---

## The preconditions, with a verdict for this system

Each row: the precondition, the source that states it, the project reading that decides it, and
yes / no / unknown. *No* rows are the load-bearing ones.

| # | Precondition | Stated by | This system's reading | Verdict |
|---|---|---|---|---|
| P1 | A **non-zero actuator→outcome response**: a motor perturbation must produce measurable displacement in an observed outcome space | Rolf et al. 2011, Eqn. 5: `w^eff = ‖Δx‖·‖Δq‖⁻¹`; Salge et al. 2014: "The lowest value for empowerment is 0, which means that an agent has no influence on the world that it can perceive" | arm travel **0.000e+00 ± 0.000e+00** per tick, T1's winner, full dome 30k (#529); frozen baseline parked at 5.042e-05 | **no** |
| P2 | An **exploratory perturbation** on the command that survives to the outcome | Rolf et al. 2011: "it is necessary to consider exploratory noise, or rather perturbations of the motor system"; Der & Martius 2015: "an initial kick acts like a dynamical germ" | motor babble tested on T2 and its reach **decayed** 10.1 → 1.82 by 10k; untrained taper 2–3 decades per interior hop (#120); deviation below the arithmetic floor never arrives at depth (#224) | **no** (tested, failed) |
| P3 | An **outcome space with more than one direction**, lower-dimensional than the motor space and redundant | Rolf et al. 2011: "Tasks … are typically substantially lower-dimensional than the motor systems themselves… If there are multiple ways to achieve the same result there is no inherent need to know all of them" | composed rim-to-apex transport rank-one (#497), **exactly 1.000** on the full dome under both T3 arms; 1.298 → ~1.08 under every T2 condition and the zero-supply control | **no** |
| P4 | Evidence whose **direction turns over**, not merely whose magnitude varies | Rolf et al. 2011 `w^dir`, an angle between observed and intended displacement; #517's own restatement of the same variable | per-cell excitation participation ratio **1.00 everywhere** (T0, ledger row 1); centred PR 1.20–1.29 against `m_e = 3` at the core–apex edges under every T2 condition, no separation beyond spread | **no** |
| P5 | A **low-level goal-reaching controller** that turns a chosen target into micro-actions | Baranes & Oudeyer 2013: "the system would then try to reach it with a lower-level goal-reaching architecture typically based on coupled inverse and forward models" | the untrained command is **one world-independent constant** (#120); the arm sits at its stops | **no** |
| P6 | A **learning rule with an escape from its zero fixed point** | Der & Martius 2015: DHL "is not able to depart from the C = 0 condition"; "often falls into the C = 0 state, which it cannot exit anymore" | zero travel under a constant command, no T2 condition above the frozen baseline; ADR-0009's falsifier fired on both clauses | **no** |
| P7 | A **world that answers**: environmental dynamics that return force/state changes contingent on recent action | Der & Martius 2015: the wheel "must be large enough so that, by its inertia, it really can give a feedback to the robotic system"; "All the robot has is the physical answer of the environment" | the world is **rank-one** (#517's P2); enriching the sandbox is **out of scope** (#127), so this is not merely unmet but closed | **no** |
| P8 | **Counterfactual evaluation** of candidate actions before acting (required by empowerment and by expected-free-energy epistemic value) | Salge et al. 2014: channel capacity is "the maximal information flow that could possibly be induced by a suitable choice of X," computed by Blahut–Arimoto over p(a) | ADR-0003: "the architecture has **no counterfactual evaluation at all**," amended into a position, not a cost; ADR-0009 makes the same refusal of the epistemic summand | **no**, and structurally barred |
| P9 | A **scalar objective a policy maximises**, with credit assignment | Pathak et al. 2017: "a policy that outputs a sequence of actions to maximize that reward signal"; Bellemare et al. 2016's "augmented Bellman equation"; Burda et al. 2018's "exploration bonus"; Oudeyer et al. 2007's "internal reward" | barred by **#5** unless its necessity bar is met; not met by this pass, which found the deficit upstream at P1/P2 | **no**, and barred |
| P10 | A **population or behaviour archive** among which novelty can select | Lehman & Stanley 2011: "the population of genomes that is central to such algorithms naturally covers a wide range of expanding behaviors" | no population; one behaviour, parked | **no** |
| P11 | Curiosity **selects among behaviours a controller already produces** (the ticket's area-3 conjecture) | true only of novelty search (above); **false** of Rolf's goal babbling (`g = const = q^home`) and of DEP (`C = 0`, one-layer network) | n/a — the conjecture does not hold of the cases that matter; the operative precondition is P1, not this | **conjecture not supported** |
| P12 | An environment **not dominated by unlearnable stochasticity** (prediction-error methods' known failure) | Burda et al. 2018: "If the transitions in the environment are random, then even with a perfect dynamics model, the expected reward will be the entropy of the transition" | not the failure mode here; this system's problem is the mirror image — nothing varies at all | **yes**, vacuously — and it buys nothing |
| P13 | **Limitations in sensing and acting** fine enough to give the intrinsic quantity structure | Salge et al. 2014: "for meaningful interaction to emerge … limitations in sensing and acting need to be present" | not read; the actuator became readable only at #506 and the conduction ratio has never been read on a world that varies | **unknown** |
| P14 | A learning signal that is **reward-free in #5's sense** (no credit assignment, no satisfaction detector, entering as evidence) | only Der & Martius 2015 qualifies: "without having to postulate … intrinsic motivation, curiosity, goal orientation, or a specific reward system" | the sole admissible formulation found — and its own precondition is P1, which reads **no** | **yes** (an admissible formulation exists), **but it is inert here** |

**Twelve no, one unknown, one yes-but-inert.** The `no` at P1 and P7 is not a gap in the design; it is
a measurement, taken twice, on two surfaces, with a control run for the purpose.

## What this pass does *not* claim

It does not claim the curiosity drive cannot be built, that it should be dropped, or that any
particular thing should be built instead — #529 forbids proposing a build and this document does not.
It does not touch ADR-0026's bar, the map's Destination, or the ordering of the apex ladder. It offers
no reading of abstraction, and every project figure quoted names its surface per #455.

It also does not claim that P1's `no` is permanent. The arm reads 0.000 **under T1's winner with the
untrained constant command and the arm at its stops**; whether the actuator would move under any other
command is a question this project has not asked in a form that would answer it, and #506's readable
actuator is the instrument that could. That is a stated unknown, not a hedge: a reading of the
actuator's response to a commanded deviation would decide P1 directly, and P1 decides P6, P7 and P14
after it.

## Gaps

- **Klyubin, Polani & Nehaniv (2005)**, the primary empowerment paper, was unreachable (the
  Hertfordshire repository refused the connection). Salge, Glackin & Polani (2014), by the same group
  and the field's standard reference text, is quoted in its place; every empowerment claim above is
  from that full text, not from the 2005 abstract.
- **Rolf, Steil & Gienger (2010)**, *IEEE TAMD* — paywalled; only the 2011 ICDL preprint was read in
  full. All goal-babbling quotations are from the 2011 preprint.
- **Baranes & Oudeyer (2013)** was read through an HTML rendering of the arXiv preprint rather than
  extracted PDF text; its quotations should be treated as one grade less reliable than the others and
  re-checked before any of them is load-bearing. None of the verdict rows depends on it alone (P5 is
  corroborated by #120).
- **Not searched**: the option-discovery and skill-diversity line (DIAYN, VIC and successors). All are
  mutual-information objectives maximised by a policy, so they would land where P9 lands, but that is
  an inference rather than a reading.
- **Der & Martius's PNAS version** returned HTTP 403; the arXiv preprint (1505.00835) of the same work
  was used. Section numbering in quotations follows the preprint.

## Sources

1. Rolf, M., Steil, J. J. & Gienger, M. (2011). *Online Goal Babbling for rapid bootstrapping of
   inverse models in high dimensions.* IEEE ICDL-EpiRob. <https://www.honda-ri.de/pubs/pdf/1732.pdf>
2. Rolf, M., Steil, J. J. & Gienger, M. (2010). *Goal Babbling Permits Direct Learning of Inverse
   Kinematics.* IEEE TAMD 2(3):216–229. DOI 10.1109/TAMD.2010.2062511
3. Der, R. & Martius, G. (2015). *Novel plasticity rule can explain the development of sensorimotor
   intelligence.* PNAS 112(45):E6224–E6232. DOI 10.1073/pnas.1508400112; arXiv:1505.00835
4. Salge, C., Glackin, C. & Polani, D. (2014). *Empowerment — An Introduction.* In *Guided
   Self-Organization: Inception*, Springer. arXiv:1310.1863
5. Klyubin, A. S., Polani, D. & Nehaniv, C. L. (2005). *Empowerment: a universal agent-centric measure
   of control.* IEEE CEC. (unreachable — see *Gaps*)
6. Oudeyer, P.-Y., Kaplan, F. & Hafner, V. V. (2007). *Intrinsic Motivation Systems for Autonomous
   Mental Development.* IEEE Trans. Evol. Comput. 11(2):265–286. DOI 10.1109/TEVC.2006.890271
7. Baranes, A. & Oudeyer, P.-Y. (2013). *Active Learning of Inverse Models with Intrinsically
   Motivated Goal Exploration in Robots.* Robotics and Autonomous Systems 61(1):49–73. arXiv:1301.4862
8. Pathak, D., Agrawal, P., Efros, A. A. & Darrell, T. (2017). *Curiosity-driven Exploration by
   Self-supervised Prediction.* ICML. arXiv:1705.05363
9. Burda, Y., Edwards, H., Pathak, D., Storkey, A., Darrell, T. & Efros, A. A. (2018). *Large-Scale
   Study of Curiosity-Driven Learning.* arXiv:1808.04355
10. Burda, Y., Edwards, H., Storkey, A. & Klimov, O. (2018). *Exploration by Random Network
    Distillation.* arXiv:1810.12894
11. Bellemare, M. G., Srinivasan, S., Ostrovski, G., Schaul, T., Saxton, D. & Munos, R. (2016).
    *Unifying Count-Based Exploration and Intrinsic Motivation.* NeurIPS. arXiv:1606.01868
12. Lehman, J. & Stanley, K. O. (2011). *Abandoning Objectives: Evolution through the Search for
    Novelty Alone.* Evolutionary Computation 19(2):189–223.
