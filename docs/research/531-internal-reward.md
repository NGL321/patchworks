# Citation pass: collapse-prevention and drives under a permitted internal reward (patchworks#531)

R-C on map [#517](https://github.com/NGL321/patchworks/issues/517), run against the scope relaxation
of 2026-09-05: *externally generated rewards remain unwanted; internally generated reward signals are
acceptable, including hardcoded ones.* This pass re-asks
[R-B](https://github.com/NGL321/patchworks/issues/529)'s question with the families R-B was obliged to
discard on #5's bar alone, and then asks the part R-B could not: **does an internally generated reward
touch the endogenous rank collapse?**

Fourteen primary sources were read in extracted full text or in the arXiv HTML rendering and are quoted
verbatim below; two were reached only at abstract level and are marked in *Gaps*. **No build is
proposed and no project code is touched**, per the ticket. R-B's readings of DEP and of IAC are cited
as R-B's where this pass did not re-derive them; where this pass did re-read a source R-B had read
(IAC, ICM, RND, pseudo-counts, SAGG-RIAC), the quotations below are this pass's own extraction.

---

## Headline: four findings, blunt

**1. The relaxation is almost entirely absorbed by an ADR that was never relaxed.** #5's definition of
a reward has three properties, the third of which is that it *"enters the learning rule as a term
modifying the update rather than as evidence"* ([#5](https://github.com/NGL321/patchworks/issues/5),
the #495 annotation comment). **ADR-0008 forbids exactly that object, independently of #5.** Every
formulation in R-B's discarded set, and every one added here, enters as an additive bonus on the RL
objective — by construction, in the authors' own equations: ICM's `r_t = r^i_t + r^e_t`, NGU's
`r_t = r^e_t + β·r^i_t`, pseudo-counts' `R^+_n(x,a) := β(N̂_n(x)+0.01)^{-1/2}` inside *"the augmented
Bellman equation"*, RND's `V = V_E + V_I`. The bar that moved was not the bar that was doing the work.

**2. The only internally generated signal this architecture can admit at the internal rim is one that
is not a reward in #5's sense at all — it is a second drive boundary cell, which the spec already
permits and which needed no relaxation.** The spec had already written the sentence:
*"Whatever eventually supplies curiosity enters as **another drive boundary cell**, not as a new
channel and not as a second error signal."* A boundary-cell assertion enters **as evidence**, through
an ordinary masked linear restriction map, holds *"a constant, not a model"*, needs no satisfaction
detector (*"Release needs no detector"*), and reaches the world only through the graph. Every one of
#5's three properties fails on it, both internal-rim bans hold, ADR-0003 is untouched, ADR-0008 is
untouched, ADR-0004 is untouched. Nothing in the relaxation was needed to get there. The one candidate
family that survives everything — hardcoded homeostatic drive reduction — survives **by becoming the
object the project already has**.

**3. Ban 1 is decisive against the learned-model families, and ICM's own defence does not save it.**
ICM trains a forward model `ϕ̂(s_{t+1}) = f(ϕ(s_t), a_t)` in a feature space where *"there is no
incentive for ϕ(s_t) to encode any environmental features that can not influence or are not influenced
by the agent's actions."* That argues the model's **domain** is narrowed to the controllable; it does
not argue that the object is not a model of the world. It is a learned map from a world state and an
action to the next world state, one tick ahead, in a learned basis. The spec closes this escape
explicitly and in advance: *"a faculty could read only what the graph gives it and still build a
forward model over that."* That sentence is written about precisely ICM's argument. **ICM fails ban 1.**

**4. The part that matters, and it inverts the ticket's hope. In the reinforcement-learning
literature, reward-driven learning is a documented *cause* of effective-rank collapse, not a cure for
it.** Kumar et al., abstract, verbatim: *"We identify an implicit under-parameterization phenomenon in
value-based deep RL methods that use bootstrapping … more gradient updates decrease the expressivity of
the current value network. We characterize this loss of expressivity via a drop in the rank of the
learned value network features."* And the causal attribution is to the reward machinery specifically:
*"By removing bootstrapped updates and instead regressing directly to Monte-Carlo estimates of the
value, the effective rank does not collapse."* Sokar et al. concur from the unit side: *"Target
non-stationarity exacerbates dormant neurons"* while *"Input non-stationarity does not appear to be a
major factor."* **Every published cure is either (a) an additive term on the objective — InFeR, DR3,
Barlow Twins, VICReg, W-MSE — which ADR-0008 bars, and which for the SSL family is a batch statistic
by construction; or (b) a reset/re-initialisation, which is not reward-driven at all.** The single
mechanism family that arrests collapse *in the learning dynamics*, online, locally, without batch
statistics and without an added objective term, is the Oja/Sanger subspace + anti-Hebbian
decorrelation line — **and it is not reward-driven, has nothing to do with this relaxation, and
belongs to [R-A](https://github.com/NGL321/patchworks/issues/528)'s side of the problem, not this
one.**

---

## The constraint set this pass judges against

Five bars, stated so each table cell can be read against one:

| bar | source | test applied here |
|---|---|---|
| **Ban 1** | spec 04, *The internal rim*: *"An internal faculty may not hold its own model of the world."* | Does the mechanism learn a map that predicts world state (raw or in a learned feature basis)? |
| **Ban 2** | same: *"An internal faculty reaches the world only through the graph. Never directly."* With the warning: *"a faculty could read only what the graph gives it and still build a forward model over that."* | Does it need a read the graph does not give it? Note ban 2 is *weaker* than ban 1 and almost every candidate passes it — which is why ban 1 does the work. |
| **ADR-0003** | *"the architecture has **no counterfactual evaluation at all**"* | Does it require evaluating candidate actions/futures before acting? |
| **ADR-0008 / no batch stats** | transport rule *"descends disagreement alone"*; permitted global signals are *"schedule-shaped rather than information-shaped"*; locality boundary *"strictly the cell"* | Does it add a term to an update, or need a statistic pooled over a batch, a window, a replay buffer or a population? |
| **ADR-0004** | restriction maps are **linear**; transport is a linear exchange | Does its ingress require a nonlinear map at the rim? |

---

## Part 1 — the candidate table

Columns as pre-registered. **survives** = the mechanism as its own paper defines it clears the bar.
Justifying quotes follow the table, keyed by candidate.

| candidate | ban 1 (no world model) | ban 2 (world only via graph) | ADR-0003 | ADR-0008 / no batch stats | needs hardcoding | changes any R-B verdict | touches endogenous collapse |
|---|---|---|---|---|---|---|---|
| **ICM** (Pathak 2017) | **no** — learned forward model in feature space | yes | **no** — `r^i` is a policy objective maximised over futures | **no** — `r_t = r^i_t + r^e_t`, additive bonus | hardcoding is vacuous (see below) | no | **no** — supply side only |
| **IAC** (Oudeyer/Kaplan/Hafner 2007) | **no** — module *M* predicts sensorimotor consequences; KGA predicts *M*'s error | yes | **no** — *"the action for which the system expects the maximal learning progress is chosen"* | **no** — windowed mean error rate, `θ ≈ 15`, smoothing `≈ 25`; region splits | **vacuous** — a hardcoded derivative of nothing | no | **no** |
| **SAGG-RIAC / competence progress** (Baranes & Oudeyer 2013) | **no** — requires *"local corresponding forward and inverse models"* | yes | **no** — goal is chosen by expected competence progress before acting | **no** — sliding-window interest over region history | **vacuous** | no | **no** |
| **Pseudo-counts** (Bellemare 2016) | **partly** — a density model over *states* is a model of the world's statistics, not its dynamics; still a learned model of world content | yes | **no** — bonus lives in *"the augmented Bellman equation"*, i.e. in a value backup | **no** — streaming density model, `R^+_n = β(N̂_n+0.01)^{-1/2}` additive | **vacuous** — a hardcoded count is a constant | no | **no** |
| **RND** (Burda 2018) | **yes, narrowly** — predicts a *fixed random network's* features, not the world's next state | yes | **no** — two value heads `V = V_E + V_I` over returns | **no** — *"a running estimate of the standard deviations of the intrinsic returns"*; *"running mean … running standard deviation"* — streaming statistics, twice | **vacuous** | no | **no** |
| **DIAYN / VIC** (skill diversity) | **no** — learned discriminator `q_φ(z\|s)` is a learned model over world states | yes | **no** — pseudo-reward `r_z(s,a) ≜ log q_φ(z\|s) − log p(z)` maximised by policy gradient | **no** — discriminator by SGD; MI objective is a distributional (population) quantity | **vacuous** — a hardcoded discriminator asserts a fixed labelling of states, i.e. a world model | no | **no** |
| **NGU / Agent57** episodic novelty | **no** — learned controllable-state embedding + episodic memory of states | yes | **no** — `r_t = r^e_t + β·r^i_t` | **no** — k-NN over an episodic memory `M`; `d_m²` *"a running average"*; RND modulator on top | **vacuous** | no | **no** |
| **Plan2Explore / disagreement ensembles** | **no** — an explicit ensemble of forward dynamics models; Plan2Explore learns a latent world model outright | yes | **no, flagrantly** — *"optimized purely from trajectories imagined under the model"* | **no** — ensemble variance is a statistic over K models; additive intrinsic reward | **vacuous** — variance of one hardcoded model is zero | no | **no** |
| **Surprise minimisation (SMiRL)** | **no** — maintains and fits `p_θ(s)`, a density model over visited world states, *as part of the agent state* | yes | **no** — `r(s_t) = log p_{θ_{t-1}}(s_t)` maximised as a return | **no** — maximum-likelihood density estimation over the visited history | **vacuous** | no | **no** |
| **Homeostatic / drive-reduction RL** (Keramati & Gutkin 2014; the Hull line) | **YES** — the drive is a fixed distance of an *internal* state from a setpoint; it holds no model of the world | **yes** | **yes**, if the drive is asserted rather than maximised over candidate outcomes | **yes**, if asserted as evidence; **no** as the paper uses it (drive reduction is fed to an RL learner) | **yes — and hardcoding is the point, not a compromise** | **no** | **no** |

### Verdict, Part 1

**One candidate survives — homeostatic drive reduction, in hardcoded form, with the RL machinery
stripped off.** What is left after stripping it is a scalar internal variable, a fixed distance
function, and an assertion. **That is the project's existing drive boundary cell.** The relaxation did
not admit it; the spec already did.

Every other candidate fails at least ban 1 *and* ADR-0003 *and* ADR-0008 — three independent bars, none
of them the reward bar that moved.

### The justifying quotes

**ICM** — Pathak, Agrawal, Efros & Darrell, "Curiosity-driven Exploration by Self-supervised
Prediction", arXiv:1705.05363 (ar5iv full text).

> "The forward model takes as inputs ϕ(s_t) and a_t and predicts the feature representation ϕ̂(s_{t+1})
> of s_{t+1}."

> "L_F(ϕ(s_t), ϕ̂(s_{t+1})) = ½‖ϕ̂(s_{t+1}) − ϕ(s_{t+1})‖²₂"

> "As there is no incentive for ϕ(s_t) to encode any environmental features that can not influence or
> are not influenced by the agent's actions, the learned exploration strategy of our agent is robust
> to uncontrollable aspects of the environment."

> "The policy sub-system is trained to maximize the sum of these two rewards r_t = r^i_t + r^e_t"

**Adjudication of ban 1 for ICM, stated plainly because the ticket asked for it.** ICM's own defence is
a claim about *what ϕ encodes*, not about *what f does*. The defence narrows the model's domain to the
agent-controllable subset of the world; it concedes, in the same breath, that ϕ encodes environmental
features the agent *can* influence — which are world features. `f: (ϕ(s_t), a_t) ↦ ϕ̂(s_{t+1})` is
therefore a learned predictor of the next state of the world, in a learned basis, held by the faculty.
Ban 1 says *"may not hold its own model of the world"*; it does not say *"may not hold a complete
model of the world"*, and the spec's stated reason — *"two models with no sheaf between them have no
defined way to disagree, so there would be no mechanism by which the graph could even discover the
imposition"* — applies with full force to a partial model. **ICM fails ban 1, and ICM's argument is an
argument about scope, not about kind.** A second, independent point: the spec's ban-2 clause was
written to catch exactly this shape — *"a faculty could read only what the graph gives it and still
build a forward model over that."* ICM is that sentence's worked example.

**IAC** — Oudeyer, Kaplan & Hafner, "Intrinsic Motivation Systems for Autonomous Mental Development",
*IEEE Trans. Evol. Comput.* 11(2):265–286, 2007. Author PDF, text extracted and quoted verbatim.

> "For each situation that the robot encounters, it is given an internal reward which is equal to the
> inverse of this difference (which also corresponds to the local derivative of the error rate curve
> of M). This internal reward is positive when the error rate decreases, and negative when it
> increases."

> "The motivation system of the robot is then a system in which the action chosen is that for which KGA
> predicts that it will lead to the greatest decrease of the mean error rate of M."

> "The mean error rate in prediction is computed at … where [θ] is a time window parameter typically
> equal to 15, and [ς] a smoothing parameter typically equal to 25."

> "The goal of the intrinsically motivated robot is then to maximize the amount of internal reward that
> it gets. Mathematically, this can be formulated as the maximization of future expected rewards (i.e.,
> maximization of the return)"

> "The action for which the system expects the maximal learning progress is chosen and executed except
> in some cases when a random action is selected (ε – greedy action selection rule)."

Three bars fail at once: a learned predictor `M` of sensorimotor consequences (ban 1); *"the action for
which the system expects the maximal learning progress is chosen"* — an evaluation of candidate actions
before acting (ADR-0003); and a windowed mean over the last 15 errors with smoothing 25, plus
incremental region splitting, which are batch/streaming statistics over a history (ADR-0008).

**On whether a hardcoded learning-progress signal is vacuous — the ticket asked, and it is.** Learning
progress is *defined* as the time-derivative of the agent's own model error: *"the local derivative of
the error rate curve of M."* To hardcode it is to fix that derivative in advance, which is to assert a
constant. A constant assertion at the internal rim is a drive, and it is a drive that carries no
information about learning progress — it has the same relationship to IAC that `DRIVE_ASSERTION = 1.0`
has to the task. **Hardcoding does not rescue any learning-progress or novelty formulation; it deletes
the formulation and leaves a drive.** The same argument applies verbatim to competence progress,
pseudo-counts, RND, disagreement variance and episodic novelty. It is the single most consequential
finding of Part 1 after finding 1, because "the user has accepted hardcoding" reads at first like a
wide door and is in fact a door onto the room the project is already in.

**SAGG-RIAC** — Baranes & Oudeyer, arXiv:1301.4862 (ar5iv).

> "Once a goal has been actively chosen at the high-level, the goal directed exploration and learning
> mechanism at the lower can be carried out in numerous ways … guide the system toward the goal by
> executing low-level actions which allow a progressive exploration of the world toward this specific
> goal and that updates at the same time the local corresponding forward and inverse models"

> "Using a sliding window in order to compute the value of interest prevents the system from keeping
> each measure of competence in its memory"

> "the interest considers the variation of competences, and by using an absolute value, it considers
> cases of increasing and decreasing competences"

**Pseudo-counts** — Bellemare et al., "Unifying Count-Based Exploration and Intrinsic Motivation",
arXiv:1606.01868 (ar5iv).

> "R⁺_n(x,a) := β(N̂_n(x) + 0.01)^{−1/2}"

> "N̂_n(x) = ρ_n(x)(1 − ρ'_n(x)) / (ρ'_n(x) − ρ_n(x))"

> "We update our density model with the states generated by following the policy"

A density model over states, updated online from the state stream, whose output enters an augmented
Bellman backup. Ban 1 is a judgement call here — a density over states is a model of the world's
*content* rather than its *dynamics* — and it is recorded as such in *Decision provenance* (c). ADR-0008
disposes of it regardless: a streaming density model is a streaming statistic and `R⁺` is an additive
bonus.

**RND** — Burda, Edwards, Storkey & Klimov, arXiv:1810.12894 (ar5iv).

> "The bonus is the error of a neural network predicting features of the observations given by a fixed
> randomly initialized neural network."

> "In order to keep the rewards on a consistent scale we normalized the intrinsic reward by dividing it
> by a running estimate of the standard deviations of the intrinsic returns."

> "we use an observation normalization scheme often used in continuous control problems whereby we
> whiten each dimension by subtracting the running mean and then dividing by the running standard
> deviation."

> "we can fit two value heads V_E and V_I separately using their respective returns, and combine them
> to give the value function V = V_E + V_I."

**RND is the one candidate that arguably clears ban 1** — its predictor targets a fixed random network,
not the world's next state — which is why it is worth stating separately that it dies twice over on
ADR-0008: *two* running normalisers, each a streaming statistic, plus an additive value head. It is the
cleanest demonstration of headline finding 1: the reward bar was never what was excluding it.

**DIAYN / VIC** — Eysenbach et al., arXiv:1802.06070 (ar5iv); Gregor et al., arXiv:1611.07507
(abstract only — *Gaps*).

> "maximize ℐ(S,Z) + ℋ[A|S] − ℐ(A;Z|S)" ; the agent receives "r_z(s,a) ≜ log q_φ(z|s) − log p(z)"

> "a method for learning useful skills without a reward function"

R-B recorded the skill-diversity line as an explicit gap and inferred it lands at P9. **That inference
was correct but for an incomplete reason.** DIAYN is barred by ban 1 before it reaches any reward bar:
`q_φ(z|s)` is a learned model over world states held by the faculty. And the mutual information
`ℐ(S,Z)` is a functional of a *distribution over states*, which is a population statistic and has no
per-tick local estimator — ADR-0008's locality boundary is *"strictly the cell"*. VIC's abstract makes
the empowerment connection explicit (*"an explicit measure of empowerment in a given state"*), which
inherits R-B's ADR-0003 bar on empowerment as well.

**NGU** — Badia et al., arXiv:2002.06038 (ar5iv).

> "At the beginning of each episode, the episodic memory starts completely empty. At every step, the
> agent computes an episodic intrinsic reward, and appends the controllable state corresponding to the
> current observation to the memory."

> "pseudo-counts are computed using the k-nearest neighbors of f(x_t) in the memory M … where d_m² is a
> running average of the squared Euclidean distance."

> "The augmented reward at time t is then defined as r_t = r^e_t + β·r^i_t"

An episodic memory of world states held outside the graph is both a world model in ban 1's sense and a
batch statistic. It is also, note, exactly the *hippocampal-analogue memory* the spec lists as a
separate fog item at the internal rim — so if this family ever returns it returns as that fog item and
under that item's own bans, not as a reward.

**Plan2Explore / disagreement** — Sekar et al., arXiv:2005.05960; Pathak, Gandhi & Gupta,
arXiv:1906.04161 (both ar5iv).

> "World models summarize past experience into a representation of the environment that enables
> predicting imagined future sequences"

> "The exploration policy is optimized purely from trajectories imagined under the model to maximize
> the intrinsic rewards computed by the model itself."

> "an ensemble of forward prediction models {f_θ1, f_θ2 …, f_θk} of the environment. Each of the model
> is trained to map a given tuple of current observation x_t and the action a_t to the resulting state
> x_{t+1}."

This family is the clearest ADR-0003 failure in the set: the intrinsic reward is *computed inside
imagined rollouts*. It is also the clearest ban-1 failure — the paper calls the object a world model.

**SMiRL** — Berseth et al., arXiv:1912.05510 (ar5iv).

> "r(s_t) = log p_{θ_{t-1}}(s_t)"

> "the parameters of the sufficient statistics are updated θ_t = U(τ_t) using a maximum likelihood
> state density estimation process"

> "SMiRL automatically discovers complex and coordinated behaviors without any reward signal"

Surprise *minimisation* was worth checking because it is the one intrinsic objective whose sign points
the way this architecture already points — a prediction-error minimiser. It fails identically: a fitted
density over visited world states is a world model under ban 1, held by the faculty and carried in the
agent's own state, and it is a maximum-likelihood estimate over a history.

**Homeostatic / drive-reduction RL** — Keramati & Gutkin, "Homeostatic reinforcement learning for
integrating reward collection and physiological stability", *eLife* 2014;3:e04811.

> "The physiological state of the animal at each time t can be represented as a point in this space,
> denoted by H_t = (h_{1,t}, h_{2,t}, .., h_{N,t})"

> the drive is "the distance of the internal state from the setpoint"

> "the rewarding value of this outcome can be defined as the consequent reduction of drive"

> "Inspired by these considerations (i.e. preservation of self-order and reduction of deviations), we
> propose a formal definition of primary reward … reminiscent of the drive-reduction theory"

**This is the ticket's one survivor, and the shape of its survival is the finding.** The drive function
is a *fixed* function of an *internal* variable. It reads no world state; it holds no dynamics model;
it needs no rollout; it has no window, buffer or batch. It is, structurally, `d(H_t, H*)` — a
hardcoded appetite, which is exactly the *"limbic-analogue appetite"* the spec names as a fog item at
the internal rim. What Keramati & Gutkin then do with it — feed drive reduction to an RL learner as a
primary reward — is the part that fails ADR-0008 and #5, and it is severable. Strip it and what
remains asserts a scalar at a boundary cell.

**And what remains is a drive, in this project's exact sense.** Cross-checked against the spec, clause
by clause: scalar stalk (*"The drive stalk is **scalar** — one dimension, and its edges are `m_e = 1`"*);
no satisfaction detector (*"Release needs no detector. The cell asserts *satisfied* forever"*); no world
model (*"it holds a constant, not a model"*); linear ingress (drive edges are *"ordinary edges with
ordinary masked linear restriction maps"*, satisfying ADR-0004); enters as evidence, not as a term
(*"Nothing is overridden. The assertion reaches those cells as ordinary disagreement"*). **Zero of #5's
three properties hold of it. It was never a reward, and the relaxation was not needed to admit it.**

---

## Part 2 — does any survivor move R-B's precondition verdicts?

**No. Not one verdict moves. A reward does not make a parked arm move.**

Stated in the terms the ticket asked for, because it is the true answer. R-B's decisive precondition
**P1** is a non-zero actuator→outcome response, read through Rolf's per-example weight
`w^eff_t = ‖x_t − x_{t−1}‖ · ‖q_t − q_{t−1}‖^{−1}`, which is zero at zero effector travel. That is an
arithmetic identity on a *measured displacement*. No term added to a policy objective, and no
assertion made at the internal rim, appears anywhere in it. An internally generated reward changes
which command the agent would like to issue; P1 asks whether issuing any command produces travel.
Those are different questions, and the second one is
[T5](https://github.com/NGL321/patchworks/issues/530)'s, which is reading it now.

Per surviving candidate, exhaustively:

| precondition | does hardcoded homeostatic drive move it? | why |
|---|---|---|
| **P1** actuator→outcome response | **no** | a scalar assertion at the internal rim is not a torque; travel is a property of the body. **Treated as OPEN** per the ticket, pending T5 — but open in *either* direction, and no reward closes it. |
| **P2** exploratory perturbation surviving to the outcome | **no** | the taper is 2–3 decades per interior hop (#120); a drive asserted at the apex is *further* from the rim than the perturbation T2 injected and failed to land |
| **P3** outcome space with >1 direction | **no** | see Part 3; this is the collapse question, and the answer there is no |
| **P4** evidence whose direction turns over | **no** — and note it *cannot*, by design | *"The drive stalk is scalar … It asserts *satisfied* and nothing else"*; a one-dimensional constant has no direction to turn over. Strength is fan-out, not width. |
| **P5** low-level goal-reaching controller | **no** | this is a controller question; the project has explicitly no subordinate executor for the PoC |
| **P6** escape from the rule's zero fixed point | **no** | the fixed point belongs to the transport rule, which ADR-0008 fixes as descending disagreement alone; a drive edge adds disagreement at the apex, it does not change the rule |
| **P7** a world that answers | **no** | closed by #127's scope, not by any signal |
| **P8** counterfactual evaluation | **no**, still barred | ADR-0003 was not relaxed |
| **P9** scalar objective a policy maximises | **still absent, and now known to be unwanted** | the surviving candidate has no policy and no maximisation; #5's necessity bar is unmet because the deficit is at P1 |
| **P10** population or archive | **no** | none, and none of the survivors need one |
| **P13** limitations in sensing/acting fine enough | **unknown**, unchanged | never read; #506's readable actuator is the instrument |
| **P14** learning signal reward-free in #5's sense | **yes — and this is where the ticket lands** | the survivor is reward-free in #5's sense. It joins DEP as a second admissible formulation, and like DEP its precondition is P1, so it is **admissible and inert** for the same reason |

**Net movement: P14 gains a second member and stays inert. Nothing else moves.** R-B's twelve `no`
remain twelve `no`.

---

## Part 3 — rank collapse in the learning dynamics

The question, restated exactly: **is there any reward-driven mechanism in the literature that arrests
rank collapse *in the learning dynamics* — that keeps a learned linear/bilinear map from collapsing to
rank one — as opposed to merely supplying more or better-motivated input?**

The map's reading that motivates the question: composed rim-to-apex effective rank reads **exactly
1.000** on the full dome under both T3 arms, and decays toward one under T2's **zero-supply control
arm** — i.e. with no exogenous supply at all. R-B's matching literature result is DEP's plain
differential-Hebbian control, which *"is not able to depart from the C = 0 condition"* and whose
linearised sensor map *"reduces to have only a single non-zero eigenvalue. This in turn means that all
future sensor values are projected onto the corresponding eigenvector and the learning dynamics cannot
depart from that"* (Der & Martius 2015, quoted from R-B — not re-read in this pass; see *Gaps*).

### 3a. The RL literature's own rank diagnostic: reward is the cause, not the cure

This is the strongest result in the pass, and it inverts the ticket's hope.

**Kumar, Agarwal, Ghosh & Levine, "Implicit Under-Parameterization Inhibits Data-Efficient Deep
Reinforcement Learning", arXiv:2010.14498.** Abstract, verbatim and in full:

> "We identify an implicit under-parameterization phenomenon in value-based deep RL methods that use
> bootstrapping: when value functions, approximated using deep neural networks, are trained with
> gradient descent using iterated regression onto target values generated by previous instances of the
> value network, more gradient updates decrease the expressivity of the current value network. We
> characterize this loss of expressivity via a drop in the rank of the learned value network features,
> and show that this typically corresponds to a performance drop. We demonstrate this phenomenon on
> Atari and Gym benchmarks, in both offline and online RL settings. We formally analyze this phenomenon
> and show that it results from a pathological interaction between bootstrapping and gradient-based
> optimization. We further show that mitigating implicit under-parameterization by controlling rank
> collapse can improve performance."

Body, on the causal attribution — this is the load-bearing sentence for this ticket:

> "By removing bootstrapped updates and instead regressing directly to Monte-Carlo estimates of the
> value, the effective rank does not collapse."

And the rank measure they use is the same *kind* of quantity the map reads:

> "srank_δ(Φ) = min{k : Σ_{i=1}^{k} σ_i(Φ) / Σ_{i=1}^{d} σ_i(Φ) ≥ 1 − δ}"

> "Theorem 4.1: … srank_δ(M_{kl}') ≤ srank_δ(M_{kl})" — effective rank decreasing monotonically across
> bootstrapping iterations.

**Read against #531's question: the machinery that turns a reward into learning — bootstrapped value
backup — is itself a documented rank-collapse mechanism, and removing it removes the collapse.** A
reward introduced to fight rank collapse would, on this literature's own finding, be introducing the
thing that causes it in the setting where the diagnostic exists.

Three corroborating primary results, all pointing the same way:

**Kumar et al., "DR3: Value-Based Deep RL Requires Explicit Regularization", arXiv:2112.04716** —
locates the mechanism inside the TD update itself:

> "R_TD(θ) additionally includes a second term that is equal to the dot product of the gradient of the
> Q-function at the current and next states."

> "the implicit preference towards maximizing the dot products of features at consecutive state-action
> tuples is what we call 'feature co-adaptation.'"

> "value functions trained with offline deep RL eventually degrade in performance … and this
> degradation is correlated with the emergence of low-rank features."

**Sokar, Agarwal, Castro & Evci, "The Dormant Neuron Phenomenon in Deep Reinforcement Learning",
arXiv:2302.12902** — the same collapse read at the unit level, with the cause isolated:

> "An algorithm exhibits the dormant neuron phenomenon if the number of τ-dormant neurons in its neural
> network increases steadily throughout training."

> "The number of dormant neurons increases as training progresses … in contrast with supervised
> learning, where the number of dormant neurons remains low throughout training."

> "Target non-stationarity exacerbates dormant neurons" — while "Input non-stationarity does not appear
> to be a major factor."

That last pair is directly on this ticket's question and deserves emphasis: **the collapse is driven by
the moving target (the reward/bootstrap machinery), not by the input distribution (the supply side).**
It is the exact opposite of the hypothesis that a better-motivated behaviour policy would relieve it.

**Lyle, Rowland & Dabney, "Understanding and Preventing Capacity Loss in Reinforcement Learning",
arXiv:2204.09560**:

> "the non-stationary prediction problems in deep RL also result in capacity loss" — manifesting as
> "representation collapse, where the feature outputs for every state in the environment inhabit a
> low-dimensional – or possibly even zero – subspace."

**One important caution against over-claiming, from the same author.** Lyle, Zheng, Nikishin, Pires,
Pascanu & Dabney, "Understanding plasticity in neural networks", arXiv:2303.01486, explicitly declines
to make feature rank the *cause*:

> "for each of four quantities, there exists a learning problem where the quantity positively
> correlates with plasticity, and one in which it exhibits a negative correlation"

> "loss of plasticity is deeply connected to changes in the curvature of the loss landscape, but that
> it often occurs in the absence of saturated units"

> "selecting a network parameterization which smooths out the loss landscape is the most effective
> means of preserving plasticity of all approaches we have considered"

So the honest statement of the RL finding is: **rank collapse in deep RL is robustly *co-occurrent
with*, and by Kumar et al.'s ablation *caused by*, bootstrapped reward learning; whether rank is the
right causal variable for the downstream capacity loss is contested within the same research group.**
The direction of the arrow between reward and collapse is not contested, and it points the wrong way
for this ticket.

### 3b. The mechanisms that *do* arrest rank collapse, and whether reward is anywhere near them

| mechanism | arrests rank collapse in the learning dynamics? | reward-driven? | admissible here? |
|---|---|---|---|
| **Oja single-neuron rule** | **no — it is the collapse.** "a neuron computes the top eigenvector of the covariance matrix and outputs the first principle component" (Pehlevan et al.) | no | n/a — it is the technical counterpart of DEP's *single non-zero eigenvalue* |
| **Sanger's GHA / generalized Hebbian** | **yes** — deflation inside the same Hebbian update makes distinct outputs converge to distinct eigenvectors in order | **no** | **partly.** Local (*"uses only local signals at each synapse"*), online, no batch statistics, no additive objective term. But it is a **different update rule**, and ADR-0008 fixes the transport rule as descending disagreement alone — so adopting it is an ADR amendment, not a term. Not this ticket's, and **not reward's**: it belongs to R-A. |
| **Oja subspace rule (symmetric, multi-unit)** | **yes**, to the principal *subspace* rather than one eigenvector — "the existing similarity matching algorithms … extract the principal subspace of the dataset" | **no** | same as GHA |
| **Anti-Hebbian lateral decorrelation (Földiák; Pehlevan/Hu/Chklovskii)** | **yes** — "the strengthening of lateral synapses is sufficient to decorrelate neuronal output and project the input to its principal eigenvectors" | **no** | **partly, and this is the most interesting row.** Updates "depend on the activity of only pre- and postsynaptic neurons" and are online, single-sample. It requires *lateral* connections between units, which in this architecture would be edges — i.e. graph topology, not a faculty. Again R-A's, not reward's. |
| **Barlow Twins redundancy reduction** | **yes** — "by trying to equate the off-diagonal elements of the cross-correlation matrix to 0, decorrelates the different vector components" | **no** | **no, twice.** (i) It is an explicit additive objective term, `ℒ_BT ≜ Σ_i(1−C_ii)² + λΣ_iΣ_{j≠i} C_ij²` — barred by ADR-0008. (ii) `C` is "computed between the outputs of the two identical networks **along the batch dimension**" — a batch statistic by construction, barred independently. Its own claim that it "does not require large batches" is about batch *size*, not about being batch-free. |
| **VICReg variance–covariance terms** | **yes** — the covariance term "decorrelates the variables of each embedding and prevents an informational collapse" | **no** | **no, twice, identically.** `ℓ(Z,Z′) = λ s(Z,Z′) + μ[v(Z)+v(Z′)] + ν[c(Z)+c(Z′)]` is additive; both regularisers are defined "over a batch". |
| **W-MSE whitening** | **yes** — "The whitening operation has a 'scattering' effect on the batch samples, avoiding degenerate solutions" | **no** | **no** — the whitening matrix is computed from a batch covariance `Σ_V = 1/(K−1) Σ_k (v_k − μ_V)(v_k − μ_V)^T` |
| **Dimensional-collapse analysis / DirectCLR (Jing et al.)** | diagnostic + partial remedy — "the embedding vectors only span a lower-dimensional subspace"; DirectCLR "directly optimizes the encoder … without relying on an explicit trainable projector" | **no** | **no** — contrastive loss over batches; also its diagnosis is about *augmentation* variance, which has no counterpart here |
| **Attention rank collapse / skip connections (Dong, Cordonnier & Loukas)** | **yes**, architecturally — pure attention's "output converges doubly exponentially to a rank-1 matrix"; "skip connections and MLPs stop the output from degeneration" | **no** | **architectural, not a signal.** Note the specific negative result: layer normalisation "plays no protective role since it only performs right-multiplication, which cannot increase matrix rank." The admissible reading here is that **rank retention is bought by topology (a path that skips layers) and by nonlinearity, not by any objective and not by any reward.** For a rim-to-apex *composition* of linear restriction maps, this is the closest structural analogue in the literature to the map's exactly-1.000 reading, and it says the fix is a path, not a term. |
| **Plasticity-loss resets: ReDo (Sokar), primacy-bias resets (Nikishin)** | **yes**, by re-initialisation — ReDo: "Periodically check in all layers whether any neurons are τ-dormant; for these, reinitialize their incoming weights and zero out the outgoing weights"; Nikishin: "we periodically re-initialize the last layers of an agent's neural networks, while maintaining the experience within the buffer" | **no** — they are *repairs of damage the reward machinery did* | **no** — both need a replay buffer or a periodic global schedule that is information-shaped, not schedule-shaped, and both discard learned parameters wholesale |
| **InFeR (Lyle et al.) / DR3 (Kumar et al.)** | **yes** — DR3 "successfully alleviates the rank collapse issue noted in prior work" | **the closest thing to a *yes* in the table** — both are terms on a *reward-driven* objective | **no** — both are explicitly additive: "the training objective with DR3 is given by: L(θ) := L_Alg(θ) + c₀ Δ(θ)"; InFeR is "an ℓ2 regularization penalty on the output-space level". ADR-0008 bars the shape directly. And note what they are *for*: they repair collapse caused by TD, in a system that has TD. This architecture has no TD to repair. |

### Verdict, Part 3

**There is no reward-driven mechanism in the literature that arrests rank collapse in the learning
dynamics.** The nearest candidates — DR3 and InFeR — are not reward-*driven*; they are additive
corrections to damage that reward-driven bootstrapping caused, they exist only inside a value-learning
system, and their shape (a term added to the objective) is the exact object ADR-0008 forbids.

Every mechanism that genuinely arrests collapse in the learning dynamics falls into one of three
families, and **none of the three is reward**:

1. **A different local update rule** — Sanger/Oja subspace, anti-Hebbian lateral decorrelation. Local,
   online, batch-free. This is the real technical counterpart to DEP's single-non-zero-eigenvalue
   collapse, and it is unsupervised. **This is the one live thread in the whole pass, and it is R-A's,
   not R-C's.**
2. **An additive decorrelation term computed over a batch** — Barlow Twins, VICReg, W-MSE. Barred
   twice over, and the batch-statistics bar is fatal on its own.
3. **Topology** — skip connections in the attention result; and, by re-initialisation, the reset
   family. Not a signal at all.

**Reward acts on the behaviour policy, i.e. on the data distribution. Collapse here is a property of
the update rule's fixed point.** The literature confirms the separation and then goes one step further
than the ticket anticipated: in the one place where the RL literature *does* carry a rank-collapse
diagnostic, the reward machinery is on the causing side.

---

## What the relaxation actually bought

**Nothing.**

Unhedged, as the ticket requires. Expanded only to say why, in three sentences: every candidate the
relaxation was meant to re-admit is barred independently by ADR-0008 (as an additive term, a batch
statistic, or both) and by ban 1 (as a held world model), neither of which was relaxed; the one
candidate that survives all remaining bars survives only in hardcoded form, and a hardcoded internal
appetite is a scalar constant asserted at the internal rim, which is the drive boundary cell the spec
already defines and already permits; and on the endogenous collapse — the map's actual problem — the
literature's finding runs the other way, with reward-driven bootstrapping named as a cause of effective-rank
collapse rather than a cure for it.

**The spec's sentence stands unamended.** *"Whatever eventually supplies curiosity enters as another
drive boundary cell, not as a new channel and not as a second error signal."* The relaxation changes
nothing about it. What the relaxation *did* do is make it possible to say that sentence is not a
restriction the project is paying a cost for — with reward permitted, the reward families still do not
get in, and they are kept out by rulings the project made for other reasons.

---

## If anything survives everything: what it would cost at ADR level

One thing survives: **a hardcoded homeostatic appetite as a second drive boundary cell.** Adopting it
is an ADR-level act and is **not this ticket's to take**. What it would cost, recorded as options, not
as a proposal:

- **No new ADR is needed for the mechanism.** Spec 04 already says *"Curiosity, fatigue, or any later
  drive arrives as an additional boundary cell, which is an ordinary structural-mask change and needs
  no new mechanism."* The cost is a structural-mask change, not an architecture change.
- **What *would* need an ADR is the appetite's state variable.** Keramati & Gutkin's drive is
  `d(H_t, H*)` where `H_t` *varies*. A varying assertion is not the current drive: #495's conceded
  point is on record that *"the capacity-zero argument … does not survive a varying assertion."* A
  drive whose scalar moves has non-zero channel capacity, and *Valence, not specification* is then no
  longer exact by arithmetic. **That is the real price, and it is a price on the spec's cleanest
  guarantee.** It would need an amendment to *Valence, not specification*, and probably to ADR-0009.
- **Where `H_t` comes from is the second ADR question, and the harder one.** Any internal variable that
  tracks something about the world's state is a world model in ban 1's sense. An internal variable that
  tracks something about the *graph* — say a scalar function of the agent's own disagreement — is not,
  but it is then read from inside the sheaf by a faculty, which is a new ingress the internal rim has
  never had, and the spec's own warning applies: *"a faculty could read only what the graph gives it
  and still build a forward model over that."*
- **None of this touches the collapse.** Per Part 3, adding a drive edge adds disagreement at the apex;
  it does not change the transport rule, whose fixed point is what reads 1.000.

---

## Gaps

- **DEP (Der & Martius 2015) was not re-read in this pass.** Its quotations — *"is not able to depart
  from the C = 0 condition"*, *"reduces to have only a single non-zero eigenvalue…"* — are taken from
  R-B's extraction and are **one grade less reliable here** than this pass's own quotations, though
  R-B recorded them as full-text extractions.
- **VIC (Gregor, Rezende & Wierstra, arXiv:1611.07507) was read at abstract level only.** The DIAYN
  row's ban-1 and ADR-0008 verdicts rest on DIAYN's full text, not on VIC's; VIC's empowerment link is
  quoted from its abstract. **One grade less reliable.**
- **Oja (1982, 1989) and Sanger (1989, *Neural Networks* 2:459–473) were not read in their original
  venues.** Sanger's rule is quoted from the author's own NeurIPS 1988 paper (*"An Optimality Principle
  for Unsupervised Learning"*), which is primary but is the earlier, shorter statement. Oja's
  single-neuron and subspace results are quoted from Pehlevan, Hu & Chklovskii (arXiv:1511.09468),
  which is a primary paper **restating** them — i.e. a primary source for the claim's current form but
  **not** the source that owns it. **The Oja rows are one grade less reliable.**
- **Földiák (1990) was not read at all.** The anti-Hebbian lateral-decorrelation row rests entirely on
  Pehlevan/Hu/Chklovskii. **Marked as inference-from-a-restating-source.**
- **Hull (1943), *Principles of Behavior*, was not read.** The drive-reduction line is quoted through
  Keramati & Gutkin's own citation of it (*"reminiscent of the drive-reduction theory"*). The
  classical psychological literature on drive reduction is therefore represented here only by its
  modern computational restatement.
- **Agent57 (arXiv:2003.13350) was not read separately**; the NGU row covers the episodic-novelty
  mechanism Agent57 inherits, and Agent57's own contribution (a meta-controller over exploration
  parameters) was not examined. It would fail ADR-0008 on the meta-controller alone, but that is an
  **inference, not a reading**.
- **Empowerment was not re-read.** R-B's ADR-0003 bar on Blahut–Arimoto is taken as standing.
- **The T2/T3 numbers are taken from the map and its tickets, not re-measured.** No compute was run;
  three 100k processes were live on the box.
- **Barlow Twins and VICReg were read in ar5iv HTML rather than extracted PDF text**, consistent with
  most of this pass; IAC alone was read as extracted PDF text.

---

## Decision provenance

The ticket pre-registered the three parts, the table's exact columns, and the deliverable's shape;
those were followed. The following rested on this agent's judgement, were never put to the user, and
are **CEDED** under the ticket's hands-off framing — flagged here rather than acted on:

**(a) Reporting the relaxation as having bought *nothing*, in one word.** The ticket permits this
answer and asks for it unhedged if true. Saying it flatly is nonetheless a stronger act than
summarising literature, and it renders a user decision — taken deliberately, on 2026-09-05, as a
flex — inert. The user should see that this is what the pass concluded, not merely that some
candidates failed.

**(b) Promoting ADR-0008 to the pass's organising bar, ahead of the two internal-rim bans the ticket
named first.** The ticket lists the bans first and ADR-0008 fourth. This pass found ADR-0008 does more
work than either ban, because #5's third property and ADR-0008's prohibition are the *same object* seen
from two sides. That framing came from the mid-pass steer of a sibling session, not from the ticket,
and it reshaped headline finding 1. Recorded so its provenance is visible.

**(c) Reading pseudo-counts' density model as a world model under ban 1 only "partly".** A density over
states models the world's *content* but not its *dynamics*, and ban 1's text (*"its own model of the
world"*) does not settle which is meant. The cell is marked *partly* rather than *no* and the verdict
does not depend on it, because ADR-0008 disposes of pseudo-counts anyway. **The general question — does
ban 1 bar static models of world content, or only predictive ones? — is left open and is worth the
user's eye**, because it also decides the hippocampal-analogue memory fog item.

**(d) Adjudicating ICM against its own defence rather than recording the disagreement.** The ticket
asked for adjudication and this pass gave one: ICM's φ argument narrows the model's domain, not its
kind, so ICM fails ban 1. A reader who takes ban 1 to mean *a model of the world beyond the agent's own
effects* would decide it the other way. The reading here is the stricter one.

**(e) Declaring hardcoded learning-progress signals *vacuous* as a general result, not case by case.**
The argument is short — learning progress is by definition a derivative of the agent's own error, so
fixing it in advance asserts a constant — and it is applied at once to IAC, SAGG-RIAC, pseudo-counts,
RND, disagreement and NGU. That is a single argument doing six rows' work. It is believed sound; it is
flagged because "the user accepted hardcoding" was the relaxation's widest-looking clause and this
argument is what closes it.

**(f) Treating homeostatic drive reduction as a *survivor* rather than as out of scope.** The ticket
named it as a candidate worth checking; it was checked and it survives. But it survives only after
severing the RL learner Keramati & Gutkin actually use, which is most of their paper. Calling the
residue a survivor is a judgement about what "the candidate" is. The alternative honest report is
*nothing survives, and the shape that would have is already in the spec* — which is the same finding
worded to give the relaxation less credit.

**(g) Reading Kumar et al. as *inverting* the ticket's hope, rather than as merely orthogonal.** The
ticket invited this ("check whether the reward is the cause of collapse there rather than the cure")
and the abstract plus the Monte-Carlo ablation support it squarely. The caution is that Kumar et al.
diagnose a *value-function feature matrix* in a TD learner, and this system has neither; the transfer
to a composed rim-to-apex restriction map is by analogy of *shape*, not by shared mechanism. **The
inversion is real about the literature; whether it is real about this system is not established.**

**(h) Admitting Lyle et al. 2303.01486 as a counterweight against the pass's own strongest result.**
It weakens headline finding 4 by contesting whether rank is the right causal variable. Including it was
a choice; a pass optimising for a clean result would have left it out. It is in because the map's own
instrument is an effective rank and the user should know the causal status of that quantity is
contested inside the group that popularised it.

**(i) Nominating the Oja/Sanger/anti-Hebbian line as "the one live thread" and then handing it to
R-A.** That is a scoping act across two parallel tickets. Nothing was written to
[#528](https://github.com/NGL321/patchworks/issues/528) and no coordination was attempted, per this
ticket's instruction to coordinate with nothing. If R-A has already covered it, this is duplication; if
R-A has not, this pass has flagged a thread it is not entitled to pull.

**(j) Reading the attention rank-collapse result as saying "the fix is a path, not a term."** Dong et
al. is about self-attention, not about sheaf restriction maps, and the composition being analysed is a
different object. The structural point — that rank retention there is bought by topology and
nonlinearity rather than by any objective — is a genuine reading of their result, but its application
to a rim-to-apex composition is **analogy, not transfer**, and is flagged as such.

**(k) Treating P1 as open in both directions and declining to speculate on T5's outcome.** The ticket
required treating P1 as open; this pass additionally declined to state what would follow if T5 returns
non-zero travel, on the grounds that it is T5's resolution to write. A reader wanting the conditional
will not find it here.

---

**Sources read.** Pathak et al. arXiv:1705.05363 · Oudeyer, Kaplan & Hafner, *IEEE TEC* 11(2), 2007
(PDF text extracted) · Baranes & Oudeyer arXiv:1301.4862 · Bellemare et al. arXiv:1606.01868 · Burda et
al. arXiv:1810.12894 · Eysenbach et al. arXiv:1802.06070 · Gregor et al. arXiv:1611.07507 (abstract) ·
Badia et al. arXiv:2002.06038 · Sekar et al. arXiv:2005.05960 · Pathak, Gandhi & Gupta
arXiv:1906.04161 · Berseth et al. arXiv:1912.05510 · Keramati & Gutkin, *eLife* 2014;3:e04811 · Kumar
et al. arXiv:2010.14498 · Kumar et al. arXiv:2112.04716 · Sokar et al. arXiv:2302.12902 · Nikishin et
al. arXiv:2205.07802 · Lyle, Rowland & Dabney arXiv:2204.09560 · Lyle et al. arXiv:2303.01486 · Zbontar
et al. arXiv:2103.03230 · Bardes, Ponce & LeCun arXiv:2105.04906 · Ermolov et al. arXiv:2007.06346 ·
Jing et al. arXiv:2110.09348 · Dong, Cordonnier & Loukas arXiv:2103.03404 · Sanger, NeurIPS 1988 ·
Pehlevan, Hu & Chklovskii arXiv:1511.09468.

No project code was modified. No compute was run. No build is proposed.
