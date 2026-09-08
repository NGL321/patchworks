# 595 — Is perceptual error cleared by acting? Rao, Friston, and the sensory row

Context: [B36 / #595](https://github.com/NGL321/patchworks/issues/595), on map
[#532](https://github.com/NGL321/patchworks/issues/532). Opened by
[B30](https://github.com/NGL321/patchworks/issues/589)'s resolution, from an objection the user
raised during that session: ADR-0003 sorts edges by **who clears the disagreement**, and a cell's
disagreement with a *sensory* boundary cell appears to have two exits, not one — the cell changes
its belief, **or** the world writes something different next tick because the agent moved.

Corpus set by the ticket: **Rajesh Rao** (active predictive coding, root Rao & Ballard 1999) and
**Karl Friston** (active inference), following citations rather than keywords. Builds on
[`026-action-boundary-citations.md`](./026-action-boundary-citations.md) and
[`045-drives-citations.md`](./045-drives-citations.md); does not restate them.

Per [ADR-0029](../adr/0029-a-problem-is-minted-by-a-human-a-proposal-is-not.md) this pass **mints nothing and proposes
nothing**. It reports.

**Provenance tags.** Every claim carries one:
`[SOURCE]` — stated at the primary source, quoted from text I read.
`[INFERENCE]` — my derivation from quoted material; no source says it.
`[ABSENT]` — searched for and not found. A negative about this search, not about the world.

**Reading-depth key** (#148's convention, kept alongside the tags):
`[FULL]` — primary text read at source (PDF text extracted locally, or HTML fetched), quotes are
from that text. `[XTRACT]` — primary source fetched and quoted through an extraction pass over the
live page; quotes are the source's words but I did not hold the whole document. `[CITE]` —
bibliographic existence and claim confirmed only through a secondary description; **no quantitative
or load-bearing claim rests on one of these.**

**No proxy stands in for a measurement.** Where a source measures something adjacent to what it is
cited for, the gap is named in the sentence that cites it — #538's precedent on this repo is that a
proxy silently substituted for the pre-registered quantity closed a falsifier that had actually
fired.

---

## Headline

**Yes, and the literature says it far more strongly than the objection did.** In Friston's
formulation, action minimising sensory prediction error is not one exit among several — it is the
**only thing action can do at all**, stated in those words. The prediction error action discharges
is the *same object* perception discharges, and Friston et al. (2010) demonstrate action driven by
**visual** prediction error, not only proprioceptive, with vision and proprioception substituting
for one another. ADR-0003's sort does not sort: on the corpus's own terms every edge is a sensory
edge, and the "motor" ones are those whose error the agent is arranged not to explain away.

**But the corpus does supply a principled discriminator, and it is not write/read.** It is
**precision** — a continuous, learned, transiently-modulated gain, not a structural kind. Brown,
Adams, Parees, Edwards & Friston (2013) make it load-bearing: movement is impossible *unless* the
precision of sensory evidence is attenuated, because otherwise perception wins the race and the
error is explained away instead of acted on. Adams, Shipp & Friston (2013) supply the second half —
an anatomical, not computational, asymmetry: proprioceptive predictions terminate on classical
reflex arcs, which *fulfil* rather than *correct*; exteroceptive predictions have no such plant.

**And there is a write/read partition in the corpus — the Markov blanket — but it is the weakest
thing here, not the strongest.** Friston (2013) defines sensory and active states exactly by which
one the world writes and which one the agent writes. That is a real, principled, load-bearing
distinction and it is the closest formal object to ADR-0003's proposed replacement. It is also the
part of the framework under the heaviest live attack (Bruineberg et al.), and **Friston et al.
(2021b) concede in their own words that the partition is not unique**.

**Four things cut against the replacement, and they are listed as prominently as the support.**

1. **Rao & Ballard (1999), the root of the whole line, has no action variable at all.** Read at
   source: perceptual error there is cleared *only* by belief change. The two-exit structure is not
   native to predictive coding; it arrives with active inference.
2. **Rao's own Active Predictive Coding does not clear perceptual error by acting.** Its action
   networks are trained by **reinforcement learning for reward**; its state networks minimise
   prediction error. The discriminating quantity in Rao's line is **the objective**, and it is
   task-dependence versus task-invariance — not who clears anything.
3. **Friston's action variable `a` is genuinely outside the model.** "The generative model has no
   notion of action"; `∂s̃/∂a` is supplied by the **generative process** — the physics — not by the
   agent. That is a real principled asymmetry, and it is the closest thing in the corpus to "the
   world clears it." It attaches to a *variable*, not to a *channel*.
4. **Nothing in the corpus is written from outside and read by no one.** In every formalism
   examined, the externally-written object — the prior preference — is precisely *what the objective
   reads*, and it is defined over **observations or states**, i.e. on the sensory row, never over
   actions.

---

## 1. Ask one — is perceptual error cleared by acting, and in what terms?

### 1.1 Friston: it is the only thing action can do

`[SOURCE]` `[FULL]` — Friston, Daunizeau, Kilner & Kiebel (2010), *Action and behavior: a
free-energy formulation*, **Biol. Cybern.** 102(3), 227–260. PDF fetched from
`fil.ion.ucl.ac.uk` and text extracted locally, §3:

> "The ﬁnal equation describes action as a gradient descent on free-energy. The only way action can
> affect free-energy is through changing the motion of sensory signals. This means action must
> suppress sensory prediction errors, ε̃_s = s̃(a) − g(μ). Equation 9 embodies a nice convergence of
> action and perception: perception tries to suppress prediction error by adjusting expectations to
> furnish better predictions of signals, while action tries to fulﬁl these predictions by changing
> those signals."

And §3.1, sharper:

> "the only thing that action can affect is the prediction error at the sensory level. This means
> action can only suppress the weighted sensory prediction error variance, ε̃_sᵀ ξ_s = ε̃_sᵀ Π_s ε̃_s."

The same paper's Appendix 3 states the mechanism:

> "Because action can only affect the free-energy through the sensory data that are sampled, it can
> only affect sensory prediction error."

`[INFERENCE]` — this is stronger than the ticket's framing. B30 asked whether acting is *one* exit
from a sensory disagreement. In Friston it is the **only** thing acting is; there is no other
disagreement in the formalism for action to clear. `ε̃_s` is one object with one definition, and the
two gradient descents in Eq. 9 — `μ̇` and `ȧ` — descend the *same* term through different arguments.
ADR-0003's "who clears the disagreement" is therefore not a partition of edges in this source; it is
a partition of **arguments of one functional over one channel**.

### 1.2 It is not only proprioception. Vision drives action, and substitutes for proprioception

`[SOURCE]` `[FULL]` — same paper, §4.2, the two-joint arm with visual and proprioceptive modalities:

> "This is because action is driven by both proprioceptive and visual prediction errors (descending
> black arrows in Fig. 7). Although the proprioceptive errors are noisy, the visual errors are precise
> and can elaborate accurate predictions of hidden states … and precise motor commands."

> "The ensuing optimization boosts visual prediction errors relative to proprioception and ensures
> that the relative contribution of both modalities is balanced in a Bayes optimal fashion (for both
> action and perception)."

> "In terms of movement, it is clear that vision can substitute for proprioception and vice versa."

`[INFERENCE]` — this is the single most direct evidence against ADR-0003's sort. A *visual* channel
— unambiguously a sensory edge under `CONTEXT.md`'s definition, since the world writes the retina —
is here shown driving action, being weighted against proprioception on a common scale, and being
**interchangeable with it** for the purpose of moving. If a visual edge and a proprioceptive edge
are substitutable inputs to the same descent, then "cleared by the cell changing its belief" versus
"cleared by the world moving" is not reading off a property of the edge.

`[SOURCE]` `[FULL]` — same paper, §3.1, on the architecture this implies:

> "In this view, the central nervous system is not divided into motor and sensory systems but is one
> perceptual inference machine that provides predictions of optimal action, in terms of its expected
> consequences."

### 1.3 The two exits are named as two, explicitly, by the same authors

`[SOURCE]` `[XTRACT]` — Adams, Shipp & Friston (2013), *Predictions not commands: active inference
in the motor system*, **Brain Struct. Funct.** 218(3), 611–643, open access at
[PMC3637647](https://pmc.ncbi.nlm.nih.gov/articles/PMC3637647/). (026 could not reach this paper
and recorded it as *abstract-level only*; it is now **read at source** and that gap is closed.)

> "The brain can minimise prediction error in one of two ways. It can either change its predictions
> to better cohere with sensory input, or change the sampling of the environment such that sensory
> samples conform to predictions."

`[INFERENCE]` — B30's objection is this sentence. The source states the two-exit structure directly
and does **not** attach the two exits to two kinds of channel: the object with two exits is "the
prediction error", singular, and the second exit is described as changing "the sampling of the
environment", not as a different edge.

### 1.4 Perceptual sampling is itself an action, and its configuration is a hidden state

`[SOURCE]` `[XTRACT]` — Friston, Adams, Perrinet & Breakspear (2012), *Perceptions as hypotheses:
saccades as experiments*, **Front. Psychol.** 3:151:

> "the physical deployment of sensory epithelia is itself a hidden state of the world that has to be
> inferred. However, these hidden states can be changed by action."

> "This induces a dual minimization with respect to action and the internal states that parameterize
> the conditional density. These minimizations correspond to action and perception respectively."

`[SOURCE]` `[XTRACT]` — Rao (2024), *A sensory–motor theory of the neocortex*, **Nat. Neurosci.**
27, 1221–1235, open access at [PMC13452464](https://pmc.ncbi.nlm.nih.gov/articles/PMC13452464/),
*Active Visual Perception*:

> "This 'perceptual' stability is enabled by the model's ability to predict the expected glimpses for
> each planned 'eye movement'."

`[INFERENCE]` — both sides of the corpus therefore treat the *perceptual* channel's own
configuration as something the agent moves. The saccade case is not an edge case of perception; in
both Friston's and Rao's active-vision work it is the normal case, exactly as the ticket says.

### 1.5 Evidence against, part one: the root paper has no action at all

`[SOURCE]` `[FULL]` — Rao & Ballard (1999), *Predictive coding in the visual cortex*, **Nat.
Neurosci.** 2(1), 79–87. PDF located and text extracted locally; verified as the correct paper from
its opening text. A term-frequency check over the extracted text:

- `"eye movement"` — **0 occurrences.**
- `"action"` — 6 occurrences, **all of them fragments of other words** (`interactions`,
  `subtraction`). No occurrence of the standalone word.
- `"motor"` — **1 occurrence**, in the Discussion, as a biological aside about a different system:
  "the sensory prediction is generated using not only recent sensory inputs but also corollary
  discharge or proprioceptive signals associated with motor commands" — describing cerebellum-like
  structures in fishes, not the model.

`[SOURCE]` — **the paper contains no action or motor variable in its model.** Nothing the network
sets changes its own input; error is cleared by `r` and `U` updating, full stop.

`[INFERENCE]` — the root of the line the ticket names *is* the one-exit picture ADR-0003 attributes
to a sensory edge, and it is one-exit because it has no body. The second exit arrives only when
active inference bolts an actuator onto the same error. That matters for how B37 reads the taxonomy:
ADR-0003's sensory row is a faithful description of **predictive coding**, and it stops being
faithful at exactly the moment the architecture becomes **active** predictive coding — which
Patchworks is.

> **Measurement gap, named.** The term counts above are a measurement over *text extracted from the
> PDF I fetched*, not over the publisher's canonical text. A word lost to extraction would not show
> up. The claim is robust to that: a model variable would appear dozens of times, not once.

---

## 2. Ask two — does anything draw a principled perceptual/motor line, and what quantity carries it?

Three candidate quantities appear in the corpus. Two are real; none of the three is *who clears the
disagreement*.

### 2.1 Precision — the quantity that actually arbitrates, and it is a gain, not a kind

`[SOURCE]` `[XTRACT]` — Brown, Adams, Parees, Edwards & Friston (2013), *Active inference, sensory
attenuation and illusions*, **Cognitive Processing** 14(4), 411–427,
[PMC3824582](https://pmc.ncbi.nlm.nih.gov/articles/PMC3824582/):

> "self-generated movements require predictions to override the sensory evidence that one is not
> actually moving. However, ignoring sensory evidence means that externally generated sensations will
> not be perceived."

> "This conflict can be resolved by attenuating the precision of sensory evidence during movement or,
> equivalently, attending away from the consequences of self-made acts."

> "sensory attenuation is necessary if prior beliefs are to supervene over sensory evidence, during
> self-generated behaviour."

`[INFERENCE]` — **this is the direct answer to ask two, and it is bad news for an edge-typed sort.**
The source states that the two exits are in genuine *competition* over one error, and that the
agent has to actively suppress the perceptual exit for the motor one to win. So there is a
principled discriminator, and it is:

- **continuous** — a precision, `Π_s`, on a scale, not a binary edge kind;
- **transient** — attenuated *during movement* and restored after, so the same channel is "motor"
  at one moment and "sensory" at the next;
- **learned/optimised** — Friston (2010) §4.2 optimises `μ_γ^pro` and `μ_γ^vis` online;
- **shared across modalities** — the same quantity weights the visual and proprioceptive channels
  against each other.

An architecture that sorts *edges* into two kinds by a structural test is asking a binary of a
quantity the corpus makes continuous and time-varying. That is a real, named cost, and it is
independent of whether write/read is a better sort than clear-by.

### 2.2 The reflex arc — an anatomical asymmetry, in the periphery, not the computation

`[SOURCE]` `[XTRACT]` — Adams, Shipp & Friston (2013), *Active inference, predictive coding and
reflexes*:

> "proprioceptive predictions should not be *corrected* but *fulfilled*, by the automatic peripheral
> transformation of proprioceptive prediction errors into movement"

> "A proprioceptive prediction error can be generated at the level of the spinal cord by the
> comparison of proprioceptive predictions (from motor cortex) and proprioceptive input… The
> prediction error can then activate the motor neuron to contract the muscle"

> "In short, peripheral proprioceptive prediction errors are (or become) motor commands."

> "if sensory systems perform hierarchal perceptual inference, where descending signals are
> predictions of sensory inputs, then the functional anatomy of the motor system can be understood in
> exactly the same way"

`[SOURCE]` — the same paper on the descending signal:

> "Descending messages in the somatomotor system are therefore predictions of proprioceptive input
> and not motor commands."

`[INFERENCE]` — the distinction this source draws is **anatomical and peripheral, and explicitly not
computational**: motor cortex is claimed to work "in exactly the same way" as sensory cortex. What
makes a proprioceptive prediction into a movement is that a *reflex arc happens to be wired to it*.
That is genuinely close in spirit to ADR-0003 — the clearing agency is downstream of the cell, and
is not a property of the cell — and it is the strongest support in the corpus for ADR-0003's core
intuition.

It is also narrower than ADR-0003 in a way B37 should not gloss: the asymmetry is between
**proprioception** and everything else, not between *sensory* and *motor*. Both rows in ADR-0003's
table are sensory channels in this source; the motor row is the sub-case with a plant attached.

> **Proxy warning.** One line in my extraction pass on this paper — "Visual prediction errors resolve
> through perception; proprioceptive errors resolve through action" — was the **extractor's gloss,
> not a quotation**, and is not treated as a source claim anywhere above. Sentences I could verify as
> the paper's own words are the ones quoted. I searched the paper for "attenuat" and "sensory
> attenuation" and found **zero occurrences** `[ABSENT]` — the attenuation argument lives in Brown et
> al. (2013), §2.1, not here.

### 2.3 The generative process, not the model — Friston's own version of "the world clears it"

`[SOURCE]` `[FULL]` — Friston et al. (2010), §3 and Appendix 3:

> "the generative model has no notion of action; it just produces predictions that action tries to
> fulﬁl"

> "The partial derivative of the error with respect to action is the partial derivative of the
> sensory data with respect to action and is speciﬁed by the generative process."

`[INFERENCE]` — **this is the corpus's genuine principled discriminator, and ADR-0003 has a version
of it.** The agent's model contains no term for `a`; the map from action to sensory consequence is
owned by the physics. In Patchworks' vocabulary: the far endpoint executes rather than restricting a
belief back. That is not nothing — it is real, first-party, and it is exactly ADR-0003's claim.

But the object it attaches to is a **variable `a`**, not a channel. Friston has one sensory channel
`s̃` and one extra variable outside the model; Patchworks has two rows of *edges*. The translation
from one to the other is not made by any source I found `[ABSENT]`, and it is where ADR-0003's
strongest support becomes an analogy rather than a citation.

### 2.4 Evidence against: in Rao's line the discriminator is the objective, not the clearing

`[SOURCE]` `[XTRACT]` — Rao (2024), *Nature Neuroscience*, Box 1 and *Active Visual Perception*:

> "The state-transition function f^s models the physics of the environment and the agent"

> a state-to-action "policy"… "which maps the current estimated state directly to an action to achieve
> the current goal"

> "State networks… were trained to minimize image-prediction errors"

> "action networks were trained using reinforcement learning for the task of image reconstruction"

`[SOURCE]` `[XTRACT]` — Rao, Gklezakos & Sathish (2022/2024), arXiv:2210.13461, abstract, quoted in
full elsewhere in 026 and re-verified here:

> "Our approach exploits hypernetworks, self-supervised learning and reinforcement learning to learn
> hierarchical world models that combine task-invariant state transition networks and task-dependent
> policy networks at multiple abstraction levels."

`[SOURCE]` — no principled criterion for what counts as a sensory input versus an action output
appears in Rao (2024); the split is stipulated architecturally and then given two different training
objectives `[ABSENT]` for the criterion.

`[INFERENCE]` — this is the second major cut against the ticket's hypothesis, and it cuts against
ADR-0003 too, in the same place 026's headline verdict already found. In Rao's APC:

- **actions are not selected to minimise prediction error.** They are selected by RL against a
  reward. Perceptual error is *not* cleared by acting in Rao's model, because the action arm is not
  descending the perceptual error at all.
- **the discriminating quantity is `task-invariant` versus `task-dependent`.** That is a statement
  about what the weights must generalise over, not about who clears a disagreement, and it has no
  translation into ADR-0003's taxonomy.

So the two halves of the ticket's own corpus disagree about ask one. Friston says perceptual error
*is* cleared by acting and that this is all action is. Rao says action is a reward-maximising policy
network beside a prediction-error-minimising state network. **ADR-0003's identity is Friston's, not
Rao's** — which is the finding 026 already recorded as its headline verdict, arrived at again from a
different direction and now with Rao's *training objectives* rather than only his *architecture* as
the evidence.

---

## 3. Ask three — does any source make write/read the discriminator?

**Yes. One does, it is central, and it is the single strongest piece of support for the
replacement — and it is also the most contested object in the corpus.**

### 3.1 The Markov blanket partition is a write/read partition, in those terms

`[SOURCE]` `[FULL]` — Friston (2013), *Life as we know it*, **J. R. Soc. Interface** 10:20130475.
PDF fetched from `fil.ion.ucl.ac.uk`, text extracted locally, §1:

> "the Markov blanket can itself be partitioned into two sets that are, and are not, children of
> external states. We will refer to these as a surface or sensory states and active states,
> respectively."

> "External states cause sensory states that inﬂuence—but are not inﬂuenced by—internal states,
> while internal states cause active states that inﬂuence—but are not inﬂuenced by—external states"

And Table 1, verbatim:

> "external states Ψ : Ψ × A × Ω → ℝ states of the world that cause sensory states and depend on
> action"
> "sensory states S : Ψ × A × Ω → ℝ the agent's sensations that constitute a probabilistic mapping
> from action and external states"
> "action states A : S × λ × Ω → ℝ an agent's action that depends on its sensory and internal states"

And §2:

> "because active states change—but are not changed by—hidden states … they will appear to place an
> upper (free energy) bound on the dispersion (entropy) of biological states"

`[INFERENCE]` — read against ADR-0003's table, the correspondence is close and it is genuinely a
write/read sort:

| | ADR-0003 sensory edge | Friston sensory state | ADR-0003 motor edge | Friston active state |
|---|---|---|---|---|
| written by | the world | external states (it is their child) | the cell | internal states |
| read by | the cell | internal states | the world | external states |

**So the answer to ask three is yes.** The one fully formal partition of the agent's boundary in the
corpus is drawn by *who causes the state and who it causes*, and not by who clears an error. Nothing
in Friston (2013) mentions error clearing in the definition at all.

### 3.2 The same source undercuts the sensory row of that partition

`[SOURCE]` `[FULL]` — Friston (2013), Table 1, again — sensory states are "a probabilistic mapping
from **action** and external states", and external states "depend on action".

`[INFERENCE]` — this is **B30's objection stated by the source itself.** In Friston's own type
signature, what the world writes onto the sensory channel is a function of what the agent did. So a
write/read sort does not escape the problem it was proposed to fix: the "world writes it" row is
already downstream of the agent's action, at the level of the definition. A sort by *who writes* is
cleaner than a sort by *who clears* only if "who writes" is itself clean, and in this source it is
not — sensory states have two parents, one of them the agent.

### 3.3 Evidence against: the partition is not unique, and its authors say so

`[SOURCE]` `[FULL]` — Bruineberg, Dołęga, Dewhurst & Baltieri, *The Emperor's New Markov Blankets*,
**Behavioral and Brain Sciences** (accepted version fetched from the corresponding author's site,
text extracted locally). Their distinction, verbatim from the short abstract:

> "We propose a distinction between instrumental Pearl blankets and realist Friston blankets. Pearl
> blankets are substantiated by the empirical literature but can do limited philosophical work.
> Friston blankets can do philosophical work, but require strong theoretical assumptions."

> "Pearl blankets exist inside of models and cannot by themselves settle questions about the
> boundaries between agents and their environments."

> "This formalizing step requires a number of non-arbitrary assumptions… For example, it is unclear
> why only electrochemical interactions are used to construct the adjacency matrix while other forms
> of inﬂuence included in the simulation (such as Newtonian forces) are ignored. If different
> thresholds were used to determine whether two nodes are connected, the adjacency matrix would look
> very different."

> "This defense of Friston blankets… faces a serious obstacle by assuming that free-energy minimizing
> systems can be identiﬁed without the help of the assumptions behind the Friston blanket construct,
> **such as the existence of unambiguously active or passive states.**"

And, decisively, **Friston et al. (2021b) quoted inside that paper** (`[CITE]` for the original —
I read this passage in Bruineberg et al.'s quotation of it, not in Friston et al. 2021b itself):

> "The nonuniqueness of the particular partition is a key practical issue. There is no pretense that
> there is any unique particular partition. There are a vast number of particular partitions for any
> given coupled dynamical system. In other words, by simply starting with different internal states
> —or indeed the number of internal states per particle—we would get a different particular
> partition."

`[INFERENCE]` — the corpus's one write/read discriminator is conceded by its own authors to be
**non-unique**, and it is attacked by a published critique precisely on the assumption ADR-0003
would need to import: *that there are unambiguously active or passive states*. Adopting write/read
does not buy a discriminator the literature regards as settled; it buys one the literature is
actively arguing about.

`[INFERENCE]` — one further mismatch, and it is structural rather than rhetorical. Friston's
partition is defined **over the whole boundary at once**, as a global conditional-independence
property of a coupled dynamical system. ADR-0003's is defined **per edge**, locally. Nothing in
Friston (2013) licenses reading the blanket edge-by-edge; the whole content of the construct is that
the *set* of blanket states shields the *set* of internal ones. Patchworks' locality doctrine (#181,
per edge never per level) points the opposite way from the object it would be borrowing. No source I
found makes a per-edge version of the partition `[ABSENT]`.

---

## 4. Ask four — a channel written from outside and read by no one

**`[ABSENT]`, and the absence has a structural reason that is worth more than the search result.**

### 4.1 What the corpus's drives actually are, and where they sit

`[SOURCE]` `[FULL]` — Da Costa, Parr, Sajid, Veselic, Neacsu & Friston (2020), *Active inference on
discrete state-spaces: a synthesis*, arXiv:2001.07203. PDF fetched, text extracted locally.

Figure 2 caption:

> "Note that the agent's preferences may be speciﬁed either in terms of states or outcomes."

Appendix C.2, the definition itself:

> "Where the vector C ∈ ℝᵐ encodes preference over states P(s_τ) = Cat(C). However, it is also
> possible to approximate this risk term over states by a risk term over outcomes (c.f., (15)), as is
> currently implemented in `spm_MDP_VB_X.m`. In this case, if C ∈ ℝⁿ denotes the preferences over
> outcomes P(o_τ) = Cat(C)"

And §2:

> "a preference is simply something an agent (believes it) is likely to work towards."

§9, on where preferences come from:

> "agents are equipped (e.g., born) with an innate generative model that entails fundamental
> preferences (e.g., essential to survival), which are not updated… the parameters of these innate
> prior distributions… have hyperpriors that are inﬁnitely precise (e.g., a Dirac delta distribution)
> and thus cannot be updated in an experience dependent fashion."

`[SOURCE]` `[FULL]` — Torresan, Kanai & Baltieri (2025), *Prior preferences in active inference
agents: soft, hard, and goal shaping*, arXiv:2512.03293. PDF fetched, text extracted locally, §2.4:

> "the agent's preferences are represented by a categorical distribution, indicated by P*(S) in the
> fully observable case, that effectively encodes an agent's goals in terms of particular
> instantiations of the random variable S."

> "the ﬁxed preference distribution indicates what the most probable state *should* be (regardless of
> any policy)."

> "This distribution needs to be speciﬁed in advance since in general active inference agents are not
> able to learn what they should do"

And the paper's own framing of the gap, from its abstract:

> "In the literature, the questions of how the preference distribution should be speciﬁed and of how
> a certain speciﬁcation impacts inference and learning in an active inference agent have been given
> hardly any attention."

`[SOURCE]` `[CITE]` — Pezzulo, Rigoli & Friston (2015), *Active Inference, homeostatic regulation and
adaptive behavioural control*, **Prog. Neurobiol.** 134, 17–35. **Not read at source** — PubMed
returned a cookie wall and ScienceDirect returned HTTP 403; characterised here from
search-engine-indexed descriptions of the abstract only. On that basis: the paper generalises active
inference by emphasising interoceptive inference and homeostatic regulation, and explains how priors
act as drives or goals to enslave action, with autonomic control generating interoceptive
predictions — homeostatic setpoints — fulfilled through autonomic reflexes as motor control fulfils
proprioceptive predictions. **No claim below rests on this source alone.**

### 4.2 What that means for the drive's shape

`[INFERENCE]` — every externally-written motivational object I found in the corpus has **three**
properties, and the drive as ADR-0009 defines it shares only the first:

1. **written from outside** — yes, and emphatically: Da Costa's innate preferences carry infinitely
   precise hyperpriors so they *cannot* be updated; Torresan et al. say the distribution "needs to be
   specified in advance". ADR-0009's drive cell matches.
2. **read, and read constantly** — the preference distribution is the second argument of the risk
   term of expected free energy; it is read on **every policy evaluation**. There is no unread
   channel anywhere in the formalism.
3. **defined over observations or states** — never over actions. Da Costa's `C` is over states or
   outcomes, and the standard SPM implementation puts it over **outcomes**. Torresan et al.'s `P*`
   is over states or observations.

`[INFERENCE]` — property 3 is the finding that bears hardest on ADR-0009. **The corpus's drive rides
the sensory row.** A prior preference is a belief about what the agent expects to *observe*; it is
compared against a predicted observation; it never lives on an action channel. ADR-0009 places the
drive edge in the motor row by ADR-0003's clear-by test. On the write/read test that B37 is
considering as a replacement, and on the corpus's own placement of the same object, the drive lands
in the **sensory** row instead — written from outside, exactly like a sensory boundary cell.

`[INFERENCE]` — and property 2 is where the ticket's phrase "read by no one" needs unpacking before
it can be compared to anything. `CONTEXT.md`'s drive boundary cell "is read by nothing" means *read
by nothing outside the sheaf* — it is very much read by the apex cells its drive edges reach.
Restated that way, the drive is: **written from outside, read from inside, never read by the world.**
That is the type signature of a **sensory state** in Friston (2013) — a child of external states
whose only consumer is internal — with one difference: it attaches at the apex rather than the rim.
No source in this corpus places an externally-written channel anywhere but at the sensory surface
`[ABSENT]`, and no source treats attachment depth as changing a channel's kind `[ABSENT]`.

`[INFERENCE]` — the structural reason the search comes back empty. In Friston (2013) the blanket
*is* the set of states through which internal and external states influence one another. A state
nothing reads is, by that definition, **not a blanket state at all** — it is external or internal.
So "written from outside and read by no one" is not a shape the formalism has an empty slot for; it
is a shape the formalism's own construction excludes. Whether that is a defect in the formalism or
in the drive is not a question this pass can answer, and per ADR-0029 it does not try.

### 4.3 Evidence against the drive's current placement is stronger than evidence for it

`[INFERENCE]` — stating this plainly because the ticket asks for the against-case as prominently as
the for-case, and here the against-case is the larger half:

- **For the motor placement:** Da Costa's preferences do only get discharged by acting, and
  Pezzulo et al. (per §4.1's `[CITE]`) describe setpoints "fulfilled" by reflexes, which is the same
  verb Adams et al. use for the motor row. So "cleared by the world moving, eventually" has real
  corpus support.
- **Against it:** the object is defined over the observation space in every formalism examined; it
  is read on every inference step; the discharge happens through the *sensory* comparison, not
  through an action channel; and the sensory/motor question is settled in the corpus by *precision*
  and by *whether a reflex arc is attached*, neither of which a drive edge has.

---

## 5. What this pass did not reach

- **Pezzulo, Rigoli & Friston (2015)** — not read at source (§4.1). PubMed cookie wall, ScienceDirect
  403. `[CITE]` only.
- **Parr & Friston, *Generalised free energy and active inference*** — located but not read; PMC
  returned a reCAPTCHA page on two attempts. A search-engine summary indicates it distinguishes
  expected free energy (where priors over outcomes sit outside the generative model) from generalised
  free energy (where they are an explicit component of it), which is directly on ask four's
  read/unread axis. **This is the most valuable unclosed gap in the pass** and it is left open rather
  than papered over. `[ABSENT]` for anything I might have drawn from it.
- **Friston et al. (2021b)**, the source of the non-uniqueness concession in §3.3, was read only
  through Bruineberg et al.'s quotation of it.
- **Parr, Pezzulo & Friston (2022)**, *Active Inference* (MIT Press) — not read, as in 026.
- **Rao, Gklezakos & Sathish (2022/2024) full text** — read at source by 026 and re-verified here
  only at abstract level plus Rao (2024)'s restatement. The training-objective quotes in §2.4 come
  from **Rao (2024)**, which is open access, not from the 2022 arXiv paper.
- **No source was found that states an alternative discriminator to precision / reflex-arc /
  Markov-blanket.** `[ABSENT]`. Three is what the corpus has.

## 6. Sources

- Rao, R.P.N., Ballard, D.H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*
  2(1), 79–87. `[FULL]`
- Friston, K., Daunizeau, J., Kilner, J., Kiebel, S.J. (2010). Action and behavior: a free-energy
  formulation. *Biological Cybernetics* 102(3), 227–260.
  https://www.fil.ion.ucl.ac.uk/~karl/Action%20and%20behavior%20A%20free-energy%20formulation.pdf
  `[FULL]`
- Friston, K., Adams, R.A., Perrinet, L., Breakspear, M. (2012). Perceptions as hypotheses: saccades
  as experiments. *Frontiers in Psychology* 3:151. `[XTRACT]`
- Friston, K. (2013). Life as we know it. *J. R. Soc. Interface* 10:20130475.
  https://www.fil.ion.ucl.ac.uk/~karl/Life%20as%20we%20know%20it.pdf `[FULL]`
- Adams, R.A., Shipp, S., Friston, K.J. (2013). Predictions not commands: active inference in the
  motor system. *Brain Structure and Function* 218(3), 611–643.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3637647/ `[XTRACT]` — **gap in 026 now closed.**
- Brown, H., Adams, R.A., Parees, I., Edwards, M., Friston, K. (2013). Active inference, sensory
  attenuation and illusions. *Cognitive Processing* 14(4), 411–427.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3824582/ `[XTRACT]`
- Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., Friston, K. (2020). Active inference on
  discrete state-spaces: a synthesis. arXiv:2001.07203 `[FULL]`
- Rao, R.P.N., Gklezakos, D.C., Sathish, V. (2022/2024). Active Predictive Coding. arXiv:2210.13461;
  *Neural Computation* 36(1). `[XTRACT, abstract]`
- Rao, R.P.N. (2024). A sensory–motor theory of the neocortex. *Nature Neuroscience* 27, 1221–1235.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC13452464/ `[XTRACT]`
- Bruineberg, J., Dołęga, K., Dewhurst, J., Baltieri, M. (2022). The Emperor's New Markov Blankets.
  *Behavioral and Brain Sciences*.
  https://manuelbaltieri.com/assets/pdf/EmperorMarkovBlanketsAccepted.pdf `[FULL]`
- Torresan, F., Kanai, R., Baltieri, M. (2025). Prior preferences in active inference agents: soft,
  hard, and goal shaping. arXiv:2512.03293 `[FULL]`
- Pezzulo, G., Rigoli, F., Friston, K. (2015). Active Inference, homeostatic regulation and adaptive
  behavioural control. *Progress in Neurobiology* 134, 17–35. `[CITE]` — **not read at source.**
- Friston, K. et al. (2021b), quoted via Bruineberg et al. `[CITE]`

---

## Decision provenance

Scoping calls I made myself, that nobody instructed. Listed so B37 can reverse any of them.

1. **I treated the ticket's "perceptual channel" as meaning *exteroceptive*, and made the
   visual-drives-action result (§1.2) the load-bearing evidence for ask one.** The alternative
   reading — perceptual meaning *any* channel the world writes, proprioception included — would make
   ask one trivially yes and uninteresting. I chose the reading that could actually fail.
2. **I read Rao & Ballard (1999) for an absence rather than a presence,** and spent a fetch and a
   local extraction establishing a negative. Nobody asked for that. I judged that "the root of the
   line has no action variable" reframes the whole taxonomy question, and it did.
3. **I promoted `precision` to the answer for ask two** over the reflex-arc account, which is the
   more obvious candidate and the one closer to ADR-0003. Both are reported; the ordering is my
   judgement, on the grounds that the reflex arc explains *which* channel moves a plant while
   precision explains *which exit wins*, and ask two asks about the exits.
4. **I went outside the two named authors for §3.3**, to Bruineberg et al. and through them to
   Friston et al. (2021b). The ticket says follow citations beyond Rao and Friston, so this is
   licensed, but the choice to make a *critique* rather than a primary source the counterweight on
   ask three is mine.
5. **I fetched Torresan, Kanai & Baltieri (2025)** — a paper nine months old at the time of writing,
   outside both named lines — because it is the only source found that treats the specification of
   the preference distribution as a question in its own right, which is ask four's subject. Its own
   abstract says the topic has "hardly any attention", which is corroboration of the `[ABSENT]`
   rather than a substitute for it.
6. **I restated "read by no one" as "never read by the world"** in §4.2 before comparing it to
   anything. `CONTEXT.md` supports that reading (the drive cell is read by the apex cells its edges
   reach), but the ticket's phrasing is looser and a different restatement would change what §4
   concludes. This is the single scoping call most likely to be worth reversing.
7. **I did not attempt to arbitrate between Friston's answer and Rao's.** §2.4 reports that the
   ticket's own corpus splits on ask one and leaves it split. Per ADR-0029 this pass proposes
   nothing, and choosing between them is B37's decision, not a research finding.
8. **I closed 026's Adams/Shipp/Friston gap opportunistically** rather than treating it as out of
   scope. It changed §2.2 from an abstract-level correspondence to a quoted one.
