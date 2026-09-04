# Citation pass: training on a corpus versus training on conversation (patchworks#446)

Opened by [#446](https://github.com/NGL321/patchworks/issues/446), part of map
[#127](https://github.com/NGL321/patchworks/issues/127). The position examined is
[`12-the-interlocutor.md`](../spec/12-the-interlocutor.md)'s, *Conversational by construction*, which
records [#129](https://github.com/NGL321/patchworks/issues/129) and rests on
[ADR-0003](../adr/0003-action-is-prediction-the-world-clears.md),
[ADR-0009](../adr/0009-a-drive-is-a-motor-edge-attached-deep.md) and
[ADR-0025](../adr/0025-coherence-is-a-motor-readback-not-a-sensory-value.md).

**Citation sequencing.** The design came first and is not on trial. #128 generalised efference copy to
a language rim and said *turn-taking is the body's refusal on a language rim* before the mistake it
guards against was available to make; #129 wrote the world down; ADR-0025 sited the coherence number
on the motor rim; #130 and #169 sized and typed the spoken stalk. This pass validates afterwards and
**seeds nothing**. Where a source agrees the document says *corroborates*; where it disagrees it says
so plainly and hands the disagreement to a ticket rather than acting on it. Nothing here edits the
spec, an ADR, or a register.

**Registers consulted:** open-problems — three rows match and one is the pass's direct subject.
[#331](https://github.com/NGL321/patchworks/issues/331) (*"The language domain has no lower bound on
its own solvability … `03` bounds its world with a scripted controller solving 14 of 72 — dumb,
privileged, and *lower*-bounding … Nothing in the language domain corresponds"*) is the live match,
and §7's second handoff is a candidate instrument for exactly it, found in the literature rather than
invented here; [#388](https://github.com/NGL321/patchworks/issues/388) (*"The wedge's rim supplies the
language domain's memory as a store, and the map's position is that memory is sustained"*) is adjacent
and is **touched** by §3, because a rim that carries an uptake flag per slot is carrying a record of
refusals as well as of characters; [#330](https://github.com/NGL321/patchworks/issues/330) (long-range
agreement forced into the core) is adjacent and untouched. proposed-solutions — **nothing matches**.
No shelf row concerns the interlocutor, the rims, or the readback; the nearest by shape,
[#314](https://github.com/NGL321/patchworks/issues/314)'s hippocampal analog, answers #388 on memory
and not on the loop. dismissed-solutions — the register holds one row,
[#346](https://github.com/NGL321/patchworks/issues/346), a cross-edge coherence term killed by
`detectability`; **nothing in this document re-proposes it**, and no idea surfaced here was previously
foreclosed. The three constants registers (`architecture.md`, `rig.md`, `world-and-build.md`) are
entirely the sandbox's; the language domain has no constants in them yet, which `architecture.md`'s
own *Stated gaps* predicts (*"a `LanguageSpec` for the second domain would arrive unseen"*). Nothing
surfaced here was foreclosed.

## Reading-depth key

#148's key, used on every source.

- **[FULL]** — paper body read (PDF text or HTML extracted).
- **[ABS]** — authoritative abstract / landing page only.
- **[CITE]** — citation confirmed to exist, text not reached.
- **[UNREACHED]** — existence not confirmed.

**Two extraction notes, because they bear on trust.** First, every fetch in this pass was rendered to
markdown and read through a summarising step, so a passage marked as a quotation is a quotation *as
returned by that step*. Second, and specific to this pass: **PDF extraction failed almost universally
in this environment** — arXiv PDFs, the PNAS mirror, the ACL Anthology PDFs and two university-hosted
copies all returned encrypted or binary streams. That is why several sources that are classics of
their fields sit at **[CITE]** rather than [ABS] or [FULL]. Where a source is [CITE], this document
**does not quote it** and does not rest a claim on its wording. The one route that worked reliably for
paywalled journal text was a full-text academic gateway, which returned verbatim body chunks for
Roseberry et al. and Myers et al.; those two are marked accordingly.

---

## Headline verdict, stated plainly

**The distinction is partially documented, and the conjunction this map draws is not documented at
all.** Two of its five limbs are established outside machine learning — formally in computational
learning theory, empirically and with the exact ablation in child development. Three limbs are absent
as this map states them. And the binding constraint that makes the map's version distinctive is not a
gap in coverage but a difference in kind: **every closed-loop language learner found expresses the
loop through a reward or a label.** This map has neither.

Five things follow, and they are separable.

1. **That an interactive learner can learn what a passive one cannot is not a conjecture. It is a
   theorem, and it is nearly forty years old.** Angluin's `L*` learns any regular set in time
   polynomial in the size of the minimum DFA from a teacher answering membership and equivalence
   queries, where finding a minimum DFA consistent with a *given* finite sample is NP-hard. Same
   target class, same alphabet; the difference is that the learner's own output determines what
   arrives next. **The map's limb 1 is a documented result and the spec can cite it.**
2. **The exact ablation the map's position implies has been run — on toddlers, not on machines, and it
   came out the map's way.** Roseberry, Hirsh-Pasek & Golinkoff put the *same* teaching content in
   front of 24–30-month-olds three ways: live, over contingent video chat, and as **yoked
   non-contingent video**. *"Results suggest that children only learned novel verbs in socially
   contingent interactions."* Their discussion is sharper still and is the sentence this ticket asked
   for: *"simply posing questions to children and pausing for the answer did not result in language
   learning if the children were not able to interact contingently with the person on video."* **Turn
   structure without the loop taught nothing.** That is limb 1 and limb 5 in one experiment.
3. **The motor half has no precedent in any learning system read here, and the reason is uniform.**
   Psycholinguistics and speech motor control own the readback — Levelt's perceptual loop, Pickering
   & Garrod's forward models, state feedback control, DIVA. But in every *learning system* found,
   what the world made of the utterance enters as a **control error, a reconstruction loss, or a
   reward**: Warlaumont & Finnegan's babbling model converts the salience of the infant's own
   vocalisation into **dopamine**; Georges et al.'s vocal-imitation agent uses its own produced sound
   as a self-supervised training target; Stöpler et al.'s interactive LM *"receives a reward if
   communicative success is achieved."* **No source read here supplies the world's reading of the
   agent's own utterance as an observation the learner must predict.** That is the pass's sharpest
   empty result and it is the limb ADR-0025 exists to protect.
4. **Turn-taking is modelled everywhere and is never an observation.** The full-duplex spoken-LM line
   is real, recent and close: LSLM consumes its own speaking channel and the listening channel
   simultaneously, and DuplexPO learns turn initiation, backchanneling and yielding. But LSLM's
   interruption signal is a token the model **emits** as a supervised target, and DuplexPO's is a
   **reward** — *"the Factorized Conversational Dynamics Reward (FCDR) to enable fine-grained temporal
   credit assignment for turn initiation, backchanneling, yielding."* Nobody hands the learner *"you
   were not taken up"* as a value on its own motor rim.
5. **The field is arriving at the question, and it arrived in 2025.** BabyLM's third challenge added
   an **Interaction** track — *"This new track encourages interactive behavior, learning from a
   teacher, and adapting the teaching material to the student."* Its rules concern token budgets and
   what the teacher may reveal; **there is no readback in them, and no motor rim.** The map's "why
   now" argument survives contact with the record: the question is one year old in this field, which
   is consistent with the map's claim that a fast local partner is what made it affordable, and it
   means "nobody has written this down" is unsurprising rather than suspicious.

**What this does not do.** It does not settle whether the domain is solvable — #331 stands untouched
as a problem, though §7 hands it a candidate instrument. It does not price the drive against a reward;
that substitution remains stipulated, and §7 hands the gap to a ticket. And it does not reopen
ADR-0025, which nothing found contradicts.

---

## 1. Limb 1 — the closed loop: the learner's output changes what arrives next

### 1.1 The formal version is a theorem, and it is the sharpest citation in the pass

**Angluin, "Learning Regular Sets from Queries and Counterexamples", *Information and Computation*
75(2):87–106, 1987.** [CITE — PDF would not extract from three mirrors; see §8]

**Angluin, "Queries and Concept Learning", *Machine Learning* 2:319–342, 1988.** [CITE]

The 1987 result is that `L*` learns any regular set, in time polynomial in the number of states of the
minimum DFA and the maximum counterexample length, from a *minimally adequate teacher* answering
membership queries and equivalence queries. The comparison class is Gold's: finding a minimum-state
DFA consistent with a **given** finite set of positive and negative examples is NP-hard. The 1988
paper generalises the setting across six query types and compares equivalence queries directly against
PAC identification under random sampling.

**Why this is the citation and not merely an analogue.** The two settings differ in exactly the axis
the map names. Same target class, same alphabet, same information in principle; what changes is
whether the learner's emission determines the next item. **The map's claim that the stream is a closed
loop rather than a fixed sequence is, for formal languages, a proved separation rather than a
position.** `12-the-interlocutor.md`'s sentence — *"it is not a property of the data, it is a property
of what is on the other end, and it cannot be recovered later by a bigger dataset"* — is the
learning-theoretic statement in the spec's own voice, and the theory agrees with it, including the
"bigger dataset" clause: the passive problem does not become tractable with more samples, it becomes
NP-hard on them.

**The honest limits, and they matter.** The teacher is *minimally adequate*: it answers a query with a
correct label and returns a counterexample. That is a **supervision channel**, not an interlocutor
with an agenda, and it is not what this map has. The separation therefore corroborates limb 1's
*shape* — output determines input, and this changes what is learnable — while using a channel the map
rules out. This is recorded as corroboration with a named gap, not as a licence.

### 1.2 The interactive-learning line in NLP has the loop and pays for it with a label

**Wang, Liang & Manning, "Learning Language Games through Interaction", ACL 2016, pp. 2368–2378,
[arXiv:1606.02447](https://arxiv.org/abs/1606.02447).** [FULL, via ar5iv]

The setting is Wittgenstein's: *"The computer initially knows nothing about language and therefore
must learn it from scratch through interaction, while the human adapts to the computer's
capabilities."* Both halves of the map's loop are present and named — the learner's emission changes
the human's next utterance, and the paper measures it: *"through this interaction, many players adapt
to the computer by becoming more consistent, more precise, and more concise."*

**And the learning signal is a label.** *"The human then chooses `yᵢ` from the list `Y` (we say the
computer is correct if `i=1`)."* The human selects the intended successor state from a ranked list.
That is supervision on each turn, in a task-completion setting, with a correctness criterion. **The
closed loop is here; the map's version of what closes it is not.**

### 1.3 Learning from the interlocutor's *reaction* rather than from a label — the nearest miss

**Chen, Gul, Chen, Geng, Wu & Artzi, "Retrospective Learning from Interactions",
[arXiv:2410.13852](https://arxiv.org/abs/2410.13852) (2024, rev. 2025).** [ABS]

The premise is the map's: *"If an LLM responds in an unexpected way to an instruction, the user is
likely to signal it by rephrasing the request, expressing frustration, or pivoting to an alternative
task."* The signal is the interlocutor's *next turn*, read as feedback on the model's own last turn,
and it is obtained *"without any external annotation"*. Task completion moved from 31% to 82%.

**Why it is a miss and not a hit.** The signal is decoded from the interlocutor's text and then used
as a training target — it is a reward extracted from the stream rather than a component of the rim.
The map's readback is not decoded from what the partner says; it is what the world writes back onto
the motor cell in the same tick. Recorded as an influence, not as support.

### 1.4 Where the loop is closed by reward, which is most of the field

**Emergent communication and Lewis signalling games.** [ABS across the line] The canonical setup —
a speaker observes a state, sends a signal, a listener acts, and *"both agents equally rewarded based
on the outcome"* — is a closed loop by construction and a reward channel by construction. The known
pathology is also on the record: agents trained this way *"tend to develop languages that display
undesirable properties from a linguistic point of view (lack of generalization, lack of
compositionality)"*, and *"without careful environmental pressures, agents develop successful but
uninterpretable communication schemes distinctly unlike human language."* Two things follow. The
literature's loop is not this map's loop. And **the failure mode the reward version has is a
degenerate protocol that scores well and is not language**, which is structurally the same worry
`11-the-language-graph.md` records under *Where the columns merge* as **echo** — a cheap way to score
well on the interlocutor's surprisal. The wedge answers it by construction (six hops); the emergent-
communication line answers it with environmental pressures. Corroborates the worry from outside; the
remedies do not transfer.

**Dialogue as a POMDP.** Young, Gašić, Thomson & Williams, "POMDP-Based Statistical Spoken Dialog
Systems: A Review", *Proceedings of the IEEE* 101(5):1160–1179, 2013,
[doi:10.1109/JPROC.2012.2225812](https://doi.org/10.1109/JPROC.2012.2225812). [CITE] The framing is
the closed loop stated as control — a belief state over what the user wants, updated by what the user
says in response to what the system said — and the policy is *"optimized via a reward-driven
process."* **This is the field's own name for "dialogue is a control loop", and the object it hands
the learner is a reward.** Where the map wants a drive, this literature has only the reward form.

**Towards developmentally plausible rewards.** Stöpler, Asadli, Nikolaus, Cotterell & Warstadt,
[arXiv:2505.05970](https://arxiv.org/abs/2505.05970) (2025). [ABS] This is the closest published work
in intent to what #446 asks about: train a language model interactively, motivated by child language
acquisition, with the signal coming from what the partner made of the utterance. *"A speaker attempts
to communicate some information to a listener in a single-turn dialogue and receives a reward if
communicative success is achieved."* Single-turn, RL, reward. The authors report that *"linguistic
evaluation improvements remain elusive"*.

**This is the pass's clearest statement about the shape of the gap, and it should be read carefully
rather than triumphantly.** The one group that set out to do developmentally-motivated interactive
language-model training reached for a reward, and it did not obviously work. That is not evidence for
the map's alternative; it is evidence that the obvious version has been tried and that the map's
version has not.

### 1.5 Limb 1 verdict

**Documented, and documented twice — once as a theorem, once as an experiment — and never in the
map's form.** The claim *"the learner's own output changes what arrives next, and this changes what is
learnable"* is established. The claim *"and it does so with no reward and no label"* is found nowhere.

---

## 2. Limb 2 — the motor consequence of speaking, carried on the rim

This is the limb `12-the-interlocutor.md` and ADR-0025 are most exposed on, and it is where the pass
returns the least.

### 2.1 The mechanism is owned by psycholinguistics and motor control, and it is well owned

**Levelt, "Monitoring and self-repair in speech", *Cognition* 14(1):41–104, 1983.** [CITE] The
perceptual loop: a speaker monitors their own speech through the comprehension system, at two levels —
the preverbal message and the articulatory buffer — and interrupts promptly on detecting trouble. The
analysis rests on 959 spontaneous self-repairs.

**Pickering & Garrod, "An integrated theory of language production and comprehension", *Behavioral and
Brain Sciences* 36(4):329–347, 2013.** [ABS] *"In rejecting this dichotomy, we instead assert that
producing and understanding are interwoven, and that this interweaving is what enables people to
predict themselves and each other."* Speakers construct **forward models of their own utterances
before executing them**; perceivers covertly imitate and construct forward models of others'.

**Houde & Nagarajan, "Speech production as state feedback control", *Frontiers in Human Neuroscience*
5:82, 2011.** [CITE] Speech as state feedback control: corollary discharge predicts the auditory
consequences of the motor command, and a mismatch between predicted and actual feedback is the error
signal.

**All three corroborate the architecture's *stalk layout* directly.** A motor boundary cell carrying
the commanded character beside what the world made of it is the same object these accounts describe,
and ADR-0025's placement — *"the same place, and for the same reason, as the arm's efference copy sits
beside its commanded torque"* — is the standard account of speech, not an analogy stretched from the
arm. **`12-the-interlocutor.md`'s readback is corroborated as a description of what speakers have.**

**And none of them is a learning system.** Levelt describes monitoring; Pickering & Garrod describe
prediction; Houde & Nagarajan describe control. In each the readback is *compared against a
prediction and used to correct*, which is a control error. That is one step from — and not the same
as — a component of an observation the learner has to predict along with everything else.

### 2.2 In every learning system found, the readback becomes a loss or a reward

Three, spanning the range, and they agree.

**Warlaumont & Finnegan, "Learning to Produce Syllabic Speech Sounds via Reward-Modulated Neural
Plasticity", *PLOS ONE*, 2016, [doi:10.1371/journal.pone.0145096](https://doi.org/10.1371/journal.pone.0145096).**
[ABS, with body passages read] A spiking network plus a human-like vocal tract acquires canonical
babbling. *"The model is rewarded when it produces a sound that is more auditorily salient than sounds
it has previously produced."* Reward receipt *"increases the level of dopamine in the neural
network"*, which raises learning rates through dopamine-modulated STDP. The paper explicitly says the
reinforcement may stand for caregiver contingency: *"It is reasonable to assume that caregivers
prefer … more salient sounds."*

**This is the closest published object to the map's coherence readback, and it is wired the way
ADR-0025 forbids.** A scalar measuring what the world made of the agent's own utterance, arriving
after the utterance, modulating learning. ADR-0025's recognisable form of the mistake — *"a scalar
that arrives after the agent acts and evaluates what it did … the label on it is irrelevant"* — is a
fair description of this model's dopamine signal. **The ADR is not tilting at a hypothetical: the
field's leading computational model of babbling is the thing it rules out.** Corroborates ADR-0025's
premise that the mistake is cheap to make; contradicts nothing in it.

**Georges, Diard, Girin, Schwartz & Hueber, "Repeat after me: Self-supervised learning of
acoustic-to-articulatory mapping by vocal imitation",
[arXiv:2204.02269](https://arxiv.org/abs/2204.02269) (2022).** [ABS] A forward model *"predicting the
sensory consequences of articulatory commands"* and an inverse model recovering commands from acoustic
input, **jointly trained self-supervised from raw acoustic data**. The agent's own produced sound
circulates as both input and training target. The loop is genuinely sensorimotor; the signal is a
reconstruction objective.

**Stöpler et al. (2025)**, §1.4 — reward.

### 2.3 Limb 2 verdict

**Absent as this map states it, and the absence is uniform enough to be structural.** The readback
exists as a *phenomenon* in three well-founded literatures and as a *mechanism* in none of the
learning systems read here — in every one it is converted into an error, a target, or a reward before
it touches learning. **No source found feeds what the world made of an utterance back as a rim
observation.** The four-part spoken stalk — commanded character, character as taken up, uptake flag,
coherence, per slot — has **no precedent found**, and this document states that as the finding rather
than as a search that ran short.

---

## 3. Limb 3 — uptake and refusal as an observation the learner receives

### 3.1 The floor is a well-described object and it is not an input

**Sacks, Schegloff & Jefferson, "A simplest systematics for the organization of turn-taking for
conversation", *Language* 50(4):696–735, 1974.** [CITE] The foundational account of turn-taking:
turn-constructional units, transition-relevance places, and rules for turn allocation. It is a
description of what conversationalists do, and the most cited article in *Language*.

It gives the map its vocabulary and none of its mechanism. *Whether an utterance was taken up* is, in
conversation analysis, an observable of the interaction available to an analyst. `12`'s uptake flag is
an observable of the interaction **written onto the agent's own motor cell every tick**. That is a
different object, and the difference is the whole of limb 3.

### 3.2 Full-duplex spoken language models: the closest engineering, and the signal points outward

**Ma, Song, Du, Cong, Chen, Wang, Wang & Chen, "Language Model Can Listen While Speaking",
[arXiv:2408.02622](https://arxiv.org/abs/2408.02622) (2024).** [ABS, with body passages read]

LSLM is a genuine structural near-hit for the wedge's two aligned columns. *"At time step `t`, all
previous information of the speaking channel `R^q_{1:t-1}` and the processed information of the
listening channel `S^p_{1:t-1}` are considered by the model simultaneously."* **Two channels, tick
aligned, one model** — which is `11-the-language-graph.md`'s *One stream, two rims* (*"A graph that
never sees its own words beside the interlocutor's is modelling two monologues"*) arrived at
independently. Corroborates #130's alignment decision from outside.

**And then the interruption signal runs the other way.** *"If a sample is selected to include an
interruption, we modify the sentence to output the IRQ token μ=0.5 seconds after the start of the
interruption and then stop outputting the remaining speaking tokens."* The model **emits** a token
announcing that it detected an interruption; it is a supervised target on the output side. The world
never writes back *"your character was refused"*. The model infers refusal from the listening channel
rather than receiving it on the speaking one.

**Li, Wu, Lin, Lee, Qin, Chen & Chen, "Decoupling Conversational Dynamics in Full-Duplex Spoken Models
through Reinforcement Learning", [arXiv:2607.07148](https://arxiv.org/abs/2607.07148) (2026).** [ABS]
The same behaviours learned as *"a separate real-time decision policy"*, optimised with a GRPO-style
objective against the **Factorized Conversational Dynamics Reward**, giving *"fine-grained temporal
credit assignment for turn initiation, backchanneling, yielding, and regularized participation."*
Turn-taking as an explicit reward, decoupled from content. The evaluation line around it — Instruct-FD
frames turn management as instruction-following, Full-Duplex-Bench as overlap handling — treats floor
behaviour as **something the system must produce correctly**, never as something the system is told.

### 3.3 Limb 3 verdict

**Absent as stated, with a strong structural near-miss on the adjacent question.** The two-channel,
tick-aligned rim has independent precedent in LSLM. The uptake flag does not: **in every system read
here, whether the utterance landed is inferred by the model or rewarded in the model, and in none is
it observed by the model.** `12`'s claim that *"turn-taking is the body's refusal on a language rim"*
and that it is *"a real answer from the world"* is unsupported by anything found, and it is
load-bearing — it is what buys the language domain its non-deterministic readback and keeps it out of
`04`'s deterministic-readback forfeit.

**One register consequence, recorded and not acted on.** Carrying uptake per slot means the rim holds
128 slots' worth of past refusals, which is a store in exactly the sense
[#388](https://github.com/NGL321/patchworks/issues/388) names. #388's failure is stated over the shift
of a record; the uptake and coherence components are part of that record. **This is not a new problem
and #388 already covers it**; noted so that a later reader of #388 knows the readback is inside its
scope rather than beside it.

---

## 4. Limb 4 — one symbol stream and one continuous readback in one representation

The thinnest limb, and the pass should say so rather than pad it.

**What exists is about action spaces, not observation stalks.** The discrete–continuous hybrid line —
HyAR ([arXiv:2109.05490](https://arxiv.org/abs/2109.05490)), and the multi-agent hybrid-action work at
[arXiv:1903.04959](https://arxiv.org/abs/1903.04959) — addresses agents that must emit a discrete
choice *and* a continuous parameter, and its contribution is a learned representation that makes the
pair tractable to optimise over. [ABS both] That is the mirror image of the map's problem: the map's
mixture is on the **inbound** stalk of one boundary cell, to be transported by one learned restriction
map, and nothing in these papers concerns transport.

**What exists on the observation side removes the mixture rather than carrying it.** The token-based
world-model line tokenises everything — *"a flexible tokenization framework supporting arbitrary
combinations of observation and action modalities"*, spanning visual, proprioceptive and symbolic
benchmarks — which resolves heterogeneity by discretising the continuous half. [ABS] The
discrete-representation line for continual RL reports that *"world models learned over discrete
representations accurately model more of the world with less capacity"*. [ABS] **Both would delete
the coherence scalar's continuity**, which is the property `12` selected the normalised surprisal for.

**Limb 4 verdict: nothing found.** No source read here carries a one-hot symbol and a continuous
readback in a single stalk that one map is learned against, or asks what that costs. `11`'s 196-wide
spoken stalk — 97 + 97 + 1 + 1 — has no precedent found, and the nearest work would counsel
discretising the two scalars, which the design has an argument against and which nothing found
addresses.

---

## 5. Limb 5 — an explicit comparison of conversation against a corpus of the same content

### 5.1 In children the comparison exists, is controlled, and comes out the map's way

**Roseberry, Hirsh-Pasek & Golinkoff, "Skype Me! Socially Contingent Interactions Help Toddlers Learn
Language", *Child Development* 85(3):956–970, 2014,
[doi:10.1111/cdev.12166](https://doi.org/10.1111/cdev.12166).** [FULL — abstract, results and
discussion read verbatim]

Thirty-six toddlers aged 24–30 months, three conditions, one teaching script: **live**, **contingent
video chat**, **yoked non-contingent video**. *"Results suggest that children only learned novel verbs
in socially contingent interactions (live interactions and video chat)."*

**The yoked condition is the experiment this ticket is about.** Same content, same speaker, same
apparent turn structure — the experimenter asks a question and pauses — and the loop removed. The
discussion states the outcome in the terms #446 uses:

> "Our results from the yoked video condition indicate that simply posing questions to children and
> pausing for the answer did not result in language learning if the children were not able to interact
> contingently with the person on video."

And the design generalises past the screen: video chat matched live interaction, so the effect is
**contingency and not co-presence**. That is precisely the map's claim that the property belongs to
what is on the other end rather than to the medium or the data.

**Corroborates `12-the-interlocutor.md`'s *Conversational by construction* about as directly as an
outside source can.** The spec says *"the partner responds because you spoke"* is *"the whole of the
domain's justification"* and that *"the cheap version of this domain is a corpus and the difference is
structural."* This is that difference, measured, with the corpus version held to the same content.

**Myers, LeWitt, Gallo & Maselli, "Baby FaceTime: can toddlers learn from online video chat?",
*Developmental Science* 20(4), 2016,
[doi:10.1111/desc.12430](https://doi.org/10.1111/desc.12430).** [ABS, with an introduction passage
read] Replicates and extends downward: 12–25 months, real-time FaceTime against pre-recorded video,
and *"only children in the FaceTime group responded to the partner in a temporally synced manner"*,
with the FaceTime group preferring and recognising the partner a week later and learning more novel
patterns. Its introduction indexes the line this sits in — Troseth, Saylor & Archer's closed-circuit
video study, and Nielsen, Simcock & Jenkins on imitation — all with the same yoked design.

**Kuhl, Tsao & Liu, "Foreign-language experience in infancy: Effects of short-term exposure and social
interaction on phonetic learning", *PNAS* 100(15), 2003,
[doi:10.1073/pnas.1532872100](https://doi.org/10.1073/pnas.1532872100).** [CITE — the publisher
returned 403 and the PDF mirror was encrypted] The social-gating experiment: 9-month-olds exposed to
Mandarin live, on video, or by audio alone. **This document does not quote it and does not rest on its
wording.** Its finding is stated here only as Roseberry et al. state it in a passage this pass did
read verbatim: *"children who heard the speakers in a live demonstration learned to discriminate
between the foreign language sounds, whereas the video display failed to confer this advantage."*
Recorded at that strength deliberately, because a citation pass that quotes a paper it could not open
is the failure mode this key exists to prevent.

### 5.2 The quantity-versus-turns separation is also on the record

**Romeo, Leonard, Robinson, West, Mackey, Rowe & Gabrieli, "Beyond the 30-Million-Word Gap: Children's
Conversational Exposure Is Associated With Language-Related Brain Function", *Psychological Science*
29(5):700–710, 2018.** [FULL, via PMC]

Home audio from 36 SES-diverse 4–6-year-olds, story-listening fMRI. Children who experienced more
conversational turns *"independently of SES, IQ, and adult-child utterances alone"* showed greater
left inferior frontal activation, which mediated the relation between exposure and verbal skill. The
statistical separation is stated outright: *"The relation between conversational turns and verbal
scores remained significant … when adult words or child utterances were added to the model, suggesting
that the number of conversational turns was not just a proxy for adult speech or child talkativeness."*
And the interpretation: *"conversational experience impacts neural language processing over and above
SES or the sheer quantity of words heard."*

**Why this belongs in the pass.** It is the observational counterpart to §5.1's experiment, and it
separates the two quantities the map's position depends on separating: **how much language arrives**
against **how much of it is a loop**. It is correlational and the paper says so; it does not establish
that turns *cause* the difference. It establishes that the turn count is not a proxy for the word
count, which is the weaker claim the map actually needs.

### 5.3 In machine learning the comparison has not been run

**Explicit negative, and it is the pass's main gap finding.** No source was found that trains a model
by interacting with a responsive partner and compares it against a model trained on the transcript of
that same interaction. The self-play and dialogue-simulation literature generates transcripts *in
order to* train on them, which is the corpus branch of the comparison without the other branch. The
BabyLM challenge holds the training budget fixed at 100M words and evaluates sample efficiency, which
is the nearest fixed-content control any of this field runs — and it is a control over corpora.

**Charpentier, Choshen, Cotterell, Gul, Hu, Liu, Jumelet, Linzen, Mueller, Ross, Shah, Warstadt,
Wilcox & Williams, "BabyLM Turns 3: Call for papers for the 2025 BabyLM workshop",
[arXiv:2502.10645](https://arxiv.org/abs/2502.10645).** [ABS, with the track section read]

The Interaction track: *"This new track encourages interactive behavior, learning from a teacher, and
adapting the teaching material to the student."* The rules are about budget and leakage — *"the
submission model must be exposed to no more than 100M word tokens … this word count includes text
generated by external models and pre-existing corpora"*, and *"the external model's weights, hidden
states, or output distribution cannot be revealed to the submission model."* The submission model
*"may not generate more than 100M words during the training process."*

Three readings, and the third is the useful one.

- **The loop is permitted and is not the object of measurement.** Nothing in the rules requires that
  the teacher's next text depend on the student's output, and nothing measures whether it did.
- **The prohibition on revealing the teacher's output distribution is, in this map's terms, a
  prohibition on exactly the quantity `12` puts on the rim.** The coherence readback *is* a functional
  of the interlocutor's next-character distribution. Under BabyLM's rules that channel is disallowed
  as leakage; under ADR-0025 it is the readback. **The two frameworks disagree about what that number
  is**, and the disagreement is instructive rather than damaging: BabyLM treats it as privileged
  information about the teacher, and this map treats it as what the world made of the agent's command.
  Handed to §7 rather than resolved here.
- **The track is one year old.** That is the record's own confirmation of the map's "why now".

### 5.4 Limb 5 verdict

**Documented in children with the exact ablation; absent in machines.** The comparison exists, is
controlled, and supports the map's position — in a literature about human infants. Nothing in machine
learning compares conversation against a corpus of the same content, and the one venue organising
around the question began doing so in 2025.

---

## 6. Influences — listed, unattributed, and not support

Per #127's standing rule, these resemble the position without stating it, and **no claim above rests
on any of them**.

- **Bender & Koller, "Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data",
  ACL 2020, pp. 5185–5198.** [ABS] *"A system trained only on form has a priori no way to learn
  meaning."* The octopus is a corpus learner and the argument is about **grounding**, not about the
  loop. The map's distinction survives even if Bender & Koller are wrong, and is not entailed if they
  are right. Listed because a reader will think of it.
- **Emergent communication / Lewis signalling games.** Closed loop, reward channel, degenerate
  protocols. §1.4.
- **Full-duplex spoken dialogue systems** — LSLM, DuplexPO, Instruct-FD, Full-Duplex-Bench. The floor
  and barge-in as engineering objects. §3.2.
- **HyAR and the discrete–continuous hybrid **action** line; token-based world models.** §4.
- **Vieira et al., "From Language Models over Tokens to Language Models over Characters",
  [arXiv:2412.03719](https://arxiv.org/abs/2412.03719).** [CITE] Already cited by `12` for the exact
  prefix marginalisation; it supplies the computation and says nothing about corpus versus
  conversation.
- **Warlaumont's infant vocal-learning line generally**, and the caregiver-contingency findings around
  it. The phenomenon the map is modelling; the mechanism it rules out. §2.2.
- **Chen et al., ReSpect.** §1.3.
- **DIVA and the speech motor-control modelling line.** §2.1. Reached only through review-level
  secondary description in this pass and therefore not cited above as a source.

---

## 7. Per-limb answers, the recommendation, and what is handed on

### Per limb

| limb | verdict |
|---|---|
| **Closed loop** — the learner's output changes what arrives next | **Documented.** Angluin's query/passive separation is the formal statement; Roseberry's yoked video is the empirical one. Never with a drive rather than a reward or label. |
| **Motor readback** — what the world made of the utterance, on the rim | **Absent as stated.** The phenomenon is well owned by psycholinguistics; in every learning system found it is converted to a control error, a reconstruction loss, or a reward before it reaches learning. |
| **Uptake / refusal as an observation** | **Absent as stated**, with a strong near-miss: LSLM's tick-aligned speaking and listening channels. Turn-taking is everywhere a thing the model must produce or is rewarded for, never a thing it is told. |
| **Mixed discrete/continuous rim in one representation** | **Nothing found.** Adjacent work is about hybrid *action* spaces or tokenises the continuous half away. |
| **Explicit corpus-versus-conversation comparison** | **Documented in children** (Roseberry; Myers; Romeo on turns versus word count); **absent in machines**. BabyLM's Interaction track, 2025, is the field arriving at the question. |

### The recommendation on `12-the-interlocutor.md` — a citation *and* a hedge, on different sentences

No decision is owed by this ticket; this is a recommendation and it is split, because the section's
two claims have different standing.

**A citation, for the loop.** *Conversational by construction* asserts that the corpus/conversation
difference *"is structural"* and cites nothing. It can now cite **Roseberry, Hirsh-Pasek & Golinkoff
(2014)** for the controlled ablation and **Romeo et al. (2018)** for the separation of turns from
sheer quantity, with **Angluin (1987)** for the formal separation. The spec's own phrasing — that the
property *"cannot be recovered later by a bigger dataset"* — is the learning-theoretic claim, and
`L*`-versus-Gold is the citation for it.

**A hedge, for the readback.** The sentence *"A corpus gives a sensory rim with no motor rim"* carries
the other three limbs, and **all three are unsupported.** No learning system found supplies the
world's reading of an utterance as an observation; nobody supplies an uptake flag to a learner; nobody
carries a symbol and a continuous readback in one stalk. `12`'s *Known exposure* is the right home,
and the honest form is one bullet saying that the readback half of the corpus/conversation argument is
this project's own and has no precedent found — in the register's `uncut` voice, where naming the debt
is the point.

**Neither is a correction.** Nothing found contradicts `12`, ADR-0025, or #129. The recommendation is
that the section stop carrying supported and unsupported claims in the same sentence.

### Handed to tickets rather than acted on

Three, in the order a reader should care.

1. **#331 may have an instrument, and it comes from §5.1's design.**
   [#331](https://github.com/NGL321/patchworks/issues/331) says the language domain has no lower bound
   on its own solvability because there is no scripted-babbler analogue of `03`'s controller solving
   14 of 72. **The yoked-video design is a different kind of bound and this domain can build it
   cheaply**: run the agent against a *recording* of an interlocutor transcript — same characters,
   same turn structure, uptake always granted, coherence replayed — and compare against the live
   interlocutor. It bounds nothing about achievability, but it is a **contrast condition that
   distinguishes an architecture that cannot use the loop from a domain in which the loop does
   nothing**, which is half of what #331 says is missing and is currently absent entirely. It also
   costs one run and is a falsification either way, in `06`'s sweep style. This is a **candidate for a
   grilling ticket, not a proposal**: ADR-0029 reserves minting to a human, and this pass mints
   nothing.
2. **The drive-not-reward substitution is stipulated and every source disagrees by construction.**
   Every closed-loop language learner read here — POMDP dialogue, emergent communication, DuplexPO,
   Stöpler et al., Warlaumont & Finnegan — closes the loop with a reward. ADR-0009 and ADR-0025 assert
   the map's alternative and neither cites a system that has done it. **That is not a contradiction
   and it is not evidence against the design**; it is the observation that the substitution has no
   external corroboration and that the one group that tried the developmentally-motivated version
   reached for a reward and reported that *"linguistic evaluation improvements remain elusive."*
   Worth a ticket asking what would count as evidence the drive does the work a reward would.
3. **BabyLM's Interaction-track rules and ADR-0025 disagree about what the interlocutor's
   next-character distribution is.** BabyLM forbids revealing the teacher's output distribution to the
   student as leakage; `12` derives the coherence readback from exactly that distribution. Both
   positions are defensible and they cannot both be the default reading. Worth recording so that a
   later reader who arrives from the BabyLM framing does not read `12`'s readback as an oversight. **A
   disagreement, handed on, not acted on.**

---

## 8. What could not be reached, stated plainly

Ranked by how much it weakens what is above.

**Costliest.**

- **Angluin (1987), *Information and Computation* 75(2):87–106, and Angluin (1988), *Machine
  Learning* 2:319–342.** Three mirrors of the 1987 PDF failed — one refused the connection, two
  returned unextractable binary. §1.1 therefore states the theorem's **content** without quoting the
  paper, and the specific pairing of `L*` with Gold's NP-hardness result was taken from a search
  step's summary of the paper rather than from the paper. The result is textbook and independently
  checkable; **the wording is not verified**, and a spec citation should be checked against the
  original before it is written down.
- **Kuhl, Tsao & Liu (2003).** PNAS returned 403 and the mirrored PDF was encrypted. §5.1 records the
  finding only as Roseberry et al. state it in a passage this pass read verbatim, and quotes nothing
  from Kuhl. Because §5.1 does not depend on it — Roseberry's own experiment carries the section —
  this costs the pass a canonical citation rather than a claim.
- **`12`'s own throughput and "why now" argument.** Nothing was searched on whether a 0.5–1B model
  streaming one character per tick is genuinely recent enough to make the experiment newly
  affordable. The map asserts it, the pass assumed it, and §5.3's finding that the BabyLM Interaction
  track is one year old is **consistent with** it rather than evidence for it. Unexamined.

**Moderate.**

- **Pickering & Garrod (2013), Levelt (1983), Houde & Nagarajan (2011), Sacks/Schegloff/Jefferson
  (1974), Young et al. (2013).** All reached at abstract or citation depth only. §2.1 and §3.1 use
  them to establish that a literature *owns* a phenomenon, which is a claim about the existence of the
  field rather than about any sentence in it, so the depth is adequate for the use. **Their
  peer-commentary and later revisions were not read**, and a full read of Pickering & Garrod in
  particular could plausibly upgrade §2.1 from "corroborates the stalk layout" to something stronger
  or weaker.
- **The DIVA model.** Reached only through review-level description. Not cited as a source.
- **BabyLM's *Findings of the Third BabyLM Challenge* (ACL Anthology 2025.babylm-main.28).** The PDF
  would not extract, so **what the Interaction track's participants actually did and whether it helped
  is not known to this pass.** §5.3 rests on the call for papers, which states the rules, not on the
  findings, which would state the outcome. This is the single most valuable thing a follow-up should
  chase: if an interaction-track entrant beat a matched corpus baseline, limb 5's machine-side verdict
  changes.
- **The full-duplex evaluation line** — Instruct-FD, Full-Duplex-Bench v1.5 and v2 — reached at search
  depth only, and listed as influences rather than used.

**Not searched at all**, recorded so the ground is not re-covered blindly: developmental robotics and
the intrinsic-motivation line (Oudeyer, Baranes), which is the field most likely to hold a
drive-shaped rather than reward-shaped closed loop; interactive task learning and instructable agents
beyond §1.2; the animal vocal-learning literature (songbird template matching), where a motor readback
against a template is the standard account and where the question "observation or error signal?" has a
literature of its own; predictive-processing and active-inference treatments of dialogue, which would
be the natural home for a readback that is predicted rather than rewarded and which this pass did not
open; and the sign-language and augmentative-communication literatures, where uptake and refusal are
unusually explicit.

**And the honest negative, which is the pass's headline.** No source was found that trains a learner
on a live turn-taking exchange in which the learner's emission changes the stream, with the motor
consequence of speaking carried on the rim beside the symbols heard, and with no reward channel. Each
limb has neighbours; **the conjunction has none.** `12-the-interlocutor.md`'s position is, on the
readback half, load-bearing and unsupported, and this document states that rather than assuming it.

---

## 9. Sources

| Source | Depth | Access |
| --- | --- | --- |
| Angluin (1987), *Learning Regular Sets from Queries and Counterexamples*, Inf. Comput. 75(2):87–106 | [CITE] | three PDF mirrors failed; content from search summaries, not quoted |
| Angluin (1988), *Queries and Concept Learning*, Machine Learning 2:319–342, [doi:10.1007/BF00116828](https://doi.org/10.1007/BF00116828) | [CITE] | landing page only |
| Gold (1978), *Complexity of Automaton Identification from Given Data*, Information and Control 37(3):302–320, [doi:10.1016/S0019-9958(78)90562-4](https://www.sciencedirect.com/science/article/pii/S0019995878905624) | [CITE] | landing page only; the comparison class in §1.1, not quoted |
| Wang, Liang & Manning (2016), *Learning Language Games through Interaction*, ACL 2368–2378, [arXiv:1606.02447](https://arxiv.org/abs/1606.02447) | [FULL] | ar5iv; abstract and feedback mechanism quoted |
| Chen, Gul, Chen, Geng, Wu & Artzi (2024), *Retrospective Learning from Interactions*, [arXiv:2410.13852](https://arxiv.org/abs/2410.13852) | [ABS] | arXiv landing page |
| Stöpler, Asadli, Nikolaus, Cotterell & Warstadt (2025), *Towards Developmentally Plausible Rewards*, [arXiv:2505.05970](https://arxiv.org/abs/2505.05970) | [ABS] | arXiv landing page |
| Charpentier et al. (2025), *BabyLM Turns 3*, [arXiv:2502.10645](https://arxiv.org/abs/2502.10645) | [ABS] | arXiv HTML; Interaction-track rules read |
| *Findings of the Third BabyLM Challenge*, ACL Anthology 2025.babylm-main.28 | [UNREACHED] | PDF would not extract — see §8 |
| Ma, Song, Du, Cong, Chen, Wang, Wang & Chen (2024), *Language Model Can Listen While Speaking*, [arXiv:2408.02622](https://arxiv.org/abs/2408.02622) | [ABS] | arXiv HTML; two body sentences quoted |
| Li, Wu, Lin, Lee, Qin, Chen & Chen (2026), *Decoupling Conversational Dynamics in Full-Duplex Spoken Models through RL*, [arXiv:2607.07148](https://arxiv.org/abs/2607.07148) | [ABS] | arXiv landing page, abstract verbatim |
| Young, Gašić, Thomson & Williams (2013), *POMDP-Based Statistical Spoken Dialog Systems: A Review*, Proc. IEEE 101(5):1160–1179 | [CITE] | IEEE paywalled |
| Warlaumont & Finnegan (2016), *Learning to Produce Syllabic Speech Sounds via Reward-Modulated Neural Plasticity*, PLOS ONE, [doi:10.1371/journal.pone.0145096](https://doi.org/10.1371/journal.pone.0145096) | [ABS] | open access; abstract verbatim plus reward passages |
| Georges, Diard, Girin, Schwartz & Hueber (2022), *Repeat after me*, [arXiv:2204.02269](https://arxiv.org/abs/2204.02269) | [ABS] | arXiv landing page |
| Levelt (1983), *Monitoring and self-repair in speech*, Cognition 14(1):41–104 | [CITE] | Elsevier paywalled |
| Pickering & Garrod (2013), *An integrated theory of language production and comprehension*, BBS 36(4):329–347 | [ABS] | Cambridge Core abstract |
| Houde & Nagarajan (2011), *Speech production as state feedback control*, Front. Hum. Neurosci. 5:82, [doi:10.3389/fnhum.2011.00082](https://doi.org/10.3389/fnhum.2011.00082) | [CITE] | landing pages only |
| Sacks, Schegloff & Jefferson (1974), *A simplest systematics for the organization of turn-taking for conversation*, Language 50(4):696–735 | [CITE] | paywalled |
| Roseberry, Hirsh-Pasek & Golinkoff (2014), *Skype Me!*, Child Development 85(3):956–970, [doi:10.1111/cdev.12166](https://doi.org/10.1111/cdev.12166) | [FULL] | full-text gateway; abstract, results and discussion read verbatim |
| Myers, LeWitt, Gallo & Maselli (2016), *Baby FaceTime*, Developmental Science 20(4), [doi:10.1111/desc.12430](https://doi.org/10.1111/desc.12430) | [ABS] | abstract verbatim plus one introduction passage |
| Kuhl, Tsao & Liu (2003), *Foreign-language experience in infancy*, PNAS 100(15), [doi:10.1073/pnas.1532872100](https://doi.org/10.1073/pnas.1532872100) | [CITE] | 403 and an encrypted PDF — **not quoted** |
| Romeo, Leonard, Robinson, West, Mackey, Rowe & Gabrieli (2018), *Beyond the 30-Million-Word Gap*, Psychological Science 29(5):700–710 | [FULL] | PMC; abstract and two results sentences verbatim |
| Bender & Koller (2020), *Climbing towards NLU*, ACL 5185–5198 | [ABS] | ACL Anthology abstract; **influence only** |
| Emergent-communication line (Lewis games; overfitting/generalization work, [arXiv:2209.15342](https://arxiv.org/abs/2209.15342)) | [ABS] | search-depth; **influence only** |
| HyAR ([arXiv:2109.05490](https://arxiv.org/abs/2109.05490)); hybrid-action MARL ([arXiv:1903.04959](https://arxiv.org/abs/1903.04959)); token-based world models | [ABS] | search-depth; **influence only** |
| Vieira et al. (2024), [arXiv:2412.03719](https://arxiv.org/abs/2412.03719) | [CITE] | already cited by `12`; not re-fetched |

## Context

Written for [#446](https://github.com/NGL321/patchworks/issues/446), on branch
`research/446-corpus-versus-conversation`. Repo documents read before searching:
`docs/spec/12-the-interlocutor.md`, `docs/spec/11-the-language-graph.md`, ADR-0025, ADR-0003,
ADR-0009, and all six registers in `docs/registers/`, plus `docs/research/394` as the form to match.
Nothing in the spec, the ADRs, the registers or `CONTEXT.md` is edited by this pass.
