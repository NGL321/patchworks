# The interlocutor

The world the agent lives in when that world is a conversational partner, and the surfaces through
which it meets that world. Independent of the architecture: nothing here assumes a sheaf, a cell, or
a local learning rule. It is the language domain's counterpart to
[`03-the-sandbox.md`](./03-the-sandbox.md), and it is deliberately the same document written for a
different world.

Terms used here are defined in [`CONTEXT.md`](../../CONTEXT.md). The decision this document records is
[#129](https://github.com/NGL321/patchworks/issues/129), with the rim's ban from
[#128](https://github.com/NGL321/patchworks/issues/128); the shape of the graph that looks at this
world is [`11-the-language-graph.md`](./11-the-language-graph.md). The one decision taken here rather
than inherited is [ADR-0025](../adr/0025-coherence-is-a-motor-readback-not-a-sensory-value.md).

**This document owns the world; `11` owns the shape of the thing looking at it.** Where the two touch
— the width of a boundary stalk, what a slot of the buffer holds — the contents are this document's
and the count is `11`'s.

## Conversational by construction

The world is a world because **the partner responds because you spoke** — not because it emits text at
you. That sentence is the whole of the domain's justification and it is worth stating before anything
else, because the cheap version of this domain is a corpus and the difference is structural.

A corpus gives a sensory rim with **no motor rim**. The text arrives regardless of what the agent
does, so there is no edge the world clears, no action by
[ADR-0003](../adr/0003-action-is-prediction-the-world-clears.md)'s only test, and no answer to the
dark room. Half the architecture would idle. This is why the unit is the **conversation** and not the
token stream, and why the map rules corpus pretraining out of scope rather than deferring it.

The commitment is the analogue of the sandbox's *planar by construction*: it is not a property of the
data, it is a property of what is on the other end, and it cannot be recovered later by a bigger
dataset.

## The partner

A **small local language model**, run through **llama.cpp** (via `llama-cpp-python`).

| | |
|---|---|
| class | instruction-tuned, **0.5–1B** parameters |
| licence | permissive |
| vocabulary | stable BPE |
| runtime | llama.cpp, streamed one character per tick |
| checkpoint | **deferred to the build** |

The requirements very nearly select the runtime rather than the runtime being chosen: the rig needs
the **full next-character distribution every tick** and **serialisable full state** for restore, and
llama.cpp supplies both as primitives. The specific checkpoint is deliberately not fixed here; the
class is, because the class is what the contract depends on.

**Small is a requirement, not a budget concession.** A knowledgeable partner is a confound rather than
a help — the map's Notes put it plainly, and the claim under test is about structure, never about what
the agent ends up knowing. It is also the only place the compute goes: the architecture is trivially
CPU-feasible, so **the interlocutor is the entire cost of this domain**, which is what forces it local
and small rather than merely recommending it.

## The alphabet

**97 symbols: the 95 printable ASCII characters, plus `idle` and a turn boundary.** One-hot, and the
**same alphabet on both rims**.

One-hot because it asserts nothing. Ninety-seven dimensions is trivial beside the sandbox's
4,096-value render, so there is no compression pressure that would justify pre-digesting the stream,
and **any structure over characters is what the graph exists to discover**.

*Considered and rejected:*

- **A learned embedding at the rim.** Rejected on
  [`04-action-and-the-boundary.md`](./04-action-and-the-boundary.md)'s membership rule — every
  transformation of information on the way in belongs to the graph. An embedding is exactly the
  compression across slices that rule bans, sitting where a cell should be.
- **Bytes.** Considered for modality-honesty and they buy nothing: the interlocutor emits text, and a
  byte rim is three times wider for the same content.

`idle` is a **value**, not an absence — #128's *silence is a value* — and it is what makes the tick
uniform when nobody is saying anything. The turn boundary is likewise a symbol the world writes rather
than an event the rig signals out of band.

## The rims

**Three boundary objects, not two.** #128 bans a sensory and a motor rim sharing a boundary cell, and
that ban is what forces the split; the split is what keeps
[ADR-0009](../adr/0009-a-drive-is-a-motor-edge-attached-deep.md)'s taxonomy intact in a domain where
both directions ride one alphabet.

| | Written / read by | Carries | Cleared by |
|---|---|---|---|
| **Heard rim** — sensory | written by the interlocutor | one character per tick, one-hot | the cell changing its belief |
| **Spoken rim** — motor | read by the interlocutor; readback written back | commanded character; readback of character-as-taken-up, uptake flag, coherence | the world moving, immediately |
| **Drive cell** — internal rim | written from outside, read by no one | scalar, asserting *understood* | the world moving, eventually |

**The drive holds no number from the world at all.** It is a constant, exactly as in ADR-0009, and it
satisfies the internal rim's ban on holding a model of the world trivially — the same way the
sandbox's drive does. The coherence number is *not* what the drive asserts; that is the whole content
of ADR-0025 and the correction #129 exists for.

### The spoken cell is read for its command and written for its readback

This is one cell, not two, and it is worth saying explicitly because the ban above is easy to
over-read.

The world **reads** the commanded character and **writes** the readback onto the same stalk, precisely
as the sandbox's actuator boundary cell is read for three commanded torques and written for three
efference components ([ADR-0006](../adr/0006-boundary-cell-stalks-are-world-shaped.md)). The ordering
is defined and is [`02-tick-semantics.md`](./02-tick-semantics.md)'s: the external write is the tick's
last word, so the command is read before the readback of it lands.

What #128's ban forbids is the **heard** stream landing on the cell the interlocutor reads. That is
the collapse a one-alphabet domain invites, and it is banned because it would make the tick's ordering
undefined — the cell would be read before the word it was going to say. A readback is not that: it is
a motor cell's own account of what the world made of its command, and every motor boundary cell in the
architecture owes one.

> **A wording repair in [ADR-0016](../adr/0016-a-boundary-cell-is-written-or-read-never-both.md).**
> That ADR's *Consequences* said the readback "arrives on a *different* cell rather than being written
> back onto the same stalk". Read literally it forbids the dome's own actuator cell, whose stalk of 6
> is three commanded and three efference, and it would forbid the spoken rim as `11` sizes it. The ban
> is about the sensory stream, not about the readback; the sentence has been corrected in place rather
> than worked around here.

### The readback is genuinely non-deterministic, and language does not need the exemption

`04`'s *Readback* permits a **deterministic** readback with a named forfeit: where the world always
complies, the readback carries no refusal and limit-learning is gone.

Language does not draw on that permission. Whether a character was taken up — dropped because the
interlocutor holds the floor, or landing mid-turn — is a real answer from the world, and **turn-taking
is the body's refusal on a language rim**. It is the exact analogue of the arm's torque clip, and it
is what the uptake flag reports.

### The spoken stalk's width is `11`'s, and this document ratifies it

[`11-the-language-graph.md`](./11-the-language-graph.md) fixes the spoken boundary stalk at **196** —
commanded character (97), character as taken up (97), uptake flag (1), coherence (1), **per slot** —
and marks the width as chosen there rather than recorded from #129. This document supplies the
contents and **ratifies that choice rather than superseding it**: the contents are four, they are
per-slot, and carrying the readback at every slot rather than only at the head is what keeps a
boundary cell's stalk dimension out of the hands of its index. Nothing here reopens the count.

## Half-duplex, and an idle run yields the floor

**One party speaks at a time**, and the **floor** is which of them it is.

**Turn length is not a tick count.** The agent emits a character every tick regardless — silence is a
value and the agent cannot fail to act — and while the interlocutor holds the floor those characters
are **not taken up**: the readback reports refused, tick after tick. An **idle run yields the floor**.
So the agent must **stop talking to be heard**.

This is the load-bearing choice of the whole contract, and its point is that it makes **silence an
action** rather than inert padding — a property this training regime has and corpus pretraining
structurally cannot.

*Considered and rejected:* **fixed-length turns**, which were the live alternative. Under them the
agent's silence changes nothing, the interlocutor replies on schedule regardless, and the degenerate
strategy of *[`04-action-and-the-boundary.md`](./04-action-and-the-boundary.md), The dark room on a
language rim* still gets answered. That is the specific failure the rule exists to make impossible.

**Interruption is out**, and deliberately deferred rather than rejected: whether the agent may be
interrupted mid-turn, and duplex generally, is held in the map's fog.

## The coherence readback

**The interlocutor's own next-character surprisal, normalised by the entropy of its next-character
distribution at that step.** One scalar per slot, on the **spoken** rim, as a component of the
readback — never on the heard rim. ADR-0025 is the decision; what follows is how the number is
produced.

*Considered and rejected:*

- **Asking the model to judge coherence.** Sparse, discrete, expensive, and reward-shaped by
  construction. This was the obvious implementation and it is the one that would have made the signal
  a reward whatever it was labelled.
- **Raw log-likelihood.** Unbounded below, and its scale is set by the interlocutor's own entropy on
  that context — so the same utterance scores differently after a predictable prefix than after an
  open one. That is a boundary stalk whose **units drift with the interlocutor's uncertainty**, which
  is a moving target for the transport rule to learn a restriction map against, and unnecessary when
  the normaliser is available for free.

**Exact prefix marginalisation, over the full vocabulary.** The interlocutor tokenises; the agent
speaks characters. Converting a token-level LM into an exact character-level one is a solved object —
Vieira et al., *From Language Models over Tokens to Language Models over Characters*
([arXiv:2412.03719](https://arxiv.org/abs/2412.03719)) — by summing token probabilities over a
vocabulary prefix trie.

**Exact rather than top-k, and the reason is the normaliser.** The entropy *is* the normaliser, and
truncation removes the tail, which is where entropy lives. A top-k entropy is systematically low by a
context-dependent amount — which reintroduces precisely the drift that normalising was chosen to
remove.

**This makes character-per-tick cheaper than it looks, not dearer.** Characters *inside* a token are
free: the next-character distribution is a masked sum over the same logit vector, so a forward pass is
needed only at **token boundaries** — roughly one per four characters. The mean token length
multiplies the effective tick rate rather than dividing it. The throughput worry that made an
API-backed interlocutor fatal is real for that case and overstated for a local one.

## The topic roster and its sampler

The interlocutor must talk *about* something, and what supplies that is a **fixed roster with a
sampler** — the direct inheritance of `03`'s own shape, line for line. `03` draws a (layout, target
puck, target zone) task from a fixed set and `reset()` rearranges the world by drawing a new one; here
a **topic** is drawn from a fixed roster and `reset()` starts a new conversation by drawing a new one.

*Considered and rejected:* a **retrieval knowledge base**. It is machinery this contract would then
have to describe, and a fixed roster injected into the system prompt gets the same constraint for
nothing.

**Topics are concrete simple subjects, not formal ones.** Bounded vocabulary with high repetition is
what a babbling stage needs. A deterministic domain such as arithmetic buys little when the agent has
no second rim to ground symbols against — it would be manipulating strings either way — and a formal
roster can be added later **without disturbing this contract**, which is why the choice is recorded as
deferrable rather than as final.

## The interlocutor's agenda

Written as a **system prompt, not a mechanism**. Four clauses:

- Talk about the drawn topic, in simple concrete language.
- Keep going when the reply is unintelligible.
- Engage with it when it is not.
- **Never remark on the fact that the partner is babbling.**

The last is a prohibition and so is the first, read the other way, and they are prohibitions for two
different reasons:

- **Remarking on incoherence** would teach the agent that a specific string follows incoherence — the
  task specification leaking back in through the sensory rim, immediately after the readback split
  worked to keep it off.
- **An agenda-only interlocutor that ploughs on regardless** is worse. If speaking does not move the
  world, the spoken rim has **nothing clearing it**, and by ADR-0003 it is not a motor edge at all.
  That is a corpus with extra steps, which is the object the map rules out of scope.

So the partner both holds an agenda **and** genuinely engages, and neither half is decoration.

## The Gymnasium contract, made continual

The env is a literal `gymnasium.Env`, with the **same three deviations** as
[`03-the-sandbox.md`](./03-the-sandbox.md), inherited unchanged. They are the contract, not
placeholders.

- **`reset()` ends the conversation and draws a new topic; it never resets the agent.** The
  interlocutor's context is **cleared**; the agent is **not touched**, and nothing announces it. The
  caregiver walks away and comes back later: what was said before is irrelevant to what is said after,
  beyond both being language. The agent finds out the way it finds out anything — its predictions stop
  working. This is structurally identical to `reset()` rearranging the world while leaving the arm
  where it is.
- **`reward` is always `0.0`; `terminated` and `truncated` are always `False`.** There is no reward
  channel and there are no episodes. **A conversation is not an episode**, because the agent is
  continuous across the boundary. The same consequence carries over: experience is generated by
  interaction, so the **sample budget is unbounded by construction** — there is no dataset and no
  fixed number of examples to be spent.
- **Privileged truth lives in `info`** — topic identity, raw token likelihoods, the interlocutor's
  hidden context, and floor state. For instrumentation and logging only; feeding any of it to the
  agent defeats the domain. **Topic identity is in `info` for the same reason goal identity is**: it
  is what lets a later demo ask whether deep cells separate by topic. That question belongs to the
  unspecified language acceptance demo, and recording the label here is what keeps it **askable
  without asserting it**.

### Snapshot and restore

Because there is no episode boundary to restart from, reproducibility comes from **snapshot/restore**
of the full state, consistent with
[ADR-0001](../adr/0001-continual-learning-applies-to-the-adapting-surface.md).

**Name the engine's own constant, never an enumeration of fields.** The state is llama.cpp's
**sequence-state blob** — `llama_state_seq_*`, wrapped as `LlamaState` / `save_state` / `load_state`
in `llama-cpp-python` — **plus the transcript and the rig's RNG**, which are the two things the
runtime does not know about. Structurally this is `mjSTATE_INTEGRATION` plus the task and the
sampler's RNG, and `03`'s hard-won lesson transfers with it: an enumeration of fields drifts the
moment the runtime gains a feature, and it drifts **silently**, as a diverging trajectory rather than
an error.

The interlocutor's full state being serialisable was the hard part when this domain was proposed —
*restoring a conversation means restoring the interlocutor's state too* — and in the chosen runtime it
is a solved primitive rather than something to design. That is a substantial part of why the runtime
is the one named above.

**A restore is not a reset.** `reset()` is **in-band**: normal operation, the agent lives through it,
and it finds out by being wrong. A **restore** rewinds the whole universe including the agent's own
adapting surface, is invisible from inside — no tick at which a cell could observe one — and never
appears in the env's contract. It is an experimenter's tool. ADR-0001, unchanged.

## The tick, and what it means when the world is silent

**Not fatal, and no new mechanism.** [`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md) commits that
every edge costs one tick and graph distance is temporal distance, which was written for a world that
responds continuously. With `idle` on the heard rim the tick stays uniform, and an interlocutor that
answers after a long pause becomes **world dynamics** — a stretch of highly predictable input — rather
than a gap in the clock. `01`'s commitment is untouched.

**The drive edge gets no special characteristic timescale.** ADR-0009 already distinguishes a motor
edge cleared *immediately* from a drive edge cleared *eventually*, and
[`05-timescales.md`](./05-timescales.md) makes timescale persistence rather than a schedule; a per-edge
rate would be new machinery contradicting both. The spectrum of a learned `K` is retained as the
**instrument** for checking this once the graph transmits — an instrument, never a mechanism, in
ADR-0005's sense.

## What this document does not decide

- **The dark room's language form** is
  [`04-action-and-the-boundary.md`](./04-action-and-the-boundary.md)'s, under *The dark room on a
  language rim*: the failure is **constant emission** rather than silence, and the observable is
  **emission entropy**. It is recorded there because it is a property of drives, not of the world.
- **The graph.** [`11-the-language-graph.md`](./11-the-language-graph.md). This document hands it the
  alphabet, a rim that is genuinely one character per tick, and the confirmation that the drive still
  attaches at an apex.
- **The drive's asserted value.** Whether it stays at `1.0` or ramps under a ceiling derived from the
  fold-margin bound is [#137](https://github.com/NGL321/patchworks/issues/137). `1.0` is the working
  value.
- **The language acceptance demo.** Unspecified, and the most important unspecified item in this
  stage. One partial analogue of `03`'s *human's hand* is available and is worth recording so it is
  not rediscovered: **changing the topic mid-conversation is a `retarget()`** — what is wanted changes
  without the world being touched. Choosing the readouts is that ticket's decision and not this one's,
  and the other two entry points, `perturb()` and `disturb_arm()`, have no obvious analogue at all.

## Known exposure

- **There is no achievability figure, and no way to produce one today.** `03` bounds its world with a
  scripted controller solving 14 of 72 tasks: dumb, privileged, and *lower*-bounding. Nothing here
  corresponds. There is no scripted babbler whose success would establish that the domain admits the
  thing being asked of it, because what is being asked has not been written down — that is the demo
  above. **The language domain therefore has no lower bound on its own solvability**, and it is
  recorded as absent rather than pending so that nobody quotes a number that was never measured.
- **The throughput figure is a build measurement and is not asserted here.** The design intent is tens
  of ticks per second on CPU, the same order as the sandbox, and the token-boundary argument above is
  the reason to expect it. It has not been measured; when it is, it belongs beside `03`'s ~400
  ticks/s.
- **What clears the tail of the spoken rim** is open —
  [#169](https://github.com/NGL321/patchworks/issues/169). Everything above describes the rim's
  contents at a slot; whether an *interior* slot of a shifting buffer is a motor cell in ADR-0003's
  sense is a question about the buffer, and it is `11`'s to answer, not this document's. Nothing here
  depends on the outcome: the contents, the alphabet, the floor rule, and the readback are the same
  either way.
- **The coherence number is computed by the interlocutor about the agent's own output**, so it is
  exactly the shape of thing that becomes a reward the moment it is put on the wrong rim. ADR-0025
  exists to stop that, and the exposure was that the mistake is *cheap to make again* — a future rig
  that logs coherence into `info` and then feeds `info` to anything has reintroduced it. `info` is
  privileged for this reason and not only for tidiness. **Enforced since
  [#348](https://github.com/NGL321/patchworks/issues/348)**: `tests/test_info_containment.py` is a
  standing guard on the channel, in two halves — a tripwire `info` that raises on any read from
  inside a tick, and a deny-by-default static check over `src/patchworks` that fails when a value
  off `info` is passed to anything but the one construction the demo surface is allowed to build.
  What remains open is the residue a guard cannot close: the rig has to *want* the number on the
  motor rim, and the guard only makes putting it anywhere else a visible edit rather than a silent
  one.
- **The topic roster is deferrable and therefore under-argued.** Concrete simple subjects are chosen
  on a babbling-stage argument, not measured; if the roster turns out to be what fails, the formal
  alternative is already named and costs no contract change.
