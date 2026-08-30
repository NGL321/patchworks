# ADR-0018: Coherence is a motor readback, not a sensory value

**Status:** accepted

## Context

Settled in [#129](https://github.com/NGL321/patchworks/issues/129), and written down with the world it
governs in [`12-the-interlocutor.md`](../spec/12-the-interlocutor.md).

The language domain needs a number that says how well the agent is being understood. The obvious
place to put it was suggested by the sandbox itself, and the analogy is a good one right up until it
is not.

**The sandbox has a two-part structure.** The goal appears **in the render** — the target zone lights
up — while a scalar at the apex asserts *satisfied*. Content in the world, valence in the drive; that
is what lets [ADR-0009](./0009-a-drive-is-a-motor-edge-attached-deep.md)'s drive carry valence without
specification.

**The proposal was to mirror it exactly.** A coherence estimate would arrive **as sensory input**,
while a scalar at the apex asserts *understood*. Same seam, different modality. That proposal was the
ticket's own, it was live, and it was the reading a reader coming from `03-the-sandbox.md` would
expect.

**The apex half is right. The sensory half is not**, and the difference is precisely the thing the
ticket existed to protect.

In the sandbox the goal is a property of **the world**, written by the world, cleared by the cell
changing its belief — a sensory edge by
[ADR-0003](./0003-action-is-prediction-the-world-clears.md)'s criterion. A likelihood of the agent's
**own output** is not that. It is *what the world made of the command*, which is
[#128](https://github.com/NGL321/patchworks/issues/128)'s generalisation of efference copy: a
**readback**. #128 saw it coming and said so — *turn-taking is the body's refusal on a language rim* —
one ticket before the mistake was available to make.

**The cost of getting it wrong is not pedantic.** A scalar that arrives after the agent acts and
evaluates what it did **is a reward**, whatever it is labelled. It would have entered through the
sensory rim, after the drive was carefully kept clean of exactly that, and it would have done so in
the one domain where the no-reward constraint is most at risk. Wiring it as a reward walks straight
into [#5](https://github.com/NGL321/patchworks/issues/5), the gap `05-timescales.md` says plainly the
timescale mechanism does not close.

## Decision

**The coherence number is a component of the spoken rim's readback. It is never written to a sensory
boundary cell, and it is never the value the drive asserts.**

Three claims, and they are one decision:

1. **It is a readback.** It is what the interlocutor made of the agent's command, and it therefore
   sits on the motor boundary cell beside the commanded character and the uptake flag — the same
   place, and for the same reason, as the arm's efference copy sits beside its commanded torque
   ([ADR-0006](./0006-boundary-cell-stalks-are-world-shaped.md)).
2. **The drive still asserts a constant.** *Understood* is asserted at the apex as valence, exactly as
   *satisfied* is in the sandbox. The drive holds **no number from the world at all**, which is what
   keeps it inside the internal rim's ban on holding a model of the world.
3. **The rim structure is three objects, not two** — heard (sensory), spoken (motor, with the
   readback), drive (internal). #128's ban on a sensory and a motor rim sharing a boundary cell forces
   the split, and the split is what keeps ADR-0009's taxonomy intact in a domain that carries both
   directions on one alphabet.

The number itself is the interlocutor's next-character **surprisal normalised by the entropy of its
next-character distribution**, computed by exact prefix marginalisation; that choice and its
alternatives are `12-the-interlocutor.md`'s, not this ADR's.

## Consequences

**The sandbox's two-part structure survives, with one part re-sited.** Content still comes from the
world and valence still comes from the drive. What changes is *which rim* carries the content: in
vision it is sensory because the goal is a fact about the world, and in language it is motor because
coherence is a fact about the agent's own utterance.

**No new mechanism.** A readback is already mandatory on every motor boundary cell
(`04-action-and-the-boundary.md`, *Readback*), and every ingredient here is one the architecture
already has. The decision costs a stalk layout and buys the constraint.

**The signal is bounded and its units do not drift.** That falls out of the decision rather than being
argued separately: a readback is a boundary stalk component that a restriction map has to be learned
against, which is what rules out raw log-likelihood, whose scale moves with the interlocutor's own
uncertainty on the context.

**The failure mode is now specific enough to watch for.** Anything that routes coherence into a
sensory cell, or into the drive's asserted value, has rebuilt the reward this ADR removes. The
recognisable form is a scalar that arrives *after* the agent acts and *evaluates* what it did; the
label on it is irrelevant.

**The drive's asserted value is untouched by this decision** and is deferred to
[#137](https://github.com/NGL321/patchworks/issues/137). `1.0` stands as the working value; whether it
ramps under a ceiling derived from the fold-margin bound is a stage-2-dependent question, because the
bound binds hardest exactly where the drive attaches.

## Alternatives considered

**Coherence as a sensory value, mirroring the render.** The live alternative, and the ticket's own
proposal. Rejected because the mirror is false at the joint: the render is the world describing
itself, and coherence is the world describing what the agent just did. Accepting it would have put a
post-hoc evaluation of the agent's action on the sensory rim, which is a reward with a different name.

**Asking the interlocutor to judge coherence.** Rejected independently of where the number lands:
sparse, discrete, expensive, and reward-shaped by construction. It is the version that makes the
mistake unmistakable, which is some argument for why the likelihood version is the dangerous one.

**Folding coherence into the drive's asserted scalar.** Rejected on ADR-0009 and on the internal rim's
first ban: a drive that carries a live number from the world holds a model of the world, and the drive
edge's disagreement would then be reducible from the drive's side — which is exactly what
`02-tick-semantics.md`'s external-write ordering exists to prevent. It would also make *valence, not
specification* false, since the number varies with what was said.

**Saying nothing and letting the graph designer place it.** Rejected on the same grounds ADR-0016 was
taken: the failure this prevents is not a bad graph but a reward channel, and the placement is hard to
reverse once the rim is built. It is also genuinely surprising to a reader who has internalised the
sandbox's analogy, which is the second bar an ADR has to clear.
