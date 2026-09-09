# The think tank

**Built on [#605](https://github.com/NGL321/patchworks/issues/605), jointly.** The room, the
notebooks and the phrasebooks were the agent's; **the prediction task, the two feedback problems,
the neighbourhood, the cliques and the notebook-page reading of privacy are the user's own**, added
in the same session when the first version turned out to describe a relay network rather than a
predictor. That division matters: a later session must not talk the user out of an image the user
built.

It is the working analogy for the whole architecture. Use it in preference to inventing a new one.

---

## The image

A huge crowd of people in a room, working **one very hard time-series problem** together, fed
information slowly.

Only a few people stand at the walls, where **portholes** deliver messages from outside. Everyone
else hears only from other people in the room.

Each person:

1. Reads what arrives — from a porthole if they are at a wall, otherwise from a handful of neighbours.
2. Interprets it **through their own specialty**, in their own private notation.
3. **Predicts the next step** of the series from that specialty.
4. Writes notes encoding that prediction — one per neighbour, each in the **phrasebook** they share
   with that particular neighbour.
5. Must work out **two** things from the same stream of notes: what is happening in the problem,
   **and whether their own predictions have been any good.** There is no separate channel for the
   second.

Nobody was told what their job is. Nobody chose their neighbours knowing what those neighbours would
turn out to specialise in. Everyone runs on the same clock — nobody waits.

## The dictionary

| in the room | in the architecture |
|---|---|
| a person | a **cell** |
| their page of 32 numbers | the **node stalk** |
| a porthole | a **boundary cell** |
| a phrasebook shared by two neighbours | a **communication lane** (the edge's stalk) |
| how wide that phrasebook is | `m_e` |
| a person's rule for writing into one phrasebook | a **restriction map** |
| the part of the page kept for their own reading | **private features** — the reserve `p` |
| everything a person can say and hear in total | the **communication bus**, `Σ_e m_e` |
| who they still bother reading | their **neighbourhood** |
| a group who all mean the same thing by something | a **community** — the user's *clique* |
| how many cliques a person is in | **abstraction** (B35) |
| narrowing a phrasebook that never helps | a **carve** |
| what the room genuinely agrees about | `H⁰` |
| porthole → room → response → world → porthole | the **world loop** |

## The five things the analogy is actually good at

### 1. Two things travel, and only one must survive unchanged

- **The content** — what a person thinks happens next. This *should* change as it travels; if it
  came back identical nobody learned anything and the room is a delay line.
- **The registration** — *which thing in the world the message is about.* This must not change.

**Holonomy measures the second, not the first.** That is not interpretation: the composed object
multiplies the **translation rules only** and never touches anybody's page ([B17](https://github.com/NGL321/patchworks/issues/565)),
so a holonomy reading contains no belief content at all. It asks whether the room's dictionaries are
mutually consistent, not whether its beliefs are unchanged.

### 2. Attenuation *is* misalignment — there is no separate "reach"

A message arrives on one phrasebook and must leave on a different one. What survives is the
**cosine of the angle between the directions the two carry** — [#533](https://github.com/NGL321/patchworks/issues/533)
established that the composed object is a product of principal-angle cosines and nothing else.
Over a chain the cosines multiply.

So [B21](https://github.com/NGL321/patchworks/issues/570)'s 20–50× loss per person and reach of 2–3
hops is not a separate fact from expressiveness. **A person saying different things to different
neighbours is exactly those angles being large.** One quantity, read from two ends.

### 3. Which makes the central trade visible, and its two cheap escapes

A product of small numbers has exactly two ways to get bigger:

| move | effect | how it went |
|---|---|---|
| **fewer terms** — shorten paths, relays, rewiring | multiply 3 cosines, not 7 | [B43](https://github.com/NGL321/patchworks/issues/607): an **unaimed random rewiring** passed the amplitude clause by seven orders. Loud; nothing understood. |
| **bigger terms** — align every lane | every cosine → 1 | [B42](https://github.com/NGL321/patchworks/issues/605): exact path-independence at any distance. Clear; nothing distinct to say. |

Both make the headline number go up. Neither is the architecture. Whether a third move exists is
[B49](https://github.com/NGL321/patchworks/issues/616).

### 4. Privacy is a split on one page, not a locked drawer

**The user's reading, and better than the agent's.** It is not a separate hidden compartment — it is
**how much of the one page a person gives to their own understanding versus what they have taken from
their neighbours' notes.** Page space spent on your own view is page space not spent on the room's.

Why it matters, all three the same thing — *being able to be a specialist*:

- **Slow belief.** Reconciliation is a continuous argument; anything exposed is under permanent
  pressure to conform. Private entries are the only place a hypothesis can persist long enough to be
  tested against a slow series.
- **Perspective.** If everything is exposed and the room reconciles, the room converges. Privacy is
  why 150 people do not become one averaged person.
- **Commitment.** You can only commit to what you cannot be argued out of.

And both extremes are degenerate ([B22](https://github.com/NGL321/patchworks/issues/571)): too little
and every idea spans all 150 — one blob, no specialists; too much and each idea spans one person —
no groups at all. **The graded middle is the heterarchy**, and it is what the project has been after
since [the motivating image](../motivating-image.md)'s Pandemonium objection.

### 5. Reversibility, and why an irreversible carve is worse than a wrong one

The room is also learning **who should talk to whom about what**. Some conversations turn out
useless and the design wants to narrow them and spend the capacity elsewhere.

[B41](https://github.com/NGL321/patchworks/issues/604): a carve is a **reallocation, never a
deletion**. Two reasons, both visible in the room:

1. The series is long and the world changes; a conversation useless for ten thousand ticks may
   become the essential one.
2. **The evidence that closing it was wrong would have had to travel down the channel that was
   closed.** An irreversible carve does not just make a bad bet — it destroys the room's ability to
   discover that it made one.

## Where it has been wrong

Kept because a corrected analogy is more trustworthy than a clean one.

- **First version had no prediction in it.** People relayed facts rather than forecasting, which made
  it a story about a communication network and quietly implied that what returns around a loop should
  be identical to what left. The user caught it. Prediction, and the fact that *"how am I doing?"*
  rides the same wires as *"what is happening?"*, are load-bearing.
- **"Everyone speaks the same language" was too strong** for the flat bundle. Across a phrasebook it
  is a genuine translation between incompatible vocabularies — one person's direction 1 can be *red,
  round, crooked ends* and their neighbour's *sweet, white, brown ovoid seeds*. What actually
  collapses is **inside** one person: they have one thing to say and say it to all their neighbours.
  The precise failure is *audience differentiation*, not global homogeneity — and getting that right
  is what turned the objection from a dimension count into the clique argument that decided B42.
- **"Clique" carries a wrong sense from graph theory**, where it means everyone connected to
  everyone — density. The record explicitly rejects that. Here the defining property is **shared
  referent**: a connected set over which one direction stays consistent. A tight huddle can agree on
  nothing; a strung-out chain can all mean the same thing.

## What it must not be used for

- **It is not evidence.** It generated B42's audience-differentiation reading and the earned-`H⁰`
  reading; both then had to be *measured*, and one of them came back against the prediction.
- **It has no levels in it, and must not grow any.** [B27](https://github.com/NGL321/patchworks/issues/576)
  called the dome wager lost; no bar, reading or ADR may appeal to `level`. A room has no storeys.
- **The apple is the user's, and it is `H⁰`.** *"Even if neither could describe it in a way the other
  could understand, they have to be describing the same thing."* That is earned agreement — and
  [B48](https://github.com/NGL321/patchworks/issues/615) exists because the trained architecture
  currently has **none**.
