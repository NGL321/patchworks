# The think tank

**Built on [#605](https://github.com/NGL321/patchworks/issues/605), jointly.** The room, the
notebooks and the phrasebooks were the agent's; **the prediction task, the two feedback problems,
the neighbourhood, the cliques and the notebook-page reading of privacy are the user's own**, added
in the same session when the first version turned out to describe a relay network rather than a
predictor. That division matters: a later session must not talk the user out of an image the user
built.

**The symbol reading is the user's own** ([#532](https://github.com/NGL321/patchworks/issues/532),
2026-09-10): the apple is a symbol, and the people are the functions that give symbols their meaning.

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
| the apple everyone is describing | a **symbol** — its identity a community's coherent direction, its activation the traffic riding it |
| what the room genuinely agrees about | **earned agreement** — the count of registered symbols (`dim H⁰` less the privacy reserve) |
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

### 2. Two different things shrink a message, and they must not be quoted as one

A message arrives on one phrasebook and must leave on a different one. What survives *that
handover* is the **cosine of the angle between the directions the two phrasebooks carry** —
[#533](https://github.com/NGL321/patchworks/issues/533) established that the composed object is a
product of principal-angle cosines and nothing else. Over a chain the cosines multiply.

**Those cosines are not small.** [#537](https://github.com/NGL321/patchworks/issues/537) read them
directly over 1578 hops × 3 angles: leading `cos θ` per hop is **0.568** at construction (min 0.474,
max 0.688) and *rises* under training to **0.792 / 0.809** at 20k. No hop is near-aligned and none
is near-orthogonal. `0.79⁷ = 0.19` — a factor of five over seven people, not orders of magnitude.

**A separate quantity, read with a separate instrument, says something much louder.**
[B21](https://github.com/NGL321/patchworks/issues/570)'s 20–50× per person is the peak deviation of a
**dynamical impulse** — a disturbance actually propagating through `sheaf.tick()` — and it reads
**people's pages**. A full-magnitude shock is already down to `9.7e-03` by the **first** person,
~100× before any composition has happened.

The two do not meet:

| | what it reads | in the room | what it says |
|---|---|---|---|
| [#533](https://github.com/NGL321/patchworks/issues/533) / [#537](https://github.com/NGL321/patchworks/issues/537) | the **transport operator** — restriction maps only, which [B17](https://github.com/NGL321/patchworks/issues/565) established **never touches the stalks** | the **phrasebooks**: how much of a topic survives translation | `cos θ` per hop 0.568 → 0.792 / 0.809 |
| [B21](https://github.com/NGL321/patchworks/issues/570) | **node stalks**, via an impulse through `sheaf.tick()` | the **pages**: how large a live disturbance still is a few people later | 20–50× per person; ~100× by hop one |

**Roughly two orders per hop separate them**, and that gap is not a rounding difference inside one
quantity — it is the signature of two. It is the tick's reconciliation gain, the pages' own
dynamics, and whatever else lives between a phrasebook and a page.

So **a person saying different things to different neighbours is not, by itself, the same fact as a
disturbance dying out.** That identity was asserted here as fact and
[B49 (#616)](https://github.com/NGL321/patchworks/issues/616) struck it on the user's ruling: it was
a **weld** of two instruments, not a finding. **Standing rule, inherited from that ruling: a reading
states which of the two objects it is on, and no argument carries a number from one to the other.**

One more number that gets quoted wrongly in the same breath: composed effective rank **1.000 is
domination, not annihilation.** #537 §4 finds the composite numerically **full rank 3** on the median
chain, with `σ₂/σ₁` 0.113 at construction and 0.005 at 20k. One direction survives well and the
others are pushed down — not *nothing arrives*.

### 3. Two cheap escapes, and why "two" is not yet known to be the whole list

**Read as a single number**, a product of small numbers has two ways to get bigger:

| move | effect | how it went |
|---|---|---|
| **fewer terms** — shorten paths, relays, rewiring | multiply 3 cosines, not 7 | [B43](https://github.com/NGL321/patchworks/issues/607): an **unaimed random rewiring** passed the amplitude clause by seven orders. Loud; nothing understood. |
| **bigger terms** — align every lane | every cosine → 1 | [B42](https://github.com/NGL321/patchworks/issues/605): exact path-independence at any distance. Clear; nothing distinct to say. |

Both make the headline number go up. Neither is the architecture.

**But two is not known to be the whole list.** A phrasebook pair does not have
*a* cosine — it has one per direction the phrasebook carries, and this map has only ever read that
spectrum through a **scalar**. A scalar cannot tell *everything is attenuating* apart from *some
directions go through intact and the rest do not*, and those are different rooms. Whether reading the
spectrum instead of the scalar yields a genuine third move is
**[B50 (#618)](https://github.com/NGL321/patchworks/issues/618)'s, and undecided** — it is a reading
of the existing surface, with nothing constructed and no candidate scored. Until it rules, the
analogy states the two moves above and does **not** claim they exhaust the list.

**One thing the table is not about: the size of the page.** The trade is drawn inside a fixed budget
— `Σ_e m_e ≤ n − 1`, phrasebook width paid for out of page space — and a budget is something you
allocate, not a wall. The claim that the budget itself could not be enlarged is **false on `main`**:
scaling page and phrasebooks together at the built ratio reads median composed effective rank
**2.047 at `n ≈ 128`** against **1.015** as built ([#537](https://github.com/NGL321/patchworks/issues/537)),
corroborated independently by [B4 (#539)](https://github.com/NGL321/patchworks/issues/539)'s closed
form at **2.195**. Bigger pages *do* buy transport. That is not a move on the table above; it moves
the table.

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

## The symbol, and where its activity lives

Added on 2026-09-10 from the user's ruling on [#532](https://github.com/NGL321/patchworks/issues/532).
The apple is a **symbol**: the thing outside the room that a set of invariants compresses around. Two
readings of it, and neither stands in for the other (§2's rule): its *identity* — which direction, over
which community — is a fact about the phrasebooks; its *activation* — how much traffic rides that
direction — is a fact about the pages, read as `A(θ)` beside the count `N(θ)`
([B52](https://github.com/NGL321/patchworks/issues/622), [B62](https://github.com/NGL321/patchworks/issues/635)).

A symbol means nothing on its own. It is the people who give it meaning, by composing the symbols they
receive into the function they are trying to approximate; that is where the dynamics are. So *symbol
activation* is the room's whole function: what matters is what in the world outside is represented
inside, never which phrasebook carried it and never in which basis. Two consequences the analogy makes
obvious: the flat room has exactly one symbol, spanning everyone, transmitted perfectly — which is why
it clears the dependence gate ([B66](https://github.com/NGL321/patchworks/issues/642)) and why it is no
architecture ([B42](https://github.com/NGL321/patchworks/issues/605)); and a room whose one agreed thing
is the standing mean ([B57](https://github.com/NGL321/patchworks/issues/629)) has one symbol too. The
bar on #532 asks for many, activated distinctly by what arrives at the portholes.

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
- **"Attenuation *is* misalignment — there is no separate reach" was a weld, and it broke.** §2 stated
  as fact that a person saying different things to different neighbours and a disturbance dying out
  were one quantity read from two ends. They are two quantities, read by two instruments on two
  objects — [B21](https://github.com/NGL321/patchworks/issues/570)'s impulse on **node stalks** and
  [#533](https://github.com/NGL321/patchworks/issues/533)/[#537](https://github.com/NGL321/patchworks/issues/537)'s
  cosines on the **transport operator**, which [B17](https://github.com/NGL321/patchworks/issues/565)
  established never touches the stalks — and they disagree by roughly **two orders per hop**. The
  section also implied cosines of 0.02–0.05, which is `1/20`–`1/50`: B21's amplitude wearing an
  angle's clothes. The measured cosines are **0.568** at construction, rising to **0.792 / 0.809**
  under training. Struck by [B49 (#616)](https://github.com/NGL321/patchworks/issues/616) on the
  user's ruling that **the weld breaks**; both refuting readings were already on `main` when this
  section was written, which is the more useful half of the lesson. **Nothing in §1 moves** — B17's
  result is what supports the content/registration split, and it is the same result that broke the
  weld.

## What it must not be used for

- **It is not evidence.** It generated B42's audience-differentiation reading and the earned-`H⁰`
  reading; both then had to be *measured*, and one of them came back against the prediction.
- **It has no levels in it, and must not grow any.** [B27](https://github.com/NGL321/patchworks/issues/576)
  called the dome wager lost; no bar, reading or ADR may appeal to `level`. A room has no storeys.
- **The apple is the user's, and it is a symbol.** *"Even if neither could describe it in a way the other
  could understand, they have to be describing the same thing."* Earned agreement counts the symbols the
  room has registered — and [B48](https://github.com/NGL321/patchworks/issues/615) exists because the
  trained architecture currently has **none**.
