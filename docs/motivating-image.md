# The motivating image

Established on [#230](https://github.com/NGL321/patchworks/issues/230) at the user's request and
written here by [#234](https://github.com/NGL321/patchworks/issues/234), as a standing reference for
[#127](https://github.com/NGL321/patchworks/issues/127)'s effort. It exists so that later sessions
**index to it rather than re-derive it**.

**What this document is.** The picture the architecture is trying to be — the world it assumes, the
constraints it accepts, and the object it takes itself to be building. It is motivation, not
specification. Nothing here overrides `CONTEXT.md`, the spec or an ADR; where it touches a decision,
the decision is the record and this is why the decision looked worth taking.

**Two standing rules on how it is written**, both the user's:

- **No attributed citations.** The formulation is the user's own, and many of its parts differ
  deliberately from the work that influenced them. Named influences appear at the foot as a **bare
  list of general influences on the theory**, never as a source for any particular claim. Nothing here
  is to be read, or later cited, as something a named author said.
- **The thermodynamic-persistence idea is motivation with thin provenance**, and is marked as such
  where it appears. The house rule is that citations validate design and never seed it, and
  [#231](https://github.com/NGL321/patchworks/issues/231) found six expired grounds in this record. If
  that idea later needs to be load-bearing it gets a `/research` pass first.

---

## The world

A fully general learning agent: a data structure standing for a cognitive system, covering artificial
general intelligence and any complex animal at once. Informed by mammalian neocortex but never by it
alone — cephalopod distributed brains force the structure not to be neocortex-shaped; arthropods enter
as a **counter-example**, largely genetically programmed and so unproductive for the general-learning
part.

**Decision-making is not the only object.** It is one of many that must fall out of the same
algorithm, and it is the privileged handle only because it is the one process known not to be
hard-coded. The nematode is the demonstration: no generalizable structure, no dynamic behaviour, a
snap reflex — which is a stabilized solver for a control problem. Sensory processing and motor control
are **not excluded**: the non-recursive parts should be simulatable from the recursive ones, and
animals' sensory preprocessing is a concession to sensor count, which is scale.

## The constraints

Continual learning. Learning by interacting with the world. Distributed. Recursive, with a canonical
repeated unit — argued **genomically rather than anatomically**: neocortex is too uniform and too
large for the genome to place cell by cell, so it likelier evolved recursively, which entails *some*
canonical microcircuit. Self-referential, with **active** predictive coding as the control model: a
control problem with learning attached. Decomposition and compression, layered. Thermodynamic, on the
position that information has an exact physical manifestation. A graph, since semantic memory looks
like one and graph tools carry the approximation that dynamical-system solving needs.

Added late: language models are evidence that a sufficiently good **embedding space** supports
enormously many purposes under unconstrained dynamics — so a fundamental embedding space plausibly
exists to be found.

## The object

A **recursive compression engine**, equally viewable as a semantic graph of dynamics, a decomposing
dynamical system, and an active-predictive-coding agent — which is what supplies purpose: model the
world, act favourably on it. Two functions in parallel and coordinated, building the model and
selecting the action, whose joint operation drives self-mutation and learning. That is the origin of
the dome's two branches meeting in a core.

## The chamber

A vessel whose shape is a **biological prior** — the dome's levels are that shaping, constraining a
heterarchical substrate so the ripples take the dynamics they ought to; not a hierarchy imposed where
a heterarchy belongs. Perturbations enter at the sensory wall; another part of the wall is receptive,
and is motor. It is a **dissipative system, so echoes are supposed to fade.** What survives is not the
ripple but the structure it assembles, and it survives because it is topologically compressed.

*(The persistence half of that sentence is the thermodynamic idea marked above as thin-provenance
motivation.)*

## The refinement, and the better image

Not viscosity — **structuredness of a vector field.** Where the field is stable it forms shapes, and
those shapes are **channels** that direct the ripples. Many signals bounce at once, decaying slowly,
decaying *less* through stable structure, with new signals constantly entering from the rim. Noisy,
with stable channels through it — areas of higher stability and areas of more noise, producing a
continual dynamical flow of multiple signals.

**This is why it is brain-like rather than system-like: it is not doing one thing.**

The record already has this object under exactly that name: ADR-0022's *learned channel*.

## The recursion

Nodes wired by **bundles** of strings rather than single strings, because a node means different
things in different contexts and so do the vibrations through it. Finite connectivity per node. Zoom
again and the same structure recurs: the whole graph is a linear model, each node is a linear model
because it must be solvable, and **the exchange between them cannot be linear** because the thing
modelled is not.

That is the patchwork.

## Why a sheaf

Compression must be **non-abelian**, therefore heterogeneous, therefore a graph with heterogeneous
connections; a sheaf over a graph is the construction that supplies it. **This is the derivation of
the project's central choice**, and before #230 it was written nowhere in the repo.

Offered alongside it and **marked as the agent's**, accepted as sensible rather than ruled: the
non-abelian object here is plausibly the **holonomy** of the restriction maps around cycles — trivial
exactly when the maps are homogeneous, with `H¹` as its obstruction. That would make heterarchy
structural rather than aesthetic, since non-abelian composition needs loops to compose around. The
dome carries **269 independent cycles** (414 cells, 682 edges), and #150 found each level a lattice
with many parallel routes into the next.

**Vocabulary warning, and it is load-bearing.** `docs/research/015-information-cohomology.md` already
citation-passed the information-cohomology line from primary texts and ruled it an **interpretive
lens, not load-bearing formalism**, with **no relation** between its `H¹` and a cellular sheaf's —
different site, coefficients, differential, and no comparison map. `CONTEXT.md` enforces that and puts
*topological invariant* on its **Avoid** list. This document selects the *idea* rather than the
formalism, which is compatible; but the two vocabularies now touch, and they must be kept apart. The
`H¹` above is the cellular sheaf's and nothing else's.

## The goal, replacing free energy

Not minimisation of an objective — **persistent information structures.** It coincides with survival,
which is why active inference appears: not a mathematical objective but a behaviour a body learns on
top of what began as a rudimentary control system. In the data structure, survival means thermodynamic
survival, and thermodynamic survival means topological compression. *(Thin-provenance motivation, per
the rule above.)*

## Attachment, recorded because it calibrates everything above

The world and the object are held **tightly**. The lowest level is held **loosely** — intuition the
user can imagine alternative formulations of, and has already let design decisions push back on. The
method is carving a Hilbert space with literature filling holes, not modelling it exactly.

---

## Named influences

**A bare list of general influences on the theory, unattributed and in no order.** None of them is a
source for any claim in this document, and nothing above is to be presented as something any of them
said.

Richard Sutton; Jeff Hawkins; Vernon Mountcastle; Karl Friston; Rajesh Rao; Bernard Koopman; Pierre
Baudot and Daniel Bennequin; Blaise Agüera y Arcas; Ilya Prigogine; Jean Piaget; Henri Poincaré; the
"compression is all you need" line of work; mixture-of-experts and tiny-recursive-model results;
transformer scaling; and comparative neuroscience across mammals, cephalopods, arthropods and
nematodes.

Piaget and Poincaré enter through schema formation in psychology, which the user reads as part of the
same topological perspective.
