# PROTOTYPE — the README front door

**Throwaway.** Three drafts of `README.md`, built to answer
[issue #73](https://github.com/NGL321/patchworks/issues/73): what the front door says, and to whom.
Only the chosen one survives, as the repo's real `README.md`. These are here to be reacted to.

## The constraint that shaped all three

[`docs/spec/10-the-demo-surface.md`](../../docs/spec/10-the-demo-surface.md) §*The front door*
already decided the hero: **two loops side by side**, captioned *I moved the puck* and *I changed
the goal*. Nothing is trained, so the hero cannot be captured. Every draft is therefore a front
door for the interval before the loops exist, and differs in what it puts in the empty slot.

## The three

| | Visitor | Empty hero slot filled with | Register |
|---|---|---|---|
| [`a-the-claim.md`](a-the-claim.md) | A technically literate stranger | Nothing — prose leads | Plain, confident, short |
| [`b-show-me.md`](b-show-me.md) | Someone who wants a picture first | A mermaid diagram of the dome, explicitly marked as a stand-in | Visual-first, minimal prose |
| [`c-the-quilt.md`](c-the-quilt.md) | A collaborator, or the author in six months | Nothing — the *method* leads | Essayistic, distinctive |

They disagree on three things, and those are the decisions to make:

1. **Does the front page commit to the thesis in the first paragraph, or show the demo first?**
   A commits. B shows. C leads with the name and the method instead of either.
2. **Does an untrained project admit that up front or at the bottom?** A and B say it early and
   flatly. C makes the absence of results into the subject.
3. **Is the design process part of the front door?** Only C claims that design-first-cite-after,
   fourteen citation passes, and two apparent novelties are what a visitor should take away.

## Numbers used, and where they come from

- ~150 predicting cells, ~264 boundary cells, apex of 8, `n = 32`, `k = 12` —
  [#8](https://github.com/NGL321/patchworks/issues/8), [#36](https://github.com/NGL321/patchworks/issues/36)
- 4×4 px tiles, pucks 4.3–6.8 px — [#8](https://github.com/NGL321/patchworks/issues/8)
- **12 of 48** tasks solved by the scripted controller —
  [#21](https://github.com/NGL321/patchworks/issues/21), which revised it down from 15/48. Note
  this number is **not yet on `action`**; it lives on `prototype/puck-dynamics`.
- Fourteen citation passes — the `wayfinder:research` tickets on
  [the map](https://github.com/NGL321/patchworks/issues/1).
