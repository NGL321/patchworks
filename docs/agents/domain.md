# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root, or
- **`CONTEXT-MAP.md`** at the repo root if it exists: it points at one `CONTEXT.md` per context. Read each one relevant to the topic.
- **`docs/adr/`**: read ADRs that touch the area you're about to work in. In multi-context repos, also check `src/<context>/docs/adr/` for context-scoped decisions.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

Single-context repo (most repos):

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-event-sourced-orders.md
│   └── 0002-postgres-for-write-model.md
└── src/
```

Multi-context repo (presence of `CONTEXT-MAP.md` at the root):

```
/
├── CONTEXT-MAP.md
├── docs/adr/                          ← system-wide decisions
└── src/
    ├── ordering/
    │   ├── CONTEXT.md
    │   └── docs/adr/                  ← context-specific decisions
    └── billing/
        ├── CONTEXT.md
        └── docs/adr/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## An ADR quoting a measured figure names its surface

> **An ADR quoting a measured figure names the surface it was taken on**, wherever the surface is
> contestable.

Adopted on [#437](https://github.com/NGL321/patchworks/issues/437). Naming the surface means saying
which build the number describes and where it was read — the rig, the branch if it is not `main`, the
horizon, and any constant whose value the number depends on. A surface is contestable whenever the
build can move under the number: a constant that might be retuned or deleted, a mechanism in review
rather than merged, a horizon a later read might extend. When in doubt, name it. The cost is a
clause; what it prevents is an argument that reads as current and is not.

**The instance.**
[ADR-0032](../adr/0032-the-maps-learn-isometric-transport-and-a-spectral-floor-expresses-it.md)
quoted its cost, its benefit and its counterweight at effective rank **1.0009**, four paragraphs
after its own *Context* said 1.0009 was the `λ = 0.4` surface
[ADR-0031](../adr/0031-the-sparsity-pressure-is-deleted.md) deleted. The decision survived the
re-read; its whole ledger did not.

This is the map-side rule on [#127](https://github.com/NGL321/patchworks/issues/127) — *a reading
quoted in this map carries the surface it was taken on* — extended by exactly one artifact class. It
lives here rather than in a map's Notes because the failure is committed by whoever writes the next
ADR, and that is not reliably someone reading a map: a map dies with its effort, and ADRs outlive it.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because…_
