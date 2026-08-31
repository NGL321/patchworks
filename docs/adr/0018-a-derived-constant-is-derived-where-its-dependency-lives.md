# ADR-0018: A derived constant is derived where its dependency lives

**Status:** accepted

## Context

Settled in [#191](https://github.com/NGL321/patchworks/issues/191), raised by
[#185](https://github.com/NGL321/patchworks/issues/185)'s edit, which typed every constant in the
scanned modules and could not resolve this without taking a decision.

[#180](https://github.com/NGL321/patchworks/issues/180) §2 kept the `derived` type on the grounds
that it was **empty** — that no constant in `src/patchworks/` was then a pure consequence of others.
It was not empty, and the first register generated from the typed sites showed it: `derived` entries
were there in the first population scanned, and one of them was **unheld**.

`bias_selection.MAP_NORM_BOUND = 2.0` said of itself: *"`rho` in `lambda_max <= rho^2 deg(v)`: the
gauge bound on a restriction map's Frobenius norm (ADR-0010), **which is 2 in the spec**."* That is
`restriction.GAUGE_RHO`, declared a second time, and nothing held the two equal. Move `GAUGE_RHO` to
3 and the slow cap goes on bounding against 2 — silently, in the module that decides every cell's
timescale at construction. This is the two-places failure the register exists to kill, found by the
register, in the first population it scanned.

**The prior question was whether they are the same number.** They are.
[`restriction.py`](../../src/patchworks/restriction.py)'s own docstring already says so: *"Interior
maps live in `[1/ρ, ρ]`; it also appears in the reconciliation gain's denominator, where `ρ² · deg(v)`
bounds the local Laplacian block's largest eigenvalue"* — which is verbatim `MAP_NORM_BOUND`'s stated
job. The gauge that bounds a map's norm is what bounds the Laplacian block assembled from those maps.
There is no reading on which they diverge legitimately: a `MAP_NORM_BOUND` of 2 against a `GAUGE_RHO`
of 3 is not a conservative cap but a **wrong** one, bounding against a band the maps no longer live
in.

`cli.DEPENDENCIES` is the same shape with a guarantee already installed: `derived` from
`pyproject.toml`, with `tests/test_cli.py` holding the table against the pins so it *"reddens rather
than drifts"*. So the repository contained both the disease and the cure, one module apart, and the
question looked like a choice between two house styles. It is not one.

## Decision

### `derived` splits on where the dependency lives, not on the constant

An equality test is wrong for two of the four `derived` entries this repository has. `MAP_NORM_BOUND`
**equals** its dependency. `CONTROL_HZ` is `PHYSICS_HZ / FRAME_SKIP`, and `DEFAULT_GAMMA` would be
`gamma_cap(floor)` — **functions** of their dependencies, where an equality test has nothing to
assert. `DEPENDENCIES` derives from `pyproject.toml`, which is not an importable Python object at
all. One rule covers all four, and it splits on the dependency:

- **Internal** — the dependency is a Python object importable in-process. The constant is then
  **derived in code**: its definition *evaluates* the dependency rather than restating its value.
  Disagreement becomes **impossible**, not merely loud. `MAP_NORM_BOUND`, `CONTROL_HZ` and a future
  `DEFAULT_GAMMA` are here.
- **External** — the dependency is a file, a spec, or the world. Nothing in-process can evaluate it,
  so a **test holds them equal** and the definition site names that test. `MINIMUM_PYTHON`,
  `DEPENDENCIES`, `PHYSICS_HZ` and `SPAWN_R` are here, and this is the arm the repository already
  ran.

This is why the two options are neither equivalent nor a free choice. The import is right for
`MAP_NORM_BOUND` **because its dependency is internal**; the test is right for `DEPENDENCIES`
**because its dependency is TOML**. `cli.DEPENDENCIES` is thereby *correct* rather than
grandfathered, and every future `derived` entry has an answer without a fresh judgement call.

**The coupling arm 1 costs is stated rather than hidden.** Importing `GAUGE_RHO` couples
`bias_selection` to `restriction`. That coupling exists in the mathematics whether or not the code
admits it, and it is cheap: `restriction` imports only `.graph`, so there is **no cycle**, and
`tick.py` already does `from .restriction import GAUGE_RHO`. The direction is precedented.

### The alias survives, because a rename is not a second declaration

`MAP_NORM_BOUND = GAUGE_RHO` keeps its name. `bias_selection` uses `rho` throughout for the
Jacobian's spectral radius, and importing a name called `GAUGE_RHO` bare into that namespace puts two
different `rho`s in one module — the collision the alias was invented to prevent. There is **one
value**, so it is not the two-places failure: an alias is a rename. It also keeps a register row, and
that row is the visible statement that the slow cap depends on the gauge; deleting the name deletes
that statement from the monitoring surface.

### The `value` column resolves

`scan_module` records `value=ast.unparse(node.value)`, so an alias would render as the string
`GAUGE_RHO`. That quietly costs the register the property it exists for: move `GAUGE_RHO` 2.0 → 3.0
and the diff shows the change on `GAUGE_RHO`'s row **only**, while the derived row reads `GAUGE_RHO`
before and after — silent on exactly the change the register was built to make visible. A `derived`
entry's value is therefore **resolved**: import the module, read the attribute, so both rows move
together, with `depends_on` carrying the fact that one follows the other. The column is called
`value` and a derived constant has one; unparsing puts the derivation in two columns and the value in
neither.

### A test enforces the rule, and not a parse-time raise

`test_a_derived_entry_says_derived_from_what` asserts a `derived` entry *names* a dependency. Nothing
asserted it is **honoured**, and that is the whole gap — `depends_on` was prose in a generated file.
`test_a_derived_entry_is_actually_derived_from_it` closes it, on the arm where it can be closed: for
each internal dependency, the definition must evaluate it.

Deliberately **not** a `MalformedProvenance` raise. That exception is for provenance the reader
cannot *parse*; this is provenance that parses cleanly and is false, which is a different failure and
deserves its own name in the output. A parse-time raise also takes down the `surveys` fixture, so an
unrelated test failing would name no constant.

## Consequences

- **`MAP_NORM_BOUND` is an import.** `bias_selection` gains `from .restriction import GAUGE_RHO`.
- **`CONTROL_HZ` is a division.** It said `@depends_on FRAME_SKIP` and restated `50.0`, which arm 1
  forbids. Its other dependency — the arena's `timestep` — had no name in Python at all, so
  `PHYSICS_HZ = 500.0` is added as the external, test-held half (`tests/test_sandbox_env.py` already
  asserts `timestep * FRAME_SKIP == 1 / CONTROL_HZ`), and `CONTROL_HZ = PHYSICS_HZ / FRAME_SKIP`.
  Making the rule executable is what found this; it was not visible before.
- **Neither arm may empty.** A rule with an unused arm is a rule nobody has tested — which is exactly
  how #180 §2 came to keep an "empty" category that was not empty — so a test pins one entry in each.
- **`source` drops the line number.** #180 §4 specified `file:line` and #185 shipped it as specified.
  It goes, and the reason is the same monitoring property:
  `test_the_checked_in_registers_are_not_stale` runs `--check` **as a test**, so inserting a paragraph
  near the top of a scanned module reddens the suite until the registers are regenerated — and the
  regeneration then rewrites every row below the insert. Measured, not hypothetical: across two
  branches the same two constants sat 33 and 3 lines apart with no semantic change, so two branches
  disagree on rows carrying no difference, in the file whose diff is the feature.
  `src/patchworks/body.py` is what remains, and a reader greps a name that is unique in the file by
  construction. If exactness is wanted back, the honest form is a permalink generated at release, not
  a line number held in a file that must not churn.
- **The generated preamble's `derived` gloss links here.** The preamble is a glossary: it says what
  the word means in one line, not why the two arms differ.

## Alternatives considered

- **A test for `MAP_NORM_BOUND` too**, matching `MINIMUM_PYTHON`'s house style. Rejected: it makes
  disagreement loud where the language can make it impossible, and it generalises badly — for
  `CONTROL_HZ` and `DEFAULT_GAMMA` the relation is a function, and an equality test would have to
  restate the function, which is the two-places failure moved into the test suite.
- **An import for everything.** Not available: `pyproject.toml` and the arena XML are not importable
  Python objects, which is the whole reason the split exists.
- **Ruling on this one constant and leaving the type's policy open.** Declined explicitly. Asked
  whether to rule the instance or the general policy, the answer was the general policy — which is
  what turned a fix into a rule and gave every later `derived` entry an answer in advance.
- **Deleting the `MAP_NORM_BOUND` name and using `GAUGE_RHO` directly.** Rejected: it puts two
  different `rho`s in one namespace, and it deletes the register row that says the slow cap depends on
  the gauge.
