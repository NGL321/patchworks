# ADR-0020: A repeated default is a constant with a definition site

**Status:** accepted

## Context

Settled in [#199](https://github.com/NGL321/patchworks/issues/199), written by
[#209](https://github.com/NGL321/patchworks/issues/209). The third invisibility the constant register
has been shown, after [#187](https://github.com/NGL321/patchworks/issues/187)'s class fields
(reachable once opted in) and [#191](https://github.com/NGL321/patchworks/issues/191)'s unheld
`derived` (visible, and false).

`bias_selection.driven_trajectory` and `bias_selection.measure` each took `drive_scale: float = 1.0`.
`scan_module` walks `ast.parse(text).body` — the module's top level — plus the fields of classes
marked `@register`. A signature default is reached by neither, so the number was not merely
unwarranted but **unflaggable**: the unmarked-constant test only sees what the scan reaches, so the
completeness check that exists to stop a constant going silent could not report this one at all.

**It had a warrant, and the warrant was one line away.** The walk is
`stalk = retention * stalk + innovation * draw()` with `innovation = sqrt(1 - retention^2)` and
`draw()` drawn at standard deviation `drive_scale`, so the stationary standard deviation of the walk
**is** `drive_scale` — and `driven_trajectory`'s own docstring already asserts *"an
Ornstein-Uhlenbeck walk with unit stationary variance"*. `1.0` is precisely what makes that docstring
true. The register admits a constant "if it has a warrant: a statable reason it is this value rather
than another", and this is one; it was written in prose immediately above the code and nowhere the
register could see it.

**It is not `agent.DRIVE_ASSERTION`.** #137 fixed *the drive's asserted scalar* at 1.0 on ADR-0009's
warrant. `drive_scale` is a different quantity wearing the same number: the amplitude of a synthetic
node stalk in the selection rig. Reading #137's warrant onto it would have been its own two-places
failure, and the register would then have carried the claim that the rig's stalk amplitude is fixed
by ADR-0009, which it is not. The shared `1.0` is the trap this constant sets for the next reader,
so the collision is written down at the definition site.

The `drive_scale` sites are **two**, not three: `sweep` takes no `drive_scale` and lets `measure`'s
default stand, and `benchmarks/timescale_selection.py` calls `measure` directly when it varies the
scale, so nothing that needs to sweep it is blocked.

## Decision

### Being a default argument exempts nothing

The register's own preamble already refused the analogous move for construction parameters: *"A
construction parameter is not a constant, and both are registered… it is a distinction in what the
number is free to be, not in whether it needs a warrant: the question this register asks is why this
value rather than another, and a value passed at construction has to answer it as much as one bound
at import."* The reasoning transfers without modification to a value bound in a signature.

The readability cost is nil, because the convention is already unanimous. The four sibling defaults
in these very signatures — `DEFAULT_TICKS`, `DEFAULT_BURN_IN`, `DEFAULT_DRIVE_CORRELATION`,
`DEFAULT_OPERATOR_SCALE` — are module-level names, and across the scanned modules thirteen parameter
names already default to a module constant, `safety_factor` across seven signatures and `ticks`
across six. `drive_scale` was the one violation of a 13-to-1 house rule, not the first case of a new
one.

### The trigger is repetition across two or more signatures

Not defaults in general. A single-site default cannot silently diverge from itself, and most defaults
are not constants — they are one-off parameters of one function, and admitting them would drown the
register in numbers with no architecture behind them. What repetition adds is a second place the
number lives, free to drift from the first, which is the failure the register exists to kill.

### The fix is a hoist, not a fourth scanner reach

`DEFAULT_DRIVE_SCALE = 1.0` at module level, both signatures defaulting to it. The existing scan
finds it, no scanner changes, and it lands as one `stipulated` row in `rig.md` beside
`DEFAULT_DRIVE_CORRELATION`. `stipulated` rather than `chosen`: the amplitude convention fixes the
value, it is not argued locally.

The alternative — teaching `scan_module` to read signature defaults — is rejected on three counts. It
admits *every* default in the scanned modules, most of them the one-off parameters the paragraph
above rules out. It collides with the scan's `name.isupper()` filter, since parameters are lowercase.
And it would be a reach built to accommodate a single outlier that the codebase's own convention says
should not exist.

**The real argument for that reach is recorded here so it can be made properly later.**
`DEFAULT_TICKS` is this register's flagship unknown *because* a rig tick count silently decided a
result ([#178](https://github.com/NGL321/patchworks/issues/178)), and a scanner reading defaults
would surface every rig knob at once. That is a larger appetite than this decision's, and it is not
foreclosed — if the register later wants every rig knob, **this ADR is what it argues against**,
rather than being extended quietly.

### A check ships with the rule

ADR-0018 shipped `test_a_derived_entry_is_actually_derived_from_it` on the reasoning that
`depends_on` had been "prose in a generated file", and that test is what caught the `IMAGE_SIZE`
collision when its branch merged. Pairing a rule with its check is the house habit, and a rule with
no check is the discipline this suite's docstring already records failing.

`test_no_bare_numeric_default_is_repeated_across_signatures` walks the scanned modules'
`FunctionDef` nodes, collects parameters whose default is a bare numeric literal, and fails on any
name appearing in two or more signatures. It returned `drive_scale` before the hoist and returns
empty after.

**It flags bare numeric literals only.** A default of `None`, `_UNSET`, or a constant name is the
settled form and stays legal — otherwise `bias_variance` (`DEFAULT_BIAS_VARIANCE`/`None`), `gamma`
(`DEFAULT_GAMMA`/`_UNSET`) and `image_size` (`IMAGE_SIZE`/`None`) redden falsely, each of them a
constant that already has a definition site. `bool` is excluded with the other non-numerics: a
repeated `flag: bool = False` is not a constant, and `True` is an `ast.Constant` of type `bool`
before it is one of type `int`.

## Consequences

- **`bias_selection.DEFAULT_DRIVE_SCALE` exists**, and `driven_trajectory` and `measure` default to
  it. One new `stipulated` row in `docs/registers/rig.md`.
- **The register now states the selection rig's amplitude convention.**
  [#181](https://github.com/NGL321/patchworks/issues/181) holds that the hop-versus-floor comparison
  is not well-formed without an amplitude convention; there is one, in the rig that measures the
  floor, and it now has a name. #181 should adopt it rather than invent a second.
- **The scanner is unchanged.** The invisibility is closed by the codebase conforming to the scan,
  not by the scan growing a reach.
- **A future rig-knob register argues against this ADR.** #178's appetite is stated above rather than
  left to be discovered, so the later ticket knows what it is overturning.

## Alternatives considered

- **A default argument is simply not a constant**, and the repetition is the price of readable
  signatures. Rejected: the register's preamble already answered this for construction parameters,
  and the readability cost is nil against a convention four sibling parameters in the same two
  signatures already keep.
- **A third scanner reach, reading signature defaults.** Rejected on scope, on the `isupper()`
  collision, and on being built for one outlier — with the honest argument for it recorded above so
  it survives the rejection.
- **Flagging every repeated default, literal or not.** Rejected: it reddens on `None`, `_UNSET` and
  constant names, which are the settled form. The check would fire on three parameters that already
  do exactly what this ADR asks.
- **Ruling the instance and leaving the policy open.** Declined, matching ADR-0018: asked whether to
  rule the constant or the rule, the answer is the rule, so the next repeated default has an answer
  in advance.
