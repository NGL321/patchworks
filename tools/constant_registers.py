"""The three constant registers, read off the definition sites (#180, #185).

The register **cannot disagree with the code, because it is a projection of
it**. Provenance is written once, as structured fields above the existing prose
at the definition site; this module parses those fields and renders
`docs/registers/`. Nothing here holds an argument — an entry carries a link back
to the source line and nothing more, because the argument already exists at the
definition site and a second copy is a second place a decision lives, which is
the failure #180 exists to kill.

The fields, on `#:` comment lines contiguously above the constant (they may sit
above the prose or inside it; the parser takes the whole `#:` run):

* ``@type`` — one of :data:`TYPES`, ordered by flexibility.
* ``@flexibility`` — required. ``unknown`` is a real value and the expected one:
  this architecture's sensitivity to its own numbers is largely unmeasured, and
  silence would collapse *nobody measured this* into *somebody measured it and
  forgot to say*, which are opposite states for a reader deciding whether to
  turn a knob.
* ``@warrant`` — where the reason lives. A spec section, an ADR, a
  `docs/research/` doc, the rig that selected it, or an issue. The literal
  ``here`` is admissible and means *argued at the definition site and nowhere
  else*, which is exactly what ``chosen`` claims; ``source`` is then the link.
* ``@depends_on`` — required on ``derived`` and meaningless elsewhere. Knowing a
  constant is derived is useless if the register will not say derived *from
  what*.
* ``@provisional <issue>`` — in place of ``@type``, for a constant resting on an
  unmet precondition. The networked half of the check watches these.
* ``@register none`` — the opt-out, at the definition site rather than in an
  allowlist file, which would be the two-places failure again. It records that
  *someone considered this and it has no warrant*, which is strictly more than
  silence.

Run ``python tools/constant_registers.py`` to regenerate; ``--check`` exits
non-zero if the checked-in files are stale.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src" / "patchworks"
OUTPUT = ROOT / "docs" / "registers"

#: The six types, **ordered by flexibility**, least flexible first. That
#: ordering is what makes a register readable at a glance rather than a table to
#: study.
TYPES = ("measured", "derived", "selected", "literature", "stipulated", "chosen")

#: The modules the completeness check scans. Not all of `src/patchworks/`:
#: `surface/` and `progress.py` are presentation, and their constants are
#: colours, glyph tables and column widths rather than numbers the architecture
#: rests on.
SCANNED = (
    "agent.py",
    "bias_selection.py",
    "body.py",
    "cli.py",
    "diagnostics.py",
    "graph.py",
    "learning.py",
    "restriction.py",
    "tick.py",
    "sandbox/env.py",
)

#: Which register each scanned module's entries land in, and what *flexibility*
#: means there. The division is #180 §1's.
REGISTERS = (
    (
        "architecture",
        "Architecture",
        "*is this a knob I may turn*",
        ("agent.py", "body.py", "learning.py", "restriction.py", "tick.py"),
    ),
    (
        "rig",
        "Rig",
        "*does the measurement change if I turn it*",
        ("bias_selection.py", "diagnostics.py"),
    ),
    (
        "world-and-build",
        "The world and the build",
        "*what breaks downstream if the world changes*",
        ("cli.py", "graph.py", "sandbox/env.py"),
    ),
)

_FIELD = re.compile(r"^#:\s*@(?P<key>\w+)(?:\s+(?P<value>.*))?$")


class MalformedProvenance(Exception):
    """A definition site whose fields cannot be read as provenance."""


@dataclass(frozen=True)
class Entry:
    """One constant's provenance, as the definition site states it."""

    name: str
    value: str
    module: str
    line: int
    type: str = ""
    provisional: str = ""
    flexibility: str = ""
    warrant: str = ""
    depends_on: str = ""

    @property
    def source(self) -> str:
        return f"src/patchworks/{self.module}:{self.line}"

    @property
    def sort_key(self) -> tuple[int, str]:
        rank = TYPES.index(self.type) if self.type in TYPES else len(TYPES)
        return (rank, self.name)


@dataclass
class Survey:
    """What one scan found: the entries, and the constants that opted out."""

    entries: list[Entry] = field(default_factory=list)
    opted_out: list[tuple[str, str]] = field(default_factory=list)
    unmarked: list[tuple[str, str]] = field(default_factory=list)


def _fields_above(lines: list[str], index: int) -> dict[str, str]:
    """The `@key value` fields in the `#:` run immediately above line *index*.

    The run is taken whole, so fields may sit above the prose or inside it. A
    key repeated is a later one winning, which no definition site does.
    """
    start = index
    while start > 0 and lines[start - 1].startswith("#:"):
        start -= 1
    found: dict[str, str] = {}
    for raw in lines[start:index]:
        match = _FIELD.match(raw)
        if match:
            found[match.group("key")] = (match.group("value") or "").strip()
    return found


def scan_module(relative: str) -> Survey:
    """Every module-level constant in one module, with what it declares."""
    path = SOURCE / relative
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    survey = Survey()
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            # Private names are implementation detail rather than the
            # architecture's numbers, and a sentinel object has no "why this
            # value" to answer.
            if name.startswith("_") or not name.isupper():
                continue
            declared = _fields_above(lines, node.lineno - 1)
            if declared.get("register") == "none":
                survey.opted_out.append((relative, name))
                continue
            if not declared:
                survey.unmarked.append((relative, name))
                continue
            survey.entries.append(
                _entry(relative, name, node, declared)
            )
    return survey


def _entry(relative: str, name: str, node: ast.stmt, declared: dict[str, str]) -> Entry:
    """One declaration, checked. Raises rather than rendering a half-read entry."""
    where = f"src/patchworks/{relative}:{node.lineno} ({name})"
    kind = declared.get("type", "")
    provisional = declared.get("provisional", "")
    if kind and provisional:
        raise MalformedProvenance(
            f"{where}: @provisional replaces @type; it does not accompany it"
        )
    if not kind and not provisional:
        raise MalformedProvenance(f"{where}: needs @type or @provisional")
    if kind and kind not in TYPES:
        raise MalformedProvenance(
            f"{where}: @type {kind!r} is not one of {', '.join(TYPES)}"
        )
    if provisional and not provisional.isdigit():
        raise MalformedProvenance(
            f"{where}: @provisional names the issue it waits on, got {provisional!r}"
        )
    flexibility = declared.get("flexibility", "")
    if not flexibility:
        raise MalformedProvenance(
            f"{where}: @flexibility is required, and `unknown` is a real value"
        )
    if not declared.get("warrant"):
        raise MalformedProvenance(f"{where}: @warrant is required")
    depends_on = declared.get("depends_on", "")
    if kind == "derived" and not depends_on:
        raise MalformedProvenance(
            f"{where}: derived is unreadable without @depends_on"
        )
    if depends_on and kind != "derived":
        raise MalformedProvenance(
            f"{where}: @depends_on belongs to derived entries, not {kind or 'provisional'}"
        )
    return Entry(
        name=name,
        value=ast.unparse(node.value) if node.value is not None else "",
        module=relative,
        line=node.lineno,
        type=kind,
        provisional=provisional,
        flexibility=flexibility,
        warrant=declared["warrant"],
        depends_on=depends_on,
    )


def survey_all() -> dict[str, Survey]:
    """Every scanned module, keyed by its path under `src/patchworks/`."""
    return {relative: scan_module(relative) for relative in SCANNED}


def provisionals(surveys: dict[str, Survey] | None = None) -> list[Entry]:
    """Every provisional entry, across all three registers."""
    surveys = surveys if surveys is not None else survey_all()
    found = [e for s in surveys.values() for e in s.entries if e.provisional]
    return sorted(found, key=lambda e: (int(e.provisional), e.name))


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

_HEADER = """<!-- Generated by tools/constant_registers.py. Do not edit by hand.

The provenance lives at the definition site; this file is a projection of it, so
it cannot disagree with the code. Edit the `#:` fields above the constant and
regenerate. -->
"""

_PREAMBLE = """
Every constant in these modules either appears here or is marked `#: @register
none` at its definition site — `tests/test_constant_registers.py` enforces that,
so a constant added later cannot silently arrive without provenance.

**A constant is admitted if it has a warrant: a statable reason it is this value
rather than another.** What that excludes is *names* — `ENV_ID`, `ENTRY_POINT`,
`MJPYTHON`, `ARM_JOINTS` — defined by MuJoCo, Gymnasium and the arena XML, where
"why that value" has no answer and an entry would be a verbatim copy of the code
carrying no information.

**Rows carry no argument.** The reason is at the definition site, which the
`source` column links; a warrant of *here* means it is argued there and nowhere
else. Rows are ordered by type, least flexible first.
"""

_TYPE_TABLE = """
| type | what it means |
|---|---|
| **measured** | read off a run. Not a knob at all. |
| **derived** | a consequence of other constants. Not settable independently. |
| **selected** | a rig chose it, against a criterion that is re-runnable. |
| **literature** | a published result. A knob you would have to beat it to turn. |
| **stipulated** | the spec or the world says so. A knob with a record attached. |
| **chosen** | argued locally, nothing external binds it. The most flexible kind. |
| **provisional #N** | resting on an unmet precondition. Not a type: a debt. |
"""

_GAPS = """
## Stated gaps

Two populations this register cannot reach, named rather than silently omitted —
being absent from a register of what the architecture rests on is worse than the
status quo, because the register would be quietly incomplete.

* **`DomeSpec`'s fifteen construction parameters** — `interior_m`,
  `touch_stalk`, `patch_grid` and the rest. Equally warranted (`touch_stalk`'s
  docstring reads *"Chosen here, not recorded"*, which is the `chosen` type
  stated verbatim before the type existed) and equally invisible to a
  module-level scan. They stay on `DomeSpec` because, unlike `n` and `k`, they
  genuinely are the dome's counts. This is a **scanner-reach** question and is
  [#187](https://github.com/NGL321/patchworks/issues/187).
* **`patchworks.body.hidden_width`** — `max{d_x + 1, d_y}`, Park et al.'s floor,
  which is a **rule rather than a number** and so has no value column to carry.
  Its warrant is in its own docstring.
"""


def render(slug: str, title: str, meaning: str, modules: tuple[str, ...],
           surveys: dict[str, Survey]) -> str:
    entries = sorted(
        (e for m in modules for e in surveys[m].entries), key=lambda e: e.sort_key
    )
    out = [_HEADER, f"# The {title.lower()} register\n"]
    out.append(f"Constants from {', '.join('`' + m + '`' for m in modules)}, where "
               f"*flexibility* means {meaning}.\n")
    out.append(_PREAMBLE)
    out.append(_TYPE_TABLE)
    out.append("\n## Entries\n")
    out.append("| name | value | type | flexibility | warrant | depends_on | source |")
    out.append("|---|---|---|---|---|---|---|")
    for e in entries:
        kind = f"provisional #{e.provisional}" if e.provisional else e.type
        out.append(
            f"| `{e.name}` | `{_cell(e.value)}` | {kind} | {_cell(e.flexibility)} "
            f"| {_cell(e.warrant)} | {_cell(e.depends_on) or '—'} | `{e.source}` |"
        )
    opted = sorted(n for m in modules for _, n in surveys[m].opted_out)
    if opted:
        out.append("\n## Marked `@register none`\n")
        out.append("No warrant to state — a name, or text the code prints.\n")
        out.append(", ".join(f"`{n}`" for n in opted))
    if slug == "architecture":
        out.append(_GAPS)
    return "\n".join(out).rstrip() + "\n"


def _cell(text: str) -> str:
    """One table cell: pipes escaped, newlines gone."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def generate() -> dict[pathlib.Path, str]:
    surveys = survey_all()
    return {
        OUTPUT / f"{slug}.md": render(slug, title, meaning, modules, surveys)
        for slug, title, meaning, modules in REGISTERS
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true",
        help="exit non-zero if the checked-in registers are stale",
    )
    args = parser.parse_args(argv)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stale = []
    for path, text in generate().items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            continue
        if args.check:
            stale.append(path)
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        names = ", ".join(str(p.relative_to(ROOT)) for p in stale)
        print(f"stale: {names}\nrun `python tools/constant_registers.py`", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
