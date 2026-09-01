"""The README's directory counts, held against the directories (#248).

`README.md` is the project's only outward-facing artifact, and three of its
counts had drifted at once: the badge said **13 ADRs** against 25 on disk, the
status block said **ten spec files** against 12, and the citation-pass count
said **fourteen** against 20. None of it was carelessness -- every number was
true when it was written, and nothing was holding it to the tree afterwards.

That is exactly the failure #180 was opened to kill and #185 built the constant
registers against: *a number in two places with nothing holding them together*.
ADR-0020 states the rule for code. This file applies it to the front door.

**A test rather than a generator.** The registers under `docs/registers/` are
generated because their content is a table with no voice of its own. The
README's counts are prose in the README's own register -- *"Twenty-five
decisions that needed a reason on the record"* -- and generating that sentence
would cost more than it holds. So the number stays hand-written and the test
holds it, which is the same guarantee at a tenth of the machinery.

**What this does and does not catch.** Each count below is anchored to the
phrasing that states it, and every anchor must match: a rewrite that drops a
site fails here rather than silently stopping being checked. A rewrite that
states a count in *new* words this file does not know about is not caught, and
that is the honest limit of the approach -- the anchors are a list, not a
parser of English.
"""

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

#: Enough of the number words to spell the counts this repository will plausibly
#: reach. A count that outgrows this list fails loudly in :func:`_quantity`
#: rather than silently passing, which is the failure mode worth having.
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def _quantity(text: str) -> int:
    """`"twenty-five"`, `"twelve"` or `"25"` as an :class:`int`."""
    word = text.strip().lower()
    if word.isdigit():
        return int(word)
    if word in _UNITS:
        return _UNITS[word]
    if word in _TENS:
        return _TENS[word]
    if "-" in word:
        tens, _, unit = word.partition("-")
        if tens in _TENS and unit in _UNITS:
            return _TENS[tens] + _UNITS[unit]
    raise AssertionError(
        f"{word!r} is not a quantity this file can read; extend _UNITS/_TENS"
    )


#: `directory -> (what it holds, [(where the README says it, pattern)])`.
#:
#: Every pattern captures exactly one quantity, and `\s+` rather than a literal
#: space throughout because the README's prose is hard-wrapped at ~100 columns
#: and a count may sit either side of a newline -- which is how *"as far as
#: fourteen\ncitation passes can tell"* survived a read-through.
COUNTS = {
    "docs/adr": (
        "architecture decision records",
        [
            ("the badge label", r"!\[decisions:\s+(\d+)\s+ADRs\]"),
            ("the badge image", r"badge/decisions-(\d+)_ADRs-"),
            ("the status block", r"([\w-]+)\s+decision\s+records"),
            ("the `Where things are` table",
             r"\(docs/adr/\)\s+\|\s+([\w-]+)\s+decisions\s+that\s+needed"),
        ],
    ),
    "docs/spec": (
        "spec files",
        [
            ("the status block", r"([\w-]+)\s+spec\s+files"),
            ("the `Where things are` table",
             r"\(docs/spec/\)\s+\|\s+([\w-]+)\s+files,\s+in\s+reading\s+order"),
        ],
    ),
    "docs/research": (
        # The table equates the two: "`docs/research/` | The citation passes".
        "citation passes",
        [("the citation-pass claim",
          r"as\s+far\s+as\s+([\w-]+)\s+citation\s+passes")],
    ),
}


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def _on_disk(directory: str) -> int:
    files = sorted((ROOT / directory).glob("*.md"))
    assert files, f"{directory} holds no markdown; the count has nothing to hold to"
    return len(files)


@pytest.mark.parametrize(
    ("directory", "site"),
    [(d, site) for d, (_, sites) in COUNTS.items() for site in sites],
    ids=lambda value: value[0] if isinstance(value, tuple) else value,
)
def test_readme_count_matches_the_directory(readme, directory, site):
    """Each stated count equals the number of files it claims to count."""
    where, pattern = site
    subject = COUNTS[directory][0]
    found = re.findall(pattern, readme)
    assert found, (
        f"README.md no longer states the {subject} count in {where} "
        f"({pattern!r} matched nothing). If the wording moved, move the "
        f"pattern with it; if the site is gone, delete it from COUNTS."
    )
    expected = _on_disk(directory)
    for raw in found:
        assert _quantity(raw) == expected, (
            f"README.md's {where} says {raw!r} {subject}, and {directory}/ "
            f"holds {expected}."
        )


def test_every_count_has_at_least_one_site():
    """A directory listed here with no anchors would check nothing."""
    for directory, (subject, sites) in COUNTS.items():
        assert sites, f"{directory} ({subject}) is held by no pattern"
