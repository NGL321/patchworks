"""The hermetic half of the provenance check (#180 §5, #185 §4).

A constant may not rest on a promise, and this repository has already
demonstrated that leaving that to discipline fails: `tick.py`'s `DEFAULT_GAMMA`
documented itself as waiting on #85's fold-margin cap *"until that check
exists"*; #85 closed, `bias_selection.FoldMarginCheck.gamma_cap` was written,
and the comment went on saying it did not exist. **The comment was true when it
was written**, which is the whole difficulty — nobody was careless.

So the guarantee is a test's rather than a comment's, which is not a new idiom
here: `cli.py:60` already records exactly that of `MINIMUM_PYTHON`'s agreement
with `pyproject.toml`.

**This file is the hermetic half only.** Whether a provisional's issue is still
open is a question for GitHub, and `tests/test_cli.py`'s locality guard exists
to break every route to a subprocess; a `gh`-shelling test would be this suite's
first network-dependent one and would fail offline — on the development laptop,
which is where #85 was closed from. That half is
`.github/workflows/constant-provenance.yml`, triggered on `issues: closed`.
"""

import pytest

import constant_registers as registers


@pytest.fixture(scope="module")
def surveys():
    """Every scanned module, parsed once.

    Parsing is itself an assertion: :func:`constant_registers.scan_module`
    raises :class:`MalformedProvenance` on a field set it cannot read as
    provenance — a type outside the six, a `derived` entry that will not say
    derived from what, a missing `@flexibility`. Every test below therefore
    rests on this fixture having been built at all.
    """
    return registers.survey_all()


class TestEveryConstantHasProvenanceOrSaysItHasNone:
    """Completeness, which is what stops the register from being decoration.

    Without it a constant added next month silently has no provenance -- the
    original disease with a register standing next to it.
    """

    def test_no_scanned_constant_is_silent(self, surveys):
        """Fields, or `@register none`. There is no third state.

        The opt-out lives at the definition site rather than in an allowlist
        file, which would be the two-places failure this mechanism exists to
        kill, and it records that *someone considered this and it has no
        warrant* -- strictly more than silence, and a real question for a
        reviewer when a pull request adds one.
        """
        unmarked = [
            f"src/patchworks/{module}: {name}"
            for survey in surveys.values()
            for module, name in survey.unmarked
        ]
        assert unmarked == [], (
            "these module-level constants declare neither provenance fields nor "
            "`#: @register none`:\n  " + "\n  ".join(unmarked)
        )

    def test_every_scanned_module_was_actually_read(self, surveys):
        """A typo in `SCANNED` would pass every other test in this file."""
        assert set(surveys) == set(registers.SCANNED)
        assert all(
            survey.entries or survey.opted_out for survey in surveys.values()
        )

    def test_the_entries_and_the_opt_outs_are_disjoint_and_named_once(self, surveys):
        for module, survey in surveys.items():
            named = [entry.name for entry in survey.entries]
            named += [name for _, name in survey.opted_out]
            assert len(named) == len(set(named)), f"{module} names a constant twice"


class TestAProvisionalIsWellFormed:
    """`@provisional` is a debt, and a debt names its creditor."""

    def test_every_provisional_names_a_positive_issue_number(self, surveys):
        """Hermetic: that it *is* an issue number, not that the issue is open.

        Whether the issue has closed is the networked half's question, and the
        one this whole mechanism was built for.
        """
        for entry in registers.provisionals(surveys):
            assert entry.provisional.isdigit(), entry.source
            assert int(entry.provisional) > 0, entry.source

    def test_a_provisional_carries_no_type(self, surveys):
        """`@provisional` replaces `@type`; a provisional constant has no type.

        That is the point of #180 §5: a constant waiting on a ticket is not
        *stipulated*. Saying both would let it read as settled.
        """
        for entry in registers.provisionals(surveys):
            assert entry.type == "", entry.source

    def test_gamma_is_the_one_this_mechanism_was_built_for(self, surveys):
        """`DEFAULT_GAMMA` is provisional on #85, and #85 is closed.

        Pinned by name rather than left to the general rules, because the
        mechanism's whole warrant is that this exact entry existed for months
        with a comment saying the opposite. If a later change makes `γ` a
        genuine `derived` entry -- `min(spec bound, fold-margin cap)`, which is
        #159's successors' work and explicitly not this ticket's -- this test is
        what says the debt was actually settled rather than quietly dropped.
        """
        gamma = [e for e in registers.provisionals(surveys) if e.name == "DEFAULT_GAMMA"]
        assert [e.provisional for e in gamma] == ["85"]


class TestTheRegistersAreAProjectionOfTheCode:
    """The register cannot disagree with the code, because it is generated."""

    def test_the_checked_in_registers_are_not_stale(self):
        """`--check`, run as a test rather than left to a hook.

        The generated files are checked into git because the diff is the
        feature: `DEFAULT_SAFETY_FACTOR: 2.6 -> 3.1` as a one-line change in a
        reviewed file is the monitoring property that motivated centralising at
        all. A checked-in file that nothing regenerates is a stale file, so the
        suite regenerates and compares.
        """
        assert registers.main(["--check"]) == 0

    def test_every_entry_lands_in_exactly_one_register(self, surveys):
        """The three registers partition the scanned modules.

        A module in none of them would have its constants checked for
        completeness and then never printed, which is the quiet incompleteness
        the stated gaps exist to avoid.
        """
        assigned = [module for _, _, _, modules in registers.REGISTERS for module in modules]
        assert sorted(assigned) == sorted(registers.SCANNED)
        assert len(assigned) == len(set(assigned))

    def test_the_types_are_ordered_by_flexibility(self):
        """The ordering is what makes a register readable at a glance.

        `measured` is not a knob at all and `chosen` is one you may simply turn,
        so the table is sorted by how free the number is rather than
        alphabetically.
        """
        assert registers.TYPES == (
            "measured",
            "derived",
            "selected",
            "literature",
            "stipulated",
            "chosen",
        )

    def test_a_derived_entry_says_derived_from_what(self, surveys):
        """Knowing `γ` is derived is useless if the register will not say from what."""
        derived = [
            entry
            for survey in surveys.values()
            for entry in survey.entries
            if entry.type == "derived"
        ]
        assert derived, "the derived category is not empty; see #185's resolution"
        for entry in derived:
            assert entry.depends_on, entry.source

    def test_a_derived_entry_is_actually_derived_from_it(self, surveys):
        """`depends_on` is honoured, not merely stated — on the arm where it can be.

        The test above asserts a `derived` entry *names* a dependency. Nothing
        asserted it is **kept**, and that gap is what produced
        [#191](https://github.com/NGL321/patchworks/issues/191):
        `MAP_NORM_BOUND` said it depended on `GAUGE_RHO` and restated the
        number, so moving the gauge would have left the slow cap bounding
        against a band the maps no longer live in — silently, in the module that
        decides every cell's timescale at construction.

        ADR-0018 splits on where the dependency lives, and so does this. Where
        it is an importable Python object the definition must **evaluate** it,
        which is what makes disagreement impossible; where it is a file or the
        world — `pyproject.toml`, the arena's ring wall — nothing in-process can
        evaluate it and a test holds the two equal instead, which is the arm
        `cli.DEPENDENCIES` and `PHYSICS_HZ` already run.

        Deliberately not a `MalformedProvenance` raise at parse time. That
        exception is for provenance that cannot be *read*; this is provenance
        that reads cleanly and is false. A parse-time raise also takes down the
        `surveys` fixture, so the failure would name no constant (#191 §4).
        """
        unheld = []
        for survey in surveys.values():
            for entry in survey.entries:
                if entry.type != "derived":
                    continue
                for name in registers.internal_dependencies(entry):
                    if name.rpartition(".")[2] not in entry.references:
                        unheld.append(f"{entry.source}: {entry.name} -> {name}")
        assert unheld == [], (
            "these `derived` constants name an importable dependency and then "
            "restate its value rather than evaluating it:\n  " + "\n  ".join(unheld)
        )

    def test_the_two_arms_are_both_populated(self, surveys):
        """Neither arm of ADR-0018 is a category with nothing in it.

        A rule with an empty arm is a rule nobody has tested, and #180 §2 kept
        `derived` on exactly that reasoning — *that it is empty* — which the
        first generated register disproved. `MAP_NORM_BOUND` and `CONTROL_HZ`
        hold the internal arm; `DEPENDENCIES` and `SPAWN_R` hold the external
        one. If a change empties either, the split is no longer earning the ADR
        and this says so rather than leaving it standing unused.
        """
        derived = [
            entry
            for survey in surveys.values()
            for entry in survey.entries
            if entry.type == "derived"
        ]
        internal = [e.name for e in derived if registers.internal_dependencies(e)]
        external = [
            e.name for e in derived if not registers.internal_dependencies(e)
        ]
        assert "MAP_NORM_BOUND" in internal
        assert "DEPENDENCIES" in external

    def test_a_derived_value_column_carries_the_number_not_the_name(self, surveys):
        """`MAP_NORM_BOUND`'s row moves when the gauge moves.

        Unparsing the definition would render the alias's value as the string
        `GAUGE_RHO`, so the register would go silent on the one change it is
        checked into git to make visible. The column is called `value` and a
        derived constant has one.
        """
        entry = next(
            e
            for e in surveys["bias_selection.py"].entries
            if e.name == "MAP_NORM_BOUND"
        )
        assert entry.value == "GAUGE_RHO"
        assert registers.rendered_value(entry) == "2.0"

    def test_source_is_the_file_and_not_the_line(self, surveys):
        """A column that churns under unrelated edits, in the file whose diff is the feature.

        `--check` runs as a test, so an insert near the top of a scanned module
        reddens the suite until the registers are regenerated — and the
        regeneration then rewrites every row below the insert, which is a merge
        conflict on rows with no semantic difference (#191 §5).
        """
        for survey in surveys.values():
            for entry in survey.entries:
                assert entry.source == f"src/patchworks/{entry.module}"
                assert ":" not in entry.source

    def test_no_entry_hides_its_flexibility(self, surveys):
        """`unknown` is a real value and the expected one.

        Silence would collapse *nobody measured this* into *somebody measured it
        and forgot to say*, which are opposite states for a reader deciding
        whether to turn a knob. Most entries read `unknown`, and that is the
        honest finding rather than a defect of the register.
        """
        for survey in surveys.values():
            for entry in survey.entries:
                assert entry.flexibility, entry.source


class TestTheReaderIsRejectingWhatItShould:
    """The parser's own behaviour, against definition sites written for the occasion."""

    def _read(self, tmp_path, monkeypatch, body):
        module = tmp_path / "sample.py"
        module.write_text(body, encoding="utf-8")
        monkeypatch.setattr(registers, "SOURCE", tmp_path)
        return registers.scan_module("sample.py")

    def test_a_type_outside_the_six_is_refused(self, tmp_path, monkeypatch):
        with pytest.raises(registers.MalformedProvenance, match="is not one of"):
            self._read(
                tmp_path,
                monkeypatch,
                "#: @type guessed\n#: @flexibility unknown\n#: @warrant here\nX = 1\n",
            )

    def test_a_missing_flexibility_is_refused(self, tmp_path, monkeypatch):
        with pytest.raises(registers.MalformedProvenance, match="@flexibility"):
            self._read(
                tmp_path, monkeypatch, "#: @type chosen\n#: @warrant here\nX = 1\n"
            )

    def test_a_derived_entry_without_depends_on_is_refused(self, tmp_path, monkeypatch):
        with pytest.raises(registers.MalformedProvenance, match="@depends_on"):
            self._read(
                tmp_path,
                monkeypatch,
                "#: @type derived\n#: @flexibility unknown\n#: @warrant here\nX = 1\n",
            )

    def test_provisional_and_type_together_are_refused(self, tmp_path, monkeypatch):
        """They are alternatives. A constant with both reads as settled and is not."""
        with pytest.raises(registers.MalformedProvenance, match="does not accompany"):
            self._read(
                tmp_path,
                monkeypatch,
                "#: @provisional 85\n#: @type stipulated\n"
                "#: @flexibility unknown\n#: @warrant here\nX = 1\n",
            )

    def test_a_provisional_naming_no_issue_is_refused(self, tmp_path, monkeypatch):
        with pytest.raises(registers.MalformedProvenance, match="names the issue"):
            self._read(
                tmp_path,
                monkeypatch,
                "#: @provisional soon\n#: @flexibility unknown\n#: @warrant here\nX = 1\n",
            )

    def test_fields_are_read_from_anywhere_in_the_comment_run(self, tmp_path, monkeypatch):
        """Fields sit above the prose, and the prose stays exactly where it was.

        The parser takes the whole contiguous `#:` run rather than requiring the
        fields to be adjacent to the assignment, so a definition site can be
        annotated without its argument being rewritten or moved.
        """
        survey = self._read(
            tmp_path,
            monkeypatch,
            "#: @type chosen\n#: @flexibility unknown\n#: @warrant here\n"
            "#: The prose that was already here, untouched.\nX = 1\n",
        )
        assert [e.name for e in survey.entries] == ["X"]
        assert survey.entries[0].type == "chosen"

    def test_a_private_or_lowercase_name_is_not_a_constant(self, tmp_path, monkeypatch):
        """A sentinel object has no "why this value" to answer.

        Requiring `@register none` on `_UNSET = _Unset()` would be noise, and
        noise in a completeness check is how a completeness check stops being
        read.
        """
        survey = self._read(tmp_path, monkeypatch, "_HIDDEN = 1\nlowercase = 2\n")
        assert survey.entries == []
        assert survey.opted_out == []
        assert survey.unmarked == []

    def test_an_unannotated_constant_is_reported_rather_than_raised(
        self, tmp_path, monkeypatch
    ):
        """Reported, so the failure message can name every one of them at once."""
        survey = self._read(tmp_path, monkeypatch, "X = 1\n")
        assert survey.unmarked == [("sample.py", "X")]
