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


class TestTheRegisterReachesIntoAMarkedClass:
    """`DomeSpec`'s fifteen, which a module-level scan could not see (#187).

    A construction parameter is a number the architecture rests on that happens
    to be a dataclass field. The register's stated question is *is this a knob I
    may turn*, so a register that cannot show the knob #14's constraint ladder
    starts on -- `interior_m` -- is not answering it.
    """

    def _fields(self, surveys):
        return {
            entry.name: entry
            for entry in surveys["graph.py"].entries
            if entry.name.startswith("DomeSpec.")
        }

    def test_all_fifteen_are_registered(self, surveys):
        """The count is pinned because the gap was one of *reach*, not of writing.

        Fifteen is every field of `DomeSpec`; if a sixteenth arrives, this test
        and the completeness check below both speak, and they say different
        things -- this one that the count moved, that one that the new field has
        no provenance.
        """
        fields = self._fields(surveys)
        assert len(fields) == 15
        assert set(fields) == {
            f"DomeSpec.{name}"
            for name in (
                "patch_grid", "vision_sides", "somatomotor_sizes", "core_sizes",
                "joints", "interior_m", "boundary_m", "drive_m", "patch_stalk",
                "proprioceptive_stalk", "touch_stalk", "actuator_stalk",
                "drive_stalk", "core_degree", "apex_degree",
            )
        }

    def test_the_class_marker_sets_the_register_and_a_field_may_override_it(
        self, surveys
    ):
        """Thirteen architecture, two the world's.

        Change the render and `patch_grid` and `patch_stalk` must follow, which
        is *what breaks downstream if the world changes* exactly, while the
        thirteen counts around them are knobs the architecture may turn. Module
        granularity cannot express that split: `graph.py` is one file.
        """
        lands = {
            name: registers.lands_in(entry)
            for name, entry in self._fields(surveys).items()
        }
        world = {n for n, slug in lands.items() if slug == "world-and-build"}
        assert world == {"DomeSpec.patch_grid", "DomeSpec.patch_stalk"}
        assert set(lands.values()) == {"architecture", "world-and-build"}

    def test_interior_m_is_stipulated_rather_than_provisional(self, surveys):
        """A ladder you may climb is not a precondition you failed to meet.

        `06`'s argument for `m = 4` stands and the value is defensible;
        `@provisional` means *resting on an unmet precondition*, and typing this
        provisional would make every value with a known upgrade path provisional
        and drain the marker. The thinness is real and lives in `flexibility`,
        citing #14 -- which is where a reader asking *may I turn this* looks.
        """
        entry = self._fields(surveys)["DomeSpec.interior_m"]
        assert entry.type == "stipulated"
        assert entry.provisional == ""
        assert "#14" in entry.flexibility

    def test_the_dome_is_not_left_looking_unwarranted(self, surveys):
        """`DEFAULT_SPEC`'s opt-out is about the name, not about the dome."""
        assert ("graph.py", "DEFAULT_SPEC") in surveys["graph.py"].opted_out


class TestTheRenderAndTheTilingAreOneNumber:
    """`IMAGE_SIZE` is `patch_grid` patches of `PATCH_PX` a side, and held to it.

    Nothing used to stop `patch_grid = 8` leaving `IMAGE_SIZE` at 64 with the
    tiling no longer covering the render; the flexibility field named
    `DomeSpec.patch_grid` in prose and the register could not link it, because
    the field was not registered. Registering it makes `@depends_on` expressible
    for the first time, so the guarantee is a test's rather than a comment's --
    the idiom `cli.py`'s `MINIMUM_PYTHON` already uses.
    """

    def test_the_render_is_exactly_the_tiling(self):
        from patchworks.graph import DEFAULT_SPEC
        from patchworks.sandbox.env import IMAGE_SIZE, PATCH_PX

        assert IMAGE_SIZE == DEFAULT_SPEC.patch_grid * PATCH_PX

    def test_a_patch_cell_stalk_is_one_patch_raw(self):
        """The world writes the stalk with no compressor in between (ADR-0006)."""
        from patchworks.graph import DEFAULT_SPEC
        from patchworks.sandbox.env import PATCH_PX, RENDER_CHANNELS

        assert DEFAULT_SPEC.patch_stalk == PATCH_PX * PATCH_PX * RENDER_CHANNELS

    def test_both_are_typed_derived_and_say_derived_from_what(self, surveys):
        """Leaving either `stipulated` would be knowingly filing a wrong row."""
        rows = {
            entry.name: entry
            for survey in surveys.values()
            for entry in survey.entries
            if entry.name in ("IMAGE_SIZE", "DomeSpec.patch_stalk")
        }
        assert {name: rows[name].type for name in rows} == {
            "IMAGE_SIZE": "derived",
            "DomeSpec.patch_stalk": "derived",
        }
        assert "patch_grid" in rows["IMAGE_SIZE"].depends_on
        assert "PATCH_PX" in rows["DomeSpec.patch_stalk"].depends_on


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

    def test_check_still_catches_a_stale_file(self, tmp_path, monkeypatch, capsys):
        """The half of `--check` the test above cannot see.

        A check that passes on a fresh tree proves the generator agrees with
        itself; this is the one that says the checked-in file is *read*. Written
        against a copy under `tmp_path`, so the real registers are untouched --
        and pinned now that class fields are in them, because a field's row is
        the sort of change most easily left ungenerated.
        """
        monkeypatch.setattr(registers, "ROOT", tmp_path)
        monkeypatch.setattr(registers, "OUTPUT", tmp_path / "registers")
        assert registers.main([]) == 0
        stale = tmp_path / "registers" / "architecture.md"
        stale.write_text(
            stale.read_text(encoding="utf-8").replace("`DomeSpec.interior_m`", "`m`"),
            encoding="utf-8",
        )
        capsys.readouterr()
        assert registers.main(["--check"]) == 1
        assert "architecture.md" in capsys.readouterr().err
        assert registers.main([]) == 0
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

    def test_a_marked_class_is_descended_into_and_its_fields_qualified(
        self, tmp_path, monkeypatch
    ):
        """The marker is the opt-in, and the row is named `Class.field`.

        Qualified because a field and a module-level constant may share a name,
        and because `IMAGE_SIZE`'s `@depends_on` has to be able to point at one.
        """
        survey = self._read(
            tmp_path,
            monkeypatch,
            "#: @register architecture\n"
            "@dataclass\n"
            "class Spec:\n"
            "    #: @type chosen\n"
            "    #: @flexibility unknown\n"
            "    #: @warrant here\n"
            "    width: int = 4\n"
            '    """Prose, below the field, exactly where it was."""\n',
        )
        assert [e.name for e in survey.entries] == ["Spec.width"]
        assert survey.entries[0].register == "architecture"
        assert survey.entries[0].value == "4"

    def test_the_marker_is_read_from_above_the_decorators(self, tmp_path, monkeypatch):
        """`class` is not the top line of a decorated class.

        Every class this reaches is a dataclass, so reading the run above
        `class` alone would find the decorator and never the marker.
        """
        survey = self._read(
            tmp_path,
            monkeypatch,
            "#: @register rig\n"
            "@dataclass\n"
            "@final\n"
            "class Spec:\n"
            "    #: @type chosen\n"
            "    #: @flexibility unknown\n"
            "    #: @warrant here\n"
            "    width: int = 4\n",
        )
        assert [e.register for e in survey.entries] == ["rig"]

    def test_a_field_may_name_its_own_register(self, tmp_path, monkeypatch):
        """Per-field placement, which is what splits `DomeSpec` in two."""
        survey = self._read(
            tmp_path,
            monkeypatch,
            "#: @register architecture\n"
            "@dataclass\n"
            "class Spec:\n"
            "    #: @register world-and-build\n"
            "    #: @type stipulated\n"
            "    #: @flexibility the world's\n"
            "    #: @warrant here\n"
            "    grid: int = 16\n"
            "    #: @type chosen\n"
            "    #: @flexibility unknown\n"
            "    #: @warrant here\n"
            "    depth: int = 5\n",
        )
        assert {e.name: e.register for e in survey.entries} == {
            "Spec.grid": "world-and-build",
            "Spec.depth": "architecture",
        }

    def test_a_field_of_a_marked_class_may_opt_out_but_not_be_silent(
        self, tmp_path, monkeypatch
    ):
        """Completeness inside the class, for the reason it holds at module level.

        Otherwise a sixteenth field arrives with no provenance and the register
        is quietly incomplete again -- the original disease with a register
        standing next to it.
        """
        survey = self._read(
            tmp_path,
            monkeypatch,
            "#: @register architecture\n"
            "@dataclass\n"
            "class Spec:\n"
            "    #: @register none\n"
            "    label: str = 'x'\n"
            "    later: int = 6\n",
        )
        assert survey.opted_out == [("sample.py", "Spec.label")]
        assert survey.unmarked == [("sample.py", "Spec.later")]

    def test_an_unmarked_class_is_not_scanned_and_not_flagged(
        self, tmp_path, monkeypatch
    ):
        """The opt-in's whole point, and the hole the architecture register names.

        Blanket descent would force `@register none` onto dozens of record
        fields, and noise in a completeness check is how a completeness check
        stops being read. The cost is that an unmarked class holding warranted
        numbers is invisible, which is stated rather than solved (#187).
        """
        survey = self._read(
            tmp_path,
            monkeypatch,
            "@dataclass\nclass Finding:\n    remedy: str = ''\n    width: int = 4\n",
        )
        assert survey.entries == []
        assert survey.opted_out == []
        assert survey.unmarked == []

    def test_a_class_marked_none_is_not_scanned(self, tmp_path, monkeypatch):
        """One key, one question everywhere it appears: which register, or none."""
        survey = self._read(
            tmp_path,
            monkeypatch,
            "#: @register none\n@dataclass\nclass Spec:\n    width: int = 4\n",
        )
        assert survey.entries == []
        assert survey.unmarked == []

    def test_a_class_marker_naming_no_register_is_refused(self, tmp_path, monkeypatch):
        """Otherwise its fields land in no register: checked, then never printed."""
        with pytest.raises(registers.MalformedProvenance, match="is not one of"):
            self._read(
                tmp_path,
                monkeypatch,
                "#: @register topology\n@dataclass\nclass Spec:\n    width: int = 4\n",
            )

    def test_a_field_override_naming_no_register_is_refused(
        self, tmp_path, monkeypatch
    ):
        with pytest.raises(registers.MalformedProvenance, match="is not one of"):
            self._read(
                tmp_path,
                monkeypatch,
                "#: @register architecture\n"
                "@dataclass\n"
                "class Spec:\n"
                "    #: @register topology\n"
                "    #: @type chosen\n"
                "    #: @flexibility unknown\n"
                "    #: @warrant here\n"
                "    width: int = 4\n",
            )
