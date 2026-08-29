"""The `patchworks` command: dispatch, `doctor`'s verdicts, `check`'s exit codes.

**No test here opens a window, and one of them is about a window.** The
`mjpython` decision is written as :func:`~patchworks.cli.window_plan`, which
returns a plan rather than acting on one, so every rule it applies can be
exercised — on this machine, and for platforms that are not this machine —
without a viewer ever being launched or a process ever being replaced.
:class:`TestNothingHereOpensAWindowOrInstallsAnything` makes that a claim the
suite holds rather than one this docstring makes: it breaks `os.execv` and
every way a subprocess can be started, and then runs the things that must not
use them.

**What rests on a human at the screen**, and is asserted nowhere:

* that `patchworks demo` actually opens a scene window, and that the window is
  usable once open. What is tested is the decision *before* it — which
  interpreter, or a refusal naming the command — and never the window.
* that the exec into `mjpython` succeeds. The plan is asserted and `os.execv`
  is asserted to be *called* with it, against a stand-in; the real exec is
  never performed, because it would replace the pytest process.

`patchworks check` is run here on the small dome (`tests/conftest.py`) against
a 16x16 render, for a handful of ticks. The full dome for three hundred ticks
is what the CLI's own defaults do and costs about four seconds; this file is in
CI on every push and takes the small one.
"""

import dataclasses
import os
import random
import runpy
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from patchworks import cli
from patchworks.graph import build_graph

# The small dome, shared (tests/conftest.py).
from conftest import SMALL

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

#: The render the small dome tiles, as `tests/test_assembled_loop.py` derives
#: it: 4x4 patch cells over 4x4 pixels each.
IMAGE_SIZE = 16

#: Enough ticks for the world to have moved and for every non-finite sweep to
#: have run on real arrays, and few enough that this file stays cheap. Nothing
#: here reads a magnitude, so there is no number this has to be large enough to
#: establish.
TICKS = 5


@pytest.fixture(scope="module")
def project():
    """`pyproject.toml`, parsed."""
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def a_finding(name="a check", passed=True, detail="what was seen", remedy=""):
    """A :class:`~patchworks.cli.Finding`, for tests about the report rather than a check."""
    return cli.Finding(name=name, passed=passed, detail=detail, remedy=remedy)


def a_liveness(**overrides):
    """A :class:`~patchworks.cli.Liveness` with plausible numbers, for format and exit-code tests."""
    fields = dict(
        ticks=300,
        seed=0,
        split="train",
        control_hz=50.0,
        commanded_mean_abs=0.295,
        applied_mean_abs=0.290,
        commanded_sd=(0.078, 0.129, 0.080),
        arm_travel=(2.26, 2.60, 2.60),
        puck_displacement=0.646,
        world_seconds=6.0,
        elapsed=1.9,
        non_finite=(),
    )
    fields.update(overrides)
    return cli.Liveness(**fields)


class TestTheOneEntryPoint:
    """Both routes reach the same dispatcher, and a bare command teaches rather than scolds."""

    def test_the_console_script_points_at_the_dispatcher(self, project):
        """`[project.scripts]` names a target that exists and is callable.

        The installed script itself cannot be asserted here: this repository's
        shared venv was installed before the entry point was declared, and a
        worktree has no console script at all. What *can* be held is that the
        declaration is not a dangling name — the failure mode where `pip
        install -e .` writes a `patchworks` that raises on the first line.
        """
        assert project["project"]["scripts"] == {"patchworks": "patchworks.cli:main"}
        module_name, _, attribute = "patchworks.cli:main".partition(":")
        import importlib

        assert callable(getattr(importlib.import_module(module_name), attribute))

    def test_python_m_patchworks_reaches_the_dispatcher_and_returns_its_exit_code(
        self, monkeypatch
    ):
        """`python -m patchworks` is `cli.main`, and its return value is the process's.

        Run through `runpy` rather than a subprocess so that the dispatcher can
        be replaced: what is being asserted is the *wiring* — that `__main__`
        calls this function and exits with what it returned — and a subprocess
        could only ever show that some exit code came back.
        """
        seen = []

        def dispatcher(argv=None):
            seen.append(argv)
            return 7

        monkeypatch.setattr(cli, "main", dispatcher)
        with pytest.raises(SystemExit) as exit:
            runpy.run_module("patchworks", run_name="__main__")
        assert exit.value.code == 7
        assert seen == [None], "the module route must not synthesise arguments of its own"

    def test_a_bare_command_prints_the_help_and_succeeds(self, capsys):
        """Typing `patchworks` is the discovery gesture, not a usage error.

        Exit 0 rather than argparse's 2: a human who does not yet know what
        there is types the command with nothing after it, and answering the
        question this entry point exists to answer with an error would be
        answering it with a rebuke.
        """
        assert cli.main([]) == 0
        printed = capsys.readouterr().out
        assert "usage: patchworks" in printed
        for subcommand in ("doctor", "check", "dome", "demo"):
            assert subcommand in printed

    def test_every_level_of_help_says_what_the_thing_is(self):
        """Not only what the flags do.

        Asked as a floor rather than by matching sentences: each description is
        prose of several lines that names the subcommand's *purpose*, and the
        cheapest way to lose that in an edit is to replace it with a flag list.
        """
        parser = cli.build_parser()
        assert "embodied graph architecture" in parser.description
        for description in (
            cli.DOCTOR_DESCRIPTION,
            cli.CHECK_DESCRIPTION,
            cli.DOME_DESCRIPTION,
            cli.DEMO_DESCRIPTION,
        ):
            assert len(description.splitlines()) >= 4
            assert description.strip()

    def test_an_unknown_subcommand_is_a_usage_error(self, capsys):
        with pytest.raises(SystemExit) as exit:
            cli.main(["nosuchthing"])
        assert exit.value.code == 2


class TestTheRecordAndTheCodeAgree:
    """`cli.py`'s copies of what `pyproject.toml` says, held against it.

    Two constants restate the packaging metadata because reading it at runtime
    would need either an installed distribution — which a worktree has not got
    — or a version-specifier parser, which needs `packaging`, which is not in
    the dependency set. Restating is fine as long as drifting is loud, and
    these are what make it loud.
    """

    def test_the_interpreter_bounds_are_requires_python(self, project):
        stated = project["project"]["requires-python"]
        assert stated == (
            f">={cli.MINIMUM_PYTHON[0]}.{cli.MINIMUM_PYTHON[1]},"
            f"<{cli.BELOW_PYTHON[0]}.{cli.BELOW_PYTHON[1]}"
        )

    def test_doctor_reports_every_declared_dependency(self, project):
        """The table `doctor` prints is `[project].dependencies`, entry for entry."""
        assert [requirement for _module, requirement in cli.DEPENDENCIES] == project[
            "project"
        ]["dependencies"]


class TestDoctorsChecks:
    """Each check, in both directions where there are two."""

    def test_this_interpreter_is_inside_the_bound(self):
        finding = cli.check_interpreter()
        assert finding.passed
        assert finding.remedy == "", "a passing check has nothing to tell anyone to do"
        assert ".".join(str(part) for part in sys.version_info[:3]) in finding.detail

    def test_an_interpreter_outside_the_bound_fails_and_says_what_to_do(
        self, monkeypatch
    ):
        monkeypatch.setattr(cli, "MINIMUM_PYTHON", (99, 0))
        monkeypatch.setattr(cli, "BELOW_PYTHON", (99, 1))
        finding = cli.check_interpreter()
        assert not finding.passed
        assert ">=99.0,<99.1" in finding.detail
        assert finding.remedy.endswith(cli.install_advice().one_line()), (
            "the remedy is whatever _venv_hint chose, with nothing composed onto it"
        )

    def test_the_interpreter_remedy_does_not_say_venv_twice(self, monkeypatch):
        """It used to prepend a venv command to one `_venv_hint` already carried.

        Outside a venv that printed `python3.12 -m venv .venv &&` twice in a
        row. Inside an out-of-bound one it was worse: make a new venv, then
        install into the old broken one with its `pip`. Only `_venv_hint` knows
        which case it is in, so only `_venv_hint` gets to answer.
        """
        monkeypatch.setattr(cli, "MINIMUM_PYTHON", (99, 0))
        monkeypatch.setattr(cli, "BELOW_PYTHON", (99, 1))
        for in_a_venv in (True, False):
            monkeypatch.setattr(cli, "in_a_virtual_environment", lambda: in_a_venv)
            remedy = cli.check_interpreter().remedy
            assert remedy.count("-m venv") <= 1, remedy
            assert remedy.count("pip install") == 1, remedy

    def test_the_dependencies_import_here_and_the_versions_are_reported_not_enforced(self):
        finding = cli.check_dependencies()
        assert finding.passed
        assert "torch" in finding.detail and "mujoco" in finding.detail
        assert "reported, not enforced" in finding.detail, (
            "the check does not resolve specifiers, and must not read as though it does"
        )

    def test_a_dependency_that_does_not_import_fails_and_names_the_install(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            cli,
            "DEPENDENCIES",
            (("torch", "torch==2.2.2"), ("no_such_module", "no-such-module==1.0")),
        )
        finding = cli.check_dependencies()
        assert not finding.passed
        assert "no-such-module==1.0" in finding.detail
        assert "torch" in finding.detail, "what did import is still worth printing"
        assert "pip" in finding.remedy

    def test_a_partial_install_is_still_reachable_by_the_dependency_check(self):
        """What `doctor` can still catch, given that it runs after the package imported.

        `patchworks/__init__.py` imports the architecture eagerly, so torch and
        numpy are already in by the time any of this runs and their checks can
        only pass. `mujoco` and `gymnasium` reach the package only through
        :mod:`patchworks.sandbox`, which `__init__` does not import — so those
        two are the ones a finding can genuinely be about, and this holds that
        that stays true. If `__init__` ever imports the sandbox, `doctor`'s
        dependency check stops being able to fail and this says so.
        """
        import ast

        source = Path(cli.__file__).parent / "__init__.py"
        imported = {
            f"patchworks.{alias.name}"
            for node in ast.walk(ast.parse(source.read_text()))
            if isinstance(node, ast.ImportFrom) and node.module is None
            for alias in node.names
        }
        assert "patchworks.sandbox" not in imported, (
            "the package's front door now pulls MuJoCo and Gymnasium, so doctor's "
            "dependency check can no longer report either as missing -- it would "
            "have crashed before running. Say so, or make the front door lazy."
        )

    def test_the_sandbox_imports(self):
        finding = cli.check_package()
        assert finding.passed
        assert "patchworks.sandbox" in finding.detail

    def test_it_asks_only_about_the_module_that_can_actually_be_missing(self):
        """`patchworks.graph` used to be checked here and could never fail.

        `patchworks/__init__.py` imports it, so by the time this runs it is in
        or the process never started. A line that cannot fail is not a check,
        and its remedy named causes -- a missing install, an unset `PYTHONPATH`
        -- that could not be the reason for the one failure it could report.
        """
        assert "patchworks.graph" not in cli.check_package().detail

    def test_a_missing_sandbox_fails_and_names_the_cause_that_is_possible(
        self, monkeypatch
    ):
        def refuse(name):
            raise ModuleNotFoundError(f"No module named {name!r}")

        monkeypatch.setattr(cli.importlib, "import_module", refuse)
        finding = cli.check_package()
        assert not finding.passed
        assert "patchworks.sandbox" in finding.detail
        for possible in ("MuJoCo", "Gymnasium"):
            assert possible in finding.remedy, (
                "a broken one of those is the only way this check can fail"
            )
        assert "PYTHONPATH" not in finding.remedy, (
            "an unresolvable package would have stopped `cli` importing, so it "
            "cannot be the cause here"
        )

    def test_the_gl_check_renders_rather_than_reading_a_variable(self):
        """It really makes a context and really renders, offscreen.

        The agent reads a rendered image every tick, so a GL path that cannot
        make a context is a run that cannot happen — and the only honest way to
        report on that is to have done it. `mujoco.Renderer` is offscreen, so
        this opens no window.
        """
        finding = cli.check_mujoco_gl()
        assert finding.passed
        assert "(64, 64, 3)" in finding.detail
        assert "MUJOCO_GL=" in finding.detail

    def test_a_gl_failure_is_reported_as_a_gl_failure(self, monkeypatch):
        import mujoco

        def refuse(*_args, **_kwargs):
            raise RuntimeError("could not create an OpenGL context")

        monkeypatch.setattr(mujoco, "Renderer", refuse)
        # Forced, because this file is itself run inside the image (`docker run
        # --entrypoint pytest ...`), where the answer is otherwise True and the
        # remedy is the other one below.
        monkeypatch.setattr(cli, "in_a_container", lambda: False)
        finding = cli.check_mujoco_gl()
        assert not finding.passed
        assert "could not create an OpenGL context" in finding.detail
        assert "MUJOCO_GL=osmesa" in finding.remedy

    def test_the_gl_remedy_in_a_container_does_not_send_anyone_after_a_driver(
        self, monkeypatch
    ):
        """The same failure, worded for where it is being read.

        ADR-0012: the host remedy -- install osmesa, or check the platform's GL
        drivers -- is advice for a host, and a human reading it inside the image
        would go looking for a driver that is not the problem. The image already
        has osmesa, so what is left to say is which tag and which backend.
        """
        import mujoco

        def refuse(*_args, **_kwargs):
            raise RuntimeError("could not create an OpenGL context")

        monkeypatch.setattr(mujoco, "Renderer", refuse)
        monkeypatch.setattr(cli, "in_a_container", lambda: True)
        finding = cli.check_mujoco_gl()
        assert not finding.passed
        assert "container" in finding.remedy
        assert "driver" not in finding.remedy, (
            "there is no host GL driver to go and look at from in here"
        )
        assert "MUJOCO_GL" in finding.remedy, (
            "the one thing that can be wrong in an image that ships osmesa"
        )


    def test_mjpython_is_not_wanted_off_macos(self):
        finding = cli.check_mjpython(platform="linux")
        assert finding.passed
        assert "not required on linux" in finding.detail

    def test_mjpython_missing_on_macos_fails_and_names_where_it_comes_from(
        self, monkeypatch
    ):
        monkeypatch.setattr(cli, "mjpython_launcher", lambda _executable: None)
        finding = cli.check_mjpython(platform="darwin")
        assert not finding.passed
        assert "mujoco" in finding.remedy, "it ships with the wheel, and that is the fix"

    def test_mjpython_present_on_macos_passes_and_says_it_is_handled(self, monkeypatch):
        monkeypatch.setattr(
            cli, "mjpython_launcher", lambda _executable: Path("/somewhere/mjpython")
        )
        finding = cli.check_mjpython(platform="darwin")
        assert finding.passed
        assert "/somewhere/mjpython" in finding.detail
        assert "re-execs into it by itself" in finding.detail


class TestTheContainerIsNoticedForWordingAndNothingElse:
    """ADR-0012's last consequence, held down.

    Being in a container is not a failure, and every line `doctor` prints is an
    observation with a verdict -- so the detection changes what one remedy says
    and must change nothing else about the report.
    """

    def test_a_runtimes_marker_file_is_what_is_looked_at(self, tmp_path):
        marker = tmp_path / ".dockerenv"
        assert not cli.in_a_container(markers=(marker,))
        marker.touch()
        assert cli.in_a_container(markers=(marker,))

    def test_both_docker_and_podman_leave_one(self):
        assert set(cli.CONTAINER_MARKERS) == {
            Path("/.dockerenv"),
            Path("/run/.containerenv"),
        }

    def test_it_adds_no_finding_and_no_line(self, monkeypatch):
        monkeypatch.setattr(cli, "in_a_container", lambda: True)
        inside = [finding.name for finding in cli.diagnose(platform="linux")]
        monkeypatch.setattr(cli, "in_a_container", lambda: False)
        outside = [finding.name for finding in cli.diagnose(platform="linux")]
        assert inside == outside
        assert not any("container" in name for name in inside)

    def test_a_container_is_not_by_itself_a_failure(self, monkeypatch):
        monkeypatch.setattr(cli, "in_a_container", lambda: True)
        assert all(finding.passed for finding in cli.diagnose(platform="linux")), (
            "this suite runs inside the image, where every check passes"
        )


class TestAnUnusableGLBackendIsAFindingNotATraceback:
    """#125's surviving criterion, verified by forcing the failure.

    `MUJOCO_GL` is read when `mujoco` is imported, so the only way to force
    this one is to start a process with the variable already wrong -- which is
    also exactly how a human hits it. What is asserted is the shape of the
    failure and not its wording: a report, a named check, a remedy, an exit
    code, and no traceback.
    """

    def test_doctor_reports_it_and_exits_one(self):
        done = subprocess.run(
            [sys.executable, "-m", "patchworks", "doctor"],
            cwd=ROOT,
            env={**os.environ, "MUJOCO_GL": "not-a-backend"},
            capture_output=True,
            text=True,
        )
        assert done.returncode == 1, done.stderr
        assert "Traceback" not in done.stderr, done.stderr
        assert "NOT READY" in done.stdout
        assert "->" in done.stdout, "a failure is rendered with its remedy"


class TestTheInstallCommandDoctorPrints:
    """The fresh clone is what this entry point is for, so its case is the one to get right."""

    def test_inside_a_venv_it_names_that_venvs_pip(self, monkeypatch):
        monkeypatch.setattr(cli, "in_a_virtual_environment", lambda: True)
        monkeypatch.setattr(sys, "prefix", "/somewhere/.venv")
        advice = cli.install_advice()
        assert advice.command == "/somewhere/.venv/bin/pip install -e '.[dev]'"
        assert advice.note == "", "there is nothing to explain about the ordinary case"

    def test_outside_one_it_says_to_make_one_first(self, monkeypatch):
        """Naming this interpreter's pip would say to install torch into the system Python."""
        monkeypatch.setattr(cli, "in_a_virtual_environment", lambda: False)
        advice = cli.install_advice()
        assert "-m venv .venv" in advice.command
        assert ".venv/bin/pip install -e '.[dev]'" in advice.command
        assert "not in a virtual environment" in advice.note

    def test_the_command_is_a_command_and_the_note_is_not_in_it(self, monkeypatch):
        """Because a refusal indents the command as a block for someone to copy.

        Folded together, that block ended in
        `(this interpreter is not in a virtual environment)` -- prose, inside
        something a fresh-clone reader was being invited to paste into a shell.
        """
        for in_a_venv in (True, False):
            monkeypatch.setattr(cli, "in_a_virtual_environment", lambda: in_a_venv)
            advice = cli.install_advice()
            assert "(" not in advice.command and ")" not in advice.command
            assert advice.one_line().startswith(advice.command)

    def test_the_venv_test_is_the_stdlibs_own(self, monkeypatch):
        monkeypatch.setattr(sys, "prefix", "/a")
        monkeypatch.setattr(sys, "base_prefix", "/a")
        assert not cli.in_a_virtual_environment()
        monkeypatch.setattr(sys, "prefix", "/a/.venv")
        assert cli.in_a_virtual_environment()

    def test_the_python_it_names_is_inside_the_bound_for_every_bound(self, monkeypatch):
        """Including bounds this project does not have yet.

        It used to take the major from `MINIMUM_PYTHON` and the minor from
        `BELOW_PYTHON`, which is two halves of two different numbers: against a
        `<4.0` it printed `python3.-1`. The old test pinned the bounds back to
        today's, so it could not have seen that. This one sweeps.
        """
        for minimum, below in (((3, 11), (3, 13)), ((3, 11), (4, 0)), ((3, 9), (3, 10))):
            monkeypatch.setattr(cli, "MINIMUM_PYTHON", minimum)
            monkeypatch.setattr(cli, "BELOW_PYTHON", below)
            named = cli.lowest_supported_python()
            major, _, minor = named.removeprefix("python").partition(".")
            assert minimum <= (int(major), int(minor)) < below, named


class TestDoctorsVerdict:
    """What the report says, and what the exit code is."""

    def test_all_passing_is_ready_and_exits_zero(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "diagnose", lambda: [a_finding(), a_finding("another")])
        assert cli._doctor(None) == 0
        printed = capsys.readouterr().out
        assert "READY: all 2 checks above passed" in printed
        assert "NOT READY" not in printed

    def test_one_failing_is_not_ready_names_it_and_exits_one(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cli,
            "diagnose",
            lambda: [
                a_finding("interpreter"),
                a_finding("mujoco gl", passed=False, remedy="set MUJOCO_GL=osmesa"),
            ],
        )
        assert cli._doctor(None) == 1
        printed = capsys.readouterr().out
        assert "NOT READY: 1 of 2 checks failed (mujoco gl)" in printed
        assert "set MUJOCO_GL=osmesa" in printed

    def test_the_verdict_prints_what_it_did_not_check(self):
        """A "ready" that reads as covering everything is the overstatement to avoid.

        The failure mode this project keeps finding against itself is prose
        claiming a guarantee it has not got, and a bare "READY" over five checks
        is exactly that shape. So the unchecked classes are printed beside the
        verdict — including on the passing path, which is the one where it
        matters.
        """
        report = cli.format_diagnosis([a_finding()])
        assert "not checked here:" in report
        assert "rests on a human at the screen" in report
        assert "READY" in report

    def test_a_failure_is_rendered_with_its_remedy_and_a_pass_without_one(self):
        failed = a_finding("gl", passed=False, detail="no context", remedy="do this")
        assert "FAIL" in failed.render() and "do this" in failed.render()
        passed = a_finding("gl", detail="a context")
        assert "ok" in passed.render() and "\n" not in passed.render()

    def test_the_real_diagnosis_runs_every_check_even_past_a_failure(self, monkeypatch):
        """A bug report wants the whole picture, not the first thing that broke."""
        monkeypatch.setattr(cli, "MINIMUM_PYTHON", (99, 0))
        monkeypatch.setattr(cli, "BELOW_PYTHON", (99, 1))
        findings = cli.diagnose(platform="linux")
        assert [finding.name for finding in findings] == [
            "interpreter",
            "dependencies",
            "package",
            "mujoco gl",
            "mjpython",
        ]
        assert not findings[0].passed
        assert all(finding.passed for finding in findings[1:])


class TestTheMjpythonProblem:
    """Exec the right interpreter, or refuse with the exact command.

    Every rule is exercised through :func:`~patchworks.cli.window_plan`, whose
    inputs are all arguments, so these run identically on any platform and none
    of them launches anything or replaces this process.
    """

    def test_off_macos_the_module_runs_under_this_interpreter(self):
        assert not cli.needs_mjpython("linux")
        plan = cli.window_plan(
            "a.module",
            ("--seed=3",),
            spoken_as="patchworks demo",
            platform="linux",
            executable="/v/bin/python",
        )
        assert plan.reexec == ("/v/bin/python", "-m", "a.module", "--seed=3")
        assert not plan.refusal

    def test_macos_needs_the_launcher(self):
        assert cli.needs_mjpython("darwin")

    def test_macos_with_a_launcher_execs_into_it(self):
        plan = cli.window_plan(
            "a.module",
            ("--seed=3",),
            spoken_as="patchworks demo",
            platform="darwin",
            launcher=Path("/venv/bin/mjpython"),
        )
        assert plan.reexec == ("/venv/bin/mjpython", "-m", "a.module", "--seed=3")
        assert not plan.refusal

    def test_macos_with_no_launcher_refuses_with_the_exact_command(self):
        plan = cli.window_plan(
            "a.module",
            ("--seed=3",),
            spoken_as="patchworks demo",
            platform="darwin",
            launcher=None,
            executable="/v/bin/python",
        )
        assert not plan.reexec
        assert "mjpython -m a.module --seed=3" in plan.refusal
        assert "/v/bin/python" in plan.refusal, "and where it looked"
        assert "pip" in plan.refusal, "and how to get one in the first place"
        assert "main thread" in plan.refusal, "and why, so it is not folklore"

    def test_the_refusal_leads_with_the_command_the_human_typed(self):
        """Not with the module path, which they have never seen and do not need.

        Someone who ran `patchworks demo` should not have to work out that
        `patchworks.surface.gestures` is the same thing. The module path is
        still offered, below, for a worktree with no console script installed.
        """
        refusal = cli.window_plan(
            "patchworks.surface.gestures",
            (),
            spoken_as="patchworks demo",
            platform="darwin",
            launcher=None,
            executable="/v/bin/python",
        ).refusal
        first, _, rest = refusal.partition("\n")
        assert "patchworks demo" in first
        assert "patchworks.surface.gestures" not in first
        assert "Then `patchworks demo` will work" in rest
        assert "mjpython -m patchworks.surface.gestures" in rest

    def test_demo_refuses_a_bad_split_before_replacing_the_process(
        self, monkeypatch, capsys
    ):
        """Past the exec there is no process left to report anything.

        `gestures.main` builds the world from `--split` too, so a typo used to
        surface as a `ValueError` traceback out of whatever `mjpython`
        inherited -- the worse version of the traceback `check` no longer has,
        because the process the human was talking to is already gone.
        """

        def must_not_exec(*_args, **_kwargs):
            raise AssertionError("a bad split must be refused before the exec")

        monkeypatch.setattr(cli, "open_window", must_not_exec)
        arguments = cli.build_parser().parse_args(["demo", "--split", "trian"])
        assert arguments.handler(arguments) == 2
        captured = capsys.readouterr()
        assert "trian" in captured.err and "train" in captured.err
        assert captured.out == ""

    def test_a_launcher_that_will_not_run_refuses_rather_than_raising(
        self, monkeypatch, capsys
    ):
        """Found, and still broken: stale, not executable, or the wrong architecture.

        `--help` promises a human never has to know about `mjpython`. An
        `OSError` traceback out of `os.execv` is that promise failing in the one
        place it is meant to hold.
        """
        monkeypatch.setattr(
            cli, "window_plan", lambda *a, **k: cli.WindowPlan(reexec=("/v/mjpython", "-m", "m"))
        )

        def refuses_to_exec(*_args):
            raise OSError(8, "Exec format error")

        monkeypatch.setattr(cli.os, "execv", refuses_to_exec)
        assert cli.open_window("m", (), spoken_as="patchworks demo") == 1
        captured = capsys.readouterr()
        assert "/v/mjpython" in captured.err
        assert "Exec format error" in captured.err
        assert "pip install" in captured.err, "and what to do about it"
        assert captured.out == ""

    def test_the_refusal_block_is_a_command_someone_can_paste(self, monkeypatch):
        """Outside a venv the advice carries a note, and a note is not a command."""
        monkeypatch.setattr(cli, "in_a_virtual_environment", lambda: False)
        refusal = cli.window_plan(
            "a.module",
            (),
            spoken_as="patchworks demo",
            platform="darwin",
            launcher=None,
            executable="/v/bin/python",
        ).refusal
        indented = [
            line.strip() for line in refusal.splitlines() if line.startswith("    ")
        ]
        assert indented, "the refusal offers commands as an indented block"
        for line in indented:
            assert not line.endswith(")"), f"not a runnable command: {line}"
        assert "not in a virtual environment" in refusal, "the note is still said"

    def test_the_demo_subcommand_says_its_own_name_in_a_refusal(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            cli,
            "open_window",
            lambda module, argv, *, spoken_as: seen.update(spoken_as=spoken_as) or 0,
        )
        arguments = cli.build_parser().parse_args(["demo"])
        arguments.handler(arguments)
        assert seen == {"spoken_as": "patchworks demo"}

    def test_launcher_none_means_there_is_none_not_go_and_look(self, monkeypatch):
        """The defect the suite found while this file was being written.

        `None` is `mjpython_launcher`'s answer for *there is no launcher*, so it
        cannot also be the not-given default: a test for the refusing path
        passed `launcher=None` and got an exec plan back, because the real
        launcher on this machine had been looked up behind its back.

        The omitted half stubs the lookup rather than calling it. CI is
        `ubuntu-latest` and `mjpython` ships only in MuJoCo's macOS wheel, so a
        real lookup here would find nothing and this file would be green on the
        development laptop and red on every push -- which is how a test ends up
        asserting the machine it ran on.
        """
        assert cli.window_plan(
            "a.module", (), spoken_as="patchworks demo", platform="darwin", launcher=None
        ).refusal
        monkeypatch.setattr(cli, "mjpython_launcher", lambda _executable: Path("/v/mjpython"))
        assert cli.window_plan(
            "a.module", (), spoken_as="patchworks demo", platform="darwin"
        ).reexec, (
            "and omitting it still looks one up"
        )

    def test_the_demo_hands_off_rather_than_importing_the_surface(self, monkeypatch):
        """The edge `TestTheArchitectureDoesNotImportTheSurface` forbids is never created.

        That guard (`tests/test_dome_panel.py`) is what stops a learning rule
        reaching for a measured persistence, and it is written as a whitelist of
        one directory. Rather than widen it for a dispatcher -- a decision about
        its scope that #119 does not carry -- `demo` execs the module that is
        already the demo's own entry point. So the import edge does not exist
        rather than being excused, and this asserts the *behaviour* that follows
        from that, not only the absence of a line.
        """
        seen = []
        monkeypatch.setattr(
            cli,
            "open_window",
            lambda module, argv, *, spoken_as: seen.append((module, argv)) or 0,
        )
        monkeypatch.delitem(sys.modules, "patchworks.surface.gestures", raising=False)
        arguments = cli.build_parser().parse_args(["demo", "--seed=2"])
        assert arguments.handler(arguments) == 0
        assert seen == [("patchworks.surface.gestures", ("--ticks=100000", "--seed=2", "--split=train"))]
        assert "patchworks.surface.gestures" not in sys.modules

    def test_an_exec_plan_is_carried_out_and_never_returns(self, monkeypatch):
        """`os.execv` replaces the process, so there is no exit code on this path."""
        monkeypatch.setattr(
            cli, "window_plan", lambda *a, **k: cli.WindowPlan(reexec=("/bin/x", "-m", "m"))
        )
        execed = []

        def fake_execv(path, argv):
            execed.append((path, argv))
            raise SystemExit(0)  # stands in for the process being replaced

        monkeypatch.setattr(cli.os, "execv", fake_execv)
        with pytest.raises(SystemExit):
            cli.open_window("m", (), spoken_as="patchworks demo")
        assert execed == [("/bin/x", ["/bin/x", "-m", "m"])]

    def test_a_refusal_goes_to_stderr_and_exits_one_without_exec(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cli, "window_plan", lambda *a, **k: cli.WindowPlan(refusal="run this instead")
        )

        def must_not_exec(*_args):
            raise AssertionError("a refused window must not exec anything anyway")

        monkeypatch.setattr(cli.os, "execv", must_not_exec)
        assert cli.open_window("m", (), spoken_as="patchworks demo") == 1
        captured = capsys.readouterr()
        assert "run this instead" in captured.err
        assert captured.out == ""

    def test_the_launcher_is_looked_for_beside_this_interpreter_first(self, monkeypatch):
        """It embeds an interpreter, so one from another venv would run another install."""
        beside = Path(sys.executable).parent / cli.MJPYTHON
        monkeypatch.setattr(Path, "exists", lambda self: self == beside)
        monkeypatch.setattr(cli.shutil, "which", lambda _name: "/elsewhere/mjpython")
        assert cli.mjpython_launcher() == beside

    def test_path_is_the_fallback_and_absence_is_none(self, monkeypatch):
        monkeypatch.setattr(Path, "exists", lambda self: False)
        monkeypatch.setattr(cli.shutil, "which", lambda _name: "/elsewhere/mjpython")
        assert cli.mjpython_launcher() == Path("/elsewhere/mjpython")
        monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
        assert cli.mjpython_launcher() is None


@pytest.fixture(scope="module")
def liveness():
    """One real run, shared: the small dome, a 16x16 render, a handful of ticks."""
    return cli.measure_liveness(
        ticks=TICKS, seed=0, dome=build_graph(SMALL), image_size=IMAGE_SIZE
    )


class TestCheckMeasuresAndExits:
    """`patchworks check`: the numbers, and the exit code that hangs off them."""

    def test_a_real_run_is_finite_and_reports_the_control_rate(self, liveness):
        assert liveness.non_finite == ()
        assert liveness.control_hz == pytest.approx(50.0)
        assert liveness.ticks == TICKS
        assert liveness.world_seconds == pytest.approx(TICKS / 50.0)

    def test_it_reports_a_torque_and_a_travel_per_joint(self, liveness):
        assert len(liveness.commanded_sd) == 3
        assert len(liveness.arm_travel) == 3
        assert liveness.commanded_mean_abs >= 0.0
        assert liveness.applied_mean_abs >= 0.0
        assert liveness.puck_displacement >= 0.0

    def test_travel_is_cumulative_not_net_displacement(self, monkeypatch):
        """The number the "arm at ~0 is broken" verdict hangs on has to mean travel.

        An arm that swings out and back ends where it started. Net displacement
        calls that motionless and tells a working installation it is broken, so
        travel accumulates |Δq| across the ticks instead. Held by driving the
        joints out and back through a real run: net is ~0, travel is not.
        """
        import dataclasses

        import patchworks.agent

        real_run = patchworks.agent.run
        swing = [0.0, 0.4, 0.8, 0.4, 0.0]
        assert len(swing) == TICKS, "the swing must return to where it started"

        def wrapped(agent, ticks, *, seed=None):
            def ticking():
                for index, outcome in enumerate(real_run(agent, ticks, seed=seed)):
                    observation = dict(outcome.observation)
                    observation["qpos"] = np.full(3, swing[index], dtype=np.float32)
                    yield dataclasses.replace(outcome, observation=observation)

            return ticking()

        monkeypatch.setattr(patchworks.agent, "run", wrapped)
        liveness = cli.measure_liveness(
            ticks=TICKS, seed=0, dome=build_graph(SMALL), image_size=IMAGE_SIZE
        )
        assert swing[0] == swing[-1], "so net displacement is exactly zero"
        # Out and back: 0.4 + 0.4 + 0.4 + 0.4 per joint.
        assert liveness.arm_travel == pytest.approx((1.6, 1.6, 1.6), abs=1e-6)

    def test_the_puck_figure_is_a_distance_not_a_rotation(self, monkeypatch):
        """A puck's third coordinate is radians and does not belong in a maximum of metres.

        `puck_pose` returns `(x, y, theta)`. A puck spun in place has not moved
        anywhere, and reporting its angle as "PUCK moved" would read as a large
        distance -- so a spin of two radians with no translation reports zero.
        """
        from patchworks.sandbox.env import PlanarPushSandbox

        spins = {"calls": 0}

        def spinning_in_place(self, i):
            spins["calls"] += 1
            # x and y never change; only the angle does.
            return np.array([0.1, 0.2, 2.0 * spins["calls"]])

        monkeypatch.setattr(PlanarPushSandbox, "puck_pose", spinning_in_place)
        liveness = cli.measure_liveness(
            ticks=TICKS, seed=0, dome=build_graph(SMALL), image_size=IMAGE_SIZE
        )
        assert liveness.puck_displacement == pytest.approx(0.0)

    def test_the_commanded_and_applied_torques_are_reported_apart(self):
        """`command` is pre-clip and `applied` is not, so one number cannot label both."""
        printed = cli.format_liveness(a_liveness())
        assert "torque |mean| asked  : 0.295   (pre-clip" in printed
        assert "torque |mean| applied: 0.290   (post-clip; 1 = saturated)" in printed

    def test_each_torque_figure_is_measured_from_its_own_field(self, monkeypatch):
        """And measured from it, not both from `command`.

        A gap between the two is the graph asking for more torque than the arm
        will give, which is the thing worth seeing -- so a `check` that read
        `command` twice would report saturation as never happening. Held by
        driving a run where the two differ by construction.
        """
        import patchworks.agent

        real_run = patchworks.agent.run

        def wrapped(agent, ticks, *, seed=None):
            def ticking():
                for outcome in real_run(agent, ticks, seed=seed):
                    yield dataclasses.replace(
                        outcome, command=np.full(3, 4.0), applied=np.full(3, 1.0)
                    )

            return ticking()

        monkeypatch.setattr(patchworks.agent, "run", wrapped)
        liveness = cli.measure_liveness(
            ticks=TICKS, seed=0, dome=build_graph(SMALL), image_size=IMAGE_SIZE
        )
        assert liveness.commanded_mean_abs == pytest.approx(4.0)
        assert liveness.applied_mean_abs == pytest.approx(1.0)

    def test_the_puck_and_arm_lines_carry_their_units(self):
        printed = cli.format_liveness(a_liveness())
        assert "ARM  travelled (rad)" in printed and "cumulative" in printed
        assert "PUCK moved (m)" in printed

    def test_the_same_seed_gives_the_same_run(self):
        """Seeded explicitly, so a bug report's numbers are somebody else's numbers too."""
        first, second = (
            cli.measure_liveness(
                ticks=TICKS, seed=1, dome=build_graph(SMALL), image_size=IMAGE_SIZE
            )
            for _ in range(2)
        )
        assert first.commanded_mean_abs == second.commanded_mean_abs
        assert first.arm_travel == second.arm_travel

    def test_nothing_is_drawn_from_a_global_rng(self):
        """Running the CLI changes no trajectory of anything run after it.

        The demo surface's rule (#77), applied here: a global draw would mean
        that whether someone had run `patchworks check` in this process decided
        the parameters of the next default-constructed `Sheaf`. #93 shipped
        exactly that bug on the panel, so it is asserted rather than intended —
        against all three global generators, since the code that must not draw
        does not get to pick which one it would have drawn from.
        """
        before = (
            torch.random.get_rng_state().clone(),
            np.random.get_state(),
            random.getstate(),
        )
        cli.measure_liveness(
            ticks=TICKS, seed=0, dome=build_graph(SMALL), image_size=IMAGE_SIZE
        )
        after = (torch.random.get_rng_state(), np.random.get_state(), random.getstate())
        assert torch.equal(before[0], after[0]), "torch's global generator moved"
        assert before[1][1].tolist() == after[1][1].tolist(), "numpy's global state moved"
        assert before[2] == after[2], "the stdlib's global random state moved"

    def test_a_finite_run_prints_the_numbers_and_exits_zero(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "measure_liveness", lambda **_: a_liveness())
        arguments = cli.build_parser().parse_args(["check"])
        assert arguments.handler(arguments) == 0
        printed = capsys.readouterr().out
        assert "control rate         : 50 Hz" in printed
        assert "non-finite           : none" in printed
        assert "untrained. Expected." in printed

    def test_anything_non_finite_exits_one_and_says_where(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cli,
            "measure_liveness",
            lambda **_: a_liveness(non_finite=("command at tick 12",)),
        )
        arguments = cli.build_parser().parse_args(["check"])
        assert arguments.handler(arguments) == 1
        printed = capsys.readouterr().out
        assert "NON-FINITE           : command at tick 12" in printed
        assert "That is a bug, not an untrained agent" in printed

    def test_the_report_carries_what_a_bug_report_needs(self):
        """Versions and platform above the numbers, because that is what it is for."""
        printed = cli.format_liveness(a_liveness())
        assert "Python" in printed and "torch" in printed and "mujoco" in printed
        assert "MUJOCO_GL=" in printed

    def test_a_bad_split_is_a_usage_error_not_a_traceback(self, capsys):
        """`--help` promises an exit code on what was measured; a traceback is not one."""
        arguments = cli.build_parser().parse_args(["check", "--split", "trian"])
        assert arguments.handler(arguments) == 2
        captured = capsys.readouterr()
        assert "trian" in captured.err
        assert "train" in captured.err, "and the message names what is allowed"
        assert captured.out == ""

    def test_the_split_is_refused_before_the_run_rather_than_caught_after(
        self, monkeypatch, capsys
    ):
        """So that a `ValueError` three hundred ticks deep is not called a bad argument.

        Wrapping the run in `except ValueError` was the first fix and the wrong
        one: exit 2 means "the arguments were wrong", and any refusal from
        inside the tick -- a `Sheaf` guard, a broadcast mismatch -- would have
        been reported as that, with the traceback and the environment line both
        lost. The one command whose purpose is a legible bug report would have
        swallowed exactly the failure it exists to capture.
        """

        def blows_up(**_kwargs):
            raise ValueError("something went wrong three hundred ticks in")

        monkeypatch.setattr(cli, "measure_liveness", blows_up)
        arguments = cli.build_parser().parse_args(["check"])
        with pytest.raises(ValueError, match="three hundred ticks in"):
            arguments.handler(arguments)

    def test_a_run_of_no_ticks_is_refused_rather_than_reported(self, capsys):
        """`range(0)` and `range(-5)` are both empty, and the verdict is not.

        A run of no ticks printed an empty spread, an empty travel, and then
        "Arm at ~0 = something is genuinely wrong" -- the line a human pastes
        into a bug report, saying the installation is broken about a run that
        never happened.
        """
        for ticks in ("0", "-5"):
            arguments = cli.build_parser().parse_args(["check", "--ticks", ticks])
            assert arguments.handler(arguments) == 2
            captured = capsys.readouterr()
            assert "--ticks must be at least 1" in captured.err
            assert captured.out == ""

    def test_the_flags_reach_the_measurement(self, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr(
            cli, "measure_liveness", lambda **kwargs: seen.update(kwargs) or a_liveness()
        )
        arguments = cli.build_parser().parse_args(
            ["check", "--ticks", "7", "--seed", "11", "--split", "heldout_pair"]
        )
        arguments.handler(arguments)
        assert seen == {"ticks": 7, "seed": 11, "split": "heldout_pair"}

    def test_the_report_names_every_place_that_went_non_finite(self):
        liveness = a_liveness(non_finite=("command, first at tick 0", "world qvel"))
        printed = cli.format_liveness(liveness)
        assert "command, first at tick 0" in printed
        assert "world qvel" in printed


class TestTheNonFiniteSweepActuallyFinds:
    """The sweep exercised on values that really are non-finite.

    Written because the exit-code tests above stand on a fake
    :class:`~patchworks.cli.Liveness`: they hold that a non-empty `non_finite`
    exits 1, and would go on passing if the sweep that fills it had been
    deleted. These inject a NaN into each of the three places the sweep looks
    and assert it comes back named.

    The injection is at `patchworks.agent.run`, wrapping the real one, so the
    world, the agent and the tick are all the real thing and only the value is
    a lie.
    """

    @staticmethod
    def measure_with(monkeypatch, corrupt):
        """Run the real thing with `corrupt` wrapped around `agent.run`."""
        import patchworks.agent

        real_run = patchworks.agent.run

        def wrapped(agent, ticks, *, seed=None):
            return corrupt(agent, real_run(agent, ticks, seed=seed))

        monkeypatch.setattr(patchworks.agent, "run", wrapped)
        return cli.measure_liveness(
            ticks=TICKS, seed=0, dome=build_graph(SMALL), image_size=IMAGE_SIZE
        )

    def test_a_non_finite_command_is_found_and_counted(self, monkeypatch):
        def corrupt(_agent, ticking):
            for outcome in ticking:
                yield dataclasses.replace(
                    outcome, command=np.full_like(np.asarray(outcome.command), np.nan)
                )

        liveness = self.measure_with(monkeypatch, corrupt)
        assert len(liveness.non_finite) == 1, "one line per place, not one per tick"
        (named,) = liveness.non_finite
        assert named.startswith("command,")
        assert "first at tick 0" in named
        assert f"on {TICKS} of {TICKS} ticks" in named

    def test_a_non_finite_observation_is_found_and_named_by_key(self, monkeypatch):
        def corrupt(_agent, ticking):
            for index, outcome in enumerate(ticking):
                if index != 1:
                    yield outcome
                    continue
                observation = dict(outcome.observation)
                key = next(
                    name
                    for name, value in observation.items()
                    if np.asarray(value).dtype.kind == "f"
                )
                observation[key] = np.full_like(np.asarray(observation[key]), np.inf)
                yield dataclasses.replace(outcome, observation=observation)

        liveness = self.measure_with(monkeypatch, corrupt)
        (named,) = liveness.non_finite
        assert named.startswith("observation[")
        assert "first at tick 1" in named
        assert f"on 1 of {TICKS} ticks" in named

    def test_a_non_finite_world_state_is_found_after_the_run(self, monkeypatch):
        def corrupt(agent, ticking):
            yield from ticking
            agent.env.unwrapped.data.qvel[0] = np.nan

        liveness = self.measure_with(monkeypatch, corrupt)
        assert liveness.non_finite == ("world qvel, at the end of the run",)

    def test_the_exit_code_follows_a_real_sweep_end_to_end(self, monkeypatch, capsys):
        """No fake anywhere on this path: a real run, a real NaN, a real 1."""

        def corrupt(_agent, ticking):
            for outcome in ticking:
                yield dataclasses.replace(
                    outcome, command=np.full_like(np.asarray(outcome.command), np.nan)
                )

        liveness = self.measure_with(monkeypatch, corrupt)
        monkeypatch.setattr(cli, "measure_liveness", lambda **_: liveness)
        arguments = cli.build_parser().parse_args(["check"])
        assert arguments.handler(arguments) == 1
        assert "NON-FINITE" in capsys.readouterr().out


class TestDomeIsPreservedNotReimplemented:
    """`patchworks dome` prints what `python -m patchworks` used to."""

    def test_it_is_the_construction_report_verbatim(self, capsys):
        arguments = cli.build_parser().parse_args(["dome"])
        assert arguments.handler(arguments) == 0
        assert capsys.readouterr().out == build_graph().report() + "\n"

    def test_it_routes_through_graphs_own_entry_point(self, monkeypatch):
        """Routed rather than reimplemented, so the two cannot drift."""
        import patchworks.graph

        called = []
        monkeypatch.setattr(patchworks.graph, "main", lambda: called.append(True))
        arguments = cli.build_parser().parse_args(["dome"])
        assert arguments.handler(arguments) == 0
        assert called == [True]


class TestNothingHereOpensAWindowOrInstallsAnything:
    """The two standing promises, asserted rather than asserted-in-prose.

    `doctor` **may offer** an install and must never perform one. That is not a
    claim about intent that a reader has to take on trust: below, every route
    out of this process is broken before `doctor` runs, so a `doctor` that
    started shelling out to pip would fail here rather than in someone's tree.
    """

    @pytest.fixture
    def no_way_out(self, monkeypatch):
        """Break every route to a subprocess, an exec, and a viewer."""

        def refuse(*args, **kwargs):
            raise AssertionError(f"this must not start anything: {args!r}")

        for name in ("run", "Popen", "call", "check_call", "check_output"):
            monkeypatch.setattr(subprocess, name, refuse)
        for name in ("system", "execv", "execvp", "execve", "posix_spawn"):
            monkeypatch.setattr(os, name, refuse)
        return refuse

    def test_doctor_reports_and_never_repairs(self, no_way_out, capsys):
        cli.diagnose()
        printed = cli.format_diagnosis(cli.diagnose())
        assert "reports and never repairs" in printed or "READY" in printed

    def test_doctors_failing_report_says_nothing_was_installed(self):
        printed = cli.format_diagnosis([a_finding("gl", passed=False, remedy="do this")])
        assert "Nothing was installed" in printed
        assert "reports and never repairs" in printed

    def test_check_starts_no_process_and_opens_no_window(self, no_way_out):
        cli.measure_liveness(
            ticks=TICKS, seed=0, dome=build_graph(SMALL), image_size=IMAGE_SIZE
        )

    def test_dome_starts_no_process(self, no_way_out, capsys):
        arguments = cli.build_parser().parse_args(["dome"])
        assert arguments.handler(arguments) == 0

    def test_the_help_starts_no_process(self, no_way_out, capsys):
        assert cli.main([]) == 0
