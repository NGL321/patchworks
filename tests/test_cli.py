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
  usable once open. What is tested is the decision *before* it — re-exec, or
  refuse by name — and that the refusal names a command.
* that `mjpython` re-exec succeeds. The plan to exec is asserted; the exec is
  not performed, because performing it would replace the pytest process.

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
        torque_mean_abs=0.295,
        torque_sd=(0.078, 0.129, 0.080),
        arm_travel=(2.26, 2.60, 2.60),
        puck_travel=0.646,
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
        assert "venv" in finding.remedy and "pip" in finding.remedy

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

    def test_the_package_modules_import(self):
        finding = cli.check_package()
        assert finding.passed
        assert "patchworks.graph" in finding.detail
        assert "patchworks.sandbox" in finding.detail

    def test_a_package_module_that_does_not_import_fails_and_mentions_pythonpath(
        self, monkeypatch
    ):
        import importlib

        def refuse(name):
            raise ModuleNotFoundError(f"No module named {name!r}")

        monkeypatch.setattr(importlib, "import_module", refuse)
        finding = cli.check_package()
        assert not finding.passed
        assert "PYTHONPATH" in finding.remedy, (
            "the worktree case is the one this check exists for"
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
        finding = cli.check_mujoco_gl()
        assert not finding.passed
        assert "could not create an OpenGL context" in finding.detail
        assert "MUJOCO_GL=osmesa" in finding.remedy

    def test_mjpython_is_not_wanted_off_macos(self):
        finding = cli.check_mjpython(platform="linux")
        assert finding.passed
        assert "not required on linux" in finding.detail

    def test_mjpython_missing_on_macos_fails_and_names_where_it_comes_from(
        self, monkeypatch
    ):
        monkeypatch.setattr(cli, "mjpython_launcher", lambda: None)
        finding = cli.check_mjpython(platform="darwin")
        assert not finding.passed
        assert "mujoco" in finding.remedy, "it ships with the wheel, and that is the fix"

    def test_mjpython_present_on_macos_passes_and_says_it_is_handled(self, monkeypatch):
        monkeypatch.setattr(cli, "mjpython_launcher", lambda: Path("/somewhere/mjpython"))
        finding = cli.check_mjpython(platform="darwin")
        assert finding.passed
        assert "/somewhere/mjpython" in finding.detail
        assert "re-execs into it by itself" in finding.detail


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
    """Re-exec, or refuse with the exact command. Never a human being told to remember.

    Every rule is exercised through :func:`~patchworks.cli.window_plan`, whose
    inputs are all arguments, so these run identically on any platform and none
    of them launches anything.
    """

    def test_off_macos_a_window_just_runs(self):
        assert not cli.needs_mjpython("linux")
        assert cli.window_plan("demo", (), platform="linux").run_here

    def test_macos_needs_it(self):
        assert cli.needs_mjpython("darwin")

    def test_already_under_the_launcher_just_runs(self):
        plan = cli.window_plan(
            "demo", (), platform="darwin", already_under=True, reexeced=False
        )
        assert plan.run_here
        assert not plan.reexec and not plan.refusal

    def test_macos_with_a_launcher_re_execs_into_it(self):
        plan = cli.window_plan(
            "demo",
            ("--seed=3",),
            platform="darwin",
            already_under=False,
            launcher=Path("/venv/bin/mjpython"),
            reexeced=False,
        )
        assert plan.reexec == (
            "/venv/bin/mjpython",
            "-m",
            "patchworks",
            "demo",
            "--seed=3",
        )
        assert not plan.run_here and not plan.refusal

    def test_macos_with_no_launcher_refuses_with_the_exact_command(self):
        plan = cli.window_plan(
            "demo",
            ("--seed=3",),
            platform="darwin",
            already_under=False,
            launcher=None,
            reexeced=False,
        )
        assert not plan.run_here and not plan.reexec
        assert "mjpython -m patchworks demo --seed=3" in plan.refusal
        assert "pip" in plan.refusal, "and how to get an mjpython in the first place"
        assert "main thread" in plan.refusal, "and why, so it is not folklore"

    def test_a_second_re_exec_refuses_rather_than_looping(self):
        """The loop guard.

        Detection reads a private MuJoCo attribute, and a private name can
        move. What that must degrade to is one refusal naming the command, not
        a process that execs itself forever — so a plan made with the marker
        already set never execs again.
        """
        plan = cli.window_plan(
            "demo",
            (),
            platform="darwin",
            already_under=False,
            launcher=Path("/venv/bin/mjpython"),
            reexeced=True,
        )
        assert not plan.reexec and not plan.run_here
        assert "stopping rather than doing it again" in plan.refusal
        assert "/venv/bin/mjpython -m patchworks demo" in plan.refusal

    def test_the_marker_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv(cli.REEXEC_MARKER, "1")
        plan = cli.window_plan(
            "demo",
            (),
            platform="darwin",
            already_under=False,
            launcher=Path("/venv/bin/mjpython"),
        )
        assert not plan.reexec

    def test_running_here_calls_the_run_and_succeeds(self, monkeypatch):
        monkeypatch.setattr(cli, "window_plan", lambda *a, **k: cli.WindowPlan(run_here=True))
        ran = []
        assert cli.open_window_with("demo", (), lambda: ran.append(True)) == 0
        assert ran == [True]

    def test_a_refusal_goes_to_stderr_and_exits_one_without_running_anything(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            cli, "window_plan", lambda *a, **k: cli.WindowPlan(refusal="run this instead")
        )

        def must_not_run():
            raise AssertionError("a refused window must not run the thing anyway")

        assert cli.open_window_with("demo", (), must_not_run) == 1
        captured = capsys.readouterr()
        assert "run this instead" in captured.err
        assert captured.out == ""

    def test_the_demo_subcommand_refuses_without_importing_the_viewer(
        self, monkeypatch, capsys
    ):
        """The refusal path never reaches `patchworks.surface.gestures`.

        Which is what makes it safe: the whole point of refusing is that this
        process cannot open a window, so it must not get as far as trying.
        """
        monkeypatch.setattr(
            cli, "window_plan", lambda *a, **k: cli.WindowPlan(refusal="not here")
        )
        monkeypatch.delitem(sys.modules, "patchworks.surface.gestures", raising=False)
        arguments = cli.build_parser().parse_args(["demo", "--seed=2"])
        assert arguments.handler(arguments) == 1
        assert "patchworks.surface.gestures" not in sys.modules
        assert "not here" in capsys.readouterr().err

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
        assert len(liveness.torque_sd) == 3
        assert len(liveness.arm_travel) == 3
        assert liveness.torque_mean_abs >= 0.0
        assert liveness.puck_travel >= 0.0

    def test_the_same_seed_gives_the_same_run(self):
        """Seeded explicitly, so a bug report's numbers are somebody else's numbers too."""
        first, second = (
            cli.measure_liveness(
                ticks=TICKS, seed=1, dome=build_graph(SMALL), image_size=IMAGE_SIZE
            )
            for _ in range(2)
        )
        assert first.torque_mean_abs == second.torque_mean_abs
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
        assert "control rate       : 50 Hz" in printed
        assert "non-finite         : none" in printed
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
        assert "NON-FINITE         : command at tick 12" in printed
        assert "That is a bug, not an untrained agent" in printed

    def test_the_report_carries_what_a_bug_report_needs(self):
        """Versions and platform above the numbers, because that is what it is for."""
        printed = cli.format_liveness(a_liveness())
        assert "Python" in printed and "torch" in printed and "mujoco" in printed
        assert "MUJOCO_GL=" in printed

    def test_the_flags_reach_the_measurement(self, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr(
            cli, "measure_liveness", lambda **kwargs: seen.update(kwargs) or a_liveness()
        )
        arguments = cli.build_parser().parse_args(
            ["check", "--ticks", "7", "--seed", "11", "--split", "test"]
        )
        arguments.handler(arguments)
        assert seen == {"ticks": 7, "seed": 11, "split": "test"}

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
