"""The `patchworks` command: one entry point, so nothing has to be remembered.

**A skeleton with the subcommands wanted today, not a finished surface** (#119).
The complete command set is a later design session's; what this file rigs up is
the shape — a dispatcher, `--help` at every level that says what a thing *is*,
and somewhere obvious for `patchworks run` (#121) and `patchworks watch` (#122)
to land.

Three things were undiscoverable before it, and each had tripped someone up:
that `python -m patchworks` printed the dome and was *not* the demo; that the
demo needs `mjpython` rather than `python` on macOS; and that a worktree needs
`PYTHONPATH`. The first is now :func:`_dome`, one subcommand among several
rather than the default. The second is :func:`window_plan`, which execs the
right interpreter or refuses by name — a human never has to know it. The third
is the venv's own `bin/patchworks`, which needs no path set at all.

**Nothing here is part of the architecture** (#77, and `CONTEXT.md`'s *Demo
surface*). No cell reads anything this computes, every seed is passed
explicitly, and nothing draws from a global RNG — `tests/test_cli.py` asserts
that last one against torch's and numpy's global state, because a CLI that
perturbed either would change the trajectory of everything run after it in the
same process.

**What `doctor` says, it checked.** The standing failure mode on this project
is prose claiming an enforcement it has not got, so each finding below reports
what was actually observed and the verdict aggregates nothing else; where a
requirement is reported rather than enforced, the output says so in those
words.

**And what `doctor` cannot reach, it does not claim.** It runs *after*
`patchworks/__init__.py`, which imports the architecture eagerly, which imports
torch — so on a clone where nothing has been installed at all, `python -m
patchworks doctor` ends in a bare `ModuleNotFoundError: No module named
'torch'` and never reaches :func:`diagnose`. The install step comes first, and
the README puts it first. Making `doctor` survive that means the package's
front door importing lazily, which is a decision about `patchworks`' public
shape rather than about this CLI, and #119 does not carry it — it is escalated
rather than taken here. What `doctor` *can* still catch is a partial
installation: `mujoco` and `gymnasium` reach the package only through
:mod:`patchworks.sandbox`, which `__init__` does not import, so either being
absent or broken arrives as a finding rather than a traceback.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the annotation only -- see `measure_liveness`
    from patchworks.graph import Dome

#: The interpreter bounds, as `pyproject.toml`'s `requires-python` states them.
#: Held as tuples rather than parsed from the specifier at runtime because
#: parsing one needs `packaging`, which is not in the dependency set and is not
#: worth adding to compare two integers. That these agree with `pyproject.toml`
#: is not left to whoever edits it: `tests/test_cli.py` reads the specifier and
#: asserts both bounds against it, so the guarantee is a test's rather than a
#: comment's.
MINIMUM_PYTHON = (3, 11)
BELOW_PYTHON = (3, 13)

#: The launcher MuJoCo's passive viewer needs on macOS.
MJPYTHON = "mjpython"


class _Unset:
    """The "not given" sentinel :func:`window_plan` needs and `None` cannot be.

    `None` is a real answer to *which mjpython* — it is what
    :func:`mjpython_launcher` returns when there is none — so a `launcher=None`
    default would make "there is no launcher" indistinguishable from "go and
    look", and a caller could not say the first. The suite is what found that:
    the test for the refusing path passed `launcher=None` and got a re-exec
    plan. Named after :class:`patchworks.agent._Unset`, which exists for the
    same reason.
    """


_UNSET = _Unset()

#: How many ticks `patchworks check` runs. 300 at 50 Hz is six seconds of
#: world and about four of wall clock end to end on the development laptop —
#: two of them the run and two the imports torch and MuJoCo cost before it
#: starts. Long enough for an untrained agent to swing the arm through whole
#: revolutions, which is the signal this output is read for, and short enough
#: that a human waits for it rather than going away.
CHECK_TICKS = 300


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One thing `doctor` looked at, and what it found.

    `passed` is the verdict and `detail` is the observation it was drawn from —
    both, always, so the output can be read without trusting the verdict.
    `remedy` is what to do about a failure and is empty when there is nothing
    to do.
    """

    name: str
    passed: bool
    detail: str
    remedy: str = ""

    def render(self) -> str:
        """Two lines when there is something to do about it, one when there is not."""
        mark = "ok  " if self.passed else "FAIL"
        first = f"  [{mark}] {self.name}: {self.detail}"
        return first if not self.remedy else f"{first}\n         -> {self.remedy}"


def in_a_virtual_environment() -> bool:
    """Is this interpreter a venv's? The stdlib's own test, from `sys`."""
    return sys.prefix != sys.base_prefix


def _venv_hint() -> str:
    """The install command to print, written against where this interpreter is.

    **Inside a venv** the command names that venv's own `pip` — read from
    `sys.prefix` rather than hard-coded as `.venv`, since a reader told
    `.venv/bin/pip` while standing in a tree whose venv is somewhere else has
    been given the wrong command, which is worse than a long one.

    **Outside one**, naming this interpreter's `pip` would say to install torch
    and MuJoCo into the system Python, so the command is to make a venv first.
    The obvious case for that branch — a fresh clone run as
    `PYTHONPATH=src python3 -m patchworks doctor` — is **not** one it reaches:
    that dies on `import torch` before `doctor` exists (see this module's
    docstring). What does reach it is an interpreter that has the dependencies
    without being a venv: a conda base environment, or a `pip install --user`.
    Narrower than it looks, and said here rather than left implied.
    """
    if not in_a_virtual_environment():
        return (
            f"python{MINIMUM_PYTHON[0]}.{BELOW_PYTHON[1] - 1} -m venv .venv "
            f"&& .venv/bin/pip install -e '.[dev]'   "
            f"(this interpreter is not in a virtual environment)"
        )
    return f"{Path(sys.prefix) / 'bin' / 'pip'} install -e '.[dev]'"


def check_interpreter() -> Finding:
    """Is this interpreter inside `requires-python`?

    A real bound check, and the only requirement `doctor` enforces rather than
    reports: the version is two integers and the bound is two integers, so
    there is nothing to approximate.
    """
    version = ".".join(str(part) for part in sys.version_info[:3])
    bound = (
        f">={MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]},"
        f"<{BELOW_PYTHON[0]}.{BELOW_PYTHON[1]}"
    )
    inside = MINIMUM_PYTHON <= sys.version_info[:2] < BELOW_PYTHON
    return Finding(
        name="interpreter",
        passed=inside,
        detail=f"Python {version} at {sys.executable}, against {bound}",
        # The remedy is `_venv_hint()` and nothing prepended. It used to add a
        # `python3.12 -m venv .venv &&` of its own, which printed the venv
        # command twice once `_venv_hint` grew its own -- and, worse, told
        # someone inside an out-of-bound venv to build a new one and then
        # install into the old one with its `pip`. One place says how to make a
        # venv, and it is the one that knows whether there is one.
        remedy="" if inside else f"this interpreter is outside {bound}: {_venv_hint()}",
    )


#: The runtime dependencies `doctor` imports, with `pyproject.toml`'s
#: requirement for each quoted verbatim beside it. `tests/test_cli.py` holds
#: this table against `[project].dependencies`, so a pin that moves there and
#: not here reddens rather than drifts.
#:
#: **These are reported, not enforced**, and `doctor` says so in those words.
#: Enforcing them means resolving a version specifier, which needs `packaging`
#: — not in the dependency set, and not worth adding for a message. What holds
#: the pins is `tests/test_package.py`, which asserts torch and MuJoCo exactly;
#: what this check enforces is that the import works at all, which is the
#: question "can this installation run" actually asks.
DEPENDENCIES = (
    ("torch", "torch==2.2.2"),
    ("mujoco", "mujoco==3.10.0"),
    ("gymnasium", "gymnasium>=1.0,<2"),
    ("numpy", "numpy<2"),
)


def check_dependencies() -> Finding:
    """Do the four runtime dependencies import, and at what versions?"""
    installed: list[str] = []
    missing: list[str] = []
    for module_name, requirement in DEPENDENCIES:
        try:
            module = importlib.import_module(module_name)
        except Exception as failure:  # an import can fail any number of ways
            missing.append(f"{requirement} ({type(failure).__name__}: {failure})")
        else:
            found = getattr(module, "__version__", "version not reported")
            installed.append(f"{module_name} {found}")
    return Finding(
        name="dependencies",
        passed=not missing,
        detail=(
            (", ".join(installed) if installed else "none imported")
            + (f"; did not import: {'; '.join(missing)}" if missing else "")
            + " (versions reported, not enforced -- pytest holds the pins)"
        ),
        remedy="" if not missing else f"install them into this interpreter: {_venv_hint()}",
    )


def check_package() -> Finding:
    """Do the modules a run actually needs import?

    `patchworks` itself is already imported by the time this runs — this file
    is inside it — so importing that would assert nothing. What is worth asking
    is whether the two modules a run reaches for come up: the dome's
    construction, which pulls torch, and the world, which pulls MuJoCo and
    Gymnasium.
    """
    wanted = ("patchworks.graph", "patchworks.sandbox")
    for module_name in wanted:
        try:
            importlib.import_module(module_name)
        except Exception as failure:
            return Finding(
                name="package",
                passed=False,
                detail=f"{module_name} did not import ({type(failure).__name__}: {failure})",
                remedy=(
                    f"install the package into this interpreter: {_venv_hint()} "
                    f"-- or, from a worktree, set PYTHONPATH to its src/"
                ),
            )
    import patchworks

    return Finding(
        name="package",
        passed=True,
        detail=f"{', '.join(wanted)} imported, from {Path(patchworks.__file__).parent}",
    )


#: The smallest world that makes MuJoCo build a GL context: one body, one geom,
#: no physics worth stepping. The check below renders it rather than the arena,
#: so that a GL failure is reported as a GL failure instead of arriving inside
#: the sandbox's own construction.
_GL_PROBE_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 1"/>
    <geom type="plane" size="1 1 0.1"/>
  </worldbody>
</mujoco>
"""


def check_mujoco_gl() -> Finding:
    """Does MuJoCo's offscreen render path work here?

    Asked by rendering, not by inspecting a variable. The agent's every tick
    reads a rendered image — the goal reaches it as perception and nothing else
    — so a GL context that cannot be made is a run that cannot happen, and the
    only honest way to report that is to have made one.

    Offscreen throughout: `mujoco.Renderer` opens no window, so this is safe on
    a headless machine and safe to call from a test.
    """
    backend = os.environ.get("MUJOCO_GL", "(unset: MuJoCo's default for this platform)")
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_string(_GL_PROBE_XML)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, 64, 64)
        try:
            renderer.update_scene(data)
            frame = renderer.render()
        finally:
            renderer.close()
    except Exception as failure:
        return Finding(
            name="mujoco gl",
            passed=False,
            detail=f"offscreen render failed with MUJOCO_GL={backend} "
            f"({type(failure).__name__}: {failure})",
            remedy=(
                "MuJoCo could not make an offscreen GL context. On a headless "
                "machine, install osmesa and set MUJOCO_GL=osmesa (this is what "
                "CI does); on a desktop, check the platform's GL drivers."
            ),
        )
    return Finding(
        name="mujoco gl",
        passed=True,
        detail=f"offscreen render returned {frame.shape} with MUJOCO_GL={backend}",
    )


def mjpython_launcher(executable: str = sys.executable) -> Path | None:
    """The `mjpython` this interpreter should use, or `None` if there is none.

    Beside `executable` first and `PATH` second, because `mjpython` must be the
    one from the same environment as the `mujoco` being imported: it embeds an
    interpreter, and one from a different venv would run a different
    installation of everything.

    `executable` is an argument, and :func:`window_plan` passes its own, so
    that where this looked and where the refusal *says* it looked cannot come
    apart.
    """
    beside = Path(executable).parent / MJPYTHON
    if beside.exists():
        return beside
    on_path = shutil.which(MJPYTHON)
    return Path(on_path) if on_path else None


def needs_mjpython(platform: str = sys.platform) -> bool:
    """Does opening a MuJoCo window on this platform require `mjpython`?

    macOS only. MuJoCo's passive viewer has to own the process's main thread
    there, which is the whole of what the `mjpython` launcher provides; on
    every other platform `mujoco.viewer` runs the window on a thread of its own
    and a plain interpreter is enough.
    """
    return platform == "darwin"


def check_mjpython(platform: str = sys.platform) -> Finding:
    """Is the launcher a window needs on this platform present?

    Reported by `doctor` even though none of `doctor`, `check` or `dome` opens
    a window, because the question a human asks it is *can this installation
    run*, and the demo is part of what this installation is for.
    """
    if not needs_mjpython(platform):
        return Finding(
            name="mjpython",
            passed=True,
            detail=f"not required on {platform}; MuJoCo's viewer takes a thread of its own there",
        )
    launcher = mjpython_launcher()
    if launcher is None:
        return Finding(
            name="mjpython",
            passed=False,
            detail=f"not found beside {sys.executable} or on PATH, and macOS's "
            f"passive viewer requires it",
            remedy=(
                f"it ships with the `mujoco` wheel, so installing the "
                f"dependencies puts it there: {_venv_hint()}"
            ),
        )
    return Finding(
        name="mjpython",
        passed=True,
        detail=f"{launcher} (macOS's passive viewer requires it; "
        f"`patchworks` re-execs into it by itself)",
    )


#: What `doctor` does *not* check, said out loud. A verdict that reads as
#: "ready" while a whole class of failure was never looked at is the overstated
#: prose this project keeps finding against itself, so the unchecked classes
#: are printed beside the verdict rather than left to be inferred from the
#: absence of a line.
NOT_CHECKED = (
    "that a window actually opens -- that rests on a human at the screen",
    "that dependency versions satisfy their specifiers -- reported only; "
    "`pytest` is what holds the pins",
    "anything about a trained agent -- there is none yet",
)


def diagnose(platform: str = sys.platform) -> list[Finding]:
    """Every check, in the order a failure would cascade.

    Ordered rather than sorted: an interpreter outside the bound explains a
    dependency that will not import, which explains a package that will not, so
    reading top-down finds the cause before its consequences. Every check runs
    regardless — a human pasting this into a bug report wants the whole picture,
    not the first failure.
    """
    return [
        check_interpreter(),
        check_dependencies(),
        check_package(),
        check_mujoco_gl(),
        check_mjpython(platform),
    ]


def format_diagnosis(findings: list[Finding]) -> str:
    """The report, verdict last."""
    failed = [finding for finding in findings if not finding.passed]
    lines = ["patchworks doctor -- can this installation run?", ""]
    lines += [finding.render() for finding in findings]
    lines += ["", "not checked here:"]
    lines += [f"  - {item}" for item in NOT_CHECKED]
    lines.append("")
    if failed:
        names = ", ".join(finding.name for finding in failed)
        lines.append(f"NOT READY: {len(failed)} of {len(findings)} checks failed ({names}).")
        lines.append("Each failure above names what to do about it. Nothing was installed:")
        lines.append("`doctor` reports and never repairs, so the commands are yours to run.")
    else:
        lines.append(f"READY: all {len(findings)} checks above passed. Next: patchworks check")
    return "\n".join(lines)


def _doctor(arguments: argparse.Namespace) -> int:
    """Run every check and print the report; 0 when all passed, 1 when any did not."""
    findings = diagnose()
    print(format_diagnosis(findings))
    return 0 if all(finding.passed for finding in findings) else 1


# ---------------------------------------------------------------------------
# the mjpython problem
# ---------------------------------------------------------------------------
#
# **A window subcommand hands off rather than importing.** `patchworks demo`
# does not import `patchworks.surface.gestures` and call its `main`; it execs
# the right interpreter on that module, which is the same thing a human typing
# the command would get. Two reasons, and the first is the load-bearing one:
#
# 1. `tests/test_dome_panel.py`'s `TestTheArchitectureDoesNotImportTheSurface`
#    holds that **nothing outside `surface/` imports the surface** -- the door
#    that stops a learning rule reaching for a measured persistence, and one
#    the naming scan in `tests/test_timescale.py` cannot see. Widening it for a
#    dispatcher is a decision about that guard's scope, and this ticket does not
#    carry it. Handing off keeps the edge from existing rather than excusing it.
# 2. On macOS the interpreter has to *be* `mjpython` before the viewer starts,
#    and there is no importing your way into that. An exec is what the platform
#    actually requires, so writing it as one is writing what is true.
#
# It also means there is no exec loop to guard against: what is exec'd is a
# different module, with no dispatcher in it to come back here.


@dataclass(frozen=True)
class WindowPlan:
    """How to launch a window subcommand: an exec, or a refusal.

    Exactly one is set. `reexec` is the argv for `os.execv`; `refusal` is what
    to print and why. Returned as data rather than acted on, which is what lets
    every rule below be tested for platforms that are not the one running the
    test -- and without a window ever opening or a process ever being replaced.
    """

    reexec: tuple[str, ...] = ()
    refusal: str = ""


#: The module `patchworks demo` hands off to: the scene window #96 built, whose
#: own `__main__` is what a human used to be told to type. Named as a string
#: rather than imported, deliberately -- see the note at the top of this
#: section.
DEMO_MODULE = "patchworks.surface.gestures"


def window_plan(
    module: str,
    argv: tuple[str, ...],
    *,
    spoken_as: str = "this",
    platform: str = sys.platform,
    launcher: Path | None | _Unset = _UNSET,
    executable: str = sys.executable,
) -> WindowPlan:
    """Decide which interpreter should run a module that opens a MuJoCo window.

    The rule the ticket asks for, in one place: **re-exec into the right
    interpreter, or refuse with the exact command**. A human never types
    `mjpython` because they were told to.

    Off macOS that interpreter is this one and the exec is a formality, kept so
    that there is one path rather than two. On macOS it is `mjpython`, and if
    there is none the refusal says what it is for, where it was looked for, how
    to get one, and what to run afterwards.

    `spoken_as` is what the human typed — `patchworks demo` — as distinct from
    `module`, which is what gets exec'd. The refusal leads with the first,
    because someone who ran one command should not have to work out that the
    thing being talked about is the same thing; the module path is offered
    after it, for a worktree where there is no console script to run.

    Every input the decision turns on is an argument with a live default, so
    the rules can be exercised for a platform that is not the one running.
    """
    if isinstance(launcher, _Unset):
        launcher = mjpython_launcher(executable)

    if not needs_mjpython(platform):
        return WindowPlan(reexec=(executable, "-m", module, *argv))
    if launcher is None:
        direct = " ".join((MJPYTHON, "-m", module, *argv))
        return WindowPlan(
            refusal=(
                f"`{spoken_as}` opens a MuJoCo window, and on macOS a MuJoCo window "
                f"has to own the process's main thread -- which only MuJoCo's "
                f"`mjpython` launcher gives it.\n\n"
                f"No `mjpython` was found beside {executable} or on PATH. It ships "
                f"with the `mujoco` wheel, so installing the dependencies puts one "
                f"there:\n\n"
                f"    {_venv_hint()}\n\n"
                f"Then `{spoken_as}` will work. From a worktree, where there is no "
                f"console script, the same thing is:\n\n    {direct}\n"
            )
        )
    return WindowPlan(reexec=(str(launcher), "-m", module, *argv))


def open_window(module: str, argv: tuple[str, ...], *, spoken_as: str) -> int:
    """Carry out a :class:`WindowPlan`: exec, or print the refusal and fail.

    The exec replaces this process, so on the success path nothing after it
    runs and there is no exit code to return. The refusal goes to stderr and
    exits 1.
    """
    plan = window_plan(module, argv, spoken_as=spoken_as)
    if plan.reexec:
        os.execv(plan.reexec[0], list(plan.reexec))
    print(plan.refusal, file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Liveness:
    """What a few hundred headless ticks showed.

    Numbers rather than sentences, so that a test can ask about the exit code's
    cause without parsing the report, and `non_finite` names *where* rather than
    saying whether: a bug report that says "the torques went non-finite" is
    worth more than one that says something did.
    """

    ticks: int
    seed: int
    split: str
    control_hz: float

    commanded_mean_abs: float
    """Mean |torque| the graph asked for, **before** the arm's limits saw it.
    `TickOutcome.command` is pre-clip, so this can exceed the action space's 1
    and is not a fraction of saturation."""

    applied_mean_abs: float
    """Mean |torque| the arm actually applied — `TickOutcome.applied`, the
    efference copy, post-clip. This one *is* a fraction of saturation, and a
    gap between it and the commanded figure is the graph asking for more than
    the arm will give."""

    commanded_sd: tuple[float, ...]
    """Per-joint spread of the commanded torque. Zero means a frozen command,
    which is the failure this number is read for."""

    arm_travel: tuple[float, ...]
    """Per joint, the **cumulative** path length in radians — the sum of
    |Δq| across the observed ticks, not the distance from start to end. An arm
    that swings out and back has travelled; its net displacement is nearly
    zero, and reporting that as travel would tell a working installation it was
    broken."""

    puck_displacement: float
    """The furthest any puck ended from where it started, in **metres**. Net
    rather than cumulative, because what is being asked is whether the agent
    put a puck somewhere. Translation only: a puck's third coordinate is a
    rotation in radians and does not belong in the same maximum."""

    world_seconds: float
    elapsed: float
    non_finite: tuple[str, ...]


def measure_liveness(
    *,
    ticks: int = CHECK_TICKS,
    seed: int = 0,
    split: str = "train",
    dome: Dome | None = None,
    image_size: int | None = None,
) -> Liveness:
    """Run the untrained agent headless and measure whether it is driving the arm.

    Grown from `scratchpad/is_it_moving.py`, whose output format proved itself
    in practice and is kept: control rate, torque magnitude, per-joint spread,
    arm travel, puck travel. What is added is the non-finite sweep, which the
    exit code hangs off, and a header of versions, because this is what gets
    pasted into a bug report.

    **Every draw is seeded explicitly and none of them is global.** The sheaf's
    parameters come from a `torch.Generator` built here, and the world's layout
    from `reset(seed=...)`, which Gymnasium keeps on the env. Nothing touches
    `torch.manual_seed` or `numpy.random`, so running this changes no trajectory
    of anything run after it -- the demo surface's rule (#77), applied to the
    CLI. `tests/test_cli.py` asserts it against both global states.

    `dome` and `image_size` exist so the suite can run this on the small dome
    against a 16x16 render rather than the full one against 64x64; the CLI
    passes neither and gets the defaults.

    The imports are inside the function, as `gestures.main` does it, and that
    buys less than it looks: `import patchworks` has already pulled torch and
    the graph in by the time anything here runs. What it does defer is
    :mod:`patchworks.sandbox`, and MuJoCo with it, so `patchworks --help` and
    `patchworks dome` never load the world.
    """
    import numpy as np
    import torch

    from patchworks.agent import Agent, run
    from patchworks.graph import build_graph
    from patchworks.sandbox import PlanarPushSandbox
    from patchworks.sandbox.env import N_PUCKS

    if dome is None:
        dome = build_graph()
    world = PlanarPushSandbox(
        split=split, **({} if image_size is None else {"image_size": image_size})
    )
    try:
        agent = Agent(world, dome=dome, generator=torch.Generator().manual_seed(seed))
        inner = world.unwrapped
        started = time.perf_counter()
        ticking = run(agent, ticks, seed=seed)

        # After `run`, before the first tick: `run` arranges the world when it
        # is called, so this is the layout the run actually starts from.
        # `puck_pose` is the env's own public accessor -- privileged state, on
        # the footing `info` is on, which is where the CLI sits by CONTEXT.md's
        # *Demo surface*. Reading `qpos` by hand instead would duplicate the
        # joint ordering the env already computes from `jnt_qposadr`, and a
        # joint added ahead of the pucks would silently move it.
        pucks_before = np.array([inner.puck_pose(i) for i in range(N_PUCKS)])

        # Summarised per place rather than listed per tick. A genuinely broken
        # run goes non-finite on *every* tick, and three hundred lines of
        # `command at tick 41` would bury the four numbers a reader came for --
        # while telling them nothing the count and the first tick do not. The
        # first tick is the useful one: it says whether the run started broken
        # or went that way.
        ticks_seen: dict[str, list[int]] = {}
        at_the_end: list[str] = []
        commanded: list[np.ndarray] = []
        applied: list[np.ndarray] = []

        # The arm's joints, read off the observation the env writes at
        # `_arm_qadr` -- again the env's index rather than a slice of this
        # file's. Travel accumulates across the observed ticks, so it is the
        # path length over `ticks - 1` intervals and undercounts by the motion
        # inside the first tick and inside each tick's physics substeps. Said
        # here because it is a floor on how far the arm went, not an estimate.
        travel = np.zeros(0)
        previous_arm: np.ndarray | None = None

        for index, outcome in enumerate(ticking):
            command = np.asarray(outcome.command, dtype=float)
            applied_now = np.asarray(outcome.applied, dtype=float)
            commanded.append(command.copy())
            applied.append(applied_now.copy())
            for label, values in (("command", command), ("applied", applied_now)):
                if not np.all(np.isfinite(values)):
                    ticks_seen.setdefault(label, []).append(index)
            for key, value in outcome.observation.items():
                array = np.asarray(value)
                if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
                    ticks_seen.setdefault(f"observation[{key!r}]", []).append(index)

            arm = np.asarray(outcome.observation["qpos"], dtype=float)
            if previous_arm is None:
                travel = np.zeros_like(arm)
            else:
                travel = travel + np.abs(arm - previous_arm)
            previous_arm = arm
        elapsed = time.perf_counter() - started

        pucks_after = np.array([inner.puck_pose(i) for i in range(N_PUCKS)])
        for label, state in (
            ("world qpos", inner.data.qpos),
            ("world qvel", inner.data.qvel),
        ):
            if not np.all(np.isfinite(state)):
                at_the_end.append(f"{label}, at the end of the run")
        non_finite = tuple(
            f"{label}, first at tick {indices[0]}, on {len(indices)} of {ticks} ticks"
            for label, indices in ticks_seen.items()
        ) + tuple(at_the_end)

        commanded_array = np.array(commanded) if commanded else np.zeros((0, 0))
        applied_array = np.array(applied) if applied else np.zeros((0, 0))
        control_hz = 1.0 / (inner.model.opt.timestep * inner.frame_skip)
        # (x, y) only: the third coordinate `puck_pose` returns is a rotation
        # in radians, and a puck that spun in place has not moved anywhere.
        moved = np.linalg.norm(pucks_after[:, :2] - pucks_before[:, :2], axis=1)
        return Liveness(
            ticks=ticks,
            seed=seed,
            split=split,
            control_hz=control_hz,
            commanded_mean_abs=float(np.abs(commanded_array).mean())
            if commanded_array.size
            else 0.0,
            applied_mean_abs=float(np.abs(applied_array).mean())
            if applied_array.size
            else 0.0,
            commanded_sd=tuple(float(value) for value in commanded_array.std(axis=0))
            if commanded_array.size
            else (),
            arm_travel=tuple(float(value) for value in travel),
            puck_displacement=float(moved.max()) if moved.size else 0.0,
            world_seconds=ticks / control_hz,
            elapsed=elapsed,
            non_finite=non_finite,
        )
    finally:
        world.close()


def _environment_line() -> str:
    """The one line a bug report needs before the numbers mean anything."""
    import platform as platform_module

    parts = [
        f"Python {'.'.join(str(part) for part in sys.version_info[:3])}",
        f"{platform_module.system()} {platform_module.machine()}",
    ]
    for module_name in ("torch", "mujoco", "gymnasium", "numpy"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            parts.append(f"{module_name} (did not import)")
        else:
            parts.append(f"{module_name} {getattr(module, '__version__', '?')}")
    parts.append(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '(unset)')}")
    return ", ".join(parts)


def format_liveness(liveness: Liveness) -> str:
    """The prototype's format, with the environment above it and a verdict below.

    The two closing sentences are the prototype's own and are the reason it was
    useful: they say what the numbers mean to someone who has never seen them
    before, which is exactly the person running this.

    Each number is labelled with what it actually is -- commanded torque and
    applied torque as two figures rather than one, travel as cumulative, the
    puck's move in metres. The prototype ran some of those together; a bug
    report is worth less when the reader has to guess which of two quantities a
    line holds.
    """
    sd = ", ".join(f"{value:.3f}" for value in liveness.commanded_sd)
    travel = ", ".join(f"{value:.2f}" for value in liveness.arm_travel)
    lines = [
        f"patchworks check -- {liveness.ticks} ticks, headless, "
        f"seed {liveness.seed}, split {liveness.split}",
        _environment_line(),
        "",
        f"control rate         : {liveness.control_hz:.0f} Hz",
        f"torque |mean| asked  : {liveness.commanded_mean_abs:.3f}"
        f"   (pre-clip; 0 = the agent is silent)",
        f"torque |mean| applied: {liveness.applied_mean_abs:.3f}"
        f"   (post-clip; 1 = saturated)",
        f"torque per-joint sd  : [{sd}]   (0 = frozen command)",
        f"ARM  travelled (rad) : [{travel}]  cumulative, over {liveness.ticks} ticks "
        f"= {liveness.world_seconds:.0f} s of world",
        f"PUCK moved (m)       : {liveness.puck_displacement:.3f}"
        f"   (furthest any puck ended from where it started)",
        f"wall clock           : {liveness.elapsed:.1f} s",
    ]
    if liveness.non_finite:
        lines.append(f"NON-FINITE           : {', '.join(liveness.non_finite)}")
        lines += [
            "",
            "Something went non-finite. That is a bug, not an untrained agent:",
            "send this whole output.",
        ]
    else:
        lines.append("non-finite           : none")
        lines += [
            "",
            "Arm travelling + puck barely = the agent is running and untrained. Expected.",
            "Arm at ~0 = something is genuinely wrong; send this output.",
        ]
    return "\n".join(lines)


def _check(arguments: argparse.Namespace) -> int:
    """Measure, print, and fail the exit code if anything went non-finite.

    A bad `--split` comes back as a `ValueError` from the world's own
    constructor, which owns the list of splits and is the only place it should
    live. Caught here and turned into a usage error rather than a traceback,
    because `--help` promises this command exits on what it measured and an
    unhandled exception is not that. Naming the splits in `choices=` would mean
    the parser importing the sandbox, so `patchworks --help` would load MuJoCo
    to print a list.
    """
    try:
        liveness = measure_liveness(
            ticks=arguments.ticks, seed=arguments.seed, split=arguments.split
        )
    except ValueError as refusal:
        print(f"patchworks check: {refusal}", file=sys.stderr)
        return 2
    print(format_liveness(liveness))
    return 1 if liveness.non_finite else 0


# ---------------------------------------------------------------------------
# dome, and the demo
# ---------------------------------------------------------------------------


def _dome(arguments: argparse.Namespace) -> int:
    """What `python -m patchworks` printed before this file existed.

    Routed to `graph.main` rather than reimplemented, so that the two cannot
    drift: this is the same call the old `__main__` made.
    """
    from patchworks.graph import main as dome_main

    dome_main()
    return 0


def _demo(arguments: argparse.Namespace) -> int:
    """The hands-on scene window, launched without anyone having to know `mjpython`.

    A launcher, not a new window: `patchworks.surface.gestures` is what #96
    built, and what a human was previously expected to reach by typing
    `mjpython -m patchworks.surface.gestures` from memory. What this adds is
    finding the launcher, or naming it -- see the note above :class:`WindowPlan`
    for why it execs rather than imports.

    The dome panel is the *other* window and is not opened here; #122 is
    building `patchworks watch` for it.
    """
    return open_window(
        DEMO_MODULE,
        (
            f"--ticks={arguments.ticks}",
            f"--seed={arguments.seed}",
            f"--split={arguments.split}",
        ),
        spoken_as="patchworks demo",
    )


# ---------------------------------------------------------------------------
# the dispatcher
# ---------------------------------------------------------------------------


DESCRIPTION = """\
patchworks -- an embodied graph architecture: many small predictors, each in its
own metric space, reconciled into a model of a world none of them sees whole.

There is no trained agent yet. What runs today is the world, the dome's
construction, and an untrained agent driving the arm -- which is enough to tell
a broken installation from a working one.

Start with `patchworks doctor`. It says whether this installation can run and,
for anything that cannot, the command to fix it. Then `patchworks check`, which
runs the thing for a few seconds and prints the numbers a bug report wants.
"""

EPILOG = """\
docs/spec/ has the architecture, CONTEXT.md the vocabulary, and docs/adr/ the
decisions that needed a reason on the record.
"""

#: Each subcommand's own `--help`, hand-wrapped. `RawDescriptionHelpFormatter`
#: is what keeps the paragraphs and the indented commands intact, and the price
#: of it is that argparse re-wraps nothing — so these are written at the width
#: they are meant to be read at rather than left as one long line.
DOCTOR_DESCRIPTION = """\
Check whether this installation can run: the interpreter against
`requires-python`, the four runtime dependencies and the package importing,
MuJoCo's offscreen render path actually rendering, and -- on macOS -- whether
the `mjpython` launcher a window needs is present.

Every failure is printed with the command that fixes it, and what was *not*
checked is printed too, so a "ready" verdict cannot be read as covering more
than it does. Nothing is ever installed or changed: this reports, and the
commands are yours to run.

This runs after the package has imported, so it cannot rescue a clone where
nothing is installed at all -- there, `python -m patchworks doctor` fails on
`import torch` before reaching any of this. Install first; the README says how.

Exit code is 0 when every check passed and 1 when any did not.
"""

CHECK_DESCRIPTION = """\
Run the untrained agent against the world for a few hundred ticks, headless
and in a few seconds, and print what it did: control rate, torque magnitude
and per-joint spread, how far the arm and the pucks moved, and whether
anything went non-finite.

This is what answers "is this thing alive", and what to paste into a bug
report -- the interpreter, platform and dependency versions are printed above
the numbers for exactly that. An untrained agent swings the arm and barely
moves a puck; that is the expected picture, not a failure.

Exit code is 0 normally, 1 if anything went non-finite -- which is a bug
rather than an untrained agent -- and 2 if the arguments were wrong.
"""

DOME_DESCRIPTION = """\
Print the graph's shape -- the taper from the two-dimensional sensorimotor
sheet to the deep core -- together with what construction recorded about it:
the cells and edges at each level, and the diagnostics build time computes.

This is what `python -m patchworks` used to print by default. The output is
unchanged; what changed is that it is now one subcommand among several rather
than the whole tool. It builds no world and opens nothing.
"""

DEMO_DESCRIPTION = """\
Open the scene window and run the agent in it, so that a human can sit in
front of it and interfere: shift-ctrl-drag a link or a puck, left-double-click
a puck and then a zone to set a goal, `r` to rearrange, `1`-`9` for the goal
pairs.

On macOS MuJoCo's viewer must own the process's main thread, and only MuJoCo's
`mjpython` launcher gives it that. This finds `mjpython` and re-execs into it
by itself; if it cannot find one it stops and prints the exact command to run.
You do not have to know any of that.

The dome panel is the other window and is not opened here.
"""


def build_parser() -> argparse.ArgumentParser:
    """The dispatcher.

    Every parser here carries a `description` that says what the thing *is*,
    not only what its flags do — the whole point being that a human who knows
    nothing about this project can type `patchworks` and find out what to do.

    `run` (#121) and `watch` (#122) are the next two subcommands and are not
    here: each is its own ticket, and adding an empty one now would be a
    registry pretending to be a feature.
    """
    parser = argparse.ArgumentParser(
        prog="patchworks",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="<subcommand>")

    doctor = subparsers.add_parser(
        "doctor",
        help="can this installation run? a line per check, and the fix",
        description=DOCTOR_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    doctor.set_defaults(handler=_doctor)

    check = subparsers.add_parser(
        "check",
        help="is it actually alive? a few hundred headless ticks, and the numbers",
        description=CHECK_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    check.add_argument(
        "--ticks",
        type=int,
        default=CHECK_TICKS,
        help=f"how many ticks to run (default: {CHECK_TICKS}, about 6 s of world)",
    )
    check.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the seed for the sheaf's draw and the world's layout (default: 0)",
    )
    check.add_argument(
        "--split", default="train", help="which task split to draw from (default: train)"
    )
    check.set_defaults(handler=_check)

    dome = subparsers.add_parser(
        "dome",
        help="print the dome's shape and what construction recorded about it",
        description=DOME_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dome.set_defaults(handler=_dome)

    demo = subparsers.add_parser(
        "demo",
        help="the scene window, drivable by hand (opens a window; needs mjpython on macOS)",
        description=DEMO_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    demo.add_argument(
        "--ticks", type=int, default=100_000, help="how many ticks to run (default: 100000)"
    )
    demo.add_argument("--seed", type=int, default=0, help="the run's seed (default: 0)")
    demo.add_argument(
        "--split", default="train", help="which task split to draw from (default: train)"
    )
    demo.set_defaults(handler=_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    """`patchworks`, and `python -m patchworks`, arrive here.

    A bare `patchworks` prints the help and exits **0**, rather than the usage
    error argparse would raise for a missing required argument. Typing the
    command with nothing after it is the discovery gesture this whole entry
    point exists for — the thing a human does when they do not yet know what
    there is — and answering it with an error would be answering the question
    the ticket was written to fix with a rebuke.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)
    handler = getattr(arguments, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return handler(arguments)
