"""Nothing downstream of `info` reaches the agent (ticket #348).

`docs/spec/03-the-sandbox.md` puts privileged truth in `info` — puck poses,
goal identity, goal distance, whether the goal is satisfied — "for logging and
for the acceptance demo's instrumentation only. Feeding it to the agent defeats
the sandbox." `docs/adr/0025-coherence-is-a-motor-readback-not-a-sensory-value.md`
rules the same thing one domain along, where it costs more: a scalar the
interlocutor computes about the agent's **own output** is a reward whatever it
is labelled, and the one channel already shaped like a place to put it is
`info`.

`docs/spec/12-the-interlocutor.md`'s *Known exposure* is not that the mistake is
likely. It is that the mistake is **cheap to make again** — a future rig that
logs coherence into `info` and then feeds `info` to anything has reintroduced a
reward, and ADR-0009's *a drive is a motor edge attached deep* and
`07-local-learning-rule.md`'s *Permitted global signals* both fail at once,
**silently**. Nothing in the tick would look different. A run with a reward in
it does not crash; it works better, which is the same asymmetry
`tests/test_perturbation.py` exists for one level down, and the same argument
for a standing test rather than a review habit.

ADR-0025 states the rule. This file is what enforces it.

## Two halves, and neither subsumes the other

**The tripwire** (:class:`TestNothingInTheTickReadsInfo`) is a runtime statement
about the rig that exists: the sandbox's `info` is replaced, in flight, by an
object that raises on every way of reading it, and a real agent runs real ticks
against that. It catches any read at all, by any route — `info["x"]`,
`info.get("x")`, `len(info)`, `for k in info`, an equality test, an attribute —
without needing to know what the reader meant to do with the number. What it
cannot do is speak about a rig that has not been written: it runs
:class:`~patchworks.sandbox.PlanarPushSandbox` through
:class:`~patchworks.agent.Agent`, and a language rim with an interlocutor on it
is neither.

**The static guard** (:class:`TestNothingDownstreamOfInfoReachesTheAgent`) is
the half still standing when the interlocutor arrives. It reads every module
under `src/patchworks`, finds each value that comes off the `info` channel,
propagates that taint through assignment, and refuses to let a tainted value go
anywhere except into a name, into a `return`, or into one of the constructions
named in :data:`PERMITTED_SINKS`. Today that list has one entry —
:class:`~patchworks.agent.TickOutcome`, the record the tick hands to logging —
and every other destination fails. `info` is read **nowhere** in
`src/patchworks`; that is the invariant, and this is what holds it down.

**Written deny-by-default, and that is the whole design.** The exposure names
the shape exactly: coherence into `info`, and then `info` "fed to anything". A
blacklist of ways for a number to reach a cell cannot be finished — a keyword
argument, a stalk slice, a closure, a dict handed over later, a
`functools.partial` — and an unfinishable blacklist that reads as complete is
worse than none, which is `tests/test_perturbation.py`'s
:class:`TestBothChecksRunInCI` ruling on the same question. So a tainted value
may take exactly three steps, all of them inert, and any fourth has to be
argued into :data:`PERMITTED_SINKS` by a human editing this file. Widening the
list is cheap; widening it silently is what is impossible.

**The principle underneath, in one line: reading spreads taint, and only
leaving is a violation.** `x = info["coherence"]` is not a leak, it is a
rename — and `x` is now the channel. What fails is the step after: `x` passed
to a call that is not a permitted sink, or `x` written *through* a subscript or
an attribute into an object that outlives the statement. That is what "reaches
the agent" means operationally, and it is why the guard does not have to
enumerate the sinks a cell is behind.

**Measured, not assumed: kill-tested against eleven leaks, and it caught all
eleven.** Seven are snippets, in :class:`TestTheAnalyserCatchesALeak`, each a
shape a future rig would plausibly write. The other four
(:class:`TestTheGuardCatchesALeakPlantedInTheRealSource`) go into
`src/patchworks/agent.py`'s own `act()`, on `tests/test_perturbation.py`'s
precedent of planting in `learning.py` rather than in a fixture: a rule that
catches a hand-written example and misses the file it is aimed at is the
failure the example was supposed to rule out.

**And it can stay silent.** :class:`TestTheAnalyserPermitsWhatIsPermitted` is
five shapes the package actually writes, each of which must keep passing, plus
the blind-spot check that closes them off; and there is a control mutation —
the clip in the same function, made a no-op multiplication — which must not be
reported either. An analyser that failed on every input would satisfy the kill
tests alone, which is what those two are for.

**One false positive was found by that planting and fixed**, and it is worth
the sentence because it is the shape a leak-detector fails in. Taint used to
follow a write target's base name, so `self.sheaf.stalks[0] += info["d"]` made
`self` the channel — and `self` appears in the `env.step` unpacking, so
`observation` came back tainted too and the guard reported the *sensory rim*
alongside the real leak. Three false findings around one true one is how a
guard gets narrowed rather than believed. Only a plain binding carries taint
now; a write through a subscript or an attribute is a violation and taints
nothing further, because there is nothing further to protect once it has
happened.

**And the guard is aimed at something.**
:meth:`TestNothingDownstreamOfInfoReachesTheAgent.test_the_guard_sees_the_channel_it_guards`
pins which modules carry the channel and how many sites each has. That is this
file's own failure mode one level down: a refactor that renames the channel
leaves the analyser finding nothing to taint, reporting no violations, and
guarding nothing — passing all the while. A guard aimed at nothing does not
fail; it stops guarding.

**Neither half needs its own CI check.** Both ride on the one `pytest`
invocation `tests/test_perturbation.py`'s :class:`TestBothChecksRunInCI`
already pins whole and unfiltered on every push, and a second copy of that
whitelist would be a second place for it to drift.
"""

import ast
import textwrap
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import pytest
import torch

from patchworks.agent import Agent, run
from patchworks.graph import build_graph

#: The package the guard reads. `src/patchworks` and nothing else: this is a
#: claim about what reaches **a cell**, and every cell, sheaf, body and agent
#: lives here. `benchmarks/` is deliberately outside it — `achievability.py`'s
#: scripted controller is *supposed* to be privileged (`03-the-sandbox.md`
#: bounds the world with it precisely because it cheats, and says so), and a
#: guard that failed on the one program written to read `info` would be read as
#: noise and switched off.
PACKAGE = Path(__file__).resolve().parents[1] / "src" / "patchworks"

#: The one place a value off the `info` channel is permitted to land, by the
#: name of the thing constructed. :class:`~patchworks.agent.TickOutcome` is the
#: record `tick()` returns, and its own docstring says what it is for —
#: "Logging and instrumentation only; no cell reads it" — so `info` arriving
#: there is the channel landing where the spec puts it rather than escaping.
#:
#: A construction named here **launders**: what comes out of it is not tainted,
#: which is why `act()` returning a `TickOutcome` does not make every caller of
#: `tick()` a suspect. Reading the field back — `outcome.info` — taints again,
#: so the laundering is of the record, not of the channel inside it.
#:
#: Adding a name here is the deliberate, reviewable act this guard exists to
#: force. It is not a list of safe functions; it is a list of constructions
#: somebody has argued about. `float`, `np.asarray` and `print` are *not* here
#: and must not be added casually: each is a way for the number to acquire an
#: ordinary-looking name one line before it is written to a stalk.
PERMITTED_SINKS = frozenset({"TickOutcome"})

#: Where the channel is, and how many sites carry it — module path relative to
#: :data:`PACKAGE`, against a count of source sites.
#:
#: `agent.py` has two: `act()`'s five-tuple unpacking of `env.step`, and
#: `run()`'s two-tuple unpacking of `env.reset`. `sandbox/env.py` has two: the
#: `self._info()` call in `reset` and the one in `step`. `timescale.py` has
#: one, the same `reset` unpacking as `run()`, because its sweep arranges the
#: world for itself. Every other module in the package has none, and that is
#: asserted too — a new module that starts carrying the channel has to say so
#: here.
KNOWN_CHANNEL_SITES = {
    "agent.py": 2,
    "sandbox/env.py": 2,
    "timescale.py": 1,
}

#: How many values `gymnasium.Env.step` returns, and which of them is `info`.
#: Named rather than typed as `4` where it is used: the position rule rests on
#: it, and it is a position rather than a name precisely because Gymnasium's
#: contract is a tuple.
STEP_ARITY = 5
STEP_INFO = 4

#: The same for `reset`, which returns `(observation, info)`.
RESET_ARITY = 2
RESET_INFO = 1


# ==========================================================================
# The static analyser
# ==========================================================================
#
# Intra-procedural and name-level, which is enough for the shape being guarded
# and honest about what it is not. Two limits are worth stating rather than
# discovering:
#
# * **Across a `self`.** A tainted value parked on an instance in one method
#   and read in another crosses a boundary this does not follow. It is a blind
#   spot only in the sense that reaching it costs a violation first —
#   `self.readback = info["x"]` writes *through an attribute*, which fails
#   here — so there is no route to the far side that this guard is silent on.
#   `test_parking_the_channel_on_self_is_itself_a_violation` holds that down.
# * **Across a module.** A helper in one module returning the channel to a
#   caller in another is not traced. Inside a module it is: a function whose
#   `return` carries taint becomes a source, so `self._readback()` taints its
#   caller. The cross-module case is left because the package has no such
#   helper and inventing the machinery for one would be untested code guarding
#   a hypothetical, which is the register's own complaint about speculative
#   generality. It is recorded here rather than closed.


def _is_info_name(name: str) -> bool:
    """Is this identifier the privileged channel, under any leading underscore?

    `info`, `_info` and `__info` are one channel; the underscore is this
    repository's discard convention (`observation, _info = env.reset(...)`),
    and a discard that is then read is exactly the route this must not miss.

    Deliberately an equality after stripping, not a suffix match. A suffix
    match taints `sys.version_info`, and a guard with a false positive in the
    standard library gets narrowed rather than believed. A future rig naming
    the channel `step_info` escapes *this* rule and is caught by the position
    rule, which is the structural one and does not read names at all.
    """
    return name.lstrip("_") == "info"


def _flat_names(target: ast.expr) -> list[ast.Name] | None:
    """The `Name` nodes of a flat tuple/list target, or `None` if it is not one."""
    if not isinstance(target, (ast.Tuple, ast.List)):
        return None
    if not all(isinstance(element, ast.Name) for element in target.elts):
        return None
    return [element for element in target.elts if isinstance(element, ast.Name)]


def _is_plain_binding(target: ast.expr) -> bool:
    """Does this target bind names, rather than write into an object?

    A `Name` or a flat tuple of them rebinds and nothing else changes. A
    `Subscript` or an `Attribute` reaches into something that outlives the
    statement, and a stalk slice is exactly that shape.
    """
    return isinstance(target, ast.Name) or _flat_names(target) is not None


def _called_attribute(node: ast.AST) -> str | None:
    """The attribute a call goes through — `env.step(...)` gives `step`."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _sink_name(node: ast.AST) -> str | None:
    """The name of the thing a call constructs, for :data:`PERMITTED_SINKS`."""
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _parameters(node: ast.AST) -> list[str]:
    """Every parameter name of a function definition, in one list."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    arguments = node.args
    named = [
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
        *([arguments.vararg] if arguments.vararg else []),
        *([arguments.kwarg] if arguments.kwarg else []),
    ]
    return [argument.arg for argument in named]


@dataclass(frozen=True)
class Violation:
    """One place a value off the `info` channel leaves."""

    module: str
    line: int
    what: str

    def __str__(self) -> str:
        return f"{self.module}:{self.line}: {self.what}"


@dataclass(frozen=True)
class Site:
    """One place a value off the `info` channel is minted."""

    module: str
    line: int
    how: str


class _Scope:
    """One function body, one class body, or the module top level.

    Scopes nest and taint flows inwards only: a name tainted in an enclosing
    function is tainted inside a closure defined in it, which is how
    `patchworks.agent.run`'s inner `ticking()` is covered. It does not flow
    outwards — a nested function cannot rebind an enclosing local without
    `nonlocal`, and a `nonlocal` naming a tainted name is refused below rather
    than modelled.
    """

    def __init__(self, node: ast.AST, parent: "_Scope | None") -> None:
        self.node = node
        self.parent = parent
        #: Every node under `node` that is not inside a nested function or
        #: class. Flat, and used for seeding and propagation, where order does
        #: not matter and a set of names is the whole state.
        self.own: list[ast.AST] = []
        self.tainted: set[str] = set()


class Analysis:
    """What one module does with the `info` channel.

    Three passes, and the middle one is a fixpoint. First the module is split
    into scopes. Then taint is seeded and propagated until nothing changes —
    across scopes too, because a function whose `return` carries taint is
    itself a source and its callers have to be re-read once that is known.
    Last, every statement is checked for taint leaving.
    """

    def __init__(self, source: str, module: str) -> None:
        self.module = module
        self.tree = ast.parse(source)
        self.scopes = self._split(self.tree)
        self.info_returning: set[str] = set()
        self.violations: list[Violation] = []
        self._resolve()
        self.sites = self._sites()
        for scope in self.scopes:
            self._check_scope(scope)

    # -- scopes ---------------------------------------------------------------

    @staticmethod
    def _split(tree: ast.Module) -> list[_Scope]:
        scopes: list[_Scope] = []

        def walk(node: ast.AST, scope: _Scope) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    inner = _Scope(child, scope)
                    scopes.append(inner)
                    walk(child, inner)
                else:
                    scope.own.append(child)
                    walk(child, scope)

        root = _Scope(tree, None)
        scopes.append(root)
        walk(tree, root)
        return scopes

    # -- taint ----------------------------------------------------------------

    def _is_source(self, node: ast.AST) -> bool:
        """Is this expression itself a value off the channel?

        Three shapes, and they are the three the package can write:

        * a call to something named `info` under its underscores —
          `self._info()`, where the sandbox mints the privileged dict;
        * an attribute read of `.info` — `outcome.info`, the record's own
          field, which is where a future reader is likeliest to pick the number
          up because it is the one that looks like ordinary logging;
        * a call to a function this module has already been found to return the
          channel from, which is the transitive case and the reason the
          taint pass is a fixpoint rather than a walk.
        """
        if isinstance(node, ast.Call):
            attribute = _called_attribute(node)
            if attribute is not None and (
                _is_info_name(attribute) or attribute in self.info_returning
            ):
                return True
            if isinstance(node.func, ast.Name) and (
                _is_info_name(node.func.id) or node.func.id in self.info_returning
            ):
                return True
        if isinstance(node, ast.Attribute) and _is_info_name(node.attr):
            return True
        return False

    def _mentions_taint(self, node: ast.AST, tainted: set[str]) -> bool:
        """Does this expression read anything downstream of the channel?

        A permitted sink is opaque here: what comes out of `TickOutcome(...)`
        is a record, not the channel, and treating it otherwise would taint
        every caller of `tick()` and turn the guard into a report on the CLI.
        """
        if _sink_name(node) in PERMITTED_SINKS:
            return False
        if self._is_source(node):
            return True
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in tainted
        ):
            return True
        return any(
            self._mentions_taint(child, tainted) for child in ast.iter_child_nodes(node)
        )

    @staticmethod
    def _bindings(node: ast.AST) -> tuple[list[ast.expr], ast.expr | None]:
        """The targets a statement binds and the value it binds them from."""
        if isinstance(node, ast.Assign):
            return list(node.targets), node.value
        if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            return [node.target], node.value
        if isinstance(node, (ast.For, ast.AsyncFor)):
            return [node.target], node.iter
        if isinstance(node, ast.withitem) and node.optional_vars is not None:
            return [node.optional_vars], node.context_expr
        return [], None

    def _seed(self, scope: _Scope) -> None:
        """Every name a scope binds straight off the channel.

        Two ways in, and the first is the one that survives a rename:

        1. **Position.** A five-tuple unpacked from anything called `step`, or
           a two-tuple from anything called `reset`, is Gymnasium's contract,
           and the `info` slot is a position in it. What the receiving name is
           called does not matter and must not.
        2. **Name.** A parameter or a target called `info` under its
           underscores.
        """
        for parameter in _parameters(scope.node):
            if _is_info_name(parameter):
                scope.tainted.add(parameter)
        for node in scope.own:
            targets, value = self._bindings(node)
            if value is None:
                continue
            for target in targets:
                names = _flat_names(target)
                if names is not None:
                    attribute = _called_attribute(value)
                    if attribute == "step" and len(names) == STEP_ARITY:
                        scope.tainted.add(names[STEP_INFO].id)
                    if attribute == "reset" and len(names) == RESET_ARITY:
                        scope.tainted.add(names[RESET_INFO].id)
                for inner in ast.walk(target):
                    if isinstance(inner, ast.Name) and _is_info_name(inner.id):
                        scope.tainted.add(inner.id)

    def _propagate(self, scope: _Scope) -> None:
        """Assignment carries taint, to a fixpoint.

        A fixpoint rather than one pass, because a loop body can assign from a
        name the loop's later statements taint, and a forward pass over a
        `while` reads the two in the order they are written rather than the
        order they run.

        **Only a plain binding carries taint.** `self.sheaf.stalks[0] = info`
        does not make `self` the channel — it makes the stalk the channel, and
        the write itself is already a violation. Tainting the base name instead
        was measured, on a leak planted in `agent.py`: `self` went tainted, the
        `env.step` unpacking mentioned `self`, and `observation` came back
        tainted with it, so the guard reported the sensory rim as a leak on the
        same line as the real one. A guard whose true finding arrives inside
        three false ones is a guard that gets narrowed.
        """
        while True:
            before = set(scope.tainted)
            for node in scope.own:
                targets, value = self._bindings(node)
                if value is None or not self._mentions_taint(value, scope.tainted):
                    continue
                for target in targets:
                    if not _is_plain_binding(target):
                        continue
                    for inner in ast.walk(target):
                        if isinstance(inner, ast.Name):
                            scope.tainted.add(inner.id)
            if scope.tainted == before:
                return

    def _returns_taint(self, scope: _Scope) -> bool:
        """Does this function hand the channel back *as* its return value?

        A container return does not count, and the exclusion is load-bearing
        rather than a convenience. `PlanarPushSandbox.step` returns a tuple
        with `info` in it — that is the contract, not a leak — and counting it
        would make `step` an info-returning function, which taints *every*
        name in every unpacking of it, `observation` included. The observation
        is the sensory rim: tainting it makes the guard report the architecture
        working. The two container shapes that do carry the channel are
        Gymnasium's, and the position rule reads them by position, which is
        both narrower and exact.

        The residue is a helper returning `{"coherence": info["c"]}` to a
        caller that digs it out again. Recorded rather than closed: it is not a
        shape the package writes, and the store it would need on the way to a
        cell is a write through a subscript, which fails here anyway.
        """
        for node in scope.own:
            if not isinstance(node, ast.Return) or node.value is None:
                continue
            if isinstance(node.value, (ast.Tuple, ast.List, ast.Dict, ast.Set)):
                continue
            if self._mentions_taint(node.value, scope.tainted):
                return True
        return False

    def _resolve(self) -> None:
        """Seed, propagate and re-read until the module stops changing."""
        while True:
            before = set(self.info_returning)
            for scope in self.scopes:
                scope.tainted = (
                    set(scope.parent.tainted) if scope.parent is not None else set()
                )
                self._seed(scope)
                self._propagate(scope)
            for scope in self.scopes:
                if isinstance(scope.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if self._returns_taint(scope):
                        self.info_returning.add(scope.node.name)
            if self.info_returning == before:
                return

    # -- where the channel is -------------------------------------------------

    def _sites(self) -> list[Site]:
        """Every place the channel is minted, for the aim check.

        Position unpackings and source expressions, and nothing else: a name
        rule firing on a dataclass field annotation is the channel being
        *declared*, not read off the world, and counting it would make the
        number a fact about how `TickOutcome` is written.
        """
        found: list[Site] = []
        # `self._info()` is a source twice over — the call, and the attribute
        # it goes through — and counting both would make the number a fact
        # about how a call is spelled. The call is the site; the attribute in
        # its callee position is the same site said again.
        callees = {id(node.func) for node in ast.walk(self.tree) if isinstance(node, ast.Call)}
        for node in ast.walk(self.tree):
            if id(node) in callees and isinstance(node, ast.Attribute):
                continue
            targets, value = self._bindings(node)
            if value is not None:
                attribute = _called_attribute(value)
                for target in targets:
                    names = _flat_names(target)
                    if names is None:
                        continue
                    if (attribute == "step" and len(names) == STEP_ARITY) or (
                        attribute == "reset" and len(names) == RESET_ARITY
                    ):
                        found.append(
                            Site(self.module, node.lineno, f"unpacked from `{attribute}`")
                        )
            if self._is_source(node) and isinstance(node, (ast.Call, ast.Attribute)):
                found.append(Site(self.module, node.lineno, "read off the channel"))
        return found

    # -- what leaves ----------------------------------------------------------

    def _fail(self, node: ast.AST, what: str) -> None:
        self.violations.append(Violation(self.module, getattr(node, "lineno", 0), what))

    def _check_scope(self, scope: _Scope) -> None:
        for node in scope.own:
            if isinstance(node, (ast.stmt, ast.withitem)):
                self._check_statement(node, scope.tainted)

    def _check_statement(self, node: ast.AST, tainted: set[str]) -> None:
        """One statement, its own expressions only.

        Nested statements are separate entries in the scope's node list and are
        checked on their own turn, so this descends through expressions and
        stops at anything that is a statement.
        """
        targets, value = self._bindings(node)
        if targets:
            carried = value is not None and self._mentions_taint(value, tainted)
            for target in targets:
                if self._mentions_taint(target, tainted):
                    self._fail(
                        node,
                        "`info` is read inside the target of an assignment, which "
                        "writes it into something that outlives the statement",
                    )
                elif carried and not _is_plain_binding(target):
                    self._fail(
                        node,
                        "`info` is written through a subscript or an attribute, "
                        "which is how a stalk gets written",
                    )
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                if name in tainted:
                    self._fail(node, f"`{name}` carries `info` out of its scope")
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.stmt, ast.withitem, ast.excepthandler)):
                continue
            self._check_expression(child, tainted)

    def _check_expression(self, node: ast.AST, tainted: set[str]) -> None:
        if isinstance(node, ast.Call):
            self._check_call(node, tainted)
            return
        for child in ast.iter_child_nodes(node):
            self._check_expression(child, tainted)

    def _check_call(self, node: ast.Call, tainted: set[str]) -> None:
        """A call is where taint leaves, unless the call is a permitted sink.

        The callee itself may be a tainted read — `info.get` — and that is a
        read, not a departure; what matters is where the *result* goes, and the
        result is tainted by propagation. So only the arguments are judged
        here, and the callee is descended into for calls nested in it.
        """
        if _sink_name(node) not in PERMITTED_SINKS:
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            for argument in arguments:
                if self._mentions_taint(argument, tainted):
                    called = _sink_name(node) or "a call"
                    self._fail(
                        node,
                        f"`info` is passed to `{called}(...)`, which is not one of "
                        f"the permitted sinks {sorted(PERMITTED_SINKS)}",
                    )
                    break
        self._check_expression(node.func, tainted)
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
            self._check_expression(argument, tainted)


def _analyse(source: str, module: str = "<snippet>") -> Analysis:
    return Analysis(textwrap.dedent(source), module)


def _package_modules() -> list[Path]:
    return sorted(path for path in PACKAGE.rglob("*.py"))


# ==========================================================================
# The tripwire
# ==========================================================================


class _InfoWasRead(AssertionError):
    """A tick read the privileged channel."""


class _Tripwire:
    """An `info` that cannot be read without saying so.

    Not a `dict` subclass. A subclass would be read through `dict`'s own C
    implementation wherever anything reached past the Python level, and the
    point is to have no level that is not this one. Everything a reader could
    plausibly do — subscript, membership, iteration, length, truth, equality,
    hashing, any attribute at all — arrives here and raises.

    `__repr__` is the one exception and returns a fixed string, so that a
    failure this class causes can be reported without the reporting itself
    tripping it. It exposes no number, which is what makes the exception safe.
    """

    __slots__ = ("reads",)

    def __init__(self) -> None:
        self.reads: list[str] = []

    def _trip(self, how: str) -> None:
        self.reads.append(how)
        raise _InfoWasRead(
            f"the tick read the privileged channel ({how}). "
            "`docs/spec/03-the-sandbox.md` puts `info` on the demo surface and "
            "ADR-0025 keeps a scalar about the agent's own output off the "
            "sensory rim; a read from inside a tick is a reward arriving."
        )

    def __getitem__(self, key: object) -> object:
        self._trip(f"info[{key!r}]")

    def __contains__(self, key: object) -> bool:
        self._trip(f"{key!r} in info")
        return False

    def __iter__(self) -> object:
        self._trip("iter(info)")

    def __len__(self) -> int:
        self._trip("len(info)")
        return 0

    def __bool__(self) -> bool:
        self._trip("bool(info)")
        return False

    def __eq__(self, other: object) -> bool:
        self._trip("info ==")
        return False

    def __hash__(self) -> int:
        self._trip("hash(info)")
        return 0

    def __getattr__(self, name: str) -> object:
        self._trip(f"info.{name}")

    def __repr__(self) -> str:
        return "<tripwire info>"


class _TripwiredInfo(gym.Wrapper):
    """The sandbox, with its `info` replaced in flight by a tripwire.

    A wrapper rather than a patched env, on the precedent of
    `tests/test_agent.py`'s `_RecordsTheAction`: the agent is handed exactly
    what the contract says it gets, and the substitution happens at the seam
    the agent reads from rather than inside the world.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.wire = _Tripwire()

    def reset(self, **kwargs: object) -> tuple[object, _Tripwire]:
        observation, _info = self.env.reset(**kwargs)
        return observation, self.wire

    def step(self, action: object) -> tuple[object, float, bool, bool, _Tripwire]:
        observation, reward, terminated, truncated, _info = self.env.step(action)
        return observation, reward, terminated, truncated, self.wire


class _LeaksInfoIntoAStalk(Agent):
    """An agent with the leak the guard exists for, planted on purpose.

    Coherence-shaped: a scalar the world computed about what the agent just
    did, added to a node stalk after the external write. Nothing about it
    crashes, and on a real `info` it would simply make the run better.
    """

    def act(self, command):  # type: ignore[no-untyped-def]
        outcome = super().act(command)
        with torch.no_grad():
            self.sheaf.stalks[0] += float(outcome.info["goal_distance"])
        return outcome


TRIPWIRE_TICKS = 3


@pytest.fixture
def tripwired():
    from patchworks.sandbox import PlanarPushSandbox

    world = _TripwiredInfo(PlanarPushSandbox(split="any"))
    yield world
    world.close()


@pytest.fixture(scope="module")
def dome():
    """The real dome, not `conftest.SMALL`.

    The tripwire runs a real agent against the real sandbox, and the patch
    cells' stalks have to be the size of the render the world produces. The
    small dome is the right size for a graph test and the wrong size for a
    world, which `Agent`'s constructor says out loud rather than tolerating.
    """
    return build_graph()


class TestNothingInTheTickReadsInfo:
    """The runtime half: a real agent, real ticks, an `info` that screams."""

    def test_a_whole_run_never_touches_it(self, tripwired, dome):
        agent = Agent(
            tripwired, dome=dome, generator=torch.Generator().manual_seed(0)
        )
        for _ in run(agent, TRIPWIRE_TICKS, seed=0):
            pass
        assert tripwired.wire.reads == []

    def test_the_reset_before_the_first_tick_never_touches_it(self, tripwired, dome):
        agent = Agent(
            tripwired, dome=dome, generator=torch.Generator().manual_seed(0)
        )
        observation, _info = tripwired.reset(seed=0)
        agent.observe(observation)
        assert tripwired.wire.reads == []

    def test_the_outcome_carries_it_out_of_the_tick_unread(self, tripwired, dome):
        """The channel still *reaches* the demo surface; it is not deleted.

        A guard that passed because `info` had stopped arriving would be
        guarding an empty pipe, and `03-the-sandbox.md` wants the pipe.
        """
        agent = Agent(
            tripwired, dome=dome, generator=torch.Generator().manual_seed(0)
        )
        observation, _info = tripwired.reset(seed=0)
        agent.observe(observation)
        outcome = agent.tick()
        assert outcome.info is tripwired.wire

    def test_a_planted_leak_trips_it(self, tripwired, dome):
        """The kill test: the same rig, with a leak in it, fails."""
        agent = _LeaksInfoIntoAStalk(
            tripwired, dome=dome, generator=torch.Generator().manual_seed(0)
        )
        observation, _info = tripwired.reset(seed=0)
        agent.observe(observation)
        with pytest.raises(_InfoWasRead):
            agent.tick()
        assert tripwired.wire.reads == ["info['goal_distance']"]

    @pytest.mark.parametrize(
        "read",
        [
            pytest.param(lambda wire: wire["goal_puck"], id="subscript"),
            pytest.param(lambda wire: wire.get("goal_puck"), id="get"),
            pytest.param(lambda wire: "goal_puck" in wire, id="membership"),
            pytest.param(lambda wire: list(wire), id="iteration"),
            pytest.param(lambda wire: len(wire), id="length"),
            pytest.param(lambda wire: bool(wire), id="truth"),
            pytest.param(lambda wire: wire == {}, id="equality"),
            pytest.param(lambda wire: wire.keys(), id="keys"),
        ],
    )
    def test_every_way_of_reading_it_trips(self, read):
        """What the tripwire is worth is the list of routes it closes.

        Enumerated because a tripwire silent on `info.get` is a tripwire that
        catches the leak nobody would write and misses the one they would.
        """
        wire = _Tripwire()
        with pytest.raises(_InfoWasRead):
            read(wire)

    def test_it_can_be_reported_without_tripping_itself(self):
        assert repr(_Tripwire()) == "<tripwire info>"


# ==========================================================================
# The static guard
# ==========================================================================


class TestNothingDownstreamOfInfoReachesTheAgent:
    """The standing half: `src/patchworks`, read as source, every module."""

    def test_no_module_lets_the_channel_leave(self):
        leaked: list[Violation] = []
        for path in _package_modules():
            module = path.relative_to(PACKAGE).as_posix()
            leaked.extend(Analysis(path.read_text(encoding="utf-8"), module).violations)
        assert not leaked, "\n".join(str(violation) for violation in leaked)

    def test_the_guard_sees_the_channel_it_guards(self):
        """The aim check: the analyser is looking at the channel, not past it.

        Counted rather than merely non-empty, because the failure this catches
        is a refactor that moves the channel somewhere the source rules do not
        recognise. That leaves nothing tainted, nothing to report, and a green
        guard over an architecture with a reward in it.
        """
        counted = {}
        for path in _package_modules():
            module = path.relative_to(PACKAGE).as_posix()
            sites = Analysis(path.read_text(encoding="utf-8"), module).sites
            if sites:
                counted[module] = len(sites)
        assert counted == KNOWN_CHANNEL_SITES

    def test_the_permitted_sink_is_the_one_the_spec_names(self):
        """`TickOutcome` is on the list because its docstring earns it.

        Read out of the source rather than asserted from memory: the entry is
        justified by what that class says it is for, and a rewrite that made it
        something a cell reads should not leave the justification standing.
        """
        source = (PACKAGE / "agent.py").read_text(encoding="utf-8")
        outcome = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ClassDef) and node.name == "TickOutcome"
        )
        assert PERMITTED_SINKS == frozenset({"TickOutcome"})
        assert "no cell reads it" in (ast.get_docstring(outcome) or "")


class TestTheAnalyserCatchesALeak:
    """Seven leaks, each a shape a future rig would plausibly write.

    Kill tests, on `tests/test_perturbation.py`'s pattern: a guard nobody has
    watched fail is a guard nobody knows the reach of. Each snippet is the
    minimum that would put a number off `info` where a cell can see it, and
    each is written the way somebody adding a feature would write it — not the
    way somebody evading a check would.
    """

    def test_it_catches_info_handed_to_the_external_write(self):
        analysis = _analyse(
            """
            def act(self, command):
                observation, _r, _t, _tr, info = self.env.step(command)
                self.write(observation, info)
            """
        )
        assert analysis.violations

    def test_it_catches_a_renamed_scalar_written_to_a_stalk(self):
        analysis = _analyse(
            """
            def act(self, command):
                observation, _r, _t, _tr, info = self.env.step(command)
                coherence = info["coherence"]
                self.sheaf.stalks[0] = coherence
            """
        )
        assert analysis.violations

    def test_it_catches_the_channel_under_an_innocuous_name(self):
        """The position rule, doing the work the name rule cannot."""
        analysis = _analyse(
            """
            def drive(agent, env, action):
                observation, _r, _t, _tr, privileged = env.step(action)
                agent.observe(privileged)
            """
        )
        assert analysis.violations

    def test_it_catches_the_channel_parked_on_an_attribute(self):
        analysis = _analyse(
            """
            def act(self, command):
                outcome = super().act(command)
                self.readback = outcome.info["goal_distance"]
            """
        )
        assert analysis.violations

    def test_it_catches_a_helper_that_returns_the_channel(self):
        """The transitive case, and the reason the taint pass is a fixpoint."""
        analysis = _analyse(
            """
            def _readback(self, env, action):
                _o, _r, _t, _tr, info = env.step(action)
                return info["coherence"]

            def act(self, command):
                self.sheaf.stalks[0] = self._readback(self.env, command)
            """
        )
        assert analysis.violations

    def test_it_catches_the_channel_carried_through_a_loop(self):
        analysis = _analyse(
            """
            def act(self, command):
                observation, _r, _t, _tr, info = self.env.step(command)
                for key in info:
                    self.write(info[key])
            """
        )
        assert analysis.violations

    def test_it_catches_the_channel_used_as_an_index(self):
        """A leak with no number in it: *which* stalk is written is the signal."""
        analysis = _analyse(
            """
            def act(self, command):
                observation, _r, _t, _tr, info = self.env.step(command)
                self.sheaf.stalks[info["goal_puck"]] = 1.0
            """
        )
        assert analysis.violations


#: Four leaks planted in `agent.py`'s own `act()`, as (find, replace) against
#: the real source. Snippets test the analyser; these test the guard —
#: `tests/test_perturbation.py`'s distinction, where five leaks went into
#: `src/patchworks/learning.py` rather than into a fixture, because a rule that
#: catches a hand-written example and misses the file it is aimed at is the
#: failure the example was supposed to rule out.
#:
#: The anchor is the external write, which is the line a leak would follow: at
#: that point in `act()` the world has answered, `info` is in scope, and the
#: stalks have just been written, so everything below is what somebody adding
#: an instrument would actually type.
PLANTED = {
    "a coherence-shaped scalar added to a stalk": (
        "        self.write(observation, applied)\n"
        '        self.sheaf.stalks[0] += float(info["goal_distance"])'
    ),
    "info handed to the external write": (
        "        self.write(observation, applied, info)"
    ),
    "info parked on the agent for a later method": (
        "        self.write(observation, applied)\n        self.privileged = info"
    ),
    "info renamed once, then written to a stalk": (
        "        self.write(observation, applied)\n"
        '        readback = info["goal_satisfied"]\n'
        "        self.sheaf.stalks[1] = readback"
    ),
}

#: The line every planted leak replaces, and the control mutates around.
ANCHOR = "        self.write(observation, applied)"

#: A change to the same function that is not a leak: the clip, made a no-op
#: multiplication. It moves the code the guard reads without moving the
#: channel, so a guard that reports it is reporting the diff rather than the
#: architecture. `tests/test_perturbation.py`'s control mutation, in miniature.
CONTROL = (
    "        applied = np.clip(command, self.action_low, self.action_high)",
    "        applied = np.clip(command, self.action_low, self.action_high) * 1.0",
)


class TestTheGuardCatchesALeakPlantedInTheRealSource:
    """The measurement: four leaks put into `agent.py`, all four caught.

    Against the real module rather than a snippet, because the analyser and the
    guard are different claims. The analyser is a rule about Python; the guard
    is a rule about *this* package, and what it has to survive is the file it
    reads having a hundred other things going on in it — a `no_grad` block, a
    clip, a `TickOutcome` construction one line down, and a `self` threaded
    through all of it.
    """

    @pytest.fixture(scope="class")
    def source(self):
        return (PACKAGE / "agent.py").read_text(encoding="utf-8")

    @pytest.mark.parametrize("leak", sorted(PLANTED), ids=sorted(PLANTED))
    def test_the_leak_is_caught(self, source, leak):
        assert source.count(ANCHOR) == 1, "the anchor the leaks are planted at moved"
        planted = source.replace(ANCHOR, PLANTED[leak], 1)
        assert Analysis(planted, "agent.py").violations

    def test_the_file_it_is_planted_in_is_clean_without_it(self, source):
        assert not Analysis(source, "agent.py").violations

    def test_a_change_that_is_not_a_leak_is_not_reported(self, source):
        """Falsifiable rather than merely tight: the guard can stay silent."""
        find, replace = CONTROL
        assert source.count(find) == 1
        assert not Analysis(source.replace(find, replace, 1), "agent.py").violations


class TestTheAnalyserPermitsWhatIsPermitted:
    """The other half of the kill test: containment, not refusal.

    An analyser that reported every module would satisfy the class above and be
    worth nothing. The first five here are the shapes the package actually
    writes, and they have to keep passing for the guard to be measuring
    anything. The sixth is the boundary of the first five: the one shape that
    looks like containment and is not.
    """

    def test_the_channel_may_land_in_the_record(self):
        analysis = _analyse(
            """
            def act(self, command):
                observation, _r, _t, _tr, info = self.env.step(command)
                return TickOutcome(command=command, observation=observation, info=info)
            """
        )
        assert not analysis.violations

    def test_the_record_is_laundered_and_its_callers_are_not_suspects(self):
        """Otherwise every reader of a `TickOutcome` fails, and the guard is noise."""
        analysis = _analyse(
            """
            def act(self, command):
                observation, _r, _t, _tr, info = self.env.step(command)
                return TickOutcome(command=command, observation=observation, info=info)

            def report(agent, command):
                outcome = agent.act(command)
                print(outcome.command)
            """
        )
        assert not analysis.violations

    def test_the_world_may_return_it(self):
        analysis = _analyse(
            """
            def step(self, action):
                return self._obs(), 0.0, False, False, self._info()
            """
        )
        assert not analysis.violations

    def test_a_discarded_channel_is_not_a_leak(self):
        analysis = _analyse(
            """
            def run(agent, ticks, seed=None):
                observation, _info = agent.env.reset(seed=seed)
                agent.observe(observation)
            """
        )
        assert not analysis.violations

    def test_a_module_that_never_meets_the_channel_is_silent(self):
        analysis = _analyse(
            """
            import sys

            def interpreter():
                return sys.version_info[:2]
            """
        )
        assert not analysis.violations
        assert not analysis.sites

    def test_parking_the_channel_on_self_is_itself_a_violation(self):
        """The documented blind spot is reached only through a violation.

        Taint is not followed across a `self` from one method to another. The
        store that would set that up is a write through an attribute, which
        fails here — so the far side of the blind spot is not reachable while
        this guard is green.
        """
        analysis = _analyse(
            """
            def act(self, command):
                observation, _r, _t, _tr, info = self.env.step(command)
                self.privileged = info
            """
        )
        assert analysis.violations
