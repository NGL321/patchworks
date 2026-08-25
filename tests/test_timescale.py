"""The clock divisor, as an instrument (ticket #87).

Three things are held down here, and the third is the point of the ticket.

* **Off is off.** With no cell divided, a tick is bit-identical to #86's --
  every tensor, bit for bit, over many ticks, and through a whole agent tick
  with the world in it.
* **On, it decimates exactly one thing**: the divided cell's own body update.
  Its chart, its prediction and its private component stand between updates;
  the message-passing phase still runs for it every tick, because
  reconciliation is what persistence leaves running too.
* **Nothing in the architecture reads a divisor or a timescale.** Two
  independent guards, both written so that a module which does not exist yet --
  #88's bias rule and #89's transport rule -- is covered the day it lands: an
  AST scan over every module of the package but the divisor's own, and a walk
  of the object graph a cell computation is handed.
"""

import ast
import pathlib
import re
import types

import numpy as np
import pytest
import torch

from patchworks.agent import Agent
from patchworks.graph import DomeSpec, build_graph
from patchworks.sandbox import PlanarPushSandbox
from patchworks.tick import Sheaf
from patchworks.timescale import ClockDivisor

SMALL = DomeSpec(
    patch_grid=4,
    vision_sides=(2,),
    somatomotor_sizes=(4,),
    core_sizes=(4, 3),
    core_degree=4,
    apex_degree=3,
)

TICKS = 12


@pytest.fixture
def dome():
    return build_graph(SMALL)


@pytest.fixture(scope="module")
def full_dome():
    """The dome the demo actually runs on, for the tests that drive the world."""
    return build_graph()


def built(dome, seed=0):
    """A sheaf, stirred, so that no equality below is an equality between zeros."""
    sheaf = Sheaf(dome, generator=torch.Generator().manual_seed(seed))
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        sheaf.stalks[: sheaf.layout.total] = torch.randn(
            sheaf.layout.total, generator=generator
        )
        sheaf.charts.normal_(0.0, 1.0, generator=generator)
    return sheaf


STATE = ("stalks", "charts", "prediction", "broadcast", "incoming")


def snapshot(sheaf):
    return {name: getattr(sheaf, name).clone() for name in STATE}


def assert_identical(left, right, when=""):
    for name in STATE:
        a, b = getattr(left, name), getattr(right, name)
        assert torch.equal(a, b), f"{name} differs {when}"
    assert left.ticks == right.ticks


def private(sheaf, cell_id):
    """The private component of one predicting cell's node stalk."""
    row = sheaf.dome.predicting.index(cell_id)
    return sheaf.stalk(cell_id) * sheaf.dome.private_projection[row]


def a_private_cell(dome):
    """A core cell with private dimension to hold, and its population row."""
    dimensions = dome.private_dimensions
    for row, cell_id in enumerate(dome.predicting):
        if dimensions[row] > 0 and dome.degrees[cell_id] > 1:
            return cell_id
    raise AssertionError("this dome has no cell with private dimension")


class TestTheDivisorIsOffByDefault:
    def test_a_divisor_constructed_empty_divides_nothing(self, dome):
        divisor = ClockDivisor(dome)
        assert divisor.divisors == {}
        assert not divisor
        assert divisor.held(0) == () and divisor.held(7) == ()

    def test_the_tick_is_bit_identical_with_the_divisor_off(self, dome):
        plain, divided = built(dome), built(dome)
        off = ClockDivisor(dome)
        for tick in range(TICKS):
            plain.tick()
            off.advance(divided)
            assert_identical(plain, divided, when=f"at tick {tick}")

    def test_a_divisor_of_one_is_indistinguishable_from_off(self, dome):
        plain, divided = built(dome), built(dome)
        every_tick = ClockDivisor.uniform(dome, 1)
        for tick in range(TICKS):
            plain.tick()
            every_tick.advance(divided)
            assert_identical(plain, divided, when=f"at tick {tick}")

    def test_switching_it_off_mid_run_returns_to_the_same_tick(self, dome):
        plain, divided = built(dome), built(dome)
        divisor = ClockDivisor(dome, {a_private_cell(dome): 3})
        for _ in range(TICKS):
            divisor.advance(divided)
        divisor.clear()
        # The two have diverged; what is asserted is that the *tick* is #86's
        # again, so from a common state they now agree bit for bit.
        for name, value in snapshot(divided).items():
            setattr(plain, name, value.clone())
        plain.ticks = divided.ticks
        for tick in range(TICKS):
            plain.tick()
            divisor.advance(divided)
            assert_identical(plain, divided, when=f"after switching off, tick {tick}")

    def test_the_divisor_never_puts_anything_on_a_tape(self, dome):
        sheaf = built(dome)
        divisor = ClockDivisor(dome, {a_private_cell(dome): 2})
        for _ in range(TICKS):
            divisor.advance(sheaf)
        sheaf.assert_no_tape()


class TestWhatADivisorHolds:
    def test_a_divided_cell_runs_its_body_every_k_ticks(self, dome):
        sheaf = built(dome)
        cell = a_private_cell(dome)
        row = dome.predicting.index(cell)
        divisor = ClockDivisor(dome, {cell: 3})
        moved = []
        for _ in range(9):
            before = sheaf.charts[row].clone()
            divisor.advance(sheaf)
            moved.append(not torch.equal(before, sheaf.charts[row]))
        assert moved == [True, False, False] * 3

    def test_an_undivided_cell_runs_its_body_every_tick(self, dome):
        sheaf = built(dome)
        divided = a_private_cell(dome)
        other = next(c for c in dome.predicting if c != divided)
        row = dome.predicting.index(other)
        divisor = ClockDivisor(dome, {divided: 3})
        for _ in range(9):
            before = sheaf.charts[row].clone()
            divisor.advance(sheaf)
            assert not torch.equal(before, sheaf.charts[row])

    def test_the_first_tick_updates_every_cell(self, dome):
        divisor = ClockDivisor.uniform(dome, 5)
        assert divisor.held(0) == ()
        assert divisor.held(1) == tuple(sorted(dome.predicting))

    def test_the_prediction_stands_while_the_cell_is_held(self, dome):
        sheaf = built(dome)
        cell = a_private_cell(dome)
        row = dome.predicting.index(cell)
        divisor = ClockDivisor(dome, {cell: 4})
        divisor.advance(sheaf)
        standing = sheaf.prediction[row].clone()
        for _ in range(3):
            divisor.advance(sheaf)
            assert torch.equal(standing, sheaf.prediction[row])

    def test_the_private_component_is_exactly_what_stands(self, dome):
        sheaf = built(dome)
        cell = a_private_cell(dome)
        divisor = ClockDivisor(dome, {cell: 4})
        divisor.advance(sheaf)
        standing = private(sheaf, cell).clone()
        for _ in range(3):
            divisor.advance(sheaf)
            assert torch.equal(standing, private(sheaf, cell))

    def test_the_held_cell_is_still_reconciled_every_tick(self, dome):
        """A schedule decimates the cell's body, never the message-passing phase.

        The reconciled component keeps moving under a divisor exactly as it does
        under persistence, which is what makes the two interchangeable -- and it
        is also what leaves the motor pathway intact, since the commanded
        components are what reconciliation fills.
        """
        sheaf = built(dome)
        cell = a_private_cell(dome)
        divisor = ClockDivisor(dome, {cell: 4})
        divisor.advance(sheaf)
        for _ in range(3):
            before = sheaf.stalk(cell).clone()
            divisor.advance(sheaf)
            assert not torch.equal(before, sheaf.stalk(cell))

    def test_a_held_cell_still_broadcasts_on_every_incident_edge(self, dome):
        sheaf = built(dome)
        cell = a_private_cell(dome)
        divisor = ClockDivisor(dome, {cell: 4})
        divisor.advance(sheaf)
        for _ in range(3):
            before = sheaf.broadcast.clone()
            divisor.advance(sheaf)
            for edge_id in dome.incident[cell]:
                edge = dome.edges[edge_id]
                side = 0 if edge.u == cell else 1
                pair = 2 * edge_id + side
                assert not torch.equal(before[pair], sheaf.broadcast[pair])


class TestTheDivisorOnlyGatesTheCommit:
    """The divisor changes *which* results are committed and nothing a cell computes.

    This is the behavioural half of the prohibition: on the same pre-tick state,
    every cell that is not held computes bit-exactly what it computes with no
    divisor in play at all, so the divisor is provably not an input to the body.
    """

    def test_an_active_cell_computes_what_it_would_have_computed(self, dome):
        undivided, divided = built(dome), built(dome)
        cell = a_private_cell(dome)
        held_row = dome.predicting.index(cell)
        divisor = ClockDivisor(dome, {cell: 2})

        undivided.ticks = divided.ticks = 1  # a tick on which the cell is held
        before = snapshot(divided)
        undivided.tick()
        divisor.advance(divided)

        assert divisor.held(1) == (cell,)
        for row in range(len(dome.predicting)):
            if row == held_row:
                continue
            assert torch.equal(undivided.charts[row], divided.charts[row])
            assert torch.equal(undivided.prediction[row], divided.prediction[row])
        assert torch.equal(divided.charts[held_row], before["charts"][held_row])
        assert torch.equal(divided.prediction[held_row], before["prediction"][held_row])

    def test_the_body_is_indifferent_to_which_cells_are_divided(self, dome):
        """Two different schedules, same state: the body's output is the same."""
        one, other = built(dome), built(dome)
        cells = list(dome.predicting)
        one.ticks = other.ticks = 1
        ClockDivisor(dome, {cells[0]: 2}).advance(one)
        ClockDivisor(dome, {cells[1]: 2, cells[2]: 3}).advance(other)
        rows = [dome.predicting.index(c) for c in cells[:3]]
        for row in range(len(dome.predicting)):
            if row in rows:
                continue
            assert torch.equal(one.charts[row], other.charts[row])
            assert torch.equal(one.prediction[row], other.prediction[row])


# -- the prohibition ------------------------------------------------------

#: The vocabulary a schedule is written in. Nothing in the architecture may name
#: any of it -- not a variable, not an attribute, not a parameter, not a
#: function. Docstrings and messages are exempt by construction: the scan reads
#: identifiers out of the syntax tree and never touches a string constant, which
#: is what lets `tick.py` go on saying that the reconciliation gain is not a
#: timescale knob.
VOCABULARY = re.compile(r"divisor|timescale|clock|schedule|cadence", re.IGNORECASE)

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "src" / "patchworks"

#: The two files the scan does not read, and why. `timescale.py` **is** the
#: instrument, and it is outside the architecture by construction. `__init__.py`
#: is the package manifest and is asserted below to hold no code at all.
EXEMPT = {"timescale.py", "__init__.py"}


def architecture_modules():
    """Every module of the package but the divisor's own, found by glob.

    Found rather than listed, so that a module which does not exist yet is
    scanned the day it is written -- which is what makes this assertion cover
    #88's bias rule and #89's transport rule rather than quietly not covering
    them.
    """
    return sorted(p for p in PACKAGE.rglob("*.py") if p.name not in EXEMPT)


def named_in(source):
    """Every identifier a module's code uses: names, attributes, parameters, keywords.

    String constants are deliberately not read.
    """
    names = set()
    for node in ast.walk(ast.parse(source)):
        for attribute in ("id", "attr", "name", "arg", "module"):
            value = getattr(node, attribute, None)
            if isinstance(value, str):
                names.add(value)
        if isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        if isinstance(node, ast.alias):
            names.update(part for part in (node.name, node.asname) if part)
    return names


def scan(source):
    return sorted(name for name in named_in(source) if VOCABULARY.search(name))


class TestNothingInTheArchitectureNamesATimescale:
    def test_no_module_names_a_divisor_or_a_timescale(self):
        offences = {
            path.relative_to(PACKAGE).as_posix(): scan(path.read_text())
            for path in architecture_modules()
        }
        assert {k: v for k, v in offences.items() if v} == {}

    def test_the_scan_covers_every_module_the_package_has(self):
        scanned = {p.relative_to(PACKAGE).as_posix() for p in architecture_modules()}
        assert {
            "agent.py",
            "body.py",
            "graph.py",
            "restriction.py",
            "tick.py",
        } <= scanned
        # Pinned so that exempting a module is a visible edit to this test
        # rather than something a later ticket can do in passing.
        assert EXEMPT == {"timescale.py", "__init__.py"}

    def test_the_exempt_manifest_holds_no_code(self):
        tree = ast.parse((PACKAGE / "__init__.py").read_text())
        assert not [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]

    @pytest.mark.parametrize(
        "source",
        [
            "def rule(sheaf, divisor):\n    return divisor\n",
            "def rule(sheaf):\n    return sheaf.clock_divisor\n",
            "def rule(sheaf):\n    return step(sheaf, timescale=1)\n",
            "class Rule:\n    schedule = None\n",
        ],
    )
    def test_the_scan_catches_a_rule_that_reads_one(self, source):
        assert scan(source), "the scan would have let this through"

    def test_the_scan_does_not_flag_a_rule_that_reads_none(self):
        assert not scan(
            "def rule(sheaf, biases):\n"
            "    return sheaf.prediction - sheaf.stalks[sheaf.layout.pad]\n"
        )


class TestADivisorIsNotReachableFromAnyCellComputation:
    """The structural half: a rule cannot read what it cannot reach.

    Both learning rules will be handed the sheaf, the body, the biases, the
    maps and the dome, because those are the only objects there are. If no
    divisor is reachable from any of them, no rule can consult one -- which is
    an assertion about #88 and #89 that holds before either exists.
    """

    def walk(self, root, limit=200_000):
        seen, stack, found = set(), [("<root>", root)], []
        while stack:
            path, obj = stack.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            assert len(seen) < limit, "the object walk ran away"
            if isinstance(obj, ClockDivisor):
                found.append(path)
                continue
            leaf = (str, bytes, int, float, bool, type(None))
            if isinstance(obj, leaf + (torch.Tensor, np.ndarray)):
                continue
            if isinstance(obj, type) or isinstance(obj, types.ModuleType):
                continue
            if isinstance(obj, (list, tuple, set, frozenset)):
                stack.extend((f"{path}[{i}]", v) for i, v in enumerate(obj))
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    stack.append((f"{path}[{key!r}]", value))
                    if VOCABULARY.search(str(key)):
                        found.append(f"{path} keyed by {key!r}")
            else:
                for name, value in getattr(obj, "__dict__", {}).items():
                    if VOCABULARY.search(name):
                        found.append(f"{path}.{name}")
                    stack.append((f"{path}.{name}", value))
        return found

    def test_nothing_a_rule_is_handed_can_reach_a_divisor(self, dome):
        sheaf = built(dome)
        ClockDivisor(dome, {a_private_cell(dome): 3}).advance(sheaf)
        for root in (sheaf, sheaf.dome, sheaf.body, sheaf.biases, sheaf.maps):
            assert self.walk(root) == []

    def test_the_walk_would_catch_a_divisor_stashed_on_the_sheaf(self, dome):
        sheaf = built(dome)
        sheaf.instrument = ClockDivisor(dome, {a_private_cell(dome): 3})
        assert self.walk(sheaf) == ["<root>.instrument"]

    def test_the_walk_would_catch_one_hidden_deeper(self, dome):
        sheaf = built(dome)
        sheaf.maps.anything = [{"held": ClockDivisor(dome)}]
        assert self.walk(sheaf)


class TestRefusals:
    def test_a_boundary_cell_cannot_be_divided(self, dome):
        with pytest.raises(ValueError, match="boundary"):
            ClockDivisor(dome, {dome.boundary[0]: 2})

    def test_a_cell_of_another_dome_is_refused(self, dome):
        with pytest.raises(ValueError, match="not a cell"):
            ClockDivisor(dome, {len(dome.cells) + 1: 2})

    @pytest.mark.parametrize("k", [0, -1, 2.0, True, "2"])
    def test_k_is_a_whole_number_of_ticks(self, dome, k):
        with pytest.raises(ValueError, match="whole number of ticks"):
            ClockDivisor(dome, {dome.predicting[0]: k})

    def test_a_sheaf_on_another_dome_is_refused(self, dome):
        other = build_graph(SMALL)
        with pytest.raises(ValueError, match="different dome"):
            ClockDivisor(dome).advance(built(other))

    def test_by_level_divides_a_level_and_leaves_the_rest_alone(self, dome):
        levels = {dome.cells[c].index.level for c in dome.predicting}
        deepest = max(levels)
        divisor = ClockDivisor.by_level(dome, {deepest: 4})
        assert set(divisor.divisors) == {
            c for c in dome.predicting if dome.cells[c].index.level == deepest
        }
        assert set(divisor.divisors.values()) == {4}


class TestTheSameGraphRunsBothWays:
    """The mode is switchable without a rebuild, against the same demo."""

    @pytest.fixture
    def world(self):
        env = PlanarPushSandbox(split="any")
        yield env
        env.close()

    @pytest.fixture
    def other_world(self):
        env = PlanarPushSandbox(split="any")
        yield env
        env.close()

    def agent(self, env, dome):
        started = Agent(env, dome=dome, generator=torch.Generator().manual_seed(0))
        observation, _info = env.reset(seed=0)
        started.observe(observation)
        return started

    def test_a_whole_agent_tick_is_bit_identical_with_the_divisor_off(
        self, world, other_world, full_dome
    ):
        plain = self.agent(world, full_dome)
        divided = self.agent(other_world, full_dome)
        off = ClockDivisor(full_dome)
        for tick in range(6):
            outcome = plain.tick()
            through_the_instrument = off.tick(divided)
            assert np.array_equal(outcome.command, through_the_instrument.command)
            assert np.array_equal(outcome.applied, through_the_instrument.applied)
            assert_identical(plain.sheaf, divided.sheaf, when=f"at tick {tick}")

    def test_the_same_graph_runs_both_ways_and_the_instrument_bites(
        self, world, other_world, full_dome
    ):
        plain = self.agent(world, full_dome)
        divided = self.agent(other_world, full_dome)
        divisor = ClockDivisor.uniform(full_dome, 2)

        # Nothing is rebuilt to switch modes: the same dome, body, biases and
        # maps drive both halves of this run.
        sheaf = divided.sheaf
        was = (divided.dome, sheaf.body, sheaf.biases, sheaf.maps)
        commands = []
        for _ in range(8):
            commands.append((plain.tick().command, divisor.tick(divided).command))
        assert any(not np.array_equal(a, b) for a, b in commands)
        assert (divided.dome, sheaf.body, sheaf.biases, sheaf.maps) == was
        assert divided.sheaf is sheaf

        # And back, mid-run, with no rebuild either.
        divisor.clear()
        divided.tick()
        divisor.tick(divided)

    def test_a_run_under_the_instrument_arranges_the_world_when_it_is_called(
        self, world, full_dome
    ):
        agent = self.agent(world, full_dome)
        divisor = ClockDivisor.uniform(full_dome, 3)
        before = agent.sheaf.stalks.clone()
        ticking = divisor.run(agent, 3, seed=1)
        # `run` resets and writes the boundary cells eagerly, exactly as
        # `patchworks.agent.run` does, so the world is arranged before the
        # first `next()`.
        assert not torch.equal(before, agent.sheaf.stalks)
        assert agent.sheaf.ticks == 0
        assert len(list(ticking)) == 3
        assert agent.sheaf.ticks == 3
