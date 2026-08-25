"""The assembled loop, smoke-tested: both rules on, a few hundred ticks (#105).

**This asserts assembly, and nothing else.** The whole tick runs here for the
first time with the adapting surface actually moving under it — the world, the
two phases of `docs/spec/02-tick-semantics.md`, the bias rule and the transport
rule, in the order `docs/spec/07-local-learning-rule.md` puts them in — and the
only questions asked of it are *did it complete* and *is anything non-finite*.

Nothing here is a result. There is no claim about the magnitude, direction or
trend of any learned quantity, nothing about learning, convergence, timescales
or where disagreement went, and no trace anyone is meant to read. **#97 is where
the architecture is asked anything**: it is the first ticket of the run phase,
an unbudgeted human-watched experiment with no compute estimate, and what this
file exists to do is make sure that experiment is not also the first time the
pieces are asked to fit together. If a reviewer can describe a result this file
establishes about the architecture, it has overreached and should be cut back.

The seams it covers are real rather than hypothetical. #88 had to change
`tick.py` to retain `prior_charts` and `prior_evidence`, state the tick was not
keeping, because the bias rule needs the inference phase's *inputs*; #89 reaches
the same way into the message-passing phase's disagreement. Two rules, two
phases, one tick — and until this file nothing ran all of it at once.

What it does **not** cover, because another file already does: cross-cell
leakage in either rule (`tests/test_perturbation.py`, the locality guard's other
half), the rules' own arithmetic (`tests/test_learning.py`,
`tests/test_transport_rule.py`), and tick ordering against the world
(`tests/test_agent.py`).
"""

import pytest
import torch

from patchworks.agent import Agent, run
from patchworks.graph import DomeSpec, build_graph
from patchworks.learning import BiasRule, TransportRule
from patchworks.sandbox import PlanarPushSandbox

#: The same small dome `tests/test_learning.py`, `tests/test_transport_rule.py`
#: and `tests/test_perturbation.py` run on: 39 cells, 15 of them predicting, 54
#: edges, built by the same rules as the real one. The full dome is not the
#: point here — every seam this exercises is present at this size, and runtime
#: is a hard constraint rather than a preference, since this stands in CI on
#: every push.
SMALL = DomeSpec(
    patch_grid=4,
    vision_sides=(2,),
    somatomotor_sizes=(4,),
    core_sizes=(4, 3),
    core_degree=4,
    apex_degree=3,
)

#: The render the small dome tiles. A patch cell's node stalk is 48 numbers —
#: a 4x4 RGB patch, written raw — and this dome lays 4x4 of them over the
#: render, so the render is 16x16 and `Agent` would refuse anything else.
IMAGE_SIZE = 16

#: A few hundred, as the ticket asks. Long enough that both rules have taken
#: hundreds of steps into each other's state, short enough to stay inside the
#: runtime constraint: the run below adds ~3s to the suite, of which ~2s is
#: MuJoCo rendering rather than anything the graph does.
TICKS = 300


@pytest.fixture
def env():
    world = PlanarPushSandbox(split="any", image_size=IMAGE_SIZE)
    yield world
    world.close()


@pytest.fixture
def agent(env):
    """Built, not started. `run` arranges the world; the order test does it itself."""
    return Agent(env, dome=build_graph(SMALL), generator=torch.Generator().manual_seed(0))


def test_the_assembled_loop_runs_with_both_rules_on(agent):
    """A few hundred ticks, both rules every tick, and nothing gone non-finite.

    The loop's shape is the one #88 named — `agent.tick()`, then the rules,
    never the reverse — and it is written unguarded on purpose. Neither
    `rule.step()` sits behind an `if` that re-derives what the rule already
    knows, so the wrong order would hit the rules' own refusals and stop this
    run rather than train on the constructor's zero placeholders. That the
    refusals are loud is asserted below, in
    :func:`test_the_wrong_order_stops_the_run_rather_than_training_on_placeholders`.

    The transport rule joins on the second tick and the prologue is where the
    **unit delay** lives: the message-passing phase reads the broadcast buffer
    as it stood before the phase, so the first tick reconciles against the
    constructor's zeros and leaves `incoming` zero. The bias rule needs one tick
    because prediction error is a cell's own quantity and crosses no edge.
    """
    bias = BiasRule(agent.sheaf)
    transport = TransportRule(agent.sheaf)

    ticking = run(agent, TICKS, seed=0)
    next(ticking)
    bias.step()
    for _ in ticking:
        bias.step()
        transport.step()

    # It completed, and both rules ran on every tick they were owed rather than
    # being skipped into a run that only looks like one.
    assert agent.sheaf.ticks == TICKS
    assert transport.steps == TICKS - 1

    # Finite, and that is the whole of what is asked. The ticket names the
    # biases, the restriction maps and the node stalks; the charts are here as
    # well because a chart is the inference phase's state carried across ticks,
    # so it is the fourth place a `nan` could sit and be carried forward.
    for name, parameter in agent.sheaf.biases.named_parameters():
        assert torch.isfinite(parameter).all(), name
    assert torch.isfinite(agent.sheaf.maps.maps).all()
    assert torch.isfinite(agent.sheaf.stalks).all()
    assert torch.isfinite(agent.sheaf.charts).all()


def test_the_wrong_order_stops_the_run_rather_than_training_on_placeholders(agent):
    """The order is `agent.tick()` then the rules, and the reverse is loud.

    The failure this rules out is the quiet one: a fresh sheaf's `prior_charts`,
    `prior_evidence` and `incoming` are zeros, which are a perfectly well-formed
    record of nothing, so a loop written the other way round would report a
    gradient rather than a mistake. The rules' refusals are unit-tested in
    `tests/test_learning.py` and `tests/test_transport_rule.py`; what is checked
    here is that the assembled loop is standing on them.
    """
    observation, _info = agent.env.reset(seed=0)
    agent.observe(observation)
    bias = BiasRule(agent.sheaf)
    transport = TransportRule(agent.sheaf)

    with pytest.raises(ValueError, match="needs a tick to learn from"):
        bias.step()
    with pytest.raises(ValueError, match="needs two ticks to learn from"):
        transport.step()

    # And after one tick the transport rule still refuses, which is what the
    # run's prologue is for rather than an off-by-one in it.
    agent.tick()
    bias.step()
    with pytest.raises(ValueError, match="needs two ticks to learn from"):
        transport.step()
