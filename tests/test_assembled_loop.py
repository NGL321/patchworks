"""The assembled loop, smoke-tested: both rules on, a few hundred ticks (#105).

**This asserts assembly, and nothing else.** The whole tick runs here for the
first time with the adapting surface actually moving under it — the world, the
two phases of `docs/spec/02-tick-semantics.md`, and then both halves of the
local learning rule as the separate phase `docs/spec/09-the-build-stack.md` §2
makes them. The order that file fixes is the one between the tick and the
rules, which #88 named and which this loop is written to stand on. It fixes no
order *between* the two rules, and this file does not need one: the bias rule
writes only the biases and the transport rule only the maps, off arrays the
tick already left behind.

The only questions asked of the loop are *did it complete*, *did both rules
move the adapting surface at all*, and *is anything non-finite*. The middle one
is there because a rule that had quietly become a no-op would satisfy the other
two, and it is asked as a bare inequality — no threshold anywhere, so not even
that middle question carries a magnitude.

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
from patchworks.learning import BiasRule, SparsityAnneal, TransportRule
from patchworks.sandbox import PlanarPushSandbox

#: The small dome the learning tests run on: 39 cells, 15 of them predicting,
#: 54 edges, built by the same rules as the real one. The full dome is not the
#: point here — every seam this exercises is present at this size, and runtime
#: is a hard constraint rather than a preference, since this stands in CI on
#: every push.
#:
#: This is the **seventh** byte-identical copy of the literal, after
#: `tests/test_tick.py`, `tests/test_restriction.py`, `tests/test_learning.py`,
#: `tests/test_transport_rule.py`, `tests/test_timescale.py` and
#: `tests/test_perturbation.py`. (`tests/test_bias_selection.py`'s `SMALL` is a
#: genuinely different dome and is not one of them.) Nothing holds the seven in
#: step: retuning one leaves the other six on the old dome silently, and
#: `tests/test_perturbation.py`'s hardcoded cell indices are derived from this
#: exact spec. One `SMALL` in a `tests/conftest.py` would fix it; that edits
#: six existing files, which #105 has no business touching while other tickets
#: are in flight.
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
#: runtime constraint, since this stands in CI on every push.
#:
#: **Measured, so that trimming it is an informed trade rather than a guess.**
#: The run below costs ~3.6s, and essentially all of it scales with this
#: constant: ~12ms a tick, of which ~5ms is physics and the 16x16 render and
#: ~7ms is the graph and the two rules. So halving `TICKS` really would buy
#: ~1.8s — it is 300 because that is what "a few hundred" asks for and 3.6s is
#: inside the budget, not because the time is unrecoverable.
#:
#: What is *not* recoverable that way is MuJoCo's GL context, created lazily
#: on the first render: ~1.2s on the development laptop, where the context is
#: process-global, so a second environment's first reset costs ~0.05s and a
#: full-suite run has already paid it elsewhere before reaching this file.
#: Both numbers are that machine's. CI runs `MUJOCO_GL=osmesa`, where the
#: context is per-renderer instead, so there the cost is paid again here — it
#: is still one render's worth rather than `TICKS` of them.
TICKS = 300


@pytest.fixture
def env():
    world = PlanarPushSandbox(split="any", image_size=IMAGE_SIZE)
    yield world
    world.close()


@pytest.fixture
def agent(env):
    """Built, not started. `run` arranges the world; the order test does it itself."""
    return Agent(
        env, dome=build_graph(SMALL), generator=torch.Generator().manual_seed(0)
    )


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
    # **Not the default anneal**, and the reason is coverage rather than taste.
    # `DEFAULT_ANNEAL_HORIZON` is 1000 steps and this run takes 299, so at the
    # default the sparsity pressure would never once reach its ceiling and the
    # loop would be smoke-tested only on the ramp -- while #97 spends
    # essentially all of its time on the flat part, at full `λ`, which is also
    # where the sparsity term pushes hardest against the gauge projection. A
    # horizon of half the run puts both regimes inside it. It configures this
    # run and changes no default.
    transport = TransportRule(agent.sheaf, anneal=SparsityAnneal(horizon=TICKS // 2))

    # The adapting surface as it was drawn, to compare against at the end.
    # Cloned, because both rules write theirs in place.
    initial_biases = {
        name: parameter.detach().clone()
        for name, parameter in agent.sheaf.biases.named_parameters()
    }
    initial_maps = agent.sheaf.maps.maps.detach().clone()

    ticking = run(agent, TICKS, seed=0)
    next(ticking)
    bias.step()
    # `transport.pressure` is the `λ` the step about to run will compose into
    # its objective, so recording it here witnesses a step that **took** the
    # ceiling rather than a schedule that merely reached it after the last one.
    pressures = []
    for _ in ticking:
        bias.step()
        pressures.append(transport.pressure)
        transport.step()

    # It completed, and a step ran on the anneal's flat part.
    assert agent.sheaf.ticks == TICKS
    assert transport.steps == TICKS - 1
    assert max(pressures) == pytest.approx(transport.anneal.pressure)

    # Both rules actually moved something. `transport.steps` counts calls, not
    # work, and the bias rule counts nothing at all, so without this a rule
    # that had become a silent no-op -- `argnums` mis-scoped to something that
    # is not the adapting surface, the **missing** gradient the `torch.func`
    # idiom was chosen to fail towards -- would leave this test green while the
    # loop it exists to assemble did nothing. Both mutations were planted and
    # both are caught here.
    #
    # The biases are compared with `torch.equal`, so the claim is that they
    # moved and not that they moved by anything in particular.
    for name, parameter in agent.sheaf.biases.named_parameters():
        assert not torch.equal(parameter, initial_biases[name]), name
    # **The maps are compared the same way, but only at the interior
    # endpoints**, and the exclusion is what keeps this an equality rather than
    # a threshold. `RestrictionMaps` scales every map to `INITIAL_NORM` at
    # construction, which in float32 lands a bit short of exactly 1; a pinned
    # map's gauge band is the single point 1, so the projection that runs after
    # every transport step rewrites its last bit -- `6e-8`, on 8 of this dome's
    # 28 pinned endpoints at the seed drawn here -- whether or not a gradient
    # ever landed. An inequality there would be satisfied by rounding, and a
    # transport rule that computed its gradient and forgot to apply it would
    # pass: planted, and it did.
    #
    # An interior map's band is `[1/rho, rho]` with the norm starting strictly
    # inside it, so `project()`'s rescale is exactly `1.0` and bit-preserving
    # until a step moves the norm. Bare inequality on those rows is therefore
    # precisely "something other than rounding happened", with no threshold in
    # it -- and so no claim about how far anything moved.
    #
    # Detached before indexing, as the rest of the suite detaches before
    # comparing: indexing a live `nn.Parameter` under ambient grad mode puts an
    # `IndexBackward0` on the tape, and this file least of all should be the
    # one that leaves a node there.
    interior = ~agent.sheaf.maps.pinned
    final_maps = agent.sheaf.maps.maps.detach()
    assert not torch.equal(final_maps[interior], initial_maps[interior])

    # Finite, and that is the whole of what is asked. The ticket names the
    # biases, the restriction maps and the node stalks; the other three are
    # here because they are the rest of what a tick carries into the next one,
    # and a `nan` sitting in any of them is a `nan` the run is already using. A
    # chart is the inference phase's state; `broadcast` is read a tick later as
    # the unit-delayed neighbour belief; `incoming` *is* the array the
    # transport rule descends on. Each would otherwise surface only through a
    # later tick's node stalk -- and on the last tick there is no later tick.
    for name, parameter in agent.sheaf.biases.named_parameters():
        assert torch.isfinite(parameter).all(), name
    assert torch.isfinite(final_maps).all()
    assert torch.isfinite(agent.sheaf.stalks).all()
    assert torch.isfinite(agent.sheaf.charts).all()
    assert torch.isfinite(agent.sheaf.broadcast).all()
    assert torch.isfinite(agent.sheaf.incoming).all()


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
