"""The whole tick: world to graph to torque to world (ticket #86).

The tracer bullet. What these hold down is the **ordering** -- external writes
land after the message-passing phase and win -- and the two things
`docs/spec/02-tick-semantics.md` says fall out of it without machinery: the
drive's standing assertion actually stands, and the motor pathway is untouched.

The body is untrained and the maps are at their initial values, so the arm
flails. That is expected and is not what any of this checks.
"""


import gymnasium as gym
import numpy as np
import pytest
import torch

from patchworks.agent import DRIVE_ASSERTION, PIXEL_SCALE, Agent, run
from patchworks.graph import CellKind, build_graph
from patchworks.sandbox import PlanarPushSandbox
from patchworks.tick import DEFAULT_GAMMA, Sheaf

TICKS = 250


@pytest.fixture(scope="module")
def dome():
    return build_graph()


@pytest.fixture
def env():
    world = PlanarPushSandbox(split="any")
    yield world
    world.close()


@pytest.fixture
def agent(env, dome):
    started = Agent(env, dome=dome, generator=torch.Generator().manual_seed(0))
    observation, _info = env.reset(seed=0)
    started.observe(observation)
    return started


class _RecordsTheAction(gym.Wrapper):
    """Catches the action in flight, between the agent and the sandbox.

    The sandbox clips internally, so `data.ctrl` is where an illegal action and
    a legal one stop being distinguishable. This looks one step earlier, which
    is the only place the difference exists.
    """

    def __init__(self, env):
        super().__init__(env)
        self.stepped = None

    def step(self, action):
        self.stepped = np.array(action, copy=True)
        return self.env.step(action)


def patch_cells(dome):
    return [c for c in dome.cells if c.kind is CellKind.PATCH]


def sensor_cells(dome, kind):
    """The somatomotor sensor cells of one kind, in joint order.

    They tile the column's construction index two to a joint, so the ordinal is
    which joint the cell reports on -- the same rule the agent's write table is
    built from, read here independently of it.
    """
    return sorted(
        (c for c in dome.cells if c.kind is kind), key=lambda c: c.index.position
    )


class TestTheOrdering:
    def test_the_world_s_write_is_the_last_word(self, agent, dome):
        outcome = agent.tick()
        observation = outcome.observation
        # Every sensory boundary cell holds what the world just wrote, not what
        # reconciliation left there a moment earlier.
        for joint, cell in enumerate(sensor_cells(dome, CellKind.PROPRIOCEPTIVE)):
            assert torch.allclose(
                agent.sheaf.stalk(cell.id),
                torch.tensor(
                    [observation["qpos"][joint], observation["qvel"][joint]]
                ),
                atol=1e-6,
            )
        for joint, cell in enumerate(sensor_cells(dome, CellKind.TOUCH)):
            assert float(agent.sheaf.stalk(cell.id)) == pytest.approx(
                observation["touch"][joint], abs=1e-6
            )

    def test_reconciliation_edits_a_boundary_stalk_and_the_write_overwrites_it(
        self, agent, dome
    ):
        # The ordering only means something because reconciliation is free to
        # edit a boundary cell's node stalk. If it were not, "the write wins"
        # would be a statement about nothing.
        cell = sensor_cells(dome, CellKind.PROPRIOCEPTIVE)[0]
        agent.tick()  # the graph's first word has to reach the edge buffers
        before = agent.sheaf.stalk(cell.id).clone()
        agent.sheaf.tick()
        assert not torch.equal(agent.sheaf.stalk(cell.id), before)

        observation, _r, _t, _tr, _i = agent.env.step(agent.command())
        agent.write(observation, agent.command())
        assert torch.allclose(
            agent.sheaf.stalk(cell.id),
            torch.tensor([observation["qpos"][0], observation["qvel"][0]]),
            atol=1e-6,
        )

    def test_the_drive_s_assertion_stands(self, agent):
        for _ in range(5):
            agent.tick()
            assert torch.equal(
                agent.sheaf.stalk(agent.drive_cell), torch.tensor([DRIVE_ASSERTION])
            )

    def test_reconciliation_would_erode_the_drive_without_the_ordering(self, agent):
        # Eight apex cells disagreeing with the drive erode it every tick. What
        # stops "the assertion stands forever" from being false is tick order,
        # not an exemption -- so disagreement on a drive edge can only ever be
        # reduced by the cell moving.
        agent.tick()
        agent.sheaf.tick()
        assert float(agent.sheaf.stalk(agent.drive_cell)) != DRIVE_ASSERTION

    def test_the_world_never_writes_the_commanded_components(self, agent):
        agent.tick()
        commanded = agent.sheaf.stalks[agent._commanded_slice].clone()
        # A write on its own, with no phases in front of it, leaves them alone.
        observation = agent.env.step(np.zeros(3, dtype=np.float32))[0]
        agent.write(observation, np.zeros(3))
        assert torch.equal(agent.sheaf.stalks[agent._commanded_slice], commanded)

    def test_the_efference_copy_is_what_was_actually_applied(self, agent):
        # Post-clip: what the world does not clear is the body's refusal, and
        # that is exactly what should arrive as disagreement on the motor edge.
        agent.sheaf.tick()
        with torch.no_grad():
            agent.sheaf.stalks[agent._commanded_slice] = torch.tensor([5.0, -5.0, 0.25])
        outcome = agent.act(agent.command())
        assert outcome.command == pytest.approx([5.0, -5.0, 0.25])
        assert outcome.applied == pytest.approx([1.0, -1.0, 0.25])
        assert torch.allclose(
            agent.sheaf.stalks[agent._efference_slice],
            torch.tensor([1.0, -1.0, 0.25]),
            atol=1e-6,
        )

    def test_the_efference_copy_is_read_against_the_world_not_only_derived(self, agent):
        # The agent computes `applied` from its action space rather than from
        # the engine, which keeps the seam a gymnasium contract rather than a
        # MuJoCo one. That is only honest while the two agree, and a torque the
        # arm never applied would be fabricated disagreement on the one edge
        # 04-action-and-the-boundary.md rests on -- so the agreement is pinned
        # here, at the saturating command, which is where it would first part.
        agent.sheaf.tick()
        with torch.no_grad():
            agent.sheaf.stalks[agent._commanded_slice] = torch.tensor([7.0, -7.0, 0.5])
        outcome = agent.act(agent.command())
        limits = agent.env.model.actuator_ctrlrange[:, 1]
        assert agent.env.data.ctrl == pytest.approx(outcome.applied * limits, abs=1e-6)


class TestTheMotorPathway:
    def test_reconciliation_fills_the_commanded_components(self, agent):
        assert torch.all(agent.sheaf.stalks[agent._commanded_slice] == 0)
        agent.tick()
        # Still nothing, and that is the unit delay rather than a fault: on the
        # first tick every cell reconciles against silence, so the graph has
        # not yet said anything to the actuator. The world is one tick away,
        # like any neighbour.
        assert torch.all(agent.sheaf.stalks[agent._commanded_slice] == 0)
        agent.tick()
        assert torch.any(agent.sheaf.stalks[agent._commanded_slice] != 0)

    def test_a_torque_reaches_the_arm(self, agent):
        agent.tick()
        outcome = agent.tick()
        limits = agent.env.model.actuator_ctrlrange[:, 1]
        assert agent.env.data.ctrl == pytest.approx(outcome.applied * limits, abs=1e-6)

    def test_the_arm_is_stepped_with_the_clipped_action(self, agent):
        # The clip is against the env's *declared* action space, so what the
        # env is stepped with has to be inside that space -- otherwise the
        # efference copy is a statement in the contract's terms about a step
        # that was not made in them. The sandbox clips internally and would
        # hide the difference, so the action is caught in flight instead.
        agent.env = _RecordsTheAction(agent.env)
        agent.sheaf.tick()
        with torch.no_grad():
            agent.sheaf.stalks[agent._commanded_slice] = torch.tensor([7.0, -7.0, 0.5])
        outcome = agent.act(agent.command())
        stepped = agent.env.stepped
        assert stepped == pytest.approx([1.0, -1.0, 0.5])
        assert stepped == pytest.approx(outcome.applied)
        assert agent.env.action_space.contains(stepped)
        # Caught in flight rather than through a contract-checking wrapper.
        # `PassiveEnvChecker` looks like the natural guard and is not one:
        # gymnasium 1.3.0 declines to check the action at all -- "for some
        # environments out-of-bounds values can be given" -- and no wrapper
        # it ships asserts `action_space.contains(action)`. A test driving a
        # saturating command through it passes identically before and after
        # this fix, so it would assert nothing while looking like coverage.

    def test_the_command_is_not_clipped_where_the_graph_produces_it(self, agent):
        with torch.no_grad():
            agent.sheaf.stalks[agent._commanded_slice] = torch.tensor([9.0, 0.0, 0.0])
        assert agent.command()[0] == pytest.approx(9.0)

    def test_there_is_no_read_out_map_from_a_cell_to_a_torque(self, agent):
        # The command is a slice of a node stalk and nothing else: no decode
        # path to torque bypasses a stalk, and no cell is designated an output.
        with torch.no_grad():
            agent.sheaf.stalks[agent._commanded_slice] = torch.tensor([0.3, -0.4, 0.5])
        assert agent.command() == pytest.approx([0.3, -0.4, 0.5])


class TestTheSensoryTiling:
    def test_a_patch_cell_holds_its_own_patch_of_the_render_raw(self, agent, dome):
        outcome = agent.tick()
        image = outcome.observation["image"]
        side = agent.patch_side
        for cell in patch_cells(dome)[:: 37]:
            r, c = cell.index.position
            block = image[r * side : (r + 1) * side, c * side : (c + 1) * side, :]
            assert torch.allclose(
                agent.sheaf.stalk(cell.id),
                torch.as_tensor(block.reshape(-1).astype(np.float32)) * PIXEL_SCALE,
                atol=1e-6,
            )

    def test_nothing_outside_the_graph_mixes_two_patches(self, agent, dome):
        # The membership rule's one narrow ban. Every component of every patch
        # cell's stalk is one pixel channel of the render, and every pixel is
        # in exactly one of them.
        pixels = agent._patch_pixels.reshape(-1)
        assert pixels.numel() == 64 * 64 * 3
        assert torch.equal(torch.sort(pixels).values, torch.arange(64 * 64 * 3))

    def test_the_agent_refuses_a_sheaf_built_on_another_dome(self, env, dome):
        # The write tables are this dome's cell ids against that sheaf's
        # layout, so a mismatch writes the render into the wrong components
        # rather than failing.
        with pytest.raises(ValueError, match="different dome"):
            Agent(env, dome=dome, sheaf=Sheaf(build_graph()))

    def test_a_sheaf_alone_brings_its_own_dome(self, env, dome):
        # Building a sheaf and handing it over is the natural thing to do, and
        # the sheaf already carries the authoritative dome. Manufacturing a
        # second one here would make this call raise on an input it accepts.
        sheaf = Sheaf(dome)
        built = Agent(env, sheaf=sheaf)
        assert built.dome is dome
        assert built.sheaf is sheaf

    def test_the_agent_refuses_a_render_its_dome_does_not_tile(self, dome):
        small = PlanarPushSandbox(split="any", image_size=17, render_obs=False)
        try:
            with pytest.raises(ValueError, match="does not tile"):
                Agent(small, dome=dome)
        finally:
            small.close()


class TestTheSheafsConstructionArguments:
    """`gamma` and `generator` build a sheaf, so a supplied sheaf refuses them (#106).

    The stake is `gamma` specifically: it is the constant #85 leaves provisional
    and the one a sweep varies, and `Agent(env, sheaf=prepared, gamma=x)` that
    quietly ran at the default would report clean numbers for the wrong `γ`.
    """

    def test_a_sheaf_and_a_gamma_is_refused_rather_than_ignored(self, env, dome):
        # Anchored on the leading token, because the refusal has to name the
        # argument it was handed rather than merely mention it in its advice.
        with pytest.raises(ValueError, match=r"^gamma belongs") as refusal:
            Agent(env, sheaf=Sheaf(dome), gamma=0.123)
        # Where it belongs instead, named: the sweep's fix is to build the
        # sheaf with the gamma it wants, not to keep looking for an Agent knob.
        assert "Sheaf" in str(refusal.value)
        assert "generator" not in str(refusal.value)

    def test_a_gamma_equal_to_the_default_is_refused_too(self, env, dome):
        # The mistake is asking the Agent to set it, not the value asked for.
        # A caller who writes the default explicitly is sweeping like any
        # other, and letting exactly `1.0` through would hide the one point of
        # a sweep that happens to sit on it.
        with pytest.raises(ValueError, match=r"^gamma belongs"):
            Agent(env, sheaf=Sheaf(dome), gamma=DEFAULT_GAMMA)

    def test_a_gamma_of_none_is_refused_too(self, env, dome):
        # `Agent(env, sheaf=prepared, gamma=cfg.gamma)` with a config that says
        # `None` for "the default" is asking for `DEFAULT_GAMMA` and would get
        # the sheaf's own -- the same wrong constant, arrived at more quietly.
        # Mentioning gamma at all alongside a sheaf is the error.
        with pytest.raises(ValueError, match=r"^gamma belongs"):
            Agent(env, sheaf=Sheaf(dome), gamma=None)

    def test_a_sheaf_and_a_generator_is_refused_on_the_same_grounds(self, env, dome):
        # The decision #106 left open. A supplied sheaf is already drawn, so
        # nothing is left to seed and a run whose author believes it is
        # reproducible is not.
        with pytest.raises(ValueError, match=r"^generator belongs") as refusal:
            Agent(env, sheaf=Sheaf(dome), generator=torch.Generator().manual_seed(0))
        assert "Sheaf" in str(refusal.value)
        assert "gamma" not in str(refusal.value)

    def test_a_sheaf_and_no_generator_is_not_a_generator(self, env, dome):
        # `None` is torch's own "no generator", so writing it out asks for
        # nothing and there is nothing to refuse. The sentinel is `gamma`'s
        # alone, because `None` is not a `γ` a caller could have meant.
        sheaf = Sheaf(dome)
        assert Agent(env, sheaf=sheaf, generator=None).sheaf is sheaf

    def test_the_refusal_names_every_argument_it_was_handed(self, env, dome):
        with pytest.raises(ValueError, match=r"^gamma and generator belong"):
            Agent(
                env,
                sheaf=Sheaf(dome),
                gamma=0.123,
                generator=torch.Generator().manual_seed(0),
            )

    def test_a_sheaf_alone_keeps_its_own_gamma(self, env, dome):
        prepared = Sheaf(dome, gamma=0.25)
        built = Agent(env, sheaf=prepared)
        assert built.sheaf.gamma == 0.25

    def test_a_gamma_of_none_with_no_sheaf_is_the_sheafs_business(self, env, dome):
        # The sentinel is the only thing read as "not given"; `None` is a
        # value, and whether a value is a legal `γ` is Sheaf's rule, asked in
        # one place. Reading `None` as "the default" here would be a second
        # answer to the same question and would send a caller who followed the
        # refusal's advice to `Sheaf(dome, gamma=None)`, which does not agree.
        # What that one rule says, from either door (#107): a refusal naming
        # `gamma` and the bound it broke, in place of the bare `TypeError` the
        # comparison inside the gain used to leak.
        names_the_rule = r"^gamma is a single global scalar"
        with pytest.raises(ValueError, match=names_the_rule) as through_the_agent:
            Agent(env, dome=dome, gamma=None)
        with pytest.raises(ValueError, match=names_the_rule) as straight_at_it:
            Sheaf(dome, gamma=None)
        assert str(through_the_agent.value) == str(straight_at_it.value)

    def test_a_gamma_with_no_sheaf_reaches_the_sheaf_that_gets_built(self, env, dome):
        built = Agent(env, dome=dome, gamma=0.25)
        assert built.sheaf.gamma == 0.25
        # Through to the gain, which is what `γ` actually is: the same graph at
        # a quarter of the reconciliation step.
        default = Agent(env, dome=dome)
        assert default.sheaf.gamma == DEFAULT_GAMMA
        assert torch.allclose(built.sheaf.gain, default.sheaf.gain * 0.25)

    def test_a_generator_with_no_sheaf_still_seeds_the_sheaf_that_gets_built(
        self, env, dome
    ):
        one = Agent(env, dome=dome, generator=torch.Generator().manual_seed(7))
        again = Agent(env, dome=dome, generator=torch.Generator().manual_seed(7))
        other = Agent(env, dome=dome, generator=torch.Generator().manual_seed(8))
        assert torch.equal(one.sheaf.maps.maps, again.sheaf.maps.maps)
        assert not torch.equal(one.sheaf.maps.maps, other.sheaf.maps.maps)


class TestASustainedRun:
    def test_the_untrained_agent_drives_the_sandbox_without_diverging(self, agent):
        for outcome in run(agent, TICKS, seed=1):
            assert np.isfinite(outcome.command).all()
        stalks, charts = agent.sheaf.stalks, agent.sheaf.charts
        assert torch.isfinite(stalks).all() and torch.isfinite(charts).all()
        # Bounded, not merely finite: a slow divergence is still a divergence.
        assert float(stalks.abs().max()) < 1e3
        assert float(charts.abs().max()) < 1e3
        assert agent.sheaf.ticks == TICKS

    def test_the_arm_flails(self, agent):
        # It is untrained, so this is all that is claimed: torque leaves the
        # actuator boundary cell, it is not the same torque every tick, and the
        # world moves in response.
        outcomes = list(run(agent, TICKS, seed=2))
        commands = np.stack([o.command for o in outcomes])
        assert np.abs(commands).max() > 0.05
        assert commands.std(axis=0).min() > 0.0
        joint_angles = np.stack([o.observation["qpos"] for o in outcomes])
        assert np.abs(joint_angles[-1] - joint_angles[0]).max() > 0.01

    def test_the_run_never_puts_anything_on_a_tape(self, agent):
        # The assertion runs inside every tick; this is the statement that a
        # long run does not accumulate its way past it.
        for _ in run(agent, 20, seed=3):
            pass
        agent.sheaf.assert_no_tape()
        assert not agent.sheaf.stalks.requires_grad


class TestBenchmark:
    """`benchmarks/agent_tick.py` is the reported per-tick wall time's provenance.

    Timings are not asserted -- a wall-clock threshold in CI measures the
    runner, not the tick. What is asserted is that the script still runs
    against the current API, so the number in `09-the-build-stack.md` keeps a
    reproduction.
    """

    def test_the_benchmark_still_runs(self):
        import agent_tick

        samples = agent_tick.measure(ticks=3)
        assert set(samples) == {
            "inference phase",
            "message-passing phase",
            "the world's read",
            "external write",
            "env.step()",
        }
        assert all(len(s) == 3 for s in samples.values())
