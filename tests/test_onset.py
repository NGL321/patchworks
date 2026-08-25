"""What the surface owes onset latency (ticket #95).

Two things, and the second is only worth anything because of the first.

* **The hands drop their markers.** Firing a hand and marking the record are
  one call, so the demo's temporal measure cannot be lost to a line somebody
  forgot -- silently, with the footage still looking fine.
* **The counter runs from the most recent marker, in ticks.** Onset is read off
  the strip as it happens rather than reconstructed afterward, and a capture
  rate cannot reach the number.
"""

import dataclasses

import numpy as np
import pytest
import torch

from patchworks.agent import Agent
from patchworks.graph import build_graph
from patchworks.sandbox import PlanarPushSandbox, restore, snapshot
from patchworks.surface import (
    CAPTURE_EVERY,
    Event,
    EventKind,
    Hands,
    OnsetCounter,
    Recorder,
)


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
    agent = Agent(env, dome=dome, generator=torch.Generator().manual_seed(0))
    observation, _info = env.reset(seed=0)
    agent.observe(observation)
    return agent


@pytest.fixture
def recorder(agent):
    return Recorder(agent, every=1)


@pytest.fixture
def hands(recorder):
    return Hands(recorder)


def tick(recorder, times=1):
    """One whole tick, watched, the way a viewer's event loop drives it."""
    records = []
    for _ in range(times):
        recorder.agent.tick()
        record = recorder.observe()
        if record is not None:
            records.append(record)
    return records


class TestTheHandsDropTheirMarkers:
    def test_each_of_the_three_hands_leaves_its_own_marker(self, hands, env):
        fired = (
            hands.disturb_arm(1, 0.05),
            hands.perturb(0, [0.32, -0.30]),
            hands.retarget(goal_puck=1, goal_zone=2),
        )
        assert [event.kind for event in fired] == [
            EventKind.DISTURB_ARM,
            EventKind.PERTURB,
            EventKind.RETARGET,
        ]
        assert hands.recorder.pending == fired

    def test_a_marker_lands_in_the_record(self, hands, recorder):
        tick(recorder, 2)  # primes the delta and gets a capture out of the way
        fired = hands.perturb(0, [0.32, -0.30])
        (record,) = tick(recorder)
        assert record.events == (fired,)

    def test_a_marker_carries_the_tick_its_hand_fired_on(self, hands, recorder):
        tick(recorder, 2)
        fired = hands.disturb_arm(0, 0.02)
        assert fired.tick == recorder.agent.sheaf.ticks
        (record,) = tick(recorder)
        assert record.tick == fired.tick + 1

    def test_the_hand_actually_fires(self, hands, env):
        """A marker is not the point on its own: the world has to have moved."""
        before = np.array(env.data.qvel)
        hands.disturb_arm(1, 0.05)
        assert not np.array_equal(env.data.qvel, before)

        hands.perturb(0, [0.32, -0.30])
        assert env.puck_pose(0)[:2] == pytest.approx([0.32, -0.30])

        task = env.task
        other = (task.goal_zone + 1) % 3
        hands.retarget(goal_zone=other)
        assert env.task.goal_zone == other

    def test_a_hand_that_is_refused_leaves_no_marker(self, hands):
        for refused in (
            lambda: hands.disturb_arm(99, 0.05),
            lambda: hands.perturb(99, [0.1, 0.1]),
            lambda: hands.retarget(goal_puck=99),
        ):
            with pytest.raises(ValueError, match="range"):
                refused()
        assert hands.recorder.pending == ()

    def test_a_retarget_marker_carries_the_goal_that_now_stands(self, hands, env):
        task = env.task
        fired = hands.retarget(goal_puck=None, goal_zone=(task.goal_zone + 1) % 3)
        assert fired.detail == (
            float(env.task.goal_puck),
            float(env.task.goal_zone),
        )

    def test_a_perturb_marker_carries_an_orientation_only_when_asked_for(self, hands):
        assert hands.perturb(0, [0.32, -0.30]).detail == (0.0, 0.32, -0.30)
        assert hands.perturb(0, [0.30, -0.28], theta=0.5).detail == (
            0.0,
            0.30,
            -0.28,
            0.5,
        )

    def test_a_marker_from_a_hand_is_written_as_plain_floats(self, hands):
        """A gesture hands over whatever it picked off an array (#96)."""
        fired = hands.perturb(np.int64(0), np.array([0.32, -0.30], dtype=np.float32))
        assert all(type(value) is float for value in fired.detail)


class TestTheCounterRunsFromTheMostRecentMarker:
    def test_there_is_nothing_to_count_before_a_hand_fires(self, recorder):
        counter = OnsetCounter()
        assert [counter.count(record) for record in tick(recorder, 3)] == [None, None]
        assert counter.since is None

    def test_the_count_starts_on_the_record_that_carries_the_marker(
        self, hands, recorder
    ):
        counter = OnsetCounter()
        for record in tick(recorder, 3):
            counter.count(record)
        fired = hands.disturb_arm(0, 0.02)
        (record,) = tick(recorder)
        assert counter.count(record) == 1
        assert counter.since == fired

    def test_the_count_climbs_one_per_tick(self, hands, recorder):
        counter = OnsetCounter()
        for record in tick(recorder, 2):
            counter.count(record)
        hands.disturb_arm(0, 0.02)
        assert [counter.count(record) for record in tick(recorder, 4)] == [1, 2, 3, 4]

    def test_the_count_is_in_ticks_and_not_in_captures(self, agent, dome):
        """A capture is decimated and a marker is not.

        The whole reason an :class:`Event` carries its own tick. Counted off
        the captures instead, the demo's temporal measure would change meaning
        with the display's rate.
        """
        recorder = Recorder(agent)  # the spec's 10 Hz, not every tick
        counter = OnsetCounter()
        for record in tick(recorder, CAPTURE_EVERY):
            counter.count(record)
        fired = Hands(recorder).disturb_arm(0, 0.02)
        counts = [counter.count(record) for record in tick(recorder, 3 * CAPTURE_EVERY)]
        assert counts[0] == 1  # the tick that fires one is always captured
        assert counts[1:] == [
            record.tick - fired.tick for record in recorder.trace[-len(counts) + 1 :]
        ]
        assert counts[-1] > CAPTURE_EVERY

    def test_a_second_marker_restarts_the_count(self, hands, recorder):
        counter = OnsetCounter()
        for record in tick(recorder, 2):
            counter.count(record)
        hands.disturb_arm(0, 0.02)
        for record in tick(recorder, 3):
            counter.count(record)
        assert counter.count(tick(recorder)[0]) == 4
        second = hands.perturb(0, [0.32, -0.30])
        assert counter.count(tick(recorder)[0]) == 1
        assert counter.since == second

    def test_a_record_carrying_two_markers_counts_from_the_later(self, recorder):
        """`08`'s three events land seconds apart in one unbroken run, so the
        two on one record is the decimated case rather than the live one --
        and the counter runs from the **most recent** marker either way."""
        counter = OnsetCounter()
        (record,) = tick(recorder, 2)
        early = Event(EventKind.PERTURB, record.tick - 3, (0.0, 0.32, -0.30))
        later = Event(EventKind.RETARGET, record.tick - 1, (1.0, 2.0))
        both = dataclasses.replace(record, events=(early, later))
        assert counter.count(both) == 1
        assert counter.since == later

    def test_a_restore_is_invisible_and_the_harness_owns_restart(
        self, hands, recorder, env
    ):
        """The limit, pinned rather than implied.

        A restore rewinds the world, the clock and the adapting surface -- and
        not `sheaf.ticks`, which counts ticks that happened. So the count goes
        on climbing across a restore, and what a harness that forgot
        :meth:`restart` gets is a plausible latency rather than an error. The
        counter cannot detect this; the test says so out loud so that nobody
        reads the guard below as covering it.
        """
        counter = OnsetCounter()
        for record in tick(recorder, 2):
            counter.count(record)
        state = snapshot(env)
        hands.disturb_arm(0, 0.02)
        for record in tick(recorder, 3):
            counter.count(record)

        restore(env, state)
        # The recorder takes the gap deliberately, which costs the one delta it
        # cannot state across the restore -- so three ticks leave two records.
        recorder.reprime()
        carried = [counter.count(record) for record in tick(recorder, 3)]
        assert carried == [5, 6], "a restore does not rewind the tick counter"
        # Which is why the harness calls this once per trial, and why the
        # docstring says the counter cannot do it for them.
        counter.restart()
        assert counter.count(tick(recorder)[0]) is None

    def test_a_record_from_a_clock_that_went_backwards_is_refused(
        self, hands, recorder
    ):
        """What the guard actually catches: a rebuilt sheaf, or a record fed
        out of order. Not a restore -- see above."""
        counter = OnsetCounter()
        for record in tick(recorder, 2):
            counter.count(record)
        hands.disturb_arm(0, 0.02)
        (record,) = tick(recorder)
        counter.count(record)
        rewound = dataclasses.replace(record, tick=record.tick - 2, events=())
        with pytest.raises(ValueError, match="restart"):
            counter.count(rewound)
        counter.restart()
        assert counter.count(rewound) is None

    def test_counts_reads_a_feed_in_order(self, hands, recorder):
        tick(recorder, 2)
        hands.disturb_arm(0, 0.02)
        records = tick(recorder, 3)
        assert [count for _record, count in OnsetCounter().counts(records)] == [1, 2, 3]
