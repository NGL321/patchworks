"""The Koopman go/no-go, and the parts of it a silent bug would corrupt.

`benchmarks/koopman_lift.py` answers ticket #126 with a number, and the number
is only worth what the protocol under it is worth. Two defects were found by
running it that a reader would not have found by reading it -- a train/score
split that measured distribution shift rather than linearity, and an upper
bound that lost to the thing it was bounding -- so what is asserted here is the
protocol: the splits are disjoint, no pair straddles a block, the fits recover
what they are given, the trained models start where they claim to start, and
the verdict fires on the clause the sweep actually tripped.

The sweep itself is not run. It is 45 minutes of fitting and it would be
measuring the world rather than the code.
"""

import numpy as np
import pytest
import torch

import koopman_lift as kl
from patchworks.sandbox import PlanarPushSandbox


class TestTheSplitIsHonest:
    """`_split` is where the first defect lived, so it is held down hardest."""

    def test_the_three_sets_are_disjoint(self):
        fitting, validation, scoring = kl._split([(0, 1000)])
        ticks = [set(range(a, b)) for a, b in fitting + validation + scoring]
        for i, first in enumerate(ticks):
            for second in ticks[i + 1 :]:
                assert not first & second

    def test_all_three_sets_are_non_empty_and_interleaved(self):
        """Interleaved, not head-and-tail: fitting on a segment's head and
        scoring on its tail measures extrapolation to a shifted distribution.
        Measured at 429x persistence on the tail against 0.83x on the head,
        which is the whole reason this function is not a fraction."""
        fitting, validation, scoring = kl._split([(0, 2000)])
        assert fitting and validation and scoring
        # A head-and-tail split would put every scoring block after every
        # fitting block. Interleaving does not.
        assert min(a for a, _ in scoring) < max(b for _, b in fitting)

    def test_a_gap_separates_every_block_from_the_next(self):
        """Adjacent ticks at 50 Hz are nearly identical, so a scored tick that
        is the immediate successor of a fitted one is partly memorisation."""
        blocks = sorted(sum(kl._split([(0, 1000)]), []))
        for (_, stop), (start, _) in zip(blocks, blocks[1:]):
            assert start - stop >= kl.BLOCK_GAP

    def test_no_pair_crosses_a_block(self):
        for horizon in (1, 5, 50):
            blocks = [(0, 100), (200, 300)]
            t, t_next = kl._pairs(blocks, horizon=horizon)
            assert np.all(t_next - t == horizon)
            for start, stop in zip(t, t_next):
                assert any(a <= start and stop < b for a, b in blocks)

    def test_a_block_shorter_than_the_horizon_yields_no_pairs(self):
        assert kl._pairs([(0, 10)], horizon=50)[0].size == 0

    def test_a_segment_too_short_to_block_is_dropped_rather_than_kept_short(self):
        assert kl._split([(0, kl.MINIMUM_BLOCK - 1)]) == ([], [], [])


class TestTheFits:
    def test_an_affine_map_is_recovered_from_its_own_data(self):
        generator = torch.Generator().manual_seed(0)
        a = torch.randn(2, 5, 7, generator=generator, dtype=kl.DTYPE)
        bias = torch.randn(2, 5, generator=generator, dtype=kl.DTYPE)
        x = torch.randn(2, 400, 7, generator=generator, dtype=kl.DTYPE)
        y = kl._apply(a, bias, x)

        fitted_a, fitted_bias = kl._affine_fit(x, y, ridge=1e-12)
        assert torch.allclose(fitted_a, a, atol=1e-6)
        assert torch.allclose(fitted_bias, bias, atol=1e-6)

    def test_the_ridge_is_chosen_per_tile_on_the_data_it_is_handed(self):
        """Not one ridge for the population: tiles differ by orders of
        magnitude in how collinear their stalks are."""
        generator = torch.Generator().manual_seed(1)
        x = torch.randn(3, 200, 4, generator=generator, dtype=kl.DTYPE)
        y = torch.randn(3, 200, 2, generator=generator, dtype=kl.DTYPE)
        _, _, chosen = kl._selected_affine(
            x, y, lambda a, b: kl._per_tile_mse(kl._apply(a, b, x), y)
        )
        assert chosen.shape == (3,)
        assert int(chosen.max()) < len(kl.RIDGE_GRID)

    def test_per_tile_error_keeps_the_tiles_apart(self):
        prediction = torch.zeros(2, 10, 3, dtype=kl.DTYPE)
        truth = torch.ones(2, 10, 3, dtype=kl.DTYPE)
        truth[1] *= 2
        assert kl._per_tile_mse(prediction, truth).tolist() == [1.0, 4.0]


class TestTheTrainedModelsAreUpperBounds:
    """The second defect: a randomly started (c) finished five times worse than
    the (b) it exists to bound, which measures Adam and not linearity."""

    def test_a_network_started_from_a_linear_fit_computes_that_fit(self):
        generator = torch.Generator().manual_seed(2)
        tiles, d_in, d_out = 3, 6, 4
        a = torch.randn(tiles, d_out, d_in, generator=generator, dtype=kl.DTYPE)
        bias = torch.randn(tiles, d_out, generator=generator, dtype=kl.DTYPE)
        source = torch.randn(tiles, 50, d_in, generator=generator, dtype=kl.DTYPE)

        net = kl.BatchedMLP(tiles, d_in, kl.hidden_width(d_in, d_out), d_out, generator)
        kl._start_from_linear(net, a, bias, source, generator)
        with torch.no_grad():
            assert torch.allclose(net(source), kl._apply(a, bias, source), atol=1e-8)

    def test_a_hidden_layer_narrower_than_its_input_is_refused(self):
        """The identity trick needs somewhere to put the identity. Refused
        rather than silently started somewhere else."""
        generator = torch.Generator().manual_seed(3)
        net = kl.BatchedMLP(1, 8, 4, 8, generator)
        with pytest.raises(ValueError, match="width >= d_in"):
            kl._start_from_linear(
                net,
                torch.zeros(1, 8, 8, dtype=kl.DTYPE),
                torch.zeros(1, 8, dtype=kl.DTYPE),
                torch.zeros(1, 5, 8, dtype=kl.DTYPE),
                generator,
            )

    def test_training_never_leaves_a_tile_worse_than_it_started(self):
        """Per-tile best-iterate selection: tiles converge at different rates,
        and stopping them together under-trains the slow ones."""
        generator = torch.Generator().manual_seed(4)
        source = torch.randn(2, 80, 3, generator=generator, dtype=kl.DTYPE)
        target = torch.sin(source[:, :, :1]).repeat(1, 1, 3)
        net = kl.BatchedMLP(2, 3, kl.hidden_width(3, 3), 3, generator)
        with torch.no_grad():
            before = kl._per_tile_mse(net(source), target)
        kl._train(net, source, target, lambda n: kl._per_tile_mse(n(source), target))
        with torch.no_grad():
            after = kl._per_tile_mse(net(source), target)
        assert torch.all(after <= before)


class TestTheVerdict:
    """The pass condition as pre-registered returns GO on the sweep that ran,
    and that is the condition misfiring rather than the variant passing.

    `recovery` is `(a - b) / (a - c)`, a ratio of differences. When (c) fails to
    beat (a) it measures nothing, and it reads 1.00 precisely *because* (b) and
    (c) agree. The ticket's second clause -- "(b) ~ (a), meaning the lift earns
    nothing" -- is the one the sweep tripped, so it is checked separately.
    """

    @staticmethod
    def _cell(kind, dmd, linear, nonlinear, *, mode="plain", k=12):
        one = np.array([dmd]), np.array([linear]), np.array([nonlinear])
        return kl.Cell(
            mode=mode,
            k=k,
            kind=kind,
            dmd=one[0],
            linear=one[1],
            nonlinear=one[2],
            unbottlenecked=one[0],
            floor=one[1],
            raw={"dmd": one[0], "linear": one[1], "nonlinear": one[2]},
            horizon={h: np.array([1.0]) for h in kl.HORIZONS},
            tau=np.array([5.0]),
            spectral_radius=0.99,
        )

    def test_a_lift_that_never_beats_raw_dmd_is_a_no_go(self):
        """The shape the sweep actually produced: (b) ~ (c) to within a
        percent, and both far worse than (a)."""
        cells = [
            self._cell("free", dmd=1.33, linear=2.74, nonlinear=2.79),
            self._cell("contact", dmd=1.94, linear=4.49, nonlinear=4.49),
        ]
        outcome, lines = kl.verdict(cells)
        assert outcome == "no-go"
        assert any("EARNS NOTHING" in line for line in lines)

    def test_recovery_alone_would_have_passed_that_sweep(self):
        """Held down because it is the trap, and the trap is subtler than it
        looks: `recovery` is formed only on tiles where (c) beats (a), so on a
        sweep where almost none do it is a median over a self-selected handful
        -- and on those few (b) and (c) agree, so it reads ~1.00 and passes.

        A class of eleven tiles the lift ruins and one it helps returns a
        passing recovery. Nothing about that number is wrong; it simply does not
        say what the pass condition reads it as saying."""
        ruined = [(1.33, 2.74, 2.79)] * 11
        helped = [(3.00, 1.00, 1.10)]
        rows = np.array(ruined + helped)
        free = kl.Cell(
            mode="plain",
            k=12,
            kind="free",
            dmd=rows[:, 0],
            linear=rows[:, 1],
            nonlinear=rows[:, 2],
            unbottlenecked=rows[:, 0],
            floor=rows[:, 1],
            raw={"dmd": rows[:, 0], "linear": rows[:, 1], "nonlinear": rows[:, 2]},
            horizon={h: np.ones(12) for h in kl.HORIZONS},
            tau=np.full(12, 5.0),
            spectral_radius=0.99,
        )
        assert free.recovery >= kl.RECOVERY_TARGET
        # ...on one tile in twelve.
        assert free.headroom_share == pytest.approx(1 / 12)
        # And the sweep is still a no-go, because the clause that matters is
        # the other one.
        assert kl.verdict([free])[0] == "no-go"

    def test_a_lift_that_earns_its_keep_can_still_go(self):
        cells = [
            self._cell("free", dmd=1.00, linear=0.40, nonlinear=0.30),
            self._cell("contact", dmd=1.20, linear=0.60, nonlinear=0.50),
        ]
        outcome, _ = kl.verdict(cells)
        assert outcome == "go"

    def test_contact_tiles_failing_alone_is_a_third_outcome(self):
        """Comment 11: a switched operator is the mature answer to exactly the
        free-motion/contact regime pair, so a partial no-go is not a full one."""
        cells = [
            self._cell("free", dmd=1.00, linear=0.40, nonlinear=0.30),
            self._cell("contact", dmd=3.00, linear=2.00, nonlinear=1.50),
        ]
        outcome, lines = kl.verdict(cells)
        assert outcome == "contact-only no-go"
        assert any("contact-only shortfall" in line for line in lines)

    def test_the_operator_overhead_divides_the_bottleneck_out(self):
        """`b/e` is what the delay conclusion rests on, so its meaning is fixed
        here. An operator that reaches its lift's floor scores 1.00 however bad
        that floor is -- which is the only way `delay`, whose floor is worse by
        construction, can be compared with `plain` at all."""
        perfect = self._cell("free", dmd=1.0, linear=9.0, nonlinear=9.0)
        perfect.floor = np.array([9.0])
        assert perfect.overhead == pytest.approx(1.0)

        wasteful = self._cell("free", dmd=1.0, linear=9.0, nonlinear=9.0)
        wasteful.floor = np.array([3.0])
        assert wasteful.overhead == pytest.approx(3.0)

    def test_a_tile_with_no_headroom_contributes_no_recovery_ratio(self):
        free = self._cell("free", dmd=1.00, linear=2.00, nonlinear=3.00)
        assert free.recovery != free.recovery  # nan
        assert free.headroom_share == 0.0


class TestThePieceIsTheOneACellGets:
    def test_a_tile_is_a_boundary_cell_stalk(self):
        """48 is `graph.py`'s `patch_stalk`, and the 16x16 tiling is the dome's
        own. A tile here is not an analysis convenience."""
        from patchworks.graph import DomeSpec

        assert kl.PATCH_STALK == DomeSpec().patch_stalk == 48
        assert kl.GRID == 16

    def test_tile_series_reads_the_tile_it_names(self):
        images = np.arange(2 * 64 * 64 * 3, dtype=np.uint8).reshape(2, 64, 64, 3)
        series = kl.tile_series(images, 3, 5)
        assert series.shape == (2, kl.PATCH_STALK)
        expected = images[:, 12:16, 20:24, :].reshape(2, kl.PATCH_STALK) / 255.0
        assert series == pytest.approx(expected)

    def test_a_still_tile_is_dropped_by_the_motion_floor(self):
        """Its persistence baseline is the denominator of every ratio reported,
        and a tile that never moves makes that a division by nearly zero."""
        images = np.zeros((50, 64, 64, 3), dtype=np.uint8)
        images[:, 0:4, 0:4, :] = np.arange(50, dtype=np.uint8)[:, None, None, None]
        _, motion = kl.tile_activity(images, [(0, 50)])
        assert motion[0, 0] > kl.MOTION_FLOOR
        assert motion[5, 5] == 0.0


class TestTheContactSplitIsNotNoise:
    def test_the_projection_lands_each_puck_on_its_own_colour(self):
        """The whole contact/free segmentation rides on this map from world
        coordinates to render pixels. If it were wrong the split would be
        random and nothing above it would mean anything -- so the benchmark
        refuses to collect until it passes, and so does this."""
        env = PlanarPushSandbox(split="any")
        try:
            observation, info = env.reset(seed=3, options={"reset_arm": True})
            assert kl.Projection(env).check(env, observation["image"], info)
        finally:
            env.close()

    def test_a_point_outside_the_render_has_no_tile(self):
        env = PlanarPushSandbox(render_obs=False)
        try:
            assert kl.Projection(env).tile((10.0, 10.0, 0.0)) is None
        finally:
            env.close()
