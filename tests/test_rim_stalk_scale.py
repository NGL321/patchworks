"""`benchmarks/rim_stalk_scale.py`: the orientation, and the two columns (#468, for #469).

**Nothing here pins a reading of a run.** The stalk ratios are readings of a
surface later tickets are expected to move, and #468 is a read rather than a
ruling. What is held is the shape of the instrument, because every way of getting
this one wrong yields a plausible number:

* **The orientation is decided once, in `geometry`, and stored.** A ratio is only
  a statement about the rim's pressure if the numerator is the end the pressure
  pushes *from*. On a boundary-incident edge that is the pinned end; on an
  interior control edge nothing is pinned, so depth to the rim decides. Getting
  either backwards turns a 2 into a 0.5 and the finding into its opposite.
* **The drive is oriented by pinning and not by depth.** The drive cell is pinned
  but sits at the apex rather than on the rim, so it is the one population where
  the two rules disagree. It is excluded from every depth-graded table for that
  reason, and an excluded population that quietly rejoined would put a 1-wide
  pinned stalk in the apex's bin and read as chaining.
* **The two columns differ by exactly the width bias.** `rms` is not a second
  opinion, it is the same ratio with `√(n_near/n_far)` divided out — so the
  identity holds per edge, and the raw column's dimension confound is a stated
  number rather than an argument.
* **Tick 0 carries no reading at all.** A fresh sheaf's node stalks are zero, so
  the ratio is `0/0`. #468 asked for construction to be reported as the axis's
  origin the way #416 did; `INITIAL_NORM` pins the maps and not the stalks, and a
  rig that reported a construction 1 here would be reporting #416's zero on a
  quantity that does not have one.
* **The metric is the name the bar is written on.** #469 cuts on
  `rim_stalk_ratio`, and a rig reporting a differently-spelled key would leave
  the problem in *cutoffs naming a rig with no recorded run* while appearing to
  be watched — the disguise that section exists to show.
"""

import cutoff_report
import numpy as np
import pytest
import rim_stalk_scale
import torch

from conftest import SMALL
from patchworks.graph import build_graph


@pytest.fixture(scope="module")
def dome():
    return build_graph(SMALL)


@pytest.fixture(scope="module")
def geometry(dome):
    return rim_stalk_scale.geometry(dome)


class TestTheOrientation:
    """Which end is the numerator, and why it may not be chosen anywhere else."""

    def test_a_boundary_incident_edge_puts_the_pinned_end_on_top(self, geometry, dome):
        seen = 0
        for edge in geometry["edges"]:
            if edge["pinned_ends"] != 1:
                continue
            seen += 1
            assert edge["orientation"] == "pinned over free"
            assert dome.cells[edge["near"]].is_boundary
            assert not dome.cells[edge["far"]].is_boundary
        assert seen, "the small dome has boundary-incident edges to orient"

    def test_an_interior_edge_puts_the_shallower_end_on_top(self, geometry):
        for edge in geometry["edges"]:
            if edge["pinned_ends"] or edge["orientation"] == "none":
                continue
            assert edge["orientation"] == "shallow over deep"
            assert edge["near_depth"] < edge["far_depth"]

    def test_no_edge_has_two_pinned_ends(self, geometry):
        # The dome has none, and the rig's orientation has no branch for one.
        assert not [e for e in geometry["edges"] if e["pinned_ends"] == 2]

    def test_an_edge_between_equal_depths_is_left_unoriented(self, geometry):
        for edge in geometry["edges"]:
            if edge["orientation"] == "none":
                assert edge["pinned_ends"] == 0
                assert edge["near_depth"] == edge["far_depth"]


class TestThePopulations:
    """#469's three-plus-one split, and that it partitions the edges."""

    def test_the_masks_partition_every_edge(self, geometry):
        record = {"geometry": geometry}
        masks = rim_stalk_scale.populations(record)
        edges = len(geometry["edges"])
        parts = masks["sensory"] | masks["drive"] | masks["actuator"]
        assert (parts == masks["boundary-incident"]).all()
        assert not (masks["boundary-incident"] & masks["interior (control)"]).any()
        assert int(masks["boundary-incident"].sum()) + int(
            masks["interior (control)"].sum()
        ) == edges

    def test_the_drive_is_pinned_but_not_on_the_rim(self, geometry):
        drive = [e for e in geometry["edges"] if e["kind"] == "drive"]
        assert drive, "the small dome carries the drive cell"
        for edge in drive:
            # Pinned end on top, as a boundary-incident edge -- and *not* the
            # shallower one, which is the whole reason it leaves the depth tables.
            assert edge["orientation"] == "pinned over free"
            assert edge["near_depth"] > edge["far_depth"]

    def test_the_drive_is_excluded_from_both_depth_tables(self, geometry, dome):
        record = {"geometry": geometry}
        frame = {
            "tick": 1,
            "norm": [1.0] * len(dome.cells),
            "rms": [1.0] * len(dome.cells),
        }
        counted = sum(r["cells"] for r in rim_stalk_scale.by_depth(record, frame)) // 2
        assert counted == len(dome.cells) - 1  # every cell but the drive
        rungs = rim_stalk_scale.graded(record, frame)
        assert sum(r["edges"] for r in rungs) == len(
            [
                e
                for e in geometry["edges"]
                if e["kind"] != "drive" and e["orientation"] != "none"
            ]
        )


class TestTheTwoColumns:
    """`rms` is the raw ratio with the width divided out, and nothing else."""

    def test_the_columns_differ_by_exactly_the_width_bias(self, geometry, dome):
        generator = torch.Generator().manual_seed(0)
        norms = torch.rand(len(dome.cells), generator=generator).double() + 0.5
        frame = {
            "tick": 1,
            "norm": norms.tolist(),
            "rms": [
                float(norms[c.id]) / np.sqrt(c.stalk) for c in dome.cells
            ],
        }
        ratios = rim_stalk_scale.edge_ratios({"geometry": geometry}, frame)
        bias = np.array([e["width_bias"] for e in geometry["edges"]])
        assert ratios["rms"] == pytest.approx(ratios["norm"] / bias, rel=1e-9)

    def test_the_width_bias_is_what_equal_per_component_scale_reads(
        self, geometry, dome
    ):
        # Every cell at per-component scale 1: the raw ratio is the bias itself
        # and the rms ratio is 1 everywhere. This is the confound, stated.
        frame = {
            "tick": 1,
            "norm": [np.sqrt(c.stalk) for c in dome.cells],
            "rms": [1.0] * len(dome.cells),
        }
        ratios = rim_stalk_scale.edge_ratios({"geometry": geometry}, frame)
        bias = np.array([e["width_bias"] for e in geometry["edges"]])
        assert ratios["norm"] == pytest.approx(bias, rel=1e-9)
        assert ratios["rms"] == pytest.approx(np.ones_like(bias), rel=1e-9)

    def test_a_sensory_edge_is_biased_by_the_patch_over_predicting_widths(
        self, geometry
    ):
        sensory = [e for e in geometry["edges"] if e["kind"] == "sensory"]
        assert sensory
        for edge in sensory:
            assert edge["width_bias"] == pytest.approx(
                np.sqrt(edge["near_stalk"] / edge["far_stalk"])
            )


class TestTheOrigin:
    """Tick 0 has no reading, and the rig does not invent #416's one."""

    def test_a_fresh_sheaf_has_no_stalk_content(self, geometry, dome):
        frame = {
            "tick": 0,
            "norm": [0.0] * len(dome.cells),
            "rms": [0.0] * len(dome.cells),
        }
        ratios = rim_stalk_scale.edge_ratios({"geometry": geometry}, frame)
        assert np.isnan(ratios["norm"]).all()

    def test_the_ladder_still_carries_tick_zero_and_tick_one(self):
        # Kept as a frame so the record shows the emptiness rather than hiding
        # it, and so tick 1 -- the measured origin -- is always on the ladder.
        assert rim_stalk_scale.ladder(30000)[:2] == [0, 1]
        assert rim_stalk_scale.ladder(100000)[-1] == 100000


class TestTheBarItIsWatched_On:
    """The rig's metric name and #469's threshold are one contract."""

    def test_the_reported_metric_is_the_one_the_register_cuts_on(self):
        register = cutoff_report.REGISTER.read_text(encoding="utf-8")
        watches = cutoff_report.watching(register, "rim_stalk_scale")
        if not watches:
            pytest.skip(
                "no open problem cuts on `rim_stalk_scale` in the checked-in "
                "register; it is a projection and may be briefly stale"
            )
        offered = {
            cutoff_report.metric_name(key)
            for key in rim_stalk_scale.readings(
                [
                    {
                        "geometry": rim_stalk_scale.geometry(build_graph(SMALL)),
                        "frames": [
                            {
                                "tick": 1,
                                "norm": [1.0] * len(build_graph(SMALL).cells),
                                "rms": [1.0] * len(build_graph(SMALL).cells),
                            }
                        ],
                    }
                ]
            )
        }
        for watch in watches:
            bar = cutoff_report.read_bar(watch.threshold)
            assert bar is not None, f"unreadable bar: {watch.threshold!r}"
            assert bar.metric in offered

    def test_a_record_with_no_frames_offers_nothing(self):
        assert rim_stalk_scale.readings([{"geometry": {"edges": []}, "frames": []}]) == {}


class TestTheScriptRuns:
    def test_read_runs_without_touching_the_tracker(self, capsys, tmp_path):
        assert (
            rim_stalk_scale.main(
                [
                    "read",
                    "--dome",
                    "small",
                    "--ticks",
                    "12",
                    "--seeds",
                    "0",
                    "--out",
                    "-",
                    "--no-file",
                ]
            )
            == 0
        )
        printed = capsys.readouterr().out
        assert "the stalks are identically zero" in printed
        assert "the chaining" in printed
