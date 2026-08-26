"""The domes the suite shares, in one place (ticket #110).

Fixtures and shared definitions only. Nothing here touches what pytest
collects, and `tests/test_perturbation.py`'s :class:`TestBothChecksRunInCI`
holds that down: a file of this name is permitted precisely because it narrows
nothing.
"""

from patchworks.graph import DomeSpec

#: The small dome most of the suite runs on: 39 cells, 15 of them predicting,
#: 54 edges. Small enough to sweep every cell in the graph twice, and built by
#: the same rules as the real one, so every seam is present at this size.
#:
#: Shared rather than copied because `tests/test_perturbation.py` reads cell
#: indices off it. A second copy drifting from this one would leave the
#: locality guard pointed at a different cell than the one its comments name,
#: and a guard aimed at the wrong cell does not fail — it stops guarding.
#: (`tests/test_bias_selection.py` and `tests/test_dome_panel.py` name domes of
#: their own; those are genuinely different domes, not copies of this one.)
SMALL = DomeSpec(
    patch_grid=4,
    vision_sides=(2,),
    somatomotor_sizes=(4,),
    core_sizes=(4, 3),
    core_degree=4,
    apex_degree=3,
)
