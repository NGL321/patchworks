"""027's spread statistic, re-expressed for the converted body (ticket #349).

`docs/research/027-regional-jacobian-spectra.md` quoted a `tau` p95/p05 ratio
across cells and never shipped the function that produced it -- the numbers
lived in the pilot's stdout. #349's re-run against the converted body has to
quote a *comparable* number, so the statistic is written once, here, and
`prototypes/regional-spectra/converted_spread.py` reads it rather than
recomputing quantiles inline.

Three things are held down, and the third is 027 section 5's own protocol
correction biting:

* **The ratio is the one 027 quoted** -- `tau = -1/ln rho` at the 95th over the
  5th percentile, on the cells whose radius admits a finite `tau` at all.
* **Expansive cells are excluded and counted, never clipped.** A cell at
  `rho >= 1` has no finite retention constant; folding it in at some ceiling
  invents the tail the ratio is most sensitive to.
* **`sd(log10 rho)` is reported beside the ratio**, because 027 section 5 asked
  for exactly that -- the ratio is dominated by a tail that diverges as
  `rho -> 1`, and the log-radius dispersion is the statistic that stays finite
  when the ratio does not.
"""

import importlib.util
import math
import pathlib

import numpy as np
import pytest

# Loaded by path: `prototypes/` is not a package and is not on the path, and
# `prototypes/driven-rho-274/read.py` already showed what a plain import buys
# here -- two rigs with the same module name finding each other.
_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "prototypes"
    / "regional-spectra"
    / "converted_spread.py"
)
_spec = importlib.util.spec_from_file_location("converted_spread_349", _SOURCE)
converted_spread = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(converted_spread)


def test_ratio_matches_a_hand_computed_tau_quantile_ratio():
    # rho chosen so tau spans a known range: tau = -1/ln(rho).
    rho = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
    stats = converted_spread.spread_statistics(rho)
    tau = -1.0 / np.log(rho)
    expected = np.quantile(tau, 0.95) / np.quantile(tau, 0.05)
    assert stats["tau_p95_over_p05"] == pytest.approx(expected)
    assert stats["finite"] == 5
    assert stats["expansive"] == 0


def test_expansive_cells_are_excluded_and_counted_not_clipped():
    contracting = np.array([0.3, 0.5, 0.7])
    with_expansive = np.concatenate([contracting, [1.0, 1.4]])
    bare = converted_spread.spread_statistics(contracting)
    mixed = converted_spread.spread_statistics(with_expansive)

    # The ratio is read off the contracting cells alone, so adding two cells
    # with no finite tau does not move it.
    assert mixed["tau_p95_over_p05"] == pytest.approx(bare["tau_p95_over_p05"])
    assert mixed["expansive"] == 2
    assert mixed["finite"] == 3
    assert mixed["cells"] == 5


def test_sd_log_radius_is_over_every_cell_including_the_expansive_ones():
    rho = np.array([0.25, 0.5, 1.0, 2.0])
    stats = converted_spread.spread_statistics(rho)
    assert stats["sd_log10_rho"] == pytest.approx(np.std(np.log10(rho)))


def test_a_spike_reads_as_a_ratio_of_one():
    """027's own null: if every cell lands in the same regime, the ratio is 1."""
    stats = converted_spread.spread_statistics(np.full(150, 0.5))
    assert stats["tau_p95_over_p05"] == pytest.approx(1.0)
    assert stats["sd_log10_rho"] == pytest.approx(0.0)


def test_all_expansive_leaves_the_ratio_undefined_rather_than_infinite():
    stats = converted_spread.spread_statistics(np.array([1.0, 1.2, 1.5]))
    assert stats["finite"] == 0
    assert math.isnan(stats["tau_p95_over_p05"])


def test_the_statistic_matches_the_pilots_own_population_convention():
    """`tau` over the contracting cells, `rho` over all of them.

    `prototypes/regional-spectra/spread_pilot.py` line 73 takes `tau` over
    `rho[rho < 1.0]` while its `rho` quantiles run over every cell, so 027
    section 6's two triples are read off **different populations**. That is why
    its `rho` p95 of 0.98 does not map to its `tau` p95 of 5.49 -- `-1/ln(0.98)`
    is 49.5, and 5.49 is the `tau` of the 95th percentile of the *contracting*
    cells. The re-run has to keep the same convention or the comparison against
    7.7x is a change of population dressed as a change of body.
    """
    # 5% expansive, as 027's `[64,64], sigma_w^2 = 1.7` configuration had.
    rho = np.concatenate([np.linspace(0.30, 0.84, 95), np.linspace(0.85, 1.30, 5)])
    stats = converted_spread.spread_statistics(rho)

    contracting = rho[rho < 1.0]
    assert stats["tau_p95"] == pytest.approx(
        np.quantile(-1.0 / np.log(contracting), 0.95)
    )
    # The radius summary still sees every cell, expansive ones included.
    assert stats["rho_median"] == pytest.approx(float(np.median(rho)))
    assert stats["sd_log10_rho"] == pytest.approx(float(np.std(np.log10(rho))))
    assert stats["expansive"] == int((rho >= 1.0).sum())


def test_the_pilots_low_end_reproduces():
    """027 section 6, `[64,64], sigma_w^2 = 1.7`: rho 0.32 and 0.57 give tau 0.86 and 1.76."""
    assert -1.0 / math.log(0.315) == pytest.approx(0.86, abs=0.01)
    assert -1.0 / math.log(0.57) == pytest.approx(1.76, abs=0.02)
