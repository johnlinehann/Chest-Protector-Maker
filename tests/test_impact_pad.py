import pytest

from chestforge.core.calibration import require_calibrated
from chestforge.core.impact_sim import simulate_bare_nts, simulate_impact
from chestforge.core.padding import PadLayer, load_material
from chestforge.core.projectile import load_projectile
from chestforge.units import mph_to_ms


@pytest.fixture(scope="module")
def setup(calibrated):
    return load_projectile("baseball"), require_calibrated("baseball")


def _run(setup, mid, mm, mph):
    ball, nts = setup
    return simulate_impact(ball, nts, mph_to_ms(mph), PadLayer(load_material(mid), mm / 1000))


def test_pad_reduces_peak_and_thicker_is_better(setup):
    bare = simulate_bare_nts(*setup, mph_to_ms(30)).peak_lbf
    peaks = [_run(setup, "poron_xrd_240", mm, 30).peak_lbf for mm in (10, 20, 30)]
    assert bare > peaks[0] > peaks[1] > peaks[2]


def test_energy_conserved_with_pad(setup):
    r = _run(setup, "eva_100", 20, 50)
    assert abs(r.energy_residual_J) < 1e-4 * r.energy_in_J
    assert r.energy_absorbed_pad_J > 0
    assert r.energy_absorbed_pad_J + r.energy_dissipated_J < r.energy_in_J


def test_bottoming_out_detected(setup):
    thin = _run(setup, "pe_45", 10, 50)
    thick = _run(setup, "pe_45", 30, 50)
    assert thin.bottomed_out and thin.max_pad_strain >= load_material("pe_45").densification_strain
    assert not thick.bottomed_out


def test_bare_path_reports_no_pad(setup):
    r = simulate_bare_nts(*setup, mph_to_ms(30))
    assert r.max_pad_compression_m == 0 and r.energy_absorbed_pad_J == 0 and not r.bottomed_out
