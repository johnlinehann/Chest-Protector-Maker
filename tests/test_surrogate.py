import pytest

from chestforge import SIM_LABEL
from chestforge.config import load_nd200
from chestforge.core import calibration
from chestforge.core.calibration import CalibrationError, require_calibrated, verify
from chestforge.core.impact_sim import simulate_bare_nts
from chestforge.core.projectile import load_projectile
from chestforge.core.surrogate import load_nts_prior
from chestforge.units import mph_to_ms


@pytest.mark.parametrize("ball", ["baseball", "lacrosse"])
def test_bare_nts_hits_appendix1_targets(calibrated, ball):
    nts = require_calibrated(ball)
    p = load_projectile(ball)
    peak = lambda mph: simulate_bare_nts(p, nts, mph_to_ms(mph)).peak_lbf
    assert peak(30.0) == pytest.approx(140, abs=2)
    assert peak(29.1) > 130
    assert peak(30.9) < 150


@pytest.mark.parametrize("ball", ["baseball", "lacrosse"])
def test_all_calibration_checks_pass(calibrated, ball):
    entry = calibrated["projectiles"][ball]
    assert entry["calibrated"] and entry["gain_in_sanity_band"]
    assert entry["checks"] and all(c["passed"] for c in entry["checks"])


def test_table_mass_within_section_9_4(calibrated):
    limit_kg = load_nd200().raw["surrogate"]["linear_bearing_table_max_weight_lb"]["value"] * 0.45359237
    assert calibrated["mechanics"]["table_mass_kg"] <= limit_kg


def test_energy_is_conserved(calibrated):
    r = simulate_bare_nts(load_projectile(), require_calibrated("baseball"), mph_to_ms(30))
    assert abs(r.energy_residual_J) < 1e-4 * r.energy_in_J
    assert 0 < r.energy_dissipated_J < r.energy_in_J


def test_results_are_labelled_and_sampled_fast_enough(calibrated):
    r = simulate_bare_nts(load_projectile(), require_calibrated("baseball"), mph_to_ms(30))
    assert r.label == SIM_LABEL
    fs = 1 / (r.t[1] - r.t[0])
    assert fs >= load_nd200().min_sample_rate_hz - 1e-6


def test_softball_runs_but_is_flagged(calibrated):
    r = simulate_bare_nts(load_projectile("softball"), require_calibrated("softball"), mph_to_ms(30))
    assert r.beyond_standard


def test_guard_refuses_when_uncalibrated(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "CALIBRATION_FILE", tmp_path / "missing.yaml")
    monkeypatch.setattr(calibration, "_verified", {})
    with pytest.raises(CalibrationError):
        require_calibrated("baseball")


def test_guard_refuses_when_config_changed(calibrated, monkeypatch):
    monkeypatch.setattr(calibration, "config_hash", lambda *a: "different")
    monkeypatch.setattr(calibration, "_verified", {})
    with pytest.raises(CalibrationError, match="config changed"):
        require_calibrated("baseball")


def test_verify_fails_for_miscalibrated_surrogate():
    p = load_projectile()
    soft = load_nts_prior(p.mass_kg).with_stiffness(5000)   # bare peak ~58 lbf
    assert not all(c.passed for c in verify(p, soft))
