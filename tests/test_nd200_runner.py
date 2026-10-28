import math

import pytest

from chestforge.core import calibration
from chestforge.core.calibration import CalibrationError
from chestforge.core.nd200_runner import ImpactRecord, ND200Report, run_nd200
from chestforge.core.padding import PadLayer, layer, load_material


@pytest.fixture(scope="module")
def report(calibrated):
    return run_nd200(PadLayer(load_material("eva_100"), 0.02), "baseball", torso_length_in=15.0)


def test_runs_full_table1_matrix(report):
    assert len(report.impacts) == 6
    assert [(i.sample, i.location) for i in report.impacts] == [
        (1, "cardiac_center"), (1, "random_1"), (1, "random_2"),
        (2, "cardiac_center"), (2, "random_1"), (2, "random_2")]
    assert {i.velocity_mph for i in report.impacts[:3]} == {30}
    assert {i.velocity_mph for i in report.impacts[3:]} == {50}
    assert {i.limit_lbf for i in report.impacts[:3]} == {90}
    assert {i.limit_lbf for i in report.impacts[3:]} == {180}


def test_report_is_labelled(report):
    assert report.label == "Simulated — not NOCSAE certified"
    assert "Simulated — not NOCSAE certified" in report.to_text()


def test_margin_formula(report):
    for i in report.impacts:
        assert i.margin_pct == pytest.approx((i.limit_lbf - i.peak_lbf) / i.limit_lbf * 100)


def _record(peak, limit=90.0):
    return ImpactRecord(1, "30mph", "cardiac_center", (0, 0), 30, peak, limit,
                        (limit - peak) / limit * 100, peak <= limit, False, result=None)


def test_pass_fail_boundary():
    at_limit = ND200Report("x", "baseball", 15, 2.0, [_record(90.0)] * 6, beyond_standard=False)
    over = ND200Report("x", "baseball", 15, 2.0, [_record(90.0)] * 5 + [_record(90.01)], beyond_standard=False)
    assert at_limit.verdict == "PASS"          # §14: "shall not exceed"
    assert over.verdict == "FAIL"              # any single impact fails the protector


@pytest.mark.parametrize("torso, shift", [(13.5, 1.75), (15.0, 2.0), (16.0, 2.25)])
def test_shifts_follow_torso_band(calibrated, torso, shift):
    rep = run_nd200(PadLayer(load_material("poron_xrd_240"), 0.03), "baseball", torso_length_in=torso)
    assert rep.max_shift_in == shift
    assert max(max(abs(c) for c in i.shift_in) for i in rep.impacts) == shift


def test_uniform_design_random_impacts_respect_spacing(report):
    for sample in (report.impacts[:3], report.impacts[3:]):
        pts = [i.shift_in for i in sample]
        assert all(math.dist(a, b) >= 1.0 for k, a in enumerate(pts) for b in pts[k + 1:])


def test_velocity_scale_limited_to_tolerance(calibrated):
    with pytest.raises(ValueError):
        run_nd200(PadLayer(load_material("eva_100"), 0.02), velocity_scale=1.05)


def test_softball_is_beyond_standard(calibrated):
    rep = run_nd200(PadLayer(load_material("poron_xrd_240"), 0.03), "softball")
    assert rep.verdict.startswith("BEYOND STANDARD")


def test_refuses_without_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "CALIBRATION_FILE", tmp_path / "none.yaml")
    monkeypatch.setattr(calibration, "_verified", {})
    with pytest.raises(CalibrationError):
        run_nd200(PadLayer(load_material("eva_100"), 0.02))


def test_worst_case_shift_finds_plate_edge(calibrated):
    from chestforge.core.layout import load_design
    rep = run_nd200(load_design("designs/example_plate_foam.yaml"), torso_length_in=15.0)
    for sample in (rep.impacts[:3], rep.impacts[3:]):
        center, r1, r2 = sample
        assert r1.peak_lbf >= r2.peak_lbf >= center.peak_lbf
        assert r1.peak_lbf > center.peak_lbf                  # edge-limited spreading is worse
        assert abs(math.hypot(*r1.shift_in) - 2.0) < 1e-6     # at the §12.2 limit
        assert math.dist(r1.shift_in, r2.shift_in) >= 1.0     # §12.2 contact spacing
        assert "plate edge" in r1.local


def test_monte_carlo_p95_in_report(calibrated):
    rep = run_nd200((layer("pc", 4), layer("poron_xrd_240", 15)), mc_samples=16, seed=3)
    assert rep.has_mc and rep.p95_passed is not None
    for i in rep.impacts:
        assert i.p95_lbf is not None and i.mc_samples == 16
    assert "P95 over 16 Monte Carlo" in rep.to_text()
