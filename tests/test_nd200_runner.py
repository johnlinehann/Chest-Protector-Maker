import pytest

from chestforge.core import calibration
from chestforge.core.calibration import CalibrationError
from chestforge.core.nd200_runner import ImpactRecord, ND200Report, random_shifts, run_nd200
from chestforge.core.padding import PadLayer, load_material


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


def test_contact_points_respect_min_spacing():
    with pytest.raises(ValueError):
        random_shifts(0.5, 1.0)
    pts = random_shifts(1.75, 1.0)
    assert pts["cardiac_center"] == (0.0, 0.0)


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
