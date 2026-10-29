import glob
import shutil

import numpy as np
import pytest
import yaml

from chestforge.validation import fit as F
from chestforge.validation.metrics import compare
from chestforge.validation.traces import align, load_trace, parse_stack, write_trace

EX = "data/experimental/examples"
TRUTH = yaml.safe_load(open(f"{EX}/GROUND_TRUTH.yaml"))


@pytest.fixture(scope="module")
def traces():
    return [load_trace(p) for p in sorted(glob.glob(f"{EX}/*.csv"))]


def test_examples_are_marked_synthetic(traces):
    assert len(traces) == 6 and all(t.synthetic for t in traces)


def test_loader_units_filter_and_alignment(tmp_path):
    t = np.arange(0, 0.03, 1 / 50000)
    f = np.where((t > 0.005) & (t < 0.011), np.sin(np.pi * (t - 0.005) / 0.006), 0.0) * 100   # 100 lbf half-sine
    write_trace(tmp_path / "a.csv", t, f * 4.4482216152605,
                {"rig": "nts", "projectile": "baseball", "velocity_mph": 30, "stack": "pc:3, eva_100:15", "filtered": "true"})
    tr = load_trace(tmp_path / "a.csv")
    assert tr.peak_N / 4.4482216152605 == pytest.approx(100, rel=1e-3)
    assert tr.t_s[np.argmax(tr.force_N)] == pytest.approx(0.003 - 0.006 * np.arcsin(0.05) / np.pi, abs=1e-4)
    assert [l.material.id for l in tr.stack] == ["pc", "eva_100"]
    assert tr.velocity_ms == pytest.approx(13.4112)


@pytest.mark.parametrize("bad, msg", [
    ({"rig": "anvil"}, "rig"), ({"rig": "nts", "velocity_mph": 30, "_units": "kN"}, "force_units"),
    ({"rig": "nts"}, "velocity"), ({"rig": "drop_tower", "velocity_ms": 3}, "impactor"),
])
def test_loader_rejects_bad_metadata(tmp_path, bad, msg):
    t = np.arange(0, 0.01, 1e-5)
    unit = bad.pop("_units", None)
    write_trace(tmp_path / "b.csv", t, np.sin(t * 300) ** 2, bad)
    if unit:
        f = tmp_path / "b.csv"
        f.write_text(f.read_text().replace("# force_units: lbf", f"# force_units: {unit}"))
    with pytest.raises(ValueError, match=msg):
        load_trace(tmp_path / "b.csv")


def test_parse_stack():
    assert parse_stack("bare") == () and parse_stack("") == ()
    with pytest.raises(ValueError):
        parse_stack("pc")


def test_metrics_identity_and_offset():
    t = np.arange(-0.002, 0.02, 5e-5)
    f = np.clip(np.sin(np.pi * t / 0.008), 0, None) * (t < 0.008) * 500
    m = compare(t, f, t, f)
    assert m["peak_err_pct"] == 0 and m["nrmse_pct"] == 0 and m["r2"] == 1
    m2 = compare(t, f, t, f * 1.1)
    assert m2["peak_err_pct"] == pytest.approx(10) and m2["impulse_err_pct"] == pytest.approx(10, rel=1e-3)


def test_align_onset():
    t = np.linspace(0, 0.01, 1001)
    f = np.clip(t - 0.004, 0, None)
    assert align(t, f)[np.argmax(f >= 0.05 * f.max())] == pytest.approx(0.0, abs=2e-5)


def test_fit_nts_recovers_ground_truth(calibrated, traces):
    fit = F.fit_nts(traces)
    tn = TRUTH["nts"]
    assert fit.params["skin_damping_ratio"] == pytest.approx(tn["skin_damping_ratio"], rel=0.05)
    assert fit.params["loading_surface_mass_kg"] == pytest.approx(tn["loading_surface_mass_kg"], rel=0.10)
    assert fit.params["skin_stiffness_N_per_m"] == pytest.approx(tn["skin_stiffness_N_per_m"], rel=0.03)
    assert fit.after["mean_abs_nrmse_pct"] < 1.0 < fit.before["mean_abs_nrmse_pct"]


def test_fit_material_recovers_ground_truth(calibrated, traces):
    fit = F.fit_material(traces, "eva_100", "eva_100_lab")
    m = fit.params["multipliers"]
    tm = TRUTH["eva_100_multipliers"]
    for k in ("A_Pa", "rate_C", "unloading_ratio"):
        assert m[k] == pytest.approx(tm[k], rel=0.10)
    assert fit.after["mean_abs_peak_err_pct"] < 1.0 < fit.before["mean_abs_peak_err_pct"]
    assert fit.params["material"].id == "eva_100_lab"


def test_apply_refuses_synthetic(calibrated, traces):
    fit = F.fit_material(traces, "eva_100")
    with pytest.raises(F.SyntheticDataError):
        F.apply_material(fit)


def test_apply_writes_override_files(calibrated, traces, tmp_path, monkeypatch):
    """With an isolated config dir: material fit becomes a new id; NTS fit overrides + recalibrates."""
    from chestforge import config
    from chestforge.core import calibration, padding
    cfg = tmp_path / "config"
    shutil.copytree(config.CONFIG_DIR, cfg)
    for mod in (config, F, padding):
        monkeypatch.setattr(mod, "CONFIG_DIR", cfg, raising=False)
    monkeypatch.setattr(calibration, "CALIBRATION_FILE", cfg / "nts_calibrated.yaml")
    config.load_nts.cache_clear(); padding._materials.cache_clear(); calibration._verified.clear()
    try:
        mfit = F.fit_material(traces, "eva_100", "eva_100_lab")
        F.apply_material(mfit, allow_synthetic=True)
        mats = padding.load_materials()
        assert "eva_100_lab" in mats and "eva_100" in mats
        assert mats["eva_100_lab"].A_Pa == pytest.approx(mats["eva_100"].A_Pa * mfit.params["multipliers"]["A_Pa"], rel=1e-5)
        nfit = F.fit_nts(traces)
        out = F.apply_nts(nfit, allow_synthetic=True)
        assert all(out["calibration"].values())                      # recalibrated to Appendix 1
        assert config.load_nts()["skin_damping_ratio"]["value"] == pytest.approx(nfit.params["skin_damping_ratio"], rel=1e-4)
        assert config.load_nts()["skin_damping_ratio"]["estimated"] is False
    finally:
        config.load_nts.cache_clear(); padding._materials.cache_clear(); calibration._verified.clear()
