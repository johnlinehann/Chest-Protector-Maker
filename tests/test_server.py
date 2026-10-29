import io
import json
import zipfile

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from chestforge import server
from chestforge.optimize.objectives import Evaluation


@pytest.fixture(scope="module")
def client():
    return TestClient(server.app)


@pytest.fixture(scope="module")
def design(client):
    return client.get("/api/meta").json()["default_design"]


def test_meta_and_ui_served(client):
    m = client.get("/api/meta").json()
    assert m["label"] == "Simulated — not NOCSAE certified"
    assert {x["id"] for x in m["materials"]} >= {"pc", "hdpe", "eva_100", "pe_45", "poron_xrd_240"}
    assert {(l["condition"], l["max_peak_lbf"]) for l in m["limits"]} == {("30mph", 90), ("50mph", 180)}
    assert client.get("/").status_code == 200
    assert "Simulated — not NOCSAE certified" in client.get("/").text
    assert client.get("/vendor/three.module.min.js").status_code == 200


@pytest.mark.parametrize("torso, shift", [(13.5, 1.75), (15, 2.0), (16, 2.25)])
def test_max_shift(client, torso, shift):
    assert client.get(f"/api/max_shift?torso_length_in={torso}").json()["max_shift_in"] == shift


def test_nd200_report_json(calibrated, client, design):
    r = client.post("/api/nd200", json={"design": design, "torso_length_in": 15, "mc_samples": 8})
    assert r.status_code == 200
    j = r.json()
    assert len(j["impacts"]) == 6 and j["verdict"] in ("PASS", "FAIL") and j["p95_verdict"] in ("PASS", "FAIL")
    assert j["label"] == "Simulated — not NOCSAE certified"
    i = j["impacts"][0]
    assert len(i["t_ms"]) == len(i["force_lbf"]) > 50
    assert max(i["force_lbf"]) == pytest.approx(i["peak_lbf"], rel=1e-3)
    assert j["metrics"]["mass_g"] > 0


def test_input_validation(client, design):
    bad = {"design": {"name": "x", "zones": [{"name": "c", "outline": {"a": 4, "b": 4},
                                               "stack": [{"material": "unobtanium", "thickness_mm": 5}]}]}}
    assert client.post("/api/nd200", json=bad).status_code == 422
    thick = json.loads(json.dumps(design))
    thick["zones"][0]["stack"][0]["thickness_mm"] = 500
    assert client.post("/api/nd200", json={"design": thick}).status_code == 422
    assert client.post("/api/nd200", json={"design": design, "mc_samples": 10_000}).status_code == 422
    assert client.post("/api/nd200", json={"design": design, "projectile": "cricket"}).status_code == 422


def test_refuses_without_calibration(client, design, tmp_path, monkeypatch):
    from chestforge.core import calibration
    monkeypatch.setattr(calibration, "CALIBRATION_FILE", tmp_path / "none.yaml")
    monkeypatch.setattr(calibration, "_verified", {})
    assert client.post("/api/nd200", json={"design": design}).status_code == 409


def test_exports(calibrated, client, design):
    req = {"design": design, "torso_length_in": 15, "mc_samples": 0}
    csv_r = client.post("/api/export/csv", json=req)
    assert csv_r.status_code == 200 and csv_r.text.startswith("# Simulated — not NOCSAE certified")
    assert csv_r.text.splitlines()[2].count(",") == 6          # t_ms + 6 impacts
    pdf = client.post("/api/export/pdf", json=req)
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"


def test_geometry_and_cad_zip(client, design):
    pytest.importorskip("build123d")
    g = client.post("/api/geometry", json={"design": design, "torso_length_in": 15}).json()
    assert len(g["layers"]) == 3
    for l in g["layers"]:
        assert len(l["positions"]) % 3 == 0 and max(l["indices"]) < len(l["positions"]) // 3
    assert len(g["landmarks"]["shift_circle"]) > 10
    z = zipfile.ZipFile(io.BytesIO(client.post("/api/export/cad", json={"design": design, "torso_length_in": 15}).content))
    assert {"protector.step", "protector.stl", "ntop_lattice.json", "summary.json"} <= set(z.namelist())


def test_runs_listing_and_traversal_guard(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RUNS", tmp_path)
    run = tmp_path / "demo"
    run.mkdir()
    p = {"plate_material": "pc", "plate_mm": 4.0, "plate_a_in": 4.0, "plate_b_in": 4.0, "outer_material": "none",
         "outer_mm": 0.0, "inner_material": "pe_45", "inner_mm": 12.0}
    ev = Evaluation(params=p, p95_50mph_lbf=100, p95_30mph_lbf=60, min_p95_margin_pct=30, min_nominal_margin_pct=32,
                    mass_g=280, thickness_mm=16, bending_Nm=15, feasible=True, bottomed=False, flags=[])
    (run / "evaluations.jsonl").write_text(json.dumps({"method": "tpe", "evaluation": ev.to_json()}) + "\n")
    assert client.get("/api/runs").json() == [{"name": "demo", "evaluations": 1, "running": False}]
    d = client.get("/api/runs/demo").json()
    assert d["points"][0]["pareto"] and d["points"][0]["design"]["zones"][0]["name"] == "cardiac"
    assert client.get("/api/runs/a.b").status_code == 422
    assert client.get("/api/runs/missing").status_code == 404
    assert client.post("/api/optimize", json={"name": "../evil"}).status_code == 422
