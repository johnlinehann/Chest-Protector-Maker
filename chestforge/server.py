"""Local web app backend: JSON API + static Three.js UI (web/). Binds to localhost only.

Run:  python -m chestforge.server      (or the `chestforge` command / desktop icon)
"""

from __future__ import annotations

import csv
import io
import re
import subprocess
import sys
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chestforge import SIM_LABEL
from chestforge.config import load_nd200, load_projectiles, load_sizes
from chestforge.core.calibration import CalibrationError
from chestforge.core.layout import Protector, protector_from_dict, protector_metrics
from chestforge.core.nd200_runner import ND200Report, run_nd200
from chestforge.core.padding import FoamMaterial, load_materials
from chestforge.optimize.objectives import Evaluation
from chestforge.optimize.optimizer import RUNS, pareto_front
from chestforge.optimize.search_space import load_optimizer_config, to_dict

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
RUN_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")

app = FastAPI(title="ChestForge", description=SIM_LABEL)


@app.on_event("startup")
def _warm_cad() -> None:
    """Import build123d/OpenCascade in the background: the first import takes several seconds."""
    def warm():
        try:
            import chestforge.geometry.protector_gen  # noqa: F401
            import build123d  # noqa: F401
        except ImportError:
            pass
    threading.Thread(target=warm, daemon=True).start()


# ---------- request models (validated; nothing reaches the physics unchecked) ----------

class EllipseIn(BaseModel):
    cx: float = Field(0.0, ge=-20, le=20)
    cy: float = Field(0.0, ge=-20, le=20)
    a: float = Field(..., gt=0, le=20)
    b: float = Field(..., gt=0, le=20)


class LayerIn(BaseModel):
    material: str = Field(..., max_length=64)
    thickness_mm: float = Field(..., gt=0, le=60)
    extent: EllipseIn | None = None


class ZoneIn(BaseModel):
    name: str = Field(..., max_length=40)
    outline: EllipseIn
    stack: list[LayerIn] = Field(..., min_length=1, max_length=8)


class DesignIn(BaseModel):
    name: str = Field("design", max_length=120)
    zones: list[ZoneIn] = Field(..., min_length=1, max_length=8)


class RunIn(BaseModel):
    design: DesignIn
    projectile: Literal["baseball", "lacrosse", "softball"] = "baseball"
    torso_length_in: float = Field(15.0, ge=8, le=24)
    mc_samples: int = Field(0, ge=0, le=500)
    seed: int = Field(0, ge=0, le=2**31 - 1)


class GeometryIn(BaseModel):
    design: DesignIn
    torso_length_in: float = Field(15.0, ge=8, le=24)


class OptimizeIn(BaseModel):
    name: str
    method: Literal["tpe", "nsga2", "surrogate", "all"] = "all"
    trials: int = Field(60, ge=4, le=1000)
    rounds: int = Field(5, ge=1, le=50)


def _protector(d: DesignIn) -> Protector:
    try:
        return protector_from_dict(d.model_dump())
    except (KeyError, ValueError) as e:
        raise HTTPException(422, f"invalid design: {e}") from e


def _run_name(name: str) -> str:
    if not RUN_NAME.match(name):
        raise HTTPException(422, "run name: letters, digits, _ and - only (max 40)")
    return name


# ---------- metadata ----------

@app.get("/api/meta")
def meta() -> dict:
    nd = load_nd200()
    mats = []
    for m in load_materials().values():
        mats.append({"id": m.id, "name": m.name, "kind": "foam" if isinstance(m, FoamMaterial) else "plate",
                     "density_kg_m3": m.density_kg_m3, "estimated": m.estimated})
    projs = [{"id": k, "nd200": v.get("nd200", True)} for k, v in load_projectiles().items() if k != "default"]
    return {
        "label": SIM_LABEL,
        "materials": mats,
        "projectiles": projs,
        "limits": [{"condition": p.name, "velocity_mph": p.velocity_mph, "max_peak_lbf": p.max_peak_lbf}
                   for p in nd.performance],
        "shift_bands": [{"band": b.torso_length_in, "max_shift_in": b.max_shift_in, "source": b.source}
                        for b in nd.shift_bands],
        "sizes": load_sizes().get("sizes", []),
        "silhouette": {k: nd.raw["surrogate"]["geometry"][k]["value"]
                       for k in ("silhouette_width_in", "silhouette_height_in")},
        "default_design": _default_design(),
    }


def _default_design() -> dict:
    p = {"plate_material": "pc", "plate_mm": 4.5, "plate_a_in": 4.0, "plate_b_in": 4.0, "outer_material": "none",
         "outer_mm": 0.0, "inner_material": "pe_45", "inner_mm": 13.0}
    return to_dict(p, load_optimizer_config())


@app.get("/api/max_shift")
def max_shift(torso_length_in: float) -> dict:
    try:
        return {"max_shift_in": load_nd200().max_shift_in(torso_length_in)}
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


# ---------- ND200 ----------

def _report_json(rep: ND200Report, prot: Protector) -> dict:
    impacts = []
    for i in rep.impacts:
        r = i.result
        f_lbf = r.force_N / 4.4482216152605
        nz = np.flatnonzero(np.abs(r.force_raw_N) > 0)
        end = min(len(r.t), (nz[-1] if len(nz) else len(r.t) - 1) + 40)
        impacts.append({
            "sample": i.sample, "condition": i.condition, "location": i.location,
            "shift_in": list(i.shift_in), "velocity_mph": i.velocity_mph,
            "peak_lbf": i.peak_lbf, "p95_lbf": i.p95_lbf, "limit_lbf": i.limit_lbf,
            "margin_pct": i.margin_pct, "p95_margin_pct": i.p95_margin_pct, "passed": i.passed,
            "local": i.local, "bottomed": list(r.bottomed_layers), "over_capacity": i.exceeds_load_cell_capacity,
            "energy_in_J": r.energy_in_J, "energy_absorbed_pad_J": r.energy_absorbed_pad_J,
            "max_pad_strain": r.max_pad_strain,
            "t_ms": np.round(r.t[:end] * 1000, 3).tolist(), "force_lbf": np.round(f_lbf[:end], 2).tolist(),
        })
    return {
        "label": rep.label, "design": rep.design, "projectile": rep.projectile,
        "torso_length_in": rep.torso_length_in, "max_shift_in": rep.max_shift_in,
        "verdict": rep.verdict, "p95_verdict": None if rep.p95_passed is None else ("PASS" if rep.p95_passed else "FAIL"),
        "min_margin_pct": rep.min_margin_pct, "min_p95_margin_pct": rep.min_p95_margin_pct,
        "beyond_standard": rep.beyond_standard, "covers_silhouette": rep.covers_silhouette, "flags": rep.flags,
        "metrics": protector_metrics(prot), "impacts": impacts, "text": rep.to_text(),
    }


def _run(req: RunIn) -> tuple[ND200Report, Protector]:
    prot = _protector(req.design)
    try:
        return run_nd200(prot, req.projectile, req.torso_length_in, mc_samples=req.mc_samples, seed=req.seed,
                         design_name=req.design.name), prot
    except CalibrationError as e:
        raise HTTPException(409, f"NTS not calibrated: {e}") from e


@app.post("/api/nd200")
def nd200(req: RunIn) -> dict:
    rep, prot = _run(req)
    return _report_json(rep, prot)


# ---------- geometry ----------

@app.post("/api/geometry")
def geometry(req: GeometryIn) -> dict:
    try:
        from chestforge.geometry.export import landmarks
        from chestforge.geometry.protector_gen import BodyFrame, build_layers
    except ImportError as e:
        raise HTTPException(501, "3D geometry needs the cad extra: pip install -e .[cad]") from e
    prot = _protector(req.design)
    try:
        layers = build_layers(prot, req.torso_length_in)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    out = []
    for ls in layers:
        verts, tris = ls.solid.tessellate(1.0)
        out.append({"name": ls.name, "zone": ls.zone, "material": ls.layer.material.id,
                    "kind": "plate" if "plate" in ls.name else "foam",
                    "positions": np.round([[v.X, v.Y, v.Z] for v in verts], 3).ravel().tolist(),
                    "indices": np.asarray(tris, dtype=int).ravel().tolist(),
                    "mass_g": ls.mass_g})
    frame = BodyFrame.for_torso(req.torso_length_in)
    return {"layers": out, "landmarks": landmarks(frame, load_nd200().max_shift_in(req.torso_length_in)),
            "torso_length_mm": req.torso_length_in * 25.4}


# ---------- exports ----------

@app.post("/api/export/cad")
def export_cad(req: GeometryIn) -> Response:
    try:
        from chestforge.geometry.export import export_design, slug
    except ImportError as e:
        raise HTTPException(501, "CAD export needs the cad extra: pip install -e .[cad]") from e
    prot = _protector(req.design)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            export_design(prot, req.torso_length_in, Path(tmp))
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(Path(tmp).iterdir()):
                z.write(f, f.name)
    name = slug(prot.name)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{name}_cad.zip"'})


@app.post("/api/export/csv")
def export_csv(req: RunIn) -> Response:
    rep, _ = _run(req)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([f"# {SIM_LABEL}"])
    w.writerow(["# design", rep.design, "projectile", rep.projectile, "torso_in", rep.torso_length_in])
    cols = [f"S{i.sample}_{i.location}_{i.velocity_mph:.0f}mph_lbf" for i in rep.impacts]
    w.writerow(["t_ms", *cols])
    t = rep.impacts[0].result.t
    traces = [i.result.force_N / 4.4482216152605 for i in rep.impacts]
    for k in range(len(t)):
        w.writerow([f"{t[k] * 1000:.3f}", *[f"{tr[k]:.3f}" for tr in traces]])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="nd200_traces.csv"'})


@app.post("/api/export/pdf")
def export_pdf(req: RunIn) -> Response:
    from chestforge.report import nd200_pdf
    rep, prot = _run(req)
    return Response(nd200_pdf(rep, protector_metrics(prot)), media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="nd200_report.pdf"'})


# ---------- optimizer runs ----------

_procs: dict[str, subprocess.Popen] = {}
_lock = threading.Lock()


def _load_evals(name: str) -> tuple[list[Evaluation], list[str]]:
    path = RUNS / name / "evaluations.jsonl"
    if not path.exists():
        raise HTTPException(404, f"no run '{name}'")
    import json
    evals, methods = [], []
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            evals.append(Evaluation(**rec["evaluation"]))
            methods.append(rec["method"])
    return evals, methods


@app.get("/api/runs")
def runs() -> list[dict]:
    out = []
    if RUNS.exists():
        for d in sorted(RUNS.iterdir()):
            f = d / "evaluations.jsonl"
            if d.is_dir() and f.exists() and RUN_NAME.match(d.name):
                out.append({"name": d.name, "evaluations": sum(1 for l in f.read_text().splitlines() if l.strip()),
                            "running": _running(d.name)})
    return out


@app.get("/api/runs/{name}")
def run_detail(name: str) -> dict:
    _run_name(name)
    cfg = load_optimizer_config()
    evals, methods = _load_evals(name)
    names = cfg["objectives"]
    front_ids = {id(e) for e in pareto_front(evals, names)} if evals else set()
    pts = []
    for k, (e, m) in enumerate(zip(evals, methods)):
        if e.error is not None:
            continue
        pts.append({"id": k, "method": m, "feasible": e.feasible, "pareto": id(e) in front_ids,
                    "objectives": {n: e.objective(n) for n in names},
                    "min_p95_margin_pct": e.min_p95_margin_pct, "design": to_dict(e.params, cfg)})
    return {"name": name, "objectives": names, "margin_target_pct": cfg["margin_target_pct"],
            "torso_length_in": cfg["torso_length_in"], "projectile": cfg["projectile"],
            "running": _running(name), "points": pts}


def _running(name: str) -> bool:
    with _lock:
        p = _procs.get(name)
        return p is not None and p.poll() is None


@app.post("/api/optimize")
def optimize(req: OptimizeIn) -> dict:
    name = _run_name(req.name)
    with _lock:
        if name in _procs and _procs[name].poll() is None:
            raise HTTPException(409, f"run '{name}' already running")
        (RUNS / name).mkdir(parents=True, exist_ok=True)
        log = open(RUNS / name / "log.txt", "a")
        _procs[name] = subprocess.Popen(
            [sys.executable, "-m", "chestforge.optimize.optimizer", "--name", name, "--method", req.method,
             "--trials", str(req.trials), "--rounds", str(req.rounds)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return {"started": name}


@app.get("/api/optimize/{name}/log")
def optimize_log(name: str) -> dict:
    _run_name(name)
    f = RUNS / name / "log.txt"
    return {"running": _running(name), "log": f.read_text()[-4000:] if f.exists() else ""}


# ---------- static UI ----------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")


def main(port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn
    import webbrowser
    url = f"http://127.0.0.1:{port}/"
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"{SIM_LABEL}\nChestForge UI at {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
