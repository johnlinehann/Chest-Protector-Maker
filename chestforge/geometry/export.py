"""Export a protector design: STEP assembly, STL per layer + combined, nTop lattice parameters,
viewer landmarks (silhouette, shift radius) and a summary with CAD vs. model mass.

Run:
  python -m chestforge.geometry.export --design designs/example_plate_foam.yaml --torso-in 15
  python -m chestforge.geometry.export --run m4 --rank 1
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from chestforge import SIM_LABEL
from chestforge.config import load_nd200
from chestforge.core.layout import Protector, load_design
from chestforge.core.padding import FoamLayer, FoamMaterial
from chestforge.geometry.protector_gen import IN, BodyFrame, LayerSolid, build_layers, load_geometry_config

ROOT = Path(__file__).resolve().parent.parent.parent


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60] or "design"


def lattice_suggestion(layer: FoamLayer) -> dict:
    """TPU lattice matching the foam's quasi-static plateau (Gibson & Ashby scaling). Not simulated."""
    g = load_geometry_config()["lattice"]
    sigma_s = g["solid"]["yield_strength_MPa"]["value"] * 1e6
    rho_s = g["solid"]["density_kg_m3"]["value"]
    m: FoamMaterial = layer.material
    t_mm = layer.thickness_m * 1000
    cs = g["cell_size_mm"]
    cells = {}
    for name, c in g["cells"].items():
        rel = (m.A_Pa / (c["C"] * sigma_s)) ** (1.0 / c["k"])
        cells[name] = {
            "relative_density": round(rel, 4),
            "lattice_density_kg_m3": round(rel * rho_s, 1),
            "mass_ratio_vs_foam": round(rel * rho_s / m.density_kg_m3, 2),
            "feasible": 0.05 <= rel <= 0.6,
        }
    return {
        "replaces": m.id,
        "target_plateau_MPa": round(m.A_Pa / 1e6, 4),
        "target_densification_strain": m.densification_strain,
        "thickness_mm": t_mm,
        "cell_size_mm": round(min(max(t_mm / 2, cs["min"]), cs["max"]), 2),
        "cells": cells,
        "estimated": True,
        "note": "Quasi-static match only; rate dependence and the ND200 response of the lattice are NOT simulated.",
    }


def landmarks(frame: BodyFrame, max_shift_in: float, n: int = 72) -> dict:
    g = load_nd200().raw["surrogate"]["geometry"]
    a, b = 0.5 * g["silhouette_width_in"]["value"], 0.5 * g["silhouette_height_in"]["value"]
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pt = lambda x, y: [round(v, 3) for v in frame.surface_point(float(x), float(y))]
    return {
        "units": "mm", "frame": "origin jugular notch on midline; +X wearer's left, +Y up, +Z out of chest",
        "jugular_notch": [0.0, 0.0, 0.0],
        "silhouette_center": pt(0, 0),
        "silhouette_outline": [pt(a * math.cos(u), b * math.sin(u)) for u in t],
        "shift_radius_in": max_shift_in,
        "shift_circle": [pt(max_shift_in * math.cos(u), max_shift_in * math.sin(u)) for u in t],
        "chest_radius_mm": round(frame.radius_mm, 3),
    }


def export_design(prot: Protector, torso_length_in: float, out: Path, model_mass_g: float | None = None) -> dict:
    from build123d import Compound, export_step, export_stl

    cfg = load_geometry_config()["export"]
    out.mkdir(parents=True, exist_ok=True)
    layers: list[LayerSolid] = build_layers(prot, torso_length_in)
    parts = []
    for ls in layers:
        ls.solid.label = ls.name
        parts.append(ls.solid)
        export_stl(ls.solid, str(out / f"{ls.name}.stl"), tolerance=cfg["stl_tolerance_mm"],
                   angular_tolerance=math.radians(cfg["stl_angular_tolerance_deg"]))
    assembly = Compound(label=slug(prot.name), children=parts)
    export_step(assembly, str(out / "protector.step"))
    export_stl(assembly, str(out / "protector.stl"), tolerance=cfg["stl_tolerance_mm"],
               angular_tolerance=math.radians(cfg["stl_angular_tolerance_deg"]))

    lattice = {ls.name: lattice_suggestion(ls.layer) for ls in layers if isinstance(ls.layer, FoamLayer)}
    (out / "ntop_lattice.json").write_text(json.dumps({"label": SIM_LABEL, "step_file": "protector.step",
                                                       "bodies": lattice}, indent=1))
    frame = BodyFrame.for_torso(torso_length_in)
    max_shift = load_nd200().max_shift_in(torso_length_in)
    (out / "landmarks.json").write_text(json.dumps(landmarks(frame, max_shift), indent=1))

    cad_mass = sum(ls.mass_g for ls in layers)
    summary = {
        "label": SIM_LABEL, "design": prot.name, "torso_length_in": torso_length_in,
        "files": ["protector.step", "protector.stl"] + [f"{ls.name}.stl" for ls in layers]
                 + ["ntop_lattice.json", "landmarks.json"],
        "layers": [{"name": ls.name, "zone": ls.zone, "material": ls.layer.material.id,
                    "thickness_mm": round(ls.layer.thickness_m * 1000, 3),
                    "radial_mm": [round(ls.r0_mm, 3), round(ls.r1_mm, 3)],
                    "volume_cm3": round(ls.volume_mm3 / 1e3, 2), "mass_g": round(ls.mass_g, 1),
                    "valid": bool(ls.solid.is_valid)} for ls in layers],
        "cad_mass_g": round(cad_mass, 1),
        "model_mass_g": None if model_mass_g is None else round(model_mass_g, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export protector geometry (STEP/STL/nTop).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--design", help="protector YAML")
    g.add_argument("--run", help="optimizer run name (uses runs/<run>/top5/design_<rank>.yaml)")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--torso-in", type=float, default=None, help="default: optimizer config torso length")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from chestforge.optimize.search_space import load_optimizer_config
    torso = args.torso_in or load_optimizer_config()["torso_length_in"]
    path = Path(args.design) if args.design else ROOT / "runs" / args.run / "top5" / f"design_{args.rank}.yaml"
    prot = load_design(path)
    out = Path(args.out) if args.out else ROOT / "exports" / slug(prot.name)
    s = export_design(prot, torso, out)
    print(f"{SIM_LABEL}\nExported '{prot.name}' (torso {torso} in) -> {out}")
    for l in s["layers"]:
        print(f"  {l['name']:34s} {l['thickness_mm']:5.1f} mm  {l['volume_cm3']:8.1f} cm3  {l['mass_g']:6.1f} g")
    print(f"  CAD mass {s['cad_mass_g']} g")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
