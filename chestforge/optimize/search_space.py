"""Design parameterisation: cardiac-zone stack + plate shape <-> Protector, Optuna and ML features."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import yaml

from chestforge.config import CONFIG_DIR, load_nd200
from chestforge.core.layout import Protector, Zone
from chestforge.core.padding import Ellipse, layer

NONE = "none"


@lru_cache
def load_optimizer_config() -> dict:
    with open(CONFIG_DIR / "optimizer.yaml") as f:
        return yaml.safe_load(f)


def _grid(spec) -> np.ndarray:
    lo, hi, step = spec
    return np.round(np.arange(lo, hi + step / 2, step), 6)


def suggest(trial, cfg: dict) -> dict:
    """Sample one design with an Optuna trial (define-by-run)."""
    sp = cfg["search_space"]
    p = {
        "plate_material": trial.suggest_categorical("plate_material", sp["plate"]["materials"]),
        "plate_mm": trial.suggest_float("plate_mm", *sp["plate"]["thickness_mm"][:2], step=sp["plate"]["thickness_mm"][2]),
        "plate_a_in": trial.suggest_float("plate_a_in", *sp["plate"]["half_width_in"][:2], step=sp["plate"]["half_width_in"][2]),
        "plate_b_in": trial.suggest_float("plate_b_in", *sp["plate"]["half_height_in"][:2], step=sp["plate"]["half_height_in"][2]),
        "outer_material": trial.suggest_categorical("outer_material", sp["outer_foam"]["materials"]),
        "outer_mm": trial.suggest_float("outer_mm", *sp["outer_foam"]["thickness_mm"][:2], step=sp["outer_foam"]["thickness_mm"][2]),
        "inner_material": trial.suggest_categorical("inner_material", sp["inner_foam"]["materials"]),
        "inner_mm": trial.suggest_float("inner_mm", *sp["inner_foam"]["thickness_mm"][:2], step=sp["inner_foam"]["thickness_mm"][2]),
    }
    return canonical(p)


def canonical(p: dict) -> dict:
    """Normalise so equivalent designs compare equal (no outer foam -> thickness 0)."""
    p = dict(p)
    for k in ("plate_material", "outer_material", "inner_material"):
        p[k] = str(p[k])                     # numpy str_ from rng.choice -> plain str (YAML/JSON safe)
    if p["outer_material"] == NONE:
        p["outer_mm"] = 0.0
    for k in ("plate_mm", "plate_a_in", "plate_b_in", "outer_mm", "inner_mm"):
        p[k] = round(float(p[k]), 4)
    return p


def random_design(rng: np.random.Generator, cfg: dict) -> dict:
    sp = cfg["search_space"]
    return canonical({
        "plate_material": rng.choice(sp["plate"]["materials"]),
        "plate_mm": rng.choice(_grid(sp["plate"]["thickness_mm"])),
        "plate_a_in": rng.choice(_grid(sp["plate"]["half_width_in"])),
        "plate_b_in": rng.choice(_grid(sp["plate"]["half_height_in"])),
        "outer_material": rng.choice(sp["outer_foam"]["materials"]),
        "outer_mm": rng.choice(_grid(sp["outer_foam"]["thickness_mm"])),
        "inner_material": rng.choice(sp["inner_foam"]["materials"]),
        "inner_mm": rng.choice(_grid(sp["inner_foam"]["thickness_mm"])),
    })


def mutate(p: dict, rng: np.random.Generator, cfg: dict, n_changes: int = 2) -> dict:
    """Change a few parameters to neighbouring values (local search around good designs)."""
    sp = cfg["search_space"]
    grids = {
        "plate_mm": _grid(sp["plate"]["thickness_mm"]), "plate_a_in": _grid(sp["plate"]["half_width_in"]),
        "plate_b_in": _grid(sp["plate"]["half_height_in"]), "outer_mm": _grid(sp["outer_foam"]["thickness_mm"]),
        "inner_mm": _grid(sp["inner_foam"]["thickness_mm"]),
    }
    cats = {"plate_material": sp["plate"]["materials"], "outer_material": sp["outer_foam"]["materials"],
            "inner_material": sp["inner_foam"]["materials"]}
    q = dict(p)
    keys = list(grids) + list(cats)
    for k in rng.choice(keys, size=n_changes, replace=False):
        if k in cats:
            q[k] = rng.choice(cats[k])
        else:
            g = grids[k]
            i = int(np.argmin(np.abs(g - (q[k] if q[k] > 0 else g[0]))))
            q[k] = g[int(np.clip(i + rng.choice([-2, -1, 1, 2]), 0, len(g) - 1))]
    if q["outer_material"] != NONE and q["outer_mm"] <= 0:   # outer foam switched on: give it a thickness
        q["outer_mm"] = rng.choice(grids["outer_mm"])
    return canonical(q)


def cardiac_zone_axes(p: dict, cfg: dict) -> tuple[float, float]:
    """Cardiac zone half-axes: covers the plate and the silhouette swept by the max shift."""
    nd = load_nd200()
    g = nd.raw["surrogate"]["geometry"]
    shift = nd.max_shift_in(cfg["torso_length_in"])
    clr = cfg["search_space"]["zone_clearance_in"]
    a = max(p["plate_a_in"], 0.5 * g["silhouette_width_in"]["value"] + shift) + clr
    b = max(p["plate_b_in"], 0.5 * g["silhouette_height_in"]["value"] + shift) + clr
    return a, b


def cardiac_stack(p: dict) -> tuple:
    stack = []
    if p["outer_material"] != NONE:
        stack.append(layer(p["outer_material"], p["outer_mm"]))
    stack.append(layer(p["plate_material"], p["plate_mm"], Ellipse(0, 0, p["plate_a_in"], p["plate_b_in"])))
    stack.append(layer(p["inner_material"], p["inner_mm"]))
    return tuple(stack)


def torso_zone(cfg: dict) -> Zone:
    t = cfg["torso_zone"]
    o = t["outline"]
    return Zone("torso", Ellipse(o.get("cx", 0), o.get("cy", 0), o["a"], o["b"]),
                tuple(layer(l["material"], l["thickness_mm"]) for l in t["stack"]))


def to_protector(p: dict, cfg: dict) -> Protector:
    a, b = cardiac_zone_axes(p, cfg)
    return Protector(describe(p), (Zone("cardiac", Ellipse(0, 0, a, b), cardiac_stack(p)), torso_zone(cfg)))


def describe(p: dict) -> str:
    outer = f"{p['outer_material']} {p['outer_mm']:g}mm / " if p["outer_material"] != NONE else ""
    return (f"{outer}{p['plate_material']} {p['plate_mm']:g}mm plate "
            f"({2 * p['plate_a_in']:g}x{2 * p['plate_b_in']:g} in) / {p['inner_material']} {p['inner_mm']:g}mm")


def to_dict(p: dict, cfg: dict) -> dict:
    """Protector YAML dict (loadable by chestforge.core.layout.protector_from_dict)."""
    a, b = cardiac_zone_axes(p, cfg)
    cardiac = []
    if p["outer_material"] != NONE:
        cardiac.append({"material": p["outer_material"], "thickness_mm": p["outer_mm"]})
    cardiac.append({"material": p["plate_material"], "thickness_mm": p["plate_mm"],
                    "extent": {"cx": 0.0, "cy": 0.0, "a": p["plate_a_in"], "b": p["plate_b_in"]}})
    cardiac.append({"material": p["inner_material"], "thickness_mm": p["inner_mm"]})
    return {"name": describe(p), "zones": [
        {"name": "cardiac", "outline": {"cx": 0.0, "cy": 0.0, "a": round(a, 4), "b": round(b, 4)}, "stack": cardiac},
        {"name": "torso", **cfg["torso_zone"]},
    ]}


def features(p: dict, cfg: dict) -> np.ndarray:
    """Numeric feature vector for the ML surrogate (one-hot materials + numerics)."""
    sp = cfg["search_space"]
    f = [p["plate_mm"], p["plate_a_in"], p["plate_b_in"], p["outer_mm"], p["inner_mm"],
         min(p["plate_a_in"], p["plate_b_in"]), p["plate_mm"] ** 3]
    for key, mats in (("plate_material", sp["plate"]["materials"]), ("outer_material", sp["outer_foam"]["materials"]),
                      ("inner_material", sp["inner_foam"]["materials"])):
        f += [1.0 if p[key] == m else 0.0 for m in mats]
    return np.array(f, dtype=float)


def ellipse_area_m2(a_in: float, b_in: float) -> float:
    return math.pi * a_in * b_in * 0.0254**2
