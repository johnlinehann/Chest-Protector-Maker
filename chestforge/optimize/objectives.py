"""Evaluate a design: ND200 at P95 (constraint) + objectives (cardiac load, mass, thickness, stiffness)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from chestforge.core.nd200_runner import ND200Report, run_nd200
from chestforge.core.padding import FoamLayer, PlateLayer
from chestforge.optimize.search_space import (cardiac_stack, cardiac_zone_axes, ellipse_area_m2, to_protector,
                                              torso_zone)

OBJECTIVE_UNITS = {"p95_50mph_lbf": "lbf", "mass_g": "g", "thickness_mm": "mm", "bending_Nm": "N·m"}


@dataclass
class Evaluation:
    params: dict
    p95_50mph_lbf: float        # worst P95 of the three 50 mph impacts
    p95_30mph_lbf: float
    min_p95_margin_pct: float   # worst of all six impacts
    min_nominal_margin_pct: float
    mass_g: float               # whole protector (cardiac zone + torso zone)
    thickness_mm: float         # cardiac-zone stack
    bending_Nm: float           # cardiac-zone stack bending stiffness (mobility proxy)
    feasible: bool              # all six impacts pass at P95 with >= margin target
    bottomed: bool
    flags: list
    error: str | None = None

    def objective(self, name: str) -> float:
        return float(getattr(self, name))

    def to_json(self) -> dict:
        return asdict(self)


def protector_mass_g(p: dict, cfg: dict) -> float:
    """Plate over its own ellipse, foams over the cardiac zone, torso stack over the rest of its outline."""
    a, b = cardiac_zone_axes(p, cfg)
    zone_area = ellipse_area_m2(a, b)
    m = 0.0
    for lay in cardiac_stack(p):
        area = ellipse_area_m2(lay.extent.a, lay.extent.b) if isinstance(lay, PlateLayer) else zone_area
        m += lay.areal_mass_kg_m2 * area
    tz = torso_zone(cfg)
    rest = max(ellipse_area_m2(tz.outline.a, tz.outline.b) - zone_area, 0.0)
    m += sum(l.areal_mass_kg_m2 for l in tz.stack) * rest
    return m * 1000.0


def stack_bending_Nm(stack) -> float:
    """Sum of layer bending stiffnesses D (unbonded layers -> lower bound of the laminate), N·m."""
    d = 0.0
    for lay in stack:
        if isinstance(lay, PlateLayer):
            d += lay.bending_stiffness
        elif isinstance(lay, FoamLayer):
            d += lay.material.E_Pa * lay.thickness_m**3 / 12.0
    return d


def evaluate(p: dict, cfg: dict, mc_samples: int | None = None) -> tuple[Evaluation, ND200Report | None]:
    stack = cardiac_stack(p)
    common = dict(
        params=p,
        mass_g=protector_mass_g(p, cfg),
        thickness_mm=sum(l.thickness_m for l in stack) * 1000.0,
        bending_Nm=stack_bending_Nm(stack),
    )
    try:
        rep = run_nd200(
            to_protector(p, cfg), cfg["projectile"], cfg["torso_length_in"],
            mc_samples=mc_samples if mc_samples is not None else cfg["mc_samples"], seed=cfg["seed"],
            ring_step_in=cfg["shift_search"]["ring_step_in"], n_angles=cfg["shift_search"]["n_angles"],
        )
    except RuntimeError as e:   # integration failure: record, never crash the study
        inf = float("inf")
        return Evaluation(p95_50mph_lbf=inf, p95_30mph_lbf=inf, min_p95_margin_pct=-inf,
                          min_nominal_margin_pct=-inf, feasible=False, bottomed=False, flags=[],
                          error=str(e), **common), None
    s50 = [i for i in rep.impacts if i.condition == "50mph"]
    s30 = [i for i in rep.impacts if i.condition == "30mph"]
    ev = Evaluation(
        p95_50mph_lbf=max(i.p95_lbf for i in s50),
        p95_30mph_lbf=max(i.p95_lbf for i in s30),
        min_p95_margin_pct=rep.min_p95_margin_pct,
        min_nominal_margin_pct=rep.min_margin_pct,
        feasible=rep.min_p95_margin_pct >= cfg["margin_target_pct"] and rep.covers_silhouette
        and not rep.beyond_standard,
        bottomed=any(i.bottomed_out for i in rep.impacts),
        flags=rep.flags,
        **common,
    )
    return ev, rep
