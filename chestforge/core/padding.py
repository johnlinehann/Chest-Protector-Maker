"""Padding layers: rate-dependent foam constitutive law and ball-on-pad force.

Milestone 2 models one homogeneous foam layer under the ball. The foam force is
    F = s(e, e_dot) * A(d)
with e = d/h the compressive strain under the ball centre and A(d) the area of a sphere
indenting to depth d (capped at the ball cross-section).

Load spreading (axisymmetric, uniform pressure): the force leaves the back of the pad over a
disc of radius a_c(d) + (h - d) tan(angle). The share falling on the cardiac silhouette
(area-equivalent circle) loads the load cell; the rest goes into the rigid surrogate body.
Plates and a proper 2D pressure distribution replace this in Milestone 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from chestforge.config import CONFIG_DIR

import yaml

E_MAX = 0.995          # strain clamp: densification term diverges at e -> 1
RATE_SMOOTH = 1.0      # 1/s, width of the load/unload blend in tanh(e_dot / RATE_SMOOTH)


@dataclass(frozen=True)
class FoamMaterial:
    id: str
    name: str
    density_kg_m3: float
    E_Pa: float
    A_Pa: float
    m: float
    B_Pa: float
    n: float
    densification_strain: float
    rate_C: float
    ref_strain_rate: float
    unloading_ratio: float
    load_spread_angle_deg: float
    estimated: bool

    def quasi_static_stress(self, e: float) -> float:
        """Avalle et al. (2001) compressive curve, Pa. Linear (E e) in tension."""
        if e <= 0.0:
            return self.E_Pa * e
        e = min(e, E_MAX)
        plateau = self.A_Pa * (1.0 - math.exp(-(self.E_Pa / self.A_Pa) * e * (1.0 - e) ** self.m))
        return plateau + self.B_Pa * (e / (1.0 - e)) ** self.n

    def stress(self, e: float, e_dot: float) -> float:
        """Rate-dependent stress with smooth hysteretic unloading, Pa (compression positive)."""
        s = self.quasi_static_stress(e)
        if e <= 0.0:
            return s
        if e_dot > self.ref_strain_rate:
            s *= 1.0 + self.rate_C * math.log(e_dot / self.ref_strain_rate)
        r = self.unloading_ratio
        return s * (0.5 * (1 + r) + 0.5 * (1 - r) * math.tanh(e_dot / RATE_SMOOTH))


@dataclass(frozen=True)
class PadLayer:
    material: FoamMaterial
    thickness_m: float

    def __post_init__(self):
        if self.thickness_m <= 0:
            raise ValueError("pad thickness must be positive")

    @property
    def areal_mass_kg_m2(self) -> float:
        return self.material.density_kg_m3 * self.thickness_m

    def footprint_mass_kg(self, ball_radius_m: float) -> float:
        """Pad mass under the ball's cross-section."""
        return self.areal_mass_kg_m2 * math.pi * ball_radius_m**2

    def force(self, d: float, d_dot: float, ball_radius_m: float) -> float:
        """Foam force for compression d (m) at rate d_dot (m/s), N."""
        h = self.thickness_m
        e, e_dot = d / h, d_dot / h
        if d <= 0.0:  # tension: bonded face, area of the footprint
            return self.material.stress(e, e_dot) * math.pi * ball_radius_m**2
        return self.material.stress(e, e_dot) * math.pi * contact_radius(d, ball_radius_m) ** 2

    def load_cell_fraction(self, d: float, ball_radius_m: float, silhouette_radius_m: float) -> float:
        """Share of the transmitted force that lands on the cardiac silhouette (centred impact)."""
        remaining = max(self.thickness_m - max(d, 0.0), 0.0)
        a = contact_radius(d, ball_radius_m) + remaining * math.tan(math.radians(self.material.load_spread_angle_deg))
        return 1.0 if a <= silhouette_radius_m else (silhouette_radius_m / a) ** 2


def contact_radius(d: float, ball_radius_m: float) -> float:
    """Radius of a sphere's footprint at indentation depth d (capped at the ball radius)."""
    a = min(max(d, 0.0), ball_radius_m)
    return math.sqrt(2.0 * ball_radius_m * a - a * a)


def _val(entry: dict) -> float:
    return entry["value"]


def load_materials() -> dict[str, FoamMaterial]:
    with open(CONFIG_DIR / "materials.yaml") as f:
        raw = yaml.safe_load(f)["materials"]
    out = {}
    for m in raw:
        if m["model"] != "avalle_jc":
            raise ValueError(f"material {m['id']}: unsupported model {m['model']!r}")
        fields = ("density_kg_m3", "E_MPa", "A_MPa", "m", "B_MPa", "n", "densification_strain",
                  "rate_C", "ref_strain_rate", "unloading_ratio", "load_spread_angle_deg")
        out[m["id"]] = FoamMaterial(
            id=m["id"],
            name=m["name"],
            density_kg_m3=_val(m["density_kg_m3"]),
            E_Pa=_val(m["E_MPa"]) * 1e6,
            A_Pa=_val(m["A_MPa"]) * 1e6,
            m=_val(m["m"]),
            B_Pa=_val(m["B_MPa"]) * 1e6,
            n=_val(m["n"]),
            densification_strain=_val(m["densification_strain"]),
            rate_C=_val(m["rate_C"]),
            ref_strain_rate=_val(m["ref_strain_rate"]),
            unloading_ratio=_val(m["unloading_ratio"]),
            load_spread_angle_deg=_val(m["load_spread_angle_deg"]),
            estimated=any(m[k].get("estimated", False) for k in fields),
        )
    return out


def load_material(material_id: str) -> FoamMaterial:
    mats = load_materials()
    if material_id not in mats:
        raise KeyError(f"unknown material {material_id!r}; available: {sorted(mats)}")
    return mats[material_id]
