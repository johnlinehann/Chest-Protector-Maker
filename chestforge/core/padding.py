"""Padding materials and layer stacks. Physics lives in kernels.py; this module holds the data.

A stack is a tuple of layers ordered from the ball side to the body side.
"""

from __future__ import annotations

import math
from functools import lru_cache
from dataclasses import dataclass
from typing import Union

import numpy as np
import yaml

from chestforge.config import CONFIG_DIR
from chestforge.core import kernels


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
    temp_coeff_per_F: float
    estimated: bool

    def params(self) -> np.ndarray:
        """Row in the kernel's foam-parameter layout."""
        p = np.zeros(kernels.N_FOAM_PAR)
        p[kernels.F_E], p[kernels.F_A], p[kernels.F_M] = self.E_Pa, self.A_Pa, self.m
        p[kernels.F_B], p[kernels.F_N], p[kernels.F_C] = self.B_Pa, self.n, self.rate_C
        p[kernels.F_REF], p[kernels.F_UNL] = self.ref_strain_rate, self.unloading_ratio
        p[kernels.F_TAN] = math.tan(math.radians(self.load_spread_angle_deg))
        return p

    def temp_factor(self, temp_F: float) -> float:
        return 1.0 - self.temp_coeff_per_F * (temp_F - 72.0)

    def quasi_static_stress(self, e: float) -> float:
        return kernels.foam_qs(e, self.E_Pa, self.A_Pa, self.m, self.B_Pa, self.n)

    def stress(self, e: float, e_dot: float, temp_F: float = 72.0) -> float:
        return kernels.foam_stress(e, e_dot, self.params(), self.temp_factor(temp_F))


@dataclass(frozen=True)
class PlateMaterial:
    id: str
    name: str
    density_kg_m3: float
    E_Pa: float
    poisson: float
    estimated: bool


@dataclass(frozen=True)
class Ellipse:
    """Axis-aligned ellipse in protector coordinates, inches. Origin = point fitted over the
    cardiac-silhouette centre (ND200 §11.1); +x toward the wearer's left, +y up."""
    cx: float
    cy: float
    a: float      # half-width, in
    b: float      # half-height, in

    def contains(self, x: float, y: float) -> bool:
        return ((x - self.cx) / self.a) ** 2 + ((y - self.cy) / self.b) ** 2 <= 1.0

    def edge_distance(self, x: float, y: float, n: int = 720) -> float:
        """Signed distance (in) from a point to the boundary: positive inside."""
        t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        d = float(np.min(np.hypot(self.cx + self.a * np.cos(t) - x, self.cy + self.b * np.sin(t) - y)))
        return d if self.contains(x, y) else -d


@dataclass(frozen=True)
class FoamLayer:
    material: FoamMaterial
    thickness_m: float

    def __post_init__(self):
        if self.thickness_m <= 0:
            raise ValueError("layer thickness must be positive")

    @property
    def areal_mass_kg_m2(self) -> float:
        return self.material.density_kg_m3 * self.thickness_m

    def footprint_mass_kg(self, ball_radius_m: float) -> float:
        """Layer mass under the ball's cross-section."""
        return self.areal_mass_kg_m2 * math.pi * ball_radius_m**2

    def describe(self) -> str:
        return f"{self.material.id} {self.thickness_m * 1000:.1f}mm"


PadLayer = FoamLayer  # Milestone 2 name


@dataclass(frozen=True)
class PlateLayer:
    material: PlateMaterial
    thickness_m: float
    extent: Ellipse | None = None    # None: plate fills its zone

    def __post_init__(self):
        if self.thickness_m <= 0:
            raise ValueError("layer thickness must be positive")

    @property
    def areal_mass_kg_m2(self) -> float:
        return self.material.density_kg_m3 * self.thickness_m

    @property
    def bending_stiffness(self) -> float:
        """D = E t^3 / (12 (1 - nu^2)), N m."""
        m = self.material
        return m.E_Pa * self.thickness_m**3 / (12.0 * (1.0 - m.poisson**2))

    def describe(self) -> str:
        return f"{self.material.id} plate {self.thickness_m * 1000:.1f}mm"


Layer = Union[FoamLayer, PlateLayer]
Stack = tuple  # tuple[Layer, ...], ball side first


def stack_areal_mass(stack: Stack) -> float:
    return sum(l.areal_mass_kg_m2 for l in stack)


def stack_thickness_m(stack: Stack) -> float:
    return sum(l.thickness_m for l in stack)


def describe_stack(stack: Stack) -> str:
    return " / ".join(l.describe() for l in stack) if stack else "bare"


def _val(entry: dict) -> float:
    return entry["value"]


def _load_raw() -> list[dict]:
    """materials.yaml plus fitted materials (config/materials_fitted.yaml, written by chestforge.validation)."""
    with open(CONFIG_DIR / "materials.yaml") as f:
        mats = yaml.safe_load(f)["materials"]
    fitted = CONFIG_DIR / "materials_fitted.yaml"
    if fitted.exists():
        extra = (yaml.safe_load(fitted.read_text()) or {}).get("materials", [])
        clash = {m["id"] for m in mats} & {m["id"] for m in extra}
        if clash:
            raise ValueError(f"materials_fitted.yaml redefines {sorted(clash)}; fitted materials need new ids")
        mats = mats + extra
    return mats


def load_materials() -> dict[str, FoamMaterial | PlateMaterial]:
    return dict(_materials())


@lru_cache
def _materials() -> dict[str, FoamMaterial | PlateMaterial]:
    """Parsed once per process (materials are frozen dataclasses)."""
    out = {}
    for m in _load_raw():
        est = any(isinstance(v, dict) and v.get("estimated", False) for v in m.values())
        if m["model"] == "avalle_jc":
            out[m["id"]] = FoamMaterial(
                id=m["id"], name=m["name"],
                density_kg_m3=_val(m["density_kg_m3"]),
                E_Pa=_val(m["E_MPa"]) * 1e6, A_Pa=_val(m["A_MPa"]) * 1e6, m=_val(m["m"]),
                B_Pa=_val(m["B_MPa"]) * 1e6, n=_val(m["n"]),
                densification_strain=_val(m["densification_strain"]),
                rate_C=_val(m["rate_C"]), ref_strain_rate=_val(m["ref_strain_rate"]),
                unloading_ratio=_val(m["unloading_ratio"]),
                load_spread_angle_deg=_val(m["load_spread_angle_deg"]),
                temp_coeff_per_F=_val(m["temp_coeff_per_F"]),
                estimated=est,
            )
        elif m["model"] == "elastic_plate":
            out[m["id"]] = PlateMaterial(
                id=m["id"], name=m["name"], density_kg_m3=_val(m["density_kg_m3"]),
                E_Pa=_val(m["E_MPa"]) * 1e6, poisson=_val(m["poisson"]), estimated=est,
            )
        else:
            raise ValueError(f"material {m['id']}: unsupported model {m['model']!r}")
    return out


def load_material(material_id: str) -> FoamMaterial | PlateMaterial:
    mats = _materials()
    if material_id not in mats:
        raise KeyError(f"unknown material {material_id!r}; available: {sorted(mats)}")
    return mats[material_id]


def layer(material_id: str, thickness_mm: float, extent: Ellipse | None = None) -> Layer:
    """Convenience constructor: foam or plate depending on the material."""
    m = load_material(material_id)
    if isinstance(m, PlateMaterial):
        return PlateLayer(m, thickness_mm / 1000, extent)
    if extent is not None:
        raise ValueError("foam layers fill their zone; extent applies to plates only")
    return FoamLayer(m, thickness_mm / 1000)
