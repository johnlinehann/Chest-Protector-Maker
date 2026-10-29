"""Parametric protector solids (build123d / OpenCascade) from a Protector layout and torso length.

Each layer becomes a curved slab: the layer's outline (wrapped onto the chest by arc length)
extruded radially and intersected with a cylindrical band [R + r0, R + r1] around the chest.
Layers stack outward from the body in reverse stack order (stacks are listed ball side first).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import yaml

from chestforge.config import CONFIG_DIR
from chestforge.core.layout import FAR_IN, Protector
from chestforge.core.padding import Ellipse, FoamLayer, PlateLayer

IN = 25.4


@lru_cache
def load_geometry_config() -> dict:
    with open(CONFIG_DIR / "geometry.yaml") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class BodyFrame:
    """Maps protector coordinates (in, origin at silhouette centre) onto the chest cylinder (mm)."""
    radius_mm: float
    sil_arc_mm: float        # arc position of the silhouette centre from the midline
    sil_y_mm: float          # height of the silhouette centre (negative: below the notch)

    @classmethod
    def for_torso(cls, torso_length_in: float) -> "BodyFrame":
        b = load_geometry_config()["body"]
        scale = torso_length_in / b["reference_torso_length_in"]["value"] if b["scale_with_torso_length"] else 1.0
        R = b["chest_radius_in"]["value"] * scale * IN
        x_sil = b["silhouette_center_from_midline_in"]["value"] * scale * IN
        return cls(R, R * math.asin(min(x_sil / R, 1.0)), -b["silhouette_center_below_notch_in"]["value"] * scale * IN)

    def to_flat(self, x_in: np.ndarray, y_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Protector (x, y) -> body (X, Y) before radial projection; arc length is preserved."""
        s = np.clip(x_in * IN + self.sil_arc_mm, -0.49 * math.pi * self.radius_mm, 0.49 * math.pi * self.radius_mm)
        return self.radius_mm * np.sin(s / self.radius_mm), y_in * IN + self.sil_y_mm

    def surface_point(self, x_in: float, y_in: float, height_mm: float = 0.0) -> tuple[float, float, float]:
        """3D point on the chest (or height_mm above it) for a protector-frame location."""
        s = x_in * IN + self.sil_arc_mm
        r = self.radius_mm + height_mm
        return r * math.sin(s / self.radius_mm), y_in * IN + self.sil_y_mm, r * math.cos(s / self.radius_mm) - self.radius_mm


@dataclass
class LayerSolid:
    name: str               # e.g. "cardiac_L2_pc_plate"
    zone: str
    layer: object           # FoamLayer | PlateLayer
    solid: object           # build123d Part
    r0_mm: float            # radial start above the chest surface
    r1_mm: float

    @property
    def volume_mm3(self) -> float:
        return float(self.solid.volume)

    @property
    def mass_g(self) -> float:
        return self.volume_mm3 * 1e-9 * self.layer.material.density_kg_m3 * 1000


def _ellipse_points(e: Ellipse, n: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return e.cx + e.a * np.cos(t), e.cy + e.b * np.sin(t)


def _prism(e: Ellipse, frame: BodyFrame, n: int):
    from build123d import Face, Plane, Vector, Wire, extrude
    X, Y = frame.to_flat(*_ellipse_points(e, n))
    z_top = 300.0
    wire = Wire.make_polygon([Vector(float(x), float(y), z_top) for x, y in zip(X, Y)], close=True)
    return extrude(Face(wire), amount=z_top + frame.radius_mm, dir=(0, 0, -1))   # down to z = -R: whole front half


def _band(frame: BodyFrame, r0: float, r1: float, height_mm: float = 2000.0):
    from build123d import Align, Cylinder, Pos, Rot
    R = frame.radius_mm
    outer = Pos(0, 0, -R) * Rot(90, 0, 0) * Cylinder(R + r1, height_mm, align=(Align.CENTER,) * 3)
    inner = Pos(0, 0, -R) * Rot(90, 0, 0) * Cylinder(R + r0, height_mm, align=(Align.CENTER,) * 3)
    return outer - inner


def build_layers(prot: Protector, torso_length_in: float) -> list[LayerSolid]:
    """Solids for every layer of every zone (zones earlier in the list cut later ones)."""
    n = load_geometry_config()["export"]["outline_points"]
    frame = BodyFrame.for_torso(torso_length_in)
    out: list[LayerSolid] = []
    claimed = None                                  # union of higher-priority zone prisms
    for z in prot.zones:
        if z.outline.a >= FAR_IN:
            raise ValueError(f"zone '{z.name}' is unbounded; geometry needs finite outlines")
        zone_prism = _prism(z.outline, frame, n)
        region = zone_prism if claimed is None else zone_prism - claimed
        r = 0.0
        for k, lay in enumerate(reversed(z.stack)):  # body side first
            t = lay.thickness_m * 1000
            footprint = region
            if isinstance(lay, PlateLayer) and lay.extent is not None:
                footprint = _prism(lay.extent, frame, n) & region
            solid = footprint & _band(frame, r, r + t)
            idx = len(z.stack) - k
            kind = "plate" if isinstance(lay, PlateLayer) else "foam"
            out.append(LayerSolid(f"{z.name}_L{idx}_{lay.material.id}_{kind}", z.name, lay, solid, r, r + t))
            r += t
        claimed = zone_prism if claimed is None else claimed + zone_prism
    return out
