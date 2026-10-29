"""Protector layout: pad zones and plate extents, and what stack sits under the impact point.

Coordinates: inches, protector frame. Origin = the protector point fitted over the cardiac-
silhouette centre (ND200 §11.1); +x toward the wearer's left, +y up. Shifting the protector by
s puts the (fixed) impact point at -s in this frame (§12.2: the ball always hits the silhouette
centre, the protector moves).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from chestforge.config import load_nd200
from chestforge.core.padding import Ellipse, PlateLayer, Stack, describe_stack, layer
from chestforge.units import IN_TO_M

FAR_IN = 1.0e4
REACH_FAR_M = 0.3   # plate reach beyond this never caps spreading -> treated as unbounded


@dataclass(frozen=True)
class Zone:
    name: str
    outline: Ellipse
    stack: Stack


@dataclass(frozen=True)
class LocalStack:
    zone: str | None
    layers: Stack
    plate_reach_m: tuple

    @property
    def signature(self) -> tuple:
        reach = tuple(round(min(r, REACH_FAR_M), 3) for r in self.plate_reach_m)
        return (self.zone, self.layers, reach)

    def describe(self) -> str:
        edge = [r for r in self.plate_reach_m if r < REACH_FAR_M]
        note = f" (plate edge {min(edge) / IN_TO_M:.2f} in away)" if edge else ""
        return f"{self.zone or 'uncovered'}: {describe_stack(self.layers)}{note}"


@dataclass(frozen=True)
class Protector:
    name: str
    zones: tuple   # first zone containing the point wins

    @classmethod
    def uniform(cls, stack: Stack, name: str | None = None) -> "Protector":
        """One stack everywhere (no edges inside any shift radius)."""
        return cls(name or describe_stack(stack), (Zone("cardiac", Ellipse(0, 0, FAR_IN, FAR_IN), tuple(stack)),))

    def zone_at(self, x: float, y: float) -> Zone | None:
        for z in self.zones:
            if z.outline.contains(x, y):
                return z
        return None

    def local_stack(self, x: float, y: float) -> LocalStack:
        z = self.zone_at(x, y)
        if z is None:
            return LocalStack(None, (), ())
        layers, reach = [], []
        for lay in z.stack:
            if isinstance(lay, PlateLayer):
                d_in = (lay.extent or z.outline).edge_distance(x, y)
                if d_in <= 0:
                    continue            # impact point is off this plate
                reach.append(min(d_in * IN_TO_M, REACH_FAR_M))
            layers.append(lay)
        return LocalStack(z.name, tuple(layers), tuple(reach))

    def edges_within(self, radius_in: float) -> list[str]:
        """Zone seams / plate edges closer than radius_in to the nominal impact point (origin)."""
        flags = []
        for z in self.zones:
            if abs(z.outline.edge_distance(0.0, 0.0)) < radius_in:
                flags.append(f"zone '{z.name}' boundary within {radius_in} in shift radius")
            for lay in z.stack:
                if isinstance(lay, PlateLayer) and lay.extent is not None \
                        and abs(lay.extent.edge_distance(0.0, 0.0)) < radius_in:
                    flags.append(f"{lay.describe()} edge (zone '{z.name}') within {radius_in} in shift radius")
        return flags

    def covers_silhouette(self, max_shift_in: float, n: int = 48) -> bool:
        """Cardiac zone (zone at the origin) contains the silhouette swept by the max shift."""
        z = self.zone_at(0.0, 0.0)
        if z is None:
            return False
        g = load_nd200().raw["surrogate"]["geometry"]
        a, b = 0.5 * g["silhouette_width_in"]["value"], 0.5 * g["silhouette_height_in"]["value"]
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for sx, sy in zip(max_shift_in * np.cos(t), max_shift_in * np.sin(t)):
            for px, py in zip(a * np.cos(t), b * np.sin(t)):
                if not z.outline.contains(px - sx, py - sy):
                    return False
        return True


def _ellipse(d: dict | None) -> Ellipse | None:
    return None if d is None else Ellipse(d.get("cx", 0.0), d.get("cy", 0.0), d["a"], d["b"])


def protector_from_dict(d: dict) -> Protector:
    zones = []
    for z in d["zones"]:
        stack = tuple(layer(l["material"], l["thickness_mm"], _ellipse(l.get("extent"))) for l in z["stack"])
        outline = _ellipse(z.get("outline")) or Ellipse(0, 0, FAR_IN, FAR_IN)
        zones.append(Zone(z["name"], outline, stack))
    return Protector(d.get("name", "unnamed"), tuple(zones))


def load_design(path: str | Path) -> Protector:
    with open(path) as f:
        return protector_from_dict(yaml.safe_load(f))


def shift_candidates(max_shift_in: float, min_radius_in: float, step_in: float = 0.25,
                     n_angles: int = 16) -> list[tuple[float, float]]:
    """Protector shifts on rings from min_radius to max_shift (inclusive), all directions."""
    radii = list(np.arange(min_radius_in, max_shift_in, step_in)) + [max_shift_in]
    out = []
    for r in radii:
        for k in range(n_angles):
            th = 2 * math.pi * k / n_angles
            out.append((round(r * math.cos(th), 6), round(r * math.sin(th), 6)))
    return out


def protector_metrics(prot: Protector, cell_in: float = 0.1) -> dict:
    """Mass (grid integration over all bounded zones; plates only inside their extent) plus the
    cardiac-zone stack thickness and summed bending stiffness."""
    from chestforge.core.padding import FoamLayer
    bounded = [z for z in prot.zones if z.outline.a < FAR_IN]
    mass_g = None
    if bounded and len(bounded) == len(prot.zones):
        x0 = min(z.outline.cx - z.outline.a for z in bounded)
        x1 = max(z.outline.cx + z.outline.a for z in bounded)
        y0 = min(z.outline.cy - z.outline.b for z in bounded)
        y1 = max(z.outline.cy + z.outline.b for z in bounded)
        xs = np.arange(x0 + cell_in / 2, x1, cell_in)
        ys = np.arange(y0 + cell_in / 2, y1, cell_in)
        X, Y = np.meshgrid(xs, ys)
        areal = np.zeros_like(X)
        claimed = np.zeros(X.shape, dtype=bool)
        for z in prot.zones:
            o = z.outline
            inside = (((X - o.cx) / o.a) ** 2 + ((Y - o.cy) / o.b) ** 2 <= 1.0) & ~claimed
            for lay in z.stack:
                cover = inside
                if isinstance(lay, PlateLayer) and lay.extent is not None:
                    e = lay.extent
                    cover = inside & (((X - e.cx) / e.a) ** 2 + ((Y - e.cy) / e.b) ** 2 <= 1.0)
                areal[cover] += lay.areal_mass_kg_m2
            claimed |= inside
        mass_g = float(areal.sum() * (cell_in * IN_TO_M) ** 2 * 1000)
    z = prot.zone_at(0.0, 0.0)
    stack = z.stack if z else ()
    bending = sum(l.bending_stiffness if isinstance(l, PlateLayer)
                  else l.material.E_Pa * l.thickness_m**3 / 12.0 for l in stack if isinstance(l, (PlateLayer, FoamLayer)))
    return {"mass_g": mass_g, "cardiac_thickness_mm": sum(l.thickness_m for l in stack) * 1000,
            "cardiac_bending_Nm": bending}
