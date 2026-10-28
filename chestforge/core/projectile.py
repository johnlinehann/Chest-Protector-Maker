"""Projectile models: mass, radius, Hunt-Crossley contact derived from the ND200 C-D spec."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from chestforge.config import Range, load_nd200, load_projectiles
from chestforge.units import G_TO_KG, IN_TO_M, LBF_TO_N


@dataclass(frozen=True)
class Projectile:
    name: str
    nd200: bool                 # False => beyond-standard (softball)
    mass_kg: float
    circumference_in: float
    cd_force_lb: float          # C-D force at the spec displacement
    cd_at: float                # C-D displacement, in units of cd_at_units
    cd_at_units: str            # "in" (baseball) or "%" of diameter (lacrosse)
    contact_exponent: float
    cor: float
    mass_range_g: Range
    circumference_range_in: Range
    cd_range_lb: Range
    cor_range: Range

    @property
    def radius_m(self) -> float:
        return self.circumference_in * IN_TO_M / (2 * math.pi)

    @property
    def cd_total_deflection_m(self) -> float:
        if self.cd_at_units == "in":
            return self.cd_at * IN_TO_M
        if self.cd_at_units == "%":
            return self.cd_at / 100 * 2 * self.radius_m
        raise ValueError(f"unknown C-D units {self.cd_at_units!r}")

    @property
    def contact_stiffness(self) -> float:
        """k in F = k d^n for one ball-on-flat contact (N/m^n).

        The C-D test compresses the ball between two plates, so each contact
        carries the full force at half the total deflection.
        """
        d = 0.5 * self.cd_total_deflection_m
        return self.cd_force_lb * LBF_TO_N / d**self.contact_exponent

    def cd_force_lb_at_spec(self) -> float:
        """Quasi-static C-D force the contact law predicts at the spec displacement."""
        d = 0.5 * self.cd_total_deflection_m
        return self.contact_stiffness * d**self.contact_exponent / LBF_TO_N

    def contact_force(self, d: float, d_dot: float, v0: float) -> float:
        """Hunt-Crossley force with Lankarani-Nikravesh damping; zero when not in contact."""
        if d <= 0.0:
            return 0.0
        alpha = 3.0 * (1.0 - self.cor**2) / (4.0 * max(abs(v0), 1e-6))
        return max(self.contact_stiffness * d**self.contact_exponent * (1.0 + alpha * d_dot), 0.0)

    def in_spec(self) -> bool:
        return (
            self.mass_range_g.contains(self.mass_kg / G_TO_KG)
            and self.circumference_range_in.contains(self.circumference_in)
            and self.cd_range_lb.contains(self.cd_force_lb_at_spec())
        )

    def sample_within_spec(self, rng: np.random.Generator) -> "Projectile":
        """Uniform draw of mass, size, C-D force and COR inside their ranges (for Monte Carlo)."""
        return replace(
            self,
            mass_kg=rng.uniform(self.mass_range_g.min, self.mass_range_g.max) * G_TO_KG,
            circumference_in=rng.uniform(self.circumference_range_in.min, self.circumference_range_in.max),
            cd_force_lb=rng.uniform(self.cd_range_lb.min, self.cd_range_lb.max),
            cor=rng.uniform(self.cor_range.min, self.cor_range.max),
        )


def load_projectile(name: str | None = None) -> Projectile:
    model = load_projectiles()
    name = name or model["default"]
    p = model[name]
    nd200 = p.get("nd200", True)
    spec = load_nd200().projectile_spec(name) if nd200 else p

    mass = Range(**spec["mass_g"])
    circ = Range(**spec["circumference_in"])
    cd = spec["compression_displacement"]
    return Projectile(
        name=name,
        nd200=nd200,
        mass_kg=mass.mid * G_TO_KG,
        circumference_in=circ.mid,
        cd_force_lb=p["cd_force_nominal_lb"]["value"],
        cd_at=cd["at"]["value"],
        cd_at_units=cd["at"]["units"],
        contact_exponent=p["contact_exponent"]["value"],
        cor=p["cor"]["value"],
        mass_range_g=mass,
        circumference_range_in=circ,
        cd_range_lb=Range(**cd["force_lb"]),
        cor_range=Range(min=p["cor"]["min"], max=p["cor"]["max"]),
    )
