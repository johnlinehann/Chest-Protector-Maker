"""NOCSAE Thoracic Surrogate (NTS) — 3-DOF lumped-parameter model.

    ball m1 --contact--> loading surface m2 --[k2, c2]--> back plate + table m3 (free on linear bearing)

The load cell sits between the loading surface and the back plate (ND200 §4.1), so the
cardiac load is the force carried by the k2/c2 branch, times a DAQ gain (Appendix 1
lets the lab adjust acquisition settings per projectile).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from chestforge.config import load_nd200, load_nts
from chestforge.units import IN_TO_M, LBF_TO_N


@dataclass(frozen=True)
class NTS:
    loading_surface_mass_kg: float
    back_mass_kg: float            # surrogate body + back plate + linear bearing table
    table_mass_kg: float
    skin_stiffness_N_per_m: float
    skin_damping_ratio: float
    reference_mass_kg: float       # mass the damping ratio is defined against (reference ball)
    gain: float = 1.0

    @property
    def skin_damping_Ns_per_m(self) -> float:
        return 2.0 * self.skin_damping_ratio * math.sqrt(self.skin_stiffness_N_per_m * self.reference_mass_kg)

    def with_stiffness(self, k2: float) -> "NTS":
        return replace(self, skin_stiffness_N_per_m=k2)

    def with_gain(self, gain: float) -> "NTS":
        return replace(self, gain=gain)

    def check_table_mass(self) -> None:
        """§9.4: linear bearing table weight <= 12.5 lb."""
        limit_lb = load_nd200().raw["surrogate"]["linear_bearing_table_max_weight_lb"]["value"]
        if self.table_mass_kg > limit_lb * 0.45359237 + 1e-9:
            raise ValueError(f"table mass {self.table_mass_kg:.2f} kg exceeds §9.4 limit {limit_lb} lb")

    def load_cell_force(self, x2: float, v2: float, x3: float, v3: float) -> float:
        """Raw (pre-gain) force through the load cell branch, N (compression positive)."""
        return self.skin_stiffness_N_per_m * (x2 - x3) + self.skin_damping_Ns_per_m * (v2 - v3)


def load_nts_prior(reference_mass_kg: float) -> NTS:
    """Uncalibrated NTS from config/nts.yaml priors."""
    c = load_nts()
    nts = NTS(
        loading_surface_mass_kg=c["loading_surface_mass_kg"]["value"],
        back_mass_kg=c["surrogate_body_mass_kg"]["value"] + c["table_mass_kg"]["value"],
        table_mass_kg=c["table_mass_kg"]["value"],
        skin_stiffness_N_per_m=c["skin_stiffness_N_per_m"]["initial"],
        skin_damping_ratio=c["skin_damping_ratio"]["value"],
        reference_mass_kg=reference_mass_kg,
    )
    nts.check_table_mass()
    return nts


def silhouette_radius_m() -> float:
    """Area-equivalent circle of the cardiac silhouette ellipse (ND200 Figure 1)."""
    g = load_nd200().raw["surrogate"]["geometry"]
    a = 0.5 * g["silhouette_width_in"]["value"]
    b = 0.5 * g["silhouette_height_in"]["value"]
    return math.sqrt(a * b) * IN_TO_M


def lbf(newtons: float) -> float:
    return newtons / LBF_TO_N
