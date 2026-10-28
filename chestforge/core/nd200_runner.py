"""Run the ND200 Table 1 matrix on a design -> per-impact PASS/FAIL + margin.

Milestone 2: a single uniform pad covers the whole silhouette, so a shifted protector presents
the same stack at the impact point; random-location shifts are still generated and checked
against §12.2 (shift limit by torso length, >= 1 in between contact points). Edge/seam effects
inside the shift radius arrive with zones in Milestone 3.

Run:  python -m chestforge.core.nd200_runner --material eva_100 --thickness-mm 20 --torso-in 15
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

from chestforge import SIM_LABEL
from chestforge.config import load_nd200
from chestforge.core.calibration import require_calibrated
from chestforge.core.impact_sim import ImpactResult, simulate_impact
from chestforge.core.padding import PadLayer, load_material
from chestforge.core.projectile import load_projectile
from chestforge.units import mph_to_ms


@dataclass
class ImpactRecord:
    sample: int
    condition: str
    location: str
    shift_in: tuple[float, float]      # protector shift (dx, dy); contact point on protector = -shift
    velocity_mph: float
    peak_lbf: float
    limit_lbf: float
    margin_pct: float                  # (limit - peak) / limit * 100
    passed: bool
    exceeds_load_cell_capacity: bool
    result: ImpactResult

    @property
    def bottomed_out(self) -> bool:
        return self.result.bottomed_out


@dataclass
class ND200Report:
    design: str
    projectile: str
    torso_length_in: float
    max_shift_in: float
    impacts: list[ImpactRecord]
    beyond_standard: bool
    label: str = field(default=SIM_LABEL)

    @property
    def passed(self) -> bool:
        return all(i.passed for i in self.impacts)

    @property
    def verdict(self) -> str:
        if self.beyond_standard:
            return "BEYOND STANDARD (not ND200)"
        return "PASS" if self.passed else "FAIL"

    @property
    def min_margin_pct(self) -> float:
        return min(i.margin_pct for i in self.impacts)

    def to_text(self) -> str:
        lines = [
            f"ND200-22m25 report — {self.label}",
            f"Design: {self.design} | projectile: {self.projectile} | torso length {self.torso_length_in} in "
            f"-> max shift {self.max_shift_in} in (§12.2)",
            "",
            f"{'S':>1} {'location':<15} {'shift (in)':>13} {'mph':>4} {'peak lbf':>9} {'limit':>6} {'margin':>8}  "
            f"{'pad strain':>10}  result",
        ]
        for i in self.impacts:
            flags = []
            if i.bottomed_out:
                flags.append("BOTTOMED OUT")
            if i.exceeds_load_cell_capacity:
                flags.append("> load cell capacity")
            lines.append(
                f"{i.sample:>1} {i.location:<15} {i.shift_in[0]:+6.2f},{i.shift_in[1]:+6.2f} {i.velocity_mph:>4.0f} "
                f"{i.peak_lbf:>9.1f} {i.limit_lbf:>6.0f} {i.margin_pct:>7.1f}%  {i.result.max_pad_strain:>10.2f}  "
                f"{'PASS' if i.passed else 'FAIL'} {' '.join(flags)}"
            )
        lines += ["", f"OVERALL: {self.verdict}   (min margin {self.min_margin_pct:.1f}%)"]
        return "\n".join(lines)


def random_shifts(max_shift_in: float, min_spacing_in: float) -> dict[str, tuple[float, float]]:
    """Two shifts at the §12.2 limit, chosen so all three contact points are >= min_spacing apart."""
    shifts = {
        "cardiac_center": (0.0, 0.0),
        "random_1": (max_shift_in, 0.0),        # protector shifted toward wearer's left
        "random_2": (0.0, max_shift_in),        # protector shifted up
    }
    pts = list(shifts.values())
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            if math.dist(pts[a], pts[b]) < min_spacing_in:
                raise ValueError(f"contact points {pts[a]} and {pts[b]} closer than {min_spacing_in} in (§12.2)")
    return shifts


def run_nd200(pad: PadLayer, projectile: str = "baseball", torso_length_in: float = 15.0,
              velocity_scale: float = 1.0, design_name: str | None = None) -> ND200Report:
    """Run all Table 1 impacts. Refuses (CalibrationError) if the NTS is not calibrated."""
    nts = require_calibrated(projectile)
    nd = load_nd200()
    ball = load_projectile(projectile)
    tol = nd.velocity_tolerance_pct / 100
    if not 1 - tol <= velocity_scale <= 1 + tol:
        raise ValueError(f"velocity_scale {velocity_scale} outside ND200 ±{nd.velocity_tolerance_pct}%")

    max_shift = nd.max_shift_in(torso_length_in)
    rl = nd.raw["random_location"]
    shifts = random_shifts(max_shift, rl["min_contact_spacing_in"]["value"])
    capacity = nd.raw["surrogate"]["load_cell_capacity_lbf"]["value"]

    records = []
    for s in nd.raw["test_matrix"]["samples"]:
        limit = nd.limit(s["condition"])
        mph = limit.velocity_mph * velocity_scale
        r = simulate_impact(ball, nts, mph_to_ms(mph), pad)  # uniform pad: a shift does not change the stack
        for loc in s["locations"]:
            peak = r.peak_lbf
            records.append(ImpactRecord(
                sample=s["sample"], condition=s["condition"], location=loc, shift_in=shifts[loc],
                velocity_mph=mph, peak_lbf=peak, limit_lbf=limit.max_peak_lbf,
                margin_pct=(limit.max_peak_lbf - peak) / limit.max_peak_lbf * 100,
                passed=peak <= limit.max_peak_lbf,
                exceeds_load_cell_capacity=peak > capacity,
                result=r,
            ))
    name = design_name or f"{pad.material.id} {pad.thickness_m * 1000:.1f} mm"
    return ND200Report(design=name, projectile=projectile, torso_length_in=torso_length_in,
                       max_shift_in=max_shift, impacts=records, beyond_standard=not ball.nd200)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the ND200 Table 1 matrix on a single-layer pad.")
    ap.add_argument("--material", default="eva_100")
    ap.add_argument("--thickness-mm", type=float, default=20.0)
    ap.add_argument("--projectile", default="baseball")
    ap.add_argument("--torso-in", type=float, default=15.0)
    args = ap.parse_args(argv)
    report = run_nd200(PadLayer(load_material(args.material), args.thickness_mm / 1000),
                       args.projectile, args.torso_in)
    print(report.to_text())
    if load_material(args.material).estimated:
        print("\nNOTE: material parameters are estimated (config/materials.yaml) — not measured data.")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
