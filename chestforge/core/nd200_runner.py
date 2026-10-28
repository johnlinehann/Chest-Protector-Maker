"""Run the ND200 Table 1 matrix on a protector design -> per-impact PASS/FAIL, margin, P95.

Random locations (§12.2): the ball always hits the silhouette centre; the protector is shifted
up to the torso-length limit in any direction. We search shifts on rings (>= 1 in so contact
points stay >= 1 in from the centre impact) in all directions, simulate every distinct local
stack, and take the two worst shifts that are >= 1 in apart as Random #1 / #2 — the worst case
governs. Monte Carlo (optional) then reports the P95 peak for each of the six impacts.

Run:
  python -m chestforge.core.nd200_runner --layer pc:3 --layer eva_100:15 --torso-in 15 --mc 200
  python -m chestforge.core.nd200_runner --design designs/example_plate_foam.yaml --mc 200
"""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from chestforge import SIM_LABEL
from chestforge.config import load_nd200
from chestforge.core.calibration import require_calibrated
from chestforge.core.impact_sim import ImpactResult, simulate_impact
from chestforge.core.layout import LocalStack, Protector, load_design, shift_candidates
from chestforge.core.monte_carlo import p95, run_monte_carlo
from chestforge.core.padding import FoamLayer, PlateLayer, layer
from chestforge.core.projectile import load_projectile
from chestforge.units import mph_to_ms


@dataclass
class ImpactRecord:
    sample: int
    condition: str
    location: str
    shift_in: tuple[float, float]      # protector shift (dx, dy); impact point on protector = -shift
    velocity_mph: float
    peak_lbf: float                    # nominal conditions
    limit_lbf: float
    margin_pct: float                  # (limit - peak) / limit * 100
    passed: bool
    exceeds_load_cell_capacity: bool
    result: ImpactResult | None
    local: str = ""                    # zone + stack under the impact point
    p95_lbf: float | None = None
    mc_samples: int = 0

    @property
    def bottomed_out(self) -> bool:
        return bool(self.result and self.result.bottomed_out)

    @property
    def p95_margin_pct(self) -> float | None:
        return None if self.p95_lbf is None else (self.limit_lbf - self.p95_lbf) / self.limit_lbf * 100


@dataclass
class ND200Report:
    design: str
    projectile: str
    torso_length_in: float
    max_shift_in: float
    impacts: list[ImpactRecord]
    beyond_standard: bool
    flags: list[str] = field(default_factory=list)
    covers_silhouette: bool = True
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
    def has_mc(self) -> bool:
        return all(i.p95_lbf is not None for i in self.impacts)

    @property
    def p95_passed(self) -> bool | None:
        return all(i.p95_lbf <= i.limit_lbf for i in self.impacts) if self.has_mc else None

    @property
    def min_margin_pct(self) -> float:
        return min(i.margin_pct for i in self.impacts)

    @property
    def min_p95_margin_pct(self) -> float | None:
        return min(i.p95_margin_pct for i in self.impacts) if self.has_mc else None

    def to_text(self) -> str:
        mc = self.has_mc
        lines = [
            f"ND200-22m25 report — {self.label}",
            f"Design: {self.design} | projectile: {self.projectile} | torso length {self.torso_length_in} in "
            f"-> max shift {self.max_shift_in} in (§12.2)",
            "",
            f"{'S':>1} {'location':<15} {'shift (in)':>13} {'mph':>4} {'peak':>7} "
            + (f"{'P95':>7} " if mc else "") + f"{'limit':>5} {'margin':>7}  result  under impact",
        ]
        for i in self.impacts:
            notes = []
            if i.bottomed_out:
                notes.append("BOTTOMED: " + ", ".join(i.result.bottomed_layers))
            if i.exceeds_load_cell_capacity:
                notes.append("> load cell capacity")
            margin = i.p95_margin_pct if mc else i.margin_pct
            lines.append(
                f"{i.sample:>1} {i.location:<15} {i.shift_in[0]:+6.2f},{i.shift_in[1]:+6.2f} {i.velocity_mph:>4.0f} "
                f"{i.peak_lbf:>7.1f} " + (f"{i.p95_lbf:>7.1f} " if mc else "")
                + f"{i.limit_lbf:>5.0f} {margin:>6.1f}%  {'PASS' if i.passed else 'FAIL':<6}  {i.local}"
                + (f"  [{'; '.join(notes)}]" if notes else "")
            )
        lines.append("")
        lines.append(f"ND200 (nominal): {self.verdict}   min margin {self.min_margin_pct:.1f}%")
        if mc:
            n = self.impacts[0].mc_samples
            lines.append(f"P95 over {n} Monte Carlo samples/impact (ball spec, ±3% velocity, 67-77 F): "
                         f"{'PASS' if self.p95_passed else 'FAIL'}   min P95 margin {self.min_p95_margin_pct:.1f}%")
        if not self.covers_silhouette:
            lines.append("WARNING: cardiac zone does not cover the silhouette + max shift radius")
        lines += [f"FLAG: {f}" for f in self.flags]
        return "\n".join(lines)


def _as_protector(design) -> Protector:
    if isinstance(design, Protector):
        return design
    if isinstance(design, (FoamLayer, PlateLayer)):
        return Protector.uniform((design,))
    return Protector.uniform(tuple(design))


def run_nd200(design, projectile: str = "baseball", torso_length_in: float = 15.0,
              velocity_scale: float = 1.0, design_name: str | None = None, mc_samples: int = 0,
              seed: int = 0, ring_step_in: float = 0.25, n_angles: int = 16) -> ND200Report:
    """Run all Table 1 impacts. Refuses (CalibrationError) if the NTS is not calibrated."""
    nts = require_calibrated(projectile)
    nd = load_nd200()
    ball = load_projectile(projectile)
    prot = _as_protector(design)
    tol = nd.velocity_tolerance_pct / 100
    if not 1 - tol <= velocity_scale <= 1 + tol:
        raise ValueError(f"velocity_scale {velocity_scale} outside ND200 ±{nd.velocity_tolerance_pct}%")

    max_shift = nd.max_shift_in(torso_length_in)
    spacing = nd.raw["random_location"]["min_contact_spacing_in"]["value"]
    if max_shift < spacing:
        raise ValueError(f"max shift {max_shift} in < contact spacing {spacing} in (§12.2)")
    capacity = nd.raw["surrogate"]["load_cell_capacity_lbf"]["value"]

    center = prot.local_stack(0.0, 0.0)
    cands = [(s, prot.local_stack(-s[0], -s[1])) for s in shift_candidates(max_shift, spacing, ring_step_in, n_angles)]
    unique: dict[tuple, LocalStack] = {center.signature: center}
    for _, loc in cands:
        unique.setdefault(loc.signature, loc)

    records = []
    mc_cache: dict[tuple, float] = {}
    for s in nd.raw["test_matrix"]["samples"]:
        limit = nd.limit(s["condition"])
        mph = limit.velocity_mph * velocity_scale
        v = mph_to_ms(mph)
        with ThreadPoolExecutor() as pool:   # kernel releases the GIL
            futs = {sig: pool.submit(simulate_impact, ball, nts, v, loc.layers, plate_reach_m=loc.plate_reach_m)
                    for sig, loc in unique.items()}
            sims = {sig: f.result() for sig, f in futs.items()}

        ranked = sorted(cands, key=lambda c: (-round(sims[c[1].signature].peak_lbf, 6), -round(math.hypot(*c[0]), 4)))
        r1 = ranked[0]
        r2 = next(c for c in ranked[1:] if math.dist(c[0], r1[0]) >= spacing)
        chosen = {"cardiac_center": ((0.0, 0.0), center), "random_1": r1, "random_2": r2}

        for loc_name in s["locations"]:
            shift, loc = chosen[loc_name]
            r = sims[loc.signature]
            rec = ImpactRecord(
                sample=s["sample"], condition=s["condition"], location=loc_name, shift_in=shift,
                velocity_mph=mph, peak_lbf=r.peak_lbf, limit_lbf=limit.max_peak_lbf,
                margin_pct=(limit.max_peak_lbf - r.peak_lbf) / limit.max_peak_lbf * 100,
                passed=r.peak_lbf <= limit.max_peak_lbf, exceeds_load_cell_capacity=r.peak_lbf > capacity,
                result=r, local=loc.describe(),
            )
            if mc_samples > 0:
                key = (loc.signature, s["condition"])
                if key not in mc_cache:
                    rng = np.random.default_rng([seed, len(mc_cache)])
                    mc_cache[key] = p95(run_monte_carlo(loc, ball, nts, v, mc_samples, rng).peak_lbf)
                rec.p95_lbf, rec.mc_samples = mc_cache[key], mc_samples
            records.append(rec)

    return ND200Report(
        design=design_name or prot.name, projectile=projectile, torso_length_in=torso_length_in,
        max_shift_in=max_shift, impacts=records, beyond_standard=not ball.nd200,
        flags=prot.edges_within(max_shift), covers_silhouette=prot.covers_silhouette(max_shift),
    )


def _parse_layer(spec: str):
    mat, _, mm = spec.partition(":")
    return layer(mat, float(mm))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the ND200 Table 1 matrix on a protector design.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--design", help="protector YAML (zones, stacks, plate extents)")
    g.add_argument("--layer", action="append", help="material:thickness_mm, ball side first (repeatable)")
    ap.add_argument("--projectile", default="baseball")
    ap.add_argument("--torso-in", type=float, default=15.0)
    ap.add_argument("--mc", type=int, default=0, help="Monte Carlo samples per impact (P95)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.design:
        design = load_design(args.design)
    else:
        design = Protector.uniform(tuple(_parse_layer(s) for s in (args.layer or ["eva_100:20"])))
    report = run_nd200(design, args.projectile, args.torso_in, mc_samples=args.mc, seed=args.seed)
    print(report.to_text())
    print("\nNOTE: material parameters in config/materials.yaml are estimated — not measured data.")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
