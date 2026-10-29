"""Auto-calibrate the NTS model to ND200 Appendix 1 and guard every design run on it.

Mirrors the lab procedure:
  * reference projectile: fit skin stiffness k2 so the bare 30 mph response is 140 lbf;
  * other ND200 projectile: keep the mechanics, fit a DAQ gain (App. 1 NOTE: settings per projectile);
  * verify Steps 3/5/6 at the worst edge of each +/-0.2 mph velocity window, and the §10.4 system
    check at 30 mph +/-3%.
Softball is beyond-standard: no calibration target exists, so it runs on the reference mechanics at
gain 1 and its results are flagged.

Run:  python -m chestforge.core.calibration
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass

import yaml
from scipy.optimize import brentq

from chestforge import SIM_LABEL
from chestforge.config import CONFIG_DIR, config_hash, load_nd200, load_nts, load_projectiles
from chestforge.core.impact_sim import simulate_bare_nts
from chestforge.core.projectile import Projectile, load_projectile
from chestforge.core.surrogate import NTS, load_nts_prior
from chestforge.units import mph_to_ms

CALIBRATION_FILE = CONFIG_DIR / "nts_calibrated.yaml"
_HASHED = ("nd200.yaml", "projectiles.yaml", "nts.yaml", "nts_fitted.yaml")


class CalibrationError(RuntimeError):
    """Raised when the NTS model is not calibrated to ND200 Appendix 1; designs must not run."""


@dataclass
class Check:
    name: str
    velocity_mph: float
    peak_lbf: float
    requirement: str
    passed: bool


def _peak(p: Projectile, nts: NTS, mph: float) -> float:
    return simulate_bare_nts(p, nts, mph_to_ms(mph)).peak_lbf


def verify(p: Projectile, nts: NTS) -> list[Check]:
    """Appendix 1 Steps 3/5/6 at both edges of +/-0.2 mph, plus §10.4 at 30 mph +/-3%."""
    nd = load_nd200()
    cal = nd.calibration
    tol = cal["velocity_tolerance_mph"]["value"]
    tgt, lo, hi = cal["target"], cal["lower"], cal["upper"]
    checks = []
    for v in (tgt["velocity_mph"] - tol, tgt["velocity_mph"], tgt["velocity_mph"] + tol):
        f = _peak(p, nts, v)
        checks.append(Check("App1 Step 3", v, f, f"{tgt['peak_lbf']} ± {tgt['tolerance_lbf']} lbf",
                            abs(f - tgt["peak_lbf"]) <= tgt["tolerance_lbf"]))
    v = lo["velocity_mph"] - tol
    f = _peak(p, nts, v)
    checks.append(Check("App1 Step 5", v, f, f"> {lo['peak_lbf_greater_than']} lbf", f > lo["peak_lbf_greater_than"]))
    v = hi["velocity_mph"] + tol
    f = _peak(p, nts, v)
    checks.append(Check("App1 Step 6", v, f, f"< {hi['peak_lbf_less_than']} lbf", f < hi["peak_lbf_less_than"]))

    sc = nd.raw["system_check"]
    for sign in (-1, 1):
        v = sc["velocity_mph"] * (1 + sign * sc["velocity_tolerance_pct"] / 100)
        f = _peak(p, nts, v)
        checks.append(Check("§10.4 system check", v, f, f"{sc['peak_lbf']['min']}-{sc['peak_lbf']['max']} lbf",
                            sc["peak_lbf"]["min"] <= f <= sc["peak_lbf"]["max"]))
    return checks


def calibrate_nts(write: bool = True) -> dict:
    """Fit the NTS model, verify it for each ND200 projectile and (optionally) write the result."""
    nts_cfg = load_nts()
    nd = load_nd200()
    target_mph = nd.calibration["target"]["velocity_mph"]
    target_lbf = nd.calibration["target"]["peak_lbf"]

    ref = load_projectile(nts_cfg["reference_projectile"])
    prior = load_nts_prior(ref.mass_kg)
    kb = nts_cfg["skin_stiffness_N_per_m"]
    k2 = brentq(lambda k: _peak(ref, prior.with_stiffness(k), target_mph) - target_lbf,
                kb["min"], kb["max"], xtol=1.0, rtol=1e-6)
    mech = prior.with_stiffness(k2)

    projectiles = {}
    band = nts_cfg["gain_sanity_band"]
    for name, model in load_projectiles().items():
        if name == "default":
            continue
        p = load_projectile(name)
        if not p.nd200:
            projectiles[name] = {"gain": 1.0, "beyond_standard": True, "calibrated": False, "checks": []}
            continue
        gain = 1.0 if name == ref.name else target_lbf / _peak(p, mech, target_mph)
        checks = verify(p, mech.with_gain(gain))
        projectiles[name] = {
            "gain": float(gain),
            "gain_in_sanity_band": band[0] <= gain <= band[1],
            "beyond_standard": False,
            "calibrated": all(c.passed for c in checks),
            "checks": [asdict(c) for c in checks],
        }

    result = {
        "label": SIM_LABEL,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config_hash": config_hash(*_HASHED),
        "reference_projectile": ref.name,
        "mechanics": {
            "loading_surface_mass_kg": mech.loading_surface_mass_kg,
            "back_mass_kg": mech.back_mass_kg,
            "table_mass_kg": mech.table_mass_kg,
            "skin_stiffness_N_per_m": float(mech.skin_stiffness_N_per_m),
            "skin_damping_ratio": mech.skin_damping_ratio,
            "skin_damping_Ns_per_m": float(mech.skin_damping_Ns_per_m),
            "reference_mass_kg": mech.reference_mass_kg,
        },
        "projectiles": projectiles,
    }
    if write:
        with open(CALIBRATION_FILE, "w") as f:
            f.write("# Generated by chestforge.core.calibration — do not edit. " + SIM_LABEL + "\n")
            yaml.safe_dump(result, f, sort_keys=False)
    return result


_verified: dict[tuple[str, str], NTS] = {}


def require_calibrated(projectile: str) -> NTS:
    """Calibrated NTS for this projectile, or CalibrationError. Every design run must call this."""
    if not CALIBRATION_FILE.exists():
        raise CalibrationError(f"{CALIBRATION_FILE.name} missing — run `python -m chestforge.core.calibration`")
    with open(CALIBRATION_FILE) as f:
        cal = yaml.safe_load(f)
    h = config_hash(*_HASHED)
    if cal.get("config_hash") != h:
        raise CalibrationError("config changed since calibration — rerun `python -m chestforge.core.calibration`")
    if projectile not in cal["projectiles"]:
        raise CalibrationError(f"no calibration entry for projectile {projectile!r}")
    entry = cal["projectiles"][projectile]
    if entry["beyond_standard"]:
        ref = cal["projectiles"][cal["reference_projectile"]]
        if not ref["calibrated"]:
            raise CalibrationError("reference projectile calibration failed; beyond-standard runs blocked too")
    elif not entry["calibrated"]:
        failed = [c for c in entry["checks"] if not c["passed"]]
        raise CalibrationError(f"NTS calibration failed for {projectile}: {failed}")

    key = (h, projectile)
    if key not in _verified:
        m = cal["mechanics"]
        nts = NTS(
            loading_surface_mass_kg=m["loading_surface_mass_kg"],
            back_mass_kg=m["back_mass_kg"],
            table_mass_kg=m["table_mass_kg"],
            skin_stiffness_N_per_m=m["skin_stiffness_N_per_m"],
            skin_damping_ratio=m["skin_damping_ratio"],
            reference_mass_kg=m["reference_mass_kg"],
            gain=entry["gain"],
        )
        nts.check_table_mass()
        if not entry["beyond_standard"]:  # re-verify rather than trust the file
            failed = [c for c in verify(load_projectile(projectile), nts) if not c.passed]
            if failed:
                raise CalibrationError(f"stored calibration no longer verifies for {projectile}: {failed}")
        _verified[key] = nts
    return _verified[key]


def main() -> int:
    res = calibrate_nts()
    print(f"{SIM_LABEL}\nNTS calibration — ND200 Appendix 1 / §10.4")
    print(f"  fitted skin stiffness k2 = {res['mechanics']['skin_stiffness_N_per_m']:.0f} N/m "
          f"(reference: {res['reference_projectile']})")
    ok = True
    for name, e in res["projectiles"].items():
        if e["beyond_standard"]:
            print(f"\n[{name}] BEYOND STANDARD — no ND200 calibration target; uses reference mechanics, gain 1.0")
            continue
        print(f"\n[{name}] gain {e['gain']:.4f}{'' if e['gain_in_sanity_band'] else '  (WARNING: outside sanity band)'}")
        for c in e["checks"]:
            print(f"  {c['name']:<20} {c['velocity_mph']:5.2f} mph  {c['peak_lbf']:7.2f} lbf  "
                  f"req {c['requirement']:<16} {'PASS' if c['passed'] else 'FAIL'}")
        ok &= e["calibrated"]
    print(f"\nCALIBRATION {'PASS' if ok else 'FAIL'} — written to {CALIBRATION_FILE}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
