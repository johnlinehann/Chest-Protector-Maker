"""Compare the model with measured traces and fit parameters to them.

* fit_nts: bare-NTS traces -> NTS skin stiffness, damping ratio and loading-surface mass.
  Only the damping ratio and surface mass are applied (config/nts_fitted.yaml); the stiffness is
  re-set by the ND200 Appendix 1 calibration afterwards, exactly like a lab adjusting its system.
* fit_material: traces with a known stack -> multipliers on one foam's Avalle/rate/hysteresis
  parameters. Applied as a NEW material id in config/materials_fitted.yaml (estimates untouched).
Fits from traces marked `synthetic: true` are refused unless allow_synthetic=True (tests only).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace

import numpy as np
import yaml
from scipy.optimize import least_squares

from chestforge.config import CONFIG_DIR, MATERIALS_FITTED, NTS_FITTED, load_nts
from chestforge.core.calibration import require_calibrated
from chestforge.core.impact_sim import Rig, simulate_impact
from chestforge.core.padding import FoamLayer, FoamMaterial
from chestforge.core.surrogate import NTS
from chestforge.validation.metrics import compare, pulse_window, summarize
from chestforge.validation.traces import Trace, align

MAT_PARAMS = ("A_Pa", "E_Pa", "B_Pa", "rate_C", "unloading_ratio")


class SyntheticDataError(ValueError):
    pass


def _rig(tr: Trace, nts: NTS) -> Rig:
    return Rig.nts(nts) if tr.rig == "nts" else Rig.drop_tower()


def simulate_trace(tr: Trace, nts: NTS, stack: tuple | None = None) -> tuple[np.ndarray, np.ndarray]:
    r = simulate_impact(tr.projectile(), nts, tr.velocity_ms, tr.stack if stack is None else stack,
                        temp_F=tr.temp_F, rig=_rig(tr, nts))
    return align(r.t, r.force_N), r.force_N


def _nts_for(tr: Trace) -> NTS:
    name = tr.meta.get("projectile", "baseball") if tr.rig == "nts" else "baseball"
    return require_calibrated(name)


def validate(traces: list[Trace]) -> dict:
    """Current model vs every trace."""
    rows = []
    for tr in traces:
        ts, fs = simulate_trace(tr, _nts_for(tr))
        rows.append({"trace": tr.name, "rig": tr.rig, "stack": _stack_str(tr.stack),
                     "velocity_ms": tr.velocity_ms, "synthetic": tr.synthetic, **compare(tr.t_s, tr.force_N, ts, fs)})
    return {"rows": rows, "summary": summarize(rows)}


def _stack_str(stack) -> str:
    return ", ".join(f"{l.material.id}:{l.thickness_m * 1000:g}" for l in stack) or "bare"


def _residual(tr: Trace, ts: np.ndarray, fs: np.ndarray, peak_weight: float = 3.0) -> np.ndarray:
    """Trace shape over the pulse window (normalised by the measured peak) + weighted peak error."""
    t0, t1 = pulse_window(tr.t_s, tr.force_N)
    w = (tr.t_s >= t0) & (tr.t_s <= t1)
    pe = tr.peak_N
    shape = (np.interp(tr.t_s[w][::4], ts, fs, left=0, right=0) - tr.force_N[w][::4]) / pe
    return np.append(shape, peak_weight * (fs.max() - pe) / pe)


@dataclass
class FitResult:
    kind: str
    params: dict
    before: dict
    after: dict
    traces: list
    synthetic: bool
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ NTS

def fit_nts(traces: list[Trace]) -> FitResult:
    bare = [t for t in traces if t.rig == "nts" and not t.stack]
    if not bare:
        raise ValueError("fit_nts needs bare-NTS traces (rig: nts, empty stack)")
    base = {t.meta.get("projectile", "baseball"): require_calibrated(t.meta.get("projectile", "baseball")) for t in bare}

    ref = next(iter(base.values()))
    x_ref = np.array([ref.skin_stiffness_N_per_m, ref.skin_damping_ratio, ref.loading_surface_mass_kg])

    def nts_from(u, nts: NTS) -> NTS:   # u = log multipliers on (k2, zeta, m2); 0 = current model
        k, z, m = x_ref * np.exp(u)
        return replace(nts, skin_stiffness_N_per_m=float(k), skin_damping_ratio=float(z), loading_surface_mass_kg=float(m))

    def res(u):
        return np.concatenate([_residual(t, *simulate_trace(t, nts_from(u, base[t.meta.get("projectile", "baseball")])))
                               for t in bare])

    lo = np.log([0.25, 0.02 / x_ref[1], 0.02 / x_ref[2]])
    hi = np.log([4.0, 1.5 / x_ref[1], 1.0 / x_ref[2]])
    # damping and surface mass trade off along a ridge: seed from a coarse grid, then refine
    seeds = [np.array([0.0, np.log(z / x_ref[1]), np.log(m / x_ref[2])])
             for z in (0.1, 0.2, 0.35, 0.6) for m in (0.05, 0.1, 0.2, 0.4)]
    start = min(seeds, key=lambda u: float(np.sum(res(np.clip(u, lo, hi)) ** 2)))
    sol = least_squares(res, np.clip(start, lo, hi), bounds=(lo, hi), diff_step=0.02, max_nfev=80, x_scale="jac")
    before = validate(bare)
    fitted = {k: nts_from(sol.x, v) for k, v in base.items()}
    after_rows = []
    for t in bare:
        ts, fs = simulate_trace(t, fitted[t.meta.get("projectile", "baseball")])
        after_rows.append({"trace": t.name, **compare(t.t_s, t.force_N, ts, fs)})
    f = fitted[next(iter(fitted))]
    return FitResult(
        kind="nts",
        params={"skin_stiffness_N_per_m": f.skin_stiffness_N_per_m, "skin_damping_ratio": f.skin_damping_ratio,
                "loading_surface_mass_kg": f.loading_surface_mass_kg},
        before=before["summary"], after=summarize(after_rows), traces=[t.name for t in bare],
        synthetic=any(t.synthetic for t in bare), extra={"rows_after": after_rows, "rows_before": before["rows"],
                                                        "cost": float(sol.cost), "nfev": int(sol.nfev)},
    )


# ------------------------------------------------------------------ materials

def _with_material(stack: tuple, mat_id: str, new: FoamMaterial) -> tuple:
    return tuple(FoamLayer(new, l.thickness_m) if isinstance(l, FoamLayer) and l.material.id == mat_id else l
                 for l in stack)


def scaled_material(m: FoamMaterial, mult: np.ndarray, new_id: str | None = None) -> FoamMaterial:
    kw = {p: getattr(m, p) * k for p, k in zip(MAT_PARAMS, mult)}
    kw["unloading_ratio"] = min(kw["unloading_ratio"], 0.95)
    return replace(m, id=new_id or m.id, **kw)


def fit_material(traces: list[Trace], material_id: str, new_id: str | None = None) -> FitResult:
    use = [t for t in traces if any(isinstance(l, FoamLayer) and l.material.id == material_id for l in t.stack)]
    if not use:
        raise ValueError(f"no trace has material {material_id!r} in its stack")
    base = next(l.material for t in use for l in t.stack if isinstance(l, FoamLayer) and l.material.id == material_id)

    def run(mult):
        m = scaled_material(base, mult)
        return [simulate_trace(t, _nts_for(t), _with_material(t.stack, material_id, m)) for t in use]

    def res(logm):
        return np.concatenate([_residual(t, ts, fs) for t, (ts, fs) in zip(use, run(np.exp(logm)))])

    lo, hi = np.log(np.full(5, 0.2)), np.log(np.full(5, 5.0))
    sol = least_squares(res, np.zeros(5), bounds=(lo, hi), diff_step=0.02, max_nfev=120)
    mult = np.exp(sol.x)
    before = validate(use)
    after_rows = [{"trace": t.name, **compare(t.t_s, t.force_N, ts, fs)} for t, (ts, fs) in zip(use, run(mult))]
    new = scaled_material(base, mult, new_id or f"{material_id}_fit")
    return FitResult(
        kind="material",
        params={"base": material_id, "new_id": new.id, "multipliers": dict(zip(MAT_PARAMS, map(float, mult))),
                "material": new, "correlation": _correlations(sol)},
        before=before["summary"], after=summarize(after_rows), traces=[t.name for t in use],
        synthetic=any(t.synthetic for t in use),
        extra={"rows_after": after_rows, "rows_before": before["rows"], "cost": float(sol.cost), "nfev": int(sol.nfev)},
    )


def _correlations(sol) -> dict:
    """Parameter pairs the data cannot separate (|correlation| > 0.9 from the Gauss-Newton covariance)."""
    try:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac)
        d = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
        corr = cov / np.outer(d, d)
    except np.linalg.LinAlgError:
        return {}
    out = {}
    for i in range(len(MAT_PARAMS)):
        for j in range(i + 1, len(MAT_PARAMS)):
            if abs(corr[i, j]) > 0.9:
                out[f"{MAT_PARAMS[i]}~{MAT_PARAMS[j]}"] = round(float(corr[i, j]), 3)
    return out


# ------------------------------------------------------------------ apply (write override files)

def _guard(fit: FitResult, allow_synthetic: bool) -> None:
    if fit.synthetic and not allow_synthetic:
        raise SyntheticDataError("fit used synthetic traces; refusing to write it into the configuration")


def _provenance(fit: FitResult) -> str:
    return (f"FITTED {dt.date.today().isoformat()} to {len(fit.traces)} measured trace(s): {', '.join(fit.traces)}; "
            f"mean |peak err| {fit.after['mean_abs_peak_err_pct']:.1f}%, mean NRMSE {fit.after['mean_abs_nrmse_pct']:.1f}%")


def _clear_caches() -> None:
    from chestforge.core import calibration, padding
    load_nts.cache_clear()
    padding._materials.cache_clear()
    calibration._verified.clear()


def apply_nts(fit: FitResult, allow_synthetic: bool = False) -> dict:
    """Write damping ratio + surface mass overrides, then recalibrate to Appendix 1."""
    _guard(fit, allow_synthetic)
    src = _provenance(fit)
    doc = {
        "fitted": {"date": dt.date.today().isoformat(), "traces": fit.traces, "before": fit.before, "after": fit.after,
                   "fitted_skin_stiffness_N_per_m": float(fit.params["skin_stiffness_N_per_m"]),
                   "synthetic": fit.synthetic},
        "overrides": {
            "skin_damping_ratio": {"value": round(float(fit.params["skin_damping_ratio"]), 5), "source": src,
                                   "estimated": False, "fitted": True},
            "loading_surface_mass_kg": {"value": round(float(fit.params["loading_surface_mass_kg"]), 5), "source": src,
                                        "estimated": False, "fitted": True},
        },
    }
    (CONFIG_DIR / NTS_FITTED).write_text("# Written by chestforge.validation — delete this file to revert to the estimates.\n"
                                         + yaml.safe_dump(doc, sort_keys=False))
    _clear_caches()
    from chestforge.core.calibration import calibrate_nts
    cal = calibrate_nts()
    return {"written": NTS_FITTED, "calibration": {k: v["calibrated"] for k, v in cal["projectiles"].items()
                                                   if not v["beyond_standard"]}}


def material_entry(m: FoamMaterial, src: str) -> dict:
    v = lambda x, unit_scale=1.0: {"value": round(float(x) * unit_scale, 6), "source": src, "estimated": False, "fitted": True}
    return {
        "id": m.id, "name": f"{m.name} — fitted", "model": "avalle_jc",
        "density_kg_m3": {"value": float(m.density_kg_m3), "source": "carried over from base material", "estimated": True},
        "E_MPa": v(m.E_Pa, 1e-6), "A_MPa": v(m.A_Pa, 1e-6), "m": {"value": float(m.m), "source": "carried over", "estimated": True},
        "B_MPa": v(m.B_Pa, 1e-6), "n": {"value": float(m.n), "source": "carried over", "estimated": True},
        "densification_strain": {"value": float(m.densification_strain), "source": "carried over", "estimated": True},
        "rate_C": v(m.rate_C), "ref_strain_rate": {"value": float(m.ref_strain_rate), "source": "carried over", "estimated": True},
        "unloading_ratio": v(m.unloading_ratio),
        "load_spread_angle_deg": {"value": float(m.load_spread_angle_deg), "source": "carried over", "estimated": True},
        "temp_coeff_per_F": {"value": float(m.temp_coeff_per_F), "source": "carried over", "estimated": True},
    }


def apply_material(fit: FitResult, allow_synthetic: bool = False) -> dict:
    """Add (or replace) the fitted material in config/materials_fitted.yaml under its new id."""
    _guard(fit, allow_synthetic)
    path = CONFIG_DIR / MATERIALS_FITTED
    doc = (yaml.safe_load(path.read_text()) if path.exists() else None) or {"materials": []}
    m: FoamMaterial = fit.params["material"]
    doc["materials"] = [x for x in doc["materials"] if x["id"] != m.id] + [material_entry(m, _provenance(fit))]
    path.write_text("# Written by chestforge.validation — fitted materials (new ids; estimates untouched).\n"
                    + yaml.safe_dump(doc, sort_keys=False))
    _clear_caches()
    return {"written": MATERIALS_FITTED, "material": m.id}
