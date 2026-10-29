"""Simulation-vs-test error metrics on onset-aligned, CFC120-filtered traces."""

from __future__ import annotations

import numpy as np

from chestforge.units import LBF_TO_N


def pulse_window(t: np.ndarray, f: np.ndarray, frac: float = 0.05) -> tuple[float, float]:
    """[onset - 1 ms, end of pulse + 2 ms]; the pulse ends where force falls below frac x peak after it."""
    k = int(np.argmax(f))
    after = np.flatnonzero(f[k:] < frac * f[k])
    t_end = t[k + after[0]] if len(after) else t[-1]
    return -1e-3, float(t_end + 2e-3)


def fwhm(t: np.ndarray, f: np.ndarray) -> float:
    above = np.flatnonzero(f >= 0.5 * f.max())
    return float(t[above[-1]] - t[above[0]]) if len(above) else 0.0


def compare(t_exp: np.ndarray, f_exp: np.ndarray, t_sim: np.ndarray, f_sim: np.ndarray) -> dict:
    """Both traces aligned to onset (t = 0). Forces in N; reported in lbf / ms / %."""
    t0, t1 = pulse_window(t_exp, f_exp)
    w = (t_exp >= t0) & (t_exp <= t1)
    te, fe = t_exp[w], f_exp[w]
    fs = np.interp(te, t_sim, f_sim, left=0.0, right=0.0)
    pe, ps = float(f_exp.max()), float(f_sim.max())
    ie, is_ = float(np.trapezoid(np.clip(fe, 0, None), te)), float(np.trapezoid(np.clip(fs, 0, None), te))
    resid = fs - fe
    ss_tot = float(np.sum((fe - fe.mean()) ** 2))
    return {
        "peak_exp_lbf": pe / LBF_TO_N, "peak_sim_lbf": ps / LBF_TO_N,
        "peak_err_pct": (ps - pe) / pe * 100,
        "ttp_exp_ms": float(t_exp[np.argmax(f_exp)]) * 1000, "ttp_sim_ms": float(t_sim[np.argmax(f_sim)]) * 1000,
        "fwhm_exp_ms": fwhm(t_exp, f_exp) * 1000, "fwhm_sim_ms": fwhm(t_sim, f_sim) * 1000,
        "impulse_err_pct": (is_ - ie) / ie * 100 if ie > 0 else float("nan"),
        "nrmse_pct": float(np.sqrt(np.mean(resid**2)) / pe * 100),
        "r2": 1 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan"),
    }


def summarize(rows: list[dict]) -> dict:
    keys = ("peak_err_pct", "nrmse_pct", "impulse_err_pct", "r2")
    out = {"n": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[f"mean_abs_{k}" if k != "r2" else "mean_r2"] = float(np.nanmean(np.abs(v) if k != "r2" else v)) if len(v) else None
    return out
