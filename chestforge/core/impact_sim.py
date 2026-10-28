"""Explicit 1D impact dynamics -> force-time trace at the NTS load cell."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from chestforge import SIM_LABEL
from chestforge.config import load_nd200, load_nts
from chestforge.core.filters import cfc_filter
from chestforge.core.projectile import Projectile
from chestforge.core.surrogate import NTS
from chestforge.units import n_to_lbf


@dataclass
class ImpactResult:
    t: np.ndarray                  # s
    force_raw_N: np.ndarray        # load cell, gain applied, unfiltered
    force_N: np.ndarray            # CFC-filtered (what ND200 scores)
    velocity_ms: float
    projectile: str
    peak_N: float
    energy_in_J: float
    energy_dissipated_J: float
    energy_residual_J: float       # conservation error of the integration
    max_ball_compression_m: float
    max_skin_compression_m: float
    beyond_standard: bool
    label: str = field(default=SIM_LABEL)

    @property
    def peak_lbf(self) -> float:
        return n_to_lbf(self.peak_N)


def simulate_bare_nts(projectile: Projectile, nts: NTS, velocity_ms: float) -> ImpactResult:
    """Projectile into the unprotected NTS (calibration / system-check configuration)."""
    solver = load_nts()["solver"]
    nd = load_nd200()
    fs = solver["output_rate_hz"]
    if fs < nd.min_sample_rate_hz:
        raise ValueError(f"output rate {fs} Hz below ND200 §9.5 minimum {nd.min_sample_rate_hz} Hz")

    m1, m2, m3 = projectile.mass_kg, nts.loading_surface_mass_kg, nts.back_mass_kg
    k2, c2 = nts.skin_stiffness_N_per_m, nts.skin_damping_Ns_per_m
    v0 = velocity_ms

    def contact(y):
        d = y[0] - y[2]
        return projectile.contact_force(d, y[1] - y[3], v0)

    def rhs(_t, y):
        x1, v1, x2, v2, x3, v3, _ = y
        fc = contact(y)
        fs_ = k2 * (x2 - x3) + c2 * (v2 - v3)
        d = x1 - x2
        hc_damp = 0.0
        if d > 0:  # power lost in the ball: contact work minus change in stored elastic energy
            elastic = projectile.contact_stiffness * d**projectile.contact_exponent
            hc_damp = (fc - elastic) * (v1 - v2)
        return [v1, -fc / m1, v2, (fc - fs_) / m2, v3, fs_ / m3, hc_damp + c2 * (v2 - v3) ** 2]

    t_eval = np.arange(0.0, solver["duration_s"], 1.0 / fs)
    y0 = [0.0, v0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sol = solve_ivp(rhs, (0.0, t_eval[-1]), y0, method="RK45", t_eval=t_eval,
                    max_step=solver["max_step_s"], rtol=solver["rtol"], atol=solver["atol"])
    if not sol.success:
        raise RuntimeError(f"impact integration failed: {sol.message}")
    x1, v1, x2, v2, x3, v3, e_diss = sol.y

    raw = nts.gain * (k2 * (x2 - x3) + c2 * (v2 - v3))
    filt = cfc_filter(raw, fs, nd.filter_cfc)

    d_ball = np.maximum(x1 - x2, 0.0)
    elastic_ball = projectile.contact_stiffness * d_ball ** (projectile.contact_exponent + 1) / (projectile.contact_exponent + 1)
    energy = 0.5 * (m1 * v1**2 + m2 * v2**2 + m3 * v3**2) + 0.5 * k2 * (x2 - x3) ** 2 + elastic_ball
    e_in = 0.5 * m1 * v0**2

    return ImpactResult(
        t=sol.t,
        force_raw_N=raw,
        force_N=filt,
        velocity_ms=v0,
        projectile=projectile.name,
        peak_N=float(filt.max()),
        energy_in_J=e_in,
        energy_dissipated_J=float(e_diss[-1]),
        energy_residual_J=float(e_in - energy[-1] - e_diss[-1]),
        max_ball_compression_m=float(d_ball.max()),
        max_skin_compression_m=float((x2 - x3).max()),
        beyond_standard=not projectile.nd200,
    )
