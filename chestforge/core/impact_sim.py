"""Explicit 1D impact dynamics -> force-time trace at the NTS load cell."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from chestforge import SIM_LABEL
from chestforge.config import load_nd200, load_nts
from chestforge.core.filters import cfc_filter
from chestforge.core.padding import PadLayer
from chestforge.core.projectile import Projectile
from chestforge.core.surrogate import NTS, silhouette_radius_m
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
    energy_dissipated_J: float     # ball + NTS skin damping
    energy_absorbed_pad_J: float   # net work done on the pad (0 when bare)
    energy_residual_J: float       # conservation error of the integration
    max_ball_compression_m: float
    max_skin_compression_m: float
    max_pad_compression_m: float
    max_pad_strain: float
    bottomed_out: bool
    beyond_standard: bool
    label: str = field(default=SIM_LABEL)

    @property
    def peak_lbf(self) -> float:
        return n_to_lbf(self.peak_N)


def simulate_bare_nts(projectile: Projectile, nts: NTS, velocity_ms: float) -> ImpactResult:
    """Projectile into the unprotected NTS (calibration / system-check configuration)."""
    return simulate_impact(projectile, nts, velocity_ms, pad=None)


def simulate_impact(projectile: Projectile, nts: NTS, velocity_ms: float,
                    pad: PadLayer | None = None) -> ImpactResult:
    """1D chain: ball -[contact]-> (pad face -[foam]->) loading surface -[k2,c2]-> back mass.

    With a pad, a third of the pad mass under the ball rides on the pad-face node and the
    rest on the loading surface. The foam force splits between the loading surface (load
    cell share f) and the surrogate body (1 - f); the pad's back face moves at the
    f-weighted velocity of the two, tracked as state xb.
    """
    solver = load_nts()["solver"]
    nd = load_nd200()
    fs = solver["output_rate_hz"]
    if fs < nd.min_sample_rate_hz:
        raise ValueError(f"output rate {fs} Hz below ND200 §9.5 minimum {nd.min_sample_rate_hz} Hz")

    R = projectile.radius_m
    m1, m2, m3 = projectile.mass_kg, nts.loading_surface_mass_kg, nts.back_mass_kg
    mp = 0.0
    if pad is not None:
        pad_mass = pad.footprint_mass_kg(R)
        mp, m2 = pad_mass / 3.0, m2 + 2.0 * pad_mass / 3.0
    k2, c2 = nts.skin_stiffness_N_per_m, nts.skin_damping_Ns_per_m
    kc, nc = projectile.contact_stiffness, projectile.contact_exponent
    v0 = velocity_ms
    r_sil = silhouette_radius_m()

    # state: x1 v1 xp vp x2 v2 x3 v3 xb E_ball E_skin W_pad   (xp/vp/xb unused when bare)
    def rhs(_t, y):
        x1, v1, xp, vp, x2, v2, x3, v3, xb = y[:9]
        if pad is None:
            xp, vp = x2, v2
        d = x1 - xp
        fc = projectile.contact_force(d, v1 - vp, v0)
        e_ball = (fc - kc * d**nc) * (v1 - vp) if d > 0 else 0.0
        fs_ = k2 * (x2 - x3) + c2 * (v2 - v3)
        if pad is None:
            return [v1, -fc / m1, 0.0, 0.0, v2, (fc - fs_) / m2, v3, fs_ / m3, 0.0,
                    e_ball, c2 * (v2 - v3) ** 2, 0.0]
        f = pad.load_cell_fraction(xp - xb, R, r_sil)
        vb = f * v2 + (1.0 - f) * v3
        fp = pad.force(xp - xb, vp - vb, R)
        return [v1, -fc / m1, vp, (fc - fp) / mp, v2, (f * fp - fs_) / m2, v3, (fs_ + (1.0 - f) * fp) / m3, vb,
                e_ball, c2 * (v2 - v3) ** 2, fp * (vp - vb)]

    t_eval = np.arange(0.0, solver["duration_s"], 1.0 / fs)
    y0 = [0.0, v0] + [0.0] * 10
    sol = solve_ivp(rhs, (0.0, t_eval[-1]), y0, method="RK45", t_eval=t_eval,
                    max_step=solver["max_step_s"], rtol=solver["rtol"], atol=solver["atol"])
    if not sol.success:
        raise RuntimeError(f"impact integration failed: {sol.message}")
    x1, v1, xp, vp, x2, v2, x3, v3, xb, e_ball, e_skin, w_pad = sol.y
    if pad is None:
        xp, vp = x2, v2

    raw = nts.gain * (k2 * (x2 - x3) + c2 * (v2 - v3))
    filt = cfc_filter(raw, fs, nd.filter_cfc)

    d_ball = np.maximum(x1 - xp, 0.0)
    elastic_ball = kc * d_ball ** (nc + 1) / (nc + 1)
    energy = (0.5 * (m1 * v1**2 + mp * vp**2 + m2 * v2**2 + m3 * v3**2)
              + 0.5 * k2 * (x2 - x3) ** 2 + elastic_ball)
    e_in = 0.5 * m1 * v0**2
    d_pad = xp - xb
    max_pad = float(d_pad.max()) if pad is not None else 0.0
    max_strain = max_pad / pad.thickness_m if pad is not None else 0.0

    return ImpactResult(
        t=sol.t,
        force_raw_N=raw,
        force_N=filt,
        velocity_ms=v0,
        projectile=projectile.name,
        peak_N=float(filt.max()),
        energy_in_J=e_in,
        energy_dissipated_J=float(e_ball[-1] + e_skin[-1]),
        energy_absorbed_pad_J=float(w_pad[-1]),
        energy_residual_J=float(e_in - energy[-1] - e_ball[-1] - e_skin[-1] - w_pad[-1]),
        max_ball_compression_m=float(d_ball.max()),
        max_skin_compression_m=float((x2 - x3).max()),
        max_pad_compression_m=max_pad,
        max_pad_strain=max_strain,
        bottomed_out=pad is not None and max_strain >= pad.material.densification_strain,
        beyond_standard=not projectile.nd200,
    )
