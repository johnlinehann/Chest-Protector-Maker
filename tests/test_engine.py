"""The compiled RK4 engine against an independent scipy integration, and multi-layer physics."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from chestforge.core.calibration import require_calibrated
from chestforge.core.filters import cfc_filter
from chestforge.core.impact_sim import simulate_bare_nts, simulate_impact
from chestforge.core.padding import layer
from chestforge.core.projectile import load_projectile
from chestforge.units import mph_to_ms, n_to_lbf


def _scipy_bare_peak(ball, nts, v0):
    m1, m2, m3 = ball.mass_kg, nts.loading_surface_mass_kg, nts.back_mass_kg
    k2, c2 = nts.skin_stiffness_N_per_m, nts.skin_damping_Ns_per_m

    def rhs(_t, y):
        x1, v1, x2, v2, x3, v3 = y
        fc = ball.contact_force(x1 - x2, v1 - v2, v0)
        fs = k2 * (x2 - x3) + c2 * (v2 - v3)
        return [v1, -fc / m1, v2, (fc - fs) / m2, v3, fs / m3]

    t = np.arange(0, 0.03, 1 / 20000)
    sol = solve_ivp(rhs, (0, t[-1]), [0, v0, 0, 0, 0, 0], t_eval=t, max_step=2e-5, rtol=1e-9, atol=1e-12)
    x1, v1, x2, v2, x3, v3 = sol.y
    raw = np.pad(nts.gain * (k2 * (x2 - x3) + c2 * (v2 - v3)), (100, 0))   # 5 ms pre-trigger, as recorded
    return n_to_lbf(cfc_filter(raw, 20000, 120).max())


@pytest.mark.parametrize("mph", [30, 50])
def test_engine_matches_independent_scipy_integration(calibrated, mph):
    ball, nts = load_projectile(), require_calibrated("baseball")
    ours = simulate_bare_nts(ball, nts, mph_to_ms(mph)).peak_lbf
    assert ours == pytest.approx(_scipy_bare_peak(ball, nts, mph_to_ms(mph)), rel=2e-3)


def test_plate_reduces_load_via_spreading(calibrated):
    ball, nts = load_projectile(), require_calibrated("baseball")
    foam = simulate_impact(ball, nts, mph_to_ms(50), (layer("eva_100", 15),))
    plated = simulate_impact(ball, nts, mph_to_ms(50), (layer("pc", 3), layer("eva_100", 15)))
    assert plated.peak_lbf < 0.8 * foam.peak_lbf
    assert plated.min_load_cell_share < 0.6 and foam.min_load_cell_share == 1.0


def test_plate_edge_increases_load(calibrated):
    ball, nts = load_projectile(), require_calibrated("baseball")
    stack = (layer("pc", 3), layer("eva_100", 15))
    interior = simulate_impact(ball, nts, mph_to_ms(50), stack, plate_reach_m=[0.3])
    edge = simulate_impact(ball, nts, mph_to_ms(50), stack, plate_reach_m=[0.0])
    assert edge.peak_lbf > interior.peak_lbf


@pytest.mark.parametrize("stack", [
    ("eva_100:5", "pc:3", "poron_xrd_240:15"),
    ("pc:2", "hdpe:2", "pe_45:10"),
    ("pe_45:10",),
    ("pc:3",),
])
def test_energy_conserved_multilayer(calibrated, stack):
    ball, nts = load_projectile(), require_calibrated("baseball")
    layers = tuple(layer(s.split(":")[0], float(s.split(":")[1])) for s in stack)
    r = simulate_impact(ball, nts, mph_to_ms(50), layers)
    assert abs(r.energy_residual_J) < 2e-3 * r.energy_in_J


def test_warm_foam_transmits_more(calibrated):
    ball, nts = load_projectile(), require_calibrated("baseball")
    stack = (layer("poron_xrd_240", 15),)
    cold = simulate_impact(ball, nts, mph_to_ms(50), stack, temp_F=67)
    warm = simulate_impact(ball, nts, mph_to_ms(50), stack, temp_F=77)
    assert warm.max_pad_strain > cold.max_pad_strain
