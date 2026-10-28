import numpy as np
import pytest

from chestforge.config import load_nd200
from chestforge.core.calibration import require_calibrated
from chestforge.core.layout import Protector
from chestforge.core.monte_carlo import p95, run_monte_carlo, sample_conditions
from chestforge.core.padding import layer
from chestforge.core.projectile import load_projectile
from chestforge.units import mph_to_ms


def test_sampled_conditions_stay_within_nd200():
    ball = load_projectile()
    balls, v, temps = sample_conditions(ball, 300, mph_to_ms(30), np.random.default_rng(1))
    assert all(b.in_spec() for b in balls)
    tol = load_nd200().velocity_tolerance_pct / 100
    assert v.min() >= mph_to_ms(30) * (1 - tol) and v.max() <= mph_to_ms(30) * (1 + tol)
    assert temps.min() >= 67 and temps.max() <= 77


def test_monte_carlo_spread_and_determinism(calibrated):
    ball, nts = load_projectile(), require_calibrated("baseball")
    loc = Protector.uniform((layer("pc", 3), layer("eva_100", 15))).local_stack(0, 0)
    a = run_monte_carlo(loc, ball, nts, mph_to_ms(50), 24, np.random.default_rng(7)).peak_lbf
    b = run_monte_carlo(loc, ball, nts, mph_to_ms(50), 24, np.random.default_rng(7)).peak_lbf
    assert np.array_equal(a, b)
    assert a.std() > 0
    assert a.min() <= p95(a) <= a.max()
