"""Monte Carlo over test-condition scatter the standard allows.

Per sample: ball properties uniform within the ND200 §13.1 spec (and COR range), impact
velocity uniform within ±3% (§8.1, Table 1), test temperature uniform within 72 ± 5 F (§6.3).
"""

from __future__ import annotations

import numpy as np

from chestforge.config import load_nd200
from chestforge.core.impact_sim import BatchResult, simulate_batch
from chestforge.core.layout import LocalStack
from chestforge.core.projectile import Projectile
from chestforge.core.surrogate import NTS


def sample_conditions(ball: Projectile, n: int, nominal_ms: float, rng: np.random.Generator):
    nd = load_nd200()
    tol = nd.velocity_tolerance_pct / 100
    t = nd.raw["conditioning"]["temperature_F"]
    balls = [ball.sample_within_spec(rng) for _ in range(n)]
    v = nominal_ms * rng.uniform(1 - tol, 1 + tol, n)
    temps = rng.uniform(t["nominal"] - t["tolerance"], t["nominal"] + t["tolerance"], n)
    return balls, v, temps


def run_monte_carlo(local: LocalStack, ball: Projectile, nts: NTS, nominal_ms: float, n: int,
                    rng: np.random.Generator) -> BatchResult:
    balls, v, temps = sample_conditions(ball, n, nominal_ms, rng)
    return simulate_batch(balls, nts, v, temps, local.layers, local.plate_reach_m)


def p95(values: np.ndarray) -> float:
    return float(np.percentile(values, 95))
