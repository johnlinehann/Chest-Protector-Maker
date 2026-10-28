"""ML surrogate + active learning.

Only the expensive simulated outputs are learned (P95 load @ 50 mph and the worst P95 margin);
mass, thickness and bending stiffness are closed-form and computed exactly for every candidate.
Gaussian-process regression (Matern 5/2 + noise) gives a mean and an uncertainty, which drive
(1) the choice of the next designs to simulate and (2) a queue of designs worth validating with
high-fidelity FEA (uncertain AND close to the margin constraint).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold

from chestforge.optimize.objectives import Evaluation, protector_mass_g, stack_bending_Nm
from chestforge.optimize.search_space import cardiac_stack, features, mutate, random_design

TARGETS = ("p95_50mph_lbf", "min_p95_margin_pct")


@dataclass
class Surrogate:
    models: dict
    x_mean: np.ndarray
    x_std: np.ndarray
    cv_r2: dict

    def predict(self, X: np.ndarray) -> dict:
        Z = (X - self.x_mean) / self.x_std
        return {t: self.models[t].predict(Z, return_std=True) for t in TARGETS}


def _gp(n_features: int) -> GaussianProcessRegressor:
    k = ConstantKernel(1.0) * Matern(length_scale=np.ones(n_features), nu=2.5) + WhiteKernel(1e-2)
    return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=1, random_state=0)


def _usable(evals: list[Evaluation]) -> list[Evaluation]:
    return [e for e in evals if e.error is None and np.isfinite(e.p95_50mph_lbf)]


def fit(evals: list[Evaluation], cfg: dict, cv_folds: int = 5) -> Surrogate:
    with warnings.catch_warnings():   # length scales hitting bounds = irrelevant feature; expected
        warnings.simplefilter("ignore", ConvergenceWarning)
        return _fit(evals, cfg, cv_folds)


def _fit(evals: list[Evaluation], cfg: dict, cv_folds: int) -> Surrogate:
    ev = _usable(evals)
    X = np.array([features(e.params, cfg) for e in ev])
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Z = (X - mean) / std
    models, r2 = {}, {}
    for t in TARGETS:
        y = np.array([getattr(e, t) for e in ev])
        pred = np.zeros_like(y)
        for tr, te in KFold(cv_folds, shuffle=True, random_state=0).split(Z):
            pred[te] = _gp(Z.shape[1]).fit(Z[tr], y[tr]).predict(Z[te])
        r2[t] = float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
        models[t] = _gp(Z.shape[1]).fit(Z, y)
    return Surrogate(models, mean, std, r2)


def _cheap(p: dict, cfg: dict) -> np.ndarray:
    stack = cardiac_stack(p)
    return np.array([protector_mass_g(p, cfg), sum(l.thickness_m for l in stack) * 1000, stack_bending_Nm(stack)])


def candidates(evals: list[Evaluation], cfg: dict, n: int, rng: np.random.Generator) -> list[dict]:
    """Half random designs, half mutations of the current feasible designs."""
    seen = {_key(e.params) for e in evals}
    feas = [e.params for e in evals if e.feasible] or [e.params for e in evals]
    out, keys = [], set()
    while len(out) < n:
        p = random_design(rng, cfg) if (len(out) % 2 == 0 or not feas) else \
            mutate(feas[rng.integers(len(feas))], rng, cfg, n_changes=int(rng.integers(1, 4)))
        k = _key(p)
        if k not in seen and k not in keys:
            keys.add(k)
            out.append(p)
    return out


def _key(p: dict) -> str:
    return repr(sorted(p.items()))


def propose(sur: Surrogate, evals: list[Evaluation], cfg: dict, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    """Next designs to simulate (random-weight Chebyshev on optimistic predictions x P(feasible)),
    and an FEA validation queue (uncertain designs near the margin constraint)."""
    sc = cfg["surrogate"]
    cands = candidates(evals, cfg, sc["candidates"], rng)
    X = np.array([features(p, cfg) for p in cands])
    pred = sur.predict(X)
    mu_p, sd_p = pred["p95_50mph_lbf"]
    mu_m, sd_m = pred["min_p95_margin_pct"]
    p_feas = norm.cdf((mu_m - cfg["margin_target_pct"]) / np.maximum(sd_m, 1e-6))
    cheap = np.array([_cheap(p, cfg) for p in cands])
    obj = np.column_stack([mu_p - sc["kappa"] * sd_p, cheap])
    lo, hi = obj.min(axis=0), obj.max(axis=0)
    norm_obj = (obj - lo) / np.where(hi > lo, hi - lo, 1.0)

    chosen = []
    for _ in range(sc["batch"]):
        w = rng.dirichlet(np.ones(obj.shape[1]))
        score = p_feas * (1.0 - (norm_obj * w).max(axis=1))
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    proposals = [cands[i] for i in chosen]

    near = np.abs(mu_m - cfg["margin_target_pct"]) < 2 * sd_m
    order = np.argsort(-(sd_m * near))[:10]
    queue = [{"params": cands[i], "pred_margin_pct": float(mu_m[i]), "std_pct": float(sd_m[i]),
              "pred_p95_50mph_lbf": float(mu_p[i])} for i in order if near[i]]
    return proposals, queue
