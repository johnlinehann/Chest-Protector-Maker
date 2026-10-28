"""Design optimizer: Optuna TPE (single objective) -> NSGA-II (Pareto) -> ML-surrogate active learning.

Hard constraint (every method): all six ND200 impacts pass at P95 with >= margin_target_pct.
Every evaluation is appended to runs/<name>/evaluations.jsonl and reused on resume.

Run:
  python -m chestforge.optimize.optimizer --name demo --method all --trials 150 --rounds 10
Outputs in runs/<name>/: evaluations.jsonl, pareto.csv, convergence.png, pareto.png,
top5.md (full ND200 reports at mc_samples_final), top5/*.yaml designs, fea_validation_queue.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import optuna
import yaml

from chestforge import SIM_LABEL
from chestforge.optimize import plots, surrogate
from chestforge.optimize.objectives import OBJECTIVE_UNITS, Evaluation, evaluate
from chestforge.optimize.search_space import describe, load_optimizer_config, suggest, to_dict, to_protector

RUNS = Path(__file__).resolve().parent.parent.parent / "runs"
BIG = 1e9


class EvalStore:
    """Evaluation cache + append-only log."""

    def __init__(self, path: Path, cfg: dict):
        self.path, self.cfg = path, cfg
        self.evals: list[Evaluation] = []
        self.method: list[str] = []
        self._by_key: dict[str, Evaluation] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                ev = Evaluation(**rec["evaluation"])
                self._add(ev, rec["method"])

    @staticmethod
    def key(p: dict) -> str:
        return json.dumps(p, sort_keys=True)

    def _add(self, ev: Evaluation, method: str):
        self.evals.append(ev)
        self.method.append(method)
        self._by_key[self.key(ev.params)] = ev

    def get(self, p: dict, method: str) -> Evaluation:
        k = self.key(p)
        if k in self._by_key:
            return self._by_key[k]
        ev, _ = evaluate(p, self.cfg)
        self._add(ev, method)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"method": method, "evaluation": ev.to_json()}) + "\n")
        return ev


def _constraint(ev: Evaluation, cfg: dict) -> float:
    m = ev.min_p95_margin_pct
    return cfg["margin_target_pct"] - m if math.isfinite(m) else BIG


def _values(ev: Evaluation, names: list[str]) -> list[float]:
    return [ev.objective(n) if math.isfinite(ev.objective(n)) else BIG for n in names]


def _study(sampler, directions) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(directions=directions, sampler=sampler)


def run_tpe(store: EvalStore, cfg: dict, trials: int, objective: str = "mass_g") -> optuna.Study:
    """Single objective (default: mass) subject to the P95 margin constraint."""
    def obj(trial):
        ev = store.get(suggest(trial, cfg), "tpe")
        trial.set_constraint("p95_margin", _constraint(ev, cfg))
        return _values(ev, [objective])[0]

    sampler = optuna.samplers.TPESampler(seed=cfg["seed"], n_startup_trials=min(20, trials))
    st = _study(sampler, ["minimize"])
    st.optimize(obj, n_trials=trials)
    return st


def run_nsga2(store: EvalStore, cfg: dict, trials: int) -> optuna.Study:
    names = cfg["objectives"]

    def obj(trial):
        ev = store.get(suggest(trial, cfg), "nsga2")
        trial.set_constraint("p95_margin", _constraint(ev, cfg))
        return _values(ev, names)

    sampler = optuna.samplers.NSGAIISampler(seed=cfg["seed"], population_size=min(40, max(trials // 4, 8)))
    st = _study(sampler, ["minimize"] * len(names))
    st.optimize(obj, n_trials=trials)
    return st


def run_surrogate(store: EvalStore, cfg: dict, rounds: int, out: Path) -> dict | None:
    """Active learning on a GP surrogate; needs cfg.surrogate.min_training evaluations first."""
    need = cfg["surrogate"]["min_training"]
    if len(store.evals) < need:
        print(f"surrogate: {len(store.evals)} evaluations < min_training {need}; skipped")
        return None
    rng = np.random.default_rng(cfg["seed"] + len(store.evals))
    info = {}
    for r in range(rounds):
        sur = surrogate.fit(store.evals, cfg)
        proposals, queue = surrogate.propose(sur, store.evals, cfg, rng)
        before = sum(e.feasible for e in store.evals)
        for p in proposals:
            store.get(p, "surrogate")
        info = {"round": r + 1, "cv_r2": sur.cv_r2, "new_feasible": sum(e.feasible for e in store.evals) - before}
        print(f"surrogate round {r + 1}/{rounds}: CV R² {sur.cv_r2}, +{info['new_feasible']} feasible")
        (out / "fea_validation_queue.json").write_text(json.dumps(queue, indent=1))
    return info


def pareto_front(evals: list[Evaluation], names: list[str]) -> list[Evaluation]:
    feas = [e for e in evals if e.feasible]
    F = np.array([[e.objective(n) for n in names] for e in feas]).reshape(len(feas), len(names))
    keep = []
    for i in range(len(feas)):
        dominated = np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1))
        if not dominated:
            keep.append(feas[i])
    return keep


def pick_top5(front: list[Evaluation], names: list[str]) -> list[tuple[str, Evaluation]]:
    """Knee (best normalised sum) plus the best of each objective, then next-best knees."""
    if not front:
        return []
    F = np.array([[e.objective(n) for n in names] for e in front])
    lo, hi = F.min(axis=0), F.max(axis=0)
    Fn = (F - lo) / np.where(hi > lo, hi - lo, 1.0)
    order = list(np.argsort(Fn.sum(axis=1)))
    picks = [("balanced (knee)", order[0])] + [(f"lowest {n}", int(np.argmin(F[:, k]))) for k, n in enumerate(names)]
    out, used = [], set()
    for why, i in picks + [("balanced runner-up", j) for j in order[1:]]:
        if i not in used:
            used.add(i)
            out.append((why, front[i]))
        if len(out) == 5:
            break
    return out


def write_outputs(store: EvalStore, cfg: dict, out: Path, objective: str) -> list[tuple[str, Evaluation]]:
    names = cfg["objectives"]
    front = pareto_front(store.evals, names)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pareto.csv", "w", newline="") as f:
        w = csv.writer(f)
        pkeys = sorted(front[0].params) if front else []
        w.writerow(["design", *names, "min_p95_margin_pct", "p95_30mph_lbf", *pkeys])
        for e in front:
            w.writerow([describe(e.params), *[f"{e.objective(n):.4g}" for n in names], f"{e.min_p95_margin_pct:.2f}",
                        f"{e.p95_30mph_lbf:.2f}", *[e.params[k] for k in pkeys]])
    plots.convergence(store.evals, objective, out / "convergence.png")
    plots.pareto(store.evals, front, out / "pareto.png")

    top = pick_top5(front, names)
    (out / "top5").mkdir(exist_ok=True)
    lines = [f"# Top designs — {SIM_LABEL}", "",
             f"{len(store.evals)} evaluations, {sum(e.feasible for e in store.evals)} feasible, "
             f"{len(front)} on the Pareto front. Constraint: all 6 ND200 impacts pass at P95 with "
             f">= {cfg['margin_target_pct']}% margin (torso {cfg['torso_length_in']} in, {cfg['projectile']}).", ""]
    from chestforge.core.nd200_runner import run_nd200
    for k, (why, e) in enumerate(top, 1):
        rep = run_nd200(to_protector(e.params, cfg), cfg["projectile"], cfg["torso_length_in"],
                        mc_samples=cfg["mc_samples_final"], seed=cfg["seed"] + 1000)
        (out / "top5" / f"design_{k}.yaml").write_text(yaml.safe_dump(to_dict(e.params, cfg), sort_keys=False))
        lines += [f"## {k}. {why}: {describe(e.params)}", "",
                  " | ".join(f"{n} {e.objective(n):.3g} {OBJECTIVE_UNITS[n]}" for n in names), "",
                  "```", rep.to_text(), "```", ""]
    (out / "top5.md").write_text("\n".join(lines))
    return top


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ChestForge design optimizer")
    ap.add_argument("--name", default="run")
    ap.add_argument("--method", choices=["tpe", "nsga2", "surrogate", "all"], default="all")
    ap.add_argument("--trials", type=int, default=100, help="trials per Optuna method")
    ap.add_argument("--rounds", type=int, default=10, help="surrogate active-learning rounds")
    ap.add_argument("--objective", default="mass_g", choices=list(OBJECTIVE_UNITS), help="TPE single objective")
    args = ap.parse_args(argv)

    cfg = load_optimizer_config()
    out = RUNS / args.name
    store = EvalStore(out / "evaluations.jsonl", cfg)
    print(f"{SIM_LABEL}\nrun '{args.name}': {len(store.evals)} evaluations loaded")
    if args.method in ("tpe", "all"):
        run_tpe(store, cfg, args.trials, args.objective)
        print(f"TPE done: {len(store.evals)} evaluations")
    if args.method in ("nsga2", "all"):
        run_nsga2(store, cfg, args.trials)
        print(f"NSGA-II done: {len(store.evals)} evaluations")
    if args.method in ("surrogate", "all"):
        run_surrogate(store, cfg, args.rounds, out)
    top = write_outputs(store, cfg, out, args.objective)
    print(f"\n{sum(e.feasible for e in store.evals)} feasible of {len(store.evals)}; results in {out}")
    for k, (why, e) in enumerate(top, 1):
        print(f"  {k}. [{why}] {describe(e.params)}: P95@50 {e.p95_50mph_lbf:.0f} lbf, {e.mass_g:.0f} g, "
              f"{e.thickness_mm:.1f} mm, margin {e.min_p95_margin_pct:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
