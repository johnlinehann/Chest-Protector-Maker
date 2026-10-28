import math

import numpy as np
import pytest

from chestforge.core.layout import protector_from_dict
from chestforge.optimize import optimizer, surrogate
from chestforge.optimize.objectives import Evaluation, evaluate, protector_mass_g, stack_bending_Nm
from chestforge.optimize.search_space import (cardiac_stack, canonical, features, load_optimizer_config, mutate,
                                              random_design, to_dict, to_protector)


@pytest.fixture(scope="module")
def cfg():
    return load_optimizer_config()


def _design(**kw):
    p = {"plate_material": "pc", "plate_mm": 3.0, "plate_a_in": 4.0, "plate_b_in": 4.5,
         "outer_material": "none", "outer_mm": 5.0, "inner_material": "eva_100", "inner_mm": 15.0}
    p.update(kw)
    return canonical(p)


def test_canonical_drops_absent_outer_foam():
    assert _design()["outer_mm"] == 0.0
    assert _design(outer_material="pe_45")["outer_mm"] == 5.0


def test_random_and_mutated_designs_stay_in_space(cfg):
    rng = np.random.default_rng(0)
    sp = cfg["search_space"]
    for _ in range(200):
        p = mutate(random_design(rng, cfg), rng, cfg, 3)
        assert sp["plate"]["thickness_mm"][0] <= p["plate_mm"] <= sp["plate"]["thickness_mm"][1]
        assert sp["inner_foam"]["thickness_mm"][0] <= p["inner_mm"] <= sp["inner_foam"]["thickness_mm"][1]
        assert p["plate_material"] in sp["plate"]["materials"]
        assert len(features(p, cfg)) == len(features(_design(), cfg))
        assert p["outer_material"] == "none" or p["outer_mm"] > 0
        cardiac_stack(p)   # always buildable


def test_every_design_covers_silhouette_plus_shift(cfg):
    from chestforge.config import load_nd200
    shift = load_nd200().max_shift_in(cfg["torso_length_in"])
    rng = np.random.default_rng(1)
    for _ in range(30):
        assert to_protector(random_design(rng, cfg), cfg).covers_silhouette(shift)


def test_design_yaml_round_trip(cfg):
    p = _design(outer_material="pe_45")
    prot = protector_from_dict(to_dict(p, cfg))
    assert prot.zones[0].stack == to_protector(p, cfg).zones[0].stack


def test_mass_and_bending_by_hand(cfg):
    p = _design()
    a_in, b_in = 4.0, 4.5                     # plate
    za, zb = 1.5 + 2.0 + 0.25, 1.73 + 2.0 + 0.25   # zone = silhouette half-axes + shift + clearance (> plate? no)
    za, zb = max(a_in, za) + 0.0, max(b_in, zb)
    za, zb = max(4.0, 1.5 + 2.0) + 0.25, max(4.5, 1.73 + 2.0) + 0.25
    in2 = 0.0254**2
    plate = 1200 * 0.003 * math.pi * a_in * b_in * in2
    foam = 100 * 0.015 * math.pi * za * zb * in2
    torso = 100 * 0.010 * (math.pi * 7 * 9 - math.pi * za * zb) * in2
    assert protector_mass_g(p, cfg) == pytest.approx((plate + foam + torso) * 1000, rel=1e-6)
    stack = cardiac_stack(p)
    expected = stack[0].bending_stiffness + 2.0e6 * 0.015**3 / 12
    assert stack_bending_Nm(stack) == pytest.approx(expected)


def test_real_evaluation(calibrated, cfg):
    ev, rep = evaluate(_design(plate_mm=4.0, inner_material="poron_xrd_240"), cfg, mc_samples=8)
    assert rep is not None and len(rep.impacts) == 6
    assert ev.p95_50mph_lbf == max(i.p95_lbf for i in rep.impacts if i.condition == "50mph")
    assert ev.feasible == (ev.min_p95_margin_pct >= cfg["margin_target_pct"])


def _fake_eval(p, cfg, mc_samples=None):
    """Smooth synthetic physics: thicker plate/foam -> lower load, more mass."""
    load = 260 - 25 * p["plate_mm"] - 2.5 * p["inner_mm"] - 1.0 * p["outer_mm"] + 4 / min(p["plate_a_in"], p["plate_b_in"])
    margin = (180 - load) / 180 * 100
    stack = cardiac_stack(p)
    ev = Evaluation(params=p, p95_50mph_lbf=load, p95_30mph_lbf=load / 2, min_p95_margin_pct=margin,
                    min_nominal_margin_pct=margin + 2, mass_g=protector_mass_g(p, cfg),
                    thickness_mm=sum(l.thickness_m for l in stack) * 1000, bending_Nm=stack_bending_Nm(stack),
                    feasible=margin >= cfg["margin_target_pct"], bottomed=False, flags=[])
    return ev, None


def test_tpe_respects_constraint_and_store_resumes(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(optimizer, "evaluate", _fake_eval)
    store = optimizer.EvalStore(tmp_path / "e.jsonl", cfg)
    st = optimizer.run_tpe(store, cfg, 40, "mass_g")
    best = st.best_trial
    assert all(v <= 0 for v in dict(best.constraints).values())   # best trial is feasible
    assert len(store.evals) <= 40
    again = optimizer.EvalStore(tmp_path / "e.jsonl", cfg)
    assert len(again.evals) == len(store.evals)
    assert [e.params for e in again.evals] == [e.params for e in store.evals]


def test_nsga2_runs_multiobjective(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(optimizer, "evaluate", _fake_eval)
    store = optimizer.EvalStore(tmp_path / "e.jsonl", cfg)
    optimizer.run_nsga2(store, cfg, 32)
    assert any(e.feasible for e in store.evals)


def _ev(load, mass, thick=10.0, bend=1.0, feasible=True):
    return Evaluation(params={"id": (load, mass, thick, bend)}, p95_50mph_lbf=load, p95_30mph_lbf=0,
                      min_p95_margin_pct=30, min_nominal_margin_pct=30, mass_g=mass, thickness_mm=thick,
                      bending_Nm=bend, feasible=feasible, bottomed=False, flags=[])


def test_pareto_front_and_top5():
    names = ["p95_50mph_lbf", "mass_g"]
    evs = [_ev(100, 300), _ev(80, 400), _ev(120, 250), _ev(110, 350), _ev(50, 100, feasible=False)]
    front = optimizer.pareto_front(evs, names)
    assert {(e.p95_50mph_lbf, e.mass_g) for e in front} == {(100, 300), (80, 400), (120, 250)}
    top = optimizer.pick_top5(front, names)
    assert len(top) == 3 and len({id(e) for _, e in top}) == 3
    assert top[0][0] == "balanced (knee)"


def test_surrogate_learns_and_proposes(cfg):
    rng = np.random.default_rng(3)
    evals = [_fake_eval(random_design(rng, cfg), cfg)[0] for _ in range(80)]
    sur = surrogate.fit(evals, cfg, cv_folds=4)
    assert sur.cv_r2["p95_50mph_lbf"] > 0.9
    small = dict(cfg, surrogate=dict(cfg["surrogate"], candidates=400, batch=5))
    proposals, queue = surrogate.propose(sur, evals, small, rng)
    seen = {optimizer.EvalStore.key(e.params) for e in evals}
    assert len(proposals) == 5 and not ({optimizer.EvalStore.key(p) for p in proposals} & seen)
    true_margin = [_fake_eval(p, cfg)[0].min_p95_margin_pct for p in proposals]
    assert np.mean(np.array(true_margin) >= cfg["margin_target_pct"]) >= 0.6   # mostly feasible picks
    for q in queue:
        assert abs(q["pred_margin_pct"] - cfg["margin_target_pct"]) < 2 * q["std_pct"]
