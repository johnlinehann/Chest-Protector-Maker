# ChestForge — Plan (Milestone 1 detail, M2–M7 outline)

## Context
- Empty repo (`claude/zen-thompson-lo9te9`, no commits). Build ChestForge: simulate + optimize catcher chest-protector padding vs NOCSAE ND200-22m25.
- M1 = config + NTS surrogate auto-calibrated to bare 140 ± 2 lbf @ 30 mph; designs refuse to run if calibration fails.
- Decisions (user): spec values verified from the NOCSAE PDF (nocsae.org must be allowed in network settings); 3-DOF lumped NTS; repo-root layout, package `chestforge`; baseball default.
- Env: Python 3.13, numpy + PyYAML + pydantic installed; PyPI reachable → add scipy, pytest.

## Layout (repo root)
```
pyproject.toml  README.md  .gitignore
config/nd200.yaml  materials.yaml (M2 stub)  sizes.yaml
chestforge/
  config.py            # pydantic-validated loaders for all YAML
  core/ projectile.py  surrogate.py  impact_sim.py  filters.py  calibration.py
  optimize/  geometry/  (M4/M5 placeholders only)
web/  data/experimental/  tests/
```

## M1 design
**config/nd200.yaml** — every value carries `value`, `units`, `source` (ND200-22m25 § ref), `verified: true/false`:
- Limits: 30 mph (13.4 m/s ±3%) ≤ 90 lbf; 50 mph (22.4 m/s ±3%) ≤ 180 lbf.
- Table 1 matrix (2 samples × {center, random1, random2}); random-shift rules (≥1 in contact spacing; shift 1.75/2.0/2.25 in by torso length bands).
- Projectiles: baseball, lacrosse (ND200); softball under `beyond_standard:` with `nd200: false`.
- NTS: bearing-table moving mass ≤ 12.5 lb; calibration 140 ± 2 lbf @30, >130 @29.1, <150 @30.9 mph.
- DAQ ≥10 kHz, CFC120; conditioning 72 ± 5 °F, ≥4 h; ≥2 samples/model.
- `sizes.yaml`: torso-length bands → max shift.

**core/projectile.py** — `Projectile` dataclass from YAML (mass, circumference → radius, C-D spec). Contact = Hunt-Crossley `F = k·δ^n (1 + 1.5·c·δ̇)`; k bounded so a quasi-static compression check lands inside the 200–300 lb @ 0.25 in band (baseball) / 115–150 lb @ 25% (lacrosse). Function `sample_within_spec(rng)` for later Monte Carlo.

**core/surrogate.py** — 3-DOF NTS:
- m1 ball → contact → m2 damped loading surface → (k2, c2) load-cell spring → m3 rigid back plate + table (≤ 12.5 lb) → table free (linear bearing).
- Load-cell force = k2·(x2−x3) + c2·(ẋ2−ẋ3).
- Free params: k2, c2, m2, surface-contact stiffness scale. Masses physically bounded.

**core/impact_sim.py** — RK45/LSODA via `scipy.integrate.solve_ivp`, output resampled at 10 kHz (configurable ≥10 kHz) → `ImpactResult(t, force_raw, force_filtered, peak_lbf, energy, max_compression)`.

**core/filters.py** — SAE J211/1 CFC filter (4-pole phaseless Butterworth via forward-backward 2-pole, J211 coefficient formulas); CFC120 default.

**core/calibration.py** — `calibrate_nts()`:
- Optimize params (scipy `least_squares`, bounded) to hit 140 lbf @ 13.41 m/s nominal, with penalty if 29.1 mph ≤ 130 or 30.9 mph ≥ 150.
- Writes `config/nts_calibrated.yaml` (params + achieved peaks + timestamp).
- `require_calibrated()` guard → raises `CalibrationError`; every design-run entry point calls it.

**Honesty**: all result objects carry `label = "Simulated — not NOCSAE certified"`; YAML values flagged `estimated`/`verified`.

## Tests (pytest)
- `test_config.py`: YAML schema; limits/velocities/tolerances; shift-limit lookup at band edges (13.5, 16.0 in).
- `test_projectile.py`: mass/circumference in spec; C-D check in spec range; softball flagged non-ND200.
- `test_filters.py`: CFC120 passes DC, −3 dB near ~200 Hz cutoff per J211 corridor, zero phase lag on a half-sine.
- `test_surrogate.py`: calibrated bare peak 140 ± 2 lbf; 29.1 > 130; 30.9 < 150; moving mass ≤ 12.5 lb; energy balance; guard raises when uncalibrated/out-of-tolerance.

## Milestones (commit + push each)
1. Config + calibrated NTS (above).
2. Single-layer pad (rate-dependent foam) + `nd200_runner` 6-impact PASS/FAIL + margin.
3. Multi-layer, plates (axisymmetric load spreading), random-shift worst case, Monte Carlo P95; `materials.yaml` with cited/estimated data.
4. Optuna TPE → NSGA-II → GP/XGBoost surrogate + active learning; zone optimization.
5. Parametric geometry + STL/STEP/nTop export.
6. Three.js UI (FastAPI backend) + PDF/CSV reports.
7. Experimental data import, material fitting, sim-vs-test dashboard.

## Verification (M1)
- `pip install -e .[dev]` → `pytest -q` all green.
- `python -m chestforge.core.calibration` prints achieved peaks at 29.1 / 30 / 30.9 mph vs targets and PASS/FAIL of calibration.
- Commit "M1: ND200 config + calibrated NTS surrogate", push to `claude/zen-thompson-lo9te9`.

## Status
- ND200 values verified against the NOCSAE PDF (Dec 2025); Figure 1 NTS geometry encoded.
- M1 complete: 44 tests passing.
- M2 complete: single-layer pad + ND200 runner, 76 tests passing. Foam-only designs (10-40 mm, 3 estimated foams) all FAIL; plates/load spreading are M3.
- M3 complete: multi-layer + plates (Winkler/Westergaard spreading), zoned layouts, worst-case shift search, Monte Carlo P95, numba RK4 engine. 100 tests.
- M4 complete: TPE / NSGA-II / GP-surrogate optimizer with P95 margin constraint, Pareto + top-5 reports.
- M5 complete: build123d geometry, STEP/STL, nTop lattice JSON, viewer landmarks.
