# Experimental data

Put measured force-time traces here (one CSV per impact). `examples/` holds **SYNTHETIC** traces
generated from known parameters (`scripts/make_synthetic_traces.py`, truth in
`examples/GROUND_TRUTH.yaml`) — they demonstrate the workflow and are never applied to the config.
`uploads/` (UI uploads) is git-ignored, so lab data is not committed by accident.

## Format
```
# chestforge-trace: 1
# rig: nts                  nts (NOCSAE thoracic surrogate) | drop_tower (pads on a rigid instrumented anvil)
# projectile: baseball      nts rig; optional ball_mass_g
# impactor_mass_kg: 2.5     drop_tower rig: hemispherical impactor (+ impactor_radius_mm, impactor_cor)
# velocity_mph: 30.1        or velocity_ms
# temperature_F: 72
# stack: pc:3, eva_100:15   ball side first, material_id:mm  (bare / empty for the bare NTS)
# force_units: lbf          lbf | N
# time_units: ms            s | ms
# filtered: false           false -> CFC120 applied on import (ND200 §9.6)
# synthetic: false
# source: lab, date, sample id
time,force
...
```
Template: `python -m chestforge.validation template`. Sample rate ≥ 10 kHz (§9.5); keep ~2 ms of pre-trigger.

## Workflow
```bash
python -m chestforge.validation validate data/experimental/*.csv                 # model vs measured
python -m chestforge.validation fit-nts data/experimental/bare_*.csv --apply     # NTS damping + surface mass
python -m chestforge.validation fit-material eva_100 data/experimental/eva_*.csv --as eva_100_lab --apply
```
- **NTS fit** writes `config/nts_fitted.yaml` (damping ratio, loading-surface mass) and re-runs the
  Appendix 1 calibration, which re-sets the skin stiffness to 140 ± 2 lbf. Delete the file to revert.
- **Material fit** adds a NEW material id to `config/materials_fitted.yaml`; the estimates stay untouched.
  Correlated parameters (not separately identifiable from your traces) are reported.
- Metrics: peak error, time-to-peak, FWHM, impulse error, NRMSE (% of measured peak), R² — on
  onset-aligned (5 % of peak), CFC120-filtered traces.
- Same features in the UI: **Validation** tab.
