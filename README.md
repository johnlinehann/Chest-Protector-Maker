# ChestForge

Simulate and optimize catcher chest-protector padding against **NOCSAE DOC (ND) 200-22m25** (commotio cordis).

> **Simulated — not NOCSAE certified.** Certification requires accredited lab testing and SEI certification.

## Quick start
```bash
pip install -e .[dev,cad]
chestforge                               # calibrates if needed, then opens the UI at http://127.0.0.1:8765
pytest -q
```

## Web UI (Milestone 6)
- **3D viewer** (Three.js, vendored — works offline): protector layers on a chest model, cardiac silhouette (red) and max-shift radius (green, dashed); hover a layer for material and mass.
- **Layer-stack editor:** zones, layers (ball side first), drag ⋮⋮ or focus it and press ↑/↓ to reorder, plate extents.
- **Run ND200:** 6-impact report card — nominal and P95 verdicts, margins, flags, CFC120 force-time charts with hover values.
- **Optimizer tab:** pick a run or start one, live Pareto chart (any two objectives), click a point to load the design.
- **Exports:** STEP/STL/nTop zip, test-style PDF report, CSV of traces, design JSON (loads with `--design`).
- Server: `python -m chestforge.server` (localhost only). `chestforge --no-ui` shows the status console only.

## Desktop icon
After the quick start, run once with the same Python:
```bash
python scripts/install_desktop_shortcut.py
```
Creates `ChestForge.lnk` (Windows), `ChestForge.app` (macOS) or `ChestForge.desktop` (Linux) on your desktop. It opens a terminal running `chestforge` (auto-calibrates if needed). Icon source: `scripts/make_icon.py`.

## Config (never hard-code spec values)
| File | Contents |
|---|---|
| `config/nd200.yaml` | The standard only — every value cites its ND200 section, `verified: true` |
| `config/projectiles.yaml` | Ball model params (COR, contact exponent); non-ND200 values `estimated: true` |
| `config/nts.yaml` | NTS 3-DOF model priors (all estimated) + solver settings |
| `config/sizes.yaml` | Your product sizes / intended torso lengths (example values) |
| `config/nts_calibrated.yaml` | Generated; hash-locked to the configs above |

## NTS model (Milestone 1)
`ball m1 → Hunt-Crossley contact → loading surface m2 → [k2, c2] load cell → back plate + table m3 (free)`

- k2 fitted so bare 30 mph baseball = 140 lbf (App. 1 Step 3); lacrosse gets a DAQ gain (App. 1 NOTE).
- Verified at the worst edge of each ±0.2 mph window (Steps 3/5/6) and §10.4 system check (30 mph ±3%, 130–150 lbf).
- Output sampled at 20 kHz (≥ 10 kHz, §9.5), SAE J211 CFC120 filtered (§9.6).
- `require_calibrated()` raises `CalibrationError` if calibration is missing, stale, or failing.

## ND200 runner
```bash
python -m chestforge.core.nd200_runner --layer pc:3 --layer poron_xrd_240:15 --torso-in 15 --mc 200
python -m chestforge.core.nd200_runner --design designs/example_plate_foam.yaml --mc 200
```
- **Stack** (ball side first): foams (Avalle 2001 curve × Johnson-Cook log rate, hysteresis, temperature) and plates (PC, HDPE).
- **Load spreading:** foam cone; plates via plate-on-elastic-foundation theory (Timoshenko), reduced near plate edges (Westergaard 1926). Only the share on the cardiac silhouette loads the load cell.
- **Random locations:** every shift direction on rings from 1 in to the §12.2 limit; worst two (≥ 1 in apart) become Random #1/#2. Flags seams/plate edges inside the shift radius and checks silhouette coverage.
- **Monte Carlo (`--mc N`):** ball within §13.1 spec, ±3% velocity, 67–77 °F → P95 peak per impact.
- **Engine:** compiled RK4 (numba), ~0.17 s/impact, parallel across samples; checked against an independent scipy integration.
- Designs: YAML with zones/stacks/plate extents — see `designs/example_plate_foam.yaml`.
- All materials are **estimated**, not measured.

## Optimizer (Milestone 4)
```bash
python -m chestforge.optimize.optimizer --name run1 --method all --trials 120 --rounds 10
```
- **Design variables** (`config/optimizer.yaml`): plate material/thickness/size, optional outer foam, inner foam. Cardiac zone always covers silhouette + max shift.
- **Hard constraint:** all 6 ND200 impacts pass at **P95** with ≥ `margin_target_pct` (default 25%).
- **Objectives:** P95 load @ 50 mph, protector mass (g), cardiac-zone thickness (mm), bending stiffness (mobility proxy).
- **Methods:** Optuna TPE (single objective, default mass) → NSGA-II (Pareto) → GP surrogate active learning after ≥ 200 evaluations (also writes an FEA validation queue of uncertain designs near the constraint).
- **Outputs** in `runs/<name>/`: `evaluations.jsonl` (resumable), `pareto.csv`, `convergence.png`, `pareto.png`, `top5.md` (full ND200 reports, 200-sample MC), `top5/*.yaml` (load with `--design`).

## Geometry + export (Milestone 5)
```bash
pip install -e .[cad]          # build123d / OpenCascade (~70 MB)
python -m chestforge.geometry.export --design designs/example_plate_foam.yaml --torso-in 15
python -m chestforge.geometry.export --run run1 --rank 1      # an optimizer top-5 design
```
- Curved pads on a cylindrical chest (radius and silhouette position from ND200 Figure 1, scaled by torso length); outlines wrapped by arc length, so plate sizes match the optimizer.
- Writes to `exports/<design>/`: `protector.step` (one named solid per layer), `protector.stl` + one STL per layer, `ntop_lattice.json` (TPU gyroid/Kelvin/honeycomb relative density matching each foam's plateau — Gibson & Ashby, **estimated, not simulated**), `landmarks.json` (silhouette + shift circle in 3D for the viewer), `summary.json` (volumes, CAD mass).

See `docs/PLAN.md` for milestone 7.
