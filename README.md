# ChestForge

Simulate and optimize catcher chest-protector padding against **NOCSAE DOC (ND) 200-22m25** (commotio cordis).

> **Simulated — not NOCSAE certified.** Certification requires accredited lab testing and SEI certification.

## Quick start
```bash
pip install -e .[dev]
python -m chestforge.core.calibration   # fit + verify the NTS model (required before any design run)
pytest -q
```

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

## ND200 runner (Milestone 2)
```bash
python -m chestforge.core.nd200_runner --material poron_xrd_240 --thickness-mm 30 --torso-in 15
```
- Single foam layer: Avalle (2001) curve × Johnson-Cook log rate factor, smooth hysteretic unloading, bottoming flag at densification strain.
- Ball indents the pad (sphere footprint); force spreads through remaining thickness; only the share on the cardiac silhouette loads the load cell.
- Runs all 6 Table 1 impacts, random shifts at the §12.2 limit for the torso length (≥ 1 in apart) → PASS/FAIL + margin per impact.
- Starter materials (EVA, PE, PORON-type) are **estimated**, not measured.

See `docs/PLAN.md` for milestones 3–7.
