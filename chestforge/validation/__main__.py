"""Experimental calibration CLI.

  python -m chestforge.validation template                       # write a CSV template
  python -m chestforge.validation validate data/experimental/*.csv
  python -m chestforge.validation fit-nts data/experimental/bare_*.csv [--apply]
  python -m chestforge.validation fit-material eva_100 data/experimental/drop_*.csv [--as eva_100_lab] [--apply]

Outputs (tables + overlay PNG) go to runs/validation/. --apply writes config/nts_fitted.yaml or
config/materials_fitted.yaml (fits from synthetic traces are refused).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from chestforge import SIM_LABEL
from chestforge.validation import fit as F
from chestforge.validation.plots import overlays
from chestforge.validation.traces import load_trace, write_trace

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs" / "validation"


def _load(patterns: list[str]):
    paths = sorted({p for pat in patterns for p in (glob.glob(pat) or [pat])})
    traces = [load_trace(p) for p in paths]
    for t in traces:
        for w in t.warnings:
            print(f"WARNING {t.name}: {w}")
        if t.synthetic:
            print(f"NOTE {t.name}: SYNTHETIC trace (generated, not measured)")
    return traces


def _table(rows: list[dict]) -> str:
    head = f"{'trace':38s} {'peak meas':>9s} {'peak sim':>9s} {'peak err':>9s} {'FWHM m/s (ms)':>14s} {'NRMSE':>7s} {'R2':>6s}"
    lines = [head]
    for r in rows:
        lines.append(f"{r['trace'][:38]:38s} {r['peak_exp_lbf']:9.1f} {r['peak_sim_lbf']:9.1f} {r['peak_err_pct']:+8.1f}% "
                     f"{r['fwhm_exp_ms']:6.2f}/{r['fwhm_sim_ms']:<6.2f} {r['nrmse_pct']:6.1f}% {r['r2']:6.3f}")
    return "\n".join(lines)


def _summary(s: dict) -> str:
    return (f"n={s['n']}  mean |peak err| {s['mean_abs_peak_err_pct']:.2f}%  mean NRMSE {s['mean_abs_nrmse_pct']:.2f}%  "
            f"mean |impulse err| {s['mean_abs_impulse_err_pct']:.2f}%  mean R² {s['mean_r2']:.4f}")


def cmd_validate(args) -> int:
    traces = _load(args.files)
    res = F.validate(traces)
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for t, r in zip(traces, res["rows"]):
        ts, fs = F.simulate_trace(t, F._nts_for(t))
        items.append({"name": t.name, "t_exp": t.t_s, "f_exp": t.force_N, "t_sim": ts, "f_sim": fs, "metrics": r})
    overlays(items, OUT / "validation.png", "Model vs measured (onset-aligned, CFC120)")
    (OUT / "validation.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"{SIM_LABEL}\n{_table(res['rows'])}\n\n{_summary(res['summary'])}\nplots: {OUT / 'validation.png'}")
    return 0


def _report_fit(fit, traces, stack_for) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{SIM_LABEL}\nfit ({fit.kind}) on {len(fit.traces)} trace(s); {fit.extra['nfev']} evaluations")
    print(f"  before: {_summary(fit.before)}\n  after:  {_summary(fit.after)}")
    print("\n" + _table(fit.extra["rows_after"]))


def cmd_fit_nts(args) -> int:
    traces = _load(args.files)
    fit = F.fit_nts(traces)
    _report_fit(fit, traces, None)
    for k, v in fit.params.items():
        print(f"  {k:28s} {v:.5g}")
    print("  (stiffness is re-set by the Appendix 1 calibration when applied; damping and surface mass are kept)")
    if args.apply:
        print(F.apply_nts(fit, allow_synthetic=args.allow_synthetic))
    return 0


def cmd_fit_material(args) -> int:
    traces = _load(args.files)
    fit = F.fit_material(traces, args.material, args.new_id)
    _report_fit(fit, traces, None)
    for k, v in fit.params["multipliers"].items():
        print(f"  x {v:6.3f}  {k}")
    if fit.params["correlation"]:
        print(f"  NOTE: strongly correlated (not separately identifiable from these traces): {fit.params['correlation']}")
    if args.apply:
        print(F.apply_material(fit, allow_synthetic=args.allow_synthetic))
    return 0


def cmd_template(args) -> int:
    import numpy as np
    path = Path(args.out)
    t = np.arange(0, 0.002, 1e-4)
    write_trace(path, t, np.zeros_like(t), {
        "rig": "nts", "projectile": "baseball", "velocity_mph": 30.0, "temperature_F": 72, "stack": "pc:3, eva_100:15",
        "filtered": "false", "synthetic": "false", "source": "lab / rig, date, sample id"})
    print(f"template written to {path} — replace the data rows with your trace (see chestforge/validation/traces.py)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m chestforge.validation", description="Experimental calibration")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("files", nargs="+"); v.set_defaults(fn=cmd_validate)
    n = sub.add_parser("fit-nts"); n.add_argument("files", nargs="+"); n.set_defaults(fn=cmd_fit_nts)
    m = sub.add_parser("fit-material"); m.add_argument("material"); m.add_argument("files", nargs="+")
    m.add_argument("--as", dest="new_id"); m.set_defaults(fn=cmd_fit_material)
    for p in (n, m):
        p.add_argument("--apply", action="store_true", help="write the fit into config/*_fitted.yaml")
        p.add_argument("--allow-synthetic", action="store_true", help=argparse.SUPPRESS)
    t = sub.add_parser("template"); t.add_argument("--out", default=str(ROOT / "data" / "experimental" / "TEMPLATE.csv"))
    t.set_defaults(fn=cmd_template)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except F.SyntheticDataError as e:
        print(f"NOT APPLIED: {e}. Synthetic examples demonstrate the workflow; fit your measured traces instead.")
        return 2
    except ValueError as e:
        print(f"ERROR: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
