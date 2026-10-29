"""Test-style ND200 PDF report (matplotlib only, no extra dependencies)."""

from __future__ import annotations

import datetime as dt
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from chestforge import SIM_LABEL  # noqa: E402
from chestforge.core.nd200_runner import ND200Report  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # centre, random #1, random #2 (validated categorical slots 1-3)
PASS_C, FAIL_C = "#0ca30c", "#d03b3b"


def _page(fig):
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.5, 0.985, SIM_LABEL.upper() + " — certification requires accredited lab testing and SEI certification",
             ha="center", va="top", fontsize=7.5, color=FAIL_C, weight="bold")


def nd200_pdf(rep: ND200Report, metrics: dict) -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # page 1: summary + table
        fig = plt.figure(figsize=(8.5, 11))
        _page(fig)
        fig.text(0.07, 0.94, "ChestForge — ND200-22m25 simulated test report", fontsize=15, color=INK, weight="bold")
        fig.text(0.07, 0.915, dt.datetime.now().strftime("%Y-%m-%d %H:%M"), fontsize=9, color=INK2)
        verdict_c = PASS_C if rep.verdict == "PASS" else (FAIL_C if rep.verdict == "FAIL" else INK2)
        lines = [
            ("Design", rep.design),
            ("Projectile", rep.projectile + ("  (BEYOND STANDARD — not an ND200 projectile)" if rep.beyond_standard else "")),
            ("Intended torso length", f"{rep.torso_length_in} in  ->  max random shift {rep.max_shift_in} in (§12.2)"),
            ("Protector mass", "n/a" if metrics.get("mass_g") is None else f"{metrics['mass_g']:.0f} g"),
            ("Cardiac-zone thickness", f"{metrics['cardiac_thickness_mm']:.1f} mm"),
            ("Silhouette coverage", "OK" if rep.covers_silhouette else "DOES NOT cover silhouette + max shift"),
        ]
        y = 0.87
        for k, v in lines:
            fig.text(0.07, y, k, fontsize=9.5, color=INK2)
            fig.text(0.33, y, v, fontsize=9.5, color=INK)
            y -= 0.025
        fig.text(0.07, y - 0.02, "Verdict (nominal):", fontsize=12, color=INK)
        fig.text(0.33, y - 0.02, f"{rep.verdict}", fontsize=12, color=verdict_c, weight="bold")
        fig.text(0.47, y - 0.02, f"min margin {rep.min_margin_pct:.1f}%", fontsize=10, color=INK2)
        if rep.has_mc:
            fig.text(0.07, y - 0.05, "Verdict (P95):", fontsize=12, color=INK)
            fig.text(0.33, y - 0.05, "PASS" if rep.p95_passed else "FAIL", fontsize=12,
                     color=PASS_C if rep.p95_passed else FAIL_C, weight="bold")
            fig.text(0.47, y - 0.05, f"min P95 margin {rep.min_p95_margin_pct:.1f}%", fontsize=10, color=INK2)
            fig.text(0.07, y - 0.075, f"P95 over {rep.impacts[0].mc_samples} Monte Carlo samples per impact "
                     "(ball within spec, ±3% velocity, 67-77 °F)", fontsize=8.5, color=INK2)
        cols = ["Sample", "Location", "Shift (in)", "mph", "Peak lbf"] + (["P95 lbf"] if rep.has_mc else []) + \
               ["Limit", "Margin", "Result"]
        rows = []
        for i in rep.impacts:
            m = i.p95_margin_pct if rep.has_mc else i.margin_pct
            rows.append([str(i.sample), i.location, f"{i.shift_in[0]:+.2f}, {i.shift_in[1]:+.2f}", f"{i.velocity_mph:.0f}",
                         f"{i.peak_lbf:.1f}"] + ([f"{i.p95_lbf:.1f}"] if rep.has_mc else []) +
                        [f"{i.limit_lbf:.0f}", f"{m:.1f}%", "PASS" if i.passed else "FAIL"])
        ax = fig.add_axes([0.07, 0.33, 0.86, 0.3])
        ax.axis("off")
        tbl = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(GRID)
            if r == 0:
                cell.set_text_props(weight="bold", color=INK)
            elif cols[c] == "Result":
                cell.set_text_props(color=PASS_C if cell.get_text().get_text() == "PASS" else FAIL_C, weight="bold")
        notes = [f"Under impact — {i.location} S{i.sample}: {i.local}" for i in rep.impacts if i.location != "cardiac_center"]
        notes += [f"FLAG: {f}" for f in rep.flags]
        notes += [f"BOTTOMED OUT ({i.location} S{i.sample}): {', '.join(i.result.bottomed_layers)}"
                  for i in rep.impacts if i.bottomed_out]
        notes += ["Method: NOCSAE ND200-22m25 Table 1 (6 impacts), §14 limits 90 lbf @ 30 mph / 180 lbf @ 50 mph,",
                  "CFC120-filtered load-cell force, NTS model calibrated to Appendix 1 (140 ± 2 lbf bare @ 30 mph).",
                  "Random locations: worst protector shifts found by search up to the §12.2 limit, >= 1 in apart.",
                  "Material parameters are ESTIMATED unless fitted to measured data."]
        y = 0.31
        for n in notes[:16]:
            fig.text(0.07, y, n[:120], fontsize=7.5, color=INK2)
            y -= 0.018
        pdf.savefig(fig)
        plt.close(fig)

        # page 2: force-time traces
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
        _page(fig)
        for ax, cond in zip(axes, ("30mph", "50mph")):
            imps = [i for i in rep.impacts if i.condition == cond]
            for k, (color, i) in enumerate(zip(SERIES, imps)):   # 3rd series dashed: identical traces stay visible
                ax.plot(i.result.t * 1000, i.result.force_N / 4.4482216152605, color=color, lw=1.6,
                        ls="--" if k == 2 else "-", label=f"{i.location} (peak {i.peak_lbf:.0f} lbf)")
            lim = imps[0].limit_lbf
            ax.axhline(lim, color=FAIL_C, lw=1, ls="--")
            ax.text(0.5, lim, f"  limit {lim:.0f} lbf", color=FAIL_C, va="bottom", fontsize=8)
            ax.set_xlim(0, 15)
            ax.set_title(f"Sample {imps[0].sample} — {imps[0].velocity_mph:.0f} mph, CFC120 load-cell force",
                         loc="left", fontsize=11, color=INK)
            ax.set_xlabel("Time (ms)", color=INK2)
            ax.set_ylabel("Cardiac load (lbf)", color=INK2)
            ax.grid(color=GRID, lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig)
        plt.close(fig)
    return buf.getvalue()
