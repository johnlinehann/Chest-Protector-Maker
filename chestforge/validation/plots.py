"""Sim-vs-test overlay small multiples (PNG). Measured = ink, model = blue, fitted model = orange."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chestforge import SIM_LABEL  # noqa: E402
from chestforge.units import LBF_TO_N  # noqa: E402

INK, INK2, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"


def overlays(items: list[dict], path: Path, title: str) -> None:
    """items: {name, t_exp, f_exp, t_sim, f_sim, [t_fit, f_fit], metrics} — forces in N, times in s (onset-aligned)."""
    n = len(items)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.2 * rows), squeeze=False, facecolor="white")
    for ax, it in zip(axes.flat, items):
        ax.plot(it["t_exp"] * 1000, it["f_exp"] / LBF_TO_N, color=INK, lw=2.2, label="measured")
        ax.plot(it["t_sim"] * 1000, it["f_sim"] / LBF_TO_N, color=BLUE, lw=1.6, label="model")
        if "t_fit" in it:
            ax.plot(it["t_fit"] * 1000, it["f_fit"] / LBF_TO_N, color=ORANGE, lw=1.6, ls="--", label="fitted model")
        m = it["metrics"]
        ax.set_title(f"{it['name']}\npeak {m['peak_err_pct']:+.1f}%  NRMSE {m['nrmse_pct']:.1f}%", fontsize=8.5, loc="left")
        t_end = 3 * m["fwhm_exp_ms"] + 2
        ax.set_xlim(-1, t_end)
        ax.set_xlabel("time from onset (ms)", fontsize=8, color=INK2)
        ax.set_ylabel("force (lbf)", fontsize=8, color=INK2)
        ax.tick_params(labelsize=7.5, colors=INK2)
        ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.005, SIM_LABEL, fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)
