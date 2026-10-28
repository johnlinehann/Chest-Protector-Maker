"""Static PNG charts for optimizer runs (the interactive versions arrive with the UI, Milestone 6).

Palette: reference categorical slots 1-2 (blue #2a78d6, orange #eb6834), validated with the dataviz
skill's validator (CVD dE 24.7, normal dE 33.6, contrast >= 3:1 on #fcfcfb). One y-axis per chart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from chestforge import SIM_LABEL  # noqa: E402
from chestforge.optimize.objectives import OBJECTIVE_UNITS, Evaluation  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LABELS = {"p95_50mph_lbf": "P95 cardiac load @ 50 mph", "mass_g": "Protector mass",
          "thickness_mm": "Cardiac-zone thickness", "bending_Nm": "Bending stiffness (mobility proxy)"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def convergence(evals: list[Evaluation], objective: str, path: Path) -> None:
    """Best feasible objective so far vs evaluation number."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    _style(ax)
    x = np.arange(1, len(evals) + 1)
    vals = np.array([e.objective(objective) if e.feasible else np.nan for e in evals])
    best = np.fmin.accumulate(np.where(np.isnan(vals), np.inf, vals))
    best[np.isinf(best)] = np.nan
    ax.scatter(x, vals, s=10, color=GRID, edgecolor=INK2, linewidth=0.3, label="feasible evaluation")
    ax.step(x, best, where="post", color=BLUE, linewidth=2, label="best so far")
    ax.set_xlabel("Evaluation #", color=INK2)
    ax.set_ylabel(f"{LABELS[objective]} ({OBJECTIVE_UNITS[objective]})", color=INK2)
    ax.set_title(f"Convergence — {LABELS[objective]}", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.text(0.01, 0.01, SIM_LABEL, color=INK2, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pareto(evals: list[Evaluation], front: list[Evaluation], path: Path) -> None:
    """Small multiples: each objective vs mass; feasible designs and the Pareto set."""
    others = ["p95_50mph_lbf", "thickness_mm", "bending_Nm"]
    feas = [e for e in evals if e.feasible]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=SURFACE)
    for ax, name in zip(axes, others):
        _style(ax)
        ax.scatter([e.mass_g for e in feas], [e.objective(name) for e in feas], s=12, color=BLUE, alpha=0.45,
                   edgecolor="none", label="feasible")
        ax.scatter([e.mass_g for e in front], [e.objective(name) for e in front], s=36, color=ORANGE,
                   edgecolor=SURFACE, linewidth=1.5, label="Pareto set (4 objectives)")
        ax.set_xlabel("Protector mass (g)", color=INK2)
        ax.set_ylabel(f"{LABELS[name]} ({OBJECTIVE_UNITS[name]})", color=INK2)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle(f"Pareto front — {len(feas)} feasible of {len(evals)} evaluated "
                 f"(all 6 impacts pass at P95 with margin)", color=INK, x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.01, SIM_LABEL, color=INK2, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
