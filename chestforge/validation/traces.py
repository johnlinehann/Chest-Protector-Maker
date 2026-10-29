"""Measured force-time traces: CSV format, loading, filtering, resampling and onset alignment.

Format (UTF-8 CSV, `#` metadata lines first, then a header row and two numeric columns):

    # chestforge-trace: 1
    # rig: nts                      nts = NOCSAE thoracic surrogate | drop_tower = rigid anvil load cell
    # projectile: baseball          nts rig: baseball | lacrosse | softball
    # ball_mass_g: 145.5            optional: measured ball mass
    # impactor_mass_kg: 2.5         drop_tower rig: hemispherical impactor mass / radius / COR
    # impactor_radius_mm: 50
    # velocity_mph: 30.1            or velocity_ms
    # temperature_F: 72
    # stack: pc:3, eva_100:15       layers ball side first, material:mm (empty = bare)
    # force_units: lbf              lbf | N
    # time_units: ms                s | ms
    # filtered: false               false -> CFC120 applied on import (ND200 §9.6)
    # synthetic: false              true ONLY for generated examples; fits from them cannot be applied
    # source: lab name, date, sample id ...
    time,force
    0.0,0.12
    ...
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chestforge.config import load_nd200
from chestforge.core.filters import cfc_filter
from chestforge.core.padding import layer
from chestforge.core.projectile import Projectile, load_projectile
from chestforge.units import LBF_TO_N, mph_to_ms

GRID_HZ = 20000.0          # comparison grid (matches the simulator output rate)
ONSET_FRACTION = 0.05      # t = 0 where the force first reaches 5 % of its peak
RIGS = ("nts", "drop_tower")


@dataclass
class Trace:
    path: str
    meta: dict
    t_s: np.ndarray            # aligned: 0 at onset, uniform GRID_HZ
    force_N: np.ndarray        # CFC120-filtered
    rig: str
    velocity_ms: float
    temp_F: float
    stack: tuple
    synthetic: bool
    warnings: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def peak_N(self) -> float:
        return float(self.force_N.max())

    def projectile(self) -> Projectile:
        m = self.meta
        if self.rig == "drop_tower":
            return make_impactor(float(m["impactor_mass_kg"]), float(m["impactor_radius_mm"]),
                                 float(m.get("impactor_cor", 0.9)))
        p = load_projectile(m.get("projectile", "baseball"))
        if "ball_mass_g" in m:
            from dataclasses import replace
            p = replace(p, mass_kg=float(m["ball_mass_g"]) / 1000)
        return p


def make_impactor(mass_kg: float, radius_mm: float, cor: float = 0.9) -> Projectile:
    """Rigid hemispherical drop-tower impactor as a (very stiff) projectile: k = 2e8 N/m^1.5."""
    from chestforge.config import Range as R
    k, d = 2.0e8, 1.0e-4                       # 0.1 mm per contact -> force = k d^1.5
    f_lb = k * d**1.5 / LBF_TO_N
    circ_in = 2 * math.pi * radius_mm / 25.4
    return Projectile(
        name="impactor", nd200=False, mass_kg=mass_kg, circumference_in=circ_in, cd_force_lb=f_lb,
        cd_at=2 * d / 0.0254, cd_at_units="in", contact_exponent=1.5, cor=cor,
        mass_range_g=R(min=mass_kg * 1000, max=mass_kg * 1000), circumference_range_in=R(min=circ_in, max=circ_in),
        cd_range_lb=R(min=f_lb, max=f_lb), cor_range=R(min=cor, max=cor),
    )


def parse_stack(spec: str) -> tuple:
    spec = (spec or "").strip()
    if not spec or spec.lower() in ("bare", "none"):
        return ()
    out = []
    for item in spec.split(","):
        mat, _, mm = item.strip().partition(":")
        if not mm:
            raise ValueError(f"stack item {item!r}: expected material:thickness_mm")
        out.append(layer(mat.strip(), float(mm)))
    return tuple(out)


def _bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def align(t: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Shift time so the first crossing of ONSET_FRACTION x peak is t = 0 (trigger offsets differ)."""
    k = int(np.argmax(f >= ONSET_FRACTION * f.max()))
    if k > 0:   # interpolate the crossing
        f0, f1 = f[k - 1], f[k]
        frac = (ONSET_FRACTION * f.max() - f0) / (f1 - f0) if f1 != f0 else 0.0
        return t - (t[k - 1] + frac * (t[k] - t[k - 1]))
    return t - t[0]


def load_trace(path: str | Path) -> Trace:
    path = Path(path)
    meta, rows = {}, []
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                key, sep, val = line[1:].partition(":")
                if sep:
                    meta[key.strip().lower()] = val.split("  ")[0].strip()
                continue
            rows.append(line)
    data = []
    for r in csv.reader(rows):
        try:
            data.append((float(r[0]), float(r[1])))
        except (ValueError, IndexError):
            continue   # header row
    if len(data) < 20:
        raise ValueError(f"{path.name}: fewer than 20 numeric rows")
    arr = np.array(data)
    t, f = arr[:, 0], arr[:, 1]
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{path.name}: time must increase strictly")

    rig = meta.get("rig", "nts")
    if rig not in RIGS:
        raise ValueError(f"{path.name}: rig must be one of {RIGS}")
    t = t / 1000.0 if meta.get("time_units", "s") == "ms" else t
    units = meta.get("force_units", "N")
    if units not in ("N", "lbf"):
        raise ValueError(f"{path.name}: force_units must be N or lbf")
    f = f * LBF_TO_N if units == "lbf" else f
    if "velocity_ms" in meta:
        v = float(meta["velocity_ms"])
    elif "velocity_mph" in meta:
        v = mph_to_ms(float(meta["velocity_mph"]))
    else:
        raise ValueError(f"{path.name}: velocity_mph or velocity_ms required")

    warnings = []
    fs = 1.0 / np.median(np.diff(t))
    if fs < load_nd200().min_sample_rate_hz:
        warnings.append(f"sample rate {fs:.0f} Hz below ND200 §9.5 minimum 10 kHz")
    if fs < 1200:
        raise ValueError(f"{path.name}: sample rate {fs:.0f} Hz too low for CFC120")
    grid = np.arange(t[0], t[-1], 1.0 / GRID_HZ)
    if fs >= GRID_HZ:   # filter at the native rate, then resample
        f = cfc_filter(f, fs, 120) if not _bool(meta.get("filtered", "false")) else f
        f = np.interp(grid, t, f)
    else:
        f = np.interp(grid, t, f)
        f = cfc_filter(f, GRID_HZ, 120) if not _bool(meta.get("filtered", "false")) else f
    if rig == "drop_tower":
        for k in ("impactor_mass_kg", "impactor_radius_mm"):
            if k not in meta:
                raise ValueError(f"{path.name}: drop_tower rig needs {k}")
    return Trace(
        path=str(path), meta=meta, t_s=align(grid, f), force_N=f, rig=rig, velocity_ms=v,
        temp_F=float(meta.get("temperature_f", 72.0)), stack=parse_stack(meta.get("stack", "")),
        synthetic=_bool(meta.get("synthetic", "false")), warnings=warnings,
    )


def write_trace(path: str | Path, t_s: np.ndarray, force: np.ndarray, meta: dict, time_units: str = "ms",
                force_units: str = "lbf") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tt = t_s * 1000 if time_units == "ms" else t_s
    ff = force / LBF_TO_N if force_units == "lbf" else force
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# chestforge-trace: 1\n")
        for k, v in {**meta, "time_units": time_units, "force_units": force_units}.items():
            fh.write(f"# {k}: {v}\n")
        fh.write("time,force\n")
        for a, b in zip(tt, ff):
            fh.write(f"{a:.6f},{b:.4f}\n")
