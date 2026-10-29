"""Load and validate the YAML configs. Spec values are read from here, never hard-coded."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class Range(_Model):
    min: float
    max: float
    source: str | None = None

    @model_validator(mode="after")
    def _ordered(self):
        if self.min > self.max:
            raise ValueError(f"range min {self.min} > max {self.max}")
        return self

    def contains(self, x: float) -> bool:
        return self.min <= x <= self.max

    @property
    def mid(self) -> float:
        return 0.5 * (self.min + self.max)


class PerformanceLimit(_Model):
    name: str
    velocity_mph: float
    velocity_ms_nominal: float
    max_peak_lbf: float
    max_peak_N: float
    source: str


class ShiftBand(_Model):
    torso_length_in: dict[str, float]
    max_shift_in: float
    source: str

    def matches(self, torso_in: float) -> bool:
        b = self.torso_length_in
        return (
            ("min_inclusive" not in b or torso_in >= b["min_inclusive"])
            and ("min_exclusive" not in b or torso_in > b["min_exclusive"])
            and ("max_inclusive" not in b or torso_in <= b["max_inclusive"])
            and ("max_exclusive" not in b or torso_in < b["max_exclusive"])
        )


class ND200(_Model):
    """Typed view of config/nd200.yaml."""

    raw: dict
    performance: list[PerformanceLimit]
    shift_bands: list[ShiftBand]

    @classmethod
    def from_dict(cls, d: dict) -> "ND200":
        return cls(
            raw=d,
            performance=d["performance"],
            shift_bands=d["random_location"]["shift_bands"],
        )

    @property
    def velocity_tolerance_pct(self) -> float:
        return self.raw["velocity_tolerance_pct"]["value"]

    def limit(self, name: str) -> PerformanceLimit:
        for p in self.performance:
            if p.name == name:
                return p
        raise KeyError(f"no ND200 performance condition {name!r}")

    def max_shift_in(self, torso_length_in: float) -> float:
        """§12.2.1-12.2.3: max protector shift for an intended torso length."""
        if torso_length_in <= 0:
            raise ValueError("torso length must be positive")
        hits = [b for b in self.shift_bands if b.matches(torso_length_in)]
        if len(hits) != 1:
            raise ValueError(f"torso length {torso_length_in} in matched {len(hits)} shift bands")
        return hits[0].max_shift_in

    @property
    def calibration(self) -> dict:
        return self.raw["calibration"]

    @property
    def min_sample_rate_hz(self) -> float:
        return self.raw["data_acquisition"]["min_sample_rate_hz"]["value"]

    @property
    def filter_cfc(self) -> int:
        return int(self.raw["data_acquisition"]["filter_cfc"]["value"])

    def projectile_spec(self, name: str) -> dict:
        return self.raw["projectiles"][name]


def _read(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


@lru_cache
def load_nd200() -> ND200:
    return ND200.from_dict(_read("nd200.yaml"))


@lru_cache
def load_projectiles() -> dict:
    return _read("projectiles.yaml")


NTS_FITTED = "nts_fitted.yaml"           # written by chestforge.validation (fit to measured traces)
MATERIALS_FITTED = "materials_fitted.yaml"


@lru_cache
def load_nts() -> dict:
    """nts.yaml with any fitted overrides (config/nts_fitted.yaml) merged over the estimates."""
    d = _read("nts.yaml")
    f = CONFIG_DIR / NTS_FITTED
    if f.exists():
        for k, v in (yaml.safe_load(f.read_text()) or {}).get("overrides", {}).items():
            if k not in d:
                raise ValueError(f"{NTS_FITTED}: unknown key {k!r}")
            d[k] = v
    return d


@lru_cache
def load_sizes() -> dict:
    return _read("sizes.yaml")


def config_hash(*names: str) -> str:
    """Hash of config files; a calibration is invalidated when any of them changes."""
    h = hashlib.sha256()
    for n in names:
        f = CONFIG_DIR / n
        h.update(n.encode() + (f.read_bytes() if f.exists() else b"<absent>"))
    return h.hexdigest()[:16]
