"""SAE J211/1 Channel Frequency Class (CFC) filter — ND200 §9.6 requires CFC120.

J211/1 Appendix C: 2-pole Butterworth run forward then backward (phaseless, 4-pole response),
with the design frequency scaled by 2.0775 so the -3 dB point lands at ~1.65 x CFC.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import filtfilt


def cfc_coefficients(cfc: float, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """(b, a) for one pass of the J211 CFC filter."""
    t = 1.0 / sample_rate_hz
    wd = 2.0 * math.pi * cfc * 2.0775
    wa = math.sin(wd * t / 2.0) / math.cos(wd * t / 2.0)
    den = 1.0 + math.sqrt(2.0) * wa + wa**2
    a0 = wa**2 / den
    b1 = -2.0 * (wa**2 - 1.0) / den
    b2 = (-1.0 + math.sqrt(2.0) * wa - wa**2) / den
    return np.array([a0, 2 * a0, a0]), np.array([1.0, -b1, -b2])


def cfc_filter(x: np.ndarray, sample_rate_hz: float, cfc: float = 120) -> np.ndarray:
    """Phaseless CFC filter along the last axis. Ends are padded by odd reflection."""
    if sample_rate_hz < 10 * cfc:
        raise ValueError(f"sample rate {sample_rate_hz} Hz too low for CFC{cfc}")
    b, a = cfc_coefficients(cfc, sample_rate_hz)
    x = np.asarray(x, dtype=float)
    return filtfilt(b, a, x, axis=-1, padtype="odd", padlen=min(x.shape[-1] - 1, 300))
