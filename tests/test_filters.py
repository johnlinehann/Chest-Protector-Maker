import numpy as np
import pytest
from scipy.signal import freqz

from chestforge.core.filters import cfc_coefficients, cfc_filter

FS = 20000.0


def _combined_gain_db(f_hz, cfc=120, fs=FS):
    b, a = cfc_coefficients(cfc, fs)
    _, h = freqz(b, a, worN=[f_hz], fs=fs)
    return 20 * np.log10(abs(h[0]) ** 2)   # forward + backward pass


def test_unity_dc_gain():
    assert _combined_gain_db(1e-3) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("f", [10, 50, 100, 120])
def test_passband_within_j211_corridor(f):
    # SAE J211/1 CFC120 passband to 120 Hz: +0.5 / -1 dB
    assert -1.0 <= _combined_gain_db(f) <= 0.5


def test_minus_3db_near_200hz():
    # J211/1: CFC120 has F_H = 200 Hz; the phaseless Butterworth is -3 dB at ~1.65 x CFC
    assert _combined_gain_db(200) == pytest.approx(-3.0, abs=0.5)


def test_strong_stopband_attenuation():
    assert _combined_gain_db(1000) < -30


def test_zero_phase_on_half_sine():
    t = np.arange(0, 0.05, 1 / FS)
    pulse = np.where((t > 0.01) & (t < 0.02), np.sin(np.pi * (t - 0.01) / 0.01), 0.0)
    y = cfc_filter(pulse, FS, 120)
    assert t[np.argmax(y)] == pytest.approx(t[np.argmax(pulse)], abs=2 / FS)
    assert y.max() == pytest.approx(1.0, rel=0.02)   # 50 Hz pulse barely attenuated


def test_rejects_too_low_sample_rate():
    with pytest.raises(ValueError):
        cfc_filter(np.zeros(100), 1000.0, 120)
