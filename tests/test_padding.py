import math

import pytest

from chestforge.core.padding import PadLayer, contact_radius, load_material, load_materials

MATS = ["eva_100", "pe_45", "poron_xrd_240"]


def test_starter_materials_present_and_flagged_estimated():
    mats = load_materials()
    assert set(MATS) <= set(mats)
    assert all(mats[m].estimated for m in MATS)   # none are fitted to measured data yet


@pytest.mark.parametrize("mid", MATS)
def test_quasi_static_curve_shape(mid):
    m = load_material(mid)
    assert m.quasi_static_stress(0.0) == 0.0
    e = 1e-5
    assert m.quasi_static_stress(e) / e == pytest.approx(m.E_Pa, rel=1e-2)   # initial slope = E
    strains = [i / 100 for i in range(1, 99)]
    s = [m.quasi_static_stress(x) for x in strains]
    assert all(b > a for a, b in zip(s, s[1:]))                              # monotonic
    assert m.quasi_static_stress(0.98) > 10 * m.quasi_static_stress(0.3)     # densification


@pytest.mark.parametrize("mid", MATS)
def test_rate_stiffening_and_hysteresis(mid):
    m = load_material(mid)
    qs = m.quasi_static_stress(0.4)
    load_fast = m.stress(0.4, 1000.0)
    unload = m.stress(0.4, -1000.0)
    assert load_fast > qs * 1.3
    assert unload == pytest.approx(qs * m.unloading_ratio, rel=1e-3)
    assert m.stress(-0.01, 0.0) == pytest.approx(-0.01 * m.E_Pa)              # linear in tension


def test_contact_radius():
    R = 0.037
    assert contact_radius(0.0, R) == 0.0
    assert contact_radius(R, R) == pytest.approx(R)
    assert contact_radius(2 * R, R) == pytest.approx(R)                      # capped
    assert contact_radius(0.005, R) == pytest.approx(math.sqrt(2 * R * 0.005 - 0.005**2))


def test_load_spreading_fraction():
    pad = PadLayer(load_material("eva_100"), 0.02)
    R, small, big = 0.037, 0.010, 0.200
    assert pad.load_cell_fraction(0.01, R, big) == 1.0       # footprint inside silhouette
    f = pad.load_cell_fraction(0.01, R, small)
    assert 0 < f < 1
    # more remaining thickness spreads further at the same indentation
    assert PadLayer(load_material("eva_100"), 0.04).load_cell_fraction(0.01, R, small) < f
    # fully compressed: spreading term gone, footprint is the ball contact only
    assert pad.load_cell_fraction(0.02, R, small) == pytest.approx((small / contact_radius(0.02, R)) ** 2)


def test_pad_mass_and_validation():
    pad = PadLayer(load_material("eva_100"), 0.02)
    assert pad.areal_mass_kg_m2 == pytest.approx(100 * 0.02)
    with pytest.raises(ValueError):
        PadLayer(load_material("eva_100"), 0.0)
    with pytest.raises(KeyError):
        load_material("unobtanium")
