import math

import pytest

import numpy as np

from chestforge.core import kernels
from chestforge.core.kernels import contact_radius
from chestforge.core.padding import (FoamMaterial, PadLayer, PlateMaterial, layer, load_material,
                                    load_materials)

MATS = ["eva_100", "pe_45", "poron_xrd_240"]


def test_starter_materials_present_and_flagged_estimated():
    mats = load_materials()
    assert set(MATS) <= set(mats)
    assert isinstance(mats["pc"], PlateMaterial) and isinstance(mats["hdpe"], PlateMaterial)
    assert all(isinstance(mats[m], FoamMaterial) for m in MATS)
    assert all(m.estimated for m in mats.values())   # none are fitted to measured data yet


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


def _spread(stack, strain=None, d_ball=0.0, reach=None, k_nts=5e6, R=0.037):
    foams = [l for l in stack if hasattr(l.material, "A_Pa")]
    plates = [l for l in stack if not hasattr(l.material, "A_Pa")]
    kinds, refs, fi, pi_ = [], [], 0, 0
    for l in stack:
        if l in foams:
            kinds.append(kernels.KIND_FOAM); refs.append(fi); fi += 1
        else:
            kinds.append(kernels.KIND_PLATE); refs.append(pi_); pi_ += 1
    e_par = np.array([f.material.params() for f in foams]).reshape(len(foams), kernels.N_FOAM_PAR)
    a_top, a_plate = np.zeros(len(foams)), np.zeros(max(len(plates), 1))
    reach = np.full(len(plates), 1e3) if reach is None else np.asarray(reach, float)
    a = kernels.spread(np.zeros(len(foams)) if strain is None else np.asarray(strain, float), d_ball, R,
                       np.array(kinds), np.array(refs), e_par, np.array([f.thickness_m for f in foams]),
                       np.ones(len(foams)), np.array([p.bending_stiffness for p in plates]), reach, k_nts,
                       a_top, a_plate)
    return a, a_top, a_plate


def test_foam_spreading_cone():
    foam = layer("eva_100", 20)
    a, a_top, _ = _spread((foam,), strain=[0.5])
    expected = contact_radius(0.01, 0.037) + 0.01 * np.tan(np.radians(35))
    assert a == pytest.approx(expected)
    # fully compressed: no thickness left to spread through
    a_full, _, _ = _spread((foam,), strain=[1.0])
    assert a_full == pytest.approx(contact_radius(0.02, 0.037))


def test_plate_spreading_follows_winkler_theory():
    """a_eq = sqrt(8/pi) (D/k)^(1/4) for a plate straight on the NTS (Timoshenko plate on elastic foundation)."""
    plate = layer("pc", 3)
    k = 5e6
    a, _, _ = _spread((plate,), k_nts=k)
    assert a == pytest.approx(np.sqrt(8 / np.pi) * (plate.bending_stiffness / k) ** 0.25, rel=1e-6)


def test_thicker_plate_spreads_more():
    a2, _, _ = _spread((layer("pc", 2),))
    a4, _, _ = _spread((layer("pc", 4),))
    assert a4 > a2


def test_plate_edge_follows_westergaard_ratio():
    interior, _, _ = _spread((layer("pc", 3),))
    at_edge, _, _ = _spread((layer("pc", 3),), reach=[0.0])
    near, _, _ = _spread((layer("pc", 3),), reach=[0.02])
    assert at_edge == pytest.approx(interior * np.sqrt(0.125 / 0.431), rel=1e-6)
    assert at_edge < near < interior


def test_soft_foam_under_plate_spreads_further():
    on_nts, _, _ = _spread((layer("pc", 3),))
    on_foam, _, _ = _spread((layer("pc", 3), layer("poron_xrd_240", 15)))
    assert on_foam > on_nts      # softer foundation -> longer characteristic length


def test_temperature_softens_foam():
    m = load_material("poron_xrd_240")
    assert m.stress(0.4, 1000.0, temp_F=77) < m.stress(0.4, 1000.0, temp_F=72) < m.stress(0.4, 1000.0, temp_F=67)


def test_pad_mass_and_validation():
    pad = PadLayer(load_material("eva_100"), 0.02)
    assert pad.areal_mass_kg_m2 == pytest.approx(100 * 0.02)
    assert layer("pc", 2).areal_mass_kg_m2 == pytest.approx(1200 * 0.002)
    with pytest.raises(ValueError):
        PadLayer(load_material("eva_100"), 0.0)
    with pytest.raises(KeyError):
        load_material("unobtanium")
