import json
import math

import numpy as np
import pytest

pytest.importorskip("build123d")

from chestforge.core.layout import Protector, Zone, load_design
from chestforge.core.padding import Ellipse, layer
from chestforge.geometry.export import export_design, lattice_suggestion
from chestforge.geometry.protector_gen import IN, BodyFrame, build_layers, load_geometry_config

DESIGN = "designs/example_plate_foam.yaml"


@pytest.fixture(scope="module")
def layers():
    return build_layers(load_design(DESIGN), 15.0)


def test_frame_scales_with_torso_and_preserves_arc_length():
    b = load_geometry_config()["body"]
    f = BodyFrame.for_torso(15.0)
    scale = 15.0 / 12.5
    assert f.radius_mm == pytest.approx(b["chest_radius_in"]["value"] * scale * IN)
    x_in = np.array([-3.0, 0.0, 2.5])
    X, Y = f.to_flat(x_in, np.zeros(3))
    arc = f.radius_mm * np.arcsin(X / f.radius_mm) - f.sil_arc_mm
    assert np.allclose(arc, x_in * IN)
    assert Y[0] == pytest.approx(-3.527 * scale * IN)


def test_solids_valid_and_ordered(layers):
    assert all(l.solid.is_valid for l in layers)
    cardiac = [l for l in layers if l.zone == "cardiac"]
    assert [l.layer.material.id for l in cardiac] == ["poron_xrd_240", "pc"]   # body side first
    assert cardiac[0].r0_mm == 0 and cardiac[1].r0_mm == pytest.approx(cardiac[0].r1_mm)


def _flat_cm3(a_in, b_in, t_mm):
    return math.pi * a_in * b_in * IN**2 * t_mm / 1e3


def test_volumes_match_flat_estimates(layers):
    by = {l.name: l.volume_mm3 / 1e3 for l in layers}
    assert by["cardiac_L1_pc_plate"] == pytest.approx(_flat_cm3(3.0, 3.5, 3), rel=0.03)
    assert by["cardiac_L2_poron_xrd_240_foam"] == pytest.approx(_flat_cm3(4.5, 5.0, 15), rel=0.03)
    torso = _flat_cm3(7.0, 9.0, 15) - _flat_cm3(4.5, 5.0, 15)
    assert by["torso_L1_eva_100_foam"] == pytest.approx(torso, rel=0.05)


def test_zones_do_not_overlap(layers):
    torso = next(l for l in layers if l.zone == "torso").solid
    for l in layers:
        if l.zone == "cardiac":
            inter = torso & l.solid
            assert (inter.volume if inter else 0.0) < 1.0      # mm^3


def test_unbounded_zone_rejected():
    with pytest.raises(ValueError):
        build_layers(Protector.uniform((layer("eva_100", 10),)), 15.0)


def test_lattice_suggestion_follows_gibson_ashby():
    s = lattice_suggestion(layer("eva_100", 15))
    g = load_geometry_config()["lattice"]
    c = g["cells"]["gyroid"]
    rel = (0.20e6 / (c["C"] * g["solid"]["yield_strength_MPa"]["value"] * 1e6)) ** (1 / c["k"])
    assert s["cells"]["gyroid"]["relative_density"] == pytest.approx(rel, abs=1e-4)
    assert s["estimated"] and "NOT simulated" in s["note"]
    assert s["cell_size_mm"] == 7.5


def test_export_writes_reimportable_step(tmp_path):
    from build123d import import_step
    prot = Protector("small", (Zone("cardiac", Ellipse(0, 0, 4, 4),
                                    (layer("pc", 3, Ellipse(0, 0, 3, 3)), layer("eva_100", 10))),))
    s = export_design(prot, 15.0, tmp_path)
    for f in s["files"]:
        assert (tmp_path / f).stat().st_size > 0
    shape = import_step(str(tmp_path / "protector.step"))
    assert len(shape.solids()) == 2
    assert sum(sl.volume for sl in shape.solids()) == pytest.approx(sum(l["volume_cm3"] for l in s["layers"]) * 1e3, rel=1e-3)
    lm = json.loads((tmp_path / "landmarks.json").read_text())
    f = BodyFrame.for_torso(15.0)
    assert lm["silhouette_center"] == pytest.approx(list(f.surface_point(0, 0)), abs=1e-3)
    assert lm["shift_radius_in"] == 2.0
