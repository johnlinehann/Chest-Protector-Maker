import pytest

from chestforge.core.layout import Protector, Zone, load_design, shift_candidates
from chestforge.core.padding import Ellipse, layer

EXAMPLE = "designs/example_plate_foam.yaml"


def test_ellipse_geometry():
    e = Ellipse(0, 0, 3, 2)
    assert e.contains(0, 0) and not e.contains(3.1, 0)
    assert e.edge_distance(0, 0) == pytest.approx(2.0, abs=1e-3)
    assert e.edge_distance(4, 0) == pytest.approx(-1.0, abs=1e-3)


def _two_zone(plate_half=3.0):
    plate = layer("pc", 3, Ellipse(0, 0, plate_half, plate_half))
    return Protector("t", (
        Zone("cardiac", Ellipse(0, 0, 4, 4), (plate, layer("eva_100", 15))),
        Zone("torso", Ellipse(0, 0, 8, 10), (layer("eva_100", 10),)),
    ))


def test_local_stack_picks_zone_and_plate_reach():
    p = _two_zone()
    c = p.local_stack(0, 0)
    assert c.zone == "cardiac" and len(c.layers) == 2
    assert c.plate_reach_m[0] == pytest.approx(3 * 0.0254, rel=1e-3)
    off_plate = p.local_stack(3.5, 0)            # inside cardiac zone, beyond the plate
    assert off_plate.zone == "cardiac" and len(off_plate.layers) == 1
    assert p.local_stack(6, 0).zone == "torso"
    assert p.local_stack(20, 0).zone is None and p.local_stack(20, 0).layers == ()


def test_edge_flags_inside_shift_radius():
    assert _two_zone(3.0).edges_within(2.0) == []
    flags = _two_zone(1.5).edges_within(2.0)
    assert any("pc plate" in f for f in flags)


def test_silhouette_coverage():
    assert _two_zone().covers_silhouette(2.0)
    tight = Protector("t", (Zone("cardiac", Ellipse(0, 0, 2, 2), (layer("eva_100", 10),)),))
    assert not tight.covers_silhouette(2.0)


def test_shift_candidates_cover_rings_and_directions():
    c = shift_candidates(2.0, 1.0, 0.25, 16)
    radii = sorted({round((x * x + y * y) ** 0.5, 3) for x, y in c})
    assert radii == [1.0, 1.25, 1.5, 1.75, 2.0]
    assert len(c) == 5 * 16


def test_load_example_design():
    p = load_design(EXAMPLE)
    assert [z.name for z in p.zones] == ["cardiac", "torso"]
    assert p.covers_silhouette(2.25)
