import numpy as np
import pytest

from chestforge.core.projectile import load_projectile
from chestforge.units import IN_TO_M


@pytest.mark.parametrize("name", ["baseball", "lacrosse"])
def test_nominal_ball_within_nd200_spec(name):
    p = load_projectile(name)
    assert p.nd200
    assert p.in_spec()
    assert p.cd_range_lb.contains(p.cd_force_lb_at_spec())


def test_baseball_spec_ranges_match_section_13_1_1():
    p = load_projectile("baseball")
    assert (p.mass_range_g.min, p.mass_range_g.max) == (142, 149)
    assert (p.circumference_range_in.min, p.circumference_range_in.max) == (9.0, 9.25)
    assert (p.cd_range_lb.min, p.cd_range_lb.max) == (200, 300)
    assert p.cd_total_deflection_m == pytest.approx(0.25 * IN_TO_M)


def test_lacrosse_spec_ranges_match_section_13_1_2():
    p = load_projectile("lacrosse")
    assert (p.mass_range_g.min, p.mass_range_g.max) == (141.8, 148.8)
    assert (p.circumference_range_in.min, p.circumference_range_in.max) == (7.75, 8.0)
    assert (p.cd_range_lb.min, p.cd_range_lb.max) == (115, 150)
    # 25% of diameter
    assert p.cd_total_deflection_m == pytest.approx(0.25 * p.circumference_in * IN_TO_M / np.pi)


def test_default_projectile_is_baseball():
    assert load_projectile().name == "baseball"


def test_softball_is_flagged_beyond_standard():
    p = load_projectile("softball")
    assert p.nd200 is False


@pytest.mark.parametrize("name", ["baseball", "lacrosse", "softball"])
def test_monte_carlo_samples_stay_in_spec(name):
    p = load_projectile(name)
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = p.sample_within_spec(rng)
        assert s.in_spec()
        assert p.cor_range.contains(s.cor)


def test_contact_force_is_unilateral_and_nonnegative():
    p = load_projectile()
    assert p.contact_force(-1e-3, -5.0, 13.4) == 0.0
    assert p.contact_force(1e-3, -1e6, 13.4) == 0.0      # huge separation speed clips, never pulls
    assert p.contact_force(1e-3, 0.0, 13.4) > 0.0
