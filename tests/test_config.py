import pytest

from chestforge.config import load_nd200


@pytest.fixture
def nd():
    return load_nd200()


def test_pass_fail_limits(nd):
    assert nd.limit("30mph").max_peak_lbf == 90 and nd.limit("30mph").max_peak_N == 400
    assert nd.limit("50mph").max_peak_lbf == 180 and nd.limit("50mph").max_peak_N == 800
    assert nd.limit("30mph").velocity_ms_nominal == 13.4
    assert nd.limit("50mph").velocity_ms_nominal == 22.4
    assert nd.velocity_tolerance_pct == 3.0


def test_table1_matrix(nd):
    samples = nd.raw["test_matrix"]["samples"]
    assert [s["condition"] for s in samples] == ["30mph", "50mph"]
    for s in samples:
        assert s["locations"] == ["cardiac_center", "random_1", "random_2"]
    assert sum(len(s["locations"]) for s in samples) == 6


@pytest.mark.parametrize("torso, shift", [
    (10.0, 1.75), (13.5, 1.75),          # §12.2.1: 13.5 in or less
    (13.5001, 2.0), (15.99, 2.0),        # §12.2.2: more than 13.5, less than 16.0
    (16.0, 2.25), (20.0, 2.25),          # §12.2.3: 16.0 or greater
])
def test_shift_limits_by_torso_length(nd, torso, shift):
    assert nd.max_shift_in(torso) == shift


def test_shift_limit_rejects_bad_torso(nd):
    with pytest.raises(ValueError):
        nd.max_shift_in(0)


def test_random_location_rules(nd):
    rl = nd.raw["random_location"]
    assert rl["min_contact_spacing_in"]["value"] == 1.0
    assert rl["aim"] == "cardiac_center"


def test_calibration_and_daq_targets(nd):
    c = nd.calibration
    assert (c["target"]["velocity_mph"], c["target"]["peak_lbf"], c["target"]["tolerance_lbf"]) == (30.0, 140, 2)
    assert (c["lower"]["velocity_mph"], c["lower"]["peak_lbf_greater_than"]) == (29.1, 130)
    assert (c["upper"]["velocity_mph"], c["upper"]["peak_lbf_less_than"]) == (30.9, 150)
    assert nd.min_sample_rate_hz == 10000
    assert nd.filter_cfc == 120
    assert nd.raw["surrogate"]["linear_bearing_table_max_weight_lb"]["value"] == 12.5


def _assert_sourced(node, path=(), sourced=False):
    if isinstance(node, dict):
        sourced = sourced or "source" in node
        if {"value", "min", "max"} & node.keys():
            assert sourced, f"unsourced value at {'.'.join(map(str, path))}"
        for k, v in node.items():
            _assert_sourced(v, path + (k,), sourced)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _assert_sourced(v, path + (i,), sourced)


def test_every_spec_value_is_sourced(nd):
    """Never invent data silently: every value/range in nd200.yaml cites a section."""
    _assert_sourced(nd.raw)


@pytest.mark.parametrize("name", ["projectiles.yaml", "nts.yaml", "materials.yaml"])
def test_model_values_are_sourced_or_estimated(name):
    import yaml
    from chestforge.config import CONFIG_DIR

    data = yaml.safe_load((CONFIG_DIR / name).read_text())

    def walk(node, path=(), flagged=False):
        if isinstance(node, dict):
            flagged = flagged or bool(node.get("estimated") or node.get("verified"))
            if "value" in node:
                assert flagged, f"{name}: {path} not sourced + flagged estimated/verified"
            for k, v in node.items():
                walk(v, path + (k,), flagged)
    walk(data)
    _assert_sourced(data)
