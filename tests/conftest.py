import pytest

from chestforge.core import calibration


@pytest.fixture(scope="session")
def calibrated(tmp_path_factory):
    """Calibrate once into a temp file so tests never depend on a stale config/nts_calibrated.yaml."""
    path = tmp_path_factory.mktemp("cal") / "nts_calibrated.yaml"
    mp = pytest.MonkeyPatch()
    mp.setattr(calibration, "CALIBRATION_FILE", path)
    calibration._verified.clear()
    res = calibration.calibrate_nts(write=True)
    yield res
    mp.undo()
    calibration._verified.clear()
