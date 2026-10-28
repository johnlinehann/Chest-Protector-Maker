"""Desktop entry point: make sure the NTS is calibrated, then show status.

The web UI (Milestone 6) will be launched from here; until then this is a status console.
"""

from __future__ import annotations

import sys

from chestforge import SIM_LABEL
from chestforge.core import calibration
from chestforge.core.calibration import CalibrationError, require_calibrated

ND200_BALLS = ("baseball", "lacrosse")


def run() -> int:
    print(f"ChestForge — NOCSAE ND200-22m25 chest-protector designer\n{SIM_LABEL}\n")
    try:
        for ball in ND200_BALLS:
            require_calibrated(ball)
        print("NTS calibration: OK (verified against Appendix 1)")
    except CalibrationError as e:
        print(f"NTS calibration needed: {e}\nCalibrating...\n")
        calibration._verified.clear()
        if calibration.main() != 0:
            print("\nCalibration FAILED — design runs are blocked until it passes.")
            return 1
    print("\nRun a design:  python -m chestforge.core.nd200_runner --layer pc:3 --layer poron_xrd_240:15 --torso-in 15 --mc 100")
    print("Optimize:      python -m chestforge.optimize.optimizer --name run1 --method all --trials 120")
    print("Materials: config/materials.yaml. The 3D UI arrives in Milestone 6.")
    return 0


def main() -> int:
    code = run()
    if sys.stdin is not None and sys.stdin.isatty():  # keep the shortcut's window open
        input("\nPress Enter to close...")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
