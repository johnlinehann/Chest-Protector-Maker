"""Desktop entry point: make sure the NTS is calibrated, then open the ChestForge web UI."""

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
    print("\nCLI:  python -m chestforge.core.nd200_runner --layer pc:3 --layer poron_xrd_240:15 --torso-in 15 --mc 100")
    print("      python -m chestforge.optimize.optimizer --name run1 --method all --trials 120")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Calibrate if needed, then open the web UI (pass --no-ui for the status console only)."""
    argv = sys.argv[1:] if argv is None else argv
    code = run()
    if code == 0 and "--no-ui" not in argv:
        from chestforge.server import main as serve
        serve()
        return 0
    if sys.stdin is not None and sys.stdin.isatty():  # keep the shortcut's window open
        input("\nPress Enter to close...")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
