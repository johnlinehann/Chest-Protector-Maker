"""Put a ChestForge icon on your desktop (Windows, macOS or Linux).

Run with the same Python you installed ChestForge into:
    python scripts/install_desktop_shortcut.py
Double-clicking the icon opens a terminal running `python -m chestforge.app`.
"""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "assets"
NAME = "ChestForge"


def desktop_dir() -> Path:
    system = platform.system()
    if system == "Windows":  # handles OneDrive-redirected desktops
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
                             capture_output=True, text=True, check=True)
        return Path(out.stdout.strip())
    if system == "Linux" and shutil.which("xdg-user-dir"):
        out = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip())
    return Path.home() / "Desktop"


def _ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def windows_powershell(desktop: Path, python: str) -> str:
    """PowerShell that writes ChestForge.lnk; python.exe keeps a console window for output."""
    lnk = desktop / f"{NAME}.lnk"
    return "; ".join([
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut(" + _ps_quote(str(lnk)) + ")",
        "$s.TargetPath = " + _ps_quote(python),
        "$s.Arguments = '-m chestforge.app'",
        "$s.WorkingDirectory = " + _ps_quote(str(REPO)),
        "$s.IconLocation = " + _ps_quote(str(ASSETS / "chestforge.ico")),
        "$s.Description = 'ChestForge - Simulated, not NOCSAE certified'",
        "$s.Save()",
    ])


def install_windows(desktop: Path, python: str) -> Path:
    subprocess.run(["powershell", "-NoProfile", "-Command", windows_powershell(desktop, python)], check=True)
    return desktop / f"{NAME}.lnk"


def install_macos(desktop: Path, python: str) -> Path:
    """Minimal .app bundle whose script opens Terminal on the launcher."""
    app = desktop / f"{NAME}.app"
    macos, res = app / "Contents" / "MacOS", app / "Contents" / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    shutil.copy(ASSETS / "chestforge.icns", res / "chestforge.icns")
    (app / "Contents" / "Info.plist").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>{NAME}</string>
  <key>CFBundleIdentifier</key><string>local.chestforge</string>
  <key>CFBundleExecutable</key><string>{NAME}</string>
  <key>CFBundleIconFile</key><string>chestforge</string>
  <key>CFBundlePackageType</key><string>APPL</string>
</dict></plist>
""")
    shell_cmd = f"cd {shlex.quote(str(REPO))} && {shlex.quote(python)} -m chestforge.app"
    applescript = shell_cmd.replace("\\", "\\\\").replace('"', '\\"')
    activate = 'tell application "Terminal" to activate'
    do_script = 'tell application "Terminal" to do script "' + applescript + '"'
    exe = macos / NAME
    exe.write_text(f"#!/bin/bash\nosascript -e {shlex.quote(activate)} -e {shlex.quote(do_script)}\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return app


def linux_desktop_entry(python: str) -> str:
    def q(s: str) -> str:  # Desktop Entry spec quoting
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$") + '"'
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        f"Name={NAME}",
        "Comment=Chest-protector designer vs NOCSAE ND200 (Simulated - not NOCSAE certified)",
        f"Exec={q(python)} -m chestforge.app",
        f"Path={REPO}",
        f"Icon={ASSETS / 'chestforge.png'}",
        "Terminal=true",
        "Categories=Science;Engineering;",
        "",
    ])


def install_linux(desktop: Path, python: str) -> Path:
    entry = linux_desktop_entry(python)
    targets = [desktop / f"{NAME}.desktop", Path.home() / ".local/share/applications/chestforge.desktop"]
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(entry)
        t.chmod(0o755)
    if shutil.which("gio"):  # GNOME: mark trusted so it launches on double-click
        subprocess.run(["gio", "set", str(targets[0]), "metadata::trusted", "true"], capture_output=True)
    return targets[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--desktop", type=Path, help="folder to place the icon in (default: your desktop)")
    args = ap.parse_args(argv)

    python = sys.executable
    try:
        subprocess.run([python, "-c", "import chestforge"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print(f"chestforge is not installed for {python}. Run: {python} -m pip install -e {REPO}")
        return 1

    desktop = args.desktop or desktop_dir()
    desktop.mkdir(parents=True, exist_ok=True)
    system = platform.system()
    installer = {"Windows": install_windows, "Darwin": install_macos, "Linux": install_linux}.get(system)
    if installer is None:
        print(f"Unsupported OS: {system}")
        return 1
    path = installer(desktop, python)
    print(f"Created {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
