import importlib.util
import platform
import sys
from pathlib import Path

import pytest

from chestforge import app

_spec = importlib.util.spec_from_file_location(
    "install_desktop_shortcut", Path(__file__).resolve().parent.parent / "scripts/install_desktop_shortcut.py")
inst = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inst)


def test_icon_assets_exist():
    for ext in ("png", "ico", "icns"):
        assert (inst.ASSETS / f"chestforge.{ext}").stat().st_size > 1000


def test_launcher_reports_calibrated(calibrated, capsys):
    assert app.run() == 0
    out = capsys.readouterr().out
    assert "Simulated — not NOCSAE certified" in out and "calibration: OK" in out


def test_linux_entry_quotes_paths_with_spaces():
    entry = inst.linux_desktop_entry("/opt/my python/bin/python3")
    assert 'Exec="/opt/my python/bin/python3" -m chestforge.app' in entry
    assert "Terminal=true" in entry and "chestforge.png" in entry


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux install path")
def test_linux_install_writes_executable_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    path = inst.install_linux(tmp_path / "Desktop", sys.executable)
    assert path.exists() and path.stat().st_mode & 0o111
    assert (tmp_path / ".local/share/applications/chestforge.desktop").exists()


def test_macos_bundle_layout(tmp_path):
    app_dir = inst.install_macos(tmp_path, "/usr/local/bin/python3")
    assert (app_dir / "Contents/Info.plist").read_text().count("chestforge") >= 1
    assert (app_dir / "Contents/Resources/chestforge.icns").exists()
    script = (app_dir / "Contents/MacOS/ChestForge").read_text()
    assert "do script" in script and "-m chestforge.app" in script


def test_windows_powershell_escapes_quotes():
    ps = inst.windows_powershell(Path("C:/Users/O'Neil/Desktop"), "C:/Python/python.exe")
    assert "O''Neil" in ps and "chestforge.ico" in ps and "-m chestforge.app" in ps
