"""Launch macOS source checkouts inside a camera-enabled application bundle."""

import hashlib
import importlib.util
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
from xml.parsers.expat import ExpatError


def _checkout() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not ((root / "desktop.spec").is_file()
            and (root / "pyproject.toml").is_file()
            and (root / "src/agent_webcam").is_dir()):
        raise RuntimeError(
            "macOS camera access needs the Agent WebCam.app bundle. "
            "Run agent-webcam from a source checkout, or open a built app bundle."
        )
    return root


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    inputs = [root / name for name in ("desktop.spec", "pyproject.toml", "uv.lock")]
    inputs.extend(sorted((root / "src").rglob("*.py")))
    for path in inputs:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _bundle_executable(root: Path) -> Path | None:
    contents = root / "dist/Agent WebCam.app/Contents"
    executable = contents / "MacOS/Agent WebCam"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        return None
    try:
        with (contents / "Info.plist").open("rb") as stream:
            info = plistlib.load(stream)
        description = info.get("NSCameraUsageDescription") if isinstance(info, dict) else None
        if isinstance(description, str) and description.strip():
            return executable
    except (OSError, ValueError, ExpatError, plistlib.InvalidFileException):
        pass
    return None


def _build_command() -> list[str]:
    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "--extra", "build", "python", "-m", "PyInstaller",
                "--noconfirm", "desktop.spec"]
    if importlib.util.find_spec("PyInstaller") is not None:
        return [sys.executable, "-m", "PyInstaller", "--noconfirm", "desktop.spec"]
    raise RuntimeError(
        "Building the macOS app needs uv or PyInstaller. From the project, run "
        "python -m pip install -e '.[build]', then run agent-webcam again."
    )


def _launch_macos() -> None:
    root = _checkout()
    fingerprint = _fingerprint(root)
    stamp = root / "build/macos-source.sha256"
    executable = _bundle_executable(root)
    try:
        current = stamp.read_text().strip() == fingerprint
    except OSError:
        current = False
    if executable is None or not current:
        print("Building Agent WebCam.app for macOS camera access…", flush=True)
        subprocess.run(_build_command(), cwd=root, check=True)
        executable = _bundle_executable(root)
        if executable is None:
            raise RuntimeError("The macOS build did not produce a camera-enabled Agent WebCam.app.")
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(fingerprint + "\n")
    os.execv(str(executable), [str(executable), *sys.argv[1:]])


def main() -> None:
    if sys.platform != "darwin" or getattr(sys, "frozen", False):
        from .app import main as run_app
        run_app()
        return
    try:
        _launch_macos()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Could not start Agent WebCam: {error}", file=sys.stderr)
        raise SystemExit(1) from error
