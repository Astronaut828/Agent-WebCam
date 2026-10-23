# Build a desktop bundle on the target operating system with PyInstaller.
import sys
from pathlib import Path

root = Path(SPECPATH)
analysis = Analysis(
    [str(root / "src/agent_webcam/__main__.py")],
    pathex=[str(root / "src")],
    binaries=[], datas=[], hiddenimports=[], hookspath=[],
    excludes=["PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtPdf"],
    noarchive=False,
)
archive = PYZ(analysis.pure)
executable = EXE(
    archive, analysis.scripts, [], exclude_binaries=True,
    name="Agent WebCam", console=False, debug=False, strip=False, upx=False,
)
folder = COLLECT(executable, analysis.binaries, analysis.datas, name="Agent WebCam", strip=False, upx=False)
if sys.platform == "darwin":
    bundle = BUNDLE(
        folder, name="Agent WebCam.app", bundle_identifier="org.agentwebcam.desktop",
        info_plist={
            "NSCameraUsageDescription": "Show your camera preview and send a frame to your local Gemma model when you ask a question.",
            "NSHighResolutionCapable": True,
        },
    )
