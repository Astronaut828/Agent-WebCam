"""Launcher checks use temporary bundles and never open a real camera."""

import contextlib
import io
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from agent_webcam import launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="agent webcam launcher ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        package = self.root / "src/agent_webcam"
        package.mkdir(parents=True)
        (package / "app.py").write_text("# synthetic app\n")
        for name in ("desktop.spec", "pyproject.toml", "uv.lock"):
            (self.root / name).write_text(f"# synthetic {name}\n")
        self.stamp = self.root / "build/macos-source.sha256"
        self.executable = self.root / "dist/Agent WebCam.app/Contents/MacOS/Agent WebCam"
        self.plist = self.executable.parents[1] / "Info.plist"
        self.launcher_file = package / "launcher.py"

    def make_bundle(self, description="Show your camera"):
        self.executable.parent.mkdir(parents=True, exist_ok=True)
        self.executable.write_text("#!/bin/sh\n")
        self.executable.chmod(0o755)
        self.plist.write_bytes(plistlib.dumps({"NSCameraUsageDescription": description}))

    def mark_current(self):
        self.stamp.parent.mkdir(parents=True, exist_ok=True)
        self.stamp.write_text(launcher._fingerprint(self.root) + "\n")

    def launch(self, build=None):
        with (patch.object(launcher, "__file__", str(self.launcher_file)),
              patch.object(launcher.shutil, "which", return_value="/tools/uv"),
              patch.object(launcher.subprocess, "run", side_effect=build) as run,
              patch.object(launcher.os, "execv") as execute,
              patch.object(sys, "argv", ["agent-webcam", "--example"])):
            launcher._launch_macos()
        return run, execute

    def test_first_launch_builds_from_checkout_then_replaces_process(self):
        def build(*args, **kwargs):
            self.make_bundle()
        run, execute = self.launch(build)
        run.assert_called_once_with(
            ["/tools/uv", "run", "--extra", "build", "python", "-m", "PyInstaller",
             "--noconfirm", "desktop.spec"], cwd=self.root, check=True)
        execute.assert_called_once_with(str(self.executable), [str(self.executable), "--example"])
        self.assertEqual(self.stamp.read_text().strip(), launcher._fingerprint(self.root))

    def test_current_valid_bundle_is_reused(self):
        self.make_bundle()
        self.mark_current()
        run, execute = self.launch()
        run.assert_not_called()
        execute.assert_called_once()

    def test_source_spec_and_dependency_changes_invalidate_fingerprint(self):
        for relative in ("src/agent_webcam/app.py", "desktop.spec", "pyproject.toml", "uv.lock"):
            with self.subTest(relative=relative):
                self.make_bundle()
                self.mark_current()
                path = self.root / relative
                path.write_text(path.read_text() + "# changed\n")
                run, execute = self.launch()
                run.assert_called_once()
                execute.assert_called_once()

    def test_invalid_camera_metadata_requires_rebuild(self):
        for description in ("", " ", None, 42):
            with self.subTest(description=description):
                self.make_bundle()
                self.mark_current()
                self.plist.write_bytes(plistlib.dumps({} if description is None else {
                    "NSCameraUsageDescription": description}))
                run, execute = self.launch(lambda *args, **kwargs: self.make_bundle())
                run.assert_called_once()
                execute.assert_called_once()

    def test_malformed_plist_or_nonexecutable_binary_is_invalid(self):
        self.make_bundle()
        self.plist.write_text("not a plist")
        self.assertIsNone(launcher._bundle_executable(self.root))
        self.make_bundle()
        self.executable.chmod(0o644)
        self.assertIsNone(launcher._bundle_executable(self.root))

    def test_build_failure_never_launches_old_bundle_or_updates_stamp(self):
        self.make_bundle()
        self.mark_current()
        previous_stamp = self.stamp.read_text()
        (self.root / "desktop.spec").write_text("# changed\n")
        with (patch.object(launcher, "__file__", str(self.launcher_file)),
              patch.object(launcher.shutil, "which", return_value="/tools/uv"),
              patch.object(launcher.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "build")),
              patch.object(launcher.os, "execv") as execute):
            with self.assertRaises(subprocess.CalledProcessError):
                launcher._launch_macos()
        execute.assert_not_called()
        self.assertEqual(self.stamp.read_text(), previous_stamp)

    def test_successful_command_without_valid_bundle_does_not_stamp_or_launch(self):
        with (patch.object(launcher, "__file__", str(self.launcher_file)),
              patch.object(launcher.shutil, "which", return_value="/tools/uv"),
              patch.object(launcher.subprocess, "run"),
              patch.object(launcher.os, "execv") as execute):
            with self.assertRaisesRegex(RuntimeError, "camera-enabled"):
                launcher._launch_macos()
        self.assertFalse(self.stamp.exists())
        execute.assert_not_called()

    def test_missing_checkout_has_actionable_error(self):
        (self.root / "desktop.spec").unlink()
        with patch.object(launcher, "__file__", str(self.launcher_file)):
            with self.assertRaisesRegex(RuntimeError, "source checkout"):
                launcher._checkout()

    def test_pip_environment_can_use_installed_pyinstaller(self):
        with (patch.object(launcher.shutil, "which", return_value=None),
              patch.object(launcher.importlib.util, "find_spec", return_value=object())):
            self.assertEqual(launcher._build_command(), [
                sys.executable, "-m", "PyInstaller", "--noconfirm", "desktop.spec"])

    def test_missing_build_tools_explains_install_command(self):
        with (patch.object(launcher.shutil, "which", return_value=None),
              patch.object(launcher.importlib.util, "find_spec", return_value=None)):
            with self.assertRaisesRegex(RuntimeError, "pip install -e"):
                launcher._build_command()

    def test_source_macos_failure_is_clear_and_does_not_import_qt(self):
        fake_app = types.ModuleType("agent_webcam.app")
        with (patch.object(sys, "platform", "darwin"),
              patch.object(sys, "frozen", False, create=True),
              patch.object(launcher, "_launch_macos", side_effect=RuntimeError("build unavailable")),
              patch.dict(sys.modules, {"agent_webcam.app": fake_app}),
              contextlib.redirect_stderr(io.StringIO()) as stderr):
            with self.assertRaises(SystemExit) as stopped:
                launcher.main()
        self.assertEqual(stopped.exception.code, 1)
        self.assertIn("Could not start Agent WebCam: build unavailable", stderr.getvalue())

    def test_other_platforms_and_frozen_macos_run_app_directly(self):
        for platform, frozen in (("linux", False), ("win32", False), ("darwin", True)):
            with self.subTest(platform=platform, frozen=frozen):
                fake_app = types.ModuleType("agent_webcam.app")
                calls = []
                fake_app.main = lambda: calls.append("app")
                with (patch.object(sys, "platform", platform),
                      patch.object(sys, "frozen", frozen, create=True),
                      patch.dict(sys.modules, {"agent_webcam.app": fake_app}),
                      patch.object(launcher, "_launch_macos") as macos):
                    launcher.main()
                self.assertEqual(calls, ["app"])
                macos.assert_not_called()


if __name__ == "__main__":
    unittest.main()
