"""Camera contracts with synthetic Qt frames; never opens a real camera."""

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtMultimedia import QtVideo, QVideoFrame
from PySide6.QtWidgets import QApplication

from agent_webcam.camera import CameraPanel, MAX_FRAME_AGE_SECONDS


class CameraPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel = CameraPanel()

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        self.app.processEvents()

    def frame(self, width=1920, height=1080):
        image = QImage(width, height, QImage.Format.Format_RGB32)
        image.fill(QColor("#a8ddbc"))
        painter = QPainter(image)
        painter.fillRect(0, 0, width // 3, height, QColor("#dc442a"))
        painter.end()
        return QVideoFrame(image)

    def feed(self, frame):
        self.panel._requested = True
        self.panel._on_frame(frame)

    def test_snapshot_scales_to_960_and_stays_unmirrored(self):
        self.feed(self.frame())
        self.assertTrue(self.panel.ready)
        jpeg = self.panel.snapshot()
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))
        image = QImage.fromData(jpeg, "JPEG")
        self.assertEqual((image.width(), image.height()), (960, 540))
        self.assertGreater(image.pixelColor(20, 20).red(), image.pixelColor(20, 20).green())
        self.assertGreater(image.pixelColor(900, 20).green(), image.pixelColor(900, 20).red())

    def test_small_frame_is_not_upscaled(self):
        self.feed(self.frame(320, 240))
        image = QImage.fromData(self.panel.snapshot(), "JPEG")
        self.assertEqual((image.width(), image.height()), (320, 240))

    def test_current_bgr_has_native_dimensions_and_owned_uint8_pixels(self):
        self.feed(self.frame(641, 479))
        pixels = self.panel.current_bgr()
        self.assertEqual(pixels.shape, (479, 641, 3))
        self.assertEqual(pixels.dtype, np.uint8)
        self.assertTrue(pixels.flags["OWNDATA"])
        self.assertGreater(int(pixels[20, 20, 2]), int(pixels[20, 20, 1]))
        self.assertGreater(int(pixels[20, 600, 1]), int(pixels[20, 600, 2]))
        self.panel.stop()
        self.assertEqual(pixels.shape, (479, 641, 3))
        with self.assertRaisesRegex(RuntimeError, "fresh camera frame"):
            self.panel.current_bgr()

    def test_portrait_frame_preserves_aspect(self):
        self.feed(self.frame(1080, 1920))
        image = QImage.fromData(self.panel.snapshot(), "JPEG")
        self.assertEqual((image.width(), image.height()), (540, 960))

    def test_new_frame_replaces_previous_snapshot(self):
        self.feed(self.frame(640, 480))
        before = self.panel.snapshot()
        image = QImage(640, 480, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.blue)
        self.panel._on_frame(QVideoFrame(image))
        after = self.panel.snapshot()
        self.assertNotEqual(before, after)
        self.assertGreater(QImage.fromData(after, "JPEG").pixelColor(20, 20).blue(), 240)

    def test_capture_uses_same_presentation_rotation_as_preview(self):
        frame = self.frame(640, 480)
        frame.setRotation(QtVideo.Rotation.Clockwise90)
        self.feed(frame)
        self.assertEqual(self.panel.current_bgr().shape, (640, 480, 3))
        image = QImage.fromData(self.panel.snapshot(), "JPEG")
        self.assertEqual((image.width(), image.height()), (480, 640))

    def test_stopped_camera_cannot_return_old_or_late_frame(self):
        self.feed(self.frame(640, 480))
        self.panel.stop()
        self.panel._on_frame(self.frame(640, 480))
        self.assertFalse(self.panel.ready)
        self.assertFalse(self.panel._latest_frame.isValid())
        with self.assertRaisesRegex(RuntimeError, "fresh camera frame"):
            self.panel.snapshot()

    def test_stale_frame_cannot_be_sent(self):
        self.feed(self.frame(640, 480))
        self.panel._frame_received_at = time.monotonic() - MAX_FRAME_AGE_SECONDS - 0.1
        self.assertFalse(self.panel.ready)
        with self.assertRaisesRegex(RuntimeError, "fresh camera frame"):
            self.panel.snapshot()
        self.assertEqual(self.panel._state, "loading")
        self.assertFalse(self.panel._latest_frame.isValid())

    def test_invalid_frame_removes_previous_frame(self):
        self.feed(self.frame(640, 480))
        self.panel._on_frame(QVideoFrame())
        self.assertFalse(self.panel.ready)
        with self.assertRaises(RuntimeError):
            self.panel.snapshot()

    def test_late_capture_from_previous_generation_is_ignored(self):
        old_generation = self.panel._generation
        self.panel.stop()
        self.panel._requested = True
        self.panel._present_frame(self.frame(), old_generation)
        self.assertFalse(self.panel.ready)
        self.assertFalse(self.panel._latest_frame.isValid())

    def test_preview_normalizes_mirroring_without_changing_capture(self):
        frame = self.frame(640, 480)
        frame.setMirrored(True)
        self.panel._requested = True
        self.panel._present_frame(frame, self.panel._generation)
        self.assertTrue(self.panel.ready)
        self.assertFalse(self.panel._latest_frame.mirrored())
        image = QImage.fromData(self.panel.snapshot(), "JPEG")
        self.assertGreater(image.pixelColor(20, 20).red(), image.pixelColor(20, 20).green())

    def test_disconnect_invalidates_snapshot(self):
        self.feed(self.frame(640, 480))
        self.panel._fail("Your camera disconnected.")
        self.assertFalse(self.panel.ready)
        self.assertEqual(self.panel._state, "error")
        self.assertFalse(self.panel._latest_frame.isValid())
        with self.assertRaises(RuntimeError):
            self.panel.snapshot()


if __name__ == "__main__":
    unittest.main()
