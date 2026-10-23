"""Recognition lifecycle tests use synthetic frames and no model or camera."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent, QObject, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from agent_webcam.recognition import MAX_NARRATION_CHARS, RecognitionController


class FakeCamera(QObject):
    state_changed = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.ready = True
        self.captures = 0
        self.error = None
        self.during_snapshot = None

    def snapshot(self):
        if self.error:
            raise RuntimeError(self.error)
        self.captures += 1
        if self.during_snapshot:
            self.during_snapshot()
        return f"synthetic-frame-{self.captures}".encode()

    def set_state(self, state, detail):
        self.ready = state == "ready"
        self.state_changed.emit(state, detail)


class FakeClient(QObject):
    state_changed = Signal(str, str)
    token = Signal(str)
    completed = Signal(float)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.ready = True
        self.busy = False
        self.calls = []
        self.cancellations = 0
        self.stale_on_cancel = False
        self.error = None

    def narrate(self, image):
        if self.error:
            raise RuntimeError(self.error)
        self.busy = True
        self.calls.append(image)

    def cancel(self):
        self.cancellations += 1
        self.busy = False
        if self.stale_on_cancel:
            self.token.emit("Stale narration")
            self.completed.emit(1.0)

    def finish(self, text):
        self.token.emit(text)
        self.busy = False
        self.completed.emit(0.5)


class RecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.camera = FakeCamera()
        self.client = FakeClient()
        self.controller = RecognitionController(self.camera, self.client)

    def tearDown(self):
        self.controller.close()
        self.controller.deleteLater()
        self.camera.deleteLater()
        self.client.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def next_frame(self):
        self.controller._timer.stop()
        self.controller._observe()

    def test_each_completed_observation_uses_one_new_frame_without_overlap(self):
        self.assertTrue(self.controller.start())
        self.assertTrue(self.controller.active)
        self.assertTrue(self.controller.in_flight)
        self.assertEqual(self.client.calls, [b"synthetic-frame-1"])
        self.controller.start()
        self.controller._observe()
        self.controller._observe()
        self.assertEqual(self.camera.captures, 1)
        self.client.finish("A metal bracket.")
        self.assertFalse(self.controller.in_flight)
        self.assertTrue(self.controller._timer.isActive())
        self.assertEqual(self.controller.text, "A metal bracket.")
        self.next_frame()
        self.assertEqual(self.client.calls, [b"synthetic-frame-1", b"synthetic-frame-2"])
        self.assertEqual(self.controller.text, "A metal bracket.")
        self.client.token.emit("A hand enters view.")
        self.assertEqual(self.controller.text, "A hand enters view.")

    def test_stop_during_gap_prevents_another_frame_and_preserves_finished_text(self):
        self.controller.start()
        self.client.finish("A blue box.")
        self.controller.stop()
        QTest.qWait(300)
        self.assertFalse(self.controller.active)
        self.assertFalse(self.controller._timer.isActive())
        self.assertEqual(self.camera.captures, 1)
        self.assertEqual(self.client.cancellations, 0)
        self.assertEqual(self.controller.text, "A blue box.")

    def test_midstream_stop_cancels_own_request_and_ignores_late_signals(self):
        self.controller.start()
        self.client.finish("The last completed view.")
        self.next_frame()
        self.client.token.emit("An unfinished view")
        self.client.stale_on_cancel = True
        self.controller.stop()
        self.client.token.emit("More stale text")
        self.client.completed.emit(2.0)
        self.client.failed.emit("Late error")
        self.assertEqual(self.client.cancellations, 1)
        self.assertEqual(self.controller.text, "The last completed view.")
        self.assertEqual(self.controller.status, "Recognition is off")
        self.assertFalse(self.controller.in_flight)
        self.assertFalse(self.controller._timer.isActive())

    def test_start_guards_camera_model_and_shared_client_without_cancelling_chat(self):
        self.camera.ready = False
        self.assertFalse(self.controller.start())
        self.camera.ready = True
        self.client.ready = False
        self.assertFalse(self.controller.start())
        self.client.ready = True
        self.client.busy = True
        self.assertFalse(self.controller.start())
        self.controller.stop()
        self.assertEqual(self.camera.captures, 0)
        self.assertEqual(self.client.cancellations, 0)
        self.assertTrue(self.client.busy)

    def test_camera_loss_stops_and_does_not_resume_when_camera_returns(self):
        self.controller.start()
        self.camera.set_state("loading", "Waiting for a fresh camera frame…")
        self.assertFalse(self.controller.active)
        self.assertEqual(self.client.cancellations, 1)
        self.assertIn("fresh camera frame", self.controller.status)
        self.camera.set_state("ready", "Ready")
        self.controller._observe()
        self.assertFalse(self.controller.active)
        self.assertEqual(len(self.client.calls), 1)

    def test_model_loss_stops_without_automatic_resume(self):
        self.controller.start()
        self.client.ready = False
        self.client.state_changed.emit("error", "Model unavailable")
        self.assertFalse(self.controller.active)
        self.assertIn("Model unavailable", self.controller.status)
        self.client.ready = True
        self.client.state_changed.emit("ready", "Ready")
        self.controller._observe()
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.cancellations, 1)

    def test_camera_state_change_during_capture_prevents_request_dispatch(self):
        self.camera.during_snapshot = lambda: self.camera.set_state("loading", "Camera stalled")
        self.assertFalse(self.controller.start())
        self.assertEqual(self.camera.captures, 1)
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.client.cancellations, 0)
        self.assertFalse(self.controller.in_flight)
        self.assertIn("Camera stalled", self.controller.status)

    def test_capture_and_request_errors_stop_without_retry(self):
        self.camera.error = "No fresh frame"
        self.assertFalse(self.controller.start())
        self.assertIn("No fresh frame", self.controller.status)
        self.assertEqual(self.client.calls, [])
        self.camera.error = None
        self.client.error = "Request rejected"
        self.assertFalse(self.controller.start())
        self.assertIn("Request rejected", self.controller.status)
        self.assertFalse(self.controller._timer.isActive())

    def test_network_failure_stops_without_scheduling_or_retaining_partial_text(self):
        self.controller.start()
        self.client.token.emit("Partial")
        self.client.busy = False
        self.client.failed.emit("Ollama timed out")
        self.assertFalse(self.controller.active)
        self.assertFalse(self.controller._timer.isActive())
        self.assertEqual(self.controller.text, "")
        self.assertIn("timed out", self.controller.status)

    def test_unexpected_shared_request_is_not_cancelled_or_replaced(self):
        self.controller.start()
        self.client.finish("A box.")
        self.client.busy = True
        self.next_frame()
        self.assertFalse(self.controller.active)
        self.assertEqual(self.camera.captures, 1)
        self.assertEqual(self.client.cancellations, 0)
        self.assertTrue(self.client.busy)

    def test_narration_replaces_previous_text_and_remains_bounded(self):
        self.controller.start()
        self.client.finish("First description")
        self.next_frame()
        for _ in range(8):
            self.client.token.emit("x" * 1000)
        self.assertEqual(self.controller.text, "x" * MAX_NARRATION_CHARS)
        self.client.busy = False
        self.client.completed.emit(0.5)
        self.assertEqual(len(self.controller.text), MAX_NARRATION_CHARS)

    def test_empty_completion_stops_instead_of_continuing(self):
        self.controller.start()
        self.client.finish("  ")
        self.assertFalse(self.controller.active)
        self.assertIn("empty narration", self.controller.status)
        self.assertFalse(self.controller._timer.isActive())

    def test_close_cancels_timers_and_request_and_cannot_restart(self):
        self.controller.start()
        self.controller.close()
        self.client.token.emit("Late")
        self.client.completed.emit(1.0)
        self.assertFalse(self.controller.start())
        self.assertFalse(self.controller.active)
        self.assertFalse(self.controller.in_flight)
        self.assertFalse(self.controller._timer.isActive())
        self.assertFalse(self.controller._status_timer.isActive())
        self.assertEqual(self.client.cancellations, 1)
        self.assertEqual(self.controller.text, "")


if __name__ == "__main__":
    unittest.main()
