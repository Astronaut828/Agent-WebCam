import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest

import cv2
import numpy as np

from PySide6.QtCore import QEvent, QObject, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from agent_webcam.app import MainWindow, bounded_history


def synthetic_image(with_object=False):
    frame = np.full((144, 192, 3), 255, dtype=np.uint8)
    if with_object:
        frame[36:108, 48:144] = (0, 0, 220)
    encoded, image = cv2.imencode(".jpg", frame)
    assert encoded
    return image.tobytes()


class FakeCamera(QWidget):
    state_changed = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.ready = True
        self.frames = [synthetic_image(), synthetic_image(with_object=True)]
        self.captures = 0
        self.stopped = False

    def snapshot(self):
        image = self.frames[min(self.captures, len(self.frames) - 1)]
        self.captures += 1
        return image

    def start(self):
        self.ready = True

    def stop(self):
        self.ready = False
        self.stopped = True
        self.state_changed.emit("stopped", "Camera off")


class FakeClient(QObject):
    state_changed = Signal(str, str)
    model_changed = Signal(str)
    token = Signal(str)
    completed = Signal(float)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.ready = True
        self.busy = False
        self.calls = []
        self.narrations = []
        self.cancelled = False
        self.closed = False

    def start(self):
        pass

    def retry(self):
        pass

    def chat(self, messages, image):
        self.busy = True
        self.calls.append((messages, image))

    def narrate(self, image):
        self.busy = True
        self.narrations.append(image)

    def cancel(self):
        self.busy = False
        self.cancelled = True

    def close(self):
        self.closed = True
        self.cancel()

    def finish(self, answer):
        self.token.emit(answer)
        self.busy = False
        self.completed.emit(0.5)


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.camera = FakeCamera()
        self.client = FakeClient()
        self.window = MainWindow(self.camera, self.client, autostart=False)

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.window.deleteLater()
        self.client.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_each_question_captures_a_new_image_and_history_is_text_only(self):
        self.window.send("What do you see?")
        self.client.finish("Three holes.")
        self.window.send("How many now?")
        self.client.finish("Four holes.")
        self.assertEqual([call[1] for call in self.client.calls], self.camera.frames)
        self.assertEqual(len(self.client.calls[1][0]), 3)
        self.assertTrue(all(set(message) == {"role", "content"} for message in self.window.history))

    def test_cancellation_excludes_partial_context_and_restores_controls(self):
        self.window.send("Inspect this.")
        self.client.token.emit("Partial answer")
        self.window.cancel()
        self.assertTrue(self.client.cancelled)
        self.assertEqual(self.window.history, [])
        self.assertIsNone(self.window.current)
        self.assertTrue(self.window.input.isEnabled())

    def test_camera_stop_cancels_answer_and_close_releases_resources(self):
        self.window.send("Inspect this.")
        self.camera.stop()
        self.assertTrue(self.client.cancelled)
        self.assertFalse(self.window.input.isEnabled())
        self.window.close()
        self.assertTrue(self.camera.stopped)
        self.assertTrue(self.client.closed)

    def test_new_chat_clears_context_and_messages(self):
        self.window.send("Remember four holes.")
        self.client.finish("Understood.")
        self.window.new_chat()
        self.assertEqual(self.window.history, [])
        self.assertEqual(self.window.message_layout.count(), 2)

    def test_context_limits_drop_old_complete_turns(self):
        messages = [{"role": role, "content": "a" * 4000}
                    for _ in range(12) for role in ("user", "assistant")]
        result = bounded_history(messages + [{"role": "user", "content": "Now?"}])
        self.assertLessEqual(len(result), 16)
        self.assertLessEqual(sum(len(row["content"]) for row in result), 16000)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[-1]["content"], "Now?")

    def start_recognition(self):
        self.window.mode_tabs.setCurrentIndex(1)
        self.window.recognition_button.click()
        self.app.processEvents()
        if not self.client.narrations:
            self.window.recognition._observe()

    def test_recognition_updates_latest_text_without_growing_chat(self):
        self.window.history = [
            {"role": "user", "content": "Keep this conversation."},
            {"role": "assistant", "content": "Understood."},
        ]
        previous_history = list(self.window.history)
        previous_message_count = self.window.message_layout.count()

        self.start_recognition()
        self.assertTrue(self.window.recognition.active)
        self.assertEqual(self.client.narrations, self.camera.frames[:1])
        self.assertFalse(self.window.input.isEnabled())
        self.client.finish("A red item rests on the table.")
        self.assertEqual(self.window.narration_label.text(), "A red item rests on the table.")

        self.window.recognition._observe()
        self.assertEqual(self.client.narrations, self.camera.frames)
        self.client.finish("A hand lifts the red item.")
        self.assertEqual(self.window.narration_label.text(), "A hand lifts the red item.")
        self.assertEqual(self.window.history, previous_history)
        self.assertEqual(self.window.message_layout.count(), previous_message_count)
        self.assertEqual(self.client.calls, [])

    def test_recognition_requires_both_camera_and_model(self):
        self.window.mode_tabs.setCurrentIndex(1)
        for resource in (self.camera, self.client):
            with self.subTest(resource=type(resource).__name__):
                resource.ready = False
                self.window.update_controls()
                self.assertFalse(self.window.recognition_button.isEnabled())
                self.window.recognition_button.click()
                self.assertFalse(self.window.recognition.active)
                self.assertEqual(self.client.narrations, [])
                resource.ready = True
                self.window.update_controls()

    def test_unchanged_scene_preserves_narration_and_keeps_stop_available(self):
        self.camera.frames = [synthetic_image()]
        self.start_recognition()
        self.client.finish("There is a solid white frame.")
        for _ in range(3):
            self.window.recognition._timer.stop()
            self.window.recognition._observe()

        self.assertEqual(self.client.narrations, self.camera.frames)
        self.assertEqual(self.window.narration_label.text(), "There is a solid white frame.")
        self.assertTrue(self.window.recognition.active)
        self.assertTrue(self.window.recognition_button.isEnabled())
        self.assertEqual(self.window.recognition_button.text(), "Stop recognition")
        self.assertFalse(self.window.input.isEnabled())
        self.assertEqual(self.window.history, [])
        self.window.recognition_button.click()
        self.assertFalse(self.window.recognition.active)
        self.assertEqual(self.window.narration_label.text(), "There is a solid white frame.")

    def test_latest_chat_answer_stays_visible_at_compact_window_size(self):
        self.window.resize(900, 600)
        self.window.show()
        self.window.send("How many holes are visible?")
        QTest.qWait(30)
        self.client.finish("Three mounting holes are visible around the blue part.")
        QTest.qWait(30)
        scrollbar = self.window.scroll.verticalScrollBar()
        self.assertEqual(scrollbar.value(), scrollbar.maximum())

    def test_switching_to_recognition_cancels_chat_before_starting(self):
        self.window.send("Inspect this item.")
        self.client.token.emit("An unfinished answer")
        self.window.mode_tabs.setCurrentIndex(1)
        self.assertTrue(self.client.cancelled)
        self.assertIsNone(self.window.current)
        self.assertEqual(self.window.history, [])
        self.assertFalse(self.window.recognition.active)
        self.assertTrue(self.window.recognition_button.isEnabled())

    def test_returning_to_chat_stops_recognition_and_ignores_late_tokens(self):
        self.start_recognition()
        self.client.token.emit("An item is visible.")
        self.window.mode_tabs.setCurrentIndex(0)
        self.assertFalse(self.window.recognition.active)
        self.assertTrue(self.client.cancelled)
        last_narration = self.window.narration_label.text()
        self.client.finish("This answer arrived after stopping.")
        self.window.recognition._observe()
        self.assertEqual(self.window.narration_label.text(), last_narration)
        self.assertEqual(self.client.narrations, self.camera.frames[:1])
        self.assertEqual(self.window.history, [])
        self.assertTrue(self.window.input.isEnabled())

    def test_camera_stop_ends_recognition(self):
        self.start_recognition()
        self.camera.stop()
        self.assertFalse(self.window.recognition.active)
        self.assertTrue(self.client.cancelled)
        self.assertFalse(self.window.recognition_button.isEnabled())

    def test_closing_window_stops_recognition_and_releases_resources(self):
        self.start_recognition()
        self.window.close()
        self.assertFalse(self.window.recognition.active)
        self.assertTrue(self.camera.stopped)
        self.assertTrue(self.client.closed)


if __name__ == "__main__":
    unittest.main()
