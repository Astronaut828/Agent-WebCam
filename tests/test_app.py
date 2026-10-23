import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QWidget

from agent_webcam.app import MainWindow, bounded_history


class FakeCamera(QWidget):
    state_changed = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.ready = True
        self.frames = [b"frame-one", b"frame-two"]
        self.captures = 0
        self.stopped = False

    def snapshot(self):
        image = self.frames[self.captures]
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
        self.cancelled = False
        self.closed = False

    def start(self):
        pass

    def retry(self):
        pass

    def chat(self, messages, image):
        self.busy = True
        self.calls.append((messages, image))

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

    def test_each_question_captures_a_new_image_and_history_is_text_only(self):
        self.window.send("What do you see?")
        self.client.finish("Three holes.")
        self.window.send("How many now?")
        self.client.finish("Four holes.")
        self.assertEqual([call[1] for call in self.client.calls], [b"frame-one", b"frame-two"])
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


if __name__ == "__main__":
    unittest.main()
