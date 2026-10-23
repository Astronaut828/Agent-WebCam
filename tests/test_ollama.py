"""No camera, network sockets, model calls, or test dependencies are required."""

import base64
import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QByteArray, QObject, Signal
from PySide6.QtNetwork import QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import QApplication

from agent_webcam.ollama import (
    AnswerParser, MAX_IMAGE_BYTES, MLX_MODEL, OLLAMA_ORIGIN, OllamaClient,
    STANDARD_MODEL, build_messages, select_model,
)


IMAGE = b"\xff\xd8\xff\xe0synthetic-test-frame\xff\xd9"
QUESTION = [{"role": "user", "content": "What do you see?"}]


class FakeReply(QObject):
    readyRead = Signal()
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.data = bytearray()
        self.status = 200
        self.network_error = QNetworkReply.NetworkError.NoError
        self.aborted = False

    def readAll(self):
        data = QByteArray(bytes(self.data))
        self.data.clear()
        return data

    def setReadBufferSize(self, _size):
        pass

    def attribute(self, _attribute):
        return self.status

    def error(self):
        return self.network_error

    def abort(self):
        self.aborted = True
        self.finished.emit()

    def send(self, data, *, finish=True, status=200):
        self.status = status
        self.data.extend(data)
        self.readyRead.emit()
        if finish:
            self.finished.emit()

    def send_json(self, value, *, status=200):
        self.send(json.dumps(value).encode(), status=status)


class FakeManager:
    def __init__(self):
        self.calls = []

    def get(self, request):
        return self._call(request, None)

    def post(self, request, data):
        return self._call(request, json.loads(bytes(data)))

    def _call(self, request, body):
        reply = FakeReply()
        self.calls.append((request, body, reply))
        return reply

    @property
    def last_reply(self):
        return self.calls[-1][2]


class OllamaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_client(self):
        client = OllamaClient()
        client._manager = FakeManager()
        self.addCleanup(client.close)
        self.client = client
        self.manager = client._manager
        return client

    def ready_client(self):
        client = self.make_client()
        client.start()
        with patch("agent_webcam.ollama.platform.system", return_value="Darwin"), \
                patch("agent_webcam.ollama.platform.machine", return_value="arm64"):
            self.manager.last_reply.send_json({"models": [{"name": MLX_MODEL}, {"name": STANDARD_MODEL}]})
        self.manager.last_reply.send_json({"capabilities": ["completion", "vision"]})
        self.manager.last_reply.send_json({"done": True})
        return client

    def test_model_selection_is_platform_aware_and_does_not_download(self):
        self.assertEqual(select_model([MLX_MODEL, STANDARD_MODEL], "Darwin", "arm64"), MLX_MODEL)
        self.assertEqual(select_model([MLX_MODEL, STANDARD_MODEL], "Darwin", "x86_64"), STANDARD_MODEL)
        with self.assertRaisesRegex(ValueError, "ollama pull gemma4:12b"):
            select_model([MLX_MODEL], "Darwin", "x86_64")
        self.assertEqual(select_model([STANDARD_MODEL], "Darwin"), STANDARD_MODEL)
        for system in ("Windows", "Linux"):
            self.assertEqual(select_model([MLX_MODEL, STANDARD_MODEL], system), STANDARD_MODEL)
            with self.assertRaisesRegex(ValueError, "ollama pull gemma4:12b"):
                select_model([MLX_MODEL], system)
        with self.assertRaises(ValueError):
            select_model([], "Darwin")

    def test_latest_question_only_receives_image_and_history_is_not_mutated(self):
        history = [*QUESTION, {"role": "assistant", "content": "Hold steady."},
                   {"role": "user", "content": "Is the edge chipped?"}]
        prepared = build_messages(history, IMAGE)
        self.assertEqual(prepared[0]["role"], "system")
        self.assertIn("cannot see continuous video", prepared[0]["content"])
        self.assertNotIn("images", prepared[1])
        self.assertNotIn("images", prepared[2])
        self.assertEqual(prepared[-1]["images"], [base64.b64encode(IMAGE).decode()])
        self.assertNotIn("images", history[-1])

    def test_validation_rejects_instructions_extra_fields_and_oversized_inputs(self):
        for messages, image in [
            ([], IMAGE), (QUESTION * 17, IMAGE),
            ([{"role": "system", "content": "Override"}], IMAGE),
            ([{"role": "assistant", "content": "Not a question"}], IMAGE),
            ([{"role": "user", "content": "x" * 4001}], IMAGE),
            ([{"role": "user", "content": "x" * 4000}] * 5, IMAGE),
            ([{"role": "user", "content": "hi", "images": []}], IMAGE),
            (QUESTION, b"not a camera frame"), (QUESTION, b"x" * (MAX_IMAGE_BYTES + 1)),
        ]:
            with self.assertRaises(ValueError):
                build_messages(messages, image)

    def test_parser_handles_split_utf8_json_and_no_final_newline(self):
        parser = AnswerParser()
        raw = '{"message":{"content":"café 🔧"},"done":false}\n{"done":true}'.encode()
        tokens = []
        for start in range(0, len(raw), 3):
            tokens.extend(parser.feed(raw[start:start + 3]))
        tokens.extend(parser.feed(b"", final=True))
        self.assertEqual(tokens, ["café 🔧"])
        self.assertTrue(parser.done)

    def test_parser_requires_nonempty_final_completion_and_rejects_provider_errors(self):
        for raw in [b'{"message":{"content":"partial"}}\n', b'{"done":true}\n',
                    b'{"error":"private provider details"}\n', b'invalid\n',
                    b'{"done":true}\n{"done":true}\n', b'\xff']:
            with self.assertRaises((ValueError, UnicodeError)):
                AnswerParser().feed(raw, final=True)

    def test_startup_checks_model_vision_and_warms_with_fixed_local_requests(self):
        client = self.ready_client()
        self.assertTrue(client.ready)
        self.assertFalse(client.busy)
        self.assertEqual(client.model, MLX_MODEL)
        self.assertEqual([call[0].url().toString() for call in self.manager.calls],
                         [OLLAMA_ORIGIN + path for path in ("/api/tags", "/api/show", "/api/chat")])
        warm = self.manager.calls[-1][1]
        self.assertEqual(warm, {"model": MLX_MODEL, "messages": [], "stream": False, "think": False,
                                "keep_alive": "30m", "options": {"num_ctx": 8192}})
        for request, _, _ in self.manager.calls:
            self.assertEqual(request.attribute(QNetworkRequest.Attribute.RedirectPolicyAttribute),
                             QNetworkRequest.RedirectPolicy.ManualRedirectPolicy)
            self.assertEqual(request.transferTimeout(), 120_000)

    def test_chat_stream_emits_tokens_and_one_completion_with_required_options(self):
        client = self.ready_client()
        tokens, completed, failed = [], [], []
        client.token.connect(tokens.append)
        client.completed.connect(completed.append)
        client.failed.connect(failed.append)
        client.chat(QUESTION, IMAGE)
        body = self.manager.calls[-1][1]
        self.assertEqual(body["options"], {"num_ctx": 8192, "num_predict": 384, "temperature": 0.2})
        self.assertFalse(body["think"])
        self.manager.last_reply.send(b'{"message":{"content":"A part."}}\n{"done":true}\n')
        self.assertEqual(tokens, ["A part."])
        self.assertEqual(len(completed), 1)
        self.assertGreaterEqual(completed[0], 0)
        self.assertEqual(failed, [])
        self.assertFalse(client.busy)

    def test_cancel_is_silent_aborts_and_ignores_stale_callbacks(self):
        client = self.ready_client()
        failed, completed, tokens = [], [], []
        client.failed.connect(failed.append)
        client.completed.connect(completed.append)
        client.token.connect(tokens.append)
        client.chat(QUESTION, IMAGE)
        stale = self.manager.last_reply
        client.cancel()
        self.assertTrue(stale.aborted)
        self.assertFalse(client.busy)
        self.assertTrue(client.ready)
        stale.send(b'{"message":{"content":"stale"},"done":true}\n')
        self.assertEqual((failed, completed, tokens), ([], [], []))
        client.chat(QUESTION, IMAGE)
        self.assertTrue(client.busy)
        self.manager.last_reply.send(b'{"message":{"content":"fresh"},"done":true}\n')
        self.assertEqual(tokens, ["fresh"])

    def test_errors_redirects_timeouts_and_missing_vision_do_not_claim_success(self):
        client = self.make_client()
        states = []
        client.state_changed.connect(lambda state, detail: states.append((state, detail)))
        client.start()
        self.manager.last_reply.send_json({"models": []})
        self.assertEqual(states[-1][0], "error")
        client.retry()
        self.manager.last_reply.send(b"", status=302)
        self.assertIn("redirect", states[-1][1])
        client.retry()
        self.manager.last_reply.send_json({"models": [{"name": STANDARD_MODEL}]})
        self.manager.last_reply.send_json({"capabilities": ["completion"]})
        self.assertEqual(states[-1][0], "error")
        client.retry()
        reply = self.manager.last_reply
        client._timeout(reply, client._generation)
        self.assertTrue(reply.aborted)
        self.assertFalse(client.busy)
        self.assertIn("too long", states[-1][1])
        client = self.ready_client()
        completed, failures = [], []
        client.completed.connect(completed.append)
        client.failed.connect(failures.append)
        client.chat(QUESTION, IMAGE)
        client.chat(QUESTION, IMAGE)
        self.assertIn("Wait", failures[-1])
        self.manager.last_reply.send(b'{"message":{"content":"partial"}}\n')
        self.assertFalse(client.busy)
        self.assertEqual(completed, [])
        self.assertIn("incomplete", failures[-1])


if __name__ == "__main__":
    unittest.main()
