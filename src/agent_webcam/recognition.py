"""Completion-driven narration of fresh camera frames, without a frame queue."""

import time

from PySide6.QtCore import QObject, QTimer, Signal


FRAME_GAP_MS = 250
MAX_NARRATION_CHARS = 4000


class RecognitionController(QObject):
    """Share one model client with chat and retain only the latest narration."""

    changed = Signal()

    def __init__(self, camera, client, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.client = client
        self._active = False
        self._in_flight = False
        self._closed = False
        self._text = ""
        self._finished_text = ""
        self._answer = ""
        self._status = "Recognition is off"
        self._started = 0.0

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(FRAME_GAP_MS)
        self._timer.timeout.connect(self._observe)
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._update_elapsed)

        self.client.token.connect(self._token)
        self.client.completed.connect(self._completed)
        self.client.failed.connect(self._failed)
        self.client.state_changed.connect(self._model_changed)
        self.camera.state_changed.connect(self._camera_changed)

    @property
    def active(self):
        return self._active

    @property
    def in_flight(self):
        return self._in_flight

    @property
    def text(self):
        return self._text

    @property
    def status(self):
        return self._status

    def start(self):
        if self._closed:
            return False
        if self._active:
            return True
        if not self.camera.ready:
            self._status = "Start your camera before recognition."
        elif not self.client.ready:
            self._status = "Wait for Gemma to be ready."
        elif self.client.busy:
            self._status = "Wait for the current answer to finish."
        else:
            self._active = True
            self._observe()
            return self._active
        self.changed.emit()
        return False

    def stop(self, detail="Recognition is off"):
        # Clear ownership before aborting: an abort may synchronously emit signals.
        owned_request = self._in_flight
        self._active = False
        self._in_flight = False
        self._timer.stop()
        self._status_timer.stop()
        self._answer = ""
        self._text = self._finished_text
        self._status = detail
        if owned_request:
            self.client.cancel()
        self.changed.emit()

    def close(self):
        if not self._closed:
            self._closed = True
            self.stop()

    def _observe(self):
        if self._closed or not self._active or self._in_flight:
            return
        if not self.camera.ready:
            self.stop("Recognition stopped: a fresh camera frame is unavailable.")
            return
        if not self.client.ready:
            self.stop("Recognition stopped: Gemma is unavailable.")
            return
        if self.client.busy:
            self.stop("Recognition stopped: Gemma is busy with another request.")
            return
        try:
            image = self.camera.snapshot()
        except (ValueError, RuntimeError) as error:
            self.stop("Recognition stopped: " + str(error))
            return
        # Snapshot validation can emit a camera state change and stop this mode.
        if not self._active or self._closed:
            return
        if self.client.busy:
            self.stop("Recognition stopped: Gemma is busy with another request.")
            return
        self._in_flight = True
        self._answer = ""
        self._started = time.monotonic()
        self._status = "Looking at the current frame…"
        self._status_timer.start()
        self.changed.emit()
        if not self._active or not self._in_flight:
            return
        try:
            self.client.narrate(image)
        except (ValueError, RuntimeError) as error:
            self._failed(str(error))

    def _token(self, text):
        if self._active and self._in_flight and not self._closed:
            self._answer = (self._answer + text)[:MAX_NARRATION_CHARS]
            self._text = self._answer
            self.changed.emit()

    def _completed(self, duration):
        if not self._active or not self._in_flight or self._closed:
            return
        self._in_flight = False
        self._status_timer.stop()
        if not self._answer.strip():
            self.stop("Recognition stopped: Gemma returned an empty narration.")
            return
        self._finished_text = self._answer.strip()
        self._text = self._finished_text
        self._answer = ""
        self._status = f"Updated in {duration:.1f}s · Looking again shortly"
        self.changed.emit()
        if self._active and not self._closed:
            self._timer.start()

    def _failed(self, detail):
        if self._active and self._in_flight and not self._closed:
            self.stop("Recognition stopped: " + detail)

    def _camera_changed(self, state, detail):
        if self._active and state != "ready":
            self.stop("Recognition stopped: " + detail)

    def _model_changed(self, state, detail):
        if self._active and state != "ready":
            self.stop("Recognition stopped: " + detail)

    def _update_elapsed(self):
        if self._active and self._in_flight and not self._closed:
            self._status = f"Looking at the current frame… {time.monotonic() - self._started:.1f}s elapsed"
            self.changed.emit()
