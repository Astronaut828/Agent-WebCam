"""Native camera preview and fresh, in-memory snapshots."""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import (
    QByteArray,
    QCameraPermission,
    QCoreApplication,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QCloseEvent, QImage, QTransform
from PySide6.QtMultimedia import (
    QCamera,
    QCameraDevice,
    QCameraFormat,
    QMediaCaptureSession,
    QMediaDevices,
    QVideoFrame,
    QVideoSink,
)
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)


MAX_FRAME_AGE_SECONDS = 2.0
START_TIMEOUT_SECONDS = 12.0


def _bgr_from_frame(frame: QVideoFrame) -> np.ndarray:
    """Convert on demand to owned OpenCV pixels; the preview stays on Qt."""
    image = frame.toImage()
    if image.isNull():
        raise RuntimeError("The camera frame could not be read. Try your question again.")

    # QVideoWidget applies frame presentation rotation; toImage() does not.
    rotation = frame.rotation()
    angle = rotation.value if hasattr(rotation, "value") else int(rotation)
    if angle:
        image = image.transformed(QTransform().rotate(angle))
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    # Respect Qt scan-line padding instead of assuming a tightly packed image.
    rows = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.sizeInBytes())
    rows = rows.reshape(image.height(), image.bytesPerLine())
    rgba = rows[:, :image.width() * 4].reshape(image.height(), image.width(), 4)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def _preferred_format(device: QCameraDevice) -> QCameraFormat | None:
    formats = [item for item in device.videoFormats() if not item.isNull()]
    if not formats:
        return None

    def score(item: QCameraFormat) -> float:
        resolution = item.resolution()
        width, height = resolution.width(), resolution.height()
        # Native formats can only select a range; Qt uses its highest rate.
        pixels = abs(width - 1280) / 1280 + abs(height - 720) / 720
        speed = abs(item.maxFrameRate() - 24) / 24
        excessive = max(0.0, item.maxFrameRate() - 30) / 15
        return pixels + speed * 0.45 + excessive

    return min(formats, key=score)


class CameraPanel(QWidget):
    """Camera-only Qt capture session. Construction never requests permission."""

    state_changed = Signal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("cameraPanel")
        self._state = "stopped"
        self._detail = "Start your camera when you’re ready to take a look."
        self._ui_state_initialized = False
        self._requested = False
        self._closing = False
        self._generation = 0
        self._camera: QCamera | None = None
        self._capture_sink: QVideoSink | None = None
        self._latest_frame = QVideoFrame()
        self._frame_received_at = 0.0
        self._started_at = 0.0
        self._selected_device_id = QByteArray()
        self._devices: list[QCameraDevice] = []

        self._session = QMediaCaptureSession(self)
        self._media_devices = QMediaDevices(self)
        self._media_devices.videoInputsChanged.connect(self._devices_changed)
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(500)
        self._watchdog.timeout.connect(self._check_frames)

        self._build_ui()
        self._refresh_devices()
        self._set_state("stopped", self._detail)

    @property
    def ready(self) -> bool:
        return (self._state == "ready" and self._requested
                and self._latest_frame.isValid()
                and time.monotonic() - self._frame_received_at <= MAX_FRAME_AGE_SECONDS)

    def _build_ui(self) -> None:
        self.setObjectName("cameraPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        heading = QVBoxLayout()
        heading.setSpacing(8)
        title_row = QHBoxLayout()
        title_row.setSpacing(12)
        title = QLabel("Camera view")
        title.setObjectName("panelTitle")
        title_row.addWidget(title)
        title_row.addStretch()
        self.status_label = QLabel("Camera off")
        self.status_label.setObjectName("cameraStatus")
        self.status_label.setAccessibleName("Camera status")
        title_row.addWidget(self.status_label)
        heading.addLayout(title_row)
        subtitle = QLabel("Live preview from your webcam")
        subtitle.setObjectName("subtle")
        heading.addWidget(subtitle)
        layout.addLayout(heading)

        self.stage = QFrame()
        self.stage.setObjectName("videoStage")
        self.stage.setMinimumSize(320, 200)
        self.stage.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._stage_layout = QStackedLayout(self.stage)
        self._stage_layout.setContentsMargins(0, 0, 0, 0)

        self.video = QVideoWidget()
        self.video.setObjectName("cameraVideo")
        self.video.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video.videoSink().videoFrameChanged.connect(self._on_frame)
        self._stage_layout.addWidget(self.video)

        self.overlay = QWidget()
        self.overlay.setObjectName("cameraOverlay")
        overlay_layout = QVBoxLayout(self.overlay)
        overlay_layout.setContentsMargins(32, 32, 32, 32)
        overlay_layout.setSpacing(16)
        overlay_layout.addStretch()
        icon = QLabel("◎")
        icon.setObjectName("cameraOverlayIcon")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_title = QLabel("Camera is off")
        self.overlay_title.setObjectName("cameraOverlayTitle")
        self.overlay_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_detail = QLabel(self._detail)
        self.overlay_detail.setObjectName("muted")
        self.overlay_detail.setWordWrap(True)
        self.overlay_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.retry_button = QPushButton("Try camera again")
        self.retry_button.setObjectName("primaryButton")
        self.retry_button.clicked.connect(self.start)
        overlay_layout.addWidget(icon)
        overlay_layout.addWidget(self.overlay_title)
        overlay_layout.addWidget(self.overlay_detail)
        overlay_layout.addWidget(self.retry_button, 0, Qt.AlignmentFlag.AlignCenter)
        overlay_layout.addStretch()
        self._stage_layout.addWidget(self.overlay)
        layout.addWidget(self.stage, 1)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(12)
        self.device_label = QLabel("Webcam")
        self.device_label.setObjectName("muted")
        self.device_label.setTextFormat(Qt.TextFormat.PlainText)
        self.device_label.setWordWrap(True)
        self.camera_combo = QComboBox()
        self.camera_combo.setObjectName("cameraSelector")
        self.camera_combo.setAccessibleName("Choose camera")
        self.camera_combo.setMinimumContentsLength(15)
        self.camera_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.camera_combo.currentIndexChanged.connect(self._select_camera)
        toolbar.addWidget(self.device_label, 1)
        toolbar.addWidget(self.camera_combo, 1)
        self.toggle_button = QPushButton("Start camera")
        self.toggle_button.setObjectName("cameraToggle")
        self.toggle_button.clicked.connect(self._toggle)
        toolbar.addWidget(self.toggle_button)
        layout.addLayout(toolbar)

    def _set_state(self, state: str, detail: str) -> None:
        changed = (state, detail) != (self._state, self._detail)
        if not changed and self._ui_state_initialized:
            return
        self._ui_state_initialized = True
        self._state, self._detail = state, detail
        labels = {"ready": "Camera on", "loading": "Starting camera",
                  "error": "Camera unavailable", "stopped": "Camera off"}
        titles = {"ready": "", "loading": "Opening your camera",
                  "error": "Let’s connect your camera", "stopped": "Camera is off"}
        self.status_label.setText(labels[state])
        self.status_label.setToolTip(detail)
        self.status_label.setProperty("state", state)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self.overlay_title.setText(titles[state])
        self.overlay_detail.setText(detail)
        self.retry_button.setVisible(state == "error")
        self.toggle_button.setText("Stop camera" if state == "ready" else
                                   "Cancel camera" if state == "loading" else "Start camera")
        self.camera_combo.setEnabled(state != "loading")
        self._stage_layout.setCurrentWidget(self.video if state == "ready" else self.overlay)
        if changed:
            self.state_changed.emit(state, detail)

    def _refresh_devices(self) -> None:
        self._devices = list(QMediaDevices.videoInputs())
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        selected_index = 0
        for index, device in enumerate(self._devices):
            self.camera_combo.addItem(device.description(), device.id())
            if device.id() == self._selected_device_id:
                selected_index = index
        if self._devices:
            self.camera_combo.setCurrentIndex(selected_index)
            self._selected_device_id = self._devices[selected_index].id()
            self.device_label.setText(self._devices[selected_index].description())
        else:
            self.device_label.setText("No camera connected")
            self._selected_device_id = QByteArray()
        self.camera_combo.blockSignals(False)
        multiple = len(self._devices) > 1
        self.camera_combo.setVisible(multiple)
        self.device_label.setVisible(not multiple)

    def _devices_changed(self) -> None:
        if self._closing:
            return
        active_id = self._camera.cameraDevice().id() if self._camera else None
        self._refresh_devices()
        if active_id is not None and not any(device.id() == active_id for device in self._devices):
            self._fail("Your camera disconnected. Reconnect it, then try again.")

    def _select_camera(self, index: int) -> None:
        if 0 <= index < len(self._devices):
            self._selected_device_id = self._devices[index].id()
            if self._requested:
                self.start()

    def _toggle(self) -> None:
        if self._requested:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        if self._closing:
            return
        self._halt()
        self._requested = True
        self._generation += 1
        generation = self._generation
        self._set_state("loading", "Allow camera access if your system asks for permission.")
        if sys.platform == "darwin":
            app = QCoreApplication.instance()
            permission = QCameraPermission()
            status = app.checkPermission(permission)
            if status == Qt.PermissionStatus.Denied:
                self._permission_denied()
                return
            if status == Qt.PermissionStatus.Undetermined:
                app.requestPermission(permission, self,
                                      lambda result: self._permission_result(result, generation))
                return
        self._open_camera(generation)

    def _permission_result(self, permission, generation: int) -> None:
        if self._closing or not self._requested or generation != self._generation:
            return
        if permission.status() == Qt.PermissionStatus.Granted:
            self._open_camera(generation)
        else:
            self._permission_denied()

    def _permission_denied(self) -> None:
        self._fail("Allow Agent WebCam in System Settings → Privacy & Security → Camera, then try again.")

    def _open_camera(self, generation: int) -> None:
        if self._closing or not self._requested or generation != self._generation:
            return
        self._refresh_devices()
        if not self._devices:
            self._fail("No camera was found. Connect a webcam, then choose Try camera again.")
            return
        device = next((item for item in self._devices if item.id() == self._selected_device_id),
                      self._devices[0])
        camera = QCamera(device, self)
        self._camera = camera
        preferred = _preferred_format(device)
        if preferred is not None:
            camera.setCameraFormat(preferred)
        camera.errorOccurred.connect(
            lambda error, detail: self._camera_error(camera, error, detail))
        camera.activeChanged.connect(lambda active: self._active_changed(camera, active))

        # Normalize presentation metadata without converting every preview frame
        # to pixels. Capture conversion happens only in snapshot().
        self._capture_sink = QVideoSink(self)
        self._capture_sink.videoFrameChanged.connect(
            lambda frame: self._present_frame(frame, generation))
        self._session.setVideoSink(self._capture_sink)
        self._session.setCamera(camera)
        self._started_at = time.monotonic()
        self._watchdog.start()
        camera.start()

    def _present_frame(self, frame: QVideoFrame, generation: int) -> None:
        if not self._requested or self._closing or generation != self._generation:
            return
        preview_frame = QVideoFrame(frame)
        preview_frame.setMirrored(False)
        self.video.videoSink().setVideoFrame(preview_frame)

    def _on_frame(self, frame: QVideoFrame) -> None:
        if not self._requested or self._closing:
            return
        if not frame.isValid():
            self._invalidate_frame()
            self._set_state("loading", "Waiting for a fresh camera frame…")
            return
        self._latest_frame = QVideoFrame(frame)
        self._frame_received_at = time.monotonic()
        self._set_state("ready", "A fresh camera frame is available.")

    def _check_frames(self) -> None:
        if not self._requested:
            return
        now = time.monotonic()
        if self._latest_frame.isValid() and now - self._frame_received_at > MAX_FRAME_AGE_SECONDS:
            self._invalidate_frame()
            self._set_state("loading", "Waiting for a fresh camera frame…")
        if now - max(self._started_at, self._frame_received_at) > START_TIMEOUT_SECONDS:
            self._fail("No camera frames arrived. Close other apps using your camera, then try again.")

    def _camera_error(self, camera: QCamera, error, detail: str) -> None:
        if camera is self._camera and self._requested and error != QCamera.Error.NoError:
            self._fail("The camera could not open. Check camera permissions and close other apps using it, then try again.")

    def _active_changed(self, camera: QCamera, active: bool) -> None:
        if camera is self._camera and self._requested and not active:
            self._fail("The camera stopped. Check that it’s connected, then try again.")

    def current_bgr(self) -> np.ndarray:
        """Return the latest fresh frame as owned BGR pixels for local processing."""
        if not self.ready:
            if self._state == "ready":
                self._invalidate_frame()
                self._set_state("loading", "Waiting for a fresh camera frame…")
            raise RuntimeError("A fresh camera frame is not available. Start your camera and try again.")
        return _bgr_from_frame(self._latest_frame)

    def snapshot(self) -> bytes:
        """Capture one fresh frame as a compact JPEG, entirely in memory."""
        image = self.current_bgr()
        height, width = image.shape[:2]
        if max(width, height) > 960:
            scale = 960 / max(width, height)
            image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                               interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise RuntimeError("The camera frame could not be captured. Try again.")
        return encoded.tobytes()

    def _invalidate_frame(self) -> None:
        self._latest_frame = QVideoFrame()
        self._frame_received_at = 0.0

    def _halt(self) -> None:
        self._requested = False
        self._generation += 1
        self._watchdog.stop()
        self._invalidate_frame()
        camera, self._camera = self._camera, None
        if camera:
            camera.stop()
            self._session.setCamera(None)
            camera.deleteLater()
        self._session.setVideoSink(None)
        if self._capture_sink:
            self._capture_sink.deleteLater()
            self._capture_sink = None
        self.video.videoSink().setVideoFrame(QVideoFrame())

    def _fail(self, detail: str) -> None:
        self._halt()
        self._set_state("error", detail)

    def stop(self) -> None:
        self._halt()
        self._set_state("stopped", "Start your camera when you’re ready to take another look.")

    def closeEvent(self, event: QCloseEvent) -> None:
        self._closing = True
        self.stop()
        super().closeEvent(event)
