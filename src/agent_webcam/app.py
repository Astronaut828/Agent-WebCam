"""The desktop camera and chat window."""

import sys
import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontDatabase, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit,
    QPushButton, QScrollArea, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from .camera import CameraPanel
from .ollama import OllamaClient


STYLE = """
QWidget { background: #111416; color: #e9edef; font-size: 14px; }
QLabel { background: transparent; border: none; }
QLabel#title { font-size: 24px; font-weight: 600; }
QLabel#subtle, QLabel#muted { color: #a1abb3; font-size: 12px; }
QLabel#localBadge { color: #a8beb3; font-size: 11px; background: #1b2421; border: 1px solid #2b3932; border-radius: 12px; padding: 6px 12px; }
QLabel#model { color: #a8beb3; font-size: 12px; }
QLabel#panelTitle { font-size: 20px; font-weight: 600; }
QLabel#cameraStatus { color: #aab5bc; background: #242b30; border: 1px solid #354047; border-radius: 10px; padding: 6px 10px; font-size: 12px; }
QLabel#cameraStatus[state="ready"] { color: #b9e6cb; background: #21362c; border-color: #355442; }
QLabel#cameraStatus[state="error"] { color: #ebc4a1; background: #372b22; border-color: #5a4431; }
QLabel#cameraOverlayIcon { color: #90b6a1; font-size: 44px; }
QLabel#cameraOverlayTitle { font-size: 20px; font-weight: 500; }
QWidget#cameraPanel, QFrame#chatPanel { background: #1a1f23; border: 1px solid #30383e; border-radius: 16px; }
QFrame#videoStage { background: #0d1113; border: 1px solid #30383e; border-radius: 10px; }
QWidget#cameraOverlay { background: #0d1113; border-radius: 10px; }
QWidget#chatContents, QScrollArea { background: #1a1f23; border: none; }
QWidget#userMessage { background: #252d33; border-radius: 10px; }
QWidget#assistantMessage { background: transparent; }
QLabel#messageRole { color: #b9e6cb; font-size: 12px; font-weight: 600; }
QLabel#messageBody { font-size: 14px; }
QFrame#composerDivider { background: #30383e; border: none; max-height: 1px; }
QPushButton { background: #2b353c; border: 1px solid #414e57; border-radius: 8px; padding: 10px 16px; }
QPushButton:hover { background: #36434b; border-color: #5b6c75; }
QPushButton:pressed { background: #20292e; }
QPushButton:disabled { color: #7c8790; border-color: #333d44; background: #242c31; }
QPushButton#send { background: #b9e6cb; color: #173727; border-color: #b9e6cb; font-weight: 600; }
QPushButton#send:hover { background: #cdf0db; border-color: #cdf0db; }
QPushButton#send:disabled { background: #354c40; color: #99b2a3; border-color: #354c40; }
QPushButton#quiet { background: transparent; border-color: #3a454d; font-size: 12px; padding: 7px 12px; }
QPushButton#quiet:hover { background: #2b353c; }
QPushButton#prompt { background: transparent; color: #bac6ce; border-color: #3a454d; padding: 9px 12px; font-size: 12px; }
QPushButton#prompt:hover { background: #2b353c; }
QPushButton#prompt:disabled { color: #7c8790; border-color: #333d44; }
QPlainTextEdit { background: #12181c; border: 1px solid #414e57; border-radius: 10px; padding: 12px; selection-background-color: #416453; }
QPlainTextEdit:focus { border-color: #a4cfb7; }
QPlainTextEdit:disabled { color: #89959e; border-color: #333d44; }
QComboBox { background: #242d33; border: 1px solid #414e57; border-radius: 8px; padding: 8px; }
QSplitter::handle { background: #111416; }
QScrollBar:vertical { background: transparent; width: 6px; margin: 4px 0; }
QScrollBar::handle:vertical { background: #4b5a63; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
"""


class MessageInput(QPlainTextEdit):
    submitted = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            event.accept()
            self.submitted.emit()
        else:
            super().keyPressEvent(event)


def bounded_history(messages):
    result = [{"role": item["role"], "content": item["content"][:4000]} for item in messages]
    while len(result) > 16 or sum(len(item["content"]) for item in result) > 16000:
        result.pop(0)
        if result and result[0]["role"] == "assistant":
            result.pop(0)
    return result


class MainWindow(QMainWindow):
    def __init__(self, camera=None, client=None, *, autostart=True):
        super().__init__()
        self.setWindowTitle("Agent WebCam")
        self.resize(1240, 800)
        self.setMinimumSize(900, 600)
        self.camera = camera if camera is not None else CameraPanel()
        self.client = client if client is not None else OllamaClient()
        self.history = []
        self.current = None
        self._closing = False

        outer = QWidget()
        layout = QVBoxLayout(outer)
        layout.setContentsMargins(28, 24, 28, 18)
        layout.setSpacing(20)
        heading = QHBoxLayout()
        title = QLabel("Agent WebCam")
        title.setObjectName("title")
        heading.addWidget(title)
        heading.addStretch()
        local = QLabel("ON DEVICE  ·  OLLAMA")
        local.setObjectName("localBadge")
        heading.addWidget(local)
        layout.addLayout(heading)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(24)
        splitter.addWidget(self.camera)
        panel = QFrame()
        panel.setObjectName("chatPanel")
        panel.setMinimumWidth(340)
        chat_layout = QVBoxLayout(panel)
        chat_layout.setContentsMargins(24, 24, 24, 24)
        chat_layout.setSpacing(20)
        chat_header = QVBoxLayout()
        chat_header.setSpacing(8)
        chat_heading = QHBoxLayout()
        chat_heading.setSpacing(12)
        caption = QLabel("Conversation")
        caption.setObjectName("panelTitle")
        chat_heading.addWidget(caption)
        chat_heading.addStretch()
        self.clear_button = QPushButton("New chat")
        self.clear_button.setObjectName("quiet")
        self.clear_button.clicked.connect(self.new_chat)
        chat_heading.addWidget(self.clear_button)
        chat_header.addLayout(chat_heading)

        model_row = QHBoxLayout()
        self.model_label = QLabel("Gemma 4 12B · Loading…")
        self.model_label.setObjectName("model")
        self.model_label.setWordWrap(True)
        self.model_label.setTextFormat(Qt.TextFormat.PlainText)
        model_row.addWidget(self.model_label, 1)
        self.retry_button = QPushButton("Retry")
        self.retry_button.setVisible(False)
        self.retry_button.clicked.connect(self.client.retry)
        model_row.addWidget(self.retry_button)
        chat_header.addLayout(model_row)
        chat_layout.addLayout(chat_header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.chat_contents = QWidget()
        self.chat_contents.setObjectName("chatContents")
        self.message_layout = QVBoxLayout(self.chat_contents)
        self.message_layout.setContentsMargins(0, 8, 10, 8)
        self.message_layout.setSpacing(20)
        self.empty_label = QLabel("Show me something.\n\nHold an item in view, then ask a question.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.empty_label.setObjectName("subtle")
        self.message_layout.addWidget(self.empty_label, 1)
        self.message_layout.addStretch()
        self.scroll.setWidget(self.chat_contents)
        chat_layout.addWidget(self.scroll, 1)

        composer = QVBoxLayout()
        composer.setSpacing(12)
        divider = QFrame()
        divider.setObjectName("composerDivider")
        divider.setFixedHeight(1)
        composer.addWidget(divider)
        composer.addSpacing(4)
        prompts = QHBoxLayout()
        prompts.setSpacing(10)
        self.prompt_buttons = []
        for label, prompt in [("What do you see?", "What do you see?"), ("Check this item", "Check this item for visible defects. Ask for any inspection criteria you need.")]:
            button = QPushButton(label)
            button.setObjectName("prompt")
            button.clicked.connect(lambda _checked=False, text=prompt: self.send(text))
            self.prompt_buttons.append(button)
            prompts.addWidget(button)
        composer.addLayout(prompts)

        self.input = MessageInput()
        self.input.setAccessibleName("Message Gemma about the camera view")
        self.input.setPlaceholderText("Ask about what's in view…")
        self.input.setFixedHeight(104)
        self.input.submitted.connect(self.send_from_input)
        self.input.textChanged.connect(self.limit_input)
        composer.addWidget(self.input)
        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        self.hint = QLabel("Starting camera and model…")
        self.hint.setObjectName("subtle")
        self.hint.setWordWrap(True)
        bottom.addWidget(self.hint, 1)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.cancel)
        self.stop_button.setVisible(False)
        bottom.addWidget(self.stop_button)
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("send")
        self.send_button.setMinimumWidth(80)
        self.send_button.clicked.connect(self.send_from_input)
        bottom.addWidget(self.send_button)
        composer.addLayout(bottom)
        chat_layout.addLayout(composer)
        splitter.addWidget(panel)
        splitter.setSizes([650, 450])
        layout.addWidget(splitter, 1)
        footer = QLabel("Processed on this device · No recordings or saved chats")
        footer.setObjectName("subtle")
        layout.addWidget(footer)
        self.setCentralWidget(outer)

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_elapsed)
        self.camera.state_changed.connect(self.camera_changed)
        self.client.state_changed.connect(self.model_changed)
        self.client.model_changed.connect(lambda model: self.model_label.setToolTip(model))
        self.client.token.connect(self.answer_token)
        self.client.completed.connect(self.answer_complete)
        self.client.failed.connect(self.answer_failed)
        self.update_controls()
        if autostart:
            QTimer.singleShot(0, self.camera.start)
            QTimer.singleShot(0, self.client.start)

    def limit_input(self):
        text = self.input.toPlainText()
        if len(text) > 4000:
            self.input.setPlainText(text[:4000])
            cursor = self.input.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.input.setTextCursor(cursor)
        self.update_controls()

    def update_controls(self):
        ready = self.camera.ready and self.client.ready and not self.client.busy and self.current is None
        self.input.setEnabled(ready)
        self.send_button.setEnabled(ready and bool(self.input.toPlainText().strip()))
        self.send_button.setVisible(self.current is None)
        self.stop_button.setVisible(self.current is not None)
        for button in self.prompt_buttons:
            button.setEnabled(ready)
        if self.current:
            self.hint.setText("Looking at the captured frame…")
        elif not self.camera.ready:
            self.hint.setText("Start your camera to ask a question")
        elif not self.client.ready:
            self.hint.setText("Waiting for Gemma…")
        else:
            self.hint.setText("Enter to send · Shift+Enter for a new line")

    def camera_changed(self, state, _detail):
        if state != "ready" and self.current:
            self.cancel()
        self.update_controls()

    def model_changed(self, state, detail):
        self.model_label.setText("Gemma 4 12B · " + ("Ready" if state == "ready" else detail))
        self.retry_button.setVisible(state == "error")
        self.update_controls()

    def add_message(self, role, text):
        self.empty_label.hide()
        widget = QWidget()
        widget.setObjectName("userMessage" if role == "user" else "assistantMessage")
        widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        item = QVBoxLayout(widget)
        item.setContentsMargins(16, 14, 16, 14)
        item.setSpacing(8)
        name = QLabel("You" if role == "user" else "Gemma")
        name.setObjectName("messageRole")
        body = QLabel(text)
        body.setObjectName("messageBody")
        body.setTextFormat(Qt.TextFormat.PlainText)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        meta = QLabel("")
        meta.setObjectName("subtle")
        meta.setWordWrap(True)
        item.addWidget(name)
        item.addWidget(body)
        item.addWidget(meta)
        self.message_layout.insertWidget(self.message_layout.count() - 1, widget)
        self.scroll_bottom()
        return body, meta

    def scroll_bottom(self):
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()))

    def send_from_input(self):
        self.send(self.input.toPlainText())

    def send(self, text):
        text = text.strip()
        if not text or len(text) > 4000 or self.current or self.client.busy or not self.client.ready or not self.camera.ready:
            return
        try:
            image = self.camera.snapshot()
        except (ValueError, RuntimeError) as error:
            self.hint.setText(str(error))
            return
        self.add_message("user", text)[1].setText("Frame captured")
        label, meta = self.add_message("assistant", "Taking a look…")
        self.current = {"question": text, "label": label, "meta": meta, "answer": "", "started": time.monotonic()}
        self.input.clear()
        self.timer.start()
        self.update_controls()
        try:
            self.client.chat(bounded_history(self.history + [{"role": "user", "content": text}]), image)
        except (ValueError, RuntimeError) as error:
            self.answer_failed(str(error))

    def answer_token(self, text):
        if self.current:
            self.current["answer"] += text
            self.current["label"].setText(self.current["answer"])
            self.scroll_bottom()

    def update_elapsed(self):
        if self.current:
            self.current["meta"].setText(f"{time.monotonic() - self.current['started']:.1f}s elapsed")

    def answer_complete(self, duration):
        if not self.current:
            return
        current = self.current
        current["meta"].setText(f"Answered in {duration:.1f}s")
        self.history = bounded_history(self.history + [
            {"role": "user", "content": current["question"]},
            {"role": "assistant", "content": current["answer"]},
        ])
        self.finish_answer()

    def answer_failed(self, error):
        if self.current:
            if not self.current["answer"]:
                self.current["label"].setText("I couldn't finish that answer.")
            self.current["meta"].setText(error)
            self.finish_answer()

    def finish_answer(self):
        self.timer.stop()
        self.current = None
        self.update_controls()
        if self.input.isEnabled():
            self.input.setFocus()

    def cancel(self):
        if self.current:
            self.current["meta"].setText("Answer stopped")
            if not self.current["answer"]:
                self.current["label"].setText("Answer stopped.")
        self.client.cancel()
        self.finish_answer()

    def new_chat(self):
        if self.current:
            self.cancel()
        self.history = []
        while self.message_layout.count() > 2:
            item = self.message_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        self.empty_label.show()
        self.input.clear()
        self.update_controls()

    def closeEvent(self, event):
        if not self._closing:
            self._closing = True
            self.timer.stop()
            self.camera.stop()
            self.client.close()
        event.accept()


def app_icon():
    image = QPixmap(128, 128)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#b8e6c8"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(4, 4, 120, 120, 28, 28)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor("#193a26"), 7))
    painter.drawRoundedRect(25, 37, 78, 55, 12, 12)
    painter.drawEllipse(48, 49, 32, 32)
    painter.end()
    return QIcon(image)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Agent WebCam")
    app.setOrganizationName("Agent WebCam")
    app.setStyle("Fusion")
    app.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont))
    app.setStyleSheet(STYLE)
    app.setWindowIcon(app_icon())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
