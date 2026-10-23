# Agent-WebCam

A desktop app with a live camera preview and chat. Show an item to your webcam
and ask Gemma what it sees. Built with Python, Qt, and OpenCV for Windows,
macOS, and Linux.

## Run

Install Python 3.11 or newer and [Ollama](https://ollama.com). Start Ollama and
download the cross-platform Gemma model:

```sh
ollama pull gemma4:12b
```

For development on Windows or Linux, from this project:

```sh
uv run agent-webcam
```

Or use Python's built-in virtual environment and pip:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or
`source .venv/bin/activate` on macOS/Linux, then:

```sh
python -m pip install -e .
agent-webcam
```

The app opens its own desktop window, starts the camera, and loads Gemma.
Allow camera access when your operating system asks. The installed
`gemma4:12b-mlx` model is reused on Apple Silicon when available; Windows and
Linux use `gemma4:12b`. Inference speed and model compatibility depend on your
hardware and Ollama version.

## Desktop builds

Build on the operating system you want to run it on:

```sh
uv run --extra build pyinstaller --noconfirm desktop.spec
```

On macOS, open `dist/Agent WebCam.app`. Use this app bundle for camera access:
it includes the camera usage description required by macOS. On Windows and
Linux, open the executable inside `dist/Agent WebCam/`. Each system needs its
own build; Ollama and the model remain separate prerequisites.

## Use

Type a question or basic inspection instructions, then send. A new camera frame
is attached to each question and the answer streams into the chat. Recent text
instructions are kept for follow-up questions. Use Stop to cancel a response,
Stop camera to release the webcam, or New chat to clear the conversation.

Only the current frame is sent to Gemma. It does not continuously analyze the
video between questions. Frames are resized to a maximum edge of 960 pixels;
thinking is disabled for faster answers. No microphone access is requested.

OpenCV handles the in-memory frame conversion and JPEG encoding. The contrib
package includes object trackers such as CSRT for the next tracking feature;
this first window provides preview and visual chat, without tracking controls.

## Local data

The app talks directly to Ollama at `127.0.0.1:11434`. It runs no web server,
opens no browser, loads no external UI assets, and saves no frames or chat
history. Current text and frames are held in memory. Ollama keeps the model
loaded for up to 30 minutes after a request; closing the app does not guarantee
immediate model unloading or erasure from system memory.

This is a visual Q&A prototype. Test representative good and defective items
before using it for quality-control decisions.

## Development

```sh
uv run python -m unittest discover -s tests -v
```

Tests use synthetic images and mocked model responses. On a headless system,
set `QT_QPA_PLATFORM=offscreen` before running them. Keep private camera data,
credentials, local settings, and generated output out of Git. Project rules
are in [AGENTS.md](AGENTS.md).
