# Agent-WebCam

A desktop app with a live camera preview, visual chat, and recognition mode.
Show an item to your webcam and ask Gemma what it sees, or let it narrate the
scene as live text.

## Local setup

### Software

| Component | Used by this project |
| --- | --- |
| Runtime | Python 3.11–3.15; tested with 3.11 |
| Desktop framework | PySide6 / Qt 6.12 for the native window, camera preview, and chat |
| Image processing | OpenCV contrib for frame encoding and scene-change detection |
| Model server | [Ollama](https://ollama.com), running locally |
| Vision model | Gemma 4 12B: `gemma4:12b`. Apple Silicon prefers `gemma4:12b-mlx` when that tag is already installed. |

Model weights are downloaded separately from this repository. The
[standard model](https://ollama.com/library/gemma4:12b) is about 8 GB; the
[MLX variant](https://ollama.com/library/gemma4:12b-mlx) is about 7.7 GB.
Only one variant is needed.

### Machine sizing

| Resource | Practical budget |
| --- | --- |
| System RAM / unified memory | 16 GB as a starting point; **32 GB recommended** for the model, app, and operating system to have headroom |
| GPU acceleration | Apple Silicon, or an [Ollama-supported GPU](https://docs.ollama.com/gpu). For a dedicated GPU, aim for **16 GB VRAM**. |
| Free storage | **20 GB** for one model, Ollama, Python dependencies, caches, and app builds; about **30 GB** if keeping both model variants |

These are planning estimates, not verified minimums or a promise of real-time
performance. [Google's hardware guidance](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/)
places this model in the 16 GB VRAM or unified-memory range. On Apple Silicon,
CPU and GPU share unified memory; do not add the RAM and VRAM figures together.
On a machine with a dedicated GPU, VRAM is separate from system RAM.
CPU-only inference may be substantially slower. Actual memory use depends on
the runtime and [context length](https://docs.ollama.com/context-length);
the app requests an 8,192-token context.

The current prebuilt dependencies target macOS 14+, Windows x64, or Linux with
glibc 2.34+ on x64 / glibc 2.39+ on ARM64. The lockfile has no prebuilt OpenCV
wheel for Windows ARM64. This is an early prototype: live camera, model, and
desktop-build testing has been performed on Apple Silicon macOS only.
Windows, Linux, and Intel Mac behavior has not been verified on hardware.

### Install and run

Install a supported Python version from the table above,
[uv](https://docs.astral.sh/uv/getting-started/installation/), and Ollama.
Download or clone this repository and
open a terminal in its project folder. Start Ollama, then download the model
if neither applicable variant is already installed:

```sh
ollama pull gemma4:12b
```

From this project on macOS, Windows, or Linux:

```sh
uv run agent-webcam
```

On macOS, this command builds and launches `dist/Agent WebCam.app`, which
contains the camera permission metadata macOS requires. The first launch
installs the build tools and takes a little longer. Subsequent launches reuse
the bundle until the source or dependencies change. Quit any older app window
before relaunching.

Or use Python's built-in virtual environment and pip:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or
`source .venv/bin/activate` on macOS/Linux. On macOS install the build tools:

```sh
python -m pip install -e ".[build]"
```

On Windows or Linux use `python -m pip install -e .`. Then launch with:

```sh
agent-webcam
```

The app opens its own desktop window, starts the camera, and loads Gemma.
Allow camera access when your operating system asks.

## Desktop builds

Build on the operating system you want to run it on:

```sh
uv run --extra build pyinstaller --noconfirm desktop.spec
```

On macOS, open `dist/Agent WebCam.app`. Use this app bundle for camera access:
it includes the camera usage description required by macOS. On Windows and
Linux, open the executable inside `dist/Agent WebCam/`. Each system needs its
own build; Ollama and the model remain separate prerequisites.

If an older macOS source window reports that `NSCameraUsageDescription` is
missing, close that window and run `uv run agent-webcam` again, or open the
app bundle above. Do not edit your Python installation's `Info.plist`.

## Use

Type a question or basic inspection instructions, then send. A new camera frame
is attached to each question and the answer streams into the chat. Recent text
instructions are kept for follow-up questions. Use Stop to cancel a response,
Stop camera to release the webcam, or New chat to clear the conversation.

Open the **Recognition** tab and choose **Start recognition** for automatic
scene narration. Gemma describes the first frame, then OpenCV watches for
visible changes. The last description stays in place while the scene is
similar; recognition stays active without repeatedly querying Gemma or
rephrasing the narration. A new object, movement, or a substantial color change
triggers another description. The status says **Watching for scene changes**
between updates.

The detector checks fresh frames about four times per second while idle,
ignoring small pixel noise. It compares each frame against the last frame
successfully described, so gradual changes can accumulate into an update.
Very subtle changes can fall below its threshold. Only one model request runs
at a time; no frames are queued and narration never adds messages to your chat
history. Stop and restart recognition to request a fresh description.
Model speed determines the delay for new descriptions. This samples the video
rather than interpreting every video frame or events between snapshots.
Select **Stop recognition** or return to
**Chat** to stop. Camera loss or a model error stops recognition until you
start it again.

Only the current frame is sent to Gemma. Frames are resized to a maximum edge
of 960 pixels; thinking is disabled and recognition descriptions have a shorter
output budget for faster updates. Narration is text only. No microphone access
is requested.

OpenCV handles the in-memory frame conversion and JPEG encoding. The contrib
package includes object trackers such as CSRT, but object tracking is not
implemented yet.

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

## License

The project code is available under the [MIT License](LICENSE). Third-party
libraries and model weights retain their own licenses.
