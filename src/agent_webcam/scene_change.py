"""Lightweight visual change detection using in-memory camera thumbnails."""

import cv2
import numpy as np


THUMBNAIL_SIZE = (192, 144)
PIXEL_CHANGE_THRESHOLD = 20
SCENE_CHANGE_FRACTION = 0.015
LOCAL_REGION_SIZE = 16
LOCAL_CHANGE_FRACTION = 0.25


class SceneChangeDetector:
    """Compare with the last narrated frame, ignoring small camera noise."""

    def __init__(self):
        self._baseline = None

    def prepare(self, image: bytes) -> np.ndarray:
        """Decode a snapshot into a small, smoothed color thumbnail."""
        if not isinstance(image, bytes) or not image:
            raise ValueError("The camera frame could not be read. Restart the camera and try again.")
        try:
            frame = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("The camera frame could not be read. Restart the camera and try again.")
            thumbnail = cv2.resize(frame, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
            return cv2.GaussianBlur(thumbnail, (3, 3), 0)
        except cv2.error as error:
            raise ValueError("The camera frame could not be read. Restart the camera and try again.") from error

    def has_changed(self, thumbnail: np.ndarray) -> bool:
        """Detect broad or localized changes without advancing the baseline."""
        if self._baseline is None:
            return True
        difference = cv2.absdiff(thumbnail, self._baseline)
        changed = np.max(difference, axis=2) >= PIXEL_CHANGE_THRESHOLD
        if np.mean(changed) >= SCENE_CHANGE_FRACTION:
            return True
        local_fraction = cv2.boxFilter(
            changed.astype(np.float32), cv2.CV_32F,
            (LOCAL_REGION_SIZE, LOCAL_REGION_SIZE),
            normalize=True, borderType=cv2.BORDER_CONSTANT,
        )
        return bool(np.max(local_fraction) >= LOCAL_CHANGE_FRACTION)

    def remember(self, thumbnail: np.ndarray) -> None:
        """Commit only a frame whose narration completed successfully."""
        self._baseline = thumbnail.copy()

    def reset(self) -> None:
        self._baseline = None
