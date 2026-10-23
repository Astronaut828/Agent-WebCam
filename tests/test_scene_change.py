"""Scene-change checks use generated JPEGs; no camera images are saved."""

import unittest

import cv2
import numpy as np

from agent_webcam.scene_change import SceneChangeDetector, THUMBNAIL_SIZE


def jpeg(pixels):
    success, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise AssertionError("Synthetic JPEG encoding failed")
    return encoded.tobytes()


def solid(value=127):
    return np.full((288, 384, 3), value, dtype=np.uint8)


class SceneChangeTests(unittest.TestCase):
    def setUp(self):
        self.detector = SceneChangeDetector()

    def prepare(self, pixels):
        return self.detector.prepare(jpeg(pixels))

    def test_first_frame_changes_and_identical_frames_stay_quiet(self):
        frame = self.prepare(solid(255))
        self.assertEqual(frame.shape, (THUMBNAIL_SIZE[1], THUMBNAIL_SIZE[0], 3))
        self.assertTrue(self.detector.has_changed(frame))
        self.detector.remember(frame)
        for _ in range(10):
            self.assertFalse(self.detector.has_changed(self.prepare(solid(255))))

    def test_jpeg_camera_noise_and_minor_brightness_changes_are_ignored(self):
        self.detector.remember(self.prepare(solid()))
        rng = np.random.default_rng(17)
        for _ in range(5):
            noise = rng.integers(-8, 9, size=(288, 384, 3))
            noisy = np.clip(solid().astype(np.int16) + noise, 0, 255).astype(np.uint8)
            self.assertFalse(self.detector.has_changed(self.prepare(noisy)))
        self.assertFalse(self.detector.has_changed(self.prepare(solid(137))))

    def test_new_object_and_its_removal_both_change_scene(self):
        empty = self.prepare(solid(255))
        pixels = solid(255)
        pixels[80:180, 120:240] = (30, 80, 180)
        item = self.prepare(pixels)
        self.detector.remember(empty)
        self.assertTrue(self.detector.has_changed(item))
        self.detector.remember(item)
        self.assertFalse(self.detector.has_changed(item))
        self.assertTrue(self.detector.has_changed(empty))

    def test_small_localized_object_crossing_tile_boundaries_is_detected(self):
        pixels = np.full((144, 192, 3), 255, dtype=np.uint8)
        self.detector.remember(self.prepare(pixels))
        # A 12-by-12 object spans a tile intersection, below the global threshold.
        pixels[58:70, 74:86] = 0
        self.assertTrue(self.detector.has_changed(self.prepare(pixels)))

    def test_color_change_with_similar_brightness_is_detected(self):
        red = solid(0)
        red[:] = (0, 0, 255)
        green = solid(0)
        green[:] = (0, 130, 0)
        self.detector.remember(self.prepare(red))
        self.assertTrue(self.detector.has_changed(self.prepare(green)))

    def test_gradual_changes_accumulate_against_narrated_baseline(self):
        self.detector.remember(self.prepare(solid(100)))
        for value in (104, 108, 112, 116):
            self.assertFalse(self.detector.has_changed(self.prepare(solid(value))))
        changed = self.prepare(solid(124))
        self.assertTrue(self.detector.has_changed(changed))
        self.assertTrue(self.detector.has_changed(changed))
        self.detector.remember(changed)
        self.assertFalse(self.detector.has_changed(changed))

    def test_baseline_is_an_owned_copy_and_reset_forces_another_observation(self):
        frame = self.prepare(solid())
        self.detector.remember(frame)
        frame[:] = 255
        self.assertTrue(self.detector.has_changed(frame))
        unchanged = self.prepare(solid())
        self.assertFalse(self.detector.has_changed(unchanged))
        self.detector.reset()
        self.assertTrue(self.detector.has_changed(unchanged))

    def test_malformed_frames_report_actionable_error(self):
        for value in (b"", b"not a JPEG", b"\xff\xd8\xff\xd9", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Restart the camera"):
                    self.detector.prepare(value)


if __name__ == "__main__":
    unittest.main()
