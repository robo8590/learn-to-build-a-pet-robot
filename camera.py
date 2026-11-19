"""Camera helpers for Sparky's "look and ..." vision feature.

Captures a still photo from the robot's camera and returns the saved file path.
Designed to work on both macOS development (AVFoundation via OpenCV) and
Raspberry Pi 5 (V4L2 / CSI camera via OpenCV) with a single code path.
"""

import os
import time
from pathlib import Path


def _import_cv2():
    """Import OpenCV lazily so Sparky can start even without a camera installed."""
    try:
        import cv2
        return cv2
    except ImportError:
        return None

# ---- Configuration (env-overridable, consistent with the portability rule) ----
CAMERA_INDEX = int(os.getenv("ROBOT_CAMERA_INDEX", "0"))
CAPTURE_DIR = Path(os.getenv("ROBOT_CAPTURE_DIR", "captures"))
CAPTURE_WIDTH = int(os.getenv("ROBOT_CAMERA_WIDTH", "1280"))
CAPTURE_HEIGHT = int(os.getenv("ROBOT_CAMERA_HEIGHT", "720"))
WARMUP_FRAMES = int(os.getenv("ROBOT_CAMERA_WARMUP_FRAMES", "5"))


def capture_photo():
    """Take a single photo and save it to the capture directory.

    Returns the absolute path to the saved JPEG file, or None when the camera
    is unavailable / the capture failed (so the caller can speak a fallback).
    """
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    cv2 = _import_cv2()
    if cv2 is None:
        print("[Camera] OpenCV (opencv-python) is not installed; cannot take photos.")
        return None

    cap = None
    try:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print(f"[Camera] Could not open camera index {CAMERA_INDEX}")
            return None

        # Request a reasonable resolution (camera may round to nearest supported).
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

        # Discard a few warmup frames so auto-exposure / white balance settle.
        for _ in range(max(0, WARMUP_FRAMES)):
            cap.read()

        ok, frame = cap.read()
        if not ok or frame is None:
            print("[Camera] Failed to grab a frame.")
            return None

        filename = f"sparky_{int(time.time())}.jpg"
        path = CAPTURE_DIR / filename
        cv2.imwrite(str(path), frame)
        print(f"[Camera] Photo saved to: {path}")
        return str(path)
    except Exception as e:
        print(f"[Camera] Error: {e}")
        return None
    finally:
        if cap is not None:
            cap.release()
