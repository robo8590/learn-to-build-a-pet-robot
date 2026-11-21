"""Camera helpers for Sparky's "look and ..." vision feature.

Captures a still photo from the robot's camera and returns the JPEG bytes.
Designed to work on both macOS development (AVFoundation via OpenCV) and
Raspberry Pi 5 (CSI camera via picamera2 when available, otherwise OpenCV /
V4L2) with a single code path.

Photos are not written to disk by default. Set ``ROBOT_KEEP_CAPTURES=1`` to
save frames under ``captures/`` for debugging (directory is git-ignored).
"""

import os
import time
from pathlib import Path

from sparky.config.paths import BASE_DIR
from sparky.config.settings import env_int

# ---- Configuration (env-overridable, consistent with the portability rule) ----
CAMERA_BACKEND = os.getenv("ROBOT_CAMERA_BACKEND", "auto").strip().lower()
CAMERA_INDEX = env_int("ROBOT_CAMERA_INDEX", 0)
CAPTURE_DIR = Path(os.getenv("ROBOT_CAPTURE_DIR", str(BASE_DIR / "captures")))
CAPTURE_WIDTH = env_int("ROBOT_CAMERA_WIDTH", 1280)
CAPTURE_HEIGHT = env_int("ROBOT_CAMERA_HEIGHT", 720)
WARMUP_FRAMES = env_int("ROBOT_CAMERA_WARMUP_FRAMES", 5)
# Max image side length sent to the vision model. Smaller = faster inference,
# fewer image tokens. Use 512 on a Pi, 768 on a Mac.
IMAGE_MAX_DIM = max(64, env_int("ROBOT_IMAGE_MAX_DIM", 768))
IMAGE_JPEG_QUALITY = min(100, max(1, env_int("ROBOT_IMAGE_JPEG_QUALITY", 85)))
KEEP_CAPTURES = os.getenv("ROBOT_KEEP_CAPTURES", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _import_cv2():
    """Import OpenCV lazily so Sparky can start even without a camera installed."""
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def _open_opencv_camera():
    """Open an OpenCV VideoCapture and warm up auto-exposure / white balance."""
    cv2 = _import_cv2()
    if cv2 is None:
        return None, None

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        return None, cv2

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    for _ in range(max(0, WARMUP_FRAMES)):
        cap.read()

    return cap, cv2


def _open_picamera2():
    """Open a Raspberry Pi CSI camera via picamera2, if available."""
    try:
        from picamera2 import Picamera2
    except ImportError:
        return None

    try:
        picam2 = Picamera2()
        # Use a still-friendly configuration.
        # Request RGB888 explicitly: the default stream format varies by
        # platform and may include an alpha channel.
        config = picam2.create_still_configuration(
            main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        # Let auto-exposure settle.
        time.sleep(0.5)
        return picam2
    except Exception:
        return None


def _capture_opencv(cap, cv2):
    """Grab a single frame from an OpenCV camera."""
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _capture_picamera2(picam2):
    """Grab a single frame from picamera2 as a BGR numpy array."""
    rgb = picam2.capture_array()
    # RGB888 should be 3 channels, but some builds return alpha anyway.
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    # picamera2 delivers RGB; OpenCV expects BGR.
    return rgb[:, :, ::-1].copy()


def _resize_and_encode(frame, max_dim: int, quality: int):
    """Downscale to ``max_dim`` on the longest side and encode as JPEG bytes."""
    import cv2

    h, w = frame.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    ok, buf = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        return None
    return buf.tobytes()


def _save_capture(jpeg_bytes: bytes):
    """Optionally persist a capture to disk for debugging."""
    if not KEEP_CAPTURES:
        return
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"sparky_{int(time.time() * 1000)}.jpg"
    path = CAPTURE_DIR / filename
    path.write_bytes(jpeg_bytes)
    print(f"[Camera] Photo saved to: {path}")


def capture_jpeg(max_dim: int | None = None, quality: int | None = None):
    """Take a single photo and return the JPEG bytes.

    Returns ``None`` when the camera is unavailable or the capture failed, so
    the caller can speak a fallback. No file is written unless
    ``ROBOT_KEEP_CAPTURES=1``.
    """
    if max_dim is None:
        max_dim = IMAGE_MAX_DIM
    if quality is None:
        quality = IMAGE_JPEG_QUALITY
    max_dim = max(64, max_dim)
    quality = min(100, max(1, quality))

    backend = CAMERA_BACKEND
    if backend not in ("auto", "picamera2", "opencv"):
        print(f"[Camera] Unknown ROBOT_CAMERA_BACKEND={backend!r}; using auto.")
        backend = "auto"

    # picamera2 is tried first for Pi CSI cameras; OpenCV is the fallback.
    picam2 = None
    cap = None
    cv2 = None

    if backend in ("auto", "picamera2"):
        picam2 = _open_picamera2()
        if picam2 is None and backend == "picamera2":
            print("[Camera] picamera2 backend requested but unavailable.")

    if picam2 is None and backend in ("auto", "opencv"):
        cap, cv2 = _open_opencv_camera()
        if cap is None and backend == "opencv":
            print("[Camera] OpenCV backend requested but unavailable.")

    if picam2 is None and cap is None:
        print("[Camera] No camera backend available (install opencv-python or picamera2).")
        return None

    try:
        if picam2 is not None:
            frame = _capture_picamera2(picam2)
        else:
            frame = _capture_opencv(cap, cv2)

        if frame is None:
            print("[Camera] Failed to grab a frame.")
            return None

        jpeg_bytes = _resize_and_encode(frame, max_dim, quality)
        if jpeg_bytes is None:
            print("[Camera] Failed to encode JPEG.")
            return None

        _save_capture(jpeg_bytes)
        return jpeg_bytes
    finally:
        if picam2 is not None:
            try:
                picam2.stop()
            except Exception:
                pass
        if cap is not None:
            cap.release()
