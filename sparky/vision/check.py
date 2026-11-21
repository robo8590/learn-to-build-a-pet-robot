"""Diagnostic check for the selected model's vision projector."""

from sparky.llm.loader import initialize_llm
from sparky.llm.vision_llm import _vision_completion


def check_vision(model_key=None):
    """Run one tiny image through the selected model's projector.

    Returns 0 on success and 1 when vision is unavailable or the call fails.
    This catches projector incompatibilities before a live conversation.
    """
    llm, vision_enabled = initialize_llm(model_key)
    if not vision_enabled:
        print("[Vision] FAIL: no usable vision projector for this model.")
        return 1

    try:
        import cv2
        import numpy as np

        image = np.full((128, 320, 3), 255, dtype=np.uint8)
        cv2.putText(
            image,
            "3 + 4 = ?",
            (12, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 0),
            4,
        )
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            print("[Vision] FAIL: could not encode the test image.")
            return 1

        reply = _vision_completion(
            llm, buf.tobytes(), "Solve the math problem. Answer briefly."
        )
        print(f"[Vision] OK: {reply}")
        return 0
    except Exception as e:
        print(f"[Vision] FAIL: {e}")
        return 1
