"""One-shot vision turn: cue, capture, answer, and speak."""

from sparky.llm.vision_llm import generate_vision_response
from sparky.tts.speaker import speak
from sparky.vision.camera import capture_jpeg


def run_vision_once(llm, vision_enabled, task, conversation=None):
    """Execute a single look-turn: cue, capture, answer, speak.

    When ``conversation`` is provided, the turn is remembered as text only so
    later replies can refer to the photo question and answer.
    """
    if not vision_enabled:
        speak("I can't see right now. Switch to a vision-capable model and try again.")
        return

    speak("Let me take a look.")
    image_bytes = capture_jpeg()
    if image_bytes is None:
        speak("I couldn't take a photo right now.")
        return

    reply = generate_vision_response(llm, image_bytes, task, conversation)
    speak(reply)
