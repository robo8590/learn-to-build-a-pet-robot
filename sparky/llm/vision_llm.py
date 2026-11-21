"""Vision-capable LLM responses for Sparky's photo feature."""

import base64
import re

from sparky.config.settings import VISION_MAX_TOKENS
from sparky.llm.prompts import VISION_SYSTEM_PROMPT


def image_bytes_to_data_url(image_bytes: bytes) -> str:
    """Convert JPEG bytes into a base64 data URL for the vision LLM."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _vision_completion(llm, image_bytes: bytes, task: str) -> str:
    """Run a vision completion, propagating errors to the caller."""
    image_url = image_bytes_to_data_url(image_bytes)
    instruction = task or "Describe what you see in this photo."
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": instruction},
                ],
            },
        ],
        temperature=0.25,
        max_tokens=VISION_MAX_TOKENS,
    )
    raw_text = response["choices"][0]["message"]["content"].strip()
    # Clean markdown asterisks or quotes for clean speech
    return re.sub(r"[\*\#\_`]", "", raw_text)


def generate_vision_response(llm, image_bytes: bytes, task: str) -> str:
    """Send a photo plus a task to the vision-capable LLM and return the answer.

    The image is placed first in the user message, which is the convention
    expected by Gemma 4 and Qwen3.5 multimodal templates.
    """
    try:
        return _vision_completion(llm, image_bytes, task)
    except Exception as e:
        print(f"Vision Generation Error: {e}")
        return "I had trouble looking at that photo. Can you try again?"
