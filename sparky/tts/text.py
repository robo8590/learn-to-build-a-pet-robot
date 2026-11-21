"""Text preprocessing and normalization for TTS."""

import re
import unicodedata


def preprocess_text_for_tts(text: str) -> str:
    """
    Cleans and normalizes text before feeding it to Piper TTS.
    - Normalizes Unicode to NFC.
    - Cleans markdown/formatting symbols.
    - Ensures clean sentence-ending punctuation for natural prosodic pauses.
    """
    if not text:
        return ""

    # 1. Unicode normalization to NFC
    text = unicodedata.normalize("NFC", text)

    # 2. Strip emojis (models sometimes emit them despite the no-emoji prompt)
    text = re.sub(r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U000024C2-\U0001F251]", "", text)
    # Strip leftover markdown artifacts, bullet points, brackets, and quotes
    text = re.sub(r"[\*\#\_`~<>]", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)  # keep (parentheses): math steps
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text)  # leading bullets/numbering

    # 3. Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    # 4. Ensure trailing punctuation so Piper handles sentence-final pitch cadence
    if text and text[-1] not in ".!?":
        text += "."

    return text
