"""Voice command parsing for Sparky."""

import re


def is_exit_command(text: str) -> bool:
    """Return True when the whole utterance is an exit phrase.

    Uses whole-utterance matching so that ``"look and read this exit sign"``
    triggers vision instead of quitting the robot.
    """
    cleaned = re.sub(r"[^\w\s']", "", text or "").strip().casefold()
    return cleaned in {"bye", "exit", "quit", "goodbye", "stop"}


LOOK_RE = re.compile(
    r"""^\s*
    (?:(?:hey|hi|ok|okay)[, ]+)?
    (?:sparky[,\s]+)?
    (?:(?:can|could|would|will)\s+you\s+)?
    (?:please\s+)?
    (?:take\s+a\s+look|look)\b
    (?:
        (?:\s*(?:,|and)\s*|\s+at\s+(?:this|that)\b\s*(?:(?:,|and)\s*)?)
        (?P<instruction>.*)
      | \s*$
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_look_command(text: str):
    """Return the task instruction when ``text`` is a "look and ..." request.

    Returns ``None`` when there is no look intent. The instruction may be empty
    (caller should default to describing the photo).
    """
    match = LOOK_RE.match(text or "")
    if not match:
        return None
    return (match.group("instruction") or "").strip()
