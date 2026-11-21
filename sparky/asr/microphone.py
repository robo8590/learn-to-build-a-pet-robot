"""Microphone capture with energy-based voice activity detection."""

import collections
import threading
import time

import numpy as np
import sounddevice as sd

from sparky.config.settings import env_float

# ---- Dynamic microphone recording (energy-based voice activity) ----
# Recording runs until the user stops talking instead of a fixed timer:
#   1. Wait for speech onset (RMS louder than REC_SILENCE_THRESHOLD).
#   2. Record until trailing silence exceeds REC_SILENCE_DURATION.
#   3. Safety caps: give up after REC_MAX_WAIT_SEC without speech onset, or
#      after REC_MAX_DURATION_SEC of total recording.
# A short pre-roll buffer is kept so the first syllable is not clipped.
REC_SAMPLE_RATE = 16000
REC_BLOCK_SEC = 0.1  # RMS analysis block size (100 ms)
REC_SILENCE_THRESHOLD = env_float("ROBOT_REC_SILENCE_THRESHOLD", 0.012)
# Longer pause tolerance for kids and non-native English speakers who need
# think-time mid-sentence; override with ROBOT_REC_SILENCE_DURATION if needed.
REC_SILENCE_DURATION = env_float("ROBOT_REC_SILENCE_DURATION", 2.0)
REC_PRE_ROLL_SEC = env_float("ROBOT_REC_PRE_ROLL_SEC", 0.5)
REC_MAX_WAIT_SEC = env_float("ROBOT_REC_MAX_WAIT_SEC", 6.0)
REC_MAX_DURATION_SEC = env_float("ROBOT_REC_MAX_DURATION_SEC", 15.0)
REC_MIN_DURATION_SEC = env_float("ROBOT_REC_MIN_DURATION_SEC", 0.5)


def record_audio(sample_rate=REC_SAMPLE_RATE):
    """Dynamically record a user utterance (length driven by their speech).

    Uses an RMS energy detector over 100 ms blocks instead of a fixed timer:
    waits for speech onset (keeping a pre-roll buffer so the first syllable is
    not clipped), records while the user talks, and stops once trailing silence
    lasts long enough to mark the end of the utterance.

    Returns the recorded audio as a 1-D float32 array, or None when nothing
    usable was captured (silence, or speech too short to transcribe).
    """
    print("\n[Listening... Speak now]")

    block_len = max(1, int(sample_rate * REC_BLOCK_SEC))
    pre_roll_blocks = max(1, int(REC_PRE_ROLL_SEC / REC_BLOCK_SEC))
    silence_limit = max(1, int(REC_SILENCE_DURATION / REC_BLOCK_SEC))
    max_wait_blocks = max(1, int(REC_MAX_WAIT_SEC / REC_BLOCK_SEC))
    max_blocks = max(1, int(REC_MAX_DURATION_SEC / REC_BLOCK_SEC))
    min_blocks = max(1, int(REC_MIN_DURATION_SEC / REC_BLOCK_SEC))

    lock = threading.Lock()
    state = {
        "started": False,       # speech onset detected?
        "elapsed_blocks": 0,    # blocks recorded before any speech
        "silence_blocks": 0,    # consecutive trailing silence blocks
        "total_blocks": 0,      # blocks recorded since speech onset
        "speech_blocks": 0,     # blocks that actually contained voice
    }
    pre_roll = collections.deque(maxlen=pre_roll_blocks)
    collected = []

    def _callback(indata, frames, time_info, status):
        block = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(block * block)))
        with lock:
            if state["started"]:
                collected.append(block)
                state["total_blocks"] += 1
                if rms >= REC_SILENCE_THRESHOLD:
                    state["silence_blocks"] = 0
                    state["speech_blocks"] += 1
                else:
                    state["silence_blocks"] += 1
            elif rms >= REC_SILENCE_THRESHOLD:
                # Speech onset: flush the pre-roll so the first syllable survives
                state["started"] = True
                collected.extend(pre_roll)
                collected.append(block)
                state["total_blocks"] = len(pre_roll) + 1
                state["speech_blocks"] = 1
                pre_roll.clear()
            else:
                pre_roll.append(block)
                state["elapsed_blocks"] += 1

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_len,
            callback=_callback,
        ):
            # Hard wall-clock cap: max wait for onset + max recording length,
            # so a stalled audio device can never hang the loop forever.
            deadline = time.time() + REC_MAX_WAIT_SEC + REC_MAX_DURATION_SEC
            while True:
                time.sleep(REC_BLOCK_SEC)
                with lock:
                    started = state["started"]
                    elapsed = state["elapsed_blocks"]
                    silence = state["silence_blocks"]
                    total = state["total_blocks"]
                if not started and elapsed >= max_wait_blocks:
                    print("[No speech detected]")
                    return None
                if started and (silence >= silence_limit or total >= max_blocks):
                    break
                if time.time() > deadline:
                    break
    except Exception as e:
        print(f"Audio Recording Error: {e}")
        return None

    with lock:
        # Require real voice content (not just pre-roll silence) plus a minimum
        # captured length, so single-block noise blips are discarded.
        usable = (
            state["started"]
            and state["speech_blocks"] >= 2
            and len(collected) >= min_blocks
        )
        audio_blocks = list(collected)

    if not usable:
        print("[Speech too short to process]")
        return None
    return np.concatenate(audio_blocks)
