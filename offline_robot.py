import argparse
import sounddevice as sd
import numpy as np
import collections
import io
import os
import re
import threading
import time
import base64
import unicodedata
import wave
from faster_whisper import WhisperModel
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from piper import PiperVoice
from piper.config import SynthesisConfig

from model_config import (
    DEFAULT_LLM_MODEL,
    PIPER_VOICE_PATH,
    WHISPER_MODEL_PATH,
    available_llm_keys,
    get_llm_config,
    is_llm_downloaded,
    is_vision_ready,
    llm_path,
    mmproj_path,
    resolve_llm_model_key,
)

from camera import capture_photo

# ==========================================
# 0. CONFIGURATION
# ==========================================
# Robot Voice Effect Profile:
#   - "cute_companion": pitch shift + subtle robot DSP (default, kid-friendly)
#   - "robot_droid": pitch shift + stronger robot DSP
#   - "natural": pure, unmodified Piper voice (no pitch shift, no DSP)
ROBOT_VOICE_STYLE = os.getenv("ROBOT_VOICE_STYLE", "cute_companion").strip().lower()


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to a default."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back to a default."""
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# LLM runtime tuning. On a Raspberry Pi 5, set ROBOT_LLM_GPU_LAYERS=0.
LLM_N_GPU_LAYERS = _env_int("ROBOT_LLM_GPU_LAYERS", -1)
LLM_N_THREADS = _env_int("ROBOT_LLM_THREADS", 4)

# Keep spoken replies short; see the latency targets in PROJECT_MISSION.md.
LLM_MAX_TOKENS = _env_int("ROBOT_LLM_MAX_TOKENS", 96)

# Initial context prompt to prime Whisper's tokenizer for natural English speech
WHISPER_INITIAL_PROMPT = "Hello, this is a spoken conversation in English."

# English-dedicated models are fast and accurate enough with a small beam size
WHISPER_BEAM_SIZE = 2

# ---- Dynamic microphone recording (energy-based voice activity) ----
# Recording runs until the user stops talking instead of a fixed timer:
#   1. Wait for speech onset (RMS louder than REC_SILENCE_THRESHOLD).
#   2. Record until trailing silence exceeds REC_SILENCE_DURATION.
#   3. Safety caps: give up after REC_MAX_WAIT_SEC without speech onset, or
#      after REC_MAX_DURATION_SEC of total recording.
# A short pre-roll buffer is kept so the first syllable is not clipped.
REC_SAMPLE_RATE = 16000
REC_BLOCK_SEC = 0.1  # RMS analysis block size (100 ms)
REC_SILENCE_THRESHOLD = _env_float("ROBOT_REC_SILENCE_THRESHOLD", 0.012)
# Longer pause tolerance for kids and non-native English speakers who need
# think-time mid-sentence; override with ROBOT_REC_SILENCE_DURATION if needed.
REC_SILENCE_DURATION = _env_float("ROBOT_REC_SILENCE_DURATION", 2.0)
REC_PRE_ROLL_SEC = _env_float("ROBOT_REC_PRE_ROLL_SEC", 0.5)
REC_MAX_WAIT_SEC = _env_float("ROBOT_REC_MAX_WAIT_SEC", 6.0)
REC_MAX_DURATION_SEC = _env_float("ROBOT_REC_MAX_DURATION_SEC", 15.0)
REC_MIN_DURATION_SEC = _env_float("ROBOT_REC_MIN_DURATION_SEC", 0.5)


# ==========================================
# 1. TEXT PREPROCESSING & NORMALIZATION FOR TTS
# ==========================================
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

    # 2. Strip leftover markdown artifacts, bullet points, brackets, and quotes
    text = re.sub(r"[\*\#\_`~<>]", " ", text)
    text = re.sub(r"\[.*?\]|\(.*?\)", " ", text)
    text = re.sub(r"^[\s\-\*\•\d+\.]+\s+", "", text)  # leading bullets/numbering

    # 3. Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    # 4. Ensure trailing punctuation so Piper handles sentence-final pitch cadence
    if text and text[-1] not in ".!?":
        text += "."

    return text


# Piper synthesis configuration (warm, friendly setting)
PIPER_SYNTHESIS_CONFIG = SynthesisConfig(length_scale=0.92, noise_scale=0.667, noise_w_scale=0.8)

# Pitch shift semitones (+N semitones raises the pitch formant into an adorable,
# youthful companion robot register without chipmunk distortion):
PITCH_SHIFT_SEMITONES = 2.5


def pitch_shift(audio_int16: np.ndarray, semitones: float, n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    """
    High-performance phase-vocoder pitch shifter in pure NumPy.
    Shifts the audio pitch by `semitones` while preserving timing and duration.
    """
    if semitones == 0:
        return audio_int16

    audio = audio_int16.astype(np.float32)
    alpha = 2.0 ** (semitones / 12.0)
    window = np.hanning(n_fft)
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    if num_frames <= 1:
        return audio_int16

    # Frame extraction with windowing
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(num_frames, n_fft),
        strides=(audio.strides[0] * hop_length, audio.strides[0]),
    ) * window

    stft = np.fft.rfft(frames, axis=-1)
    time_steps = np.arange(0, num_frames - 1, 1.0 / alpha)
    stretched_stft = np.zeros((len(time_steps), stft.shape[1]), dtype=complex)

    phase_adv = np.linspace(0, np.pi * hop_length, stft.shape[1])
    phase_accum = np.angle(stft[0])
    stretched_stft[0] = stft[0]

    for i, t in enumerate(time_steps[1:], 1):
        idx = int(t)
        frac = t - idx
        mag = (1 - frac) * np.abs(stft[idx]) + frac * np.abs(stft[min(idx + 1, num_frames - 1)])
        dphase = np.angle(stft[min(idx + 1, num_frames - 1)]) - np.angle(stft[idx]) - phase_adv
        dphase = dphase - 2 * np.pi * np.round(dphase / (2 * np.pi))
        phase_accum += phase_adv + dphase
        stretched_stft[i] = mag * np.exp(1j * phase_accum)

    # ISTFT
    stretched_frames = np.fft.irfft(stretched_stft, axis=-1) * window
    out_len = (len(stretched_frames) - 1) * hop_length + n_fft
    out_audio = np.zeros(out_len, dtype=np.float32)
    window_sum = np.zeros(out_len, dtype=np.float32)

    for i in range(len(stretched_frames)):
        start = i * hop_length
        out_audio[start : start + n_fft] += stretched_frames[i]
        window_sum[start : start + n_fft] += window**2

    nonzero = window_sum > 1e-4
    out_audio[nonzero] /= window_sum[nonzero]

    # Resample to match original duration and shift pitch
    target_len = len(audio)
    resampled_indices = np.linspace(0, len(out_audio) - 1, target_len)
    shifted = np.interp(resampled_indices, np.arange(len(out_audio)), out_audio)
    return np.clip(shifted, -32768, 32767).astype(np.int16)


def apply_robot_dsp(
    audio_int16: np.ndarray,
    samplerate: int,
    style: str = ROBOT_VOICE_STYLE,
) -> np.ndarray:
    """
    Applies real-time DSP acoustic effects in pure NumPy to transform human voice
    into a cute, friendly companion robot:
    - 'cute_companion': Subtle metallic comb resonance + subtle ring modulation.
      Sounds like a high-tech, warm robotic buddy (Wall-E / BB-8 inspired warmth).
    - 'robot_droid': Distinct ring modulation & robotic resonance.
    - 'natural': Pure, unmodified voice without pitch shift or metallic modulation.
    """
    if style == "natural" or len(audio_int16) == 0:
        return audio_int16

    audio = audio_int16.astype(np.float32)

    # Effect parameters tuned for child-friendly robot
    if style == "robot_droid":
        mod_freq = 42.0
        ring_depth = 0.40
        delay_ms = 3.5
        comb_feedback = 0.45
        comb_blend = 0.40
    else:  # 'cute_companion' (default)
        mod_freq = 32.0
        ring_depth = 0.22
        delay_ms = 2.6
        comb_feedback = 0.32
        comb_blend = 0.28

    # 1. Subtle Ring Modulation (adds warm robotic carrier harmonics)
    t = np.arange(len(audio)) / float(samplerate)
    carrier = np.sin(2.0 * np.pi * mod_freq * t)
    ring = (1.0 - ring_depth) * audio + ring_depth * (audio * carrier)

    # 2. Resonant Metallic Comb Filter (creates robot chassis acoustic cavity)
    delay_samples = max(1, int((delay_ms / 1000.0) * samplerate))
    comb = np.copy(ring)
    for k in range(1, 4):
        shift = k * delay_samples
        if shift < len(ring):
            comb[shift:] += (comb_feedback**k) * ring[:-shift]

    out = (1.0 - comb_blend) * ring + comb_blend * comb

    # Normalization to prevent digital clipping
    max_val = np.max(np.abs(out))
    if max_val > 32000:
        out = (out / max_val) * 32000

    return np.clip(out, -32768, 32767).astype(np.int16)


# ==========================================
# 2. INITIALIZE OFFLINE COMPONENTS
# ==========================================
_piper_voice = None


def get_piper_voice():
    """Load (and cache) the local English Piper voice."""
    global _piper_voice

    if not PIPER_VOICE_PATH.is_file():
        raise FileNotFoundError(
            f"Local Piper voice model not found at '{PIPER_VOICE_PATH}'. "
            "Run 'python download_models.py' to download required voices."
        )

    if _piper_voice is None:
        print("Loading local Piper voice...")
        _piper_voice = PiperVoice.load(str(PIPER_VOICE_PATH))

    if _piper_voice is None:
        raise RuntimeError("Piper voice failed to load.")
    return _piper_voice


def speak(text):
    """Speak text with the local English voice and the companion robot effect."""
    print(f"\nRobot: {text}")
    try:
        clean_text = preprocess_text_for_tts(text)
        if not clean_text:
            return

        voice = get_piper_voice()

        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            voice.synthesize_wav(clean_text, wav_file, syn_config=PIPER_SYNTHESIS_CONFIG)

        audio_buffer.seek(0)
        with wave.open(audio_buffer, "rb") as wav_file:
            audio = np.frombuffer(
                wav_file.readframes(wav_file.getnframes()), dtype=np.int16
            )
            samplerate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()

            # Apply child-like companion robot pitch-shift
            # (skipped when ROBOT_VOICE_STYLE == "natural")
            shift_val = PITCH_SHIFT_SEMITONES
            if shift_val != 0.0 and len(audio) > 0 and ROBOT_VOICE_STYLE != "natural":
                if num_channels > 1:
                    audio = audio.reshape(-1, num_channels)
                    channels_shifted = [
                        pitch_shift(audio[:, ch], shift_val) for ch in range(num_channels)
                    ]
                    audio = np.column_stack(channels_shifted)
                else:
                    audio = pitch_shift(audio, shift_val)
            elif num_channels > 1:
                audio = audio.reshape(-1, num_channels)

            # Apply robotic DSP (resonance + ring modulation) to make it unmistakably a robot
            if len(audio) > 0 and ROBOT_VOICE_STYLE != "natural":
                if num_channels > 1:
                    channels_robot = [
                        apply_robot_dsp(audio[:, ch], samplerate, ROBOT_VOICE_STYLE)
                        for ch in range(num_channels)
                    ]
                    audio = np.column_stack(channels_robot)
                else:
                    audio = apply_robot_dsp(audio, samplerate, ROBOT_VOICE_STYLE)

            sd.play(audio, samplerate=samplerate)
            sd.wait()
    except Exception as e:
        print(f"TTS Error: {e}")


def resolve_whisper_path():
    if WHISPER_MODEL_PATH.is_dir() and (WHISPER_MODEL_PATH / "model.bin").is_file():
        return str(WHISPER_MODEL_PATH)
    raise FileNotFoundError(
        f"Whisper model directory not found at '{WHISPER_MODEL_PATH}'. "
        "Please run 'python download_models.py' first."
    )


def initialize_whisper():
    model_path = resolve_whisper_path()
    print(f"Loading local Whisper model from: {model_path}...")
    return WhisperModel(
        model_path,
        device="cpu",
        compute_type="int8",
    )


def initialize_llm(model_key=None):
    """Load the selected local GGUF model.

    Selection precedence: ``model_key`` (CLI) -> ``ROBOT_LLM_MODEL`` env var ->
    ``DEFAULT_LLM_MODEL``.

    Returns ``(llm, vision_enabled)`` where ``vision_enabled`` is True when the
    model supports vision and its mmproj projector was loaded, so image input
    is usable for the "look and ..." feature.
    """
    try:
        resolved_key = resolve_llm_model_key(
            cli_value=model_key,
            env_value=os.getenv("ROBOT_LLM_MODEL"),
        )
    except KeyError as e:
        raise SystemExit(str(e))

    config = get_llm_config(resolved_key)
    path = llm_path(resolved_key)

    if not path.is_file():
        raise SystemExit(
            f"LLM model '{resolved_key}' not found at '{path}'.\n"
            f"Download it with: python download_models.py --llm {resolved_key}\n"
            f"Available models: {', '.join(available_llm_keys())}"
        )

    print(f"Loading local LLM: {config['label']} ({resolved_key})...")

    chat_handler = None
    if is_vision_ready(resolved_key):
        projector = mmproj_path(resolved_key)
        print(f"Loading vision projector: {projector}...")
        chat_handler = Llava15ChatHandler(clip_model_path=str(projector))

    llm = Llama(
        model_path=str(path),
        n_ctx=config["n_ctx"],
        n_threads=LLM_N_THREADS,
        n_gpu_layers=LLM_N_GPU_LAYERS,
        chat_handler=chat_handler,
        verbose=False,
    )

    vision_enabled = chat_handler is not None
    return llm, vision_enabled


# ==========================================
# 3. PERSONALITY & PROMPTS
# ==========================================
SYSTEM_PROMPT = (
    "You are Sparky, a cheerful, cute, and encouraging companion robot friend for kids! "
    "Always reply in English with warmth, enthusiasm, and child-friendly curiosity. "
    "Keep answers short (1-3 sentences maximum). "
    "Never use formatting symbols, markdown, emojis, or bullet points."
)

VISION_SYSTEM_PROMPT = (
    "You are Sparky, a cheerful, cute, and encouraging companion robot. "
    "The user has shared a photo and asked you to do something with it. "
    "Answer in English with warmth and kid-friendly curiosity. "
    "Keep answers short (1-3 sentences maximum). "
    "Never use formatting symbols, markdown, emojis, or bullet points. "
    "If the photo is not clear enough to answer, say so gently."
)

# ==========================================
# 4. CORE AUDIO & INFERENCE FUNCTIONS
# ==========================================
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


def transcribe_audio(whisper_model, audio_data):
    try:
        segments, info = whisper_model.transcribe(
            audio_data,
            beam_size=WHISPER_BEAM_SIZE,
            language="en",
            initial_prompt=WHISPER_INITIAL_PROMPT,
            vad_filter=True,  # Filter out silence and noise hallucinations
        )
        text = " ".join([segment.text for segment in segments]).strip()
        print(f"You said: {text}")
        return text
    except Exception as e:
        print(f"Transcription Error: {e}")
        return ""


def generate_response(llm, user_text):
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            temperature=0.7,
            max_tokens=LLM_MAX_TOKENS,
        )
        raw_text = response["choices"][0]["message"]["content"].strip()
        # Clean markdown asterisks or quotes for clean speech
        cleaned_text = re.sub(r"[\*\#\_`]", "", raw_text)
        return cleaned_text
    except Exception as e:
        print(f"LLM Generation Error: {e}")
        return "Oops! My circuits got tangled for a second. Can you say that again?"


def image_to_data_url(image_path: str) -> str:
    """Convert a local image file into a base64 data URL for the vision LLM."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def generate_vision_response(llm, image_path: str, task: str) -> str:
    """Send a photo plus a task to the vision-capable LLM and return the answer."""
    try:
        image_url = image_to_data_url(image_path)
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            temperature=0.7,
            max_tokens=LLM_MAX_TOKENS,
        )
        raw_text = response["choices"][0]["message"]["content"].strip()
        # Clean markdown asterisks or quotes for clean speech
        cleaned_text = re.sub(r"[\*\#\_`]", "", raw_text)
        return cleaned_text
    except Exception as e:
        print(f"Vision Generation Error: {e}")
        return "I had trouble looking at that photo. Can you try again?"


# ==========================================
# 5. COMMAND-LINE INTERFACE
# ==========================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Sparky - offline voice companion and tutor robot.",
    )
    parser.add_argument(
        "--model",
        metavar="KEY",
        help="LLM model to load (overrides the ROBOT_LLM_MODEL env var).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available LLM models and exit.",
    )
    return parser.parse_args(argv)


def print_models_table():
    print("Available LLM models:")
    for key in available_llm_keys():
        config = get_llm_config(key)
        marker = " *default" if key == DEFAULT_LLM_MODEL else ""
        vision = " [vision]" if get_llm_config(key).get("supports_vision", False) else ""
        status = "downloaded" if is_llm_downloaded(key) else "not downloaded"
        print(f"  {key}{marker}{vision}")
        print(
            f"    {config['label']} | {config['size_gb']:.2f} GB | "
            f"{status} | {config['notes']}"
        )
    print("\nDownload a model with: python download_models.py --llm KEY")
    print("Download a vision projector with: python download_models.py --mmproj KEY")


# ==========================================
# 6. MAIN INTERACTIVE LOOP
# ==========================================
if __name__ == "__main__":
    args = parse_args()

    if args.list_models:
        print_models_table()
        raise SystemExit(0)

    print("=" * 50)
    print(" Starting Sparky Robot")
    print("=" * 50)

    llm, vision_enabled = initialize_llm(args.model)
    whisper_model = initialize_whisper()

    # Friendly greeting
    speak("Hello! I am Sparky, your offline robot assistant. Let's talk!")

    exit_phrases = ("bye", "exit", "quit", "goodbye")

    while True:
        audio = record_audio()
        if audio is not None:
            user_speech = transcribe_audio(whisper_model, audio)

            if user_speech:
                # Check for exit commands
                lower_speech = user_speech.casefold()
                if any(re.search(rf"\b{re.escape(phrase)}\b", lower_speech) for phrase in exit_phrases):
                    speak("Goodbye! Have a wonderful day!")
                    break

                # "look and <task>" vision trigger: take a photo and ask the LLM
                look_match = re.search(r"\blook and\b\s+(.+)", user_speech, re.IGNORECASE)
                if look_match:
                    task = look_match.group(1).strip()
                    if not vision_enabled:
                        # Current model cannot process images.
                        speak("I can't see right now")
                        continue
                    speak("Let me take a look.")
                    image_path = capture_photo()
                    if image_path is None:
                        speak("I couldn't take a photo right now.")
                        continue
                    reply = generate_vision_response(llm, image_path, task)
                    speak(reply)
                    continue

                # Generate AI reply and speak it
                reply = generate_response(llm, user_speech)
                speak(reply)
