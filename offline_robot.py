import sounddevice as sd
import numpy as np
import gc
import io
import os
import re
import sys
import unicodedata
import wave
from pathlib import Path
from faster_whisper import WhisperModel
from llama_cpp import Llama
from piper import PiperVoice
from piper.config import SynthesisConfig

# ==========================================
# 0. CONFIGURATION & PRE-SET LANGUAGE
# ==========================================
# Preset language for the robot: "vi" or "en"
# Can be overridden via environment variable: export ROBOT_LANG=vi
ROBOT_LANG = os.getenv("ROBOT_LANG", "vi").strip().lower()
if ROBOT_LANG not in ("vi", "en"):
    print(f"Warning: Unsupported ROBOT_LANG='{ROBOT_LANG}'. Defaulting to 'vi'.")
    ROBOT_LANG = "vi"

# Robot Voice Effect Profile:
#   - "cute_companion": pitch shift + subtle robot DSP (default, kid-friendly)
#   - "robot_droid": pitch shift + stronger robot DSP
#   - "natural": pure, unmodified Piper voice (no pitch shift, no DSP)
ROBOT_VOICE_STYLE = os.getenv("ROBOT_VOICE_STYLE", "cute_companion").strip().lower()

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Language-specific Whisper paths (PhoWhisper for 'vi', tiny.en/base.en for 'en')
WHISPER_MODEL_PATHS = {
    "vi": MODELS_DIR / "whisper" / "vi",
    "en": MODELS_DIR / "whisper" / "en",
}

VOICE_MODELS = {
    "en": MODELS_DIR / "piper" / "en_US-lessac-medium.onnx",
}

# ==========================================
# VIETNAMESE VOICE PROFILES (kid-friendly companion robot)
# ==========================================
# Each profile picks a Piper model + optional speaker_id + synthesis tuning so
# Sparky sounds like a friendly, encouraging, cute companion robot for kids.
#
# Available Vietnamese profiles:
#   - vais1000_female : warm, clear female voice (Vais1000 dataset) - the previous default
#   - csa_child       : young & cute (CSA Voice V3 speaker 4, Yen Nhi)  ★ best for kids
#   - csa_gentle      : gentle female (CSA Voice V3 speaker 0, Ngoc Lan)
#   - csa_cheerful    : cheerful female (CSA Voice V3 speaker 1, Minh Hanh)
#   - csa_bright      : bright & curious female (CSA Voice V3 speaker 3, Thu Ha)
#   - single_male     : male voice (25 hours single, Quang Huy, lower quality)
#
# Select via env var:  export VI_VOICE_PROFILE=csa_child

VI_VOICE_PROFILES = {
    "vais1000_female": {
        "model": "vi_VN-vais1000-medium.onnx",
        "speaker_id": None,
        "length_scale": 0.90,
        "noise_scale": 0.75,
        "noise_w_scale": 0.85,
        "pitch_semitones": 3.6,
        "desc": "Nồng ấm, rõ ràng - giọng nữ ấm áp (Vais1000)",
    },
    "csa_child": {
        "model": "vi_VN-csa-voice-piper-v3-medium.onnx",
        "speaker_id": 4,
        "length_scale": 0.92,
        "noise_scale": 0.78,
        "noise_w_scale": 0.88,
        "pitch_semitones": 3.2,
        "desc": "Ngầu, đáng yêu, giọng trẻ em (Yen Nhi - CSA Voice speaker 4) ★ khuyên dùng cho trẻ em",
    },
    "csa_gentle": {
        "model": "vi_VN-csa-voice-piper-v3-medium.onnx",
        "speaker_id": 0,
        "length_scale": 0.90,
        "noise_scale": 0.70,
        "noise_w_scale": 0.82,
        "pitch_semitones": 3.0,
        "desc": "Dịu dàng, nhẹ nhàng (Ngoc Lan - CSA Voice speaker 0)",
    },
    "csa_cheerful": {
        "model": "vi_VN-csa-voice-piper-v3-medium.onnx",
        "speaker_id": 1,
        "length_scale": 0.88,
        "noise_scale": 0.80,
        "noise_w_scale": 0.90,
        "pitch_semitones": 3.4,
        "desc": "Vui tươi, hào hứng (Minh Hanh - CSA Voice speaker 1)",
    },
    "csa_bright": {
        "model": "vi_VN-csa-voice-piper-v3-medium.onnx",
        "speaker_id": 3,
        "length_scale": 0.89,
        "noise_scale": 0.76,
        "noise_w_scale": 0.86,
        "pitch_semitones": 3.2,
        "desc": "Sáng tạo, tò mò (Thu Ha - CSA Voice speaker 3)",
    },
    "single_male": {
        "model": "vi_VN-25hours_single-low.onnx",
        "speaker_id": None,
        "length_scale": 0.95,
        "noise_scale": 0.667,
        "noise_w_scale": 0.80,
        "pitch_semitones": 2.0,
        "desc": "Giọng nam (Quang Huy - 25 giờ đơn, chất lượng thấp hơn)",
    },
}

# Pick the Vietnamese voice profile (default: cute child voice for kids)
VI_VOICE_PROFILE = os.getenv("VI_VOICE_PROFILE", "csa_child").strip().lower()
if VI_VOICE_PROFILE not in VI_VOICE_PROFILES:
    print(f"Warning: Unknown VI_VOICE_PROFILE='{VI_VOICE_PROFILE}'. Defaulting to 'csa_child'.")
    VI_VOICE_PROFILE = "csa_child"


def vi_voice_model_path(profile=VI_VOICE_PROFILE):
    return MODELS_DIR / "piper" / VI_VOICE_PROFILES[profile]["model"]


def vi_voice_speaker_id(profile=VI_VOICE_PROFILE):
    return VI_VOICE_PROFILES[profile].get("speaker_id")


def vi_voice_synthesis_config(profile=VI_VOICE_PROFILE):
    p = VI_VOICE_PROFILES[profile]
    return SynthesisConfig(
        speaker_id=p.get("speaker_id"),
        length_scale=p["length_scale"],
        noise_scale=p["noise_scale"],
        noise_w_scale=p["noise_w_scale"],
    )


def vi_voice_pitch_semitones(profile=VI_VOICE_PROFILE):
    return VI_VOICE_PROFILES[profile]["pitch_semitones"]


def list_vi_voices():
    """Print all available Vietnamese voice profiles with descriptions."""
    print("\n" + "=" * 60)
    print("  Available Vietnamese voice profiles (VI_VOICE_PROFILE):")
    print("=" * 60)
    for key, p in VI_VOICE_PROFILES.items():
        marker = "  ★ DEFAULT" if key == VI_VOICE_PROFILE else ""
        speaker = f" (speaker_id={p['speaker_id']})" if p.get("speaker_id") is not None else " (single speaker)"
        print(f"  - {key:18s}{speaker:24s} {p['desc']}{marker}")
    print("  Select with:  export VI_VOICE_PROFILE=<name>")
    print("=" * 60 + "\n")


# Initial context prompt to prime Whisper tokenizer for Vietnamese diacritics
INITIAL_PROMPTS = {
    "vi": "Xin chào, đây là cuộc trò chuyện bằng tiếng Việt có dấu đầy đủ.",
    "en": "Hello, this is a spoken conversation in English.",
}

# Adaptive beam size:
# - vi: beam_size=5 (essential for tonal accuracy)
# - en: beam_size=2 or 3 (fast and accurate enough on English-dedicated models)
BEAM_SIZES = {
    "vi": 5,
    "en": 2,
}

# ==========================================
# 1. TEXT PREPROCESSING & NORMALIZATION FOR TTS
# ==========================================
def _convert_small_number_to_vi(match):
    val = int(match.group(0))
    if val > 99:
        return str(val)
    units = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    if val < 10:
        return units[val]
    if val == 10:
        return "mười"
    if val < 20:
        remainder = val % 10
        if remainder == 5:
            return "mười lăm"
        return f"mười {units[remainder]}"
    tens = val // 10
    remainder = val % 10
    tens_str = "hai mươi" if tens == 2 else f"{units[tens]} mươi"
    if remainder == 0:
        return tens_str
    elif remainder == 1:
        return f"{tens_str} mốt"
    elif remainder == 4:
        return f"{tens_str} tư"
    elif remainder == 5:
        return f"{tens_str} lăm"
    return f"{tens_str} {units[remainder]}"


def preprocess_text_for_tts(text: str, language: str = ROBOT_LANG) -> str:
    """
    Cleans and normalizes text before feeding it to Piper TTS.
    Especially crucial for Vietnamese:
    - Normalizes Unicode to NFC (prevents decomposed diacritics from breaking eSpeak-ng).
    - Cleans markdown/formatting symbols.
    - Expands small numbers and common units into spoken Vietnamese.
    - Ensures clean sentence-ending punctuation for natural prosodic pauses.
    """
    if not text:
        return ""

    # 1. Unicode normalization to NFC (vital for Vietnamese diacritics and tones)
    text = unicodedata.normalize("NFC", text)

    # 2. Strip leftover markdown artifacts, bullet points, brackets, and quotes
    text = re.sub(r"[\*\#\_`~<>]", " ", text)
    text = re.sub(r"\[.*?\]|\(.*?\)", " ", text)
    text = re.sub(r"^[\s\-\*\•\d+\.]+\s+", "", text)  # leading bullets/numbering

    if language == "vi":
        # Expand percentage and temperature units
        text = re.sub(r"(\d+)\s*%", r"\1 phần trăm", text)
        text = re.sub(r"(\d+)\s*°\s*C\b", r"\1 độ C", text, flags=re.IGNORECASE)
        text = re.sub(r"(\d+)\s*km/h\b", r"\1 ki-lô-mét một giờ", text, flags=re.IGNORECASE)

        # Expand isolated small integers (0-99) into Vietnamese words for natural tonal flow
        text = re.sub(r"\b\d{1,2}\b", _convert_small_number_to_vi, text)

    # 3. Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    # 4. Ensure trailing punctuation so Piper handles sentence-final pitch cadence
    if text and text[-1] not in ".!?":
        text += "."

    return text


# Piper synthesis configurations per language:
# - vi: now profile-driven (see VI_VOICE_PROFILES) for kid-friendly expressiveness
# - en: fixed warm setting
PIPER_SYNTHESIS_CONFIGS = {
    "en": SynthesisConfig(length_scale=0.92, noise_scale=0.667, noise_w_scale=0.8),
}

# Pitch shift semitones per language (+N semitones raises the pitch formant
# into an adorable, youthful companion robot register without chipmunk distortion):
# - vi: now profile-driven (see VI_VOICE_PROFILES / vi_voice_pitch_semitones)
PITCH_SHIFT_SEMITONES = {
    "en": 2.5,
}


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
_piper_voice_profile = None  # cache key: "en" or "vi:<profile>"


def get_piper_voice(language=ROBOT_LANG, profile=None):
    """
    Load a Piper voice.
    - language='en'  -> English Lessac voice (cached by "en")
    - language='vi'  -> Vietnamese voice selected by `profile` (defaults to VI_VOICE_PROFILE),
                        cached by "vi:<profile>"
    """
    global _piper_voice, _piper_voice_profile

    if language == "en":
        voice_path = VOICE_MODELS["en"]
        cache_key = "en"
    elif language == "vi":
        profile = profile or VI_VOICE_PROFILE
        if profile not in VI_VOICE_PROFILES:
            raise ValueError(f"Unsupported Vietnamese voice profile: '{profile}'.")
        voice_path = vi_voice_model_path(profile)
        cache_key = f"vi:{profile}"
    else:
        raise ValueError(f"Unsupported Piper voice language: '{language}'.")

    if not voice_path.is_file():
        raise FileNotFoundError(
            f"Local Piper voice model not found at '{voice_path}'. "
            "Run 'python download_models.py' to download required voices."
        )

    if cache_key != _piper_voice_profile:
        _piper_voice = None
        _piper_voice_profile = None
        gc.collect()
        label = f"vi profile '{profile}'" if language == "vi" else f"'{language}'"
        print(f"Loading local Piper voice for {label}...")
        _piper_voice = PiperVoice.load(str(voice_path))
        _piper_voice_profile = cache_key

    if _piper_voice is None:
        raise RuntimeError("Piper voice failed to load.")
    return _piper_voice


def speak(text, language=ROBOT_LANG, profile=None):
    """
    Speak text with the selected voice profile.
    - language='vi' uses profile (defaults to VI_VOICE_PROFILE, or `profile` if given)
    - language='en' uses the English Lessac voice
    """
    print(f"\nRobot: {text}")
    try:
        clean_text = preprocess_text_for_tts(text, language)
        if not clean_text:
            return

        if language == "vi":
            profile = profile or VI_VOICE_PROFILE
            voice = get_piper_voice("vi", profile=profile)
            syn_config = vi_voice_synthesis_config(profile)
            shift_val = vi_voice_pitch_semitones(profile)
        else:
            voice = get_piper_voice("en")
            syn_config = PIPER_SYNTHESIS_CONFIGS.get("en", None)
            shift_val = PITCH_SHIFT_SEMITONES.get("en", 0.0)

        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            voice.synthesize_wav(clean_text, wav_file, syn_config=syn_config)

        audio_buffer.seek(0)
        with wave.open(audio_buffer, "rb") as wav_file:
            audio = np.frombuffer(
                wav_file.readframes(wav_file.getnframes()), dtype=np.int16
            )
            samplerate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()

            # Apply child-like companion robot pitch-shift
            # (skipped when ROBOT_VOICE_STYLE == "natural")
            shift_val = PITCH_SHIFT_SEMITONES.get(language, 0.0)
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


def resolve_whisper_path(language):
    target_path = WHISPER_MODEL_PATHS.get(language)
    if target_path and target_path.is_dir() and (target_path / "model.bin").is_file():
        return str(target_path)
    raise FileNotFoundError(
        f"Whisper model directory not found for language '{language}'. "
        f"Please run 'python download_models.py --lang {language}' first."
    )


def initialize_whisper(language=ROBOT_LANG):
    model_path = resolve_whisper_path(language)
    print(f"Loading local Whisper model for '{language}' from: {model_path}...")
    return WhisperModel(
        model_path,
        device="cpu",
        compute_type="int8",
    )


def initialize_llm():
    llm_path = MODELS_DIR / "llama-3.2-1b-instruct.Q4_K_M.gguf"
    if not llm_path.is_file():
        raise FileNotFoundError(
            f"LLM model not found at '{llm_path}'. Please run 'python download_models.py'."
        )

    print("Loading local LLM (Llama 3.2 1B)...")
    return Llama(
        model_path=str(llm_path),
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=-1,  # -1 for Mac GPU; set to 0 when deploying to Raspberry Pi 5
        verbose=False,
    )


# ==========================================
# 3. PERSONALITY & PROMPTS
# ==========================================
SYSTEM_PROMPTS = {
    "en": (
        "You are Sparky, a cheerful, cute, and encouraging companion robot friend for kids! "
        "Always reply in English with warmth, enthusiasm, and child-friendly curiosity. "
        "Keep answers short (1-3 sentences maximum). "
        "Never use formatting symbols, markdown, emojis, or bullet points."
    ),
    "vi": (
        "Bạn là Sparky, một bạn robot nhỏ nhắn, siêu đáng yêu, vui tươi và luôn khích lệ các bạn nhỏ! "
        "Xưng hô là 'mình' và gọi trẻ là 'bạn' hoặc 'bạn nhỏ'. Giọng điệu hào hứng, ấm áp, ngộ nghĩnh và gần gũi. "
        "Luôn trả lời bằng tiếng Việt có dấu đầy đủ, chuẩn chính tả. "
        "Trả lời thật ngắn gọn và sinh động (chỉ 1 đến 2 câu). "
        "Tuyệt đối không dùng ký hiệu định dạng, Markdown, biểu tượng cảm xúc hoặc gạch đầu dòng. "
        "Viết các số đếm thành chữ tiếng Việt để giọng đọc truyền cảm nhất."
    ),
}

# ==========================================
# 4. CORE AUDIO & INFERENCE FUNCTIONS
# ==========================================
def record_audio(duration=5, sample_rate=16000):
    print("\n[Listening... Speak now]")
    try:
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return audio_data.flatten()
    except Exception as e:
        print(f"Audio Recording Error: {e}")
        return None


def transcribe_audio(whisper_model, audio_data, language=ROBOT_LANG):
    try:
        beam_size = BEAM_SIZES.get(language, 5)
        initial_prompt = INITIAL_PROMPTS.get(language, None)

        segments, info = whisper_model.transcribe(
            audio_data,
            beam_size=beam_size,
            language=language,
            initial_prompt=initial_prompt,
            vad_filter=True,  # Filter out silence and noise hallucinations
        )
        text = " ".join([segment.text for segment in segments]).strip()
        print(f"You said: {text}")
        return text
    except Exception as e:
        print(f"Transcription Error: {e}")
        return ""


def generate_response(llm, user_text, language=ROBOT_LANG):
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[language]},
                {"role": "user", "content": user_text},
            ],
            temperature=0.7,
            max_tokens=150,
        )
        raw_text = response["choices"][0]["message"]["content"].strip()
        # Clean markdown asterisks or quotes for clean speech
        cleaned_text = re.sub(r"[\*\#\_`]", "", raw_text)
        return cleaned_text
    except Exception as e:
        print(f"LLM Generation Error: {e}")
        if language == "vi":
            return "Ôi, mạch điện của mình bị rối mất rồi. Bạn nói lại nhé?"
        return "Oops! My circuits got tangled for a second. Can you say that again?"


# ==========================================
# 5. MAIN INTERACTIVE LOOP
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print(f" Starting Sparky Robot (Language Mode: {ROBOT_LANG.upper()})")
    print("=" * 50)

    # Optionally list all Vietnamese voices and exit (ROBOT_LIST_VOICES=1)
    if os.getenv("ROBOT_LIST_VOICES", "0") == "1" and ROBOT_LANG == "vi":
        list_vi_voices()
        sys.exit(0)

    whisper_model = initialize_whisper(ROBOT_LANG)
    llm = initialize_llm()

    # Announce active voice profile
    if ROBOT_LANG == "vi":
        p = VI_VOICE_PROFILES[VI_VOICE_PROFILE]
        print(f"  Vietnamese voice profile: {VI_VOICE_PROFILE} — {p['desc']}")

    # Friendly greeting
    if ROBOT_LANG == "vi":
        speak("Xin chào! Mình là Sparky, robot ngoại tuyến của bạn. Cùng trò chuyện nhé!", ROBOT_LANG)
    else:
        speak("Hello! I am Sparky, your offline robot assistant. Let's talk!", ROBOT_LANG)

    exit_phrases = {
        "en": ("bye", "exit", "quit", "goodbye"),
        "vi": ("tạm biệt", "thoát", "kết thúc", "nghỉ thôi"),
    }[ROBOT_LANG]

    while True:
        audio = record_audio(duration=5)
        if audio is not None:
            user_speech = transcribe_audio(whisper_model, audio, language=ROBOT_LANG)

            if user_speech:
                # Check for exit commands
                lower_speech = user_speech.casefold()
                if any(re.search(rf"\b{re.escape(phrase)}\b", lower_speech) for phrase in exit_phrases):
                    if ROBOT_LANG == "vi":
                        speak("Tạm biệt! Chúc bạn một ngày tuyệt vời!", ROBOT_LANG)
                    else:
                        speak("Goodbye! Have a wonderful day!", ROBOT_LANG)
                    break

                # Generate AI reply and speak it
                reply = generate_response(llm, user_speech, language=ROBOT_LANG)
                speak(reply, ROBOT_LANG)
