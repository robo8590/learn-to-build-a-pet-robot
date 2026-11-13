import sounddevice as sd
import numpy as np
import gc
import io
import os
import re
import wave
from pathlib import Path
from faster_whisper import WhisperModel
from llama_cpp import Llama
from piper import PiperVoice

# ==========================================
# 0. CONFIGURATION & PRE-SET LANGUAGE
# ==========================================
# Preset language for the robot: "vi" or "en"
# Can be overridden via environment variable: export ROBOT_LANG=vi
ROBOT_LANG = os.getenv("ROBOT_LANG", "vi").strip().lower()
if ROBOT_LANG not in ("vi", "en"):
    print(f"Warning: Unsupported ROBOT_LANG='{ROBOT_LANG}'. Defaulting to 'vi'.")
    ROBOT_LANG = "vi"

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Language-specific Whisper paths (PhoWhisper for 'vi', tiny.en/base.en for 'en')
WHISPER_MODEL_PATHS = {
    "vi": MODELS_DIR / "whisper" / "vi",
    "en": MODELS_DIR / "whisper" / "en",
}

VOICE_MODELS = {
    "en": MODELS_DIR / "piper" / "en_US-lessac-medium.onnx",
    "vi": MODELS_DIR / "piper" / "vi_VN-vais1000-medium.onnx",
}

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
# 1. INITIALIZE OFFLINE COMPONENTS
# ==========================================
_piper_voice = None
_piper_voice_language = None


def get_piper_voice(language=ROBOT_LANG):
    global _piper_voice, _piper_voice_language

    if language not in VOICE_MODELS:
        raise ValueError(f"Unsupported Piper voice language: '{language}'.")

    if language != _piper_voice_language:
        voice_path = VOICE_MODELS[language]
        if not voice_path.is_file():
            raise FileNotFoundError(
                f"Local Piper voice model not found at '{voice_path}'. "
                "Run 'python download_models.py' to download required voices."
            )

        _piper_voice = None
        _piper_voice_language = None
        gc.collect()
        print(f"Loading local Piper voice for '{language}'...")
        _piper_voice = PiperVoice.load(str(voice_path))
        _piper_voice_language = language

    if _piper_voice is None:
        raise RuntimeError("Piper voice failed to load.")
    return _piper_voice


def speak(text, language=ROBOT_LANG):
    print(f"\nRobot: {text}")
    try:
        voice = get_piper_voice(language)
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        audio_buffer.seek(0)
        with wave.open(audio_buffer, "rb") as wav_file:
            audio = np.frombuffer(
                wav_file.readframes(wav_file.getnframes()), dtype=np.int16
            )
            if wav_file.getnchannels() > 1:
                audio = audio.reshape(-1, wav_file.getnchannels())
            sd.play(audio, samplerate=wav_file.getframerate())
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
# 2. PERSONALITY & PROMPTS
# ==========================================
SYSTEM_PROMPTS = {
    "en": (
        "You are Sparky, a friendly, encouraging, and cute companion robot for kids. "
        "Always reply in English. Keep answers short (1-3 sentences maximum). "
        "Never use formatting symbols, markdown, emojis, or bullet points."
    ),
    "vi": (
        "Bạn là Sparky, một người bạn robot thân thiện, luôn khích lệ và đáng yêu "
        "dành cho trẻ em. Luôn trả lời bằng tiếng Việt. Trả lời ngắn gọn "
        "(tối đa 1-3 câu). Không dùng ký hiệu định dạng, Markdown, biểu tượng cảm xúc hoặc gạch đầu dòng."
    ),
}

# ==========================================
# 3. CORE AUDIO & INFERENCE FUNCTIONS
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
# 4. MAIN INTERACTIVE LOOP
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print(f" Starting Sparky Robot (Language Mode: {ROBOT_LANG.upper()})")
    print("=" * 50)

    whisper_model = initialize_whisper(ROBOT_LANG)
    llm = initialize_llm()

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
