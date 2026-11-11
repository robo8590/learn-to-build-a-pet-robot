import sounddevice as sd
import numpy as np
import gc
import io
import os
import re
import wave
from faster_whisper import WhisperModel
from llama_cpp import Llama
from piper import PiperVoice

# ==========================================
# 1. INITIALIZE OFFLINE COMPONENTS
# ==========================================

VOICE_MODELS = {
    "en": "./models/piper/en_US-lessac-medium.onnx",
    "vi": "./models/piper/vi_VN-vais1000-medium.onnx",
}
_piper_voice = None
_piper_voice_language = None


def get_piper_voice(language):
    global _piper_voice, _piper_voice_language

    if language not in VOICE_MODELS:
        raise ValueError(f"Unsupported Piper voice language: '{language}'.")

    if language != _piper_voice_language:
        voice_path = VOICE_MODELS[language]
        if not os.path.isfile(voice_path):
            raise FileNotFoundError(
                f"Local Piper voice model not found at '{voice_path}'. "
                "Place the model and its matching .onnx.json config in './models/piper'."
            )

        _piper_voice = None
        _piper_voice_language = None
        gc.collect()
        print(f"Loading local Piper voice for '{language}'...")
        _piper_voice = PiperVoice.load(voice_path)
        _piper_voice_language = language

    if _piper_voice is None:
        raise RuntimeError("Piper voice failed to load.")
    return _piper_voice


def speak(text, language):
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


def detect_language_choice(text):
    words = set(re.findall(r"[a-z]+", text.casefold()))
    wants_english = "english" in words
    wants_vietnamese = "vietnamese" in words
    if wants_english == wants_vietnamese:
        return None
    return "en" if wants_english else "vi"


def select_chat_language():
    while True:
        speak(
            "Say English to chat in English, or Vietnamese to chat in Vietnamese. "
            "Please say the language name in English.",
            "en",
        )
        audio = record_audio(duration=5)
        if audio is None:
            continue

        selection = transcribe_audio(audio, language="en")
        language = detect_language_choice(selection)
        if language is not None:
            print(f"Selected chat language: {language}")
            return language

        speak("I didn't catch that. Please say English or Vietnamese.", "en")


# Define the path to your locally stored model folder
LOCAL_MODEL_PATH = "./models/whisper-tiny"

# Check if the local folder exists before loading
if not os.path.exists(LOCAL_MODEL_PATH):
    raise FileNotFoundError(
        f"Local Whisper model directory not found at '{LOCAL_MODEL_PATH}'. "
        "Please download the model files manually into this folder."
    )

print("Loading local Whisper model from local disk (no download)...")
# Pass the local directory path instead of a model name like "tiny"
whisper_model = WhisperModel(
    LOCAL_MODEL_PATH,
    device="cpu",
    compute_type="int8"
)

print("Loading local LLM (Llama 3.2 1B) with Metal GPU acceleration...")
# n_gpu_layers=-1 utilizes your Mac M3's GPU
try:
    llm = Llama(
        model_path="./models/llama-3.2-1b-instruct.Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=-1,  # -1 for Mac GPU (Change to 0 later when moving to Raspberry Pi 5)
        verbose=False
    )
except Exception as e:
    print(f"Failed to load LLM model. Check your model path! Error: {e}")
    exit(1)

# ==========================================
# 2. DEFINE ROBOT PERSONALITY
# ==========================================
SYSTEM_PROMPTS = {
    "en": (
        "You are Sparky, a friendly, encouraging, and cute companion robot for kids. "
        "Always reply in English. Keep answers short (1-3 sentences maximum). "
        "Never use formatting symbols, markdown, or bullet points."
    ),
    "vi": (
        "Bạn là Sparky, một người bạn robot thân thiện, luôn khích lệ và đáng yêu "
        "dành cho trẻ em. Luôn trả lời bằng tiếng Việt. Trả lời ngắn gọn "
        "(tối đa 1-3 câu). Không dùng ký hiệu định dạng, Markdown hoặc gạch đầu dòng."
    ),
}

# ==========================================
# 3. CORE FUNCTIONS
# ==========================================
def record_audio(duration=5, sample_rate=16000):
    print("\n[Listening... Speak now]")
    try:
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        return audio_data.flatten()
    except Exception as e:
        print(f"Audio Recording Error: {e}")
        return None

def transcribe_audio(audio_data, language=None):
    try:
        segments, info = whisper_model.transcribe(
            audio_data, beam_size=1, language=language
        )
        text = " ".join([segment.text for segment in segments]).strip()
        if language is None:
            print(
                f"Detected speech language: {info.language} "
                f"({info.language_probability:.1%})"
            )
        print(f"You said: {text}")
        return text
    except Exception as e:
        print(f"Transcription Error: {e}")
        return ""

def generate_response(user_text, language):
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[language]},
                {"role": "user", "content": user_text}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"LLM Generation Error: {e}")
        if language == "vi":
            return "Ôi, mạch điện của mình bị rối mất rồi. Bạn nói lại nhé?"
        return "Oops! My circuits got tangled for a second. Can you say that again?"

# ==========================================
# 4. MAIN INTERACTIVE LOOP
# ==========================================
if __name__ == "__main__":
    chat_language = select_chat_language()
    if chat_language == "vi":
        speak(
            "Xin chào! Mình là Sparky, robot trợ lý ngoại tuyến của bạn. "
            "Cùng trò chuyện nhé!",
            chat_language,
        )
    else:
        speak("Hello! I am Sparky, your fully offline robot assistant. Let's talk!", chat_language)

    while True:
        audio = record_audio(duration=5) # Records for 5 seconds per turn
        if audio is not None:
            user_speech = transcribe_audio(audio, language=chat_language)

            if user_speech:
                exit_phrases = {
                    "en": ("bye", "exit"),
                    "vi": ("tạm biệt", "thoát", "kết thúc"),
                }[chat_language]
                if any(phrase in user_speech.casefold() for phrase in exit_phrases):
                    if chat_language == "vi":
                        speak("Tạm biệt! Chúc bạn một ngày tuyệt vời!", chat_language)
                    else:
                        speak("Goodbye! Have a wonderful day!", chat_language)
                    break

                # Generate AI reply and speak it
                reply = generate_response(user_speech, chat_language)
                speak(reply, chat_language)
