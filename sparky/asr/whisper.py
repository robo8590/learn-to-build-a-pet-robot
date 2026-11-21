"""faster-whisper speech-to-text loading and transcription."""

from faster_whisper import WhisperModel

from sparky.config.paths import WHISPER_MODEL_PATH

# Initial context prompt to prime Whisper's tokenizer for natural English speech
WHISPER_INITIAL_PROMPT = "Hello, this is a spoken conversation in English."

# English-dedicated models are fast and accurate enough with a small beam size
WHISPER_BEAM_SIZE = 2


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
