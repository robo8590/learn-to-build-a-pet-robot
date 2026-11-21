"""Speak text through Piper with Sparky's robot voice effect."""

import io
import wave

import numpy as np
import sounddevice as sd

from sparky.config.settings import ROBOT_VOICE_STYLE
from sparky.tts.effects import PITCH_SHIFT_SEMITONES, apply_robot_dsp, pitch_shift
from sparky.tts.text import preprocess_text_for_tts
from sparky.tts.voice import PIPER_SYNTHESIS_CONFIG, get_piper_voice


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
