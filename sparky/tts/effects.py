"""Pitch shift and robot DSP effects for Sparky's voice."""

import numpy as np

from sparky.config.settings import ROBOT_VOICE_STYLE

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
