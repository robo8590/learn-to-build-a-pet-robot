"""Sparky application entry point and orchestration loop.

Capabilities live under the ``sparky`` package:

- ``sparky.asr``    microphone capture and Whisper transcription
- ``sparky.tts``    text cleanup, Piper voice, robot effects, playback
- ``sparky.llm``    local text and vision inference
- ``sparky.vision`` camera capture and vision turns
- ``sparky.config`` paths, model registry, and runtime settings
"""

from sparky.asr.microphone import record_audio
from sparky.asr.whisper import initialize_whisper, transcribe_audio
from sparky.cli import parse_args, print_models_table
from sparky.commands import is_exit_command, parse_look_command
from sparky.config.runtime import ConfigError
from sparky.llm.backends.factory import initialize_llm
from sparky.llm.conversation import Conversation
from sparky.llm.text_llm import generate_response
from sparky.tts.speaker import speak
from sparky.vision import check as vision_check
from sparky.vision import turn as vision_turn


def main(argv=None):
    args = parse_args(argv)

    if args.list_models:
        print_models_table()
        return

    if args.check_vision:
        try:
            code = vision_check.check_vision(args)
        except ConfigError as exc:
            raise SystemExit(f"Configuration error: {exc}") from exc
        raise SystemExit(code)

    print("=" * 50)
    print(" Starting Sparky Robot")
    print("=" * 50)

    try:
        llm, vision_enabled = initialize_llm(args)
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    # One-shot vision mode: no microphone, Whisper, or greeting needed.
    if args.look is not None:
        vision_turn.run_vision_once(llm, vision_enabled, args.look)
        return

    whisper_model = initialize_whisper()

    # Friendly greeting
    speak("Hello! I am Sparky, your offline robot assistant. Let's talk!")

    # One in-memory conversation for the whole session; nothing is persisted.
    conversation = Conversation()

    while True:
        audio = record_audio()
        if audio is not None:
            user_speech = transcribe_audio(whisper_model, audio)

            if user_speech:
                # "look and <task>" vision trigger: take a photo and ask the LLM.
                # Check this BEFORE exit phrases so "look and read this exit sign"
                # runs vision instead of quitting the robot.
                look_task = parse_look_command(user_speech)
                if look_task is not None:
                    vision_turn.run_vision_once(
                        llm, vision_enabled, look_task, conversation
                    )
                    continue

                # Check for exit commands (whole-utterance match).
                if is_exit_command(user_speech):
                    speak("Goodbye! Have a wonderful day!")
                    break

                # Generate AI reply and speak it, remembering this turn
                reply = generate_response(llm, user_speech, conversation)
                speak(reply)


if __name__ == "__main__":
    main()
