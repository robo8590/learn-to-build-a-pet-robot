import importlib
import os
import unittest
from unittest import mock

import camera
import offline_robot
from model_config import (
    available_llm_keys,
    get_llm_config,
    is_vision_capable,
    is_vision_ready,
)


class LookCommandTests(unittest.TestCase):
    def test_false_positives(self):
        for text in (
            "looks good",
            "look out!",
            "look for the answer",
            "look up",
            "I will look at this",
            "let me look",
        ):
            with self.subTest(text=text):
                self.assertIsNone(offline_robot.parse_look_command(text))

    def test_trigger_forms(self):
        cases = {
            "look and solve this math": "solve this math",
            "Look and read this.": "read this.",
            "look at this and solve": "solve",
            "take a look and tell me": "tell me",
            "sparky, can you take a look and solve number 3": "solve number 3",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(offline_robot.parse_look_command(text), expected)

    def test_bare_look_and_at_this(self):
        self.assertEqual(offline_robot.parse_look_command("look"), "")
        self.assertEqual(offline_robot.parse_look_command("take a look"), "")
        self.assertEqual(offline_robot.parse_look_command("look at this"), "")


class ExitCommandTests(unittest.TestCase):
    def test_exit_phrases(self):
        for text in ("bye", "Bye!", "exit", "quit", "goodbye", "stop"):
            with self.subTest(text=text):
                self.assertTrue(offline_robot.is_exit_command(text))

    def test_look_sentence_is_not_exit(self):
        self.assertFalse(offline_robot.is_exit_command("look and read this exit sign"))
        self.assertFalse(offline_robot.is_exit_command("exit sign"))

    def test_look_runs_before_exit(self):
        text = "look and read this exit sign"
        self.assertIsNotNone(offline_robot.parse_look_command(text))
        self.assertFalse(offline_robot.is_exit_command(text))


class TtsTests(unittest.TestCase):
    def test_parentheses_preserved(self):
        text = "12 times (10 plus 2) is 96."
        self.assertEqual(offline_robot.preprocess_text_for_tts(text), text)

    def test_leading_math_number_preserved(self):
        self.assertEqual(
            offline_robot.preprocess_text_for_tts("12 x 8 = 96."),
            "12 x 8 = 96.",
        )

    def test_bracket_annotations_removed(self):
        self.assertEqual(
            offline_robot.preprocess_text_for_tts("[laughs] 3 + 4 = 7"),
            "3 + 4 = 7.",
        )

    def test_list_prefix_removed(self):
        self.assertEqual(
            offline_robot.preprocess_text_for_tts("1. First step"),
            "First step.",
        )
        self.assertEqual(
            offline_robot.preprocess_text_for_tts("2) Second step"),
            "Second step.",
        )

    def test_emoji_removed(self):
        self.assertEqual(
            offline_robot.preprocess_text_for_tts("Great job! 🎉"),
            "Great job!",
        )


class VisionRegistryTests(unittest.TestCase):
    def test_capability_flags(self):
        self.assertTrue(is_vision_capable("qwen35-2b"))
        self.assertTrue(is_vision_capable("gemma4-e2b"))
        self.assertTrue(is_vision_capable("qwen35-4b"))
        self.assertFalse(is_vision_capable("qwen25-1.5b"))
        self.assertFalse(is_vision_capable("llama32-1b"))

    def test_vision_models_have_projector_metadata(self):
        for key in available_llm_keys():
            config = get_llm_config(key)
            if config.get("supports_vision"):
                with self.subTest(key=key):
                    self.assertTrue(config.get("mmproj_filename"))
                    self.assertTrue(config.get("mmproj_local_filename"))

    def test_text_only_model_is_not_vision_ready(self):
        self.assertFalse(is_vision_ready("qwen25-1.5b"))


class CameraEnvTests(unittest.TestCase):
    def test_case_insensitive_capture_flag_and_clamps(self):
        with mock.patch.dict(
            os.environ,
            {
                "ROBOT_KEEP_CAPTURES": "True",
                "ROBOT_IMAGE_MAX_DIM": "0",
                "ROBOT_IMAGE_JPEG_QUALITY": "500",
            },
            clear=False,
        ):
            module = importlib.reload(camera)
            self.assertTrue(module.KEEP_CAPTURES)
            self.assertEqual(module.IMAGE_MAX_DIM, 64)
            self.assertEqual(module.IMAGE_JPEG_QUALITY, 100)
        importlib.reload(camera)

    def test_data_url_prefix(self):
        url = offline_robot.image_bytes_to_data_url(b"\xff\xd8\xff")
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))


class StartupOrderTests(unittest.TestCase):
    def test_look_skips_whisper_and_greeting(self):
        with (
            mock.patch.object(
                offline_robot, "initialize_llm", return_value=(object(), True)
            ),
            mock.patch.object(
                offline_robot,
                "initialize_whisper",
                side_effect=AssertionError("Whisper should not load"),
            ) as whisper,
            mock.patch.object(offline_robot, "_run_vision_once") as run,
            mock.patch.object(offline_robot, "speak") as speak,
            mock.patch("builtins.print"),
        ):
            offline_robot.main(["--look", "solve this"])
        whisper.assert_not_called()
        run.assert_called_once()
        speak.assert_not_called()

    def test_check_vision_exits_before_model_or_whisper(self):
        with (
            mock.patch.object(offline_robot, "check_vision", return_value=0) as check,
            mock.patch.object(offline_robot, "initialize_llm") as init,
            mock.patch.object(offline_robot, "initialize_whisper") as whisper,
        ):
            with self.assertRaises(SystemExit) as raised:
                offline_robot.main(["--check-vision"])
        self.assertEqual(raised.exception.code, 0)
        check.assert_called_once_with(None)
        init.assert_not_called()
        whisper.assert_not_called()


class CheckVisionTests(unittest.TestCase):
    def test_missing_projector_fails(self):
        with (
            mock.patch.object(
                offline_robot, "initialize_llm", return_value=(object(), False)
            ),
            mock.patch("builtins.print"),
        ):
            self.assertEqual(offline_robot.check_vision(), 1)

    def test_completion_error_fails(self):
        with (
            mock.patch.object(
                offline_robot, "initialize_llm", return_value=(object(), True)
            ),
            mock.patch.object(
                offline_robot,
                "_vision_completion",
                side_effect=RuntimeError("projector failed"),
            ),
            mock.patch("builtins.print"),
        ):
            self.assertEqual(offline_robot.check_vision(), 1)


class CameraBackendTests(unittest.TestCase):
    def _run(self, backend):
        with (
            mock.patch.object(camera, "CAMERA_BACKEND", backend),
            mock.patch.object(camera, "_open_picamera2", return_value=None) as picam,
            mock.patch.object(
                camera, "_open_opencv_camera", return_value=(None, None)
            ) as opencv,
            mock.patch("builtins.print"),
        ):
            result = camera.capture_jpeg()
        return result, picam.called, opencv.called

    def test_explicit_picamera2_does_not_fall_back(self):
        result, picam, opencv = self._run("picamera2")
        self.assertIsNone(result)
        self.assertTrue(picam)
        self.assertFalse(opencv)

    def test_auto_falls_back_to_opencv(self):
        result, picam, opencv = self._run("auto")
        self.assertIsNone(result)
        self.assertTrue(picam)
        self.assertTrue(opencv)

    def test_unknown_backend_uses_auto(self):
        result, picam, opencv = self._run("bogus")
        self.assertIsNone(result)
        self.assertTrue(picam)
        self.assertTrue(opencv)


if __name__ == "__main__":
    unittest.main()
