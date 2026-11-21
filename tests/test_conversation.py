import unittest
from unittest import mock

import offline_robot
import sparky.vision.turn as vision_turn
from sparky.llm import text_llm
from sparky.llm.conversation import Conversation
from sparky.llm.prompts import SYSTEM_PROMPT, VISION_SYSTEM_PROMPT
from sparky.llm.vision_llm import generate_vision_response


def _fake_llm(reply="Hello there!"):
    llm = mock.Mock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": reply}}]
    }
    return llm


class ConversationTests(unittest.TestCase):
    def test_starts_with_only_system_message(self):
        self.assertEqual(
            Conversation().messages(), [{"role": "system", "content": SYSTEM_PROMPT}]
        )

    def test_exchange_is_recorded(self):
        conversation = Conversation()
        conversation.add_exchange("My name is Sam.", "Nice to meet you, Sam!")
        self.assertEqual(conversation.turn_count, 1)
        self.assertEqual(
            conversation.messages(),
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "My name is Sam."},
                {"role": "assistant", "content": "Nice to meet you, Sam!"},
            ],
        )

    def test_clear_forgets_everything(self):
        conversation = Conversation()
        conversation.add_exchange("hi", "hello")
        conversation.clear()
        self.assertEqual(conversation.turn_count, 0)
        self.assertEqual(conversation.messages()[0]["role"], "system")

    def test_max_turns_evicts_oldest(self):
        conversation = Conversation(max_turns=2)
        for i in range(3):
            conversation.add_exchange(f"question {i}", f"answer {i}")
        contents = [m["content"] for m in conversation.messages()]
        self.assertNotIn("question 0", contents)
        self.assertIn("question 1", contents)
        self.assertIn("answer 2", contents)
        self.assertEqual(conversation.messages()[0]["role"], "system")
        self.assertEqual(conversation.turn_count, 2)

    def test_token_budget_evicts_oldest_but_keeps_latest(self):
        conversation = Conversation(context_tokens=1024, max_turns=20)
        for i in range(10):
            conversation.add_exchange("q" * 200 + str(i), "a" * 200 + str(i))
        messages = conversation.messages()
        contents = [m["content"] for m in messages]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("q" * 200 + "9", contents)  # latest kept
        self.assertNotIn("q" * 200 + "0", contents)  # oldest evicted
        self.assertLessEqual(
            sum(len(content) // 4 for content in contents),
            conversation._token_budget(),
        )

    def test_single_oversized_turn_is_never_dropped(self):
        conversation = Conversation(context_tokens=256, max_turns=5)
        conversation.add_exchange("q" * 4000, "a" * 4000)
        self.assertEqual(len(conversation.messages()), 3)

    def test_history_messages_exclude_system_message(self):
        conversation = Conversation()
        conversation.add_exchange("q", "a")
        self.assertEqual(
            conversation.history_messages(),
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_messages_accepts_custom_system_prompt(self):
        conversation = Conversation()
        self.assertEqual(
            conversation.messages(system_prompt="VISION PROMPT"),
            [{"role": "system", "content": "VISION PROMPT"}],
        )


class GenerateResponseMemoryTests(unittest.TestCase):
    def test_prior_turns_are_sent_to_the_model(self):
        conversation = Conversation()
        conversation.add_exchange("My name is Sam.", "Nice to meet you, Sam!")
        llm = _fake_llm()
        text_llm.generate_response(llm, "What is my name?", conversation)

        messages = llm.create_chat_completion.call_args.kwargs["messages"]
        self.assertEqual(
            [m["content"] for m in messages],
            [
                SYSTEM_PROMPT,
                "My name is Sam.",
                "Nice to meet you, Sam!",
                "What is my name?",
            ],
        )

    def test_successful_reply_is_recorded(self):
        conversation = Conversation()
        reply = text_llm.generate_response(
            _fake_llm("**Great!**"), "Hello!", conversation
        )
        self.assertEqual(reply, "Great!")
        self.assertEqual(conversation.turn_count, 1)
        self.assertEqual(conversation.messages()[-1]["content"], "Great!")

    def test_error_does_not_change_history(self):
        conversation = Conversation()
        conversation.add_exchange("earlier", "remembered")
        llm = mock.Mock()
        llm.create_chat_completion.side_effect = RuntimeError("boom")

        with mock.patch("builtins.print"):
            reply = text_llm.generate_response(llm, "new question", conversation)

        self.assertIn("circuits got tangled", reply)
        self.assertEqual(conversation.turn_count, 1)
        self.assertNotIn("new question", [m["content"] for m in conversation.messages()])

    def test_without_conversation_keeps_old_behavior(self):
        llm = _fake_llm()
        text_llm.generate_response(llm, "hi")
        self.assertEqual(
            llm.create_chat_completion.call_args.kwargs["messages"],
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "hi"},
            ],
        )


class VisionMemoryTests(unittest.TestCase):
    def test_vision_reply_is_remembered_as_text_only(self):
        conversation = Conversation()
        reply = generate_vision_response(
            _fake_llm("It is a cat."), b"jpeg", "what is this?", conversation
        )
        self.assertEqual(reply, "It is a cat.")
        self.assertEqual(conversation.turn_count, 1)
        contents = [m["content"] for m in conversation.messages()]
        self.assertIn("Photo question: what is this?", contents)
        self.assertIn("It is a cat.", contents)
        joined = " ".join(str(content) for content in contents)
        self.assertNotIn("data:image", joined)
        self.assertNotIn("base64", joined)

    def test_vision_prompt_uses_vision_system_prompt_and_prior_text_turns(self):
        conversation = Conversation()
        conversation.add_exchange("My name is Sam.", "Nice to meet you, Sam!")
        llm = _fake_llm("It is a cat.")
        generate_vision_response(llm, b"jpeg", "what is this?", conversation)

        messages = llm.create_chat_completion.call_args.kwargs["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": VISION_SYSTEM_PROMPT})
        self.assertEqual(messages[1], {"role": "user", "content": "My name is Sam."})
        self.assertEqual(
            messages[2], {"role": "assistant", "content": "Nice to meet you, Sam!"}
        )
        self.assertEqual(messages[3]["role"], "user")
        content = messages[3]["content"]
        self.assertTrue(
            content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )
        self.assertEqual(content[1]["text"], "what is this?")

    def test_later_text_reply_sees_earlier_vision_answer(self):
        conversation = Conversation()
        generate_vision_response(
            _fake_llm("It is a cat."), b"jpeg", "what is this?", conversation
        )
        llm = _fake_llm("Yes, it was a cat.")
        text_llm.generate_response(llm, "Was that a cat?", conversation)

        messages = llm.create_chat_completion.call_args.kwargs["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[1]["content"], "Photo question: what is this?")
        self.assertEqual(messages[2]["content"], "It is a cat.")
        self.assertEqual(messages[3]["content"], "Was that a cat?")

    def test_vision_error_does_not_change_history(self):
        conversation = Conversation()
        conversation.add_exchange("earlier", "remembered")
        llm = mock.Mock()
        llm.create_chat_completion.side_effect = RuntimeError("boom")
        with mock.patch("builtins.print"):
            reply = generate_vision_response(llm, b"jpeg", "what is this?", conversation)
        self.assertIn("trouble looking", reply)
        self.assertEqual(conversation.turn_count, 1)

    def test_vision_without_conversation_keeps_old_behavior(self):
        llm = _fake_llm("It is a cat.")
        generate_vision_response(llm, b"jpeg", "what is this?")
        messages = llm.create_chat_completion.call_args.kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

    def test_run_vision_once_forwards_conversation(self):
        conversation = Conversation()
        llm = object()
        with (
            mock.patch.object(vision_turn, "speak"),
            mock.patch.object(vision_turn, "capture_jpeg", return_value=b"jpeg"),
            mock.patch.object(
                vision_turn, "generate_vision_response", return_value="It is a cat."
            ) as generate,
        ):
            vision_turn.run_vision_once(llm, True, "what is this?", conversation)
        generate.assert_called_once_with(llm, b"jpeg", "what is this?", conversation)


class MainLoopMemoryTests(unittest.TestCase):
    def test_same_conversation_is_reused_across_turns(self):
        seen = []

        def fake_generate(llm, text, conversation=None):
            seen.append((text, conversation))
            return "reply"

        with (
            mock.patch.object(
                offline_robot, "initialize_llm", return_value=(object(), False)
            ),
            mock.patch.object(
                offline_robot, "initialize_whisper", return_value=object()
            ),
            mock.patch.object(offline_robot, "record_audio", return_value=b"audio"),
            mock.patch.object(
                offline_robot,
                "transcribe_audio",
                side_effect=["my name is Sam", "what is my name?", "bye"],
            ),
            mock.patch.object(
                offline_robot, "generate_response", side_effect=fake_generate
            ),
            mock.patch.object(offline_robot, "speak"),
            mock.patch("builtins.print"),
        ):
            offline_robot.main([])

        self.assertEqual([text for text, _ in seen], ["my name is Sam", "what is my name?"])
        self.assertIs(seen[0][1], seen[1][1])
        self.assertIsInstance(seen[0][1], Conversation)

    def test_vision_and_text_turns_share_one_conversation(self):
        text_conversations = []
        vision_calls = []

        def fake_generate(llm, text, conversation=None):
            text_conversations.append(conversation)
            return "text reply"

        def fake_vision(llm, vision_enabled, task, conversation=None):
            vision_calls.append((task, conversation))

        with (
            mock.patch.object(
                offline_robot, "initialize_llm", return_value=(object(), True)
            ),
            mock.patch.object(
                offline_robot, "initialize_whisper", return_value=object()
            ),
            mock.patch.object(offline_robot, "record_audio", return_value=b"audio"),
            mock.patch.object(
                offline_robot,
                "transcribe_audio",
                side_effect=["look and read this", "what did it say?", "bye"],
            ),
            mock.patch.object(
                offline_robot, "generate_response", side_effect=fake_generate
            ),
            mock.patch.object(
                vision_turn, "run_vision_once", side_effect=fake_vision
            ),
            mock.patch.object(offline_robot, "speak"),
            mock.patch("builtins.print"),
        ):
            offline_robot.main([])

        self.assertEqual(len(vision_calls), 1)
        self.assertEqual(vision_calls[0][0], "read this")
        self.assertEqual(len(text_conversations), 1)
        self.assertIs(vision_calls[0][1], text_conversations[0])


if __name__ == "__main__":
    unittest.main()
