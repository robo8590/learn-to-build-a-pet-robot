"""In-memory multi-turn conversation history for Sparky.

This lets the LLM see earlier parts of a session (for example, the user's
name) when answering a new turn. History lives only in RAM for the lifetime of
the process: nothing is written to disk, and photos are never stored here.

The stored history is bounded by a turn count and an estimated token budget so
that growing conversations cannot overflow the model's context window.
"""

from sparky.config.settings import LLM_MEMORY_MAX_TURNS
from sparky.llm.prompts import SYSTEM_PROMPT


class Conversation:
    """A bounded, in-memory record of user/assistant exchanges."""

    def __init__(self, system_prompt=SYSTEM_PROMPT, max_turns=None, context_tokens=2048):
        self.system_prompt = system_prompt
        self.max_turns = max(0, LLM_MEMORY_MAX_TURNS if max_turns is None else max_turns)
        # Use the smallest context window in the registry (2048) as the default
        # so the cap is safe even for the text-only fallback models.
        self.context_tokens = max(256, context_tokens)
        self._turns = []  # list of (user_text, assistant_text) pairs

    @property
    def turn_count(self):
        """Number of remembered exchanges."""
        return len(self._turns)

    def add_exchange(self, user_text, assistant_text):
        """Remember one user/assistant exchange, evicting the oldest if needed."""
        self._turns.append((user_text, assistant_text))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]

    def clear(self):
        """Forget all remembered turns."""
        self._turns.clear()

    def _token_budget(self):
        """Leave roughly half the context for the system prompt, new input, reply."""
        return max(256, self.context_tokens // 2)

    def _estimated_tokens(self, turns):
        total = len(self.system_prompt) // 4
        for user_text, assistant_text in turns:
            total += len(user_text) // 4 + len(assistant_text) // 4
        return total

    def history_messages(self):
        """Return trimmed user/assistant turns, without any system message.

        Callers pair these with their own system prompt (chat vs. vision).
        """
        turns = self._turns
        while len(turns) > 1 and self._estimated_tokens(turns) > self._token_budget():
            turns = turns[1:]

        messages = []
        for user_text, assistant_text in turns:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        return messages

    def messages(self, system_prompt=None):
        """Return chat messages for the LLM, trimmed to fit the token budget.

        The system message is always first, and the most recent exchange is
        always kept even if it alone exceeds the budget. Pass ``system_prompt``
        to reuse this history with a different prompt (e.g. the vision one).
        """
        prompt = self.system_prompt if system_prompt is None else system_prompt
        return [{"role": "system", "content": prompt}] + self.history_messages()
