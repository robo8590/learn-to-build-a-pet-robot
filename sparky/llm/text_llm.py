"""Conversational text responses from the local LLM."""

import re

from sparky.config.settings import LLM_MAX_TOKENS
from sparky.llm.prompts import SYSTEM_PROMPT


def generate_response(llm, user_text):
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            temperature=0.7,
            max_tokens=LLM_MAX_TOKENS,
        )
        raw_text = response["choices"][0]["message"]["content"].strip()
        # Clean markdown asterisks or quotes for clean speech
        cleaned_text = re.sub(r"[\*\#\_`]", "", raw_text)
        return cleaned_text
    except Exception as e:
        print(f"LLM Generation Error: {e}")
        return "Oops! My circuits got tangled for a second. Can you say that again?"
