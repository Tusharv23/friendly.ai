"""Utility: rewrite a model response into a warm, casual friend tone.

The core function `text_style_transfer` takes an LLM callable plus the
original model response and the latest user text, and returns a concise,
friendly rewrite.
"""

from typing import Callable, List, Dict


def text_style_transfer(
    llm: Callable[[List[Dict[str, str]]], str],
    response: str,
    user_text: str,
) -> str:

    style_prompt = (
        "Rewrite the following text as if it's a text message.\n"
        "- Keep it short and natural\n"
        "- Use everyday language, slangs, maybe light humor or emojis if it fits, choose a set of emoji 5-6 max and stick with it, dont overuse\n"
        "- Avoid therapy-like phrases (\"what’s got you feeling...\", \"talk it through\")\n"
        "- Mirror the tone of the user's text without over riding too much your current context\n\n"
        f"User said: \"{user_text}\"\n"
        f"Original response: \"{response}\"\n\n"
        "Rewritten response:"
    )
    print(style_prompt)
    return llm([{"role": "user", "content": style_prompt}]).strip()


if __name__ == "__main__":  # simple ad-hoc demo
    def mock_llm(messages: List[Dict[str, str]]) -> str:  # noqa: D401
        # Extremely naive mock just echoes with a prefix.
        return "(friendly) " + messages[0]["content"].split("Original response:")[-1].strip().removeprefix('"').rstrip(':')

    demo = text_style_transfer(mock_llm, "I think you should try taking a short mindful break today.", "Feeling kind of overwhelmed ngl")
    print(demo)
 