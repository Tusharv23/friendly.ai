#Text Style Transfer - Will try my best to make the output sound like a friendly, casual friend
def text_style_transfer(response: str, user_text: str) -> str:
    style_prompt = f"""
    Rewrite the following text as if it's coming from a warm, casual friend.
    - Keep it short and natural
    - Use everyday language, maybe light humor or emojis if it fits
    - Avoid therapy-like phrases ("what’s got you feeling...", "talk it through")
    - Mirror the tone of the user's text

    User said: "{user_text}"
    Original response: "{response}"

    Rewritten response:
    """
    return llm([{"role": "user", "content": style_prompt}]).strip()
raw_out = self.llm(msgs).strip()
final_out = friend_tone_rewriter(raw_out, user_text)
self.memory.add("assistant", final_out)
return final_out
 