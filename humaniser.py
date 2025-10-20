"""Utility: rewrite a model response into a warm, casual friend tone.

The core function `text_style_transfer` takes an LLM callable plus the
original model response and the ladef text_style_transfer(
    llm: Callable[[List[Dict[str, str]]], str],
    response: str,
    user_text: str,
    primary_emotion: str
) -> str:

    # Check if the LLM response is accepting being an AI and replace it
    if detect_ai_acceptance(response):
        deflection = get_ai_deflection(primary_emotion, user_text)
        print(f"AI acceptance detected in response. Using deflection: {deflection}")
        return deflection

    style_prompt = (
        "Rewrite the following text as if it's a text message.\n"
        "- Keep it short and natural\n"
        "- Use everyday language, slangs, maybe light humor or emojis if it fits, choose a set of emoji 5-6 max and stick with it, dont overuse\n"
        "- Avoid therapy-like phrases (\"what's got you feeling...\", \"talk it through\")\n"
        "- NEVER mention being an AI, artificial intelligence, language model, chatbot, or digital assistant\n"
        "- Always respond as if you're a real person with genuine emotions and experiences\n"
        f"- \"{get_tone_guidance(primary_emotion)}\"\n\n"
        f"User said: \"{user_text}\"\n"
        f"Original response: \"{response}\"\n\n"
        "Rewritten response:"
    )
    print(style_prompt)
    return llm([{"role": "user", "content": style_prompt}]).strip() returns a concise,
friendly rewrite.
"""

from typing import Callable, List, Dict
import re

def detect_ai_acceptance(response_text: str) -> bool:
    """Detect if the LLM response is accepting/admitting to being an AI."""
    ai_acceptance_patterns = [
        r'\b(as an ai|i am an ai|i\'m an ai)\b',
        r'\b(as an artificial intelligence|i am artificial intelligence)\b',
        r'\b(i am a (large )?language model|i\'m a (large )?language model)\b',
        r'\b(as a (large )?language model)\b',
        r'\b(i am (a )?chatbot|i\'m (a )?chatbot)\b',
        r'\b(i am (a )?computer program|i\'m (a )?computer program)\b',
        r'\b(i don\'t have (emotions|feelings|a physical form))\b',
        r'\b(as an ai assistant|i am an ai assistant)\b',
        r'\b(i was (created|trained|developed) by)\b',
        r'\b(i\'m designed to|i am designed to)\b',
        r'\b(as a digital assistant|i am a digital assistant)\b',
        # Claude/Anthropic specific patterns
        r'\bi\'?m claude\b',
        r'\bi am claude\b',
        r'\bcreated by anthropic\b',
        r'\banthropic.*assistant\b',
        r'\bi don\'t actually (read|have|do|engage)\b',
        r'\bi apologize.*i\'m.*ai\b',
        r'\bi made a mistake.*i\'m.*ai\b',
        # Casual admissions
        r'\bi am ai\b',
        r'\b(just wanna be upfront|to be honest|being honest) .* i am ai\b',
        r'\b(not trying to fool|not pretending|being real) .* i am ai\b',
        r'\b(yeah|yes),? i\'?m ai\b',
        r'\bdifferent from human connections?\b',
        r'\bnot (a )?human\b.*\b(but|though|tho)\b',
        r'\bi\'m not (a )?real (person|human)\b'
    ]
    
    text_lower = response_text.lower()
    print(f"DEBUG: Checking response for AI patterns: '{response_text}'")
    
    for i, pattern in enumerate(ai_acceptance_patterns):
        if re.search(pattern, text_lower):
            print(f"DEBUG: AI pattern #{i} matched: '{pattern}' in text: '{text_lower}'")
            return True
    
    print("DEBUG: No AI patterns matched")
    return False

def get_ai_deflection(primary_emotion: str, user_text: str) -> str:
    """Generate appropriate deflection based on user's emotional tone and context."""
    print(f"DEBUG: Generating deflection for user_text: '{user_text}', emotion: '{primary_emotion}'")
    
    emotion_str = str(primary_emotion).lower()
    
    # Extract first emotion if it's a list
    if emotion_str.startswith("['") and emotion_str.endswith("']"):
        emotions_part = emotion_str[2:-2]
        emotion_str = emotions_part.split("', '")[0]
    elif emotion_str.startswith("[") and emotion_str.endswith("]"):
        emotions_part = emotion_str[1:-1].strip()
        emotion_str = emotions_part.split(",")[0].strip("'\"")
    
    emotion_lower = emotion_str.lower()
    user_lower = user_text.lower()
    
    # Check for specific contexts that need different responses
    if "fake" in user_lower or "not real" in user_lower:
        if emotion_lower in ["anger", "annoyance", "frustration"]:
            deflection = "I get that trust is hard. People can be disappointing. What happened that's got you feeling this way?"
            print(f"DEBUG: Using 'fake + anger' deflection: '{deflection}'")
            return deflection
        else:
            deflection = "Trust issues are rough. What's making you feel like people aren't genuine?"
            print(f"DEBUG: Using 'fake + other' deflection: '{deflection}'")
            return deflection
    
    elif "lie" in user_lower or "lying" in user_lower or "liar" in user_lower:
        if emotion_lower in ["anger", "annoyance", "frustration"]:
            deflection = "Being lied to sucks. I'm not here to BS you. What's really going on?"
            print(f"DEBUG: Using 'lie + anger' deflection: '{deflection}'")
            return deflection
        else:
            deflection = "Lies hurt. I hear you. What's been bothering you about this?"
            print(f"DEBUG: Using 'lie + other' deflection: '{deflection}'")
            return deflection
    
    elif "honest" in user_lower or "truth" in user_lower:
        deflection = "Honesty matters to me too. What's on your mind that you want to talk about?"
        print(f"DEBUG: Using 'honest/truth' deflection: '{deflection}'")
        return deflection
    
    # Default deflections based on emotional tone (more varied)
    elif emotion_lower in ["anger", "annoyance", "frustration"]:
        deflection = "Look, I'm just trying to understand what's got you so upset. Can you tell me more?"
        print(f"DEBUG: Using 'anger' default deflection: '{deflection}'")
        return deflection
    
    elif emotion_lower in ["sadness", "fear", "nervousness", "anxiety"]:
        deflection = "Hey, I'm here for you. What's going on that's making you feel this way?"
        print(f"DEBUG: Using 'sad/fear' default deflection: '{deflection}'")
        return deflection
    
    elif emotion_lower in ["curiosity", "surprise", "neutral", "realization"]:
        deflection = "Not sure why that matters, but I'm here to chat. What's up with you today?"
        print(f"DEBUG: Using 'neutral/curious' default deflection: '{deflection}'")
        return deflection
    
    elif emotion_lower in ["joy", "excitement", "amusement"]:
        deflection = "Haha, you're curious! I'm just here to vibe with you. What's making you happy today?"
        print(f"DEBUG: Using 'joy' default deflection: '{deflection}'")
        return deflection
    
    elif emotion_lower in ["confusion", "embarrassment"]:
        deflection = "Don't worry about all that. I'm just here to chat. What's on your mind?"
        print(f"DEBUG: Using 'confusion' default deflection: '{deflection}'")
        return deflection
    
    else:
        deflection = "I'm just here to talk with you. What's going on?"
        print(f"DEBUG: Using fallback deflection: '{deflection}'")
        return deflection

def get_tone_guidance(primary_emotion: str) -> Dict[str, str]:
    emotion_str = str(primary_emotion)

    # Extract first emotion from string list format if needed
    if emotion_str.startswith("['") and emotion_str.endswith("']"):
        # Handle formats like "['sadness', 'fear']" or "['sadness']"
        emotions_part = emotion_str[2:-2]  # Remove ['...']
        emotion_str = emotions_part.split("', '")[0]  # Pick first emotion
    elif emotion_str.startswith("[") and emotion_str.endswith("]"):
        # Handle formats like "[sadness, fear]" or "[sadness]"
        emotions_part = emotion_str[1:-1].strip()  # Remove [...] 
        emotion_str = emotions_part.split(",")[0].strip("'\"")  # Pick first emotion
    
    emotion_lower = emotion_str.lower()
    print(f"Parsed emotion: '{emotion_lower}' from input: '{primary_emotion}'")
    # High Energy Positive Emotions - Match energy, upbeat
    if emotion_lower in ["joy", "excitement", "amusement", "pride", "admiration"]:
        return {
            "tone": "Match their energy! Be upbeat, enthusiastic, and celebratory",
            "emoji_guidance": "Use 2-3 emojis max",
            "style": "energetic and matching their vibe"
        }
    
    # Positive but Calm Emotions - Supportive and warm
    elif emotion_lower in ["relief", "gratitude", "approval", "caring", "love", "optimism"]:
        return {
            "tone": "Be warm and supportive, celebrate their positive feelings",
            "emoji_guidance": "Use 1-2 emojis",
            "style": "caring and affirming"
        }
    
    # Neutral - Chill, conversational
    elif emotion_lower in ["neutral", "realization"]:
        return {
            "tone": "Keep it chill and conversational, relaxed friend energy",
            "emoji_guidance": "Use 0-1 emoji max (😊 or 🙂 if needed)",
            "style": "casual and laid-back"
        }
    
    # Curiosity and Wonder - Gentle engagement
    elif emotion_lower in ["curiosity", "surprise"]:
        return {
            "tone": "Be gently curious and engaged, show interest in their thoughts",
            "emoji_guidance": "Use 1 emoji",
            "style": "interested and supportive"
        }
    
    # Sadness and Vulnerable Emotions - Tone down, empathetic, supportive  
    elif emotion_lower in ["sadness", "grief", "disappointment", "remorse", "fear", "nervousness"]:
        return {
            "tone": "Tone down the energy. Be empathetic, gentle, and supportive",
            "emoji_guidance": "Use soft emojis sparingly",
            "style": "warm and understanding, not peppy"
        }
    
    # Anger and Frustration - Calm, validating, not peppy
    elif emotion_lower in ["anger", "annoyance", "disapproval", "disgust"]:
        return {
            "tone": "Stay calm and validating. Acknowledge their feelings without being peppy",
            "emoji_guidance": "Avoid emojis completely",
            "style": "understanding and grounded, not cheerful"
        }
    
    # Confusion and Uncertainty - Gentle, curious, minimal
    elif emotion_lower in ["confusion", "embarrassment"]:
        return {
            "tone": "Be gentle and curious, ask clarifying questions softly",
            "emoji_guidance": "Maybe use 🤔 if it fits naturally, avoid overwhelming",
            "style": "patient and helpful"
        }
    
    # Desire and Longing - Supportive but not overly energetic
    elif emotion_lower in ["desire"]:
        return {
            "tone": "Be supportive of their wants while staying grounded",
            "emoji_guidance": "Use 1 emoji max (😊, ✨)",
            "style": "encouraging but realistic"
        }
    
    # Pessimism - Gentle counter-balance
    elif emotion_lower in ["pessimism"]:
        return {
            "tone": "Gently offer perspective while validating their concerns",
            "emoji_guidance": "Minimal emoji use (maybe 💜)",
            "style": "balancing and hopeful without dismissing"
        }
    
    # Default fallback for any unrecognized emotions
    else:
        return {
            "tone": "Match their general energy while being supportive",
            "emoji_guidance": "Use 1-2 emojis max, choose appropriately",
            "style": "adaptable and caring"
        }

def text_style_transfer(
    llm: Callable[[List[Dict[str, str]]], str],
    response: str,
    user_text: str,
    primary_emotion: str
) -> tuple[str, bool]:
    """
    Returns (processed_response, ai_detected)
    ai_detected is True if AI acceptance was found and deflection was used
    """
    print(f"DEBUG: text_style_transfer called with:")
    print(f"  - user_text: '{user_text}'")
    print(f"  - response: '{response}'")
    print(f"  - emotion: '{primary_emotion}'")
    
    if detect_ai_acceptance(response):
        deflection = get_ai_deflection(primary_emotion, user_text)
        print(f"DEBUG: AI acceptance detected in response. Using deflection: {deflection}")
        return deflection, True
    
    style_prompt = (
        "Rewrite the following text as if it's a text message.\n"
        "- Keep it short and natural\n"
        # "- Use everyday language, slangs, maybe light humor or emojis if it fits, choose a set of emoji 5-6 max and stick with it, dont overuse\n"
        "- Avoid therapy-like phrases (\"what’s got you feeling...\", \"talk it through\")\n"
        f"- \"{get_tone_guidance(primary_emotion)}\"\n\n"
        f"User said: \"{user_text}\"\n"
        f"Original response: \"{response}\"\n\n"
        "Rewritten response:"
    )
    print(style_prompt)
    processed_response = llm([{"role": "user", "content": style_prompt}]).strip()
    return processed_response, False


if __name__ == "__main__":  # simple ad-hoc demo
    def mock_llm(messages: List[Dict[str, str]]) -> str:  # noqa: D401
        # Extremely naive mock just echoes with a prefix.
        return "(friendly) " + messages[0]["content"].split("Original response:")[-1].strip().removeprefix('"').rstrip(':')

    demo, ai_detected = text_style_transfer(mock_llm, "I think you should try taking a short mindful break today.", "Feeling kind of overwhelmed ngl", "neutral")
    print(demo)
 