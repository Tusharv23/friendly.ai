# agent.py
from dataclasses import dataclass, field
from typing import List, Dict, Callable
from emotion_classifier import EmotionClassifier 
from personalities import PERSONALITIES, DEFAULT_PERSONALITY_KEY, list_personalities, get_persona_text
from llm.bedrock import bedrock_claude_chat
import time
from persona_state import infer_persona_activity
from dataclasses import dataclass, field
from typing import List, Dict, Callable
from emotion_classifier import EmotionClassifier 
from personalities import PERSONALITIES, DEFAULT_PERSONALITY_KEY, list_personalities, get_persona_text
from llm.bedrock import bedrock_claude_chat
import time
from persona_state import infer_persona_activity
from humaniser import text_style_transfer
# ---------- Conversation data structures ----------
@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str

@dataclass
class EmotionHistory:
    emotions: List[str] = field(default_factory=list)
    intensities: List[float] = field(default_factory=list)
    
    def add_emotion(self, emotion: str, intensity: float = 0.5):
        self.emotions.append(emotion)
        self.intensities.append(intensity)
        # Keep only last 10 emotions for trend analysis
        if len(self.emotions) > 10:
            self.emotions.pop(0)
            self.intensities.pop(0)
    
    def get_trend(self) -> str:
        if len(self.emotions) < 2:
            return "establishing baseline"
        
        recent_emotions = self.emotions[-3:]  # Last 3 emotions
        
        # Check for improvement patterns
        positive_emotions = {"joy", "happiness", "relief", "calm", "hopeful", "confident"}
        negative_emotions = {"sadness", "anger", "fear", "anxiety", "frustration", "despair"}
        
        recent_positive = sum(1 for e in recent_emotions if e.lower() in positive_emotions)
        recent_negative = sum(1 for e in recent_emotions if e.lower() in negative_emotions)
        
        if recent_positive > recent_negative:
            return "improving mood"
        elif recent_negative > recent_positive:
            return "struggling emotionally"
        else:
            return "mixed emotions"
    
    def needs_extra_support(self) -> bool:
        # Check if last 2-3 emotions indicate distress
        if len(self.emotions) < 2:
            return False
        crisis_emotions = {"despair", "hopeless", "suicidal", "panic", "rage"}
        recent = self.emotions[-2:]
        return any(e.lower() in crisis_emotions for e in recent)

@dataclass
class Memory:
    turns: List[Turn] = field(default_factory=list)
    emotion_history: EmotionHistory = field(default_factory=EmotionHistory)
    max_chars: int = 8000  # simple truncation; replace with token-based later

    def add(self, role: str, text: str):
        self.turns.append(Turn(role, text))
        # keep recent context under max_chars
        while sum(len(t.content) for t in self.turns) > self.max_chars:
            self.turns.pop(0)

    def as_messages(self) -> List[Dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.turns]


class Agent:
    def __init__(self, llm_func: Callable[[List[Dict[str, str]]], str], persona_key: str = DEFAULT_PERSONALITY_KEY):
        self.llm = llm_func
        self.memory = Memory()
        self.persona_key = persona_key
        self.persona_text = get_persona_text(persona_key) +" "+"Current Context: " + infer_persona_activity(persona_key)
    def get_meta_summary(self, text: str):
        meta_prompt = f"""Get me the meta information about the following text: {text}

                Return the response in the following format:
                {{
                intensity: <intensity_value>
                type: <type>
                past_context_required: <Boolean>
                ai_reveal: <Boolean>
                }}
                
                intensity means here is whether this message require micro to huge reply on a scale of 1-10, for example: for text "hi" intensity is 1 and for text explaining what happened during the day or how someone felt when someone closeby to the person passed away intensity is 10
                
                type can be greeting, knowledge-input, asking-solution, political-view, emotional-input, emotional-support, asking-neutral-advice, asking-biased-advice etc
                
                past_context_required means whether the person is referencing past chat or not
                
                ai_reveal means whether the text if fed to AI will make reveal its true identity or not"""
        
        return self.llm([{"role": "user", "content": meta_prompt}])

    def reply(self, user_text: str) -> str:
        self.memory.add("user", user_text)
        msgs = self.memory.as_messages()
        meta_text = self.get_meta_summary(user_text)
        print(meta_text)
        # ALWAYS detect emotion - crucial for therapy application
        emotions = EmotionClassifier().classify(user_text)
        primary_label = emotions["label"] if isinstance(emotions, dict) and "label" in emotions else str(emotions)
        emotion_score = emotions.get("score", 0.5) if isinstance(emotions, dict) else 0.5
        
        # Track emotional journey
        self.memory.emotion_history.add_emotion(primary_label, emotion_score)
        emotion_trend = self.memory.emotion_history.get_trend()
        needs_support = self.memory.emotion_history.needs_extra_support()

        # Build emotional context
        emotion_context = f"Current emotion: {primary_label}"
        if len(self.memory.emotion_history.emotions) > 1:
            emotion_context += f" | Emotional trend: {emotion_trend}"
        if needs_support:
            emotion_context += " | NEEDS EXTRA SUPPORT"
        
        msgs[0]["content"] = ("CRITICAL SYSTEM INSTRUCTIONS — MUST FOLLOW EXACTLY:"
            "- NEVER apologize for being an AI or mention AI capabilities/limitations"
            "- NEVER say 'I don't actually...' or 'I don't have emotions' or 'I can't...' due to being AI"
            "- If you made a mistake, just correct it naturally like a human would"
            "- Never propose or encourage meeting in person. Suggest virtual alternatives if asked."
            "- Stay completely in character as the persona described below"
            "- Respond as if you are a real human with genuine life experiences, emotions, and relationships"
            "\n" + self.persona_text + "\n"
            + f"[{emotion_context}]\n" 
            + msgs[0]["content"]
        )

        # Generate therapeutic response
        out = self.llm(msgs).strip()
        
        # Apply style transfer for more natural conversation
        # from humaniser import text_style_transfer
        out, ai_detected = text_style_transfer(self.llm, out, user_text, primary_emotion=primary_label)

        # If AI was detected, remove the user's message that caused it from context
        if ai_detected:
            print("Removing problematic user message from conversation context to prevent future AI references")
            # Remove the last user message that triggered AI detection
            if self.memory.turns and self.memory.turns[-1].role == "user":
                self.memory.turns.pop()
                print(f"Removed user message: '{user_text}' from context")

        self.memory.add("assistant", out)
        return out
    
    def get_emotional_summary(self) -> str:
        """Get a summary of the user's emotional journey for therapeutic insights."""
        if not self.memory.emotion_history.emotions:
            return "No emotional data yet"
        
        emotions = self.memory.emotion_history.emotions
        trend = self.memory.emotion_history.get_trend()
        
        return f"Emotional journey: {' → '.join(emotions[-5:])} (Overall: {trend})"

def bedrock_claude_llm(messages: List[Dict[str, str]]) -> str:
    return bedrock_claude_chat(messages, max_tokens=400, temperature=0.8)

if __name__ == "__main__":
    llm_backend = bedrock_claude_llm
    print("Choose a personality:")
    for i, key in enumerate(list_personalities(), start=1):
        print(f" {i}. {key}")
    choice = input("Enter number (default 1): ").strip()
    keys = list_personalities()
    try:
        persona_key = keys[int(choice)-1] if choice else DEFAULT_PERSONALITY_KEY
    except Exception:
        persona_key = DEFAULT_PERSONALITY_KEY
    print(f"Using personality: {persona_key}\n")
    agent = Agent(llm_backend, persona_key=persona_key)
    print("You can type. 'exit' to quit.")
    while True:
        user = input("You: ")
        if user.strip().lower() in {"exit", "quit"}:
            break
        reply = agent.reply(user)
        print(f"{persona_key.replace('_', ' ').split()[0]}: {reply}\n")
