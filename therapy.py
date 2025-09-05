# agent.py
from dataclasses import dataclass, field
from typing import List, Dict, Callable
from emotion_classifier import EmotionClassifier 
from personalities import PERSONALITIES, DEFAULT_PERSONALITY_KEY, list_personalities, get_persona_text
from llm import bedrock_claude_chat
import time
from persona_state import infer_persona_activity

# ---------- Conversation data structures ----------
@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str

@dataclass
class Memory:
    turns: List[Turn] = field(default_factory=list)
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

    def reply(self, user_text: str) -> str:
        self.memory.add("user", user_text)
        msgs = self.memory.as_messages()
        # For Bedrock Claude, prepend persona to the first user message
        if msgs and msgs[0]["role"] == "user":
            emotions = EmotionClassifier().classify(user_text)
            primary_label = emotions["label"] if isinstance(emotions, dict) and "label" in emotions else str(emotions)
            print(primary_label +" and "+self.persona_text)
            msgs[0]["content"] = (
                self.persona_text
                + "\nEmotion detected: " + primary_label
                + "\n" + msgs[0]["content"]
            )
        out = self.llm(msgs).strip()
        self.memory.add("assistant", out)
        return out

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
