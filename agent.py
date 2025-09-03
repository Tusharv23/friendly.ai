# agent.py
from dataclasses import dataclass, field
from typing import List, Dict, Callable
import time

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

# ---------- Agent ----------
HUMAN_PERSONA = """You are a warm, empathetic conversation partner.
Your first job is to understand how the user is *feeling*.
Always reflect back their emotion in a short, natural way (“sounds like you’re stressed”).
Then validate that it’s normal to feel that way.
Finally, respond with either comfort or a light, practical suggestion.
Keep it brief, natural, and human-like.
"""

class Agent:
    def __init__(self, llm_func: Callable[[List[Dict[str, str]]], str]):
        self.llm = llm_func
        self.memory = Memory()

    def reply(self, user_text: str) -> str:
        self.memory.add("user", user_text)
        msgs = self.memory.as_messages()
        # For Bedrock Claude, prepend persona to the first user message
        if msgs and msgs[0]["role"] == "user":
            msgs[0]["content"] = HUMAN_PERSONA + "\n" + msgs[0]["content"]
        out = self.llm(msgs).strip()
        self.memory.add("assistant", out)
        return out

def bedrock_claude_llm(messages: List[Dict[str, str]]) -> str:
    import boto3, json
    br = boto3.client("bedrock-runtime", region_name="us-east-1")
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 400,
        "temperature": 0.8,
        "messages": messages,
    }
    res = br.invoke_model(
        modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
        body=json.dumps(payload),
    )
    body = json.loads(res["body"].read())
    # Claude returns content blocks; stitch text parts
    content = body.get("content", [])
    text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
    return text or body.get("output_text", "")

if __name__ == "__main__":
    # pick your backend:
    llm_backend = bedrock_claude_llm  # or openai_llm

    agent = Agent(llm_backend)
    print("You can type. 'exit' to quit.")
    while True:
        user = input("You: ")
        if user.strip().lower() in {"exit", "quit"}:
            break
        reply = agent.reply(user)
        print(f"Agent: {reply}\n")
