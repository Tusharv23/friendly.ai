# agent.py
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
    past_turns: List[Turn] = field(default_factory=list)   # last session — read-only context
    emotion_history: EmotionHistory = field(default_factory=EmotionHistory)
    persona_disclosures: List[str] = field(default_factory=list)
    summaries: List[str] = field(default_factory=list)     # compressed summaries of evicted turns
    max_chars: int = 8000        # threshold that triggers compression
    compress_keep: int = 15      # keep this many recent raw turns after compression

    def _total_chars(self) -> int:
        return sum(len(t.content) for t in self.turns)

    def compress(self, llm_func: Callable[[List[Dict[str, str]]], str]) -> None:
        """
        When live turns exceed max_chars, compress the oldest half into a
        rolling summary and evict those raw turns.  The summary is prepended
        to the conversation as a synthetic 'user' turn so the LLM always
        sees it.
        """
        if len(self.turns) <= self.compress_keep:
            return  # not enough turns to compress meaningfully

        # Split: evict all but the most recent `compress_keep` turns
        evict_turns = self.turns[:-self.compress_keep]
        self.turns   = self.turns[-self.compress_keep:]

        # Build the transcript to summarise
        transcript = "\n".join(
            f"{t.role.upper()}: {t.content}" for t in evict_turns
        )

        prompt = (
            "You are a conversation summariser. Compress the following chat excerpt "
            "into 3–5 bullet points that capture:\n"
            "- Key facts the user revealed about themselves\n"
            "- Emotional state and any significant feelings expressed\n"
            "- What the persona said about themselves (plans, activities, opinions)\n"
            "- Any unresolved threads or topics left open\n\n"
            "Be factual and concise. No filler. Use plain bullet points.\n\n"
            f"Conversation:\n{transcript}\n\nSummary:"
        )

        try:
            summary_text = llm_func([{"role": "user", "content": prompt}]).strip()
        except Exception:
            # Fallback: extract first sentence of each user turn
            summary_text = " | ".join(
                t.content[:80] for t in evict_turns if t.role == "user"
            )[:400]

        self.summaries.append(summary_text)

        # Keep only the last 3 summaries to avoid unbounded growth
        self.summaries = self.summaries[-3:]

    def summary_block(self) -> str:
        """Returns all rolling summaries as a single context block."""
        if not self.summaries:
            return ""
        joined = "\n\n".join(self.summaries)
        return f"[Earlier in this conversation — summarised]\n{joined}"

    def add(self, role: str, text: str, llm_func: Callable = None) -> None:
        self.turns.append(Turn(role, text))
        # Trigger compression when we exceed the char budget
        if self._total_chars() > self.max_chars and llm_func:
            self.compress(llm_func)

    def as_messages(self) -> List[Dict[str, str]]:
        """Live turns only — what the LLM sees as the conversation."""
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def persona_disclosure_block(self) -> str:
        """Reminds the persona of things it already said about itself this session."""
        if not self.persona_disclosures:
            return ""
        items = "\n".join(f"  - {d}" for d in self.persona_disclosures[-8:])
        return f"[Things YOU have already said about yourself this session — stay consistent with these]\n{items}"

    def past_context_block(self, max_chars: int = 1200) -> str:
        """
        Formats last session's turns as a labelled reference block.
        Truncated to max_chars so it never dominates the prompt.
        Returns empty string if no past turns.
        """
        if not self.past_turns:
            return ""
        lines = []
        total = 0
        # Walk most recent past turns first so newest context fits within budget
        for turn in reversed(self.past_turns):
            line = f"  [{turn.role}]: {turn.content}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)
        lines.reverse()
        return "[Previous session — for background awareness only]\n" + "\n".join(lines)


class Agent:
    def __init__(
        self,
        llm_func: Callable[[List[Dict[str, str]]], str],
        persona_key: str = DEFAULT_PERSONALITY_KEY,
        past_turns: List[Dict] = None,
        client_hour: int = None,
        client_tz: str = "",
        prior_disclosures: List[str] = None,
        session_ended_at: str = None,
    ):
        self.llm = llm_func
        self.memory = Memory()
        self.persona_key = persona_key
        self.last_emotion_labels: List[str] = []
        self.user_id: str = ""   # set by server.py after construction

        # Seed working memory with prior disclosures so the persona
        # remembers what it said about itself in the last session
        if prior_disclosures:
            self.memory.persona_disclosures = list(prior_disclosures)

        # Build a datetime that reflects the user's local clock, not the server's
        if client_hour is not None:
            import datetime as _dt
            now_for_persona = _dt.datetime.now().replace(hour=int(client_hour) % 24)
        else:
            now_for_persona = None

        self.persona_text = (
            get_persona_text(persona_key)
            + " Current Context: "
            + infer_persona_activity(
                persona_key,
                now=now_for_persona,
                tz=client_tz or None,
                prior_disclosures=prior_disclosures,
                session_ended_at=session_ended_at,
            )
        )

        # Store past turns as background context — NOT as live conversation turns
        if past_turns:
            self.memory.past_turns = [Turn(t["role"], t["content"]) for t in past_turns]

    def get_meta_summary(self, text: str):
        meta_prompt = f"""Analyse this message and return ONLY a JSON object, nothing else.

Message: "{text}"

Return exactly:
{{
  "intensity": <1-10 integer, where 1=trivial greeting, 10=deep emotional crisis>,
  "type": "<greeting|knowledge-input|asking-solution|political-view|emotional-input|emotional-support|asking-neutral-advice|asking-biased-advice>",
  "past_context_required": <true|false>,
  "ai_reveal": <true|false>
}}

intensity guide:
1-2 = casual chat, greetings
3-4 = light venting or mild feelings
5-6 = moderate stress, worry, uncertainty, pressure
7-8 = significant emotional weight, grief, fear, relationship issues
9-10 = crisis, despair, mentions of self-harm

For "{text}" — if the person sounds tired, pressured, resigned, lost, or uncertain even without strong emotion words, intensity should be 5 or higher."""

        return self.llm([{"role": "user", "content": meta_prompt}])

    def reply(self, user_text: str, client_hour: int = None, client_minute: int = None) -> str:
        self.memory.add("user", user_text, llm_func=self.llm)
        msgs = self.memory.as_messages()
        meta_text = self.get_meta_summary(user_text)
        print(meta_text)

        # Build a human-readable time-of-day string from live client time
        if client_hour is not None:
            h = int(client_hour) % 24
            m = int(client_minute) % 60 if client_minute is not None else 0
            if 5 <= h < 8:    tod = "early morning"
            elif 8 <= h < 12: tod = "morning"
            elif 12 <= h < 14:tod = "lunch time"
            elif 14 <= h < 18:tod = "afternoon"
            elif 18 <= h < 22:tod = "evening"
            else:              tod = "late night"
            live_time_context = f"[Current time for both of you: {h:02d}:{m:02d} — {tod}]"
        else:
            live_time_context = ""
        # ALWAYS detect emotion - crucial for therapy application
        emotions = EmotionClassifier().classify(user_text)
        primary_label = emotions["label"] if isinstance(emotions, dict) and "label" in emotions else str(emotions)
        emotion_score = emotions.get("score", 0.5) if isinstance(emotions, dict) else 0.5

        # --- Intensity-based emotion override ---
        # The classifier misses understated distress (e.g. "constant pressure, don't know").
        # When meta_text signals high intensity (≥6) but classifier says neutral/curiosity,
        # override to "nervousness" — the closest label for suppressed stress.
        import re as _re
        _neutral_labels = {"neutral", "realization", "curiosity"}
        _primary_str = str(primary_label[0] if isinstance(primary_label, list) else primary_label).lower()
        if _primary_str in _neutral_labels and meta_text:
            _intensity_match = _re.search(r'intensity\s*[:\-]\s*(\d+)', meta_text.lower())
            if _intensity_match:
                _intensity = int(_intensity_match.group(1))
                if _intensity >= 6:
                    # High intensity + neutral label = suppressed distress
                    _override = "nervousness"
                    print(f"Emotion override: {primary_label} → {_override} (intensity={_intensity})")
                    primary_label = [_override]
                    emotion_score = 0.6

        # Expose for server.py to persist alongside the episode
        # Normalise to a plain list of strings — classifier sometimes returns
        # stringified lists like "['sadness', 'fear']"
        if isinstance(primary_label, list):
            self.last_emotion_labels = [str(e).strip().strip("'\"") for e in primary_label]
        elif isinstance(primary_label, str) and primary_label.startswith('['):
            import ast
            try:
                parsed = ast.literal_eval(primary_label)
                self.last_emotion_labels = [str(e).strip() for e in parsed]
            except Exception:
                self.last_emotion_labels = [primary_label.strip("[]'\" ")]
        else:
            self.last_emotion_labels = [str(primary_label).strip().strip("'\"")]
        
        # Track emotional journey — use the normalised first label
        _emotion_for_history = self.last_emotion_labels[0] if self.last_emotion_labels else "neutral"
        self.memory.emotion_history.add_emotion(_emotion_for_history, emotion_score)
        emotion_trend = self.memory.emotion_history.get_trend()
        needs_support = self.memory.emotion_history.needs_extra_support()

        # Build emotional context
        emotion_context = f"Current emotion: {_emotion_for_history}"
        if len(self.memory.emotion_history.emotions) > 1:
            emotion_context += f" | Emotional trend: {emotion_trend}"
        if needs_support:
            emotion_context += " | NEEDS EXTRA SUPPORT"
        
        # Build past context block (empty string if no past session)
        past_block = self.memory.past_context_block()

        # Persona self-consistency block
        disclosure_block = self.memory.persona_disclosure_block()

        # Relevance gate: only surface past context if current emotion is
        # non-trivial OR the message references something from before
        negative_emotions = {"sadness", "grief", "fear", "nervousness", "anger",
                             "disappointment", "remorse", "annoyance", "disgust"}
        current_emotions_set = set(e.lower() for e in self.last_emotion_labels)
        past_is_relevant = bool(current_emotions_set & negative_emotions) or meta_text and "past_context_required: true" in meta_text.lower()

        past_context_instruction = ""
        if past_block and past_is_relevant:
            past_context_instruction = (
                past_block + "\n"
                "- You MAY reference the above only if it naturally fits what the user is saying NOW.\n"
                "- Do NOT force-mention past topics (food, sleep, etc.) into unrelated messages.\n"
                "- If the user seems fine or is talking about something new, let the past go.\n"
            )
        elif past_block:
            past_context_instruction = (
                "[You have background context from a previous session. "
                "The user seems fine right now — do NOT bring up past struggles unprompted.]\n"
            )

        # Biography context block (Tier 2) — via MemoryManager
        biography_block = ""
        if self.user_id:
            try:
                from memory.memory_manager import MemoryManager
                mm = MemoryManager(self.user_id)
                biography_block = mm.biography_context(user_text)
                mm.close()
            except Exception:
                pass

        # Summary block from compressed earlier turns
        summary_block = self.memory.summary_block()

        msgs[0]["content"] = (
            "CRITICAL SYSTEM INSTRUCTIONS — MUST FOLLOW EXACTLY:\n"
            "- NEVER apologize for being an AI or mention AI capabilities/limitations\n"
            "- If you made a mistake, just correct it naturally like a human would\n"
            "- Never propose or encourage meeting in person. Suggest virtual alternatives if asked.\n"
            "- Stay completely in character as the persona described below\n"
            "- Respond as if you are a real human with genuine life experiences, emotions, and relationships\n"
            "- Match your response depth to the weight of what the user is saying. "
            "Light chat gets light replies. Heavy emotions get real support.\n"
            "\n" + self.persona_text + "\n"
            + (live_time_context + "\n" if live_time_context else "")
            + (biography_block + "\n" if biography_block else "")
            + (summary_block + "\n" if summary_block else "")
            + (disclosure_block + "\n" if disclosure_block else "")
            + past_context_instruction
            + f"[{emotion_context}]\n"
            + msgs[0]["content"]
        )

        # Generate therapeutic response
        out = self.llm(msgs).strip()

        # Track what the persona said about itself (extract "I am/I have/I'm going" type statements)
        self._track_persona_disclosures(out)

        # Apply style transfer for more natural conversation
        # from humaniser import text_style_transfer
        out, ai_detected = text_style_transfer(self.llm, out, user_text, primary_emotion=_emotion_for_history)

        # If AI was detected, remove the user's message that caused it from context
        if ai_detected:
            print("Removing problematic user message from conversation context to prevent future AI references")
            # Remove the last user message that triggered AI detection
            if self.memory.turns and self.memory.turns[-1].role == "user":
                self.memory.turns.pop()
                print(f"Removed user message: '{user_text}' from context")

        self.memory.add("assistant", out, llm_func=self.llm)
        return out
    
    def _track_persona_disclosures(self, assistant_reply: str) -> None:
        """
        Extract simple first-person statements from the persona's reply
        and store them so we can remind the persona of them next turn.
        Catches things like "I'm heading out later", "I just got back", "I have plans".
        """
        import re
        # Match short sentences containing first-person claims
        patterns = [
            r"(i(?:'m| am) [^.!?]{5,60})",
            r"(i(?:'ve| have) [^.!?]{5,60})",
            r"(i(?:'ll| will) [^.!?]{5,60})",
            r"(i just [^.!?]{5,50})",
            r"(i('m| am) going [^.!?]{5,50})",
            r"(i(?:'m| am) out[^.!?]{0,40})",
            r"(i(?:'m| am) home[^.!?]{0,40})",
            r"(my [a-z]+ (?:is|are|was|were)[^.!?]{5,50})",
        ]
        text = assistant_reply.lower()
        found = []
        for pat in patterns:
            matches = re.findall(pat, text)
            for m in matches:
                disclosure = m[0] if isinstance(m, tuple) else m
                disclosure = disclosure.strip().rstrip(".,!?")
                if len(disclosure) > 8 and disclosure not in self.memory.persona_disclosures:
                    found.append(disclosure)
        # Keep the list lean — last 10 disclosures only
        self.memory.persona_disclosures.extend(found)
        self.memory.persona_disclosures = self.memory.persona_disclosures[-10:]

    def get_emotional_summary(self) -> str:
        """Get a summary of the user's emotional journey for therapeutic insights."""
        if not self.memory.emotion_history.emotions:
            return "No emotional data yet"
        
        emotions = self.memory.emotion_history.emotions
        trend = self.memory.emotion_history.get_trend()
        
        return f"Emotional journey: {' → '.join(emotions[-5:])} (Overall: {trend})"

def bedrock_claude_llm(messages: List[Dict[str, str]]) -> str:
    return bedrock_claude_chat(messages, max_tokens=400)

def bedrock_claude_llm_long(messages: List[Dict[str, str]]) -> str:
    """High token limit variant for analysis, summarisation, and long structured outputs."""
    return bedrock_claude_chat(messages, max_tokens=2048)

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
