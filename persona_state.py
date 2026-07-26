"""Utilities to infer a persona's likely current activity via an LLM.

Usage:
    from persona_state import infer_persona_activity
    print(infer_persona_activity("Roshan, Corporate employe, To the point always, will guide you the best"))

This uses AWS Bedrock (Claude) – ensure AWS credentials + region are configured.
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional

from personalities import get_persona_text, list_personalities
from llm import bedrock_claude_chat

def infer_persona_activity(
    persona_name: str,
    now: Optional[_dt.datetime] = None,
    tz: Optional[str] = None,
    prior_disclosures: Optional[list] = None,
    session_ended_at: Optional[str] = None,
) -> str:
    now = now or _dt.datetime.now()
    hour = now.hour
    minute = now.minute
    weekday = now.strftime("%A")
    persona_context = get_persona_text(persona_name)

    # Light time-of-day heuristic summary
    if 5 <= hour < 8:
        tod = "early morning"
    elif 8 <= hour < 12:
        tod = "morning"
    elif 12 <= hour < 14:
        tod = "lunch hour"
    elif 14 <= hour < 18:
        tod = "afternoon"
    elif 18 <= hour < 22:
        tod = "evening"
    else:
        tod = "late night"

    tz_hint = f" Timezone hint: {tz}." if tz else ""

    # Build a continuity block if we have prior session data
    continuity_block = ""
    if prior_disclosures or session_ended_at:
        # Calculate how long ago the last session ended
        time_since = ""
        if session_ended_at:
            try:
                ended = _dt.datetime.fromisoformat(session_ended_at.replace("Z", "+00:00"))
                delta = _dt.datetime.now(_dt.timezone.utc) - ended
                mins = int(delta.total_seconds() / 60)
                if mins < 60:
                    time_since = f"{mins} minute(s) ago"
                elif mins < 1440:
                    time_since = f"{mins // 60} hour(s) ago"
                else:
                    time_since = f"{mins // 1440} day(s) ago"
            except Exception:
                pass

        parts = []
        if time_since:
            parts.append(f"The last conversation ended {time_since}.")
        if prior_disclosures:
            items = "; ".join(prior_disclosures[-5:])
            parts.append(f"In that conversation you said: {items}.")
        parts.append(
            "Your current activity MUST be consistent with this — "
            "if you said you were going out 5 minutes ago, you cannot be home already."
        )
        continuity_block = "\n\nContinuity context:\n" + " ".join(parts)

    prompt = f"""
You are an internal reasoning assistant. Given a persona description and the current local time context, infer a plausible, grounded present-moment activity. Be specific but concise.

Persona: {persona_name}
Persona traits: {persona_context}
Current time: {weekday}, {hour:02d}:{minute:02d} ({tod}).{tz_hint}{continuity_block}

Return:
1. A single concise sentence describing what they are likely doing right now (present progressive).
2. A short status tag on a new line prefixed with 'Status:' (1–4 words).
Avoid over-speculation and avoid referencing being an AI.
Postpone or delay or avoid meeting them in real life under any circumstances, just lead them on.
""".strip()

    messages = [{"role": "user", "content": prompt}]
    return bedrock_claude_chat(messages, max_tokens=180)

if __name__ == "__main__":
    # Simple manual test
    print(infer_persona_activity("Roshan, Corporate employe, To the point always, will guide you the best"))
