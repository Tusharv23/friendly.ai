"""
Analysis Engine
----------------
The intelligence core of Tier 2.  Runs at session end OR every 100
messages (whichever comes first).

What it does in one LLM call per analysis run:
  1. Reads all turns from the session being analysed
  2. Checks every pending prediction against what actually happened
  3. Extracts new facts, patterns, emotional phases
  4. Generates new predictions with confidence scores
  5. Writes the session crux (2-3 sentence summary)
  6. Computes an EQ delta (did this session help or hurt emotional growth?)

The engine talks to:
  - EpisodeStore  → read source turns
  - BiographyStore → read existing biography, write new/updated entries
  - LLM           → one structured analysis call per run

Output is a structured JSON object parsed and written to BiographyStore.
All writes are additive — nothing is deleted without explicit deactivation.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Callable, List, Dict, Optional

from memory.episode_store import EpisodeStore
from memory.biography_store import BiographyStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_ANALYSIS_PROMPT = """You are an analytical assistant helping build a deep psychological profile of a user
based on their conversation with an AI companion. Your job is to be insightful, precise, and human.

USER ID: {user_id}
SESSION ID: {session_id}

EXISTING BIOGRAPHY (what we already know):
{existing_biography}

PENDING PREDICTIONS TO CHECK:
{pending_predictions}

SESSION TRANSCRIPT:
{transcript}

EMOTION ARC THIS SESSION:
{emotion_arc}

---

Analyse this session and return a JSON object with exactly this structure:

{{
  "crux": "2-3 sentence summary of what this session was really about emotionally and factually",

  "prediction_checks": [
    {{
      "fact_id": "<fact_id of the prediction being checked>",
      "outcome": "verified" | "defied" | "inconclusive",
      "reasoning": "one sentence explaining why",
      "defied_by": "what actually happened (only if defied, else null)"
    }}
  ],

  "new_entries": [
    {{
      "type": "fact" | "pattern" | "phase" | "prediction",
      "text": "clear, specific, human-readable statement about the user",
      "confidence": 0.0-1.0,
      "reasoning": "one sentence explaining the evidence for this"
    }}
  ],

  "deactivate_ids": ["<fact_id>"],

  "eq_delta": -2 to +2,
  "eq_reasoning": "one sentence on why EQ moved this direction"
}}

RULES:
- facts: things directly stated or strongly implied by the user ("has a job interview next week")
- patterns: recurring behaviours across messages ("deflects with humour when uncomfortable")
- phases: current emotional arc stage ("moving from denial into anger about the breakup")
- predictions: specific, testable things you expect in the NEXT 1-3 sessions with confidence
- Only generate entries you can justify from the transcript — no speculation beyond evidence
- predictions must be falsifiable: "Will feel X" or "Will mention Y" — not vague
- deactivate_ids: fact_ids of existing entries that are now clearly outdated or contradicted
- eq_delta: +2 breakthrough session, +1 slight improvement, 0 neutral, -1 struggling, -2 crisis
- Return ONLY the JSON object, no preamble or explanation outside it
- CRITICAL: all string values must use straight double quotes only. Do NOT use apostrophes or single quotes inside string values — write "does not" instead of "doesn't", "it is" instead of "it's" etc."""


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------

class AnalysisEngine:
    """
    Runs the biography analysis for one session.

    Usage:
        engine = AnalysisEngine(llm_func, user_id="tushar")
        result = engine.run(session_id)
    """

    def __init__(
        self,
        llm_func: Callable[[List[Dict[str, str]]], str],
        user_id: str,
    ) -> None:
        self.llm = llm_func
        self.user_id = user_id.strip().lower()
        self.episode_store   = EpisodeStore(user_id)
        self.biography_store = BiographyStore(user_id)

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_transcript(self, turns: list[dict]) -> str:
        """Format turns as a readable transcript."""
        lines = []
        for t in turns:
            role = t.get("role", "user").upper()
            content = t.get("content", "")
            ts = t.get("timestamp", "")[:16]   # YYYY-MM-DDTHH:MM
            emotions = t.get("emotion_labels", [])
            emotion_str = f" [{', '.join(emotions)}]" if emotions else ""
            lines.append(f"[{ts}] {role}{emotion_str}: {content}")
        return "\n".join(lines) if lines else "(no turns)"

    def _build_emotion_arc(self, turns: list[dict]) -> str:
        """Summarise the emotional progression through the session."""
        user_emotions = [
            (t.get("timestamp", "")[:16], t.get("emotion_labels", []))
            for t in turns
            if t.get("role") == "user" and t.get("emotion_labels")
        ]
        if not user_emotions:
            return "(no emotion data)"
        arc = " → ".join(
            f"{e[1][0]}" for e in user_emotions if e[1]
        )
        return arc

    def _build_existing_biography(self) -> str:
        """Summarise existing biography entries for the prompt."""
        all_entries = self.biography_store.get_all(active_only=True)
        if not all_entries:
            return "(no existing biography)"
        lines = []
        for e in all_entries[:20]:   # cap at 20 to stay within prompt budget
            conf = f"{e.get('confidence', 0):.0%}"
            lines.append(
                f"  [{e['type'].upper()}] {e['text']}  "
                f"(confidence: {conf}, id: {e['fact_id'][:8]}...)"
            )
        return "\n".join(lines)

    def _build_pending_predictions(self) -> str:
        """Format pending predictions for the check."""
        preds = self.biography_store.get_active_predictions()
        if not preds:
            return "(no pending predictions)"
        lines = []
        for p in preds:
            lines.append(
                f"  [id: {p['fact_id'][:8]}...] {p['text']}  "
                f"(confidence: {p.get('confidence', 0):.0%})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM call + JSON parsing
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> Optional[dict]:
        """Call LLM and parse JSON response. Returns None on failure."""
        try:
            response = self.llm([{"role": "user", "content": prompt}])

            # Extract JSON block — handle markdown fences if present
            match = re.search(r'\{[\s\S]*\}', response)
            if not match:
                logger.warning("AnalysisEngine: no JSON found in LLM response")
                return None

            raw_json = match.group()

            def clean_json(s: str) -> str:
                """
                Fix common LLM JSON issues:
                1. Literal newlines inside string values → space
                2. Unescaped apostrophes → escaped
                """
                result = []
                in_string = False
                i = 0
                while i < len(s):
                    c = s[i]
                    if c == '\\' and i + 1 < len(s):
                        # Keep escape sequences as-is
                        result.append(c)
                        result.append(s[i + 1])
                        i += 2
                        continue
                    if c == '"':
                        in_string = not in_string
                        result.append(c)
                    elif in_string:
                        if c == '\n':
                            result.append(' ')   # collapse newline to space
                        elif c == '\r':
                            pass                 # drop carriage return
                        elif c == "'":
                            result.append("'")   # apostrophes are fine in JSON strings
                        else:
                            result.append(c)
                    else:
                        result.append(c)
                    i += 1
                return ''.join(result)

            # First try strict parse
            try:
                return json.loads(raw_json)
            except json.JSONDecodeError:
                pass

            # Fallback 1: clean and retry
            try:
                return json.loads(clean_json(raw_json))
            except json.JSONDecodeError:
                pass

            # Fallback 2: ask the LLM to repair its own JSON
            try:
                repair_prompt = (
                    "The following JSON is malformed. Fix it and return ONLY valid JSON, nothing else. "
                    "Replace any apostrophes inside string values with the word form (e.g. 'doesn't' -> 'does not').\n\n"
                    f"{raw_json}"
                )
                repaired = self.llm([{"role": "user", "content": repair_prompt}])
                repair_match = re.search(r'\{[\s\S]*\}', repaired)
                if repair_match:
                    return json.loads(repair_match.group())
            except Exception:
                pass

            # Fallback 3: use ast.literal_eval on a python-dict-like string
            import ast
            try:
                return ast.literal_eval(raw_json)
            except Exception:
                pass

            # Fallback 3: extract fields manually with regex — get what we can
            logger.warning("AnalysisEngine: JSON malformed, attempting partial extraction")
            result: dict = {}

            crux_m = re.search(r'"crux"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_json, re.DOTALL)
            if crux_m:
                result["crux"] = crux_m.group(1).replace('\n', ' ').strip()

            eq_m = re.search(r'"eq_delta"\s*:\s*(-?\d+)', raw_json)
            if eq_m:
                result["eq_delta"] = int(eq_m.group(1))

            eq_r = re.search(r'"eq_reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_json, re.DOTALL)
            if eq_r:
                result["eq_reasoning"] = eq_r.group(1).replace('\n', ' ').strip()

            # Extract new_entries array manually
            entries = []
            for em in re.finditer(
                r'\{\s*"type"\s*:\s*"(\w+)"\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"confidence"\s*:\s*([\d.]+)',
                raw_json, re.DOTALL
            ):
                entries.append({
                    "type":       em.group(1),
                    "text":       em.group(2).replace('\n', ' ').strip(),
                    "confidence": float(em.group(3)),
                    "reasoning":  "",
                })
            result["new_entries"] = entries

            result.setdefault("prediction_checks", [])
            result.setdefault("deactivate_ids", [])
            result.setdefault("eq_delta", 0)

            if result.get("crux"):
                logger.info("AnalysisEngine: partial extraction succeeded (crux + eq)")
                return result

            logger.warning("AnalysisEngine: partial extraction failed too — raw response:\n%s", raw_json[:800])
            return None

        except Exception as e:
            logger.warning("AnalysisEngine: LLM call failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Result processing
    # ------------------------------------------------------------------

    def _process_result(self, result: dict, session_id: str) -> dict:
        """
        Write the analysis result to BiographyStore.
        Returns a summary of what was written.
        """
        written = {
            "crux_id": None,
            "prediction_checks": [],
            "new_entry_ids": [],
            "deactivated_ids": [],
            "eq_delta": 0,
        }

        # 1. Write crux — deactivate any previous crux first
        crux_text = result.get("crux", "").strip()
        if crux_text:
            old_crux = self.biography_store.get_crux()
            if old_crux:
                self.biography_store.deactivate(old_crux["fact_id"])
            crux_id = self.biography_store.upsert({
                "type": "crux",
                "text": crux_text,
                "confidence": 1.0,
                "source_sessions": [session_id],
            })
            written["crux_id"] = crux_id

        # 2. Process prediction checks
        for check in result.get("prediction_checks", []):
            fact_id = check.get("fact_id", "")
            # Resolve short IDs (first 8 chars) to full IDs
            if len(fact_id) < 36:
                fact_id = self._resolve_short_id(fact_id)
            if not fact_id:
                continue

            outcome = check.get("outcome", "inconclusive")
            if outcome == "verified":
                self.biography_store.mark_verified(fact_id)
                written["prediction_checks"].append({"id": fact_id, "outcome": "verified"})
            elif outcome == "defied":
                self.biography_store.mark_defied(
                    fact_id,
                    check.get("defied_by") or check.get("reasoning", ""),
                )
                written["prediction_checks"].append({"id": fact_id, "outcome": "defied"})
            else:
                # Inconclusive — slight confidence decay to avoid stale predictions
                self.biography_store.update_confidence(fact_id, -0.05)
                written["prediction_checks"].append({"id": fact_id, "outcome": "inconclusive"})

        # 3. Write new entries
        for entry_data in result.get("new_entries", []):
            if not entry_data.get("text", "").strip():
                continue

            entry_type = entry_data.get("type", "fact")

            # Deactivate old phase entries if a new phase is being set
            if entry_type == "phase":
                for old_phase in self.biography_store.get_active_phases():
                    self.biography_store.deactivate(old_phase["fact_id"])

            # Deduplication: check cosine similarity against existing entries
            # of the same type — if too similar, boost confidence instead of inserting
            new_text = entry_data["text"].strip()
            existing = self.biography_store.search(
                new_text, top_k=1, types=[entry_type], min_confidence=0.0
            )
            if existing and existing[0]["_score"] >= 0.65:
                # Similar entry already exists — nudge confidence up and skip insert
                existing_entry = existing[0]
                delta = (entry_data.get("confidence", 0.7) - existing_entry["confidence"]) * 0.3
                self.biography_store.update_confidence(existing_entry["fact_id"], delta)
                # Add this session to source_sessions
                src = existing_entry.get("source_sessions", [])
                if session_id not in src:
                    src.append(session_id)
                    existing_entry["source_sessions"] = src
                    self.biography_store._save_metadata()
                logger.info(
                    "Dedup: merged new %s entry into existing (score=%.2f)",
                    entry_type, existing[0]["_score"]
                )
                continue

            fact_id = self.biography_store.upsert({
                "type":            entry_type,
                "text":            new_text,
                "confidence":      entry_data.get("confidence", 0.7),
                "source_sessions": [session_id],
            })
            if fact_id:
                written["new_entry_ids"].append(fact_id)

        # 4. Deactivate stale entries
        for fact_id in result.get("deactivate_ids", []):
            if len(fact_id) < 36:
                fact_id = self._resolve_short_id(fact_id)
            if fact_id:
                self.biography_store.deactivate(fact_id)
                written["deactivated_ids"].append(fact_id)

        # 5. EQ delta
        written["eq_delta"] = result.get("eq_delta", 0)

        # 6. Deduplicate biography after every write
        removed = self.biography_store.deduplicate(threshold=0.65)
        if removed:
            logger.info("BiographyStore: deduped %d entries", removed)

        return written

    def _resolve_short_id(self, short_id: str) -> str:
        """Match a short (8-char) fact_id prefix to a full UUID."""
        short_id = short_id.rstrip(".")
        all_entries = self.biography_store.get_all(active_only=False)
        for entry in all_entries:
            if entry.get("fact_id", "").startswith(short_id):
                return entry["fact_id"]
        return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, session_id: str) -> Optional[dict]:
        """
        Run the full analysis for a session.
        Returns the written summary dict, or None if analysis could not run.
        """
        # Load the session turns
        turns = self.episode_store.get_session_turns(session_id)
        if not turns:
            logger.info("AnalysisEngine.run: no turns found for session %r", session_id)
            return None

        # Only analyse user turns (assistant turns are context, not subject)
        user_turns = [t for t in turns if t.get("role") == "user"]
        if len(user_turns) < 2:
            logger.info(
                "AnalysisEngine.run: only %d user turn(s) in session %r — skipping",
                len(user_turns), session_id,
            )
            return None

        logger.info(
            "AnalysisEngine.run: analysing session %r (%d turns) for user %r",
            session_id, len(turns), self.user_id,
        )

        # Build the prompt
        prompt = _ANALYSIS_PROMPT.format(
            user_id=self.user_id,
            session_id=session_id,
            existing_biography=self._build_existing_biography(),
            pending_predictions=self._build_pending_predictions(),
            transcript=self._build_transcript(turns),
            emotion_arc=self._build_emotion_arc(turns),
        )

        # Call LLM
        result = self._call_llm(prompt)
        if not result:
            logger.warning(
                "AnalysisEngine.run: LLM returned no valid result for session %r", session_id
            )
            return None

        # Process and write
        written = self._process_result(result, session_id)
        written["session_id"] = session_id
        written["analysed_at"] = self._now_utc()
        written["raw_result"]  = result   # keep for debugging

        logger.info(
            "AnalysisEngine.run: session %r — crux=%s, new_entries=%d, checks=%d, eq_delta=%s",
            session_id,
            bool(written["crux_id"]),
            len(written["new_entry_ids"]),
            len(written["prediction_checks"]),
            written["eq_delta"],
        )

        return written

    def close(self) -> None:
        self.episode_store.close()
