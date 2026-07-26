"""
Memory Manager — unified facade for all three memory tiers.

The Agent calls this instead of touching EpisodeStore, BiographyStore,
or AnalysisEngine directly.  This keeps therapy.py clean and makes the
memory system swappable without touching agent code.

Tiers:
    Tier 1  EpisodeStore     — raw turn log (TinyDB)
    Tier 2  BiographyStore   — vector-indexed biographical facts
    Tier 3  Working Memory   — lives in Memory dataclass inside Agent

Public interface used by Agent:
    mm = MemoryManager(user_id)
    mm.append_turn(session_id, index, role, text, emotions, persona_key)
    mm.save_persona_state(persona_key, disclosures)
    mm.biography_context(query) -> str   # prompt block
    mm.get_past_session() -> (session_id, turns)
    mm.load_persona_state(persona_key) -> dict
    mm.trigger_analysis(session_id)      # non-blocking background thread
    mm.close()
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, List, Dict, Optional

from memory.episode_store import EpisodeStore
from memory.biography_store import BiographyStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Single entry point for all persistent memory operations.

    Usage:
        mm = MemoryManager("tushar", llm_func=bedrock_claude_llm)
        mm.append_turn(session_id, 0, "user", "hey", ["neutral"], persona_key)
        block = mm.biography_context("feeling overwhelmed")
    """

    def __init__(
        self,
        user_id: str,
        llm_func: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ) -> None:
        self.user_id = user_id.strip().lower()
        self._llm = llm_func
        self._episodes = EpisodeStore(self.user_id)
        self._biography = BiographyStore(self.user_id)

    # ------------------------------------------------------------------
    # Tier 1 — Episode operations
    # ------------------------------------------------------------------

    def append_turn(
        self,
        session_id: str,
        turn_index: int,
        role: str,
        content: str,
        emotion_labels: List[str],
        persona_key: str,
    ) -> None:
        """Persist one turn. Never raises."""
        self._episodes.append(
            session_id, turn_index, role, content, emotion_labels, persona_key
        )

    def get_past_session(self) -> tuple[Optional[str], List[dict]]:
        """Return (session_id, turns) of the most recent session."""
        return self._episodes.get_last_session()

    def get_message_count(self, session_id: str) -> int:
        return self._episodes.get_message_count(session_id)

    def save_persona_state(self, persona_key: str, disclosures: List[str]) -> None:
        self._episodes.save_persona_state(persona_key, disclosures)

    def load_persona_state(self, persona_key: str) -> dict:
        return self._episodes.load_persona_state(persona_key)

    def get_full_history(self) -> List[dict]:
        """All turns across all sessions, oldest first."""
        all_sessions = self._episodes.get_all_sessions()
        messages = []
        for sid in all_sessions:
            for turn in self._episodes.get_session_turns(sid):
                messages.append({
                    "role":           turn["role"],
                    "content":        turn["content"],
                    "session_id":     turn["session_id"],
                    "timestamp":      turn.get("timestamp", ""),
                    "emotion_labels": turn.get("emotion_labels", []),
                })
        return messages

    # ------------------------------------------------------------------
    # Tier 2 — Biography operations
    # ------------------------------------------------------------------

    def biography_context(self, query: str, max_chars: int = 800) -> str:
        """
        Build and return the [Biography context] prompt block for the
        current user message.  Returns empty string if biography is empty.
        """
        try:
            return self._biography.build_context_block(query, max_chars=max_chars)
        except Exception as exc:
            logger.warning("MemoryManager.biography_context failed: %s", exc)
            return ""

    def biography_summary(self) -> dict:
        return self._biography.summary()

    def get_dashboard_data(self) -> dict:
        """Aggregate everything needed for the dashboard endpoint."""
        all_sessions = self._episodes.get_all_sessions()
        emotion_timeline = []
        for sid in all_sessions:
            for turn in self._episodes.get_session_turns(sid):
                if turn.get("role") == "user" and turn.get("emotion_labels"):
                    emotion_timeline.append({
                        "timestamp":    turn.get("timestamp", ""),
                        "session_id":   sid[:8],
                        "emotion":      turn["emotion_labels"][0],
                        "all_emotions": turn["emotion_labels"],
                    })

        biography = {
            "crux":        self._biography.get_crux(),
            "facts":       self._biography.get_all("fact"),
            "patterns":    self._biography.get_all("pattern"),
            "phases":      self._biography.get_active_phases(),
            "predictions": self._biography.get_active_predictions(),
        }

        return {
            "user_id":          self.user_id,
            "session_count":    len(all_sessions),
            "emotion_timeline": emotion_timeline,
            "biography":        biography,
            "bio_summary":      self._biography.summary(),
        }

    # ------------------------------------------------------------------
    # Analysis trigger
    # ------------------------------------------------------------------

    def trigger_analysis(self, session_id: str) -> None:
        """
        Fire-and-forget: run the analysis engine for this session in a
        background thread.  Requires llm_func to have been provided.
        """
        if not self._llm:
            logger.warning("MemoryManager.trigger_analysis: no llm_func provided")
            return

        llm = self._llm
        user_id = self.user_id

        def _run() -> None:
            try:
                from memory.analysis_engine import AnalysisEngine
                engine = AnalysisEngine(llm, user_id)
                result = engine.run(session_id)
                if result:
                    logger.info(
                        "Analysis done — session=%s eq_delta=%s new_entries=%d",
                        session_id[:8],
                        result.get("eq_delta"),
                        len(result.get("new_entry_ids", [])),
                    )
                engine.close()
            except Exception as exc:
                logger.warning(
                    "MemoryManager analysis thread failed session=%r: %s",
                    session_id, exc,
                )

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._episodes.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"MemoryManager(user_id={self.user_id!r})"
